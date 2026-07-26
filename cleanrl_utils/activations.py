from typing import Any
from collections.abc import Callable
import torch
import torch.nn as nn
import logging
import numpy as np
from torchmetrics import Metric

from gromo.containers.growing_container import GrowingModule
from gromo.utils.utils import global_device
from gromo.utils.training_utils import (
    AverageMeter,
    DummyMetric,
    enumerate_dataloader
)


class _ReLUDerivativeOneAtZero(torch.autograd.Function):
    """ReLU forward; backward uses (x >= 0) so the subgradient at 0 is 1,
    not 0.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return torch.clamp(x, min=0.0)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        mask = (x >= 0).to(dtype=grad_output.dtype)
        return grad_output * mask


class ReLUDerivativeOneAtZero(nn.Module):
    """
    ReLU with the convention :math:`f'(0)=1` (and :math:`f(0)=0`).

    Forward matches :class:`torch.nn.ReLU`. For the backward, gradients use the
    mask ``x >= 0`` instead of PyTorch's ``x > 0``, so a pre-activation exactly
    zero is treated like the linear branch with unit slope—consistent with
    assumptions that :math:`f(0)=0` and :math:`f'(0)=1` when zero lies in the
    linear part. Standard ``nn.ReLU`` yields zero gradient at :math:`x=0`.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ReLU with derivative set to 1 at :math:`x=0`."""
        return _ReLUDerivativeOneAtZero.apply(x)


class _ReLUDerivativeOneAtZeroFunctorch(torch.autograd.Function):
    """Same as _ReLUDerivativeOneAtZero but using the new-style API
    (setup_context) required for compatibility with functorch transforms
    (torch.func.grad, vmap, ...) used by gromo's first_order_improvement.
    """

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0.0)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple, output: torch.Tensor) -> None:
        (x,) = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        mask = (x >= 0).to(dtype=grad_output.dtype)
        return grad_output * mask


class ReLUDerivativeOneAtZeroFunctorch(nn.Module):
    """
    Functorch-compatible variant of ReLUDerivativeOneAtZero.
    Use this as post_layer_function in gromo LinearGrowingModule so that
    first_order_improvement can be computed via torch.func.grad.
    Identical forward/backward to ReLUDerivativeOneAtZero.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _ReLUDerivativeOneAtZeroFunctorch.apply(x)


@torch.no_grad()
def evaluate_layer(
    model: nn.Module,
    layer: GrowingModule,
    dataloader: torch.utils.data.DataLoader,
    loss_function: nn.Module | Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    metrics: Metric | None = None,
    batch_limit: int | None = None,
    dataloader_seed: int | None = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[float, float]:
    """
    Evaluate the extended forward pass of the growing layer in the context of
    the full model, using a dataloader of (observations, targets) pairs.

    Performs the chain:
        conv → flat → encoder.extended_forward → q_head.extended_forward

    Parameters
    ----------
    model : nn.Module
        Full Q-network (must expose .conv, .flat, .encoder, .q_head).
    layer : GrowingModule
        The growing layer being tested. Not called directly here — the
        scaling_factor set on it is picked up by the extended_forward chain.
    dataloader : torch.utils.data.DataLoader
        Yields (observations [B,H,W,C], targets) batches.
    loss_function : nn.Module | Callable
        Applied to (q_vals, targets). Must have reduction="mean".
    metrics : Metric | None, optional
        Auxiliary metric. Default is None.
    batch_limit : int | None, optional
        Maximum number of batches. None = no limit.
    dataloader_seed : int | None, optional
        Seed for the dataloader generator.
    device : torch.device, optional
        Device to use. Default is cpu.

    Returns
    -------
    tuple[float, float]
        (average_loss, metrics_value).
    """
    assert (
        not isinstance(loss_function, nn.Module) or loss_function.reduction == "mean"
    ), "The loss function should be averaged over the batch"

    loss_meter = AverageMeter()
    if metrics is None:
        metrics = DummyMetric()
    else:
        metrics.reset()
        metrics = metrics.to(device)

    model.eval()
    for _, (x, y) in enumerate_dataloader(
        dataloader, dataloader_seed=dataloader_seed, batch_limit=batch_limit
    ):
        x, y = x.to(device), y.to(device)
        flat_x = model.flat(model.conv(x.permute(0, 3, 1, 2).float()))
        main_enc, ext_enc = model.encoder.extended_forward(flat_x)
        y_pred, _ = model.q_head.extended_forward(main_enc, x_ext=ext_enc)
        loss = loss_function(y_pred, y)
        loss_meter.update(loss, x.size(0))
        metrics.update(y_pred, y)

    return loss_meter.compute().item(), metrics.compute().item()


def line_search(
    model: nn.Module,
    layer: GrowingModule,
    dataloader: torch.utils.data.DataLoader,
    dataloader_seed: int | None = None,
    loss_function: nn.Module = nn.MSELoss(reduction="sum"),
    aux_loss_function: Metric | None = None,
    batch_limit: int = -1,
    initial_loss: float | None = None,
    initial_aux_loss: float | None = None,
    first_order_improvement: float | torch.Tensor = 1,
    alpha: float = 0.1,
    beta: float = 0.5,
    t0: float | None = None,
    max_gamma: float | None = None,
    extended_search: bool = False,
    max_iter: int = 20,
    epsilon: float = 1e-7,
    verbose: bool = False,
    device: torch.device | None = None,
) -> tuple[float, float, float, list[float], list[float], list[float]]:
    """
    Perform line search to find optimal scaling factor for the currently updated layer.

    This function performs a backtracking line search with Armijo condition to find
    the optimal scaling factor (gamma) for the newly added neurons in the currently
    updated layer. The search operates on the square root of gamma (t) for numerical
    stability, where gamma = t^2.

    Parameters
    ----------
    layer : GrowingModule
        An updated layer that has a scaling_factor attribute.
    dataloader : torch.utils.data.DataLoader
        DataLoader to evaluate the layer on during the line search.
    dataloader_seed : int | None, optional
        Seed for shuffling the dataloader during evaluation. If None, no reseeding is done.
        Default is None.
    loss_function : nn.Module, optional
        Loss function to minimize. Should use reduction="sum".
        Default is nn.MSELoss(reduction="sum").
    aux_loss_function : Metric | None, optional
        A Metric instance to track auxiliary metrics (e.g., accuracy).
        Default is None.
    batch_limit : int, optional
        Maximum number of batches to use for evaluation. Use -1 for no limit.
        Default is -1.
    initial_loss : float | None, optional
        Initial loss at gamma=0. If None, it will be computed.
        Default is None.
    initial_aux_loss : float | None, optional
        Initial auxiliary loss at gamma=0. If None, will be replaced by 0.
        Default is None.
    first_order_improvement : float | torch.Tensor, optional
        Expected first-order improvement in loss. Used to initialize the search
        and for the Armijo condition.
        Default is 1.
    alpha : float, optional
        Armijo condition parameter. The sufficient decrease condition is:
        loss < initial_loss - alpha * gamma * first_order_improvement.
        Default is 0.1.
    beta : float, optional
        Step size reduction factor. The actual factor applied is sqrt(beta) since
        we work with t = sqrt(gamma). Typical values: 0.5 for aggressive, 0.8 for conservative.
        Default is 0.5.
    t0 : float | None, optional
        Initial value for t (sqrt of initial gamma). If None, computed as:
        t0 = sqrt(2 * initial_loss / first_order_improvement).
        Default is None.
    max_gamma : float | None, optional
        Maximum allowable gamma value. If None, no maximum is enforced.
    extended_search : bool, optional
        If True, extends the search in both directions (increase and decrease gamma)
        to find a local minimum. If False, only performs backtracking.
        Default is True.
    max_iter : int, optional
        Maximum number of iterations for the line search.
        Default is 100.
    epsilon : float, optional
        Minimum value for t (sqrt of gamma) below which search stops.
        Actual threshold is sqrt(epsilon).
        Default is 1e-7.
    verbose : bool, optional
        If True, prints detailed information about each tested gamma value.
        Default is False.
    device : torch.device | None, optional
        Device to perform computations on. If None, uses global_device().
        Default is None.

    Returns
    -------
    tuple[float, float, float, list[float], list[float], list[float]]
        A tuple containing:
        - gamma (float): The optimal scaling factor (t^2).
        - loss (float): The loss at the optimal gamma.
        - aux_loss (float): The auxiliary loss at the optimal gamma.
        - gammas (list[float]): List of all tested gamma values.
        - losses (list[float]): List of losses corresponding to tested gammas.
        - aux_losses (list[float]): List of auxiliary losses corresponding to tested gammas.

    Raises
    ------
    AssertionError
        If layer is None.

    Notes
    -----
    - The function works with t = sqrt(gamma) for numerical stability.
    - The Armijo condition ensures sufficient decrease in the loss function.
    - The extended_search option allows finding better local minima by exploring
      beyond the first acceptable point.
    - The scaling_factor of the currently_updated_layer is set to the optimal
      value (t, not gamma) at the end of the function.

    Examples
    --------
    >>> from tools.metrics import TopKAccuracy
    >>> gamma, loss, aux_loss, gammas, losses, aux_losses = line_search(
    ...     layer=growing_layer,
    ...     dataloader=train_loader,
    ...     loss_function=nn.CrossEntropyLoss(reduction="sum"),
    ...     aux_loss_function=TopKAccuracy(k=1),
    ...     initial_loss=100.0,
    ...     first_order_improvement=5.0,
    ...     alpha=0.1,
    ...     beta=0.5,
    ...     verbose=True
    ... )
    """
    logger = logging.getLogger(__name__)
    assert layer is not None, "No currently updated layer"

    if device is None:
        device = global_device()

    gammas = []
    losses = []
    aux_losses = []
    beta = np.sqrt(beta)
    epsilon = np.sqrt(epsilon)
    if isinstance(first_order_improvement, torch.Tensor):
        first_order_improvement = first_order_improvement.item()
    if isinstance(initial_loss, torch.Tensor):
        initial_loss = initial_loss.item()

    def test_gamma(sqrt_gamma):
        layer.set_scaling_factor(sqrt_gamma)
        loss, aux_loss = evaluate_layer(
            model=model,
            layer=layer,
            dataloader=dataloader,
            dataloader_seed=dataloader_seed,
            loss_function=loss_function,
            metrics=aux_loss_function,
            batch_limit=batch_limit,
            device=device,
        )
        gammas.append(sqrt_gamma**2)
        losses.append(loss)
        aux_losses.append(aux_loss)
        if verbose:
            logger.info(
                f"gamma n° {len(gammas)}: {sqrt_gamma**2:.3e} -> Loss: {loss:.6e} (aux_loss: {aux_loss * 100:.2f}%)"
            )
        return loss, aux_loss

    if initial_loss is None:
        logger.warning("Initial loss is not provided, computing it")
        initial_loss, initial_aux_loss = test_gamma(0.0)
        logger.info(
            f"Initial loss: {initial_loss:.3e} (aux_loss: {initial_aux_loss * 100:.2f}%)"
        )
    gammas.append(0.0)
    losses.append(initial_loss)
    initial_aux_loss = initial_aux_loss if initial_aux_loss is not None else 0.0
    aux_losses.append(initial_aux_loss)
    if verbose:
        logger.info(
            f"gamma n° {len(gammas)}: {0.0:.3e} -> Loss: {initial_loss:.3e} (aux_loss: {initial_aux_loss * 100:.2f}%)"
        )

    def under_bound(sqrt_gamma: float, loss: float):
        return loss < initial_loss - alpha * sqrt_gamma**2 * first_order_improvement

    # gamma = t ** 2
    if t0 is None:
        t = np.sqrt(2 * (initial_loss / first_order_improvement))
    else:
        t = np.sqrt(t0)
    if max_gamma is not None:
        max_t = np.sqrt(max_gamma)
        t = min(t, max_t)
    l0, l0_aux = test_gamma(t)
    l1, l1_aux = l0, l0_aux
    i = 0
    if under_bound(t, l0):
        if extended_search:
            go = t / beta < max_t
            while go:
                l0, l0_aux = l1, l1_aux
                t /= beta
                l1, l1_aux = test_gamma(t)
                go = l1 < l0 and i < max_iter and t < max_t
                i += 1
            t *= beta
        layer.set_scaling_factor(t)
    else:
        go = True
        while go:
            l0, l0_aux = l1, l1_aux
            t *= beta
            l1, l1_aux = test_gamma(t)
            go = (
                ((not under_bound(t, l1)) or (l1 < l0 and extended_search))
                and i < max_iter
                and t > epsilon
            )
            i += 1
        t /= beta
        layer.set_scaling_factor(t)

    # select best gamma found
    min_loss = float("inf")
    best_idx = -1
    for idx, loss in enumerate(losses):
        if loss < min_loss:
            min_loss = loss
            best_idx = idx
    t = np.sqrt(gammas[best_idx])
    layer.set_scaling_factor(t)
    l0 = losses[best_idx]
    l0_aux = aux_losses[best_idx]

    if verbose:
        logger.info(
            f"Line search completed: optimal gamma = {t**2:.3e}, loss = {l0:.3e}, aux_loss = {l0_aux * 100:.2f}%"
        )

    return t**2, l0, l0_aux, gammas, losses, aux_losses
