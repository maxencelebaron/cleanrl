from typing import Any
import torch
import torch.nn as nn


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
