import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _load(run_dir: str, tag: str) -> dict[int, float]:
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return {}
    return {e.step: e.value for e in ea.Scalars(tag)}


def _final(run_dir: str, tag: str) -> float | None:
    series = _load(run_dir, tag)
    return series[max(series)] if series else None


def scatter_within_runs(run_dir: str, output_path: str):
    """
    One point per monitoring checkpoint (charts/feature_rank vs charts/plasticity_loss).
    """
    ranks = _load(run_dir, "features/rank")
    losses = _load(run_dir, "charts/plasticity_loss")
    common_steps = sorted(set(ranks) & set(losses))
    if not common_steps:
        print(f"scatter_within_runs: no data in {run_dir}")
        return
    x = np.array([ranks[s] for s in common_steps])
    y = np.array([losses[s] for s in common_steps])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, zorder=3)
    if len(x) >= 2:
        coeffs = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_line, np.polyval(coeffs, x_line), color="red", linestyle="--",
                label=f"fit: y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}")
        ax.legend()
    ax.set_xlabel("Feature rank")
    ax.set_ylabel("Plasticity loss  P(θ_t) - P(θ_0)")
    ax.set_title("Plasticity loss vs Feature rank")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def scatter_cross_runs(run_dirs: list[str], output_path: str):
    """One point per run (charts/final_feature_rank vs charts/final_plasticity_loss)."""
    x, y = [], []
    for run_dir in run_dirs:
        rank = _final(run_dir, "features/rank")
        loss = _final(run_dir, "charts/plasticity_loss")
        if rank is None or loss is None:
            print(f"scatter_cross_runs: skip {run_dir} — missing final metrics")
            continue
        x.append(rank)
        y.append(loss)
    if not x:
        print("scatter_cross_runs: no valid runs")
        return
    x, y = np.array(x), np.array(y)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, zorder=3)
    if len(x) >= 2:
        coeffs = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_line, np.polyval(coeffs, x_line), color="red",
                label=f"fit: y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}")
        ax.legend()
    ax.set_xlabel("Final feature rank")
    ax.set_ylabel("Final plasticity loss  P(θ_T) - P(θ_0)")
    ax.set_title("Plasticity loss vs Feature rank")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
