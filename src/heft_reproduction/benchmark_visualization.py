"""Visualization for Phase 4B dynamic scheduling benchmark sweeps."""

from __future__ import annotations

from pathlib import Path

from .dynamic_benchmark import DynamicBenchmarkResult
from .dynamic_visualization import POLICY_COLORS, POLICY_LABELS


def save_benchmark_plot(
    benchmark: DynamicBenchmarkResult,
    output_path: str | Path,
) -> Path:
    """Plot mean JCT against offered load for each size and runtime CV."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required for plotting. Install with "
            "`pip install -e '.[visualization]'`."
        ) from exc

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(benchmark.sizes),
        len(benchmark.cvs),
        figsize=(5.2 * len(benchmark.cvs), 4.2 * len(benchmark.sizes)),
        squeeze=False,
        constrained_layout=True,
    )
    aggregate_index = {
        (
            aggregate.size,
            aggregate.runtime_cv,
            aggregate.offered_load,
            aggregate.policy,
        ): aggregate
        for aggregate in benchmark.aggregates
    }

    for row, size in enumerate(benchmark.sizes):
        for column, cv in enumerate(benchmark.cvs):
            axis = axes[row][column]
            for policy in benchmark.policies:
                means = []
                intervals = []
                for load in benchmark.loads:
                    aggregate = aggregate_index[(size, cv, load, policy)]
                    statistic = aggregate.statistics["mean_jct"]
                    means.append(statistic.mean)
                    intervals.append(statistic.ci95_half_width)
                axis.errorbar(
                    benchmark.loads,
                    means,
                    yerr=intervals,
                    marker="o",
                    capsize=3,
                    linewidth=1.5,
                    color=POLICY_COLORS[policy],
                    label=POLICY_LABELS[policy],
                )
            axis.set_title(f"{size.capitalize()} workflows | CV={cv:g}")
            axis.set_xlabel("Normalized offered load")
            axis.set_ylabel("Mean JCT (simulated seconds)")
            axis.grid(axis="y", color="#d7dce0", linewidth=0.7)
            if row == 0 and column == len(benchmark.cvs) - 1:
                axis.legend(fontsize=7.5)

    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output
