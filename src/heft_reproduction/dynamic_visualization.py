"""Policy-comparison visualization for dynamic workflow experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .dynamic_models import DynamicSimulationResult


POLICY_LABELS = {
    "online-greedy-eft": "Online Greedy-EFT",
    "per-workflow-static-heft": "Per-Workflow Static HEFT",
    "rolling-heft": "Rolling HEFT",
    "aging-rolling-heft": "Aging Rolling HEFT",
    "shortest-remaining-work": "Shortest Remaining Work",
}
POLICY_COLORS = {
    "online-greedy-eft": "#2f6f9f",
    "per-workflow-static-heft": "#b64c54",
    "rolling-heft": "#438a66",
    "aging-rolling-heft": "#8a5ca8",
    "shortest-remaining-work": "#9a7b31",
}


def save_dynamic_comparison_plot(
    results: Sequence[DynamicSimulationResult],
    output_path: str | Path,
) -> Path:
    """Save workflow-level JCT curves and aggregate policy comparisons."""

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required for plotting. Install with "
            "`pip install -e '.[visualization]'`."
        ) from exc

    if not results:
        raise ValueError("at least one dynamic result is required")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, (jct_ax, summary_ax) = plt.subplots(
        1,
        2,
        figsize=(13.5, 4.8),
        constrained_layout=True,
    )
    for result in results:
        indices = range(1, len(result.workflows) + 1)
        jcts = [workflow.jct for workflow in result.workflows]
        jct_ax.plot(
            indices,
            jcts,
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            color=POLICY_COLORS[result.policy],
            label=POLICY_LABELS[result.policy],
        )
    jct_ax.set_title("Workflow JCT by arrival order")
    jct_ax.set_xlabel("Workflow arrival order")
    jct_ax.set_ylabel("Job completion time (simulated seconds)")
    jct_ax.grid(axis="y", color="#d7dce0", linewidth=0.7)
    jct_ax.legend(fontsize=8)

    x = np.arange(len(results))
    width = 0.34
    means = [result.metrics.mean_jct for result in results]
    p95s = [result.metrics.p95_jct for result in results]
    mean_bars = summary_ax.bar(
        x - width / 2,
        means,
        width,
        color="#597d8c",
        label="Mean JCT",
    )
    p95_bars = summary_ax.bar(
        x + width / 2,
        p95s,
        width,
        color="#d97732",
        label="P95 JCT",
    )
    summary_ax.set_xticks(
        x,
        [
            POLICY_LABELS[result.policy].replace(" ", "\n", 1)
            for result in results
        ],
        fontsize=8,
    )
    summary_ax.set_title("Dynamic scheduling policy comparison")
    summary_ax.set_ylabel("Simulated seconds")
    summary_ax.grid(axis="y", color="#d7dce0", linewidth=0.7)
    summary_ax.legend(fontsize=8)
    summary_ax.bar_label(mean_bars, fmt="%.1f", padding=3, fontsize=7.5)
    summary_ax.bar_label(p95_bars, fmt="%.1f", padding=3, fontsize=7.5)

    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output
