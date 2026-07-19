"""Visualize aggregate runtime-uncertainty experiment results."""

from __future__ import annotations

from pathlib import Path

from .uncertainty import UncertaintyExperiment


def save_uncertainty_summary_plot(
    experiment: UncertaintyExperiment,
    output_path: str | Path,
) -> Path:
    """Save makespan and fixed-plan gap summaries by uncertainty level."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required for plotting. Install with "
            "`pip install -e '.[visualization]'`."
        ) from exc

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summaries = experiment.summaries
    cvs = [summary.cv for summary in summaries]
    static_means = [summary.static_plan_mean_makespan for summary in summaries]
    static_ci = [summary.static_plan_ci95_half_width for summary in summaries]
    perfect_means = [
        summary.perfect_information_heft_mean_makespan
        for summary in summaries
    ]
    gaps = [summary.mean_static_plan_gap for summary in summaries]

    fig, (makespan_ax, gap_ax) = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
        constrained_layout=True,
    )
    makespan_ax.errorbar(
        cvs,
        static_means,
        yerr=static_ci,
        marker="o",
        capsize=4,
        color="#b64c54",
        label="Fixed static HEFT plan",
    )
    makespan_ax.plot(
        cvs,
        perfect_means,
        marker="o",
        color="#2f6f9f",
        label="Perfect-information HEFT reference",
    )
    makespan_ax.axhline(
        experiment.planned_makespan,
        color="#48555f",
        linestyle="--",
        linewidth=1.1,
        label="Estimated-plan makespan",
    )
    makespan_ax.set_title("Mean realized makespan")
    makespan_ax.set_xlabel("Runtime uncertainty (CV)")
    makespan_ax.set_ylabel("Simulated seconds")
    makespan_ax.grid(axis="y", color="#d7dce0", linewidth=0.7)
    makespan_ax.legend(fontsize=8)

    gap_ax.plot(cvs, gaps, marker="o", color="#d97732")
    gap_ax.axhline(0.0, color="#48555f", linewidth=1.0)
    gap_ax.set_title("Fixed-plan gap to perfect-information HEFT")
    gap_ax.set_xlabel("Runtime uncertainty (CV)")
    gap_ax.set_ylabel("Mean simulated seconds")
    gap_ax.grid(axis="y", color="#d7dce0", linewidth=0.7)

    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output
