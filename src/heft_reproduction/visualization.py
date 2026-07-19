"""Optional schedule visualization."""

from __future__ import annotations

from pathlib import Path

from .models import Schedule, Workflow, processor_timelines


def save_gantt_chart(
    workflow: Workflow,
    schedule: Schedule,
    output_path: str | Path,
) -> Path:
    """Save a compact processor-timeline chart using Matplotlib."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required for plotting. Install with "
            "`pip install -e '.[visualization]'`."
        ) from exc

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    timelines = processor_timelines(schedule, workflow.processors)
    colors = ("#2f6f9f", "#d97732", "#438a66", "#8a5ca8", "#b64c54")

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    for row, processor in enumerate(workflow.processors):
        for index, entry in enumerate(timelines[processor]):
            ax.barh(
                row,
                entry.duration,
                left=entry.start,
                height=0.55,
                color=colors[index % len(colors)],
                edgecolor="#20252b",
                linewidth=0.8,
            )
            ax.text(
                entry.start + entry.duration / 2,
                row,
                f"T{entry.task}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_yticks(range(len(workflow.processors)), workflow.processors)
    ax.invert_yaxis()
    ax.set_xlabel("Time")
    ax.set_ylabel("Processor")
    ax.set_title("HEFT Paper Example Schedule")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#d7dce0", linewidth=0.7)
    ax.set_xlim(left=0)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output
