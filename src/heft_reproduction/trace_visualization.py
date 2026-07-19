"""Visualization for source-aware trace schedules."""

from __future__ import annotations

from pathlib import Path

from .models import Schedule, processor_timelines
from .trace_model import ModeledTraceWorkflow


PROGRAM_COLORS = (
    "#2f6f9f",
    "#d97732",
    "#438a66",
    "#8a5ca8",
    "#b64c54",
    "#597d8c",
    "#9a7b31",
    "#5368a6",
)


def save_trace_gantt_chart(
    modeled: ModeledTraceWorkflow,
    schedule: Schedule,
    output_path: str | Path,
) -> Path:
    """Save a readable Gantt chart for a multi-task trace schedule."""

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required for plotting. Install with "
            "`pip install -e '.[visualization]'`."
        ) from exc

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    workflow = modeled.workflow
    timelines = processor_timelines(schedule, workflow.processors)
    programs = sorted(
        {metadata.program for metadata in modeled.task_metadata.values()}
    )
    colors = {
        program: PROGRAM_COLORS[index % len(PROGRAM_COLORS)]
        for index, program in enumerate(programs)
    }
    makespan = max((entry.finish for entry in schedule.values()), default=0.0)

    fig, ax = plt.subplots(figsize=(16, 5.8), constrained_layout=True)
    for row, processor in enumerate(workflow.processors):
        for entry in timelines[processor]:
            metadata = modeled.task_metadata[entry.task]
            ax.barh(
                row,
                entry.duration,
                left=entry.start,
                height=0.58,
                color=colors[metadata.program],
                edgecolor="#20252b",
                linewidth=0.55,
            )
            if makespan and entry.duration >= makespan * 0.025:
                ax.text(
                    entry.start + entry.duration / 2,
                    row,
                    f"T{entry.task}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=6.5,
                    fontweight="bold",
                )

    ax.set_yticks(range(len(workflow.processors)), workflow.processors)
    ax.invert_yaxis()
    ax.set_xlabel("Simulated time (seconds)")
    ax.set_ylabel("Simulated worker")
    ax.set_title("Trace-Driven HEFT Schedule: Montage 2MASS 0.05-degree")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#d7dce0", linewidth=0.7)
    ax.set_xlim(left=0)
    legend = [
        Patch(facecolor=colors[program], edgecolor="#20252b", label=program)
        for program in programs
    ]
    ax.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=min(4, len(legend)),
        frameon=False,
        fontsize=8,
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output
