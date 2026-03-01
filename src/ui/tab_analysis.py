import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..db import Database


def _load_analysis(db):
    stats = db.get_analysis_stats()

    summary = (
        f"**Total Detections:** {stats.total_detections}  \n"
        f"**Total Labels:** {stats.total_labels}  \n"
        f"**Total Reviewed:** {stats.total_reviewed}  \n"
        f"**Accuracy Rate:** {stats.accuracy_rate:.1%}  \n"
        f"**Wrong Reason Cases:** {stats.wrong_reason_count}"
    )

    conf_fig = _plot_confidence_distribution(stats.confidence_distribution)
    misclass_data = [
        [m["detected"], m["expected"], m["count"]]
        for m in stats.common_misclassifications
    ]

    return summary, conf_fig, misclass_data


def _plot_confidence_distribution(distribution):
    fig, ax = plt.subplots(figsize=(8, 4))

    if not distribution:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center", fontsize=14)
        ax.set_title("Confidence Distribution")
        plt.tight_layout()
        return fig

    buckets = [d["bucket"] for d in distribution]
    totals = [d["total"] for d in distribution]
    correct = [d["correct"] or 0 for d in distribution]
    incorrect = [d["incorrect"] or 0 for d in distribution]

    x = range(len(buckets))
    bar_width = 0.35

    ax.bar([i - bar_width / 2 for i in x], correct, bar_width, label="Correct", color="#4CAF50")
    ax.bar([i + bar_width / 2 for i in x], incorrect, bar_width, label="Incorrect", color="#F44336")

    ax.set_xlabel("Confidence Range (%)")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution: Correct vs Incorrect")
    ax.set_xticks(list(x))
    ax.set_xticklabels(buckets, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    return fig


def create_tab(db: Database) -> gr.Blocks:
    with gr.Blocks() as tab:
        gr.Markdown("## Analysis Dashboard")
        refresh_btn = gr.Button("Refresh")

        summary_md = gr.Markdown("Click Refresh to load analysis data")

        with gr.Row():
            with gr.Column(scale=1):
                conf_plot = gr.Plot(label="Confidence Distribution")
            with gr.Column(scale=1):
                misclass_table = gr.Dataframe(
                    headers=["Detected Label", "Expected Label", "Count"],
                    label="Common Misclassifications",
                )

        refresh_btn.click(
            fn=lambda: _load_analysis(db),
            outputs=[summary_md, conf_plot, misclass_table],
        )

    return tab
