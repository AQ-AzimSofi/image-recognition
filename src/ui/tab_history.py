import gradio as gr
from pathlib import Path

from ..db import Database
from ..image_utils import draw_labels_on_image
from ..models import DetectedLabel, DetectedInstance


def _load_history(label_filter, date_from, date_to, db):
    label_f = label_filter if label_filter else None
    date_f = date_from if date_from else None
    date_t = date_to if date_to else None

    summaries = db.get_history(limit=100, label_filter=label_f, date_from=date_f, date_to=date_t)

    table_data = []
    for s in summaries:
        reviewed = "Yes" if s.has_feedback else "No"
        labels_str = ", ".join(s.top_labels[:3])
        table_data.append([
            s.id,
            s.image_filename,
            s.detected_at,
            s.label_count,
            labels_str,
            reviewed,
        ])

    return table_data


def _load_detail(selection, db):
    if selection is None or len(selection) == 0:
        return None, []

    row = selection.iloc[0] if hasattr(selection, "iloc") else selection[0]
    detection_id = int(row[0]) if isinstance(row, (list, tuple)) else int(row)

    detection = db.get_detection(detection_id)
    if not detection:
        return None, []

    image_path = Path(detection.image_path)
    annotated = None
    if image_path.exists():
        labels = []
        for l in detection.labels:
            instances = []
            if l.has_bounding_box and l.bbox:
                instances.append(
                    DetectedInstance(
                        bbox_left=l.bbox["left"],
                        bbox_top=l.bbox["top"],
                        bbox_width=l.bbox["width"],
                        bbox_height=l.bbox["height"],
                        confidence=l.confidence,
                    )
                )
            labels.append(
                DetectedLabel(name=l.name, confidence=l.confidence, instances=instances)
            )
        annotated = draw_labels_on_image(image_path, labels)

    detail_data = []
    for l in detection.labels:
        status = l.feedback_status or "Not reviewed"
        detail_data.append([l.id, l.name, f"{l.confidence:.1f}%", l.has_bounding_box, status])

    return annotated, detail_data


def create_tab(db: Database) -> gr.Blocks:
    with gr.Blocks() as tab:
        gr.Markdown("## Detection History")

        with gr.Row():
            label_filter = gr.Textbox(label="Filter by label", placeholder="e.g. vacuum")
            date_from = gr.Textbox(label="From (YYYY-MM-DD)", placeholder="2026-01-01")
            date_to = gr.Textbox(label="To (YYYY-MM-DD)", placeholder="2026-12-31")
            refresh_btn = gr.Button("Refresh")

        history_table = gr.Dataframe(
            headers=["ID", "Filename", "Date", "Labels", "Top Labels", "Reviewed"],
            label="Detection History",
            interactive=False,
        )

        gr.Markdown("### Detection Detail")
        with gr.Row():
            detail_image = gr.Image(type="pil", label="Annotated Image")
            detail_table = gr.Dataframe(
                headers=["Label ID", "Name", "Confidence", "Has BBox", "Status"],
                label="Labels",
            )

        refresh_btn.click(
            fn=lambda lf, df, dt: _load_history(lf, df, dt, db),
            inputs=[label_filter, date_from, date_to],
            outputs=[history_table],
        )

        history_table.select(
            fn=lambda sel: _load_detail(sel, db),
            inputs=[history_table],
            outputs=[detail_image, detail_table],
        )

    return tab
