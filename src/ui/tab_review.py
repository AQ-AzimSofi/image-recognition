import gradio as gr
from pathlib import Path

from ..db import Database
from ..image_utils import draw_labels_on_image, crop_bounding_box
from ..models import DetectedLabel, DetectedInstance, FeedbackEntry


def _load_detection(detection_id, db):
    if not detection_id:
        return None, [], gr.update(choices=[]), "No detection loaded"

    detection = db.get_detection(int(detection_id))
    if not detection:
        return None, [], gr.update(choices=[]), "Detection not found"

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

    label_choices = [f"{l.id}: {l.name} ({l.confidence:.1f}%)" for l in detection.labels]
    table_data = []
    for l in detection.labels:
        status = l.feedback_status or "Not reviewed"
        table_data.append([l.id, l.name, f"{l.confidence:.1f}%", l.has_bounding_box, status])

    info = f"Detection #{detection.id} - {detection.image_filename} - {detection.label_count} labels"

    return annotated, table_data, gr.update(choices=label_choices, value=label_choices[0] if label_choices else None), info


def _load_label_crop(label_selection, detection_id, db):
    if not label_selection or not detection_id:
        return None

    label_id = int(label_selection.split(":")[0])
    detection = db.get_detection(int(detection_id))
    if not detection:
        return None

    image_path = Path(detection.image_path)
    if not image_path.exists():
        return None

    for l in detection.labels:
        if l.id == label_id and l.has_bounding_box and l.bbox:
            inst = DetectedInstance(
                bbox_left=l.bbox["left"],
                bbox_top=l.bbox["top"],
                bbox_width=l.bbox["width"],
                bbox_height=l.bbox["height"],
                confidence=l.confidence,
            )
            return crop_bounding_box(image_path, inst)

    return None


def _find_next_unreviewed(db):
    summaries = db.get_history(limit=200)
    for s in summaries:
        if not s.has_feedback:
            return str(s.id)
    return ""


def _submit_feedback(detection_id, label_selection, correctness, wrong_reason, expected_label, notes, db):
    if not detection_id or not label_selection:
        return "Please select a detection and label first"

    label_id = int(label_selection.split(":")[0])

    is_correct = None
    if correctness == "Correct":
        is_correct = True
    elif correctness == "Incorrect":
        is_correct = False

    entry = FeedbackEntry(
        label_id=label_id,
        detection_id=int(detection_id),
        is_correct=is_correct,
        is_wrong_reason=wrong_reason,
        expected_label=expected_label if expected_label else None,
        reviewer_notes=notes if notes else None,
    )
    db.save_feedback(entry)
    return f"Feedback saved for label {label_id}"


def create_tab(db: Database) -> gr.Blocks:
    with gr.Blocks() as tab:
        gr.Markdown("## Review & Feedback (Kaizen)")

        with gr.Row():
            detection_id_input = gr.Textbox(label="Detection ID", placeholder="Enter detection ID")
            load_btn = gr.Button("Load")
            next_btn = gr.Button("Next Unreviewed")

        info_text = gr.Markdown("")

        with gr.Row():
            with gr.Column(scale=1):
                annotated_image = gr.Image(type="pil", label="Detection (with BoundingBoxes)")

            with gr.Column(scale=1):
                labels_table = gr.Dataframe(
                    headers=["Label ID", "Name", "Confidence", "Has BBox", "Status"],
                    label="Detected Labels",
                )

        gr.Markdown("### Label Review")
        with gr.Row():
            with gr.Column(scale=1):
                label_selector = gr.Dropdown(label="Select Label to Review", choices=[])
                cropped_image = gr.Image(type="pil", label="Cropped BBox Region")

            with gr.Column(scale=1):
                correctness = gr.Radio(
                    choices=["Correct", "Incorrect", "Not Sure"],
                    label="Is this label correct?",
                )
                wrong_reason = gr.Checkbox(
                    label="Right Answer, Wrong Reason (BBox covers wrong object)",
                )
                expected_label = gr.Textbox(
                    label="Expected Label (if incorrect)",
                    placeholder="What should the label be?",
                )
                notes = gr.Textbox(label="Reviewer Notes", lines=2)
                submit_btn = gr.Button("Submit Feedback", variant="primary")
                feedback_status = gr.Markdown("")

        load_btn.click(
            fn=lambda did: _load_detection(did, db),
            inputs=[detection_id_input],
            outputs=[annotated_image, labels_table, label_selector, info_text],
        )

        next_btn.click(
            fn=lambda: _find_next_unreviewed(db),
            outputs=[detection_id_input],
        ).then(
            fn=lambda did: _load_detection(did, db),
            inputs=[detection_id_input],
            outputs=[annotated_image, labels_table, label_selector, info_text],
        )

        label_selector.change(
            fn=lambda sel, did: _load_label_crop(sel, did, db),
            inputs=[label_selector, detection_id_input],
            outputs=[cropped_image],
        )

        submit_btn.click(
            fn=lambda did, sel, cor, wr, exp, n: _submit_feedback(did, sel, cor, wr, exp, n, db),
            inputs=[detection_id_input, label_selector, correctness, wrong_reason, expected_label, notes],
            outputs=[feedback_status],
        ).then(
            fn=lambda did: _load_detection(did, db),
            inputs=[detection_id_input],
            outputs=[annotated_image, labels_table, label_selector, info_text],
        )

    return tab
