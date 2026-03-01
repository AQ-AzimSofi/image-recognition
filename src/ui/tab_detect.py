import gradio as gr
from PIL import Image

from ..db import Database
from ..image_utils import draw_labels_on_image, get_image_dimensions
from ..rekognition import RekognitionClient, RekognitionError


def _run_detection(image_path, min_confidence, max_labels, client, db):
    if image_path is None:
        return None, [], "{}"

    try:
        result = client.detect_labels_from_path(image_path, min_confidence, max_labels)
    except RekognitionError as e:
        return None, [[str(e), "", "", ""]], "{}"

    width, height = get_image_dimensions(image_path)
    result.image_width = width
    result.image_height = height

    labels_data = []
    for label in result.labels:
        if label.instances:
            for inst in label.instances:
                labels_data.append({
                    "name": label.name,
                    "confidence": label.confidence,
                    "category": ", ".join(label.categories),
                    "parents": ", ".join(label.parents),
                    "has_bounding_box": 1,
                    "bbox_left": inst.bbox_left,
                    "bbox_top": inst.bbox_top,
                    "bbox_width": inst.bbox_width,
                    "bbox_height": inst.bbox_height,
                    "instance_confidence": inst.confidence,
                })
        else:
            labels_data.append({
                "name": label.name,
                "confidence": label.confidence,
                "category": ", ".join(label.categories),
                "parents": ", ".join(label.parents),
                "has_bounding_box": 0,
            })

    detection_id = db.save_detection(
        image_filename=image_path.split("/")[-1] if "/" in image_path else image_path.split("\\")[-1],
        image_path=image_path,
        image_width=width,
        image_height=height,
        labels_data=labels_data,
        raw_response=result.raw_response,
        min_confidence=min_confidence,
        max_labels=int(max_labels),
    )

    annotated = draw_labels_on_image(image_path, result.labels)

    table_data = []
    for label in result.labels:
        has_bbox = "Yes" if label.instances else "No"
        bbox_info = ""
        if label.instances:
            inst = label.instances[0]
            bbox_info = f"({inst.bbox_left:.2f}, {inst.bbox_top:.2f}, {inst.bbox_width:.2f}, {inst.bbox_height:.2f})"
        table_data.append([
            label.name,
            f"{label.confidence:.1f}%",
            has_bbox,
            bbox_info,
        ])

    import json
    raw_json = json.dumps(result.raw_response, indent=2, default=str)

    return annotated, table_data, raw_json


def create_tab(client: RekognitionClient, db: Database) -> gr.Blocks:
    with gr.Blocks() as tab:
        gr.Markdown("## Object Detection")
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="filepath", label="Upload Image")
                min_conf = gr.Slider(0, 100, value=50, step=1, label="Min Confidence (%)")
                max_lab = gr.Slider(1, 50, value=20, step=1, label="Max Labels")
                detect_btn = gr.Button("Detect", variant="primary")

            with gr.Column(scale=1):
                annotated_output = gr.Image(type="pil", label="Annotated Image")
                results_table = gr.Dataframe(
                    headers=["Label", "Confidence", "Has BBox", "BBox Region"],
                    label="Detection Results",
                )

        with gr.Accordion("Raw JSON Response", open=False):
            raw_json = gr.Code(language="json", label="Rekognition Response")

        detect_btn.click(
            fn=lambda img, mc, ml: _run_detection(img, mc, ml, client, db),
            inputs=[image_input, min_conf, max_lab],
            outputs=[annotated_output, results_table, raw_json],
        )

    return tab
