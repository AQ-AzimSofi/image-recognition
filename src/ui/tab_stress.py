import gradio as gr
from PIL import Image

from ..db import Database
from ..image_utils import (
    apply_degradation,
    draw_labels_on_image,
    image_to_bytes,
    get_image_dimensions,
    DEGRADATION_TYPES,
)
from ..rekognition import RekognitionClient, RekognitionError


def _run_stress_test(image_path, degradation_type, level, min_confidence, client, db):
    if image_path is None:
        return None, None, [], "Upload an image first"

    try:
        original_result = client.detect_labels_from_path(image_path, min_confidence)
    except RekognitionError as e:
        return None, None, [], f"Error: {e}"

    original_annotated = draw_labels_on_image(image_path, original_result.labels)

    img = Image.open(image_path).convert("RGB")
    degraded = apply_degradation(img, degradation_type, level)
    degraded_bytes = image_to_bytes(degraded)

    try:
        degraded_result = client.detect_labels(degraded_bytes, min_confidence)
    except RekognitionError as e:
        return original_annotated, None, [], f"Degraded image error: {e}"

    degraded_annotated = draw_labels_on_image_from_pil(degraded, degraded_result.labels)

    original_names = {l.name for l in original_result.labels}
    degraded_names = {l.name for l in degraded_result.labels}

    comparison = []
    for name in sorted(original_names | degraded_names):
        orig_conf = next((l.confidence for l in original_result.labels if l.name == name), 0)
        deg_conf = next((l.confidence for l in degraded_result.labels if l.name == name), 0)

        if name in original_names and name not in degraded_names:
            status = "LOST"
        elif name not in original_names and name in degraded_names:
            status = "GAINED"
        else:
            status = "KEPT"

        delta = deg_conf - orig_conf
        comparison.append([name, f"{orig_conf:.1f}%", f"{deg_conf:.1f}%", f"{delta:+.1f}%", status])

    lost = len(original_names - degraded_names)
    gained = len(degraded_names - original_names)
    kept = len(original_names & degraded_names)
    info = f"**Results:** {kept} kept, {lost} lost, {gained} gained"

    width, height = get_image_dimensions(image_path)
    original_labels_data = _result_to_labels_data(original_result)
    original_id = db.save_detection(
        image_filename=image_path.split("/")[-1] if "/" in image_path else image_path.split("\\")[-1],
        image_path=image_path,
        image_width=width,
        image_height=height,
        labels_data=original_labels_data,
        raw_response=original_result.raw_response,
        min_confidence=min_confidence,
        max_labels=20,
    )

    degraded_labels_data = _result_to_labels_data(degraded_result)
    degraded_id = db.save_detection(
        image_filename=f"stress_{degradation_type}_{level}",
        image_path="in-memory",
        image_width=width,
        image_height=height,
        labels_data=degraded_labels_data,
        raw_response=degraded_result.raw_response,
        min_confidence=min_confidence,
        max_labels=20,
    )

    label_diff = {
        "labels_lost": sorted(original_names - degraded_names),
        "labels_gained": sorted(degraded_names - original_names),
        "labels_kept": sorted(original_names & degraded_names),
    }
    db.save_stress_test(original_id, degradation_type, level, degraded_id, label_diff)

    return original_annotated, degraded_annotated, comparison, info


def draw_labels_on_image_from_pil(pil_image, labels):
    from ..image_utils import generate_color_map
    from PIL import ImageDraw, ImageFont

    img = pil_image.copy()
    draw = ImageDraw.Draw(img)
    width, height = img.size

    bbox_labels = [l for l in labels if l.instances]
    if not bbox_labels:
        return img

    all_names = [l.name for l in bbox_labels]
    color_map = generate_color_map(all_names)

    try:
        font = ImageFont.truetype("arial.ttf", max(14, height // 40))
    except (OSError, IOError):
        font = ImageFont.load_default()

    for label in bbox_labels:
        color = color_map.get(label.name, (255, 0, 0))
        for inst in label.instances:
            x1 = int(inst.bbox_left * width)
            y1 = int(inst.bbox_top * height)
            x2 = int((inst.bbox_left + inst.bbox_width) * width)
            y2 = int((inst.bbox_top + inst.bbox_height) * height)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            text = f"{label.name} ({inst.confidence:.1f}%)"
            bbox = font.getbbox(text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            text_bg_y1 = max(0, y1 - text_h - 6)
            draw.rectangle([x1, text_bg_y1, x1 + text_w + 8, y1], fill=color)
            draw.text((x1 + 4, text_bg_y1 + 1), text, fill=(255, 255, 255), font=font)

    return img


def _result_to_labels_data(result):
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
    return labels_data


def create_tab(client: RekognitionClient, db: Database) -> gr.Blocks:
    with gr.Blocks() as tab:
        gr.Markdown("## Stress Test")
        gr.Markdown("Test how image degradation affects detection accuracy. Each test costs ~$0.002 (2 API calls).")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="filepath", label="Upload Image")
                deg_type = gr.Dropdown(
                    choices=DEGRADATION_TYPES,
                    value="blur",
                    label="Degradation Type",
                )
                level_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="Severity Level")
                min_conf = gr.Slider(0, 100, value=50, step=1, label="Min Confidence (%)")
                run_btn = gr.Button("Run Stress Test", variant="primary")

        info_md = gr.Markdown("")

        with gr.Row():
            original_image = gr.Image(type="pil", label="Original")
            degraded_image = gr.Image(type="pil", label="Degraded")

        comparison_table = gr.Dataframe(
            headers=["Label", "Original Conf", "Degraded Conf", "Delta", "Status"],
            label="Label Comparison",
        )

        run_btn.click(
            fn=lambda img, dt, lv, mc: _run_stress_test(img, dt, lv, mc, client, db),
            inputs=[image_input, deg_type, level_slider, min_conf],
            outputs=[original_image, degraded_image, comparison_table, info_md],
        )

    return tab
