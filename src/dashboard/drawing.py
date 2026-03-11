from __future__ import annotations

import colorsys

from PIL import Image, ImageDraw, ImageFont

from ..providers.base import DetectionBox


CATEGORY_COLORS = {
    "Baseline": (220, 53, 69),
    "Single-Step LLM": (13, 110, 253),
    "Crop & Classify": (25, 135, 84),
    "Identify & Ground": (255, 193, 7),
}


def generate_box_colors(boxes: list[DetectionBox]) -> list[tuple[int, int, int]]:
    n = max(len(boxes), 1)
    colors = []
    for i in range(len(boxes)):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.9)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def draw_detections(
    image_path: str,
    boxes: list[DetectionBox],
    title: str = "",
) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    boxes_with_coords = [b for b in boxes if b.x_max > 0]
    if not boxes_with_coords:
        return img

    colors = generate_box_colors(boxes_with_coords)

    try:
        font_size = max(12, min(h // 35, 24))
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("arial.ttf", max(12, h // 35))
        except (OSError, IOError):
            font = ImageFont.load_default()

    for i, box in enumerate(boxes_with_coords):
        color = colors[i]
        x1, y1, x2, y2 = box.to_pixel_coords(w, h)

        for offset in range(3):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=color,
            )

        label_parts = [box.label]
        if box.brand and box.brand != box.label:
            label_parts.insert(0, box.brand)
        if box.product_name and box.product_name != box.label:
            label_parts.append(f"({box.product_name})")
        text = " | ".join(label_parts[:2])
        text += f" {box.confidence:.0f}%"

        bbox = font.getbbox(text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        padding = 4

        bg_y1 = max(0, y1 - text_h - padding * 2)
        bg_y2 = y1
        bg_x2 = min(w, x1 + text_w + padding * 2)

        draw.rectangle([x1, bg_y1, bg_x2, bg_y2], fill=color)
        draw.text(
            (x1 + padding, bg_y1 + padding // 2),
            text,
            fill=(255, 255, 255),
            font=font,
        )

    return img
