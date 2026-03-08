import io
import colorsys
from pathlib import Path

import numpy as np
import pillow_heif
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

pillow_heif.register_heif_opener()

from .models import DetectedLabel, DetectedInstance


def generate_color_map(label_names: list[str]) -> dict[str, tuple[int, int, int]]:
    colors = {}
    n = max(len(label_names), 1)
    for i, name in enumerate(label_names):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors[name] = (int(r * 255), int(g * 255), int(b * 255))
    return colors


def draw_labels_on_image(
    image_path: Path | str,
    labels: list[DetectedLabel],
    color_map: dict[str, tuple[int, int, int]] | None = None,
) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    bbox_labels = [l for l in labels if l.instances]
    if not bbox_labels:
        return img

    all_names = [l.name for l in bbox_labels]
    if color_map is None:
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
            text_bg_y2 = y1
            draw.rectangle([x1, text_bg_y1, x1 + text_w + 8, text_bg_y2], fill=color)
            draw.text((x1 + 4, text_bg_y1 + 1), text, fill=(255, 255, 255), font=font)

    return img


def crop_bounding_box(
    image_path: Path | str,
    instance: DetectedInstance,
    padding: float = 0.05,
) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    x1 = max(0, int((instance.bbox_left - padding) * width))
    y1 = max(0, int((instance.bbox_top - padding) * height))
    x2 = min(width, int((instance.bbox_left + instance.bbox_width + padding) * width))
    y2 = min(height, int((instance.bbox_top + instance.bbox_height + padding) * height))

    return img.crop((x1, y1, x2, y2))


def get_image_dimensions(image_path: Path | str) -> tuple[int, int]:
    with Image.open(image_path) as img:
        return img.size


def apply_degradation(
    image: Image.Image,
    degradation_type: str,
    level: float,
) -> Image.Image:
    level = max(0.0, min(1.0, level))
    degradations = {
        "blur": _blur,
        "darken": _darken,
        "brighten": _brighten,
        "noise": _add_noise,
        "crop": _crop_center,
        "jpeg_compress": _jpeg_compress,
        "resize_down": _resize_down,
    }
    func = degradations.get(degradation_type)
    if func is None:
        raise ValueError(
            f"Unknown degradation type: {degradation_type}. "
            f"Available: {list(degradations.keys())}"
        )
    return func(image.copy(), level)


DEGRADATION_TYPES = ["blur", "darken", "brighten", "noise", "crop", "jpeg_compress", "resize_down"]


def _blur(image: Image.Image, level: float) -> Image.Image:
    radius = level * 20
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def _darken(image: Image.Image, level: float) -> Image.Image:
    factor = 1.0 - (level * 0.95)
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def _brighten(image: Image.Image, level: float) -> Image.Image:
    factor = 1.0 + (level * 3.0)
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def _add_noise(image: Image.Image, level: float) -> Image.Image:
    arr = np.array(image, dtype=np.float32)
    noise_std = level * 100
    noise = np.random.normal(0, noise_std, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _crop_center(image: Image.Image, level: float) -> Image.Image:
    width, height = image.size
    crop_ratio = max(0.1, 1.0 - level * 0.9)
    new_w = int(width * crop_ratio)
    new_h = int(height * crop_ratio)
    left = (width - new_w) // 2
    top = (height - new_h) // 2
    cropped = image.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((width, height), Image.LANCZOS)


def _jpeg_compress(image: Image.Image, level: float) -> Image.Image:
    quality = max(1, int(95 - level * 94))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _resize_down(image: Image.Image, level: float) -> Image.Image:
    width, height = image.size
    scale = max(0.05, 1.0 - level * 0.95)
    small_w = max(1, int(width * scale))
    small_h = max(1, int(height * scale))
    small = image.resize((small_w, small_h), Image.LANCZOS)
    return small.resize((width, height), Image.NEAREST)


def image_to_bytes(image: Image.Image, format: str = "JPEG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()
