from __future__ import annotations

import base64
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


@dataclass
class DetectionBox:
    label: str
    confidence: float
    x_min: float  # normalized 0-1
    y_min: float  # normalized 0-1
    x_max: float  # normalized 0-1
    y_max: float  # normalized 0-1
    brand: str | None = None
    product_name: str | None = None

    def to_pixel_coords(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (
            int(self.x_min * width),
            int(self.y_min * height),
            int(self.x_max * width),
            int(self.y_max * height),
        )


@dataclass
class ProviderResult:
    provider_name: str
    boxes: list[DetectionBox] = field(default_factory=list)
    latency_ms: float = 0.0
    cost_estimate: float = 0.0
    raw_response: dict | None = None
    error: str | None = None


class Provider(ABC):
    name: str

    @abstractmethod
    def detect(self, image_path: str | Path) -> ProviderResult:
        ...

    def _timed_detect(self, image_path: str | Path) -> ProviderResult:
        start = time.perf_counter()
        result = self.detect(image_path)
        elapsed = (time.perf_counter() - start) * 1000
        result.latency_ms = elapsed
        return result


def _needs_conversion(image_path: str | Path) -> bool:
    return Path(image_path).suffix.lower() not in SUPPORTED_FORMATS


def _convert_to_jpeg_bytes(image_path: str | Path) -> bytes:
    import io
    img = Image.open(image_path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def encode_image_base64(image_path: str | Path) -> str:
    if _needs_conversion(image_path):
        return base64.b64encode(_convert_to_jpeg_bytes(image_path)).decode("utf-8")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path: str | Path) -> str:
    suffix = Path(image_path).suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        return "image/jpeg"
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return media_types.get(suffix, "image/jpeg")


def load_image_bytes(image_path: str | Path) -> bytes:
    if _needs_conversion(image_path):
        return _convert_to_jpeg_bytes(image_path)
    with open(image_path, "rb") as f:
        return f.read()


def detect_box_scale(items: list[dict]) -> float:
    """Detect whether LLM returned box_2d in 0-1 or 0-1000 scale.

    Some models ignore the "0-1000 scale" instruction and return 0-1 floats.
    If the max coordinate across all items is <= 1.0, assume 0-1 scale.
    """
    all_coords = []
    for item in items:
        box_2d = item.get("box_2d", [])
        if len(box_2d) == 4:
            all_coords.extend(box_2d)
    if all_coords and max(all_coords) <= 1.0:
        return 1.0
    return 1000.0
