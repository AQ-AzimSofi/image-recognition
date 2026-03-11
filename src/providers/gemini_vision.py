from __future__ import annotations

import json
import os
from pathlib import Path

from .base import DetectionBox, Provider, ProviderResult, detect_box_scale

COST_PER_IMAGE_PRO = 0.002
COST_PER_IMAGE_FLASH = 0.00006

DETECTION_PROMPT = """You are an expert in object and product identification.
Analyze this image and identify ALL visible objects, products, brands, and items.

For each detected object, return a JSON array with this exact format:
[
  {
    "product_name": "specific product name",
    "brand": "brand/manufacturer name",
    "label": "concise label for display",
    "confidence": 0.95,
    "box_2d": [ymin, xmin, ymax, xmax]
  }
]

IMPORTANT:
- Coordinates must be on a 0-1000 scale (normalized).
- Be specific: instead of "Bottle", say the actual product name.
- Include any text you can read on labels.
- confidence should be between 0.0 and 1.0.
- Return ONLY the JSON array, no other text."""

CLASSIFY_PROMPT = """You are an expert in object and product identification. Identify the item in this cropped image.
Return a JSON object:
{
  "product_name": "specific product name",
  "brand": "brand/manufacturer",
  "label": "concise display label",
  "confidence": 0.95
}
Return ONLY the JSON object, no other text."""


class GeminiVisionProvider(Provider):
    name = "Gemini 3.1 Pro"

    def __init__(self, model: str = "gemini-2.5-flash-preview-04-17"):
        self.model = model
        self._client = None

        if "flash" in model.lower():
            self.name = "Gemini Flash"
            self._cost = COST_PER_IMAGE_FLASH
        else:
            self._cost = COST_PER_IMAGE_PRO

    @property
    def client(self):
        if self._client is None:
            import google.generativeai as genai
            genai.configure(api_key=os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY"))
            self._client = genai.GenerativeModel(self.model)
        return self._client

    def detect(self, image_path: str | Path) -> ProviderResult:
        from PIL import Image

        try:
            img = Image.open(image_path)
            response = self.client.generate_content(
                [DETECTION_PROMPT, img],
                generation_config={"temperature": 0.1},
            )
        except Exception as e:
            return ProviderResult(
                provider_name=self.name,
                error=str(e),
            )

        text = response.text
        boxes = self._parse_response(text)

        return ProviderResult(
            provider_name=self.name,
            boxes=boxes,
            cost_estimate=self._cost,
            raw_response={"text": text},
        )

    def classify_crop(self, crop_image) -> dict:
        try:
            response = self.client.generate_content(
                [CLASSIFY_PROMPT, crop_image],
                generation_config={"temperature": 0.1},
            )
            text = response.text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1])
            return json.loads(text)
        except Exception:
            return {"product_name": "Unknown", "brand": "Unknown", "label": "Unknown", "confidence": 0.0}

    def _parse_response(self, text: str) -> list[DetectionBox]:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])

        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                try:
                    items = json.loads(text[start:end])
                except json.JSONDecodeError:
                    return []
            else:
                return []

        scale = detect_box_scale(items)

        boxes = []
        for item in items:
            box_2d = item.get("box_2d", [0, 0, 0, 0])
            if len(box_2d) != 4:
                continue

            ymin, xmin, ymax, xmax = box_2d
            boxes.append(
                DetectionBox(
                    label=item.get("label", item.get("product_name", "Unknown")),
                    confidence=item.get("confidence", 0.5) * 100,
                    x_min=xmin / scale,
                    y_min=ymin / scale,
                    x_max=xmax / scale,
                    y_max=ymax / scale,
                    brand=item.get("brand"),
                    product_name=item.get("product_name"),
                )
            )
        return boxes
