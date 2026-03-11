from __future__ import annotations

import json
import os
from pathlib import Path

from .base import (
    DetectionBox,
    Provider,
    ProviderResult,
    detect_box_scale,
    encode_image_base64,
    get_image_media_type,
)

COST_PER_IMAGE_SONNET = 0.003

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


class ClaudeVisionProvider(Provider):
    name = "Claude 4.6 Sonnet"

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY")
            )
        return self._client

    def detect(self, image_path: str | Path) -> ProviderResult:
        image_b64 = encode_image_base64(image_path)
        media_type = get_image_media_type(image_path)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": DETECTION_PROMPT,
                            },
                        ],
                    }
                ],
            )
        except Exception as e:
            return ProviderResult(
                provider_name=self.name,
                error=str(e),
            )

        text = response.content[0].text
        boxes = self._parse_response(text)

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = (input_tokens * 3 / 1_000_000) + (output_tokens * 15 / 1_000_000)

        return ProviderResult(
            provider_name=self.name,
            boxes=boxes,
            cost_estimate=cost,
            raw_response={"text": text, "usage": {"input": input_tokens, "output": output_tokens}},
        )

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
