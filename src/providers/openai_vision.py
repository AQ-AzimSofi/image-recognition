from __future__ import annotations

import json
import logging
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

logger = logging.getLogger(__name__)

COST_PER_IMAGE = 0.003

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


class OpenAIVisionProvider(Provider):
    name = "GPT-4.1"

    def __init__(self, model: str = "gpt-4.1"):
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return self._client

    def detect(self, image_path: str | Path) -> ProviderResult:
        image_b64 = encode_image_base64(image_path)
        media_type = get_image_media_type(image_path)
        data_url = f"data:{media_type};base64,{image_b64}"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url, "detail": "high"},
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

        text = response.choices[0].message.content
        logger.info("[OpenAI/%s] Raw response text:\n%s", self.model, text)
        boxes = self._parse_response(text)

        usage = response.usage
        cost = (usage.prompt_tokens * 2 / 1_000_000) + (usage.completion_tokens * 8 / 1_000_000)

        return ProviderResult(
            provider_name=self.name,
            boxes=boxes,
            cost_estimate=cost,
            raw_response={"text": text, "usage": {"prompt": usage.prompt_tokens, "completion": usage.completion_tokens}},
        )

    def _parse_response(self, text: str) -> list[DetectionBox]:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
            logger.info("[OpenAI/%s] Stripped markdown code fences", self.model)

        try:
            items = json.loads(text)
            logger.info("[OpenAI/%s] JSON parsed OK, %d items", self.model, len(items))
        except json.JSONDecodeError as e:
            logger.warning("[OpenAI/%s] JSON parse failed: %s", self.model, e)
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                try:
                    items = json.loads(text[start:end])
                    logger.info(
                        "[OpenAI/%s] Fallback JSON parse OK, %d items",
                        self.model, len(items),
                    )
                except json.JSONDecodeError as e2:
                    logger.error(
                        "[OpenAI/%s] Fallback JSON parse also failed: %s",
                        self.model, e2,
                    )
                    return []
            else:
                logger.error(
                    "[OpenAI/%s] No JSON array found in response", self.model
                )
                return []

        scale = detect_box_scale(items)
        logger.info("[OpenAI/%s] Detected coordinate scale: %s", self.model, scale)

        boxes = []
        for i, item in enumerate(items):
            box_2d = item.get("box_2d", [0, 0, 0, 0])
            has_box_field = "box_2d" in item
            logger.info(
                "[OpenAI/%s] Item #%d: label=%r, confidence=%s, "
                "has_box_2d=%s, box_2d=%s",
                self.model, i,
                item.get("label", item.get("product_name", "?")),
                item.get("confidence"),
                has_box_field, box_2d,
            )

            if len(box_2d) != 4:
                logger.warning(
                    "[OpenAI/%s]   -> SKIPPED: box_2d has %d values (need 4)",
                    self.model, len(box_2d),
                )
                continue

            ymin, xmin, ymax, xmax = box_2d
            is_zero_box = all(v == 0 for v in box_2d)
            if is_zero_box:
                logger.warning(
                    "[OpenAI/%s]   -> NOTE: box_2d is all zeros (no spatial data)",
                    self.model,
                )

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

        zero_box_count = sum(
            1 for b in boxes
            if b.x_min == 0 and b.y_min == 0 and b.x_max == 0 and b.y_max == 0
        )
        logger.info(
            "[OpenAI/%s] Final: %d boxes total, %d with zero coordinates",
            self.model, len(boxes), zero_box_count,
        )
        return boxes
