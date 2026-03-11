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

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODEL_ALIASES = {
    "claude-sonnet": "anthropic/claude-sonnet-4-6",
    "claude-haiku": "anthropic/claude-haiku-4-5",
    "gemini-pro": "google/gemini-2.5-pro-preview",
    "gemini-flash": "google/gemini-2.5-flash",
    "gpt-4.1": "openai/gpt-4.1",
    "gpt-4.1-mini": "openai/gpt-4.1-mini",
}

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


class OpenRouterProvider(Provider):
    def __init__(
        self,
        model: str,
        display_name: str | None = None,
        api_key: str | None = None,
    ):
        resolved = MODEL_ALIASES.get(model, model)
        self.model = resolved
        self.name = display_name or _friendly_name(resolved)
        self._api_key = api_key
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=self._api_key or os.environ.get("OPENROUTER_API_KEY"),
            )
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
                                "image_url": {"url": data_url},
                            },
                            {
                                "type": "text",
                                "text": DETECTION_PROMPT,
                            },
                        ],
                    }
                ],
                extra_headers={
                    "HTTP-Referer": "https://github.com/kjm-image-recognition",
                    "X-Title": "KJM Vision POC",
                },
            )
        except Exception as e:
            return ProviderResult(
                provider_name=self.name,
                error=str(e),
            )

        text = response.choices[0].message.content
        logger.info("[OpenRouter/%s] Raw response text:\n%s", self.name, text)
        boxes = _parse_llm_response(text, self.name)

        usage = response.usage
        cost = _estimate_cost(self.model, usage.prompt_tokens, usage.completion_tokens)

        return ProviderResult(
            provider_name=self.name,
            boxes=boxes,
            cost_estimate=cost,
            raw_response={
                "model": self.model,
                "text": text,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                },
            },
        )

    def classify_crop(self, crop_image) -> dict:
        import io
        import base64

        buf = io.BytesIO()
        crop_image.save(buf, format="JPEG")
        crop_b64 = base64.b64encode(buf.getvalue()).decode()
        data_url = f"data:image/jpeg;base64,{crop_b64}"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=512,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": CLASSIFY_PROMPT},
                        ],
                    }
                ],
                extra_headers={
                    "HTTP-Referer": "https://github.com/kjm-image-recognition",
                    "X-Title": "KJM Vision POC",
                },
            )
            text = response.choices[0].message.content.strip()
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:-1])
            return json.loads(text)
        except Exception:
            return {
                "product_name": "Unknown",
                "brand": "Unknown",
                "label": "Unknown",
                "confidence": 0.0,
            }


def _friendly_name(model_id: str) -> str:
    names = {
        "anthropic/claude-sonnet-4-6": "Claude 4.6 Sonnet",
        "anthropic/claude-haiku-4-5": "Claude 4.5 Haiku",
        "google/gemini-2.5-pro-preview": "Gemini 2.5 Pro",
        "google/gemini-2.5-flash": "Gemini 2.5 Flash",
        "openai/gpt-4.1": "GPT-4.1",
        "openai/gpt-4.1-mini": "GPT-4.1 Mini",
    }
    return names.get(model_id, model_id.split("/")[-1])


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = {
        "anthropic/claude-sonnet-4-6": (3.0, 15.0),
        "anthropic/claude-haiku-4-5": (0.8, 4.0),
        "google/gemini-2.5-pro-preview": (1.25, 10.0),
        "google/gemini-2.5-flash": (0.15, 0.6),
        "openai/gpt-4.1": (2.0, 8.0),
        "openai/gpt-4.1-mini": (0.4, 1.6),
    }
    rates = pricing.get(model, (2.0, 8.0))
    return (input_tokens * rates[0] / 1_000_000) + (output_tokens * rates[1] / 1_000_000)


def _parse_llm_response(text: str, provider_name: str = "LLM") -> list[DetectionBox]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
        logger.info("[OpenRouter/%s] Stripped markdown code fences", provider_name)

    try:
        items = json.loads(text)
        logger.info("[OpenRouter/%s] JSON parsed OK, %d items", provider_name, len(items))
    except json.JSONDecodeError as e:
        logger.warning("[OpenRouter/%s] JSON parse failed: %s", provider_name, e)
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                items = json.loads(text[start:end])
                logger.info(
                    "[OpenRouter/%s] Fallback JSON parse OK, %d items",
                    provider_name, len(items),
                )
            except json.JSONDecodeError as e2:
                logger.error(
                    "[OpenRouter/%s] Fallback JSON parse also failed: %s",
                    provider_name, e2,
                )
                return []
        else:
            logger.error("[OpenRouter/%s] No JSON array found in response", provider_name)
            return []

    scale = detect_box_scale(items)
    logger.info(
        "[OpenRouter/%s] Detected coordinate scale: %s",
        provider_name, scale,
    )

    boxes = []
    for i, item in enumerate(items):
        box_2d = item.get("box_2d", [0, 0, 0, 0])
        has_box_field = "box_2d" in item
        logger.info(
            "[OpenRouter/%s] Item #%d: label=%r, confidence=%s, "
            "has_box_2d=%s, box_2d=%s",
            provider_name, i,
            item.get("label", item.get("product_name", "?")),
            item.get("confidence"),
            has_box_field, box_2d,
        )

        if len(box_2d) != 4:
            logger.warning(
                "[OpenRouter/%s]   -> SKIPPED: box_2d has %d values (need 4)",
                provider_name, len(box_2d),
            )
            continue

        ymin, xmin, ymax, xmax = box_2d
        is_zero_box = all(v == 0 for v in box_2d)
        if is_zero_box:
            logger.warning(
                "[OpenRouter/%s]   -> NOTE: box_2d is all zeros (no spatial data)",
                provider_name,
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
        "[OpenRouter/%s] Final: %d boxes total, %d with zero coordinates",
        provider_name, len(boxes), zero_box_count,
    )
    return boxes
