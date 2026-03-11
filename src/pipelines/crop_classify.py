from __future__ import annotations

from pathlib import Path

from PIL import Image

from .base import Pipeline, PipelineResult
from ..providers.base import DetectionBox
from ..providers.rekognition_provider import RekognitionProvider
from ..providers.google_vision import GoogleVisionProvider
from ..providers.openrouter import OpenRouterProvider


def _crop_box(img: Image.Image, box: DetectionBox, padding: float = 0.03) -> Image.Image:
    w, h = img.size
    x1 = max(0, int((box.x_min - padding) * w))
    y1 = max(0, int((box.y_min - padding) * h))
    x2 = min(w, int((box.x_max + padding) * w))
    y2 = min(h, int((box.y_max + padding) * h))
    return img.crop((x1, y1, x2, y2))


def _classify_crops(
    img: Image.Image,
    boxes: list[DetectionBox],
    classifier: OpenRouterProvider,
    base_cost: float,
) -> PipelineResult:
    boxes_with_coords = [b for b in boxes if b.x_max > 0]
    if not boxes_with_coords:
        return None

    enriched = []
    total_cost = base_cost

    for box in boxes_with_coords:
        crop = _crop_box(img, box)
        classification = classifier.classify_crop(crop)

        enriched.append(
            DetectionBox(
                label=classification.get("label", box.label),
                confidence=classification.get("confidence", box.confidence / 100) * 100,
                x_min=box.x_min,
                y_min=box.y_min,
                x_max=box.x_max,
                y_max=box.y_max,
                brand=classification.get("brand"),
                product_name=classification.get("product_name"),
            )
        )

    return enriched, total_cost


class GoogleVisionGeminiFlash(Pipeline):
    name = "Google Vision + Gemini Flash"
    category = "Crop & Classify"
    description = "Ultra-low-cost hybrid: Google draws boxes, Gemini Flash classifies each crop"

    def __init__(self):
        self.box_provider = GoogleVisionProvider()
        self.classifier = OpenRouterProvider("gemini-flash", "Gemini Flash (classifier)")

    def run(self, image_path: str | Path) -> PipelineResult:
        box_result = self.box_provider.detect(image_path)
        if box_result.error:
            return PipelineResult(
                pipeline_name=self.name,
                category=self.category,
                error=box_result.error,
            )

        boxes_with_coords = [b for b in box_result.boxes if b.x_max > 0]
        if not boxes_with_coords:
            return PipelineResult(
                pipeline_name=self.name,
                category=self.category,
                boxes=box_result.boxes,
                cost_estimate=box_result.cost_estimate,
            )

        img = Image.open(image_path).convert("RGB")
        result = _classify_crops(img, box_result.boxes, self.classifier, box_result.cost_estimate)
        if result is None:
            return PipelineResult(
                pipeline_name=self.name,
                category=self.category,
                boxes=box_result.boxes,
                cost_estimate=box_result.cost_estimate,
            )

        enriched, total_cost = result
        return PipelineResult(
            pipeline_name=self.name,
            category=self.category,
            boxes=enriched,
            cost_estimate=total_cost,
            raw_responses=[box_result.raw_response] if box_result.raw_response else [],
        )


class RekognitionClaudeHaiku(Pipeline):
    name = "Rekognition + Claude Haiku"
    category = "Crop & Classify"
    description = "AWS ecosystem hybrid: Rekognition boxes + Claude Haiku for detailed classification"

    def __init__(self):
        self.box_provider = RekognitionProvider()
        self.classifier = OpenRouterProvider("claude-haiku", "Claude Haiku (classifier)")

    def run(self, image_path: str | Path) -> PipelineResult:
        box_result = self.box_provider.detect(image_path)
        if box_result.error:
            return PipelineResult(
                pipeline_name=self.name,
                category=self.category,
                error=box_result.error,
            )

        boxes_with_coords = [b for b in box_result.boxes if b.x_max > 0]
        if not boxes_with_coords:
            return PipelineResult(
                pipeline_name=self.name,
                category=self.category,
                boxes=box_result.boxes,
                cost_estimate=box_result.cost_estimate,
            )

        img = Image.open(image_path).convert("RGB")
        result = _classify_crops(img, box_result.boxes, self.classifier, box_result.cost_estimate)
        if result is None:
            return PipelineResult(
                pipeline_name=self.name,
                category=self.category,
                boxes=box_result.boxes,
                cost_estimate=box_result.cost_estimate,
            )

        enriched, total_cost = result
        return PipelineResult(
            pipeline_name=self.name,
            category=self.category,
            boxes=enriched,
            cost_estimate=total_cost,
        )


class GoogleVisionClaudeHaiku(Pipeline):
    name = "Google Vision + Claude Haiku"
    category = "Crop & Classify"
    description = "Balanced hybrid: Google's precise boxes + Claude Haiku's fast identification"

    def __init__(self):
        self.box_provider = GoogleVisionProvider()
        self.classifier = OpenRouterProvider("claude-haiku", "Claude Haiku (classifier)")

    def run(self, image_path: str | Path) -> PipelineResult:
        box_result = self.box_provider.detect(image_path)
        if box_result.error:
            return PipelineResult(
                pipeline_name=self.name,
                category=self.category,
                error=box_result.error,
            )

        boxes_with_coords = [b for b in box_result.boxes if b.x_max > 0]
        if not boxes_with_coords:
            return PipelineResult(
                pipeline_name=self.name,
                category=self.category,
                boxes=box_result.boxes,
                cost_estimate=box_result.cost_estimate,
            )

        img = Image.open(image_path).convert("RGB")
        result = _classify_crops(img, box_result.boxes, self.classifier, box_result.cost_estimate)
        if result is None:
            return PipelineResult(
                pipeline_name=self.name,
                category=self.category,
                boxes=box_result.boxes,
                cost_estimate=box_result.cost_estimate,
            )

        enriched, total_cost = result
        return PipelineResult(
            pipeline_name=self.name,
            category=self.category,
            boxes=enriched,
            cost_estimate=total_cost,
        )
