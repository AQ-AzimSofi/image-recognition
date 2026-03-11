from __future__ import annotations

from pathlib import Path

from .base import Pipeline, PipelineResult
from ..providers.rekognition_provider import RekognitionProvider
from ..providers.google_vision import GoogleVisionProvider


class RekognitionBaseline(Pipeline):
    name = "AWS Rekognition Only"
    category = "Baseline"
    description = "Current state: fast bounding boxes but generic English labels (e.g. 'Bottle', 'Box')"

    def __init__(self):
        self.provider = RekognitionProvider()

    def run(self, image_path: str | Path) -> PipelineResult:
        result = self.provider.detect(image_path)
        return PipelineResult(
            pipeline_name=self.name,
            category=self.category,
            boxes=result.boxes,
            cost_estimate=result.cost_estimate,
            raw_responses=[result.raw_response] if result.raw_response else [],
            error=result.error,
        )


class GoogleVisionBaseline(Pipeline):
    name = "Google Cloud Vision Only"
    category = "Baseline"
    description = "Object localization with slightly better consumer goods detection, still generic labels"

    def __init__(self):
        self.provider = GoogleVisionProvider()

    def run(self, image_path: str | Path) -> PipelineResult:
        result = self.provider.detect(image_path)
        return PipelineResult(
            pipeline_name=self.name,
            category=self.category,
            boxes=result.boxes,
            cost_estimate=result.cost_estimate,
            raw_responses=[result.raw_response] if result.raw_response else [],
            error=result.error,
        )
