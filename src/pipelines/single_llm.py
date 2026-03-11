from __future__ import annotations

from pathlib import Path

from .base import Pipeline, PipelineResult
from ..providers.openrouter import OpenRouterProvider


class ClaudeSingleShot(Pipeline):
    name = "Claude 4.5 Haiku"
    category = "Single-Step LLM"
    description = "Fast and cheap; good contextual understanding with bounding box coordinates in one call"

    def __init__(self):
        self.provider = OpenRouterProvider("claude-haiku")

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


class GeminiSingleShot(Pipeline):
    name = "Gemini 2.5 Flash"
    category = "Single-Step LLM"
    description = "Fast and cheap with native bounding box output, strong multilingual support"

    def __init__(self):
        self.provider = OpenRouterProvider("gemini-flash")

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


class GPT41SingleShot(Pipeline):
    name = "GPT-4.1 Mini"
    category = "Single-Step LLM"
    description = "Cost-effective mid-tier model; good balance of vision quality and price"

    def __init__(self):
        self.provider = OpenRouterProvider("gpt-4.1-mini")

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
