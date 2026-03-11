from __future__ import annotations

from .base import Pipeline
from .baseline import RekognitionBaseline, GoogleVisionBaseline
from .single_llm import ClaudeSingleShot, GeminiSingleShot, GPT41SingleShot
from .crop_classify import (
    GoogleVisionGeminiFlash,
    RekognitionClaudeHaiku,
    GoogleVisionClaudeHaiku,
)

PIPELINE_REGISTRY: dict[str, type[Pipeline]] = {
    "rekognition": RekognitionBaseline,
    "google_vision": GoogleVisionBaseline,
    "claude_haiku": ClaudeSingleShot,
    "gemini_flash": GeminiSingleShot,
    "gpt41_mini": GPT41SingleShot,
    "gv_gemini_flash": GoogleVisionGeminiFlash,
    "rek_claude_haiku": RekognitionClaudeHaiku,
    "gv_claude_haiku": GoogleVisionClaudeHaiku,
}


def get_pipeline(key: str) -> Pipeline:
    cls = PIPELINE_REGISTRY.get(key)
    if cls is None:
        raise KeyError(f"Unknown pipeline: {key}. Available: {list(PIPELINE_REGISTRY.keys())}")
    return cls()


def list_pipelines() -> list[dict]:
    return [
        {
            "key": key,
            "name": cls.name,
            "category": cls.category,
            "description": cls.description,
        }
        for key, cls in PIPELINE_REGISTRY.items()
    ]
