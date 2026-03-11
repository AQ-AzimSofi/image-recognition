from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from ..providers.base import DetectionBox


@dataclass
class PipelineResult:
    pipeline_name: str
    category: str
    boxes: list[DetectionBox] = field(default_factory=list)
    latency_ms: float = 0.0
    cost_estimate: float = 0.0
    raw_responses: list[dict] = field(default_factory=list)
    error: str | None = None

    @property
    def box_count(self) -> int:
        return len([b for b in self.boxes if b.x_max > 0])

    @property
    def cost_per_1000(self) -> float:
        return self.cost_estimate * 1000


class Pipeline(ABC):
    name: str
    category: str
    description: str

    @abstractmethod
    def run(self, image_path: str | Path) -> PipelineResult:
        ...

    def execute(self, image_path: str | Path) -> PipelineResult:
        start = time.perf_counter()
        result = self.run(image_path)
        result.latency_ms = (time.perf_counter() - start) * 1000
        return result
