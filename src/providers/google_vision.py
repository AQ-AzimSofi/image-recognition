from __future__ import annotations

import logging
from pathlib import Path

from .base import DetectionBox, Provider, ProviderResult, load_image_bytes

logger = logging.getLogger(__name__)

COST_PER_IMAGE = 0.00225


class GoogleVisionProvider(Provider):
    name = "Google Cloud Vision"

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from google.cloud import vision
            self._client = vision.ImageAnnotatorClient()
        return self._client

    def detect(self, image_path: str | Path) -> ProviderResult:
        from google.cloud import vision

        image_bytes = load_image_bytes(image_path)
        image = vision.Image(content=image_bytes)

        try:
            response = self.client.object_localization(image=image)
        except Exception as e:
            return ProviderResult(
                provider_name=self.name,
                error=str(e),
            )

        annotations = response.localized_object_annotations
        logger.info(
            "[GoogleVision] Total annotations returned: %d", len(annotations)
        )

        boxes = []
        for i, obj in enumerate(annotations):
            vertices = obj.bounding_poly.normalized_vertices
            vertex_coords = [(v.x, v.y) for v in vertices]
            logger.info(
                "[GoogleVision] Object #%d: name=%r, score=%.3f, "
                "vertex_count=%d, vertices=%s",
                i, obj.name, obj.score, len(vertices), vertex_coords,
            )

            if obj.score < self.min_confidence:
                logger.info(
                    "[GoogleVision]   -> SKIPPED: score %.3f < min_confidence %.3f",
                    obj.score, self.min_confidence,
                )
                continue

            if len(vertices) >= 4:
                x_min = vertices[0].x
                y_min = vertices[0].y
                x_max = vertices[2].x
                y_max = vertices[2].y
                logger.info(
                    "[GoogleVision]   -> BOX: x_min=%.4f, y_min=%.4f, "
                    "x_max=%.4f, y_max=%.4f",
                    x_min, y_min, x_max, y_max,
                )
            else:
                logger.warning(
                    "[GoogleVision]   -> SKIPPED: only %d vertices (need 4+)",
                    len(vertices),
                )
                continue

            boxes.append(
                DetectionBox(
                    label=obj.name,
                    confidence=obj.score * 100,
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                )
            )

        logger.info(
            "[GoogleVision] Final result: %d boxes from %d annotations",
            len(boxes), len(annotations),
        )

        raw = {
            "objects": [
                {
                    "name": obj.name,
                    "score": obj.score,
                    "vertices": [
                        {"x": v.x, "y": v.y}
                        for v in obj.bounding_poly.normalized_vertices
                    ],
                }
                for obj in response.localized_object_annotations
            ]
        }

        return ProviderResult(
            provider_name=self.name,
            boxes=boxes,
            cost_estimate=COST_PER_IMAGE,
            raw_response=raw,
        )
