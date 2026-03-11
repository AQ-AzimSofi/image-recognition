from __future__ import annotations

from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from .base import DetectionBox, Provider, ProviderResult, load_image_bytes

COST_PER_IMAGE = 0.001


class RekognitionProvider(Provider):
    name = "AWS Rekognition"

    def __init__(
        self,
        region_name: str = "ap-northeast-1",
        min_confidence: float = 50.0,
        max_labels: int = 20,
    ):
        self.min_confidence = min_confidence
        self.max_labels = max_labels
        self.client = boto3.client("rekognition", region_name=region_name)

    def detect(self, image_path: str | Path) -> ProviderResult:
        image_bytes = load_image_bytes(image_path)

        try:
            response = self.client.detect_labels(
                Image={"Bytes": image_bytes},
                MinConfidence=self.min_confidence,
                MaxLabels=self.max_labels,
                Features=["GENERAL_LABELS"],
            )
        except (ClientError, NoCredentialsError) as e:
            return ProviderResult(
                provider_name=self.name,
                error=str(e),
            )

        boxes = []
        for label_data in response.get("Labels", []):
            label_name = label_data.get("Name", "")
            confidence = label_data.get("Confidence", 0)

            instances = label_data.get("Instances", [])
            if instances:
                for inst in instances:
                    bbox = inst.get("BoundingBox", {})
                    left = bbox.get("Left", 0)
                    top = bbox.get("Top", 0)
                    width = bbox.get("Width", 0)
                    height = bbox.get("Height", 0)
                    boxes.append(
                        DetectionBox(
                            label=label_name,
                            confidence=inst.get("Confidence", confidence),
                            x_min=left,
                            y_min=top,
                            x_max=left + width,
                            y_max=top + height,
                        )
                    )
            else:
                boxes.append(
                    DetectionBox(
                        label=label_name,
                        confidence=confidence,
                        x_min=0,
                        y_min=0,
                        x_max=0,
                        y_max=0,
                    )
                )

        return ProviderResult(
            provider_name=self.name,
            boxes=boxes,
            cost_estimate=COST_PER_IMAGE,
            raw_response=response,
        )
