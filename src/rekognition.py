from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from .config import DEFAULT_CONFIG
from .models import DetectedInstance, DetectedLabel, DetectionResult


MAX_IMAGE_BYTES = 5 * 1024 * 1024


class RekognitionError(Exception):
    pass


class CredentialsError(RekognitionError):
    pass


class ImageTooLargeError(RekognitionError):
    pass


class InvalidImageError(RekognitionError):
    pass


class RekognitionClient:
    def __init__(
        self,
        region_name: str | None = None,
        min_confidence: float | None = None,
        max_labels: int | None = None,
    ):
        config = DEFAULT_CONFIG.rekognition
        self.region_name = region_name or config.region_name
        self.min_confidence = min_confidence or config.min_confidence
        self.max_labels = max_labels or config.max_labels

        try:
            self.client = boto3.client("rekognition", region_name=self.region_name)
        except NoCredentialsError:
            raise CredentialsError(
                "AWS credentials not found. Configure via environment variables, "
                "~/.aws/credentials, or IAM role."
            )

    def detect_labels(
        self,
        image_bytes: bytes,
        min_confidence: float | None = None,
        max_labels: int | None = None,
    ) -> DetectionResult:
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise ImageTooLargeError(
                f"Image size ({len(image_bytes)} bytes) exceeds Rekognition limit "
                f"of {MAX_IMAGE_BYTES} bytes (5MB). Resize or compress the image."
            )

        try:
            response = self.client.detect_labels(
                Image={"Bytes": image_bytes},
                MinConfidence=min_confidence or self.min_confidence,
                MaxLabels=max_labels or self.max_labels,
                Features=["GENERAL_LABELS"],
            )
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "InvalidImageFormatException":
                raise InvalidImageError("Image format not supported by Rekognition.")
            if error_code == "ImageTooLargeException":
                raise ImageTooLargeError("Image too large for Rekognition.")
            raise RekognitionError(f"Rekognition API error: {e}")
        except NoCredentialsError:
            raise CredentialsError("AWS credentials expired or invalid.")

        labels = self._parse_labels(response)

        return DetectionResult(
            labels=labels,
            raw_response=response,
            image_width=0,
            image_height=0,
        )

    def detect_labels_from_path(
        self,
        image_path: Path,
        min_confidence: float | None = None,
        max_labels: int | None = None,
    ) -> DetectionResult:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image_bytes = image_path.read_bytes()
        return self.detect_labels(image_bytes, min_confidence, max_labels)

    def _parse_labels(self, response: dict) -> list[DetectedLabel]:
        labels = []
        for label_data in response.get("Labels", []):
            instances = []
            for inst in label_data.get("Instances", []):
                bbox = inst.get("BoundingBox", {})
                instances.append(
                    DetectedInstance(
                        bbox_left=bbox.get("Left", 0),
                        bbox_top=bbox.get("Top", 0),
                        bbox_width=bbox.get("Width", 0),
                        bbox_height=bbox.get("Height", 0),
                        confidence=inst.get("Confidence", 0),
                    )
                )

            categories = [
                c.get("Name", "") for c in label_data.get("Categories", [])
            ]
            parents = [
                p.get("Name", "") for p in label_data.get("Parents", [])
            ]

            labels.append(
                DetectedLabel(
                    name=label_data.get("Name", ""),
                    confidence=label_data.get("Confidence", 0),
                    categories=categories,
                    parents=parents,
                    instances=instances,
                )
            )

        return labels
