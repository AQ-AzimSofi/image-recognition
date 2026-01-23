#!/usr/bin/env python3
"""
Auto-labeling module using YOLO.

Pre-annotates images using existing YOLO models to speed up manual annotation.
Exports in various formats (YOLO, COCO, Label Studio).
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import uuid


@dataclass
class Detection:
    """Single detection result."""
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str


@dataclass
class ImageAnnotation:
    """Annotations for a single image."""
    image_path: str
    width: int
    height: int
    detections: List[Detection]


class AutoLabeler:
    """
    Auto-label images using YOLO models.

    Features:
    - Pre-annotate images with existing YOLO model
    - Export to YOLO, COCO, or Label Studio format
    - Confidence filtering
    - Batch processing
    """

    def __init__(
        self,
        model_path: str = None,
        confidence: float = 0.5,
        iou: float = 0.45,
        classes: List[int] = None
    ):
        """
        Initialize auto-labeler.

        Args:
            model_path: Path to YOLO model
            confidence: Confidence threshold
            iou: NMS IoU threshold
            classes: Class IDs to detect (None = all, [0] = person only)
        """
        from ultralytics import YOLO
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from config import get_model_path

        model_path = model_path or get_model_path()
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou = iou
        self.classes = classes if classes is not None else [0]  # Default: person only
        self.class_names = self.model.names

    def detect_single(self, image_path: str) -> ImageAnnotation:
        """
        Detect objects in a single image.

        Args:
            image_path: Path to image

        Returns:
            ImageAnnotation with detections
        """
        results = self.model(
            image_path,
            conf=self.confidence,
            iou=self.iou,
            classes=self.classes,
            verbose=False
        )

        detections = []
        img_width = int(results[0].orig_shape[1])
        img_height = int(results[0].orig_shape[0])

        for r in results:
            for box in r.boxes:
                class_id = int(box.cls)
                detections.append(Detection(
                    bbox=box.xyxy[0].cpu().tolist(),
                    confidence=float(box.conf),
                    class_id=class_id,
                    class_name=self.class_names[class_id]
                ))

        return ImageAnnotation(
            image_path=str(image_path),
            width=img_width,
            height=img_height,
            detections=detections
        )

    def detect_batch(
        self,
        image_dir: str,
        recursive: bool = False,
        extensions: List[str] = None
    ) -> List[ImageAnnotation]:
        """
        Detect objects in all images in a directory.

        Args:
            image_dir: Directory containing images
            recursive: Search subdirectories
            extensions: File extensions to process

        Returns:
            List of ImageAnnotation
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

        image_dir = Path(image_dir)
        pattern = "**/*" if recursive else "*"

        image_paths = [
            p for p in image_dir.glob(pattern)
            if p.suffix.lower() in extensions
        ]

        print(f"Processing {len(image_paths)} images...")

        annotations = []
        for i, img_path in enumerate(sorted(image_paths)):
            ann = self.detect_single(str(img_path))
            annotations.append(ann)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(image_paths)}")

        total_detections = sum(len(a.detections) for a in annotations)
        print(f"Done. Total detections: {total_detections}")

        return annotations

    def export_yolo(
        self,
        annotations: List[ImageAnnotation],
        output_dir: str,
        copy_images: bool = False
    ) -> str:
        """
        Export annotations in YOLO format.

        Args:
            annotations: List of annotations
            output_dir: Output directory
            copy_images: Copy images to output directory

        Returns:
            Path to output directory
        """
        import shutil

        output_dir = Path(output_dir)
        labels_dir = output_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        if copy_images:
            images_dir = output_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

        for ann in annotations:
            img_path = Path(ann.image_path)
            label_path = labels_dir / f"{img_path.stem}.txt"

            with open(label_path, "w") as f:
                for det in ann.detections:
                    # Convert to YOLO format: class x_center y_center width height
                    x1, y1, x2, y2 = det.bbox
                    x_center = ((x1 + x2) / 2) / ann.width
                    y_center = ((y1 + y2) / 2) / ann.height
                    width = (x2 - x1) / ann.width
                    height = (y2 - y1) / ann.height

                    f.write(f"{det.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            if copy_images:
                shutil.copy2(img_path, images_dir / img_path.name)

        # Write classes.txt
        classes_file = output_dir / "classes.txt"
        class_ids = sorted(set(d.class_id for a in annotations for d in a.detections))
        with open(classes_file, "w") as f:
            for cid in class_ids:
                f.write(f"{self.class_names[cid]}\n")

        print(f"Exported YOLO format to: {output_dir}")
        return str(output_dir)

    def export_coco(
        self,
        annotations: List[ImageAnnotation],
        output_path: str
    ) -> str:
        """
        Export annotations in COCO format.

        Args:
            annotations: List of annotations
            output_path: Output JSON file path

        Returns:
            Path to output file
        """
        coco = {
            "info": {
                "description": "Auto-labeled dataset",
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().isoformat()
            },
            "licenses": [],
            "categories": [],
            "images": [],
            "annotations": []
        }

        # Build categories
        class_ids = sorted(set(d.class_id for a in annotations for d in a.detections))
        for cid in class_ids:
            coco["categories"].append({
                "id": cid,
                "name": self.class_names[cid],
                "supercategory": ""
            })

        annotation_id = 0
        for img_id, ann in enumerate(annotations):
            coco["images"].append({
                "id": img_id,
                "file_name": Path(ann.image_path).name,
                "width": ann.width,
                "height": ann.height
            })

            for det in ann.detections:
                x1, y1, x2, y2 = det.bbox
                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": det.class_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO: [x, y, width, height]
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                    "score": det.confidence
                })
                annotation_id += 1

        with open(output_path, "w") as f:
            json.dump(coco, f, indent=2)

        print(f"Exported COCO format to: {output_path}")
        return output_path

    def export_label_studio(
        self,
        annotations: List[ImageAnnotation],
        output_path: str,
        image_url_prefix: str = "/data/local-files/?d="
    ) -> str:
        """
        Export annotations in Label Studio format.

        Args:
            annotations: List of annotations
            output_path: Output JSON file path
            image_url_prefix: URL prefix for images

        Returns:
            Path to output file
        """
        tasks = []

        for ann in annotations:
            img_path = Path(ann.image_path)

            predictions = []
            for det in ann.detections:
                x1, y1, x2, y2 = det.bbox

                # Label Studio uses percentages
                x_pct = (x1 / ann.width) * 100
                y_pct = (y1 / ann.height) * 100
                w_pct = ((x2 - x1) / ann.width) * 100
                h_pct = ((y2 - y1) / ann.height) * 100

                predictions.append({
                    "id": str(uuid.uuid4())[:8],
                    "type": "rectanglelabels",
                    "from_name": "label",
                    "to_name": "image",
                    "original_width": ann.width,
                    "original_height": ann.height,
                    "value": {
                        "x": x_pct,
                        "y": y_pct,
                        "width": w_pct,
                        "height": h_pct,
                        "rotation": 0,
                        "rectanglelabels": [det.class_name]
                    },
                    "score": det.confidence
                })

            task = {
                "data": {
                    "image": f"{image_url_prefix}{img_path.name}"
                },
                "predictions": [{
                    "model_version": "yolo_auto",
                    "score": sum(d.confidence for d in ann.detections) / len(ann.detections) if ann.detections else 0,
                    "result": predictions
                }] if predictions else []
            }
            tasks.append(task)

        with open(output_path, "w") as f:
            json.dump(tasks, f, indent=2)

        print(f"Exported Label Studio format to: {output_path}")
        return output_path

    def get_statistics(self, annotations: List[ImageAnnotation]) -> Dict:
        """
        Get annotation statistics.

        Args:
            annotations: List of annotations

        Returns:
            Statistics dict
        """
        total_images = len(annotations)
        total_detections = sum(len(a.detections) for a in annotations)
        images_with_detections = sum(1 for a in annotations if a.detections)

        class_counts = {}
        confidence_sum = 0

        for ann in annotations:
            for det in ann.detections:
                class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
                confidence_sum += det.confidence

        return {
            "total_images": total_images,
            "total_detections": total_detections,
            "images_with_detections": images_with_detections,
            "images_without_detections": total_images - images_with_detections,
            "avg_detections_per_image": total_detections / total_images if total_images > 0 else 0,
            "avg_confidence": confidence_sum / total_detections if total_detections > 0 else 0,
            "class_distribution": class_counts
        }


def main():
    """CLI for auto-labeling."""
    import argparse

    parser = argparse.ArgumentParser(description="Auto-label images with YOLO")
    parser.add_argument("input", help="Image file or directory")
    parser.add_argument("--model", "-m", default="yolo11n.pt", help="YOLO model")
    parser.add_argument("--confidence", "-c", type=float, default=0.5)
    parser.add_argument("--output", "-o", required=True, help="Output path")
    parser.add_argument("--format", "-f", choices=["yolo", "coco", "labelstudio"], default="yolo")
    parser.add_argument("--copy-images", action="store_true")

    args = parser.parse_args()

    labeler = AutoLabeler(
        model_path=args.model,
        confidence=args.confidence
    )

    input_path = Path(args.input)
    if input_path.is_dir():
        annotations = labeler.detect_batch(str(input_path))
    else:
        annotations = [labeler.detect_single(str(input_path))]

    # Show stats
    stats = labeler.get_statistics(annotations)
    print(f"\nStatistics:")
    print(f"  Images: {stats['total_images']}")
    print(f"  Detections: {stats['total_detections']}")
    print(f"  Avg confidence: {stats['avg_confidence']:.3f}")

    # Export
    if args.format == "yolo":
        labeler.export_yolo(annotations, args.output, args.copy_images)
    elif args.format == "coco":
        labeler.export_coco(annotations, args.output)
    elif args.format == "labelstudio":
        labeler.export_label_studio(annotations, args.output)


if __name__ == "__main__":
    main()
