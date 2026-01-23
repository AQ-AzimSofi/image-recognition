#!/usr/bin/env python3
"""
Annotation storage module.

Stores and manages annotations (bounding boxes, labels) for images.
Supports multiple formats and provides conversion utilities.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum


class AnnotationFormat(Enum):
    """Supported annotation formats."""
    YOLO = "yolo"
    COCO = "coco"
    PASCAL_VOC = "pascal_voc"
    INTERNAL = "internal"


@dataclass
class BoundingBox:
    """Bounding box annotation."""
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int = 0
    class_name: str = "person"
    confidence: float = 1.0
    person_id: Optional[str] = None

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_yolo(self, img_width: int, img_height: int) -> str:
        """Convert to YOLO format (class x_center y_center width height)."""
        x_center = (self.x1 + self.x2) / 2 / img_width
        y_center = (self.y1 + self.y2) / 2 / img_height
        width = self.width / img_width
        height = self.height / img_height
        return f"{self.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    @classmethod
    def from_yolo(
        cls,
        line: str,
        img_width: int,
        img_height: int,
        class_names: List[str] = None
    ) -> "BoundingBox":
        """Create from YOLO format."""
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1]) * img_width
        y_center = float(parts[2]) * img_height
        width = float(parts[3]) * img_width
        height = float(parts[4]) * img_height

        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        class_name = class_names[class_id] if class_names and class_id < len(class_names) else "person"

        return cls(
            x1=x1, y1=y1, x2=x2, y2=y2,
            class_id=class_id,
            class_name=class_name
        )


@dataclass
class Annotation:
    """Image annotation containing multiple bounding boxes."""
    image_path: str
    image_width: int
    image_height: int
    boxes: List[BoundingBox] = field(default_factory=list)
    source: str = "manual"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.boxes)


class AnnotationStore:
    """
    Store and manage annotations.

    Features:
    - Multiple format support (YOLO, COCO, Pascal VOC)
    - Automatic format detection
    - Conversion between formats
    - Validation and statistics
    """

    def __init__(
        self,
        store_dir: str = "data/annotations",
        class_names: List[str] = None
    ):
        """
        Initialize annotation store.

        Args:
            store_dir: Directory for storing annotations
            class_names: List of class names (default: ["person"])
        """
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names or ["person"]
        self._annotations: Dict[str, Annotation] = {}
        self._load_index()

    def _load_index(self):
        """Load annotation index."""
        index_path = self.store_dir / "index.json"
        if index_path.exists():
            with open(index_path, "r") as f:
                index = json.load(f)
                for path, data in index.get("annotations", {}).items():
                    boxes = [BoundingBox(**b) for b in data.get("boxes", [])]
                    self._annotations[path] = Annotation(
                        image_path=data["image_path"],
                        image_width=data["image_width"],
                        image_height=data["image_height"],
                        boxes=boxes,
                        source=data.get("source", "unknown"),
                        created_at=data.get("created_at", ""),
                        updated_at=data.get("updated_at", ""),
                        metadata=data.get("metadata", {})
                    )

    def _save_index(self):
        """Save annotation index."""
        index = {
            "updated_at": datetime.now().isoformat(),
            "count": len(self._annotations),
            "class_names": self.class_names,
            "annotations": {}
        }

        for path, ann in self._annotations.items():
            index["annotations"][path] = {
                "image_path": ann.image_path,
                "image_width": ann.image_width,
                "image_height": ann.image_height,
                "boxes": [asdict(b) for b in ann.boxes],
                "source": ann.source,
                "created_at": ann.created_at,
                "updated_at": ann.updated_at,
                "metadata": ann.metadata
            }

        with open(self.store_dir / "index.json", "w") as f:
            json.dump(index, f, indent=2)

    def add(self, annotation: Annotation) -> str:
        """
        Add annotation to store.

        Args:
            annotation: Annotation object

        Returns:
            Annotation ID (image path)
        """
        key = annotation.image_path
        annotation.updated_at = datetime.now().isoformat()
        self._annotations[key] = annotation
        self._save_index()
        return key

    def get(self, image_path: str) -> Optional[Annotation]:
        """
        Get annotation for image.

        Args:
            image_path: Path to image

        Returns:
            Annotation or None
        """
        return self._annotations.get(image_path)

    def remove(self, image_path: str) -> bool:
        """
        Remove annotation.

        Args:
            image_path: Path to image

        Returns:
            True if removed
        """
        if image_path in self._annotations:
            del self._annotations[image_path]
            self._save_index()
            return True
        return False

    def list_all(self) -> List[str]:
        """List all annotated image paths."""
        return list(self._annotations.keys())

    def import_yolo(
        self,
        labels_dir: str,
        images_dir: str,
        recursive: bool = False
    ) -> int:
        """
        Import annotations from YOLO format.

        Args:
            labels_dir: Directory containing .txt label files
            images_dir: Directory containing images
            recursive: Search subdirectories

        Returns:
            Number of annotations imported
        """
        labels_dir = Path(labels_dir)
        images_dir = Path(images_dir)
        imported = 0

        pattern = "**/*.txt" if recursive else "*.txt"

        for label_path in labels_dir.glob(pattern):
            stem = label_path.stem

            img_path = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                candidate = images_dir / f"{stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break

            if not img_path:
                continue

            from PIL import Image
            with Image.open(img_path) as img:
                width, height = img.size

            boxes = []
            with open(label_path, "r") as f:
                for line in f:
                    if line.strip():
                        box = BoundingBox.from_yolo(line, width, height, self.class_names)
                        boxes.append(box)

            annotation = Annotation(
                image_path=str(img_path),
                image_width=width,
                image_height=height,
                boxes=boxes,
                source="yolo_import"
            )

            self.add(annotation)
            imported += 1

        print(f"Imported {imported} annotations from YOLO format")
        return imported

    def export_yolo(
        self,
        output_dir: str,
        image_paths: List[str] = None
    ) -> int:
        """
        Export annotations to YOLO format.

        Args:
            output_dir: Output directory for label files
            image_paths: Optional list of images to export (default: all)

        Returns:
            Number of annotations exported
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = image_paths or list(self._annotations.keys())
        exported = 0

        for img_path in paths:
            ann = self._annotations.get(img_path)
            if not ann:
                continue

            label_path = output_dir / f"{Path(img_path).stem}.txt"

            with open(label_path, "w") as f:
                for box in ann.boxes:
                    line = box.to_yolo(ann.image_width, ann.image_height)
                    f.write(line + "\n")

            exported += 1

        print(f"Exported {exported} annotations to YOLO format")
        return exported

    def import_from_detections(
        self,
        detection_results: Dict[str, List[List[float]]],
        source: str = "yolo_detection"
    ) -> int:
        """
        Import annotations from detection results.

        Args:
            detection_results: Dict mapping image_path to list of boxes [x1,y1,x2,y2,conf]
            source: Source identifier

        Returns:
            Number of annotations imported
        """
        imported = 0

        for img_path, boxes_raw in detection_results.items():
            img_path = Path(img_path)
            if not img_path.exists():
                continue

            from PIL import Image
            with Image.open(img_path) as img:
                width, height = img.size

            boxes = []
            for box_data in boxes_raw:
                box = BoundingBox(
                    x1=box_data[0],
                    y1=box_data[1],
                    x2=box_data[2],
                    y2=box_data[3],
                    confidence=box_data[4] if len(box_data) > 4 else 1.0,
                    class_id=0,
                    class_name="person"
                )
                boxes.append(box)

            annotation = Annotation(
                image_path=str(img_path),
                image_width=width,
                image_height=height,
                boxes=boxes,
                source=source
            )

            self.add(annotation)
            imported += 1

        return imported

    def get_statistics(self) -> Dict:
        """
        Get annotation statistics.

        Returns:
            Statistics dict
        """
        total_boxes = 0
        class_counts = {}
        sources = {}
        box_sizes = []

        for ann in self._annotations.values():
            total_boxes += ann.count

            for box in ann.boxes:
                class_counts[box.class_name] = class_counts.get(box.class_name, 0) + 1
                box_sizes.append(box.area)

            sources[ann.source] = sources.get(ann.source, 0) + 1

        return {
            "total_images": len(self._annotations),
            "total_boxes": total_boxes,
            "avg_boxes_per_image": total_boxes / len(self._annotations) if self._annotations else 0,
            "class_distribution": class_counts,
            "sources": sources,
            "avg_box_area": sum(box_sizes) / len(box_sizes) if box_sizes else 0
        }

    def validate(self) -> Dict:
        """
        Validate annotations.

        Returns:
            Validation results
        """
        issues = {
            "missing_images": [],
            "invalid_boxes": [],
            "out_of_bounds": []
        }

        for img_path, ann in self._annotations.items():
            if not Path(img_path).exists():
                issues["missing_images"].append(img_path)
                continue

            for i, box in enumerate(ann.boxes):
                if box.x1 >= box.x2 or box.y1 >= box.y2:
                    issues["invalid_boxes"].append({
                        "image": img_path,
                        "box_index": i,
                        "reason": "zero or negative dimensions"
                    })

                if (box.x1 < 0 or box.y1 < 0 or
                    box.x2 > ann.image_width or box.y2 > ann.image_height):
                    issues["out_of_bounds"].append({
                        "image": img_path,
                        "box_index": i
                    })

        issues["valid"] = (
            len(issues["missing_images"]) == 0 and
            len(issues["invalid_boxes"]) == 0
        )

        return issues


def main():
    """CLI for annotation management."""
    import argparse

    parser = argparse.ArgumentParser(description="Annotation Store")
    subparsers = parser.add_subparsers(dest="command")

    import_parser = subparsers.add_parser("import", help="Import annotations")
    import_parser.add_argument("--format", "-f", choices=["yolo"], default="yolo")
    import_parser.add_argument("--labels", "-l", required=True, help="Labels directory")
    import_parser.add_argument("--images", "-i", required=True, help="Images directory")

    export_parser = subparsers.add_parser("export", help="Export annotations")
    export_parser.add_argument("--format", "-f", choices=["yolo"], default="yolo")
    export_parser.add_argument("--output", "-o", required=True, help="Output directory")

    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    validate_parser = subparsers.add_parser("validate", help="Validate annotations")

    args = parser.parse_args()
    store = AnnotationStore()

    if args.command == "import":
        if args.format == "yolo":
            store.import_yolo(args.labels, args.images)

    elif args.command == "export":
        if args.format == "yolo":
            store.export_yolo(args.output)

    elif args.command == "stats":
        stats = store.get_statistics()
        print(json.dumps(stats, indent=2))

    elif args.command == "validate":
        results = store.validate()
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
