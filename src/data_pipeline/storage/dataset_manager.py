#!/usr/bin/env python3
"""
Dataset management module.

Manages dataset structure, splits (train/val/test),
and provides utilities for dataset operations.
"""

import shutil
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class DatasetSplit:
    """Dataset split information."""
    name: str
    images: List[str] = field(default_factory=list)
    annotations: List[str] = field(default_factory=list)
    count: int = 0


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str
    version: str = "1.0"
    description: str = ""
    classes: List[str] = field(default_factory=lambda: ["person"])
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1


class DatasetManager:
    """
    Manage datasets for training.

    Features:
    - Dataset creation and versioning
    - Train/val/test splitting
    - YOLO format export
    - COCO format export
    - Dataset merging
    """

    def __init__(self, base_dir: str = "data/training_datasets"):
        """
        Initialize dataset manager.

        Args:
            base_dir: Base directory for datasets
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_dataset(
        self,
        name: str,
        config: DatasetConfig = None
    ) -> Path:
        """
        Create a new dataset structure.

        Args:
            name: Dataset name
            config: Dataset configuration

        Returns:
            Path to dataset directory
        """
        config = config or DatasetConfig(name=name)
        dataset_dir = self.base_dir / name

        if dataset_dir.exists():
            raise ValueError(f"Dataset already exists: {name}")

        for split in ["train", "val", "test"]:
            (dataset_dir / split / "images").mkdir(parents=True)
            (dataset_dir / split / "labels").mkdir(parents=True)

        config_data = {
            "name": config.name,
            "version": config.version,
            "description": config.description,
            "classes": config.classes,
            "created_at": datetime.now().isoformat(),
            "splits": {
                "train": config.train_ratio,
                "val": config.val_ratio,
                "test": config.test_ratio
            }
        }

        with open(dataset_dir / "dataset.json", "w") as f:
            json.dump(config_data, f, indent=2)

        self._create_yolo_yaml(dataset_dir, config)

        print(f"Created dataset: {dataset_dir}")
        return dataset_dir

    def _create_yolo_yaml(self, dataset_dir: Path, config: DatasetConfig):
        """Create YOLO dataset.yaml file."""
        yaml_content = f"""# YOLO Dataset Configuration
# Generated: {datetime.now().isoformat()}

path: {dataset_dir.absolute()}
train: train/images
val: val/images
test: test/images

nc: {len(config.classes)}
names: {config.classes}
"""
        with open(dataset_dir / "dataset.yaml", "w") as f:
            f.write(yaml_content)

    def add_images(
        self,
        dataset_name: str,
        image_paths: List[str],
        annotation_paths: List[str] = None,
        split: str = None,
        auto_split: bool = True
    ) -> Dict[str, int]:
        """
        Add images to dataset.

        Args:
            dataset_name: Dataset name
            image_paths: List of image paths
            annotation_paths: List of annotation paths (YOLO format)
            split: Target split (train/val/test) or None for auto
            auto_split: Automatically distribute to splits

        Returns:
            Dict with counts per split
        """
        dataset_dir = self.base_dir / dataset_name

        if not dataset_dir.exists():
            raise ValueError(f"Dataset not found: {dataset_name}")

        with open(dataset_dir / "dataset.json", "r") as f:
            config = json.load(f)

        if annotation_paths is None:
            annotation_paths = [None] * len(image_paths)

        pairs = list(zip(image_paths, annotation_paths))

        if auto_split and split is None:
            random.shuffle(pairs)
            n = len(pairs)
            train_end = int(n * config["splits"]["train"])
            val_end = train_end + int(n * config["splits"]["val"])

            split_assignments = (
                ["train"] * train_end +
                ["val"] * (val_end - train_end) +
                ["test"] * (n - val_end)
            )
        else:
            split_assignments = [split or "train"] * len(pairs)

        counts = {"train": 0, "val": 0, "test": 0}

        for (img_path, ann_path), target_split in zip(pairs, split_assignments):
            img_path = Path(img_path)
            dest_img = dataset_dir / target_split / "images" / img_path.name

            shutil.copy2(img_path, dest_img)

            if ann_path:
                ann_path = Path(ann_path)
                dest_ann = dataset_dir / target_split / "labels" / ann_path.name
                shutil.copy2(ann_path, dest_ann)

            counts[target_split] += 1

        self._update_dataset_stats(dataset_dir)
        return counts

    def _update_dataset_stats(self, dataset_dir: Path):
        """Update dataset statistics."""
        with open(dataset_dir / "dataset.json", "r") as f:
            config = json.load(f)

        stats = {}
        for split in ["train", "val", "test"]:
            images_dir = dataset_dir / split / "images"
            labels_dir = dataset_dir / split / "labels"

            image_count = len(list(images_dir.glob("*")))
            label_count = len(list(labels_dir.glob("*.txt")))

            stats[split] = {
                "images": image_count,
                "labels": label_count
            }

        config["stats"] = stats
        config["updated_at"] = datetime.now().isoformat()

        with open(dataset_dir / "dataset.json", "w") as f:
            json.dump(config, f, indent=2)

    def get_dataset_info(self, dataset_name: str) -> Dict:
        """
        Get dataset information.

        Args:
            dataset_name: Dataset name

        Returns:
            Dataset info dict
        """
        dataset_dir = self.base_dir / dataset_name

        if not dataset_dir.exists():
            raise ValueError(f"Dataset not found: {dataset_name}")

        with open(dataset_dir / "dataset.json", "r") as f:
            return json.load(f)

    def list_datasets(self) -> List[Dict]:
        """
        List all datasets.

        Returns:
            List of dataset info dicts
        """
        datasets = []

        for dataset_dir in self.base_dir.iterdir():
            if dataset_dir.is_dir():
                config_path = dataset_dir / "dataset.json"
                if config_path.exists():
                    with open(config_path, "r") as f:
                        datasets.append(json.load(f))

        return datasets

    def get_split_files(
        self,
        dataset_name: str,
        split: str
    ) -> Tuple[List[Path], List[Path]]:
        """
        Get image and label files for a split.

        Args:
            dataset_name: Dataset name
            split: Split name (train/val/test)

        Returns:
            Tuple of (image_paths, label_paths)
        """
        dataset_dir = self.base_dir / dataset_name
        images_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"

        images = sorted(images_dir.glob("*"))
        labels = []

        for img in images:
            label_path = labels_dir / f"{img.stem}.txt"
            labels.append(label_path if label_path.exists() else None)

        return images, labels

    def export_to_coco(
        self,
        dataset_name: str,
        output_path: str = None
    ) -> str:
        """
        Export dataset to COCO format.

        Args:
            dataset_name: Dataset name
            output_path: Output JSON path

        Returns:
            Path to COCO JSON file
        """
        dataset_dir = self.base_dir / dataset_name

        if output_path is None:
            output_path = dataset_dir / "annotations_coco.json"

        with open(dataset_dir / "dataset.json", "r") as f:
            config = json.load(f)

        coco = {
            "info": {
                "description": config.get("description", ""),
                "version": config.get("version", "1.0"),
                "year": datetime.now().year,
                "date_created": datetime.now().isoformat()
            },
            "licenses": [],
            "categories": [
                {"id": i, "name": name, "supercategory": ""}
                for i, name in enumerate(config.get("classes", ["person"]))
            ],
            "images": [],
            "annotations": []
        }

        image_id = 0
        annotation_id = 0

        for split in ["train", "val", "test"]:
            images, labels = self.get_split_files(dataset_name, split)

            for img_path, label_path in zip(images, labels):
                if not img_path.exists():
                    continue

                from PIL import Image
                with Image.open(img_path) as img:
                    width, height = img.size

                coco["images"].append({
                    "id": image_id,
                    "file_name": str(img_path.relative_to(dataset_dir)),
                    "width": width,
                    "height": height
                })

                if label_path and label_path.exists():
                    with open(label_path, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                w = float(parts[3])
                                h = float(parts[4])

                                x = (x_center - w / 2) * width
                                y = (y_center - h / 2) * height
                                box_w = w * width
                                box_h = h * height

                                coco["annotations"].append({
                                    "id": annotation_id,
                                    "image_id": image_id,
                                    "category_id": class_id,
                                    "bbox": [x, y, box_w, box_h],
                                    "area": box_w * box_h,
                                    "iscrowd": 0
                                })
                                annotation_id += 1

                image_id += 1

        with open(output_path, "w") as f:
            json.dump(coco, f, indent=2)

        print(f"Exported COCO format to: {output_path}")
        return str(output_path)

    def merge_datasets(
        self,
        source_datasets: List[str],
        target_name: str,
        rebalance: bool = True
    ) -> Path:
        """
        Merge multiple datasets into one.

        Args:
            source_datasets: List of source dataset names
            target_name: Name for merged dataset
            rebalance: Rebalance train/val/test splits

        Returns:
            Path to merged dataset
        """
        all_images = []
        all_labels = []

        for ds_name in source_datasets:
            for split in ["train", "val", "test"]:
                images, labels = self.get_split_files(ds_name, split)
                for img, lbl in zip(images, labels):
                    if img.exists():
                        all_images.append(str(img))
                        all_labels.append(str(lbl) if lbl and lbl.exists() else None)

        config = DatasetConfig(name=target_name)
        target_dir = self.create_dataset(target_name, config)

        self.add_images(
            target_name,
            all_images,
            all_labels,
            auto_split=rebalance
        )

        return target_dir

    def delete_dataset(self, dataset_name: str, confirm: bool = False):
        """
        Delete a dataset.

        Args:
            dataset_name: Dataset name
            confirm: Require confirmation
        """
        dataset_dir = self.base_dir / dataset_name

        if not dataset_dir.exists():
            raise ValueError(f"Dataset not found: {dataset_name}")

        if not confirm:
            raise ValueError("Set confirm=True to delete dataset")

        shutil.rmtree(dataset_dir)
        print(f"Deleted dataset: {dataset_name}")


def main():
    """CLI for dataset management."""
    import argparse

    parser = argparse.ArgumentParser(description="Dataset Management")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    create_parser = subparsers.add_parser("create", help="Create dataset")
    create_parser.add_argument("name", help="Dataset name")
    create_parser.add_argument("--description", "-d", default="")

    list_parser = subparsers.add_parser("list", help="List datasets")

    info_parser = subparsers.add_parser("info", help="Dataset info")
    info_parser.add_argument("name", help="Dataset name")

    add_parser = subparsers.add_parser("add", help="Add images")
    add_parser.add_argument("name", help="Dataset name")
    add_parser.add_argument("images", help="Image directory")
    add_parser.add_argument("--labels", "-l", help="Labels directory")
    add_parser.add_argument("--split", "-s", choices=["train", "val", "test"])

    export_parser = subparsers.add_parser("export", help="Export to COCO")
    export_parser.add_argument("name", help="Dataset name")
    export_parser.add_argument("--output", "-o", help="Output path")

    args = parser.parse_args()
    manager = DatasetManager()

    if args.command == "create":
        config = DatasetConfig(name=args.name, description=args.description)
        manager.create_dataset(args.name, config)

    elif args.command == "list":
        datasets = manager.list_datasets()
        for ds in datasets:
            stats = ds.get("stats", {})
            total = sum(s.get("images", 0) for s in stats.values())
            print(f"  {ds['name']}: {total} images")

    elif args.command == "info":
        info = manager.get_dataset_info(args.name)
        print(json.dumps(info, indent=2))

    elif args.command == "add":
        images_dir = Path(args.images)
        images = sorted([str(p) for p in images_dir.glob("*") if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])

        labels = None
        if args.labels:
            labels_dir = Path(args.labels)
            labels = [str(labels_dir / f"{Path(img).stem}.txt") for img in images]

        counts = manager.add_images(args.name, images, labels, split=args.split)
        print(f"Added: {counts}")

    elif args.command == "export":
        manager.export_to_coco(args.name, args.output)


if __name__ == "__main__":
    main()
