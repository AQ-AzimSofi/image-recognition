#!/usr/bin/env python3
"""
Re-ID data preparation module.

Prepares training data for person re-identification:
- Crop persons from images
- Cluster similar persons
- Manual/semi-automatic identity assignment
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import shutil


@dataclass
class PersonCrop:
    """Cropped person image info."""
    crop_path: str
    source_image: str
    bbox: List[float]
    confidence: float
    features: Optional[np.ndarray] = None
    cluster_id: Optional[int] = None
    person_id: Optional[str] = None


class ReIDDataPrep:
    """
    Prepare data for Re-ID training.

    Features:
    - Crop persons from detection results
    - Extract features for clustering
    - Cluster similar persons
    - Export in Re-ID training format
    """

    def __init__(
        self,
        output_dir: str = "data/reid_prep",
        min_crop_size: int = 64,
        padding: float = 0.1
    ):
        """
        Initialize Re-ID data preparation.

        Args:
            output_dir: Output directory for crops
            min_crop_size: Minimum crop dimension (skip smaller)
            padding: Padding ratio around bounding box
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_crop_size = min_crop_size
        self.padding = padding

        self.crops: List[PersonCrop] = []
        self.feature_extractor = None

    def _load_feature_extractor(self):
        """Load feature extractor for clustering."""
        if self.feature_extractor is not None:
            return

        import torch
        from torchvision import models, transforms

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        self.feature_extractor = self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def crop_from_image(
        self,
        image_path: str,
        boxes: List[List[float]],
        output_subdir: str = None
    ) -> List[PersonCrop]:
        """
        Crop persons from a single image.

        Args:
            image_path: Path to image
            boxes: List of bounding boxes [x1, y1, x2, y2, conf]
            output_subdir: Subdirectory for crops

        Returns:
            List of PersonCrop objects
        """
        img = cv2.imread(image_path)
        if img is None:
            return []

        h, w = img.shape[:2]
        crops_dir = self.output_dir / "crops"
        if output_subdir:
            crops_dir = crops_dir / output_subdir
        crops_dir.mkdir(parents=True, exist_ok=True)

        crops = []
        img_name = Path(image_path).stem

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            conf = box[4] if len(box) > 4 else 1.0

            # Add padding
            box_w = x2 - x1
            box_h = y2 - y1

            if box_w < self.min_crop_size or box_h < self.min_crop_size:
                continue

            pad_x = int(box_w * self.padding)
            pad_y = int(box_h * self.padding)

            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)

            # Crop
            crop = img[y1:y2, x1:x2]

            # Save crop
            crop_filename = f"{img_name}_{i:03d}.jpg"
            crop_path = crops_dir / crop_filename
            cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

            person_crop = PersonCrop(
                crop_path=str(crop_path),
                source_image=image_path,
                bbox=box[:4],
                confidence=conf
            )
            crops.append(person_crop)
            self.crops.append(person_crop)

        return crops

    def crop_from_detections(
        self,
        image_dir: str,
        detection_func=None,
        confidence: float = 0.5
    ) -> List[PersonCrop]:
        """
        Detect and crop persons from all images in directory.

        Args:
            image_dir: Directory containing images
            detection_func: Optional custom detection function
            confidence: Detection confidence threshold

        Returns:
            List of all PersonCrop objects
        """
        from ultralytics import YOLO

        image_dir = Path(image_dir)
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

        image_paths = sorted([
            p for p in image_dir.glob("*")
            if p.suffix.lower() in extensions
        ])

        print(f"Processing {len(image_paths)} images...")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from config import get_model_path

        model = YOLO(get_model_path())
        all_crops = []

        for i, img_path in enumerate(image_paths):
            results = model(str(img_path), conf=confidence, classes=[0], verbose=False)

            boxes = []
            for r in results:
                for box in r.boxes:
                    boxes.append(box.xyxy[0].cpu().tolist() + [float(box.conf)])

            crops = self.crop_from_image(str(img_path), boxes)
            all_crops.extend(crops)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(image_paths)}, crops: {len(all_crops)}")

        print(f"Done. Total crops: {len(all_crops)}")
        return all_crops

    def extract_features(self, crops: List[PersonCrop] = None) -> np.ndarray:
        """
        Extract features for all crops.

        Args:
            crops: List of crops (default: all crops)

        Returns:
            Feature matrix (N x D)
        """
        import torch

        self._load_feature_extractor()

        crops = crops or self.crops
        if not crops:
            return np.array([])

        print(f"Extracting features for {len(crops)} crops...")

        features = []
        batch_size = 32

        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i:i + batch_size]
            batch_images = []

            for crop in batch_crops:
                img = cv2.imread(crop.crop_path)
                if img is None:
                    batch_images.append(torch.zeros(3, 256, 128))
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tensor = self.transform(img)
                batch_images.append(tensor)

            batch_tensor = torch.stack(batch_images).to(self.device)

            with torch.no_grad():
                batch_features = self.feature_extractor(batch_tensor)
                batch_features = batch_features.squeeze(-1).squeeze(-1)
                batch_features = torch.nn.functional.normalize(batch_features, p=2, dim=1)

            features.append(batch_features.cpu().numpy())

            for j, crop in enumerate(batch_crops):
                crop.features = batch_features[j].cpu().numpy()

        feature_matrix = np.vstack(features)
        print(f"Extracted features: {feature_matrix.shape}")

        return feature_matrix

    def cluster_crops(
        self,
        n_clusters: int = None,
        min_cluster_size: int = 2,
        method: str = "agglomerative"
    ) -> Dict[int, List[PersonCrop]]:
        """
        Cluster crops by visual similarity.

        Args:
            n_clusters: Number of clusters (auto if None)
            min_cluster_size: Minimum samples per cluster
            method: Clustering method (agglomerative, kmeans, dbscan)

        Returns:
            Dict mapping cluster_id to list of crops
        """
        if not self.crops:
            return {}

        # Extract features if not done
        features = self.extract_features()

        print(f"Clustering {len(self.crops)} crops...")

        if method == "agglomerative":
            from sklearn.cluster import AgglomerativeClustering

            if n_clusters is None:
                # Estimate clusters based on feature similarity
                n_clusters = max(2, len(self.crops) // 5)

            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric="cosine",
                linkage="average"
            )
            labels = clustering.fit_predict(features)

        elif method == "kmeans":
            from sklearn.cluster import KMeans

            if n_clusters is None:
                n_clusters = max(2, len(self.crops) // 5)

            clustering = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clustering.fit_predict(features)

        elif method == "dbscan":
            from sklearn.cluster import DBSCAN

            clustering = DBSCAN(eps=0.3, min_samples=min_cluster_size, metric="cosine")
            labels = clustering.fit_predict(features)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Assign cluster IDs
        clusters = {}
        for i, (crop, label) in enumerate(zip(self.crops, labels)):
            crop.cluster_id = int(label)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(crop)

        # Filter small clusters
        clusters = {k: v for k, v in clusters.items() if len(v) >= min_cluster_size}

        print(f"Found {len(clusters)} clusters")
        for cid, crops in sorted(clusters.items()):
            print(f"  Cluster {cid}: {len(crops)} crops")

        return clusters

    def export_for_review(
        self,
        clusters: Dict[int, List[PersonCrop]],
        output_dir: str = None
    ) -> str:
        """
        Export clustered crops for manual review.

        Args:
            clusters: Clustered crops
            output_dir: Output directory

        Returns:
            Path to output directory
        """
        output_dir = Path(output_dir or self.output_dir / "review")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create directories for each cluster
        for cluster_id, crops in clusters.items():
            cluster_dir = output_dir / f"cluster_{cluster_id:03d}"
            cluster_dir.mkdir(exist_ok=True)

            for i, crop in enumerate(crops):
                src = Path(crop.crop_path)
                dst = cluster_dir / f"{i:03d}_{src.name}"
                shutil.copy2(src, dst)

        # Create review instructions
        instructions = output_dir / "README.txt"
        with open(instructions, "w") as f:
            f.write("""Re-ID Data Review Instructions
==============================

Each folder (cluster_XXX) contains crops that the system thinks
are the same person.

Your tasks:
1. Review each cluster folder
2. Remove incorrect crops (different persons)
3. Merge folders if they contain the same person
4. Rename folders to person IDs (e.g., person_001, person_002)

After review:
- Run the export command to create the training dataset
- Each folder name becomes the person ID

Tips:
- Look for clothing, body shape, accessories
- Consider time/location if available
- When unsure, keep separate (better to under-merge)
""")

        # Create summary
        summary = {
            "created_at": datetime.now().isoformat(),
            "total_clusters": len(clusters),
            "total_crops": sum(len(c) for c in clusters.values()),
            "clusters": {
                f"cluster_{cid:03d}": len(crops)
                for cid, crops in sorted(clusters.items())
            }
        }

        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Exported for review: {output_dir}")
        print(f"  Clusters: {len(clusters)}")
        print(f"  Total crops: {sum(len(c) for c in clusters.values())}")

        return str(output_dir)

    def export_training_dataset(
        self,
        reviewed_dir: str,
        output_dir: str = None,
        train_ratio: float = 0.8
    ) -> str:
        """
        Export reviewed data as Re-ID training dataset.

        Args:
            reviewed_dir: Directory with reviewed/renamed clusters
            output_dir: Output directory
            train_ratio: Train/query split ratio

        Returns:
            Path to training dataset
        """
        import random

        reviewed_dir = Path(reviewed_dir)
        output_dir = Path(output_dir or self.output_dir / "reid_dataset")

        train_dir = output_dir / "train"
        query_dir = output_dir / "query"
        gallery_dir = output_dir / "gallery"

        for d in [train_dir, query_dir, gallery_dir]:
            d.mkdir(parents=True, exist_ok=True)

        stats = {"persons": 0, "train": 0, "query": 0, "gallery": 0}

        for person_dir in sorted(reviewed_dir.iterdir()):
            if not person_dir.is_dir() or person_dir.name.startswith("."):
                continue

            person_id = person_dir.name
            images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))

            if len(images) < 2:
                continue

            stats["persons"] += 1

            # Create person directories
            (train_dir / person_id).mkdir(exist_ok=True)
            (query_dir / person_id).mkdir(exist_ok=True)
            (gallery_dir / person_id).mkdir(exist_ok=True)

            # Split images
            random.shuffle(images)
            n_train = max(1, int(len(images) * train_ratio))

            train_images = images[:n_train]
            test_images = images[n_train:]

            # Copy train images
            for i, img in enumerate(train_images):
                dst = train_dir / person_id / f"{i:04d}.jpg"
                shutil.copy2(img, dst)
                stats["train"] += 1

            # Split test into query and gallery
            if test_images:
                query_img = test_images[0]
                gallery_imgs = test_images[1:] if len(test_images) > 1 else test_images

                shutil.copy2(query_img, query_dir / person_id / "0000.jpg")
                stats["query"] += 1

                for i, img in enumerate(gallery_imgs):
                    shutil.copy2(img, gallery_dir / person_id / f"{i:04d}.jpg")
                    stats["gallery"] += 1

        # Save dataset info
        info = {
            "created_at": datetime.now().isoformat(),
            "statistics": stats,
            "splits": {
                "train": str(train_dir),
                "query": str(query_dir),
                "gallery": str(gallery_dir)
            }
        }

        with open(output_dir / "dataset_info.json", "w") as f:
            json.dump(info, f, indent=2)

        print(f"Created Re-ID dataset: {output_dir}")
        print(f"  Persons: {stats['persons']}")
        print(f"  Train: {stats['train']} images")
        print(f"  Query: {stats['query']} images")
        print(f"  Gallery: {stats['gallery']} images")

        return str(output_dir)

    def get_statistics(self) -> Dict:
        """Get current statistics."""
        cluster_counts = {}
        for crop in self.crops:
            if crop.cluster_id is not None:
                cluster_counts[crop.cluster_id] = cluster_counts.get(crop.cluster_id, 0) + 1

        return {
            "total_crops": len(self.crops),
            "crops_with_features": sum(1 for c in self.crops if c.features is not None),
            "crops_clustered": sum(1 for c in self.crops if c.cluster_id is not None),
            "num_clusters": len(cluster_counts),
            "cluster_sizes": dict(sorted(cluster_counts.items()))
        }


def main():
    """CLI for Re-ID data preparation."""
    import argparse

    parser = argparse.ArgumentParser(description="Re-ID Data Preparation")
    subparsers = parser.add_subparsers(dest="command")

    crop_parser = subparsers.add_parser("crop", help="Crop persons from images")
    crop_parser.add_argument("image_dir", help="Image directory")
    crop_parser.add_argument("--output", "-o", default="data/reid_prep")
    crop_parser.add_argument("--confidence", "-c", type=float, default=0.5)

    cluster_parser = subparsers.add_parser("cluster", help="Cluster crops")
    cluster_parser.add_argument("crops_dir", help="Crops directory")
    cluster_parser.add_argument("--n-clusters", "-n", type=int, default=None)
    cluster_parser.add_argument("--method", "-m", default="agglomerative")
    cluster_parser.add_argument("--output", "-o", default="data/reid_prep/review")

    export_parser = subparsers.add_parser("export", help="Export training dataset")
    export_parser.add_argument("reviewed_dir", help="Reviewed clusters directory")
    export_parser.add_argument("--output", "-o", default="data/reid_dataset")

    args = parser.parse_args()

    if args.command == "crop":
        prep = ReIDDataPrep(output_dir=args.output)
        prep.crop_from_detections(args.image_dir, confidence=args.confidence)

    elif args.command == "cluster":
        prep = ReIDDataPrep(output_dir=Path(args.crops_dir).parent)
        # Load existing crops
        crops_dir = Path(args.crops_dir)
        for crop_path in sorted(crops_dir.glob("**/*.jpg")):
            prep.crops.append(PersonCrop(
                crop_path=str(crop_path),
                source_image="",
                bbox=[0, 0, 0, 0],
                confidence=1.0
            ))

        clusters = prep.cluster_crops(n_clusters=args.n_clusters, method=args.method)
        prep.export_for_review(clusters, args.output)

    elif args.command == "export":
        prep = ReIDDataPrep()
        prep.export_training_dataset(args.reviewed_dir, args.output)


if __name__ == "__main__":
    main()
