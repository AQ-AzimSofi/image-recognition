#!/usr/bin/env python3
"""
Person Re-Identification module.

Uses deep learning features to identify the same person across different images/cameras.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from utils import load_image


@dataclass
class PersonDetection:
    """Represents a detected person with their features."""
    box: List[float]
    confidence: float
    features: Optional[np.ndarray] = None
    person_id: Optional[str] = None
    image_path: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class MatchResult:
    """Result of matching persons between two sets."""
    matches: List[Tuple[int, int, float]]
    unmatched_first: List[int] = field(default_factory=list)
    unmatched_second: List[int] = field(default_factory=list)


class PersonReID:
    """
    Person Re-Identification using ResNet50 features.

    Extracts visual features from person crops and matches them
    across different images/cameras using cosine similarity.
    """

    def __init__(self, model_name: str = "resnet50", device: str = None):
        """
        Initialize the Re-ID model.

        Args:
            model_name: Feature extraction model (resnet50, resnet18, efficientnet)
            device: Device to run on (cuda/cpu, auto-detect if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = self._load_model(model_name)
        self.transform = self._get_transform()
        self.feature_dim = self._get_feature_dim()

    def _load_model(self, model_name: str) -> nn.Module:
        """Load pre-trained model for feature extraction."""
        if model_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            model = nn.Sequential(*list(model.children())[:-1])
        elif model_name == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            model = nn.Sequential(*list(model.children())[:-1])
        elif model_name == "efficientnet":
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            model = nn.Sequential(*list(model.children())[:-1])
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model = model.to(self.device)
        model.eval()
        return model

    def _get_transform(self) -> transforms.Compose:
        """Get image transformation pipeline."""
        return transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _get_feature_dim(self) -> int:
        """Get feature vector dimension."""
        if self.model_name in ["resnet50"]:
            return 2048
        elif self.model_name == "resnet18":
            return 512
        elif self.model_name == "efficientnet":
            return 1280
        return 2048

    def _crop_person(self, image: np.ndarray, box: List[float], padding: float = 0.1) -> Image.Image:
        """
        Crop person region from image with padding.

        Args:
            image: Full image (BGR format from OpenCV)
            box: Bounding box [x1, y1, x2, y2, ...]
            padding: Padding ratio to add around the box
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box[:4]]

        box_w = x2 - x1
        box_h = y2 - y1
        pad_x = int(box_w * padding)
        pad_y = int(box_h * padding)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        crop = image[y1:y2, x1:x2]
        crop_rgb = crop[:, :, ::-1]
        return Image.fromarray(crop_rgb)

    @torch.no_grad()
    def extract_features(self, image: np.ndarray, boxes: List[List[float]]) -> List[np.ndarray]:
        """
        Extract feature vectors for each detected person.

        Args:
            image: Full image (BGR format)
            boxes: List of bounding boxes [[x1, y1, x2, y2, conf], ...]

        Returns:
            List of feature vectors (one per person)
        """
        if not boxes:
            return []

        features = []
        batch = []

        for box in boxes:
            crop = self._crop_person(image, box)
            tensor = self.transform(crop)
            batch.append(tensor)

        batch_tensor = torch.stack(batch).to(self.device)

        output = self.model(batch_tensor)
        output = output.squeeze(-1).squeeze(-1)

        output = nn.functional.normalize(output, p=2, dim=1)

        features = output.cpu().numpy()
        return [f for f in features]

    def extract_features_from_path(self, image_path: str, boxes: List[List[float]]) -> List[np.ndarray]:
        """Extract features from image file path."""
        image = load_image(image_path)
        return self.extract_features(image, boxes)

    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Compute cosine similarity between two feature vectors.

        Args:
            feat1: First feature vector
            feat2: Second feature vector

        Returns:
            Similarity score (0.0 to 1.0)
        """
        return float(np.dot(feat1, feat2))

    def compute_similarity_matrix(self, features1: List[np.ndarray], features2: List[np.ndarray]) -> np.ndarray:
        """
        Compute pairwise similarity matrix between two sets of features.

        Args:
            features1: First set of feature vectors
            features2: Second set of feature vectors

        Returns:
            Similarity matrix (len(features1) x len(features2))
        """
        if not features1 or not features2:
            return np.array([])

        f1 = np.stack(features1)
        f2 = np.stack(features2)

        return np.dot(f1, f2.T)

    def match_persons(
        self,
        features1: List[np.ndarray],
        features2: List[np.ndarray],
        threshold: float = 0.6
    ) -> MatchResult:
        """
        Match persons between two sets using Hungarian algorithm.

        Args:
            features1: Features from first image/camera
            features2: Features from second image/camera
            threshold: Minimum similarity to consider a match

        Returns:
            MatchResult with matches and unmatched indices
        """
        if not features1 or not features2:
            return MatchResult(
                matches=[],
                unmatched_first=list(range(len(features1))),
                unmatched_second=list(range(len(features2)))
            )

        sim_matrix = self.compute_similarity_matrix(features1, features2)

        from scipy.optimize import linear_sum_assignment
        cost_matrix = 1 - sim_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_first = set(range(len(features1)))
        unmatched_second = set(range(len(features2)))

        for i, j in zip(row_ind, col_ind):
            similarity = sim_matrix[i, j]
            if similarity >= threshold:
                matches.append((i, j, float(similarity)))
                unmatched_first.discard(i)
                unmatched_second.discard(j)

        return MatchResult(
            matches=matches,
            unmatched_first=sorted(unmatched_first),
            unmatched_second=sorted(unmatched_second)
        )


class PersonTracker:
    """
    Track persons across a sequence of images.

    Assigns consistent IDs to the same person appearing in multiple frames.
    """

    def __init__(self, reid_model: PersonReID, similarity_threshold: float = 0.6):
        """
        Initialize tracker.

        Args:
            reid_model: PersonReID instance for feature extraction
            similarity_threshold: Minimum similarity to match persons
        """
        self.reid = reid_model
        self.threshold = similarity_threshold
        self.next_id = 1
        self.person_gallery: Dict[str, np.ndarray] = {}
        self.id_last_seen: Dict[str, int] = {}

    def _generate_id(self) -> str:
        """Generate a new unique person ID."""
        person_id = f"P{self.next_id:04d}"
        self.next_id += 1
        return person_id

    def process_frame(
        self,
        image: np.ndarray,
        boxes: List[List[float]],
        frame_index: int = 0
    ) -> List[PersonDetection]:
        """
        Process a single frame and assign person IDs.

        Args:
            image: Frame image (BGR format)
            boxes: Detected person boxes
            frame_index: Current frame index (for tracking)

        Returns:
            List of PersonDetection with assigned IDs
        """
        if not boxes:
            return []

        features = self.reid.extract_features(image, boxes)

        detections = []
        for box, feat in zip(boxes, features):
            detection = PersonDetection(
                box=box[:4],
                confidence=box[4] if len(box) > 4 else 1.0,
                features=feat
            )
            detections.append(detection)

        if not self.person_gallery:
            for det in detections:
                person_id = self._generate_id()
                det.person_id = person_id
                self.person_gallery[person_id] = det.features
                self.id_last_seen[person_id] = frame_index
        else:
            gallery_ids = list(self.person_gallery.keys())
            gallery_features = [self.person_gallery[pid] for pid in gallery_ids]
            current_features = [d.features for d in detections]

            match_result = self.reid.match_persons(
                current_features,
                gallery_features,
                self.threshold
            )

            for curr_idx, gal_idx, sim in match_result.matches:
                person_id = gallery_ids[gal_idx]
                detections[curr_idx].person_id = person_id
                alpha = 0.3
                self.person_gallery[person_id] = (
                    alpha * detections[curr_idx].features +
                    (1 - alpha) * self.person_gallery[person_id]
                )
                self.person_gallery[person_id] /= np.linalg.norm(self.person_gallery[person_id])
                self.id_last_seen[person_id] = frame_index

            for curr_idx in match_result.unmatched_first:
                person_id = self._generate_id()
                detections[curr_idx].person_id = person_id
                self.person_gallery[person_id] = detections[curr_idx].features
                self.id_last_seen[person_id] = frame_index

        return detections

    def process_sequence(
        self,
        image_paths: List[str],
        detections_per_frame: List[List[List[float]]]
    ) -> Dict[str, List[PersonDetection]]:
        """
        Process a sequence of frames.

        Args:
            image_paths: List of image file paths
            detections_per_frame: List of detection boxes per frame

        Returns:
            Dict mapping image path to list of PersonDetections
        """
        results = {}

        for idx, (path, boxes) in enumerate(zip(image_paths, detections_per_frame)):
            image = load_image(path)
            detections = self.process_frame(image, boxes, idx)

            for det in detections:
                det.image_path = path

            results[path] = detections

        return results

    def get_unique_person_count(self) -> int:
        """Get total number of unique persons tracked."""
        return len(self.person_gallery)

    def reset(self):
        """Reset tracker state."""
        self.next_id = 1
        self.person_gallery.clear()
        self.id_last_seen.clear()


def match_across_cameras(
    reid_model: PersonReID,
    camera_detections: Dict[str, List[PersonDetection]],
    threshold: float = 0.5
) -> Dict[str, str]:
    """
    Match persons across multiple cameras at the same time.

    Args:
        reid_model: PersonReID instance
        camera_detections: Dict mapping camera_id to list of detections
        threshold: Similarity threshold for matching

    Returns:
        Dict mapping local person_id to global person_id
    """
    all_detections = []
    detection_sources = []

    for cam_id, detections in camera_detections.items():
        for det in detections:
            all_detections.append(det)
            detection_sources.append(cam_id)

    if len(all_detections) <= 1:
        return {d.person_id: d.person_id for d in all_detections if d.person_id}

    features = [d.features for d in all_detections]
    sim_matrix = reid_model.compute_similarity_matrix(features, features)

    n = len(all_detections)
    global_id_map = {}
    next_global_id = 1
    assigned = [False] * n

    for i in range(n):
        if assigned[i]:
            continue

        global_id = f"G{next_global_id:04d}"
        next_global_id += 1

        if all_detections[i].person_id:
            global_id_map[all_detections[i].person_id] = global_id
        assigned[i] = True

        for j in range(i + 1, n):
            if assigned[j]:
                continue
            if detection_sources[i] == detection_sources[j]:
                continue
            if sim_matrix[i, j] >= threshold:
                if all_detections[j].person_id:
                    global_id_map[all_detections[j].person_id] = global_id
                assigned[j] = True

    for i, det in enumerate(all_detections):
        if not assigned[i] and det.person_id and det.person_id not in global_id_map:
            global_id = f"G{next_global_id:04d}"
            next_global_id += 1
            global_id_map[det.person_id] = global_id

    return global_id_map


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Person Re-ID")
    parser.add_argument("image1", help="First image path")
    parser.add_argument("image2", help="Second image path")
    parser.add_argument("--threshold", type=float, default=0.6, help="Similarity threshold")
    args = parser.parse_args()

    from detect import detect_persons
    from ultralytics import YOLO

    print("Loading models...")
    yolo = YOLO("yolo11n.pt")
    reid = PersonReID()

    print(f"\nProcessing {args.image1}...")
    count1, boxes1 = detect_persons(yolo, args.image1)
    img1 = load_image(args.image1)
    features1 = reid.extract_features(img1, boxes1)
    print(f"  Detected {count1} persons")

    print(f"\nProcessing {args.image2}...")
    count2, boxes2 = detect_persons(yolo, args.image2)
    img2 = load_image(args.image2)
    features2 = reid.extract_features(img2, boxes2)
    print(f"  Detected {count2} persons")

    print(f"\nMatching persons (threshold={args.threshold})...")
    result = reid.match_persons(features1, features2, args.threshold)

    print(f"\nMatches found: {len(result.matches)}")
    for i, j, sim in result.matches:
        print(f"  Image1 Person {i+1} <-> Image2 Person {j+1} (similarity: {sim:.3f})")

    if result.unmatched_first:
        print(f"\nUnmatched in Image1: {[i+1 for i in result.unmatched_first]}")
    if result.unmatched_second:
        print(f"Unmatched in Image2: {[i+1 for i in result.unmatched_second]}")
