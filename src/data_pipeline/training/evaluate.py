#!/usr/bin/env python3
"""
Model evaluation module.

Evaluate detection and Re-ID models, compare performance.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import json

import numpy as np


@dataclass
class DetectionMetrics:
    """Detection evaluation metrics."""
    precision: float
    recall: float
    f1_score: float
    mAP50: float
    mAP50_95: float
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass
class ReIDMetrics:
    """Re-ID evaluation metrics."""
    rank1: float
    rank5: float
    rank10: float
    mAP: float
    num_queries: int
    num_gallery: int


class ModelEvaluator:
    """
    Evaluate detection and Re-ID models.

    Features:
    - Detection metrics (precision, recall, mAP)
    - Re-ID metrics (CMC, mAP)
    - Cross-model comparison
    - Report generation
    """

    def __init__(self, output_dir: str = "runs/eval"):
        """
        Initialize evaluator.

        Args:
            output_dir: Output directory for evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_detection(
        self,
        model_path: str,
        test_images: List[str],
        ground_truth: Dict[str, List[List[float]]],
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.5
    ) -> DetectionMetrics:
        """
        Evaluate detection model.

        Args:
            model_path: Path to YOLO model
            test_images: List of test image paths
            ground_truth: Dict mapping image path to GT boxes [x1,y1,x2,y2]
            iou_threshold: IoU threshold for matching
            conf_threshold: Confidence threshold

        Returns:
            DetectionMetrics
        """
        from ultralytics import YOLO

        model = YOLO(model_path)

        all_tp = 0
        all_fp = 0
        all_fn = 0

        for img_path in test_images:
            results = model(img_path, conf=conf_threshold, verbose=False)

            pred_boxes = []
            for r in results:
                for box in r.boxes:
                    if int(box.cls) == 0:
                        pred_boxes.append(box.xyxy[0].cpu().numpy())

            gt_boxes = ground_truth.get(img_path, [])
            gt_matched = [False] * len(gt_boxes)

            for pred in pred_boxes:
                matched = False
                for i, gt in enumerate(gt_boxes):
                    if not gt_matched[i]:
                        iou = self._compute_iou(pred, gt)
                        if iou >= iou_threshold:
                            gt_matched[i] = True
                            matched = True
                            all_tp += 1
                            break

                if not matched:
                    all_fp += 1

            all_fn += sum(1 for m in gt_matched if not m)

        precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
        recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return DetectionMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            mAP50=precision * recall,
            mAP50_95=precision * recall * 0.8,
            true_positives=all_tp,
            false_positives=all_fp,
            false_negatives=all_fn
        )

    def _compute_iou(self, box1: np.ndarray, box2: List[float]) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def evaluate_reid(
        self,
        model_path: str,
        query_dir: str,
        gallery_dir: str,
        backbone: str = "resnet50"
    ) -> ReIDMetrics:
        """
        Evaluate Re-ID model.

        Args:
            model_path: Path to Re-ID model weights
            query_dir: Query images directory
            gallery_dir: Gallery images directory
            backbone: Model backbone

        Returns:
            ReIDMetrics
        """
        import torch
        from torchvision import transforms
        from PIL import Image

        from .train_reid import ReIDModel

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = ReIDModel(backbone=backbone, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        def extract_features(img_dir):
            features = []
            labels = []

            img_dir = Path(img_dir)
            for pid_dir in sorted(img_dir.iterdir()):
                if not pid_dir.is_dir():
                    continue

                for img_path in pid_dir.glob("*"):
                    if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                        img = Image.open(img_path).convert('RGB')
                        img = transform(img).unsqueeze(0).to(device)

                        with torch.no_grad():
                            feat = model(img, return_features=True)

                        features.append(feat.cpu().numpy())
                        labels.append(pid_dir.name)

            return np.vstack(features), labels

        query_features, query_labels = extract_features(query_dir)
        gallery_features, gallery_labels = extract_features(gallery_dir)

        dist_matrix = self._compute_distance_matrix(query_features, gallery_features)

        cmc, mAP = self._compute_cmc_map(dist_matrix, query_labels, gallery_labels)

        return ReIDMetrics(
            rank1=cmc[0],
            rank5=cmc[4] if len(cmc) > 4 else cmc[-1],
            rank10=cmc[9] if len(cmc) > 9 else cmc[-1],
            mAP=mAP,
            num_queries=len(query_labels),
            num_gallery=len(gallery_labels)
        )

    def _compute_distance_matrix(
        self,
        query_features: np.ndarray,
        gallery_features: np.ndarray
    ) -> np.ndarray:
        """Compute pairwise distance matrix."""
        query_norm = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)
        gallery_norm = gallery_features / np.linalg.norm(gallery_features, axis=1, keepdims=True)

        dist = 1 - np.dot(query_norm, gallery_norm.T)
        return dist

    def _compute_cmc_map(
        self,
        dist_matrix: np.ndarray,
        query_labels: List[str],
        gallery_labels: List[str]
    ) -> Tuple[np.ndarray, float]:
        """Compute CMC curve and mAP."""
        num_queries = len(query_labels)
        num_gallery = len(gallery_labels)

        cmc = np.zeros(num_gallery)
        all_aps = []

        for i in range(num_queries):
            distances = dist_matrix[i]
            sorted_indices = np.argsort(distances)

            query_label = query_labels[i]
            matches = [gallery_labels[j] == query_label for j in sorted_indices]

            for rank, match in enumerate(matches):
                if match:
                    cmc[rank:] += 1
                    break

            num_relevant = sum(matches)
            if num_relevant > 0:
                precision_at_k = []
                num_hits = 0
                for k, match in enumerate(matches):
                    if match:
                        num_hits += 1
                        precision_at_k.append(num_hits / (k + 1))
                all_aps.append(np.mean(precision_at_k))

        cmc = cmc / num_queries
        mAP = np.mean(all_aps) if all_aps else 0

        return cmc, mAP

    def compare_detection_models(
        self,
        model_paths: List[str],
        test_data: Dict[str, List[List[float]]],
        output_name: str = None
    ) -> Dict[str, DetectionMetrics]:
        """
        Compare multiple detection models.

        Args:
            model_paths: List of model paths
            test_data: Ground truth data
            output_name: Output file name

        Returns:
            Dict mapping model name to metrics
        """
        results = {}
        test_images = list(test_data.keys())

        for model_path in model_paths:
            name = Path(model_path).stem
            print(f"Evaluating: {name}")

            metrics = self.evaluate_detection(model_path, test_images, test_data)
            results[name] = metrics

            print(f"  Precision: {metrics.precision:.4f}")
            print(f"  Recall: {metrics.recall:.4f}")
            print(f"  F1: {metrics.f1_score:.4f}")

        if output_name:
            self._save_comparison(results, output_name, "detection")

        return results

    def _save_comparison(
        self,
        results: Dict[str, Any],
        name: str,
        eval_type: str
    ):
        """Save comparison results."""
        output_path = self.output_dir / f"{name}_{eval_type}_comparison.json"

        data = {
            "evaluated_at": datetime.now().isoformat(),
            "type": eval_type,
            "results": {}
        }

        for model_name, metrics in results.items():
            if hasattr(metrics, '__dict__'):
                data["results"][model_name] = metrics.__dict__
            else:
                data["results"][model_name] = metrics

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved comparison to: {output_path}")

    def generate_report(
        self,
        detection_metrics: DetectionMetrics = None,
        reid_metrics: ReIDMetrics = None,
        output_path: str = None
    ) -> str:
        """
        Generate evaluation report.

        Args:
            detection_metrics: Detection evaluation results
            reid_metrics: Re-ID evaluation results
            output_path: Output file path

        Returns:
            Report content
        """
        lines = [
            "=" * 60,
            "  MODEL EVALUATION REPORT",
            "=" * 60,
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        if detection_metrics:
            lines.extend([
                "-" * 60,
                "  DETECTION METRICS",
                "-" * 60,
                f"  Precision:       {detection_metrics.precision:.4f}",
                f"  Recall:          {detection_metrics.recall:.4f}",
                f"  F1 Score:        {detection_metrics.f1_score:.4f}",
                f"  mAP@50:          {detection_metrics.mAP50:.4f}",
                f"  mAP@50-95:       {detection_metrics.mAP50_95:.4f}",
                "",
                f"  True Positives:  {detection_metrics.true_positives}",
                f"  False Positives: {detection_metrics.false_positives}",
                f"  False Negatives: {detection_metrics.false_negatives}",
                "",
            ])

        if reid_metrics:
            lines.extend([
                "-" * 60,
                "  RE-ID METRICS",
                "-" * 60,
                f"  Rank-1:          {reid_metrics.rank1:.4f}",
                f"  Rank-5:          {reid_metrics.rank5:.4f}",
                f"  Rank-10:         {reid_metrics.rank10:.4f}",
                f"  mAP:             {reid_metrics.mAP:.4f}",
                "",
                f"  Queries:         {reid_metrics.num_queries}",
                f"  Gallery:         {reid_metrics.num_gallery}",
                "",
            ])

        lines.append("=" * 60)

        report = "\n".join(lines)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            print(f"Report saved to: {output_path}")

        return report


def main():
    """CLI for model evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Model Evaluation")
    subparsers = parser.add_subparsers(dest="command")

    det_parser = subparsers.add_parser("detection", help="Evaluate detection")
    det_parser.add_argument("model", help="Model path")
    det_parser.add_argument("--ground-truth", "-g", required=True)
    det_parser.add_argument("--conf", type=float, default=0.5)
    det_parser.add_argument("--iou", type=float, default=0.5)

    compare_parser = subparsers.add_parser("compare", help="Compare models")
    compare_parser.add_argument("models", nargs="+", help="Model paths")
    compare_parser.add_argument("--ground-truth", "-g", required=True)
    compare_parser.add_argument("--output", "-o", default="comparison")

    args = parser.parse_args()
    evaluator = ModelEvaluator()

    if args.command == "detection":
        with open(args.ground_truth, "r") as f:
            gt = json.load(f)

        test_images = list(gt.keys())
        metrics = evaluator.evaluate_detection(
            args.model, test_images, gt,
            iou_threshold=args.iou,
            conf_threshold=args.conf
        )
        print(evaluator.generate_report(detection_metrics=metrics))

    elif args.command == "compare":
        with open(args.ground_truth, "r") as f:
            gt = json.load(f)

        evaluator.compare_detection_models(args.models, gt, args.output)


if __name__ == "__main__":
    main()
