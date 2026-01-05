#!/usr/bin/env python3
"""
Calculate detection accuracy by comparing YOLO results against ground truth.

Usage:
    python src/accuracy.py
    python src/accuracy.py --ground-truth data/ground_truth.json
    python src/accuracy.py --confidence 0.3
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from detect import detect_persons


def load_ground_truth(ground_truth_path: str) -> dict:
    """Load ground truth data from JSON file."""
    with open(ground_truth_path, "r") as f:
        data = json.load(f)
    return {item["image"]: item["actual_count"] for item in data["images"]}


def calculate_metrics(detected: int, actual: int) -> dict:
    """
    Calculate detection metrics for a single image.

    Returns dict with:
        - correct: bool (exact match)
        - difference: int (detected - actual)
        - false_negatives: int (missed detections)
        - false_positives: int (extra detections)
    """
    diff = detected - actual
    return {
        "correct": detected == actual,
        "difference": diff,
        "false_negatives": max(0, actual - detected),
        "false_positives": max(0, detected - actual)
    }


def run_accuracy_test(
    model: YOLO,
    ground_truth: dict,
    confidence_threshold: float,
    images_dir: str
) -> dict:
    """
    Run detection on all images and compare against ground truth.

    Returns comprehensive accuracy report.
    """
    results = []
    total_actual = 0
    total_detected = 0
    total_correct = 0
    total_fn = 0
    total_fp = 0

    print(f"\n{'='*70}")
    print(f"{'Image':<30} {'Actual':>8} {'Detected':>10} {'Status':>10}")
    print(f"{'='*70}")

    for image_name, actual_count in ground_truth.items():
        image_path = Path(images_dir) / image_name

        if not image_path.exists():
            print(f"{image_name:<30} {'MISSING':>8} {'-':>10} {'SKIP':>10}")
            continue

        detected_count, boxes = detect_persons(model, str(image_path), confidence_threshold)

        metrics = calculate_metrics(detected_count, actual_count)

        status = "OK" if metrics["correct"] else f"OFF BY {metrics['difference']:+d}"

        print(f"{image_name:<30} {actual_count:>8} {detected_count:>10} {status:>10}")

        results.append({
            "image": image_name,
            "actual": actual_count,
            "detected": detected_count,
            "metrics": metrics
        })

        total_actual += actual_count
        total_detected += detected_count
        if metrics["correct"]:
            total_correct += 1
        total_fn += metrics["false_negatives"]
        total_fp += metrics["false_positives"]

    print(f"{'='*70}")

    if not results:
        return {"error": "No images found"}

    exact_accuracy = (total_correct / len(results)) * 100

    if total_actual > 0:
        detection_rate = ((total_actual - total_fn) / total_actual) * 100
    else:
        detection_rate = 100.0

    if total_detected > 0:
        precision = ((total_detected - total_fp) / total_detected) * 100
    else:
        precision = 100.0

    report = {
        "summary": {
            "images_tested": len(results),
            "total_actual_persons": total_actual,
            "total_detected_persons": total_detected,
            "exact_match_accuracy": round(exact_accuracy, 2),
            "detection_rate": round(detection_rate, 2),
            "precision": round(precision, 2),
            "total_false_negatives": total_fn,
            "total_false_positives": total_fp
        },
        "confidence_threshold": confidence_threshold,
        "details": results
    }

    return report


def print_report(report: dict):
    """Print formatted accuracy report."""
    if "error" in report:
        print(f"\nError: {report['error']}")
        return

    s = report["summary"]

    print(f"\n{'='*50}")
    print("ACCURACY REPORT")
    print(f"{'='*50}")
    print(f"Confidence Threshold: {report['confidence_threshold']}")
    print(f"\nImages Tested:        {s['images_tested']}")
    print(f"Total Actual Persons: {s['total_actual_persons']}")
    print(f"Total Detected:       {s['total_detected_persons']}")
    print(f"\n--- Metrics ---")
    print(f"Exact Match Accuracy: {s['exact_match_accuracy']:.1f}%")
    print(f"Detection Rate:       {s['detection_rate']:.1f}%  (target: >= 95%)")
    print(f"Precision:            {s['precision']:.1f}%")
    print(f"\nFalse Negatives (Missed):  {s['total_false_negatives']}")
    print(f"False Positives (Extra):   {s['total_false_positives']}")

    if s['detection_rate'] >= 95:
        print(f"\n[PASS] Detection rate meets the 95% daytime target!")
    else:
        print(f"\n[FAIL] Detection rate below 95% target. Consider:")
        print(f"       - Lowering confidence threshold")
        print(f"       - Using a larger YOLO model (yolo11s.pt, yolo11m.pt)")
        print(f"       - Checking image quality")


def main():
    parser = argparse.ArgumentParser(description="Calculate YOLO detection accuracy")
    parser.add_argument(
        "--ground-truth", "-g",
        default="data/ground_truth.json",
        help="Path to ground truth JSON file"
    )
    parser.add_argument(
        "--images-dir", "-i",
        default="images/input",
        help="Directory containing test images"
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--model", "-m",
        default="yolo11n.pt",
        help="YOLO model to use (default: yolo11n.pt)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Save report to JSON file"
    )

    args = parser.parse_args()

    if not Path(args.ground_truth).exists():
        print(f"Error: Ground truth file not found: {args.ground_truth}")
        print("\nPlease create a ground_truth.json file with format:")
        print('''
{
    "images": [
        {"image": "photo1.jpg", "actual_count": 3},
        {"image": "photo2.jpg", "actual_count": 5}
    ]
}
        ''')
        return

    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)

    ground_truth = load_ground_truth(args.ground_truth)
    print(f"Loaded ground truth for {len(ground_truth)} images")

    report = run_accuracy_test(
        model,
        ground_truth,
        args.confidence,
        args.images_dir
    )

    print_report(report)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
