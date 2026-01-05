#!/usr/bin/env python3
"""
YOLO-based person detection script.

Usage:
    python src/detect.py <image_path> [--confidence 0.5] [--save]
    python src/detect.py images/input/test.jpg --save
    python src/detect.py images/input/  # Process all images in directory
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from utils import (
    load_image,
    save_image,
    draw_bounding_boxes,
    add_count_label,
    format_detection_result
)


PERSON_CLASS_ID = 0


def detect_persons(
    model: YOLO,
    image_path: str,
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.7
) -> tuple[int, list]:
    """
    Detect persons in an image using YOLO.

    Args:
        model: Loaded YOLO model
        image_path: Path to input image
        confidence_threshold: Minimum confidence to count as detection
        iou_threshold: IoU threshold for NMS (lower = more aggressive duplicate removal)

    Returns:
        Tuple of (person_count, list_of_boxes)
        Each box is [x1, y1, x2, y2, confidence]
    """
    results = model(image_path, verbose=False, iou=iou_threshold)

    boxes = []
    for result in results:
        for box in result.boxes:
            if int(box.cls) == PERSON_CLASS_ID and float(box.conf) >= confidence_threshold:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf)
                boxes.append([x1, y1, x2, y2, conf])

    return len(boxes), boxes


def process_single_image(
    model: YOLO,
    image_path: str,
    confidence_threshold: float,
    iou_threshold: float,
    save_output: bool,
    output_dir: str
) -> dict:
    """Process a single image and optionally save annotated result."""
    count, boxes = detect_persons(model, image_path, confidence_threshold, iou_threshold)

    result = format_detection_result(image_path, count, boxes)

    print(f"\n{'='*50}")
    print(f"Image: {image_path}")
    print(f"Persons detected: {count}")
    print(f"Confidence threshold: {confidence_threshold}")

    if boxes:
        print("\nDetections:")
        for i, box in enumerate(boxes):
            print(f"  Person {i+1}: confidence={box[4]:.2f}, box={[int(b) for b in box[:4]]}")

    if save_output:
        img = load_image(image_path)
        img = draw_bounding_boxes(img, boxes)
        img = add_count_label(img, count)

        output_path = Path(output_dir) / f"detected_{Path(image_path).name}"
        save_image(img, str(output_path))
        print(f"\nSaved annotated image to: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Detect persons in images using YOLO")
    parser.add_argument(
        "input",
        help="Path to image file or directory containing images"
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.5,
        help="Confidence threshold (0.0-1.0, default: 0.5)"
    )
    parser.add_argument(
        "--save", "-s",
        action="store_true",
        help="Save annotated output images"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="images/output",
        help="Output directory for annotated images (default: images/output)"
    )
    parser.add_argument(
        "--model", "-m",
        default="yolo11n.pt",
        help="YOLO model to use (default: yolo11n.pt)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (0.0-1.0, default: 0.45). Lower = fewer duplicates"
    )

    args = parser.parse_args()

    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)

    input_path = Path(args.input)
    results = []

    if input_path.is_file():
        result = process_single_image(
            model, str(input_path), args.confidence, args.iou, args.save, args.output_dir
        )
        results.append(result)
    elif input_path.is_dir():
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_files = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        if not image_files:
            print(f"No images found in {input_path}")
            return

        print(f"Found {len(image_files)} images to process")

        for img_file in sorted(image_files):
            result = process_single_image(
                model, str(img_file), args.confidence, args.iou, args.save, args.output_dir
            )
            results.append(result)

        total_persons = sum(r["person_count"] for r in results)
        print(f"\n{'='*50}")
        print(f"SUMMARY")
        print(f"{'='*50}")
        print(f"Images processed: {len(results)}")
        print(f"Total persons detected: {total_persons}")
        print(f"Average per image: {total_persons/len(results):.1f}")
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        return

    if args.save:
        results_path = Path(args.output_dir) / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
