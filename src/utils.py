import cv2
import numpy as np
from pathlib import Path


def load_image(image_path: str) -> np.ndarray:
    """Load an image from path."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return img


def save_image(image: np.ndarray, output_path: str) -> None:
    """Save an image to path."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, image)


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: list,
    color: tuple = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding boxes on image.

    Args:
        image: Input image (BGR format)
        boxes: List of boxes, each with (x1, y1, x2, y2, confidence)
        color: BGR color tuple
        thickness: Line thickness

    Returns:
        Image with bounding boxes drawn
    """
    img_copy = image.copy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        conf = box[4] if len(box) > 4 else 0.0

        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)

        label = f"Person {i+1}: {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        cv2.rectangle(
            img_copy,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            color,
            -1
        )
        cv2.putText(
            img_copy,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )

    return img_copy


def add_count_label(image: np.ndarray, count: int) -> np.ndarray:
    """Add total person count label to top-left of image."""
    img_copy = image.copy()
    label = f"Total Persons: {count}"

    cv2.rectangle(img_copy, (10, 10), (200, 40), (0, 0, 0), -1)
    cv2.putText(
        img_copy,
        label,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return img_copy


def format_detection_result(image_path: str, count: int, boxes: list) -> dict:
    """Format detection result as dictionary."""
    return {
        "image": str(image_path),
        "person_count": count,
        "detections": [
            {
                "box": [float(b) for b in box[:4]],
                "confidence": float(box[4]) if len(box) > 4 else 0.0
            }
            for box in boxes
        ]
    }
