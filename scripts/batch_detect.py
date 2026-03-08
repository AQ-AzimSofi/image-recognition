"""Batch detect objects in images from a directory, with HEIC support."""

import io
import json
import sys
from pathlib import Path

import pillow_heif
from PIL import Image

pillow_heif.register_heif_opener()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rekognition import RekognitionClient


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".heif"}
MAX_JPEG_BYTES = 5 * 1024 * 1024


def convert_to_jpeg_bytes(image_path: Path, quality: int = 85) -> bytes:
    img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    data = buf.getvalue()

    if len(data) > MAX_JPEG_BYTES:
        ratio = (MAX_JPEG_BYTES / len(data)) ** 0.5
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        data = buf.getvalue()

    return data


def detect_image(client: RekognitionClient, image_path: Path, save_annotated: bool = False) -> dict:
    jpeg_bytes = convert_to_jpeg_bytes(image_path)
    result = client.detect_labels(jpeg_bytes)

    if save_annotated:
        from src.image_utils import draw_labels_on_image
        annotated = draw_labels_on_image(image_path, result.labels)
        out_dir = image_path.parent / "annotated"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"{image_path.stem}_annotated.jpg"
        annotated.save(out_path, "JPEG", quality=90)

    return {
        "file": image_path.name,
        "label_count": len(result.labels),
        "labels": [
            {
                "name": l.name,
                "confidence": round(l.confidence, 1),
                "categories": l.categories,
                "instances": len(l.instances),
            }
            for l in result.labels
        ],
    }


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <image_directory> [min_confidence] [--annotate]")
        sys.exit(1)

    image_dir = Path(sys.argv[1])
    if not image_dir.is_dir():
        print(f"Error: {image_dir} is not a directory")
        sys.exit(1)

    save_annotated = "--annotate" in sys.argv
    args = [a for a in sys.argv[2:] if a != "--annotate"]
    min_confidence = float(args[0]) if args else 50.0

    images = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not images:
        print(f"No supported images found in {image_dir}")
        sys.exit(1)

    print(f"Found {len(images)} images in {image_dir}")
    print(f"Min confidence: {min_confidence}%\n")

    client = RekognitionClient(min_confidence=min_confidence)
    results = []

    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] Processing {img_path.name}...")
        try:
            result = detect_image(client, img_path, save_annotated)
            results.append(result)
            top_labels = ", ".join(
                f"{l['name']} ({l['confidence']}%)" for l in result["labels"][:5]
            )
            print(f"  -> {result['label_count']} labels: {top_labels}")
        except Exception as e:
            print(f"  -> ERROR: {e}")
            results.append({"file": img_path.name, "error": str(e)})

    output_path = image_dir / "detection_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
