"""
Extract frames from timelapse video for YOLO detection testing.

Usage:
    python src/extract_frames.py                    # 1 frame per second
    python src/extract_frames.py --all              # Every frame
    python src/extract_frames.py --interval 0.5    # 1 frame every 0.5 seconds
"""

import argparse
import cv2
from pathlib import Path


def extract_frames(
    video_path: str,
    output_dir: str,
    all_frames: bool = False,
    interval: float = 1.0
) -> list[str]:
    """
    Extract frames from video.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        all_frames: If True, extract every single frame
        interval: Extract 1 frame every N seconds (ignored if all_frames=True)

    Returns:
        List of saved frame filenames
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video: {video_path.name}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")

    if all_frames:
        frame_interval = 1
        print(f"Extracting ALL {total_frames} frames...")
    else:
        frame_interval = max(1, int(fps * interval))
        expected = total_frames // frame_interval
        print(f"Extracting 1 frame per {interval} second(s) (~{expected} frames)...")
    print("-" * 40)

    saved_files = []
    frame_number = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_interval == 0:
            extracted_count += 1
            filename = f"frame_{extracted_count:04d}.jpg"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), frame)
            saved_files.append(filename)

            if all_frames and extracted_count % 100 == 0:
                print(f"  Progress: {extracted_count}/{total_frames}")
            elif not all_frames:
                print(f"  Saved: {filename}")

        frame_number += 1

    cap.release()

    print("-" * 40)
    print(f"Extracted {len(saved_files)} frames to {output_dir}/")
    return saved_files


def main():
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("--all", action="store_true", help="Extract every frame")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Extract 1 frame every N seconds (default: 1.0)")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--output", type=str, help="Output directory")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    video_path = args.video or project_root / "private" / "data" / "TLC00046.AVI"
    output_dir = args.output or project_root / "images" / "input" / "frames"

    try:
        frames = extract_frames(
            video_path,
            output_dir,
            all_frames=args.all,
            interval=args.interval
        )
        print(f"\nExtracted to: {output_dir}/")
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
