#!/usr/bin/env python3
"""
Video to frames extraction module.

Extracts frames from video files at specified intervals.
Supports various video formats and extraction strategies.
"""

import cv2
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import json


@dataclass
class ExtractionConfig:
    """Configuration for frame extraction."""
    interval_seconds: float = 60.0
    max_frames: Optional[int] = None
    start_time: float = 0.0
    end_time: Optional[float] = None
    resize: Optional[tuple] = None
    quality: int = 95


@dataclass
class FrameInfo:
    """Information about an extracted frame."""
    frame_number: int
    timestamp_seconds: float
    output_path: str
    video_source: str
    extracted_at: str


class VideoFrameExtractor:
    """
    Extract frames from video files.

    Supports:
    - Fixed interval extraction (e.g., every 60 seconds)
    - Frame count-based extraction
    - Time range selection
    - Multiple video formats (mp4, avi, mov, mkv)
    """

    SUPPORTED_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}

    def __init__(self, output_dir: str = "data/frames"):
        """
        Initialize extractor.

        Args:
            output_dir: Base directory for extracted frames
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_from_video(
        self,
        video_path: str,
        config: ExtractionConfig = None,
        output_subdir: str = None,
        progress_callback: Callable[[int, int], None] = None
    ) -> List[FrameInfo]:
        """
        Extract frames from a single video file.

        Args:
            video_path: Path to video file
            config: Extraction configuration
            output_subdir: Subdirectory name (default: video filename)
            progress_callback: Optional callback(current, total)

        Returns:
            List of FrameInfo for extracted frames
        """
        config = config or ExtractionConfig()
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if video_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {video_path.suffix}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        if output_subdir is None:
            output_subdir = video_path.stem

        output_path = self.output_dir / output_subdir
        output_path.mkdir(parents=True, exist_ok=True)

        interval_frames = int(config.interval_seconds * fps)
        start_frame = int(config.start_time * fps)
        end_frame = int(config.end_time * fps) if config.end_time else total_frames

        extracted = []
        frame_count = 0
        current_frame = start_frame

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        while current_frame < end_frame:
            if config.max_frames and frame_count >= config.max_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()

            if not ret:
                break

            if config.resize:
                frame = cv2.resize(frame, config.resize)

            timestamp = current_frame / fps
            filename = f"frame_{frame_count:06d}_{timestamp:.1f}s.jpg"
            frame_path = output_path / filename

            cv2.imwrite(
                str(frame_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, config.quality]
            )

            info = FrameInfo(
                frame_number=frame_count,
                timestamp_seconds=timestamp,
                output_path=str(frame_path),
                video_source=str(video_path),
                extracted_at=datetime.now().isoformat()
            )
            extracted.append(info)

            if progress_callback:
                progress_callback(frame_count + 1, (end_frame - start_frame) // interval_frames)

            frame_count += 1
            current_frame += interval_frames

        cap.release()

        metadata_path = output_path / "extraction_metadata.json"
        self._save_metadata(metadata_path, video_path, config, extracted, fps, duration)

        print(f"Extracted {len(extracted)} frames to {output_path}")
        return extracted

    def extract_from_directory(
        self,
        video_dir: str,
        config: ExtractionConfig = None,
        recursive: bool = False
    ) -> Dict[str, List[FrameInfo]]:
        """
        Extract frames from all videos in a directory.

        Args:
            video_dir: Directory containing videos
            config: Extraction configuration
            recursive: Search subdirectories

        Returns:
            Dict mapping video path to list of FrameInfo
        """
        video_dir = Path(video_dir)
        results = {}

        pattern = "**/*" if recursive else "*"
        for video_path in video_dir.glob(pattern):
            if video_path.suffix.lower() in self.SUPPORTED_FORMATS:
                try:
                    frames = self.extract_from_video(video_path, config)
                    results[str(video_path)] = frames
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")

        return results

    def extract_at_timestamps(
        self,
        video_path: str,
        timestamps: List[float],
        output_subdir: str = None
    ) -> List[FrameInfo]:
        """
        Extract frames at specific timestamps.

        Args:
            video_path: Path to video
            timestamps: List of timestamps in seconds
            output_subdir: Output subdirectory

        Returns:
            List of FrameInfo
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)

        if output_subdir is None:
            output_subdir = video_path.stem

        output_path = self.output_dir / output_subdir
        output_path.mkdir(parents=True, exist_ok=True)

        extracted = []

        for idx, ts in enumerate(sorted(timestamps)):
            frame_num = int(ts * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if not ret:
                continue

            filename = f"frame_{idx:06d}_{ts:.1f}s.jpg"
            frame_path = output_path / filename

            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            info = FrameInfo(
                frame_number=idx,
                timestamp_seconds=ts,
                output_path=str(frame_path),
                video_source=str(video_path),
                extracted_at=datetime.now().isoformat()
            )
            extracted.append(info)

        cap.release()
        return extracted

    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get video file information.

        Args:
            video_path: Path to video

        Returns:
            Dict with video properties
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        info = {
            "path": str(video_path),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
        }

        info["duration_seconds"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
        info["duration_formatted"] = str(timedelta(seconds=int(info["duration_seconds"])))

        cap.release()
        return info

    def _save_metadata(
        self,
        path: Path,
        video_path: Path,
        config: ExtractionConfig,
        frames: List[FrameInfo],
        fps: float,
        duration: float
    ):
        """Save extraction metadata to JSON."""
        metadata = {
            "source_video": str(video_path),
            "video_fps": fps,
            "video_duration": duration,
            "extraction_config": {
                "interval_seconds": config.interval_seconds,
                "max_frames": config.max_frames,
                "start_time": config.start_time,
                "end_time": config.end_time,
                "resize": config.resize,
                "quality": config.quality
            },
            "extracted_frames": len(frames),
            "frames": [
                {
                    "frame_number": f.frame_number,
                    "timestamp_seconds": f.timestamp_seconds,
                    "output_path": f.output_path
                }
                for f in frames
            ],
            "extracted_at": datetime.now().isoformat()
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)


def main():
    """CLI for video frame extraction."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("video", help="Video file or directory")
    parser.add_argument("--interval", "-i", type=float, default=60.0,
                        help="Extraction interval in seconds (default: 60)")
    parser.add_argument("--max-frames", "-n", type=int, default=None,
                        help="Maximum frames to extract")
    parser.add_argument("--start", type=float, default=0.0,
                        help="Start time in seconds")
    parser.add_argument("--end", type=float, default=None,
                        help="End time in seconds")
    parser.add_argument("--output", "-o", default="data/frames",
                        help="Output directory")
    parser.add_argument("--info", action="store_true",
                        help="Show video info only")

    args = parser.parse_args()

    extractor = VideoFrameExtractor(output_dir=args.output)

    if args.info:
        info = extractor.get_video_info(args.video)
        print(f"Video: {info['path']}")
        print(f"Resolution: {info['width']}x{info['height']}")
        print(f"FPS: {info['fps']:.2f}")
        print(f"Duration: {info['duration_formatted']}")
        print(f"Frames: {info['frame_count']}")
        return

    config = ExtractionConfig(
        interval_seconds=args.interval,
        max_frames=args.max_frames,
        start_time=args.start,
        end_time=args.end
    )

    video_path = Path(args.video)
    if video_path.is_dir():
        extractor.extract_from_directory(args.video, config)
    else:
        extractor.extract_from_video(args.video, config)


if __name__ == "__main__":
    main()
