#!/usr/bin/env python3
"""
Integrated pipeline for worker counting POC.

Combines detection, Re-ID, aggregation, visualization, and reporting
into a single workflow.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

from ultralytics import YOLO

from detect import detect_persons
from reid import PersonReID, PersonTracker
from aggregator import WorkerAggregator, DailyAggregation
from visualizer import WorkerVisualizer
from reporter import WorkerReporter
from utils import load_image
from config import get_model_path, DEFAULT_DETECTION_CONFIG, DEFAULT_REID_CONFIG


class WorkerCountingPipeline:
    """
    End-to-end pipeline for construction site worker counting.

    Workflow:
    1. Load images from directory
    2. Detect persons using YOLO
    3. Extract features and assign Re-ID
    4. Aggregate counts over time
    5. Generate visualizations
    6. Create reports
    """

    def __init__(
        self,
        yolo_model: str = None,
        reid_model: str = None,
        confidence_threshold: float = None,
        iou_threshold: float = None,
        reid_threshold: float = None,
        output_dir: str = "reports"
    ):
        """
        Initialize pipeline.

        Args:
            yolo_model: YOLO model file
            reid_model: Re-ID model name
            confidence_threshold: Detection confidence threshold
            iou_threshold: NMS IoU threshold
            reid_threshold: Re-ID similarity threshold
            output_dir: Output directory for reports
        """
        yolo_model = yolo_model or get_model_path()
        reid_model = reid_model or DEFAULT_REID_CONFIG.backbone
        confidence_threshold = confidence_threshold or DEFAULT_DETECTION_CONFIG.confidence
        iou_threshold = iou_threshold or DEFAULT_DETECTION_CONFIG.iou_threshold
        reid_threshold = reid_threshold or DEFAULT_REID_CONFIG.threshold

        print("Initializing pipeline...")

        print(f"  Loading YOLO model: {yolo_model}")
        self.yolo = YOLO(yolo_model)

        print(f"  Loading Re-ID model: {reid_model}")
        self.reid = PersonReID(model_name=reid_model)

        self.aggregator = WorkerAggregator(
            reid_model=self.reid,
            similarity_threshold=reid_threshold
        )

        self.visualizer = WorkerVisualizer()
        self.reporter = WorkerReporter(output_dir=output_dir)

        self.confidence = confidence_threshold
        self.iou = iou_threshold
        self.output_dir = Path(output_dir)

        print("Pipeline initialized successfully")

    def _get_image_files(self, input_path: Path) -> List[Path]:
        """Get sorted list of image files."""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

        if input_path.is_file():
            return [input_path]

        files = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in extensions
        ]
        return sorted(files)

    def _detect_all(self, image_paths: List[Path]) -> Dict[str, List[List[float]]]:
        """Run detection on all images."""
        results = {}

        for path in image_paths:
            count, boxes = detect_persons(
                self.yolo,
                str(path),
                self.confidence,
                self.iou
            )
            results[str(path)] = boxes

        return results

    def run(
        self,
        input_path: str,
        site_id: str = "site01",
        date: str = None,
        save_annotated: bool = True,
        generate_charts: bool = True,
        generate_reports: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline.

        Args:
            input_path: Path to image file or directory
            site_id: Site identifier
            date: Date string (default: today)
            save_annotated: Whether to save annotated images
            generate_charts: Whether to generate charts
            generate_reports: Whether to generate reports

        Returns:
            Dict with pipeline results
        """
        start_time = datetime.now()
        print(f"\n{'='*60}")
        print(f"  WORKER COUNTING PIPELINE")
        print(f"{'='*60}")
        print(f"  Input: {input_path}")
        print(f"  Site: {site_id}")
        print(f"{'='*60}\n")

        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        input_path = Path(input_path)
        image_files = self._get_image_files(input_path)

        if not image_files:
            print("No images found!")
            return {"status": "error", "message": "No images found"}

        print(f"Step 1: Detecting persons in {len(image_files)} images...")
        detection_results = self._detect_all(image_files)

        total_detections = sum(len(boxes) for boxes in detection_results.values())
        print(f"  Total detections: {total_detections}")

        print("\nStep 2: Assigning person IDs with Re-ID...")
        image_paths = list(detection_results.keys())
        boxes_per_image = list(detection_results.values())

        tracked_results = self.aggregator.process_image_sequence(
            image_paths, boxes_per_image
        )

        unique_count = self.aggregator.tracker.get_unique_person_count()
        print(f"  Unique persons identified: {unique_count}")

        print("\nStep 3: Aggregating time series...")
        aggregation = self.aggregator.aggregate_daily(
            tracked_results, site_id, date
        )

        print(f"  Man-hours: {aggregation.man_hours:.1f}")
        print(f"  Peak: {aggregation.peak_count} workers at {aggregation.peak_time}")

        output_files = {
            "annotated_images": [],
            "charts": [],
            "reports": []
        }

        if save_annotated:
            print("\nStep 4: Saving annotated images...")
            annotated_dir = self.output_dir / "annotated" / site_id / date
            annotated_dir.mkdir(parents=True, exist_ok=True)

            from utils import draw_bounding_boxes, add_count_label, save_image

            for path, detections in tracked_results.items():
                img = load_image(path)

                boxes_with_ids = []
                for det in detections:
                    box = det.box + [det.confidence]
                    boxes_with_ids.append((box, det.person_id))

                for box, pid in boxes_with_ids:
                    import cv2
                    x1, y1, x2, y2 = [int(v) for v in box[:4]]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = pid if pid else "Person"
                    cv2.putText(img, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                img = add_count_label(img, len(detections))

                output_path = annotated_dir / f"annotated_{Path(path).name}"
                save_image(img, str(output_path))
                output_files["annotated_images"].append(str(output_path))

            print(f"  Saved {len(output_files['annotated_images'])} annotated images")

        if generate_charts:
            print("\nStep 5: Generating visualizations...")
            charts_dir = self.output_dir / "graphs" / site_id

            chart_line = self.visualizer.plot_hourly_counts(
                aggregation,
                str(charts_dir / f"timeline_{date}.png")
            )
            output_files["charts"].append(chart_line)

            chart_bar = self.visualizer.plot_hourly_bar(
                aggregation,
                str(charts_dir / f"hourly_{date}.png")
            )
            output_files["charts"].append(chart_bar)

            chart_summary = self.visualizer.plot_daily_summary(
                aggregation,
                str(charts_dir / f"summary_{date}.png")
            )
            output_files["charts"].append(chart_summary)

            print(f"  Generated {len(output_files['charts'])} charts")

        if generate_reports:
            print("\nStep 6: Generating reports...")
            reports = self.reporter.generate_all_reports(aggregation)
            output_files["reports"] = list(reports.values())
            print(f"  Generated {len(output_files['reports'])} reports")

        elapsed = (datetime.now() - start_time).total_seconds()

        print(f"\n{'='*60}")
        print(f"  PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"  Total time: {elapsed:.1f} seconds")
        print(f"  Images processed: {len(image_files)}")
        print(f"  Unique workers: {aggregation.total_unique_workers}")
        print(f"  Man-hours: {aggregation.man_hours:.1f}")
        print(f"{'='*60}\n")

        return {
            "status": "success",
            "site_id": site_id,
            "date": date,
            "images_processed": len(image_files),
            "total_detections": total_detections,
            "unique_workers": aggregation.total_unique_workers,
            "man_hours": aggregation.man_hours,
            "peak_count": aggregation.peak_count,
            "peak_time": aggregation.peak_time,
            "hourly_counts": aggregation.hourly_counts,
            "output_files": output_files,
            "elapsed_seconds": elapsed
        }

    def run_batch(
        self,
        input_dirs: List[str],
        site_ids: List[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run pipeline on multiple directories/sites.

        Args:
            input_dirs: List of input directories
            site_ids: List of site IDs (one per directory)
            **kwargs: Additional arguments for run()

        Returns:
            List of results
        """
        if site_ids is None:
            site_ids = [f"site{i:02d}" for i in range(1, len(input_dirs) + 1)]

        results = []
        for input_dir, site_id in zip(input_dirs, site_ids):
            result = self.run(input_dir, site_id=site_id, **kwargs)
            results.append(result)

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Worker Counting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/pipeline.py images/input/frames/ --site site01
  python src/pipeline.py images/input/ --confidence 0.4 --no-charts
  python src/pipeline.py images/input/ --output reports/test/
        """
    )

    parser.add_argument(
        "input",
        help="Path to image file or directory"
    )
    parser.add_argument(
        "--site", "-s",
        default="site01",
        help="Site identifier (default: site01)"
    )
    parser.add_argument(
        "--date", "-d",
        default=None,
        help="Date string (default: today)"
    )
    parser.add_argument(
        "--yolo-model",
        default=None,
        help="YOLO model file (default: from config)"
    )
    parser.add_argument(
        "--reid-model",
        default=None,
        choices=["resnet50", "resnet18", "efficientnet"],
        help="Re-ID model (default: from config)"
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=None,
        help="Detection confidence threshold (default: from config)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help="NMS IoU threshold (default: from config)"
    )
    parser.add_argument(
        "--reid-threshold",
        type=float,
        default=None,
        help="Re-ID similarity threshold (default: from config)"
    )
    parser.add_argument(
        "--output", "-o",
        default="reports",
        help="Output directory (default: reports)"
    )
    parser.add_argument(
        "--no-annotated",
        action="store_true",
        help="Skip saving annotated images"
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip generating charts"
    )
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Skip generating reports"
    )
    parser.add_argument(
        "--json-output",
        help="Save pipeline results to JSON file"
    )

    args = parser.parse_args()

    pipeline = WorkerCountingPipeline(
        yolo_model=args.yolo_model,
        reid_model=args.reid_model,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        reid_threshold=args.reid_threshold,
        output_dir=args.output
    )

    results = pipeline.run(
        input_path=args.input,
        site_id=args.site,
        date=args.date,
        save_annotated=not args.no_annotated,
        generate_charts=not args.no_charts,
        generate_reports=not args.no_reports
    )

    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.json_output}")


if __name__ == "__main__":
    main()
