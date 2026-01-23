#!/usr/bin/env python3
"""
Report generation module for worker counting results.

Generates JSON, CSV, and summary reports.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import asdict

from aggregator import DailyAggregation, TimeSlot


class WorkerReporter:
    """
    Generates reports from worker counting data.

    Supports:
    - JSON reports (detailed)
    - CSV exports (tabular)
    - Summary text reports
    - KPI evaluation reports
    """

    def __init__(self, output_dir: str = "reports"):
        """
        Initialize reporter.

        Args:
            output_dir: Base directory for report output
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _serialize_aggregation(self, agg: DailyAggregation) -> Dict:
        """Convert DailyAggregation to serializable dict."""
        return {
            "date": agg.date,
            "site_id": agg.site_id,
            "summary": {
                "total_unique_workers": agg.total_unique_workers,
                "man_hours": round(agg.man_hours, 2),
                "peak_count": agg.peak_count,
                "peak_time": agg.peak_time
            },
            "hourly_counts": agg.hourly_counts,
            "time_slots": [
                {
                    "start_time": slot.start_time.isoformat(),
                    "end_time": slot.end_time.isoformat(),
                    "unique_workers": slot.unique_workers,
                    "total_detections": slot.total_detections,
                    "person_ids": slot.person_ids,
                    "images_count": len(slot.images)
                }
                for slot in agg.time_slots
            ]
        }

    def generate_json_report(
        self,
        aggregation: DailyAggregation,
        filename: str = None,
        include_details: bool = True
    ) -> str:
        """
        Generate detailed JSON report.

        Args:
            aggregation: DailyAggregation result
            filename: Output filename (auto-generated if None)
            include_details: Whether to include time slot details

        Returns:
            Path to generated report
        """
        if filename is None:
            filename = f"report_{aggregation.site_id}_{aggregation.date}.json"

        output_path = self.output_dir / "daily" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report_data = self._serialize_aggregation(aggregation)

        report_data["metadata"] = {
            "generated_at": datetime.now().isoformat(),
            "report_type": "daily_aggregation",
            "version": "1.0"
        }

        if not include_details:
            del report_data["time_slots"]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print(f"JSON report saved: {output_path}")
        return str(output_path)

    def generate_csv_report(
        self,
        aggregation: DailyAggregation,
        filename: str = None
    ) -> str:
        """
        Generate CSV report with hourly data.

        Args:
            aggregation: DailyAggregation result
            filename: Output filename

        Returns:
            Path to generated report
        """
        if filename is None:
            filename = f"hourly_{aggregation.site_id}_{aggregation.date}.csv"

        output_path = self.output_dir / "csv" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow([
                "date", "site_id", "time_slot", "unique_workers",
                "total_detections", "person_ids"
            ])

            for slot in aggregation.time_slots:
                writer.writerow([
                    aggregation.date,
                    aggregation.site_id,
                    slot.start_time.strftime("%H:%M"),
                    slot.unique_workers,
                    slot.total_detections,
                    ";".join(slot.person_ids)
                ])

        print(f"CSV report saved: {output_path}")
        return str(output_path)

    def generate_summary_csv(
        self,
        aggregations: List[DailyAggregation],
        filename: str = "daily_summary.csv"
    ) -> str:
        """
        Generate summary CSV for multiple days.

        Args:
            aggregations: List of DailyAggregation results
            filename: Output filename

        Returns:
            Path to generated report
        """
        output_path = self.output_dir / "csv" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow([
                "date", "site_id", "total_unique_workers", "man_hours",
                "peak_count", "peak_time", "active_hours"
            ])

            for agg in aggregations:
                active_hours = sum(1 for c in agg.hourly_counts.values() if c > 0)
                writer.writerow([
                    agg.date,
                    agg.site_id,
                    agg.total_unique_workers,
                    round(agg.man_hours, 2),
                    agg.peak_count,
                    agg.peak_time,
                    active_hours
                ])

        print(f"Summary CSV saved: {output_path}")
        return str(output_path)

    def generate_text_summary(
        self,
        aggregation: DailyAggregation,
        filename: str = None
    ) -> str:
        """
        Generate human-readable text summary.

        Args:
            aggregation: DailyAggregation result
            filename: Output filename

        Returns:
            Path to generated report
        """
        if filename is None:
            filename = f"summary_{aggregation.site_id}_{aggregation.date}.txt"

        output_path = self.output_dir / "text" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "=" * 60,
            f"  DAILY WORKER COUNT REPORT",
            "=" * 60,
            "",
            f"  Date:     {aggregation.date}",
            f"  Site:     {aggregation.site_id}",
            "",
            "-" * 60,
            "  SUMMARY",
            "-" * 60,
            f"  Total Unique Workers:  {aggregation.total_unique_workers}",
            f"  Total Man-hours:       {aggregation.man_hours:.1f}",
            f"  Peak Count:            {aggregation.peak_count} workers",
            f"  Peak Time:             {aggregation.peak_time}",
            "",
            "-" * 60,
            "  HOURLY BREAKDOWN",
            "-" * 60,
        ]

        for time_slot, count in sorted(aggregation.hourly_counts.items()):
            bar = "#" * count
            lines.append(f"  {time_slot}  |{bar:<15}| {count}")

        lines.extend([
            "",
            "-" * 60,
            f"  Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60
        ])

        content = "\n".join(lines)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Text summary saved: {output_path}")
        return str(output_path)

    def evaluate_kpi(
        self,
        aggregation: DailyAggregation,
        ground_truth: Dict[str, Any] = None,
        target_kpis: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Evaluate KPIs against targets.

        Args:
            aggregation: DailyAggregation result
            ground_truth: Optional ground truth data for accuracy
            target_kpis: Target KPI values

        Returns:
            KPI evaluation results
        """
        if target_kpis is None:
            target_kpis = {
                "detection_rate_day": 0.95,
                "detection_rate_night": 0.90,
                "duplicate_rate": 0.05,
                "man_hour_accuracy": 0.10
            }

        evaluation = {
            "date": aggregation.date,
            "site_id": aggregation.site_id,
            "metrics": {
                "total_unique_workers": aggregation.total_unique_workers,
                "man_hours": aggregation.man_hours,
                "peak_count": aggregation.peak_count
            },
            "kpi_targets": target_kpis,
            "kpi_results": {},
            "overall_status": "PASS"
        }

        if ground_truth:
            actual_workers = ground_truth.get("actual_workers", aggregation.total_unique_workers)
            if actual_workers > 0:
                detection_rate = aggregation.total_unique_workers / actual_workers
                evaluation["kpi_results"]["detection_rate"] = {
                    "value": round(detection_rate, 3),
                    "target": target_kpis["detection_rate_day"],
                    "status": "PASS" if detection_rate >= target_kpis["detection_rate_day"] else "FAIL"
                }

            actual_man_hours = ground_truth.get("actual_man_hours")
            if actual_man_hours:
                error_rate = abs(aggregation.man_hours - actual_man_hours) / actual_man_hours
                evaluation["kpi_results"]["man_hour_accuracy"] = {
                    "value": round(error_rate, 3),
                    "target": target_kpis["man_hour_accuracy"],
                    "status": "PASS" if error_rate <= target_kpis["man_hour_accuracy"] else "FAIL"
                }

        for result in evaluation["kpi_results"].values():
            if result.get("status") == "FAIL":
                evaluation["overall_status"] = "FAIL"
                break

        return evaluation

    def generate_kpi_report(
        self,
        aggregation: DailyAggregation,
        ground_truth: Dict[str, Any] = None,
        filename: str = None
    ) -> str:
        """
        Generate KPI evaluation report.

        Args:
            aggregation: DailyAggregation result
            ground_truth: Ground truth data
            filename: Output filename

        Returns:
            Path to generated report
        """
        if filename is None:
            filename = f"kpi_{aggregation.site_id}_{aggregation.date}.json"

        output_path = self.output_dir / "accuracy" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        evaluation = self.evaluate_kpi(aggregation, ground_truth)

        evaluation["metadata"] = {
            "generated_at": datetime.now().isoformat(),
            "report_type": "kpi_evaluation"
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(evaluation, f, indent=2)

        print(f"KPI report saved: {output_path}")
        return str(output_path)

    def generate_all_reports(
        self,
        aggregation: DailyAggregation,
        ground_truth: Dict[str, Any] = None
    ) -> Dict[str, str]:
        """
        Generate all report types.

        Args:
            aggregation: DailyAggregation result
            ground_truth: Optional ground truth

        Returns:
            Dict mapping report type to file path
        """
        reports = {}

        reports["json"] = self.generate_json_report(aggregation)
        reports["csv"] = self.generate_csv_report(aggregation)
        reports["text"] = self.generate_text_summary(aggregation)
        reports["kpi"] = self.generate_kpi_report(aggregation, ground_truth)

        return reports


def demo_reporter():
    """Generate demo reports."""
    from aggregator import create_test_aggregation

    aggregation = create_test_aggregation()

    reporter = WorkerReporter()
    reports = reporter.generate_all_reports(aggregation)

    print("\nGenerated reports:")
    for report_type, path in reports.items():
        print(f"  {report_type}: {path}")


if __name__ == "__main__":
    demo_reporter()
