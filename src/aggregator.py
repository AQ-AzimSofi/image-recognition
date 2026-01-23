#!/usr/bin/env python3
"""
Time-series aggregation module for worker counting.

Aggregates detection results over time, removes duplicates using Re-ID,
and calculates man-hours.
"""

import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

from reid import PersonReID, PersonTracker, PersonDetection


@dataclass
class TimeSlot:
    """Represents a time slot with worker counts."""
    start_time: datetime
    end_time: datetime
    unique_workers: int = 0
    total_detections: int = 0
    person_ids: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)


@dataclass
class DailyAggregation:
    """Daily aggregation result."""
    date: str
    site_id: str
    total_unique_workers: int
    man_hours: float
    peak_count: int
    peak_time: str
    hourly_counts: Dict[str, int]
    time_slots: List[TimeSlot]


@dataclass
class SiteAggregation:
    """Aggregation for a single site/camera."""
    site_id: str
    camera_ids: List[str]
    detections: List[PersonDetection]
    unique_count: int


class WorkerAggregator:
    """
    Aggregates worker detections over time.

    Handles:
    - Time-series counting with Re-ID
    - Duplicate removal across frames
    - Multi-camera fusion
    - Man-hour calculation
    """

    def __init__(
        self,
        reid_model: PersonReID = None,
        similarity_threshold: float = 0.6,
        time_slot_minutes: int = 60
    ):
        """
        Initialize aggregator.

        Args:
            reid_model: PersonReID instance (creates one if None)
            similarity_threshold: Threshold for person matching
            time_slot_minutes: Duration of each time slot in minutes
        """
        self.reid = reid_model or PersonReID()
        self.threshold = similarity_threshold
        self.slot_minutes = time_slot_minutes
        self.tracker = PersonTracker(self.reid, similarity_threshold)

    def parse_timestamp_from_filename(self, filename: str) -> Optional[datetime]:
        """
        Extract timestamp from filename.

        Supports formats:
        - frame_0001.jpg -> index-based
        - 2026-01-21_08-30-00.jpg -> datetime
        - TLC00046_20260121_083000.jpg -> datetime
        """
        name = Path(filename).stem

        datetime_pattern = r'(\d{4}[-_]?\d{2}[-_]?\d{2})[_T]?(\d{2}[-_:]?\d{2}[-_:]?\d{2})'
        match = re.search(datetime_pattern, name)
        if match:
            date_str = match.group(1).replace('-', '').replace('_', '')
            time_str = match.group(2).replace('-', '').replace('_', '').replace(':', '')
            try:
                return datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
            except ValueError:
                pass

        frame_pattern = r'frame[_-]?(\d+)'
        match = re.search(frame_pattern, name, re.IGNORECASE)
        if match:
            frame_num = int(match.group(1))
            base_time = datetime(2026, 1, 21, 8, 0, 0)
            return base_time + timedelta(minutes=frame_num)

        return None

    def parse_timestamp_from_index(
        self,
        index: int,
        base_time: datetime = None,
        interval_minutes: int = 1
    ) -> datetime:
        """
        Generate timestamp from frame index.

        Args:
            index: Frame index
            base_time: Starting time (default: 8:00 AM today)
            interval_minutes: Minutes between frames
        """
        if base_time is None:
            today = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
            base_time = today

        return base_time + timedelta(minutes=index * interval_minutes)

    def group_by_time_slot(
        self,
        detections: Dict[str, List[PersonDetection]],
        slot_minutes: int = None
    ) -> Dict[str, List[Tuple[str, List[PersonDetection]]]]:
        """
        Group detections by time slot.

        Args:
            detections: Dict mapping image_path to list of detections
            slot_minutes: Minutes per slot (default: self.slot_minutes)

        Returns:
            Dict mapping slot key (e.g., "08:00") to list of (path, detections)
        """
        slot_minutes = slot_minutes or self.slot_minutes
        slots = defaultdict(list)

        for idx, (path, dets) in enumerate(sorted(detections.items())):
            timestamp = self.parse_timestamp_from_filename(path)
            if timestamp is None:
                timestamp = self.parse_timestamp_from_index(idx)

            slot_hour = timestamp.hour
            slot_minute = (timestamp.minute // slot_minutes) * slot_minutes
            slot_key = f"{slot_hour:02d}:{slot_minute:02d}"

            for det in dets:
                det.timestamp = timestamp.isoformat()

            slots[slot_key].append((path, dets))

        return dict(slots)

    def count_unique_in_slot(
        self,
        slot_detections: List[Tuple[str, List[PersonDetection]]]
    ) -> Tuple[int, List[str]]:
        """
        Count unique persons in a time slot.

        Args:
            slot_detections: List of (image_path, detections) in the slot

        Returns:
            Tuple of (unique_count, list of person_ids)
        """
        all_ids = set()

        for path, dets in slot_detections:
            for det in dets:
                if det.person_id:
                    all_ids.add(det.person_id)

        return len(all_ids), sorted(all_ids)

    def process_image_sequence(
        self,
        image_paths: List[str],
        boxes_per_image: List[List[List[float]]]
    ) -> Dict[str, List[PersonDetection]]:
        """
        Process a sequence of images with detection boxes.

        Args:
            image_paths: List of image file paths
            boxes_per_image: Detection boxes for each image

        Returns:
            Dict mapping image path to list of PersonDetections with IDs
        """
        self.tracker.reset()
        return self.tracker.process_sequence(image_paths, boxes_per_image)

    def aggregate_daily(
        self,
        detections: Dict[str, List[PersonDetection]],
        site_id: str = "site01",
        date: str = None
    ) -> DailyAggregation:
        """
        Aggregate detections for a single day.

        Args:
            detections: Dict mapping image_path to list of detections
            site_id: Site identifier
            date: Date string (default: today)

        Returns:
            DailyAggregation with hourly counts and summary
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        slots = self.group_by_time_slot(detections)

        hourly_counts = {}
        time_slots = []
        all_person_ids = set()

        for slot_key in sorted(slots.keys()):
            slot_data = slots[slot_key]
            unique_count, person_ids = self.count_unique_in_slot(slot_data)

            hourly_counts[slot_key] = unique_count
            all_person_ids.update(person_ids)

            hour, minute = map(int, slot_key.split(':'))
            start_time = datetime.strptime(f"{date} {slot_key}", "%Y-%m-%d %H:%M")
            end_time = start_time + timedelta(minutes=self.slot_minutes)

            time_slot = TimeSlot(
                start_time=start_time,
                end_time=end_time,
                unique_workers=unique_count,
                total_detections=sum(len(d) for _, d in slot_data),
                person_ids=person_ids,
                images=[p for p, _ in slot_data]
            )
            time_slots.append(time_slot)

        peak_count = max(hourly_counts.values()) if hourly_counts else 0
        peak_time = max(hourly_counts, key=hourly_counts.get) if hourly_counts else "N/A"

        man_hours = self.calculate_man_hours(hourly_counts)

        return DailyAggregation(
            date=date,
            site_id=site_id,
            total_unique_workers=len(all_person_ids),
            man_hours=man_hours,
            peak_count=peak_count,
            peak_time=peak_time,
            hourly_counts=hourly_counts,
            time_slots=time_slots
        )

    def calculate_man_hours(
        self,
        hourly_counts: Dict[str, int],
        hours_per_slot: float = None
    ) -> float:
        """
        Calculate total man-hours from hourly counts.

        Args:
            hourly_counts: Dict mapping time slot to worker count
            hours_per_slot: Hours per time slot (default: slot_minutes/60)

        Returns:
            Total man-hours
        """
        if hours_per_slot is None:
            hours_per_slot = self.slot_minutes / 60.0

        return sum(count * hours_per_slot for count in hourly_counts.values())

    def merge_cameras(
        self,
        camera_results: Dict[str, Dict[str, List[PersonDetection]]],
        merge_threshold: float = None
    ) -> Dict[str, List[PersonDetection]]:
        """
        Merge detection results from multiple cameras.

        Args:
            camera_results: Dict mapping camera_id to their detection results
            merge_threshold: Similarity threshold for cross-camera matching

        Returns:
            Merged detections with globally unique IDs
        """
        if merge_threshold is None:
            merge_threshold = self.threshold

        merged = defaultdict(list)
        global_id_counter = 1
        camera_to_global_id = {}

        cameras = list(camera_results.keys())

        if len(cameras) == 1:
            return camera_results[cameras[0]]

        all_timestamps = set()
        for cam_id, cam_dets in camera_results.items():
            for path in cam_dets.keys():
                ts = self.parse_timestamp_from_filename(path)
                if ts:
                    all_timestamps.add(ts)

        for timestamp in sorted(all_timestamps):
            frame_detections = {}

            for cam_id, cam_dets in camera_results.items():
                for path, dets in cam_dets.items():
                    path_ts = self.parse_timestamp_from_filename(path)
                    if path_ts and abs((path_ts - timestamp).total_seconds()) < 60:
                        frame_detections[cam_id] = dets
                        break

            if len(frame_detections) > 1:
                from reid import match_across_cameras
                local_to_global = match_across_cameras(
                    self.reid,
                    frame_detections,
                    merge_threshold
                )

                for cam_id, dets in frame_detections.items():
                    for det in dets:
                        if det.person_id in local_to_global:
                            det.person_id = local_to_global[det.person_id]

            for cam_id, dets in frame_detections.items():
                for path, cam_dets in camera_results[cam_id].items():
                    path_ts = self.parse_timestamp_from_filename(path)
                    if path_ts and abs((path_ts - timestamp).total_seconds()) < 60:
                        if path not in merged:
                            merged[path] = []
                        merged[path].extend(dets)
                        break

        return dict(merged)

    def get_worker_presence(
        self,
        detections: Dict[str, List[PersonDetection]]
    ) -> Dict[str, List[str]]:
        """
        Get presence timeline for each worker.

        Args:
            detections: Detection results

        Returns:
            Dict mapping person_id to list of timestamps when they were seen
        """
        presence = defaultdict(list)

        for path, dets in sorted(detections.items()):
            for det in dets:
                if det.person_id and det.timestamp:
                    presence[det.person_id].append(det.timestamp)

        return dict(presence)


def create_test_aggregation():
    """Create test data for demonstration."""
    aggregator = WorkerAggregator()

    test_detections = {}
    for hour in range(8, 17):
        for minute in [0, 30]:
            path = f"frame_{hour:02d}{minute:02d}.jpg"
            count = 3 if 9 <= hour <= 15 else 2

            dets = []
            for i in range(count):
                det = PersonDetection(
                    box=[100 + i*100, 100, 200 + i*100, 400],
                    confidence=0.9,
                    person_id=f"P{i+1:04d}"
                )
                dets.append(det)

            test_detections[path] = dets

    result = aggregator.aggregate_daily(test_detections, "test_site")

    print(f"Date: {result.date}")
    print(f"Site: {result.site_id}")
    print(f"Total unique workers: {result.total_unique_workers}")
    print(f"Man-hours: {result.man_hours:.1f}")
    print(f"Peak: {result.peak_count} workers at {result.peak_time}")
    print("\nHourly counts:")
    for slot, count in sorted(result.hourly_counts.items()):
        print(f"  {slot}: {count} workers")

    return result


if __name__ == "__main__":
    create_test_aggregation()
