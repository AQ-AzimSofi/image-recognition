import json
import sqlite3
from pathlib import Path
from contextlib import contextmanager

from .config import DEFAULT_CONFIG
from .models import (
    DetectionSummary,
    DetectionDetail,
    LabelDetail,
    FeedbackEntry,
    AnalysisStats,
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_filename TEXT NOT NULL,
    image_path TEXT NOT NULL,
    image_width INTEGER,
    image_height INTEGER,
    detected_at TEXT NOT NULL DEFAULT (datetime('now')),
    min_confidence REAL,
    max_labels INTEGER,
    raw_response TEXT,
    label_count INTEGER NOT NULL DEFAULT 0,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id INTEGER NOT NULL REFERENCES detections(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    confidence REAL NOT NULL,
    category TEXT,
    parents TEXT,
    has_bounding_box INTEGER NOT NULL DEFAULT 0,
    bbox_left REAL,
    bbox_top REAL,
    bbox_width REAL,
    bbox_height REAL,
    instance_confidence REAL
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label_id INTEGER NOT NULL REFERENCES labels(id) ON DELETE CASCADE,
    detection_id INTEGER NOT NULL REFERENCES detections(id) ON DELETE CASCADE,
    is_correct INTEGER,
    is_wrong_reason INTEGER NOT NULL DEFAULT 0,
    expected_label TEXT,
    reviewer_notes TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS stress_tests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_detection_id INTEGER NOT NULL REFERENCES detections(id),
    degradation_type TEXT NOT NULL,
    degradation_level REAL NOT NULL,
    result_detection_id INTEGER NOT NULL REFERENCES detections(id),
    label_diff TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_detections_detected_at ON detections(detected_at);
CREATE INDEX IF NOT EXISTS idx_labels_detection_id ON labels(detection_id);
CREATE INDEX IF NOT EXISTS idx_labels_name ON labels(name);
CREATE INDEX IF NOT EXISTS idx_feedback_detection_id ON feedback(detection_id);
CREATE INDEX IF NOT EXISTS idx_feedback_label_id ON feedback(label_id);
"""


class Database:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DEFAULT_CONFIG.database.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def save_detection(
        self,
        image_filename: str,
        image_path: str,
        image_width: int,
        image_height: int,
        labels_data: list[dict],
        raw_response: dict,
        min_confidence: float,
        max_labels: int,
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO detections
                   (image_filename, image_path, image_width, image_height,
                    raw_response, label_count, min_confidence, max_labels)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    image_filename,
                    image_path,
                    image_width,
                    image_height,
                    json.dumps(raw_response),
                    len(labels_data),
                    min_confidence,
                    max_labels,
                ),
            )
            detection_id = cursor.lastrowid

            for label in labels_data:
                conn.execute(
                    """INSERT INTO labels
                       (detection_id, name, confidence, category, parents,
                        has_bounding_box, bbox_left, bbox_top, bbox_width,
                        bbox_height, instance_confidence)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        detection_id,
                        label["name"],
                        label["confidence"],
                        label.get("category"),
                        label.get("parents"),
                        label.get("has_bounding_box", 0),
                        label.get("bbox_left"),
                        label.get("bbox_top"),
                        label.get("bbox_width"),
                        label.get("bbox_height"),
                        label.get("instance_confidence"),
                    ),
                )

            return detection_id

    def get_detection(self, detection_id: int) -> DetectionDetail | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM detections WHERE id = ?", (detection_id,)
            ).fetchone()
            if not row:
                return None

            label_rows = conn.execute(
                "SELECT * FROM labels WHERE detection_id = ?", (detection_id,)
            ).fetchall()

            feedback_rows = conn.execute(
                "SELECT label_id, is_correct, is_wrong_reason FROM feedback WHERE detection_id = ?",
                (detection_id,),
            ).fetchall()
            feedback_map = {f["label_id"]: f for f in feedback_rows}

            labels = []
            for lr in label_rows:
                fb = feedback_map.get(lr["id"])
                if fb:
                    if fb["is_wrong_reason"]:
                        status = "wrong_reason"
                    elif fb["is_correct"]:
                        status = "correct"
                    elif fb["is_correct"] == 0:
                        status = "incorrect"
                    else:
                        status = None
                else:
                    status = None

                bbox = None
                if lr["has_bounding_box"]:
                    bbox = {
                        "left": lr["bbox_left"],
                        "top": lr["bbox_top"],
                        "width": lr["bbox_width"],
                        "height": lr["bbox_height"],
                    }

                labels.append(
                    LabelDetail(
                        id=lr["id"],
                        name=lr["name"],
                        confidence=lr["confidence"],
                        has_bounding_box=bool(lr["has_bounding_box"]),
                        bbox=bbox,
                        feedback_status=status,
                    )
                )

            return DetectionDetail(
                id=row["id"],
                image_filename=row["image_filename"],
                image_path=row["image_path"],
                image_width=row["image_width"],
                image_height=row["image_height"],
                detected_at=row["detected_at"],
                min_confidence=row["min_confidence"],
                max_labels=row["max_labels"],
                label_count=row["label_count"],
                labels=labels,
                notes=row["notes"],
            )

    def get_history(
        self,
        limit: int = 50,
        offset: int = 0,
        label_filter: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[DetectionSummary]:
        query = "SELECT d.* FROM detections d"
        params = []

        if label_filter:
            query += " INNER JOIN labels l ON l.detection_id = d.id AND l.name LIKE ?"
            params.append(f"%{label_filter}%")

        conditions = []
        if date_from:
            conditions.append("d.detected_at >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("d.detected_at <= ?")
            params.append(date_to)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        if label_filter:
            query += " GROUP BY d.id"

        query += " ORDER BY d.detected_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            summaries = []
            for row in rows:
                top_labels_rows = conn.execute(
                    "SELECT name FROM labels WHERE detection_id = ? ORDER BY confidence DESC LIMIT 5",
                    (row["id"],),
                ).fetchall()
                top_labels = [r["name"] for r in top_labels_rows]

                has_fb = conn.execute(
                    "SELECT COUNT(*) as c FROM feedback WHERE detection_id = ?",
                    (row["id"],),
                ).fetchone()["c"] > 0

                summaries.append(
                    DetectionSummary(
                        id=row["id"],
                        image_filename=row["image_filename"],
                        detected_at=row["detected_at"],
                        label_count=row["label_count"],
                        top_labels=top_labels,
                        has_feedback=has_fb,
                    )
                )
            return summaries

    def save_feedback(self, entry: FeedbackEntry) -> int:
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM feedback WHERE label_id = ?", (entry.label_id,)
            ).fetchone()

            if existing:
                conn.execute(
                    """UPDATE feedback
                       SET is_correct = ?, is_wrong_reason = ?, expected_label = ?,
                           reviewer_notes = ?, created_at = datetime('now')
                       WHERE label_id = ?""",
                    (
                        int(entry.is_correct) if entry.is_correct is not None else None,
                        int(entry.is_wrong_reason),
                        entry.expected_label,
                        entry.reviewer_notes,
                        entry.label_id,
                    ),
                )
                return existing["id"]

            cursor = conn.execute(
                """INSERT INTO feedback
                   (label_id, detection_id, is_correct, is_wrong_reason,
                    expected_label, reviewer_notes)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    entry.label_id,
                    entry.detection_id,
                    int(entry.is_correct) if entry.is_correct is not None else None,
                    int(entry.is_wrong_reason),
                    entry.expected_label,
                    entry.reviewer_notes,
                ),
            )
            return cursor.lastrowid

    def get_analysis_stats(self) -> AnalysisStats:
        with self._connect() as conn:
            total_detections = conn.execute(
                "SELECT COUNT(*) as c FROM detections"
            ).fetchone()["c"]

            total_labels = conn.execute(
                "SELECT COUNT(*) as c FROM labels"
            ).fetchone()["c"]

            total_reviewed = conn.execute(
                "SELECT COUNT(*) as c FROM feedback WHERE is_correct IS NOT NULL"
            ).fetchone()["c"]

            correct_count = conn.execute(
                "SELECT COUNT(*) as c FROM feedback WHERE is_correct = 1"
            ).fetchone()["c"]

            wrong_reason_count = conn.execute(
                "SELECT COUNT(*) as c FROM feedback WHERE is_wrong_reason = 1"
            ).fetchone()["c"]

            accuracy_rate = correct_count / total_reviewed if total_reviewed > 0 else 0.0

            misclass_rows = conn.execute(
                """SELECT l.name as detected, f.expected_label as expected, COUNT(*) as count
                   FROM feedback f
                   JOIN labels l ON l.id = f.label_id
                   WHERE f.is_correct = 0 AND f.expected_label IS NOT NULL
                   GROUP BY l.name, f.expected_label
                   ORDER BY count DESC
                   LIMIT 20""",
            ).fetchall()

            common_misclassifications = [
                {"detected": r["detected"], "expected": r["expected"], "count": r["count"]}
                for r in misclass_rows
            ]

            conf_rows = conn.execute(
                """SELECT
                       CAST(l.confidence / 10 AS INTEGER) * 10 as bucket,
                       COUNT(*) as total,
                       SUM(CASE WHEN f.is_correct = 1 THEN 1 ELSE 0 END) as correct,
                       SUM(CASE WHEN f.is_correct = 0 THEN 1 ELSE 0 END) as incorrect
                   FROM labels l
                   LEFT JOIN feedback f ON f.label_id = l.id
                   GROUP BY bucket
                   ORDER BY bucket""",
            ).fetchall()

            confidence_distribution = [
                {
                    "bucket": f"{r['bucket']}-{r['bucket'] + 10}",
                    "total": r["total"],
                    "correct": r["correct"],
                    "incorrect": r["incorrect"],
                }
                for r in conf_rows
            ]

            return AnalysisStats(
                total_detections=total_detections,
                total_labels=total_labels,
                total_reviewed=total_reviewed,
                accuracy_rate=accuracy_rate,
                wrong_reason_count=wrong_reason_count,
                common_misclassifications=common_misclassifications,
                confidence_distribution=confidence_distribution,
            )

    def delete_detection(self, detection_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM detections WHERE id = ?", (detection_id,)
            )
            return cursor.rowcount > 0

    def save_stress_test(
        self,
        source_detection_id: int,
        degradation_type: str,
        degradation_level: float,
        result_detection_id: int,
        label_diff: dict,
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO stress_tests
                   (source_detection_id, degradation_type, degradation_level,
                    result_detection_id, label_diff)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    source_detection_id,
                    degradation_type,
                    degradation_level,
                    result_detection_id,
                    json.dumps(label_diff),
                ),
            )
            return cursor.lastrowid

    def get_stress_tests(self, source_detection_id: int | None = None) -> list[dict]:
        query = """
            SELECT st.*, d1.image_filename as source_image, d2.image_filename as result_image
            FROM stress_tests st
            JOIN detections d1 ON d1.id = st.source_detection_id
            JOIN detections d2 ON d2.id = st.result_detection_id
        """
        params = []
        if source_detection_id:
            query += " WHERE st.source_detection_id = ?"
            params.append(source_detection_id)
        query += " ORDER BY st.created_at DESC"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
