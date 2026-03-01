from dataclasses import dataclass, field


@dataclass
class DetectedInstance:
    bbox_left: float
    bbox_top: float
    bbox_width: float
    bbox_height: float
    confidence: float


@dataclass
class DetectedLabel:
    name: str
    confidence: float
    categories: list[str] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)
    instances: list[DetectedInstance] = field(default_factory=list)


@dataclass
class DetectionResult:
    labels: list[DetectedLabel]
    raw_response: dict
    image_width: int
    image_height: int


@dataclass
class DetectionSummary:
    id: int
    image_filename: str
    detected_at: str
    label_count: int
    top_labels: list[str]
    has_feedback: bool


@dataclass
class LabelDetail:
    id: int
    name: str
    confidence: float
    has_bounding_box: bool
    bbox: dict | None
    feedback_status: str | None


@dataclass
class DetectionDetail:
    id: int
    image_filename: str
    image_path: str
    image_width: int
    image_height: int
    detected_at: str
    min_confidence: float
    max_labels: int
    label_count: int
    labels: list[LabelDetail]
    notes: str | None


@dataclass
class FeedbackEntry:
    label_id: int
    detection_id: int
    is_correct: bool | None
    is_wrong_reason: bool = False
    expected_label: str | None = None
    reviewer_notes: str | None = None


@dataclass
class AnalysisStats:
    total_detections: int
    total_labels: int
    total_reviewed: int
    accuracy_rate: float
    wrong_reason_count: int
    common_misclassifications: list[dict]
    confidence_distribution: list[dict]
