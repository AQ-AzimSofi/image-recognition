#!/usr/bin/env python3
"""
Centralized configuration for Worker Counting System.

All configurable parameters should be defined here.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import os


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Directory paths
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = PROJECT_ROOT / "images"
REPORTS_DIR = PROJECT_ROOT / "reports"
RUNS_DIR = PROJECT_ROOT / "runs"


@dataclass
class ModelConfig:
    """Model file paths and settings."""
    models_dir: Path = MODELS_DIR
    default_yolo: str = "yolo11n.pt"
    default_reid: str = "reid_resnet50.pt"

    @property
    def yolo_path(self) -> Path:
        return self.models_dir / self.default_yolo

    @property
    def reid_path(self) -> Path:
        return self.models_dir / self.default_reid

    def get_yolo_model(self, name: str = None) -> str:
        """Get full path to YOLO model."""
        model_name = name or self.default_yolo
        path = self.models_dir / model_name
        if path.exists():
            return str(path)
        return model_name


@dataclass
class DetectionConfig:
    """Detection parameters."""
    confidence: float = 0.5
    iou_threshold: float = 0.45
    person_class_id: int = 0
    image_size: int = 640


@dataclass
class ReIDConfig:
    """Re-ID parameters."""
    threshold: float = 0.6
    embedding_dim: int = 512
    backbone: str = "resnet50"


@dataclass
class TrainingConfig:
    """Training parameters."""
    yolo_epochs: int = 100
    yolo_batch_size: int = 16
    yolo_image_size: int = 640

    reid_epochs: int = 60
    reid_batch_size: int = 32
    reid_learning_rate: float = 0.0003


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    reid: ReIDConfig = field(default_factory=ReIDConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    generate_charts: bool = True
    generate_reports: bool = True
    output_dir: Path = REPORTS_DIR


@dataclass
class WebUIConfig:
    """Web UI configuration."""
    host: str = "0.0.0.0"
    port: int = 7860
    share: bool = False


# Default configurations
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_DETECTION_CONFIG = DetectionConfig()
DEFAULT_REID_CONFIG = ReIDConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_PIPELINE_CONFIG = PipelineConfig()
DEFAULT_WEBUI_CONFIG = WebUIConfig()


def get_model_path(model_name: str = None) -> str:
    """
    Get model path, checking models/ directory first.

    Args:
        model_name: Model filename (e.g., "yolo11n.pt")

    Returns:
        Full path if exists in models/, otherwise just the name
    """
    if model_name is None:
        model_name = DEFAULT_MODEL_CONFIG.default_yolo

    model_path = MODELS_DIR / model_name
    if model_path.exists():
        return str(model_path)

    # Fallback to root directory (for backwards compatibility)
    root_path = PROJECT_ROOT / model_name
    if root_path.exists():
        return str(root_path)

    return model_name


def ensure_directories():
    """Create required directories if they don't exist."""
    directories = [
        MODELS_DIR,
        DATA_DIR,
        IMAGES_DIR / "input",
        IMAGES_DIR / "output",
        REPORTS_DIR,
        RUNS_DIR / "train",
        RUNS_DIR / "reid",
        RUNS_DIR / "eval",
    ]
    for d in directories:
        d.mkdir(parents=True, exist_ok=True)
