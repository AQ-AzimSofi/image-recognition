from pathlib import Path
from dataclasses import dataclass, field


PROJECT_ROOT = Path(__file__).parent.parent

IMAGES_DIR = PROJECT_ROOT / "images"
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
DB_PATH = DATA_DIR / "detections.db"


@dataclass
class RekognitionConfig:
    region_name: str = "ap-northeast-1"
    min_confidence: float = 50.0
    max_labels: int = 20


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False


@dataclass
class UIConfig:
    host: str = "0.0.0.0"
    port: int = 7860
    share: bool = False


@dataclass
class DatabaseConfig:
    db_path: Path = DB_PATH


@dataclass
class AppConfig:
    rekognition: RekognitionConfig = field(default_factory=RekognitionConfig)
    api: APIConfig = field(default_factory=APIConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    images_dir: Path = IMAGES_DIR
    data_dir: Path = DATA_DIR
    reports_dir: Path = REPORTS_DIR


DEFAULT_CONFIG = AppConfig()


def ensure_directories():
    directories = [
        IMAGES_DIR / "input",
        IMAGES_DIR / "output",
        DATA_DIR,
        REPORTS_DIR,
    ]
    for d in directories:
        d.mkdir(parents=True, exist_ok=True)
