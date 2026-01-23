#!/usr/bin/env python3
"""
YOLO training pipeline.

Fine-tune YOLO models on custom datasets for person detection.
"""

import shutil
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import yaml


@dataclass
class YOLOTrainingConfig:
    """YOLO training configuration."""
    model: str = "yolo11n.pt"
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    patience: int = 50
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5
    augment: bool = True
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    device: str = ""
    workers: int = 8
    project: str = "runs/train"
    name: str = "exp"
    exist_ok: bool = False
    pretrained: bool = True
    optimizer: str = "auto"
    seed: int = 0
    deterministic: bool = True
    single_cls: bool = True
    rect: bool = False
    cos_lr: bool = False
    close_mosaic: int = 10
    resume: bool = False
    amp: bool = True
    fraction: float = 1.0
    freeze: Optional[int] = None
    multi_scale: bool = False
    overlap_mask: bool = True
    mask_ratio: int = 4
    dropout: float = 0.0
    val: bool = True
    save: bool = True
    save_period: int = -1
    cache: bool = False
    verbose: bool = True
    plots: bool = True


class YOLOTrainer:
    """
    YOLO model trainer.

    Features:
    - Fine-tuning from pretrained models
    - Custom dataset support
    - Hyperparameter management
    - Training monitoring
    - Model export
    """

    AVAILABLE_MODELS = [
        "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
    ]

    def __init__(self, output_dir: str = "runs/train"):
        """
        Initialize trainer.

        Args:
            output_dir: Base directory for training outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.training_history: List[Dict] = []

    def train(
        self,
        dataset_yaml: str,
        config: YOLOTrainingConfig = None,
        run_name: str = None
    ) -> Dict[str, Any]:
        """
        Train YOLO model.

        Args:
            dataset_yaml: Path to dataset YAML file
            config: Training configuration
            run_name: Name for this training run

        Returns:
            Training results dict
        """
        from ultralytics import YOLO

        config = config or YOLOTrainingConfig()

        if run_name is None:
            run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        config.project = str(self.output_dir)
        config.name = run_name

        print(f"Loading model: {config.model}")
        model = YOLO(config.model)

        print(f"Starting training: {run_name}")
        print(f"  Dataset: {dataset_yaml}")
        print(f"  Epochs: {config.epochs}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Image size: {config.image_size}")

        training_args = {
            "data": dataset_yaml,
            "epochs": config.epochs,
            "batch": config.batch_size,
            "imgsz": config.image_size,
            "patience": config.patience,
            "lr0": config.lr0,
            "lrf": config.lrf,
            "momentum": config.momentum,
            "weight_decay": config.weight_decay,
            "warmup_epochs": config.warmup_epochs,
            "warmup_momentum": config.warmup_momentum,
            "warmup_bias_lr": config.warmup_bias_lr,
            "box": config.box,
            "cls": config.cls,
            "dfl": config.dfl,
            "augment": config.augment,
            "hsv_h": config.hsv_h,
            "hsv_s": config.hsv_s,
            "hsv_v": config.hsv_v,
            "degrees": config.degrees,
            "translate": config.translate,
            "scale": config.scale,
            "shear": config.shear,
            "perspective": config.perspective,
            "flipud": config.flipud,
            "fliplr": config.fliplr,
            "mosaic": config.mosaic,
            "mixup": config.mixup,
            "copy_paste": config.copy_paste,
            "workers": config.workers,
            "project": config.project,
            "name": config.name,
            "exist_ok": config.exist_ok,
            "pretrained": config.pretrained,
            "optimizer": config.optimizer,
            "seed": config.seed,
            "deterministic": config.deterministic,
            "single_cls": config.single_cls,
            "rect": config.rect,
            "cos_lr": config.cos_lr,
            "close_mosaic": config.close_mosaic,
            "resume": config.resume,
            "amp": config.amp,
            "fraction": config.fraction,
            "multi_scale": config.multi_scale,
            "overlap_mask": config.overlap_mask,
            "mask_ratio": config.mask_ratio,
            "dropout": config.dropout,
            "val": config.val,
            "save": config.save,
            "save_period": config.save_period,
            "cache": config.cache,
            "verbose": config.verbose,
            "plots": config.plots,
        }

        if config.device:
            training_args["device"] = config.device

        if config.freeze:
            training_args["freeze"] = config.freeze

        results = model.train(**training_args)

        run_dir = self.output_dir / run_name
        result_data = {
            "run_name": run_name,
            "dataset": dataset_yaml,
            "model": config.model,
            "epochs_completed": config.epochs,
            "output_dir": str(run_dir),
            "best_weights": str(run_dir / "weights" / "best.pt"),
            "last_weights": str(run_dir / "weights" / "last.pt"),
            "started_at": datetime.now().isoformat(),
            "config": {
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "image_size": config.image_size,
                "lr0": config.lr0
            }
        }

        if hasattr(results, 'results_dict'):
            result_data["metrics"] = results.results_dict

        self._save_run_info(run_dir, result_data, config)

        self.training_history.append(result_data)

        print(f"\nTraining complete: {run_dir}")
        print(f"  Best weights: {result_data['best_weights']}")

        return result_data

    def _save_run_info(
        self,
        run_dir: Path,
        result_data: Dict,
        config: YOLOTrainingConfig
    ):
        """Save training run information."""
        with open(run_dir / "training_info.json", "w") as f:
            json.dump(result_data, f, indent=2, default=str)

        config_dict = {
            k: v for k, v in config.__dict__.items()
            if not k.startswith("_")
        }
        with open(run_dir / "training_config.yaml", "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def validate(
        self,
        model_path: str,
        dataset_yaml: str,
        image_size: int = 640,
        batch_size: int = 16,
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """
        Validate model on dataset.

        Args:
            model_path: Path to model weights
            dataset_yaml: Path to dataset YAML
            image_size: Image size
            batch_size: Batch size
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS

        Returns:
            Validation results
        """
        from ultralytics import YOLO

        print(f"Validating: {model_path}")
        model = YOLO(model_path)

        results = model.val(
            data=dataset_yaml,
            imgsz=image_size,
            batch=batch_size,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=True
        )

        metrics = {
            "mAP50": results.box.map50,
            "mAP50-95": results.box.map,
            "precision": results.box.mp,
            "recall": results.box.mr,
        }

        print(f"\nValidation Results:")
        print(f"  mAP50: {metrics['mAP50']:.4f}")
        print(f"  mAP50-95: {metrics['mAP50-95']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")

        return metrics

    def export(
        self,
        model_path: str,
        format: str = "onnx",
        image_size: int = 640,
        output_dir: str = None
    ) -> str:
        """
        Export model to different format.

        Args:
            model_path: Path to model weights
            format: Export format (onnx, torchscript, tflite, etc.)
            image_size: Image size
            output_dir: Output directory

        Returns:
            Path to exported model
        """
        from ultralytics import YOLO

        print(f"Exporting model: {model_path} -> {format}")
        model = YOLO(model_path)

        export_path = model.export(
            format=format,
            imgsz=image_size,
            simplify=True
        )

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            dest = output_dir / Path(export_path).name
            shutil.move(export_path, dest)
            export_path = str(dest)

        print(f"Exported to: {export_path}")
        return export_path

    def resume_training(self, run_dir: str) -> Dict[str, Any]:
        """
        Resume interrupted training.

        Args:
            run_dir: Directory of interrupted training run

        Returns:
            Training results
        """
        from ultralytics import YOLO

        run_dir = Path(run_dir)
        last_weights = run_dir / "weights" / "last.pt"

        if not last_weights.exists():
            raise FileNotFoundError(f"No checkpoint found: {last_weights}")

        print(f"Resuming training from: {last_weights}")
        model = YOLO(str(last_weights))
        results = model.train(resume=True)

        return {"status": "resumed", "run_dir": str(run_dir)}

    def get_training_history(self) -> List[Dict]:
        """Get training history."""
        return self.training_history

    def compare_models(
        self,
        model_paths: List[str],
        dataset_yaml: str
    ) -> Dict[str, Dict]:
        """
        Compare multiple models.

        Args:
            model_paths: List of model paths
            dataset_yaml: Dataset YAML

        Returns:
            Comparison results
        """
        results = {}

        for model_path in model_paths:
            name = Path(model_path).stem
            metrics = self.validate(model_path, dataset_yaml)
            results[name] = metrics

        return results


def create_dataset_yaml(
    dataset_dir: str,
    classes: List[str] = None,
    output_path: str = None
) -> str:
    """
    Create dataset YAML for YOLO training.

    Args:
        dataset_dir: Dataset directory
        classes: Class names
        output_path: Output path for YAML

    Returns:
        Path to YAML file
    """
    dataset_dir = Path(dataset_dir)
    classes = classes or ["person"]

    if output_path is None:
        output_path = dataset_dir / "dataset.yaml"

    yaml_content = {
        "path": str(dataset_dir.absolute()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(classes),
        "names": classes
    }

    with open(output_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    return str(output_path)


def main():
    """CLI for YOLO training."""
    import argparse

    parser = argparse.ArgumentParser(description="YOLO Training Pipeline")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("dataset", help="Dataset YAML path")
    train_parser.add_argument("--model", "-m", default="yolo11n.pt")
    train_parser.add_argument("--epochs", "-e", type=int, default=100)
    train_parser.add_argument("--batch", "-b", type=int, default=16)
    train_parser.add_argument("--imgsz", "-i", type=int, default=640)
    train_parser.add_argument("--name", "-n", help="Run name")
    train_parser.add_argument("--device", "-d", default="")

    val_parser = subparsers.add_parser("validate", help="Validate model")
    val_parser.add_argument("model", help="Model path")
    val_parser.add_argument("dataset", help="Dataset YAML path")

    export_parser = subparsers.add_parser("export", help="Export model")
    export_parser.add_argument("model", help="Model path")
    export_parser.add_argument("--format", "-f", default="onnx")
    export_parser.add_argument("--output", "-o", help="Output directory")

    resume_parser = subparsers.add_parser("resume", help="Resume training")
    resume_parser.add_argument("run_dir", help="Training run directory")

    args = parser.parse_args()
    trainer = YOLOTrainer()

    if args.command == "train":
        config = YOLOTrainingConfig(
            model=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            image_size=args.imgsz,
            device=args.device
        )
        trainer.train(args.dataset, config, args.name)

    elif args.command == "validate":
        trainer.validate(args.model, args.dataset)

    elif args.command == "export":
        trainer.export(args.model, args.format, output_dir=args.output)

    elif args.command == "resume":
        trainer.resume_training(args.run_dir)


if __name__ == "__main__":
    main()
