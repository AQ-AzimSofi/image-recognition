#!/usr/bin/env python3
"""
Re-ID training pipeline.

Train person re-identification models for tracking across cameras.
"""

import shutil
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np


@dataclass
class ReIDTrainingConfig:
    """Re-ID training configuration."""
    backbone: str = "resnet50"
    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 0.0003
    weight_decay: float = 5e-4
    lr_scheduler: str = "step"
    lr_step_size: int = 20
    lr_gamma: float = 0.1
    warmup_epochs: int = 5
    input_size: Tuple[int, int] = (256, 128)
    embedding_dim: int = 512
    margin: float = 0.3
    loss_type: str = "triplet"
    hard_mining: bool = True
    num_instances: int = 4
    label_smoothing: float = 0.1
    dropout: float = 0.5
    pretrained: bool = True
    freeze_backbone_epochs: int = 0
    augmentation: bool = True
    random_erasing: float = 0.5
    color_jitter: bool = True
    horizontal_flip: float = 0.5
    device: str = ""
    num_workers: int = 4
    save_freq: int = 10
    eval_freq: int = 5
    project: str = "runs/reid"
    name: str = "exp"


class TripletDataset(Dataset):
    """Dataset for triplet loss training."""

    def __init__(
        self,
        root_dir: str,
        transform=None,
        num_instances: int = 4
    ):
        """
        Initialize dataset.

        Args:
            root_dir: Root directory with person_id subdirectories
            transform: Image transforms
            num_instances: Number of instances per identity in batch
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.num_instances = num_instances

        self.data = []
        self.pid_to_indices = {}
        self.pids = []

        self._load_data()

    def _load_data(self):
        """Load dataset structure."""
        pid_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])

        for pid_idx, pid_dir in enumerate(pid_dirs):
            pid = pid_dir.name
            self.pids.append(pid)
            self.pid_to_indices[pid] = []

            for img_path in pid_dir.glob("*"):
                if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                    idx = len(self.data)
                    self.data.append({
                        'path': str(img_path),
                        'pid': pid,
                        'pid_idx': pid_idx
                    })
                    self.pid_to_indices[pid].append(idx)

        print(f"Loaded {len(self.data)} images from {len(self.pids)} identities")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        img = Image.open(item['path']).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, item['pid_idx'], item['pid']

    def get_num_classes(self):
        return len(self.pids)


class TripletBatchSampler:
    """Batch sampler for triplet loss (P identities x K instances)."""

    def __init__(self, dataset: TripletDataset, batch_size: int, num_instances: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances

        self.valid_pids = [
            pid for pid in dataset.pids
            if len(dataset.pid_to_indices[pid]) >= num_instances
        ]

    def __iter__(self):
        np.random.shuffle(self.valid_pids)

        batch = []
        for pid in self.valid_pids:
            indices = self.dataset.pid_to_indices[pid]
            if len(indices) >= self.num_instances:
                selected = np.random.choice(indices, self.num_instances, replace=False)
                batch.extend(selected)

                if len(batch) >= self.batch_size:
                    yield batch[:self.batch_size]
                    batch = batch[self.batch_size:]

    def __len__(self):
        return len(self.valid_pids) * self.num_instances // self.batch_size


class ReIDModel(nn.Module):
    """Re-ID model with embedding head."""

    def __init__(
        self,
        backbone: str = "resnet50",
        embedding_dim: int = 512,
        num_classes: int = None,
        dropout: float = 0.5,
        pretrained: bool = True
    ):
        super().__init__()

        if backbone == "resnet50":
            base = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            )
            self.backbone = nn.Sequential(*list(base.children())[:-2])
            self.feature_dim = 2048
        elif backbone == "resnet18":
            base = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.backbone = nn.Sequential(*list(base.children())[:-2])
            self.feature_dim = 512
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.bottleneck = nn.Sequential(
            nn.Linear(self.feature_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.embedding_dim = embedding_dim

        self.classifier = None
        if num_classes:
            self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, return_features=False):
        features = self.backbone(x)
        features = self.gap(features)
        features = features.view(features.size(0), -1)

        embeddings = self.bottleneck(features)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        if return_features:
            return embeddings

        if self.classifier is not None:
            logits = self.classifier(embeddings)
            return embeddings, logits

        return embeddings


class TripletLoss(nn.Module):
    """Triplet loss with hard mining."""

    def __init__(self, margin: float = 0.3, hard_mining: bool = True):
        super().__init__()
        self.margin = margin
        self.hard_mining = hard_mining

    def forward(self, embeddings, labels):
        dist_mat = self._compute_distance_matrix(embeddings)

        if self.hard_mining:
            return self._hard_mining_triplet_loss(dist_mat, labels)
        else:
            return self._batch_all_triplet_loss(dist_mat, labels)

    def _compute_distance_matrix(self, embeddings):
        n = embeddings.size(0)
        dist = torch.pow(embeddings, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist = dist - 2 * torch.mm(embeddings, embeddings.t())
        dist = dist.clamp(min=1e-12).sqrt()
        return dist

    def _hard_mining_triplet_loss(self, dist_mat, labels):
        n = dist_mat.size(0)

        mask_pos = labels.expand(n, n).eq(labels.expand(n, n).t())
        mask_neg = labels.expand(n, n).ne(labels.expand(n, n).t())

        dist_ap = []
        dist_an = []

        for i in range(n):
            pos_mask = mask_pos[i]
            neg_mask = mask_neg[i]

            if pos_mask.sum() > 1:
                dist_ap.append(dist_mat[i][pos_mask].max())
            else:
                dist_ap.append(dist_mat[i][pos_mask].mean())

            if neg_mask.sum() > 0:
                dist_an.append(dist_mat[i][neg_mask].min())
            else:
                dist_an.append(torch.tensor(0.0, device=dist_mat.device))

        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)

        loss = torch.clamp(dist_ap - dist_an + self.margin, min=0)
        return loss.mean()

    def _batch_all_triplet_loss(self, dist_mat, labels):
        n = dist_mat.size(0)

        mask_pos = labels.expand(n, n).eq(labels.expand(n, n).t())
        mask_neg = labels.expand(n, n).ne(labels.expand(n, n).t())

        anchor_pos = dist_mat.unsqueeze(2)
        anchor_neg = dist_mat.unsqueeze(1)

        triplet_loss = anchor_pos - anchor_neg + self.margin

        mask = mask_pos.unsqueeze(2) * mask_neg.unsqueeze(1)
        triplet_loss = triplet_loss * mask.float()

        triplet_loss = torch.clamp(triplet_loss, min=0)

        valid_triplets = (triplet_loss > 1e-16).float().sum()
        if valid_triplets > 0:
            return triplet_loss.sum() / valid_triplets
        return torch.tensor(0.0, device=dist_mat.device)


class ReIDTrainer:
    """
    Re-ID model trainer.

    Features:
    - Triplet loss training
    - Hard mining
    - Cross-entropy loss option
    - Model evaluation
    """

    def __init__(self, output_dir: str = "runs/reid"):
        """
        Initialize trainer.

        Args:
            output_dir: Output directory for training runs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_transforms(self, config: ReIDTrainingConfig, is_train: bool):
        """Get data transforms."""
        if is_train and config.augmentation:
            transform_list = [
                transforms.Resize(config.input_size),
                transforms.RandomHorizontalFlip(p=config.horizontal_flip),
            ]

            if config.color_jitter:
                transform_list.append(
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.1
                    )
                )

            transform_list.extend([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            if config.random_erasing > 0:
                transform_list.append(
                    transforms.RandomErasing(p=config.random_erasing)
                )
        else:
            transform_list = [
                transforms.Resize(config.input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]

        return transforms.Compose(transform_list)

    def train(
        self,
        train_dir: str,
        val_dir: str = None,
        config: ReIDTrainingConfig = None,
        run_name: str = None
    ) -> Dict[str, Any]:
        """
        Train Re-ID model.

        Args:
            train_dir: Training data directory (person_id subdirs)
            val_dir: Validation data directory
            config: Training configuration
            run_name: Name for this run

        Returns:
            Training results
        """
        config = config or ReIDTrainingConfig()

        if run_name is None:
            run_name = f"reid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        run_dir = self.output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        device = torch.device(
            config.device if config.device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Using device: {device}")

        train_transform = self._get_transforms(config, is_train=True)
        train_dataset = TripletDataset(train_dir, train_transform, config.num_instances)

        train_sampler = TripletBatchSampler(
            train_dataset,
            config.batch_size,
            config.num_instances
        )

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=config.num_workers,
            pin_memory=True
        )

        num_classes = train_dataset.get_num_classes()

        model = ReIDModel(
            backbone=config.backbone,
            embedding_dim=config.embedding_dim,
            num_classes=num_classes if config.loss_type == "softmax" else None,
            dropout=config.dropout,
            pretrained=config.pretrained
        )
        model = model.to(device)

        triplet_loss = TripletLoss(margin=config.margin, hard_mining=config.hard_mining)
        ce_loss = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        if config.lr_scheduler == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.lr_step_size,
                gamma=config.lr_gamma
            )
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.epochs
            )

        print(f"Starting Re-ID training: {run_name}")
        print(f"  Train identities: {num_classes}")
        print(f"  Train images: {len(train_dataset)}")
        print(f"  Epochs: {config.epochs}")
        print(f"  Batch size: {config.batch_size}")

        history = {"train_loss": [], "lr": []}
        best_loss = float("inf")

        for epoch in range(config.epochs):
            model.train()
            epoch_loss = 0
            num_batches = 0

            for batch_idx, (images, labels, _) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                if config.loss_type == "triplet":
                    embeddings = model(images, return_features=True)
                    loss = triplet_loss(embeddings, labels)
                elif config.loss_type == "softmax":
                    embeddings, logits = model(images)
                    loss = ce_loss(logits, labels)
                else:
                    embeddings, logits = model(images)
                    loss = triplet_loss(embeddings, labels) + ce_loss(logits, labels)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            scheduler.step()

            avg_loss = epoch_loss / max(num_batches, 1)
            current_lr = optimizer.param_groups[0]['lr']

            history["train_loss"].append(avg_loss)
            history["lr"].append(current_lr)

            print(f"Epoch {epoch+1}/{config.epochs} - Loss: {avg_loss:.4f} - LR: {current_lr:.6f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), run_dir / "best.pt")

            if (epoch + 1) % config.save_freq == 0:
                torch.save(model.state_dict(), run_dir / f"epoch_{epoch+1}.pt")

        torch.save(model.state_dict(), run_dir / "last.pt")

        result_data = {
            "run_name": run_name,
            "train_dir": train_dir,
            "num_identities": num_classes,
            "num_images": len(train_dataset),
            "epochs_completed": config.epochs,
            "best_loss": best_loss,
            "output_dir": str(run_dir),
            "best_weights": str(run_dir / "best.pt"),
            "config": {
                "backbone": config.backbone,
                "embedding_dim": config.embedding_dim,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate
            },
            "history": history
        }

        with open(run_dir / "training_info.json", "w") as f:
            json.dump(result_data, f, indent=2)

        print(f"\nTraining complete: {run_dir}")
        print(f"  Best loss: {best_loss:.4f}")
        print(f"  Best weights: {run_dir / 'best.pt'}")

        return result_data

    def prepare_dataset_from_detections(
        self,
        detection_results: Dict[str, List],
        output_dir: str,
        min_size: int = 50
    ) -> str:
        """
        Prepare Re-ID dataset from detection results.

        Args:
            detection_results: Dict mapping image_path to PersonDetection list
            output_dir: Output directory
            min_size: Minimum crop size

        Returns:
            Path to prepared dataset
        """
        import cv2

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for img_path, detections in detection_results.items():
            img = cv2.imread(img_path)
            if img is None:
                continue

            for det in detections:
                if not det.person_id:
                    continue

                x1, y1, x2, y2 = [int(v) for v in det.box[:4]]
                if x2 - x1 < min_size or y2 - y1 < min_size:
                    continue

                crop = img[y1:y2, x1:x2]

                person_dir = output_dir / det.person_id
                person_dir.mkdir(exist_ok=True)

                existing = len(list(person_dir.glob("*.jpg")))
                crop_path = person_dir / f"{existing:04d}.jpg"
                cv2.imwrite(str(crop_path), crop)

        stats = {
            pid.name: len(list(pid.glob("*.jpg")))
            for pid in output_dir.iterdir()
            if pid.is_dir()
        }

        print(f"Prepared Re-ID dataset: {output_dir}")
        print(f"  Identities: {len(stats)}")
        print(f"  Total crops: {sum(stats.values())}")

        return str(output_dir)


def main():
    """CLI for Re-ID training."""
    import argparse

    parser = argparse.ArgumentParser(description="Re-ID Training Pipeline")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("train_dir", help="Training data directory")
    train_parser.add_argument("--val", help="Validation directory")
    train_parser.add_argument("--backbone", default="resnet50")
    train_parser.add_argument("--epochs", "-e", type=int, default=60)
    train_parser.add_argument("--batch", "-b", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=0.0003)
    train_parser.add_argument("--name", "-n", help="Run name")

    prepare_parser = subparsers.add_parser("prepare", help="Prepare dataset")
    prepare_parser.add_argument("--output", "-o", required=True)

    args = parser.parse_args()
    trainer = ReIDTrainer()

    if args.command == "train":
        config = ReIDTrainingConfig(
            backbone=args.backbone,
            epochs=args.epochs,
            batch_size=args.batch,
            learning_rate=args.lr
        )
        trainer.train(args.train_dir, args.val, config, args.name)


if __name__ == "__main__":
    main()
