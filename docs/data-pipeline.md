# Data Pipeline Documentation

データ管理・学習パイプラインのドキュメント

## 概要

```
src/data_pipeline/
├── ingest/           # データ取り込み
│   ├── video_to_frames.py    # 動画→フレーム抽出
│   ├── image_importer.py     # 画像整理・インポート
│   └── metadata_extractor.py # メタデータ抽出
│
├── storage/          # データ管理
│   ├── dataset_manager.py    # データセット構造管理
│   └── annotation_store.py   # アノテーション保存
│
└── training/         # 学習パイプライン
    ├── train_yolo.py         # YOLO fine-tuning
    ├── train_reid.py         # Re-ID学習
    └── evaluate.py           # モデル評価
```

---

## 1. データ取り込み (Ingest)

### 動画からフレーム抽出

```python
from src.data_pipeline.ingest import VideoFrameExtractor, ExtractionConfig

extractor = VideoFrameExtractor(output_dir="data/frames")

# 60秒ごとにフレーム抽出
config = ExtractionConfig(
    interval_seconds=60.0,
    max_frames=100,
    quality=95
)

frames = extractor.extract_from_video("video.mp4", config)
print(f"Extracted {len(frames)} frames")
```

CLI:
```bash
python src/data_pipeline/ingest/video_to_frames.py video.mp4 --interval 60
python src/data_pipeline/ingest/video_to_frames.py video.mp4 --info  # 動画情報のみ
```

### 画像インポート

```python
from src.data_pipeline.ingest import ImageImporter

importer = ImageImporter(dataset_dir="data/datasets")

# ディレクトリ一括インポート
results = importer.import_directory(
    "raw_images/",
    site_id="site01",
    camera_id="cam01"
)

# 自動カメラグルーピング（ファイル名から推測）
results = importer.import_with_auto_grouping("raw_images/", site_id="site01")

# 統計情報
stats = importer.get_dataset_stats()
print(f"Total images: {stats['total_images']}")
```

CLI:
```bash
python src/data_pipeline/ingest/image_importer.py raw_images/ --site site01 --camera cam01
python src/data_pipeline/ingest/image_importer.py raw_images/ --auto-group
python src/data_pipeline/ingest/image_importer.py --stats
```

### メタデータ抽出

```python
from src.data_pipeline.ingest import MetadataExtractor

extractor = MetadataExtractor()

# ディレクトリ全体のメタデータ
results = extractor.extract_directory("images/", output_json="metadata.json")

# サマリー
summary = extractor.get_summary(results)
print(f"Cameras: {summary['cameras']}")
print(f"Date range: {summary['date_range']}")
```

CLI:
```bash
python src/data_pipeline/ingest/metadata_extractor.py images/ --output metadata.json
python src/data_pipeline/ingest/metadata_extractor.py images/ --summary
```

---

## 2. データ管理 (Storage)

### データセット管理

```python
from src.data_pipeline.storage import DatasetManager, DatasetConfig

manager = DatasetManager(base_dir="data/training_datasets")

# データセット作成
config = DatasetConfig(
    name="workers_v1",
    description="Construction site worker detection",
    classes=["person"],
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1
)
manager.create_dataset("workers_v1", config)

# 画像追加（自動分割）
manager.add_images(
    "workers_v1",
    image_paths=["img1.jpg", "img2.jpg"],
    annotation_paths=["img1.txt", "img2.txt"],
    auto_split=True
)

# COCO形式エクスポート
manager.export_to_coco("workers_v1", "annotations_coco.json")
```

CLI:
```bash
python src/data_pipeline/storage/dataset_manager.py create workers_v1
python src/data_pipeline/storage/dataset_manager.py add workers_v1 images/ --labels labels/
python src/data_pipeline/storage/dataset_manager.py list
python src/data_pipeline/storage/dataset_manager.py export workers_v1 --output coco.json
```

### アノテーション管理

```python
from src.data_pipeline.storage import AnnotationStore, BoundingBox, Annotation

store = AnnotationStore(store_dir="data/annotations")

# YOLOフォーマットからインポート
store.import_yolo(labels_dir="labels/", images_dir="images/")

# 検出結果からインポート
store.import_from_detections(detection_results, source="yolo_detection")

# YOLOフォーマットにエクスポート
store.export_yolo(output_dir="yolo_labels/")

# 統計
stats = store.get_statistics()
print(f"Total boxes: {stats['total_boxes']}")
```

---

## 3. 学習パイプライン (Training)

### YOLO Fine-tuning

```python
from src.data_pipeline.training import YOLOTrainer, YOLOTrainingConfig

trainer = YOLOTrainer(output_dir="runs/train")

config = YOLOTrainingConfig(
    model="yolo11n.pt",
    epochs=100,
    batch_size=16,
    image_size=640,
    single_cls=True,  # person only
    augment=True
)

results = trainer.train(
    dataset_yaml="data/training_datasets/workers_v1/dataset.yaml",
    config=config,
    run_name="workers_yolo_v1"
)

# 検証
metrics = trainer.validate(
    model_path="runs/train/workers_yolo_v1/weights/best.pt",
    dataset_yaml="dataset.yaml"
)

# エクスポート
trainer.export(
    model_path="runs/train/workers_yolo_v1/weights/best.pt",
    format="onnx"
)
```

CLI:
```bash
python src/data_pipeline/training/train_yolo.py train dataset.yaml --epochs 100 --batch 16
python src/data_pipeline/training/train_yolo.py validate model.pt dataset.yaml
python src/data_pipeline/training/train_yolo.py export model.pt --format onnx
```

### Re-ID学習

```python
from src.data_pipeline.training import ReIDTrainer, ReIDTrainingConfig

trainer = ReIDTrainer(output_dir="runs/reid")

# 検出結果からRe-IDデータセット作成
dataset_path = trainer.prepare_dataset_from_detections(
    detection_results,
    output_dir="data/reid_dataset"
)

config = ReIDTrainingConfig(
    backbone="resnet50",
    epochs=60,
    batch_size=32,
    embedding_dim=512,
    loss_type="triplet",
    hard_mining=True
)

results = trainer.train(
    train_dir="data/reid_dataset",
    config=config,
    run_name="workers_reid_v1"
)
```

CLI:
```bash
python src/data_pipeline/training/train_reid.py train data/reid_dataset --epochs 60
```

### モデル評価

```python
from src.data_pipeline.training import ModelEvaluator

evaluator = ModelEvaluator(output_dir="runs/eval")

# 検出モデル評価
det_metrics = evaluator.evaluate_detection(
    model_path="yolo_model.pt",
    test_images=["test1.jpg", "test2.jpg"],
    ground_truth={"test1.jpg": [[x1,y1,x2,y2], ...]}
)

# Re-IDモデル評価
reid_metrics = evaluator.evaluate_reid(
    model_path="reid_model.pt",
    query_dir="data/query",
    gallery_dir="data/gallery"
)

# レポート生成
report = evaluator.generate_report(
    detection_metrics=det_metrics,
    reid_metrics=reid_metrics,
    output_path="evaluation_report.txt"
)

# モデル比較
evaluator.compare_detection_models(
    model_paths=["model_v1.pt", "model_v2.pt"],
    test_data=ground_truth,
    output_name="model_comparison"
)
```

---

## データフロー

```
[お客様データ]
     │
     ▼
┌─────────────────┐
│  1. Ingest      │
│  ├─ 動画分解    │
│  ├─ 画像整理    │
│  └─ メタデータ  │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  2. Storage     │
│  ├─ データセット │
│  └─ アノテーション │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  3. Training    │
│  ├─ YOLO学習    │
│  ├─ Re-ID学習   │
│  └─ 評価        │
└─────────────────┘
     │
     ▼
[Fine-tuned Models]
```

---

## ディレクトリ構造（データ来た後）

```
data/
├── raw/                    # お客様から受け取った生データ
│   ├── videos/
│   └── images/
│
├── frames/                 # 動画から抽出したフレーム
│   └── {video_name}/
│
├── datasets/               # 整理されたデータセット
│   └── {site_id}/{camera_id}/{date}/
│
├── training_datasets/      # 学習用データセット
│   └── {dataset_name}/
│       ├── train/images/
│       ├── train/labels/
│       ├── val/images/
│       ├── val/labels/
│       └── dataset.yaml
│
├── annotations/            # アノテーションストア
│   └── index.json
│
└── reid_datasets/          # Re-ID学習データ
    └── {dataset_name}/
        └── {person_id}/*.jpg
```

---

## 次のステップ

1. **お客様データ受領時**:
   ```bash
   # 動画の場合
   python src/data_pipeline/ingest/video_to_frames.py data/raw/video.mp4 --interval 60

   # 画像の場合
   python src/data_pipeline/ingest/image_importer.py data/raw/images/ --auto-group
   ```

2. **データセット作成**:
   ```bash
   python src/data_pipeline/storage/dataset_manager.py create workers_v1
   python src/data_pipeline/storage/dataset_manager.py add workers_v1 data/frames/
   ```

3. **学習実行**:
   ```bash
   python src/data_pipeline/training/train_yolo.py train dataset.yaml --epochs 100
   ```
