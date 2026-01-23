# API リファレンス

> 現在実装済みのモジュールの詳細仕様

---

## 1. detect.py - 人物検出モジュール

### 概要
YOLOモデルを使用して画像から人物を検出する。

### CLI使用方法

```bash
# 単一画像の検出
python src/detect.py images/input/photo.jpg --save

# ディレクトリ一括処理
python src/detect.py images/input/online/ --save

# パラメータ指定
python src/detect.py images/input/ --confidence 0.3 --iou 0.4 --model yolo11m.pt --save
```

### コマンドライン引数

| 引数 | 型 | デフォルト | 説明 |
|------|-----|----------|------|
| `input` | str | 必須 | 画像ファイルまたはディレクトリのパス |
| `--confidence` | float | 0.5 | 検出confidence閾値（0.0-1.0） |
| `--iou` | float | 0.45 | NMS IoU閾値（低いほど重複除去が厳しい） |
| `--model` | str | yolo11n.pt | 使用するYOLOモデル |
| `--save` / `-s` | flag | False | アノテーション画像を保存 |
| `--output-dir` | str | images/output | 出力ディレクトリ |

### Python API

```python
from detect import detect_persons, process_single_image

# 基本的な検出
count, boxes = detect_persons(
    image_path="images/input/photo.jpg",
    confidence_threshold=0.5,
    iou_threshold=0.45,
    model_path="yolo11n.pt"
)

# 戻り値
# count: int - 検出された人数
# boxes: List[List[float]] - [[x1, y1, x2, y2, confidence], ...]

# 画像処理（検出 + 保存）
process_single_image(
    image_path="images/input/photo.jpg",
    confidence_threshold=0.5,
    iou_threshold=0.45,
    model_path="yolo11n.pt",
    save_output=True,
    output_dir="images/output"
)
```

### 出力例

**コンソール出力:**
```
==================================================
Image: images/input/site1.jpg
Persons detected: 3
Confidence threshold: 0.5

Detections:
  Person 1: confidence=0.92, box=[120, 85, 280, 450]
  Person 2: confidence=0.87, box=[350, 100, 480, 430]
  Person 3: confidence=0.61, box=[500, 200, 600, 400]
```

**保存画像:**
- 緑色のバウンディングボックス
- 各ボックスに "Person N" ラベル
- 左上に "Total Persons: N" 表示

### 利用可能なモデル

| モデル | サイズ | 速度 | 精度 |
|--------|--------|------|------|
| yolo11n.pt | 5.4MB | 最速 | 低 |
| yolo11s.pt | 19MB | 速い | 中低 |
| yolo11m.pt | 39MB | 中 | 中 |
| yolo11l.pt | 50MB | 遅い | 中高 |
| yolo11x.pt | 110MB | 最遅 | 高 |

---

## 2. accuracy.py - 精度評価モジュール

### 概要
検出結果をGround Truthと比較し、精度メトリクスを算出する。

### CLI使用方法

```bash
# 基本的な精度テスト
python src/accuracy.py --images-dir images/input/actual-TLC00046/

# カスタムGround Truth使用
python src/accuracy.py --ground-truth data/custom_truth.json --images-dir images/input/

# 結果をJSONに保存
python src/accuracy.py --images-dir images/input/ --output reports/accuracy.json
```

### コマンドライン引数

| 引数 | 型 | デフォルト | 説明 |
|------|-----|----------|------|
| `--ground-truth` | str | data/ground_truth.json | Ground Truthファイル |
| `--images-dir` | str | images/input | 画像ディレクトリ |
| `--confidence` | float | 0.5 | 検出confidence閾値 |
| `--model` | str | yolo11n.pt | 使用するYOLOモデル |
| `--output` / `-o` | str | None | 結果JSONの保存先 |

### Ground Truthフォーマット

```json
{
  "images": [
    {
      "filename": "frame_01.jpg",
      "actual_count": 3,
      "notes": "朝8時、3名作業中"
    },
    {
      "filename": "frame_02.jpg",
      "actual_count": 5,
      "notes": "10時、ピーク時間帯"
    }
  ]
}
```

### Python API

```python
from accuracy import load_ground_truth, calculate_metrics, run_accuracy_test

# Ground Truth読み込み
ground_truth = load_ground_truth("data/ground_truth.json")

# 単一画像の評価
metrics = calculate_metrics(
    detected_count=3,
    actual_count=4
)
# 戻り値: {
#   "correct": False,
#   "difference": -1,
#   "false_negatives": 1,
#   "false_positives": 0
# }

# 全体テスト実行
results = run_accuracy_test(
    ground_truth_path="data/ground_truth.json",
    images_dir="images/input/",
    confidence_threshold=0.5,
    model_path="yolo11n.pt"
)
```

### 出力メトリクス

| メトリクス | 説明 | 計算式 |
|----------|------|--------|
| exact_match_accuracy | 完全一致率 | 一致画像数 / 総画像数 |
| detection_rate | 検出率 | 検出人数 / 実際人数 |
| precision | 精度 | 正検出 / (正検出 + 誤検出) |
| false_negatives | 見逃し数 | 実際 - 検出（負の場合0） |
| false_positives | 誤検出数 | 検出 - 実際（負の場合0） |

### 出力例

**コンソール出力:**
```
============================================================
                    ACCURACY TEST REPORT
============================================================

Summary:
  Images tested:        24
  Total actual persons: 50
  Total detected:       48

Metrics:
  Exact match accuracy: 75.00%
  Detection rate:       96.00%
  Precision:            97.92%

Errors:
  Total false negatives: 2
  Total false positives: 0

============================================================
```

---

## 3. extract_frames.py - フレーム抽出モジュール

### 概要
動画ファイルから静止画フレームを抽出する。

### CLI使用方法

```bash
# デフォルト動画からフレーム抽出（1秒間隔）
python src/extract_frames.py

# 全フレーム抽出
python src/extract_frames.py --all

# カスタム設定
python src/extract_frames.py --video path/to/video.mp4 --output frames/ --interval 5.0
```

### コマンドライン引数

| 引数 | 型 | デフォルト | 説明 |
|------|-----|----------|------|
| `--video` | str | private/data/TLC00046.AVI | 入力動画ファイル |
| `--output` | str | images/input/frames/ | 出力ディレクトリ |
| `--all` | flag | False | 全フレームを抽出 |
| `--interval` | float | 1.0 | 抽出間隔（秒） |

### Python API

```python
from extract_frames import extract_frames

# 指定間隔でフレーム抽出
filenames = extract_frames(
    video_path="private/data/TLC00046.AVI",
    output_dir="images/input/frames/",
    all_frames=False,
    interval=1.0
)
# 戻り値: ["frame_0001.jpg", "frame_0002.jpg", ...]

# 全フレーム抽出
filenames = extract_frames(
    video_path="video.mp4",
    output_dir="output/",
    all_frames=True
)
```

### 出力

- ファイル名形式: `frame_NNNN.jpg`（4桁ゼロ埋め）
- 画質: JPEG（OpenCVデフォルト品質）

---

## 4. utils.py - ユーティリティモジュール

### 概要
画像処理のヘルパー関数群。

### Python API

```python
from utils import load_image, save_image, draw_bounding_boxes, add_count_label

# 画像読み込み
image = load_image("path/to/image.jpg")
# 戻り値: numpy.ndarray (BGR形式)

# 画像保存
save_image(image, "path/to/output.jpg")
# ディレクトリが存在しない場合は自動作成

# バウンディングボックス描画
boxes = [[100, 100, 200, 300, 0.95], [250, 150, 350, 400, 0.87]]
annotated = draw_bounding_boxes(
    image,
    boxes,
    color=(0, 255, 0),  # BGR: 緑
    thickness=2
)
# 各ボックスに "Person N (conf)" ラベルを追加

# カウントラベル追加
labeled = add_count_label(image, count=3)
# 左上に "Total Persons: 3" を追加

# 検出結果のフォーマット
result = format_detection_result(
    image_path="input.jpg",
    count=3,
    boxes=[[100, 100, 200, 300, 0.95], ...]
)
# 戻り値: {
#   "image": "input.jpg",
#   "count": 3,
#   "detections": [{"box": [...], "confidence": 0.95}, ...]
# }
```

---

## 5. 定数・設定

### detect.py 内の定数

```python
PERSON_CLASS_ID = 0  # YOLOの人物クラスID
```

### 推奨パラメータ

| ユースケース | confidence | iou | model |
|-------------|------------|-----|-------|
| 高精度重視 | 0.5 | 0.45 | yolo11x.pt |
| バランス | 0.4 | 0.45 | yolo11m.pt |
| 高速処理 | 0.5 | 0.5 | yolo11n.pt |
| 見逃し防止 | 0.3 | 0.4 | yolo11l.pt |

---

## 6. エラーハンドリング

### 共通エラー

| エラー | 原因 | 対処 |
|--------|------|------|
| FileNotFoundError | 画像/動画が見つからない | パスを確認 |
| ValueError | 無効なパラメータ | 範囲を確認（0.0-1.0） |
| cv2.error | 画像読み込み失敗 | ファイル形式を確認 |

### モデルエラー

| エラー | 原因 | 対処 |
|--------|------|------|
| ModuleNotFoundError | ultralytics未インストール | `pip install ultralytics` |
| RuntimeError | モデルファイル破損 | モデルを再ダウンロード |

---

## 7. 今後追加予定のAPI

### reid.py（Phase 2）

```python
class PersonReID:
    def __init__(self, model_name: str = "osnet"):
        pass

    def extract_features(self, image: np.ndarray, boxes: List) -> List[np.ndarray]:
        """人物領域から特徴ベクトルを抽出"""
        pass

    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """2つの特徴ベクトル間の類似度を計算"""
        pass

    def match_persons(self, features1: List, features2: List, threshold: float = 0.7) -> List[Tuple]:
        """2セット間の人物マッチング"""
        pass
```

### aggregator.py（Phase 3）

```python
class WorkerAggregator:
    def __init__(self, reid_model: PersonReID):
        pass

    def process_sequence(self, image_paths: List[str]) -> Dict:
        """時系列画像を処理して集計"""
        pass

    def count_unique_workers(self, detections: List) -> int:
        """ユニークな作業員数を算出"""
        pass

    def calculate_man_hours(self, hourly_counts: Dict) -> float:
        """実人工を算出"""
        pass
```

---

*最終更新: 2026-01-21*
