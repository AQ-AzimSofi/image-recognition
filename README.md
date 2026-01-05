# YOLO Person Detection for Construction Sites

Person detection system using YOLO for counting workers on construction sites from static images.

## Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Detect persons in images

```bash
# Single image
python src/detect.py images/input/photo.jpg

# Directory of images (with annotated output)
python src/detect.py images/input/online/ --save

# Adjust confidence threshold
python src/detect.py images/input/ --confidence 0.3 --save

# Adjust NMS to reduce duplicate detections
python src/detect.py images/input/ --iou 0.4 --save
```

Output images with bounding boxes are saved to `images/output/`.

### Test accuracy against ground truth

1. Add your test images to `images/input/`
2. Manually count persons in each image
3. Update `data/ground_truth.json`:
   ```json
   {
       "images": [
           {"image": "photo1.jpg", "actual_count": 3},
           {"image": "photo2.jpg", "actual_count": 5}
       ]
   }
   ```
4. Run accuracy test:
   ```bash
   python src/accuracy.py
   ```

### Extract frames from video

```bash
python src/extract_frames.py
```

Extracts frames from timelapse video in `private-data/` to `images/input/`.

## Project Structure

```
src/
  detect.py         # Main detection script
  accuracy.py       # Accuracy testing against ground truth
  extract_frames.py # Video frame extraction
  utils.py          # Image manipulation helpers

images/
  input/            # Test images
  output/           # Detection results with bounding boxes

data/
  ground_truth.json # Manual counts for accuracy testing
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--confidence` | 0.5 | Minimum detection confidence (0.0-1.0) |
| `--iou` | 0.45 | NMS threshold - lower removes more duplicates |
| `--model` | yolo11n.pt | YOLO model (n=nano, s=small, m=medium) |

## Example Output

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
