# Object Detection Kaizen System

AWS Rekognition-based object detection system with mislabel analysis and stress testing for office and construction site environments.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

### AWS Credentials

Configure AWS credentials via one of:
- Environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- AWS credentials file: `~/.aws/credentials`
- IAM role (if running on AWS infrastructure)

## Usage

### Run both API server and Gradio UI

```bash
python -m src.main --mode both
```

- Gradio UI: http://localhost:7860
- FastAPI Swagger UI: http://localhost:8000/docs

### Run API server only

```bash
python -m src.main --mode api
```

### Run Gradio UI only

```bash
python -m src.main --mode ui
```

## Project Structure

```
src/
    config.py           # Configuration (AWS region, thresholds, paths)
    models.py           # Shared data models
    db.py               # SQLite database layer
    rekognition.py      # AWS Rekognition client wrapper
    image_utils.py      # BoundingBox drawing, image degradation
    main.py             # Entry point

    api/                # FastAPI server
        app.py          # App factory
        routes_detect.py
        routes_history.py
        routes_feedback.py
        routes_analysis.py
        routes_stress.py

    ui/                 # Gradio demo UI
        app.py          # Tab assembly
        tab_detect.py
        tab_history.py
        tab_review.py
        tab_analysis.py
        tab_stress.py
        tab_capabilities.py
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/detect` | Upload image for object detection |
| GET | `/api/detections` | List detection history |
| GET | `/api/detections/{id}` | Get detection detail |
| GET | `/api/detections/{id}/annotated` | Get annotated image |
| DELETE | `/api/detections/{id}` | Delete detection |
| POST | `/api/feedback` | Submit label feedback |
| GET | `/api/analysis/summary` | Get accuracy statistics |
| POST | `/api/stress-test` | Run image degradation test |

## Gradio UI Tabs

1. **Detection** - Upload image, adjust confidence, see annotated results
2. **History** - Browse past detections with filters
3. **Review** - Per-label feedback (correct/incorrect/wrong-reason)
4. **Analysis** - Accuracy dashboard with charts
5. **Stress Test** - Image degradation testing
6. **Capabilities** - Explore Rekognition's detection range
