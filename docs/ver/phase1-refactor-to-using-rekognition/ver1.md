# Phase 1: Refactor to AWS Rekognition Object Detection System

## Background & Motivation

The original codebase was a YOLO-based person detection system for construction sites (worker counting with Re-ID tracking). The project scope shifted to a general-purpose **object detection system** using AWS Rekognition, targeting office and construction site environments.

The core requirement is not just object detection, but a **kaizen (improvement) system** that can:
1. Detect when the AI mislabels objects
2. Detect when the AI gets the right answer but from wrong reasoning (e.g., bounding box covers the wrong object)
3. Stress test the AI to understand its limits

### Why AWS Rekognition over YOLO

- Team has AWS expertise (AIF-C01, MLA-C01 certified, AWS partner company)
- Managed service means no model training/hosting infrastructure
- Pay-per-image pricing ($0.001/image) ideal for POC
- DetectLabels API provides bounding boxes + confidence scores out of the box
- Custom Labels available for domain-specific equipment detection later

## What Changed

### Removed Files (old YOLO system)

All source files from the previous YOLO-based person detection system were removed:

| File | Purpose (old) |
|------|--------------|
| `src/detect.py` | YOLO person detection |
| `src/reid.py` | Re-ID feature extraction and person matching |
| `src/aggregator.py` | Time-series aggregation for worker counting |
| `src/visualizer.py` | Matplotlib chart generation |
| `src/reporter.py` | JSON/CSV report generation |
| `src/pipeline.py` | End-to-end orchestrator |
| `src/accuracy.py` | Ground truth comparison |
| `src/utils.py` | Image processing helpers (OpenCV-based) |
| `src/web_ui.py` | Gradio web interface for YOLO pipeline |
| `src/data_pipeline/` | Full ML data pipeline (ingest, storage, labeling, training) |

### New Files Created

#### Core Modules (Phase 1 Foundation)

**`src/config.py`** - Rewritten with new dataclass configs:
- `RekognitionConfig` (region, min_confidence, max_labels) replaces `ModelConfig`, `DetectionConfig`, `ReIDConfig`
- `APIConfig` (host, port for FastAPI) and `UIConfig` (host, port for Gradio) added
- `DatabaseConfig` (SQLite path) added
- `AppConfig` composes all configs, replacing `PipelineConfig`
- Design decision: AWS region defaults to `ap-northeast-1`

**`src/models.py`** - Shared dataclasses used across API and UI:
- `DetectedInstance` / `DetectedLabel` / `DetectionResult` - Rekognition response models
- `DetectionSummary` / `DetectionDetail` / `LabelDetail` - Database view models
- `FeedbackEntry` - Human review input model
- `AnalysisStats` - Analytics aggregation model
- Design decision: Used Python 3.10+ union syntax (`dict | None`) for cleaner type hints

**`src/db.py`** - SQLite database layer with full CRUD:
- 4 tables: `detections`, `labels`, `feedback`, `stress_tests`
- Context manager for connection lifecycle with auto-commit/rollback
- `save_detection()` stores image metadata + all detected labels in a single transaction
- `get_detection()` joins labels with feedback to produce a complete view
- `get_history()` supports filtering by label name, date range, with pagination
- `save_feedback()` uses upsert pattern (update if label already reviewed)
- `get_analysis_stats()` computes accuracy rate, confidence distribution, common misclassifications
- Design decision: Used raw `sqlite3` instead of SQLAlchemy (no extra dependency, sufficient for POC)

**`src/rekognition.py`** - AWS Rekognition client wrapper:
- Wraps `boto3.client('rekognition')` with custom error hierarchy (`RekognitionError`, `CredentialsError`, `ImageTooLargeError`, `InvalidImageError`)
- `detect_labels()` sends image bytes directly (no S3 required)
- `detect_labels_from_path()` convenience method for file-based detection
- `_parse_labels()` converts Rekognition JSON response to typed dataclasses
- Pre-flight check: validates image size < 5MB before API call
- Design decision: Images sent as bytes, not via S3, to keep demo simple

**`src/image_utils.py`** - Pillow-based image processing:
- `draw_labels_on_image()` - Overlay colored bounding boxes with label text
- `crop_bounding_box()` - Crop and zoom into a specific detection region
- `generate_color_map()` - HSV-based distinct colors per label
- `get_image_dimensions()` - Read dimensions without loading full image
- `image_to_bytes()` - Convert PIL Image to bytes for API calls
- 7 degradation functions for stress testing: blur, darken, brighten, noise, crop, jpeg_compress, resize_down
- Each degradation takes a level (0.0-1.0) for consistent parameterization
- Design decision: Pillow over OpenCV - cleaner API, no BGR/RGB confusion, Gradio native compatibility

#### FastAPI Server (Phase 2)

**`src/api/app.py`** - FastAPI application factory:
- Creates app with title, description, and auto-generated Swagger UI at `/docs`
- Includes all route modules via `include_router()`
- Health check endpoint at `/health`

**`src/api/dependencies.py`** - Dependency injection:
- `get_db()` and `get_rekognition_client()` as `lru_cache` singletons
- Injected into route handlers via FastAPI's `Depends()`

**`src/api/routes_detect.py`** - Core detection endpoint:
- `POST /api/detect` - Accepts multipart file upload, optional confidence/max_labels params
- Saves uploaded image to `images/input/` with timestamp to avoid collisions
- Calls Rekognition, stores results in SQLite, returns structured JSON
- Handles both labels with bounding boxes (physical objects) and without (abstract/scene)

**`src/api/routes_history.py`** - History browsing:
- `GET /api/detections` - Paginated list with label/date filters
- `GET /api/detections/{id}` - Full detail with label feedback status
- `GET /api/detections/{id}/image` - Serve original image file
- `GET /api/detections/{id}/annotated` - On-the-fly annotated image generation (bounding boxes drawn fresh each time from stored label data)
- `DELETE /api/detections/{id}` - Cascade deletes labels and feedback

**`src/api/routes_feedback.py`** - Kaizen review:
- `POST /api/feedback` - Submit per-label feedback (correct/incorrect/wrong_reason)
- Validates that label belongs to the specified detection
- `GET /api/detections/{id}/feedback` - Get feedback status for all labels in a detection

**`src/api/routes_analysis.py`** - Analytics:
- `GET /api/analysis/summary` - Total counts, accuracy rate, wrong reason count
- `GET /api/analysis/misclassifications` - Top common misclassification pairs
- `GET /api/analysis/confidence-distribution` - Bucketed by 10% intervals

**`src/api/routes_stress.py`** - Stress testing:
- `POST /api/stress-test` - Upload image, specify degradation type + level, get comparison
- Makes 2 Rekognition API calls (original + degraded), computes label diff (lost/gained/kept)
- Stores both detections and the stress test relationship in SQLite
- `GET /api/stress-tests` - List past stress test results

#### Gradio UI (Phase 3)

**`src/ui/app.py`** - Assembles 6 tabs into a single Gradio Blocks application.

**`src/ui/tab_detect.py`** - Detection tab:
- Image upload + confidence/max_labels sliders
- Shows annotated image with bounding boxes and results table
- Collapsible raw JSON response accordion
- Every detection auto-saves to SQLite

**`src/ui/tab_history.py`** - History tab:
- Filter by label name, date range
- Shows detection list, click to load detail with annotated image
- Shows per-label feedback status

**`src/ui/tab_review.py`** - Review tab (core kaizen functionality):
- Load by detection ID or "Next Unreviewed" button
- Full annotated image with colored per-label bounding boxes
- Dropdown to select individual label for review
- Cropped bounding box preview (zoomed in to see what the AI "saw")
- Radio: Correct / Incorrect / Not Sure
- Checkbox: "Right Answer, Wrong Reason" flag
- Expected label text field (for mislabeling cases)
- Reviewer notes
- Auto-refreshes after feedback submission

**`src/ui/tab_analysis.py`** - Analysis dashboard:
- Summary stats: total detections, labels, reviewed count, accuracy rate, wrong reason count
- Confidence distribution chart (matplotlib bar chart, correct vs incorrect by confidence bucket)
- Common misclassifications table

**`src/ui/tab_stress.py`** - Stress test tab:
- Upload image, select degradation type and severity level
- Side-by-side comparison: original vs degraded (both annotated)
- Comparison table: label name, original confidence, degraded confidence, delta, status (kept/lost/gained)
- Cost warning: each test costs ~$0.002 (2 API calls)

**`src/ui/tab_capabilities.py`** - Capabilities explorer:
- Runs detection at minimum confidence (1%) with max labels (100)
- Shows everything Rekognition can detect in the image
- Summary: total labels, with/without bounding box breakdown, categories found
- Useful for understanding the label taxonomy and what's in scope

#### Entry Point (Phase 4)

**`src/main.py`** - CLI entry point:
- `--mode api` - FastAPI server only (port 8000)
- `--mode ui` - Gradio UI only (port 7860)
- `--mode both` - Both servers (FastAPI on daemon thread, Gradio on main thread)
- Initializes directories, database, and Rekognition client on startup

### Modified Files

**`requirements.txt`** - Dependencies updated:
- Removed: ultralytics, opencv-python, torch, torchvision, scikit-learn, pandas
- Added: boto3, fastapi, uvicorn, python-multipart
- Kept: gradio, Pillow, numpy, matplotlib

**`.gitignore`** - Added `*.db` to ignore SQLite database files.

**`README.md`** - Completely rewritten for the new project.

## Architecture Decisions

### Why Two Servers (FastAPI + Gradio)

- FastAPI provides the actual API endpoints that a production frontend would consume. Swagger UI comes free for API testing.
- Gradio provides a rich demo/testing interface for the PM and kaizen analysis. It calls core modules directly (no HTTP overhead).
- Both share the same `rekognition.py`, `db.py`, and `image_utils.py` core.

### Why SQLite

- Zero setup, no external database server needed
- Built into Python, no extra dependency
- Sufficient for POC/demo volume
- Easy to inspect with any SQLite client
- Can migrate to DynamoDB or RDS later if needed

### Why Pillow over OpenCV

- Cleaner API (no BGR/RGB confusion)
- Better text and shape drawing for bounding box overlays
- Gradio natively works with PIL Images
- Lighter dependency (no C++ build toolchain needed)

### BoundingBox Storage Strategy

Rekognition returns bounding box coordinates as ratios (0.0-1.0) relative to image dimensions. These ratios are stored as-is in SQLite. Conversion to pixel coordinates happens only at rendering time in `image_utils.py`. This keeps the stored data resolution-independent.

### Image Storage Strategy

Uploaded images are saved locally to `images/input/` with a timestamp suffix for uniqueness. The file path is stored in the `detections` table. Annotated images (with bounding boxes) are generated on-the-fly from the original image + stored label data, not persisted separately. This avoids stale overlays if feedback changes.

## Verification

After each phase:
- Phase 1: Run a Python script to send an image to Rekognition, store in SQLite, draw bounding boxes
- Phase 2: Open Swagger UI at `localhost:8000/docs`, test each endpoint
- Phase 3: Open Gradio at `localhost:7860`, test each tab end-to-end
- Phase 4: Run `python -m src.main --mode both` and verify both servers start
