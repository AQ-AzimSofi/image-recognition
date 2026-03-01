import shutil
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, Query, UploadFile, HTTPException

from ..config import IMAGES_DIR
from ..db import Database
from ..image_utils import get_image_dimensions
from ..rekognition import RekognitionClient, RekognitionError
from .dependencies import get_db, get_rekognition_client

router = APIRouter(prefix="/api", tags=["detection"])


@router.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    min_confidence: float = Query(50.0, ge=0, le=100),
    max_labels: int = Query(20, ge=1, le=100),
    db: Database = Depends(get_db),
    client: RekognitionClient = Depends(get_rekognition_client),
):
    contents = await file.read()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_stem = Path(file.filename).stem if file.filename else "upload"
    ext = Path(file.filename).suffix if file.filename else ".jpg"
    saved_filename = f"{original_stem}_{timestamp}{ext}"
    saved_path = IMAGES_DIR / "input" / saved_filename
    saved_path.parent.mkdir(parents=True, exist_ok=True)
    saved_path.write_bytes(contents)

    try:
        result = client.detect_labels(contents, min_confidence, max_labels)
    except RekognitionError as e:
        raise HTTPException(status_code=400, detail=str(e))

    width, height = get_image_dimensions(saved_path)
    result.image_width = width
    result.image_height = height

    labels_data = []
    for label in result.labels:
        if label.instances:
            for inst in label.instances:
                labels_data.append({
                    "name": label.name,
                    "confidence": label.confidence,
                    "category": ", ".join(label.categories),
                    "parents": ", ".join(label.parents),
                    "has_bounding_box": 1,
                    "bbox_left": inst.bbox_left,
                    "bbox_top": inst.bbox_top,
                    "bbox_width": inst.bbox_width,
                    "bbox_height": inst.bbox_height,
                    "instance_confidence": inst.confidence,
                })
        else:
            labels_data.append({
                "name": label.name,
                "confidence": label.confidence,
                "category": ", ".join(label.categories),
                "parents": ", ".join(label.parents),
                "has_bounding_box": 0,
            })

    detection_id = db.save_detection(
        image_filename=saved_filename,
        image_path=str(saved_path),
        image_width=width,
        image_height=height,
        labels_data=labels_data,
        raw_response=result.raw_response,
        min_confidence=min_confidence,
        max_labels=max_labels,
    )

    detection = db.get_detection(detection_id)
    return {
        "detection_id": detection_id,
        "image_filename": saved_filename,
        "labels": [
            {
                "id": l.id,
                "name": l.name,
                "confidence": l.confidence,
                "has_bounding_box": l.has_bounding_box,
                "bbox": l.bbox,
            }
            for l in detection.labels
        ],
    }
