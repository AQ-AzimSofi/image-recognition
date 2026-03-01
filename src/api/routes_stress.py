from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, Query, UploadFile, HTTPException
from PIL import Image

from ..config import IMAGES_DIR
from ..db import Database
from ..image_utils import apply_degradation, image_to_bytes, get_image_dimensions, DEGRADATION_TYPES
from ..rekognition import RekognitionClient, RekognitionError
from .dependencies import get_db, get_rekognition_client

router = APIRouter(prefix="/api", tags=["stress-test"])


@router.post("/stress-test")
async def run_stress_test(
    file: UploadFile = File(...),
    degradation_type: str = Query(..., enum=DEGRADATION_TYPES),
    level: float = Query(0.5, ge=0.0, le=1.0),
    min_confidence: float = Query(50.0, ge=0, le=100),
    db: Database = Depends(get_db),
    client: RekognitionClient = Depends(get_rekognition_client),
):
    contents = await file.read()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_stem = Path(file.filename).stem if file.filename else "upload"
    ext = Path(file.filename).suffix if file.filename else ".jpg"

    original_path = IMAGES_DIR / "input" / f"{original_stem}_{timestamp}{ext}"
    original_path.parent.mkdir(parents=True, exist_ok=True)
    original_path.write_bytes(contents)

    try:
        original_result = client.detect_labels(contents, min_confidence)
    except RekognitionError as e:
        raise HTTPException(status_code=400, detail=str(e))

    width, height = get_image_dimensions(original_path)

    original_labels_data = _result_to_labels_data(original_result)
    original_id = db.save_detection(
        image_filename=original_path.name,
        image_path=str(original_path),
        image_width=width,
        image_height=height,
        labels_data=original_labels_data,
        raw_response=original_result.raw_response,
        min_confidence=min_confidence,
        max_labels=20,
    )

    img = Image.open(original_path).convert("RGB")
    degraded = apply_degradation(img, degradation_type, level)
    degraded_bytes = image_to_bytes(degraded)

    degraded_filename = f"{original_stem}_{timestamp}_{degradation_type}_{level}{ext}"
    degraded_path = IMAGES_DIR / "input" / degraded_filename
    degraded_path.write_bytes(degraded_bytes)

    try:
        degraded_result = client.detect_labels(degraded_bytes, min_confidence)
    except RekognitionError as e:
        raise HTTPException(status_code=400, detail=str(e))

    degraded_labels_data = _result_to_labels_data(degraded_result)
    degraded_id = db.save_detection(
        image_filename=degraded_filename,
        image_path=str(degraded_path),
        image_width=width,
        image_height=height,
        labels_data=degraded_labels_data,
        raw_response=degraded_result.raw_response,
        min_confidence=min_confidence,
        max_labels=20,
    )

    original_names = {l.name for l in original_result.labels}
    degraded_names = {l.name for l in degraded_result.labels}

    label_diff = {
        "labels_lost": sorted(original_names - degraded_names),
        "labels_gained": sorted(degraded_names - original_names),
        "labels_kept": sorted(original_names & degraded_names),
    }

    db.save_stress_test(original_id, degradation_type, level, degraded_id, label_diff)

    return {
        "source_detection_id": original_id,
        "degraded_detection_id": degraded_id,
        "degradation_type": degradation_type,
        "level": level,
        "label_diff": label_diff,
        "original_label_count": len(original_names),
        "degraded_label_count": len(degraded_names),
    }


@router.get("/stress-tests")
async def list_stress_tests(
    source_detection_id: int | None = Query(None),
    db: Database = Depends(get_db),
):
    return db.get_stress_tests(source_detection_id)


def _result_to_labels_data(result) -> list[dict]:
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
    return labels_data
