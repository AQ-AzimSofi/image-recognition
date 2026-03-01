from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from ..db import Database
from ..image_utils import draw_labels_on_image
from ..models import DetectedLabel, DetectedInstance
from .dependencies import get_db

router = APIRouter(prefix="/api", tags=["history"])


@router.get("/detections")
async def list_detections(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    label: str | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    db: Database = Depends(get_db),
):
    summaries = db.get_history(limit, offset, label, date_from, date_to)
    return [
        {
            "id": s.id,
            "image_filename": s.image_filename,
            "detected_at": s.detected_at,
            "label_count": s.label_count,
            "top_labels": s.top_labels,
            "has_feedback": s.has_feedback,
        }
        for s in summaries
    ]


@router.get("/detections/{detection_id}")
async def get_detection(detection_id: int, db: Database = Depends(get_db)):
    detection = db.get_detection(detection_id)
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")
    return {
        "id": detection.id,
        "image_filename": detection.image_filename,
        "image_path": detection.image_path,
        "image_width": detection.image_width,
        "image_height": detection.image_height,
        "detected_at": detection.detected_at,
        "min_confidence": detection.min_confidence,
        "max_labels": detection.max_labels,
        "label_count": detection.label_count,
        "labels": [
            {
                "id": l.id,
                "name": l.name,
                "confidence": l.confidence,
                "has_bounding_box": l.has_bounding_box,
                "bbox": l.bbox,
                "feedback_status": l.feedback_status,
            }
            for l in detection.labels
        ],
        "notes": detection.notes,
    }


@router.get("/detections/{detection_id}/image")
async def get_detection_image(detection_id: int, db: Database = Depends(get_db)):
    detection = db.get_detection(detection_id)
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")

    image_path = Path(detection.image_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    return FileResponse(str(image_path))


@router.get("/detections/{detection_id}/annotated")
async def get_annotated_image(detection_id: int, db: Database = Depends(get_db)):
    detection = db.get_detection(detection_id)
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")

    image_path = Path(detection.image_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    labels = []
    for l in detection.labels:
        instances = []
        if l.has_bounding_box and l.bbox:
            instances.append(
                DetectedInstance(
                    bbox_left=l.bbox["left"],
                    bbox_top=l.bbox["top"],
                    bbox_width=l.bbox["width"],
                    bbox_height=l.bbox["height"],
                    confidence=l.confidence,
                )
            )
        labels.append(
            DetectedLabel(
                name=l.name,
                confidence=l.confidence,
                instances=instances,
            )
        )

    annotated = draw_labels_on_image(image_path, labels)

    from io import BytesIO
    from fastapi.responses import StreamingResponse

    buffer = BytesIO()
    annotated.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")


@router.delete("/detections/{detection_id}")
async def delete_detection(detection_id: int, db: Database = Depends(get_db)):
    deleted = db.delete_detection(detection_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Detection not found")
    return {"deleted": True}
