from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..db import Database
from ..models import FeedbackEntry
from .dependencies import get_db

router = APIRouter(prefix="/api", tags=["feedback"])


class FeedbackRequest(BaseModel):
    label_id: int
    detection_id: int
    is_correct: bool | None = None
    is_wrong_reason: bool = False
    expected_label: str | None = None
    reviewer_notes: str | None = None


@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    db: Database = Depends(get_db),
):
    detection = db.get_detection(request.detection_id)
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")

    label_ids = {l.id for l in detection.labels}
    if request.label_id not in label_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Label {request.label_id} does not belong to detection {request.detection_id}",
        )

    entry = FeedbackEntry(
        label_id=request.label_id,
        detection_id=request.detection_id,
        is_correct=request.is_correct,
        is_wrong_reason=request.is_wrong_reason,
        expected_label=request.expected_label,
        reviewer_notes=request.reviewer_notes,
    )
    feedback_id = db.save_feedback(entry)
    return {"feedback_id": feedback_id}


@router.get("/detections/{detection_id}/feedback")
async def get_detection_feedback(
    detection_id: int,
    db: Database = Depends(get_db),
):
    detection = db.get_detection(detection_id)
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")

    return [
        {
            "label_id": l.id,
            "label_name": l.name,
            "confidence": l.confidence,
            "feedback_status": l.feedback_status,
        }
        for l in detection.labels
    ]
