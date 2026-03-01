from fastapi import APIRouter, Depends

from ..db import Database
from .dependencies import get_db

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


@router.get("/summary")
async def get_summary(db: Database = Depends(get_db)):
    stats = db.get_analysis_stats()
    return {
        "total_detections": stats.total_detections,
        "total_labels": stats.total_labels,
        "total_reviewed": stats.total_reviewed,
        "accuracy_rate": round(stats.accuracy_rate, 4),
        "wrong_reason_count": stats.wrong_reason_count,
    }


@router.get("/misclassifications")
async def get_misclassifications(db: Database = Depends(get_db)):
    stats = db.get_analysis_stats()
    return stats.common_misclassifications


@router.get("/confidence-distribution")
async def get_confidence_distribution(db: Database = Depends(get_db)):
    stats = db.get_analysis_stats()
    return stats.confidence_distribution
