from fastapi import FastAPI

from .routes_detect import router as detect_router
from .routes_history import router as history_router
from .routes_feedback import router as feedback_router
from .routes_analysis import router as analysis_router
from .routes_stress import router as stress_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Object Detection Kaizen API",
        description="AWS Rekognition object detection with mislabel analysis and stress testing",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.include_router(detect_router)
    app.include_router(history_router)
    app.include_router(feedback_router)
    app.include_router(analysis_router)
    app.include_router(stress_router)

    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    return app


app = create_app()
