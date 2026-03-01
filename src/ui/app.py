import gradio as gr

from ..db import Database
from ..rekognition import RekognitionClient
from .tab_detect import create_tab as create_detect_tab
from .tab_history import create_tab as create_history_tab
from .tab_review import create_tab as create_review_tab
from .tab_analysis import create_tab as create_analysis_tab
from .tab_stress import create_tab as create_stress_tab
from .tab_capabilities import create_tab as create_capabilities_tab


def create_ui(client: RekognitionClient, db: Database) -> gr.Blocks:
    with gr.Blocks(title="Object Detection Kaizen") as app:
        gr.Markdown("# Object Detection Kaizen System")
        gr.Markdown("AWS Rekognition object detection with mislabel analysis and stress testing")

        with gr.Tabs():
            with gr.Tab("Detection"):
                create_detect_tab(client, db)

            with gr.Tab("History"):
                create_history_tab(db)

            with gr.Tab("Review"):
                create_review_tab(db)

            with gr.Tab("Analysis"):
                create_analysis_tab(db)

            with gr.Tab("Stress Test"):
                create_stress_tab(client, db)

            with gr.Tab("Capabilities"):
                create_capabilities_tab(client)

    return app
