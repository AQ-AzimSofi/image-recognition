import argparse
import logging

from .config import DEFAULT_CONFIG, ensure_directories
from .db import Database
from .rekognition import RekognitionClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


def run_api(db, client):
    import uvicorn
    from .api.app import app

    config = DEFAULT_CONFIG.api
    uvicorn.run(app, host=config.host, port=config.port, reload=config.reload)


def run_ui(db, client):
    from .ui.app import create_ui

    config = DEFAULT_CONFIG.ui
    app = create_ui(client, db)
    app.launch(server_name=config.host, server_port=config.port, share=config.share)


def run_both(db, client):
    import threading
    import uvicorn
    from .api.app import app
    from .ui.app import create_ui

    api_config = DEFAULT_CONFIG.api
    ui_config = DEFAULT_CONFIG.ui

    api_thread = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": api_config.host, "port": api_config.port},
        daemon=True,
    )
    api_thread.start()
    print(f"API server started at http://{api_config.host}:{api_config.port}")
    print(f"Swagger UI at http://{api_config.host}:{api_config.port}/docs")

    ui_app = create_ui(client, db)
    ui_app.launch(server_name=ui_config.host, server_port=ui_config.port, share=ui_config.share)


def main():
    parser = argparse.ArgumentParser(description="Object Detection Kaizen System")
    parser.add_argument(
        "--mode",
        choices=["api", "ui", "both"],
        default="both",
        help="Run mode: api (FastAPI only), ui (Gradio only), both (default)",
    )
    args = parser.parse_args()

    ensure_directories()
    db = Database()
    client = RekognitionClient()

    runners = {"api": run_api, "ui": run_ui, "both": run_both}
    runners[args.mode](db, client)


if __name__ == "__main__":
    main()
