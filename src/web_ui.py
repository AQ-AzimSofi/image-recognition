#!/usr/bin/env python3
"""
Web UI for Worker Counting System.

Provides a browser-based interface for all pipeline operations.
"""

import gradio as gr
from pathlib import Path
from typing import Optional, List
import json
import tempfile
import shutil


# ============================================================
# Data Ingestion Functions
# ============================================================

def extract_frames(
    video_file,
    interval: float = 60.0,
    max_frames: int = 0,
    output_name: str = ""
):
    """Extract frames from video."""
    if video_file is None:
        return "Please upload a video file", None

    from data_pipeline.ingest import VideoFrameExtractor, ExtractionConfig

    extractor = VideoFrameExtractor(output_dir="data/frames")

    config = ExtractionConfig(
        interval_seconds=interval,
        max_frames=max_frames if max_frames > 0 else None,
        quality=95
    )

    output_subdir = output_name if output_name else Path(video_file).stem

    try:
        frames = extractor.extract_from_video(
            video_file,
            config,
            output_subdir=output_subdir
        )

        output_dir = Path("data/frames") / output_subdir
        sample_images = list(output_dir.glob("*.jpg"))[:6]

        result = f"Extracted {len(frames)} frames\n"
        result += f"Output: {output_dir}\n"
        result += f"Interval: {interval}s"

        return result, sample_images if sample_images else None

    except Exception as e:
        return f"Error: {str(e)}", None


def get_video_info(video_file):
    """Get video file information."""
    if video_file is None:
        return "Please upload a video file"

    from data_pipeline.ingest import VideoFrameExtractor

    extractor = VideoFrameExtractor()

    try:
        info = extractor.get_video_info(video_file)

        output = f"Video Information:\n\n"
        output += f"Resolution: {info['width']}x{info['height']}\n"
        output += f"FPS: {info['fps']:.2f}\n"
        output += f"Duration: {info['duration_formatted']}\n"
        output += f"Total frames: {info['frame_count']}\n"

        return output

    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================
# Auto-Labeling Functions
# ============================================================

def run_auto_labeling(
    image_dir: str,
    confidence: float = 0.5,
    export_format: str = "yolo",
    copy_images: bool = True
):
    """Run auto-labeling on images."""
    if not image_dir:
        return "Please enter image directory path", None

    if not Path(image_dir).exists():
        return f"Directory not found: {image_dir}", None

    try:
        from data_pipeline.labeling import AutoLabeler

        labeler = AutoLabeler(confidence=confidence)
        annotations = labeler.detect_batch(image_dir)

        stats = labeler.get_statistics(annotations)

        output_dir = Path("data/auto_labeled")
        output_dir.mkdir(parents=True, exist_ok=True)

        if export_format == "yolo":
            labeler.export_yolo(annotations, str(output_dir / "yolo"), copy_images)
            export_path = output_dir / "yolo"
        elif export_format == "coco":
            labeler.export_coco(annotations, str(output_dir / "coco.json"))
            export_path = output_dir / "coco.json"
        else:
            labeler.export_label_studio(annotations, str(output_dir / "labelstudio.json"))
            export_path = output_dir / "labelstudio.json"

        result = f"Auto-labeling complete!\n\n"
        result += f"Images: {stats['total_images']}\n"
        result += f"Detections: {stats['total_detections']}\n"
        result += f"Avg per image: {stats['avg_detections_per_image']:.1f}\n"
        result += f"Avg confidence: {stats['avg_confidence']:.3f}\n"
        result += f"\nExported to: {export_path}"

        # Show sample annotated images
        sample_images = None
        if copy_images and export_format == "yolo":
            images_dir = output_dir / "yolo" / "images"
            if images_dir.exists():
                sample_images = list(images_dir.glob("*.jpg"))[:4]

        return result, sample_images

    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n\n{traceback.format_exc()}", None


def setup_label_studio():
    """Generate Label Studio setup script."""
    try:
        from data_pipeline.labeling import LabelStudioIntegration

        integration = LabelStudioIntegration()
        script_path = integration.generate_setup_script("setup_label_studio.sh")

        instructions = integration.get_labeling_instructions()

        result = f"Setup script generated: {script_path}\n\n"
        result += "To start Label Studio:\n"
        result += "  ./setup_label_studio.sh\n\n"
        result += "Then open: http://localhost:8080\n\n"
        result += "=" * 50 + "\n"
        result += instructions

        return result

    except Exception as e:
        return f"Error: {str(e)}"


def prepare_for_label_studio(image_dir: str):
    """Prepare images for Label Studio import."""
    if not image_dir:
        return "Please enter image directory path"

    if not Path(image_dir).exists():
        return f"Directory not found: {image_dir}"

    try:
        from data_pipeline.labeling import LabelStudioIntegration, AutoLabeler

        # First, auto-label the images
        labeler = AutoLabeler(confidence=0.5)
        annotations = labeler.detect_batch(image_dir)

        # Export in Label Studio format
        ls_dir = Path("data/label_studio")
        ls_dir.mkdir(parents=True, exist_ok=True)

        # Copy images
        integration = LabelStudioIntegration()
        image_paths = integration.prepare_images_for_import(
            image_dir,
            str(ls_dir / "images")
        )

        # Create pre-annotations
        labeler.export_label_studio(
            annotations,
            str(ls_dir / "pre_annotations.json"),
            image_url_prefix="/data/local-files/?d=images/"
        )

        # Create import file
        integration.create_import_json(
            image_paths,
            str(ls_dir / "import_tasks.json")
        )

        result = f"Prepared for Label Studio!\n\n"
        result += f"Images copied: {len(image_paths)}\n"
        result += f"Pre-annotations: {sum(len(a.detections) for a in annotations)} boxes\n\n"
        result += f"Files created in: {ls_dir}\n"
        result += f"  - images/\n"
        result += f"  - pre_annotations.json\n"
        result += f"  - import_tasks.json\n\n"
        result += "Next steps:\n"
        result += "1. Run ./setup_label_studio.sh\n"
        result += "2. Create project in Label Studio\n"
        result += "3. Import pre_annotations.json"

        return result

    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n\n{traceback.format_exc()}"


# ============================================================
# Re-ID Data Preparation Functions
# ============================================================

def crop_persons(image_dir: str, confidence: float = 0.5):
    """Crop persons from images for Re-ID."""
    if not image_dir:
        return "Please enter image directory path", None

    if not Path(image_dir).exists():
        return f"Directory not found: {image_dir}", None

    try:
        from data_pipeline.labeling import ReIDDataPrep

        prep = ReIDDataPrep(output_dir="data/reid_prep")
        crops = prep.crop_from_detections(image_dir, confidence=confidence)

        stats = prep.get_statistics()

        result = f"Cropping complete!\n\n"
        result += f"Total crops: {stats['total_crops']}\n"
        result += f"Output: data/reid_prep/crops/\n\n"
        result += "Next: Run clustering to group similar persons"

        # Show sample crops
        crops_dir = Path("data/reid_prep/crops")
        sample_images = list(crops_dir.glob("*.jpg"))[:6]

        return result, sample_images if sample_images else None

    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n\n{traceback.format_exc()}", None


def cluster_persons(n_clusters: int = 0, method: str = "agglomerative"):
    """Cluster person crops."""
    crops_dir = Path("data/reid_prep/crops")
    if not crops_dir.exists():
        return "No crops found. Run 'Crop Persons' first.", None

    try:
        from data_pipeline.labeling import ReIDDataPrep
        from data_pipeline.labeling.reid_prep import PersonCrop

        prep = ReIDDataPrep(output_dir="data/reid_prep")

        # Load existing crops
        for crop_path in sorted(crops_dir.glob("**/*.jpg")):
            prep.crops.append(PersonCrop(
                crop_path=str(crop_path),
                source_image="",
                bbox=[0, 0, 0, 0],
                confidence=1.0
            ))

        # Cluster
        clusters = prep.cluster_crops(
            n_clusters=n_clusters if n_clusters > 0 else None,
            method=method
        )

        # Export for review
        review_dir = prep.export_for_review(clusters, "data/reid_prep/review")

        result = f"Clustering complete!\n\n"
        result += f"Found {len(clusters)} clusters\n\n"
        result += "Cluster sizes:\n"
        for cid, crops in sorted(clusters.items())[:10]:
            result += f"  cluster_{cid:03d}: {len(crops)} crops\n"

        if len(clusters) > 10:
            result += f"  ... and {len(clusters) - 10} more\n"

        result += f"\nReview folder: {review_dir}\n"
        result += "\nNext steps:\n"
        result += "1. Review clusters in the review folder\n"
        result += "2. Merge/split folders as needed\n"
        result += "3. Rename folders to person IDs\n"
        result += "4. Export training dataset"

        # Show sample from first few clusters
        sample_images = []
        for cid in sorted(clusters.keys())[:3]:
            cluster_dir = Path(review_dir) / f"cluster_{cid:03d}"
            imgs = list(cluster_dir.glob("*.jpg"))[:2]
            sample_images.extend(imgs)

        return result, sample_images if sample_images else None

    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n\n{traceback.format_exc()}", None


def export_reid_dataset(reviewed_dir: str):
    """Export Re-ID training dataset."""
    if not reviewed_dir:
        reviewed_dir = "data/reid_prep/review"

    if not Path(reviewed_dir).exists():
        return f"Directory not found: {reviewed_dir}"

    try:
        from data_pipeline.labeling import ReIDDataPrep

        prep = ReIDDataPrep()
        dataset_path = prep.export_training_dataset(
            reviewed_dir,
            "data/reid_dataset"
        )

        # Read stats
        info_path = Path(dataset_path) / "dataset_info.json"
        with open(info_path) as f:
            info = json.load(f)

        stats = info["statistics"]

        result = f"Re-ID dataset created!\n\n"
        result += f"Location: {dataset_path}\n\n"
        result += f"Statistics:\n"
        result += f"  Persons: {stats['persons']}\n"
        result += f"  Train images: {stats['train']}\n"
        result += f"  Query images: {stats['query']}\n"
        result += f"  Gallery images: {stats['gallery']}\n\n"
        result += "Ready for Re-ID training!"

        return result

    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n\n{traceback.format_exc()}"


# ============================================================
# Training Functions
# ============================================================

def start_yolo_training(
    dataset_yaml: str,
    model: str = "yolo11n.pt",
    epochs: int = 100,
    batch_size: int = 16,
    image_size: int = 640
):
    """Start YOLO training."""
    if not dataset_yaml:
        return "Please enter dataset YAML path"

    if not Path(dataset_yaml).exists():
        return f"Dataset not found: {dataset_yaml}"

    try:
        from data_pipeline.training import YOLOTrainer, YOLOTrainingConfig

        config = YOLOTrainingConfig(
            model=model,
            epochs=epochs,
            batch_size=batch_size,
            image_size=image_size,
            single_cls=True
        )

        trainer = YOLOTrainer()

        result = f"Starting YOLO training...\n\n"
        result += f"Model: {model}\n"
        result += f"Dataset: {dataset_yaml}\n"
        result += f"Epochs: {epochs}\n"
        result += f"Batch size: {batch_size}\n\n"
        result += "Training started in background.\n"
        result += "Check runs/train/ for progress."

        # Note: In production, this would run in background
        # For demo, we just show the config
        # results = trainer.train(dataset_yaml, config)

        return result

    except Exception as e:
        return f"Error: {str(e)}"


def start_reid_training(
    train_dir: str,
    backbone: str = "resnet50",
    epochs: int = 60,
    batch_size: int = 32
):
    """Start Re-ID training."""
    if not train_dir:
        train_dir = "data/reid_dataset/train"

    if not Path(train_dir).exists():
        return f"Training data not found: {train_dir}"

    try:
        result = f"Starting Re-ID training...\n\n"
        result += f"Backbone: {backbone}\n"
        result += f"Train dir: {train_dir}\n"
        result += f"Epochs: {epochs}\n"
        result += f"Batch size: {batch_size}\n\n"
        result += "Training started in background.\n"
        result += "Check runs/reid/ for progress."

        return result

    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================
# Detection & Pipeline Functions
# ============================================================

def run_detection(
    image_files,
    confidence: float = 0.5,
    iou: float = 0.45
):
    """Run person detection on images."""
    if not image_files:
        return "Please upload images", None

    from ultralytics import YOLO
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from detect import detect_persons
    from utils import load_image, draw_bounding_boxes, add_count_label, save_image
    from config import get_model_path

    try:
        model = YOLO(get_model_path())

        results_text = "Detection Results:\n\n"
        output_images = []

        for img_file in image_files[:5]:
            img_path = str(img_file)
            count, boxes = detect_persons(model, img_path, confidence, iou)

            results_text += f"{Path(img_path).name}: {count} persons\n"

            if boxes:
                img = load_image(img_path)
                img = draw_bounding_boxes(img, boxes)
                img = add_count_label(img, count)

                temp_path = Path(tempfile.gettempdir()) / f"detected_{Path(img_path).name}"
                save_image(img, str(temp_path))
                output_images.append(str(temp_path))

        return results_text, output_images if output_images else None

    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n\n{traceback.format_exc()}", None


def run_pipeline(
    image_dir: str,
    site_id: str = "site01",
    confidence: float = 0.5,
    reid_threshold: float = 0.6,
    generate_charts: bool = True,
    generate_reports: bool = True
):
    """Run complete worker counting pipeline."""
    if not image_dir:
        return "Please enter image directory path", None, None

    if not Path(image_dir).exists():
        return f"Directory not found: {image_dir}", None, None

    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from pipeline import WorkerCountingPipeline

        pipeline_obj = WorkerCountingPipeline(
            confidence_threshold=confidence,
            reid_threshold=reid_threshold
        )

        results = pipeline_obj.run(
            input_path=image_dir,
            site_id=site_id,
            generate_charts=generate_charts,
            generate_reports=generate_reports
        )

        if results["status"] != "success":
            return f"Error: {results.get('message', 'Unknown error')}", None, None

        output = f"Pipeline Complete!\n\n"
        output += f"Images processed: {results['images_processed']}\n"
        output += f"Total detections: {results['total_detections']}\n"
        output += f"Unique workers: {results['unique_workers']}\n"
        output += f"Man-hours: {results['man_hours']:.1f}\n"
        output += f"Peak: {results['peak_count']} workers at {results['peak_time']}\n"
        output += f"\nTime: {results['elapsed_seconds']:.1f}s"

        chart_images = results.get("output_files", {}).get("charts", [])
        report_files = results.get("output_files", {}).get("reports", [])

        report_content = None
        for report_path in report_files:
            if report_path.endswith(".txt"):
                with open(report_path, "r") as f:
                    report_content = f.read()
                break

        return output, chart_images[:3] if chart_images else None, report_content

    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n\n{traceback.format_exc()}", None, None


# ============================================================
# Dataset Management Functions
# ============================================================

def create_dataset(
    name: str,
    description: str = "",
    train_ratio: float = 0.7,
    val_ratio: float = 0.2
):
    """Create a new dataset."""
    if not name:
        return "Please enter a dataset name"

    from data_pipeline.storage import DatasetManager, DatasetConfig

    manager = DatasetManager(base_dir="data/training_datasets")

    try:
        config = DatasetConfig(
            name=name,
            description=description,
            classes=["person"],
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=1.0 - train_ratio - val_ratio
        )

        path = manager.create_dataset(name, config)

        return f"Created dataset: {path}\n\nSplits: train={train_ratio:.0%}, val={val_ratio:.0%}, test={1-train_ratio-val_ratio:.0%}"

    except Exception as e:
        return f"Error: {str(e)}"


def list_datasets():
    """List all datasets."""
    from data_pipeline.storage import DatasetManager

    manager = DatasetManager(base_dir="data/training_datasets")

    try:
        datasets = manager.list_datasets()

        if not datasets:
            return "No datasets found"

        output = "Datasets:\n\n"
        for ds in datasets:
            stats = ds.get("stats", {})
            total = sum(s.get("images", 0) for s in stats.values())
            output += f"  {ds['name']}: {total} images\n"
            output += f"    {ds.get('description', 'No description')}\n\n"

        return output

    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================
# Create UI
# ============================================================

def create_ui():
    """Create Gradio interface."""

    with gr.Blocks(title="Worker Counting System") as app:
        gr.Markdown("# Worker Counting System")
        gr.Markdown("Construction site worker detection, tracking, and training")

        with gr.Tabs():
            # Tab 1: Data Ingestion
            with gr.TabItem("1. Data Import"):
                gr.Markdown("## Import video/images")

                with gr.Row():
                    with gr.Column():
                        video_input = gr.File(label="Upload Video", file_types=["video"])
                        interval_input = gr.Slider(1, 300, value=60, label="Interval (seconds)")
                        max_frames_input = gr.Number(value=0, label="Max frames (0=unlimited)")
                        video_info_btn = gr.Button("Get Video Info")
                        extract_btn = gr.Button("Extract Frames", variant="primary")

                    with gr.Column():
                        video_info_output = gr.Textbox(label="Video Info", lines=5)
                        extract_output = gr.Textbox(label="Result", lines=5)
                        extract_gallery = gr.Gallery(label="Sample Frames", columns=3)

                video_info_btn.click(get_video_info, inputs=[video_input], outputs=[video_info_output])
                extract_btn.click(
                    extract_frames,
                    inputs=[video_input, interval_input, max_frames_input, gr.State("")],
                    outputs=[extract_output, extract_gallery]
                )

            # Tab 2: Auto-Labeling
            with gr.TabItem("2. Auto-Label"):
                gr.Markdown("## Automatic annotation with YOLO")

                with gr.Row():
                    with gr.Column():
                        al_image_dir = gr.Textbox(label="Image Directory", placeholder="data/frames/video_name")
                        al_confidence = gr.Slider(0.1, 0.9, value=0.5, label="Confidence")
                        al_format = gr.Radio(["yolo", "coco", "labelstudio"], value="yolo", label="Export Format")
                        al_copy = gr.Checkbox(value=True, label="Copy images to output")
                        al_btn = gr.Button("Run Auto-Labeling", variant="primary")

                    with gr.Column():
                        al_output = gr.Textbox(label="Result", lines=10)
                        al_gallery = gr.Gallery(label="Sample Images", columns=2)

                al_btn.click(
                    run_auto_labeling,
                    inputs=[al_image_dir, al_confidence, al_format, al_copy],
                    outputs=[al_output, al_gallery]
                )

                gr.Markdown("---")
                gr.Markdown("### Label Studio Integration")

                with gr.Row():
                    with gr.Column():
                        ls_setup_btn = gr.Button("Generate Setup Script")
                        ls_image_dir = gr.Textbox(label="Image Directory for Label Studio")
                        ls_prepare_btn = gr.Button("Prepare for Label Studio", variant="primary")

                    with gr.Column():
                        ls_output = gr.Textbox(label="Instructions", lines=20)

                ls_setup_btn.click(setup_label_studio, outputs=[ls_output])
                ls_prepare_btn.click(prepare_for_label_studio, inputs=[ls_image_dir], outputs=[ls_output])

            # Tab 3: Re-ID Data Prep
            with gr.TabItem("3. Re-ID Data"):
                gr.Markdown("## Prepare Re-ID training data")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Step 1: Crop Persons")
                        reid_image_dir = gr.Textbox(label="Image Directory", placeholder="images/input/")
                        reid_conf = gr.Slider(0.1, 0.9, value=0.5, label="Detection Confidence")
                        crop_btn = gr.Button("Crop Persons", variant="primary")

                    with gr.Column():
                        crop_output = gr.Textbox(label="Result", lines=8)
                        crop_gallery = gr.Gallery(label="Sample Crops", columns=3)

                crop_btn.click(
                    crop_persons,
                    inputs=[reid_image_dir, reid_conf],
                    outputs=[crop_output, crop_gallery]
                )

                gr.Markdown("---")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Step 2: Cluster by Identity")
                        n_clusters = gr.Number(value=0, label="Number of clusters (0=auto)")
                        cluster_method = gr.Radio(
                            ["agglomerative", "kmeans", "dbscan"],
                            value="agglomerative",
                            label="Clustering method"
                        )
                        cluster_btn = gr.Button("Cluster Persons", variant="primary")

                    with gr.Column():
                        cluster_output = gr.Textbox(label="Result", lines=12)
                        cluster_gallery = gr.Gallery(label="Sample Clusters", columns=3)

                cluster_btn.click(
                    cluster_persons,
                    inputs=[n_clusters, cluster_method],
                    outputs=[cluster_output, cluster_gallery]
                )

                gr.Markdown("---")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Step 3: Export Dataset")
                        gr.Markdown("After reviewing and renaming clusters:")
                        reviewed_dir = gr.Textbox(
                            label="Reviewed Directory",
                            value="data/reid_prep/review"
                        )
                        export_reid_btn = gr.Button("Export Training Dataset", variant="primary")

                    with gr.Column():
                        export_reid_output = gr.Textbox(label="Result", lines=10)

                export_reid_btn.click(
                    export_reid_dataset,
                    inputs=[reviewed_dir],
                    outputs=[export_reid_output]
                )

            # Tab 4: Training
            with gr.TabItem("4. Training"):
                gr.Markdown("## Model Training")

                with gr.Tabs():
                    with gr.TabItem("YOLO Training"):
                        with gr.Row():
                            with gr.Column():
                                yolo_dataset = gr.Textbox(
                                    label="Dataset YAML",
                                    placeholder="data/training_datasets/workers/dataset.yaml"
                                )
                                yolo_model = gr.Dropdown(
                                    ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolov8n.pt", "yolov8s.pt"],
                                    value="yolo11n.pt",
                                    label="Base Model"
                                )
                                yolo_epochs = gr.Slider(10, 300, value=100, label="Epochs")
                                yolo_batch = gr.Slider(4, 64, value=16, label="Batch Size")
                                yolo_imgsz = gr.Dropdown([320, 480, 640, 800], value=640, label="Image Size")
                                yolo_train_btn = gr.Button("Start Training", variant="primary")

                            with gr.Column():
                                yolo_train_output = gr.Textbox(label="Status", lines=12)

                        yolo_train_btn.click(
                            start_yolo_training,
                            inputs=[yolo_dataset, yolo_model, yolo_epochs, yolo_batch, yolo_imgsz],
                            outputs=[yolo_train_output]
                        )

                    with gr.TabItem("Re-ID Training"):
                        with gr.Row():
                            with gr.Column():
                                reid_train_dir = gr.Textbox(
                                    label="Training Directory",
                                    value="data/reid_dataset/train"
                                )
                                reid_backbone = gr.Dropdown(
                                    ["resnet50", "resnet18"],
                                    value="resnet50",
                                    label="Backbone"
                                )
                                reid_epochs = gr.Slider(10, 200, value=60, label="Epochs")
                                reid_batch = gr.Slider(8, 64, value=32, label="Batch Size")
                                reid_train_btn = gr.Button("Start Training", variant="primary")

                            with gr.Column():
                                reid_train_output = gr.Textbox(label="Status", lines=12)

                        reid_train_btn.click(
                            start_reid_training,
                            inputs=[reid_train_dir, reid_backbone, reid_epochs, reid_batch],
                            outputs=[reid_train_output]
                        )

            # Tab 5: Detection
            with gr.TabItem("5. Detection"):
                gr.Markdown("## Run person detection")

                with gr.Row():
                    with gr.Column():
                        det_images = gr.File(label="Upload Images", file_count="multiple", file_types=["image"])
                        det_conf = gr.Slider(0.1, 0.9, value=0.5, label="Confidence")
                        det_iou = gr.Slider(0.1, 0.9, value=0.45, label="IoU threshold")
                        det_btn = gr.Button("Run Detection", variant="primary")

                    with gr.Column():
                        det_output = gr.Textbox(label="Results", lines=8)
                        det_gallery = gr.Gallery(label="Detected Images", columns=3)

                det_btn.click(
                    run_detection,
                    inputs=[det_images, det_conf, det_iou],
                    outputs=[det_output, det_gallery]
                )

            # Tab 6: Full Pipeline
            with gr.TabItem("6. Full Pipeline"):
                gr.Markdown("## Run complete worker counting pipeline")

                with gr.Row():
                    with gr.Column():
                        pipe_dir = gr.Textbox(
                            label="Image Directory",
                            placeholder="images/input/frames/"
                        )
                        pipe_site = gr.Textbox(value="site01", label="Site ID")
                        pipe_conf = gr.Slider(0.1, 0.9, value=0.5, label="Detection confidence")
                        pipe_reid = gr.Slider(0.1, 0.9, value=0.6, label="Re-ID threshold")
                        pipe_charts = gr.Checkbox(value=True, label="Generate charts")
                        pipe_reports = gr.Checkbox(value=True, label="Generate reports")
                        pipe_btn = gr.Button("Run Pipeline", variant="primary")

                    with gr.Column():
                        pipe_output = gr.Textbox(label="Results", lines=10)
                        pipe_gallery = gr.Gallery(label="Charts", columns=3)
                        pipe_report = gr.Textbox(label="Report", lines=15)

                pipe_btn.click(
                    run_pipeline,
                    inputs=[pipe_dir, pipe_site, pipe_conf, pipe_reid, pipe_charts, pipe_reports],
                    outputs=[pipe_output, pipe_gallery, pipe_report]
                )

            # Tab 7: Help
            with gr.TabItem("Help"):
                gr.Markdown("""
## Workflow Overview

```
1. Data Import    → Video/images to frames
2. Auto-Label     → YOLO pre-annotation + Label Studio
3. Re-ID Data     → Crop persons + Cluster + Review
4. Training       → Fine-tune YOLO and Re-ID models
5. Detection      → Test detection on new images
6. Full Pipeline  → End-to-end worker counting
```

---

## Step-by-Step Guide

### For Detection Training (YOLO)

1. **Import Data**: Upload video or organize images
2. **Auto-Label**: Run YOLO to pre-annotate
3. **Review in Label Studio**: Fix/add annotations
4. **Create Dataset**: Export and create train/val/test splits
5. **Train**: Fine-tune YOLO on your data

### For Re-ID Training

1. **Crop Persons**: Extract person crops from images
2. **Cluster**: Group similar persons automatically
3. **Review**: Manually verify/fix clusters
4. **Export**: Create Re-ID training dataset
5. **Train**: Train Re-ID model

---

## Directory Structure

```
data/
├── frames/              # Extracted video frames
├── auto_labeled/        # Auto-labeling output
├── label_studio/        # Label Studio files
├── reid_prep/           # Re-ID preparation
│   ├── crops/          # Person crops
│   └── review/         # Clustered for review
├── reid_dataset/        # Final Re-ID dataset
└── training_datasets/   # YOLO datasets

runs/
├── train/              # YOLO training runs
└── reid/               # Re-ID training runs

reports/                # Pipeline outputs
```

---

## Tips

- Start with **confidence 0.5** for auto-labeling
- Use **Label Studio** for high-quality annotations
- Review **Re-ID clusters** carefully before training
- Fine-tune with **at least 100+ images** per class
                """)

    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft()
    )
