#!/usr/bin/env python3
"""
Label Studio integration module.

Provides utilities for working with Label Studio:
- Project setup
- Import/Export
- API integration
"""

from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import json
import subprocess
import time


@dataclass
class LabelStudioConfig:
    """Label Studio configuration."""
    url: str = "http://localhost:8080"
    api_key: str = ""
    project_name: str = "Worker Detection"
    label_config: str = ""


class LabelStudioIntegration:
    """
    Label Studio integration utilities.

    Features:
    - Project setup with proper labeling interface
    - Import pre-annotations
    - Export annotations in various formats
    - Local storage configuration
    """

    DEFAULT_LABEL_CONFIG = """
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="person" background="green"/>
  </RectangleLabels>
</View>
    """.strip()

    def __init__(self, config: LabelStudioConfig = None):
        """
        Initialize integration.

        Args:
            config: Label Studio configuration
        """
        self.config = config or LabelStudioConfig()
        if not self.config.label_config:
            self.config.label_config = self.DEFAULT_LABEL_CONFIG

    def generate_setup_script(self, output_path: str = "setup_label_studio.sh") -> str:
        """
        Generate shell script to set up Label Studio.

        Args:
            output_path: Output script path

        Returns:
            Path to generated script
        """
        script = '''#!/bin/bash
# Label Studio Setup Script
# Generated for Worker Detection Project

echo "Setting up Label Studio..."

# Check if Label Studio is installed
if ! command -v label-studio &> /dev/null; then
    echo "Installing Label Studio..."
    pip install label-studio
fi

# Create data directory
mkdir -p data/label_studio
mkdir -p data/label_studio/images

# Set environment variables
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=$(pwd)/data/label_studio

echo ""
echo "==================================="
echo "Starting Label Studio..."
echo "==================================="
echo ""
echo "1. Open http://localhost:8080 in browser"
echo "2. Create account (first time only)"
echo "3. Create new project"
echo "4. Use this labeling config:"
echo ""
cat << 'EOF'
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="person" background="green"/>
  </RectangleLabels>
</View>
EOF
echo ""
echo "5. Import images from: data/label_studio/images/"
echo "6. Import pre-annotations from: data/label_studio/pre_annotations.json"
echo ""

# Start Label Studio
label-studio start --data-dir data/label_studio
'''
        with open(output_path, "w") as f:
            f.write(script)

        # Make executable
        Path(output_path).chmod(0o755)

        print(f"Generated setup script: {output_path}")
        return output_path

    def prepare_images_for_import(
        self,
        image_dir: str,
        output_dir: str = "data/label_studio/images"
    ) -> List[str]:
        """
        Prepare images for Label Studio import.

        Args:
            image_dir: Source image directory
            output_dir: Label Studio images directory

        Returns:
            List of prepared image paths
        """
        import shutil

        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        prepared = []

        for img_path in sorted(image_dir.glob("*")):
            if img_path.suffix.lower() in extensions:
                dest = output_dir / img_path.name
                shutil.copy2(img_path, dest)
                prepared.append(str(dest))

        print(f"Prepared {len(prepared)} images in {output_dir}")
        return prepared

    def create_import_json(
        self,
        image_paths: List[str],
        output_path: str = "data/label_studio/import_tasks.json",
        image_url_prefix: str = "/data/local-files/?d=images/"
    ) -> str:
        """
        Create JSON file for importing images to Label Studio.

        Args:
            image_paths: List of image file paths
            output_path: Output JSON path
            image_url_prefix: URL prefix for local files

        Returns:
            Path to output file
        """
        tasks = []

        for img_path in image_paths:
            img_name = Path(img_path).name
            tasks.append({
                "data": {
                    "image": f"{image_url_prefix}{img_name}"
                }
            })

        with open(output_path, "w") as f:
            json.dump(tasks, f, indent=2)

        print(f"Created import file: {output_path} ({len(tasks)} tasks)")
        return output_path

    def create_pre_annotated_import(
        self,
        annotations_path: str,
        output_path: str = "data/label_studio/pre_annotations.json"
    ) -> str:
        """
        Convert auto-labeled annotations to Label Studio import format.

        Args:
            annotations_path: Path to Label Studio format annotations (from auto_labeler)
            output_path: Output path

        Returns:
            Path to output file
        """
        import shutil
        shutil.copy2(annotations_path, output_path)
        print(f"Prepared pre-annotations: {output_path}")
        return output_path

    def export_to_yolo(
        self,
        export_json_path: str,
        output_dir: str,
        images_dir: str
    ) -> str:
        """
        Convert Label Studio export to YOLO format.

        Args:
            export_json_path: Path to Label Studio JSON export
            output_dir: Output directory
            images_dir: Directory containing images

        Returns:
            Path to output directory
        """
        with open(export_json_path, "r") as f:
            tasks = json.load(f)

        output_dir = Path(output_dir)
        labels_dir = output_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        class_names = set()

        for task in tasks:
            image_url = task.get("data", {}).get("image", "")
            img_name = Path(image_url).name.split("?")[0]
            if "=" in img_name:
                img_name = img_name.split("=")[-1]

            # Get annotations
            annotations = task.get("annotations", [])
            if not annotations:
                continue

            results = annotations[0].get("result", [])

            # Find image dimensions from first result
            orig_width = 100
            orig_height = 100
            for r in results:
                if "original_width" in r:
                    orig_width = r["original_width"]
                    orig_height = r["original_height"]
                    break

            # Write YOLO label file
            label_path = labels_dir / f"{Path(img_name).stem}.txt"

            with open(label_path, "w") as f:
                for r in results:
                    if r.get("type") != "rectanglelabels":
                        continue

                    value = r.get("value", {})
                    x_pct = value.get("x", 0)
                    y_pct = value.get("y", 0)
                    w_pct = value.get("width", 0)
                    h_pct = value.get("height", 0)
                    labels = value.get("rectanglelabels", [])

                    if not labels:
                        continue

                    label = labels[0]
                    class_names.add(label)

                    # Convert to YOLO format (normalized center x, y, width, height)
                    x_center = (x_pct + w_pct / 2) / 100
                    y_center = (y_pct + h_pct / 2) / 100
                    width = w_pct / 100
                    height = h_pct / 100

                    # Class ID (0 for person)
                    class_id = 0 if label == "person" else list(class_names).index(label)

                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        # Write classes.txt
        with open(output_dir / "classes.txt", "w") as f:
            for name in sorted(class_names):
                f.write(f"{name}\n")

        print(f"Exported to YOLO format: {output_dir}")
        print(f"  Labels: {len(list(labels_dir.glob('*.txt')))}")
        print(f"  Classes: {sorted(class_names)}")

        return str(output_dir)

    def get_labeling_instructions(self) -> str:
        """
        Get labeling instructions for annotators.

        Returns:
            Instruction text
        """
        return """
# Worker Detection Labeling Instructions

## Setup
1. Run: `./setup_label_studio.sh`
2. Open: http://localhost:8080
3. Create account (first time)
4. Create project with the provided labeling config

## Labeling Guidelines

### What to Label
- Label ALL visible persons in each image
- Include partially visible persons (at least 50% visible)
- Include persons at any distance

### How to Label
1. Click and drag to draw bounding box
2. Box should tightly fit the person (head to feet)
3. Include some padding (~5-10%)
4. Select "person" label

### Do NOT Label
- Mannequins or statues
- People in photos/posters
- Reflections
- Persons less than 50% visible

### Quality Tips
- Zoom in for distant/small persons
- Use keyboard shortcuts (press 1 for person)
- Review your annotations before submitting

## Keyboard Shortcuts
- 1: Select "person" label
- Ctrl+Z: Undo
- Space: Submit and next
"""

    def check_label_studio_status(self) -> Dict[str, Any]:
        """
        Check if Label Studio is running.

        Returns:
            Status dict
        """
        import urllib.request
        import urllib.error

        try:
            url = f"{self.config.url}/health"
            req = urllib.request.Request(url, method='GET')
            response = urllib.request.urlopen(req, timeout=5)
            return {
                "running": True,
                "url": self.config.url,
                "status": "healthy"
            }
        except urllib.error.URLError:
            return {
                "running": False,
                "url": self.config.url,
                "status": "not running"
            }
        except Exception as e:
            return {
                "running": False,
                "url": self.config.url,
                "status": str(e)
            }


def main():
    """CLI for Label Studio integration."""
    import argparse

    parser = argparse.ArgumentParser(description="Label Studio Integration")
    subparsers = parser.add_subparsers(dest="command")

    setup_parser = subparsers.add_parser("setup", help="Generate setup script")
    setup_parser.add_argument("--output", "-o", default="setup_label_studio.sh")

    prepare_parser = subparsers.add_parser("prepare", help="Prepare images")
    prepare_parser.add_argument("image_dir", help="Image directory")
    prepare_parser.add_argument("--output", "-o", default="data/label_studio/images")

    export_parser = subparsers.add_parser("export", help="Export to YOLO")
    export_parser.add_argument("export_json", help="Label Studio export JSON")
    export_parser.add_argument("--output", "-o", required=True)
    export_parser.add_argument("--images", "-i", required=True)

    instructions_parser = subparsers.add_parser("instructions", help="Show labeling instructions")

    status_parser = subparsers.add_parser("status", help="Check Label Studio status")

    args = parser.parse_args()
    integration = LabelStudioIntegration()

    if args.command == "setup":
        integration.generate_setup_script(args.output)

    elif args.command == "prepare":
        paths = integration.prepare_images_for_import(args.image_dir, args.output)
        integration.create_import_json(paths)

    elif args.command == "export":
        integration.export_to_yolo(args.export_json, args.output, args.images)

    elif args.command == "instructions":
        print(integration.get_labeling_instructions())

    elif args.command == "status":
        status = integration.check_label_studio_status()
        print(f"Label Studio: {status['status']}")
        print(f"URL: {status['url']}")


if __name__ == "__main__":
    main()
