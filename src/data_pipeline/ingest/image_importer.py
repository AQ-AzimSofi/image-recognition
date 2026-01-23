#!/usr/bin/env python3
"""
Image import and organization module.

Imports images from various sources, renames them consistently,
and organizes them into a structured directory layout.
"""

import shutil
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import re


@dataclass
class ImportedImage:
    """Information about an imported image."""
    original_path: str
    new_path: str
    original_name: str
    new_name: str
    file_hash: str
    file_size: int
    imported_at: str
    metadata: Dict


class ImageImporter:
    """
    Import and organize images into a structured dataset.

    Features:
    - Consistent naming convention
    - Duplicate detection (by hash)
    - Directory organization (by date/camera/site)
    - Metadata preservation
    """

    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}

    def __init__(
        self,
        dataset_dir: str = "data/datasets",
        naming_pattern: str = "{site}_{camera}_{date}_{time}_{index:06d}"
    ):
        """
        Initialize importer.

        Args:
            dataset_dir: Base directory for datasets
            naming_pattern: Pattern for renamed files
        """
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.naming_pattern = naming_pattern
        self.hash_index: Dict[str, str] = {}
        self._load_hash_index()

    def _load_hash_index(self):
        """Load existing file hash index."""
        index_path = self.dataset_dir / ".hash_index.json"
        if index_path.exists():
            with open(index_path, "r") as f:
                self.hash_index = json.load(f)

    def _save_hash_index(self):
        """Save file hash index."""
        index_path = self.dataset_dir / ".hash_index.json"
        with open(index_path, "w") as f:
            json.dump(self.hash_index, f, indent=2)

    def _compute_hash(self, file_path: Path) -> str:
        """Compute MD5 hash of file."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _parse_filename_info(self, filename: str) -> Dict:
        """
        Extract information from filename.

        Supports patterns:
        - TLC00046_20260121_083000.jpg
        - 2026-01-21_08-30-00.jpg
        - frame_0001.jpg
        - IMG_20260121_083000.jpg
        """
        stem = Path(filename).stem
        info = {
            "date": None,
            "time": None,
            "camera": None,
            "index": None
        }

        patterns = [
            (r'([A-Z]+\d+)_(\d{8})_(\d{6})', ['camera', 'date', 'time']),
            (r'(\d{4})-?(\d{2})-?(\d{2})[_T](\d{2})-?(\d{2})-?(\d{2})',
             ['year', 'month', 'day', 'hour', 'minute', 'second']),
            (r'frame[_-]?(\d+)', ['index']),
            (r'IMG_(\d{8})_(\d{6})', ['date', 'time']),
        ]

        for pattern, fields in patterns:
            match = re.search(pattern, stem, re.IGNORECASE)
            if match:
                groups = match.groups()
                if 'year' in fields:
                    info['date'] = f"{groups[0]}{groups[1]}{groups[2]}"
                    info['time'] = f"{groups[3]}{groups[4]}{groups[5]}"
                else:
                    for i, field in enumerate(fields):
                        if i < len(groups):
                            info[field] = groups[i]
                break

        return info

    def import_single(
        self,
        source_path: str,
        site_id: str = "site01",
        camera_id: str = "cam01",
        copy: bool = True,
        skip_duplicates: bool = True
    ) -> Optional[ImportedImage]:
        """
        Import a single image.

        Args:
            source_path: Path to source image
            site_id: Site identifier
            camera_id: Camera identifier
            copy: Copy file (True) or move (False)
            skip_duplicates: Skip if duplicate hash exists

        Returns:
            ImportedImage info or None if skipped
        """
        source = Path(source_path)

        if not source.exists():
            raise FileNotFoundError(f"Image not found: {source}")

        if source.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {source.suffix}")

        file_hash = self._compute_hash(source)

        if skip_duplicates and file_hash in self.hash_index:
            print(f"Skipping duplicate: {source.name}")
            return None

        filename_info = self._parse_filename_info(source.name)

        date_str = filename_info.get('date') or datetime.now().strftime("%Y%m%d")
        time_str = filename_info.get('time') or datetime.now().strftime("%H%M%S")

        if len(date_str) == 8:
            date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        else:
            date_formatted = date_str

        dest_dir = self.dataset_dir / site_id / camera_id / date_formatted
        dest_dir.mkdir(parents=True, exist_ok=True)

        existing_count = len(list(dest_dir.glob(f"*{source.suffix}")))

        new_name = self.naming_pattern.format(
            site=site_id,
            camera=camera_id,
            date=date_str,
            time=time_str,
            index=existing_count
        ) + source.suffix.lower()

        dest_path = dest_dir / new_name

        if copy:
            shutil.copy2(source, dest_path)
        else:
            shutil.move(source, dest_path)

        self.hash_index[file_hash] = str(dest_path)
        self._save_hash_index()

        return ImportedImage(
            original_path=str(source),
            new_path=str(dest_path),
            original_name=source.name,
            new_name=new_name,
            file_hash=file_hash,
            file_size=dest_path.stat().st_size,
            imported_at=datetime.now().isoformat(),
            metadata=filename_info
        )

    def import_directory(
        self,
        source_dir: str,
        site_id: str = "site01",
        camera_id: str = "cam01",
        recursive: bool = True,
        copy: bool = True,
        skip_duplicates: bool = True
    ) -> List[ImportedImage]:
        """
        Import all images from a directory.

        Args:
            source_dir: Source directory
            site_id: Site identifier
            camera_id: Camera identifier
            recursive: Search subdirectories
            copy: Copy files (True) or move (False)
            skip_duplicates: Skip duplicate files

        Returns:
            List of ImportedImage
        """
        source_dir = Path(source_dir)
        imported = []

        pattern = "**/*" if recursive else "*"
        image_files = sorted([
            f for f in source_dir.glob(pattern)
            if f.suffix.lower() in self.SUPPORTED_FORMATS
        ])

        print(f"Found {len(image_files)} images in {source_dir}")

        for img_path in image_files:
            try:
                result = self.import_single(
                    str(img_path),
                    site_id=site_id,
                    camera_id=camera_id,
                    copy=copy,
                    skip_duplicates=skip_duplicates
                )
                if result:
                    imported.append(result)
            except Exception as e:
                print(f"Error importing {img_path}: {e}")

        print(f"Imported {len(imported)} images")
        return imported

    def import_with_auto_grouping(
        self,
        source_dir: str,
        site_id: str = "site01",
        copy: bool = True
    ) -> Dict[str, List[ImportedImage]]:
        """
        Import images with automatic camera grouping based on filename.

        Args:
            source_dir: Source directory
            site_id: Site identifier
            copy: Copy files

        Returns:
            Dict mapping camera_id to list of ImportedImage
        """
        source_dir = Path(source_dir)
        results = {}

        image_files = sorted([
            f for f in source_dir.glob("**/*")
            if f.suffix.lower() in self.SUPPORTED_FORMATS
        ])

        for img_path in image_files:
            info = self._parse_filename_info(img_path.name)
            camera_id = info.get('camera') or 'cam01'

            result = self.import_single(
                str(img_path),
                site_id=site_id,
                camera_id=camera_id,
                copy=copy
            )

            if result:
                if camera_id not in results:
                    results[camera_id] = []
                results[camera_id].append(result)

        return results

    def get_dataset_stats(self, site_id: str = None) -> Dict:
        """
        Get statistics about the dataset.

        Args:
            site_id: Optional site filter

        Returns:
            Dict with dataset statistics
        """
        stats = {
            "total_images": 0,
            "total_size_bytes": 0,
            "sites": {},
            "cameras": set(),
            "dates": set()
        }

        search_path = self.dataset_dir / site_id if site_id else self.dataset_dir

        for img_path in search_path.glob("**/*"):
            if img_path.suffix.lower() in self.SUPPORTED_FORMATS:
                stats["total_images"] += 1
                stats["total_size_bytes"] += img_path.stat().st_size

                parts = img_path.relative_to(self.dataset_dir).parts
                if len(parts) >= 3:
                    site, camera, date = parts[:3]
                    if site not in stats["sites"]:
                        stats["sites"][site] = {"cameras": {}, "total": 0}
                    if camera not in stats["sites"][site]["cameras"]:
                        stats["sites"][site]["cameras"][camera] = 0
                    stats["sites"][site]["cameras"][camera] += 1
                    stats["sites"][site]["total"] += 1
                    stats["cameras"].add(camera)
                    stats["dates"].add(date)

        stats["cameras"] = sorted(stats["cameras"])
        stats["dates"] = sorted(stats["dates"])
        stats["total_size_mb"] = stats["total_size_bytes"] / (1024 * 1024)

        return stats

    def export_manifest(self, output_path: str = None) -> str:
        """
        Export dataset manifest to JSON.

        Args:
            output_path: Output file path

        Returns:
            Path to manifest file
        """
        if output_path is None:
            output_path = self.dataset_dir / "manifest.json"

        manifest = {
            "created_at": datetime.now().isoformat(),
            "dataset_dir": str(self.dataset_dir),
            "stats": self.get_dataset_stats(),
            "files": []
        }

        for img_path in self.dataset_dir.glob("**/*"):
            if img_path.suffix.lower() in self.SUPPORTED_FORMATS:
                rel_path = img_path.relative_to(self.dataset_dir)
                manifest["files"].append({
                    "path": str(rel_path),
                    "size": img_path.stat().st_size,
                    "modified": datetime.fromtimestamp(
                        img_path.stat().st_mtime
                    ).isoformat()
                })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return str(output_path)


def main():
    """CLI for image import."""
    import argparse

    parser = argparse.ArgumentParser(description="Import images to dataset")
    parser.add_argument("source", help="Source file or directory")
    parser.add_argument("--site", "-s", default="site01", help="Site ID")
    parser.add_argument("--camera", "-c", default="cam01", help="Camera ID")
    parser.add_argument("--output", "-o", default="data/datasets", help="Output directory")
    parser.add_argument("--move", action="store_true", help="Move instead of copy")
    parser.add_argument("--auto-group", action="store_true",
                        help="Auto-group by camera from filename")
    parser.add_argument("--stats", action="store_true", help="Show dataset stats")
    parser.add_argument("--manifest", action="store_true", help="Export manifest")

    args = parser.parse_args()

    importer = ImageImporter(dataset_dir=args.output)

    if args.stats:
        stats = importer.get_dataset_stats()
        print(f"Total images: {stats['total_images']}")
        print(f"Total size: {stats['total_size_mb']:.1f} MB")
        print(f"Sites: {list(stats['sites'].keys())}")
        print(f"Cameras: {stats['cameras']}")
        print(f"Dates: {stats['dates']}")
        return

    if args.manifest:
        path = importer.export_manifest()
        print(f"Manifest exported to: {path}")
        return

    source = Path(args.source)

    if source.is_dir():
        if args.auto_group:
            results = importer.import_with_auto_grouping(
                str(source),
                site_id=args.site,
                copy=not args.move
            )
            for camera, images in results.items():
                print(f"  {camera}: {len(images)} images")
        else:
            importer.import_directory(
                str(source),
                site_id=args.site,
                camera_id=args.camera,
                copy=not args.move
            )
    else:
        result = importer.import_single(
            str(source),
            site_id=args.site,
            camera_id=args.camera,
            copy=not args.move
        )
        if result:
            print(f"Imported: {result.new_path}")


if __name__ == "__main__":
    main()
