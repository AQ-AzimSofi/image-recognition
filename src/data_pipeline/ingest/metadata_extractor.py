#!/usr/bin/env python3
"""
Metadata extraction module.

Extracts metadata from images including EXIF, filename patterns,
and generates structured metadata for the dataset.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json

try:
    from PIL import Image
    from PIL.ExifTags import TAGS, GPSTAGS
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@dataclass
class ImageMetadata:
    """Extracted image metadata."""
    file_path: str
    file_name: str
    file_size: int
    width: int
    height: int
    format: str
    timestamp: Optional[str] = None
    camera_id: Optional[str] = None
    site_id: Optional[str] = None
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    exif: Optional[Dict] = None
    custom: Optional[Dict] = None


class MetadataExtractor:
    """
    Extract and manage image metadata.

    Features:
    - EXIF data extraction
    - Filename pattern parsing
    - GPS coordinate extraction
    - Timestamp normalization
    - Batch processing
    """

    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.webp'}

    FILENAME_PATTERNS = [
        (r'([A-Z]+\d+)_(\d{8})_(\d{6})', {
            'camera_id': 1, 'date': 2, 'time': 3
        }),
        (r'(\d{4})-?(\d{2})-?(\d{2})[_T](\d{2})-?(\d{2})-?(\d{2})', {
            'year': 1, 'month': 2, 'day': 3, 'hour': 4, 'minute': 5, 'second': 6
        }),
        (r'frame[_-]?(\d+)', {'frame_number': 1}),
        (r'site(\d+)[_-]cam(\d+)', {'site_num': 1, 'cam_num': 2}),
    ]

    def __init__(self, cache_dir: str = None):
        """
        Initialize extractor.

        Args:
            cache_dir: Directory to cache extracted metadata
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, ImageMetadata] = {}

    def extract(self, image_path: str, use_cache: bool = True) -> ImageMetadata:
        """
        Extract metadata from a single image.

        Args:
            image_path: Path to image file
            use_cache: Use cached metadata if available

        Returns:
            ImageMetadata object
        """
        image_path = Path(image_path)

        if use_cache and str(image_path) in self._cache:
            return self._cache[str(image_path)]

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        stat = image_path.stat()

        width, height, img_format, exif_data = self._get_image_info(image_path)

        filename_info = self._parse_filename(image_path.name)

        timestamp = self._extract_timestamp(exif_data, filename_info)

        gps_lat, gps_lon = self._extract_gps(exif_data)

        metadata = ImageMetadata(
            file_path=str(image_path.absolute()),
            file_name=image_path.name,
            file_size=stat.st_size,
            width=width,
            height=height,
            format=img_format,
            timestamp=timestamp,
            camera_id=filename_info.get('camera_id'),
            site_id=filename_info.get('site_id'),
            gps_lat=gps_lat,
            gps_lon=gps_lon,
            exif=exif_data,
            custom=filename_info
        )

        self._cache[str(image_path)] = metadata
        return metadata

    def _get_image_info(self, image_path: Path) -> tuple:
        """Get basic image info and EXIF data."""
        if not HAS_PIL:
            import cv2
            img = cv2.imread(str(image_path))
            if img is None:
                return 0, 0, "unknown", {}
            h, w = img.shape[:2]
            return w, h, image_path.suffix.lower()[1:], {}

        try:
            with Image.open(image_path) as img:
                width, height = img.size
                img_format = img.format or "unknown"

                exif_data = {}
                if hasattr(img, '_getexif') and img._getexif():
                    raw_exif = img._getexif()
                    for tag_id, value in raw_exif.items():
                        tag = TAGS.get(tag_id, tag_id)
                        if isinstance(value, bytes):
                            continue
                        try:
                            json.dumps(value)
                            exif_data[tag] = value
                        except (TypeError, ValueError):
                            exif_data[tag] = str(value)

                return width, height, img_format, exif_data
        except Exception as e:
            print(f"Error reading {image_path}: {e}")
            return 0, 0, "unknown", {}

    def _parse_filename(self, filename: str) -> Dict[str, Any]:
        """Parse filename for embedded information."""
        info = {}
        stem = Path(filename).stem

        for pattern, groups in self.FILENAME_PATTERNS:
            match = re.search(pattern, stem, re.IGNORECASE)
            if match:
                for name, idx in groups.items():
                    if idx <= len(match.groups()):
                        info[name] = match.group(idx)

        if 'date' in info and 'time' in info:
            date_str = info['date']
            time_str = info['time']
            try:
                dt = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
                info['parsed_datetime'] = dt.isoformat()
            except ValueError:
                pass

        if 'year' in info:
            try:
                dt = datetime(
                    int(info['year']), int(info['month']), int(info['day']),
                    int(info['hour']), int(info['minute']), int(info['second'])
                )
                info['parsed_datetime'] = dt.isoformat()
            except (ValueError, KeyError):
                pass

        if 'site_num' in info:
            info['site_id'] = f"site{info['site_num']:0>2}"
        if 'cam_num' in info:
            info['camera_id'] = f"cam{info['cam_num']:0>2}"

        return info

    def _extract_timestamp(
        self,
        exif_data: Dict,
        filename_info: Dict
    ) -> Optional[str]:
        """Extract timestamp from EXIF or filename."""
        exif_date_tags = [
            'DateTimeOriginal',
            'DateTime',
            'DateTimeDigitized'
        ]

        for tag in exif_date_tags:
            if tag in exif_data:
                try:
                    dt_str = exif_data[tag]
                    dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
                    return dt.isoformat()
                except (ValueError, TypeError):
                    continue

        if 'parsed_datetime' in filename_info:
            return filename_info['parsed_datetime']

        return None

    def _extract_gps(self, exif_data: Dict) -> tuple:
        """Extract GPS coordinates from EXIF."""
        if 'GPSInfo' not in exif_data:
            return None, None

        gps_info = exif_data['GPSInfo']

        def convert_to_degrees(value):
            if isinstance(value, tuple) and len(value) == 3:
                d, m, s = value
                if isinstance(d, tuple):
                    d = d[0] / d[1]
                if isinstance(m, tuple):
                    m = m[0] / m[1]
                if isinstance(s, tuple):
                    s = s[0] / s[1]
                return d + m / 60 + s / 3600
            return None

        lat = None
        lon = None

        if 2 in gps_info:
            lat = convert_to_degrees(gps_info[2])
            if lat and 1 in gps_info and gps_info[1] == 'S':
                lat = -lat

        if 4 in gps_info:
            lon = convert_to_degrees(gps_info[4])
            if lon and 3 in gps_info and gps_info[3] == 'W':
                lon = -lon

        return lat, lon

    def extract_batch(
        self,
        image_paths: List[str],
        parallel: bool = False
    ) -> List[ImageMetadata]:
        """
        Extract metadata from multiple images.

        Args:
            image_paths: List of image paths
            parallel: Use parallel processing

        Returns:
            List of ImageMetadata
        """
        results = []

        for path in image_paths:
            try:
                metadata = self.extract(path)
                results.append(metadata)
            except Exception as e:
                print(f"Error processing {path}: {e}")

        return results

    def extract_directory(
        self,
        directory: str,
        recursive: bool = True,
        output_json: str = None
    ) -> List[ImageMetadata]:
        """
        Extract metadata from all images in a directory.

        Args:
            directory: Directory path
            recursive: Search subdirectories
            output_json: Optional output JSON file

        Returns:
            List of ImageMetadata
        """
        directory = Path(directory)
        pattern = "**/*" if recursive else "*"

        image_paths = [
            str(p) for p in directory.glob(pattern)
            if p.suffix.lower() in self.SUPPORTED_FORMATS
        ]

        print(f"Processing {len(image_paths)} images...")
        results = self.extract_batch(image_paths)

        if output_json:
            self.save_to_json(results, output_json)

        return results

    def save_to_json(self, metadata_list: List[ImageMetadata], output_path: str):
        """Save metadata list to JSON file."""
        data = {
            "extracted_at": datetime.now().isoformat(),
            "count": len(metadata_list),
            "images": [asdict(m) for m in metadata_list]
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        print(f"Saved metadata to: {output_path}")

    def get_summary(self, metadata_list: List[ImageMetadata]) -> Dict:
        """
        Get summary statistics from metadata.

        Args:
            metadata_list: List of ImageMetadata

        Returns:
            Summary dictionary
        """
        if not metadata_list:
            return {}

        cameras = set()
        sites = set()
        timestamps = []
        total_size = 0
        resolutions = set()

        for m in metadata_list:
            if m.camera_id:
                cameras.add(m.camera_id)
            if m.site_id:
                sites.add(m.site_id)
            if m.timestamp:
                timestamps.append(m.timestamp)
            total_size += m.file_size
            resolutions.add(f"{m.width}x{m.height}")

        return {
            "total_images": len(metadata_list),
            "total_size_mb": total_size / (1024 * 1024),
            "cameras": sorted(cameras),
            "sites": sorted(sites),
            "resolutions": sorted(resolutions),
            "date_range": {
                "earliest": min(timestamps) if timestamps else None,
                "latest": max(timestamps) if timestamps else None
            },
            "images_with_timestamp": len(timestamps),
            "images_with_gps": sum(1 for m in metadata_list if m.gps_lat)
        }


def main():
    """CLI for metadata extraction."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract image metadata")
    parser.add_argument("input", help="Image file or directory")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="Search subdirectories")
    parser.add_argument("--summary", "-s", action="store_true",
                        help="Show summary only")

    args = parser.parse_args()

    extractor = MetadataExtractor()
    input_path = Path(args.input)

    if input_path.is_dir():
        results = extractor.extract_directory(
            str(input_path),
            recursive=args.recursive,
            output_json=args.output
        )
    else:
        results = [extractor.extract(str(input_path))]
        if args.output:
            extractor.save_to_json(results, args.output)

    if args.summary or not args.output:
        summary = extractor.get_summary(results)
        print(f"\nSummary:")
        print(f"  Total images: {summary.get('total_images', 0)}")
        print(f"  Total size: {summary.get('total_size_mb', 0):.1f} MB")
        print(f"  Cameras: {summary.get('cameras', [])}")
        print(f"  Sites: {summary.get('sites', [])}")
        print(f"  Resolutions: {summary.get('resolutions', [])}")
        if summary.get('date_range', {}).get('earliest'):
            print(f"  Date range: {summary['date_range']['earliest']} to {summary['date_range']['latest']}")


if __name__ == "__main__":
    main()
