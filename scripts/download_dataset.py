import json
import os
import logging
from pathlib import Path
from icrawler.builtin import BingImageCrawler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

DATASET_DIR = Path(__file__).resolve().parent.parent / "datasets" / "everyday-items"

CATEGORIES = {
    "fire_extinguisher": "fire extinguisher in hallway at work reddit",
    "office_chair": "my office chair at work reddit",
    "ladder": "ladder on job site construction reddit",
    "hard_hat": "hard hat on construction site reddit",
    "safety_vest": "hi vis safety vest at work reddit",
    "toolbox": "my toolbox at work reddit",
    "whiteboard": "whiteboard in office meeting room reddit",
    "filing_cabinet": "filing cabinet in office reddit",
    "printer": "office printer at work reddit",
    "monitor": "my work desk monitor setup reddit",
    "clipboard": "clipboard on job site construction reddit",
    "traffic_cone": "traffic cone on road work site reddit",
    "power_drill": "power drill at work site reddit",
    "first_aid_kit": "first aid kit on wall at work reddit",
    "stack_of_boxes": "boxes stacked storage room warehouse reddit",
    "dolly_cart": "hand truck dolly cart warehouse reddit",
    "tape_measure": "tape measure at construction site reddit",
    "extension_cord": "extension cord on floor at work reddit",
    "water_cooler": "water cooler break room office reddit",
    "desk_phone": "office desk phone at work reddit",
}

IMAGES_PER_CATEGORY = 3
TOTAL_TARGET = 50


def download_category(category_name: str, query: str, max_num: int) -> int:
    out_dir = DATASET_DIR / category_name
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(out_dir.glob("*.jpg"))) + len(list(out_dir.glob("*.png")))
    if existing >= max_num:
        log.info(f"Skipping {category_name} - already has {existing} images")
        return existing

    crawler = BingImageCrawler(
        storage={"root_dir": str(out_dir)},
        log_level=logging.WARNING,
    )
    crawler.crawl(
        keyword=query,
        max_num=max_num,
        min_size=(200, 200),
        file_idx_offset="auto",
    )

    downloaded = len(list(out_dir.glob("*.jpg"))) + len(list(out_dir.glob("*.png")))
    log.info(f"{category_name}: {downloaded} images")
    return downloaded


def build_manifest():
    manifest = []
    for category_dir in sorted(DATASET_DIR.iterdir()):
        if not category_dir.is_dir():
            continue
        for img_path in sorted(category_dir.iterdir()):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
                manifest.append({
                    "file": f"{category_dir.name}/{img_path.name}",
                    "category": category_dir.name,
                })

    manifest_path = DATASET_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"Manifest written: {len(manifest)} images across {len(set(m['category'] for m in manifest))} categories")
    return manifest


def main():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading ~{TOTAL_TARGET} everyday item images to {DATASET_DIR}")

    total = 0
    for category_name, query in CATEGORIES.items():
        count = download_category(category_name, query, IMAGES_PER_CATEGORY)
        total += count
        if total >= TOTAL_TARGET:
            log.info(f"Reached target of {TOTAL_TARGET} images (got {total})")
            break

    manifest = build_manifest()
    log.info(f"Done. Total: {len(manifest)} images")


if __name__ == "__main__":
    main()
