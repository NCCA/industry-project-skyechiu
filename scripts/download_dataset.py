#!/usr/bin/env python3
"""
Step 1: Download raw face images from Kaggle (FFHQ subset).

Matches facelift_pipeline.ipynb Cell 3.

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --max-images 50
    python scripts/download_dataset.py --manual
"""

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "pipeline_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Download face dataset from Kaggle")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Kaggle dataset slug (default: from config)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: from config)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Max images to keep (for quick testing)")
    parser.add_argument("--manual", action="store_true",
                        help="Print manual download instructions only")
    args = parser.parse_args()

    config = load_config()
    dataset_slug = args.dataset or config["kaggle"]["dataset_slug"]
    raw_dir = Path(args.output_dir or config["paths"]["dataset_raw"]).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    max_images = args.max_images or config["kaggle"].get("max_images")

    if args.manual:
        print("=" * 60)
        print("Manual download:")
        print(f"  1. Visit https://www.kaggle.com/datasets/{dataset_slug}")
        print(f"  2. Download and extract images into: {raw_dir}")
        print("=" * 60)
        return

    # --- Kaggle API download ---
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("ERROR: kaggle package not installed.")
        print("  pip install kaggle")
        print("  Also place kaggle.json in ~/.kaggle/")
        print(f"\nOr download manually from: https://www.kaggle.com/datasets/{dataset_slug}")
        sys.exit(1)

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading: {dataset_slug}")
    print(f"Target dir:  {raw_dir}")
    api.dataset_download_files(dataset_slug, path=str(raw_dir), unzip=True)
    print("Download complete!")

    # --- Enumerate images ---
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    raw_images = sorted([f for f in raw_dir.rglob("*") if f.suffix.lower() in image_exts])
    print(f"Raw images found: {len(raw_images)}")

    # Limit if requested
    if max_images and len(raw_images) > max_images:
        print(f"Keeping first {max_images} images, removing {len(raw_images) - max_images}")
        for img in raw_images[max_images:]:
            img.unlink()
        raw_images = raw_images[:max_images]

    for img in raw_images[:5]:
        print(f"  {img.name}")
    if len(raw_images) > 5:
        print(f"  ... and {len(raw_images) - 5} more")

    print(f"\nNext step: python scripts/prepare_images.py")


if __name__ == "__main__":
    main()
