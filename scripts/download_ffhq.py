"""
Download 1000 FFHQ images from Kaggle for FaceLift pipeline.
Run this in your conda environment:
    cd D:\zmm\facelift_pipeline
    python scripts/download_ffhq.py
"""
import os
import sys
import shutil
import random
from pathlib import Path

# Config
NUM_IMAGES = 1000
RAW_DIR = Path(r"D:\zmm\facelift_pipeline\data\raw_faces")
KAGGLE_DATASET = "arnaud58/flickrfaceshq-dataset-ffhq"  # Full resolution 1024x1024
TEMP_DIR = Path(r"D:\zmm\facelift_pipeline\data\_ffhq_temp")

def main():
    print("=" * 50)
    print(f"Downloading {NUM_IMAGES} FFHQ images")
    print("=" * 50)

    # Check kaggle
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("Kaggle API: OK")
    except Exception as e:
        print(f"Kaggle API error: {e}")
        print("Make sure kaggle is installed and kaggle.json is configured.")
        print("  pip install kaggle")
        print("  Place kaggle.json in C:\\Users\\<you>\\.kaggle\\")
        sys.exit(1)

    # Backup existing raw_faces
    existing = list(RAW_DIR.glob("*"))
    if existing:
        backup = RAW_DIR.parent / "raw_faces_old_backup"
        backup.mkdir(exist_ok=True)
        print(f"Backing up {len(existing)} existing images to {backup}")
        for f in existing:
            if f.is_file():
                shutil.move(str(f), str(backup / f.name))

    # Download FFHQ from Kaggle
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading FFHQ dataset to {TEMP_DIR}...")
    print("This may take a while (dataset is large)...")

    try:
        api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(TEMP_DIR),
            unzip=True,
            quiet=False,
        )
        print("Download complete!")
    except Exception as e:
        print(f"Download failed: {e}")
        print("\nAlternative: try this dataset instead:")
        print("  kaggle datasets download -d greatgamedota/ffhq-face-data-set")
        sys.exit(1)

    # Find all images
    all_images = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        all_images.extend(TEMP_DIR.rglob(ext))

    print(f"\nFound {len(all_images)} total images in download")

    if len(all_images) == 0:
        print("No images found! Check the download.")
        sys.exit(1)

    # Randomly select NUM_IMAGES
    random.seed(42)
    if len(all_images) > NUM_IMAGES:
        selected = random.sample(all_images, NUM_IMAGES)
    else:
        selected = all_images
        print(f"Warning: only {len(selected)} images available (wanted {NUM_IMAGES})")

    # Copy to raw_faces
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nCopying {len(selected)} images to {RAW_DIR}...")

    for i, img_path in enumerate(selected):
        # Rename to ffhq_XXXXX.png
        ext = img_path.suffix
        dst = RAW_DIR / f"ffhq_{i:05d}{ext}"
        shutil.copy2(str(img_path), str(dst))
        if (i + 1) % 100 == 0:
            print(f"  Copied {i+1}/{len(selected)}")

    print(f"\nDone! {len(selected)} FFHQ images in {RAW_DIR}")
    print(f"\nNext steps:")
    print(f"  1. Check the images in {RAW_DIR}, delete any bad ones")
    print(f"  2. Run the notebook: Step 2 -> Step 3 (TEST_LIMIT=100) -> Step 4 -> Step 5")

    # Clean up temp (optional - ask user)
    print(f"\nTemp download folder: {TEMP_DIR}")
    print(f"You can delete it manually to save disk space.")


if __name__ == "__main__":
    main()
