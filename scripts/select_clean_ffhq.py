"""
Select N FFHQ images with clean backgrounds.

Strategy:
- Sample border pixels (top/bottom/left/right strips, ~20% of image edges)
- Compute color variance in those border regions
- Low variance = clean/uniform background
- Sort by cleanliness, take top candidates, then randomly sample N

CLI usage:
    python scripts/select_clean_ffhq.py --num-select 1400
    python scripts/select_clean_ffhq.py --num-select 3000 --random-seed 123
    python scripts/select_clean_ffhq.py                        # defaults: 1000, seed=42

The script is resume-safe: files with names already in raw_faces/ are skipped,
so you can run it multiple times with different --num-select to incrementally
grow the dataset without overwriting existing work.
"""

import argparse
import os
import sys
import random
import shutil
import numpy as np
from PIL import Image
from pathlib import Path

# === Config (defaults - can be overridden via CLI) ===
# Auto-detect path (Linux mount vs Windows)
_base = "/sessions/magical-focused-noether/mnt/facelift_pipeline/data"
if not os.path.exists(_base):
    _base = "/sessions/epic-zen-shannon/mnt/facelift_pipeline/data"
if not os.path.exists(_base):
    _base = r"D:\zmm\facelift_pipeline\data"
SRC_DIR = os.path.join(_base, "_ffhq_temp", "archive")
DST_DIR = os.path.join(_base, "raw_faces")
DEFAULT_NUM_SELECT = 1000
DEFAULT_RANDOM_SEED = 42
BORDER_WIDTH = 40  # pixels from edge to sample
# Max variance threshold - lower = cleaner background
# We'll use adaptive selection: sort by score, take top N

# Will be filled in by main() from CLI args
NUM_SELECT = DEFAULT_NUM_SELECT
RANDOM_SEED = DEFAULT_RANDOM_SEED

def get_background_score(img_path):
    """
    Lower score = cleaner background.
    Samples border pixels and computes color standard deviation.
    """
    try:
        img = Image.open(img_path).convert('RGB')
        arr = np.array(img)
        h, w = arr.shape[:2]
        bw = BORDER_WIDTH

        # Collect border strips
        top = arr[:bw, :, :]           # top strip
        bottom = arr[h-bw:, :, :]      # bottom strip
        left = arr[bw:h-bw, :bw, :]    # left strip (excluding corners)
        right = arr[bw:h-bw, w-bw:, :] # right strip (excluding corners)

        # Combine all border pixels
        border_pixels = np.concatenate([
            top.reshape(-1, 3),
            bottom.reshape(-1, 3),
            left.reshape(-1, 3),
            right.reshape(-1, 3)
        ], axis=0)

        # Score = mean of per-channel std deviation
        score = np.mean(np.std(border_pixels.astype(np.float32), axis=0))
        return score
    except Exception as e:
        print("Error processing " + str(img_path) + ": " + str(e))
        return 9999.0  # bad score


def main():
    global NUM_SELECT, RANDOM_SEED

    parser = argparse.ArgumentParser(description="Select N clean-background FFHQ images")
    parser.add_argument("--num-select", type=int, default=DEFAULT_NUM_SELECT,
                        help="Number of images to select (default: 1000)")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED,
                        help="Random seed for sampling (default: 42)")
    parser.add_argument("--src-dir", type=str, default=SRC_DIR,
                        help="Source FFHQ archive directory")
    parser.add_argument("--dst-dir", type=str, default=DST_DIR,
                        help="Destination raw_faces directory")
    args = parser.parse_args()

    NUM_SELECT = args.num_select
    RANDOM_SEED = args.random_seed

    print("=" * 60)
    print("  FFHQ Clean-Background Selector")
    print("=" * 60)
    print("  NUM_SELECT  : " + str(NUM_SELECT))
    print("  RANDOM_SEED : " + str(RANDOM_SEED))
    print("  SRC         : " + str(args.src_dir))
    print("  DST         : " + str(args.dst_dir))
    print("=" * 60)

    src = Path(args.src_dir)
    dst = Path(args.dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    # Get all PNG files
    all_files = sorted([f for f in os.listdir(src) if f.lower().endswith('.png')])
    total = len(all_files)
    print("Total FFHQ images: " + str(total))

    if total == 0:
        print("ERROR: No images found in " + str(src))
        sys.exit(1)

    # Score all images
    print("Scoring background cleanliness...")
    scores = []
    for i, fname in enumerate(all_files):
        score = get_background_score(src / fname)
        scores.append((fname, score))
        if (i + 1) % 200 == 0:
            print("  Scored " + str(i + 1) + "/" + str(total) + " images...")

    # Sort by score (lower = cleaner)
    scores.sort(key=lambda x: x[1])

    # Print score distribution
    all_scores = [s for _, s in scores]
    print("\nScore distribution:")
    print("  Min:    " + str(round(min(all_scores), 2)))
    print("  25th:   " + str(round(np.percentile(all_scores, 25), 2)))
    print("  Median: " + str(round(np.median(all_scores), 2)))
    print("  75th:   " + str(round(np.percentile(all_scores, 75), 2)))
    print("  Max:    " + str(round(max(all_scores), 2)))

    # Take top 1500 cleanest (buffer for random selection)
    pool_size = min(len(scores), int(NUM_SELECT * 1.5))
    clean_pool = scores[:pool_size]
    print("\nClean pool size (top " + str(pool_size) + "):")
    print("  Score range: " + str(round(clean_pool[0][1], 2)) + " ~ " + str(round(clean_pool[-1][1], 2)))

    # Randomly select 1000 from the clean pool
    random.seed(RANDOM_SEED)
    selected = random.sample(clean_pool, min(NUM_SELECT, len(clean_pool)))
    print("\nSelected " + str(len(selected)) + " images")

    # Check existing files
    existing = set(os.listdir(dst))
    print("Existing images in raw_faces: " + str(len(existing)))

    # Copy with ffhq_ prefix
    copied = 0
    skipped = 0
    for fname, score in selected:
        new_name = "ffhq_" + fname
        if new_name in existing:
            skipped += 1
            continue
        shutil.copy2(str(src / fname), str(dst / new_name))
        copied += 1
        if copied % 100 == 0:
            print("  Copied " + str(copied) + " files...")

    final_count = len(os.listdir(dst))
    print("\nDone!")
    print("  Copied: " + str(copied) + " new FFHQ images")
    print("  Skipped (already exist): " + str(skipped))
    print("  Total images in raw_faces: " + str(final_count))


if __name__ == "__main__":
    main()
