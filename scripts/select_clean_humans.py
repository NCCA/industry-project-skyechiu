"""
Select 1000 clean-background images from Human Faces dataset.

1. Deduplicate (same base name, different extensions → keep largest file)
2. Score background cleanliness (border pixel variance)
3. Take top 1500 cleanest → randomly sample 1000
4. Copy to raw_faces with 'hf_' prefix
"""

import os
import sys
import random
import shutil
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict

# === Config ===
# Auto-detect path
_base = "/sessions/epic-zen-shannon/mnt/facelift_pipeline/data"
if not os.path.exists(_base):
    _base = r"D:\zmm\facelift_pipeline\data"

SRC_DIR = os.path.join(_base, "_ffhq_temp", "Humans")
DST_DIR = os.path.join(_base, "raw_faces")
NUM_SELECT = 1000
BORDER_WIDTH = 40
RANDOM_SEED = 42


def deduplicate(src_dir):
    """Keep only the largest file for each base name."""
    groups = defaultdict(list)
    for f in os.listdir(src_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext not in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
            continue
        base = os.path.splitext(f)[0]
        fpath = os.path.join(src_dir, f)
        size = os.path.getsize(fpath)
        groups[base].append((f, size))

    # For each group, keep the largest file
    unique = []
    for base, files in groups.items():
        files.sort(key=lambda x: -x[1])  # largest first
        unique.append(files[0][0])

    return sorted(unique)


def get_background_score(img_path):
    """Lower score = cleaner background."""
    try:
        img = Image.open(img_path).convert('RGB')
        arr = np.array(img)
        h, w = arr.shape[:2]
        bw = min(BORDER_WIDTH, h // 5, w // 5)
        if bw < 5:
            return 9999.0

        top = arr[:bw, :, :]
        bottom = arr[h-bw:, :, :]
        left = arr[bw:h-bw, :bw, :]
        right = arr[bw:h-bw, w-bw:, :]

        border_pixels = np.concatenate([
            top.reshape(-1, 3),
            bottom.reshape(-1, 3),
            left.reshape(-1, 3),
            right.reshape(-1, 3)
        ], axis=0)

        score = np.mean(np.std(border_pixels.astype(np.float32), axis=0))
        return score
    except Exception as e:
        print("Error: " + str(img_path) + " - " + str(e))
        return 9999.0


def main():
    src = Path(SRC_DIR)
    dst = Path(DST_DIR)
    dst.mkdir(parents=True, exist_ok=True)

    # Step 1: Deduplicate
    print("Step 1: Deduplicating...")
    unique_files = deduplicate(str(src))
    print("  Total files: " + str(len(os.listdir(src))))
    print("  After dedup: " + str(len(unique_files)))

    # Step 2: Score background cleanliness
    print("\nStep 2: Scoring background cleanliness...")
    scores = []
    for i, fname in enumerate(unique_files):
        score = get_background_score(str(src / fname))
        scores.append((fname, score))
        if (i + 1) % 500 == 0:
            print("  Scored " + str(i + 1) + "/" + str(len(unique_files)))

    # Sort by score (lower = cleaner)
    scores.sort(key=lambda x: x[1])

    # Print distribution
    all_scores = [s for _, s in scores if s < 9999]
    print("\nScore distribution:")
    print("  Min:    " + str(round(min(all_scores), 2)))
    print("  25th:   " + str(round(np.percentile(all_scores, 25), 2)))
    print("  Median: " + str(round(np.median(all_scores), 2)))
    print("  75th:   " + str(round(np.percentile(all_scores, 75), 2)))
    print("  Max:    " + str(round(max(all_scores), 2)))

    # Step 3: Select top 1500, then random sample 1000
    pool_size = min(len(scores), int(NUM_SELECT * 1.5))
    clean_pool = scores[:pool_size]
    print("\nStep 3: Clean pool (top " + str(pool_size) + "):")
    print("  Score range: " + str(round(clean_pool[0][1], 2)) + " ~ " + str(round(clean_pool[-1][1], 2)))

    random.seed(RANDOM_SEED)
    selected = random.sample(clean_pool, min(NUM_SELECT, len(clean_pool)))
    print("  Selected: " + str(len(selected)) + " images")

    # Step 4: Copy to raw_faces
    print("\nStep 4: Copying to " + str(dst))
    existing = set(os.listdir(dst))
    print("  Existing in raw_faces: " + str(len(existing)))

    copied = 0
    skipped = 0
    for fname, score in selected:
        new_name = "hf_" + fname
        if new_name in existing:
            skipped += 1
            continue
        shutil.copy2(str(src / fname), str(dst / new_name))
        copied += 1
        if copied % 100 == 0:
            print("  Copied " + str(copied) + "...")

    final_count = len(os.listdir(dst))
    print("\nDone!")
    print("  Copied: " + str(copied) + " new images")
    print("  Skipped: " + str(skipped))
    print("  Total in raw_faces: " + str(final_count))


if __name__ == "__main__":
    main()
