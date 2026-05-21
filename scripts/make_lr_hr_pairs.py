"""
Generate Low-Resolution / High-Resolution depth map pairs for the Up-Res task.

For every depth map in data/dataset/{train,val}/depth/ (1024x1024, 16-bit):
  - Create a 256x256 16-bit version (pure spatial down-sample, area interp)
  - Create a 256x256 8-bit version (down-sample + bit-depth quantization,
    simulating realistic low-quality depth sensor output)

Output layout:
    data/dataset/
        train/
            depth_lr_16bit/    256x256, uint16
            depth_lr_8bit/     256x256, uint8
        val/
            depth_lr_16bit/, depth_lr_8bit/

Usage:
    python scripts/make_lr_hr_pairs.py
    python scripts/make_lr_hr_pairs.py --target_size 256
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def downsample_pair(hr_path: Path, out_16: Path, out_8: Path, target_size: int):
    hr = cv2.imread(str(hr_path), cv2.IMREAD_UNCHANGED)
    if hr is None:
        print(f"Skip (read failed): {hr_path}")
        return False
    if hr.dtype != np.uint16:
        # Promote uint8 to uint16 for consistent processing
        hr = (hr.astype(np.float32) / 255.0 * 65535.0).astype(np.uint16)

    # 1) 16-bit LR: keep precision, just spatial downsample
    lr16 = cv2.resize(hr, (target_size, target_size), interpolation=cv2.INTER_AREA)
    lr16 = lr16.astype(np.uint16)
    cv2.imwrite(str(out_16), lr16)

    # 2) 8-bit LR: simulate sensor with crude quantization
    lr_f = cv2.resize(hr.astype(np.float32) / 65535.0, (target_size, target_size),
                      interpolation=cv2.INTER_AREA)
    lr8 = np.clip(lr_f * 255.0 + 0.5, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out_8), lr8)
    return True


def process_split(split_dir: Path, target_size: int):
    hr_dir = split_dir / "depth"
    if not hr_dir.exists():
        print(f"  no depth dir at {hr_dir}, skip split")
        return 0

    out16 = split_dir / "depth_lr_16bit"
    out8 = split_dir / "depth_lr_8bit"
    out16.mkdir(exist_ok=True)
    out8.mkdir(exist_ok=True)

    n_ok = 0
    files = sorted(hr_dir.glob("*.png"))
    for i, f in enumerate(files):
        if downsample_pair(f, out16 / f.name, out8 / f.name, target_size):
            n_ok += 1
        if (i + 1) % 50 == 0 or (i + 1) == len(files):
            print(f"  [{i+1}/{len(files)}]")
    return n_ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/dataset", type=Path)
    p.add_argument("--target_size", default=256, type=int,
                   help="LR resolution (default 256)")
    args = p.parse_args()

    if not args.root.exists():
        raise SystemExit(f"Dataset root not found: {args.root}\n"
                         f"Run scripts/reorganize_dataset.py first.")

    # Sanity check: HR depth must be 1024x1024 (the up-res task target)
    sample_dir = args.root / "train" / "depth"
    sample_files = sorted(sample_dir.glob("*.png"))
    if not sample_files:
        raise SystemExit(f"No HR depth files found in {sample_dir}")
    sample_hr = cv2.imread(str(sample_files[0]), cv2.IMREAD_UNCHANGED)
    actual_h, actual_w = sample_hr.shape[:2]
    print(f"Detected HR depth size: {actual_w}x{actual_h}, dtype={sample_hr.dtype}")
    if (actual_h, actual_w) != (1024, 1024):
        raise SystemExit(
            f"\nERROR: HR depth is {actual_w}x{actual_h}, expected 1024x1024.\n"
            f"This means data/postprocessed/ was not regenerated at 1024.\n"
            f"FIX:\n"
            f"  1. Open FaceLift/facelift_pipeline.ipynb\n"
            f"  2. Restart kernel\n"
            f"  3. Re-run Step 4.5 (cell 19) to regenerate postprocessed/ at 1024\n"
            f"  4. Re-run scripts/reorganize_dataset.py\n"
            f"  5. Re-run this script\n"
        )

    print(f"Generating LR pairs at {args.target_size}x{args.target_size}")
    print(f"  HR: 1024x1024 16-bit | LR: {args.target_size}x{args.target_size} (16-bit + 8-bit)")

    for split_name in ("train", "val"):
        split_dir = args.root / split_name
        if split_dir.exists():
            print(f"\n[{split_name}]")
            n = process_split(split_dir, args.target_size)
            print(f"  done: {n} samples")

    print("\nDone. New dirs created in each split:")
    print("  depth_lr_16bit/  - 256x256 uint16 (clean LR)")
    print("  depth_lr_8bit/   - 256x256 uint8  (quantized LR)")


if __name__ == "__main__":
    main()
