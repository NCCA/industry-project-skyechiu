"""
Generate foreground face masks for every sample in data/dataset/{train,val}/depth/
by running a morphology-based mask builder on the matching data/cropped_faces/<name>.png.

Outputs:
    data/dataset/{train,val}/mask/<name>.png       1024×1024 uint8 {0,255}
    data/dataset/{train,val}/mask_lr/<name>.png     256×256   uint8 {0,255}

Replaces manually running render_improve.ipynb section 3.1. Self-contained: no
globals from notebooks required. Kept in sync with the canonical
_build_face_mask_from_cropped in render_improve.ipynb section 2.3 — if you edit
one, edit the other.

Pure CPU morphology (cv2). About 30-60s for 1288 samples.

Usage:
    python scripts/make_face_masks.py
    python scripts/make_face_masks.py --data_root data --white_tol 12 --erode_px 0

Skip logic: if every stem in depth/ already has a matching mask/ + mask_lr/
file (non-empty), the script exits without re-running.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# Loose profile (matches render_improve.ipynb section 3.1 defaults):
#   WHITE_TOL=12  keeps matting boundary + hair (dataset mask should include
#                 the whole person region, not just the tight face core)
#   ERODE_PX=0    so the mask edge isn't pulled in
DEFAULT_WHITE_TOL = 12
DEFAULT_K_CLOSE   = 7
DEFAULT_K_OPEN    = 3
DEFAULT_ERODE_PX  = 0
HR_SIZE = 1024
LR_SIZE = 256


def build_face_mask_from_cropped(rgb, res, white_tol, k_close, k_open, erode_px):
    """Build a binary (res,res) face mask from a source RGB image.

    Pipeline:
      1. White-distance threshold → raw foreground
      2. Morphology close + open  → clean noise
      3. Keep largest connected component
      4. Flood-fill interior holes (eye sclera, teeth, specular highlights)
      5. Optional erosion
      6. Resize to `res` with NEAREST + re-binarize
    """
    dist = (255 - rgb.astype(np.int16)).max(axis=-1)
    m = (dist > white_tol).astype(np.uint8) * 255

    if k_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if k_open > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (m > 0).astype(np.uint8), 8,
    )
    if n_labels > 1:
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = np.where(labels == best, 255, 0).astype(np.uint8)

    # Fill interior holes
    h, w = m.shape
    flood = m.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    m[flood == 0] = 255

    if erode_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1),
        )
        m = cv2.erode(m, k, iterations=1)

    if m.shape[0] != res or m.shape[1] != res:
        m = cv2.resize(m, (res, res), interpolation=cv2.INTER_NEAREST)

    return m > 127


def load_cropped_rgb(cropped_dir, stem):
    """Load cropped_faces/<stem>.{png,jpg,jpeg,webp,bmp} if present."""
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        p = cropped_dir / f"{stem}{ext}"
        if p.exists():
            return np.array(Image.open(p).convert("RGB"))
    return None


def process_split(data_root, split, cropped_dir, args):
    hr_dir      = data_root / "dataset" / split / "depth"
    mask_dir    = data_root / "dataset" / split / "mask"
    mask_lr_dir = data_root / "dataset" / split / "mask_lr"
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_lr_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted(p.stem for p in hr_dir.glob("*.png"))
    if not stems:
        print(f"[{split}] no depth samples at {hr_dir} — skipping")
        return 0, []

    # Skip-if-all-already-done guard
    existing_hr = {p.stem for p in mask_dir.glob("*.png") if p.stat().st_size > 0}
    existing_lr = {p.stem for p in mask_lr_dir.glob("*.png") if p.stat().st_size > 0}
    missing = [s for s in stems if s not in existing_hr or s not in existing_lr]
    if not missing:
        print(f"[{split}] SKIP — {len(stems)}/{len(stems)} masks already present.")
        return len(stems), []

    if existing_hr or existing_lr:
        print(f"[{split}] RESUME — {len(stems) - len(missing)} already done, "
              f"processing {len(missing)} missing")
        stems = missing
    else:
        print(f"[{split}] {len(stems)} samples to process")

    n_done = 0
    n_missing_cropped = 0
    fg_ratios = []
    t0 = time.time()

    for i, stem in enumerate(stems):
        rgb = load_cropped_rgb(cropped_dir, stem)
        if rgb is None:
            n_missing_cropped += 1
            continue

        try:
            m_hr = build_face_mask_from_cropped(
                rgb, HR_SIZE,
                white_tol=args.white_tol, k_close=args.k_close,
                k_open=args.k_open, erode_px=args.erode_px,
            )
            m_lr = build_face_mask_from_cropped(
                rgb, LR_SIZE,
                white_tol=args.white_tol, k_close=args.k_close,
                k_open=args.k_open, erode_px=args.erode_px,
            )
        except Exception as e:
            print(f"  FAIL [{stem}]: {e}")
            continue

        Image.fromarray((m_hr.astype(np.uint8) * 255), mode="L").save(mask_dir / f"{stem}.png")
        Image.fromarray((m_lr.astype(np.uint8) * 255), mode="L").save(mask_lr_dir / f"{stem}.png")
        fg_ratios.append(float(m_hr.mean()))
        n_done += 1

        if (i + 1) % 100 == 0 or (i + 1) == len(stems):
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (len(stems) - (i + 1)) / max(rate, 1e-6)
            print(f"  [{split} {i+1}/{len(stems)}] "
                  f"rate={rate:.1f}/s  ETA={eta:.0f}s")

    if fg_ratios:
        a = np.array(fg_ratios)
        print(f"[{split}] wrote {n_done} masks  "
              f"fg_ratio mean={a.mean():.3f} min={a.min():.3f} max={a.max():.3f}")
    if n_missing_cropped:
        print(f"[{split}] WARNING: {n_missing_cropped} samples with no "
              f"cropped_faces source — their mask_dir will fall back to all-ones")

    return n_done, fg_ratios


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   default="data", type=Path,
                   help="Project data root (default: ./data)")
    p.add_argument("--cropped_dir", default=None, type=Path,
                   help="Override cropped_faces location (default: <data_root>/cropped_faces)")
    p.add_argument("--white_tol",   default=DEFAULT_WHITE_TOL, type=int,
                   help="Distance-from-white threshold. Larger = more inclusive.")
    p.add_argument("--k_close",     default=DEFAULT_K_CLOSE, type=int)
    p.add_argument("--k_open",      default=DEFAULT_K_OPEN,  type=int)
    p.add_argument("--erode_px",    default=DEFAULT_ERODE_PX, type=int,
                   help="Erode radius in pixels (0 = no erode, matches section 3.1 default)")
    args = p.parse_args()

    data_root = args.data_root.resolve()
    cropped_dir = (args.cropped_dir or (data_root / "cropped_faces")).resolve()

    if not cropped_dir.exists():
        raise SystemExit(f"Missing cropped_faces dir: {cropped_dir}")

    print(f"data_root   = {data_root}")
    print(f"cropped_dir = {cropped_dir}")
    print(f"params      = white_tol={args.white_tol} k_close={args.k_close} "
          f"k_open={args.k_open} erode_px={args.erode_px}")
    print(f"sizes       = HR {HR_SIZE}, LR {LR_SIZE}")

    total = 0
    for split in ("train", "val"):
        n, _ = process_split(data_root, split, cropped_dir, args)
        total += n
    print(f"\nDone. Total masks written this run: {total}")


if __name__ == "__main__":
    main()
