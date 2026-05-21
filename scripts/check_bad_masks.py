"""Scan for low-quality samples using multiple criteria."""
from pathlib import Path
from PIL import Image
import numpy as np

MASK_RATIO_LO = 0.15    # mask too small
MASK_RATIO_HI = 0.85    # mask too large (extreme close-up)
CENTER_THRESH = 0.25    # centroid too far from center
BBOX_MIN = 0.20         # bbox width or height < 20% of image (tiny face)
BBOX_ASPECT_MAX = 3.0   # bbox aspect ratio too extreme (partial face crop)

for split in ["train", "val"]:
    d = Path(f"data/dataset/{split}/mask")
    bad = []
    total = 0
    for p in sorted(d.glob("*.png")):
        total += 1
        m = np.array(Image.open(p))
        binary = m > 127
        ratio = binary.mean()
        reason = []

        if not binary.any():
            reason.append("empty_mask")
            bad.append((p.stem, ratio, ", ".join(reason)))
            continue

        # centroid offset
        ys, xs = np.where(binary)
        cy, cx = ys.mean() / m.shape[0], xs.mean() / m.shape[1]
        off = ((cy - 0.5)**2 + (cx - 0.5)**2) ** 0.5

        # bounding box
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        bh = (y1 - y0) / m.shape[0]
        bw = (x1 - x0) / m.shape[1]
        aspect = max(bh, bw) / (min(bh, bw) + 1e-6)

        if ratio < MASK_RATIO_LO:
            reason.append(f"small={ratio:.3f}")
        if ratio > MASK_RATIO_HI:
            reason.append(f"closeup={ratio:.3f}")
        if off > CENTER_THRESH:
            reason.append(f"off_center={off:.2f}")
        if bh < BBOX_MIN or bw < BBOX_MIN:
            reason.append(f"tiny_bbox={bw:.2f}x{bh:.2f}")
        if aspect > BBOX_ASPECT_MAX:
            reason.append(f"bad_aspect={aspect:.1f}")

        if reason:
            bad.append((p.stem, ratio, ", ".join(reason)))

    print(f"\n{split}: {len(bad)} bad / {total} total")
    for name, r, reasons in bad:
        print(f"  {name}  mask={r:.3f}  [{reasons}]")
