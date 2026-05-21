"""
Reorganize post-processed maps into a standard train/val dataset structure.

Source layout:
    data/postprocessed/
        rgb/<sample>.png
        depth/<sample>.png       (16-bit, 1024x1024)
        normal/<sample>.png
        opacity/<sample>.png

Output layout:
    data/dataset/
        train/
            image/   (RGB)
            depth/   (16-bit)
            normal/
            opacity/
        val/
            image/, depth/, normal/, opacity/
        manifest.json   (records the split)

Usage:
    python scripts/reorganize_dataset.py
    python scripts/reorganize_dataset.py --val_ratio 0.1 --seed 42
"""

import argparse
import json
import random
import shutil
from pathlib import Path


MODALITY_MAP = {
    "rgb": "image",
    "depth": "depth",
    "normal": "normal",
    "opacity": "opacity",
}


def reorganize(src_root: Path, dst_root: Path, val_ratio: float, seed: int):
    src_rgb = src_root / "rgb"
    if not src_rgb.exists():
        raise SystemExit(f"Source missing: {src_rgb}")

    samples = sorted(p.stem for p in src_rgb.glob("*.png"))
    if not samples:
        raise SystemExit("No samples found in source rgb folder.")

    # Verify all modalities have the same samples
    for sub in ("depth", "normal", "opacity"):
        present = {p.stem for p in (src_root / sub).glob("*.png")}
        missing = set(samples) - present
        if missing:
            print(f"WARN [{sub}] missing {len(missing)} samples: {sorted(missing)[:5]}...")
            samples = [s for s in samples if s not in missing]

    rng = random.Random(seed)
    rng.shuffle(samples)
    n_val = max(1, int(round(len(samples) * val_ratio)))
    val_samples = sorted(samples[:n_val])
    train_samples = sorted(samples[n_val:])
    print(f"Total: {len(samples)} | train: {len(train_samples)} | val: {len(val_samples)}")

    # Create destination structure
    if dst_root.exists():
        print(f"Removing existing {dst_root}")
        shutil.rmtree(dst_root)
    for split in ("train", "val"):
        for modality in MODALITY_MAP.values():
            (dst_root / split / modality).mkdir(parents=True, exist_ok=True)

    def copy_split(split_name: str, names):
        for name in names:
            for src_sub, dst_sub in MODALITY_MAP.items():
                src = src_root / src_sub / f"{name}.png"
                dst = dst_root / split_name / dst_sub / f"{name}.png"
                if src.exists():
                    shutil.copy2(src, dst)
        print(f"  {split_name}: copied {len(names)} samples x 4 modalities")

    print("Copying...")
    copy_split("train", train_samples)
    copy_split("val", val_samples)

    manifest = {
        "source": str(src_root),
        "destination": str(dst_root),
        "val_ratio": val_ratio,
        "seed": seed,
        "total": len(samples),
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "train_samples": train_samples,
        "val_samples": val_samples,
        "modalities": list(MODALITY_MAP.values()),
    }
    with open(dst_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest to {dst_root/'manifest.json'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/postprocessed", type=Path)
    p.add_argument("--dst", default="data/dataset", type=Path)
    p.add_argument("--val_ratio", default=0.10, type=float)
    p.add_argument("--seed", default=42, type=int)
    args = p.parse_args()
    reorganize(args.src, args.dst, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
