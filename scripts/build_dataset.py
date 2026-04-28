#!/usr/bin/env python3
"""
Step 5: Verify rendered outputs and print dataset statistics.

After rendering (step 4), this script checks that all splats have
corresponding depth/RGB/normal/opacity maps, and reports any mismatches.

This is the validation step between rendering and postprocessing.
The actual train/val split is handled by reorganize_dataset.py.

Matches render_improve.ipynb Cell 3 + verification logic.

Usage:
    python scripts/build_dataset.py
"""

import argparse
import sys
from pathlib import Path

import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "pipeline_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _count_png(d):
    return len(list(d.glob("*.png"))) if d.exists() else 0


def _stems(d):
    return {p.stem for p in d.glob("*.png")} if d.exists() else set()


def main():
    config = load_config()

    splat_dir = Path(config["paths"]["splat_output"]).resolve()
    depth_dir = Path(config["paths"]["depth_output"]).resolve()
    normal_dir = Path(config["paths"]["normal_output"]).resolve()
    opacity_dir = Path(config["paths"]["opacity_output"]).resolve()
    rgb_dir = Path(config["paths"]["rgb_output"]).resolve()
    cropped_dir = Path(config["paths"]["dataset_cropped"]).resolve()

    print("=" * 60)
    print("  Dataset Verification")
    print("=" * 60)

    # Count splats
    splat_folders = sorted([
        d for d in splat_dir.iterdir()
        if d.is_dir() and (d / "gaussians.ply").exists()
    ]) if splat_dir.exists() else []
    splat_stems = {d.name for d in splat_folders}

    # Count rendered outputs
    dirs = {
        "Cropped faces": cropped_dir,
        "Splats": splat_dir,
        "RGB rendered": rgb_dir,
        "Depth maps": depth_dir,
        "Normal maps": normal_dir,
        "Opacity maps": opacity_dir,
    }

    for label, d in dirs.items():
        if label == "Splats":
            print(f"  {label:20s}: {len(splat_folders):5d}  ({splat_dir})")
        else:
            n = _count_png(d)
            print(f"  {label:20s}: {n:5d}  ({d})")

    # Check consistency: all 4 map types should have matching stems
    rgb_stems = _stems(rgb_dir)
    depth_stems = _stems(depth_dir)
    normal_stems = _stems(normal_dir)
    opacity_stems = _stems(opacity_dir)

    common = rgb_stems & depth_stems & normal_stems & opacity_stems
    all_stems = rgb_stems | depth_stems | normal_stems | opacity_stems

    print(f"\n  Fully rendered (all 4 maps): {len(common)}")

    if common != all_stems:
        orphan = all_stems - common
        print(f"  Incomplete renders: {len(orphan)}")
        for stem in sorted(orphan)[:10]:
            missing = []
            if stem not in rgb_stems:
                missing.append("RGB")
            if stem not in depth_stems:
                missing.append("depth")
            if stem not in normal_stems:
                missing.append("normal")
            if stem not in opacity_stems:
                missing.append("opacity")
            print(f"    {stem}: missing {', '.join(missing)}")
        if len(orphan) > 10:
            print(f"    ... and {len(orphan) - 10} more")

    # Check splats without renders
    missing_renders = splat_stems - common
    if missing_renders:
        print(f"\n  Splats without renders: {len(missing_renders)}")
        print(f"  Run scripts/export_depth.py to render them.")

    # Sample resolution check
    if depth_stems:
        sample = sorted(depth_dir.glob("*.png"))[0]
        img = Image.open(sample)
        print(f"\n  Sample depth: {img.size} mode={img.mode} ({sample.name})")

    if rgb_stems:
        sample = sorted(rgb_dir.glob("*.png"))[0]
        img = Image.open(sample)
        print(f"  Sample RGB:   {img.size} mode={img.mode} ({sample.name})")

    print(f"\nNext step: python scripts/run_postprocess.py")


if __name__ == "__main__":
    main()
