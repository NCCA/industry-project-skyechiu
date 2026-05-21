"""
Standalone post-processing runner (replaces notebook Step 4.5).

Reads raw rendered maps from data/{rgb_rendered,depth_maps,normal_maps,opacity_maps}/
and the original cropped face from data/cropped_faces/, runs the full
postprocess_maps.postprocess_single() pipeline, and writes results to
data/postprocessed/{rgb,depth,normal,opacity}/.

Usage:
    python scripts/run_postprocess.py
    python scripts/run_postprocess.py --workers 4
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Make scripts/ importable
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from postprocess_maps import LandmarkAligner, postprocess_single  # noqa: E402


def save_outputs(result: dict, sample: str, out_dirs: dict):
    # RGB (uint8)
    Image.fromarray(result["render_rgb"]).save(out_dirs["rgb"] / f"{sample}.png")

    # Depth (16-bit)
    depth = result["depth"]
    depth16 = (np.clip(depth, 0, 1) * 65535).astype(np.uint16)
    Image.fromarray(depth16, mode="I;16").save(out_dirs["depth"] / f"{sample}.png")

    # Normal (uint8 RGB)
    Image.fromarray(result["normal"]).save(out_dirs["normal"] / f"{sample}.png")

    # Opacity (L mode 8-bit)
    Image.fromarray(result["opacity"], mode="L").save(out_dirs["opacity"] / f"{sample}.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data", type=Path)
    p.add_argument("--out_dir", default="data/postprocessed", type=Path)
    p.add_argument("--limit", default=0, type=int,
                   help="Process only first N samples (0 = all)")
    args = p.parse_args()

    src = {
        "cropped": args.data_root / "cropped_faces",
        "rgb": args.data_root / "rgb_rendered",
        "depth": args.data_root / "depth_maps",
        "normal": args.data_root / "normal_maps",
        "opacity": args.data_root / "opacity_maps",
    }

    # Verify all source dirs exist
    for k, d in src.items():
        if not d.exists():
            raise SystemExit(f"Missing source dir: {d}")

    out_dirs = {
        "rgb": args.out_dir / "rgb",
        "depth": args.out_dir / "depth",
        "normal": args.out_dir / "normal",
        "opacity": args.out_dir / "opacity",
    }
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Build sample list (intersect cropped + rendered)
    cropped_stems = {p.stem for p in src["cropped"].glob("*.png")}
    rendered_stems = {p.stem for p in src["rgb"].glob("*.png")}
    samples = sorted(cropped_stems & rendered_stems)
    if args.limit > 0:
        samples = samples[: args.limit]
    print(f"Processing {len(samples)} samples")

    # Sanity check resolutions
    sample_rgb = Image.open(src["rgb"] / f"{samples[0]}.png")
    sample_depth = Image.open(src["depth"] / f"{samples[0]}.png")
    print(f"  RGB rendered:   {sample_rgb.size} ({sample_rgb.mode})")
    print(f"  Depth rendered: {sample_depth.size} ({sample_depth.mode})")
    if sample_rgb.size != (1024, 1024):
        print(f"  WARNING: rendered size is {sample_rgb.size}, expected 1024x1024")
        print(f"  (Make sure you re-rendered with config render_resolution: 1024)")

    aligner = LandmarkAligner()
    pp_config = {
        "opacity_threshold": 0.3,
        "min_component_area": 100,
        "use_ecc_refine": True,
        "fill_holes": True,
        "max_hole_area": 5000,
        "normals_from_depth": True,
        "camera_hfov": 50.0,
        "bilateral_d": 9,
        "bilateral_sigma_color": 30.0,
        "bilateral_sigma_space": 30.0,
    }

    n_ok = 0
    n_fail = 0
    consistency_scores = []
    t0 = time.time()

    for i, name in enumerate(samples):
        try:
            result = postprocess_single(
                str(src["cropped"] / f"{name}.png"),
                str(src["rgb"] / f"{name}.png"),
                str(src["depth"] / f"{name}.png"),
                str(src["normal"] / f"{name}.png"),
                str(src["opacity"] / f"{name}.png"),
                aligner, pp_config,
            )
            save_outputs(result, name, out_dirs)
            consistency_scores.append(result.get("consistency_score", 0.0))
            n_ok += 1
        except Exception as e:
            print(f"  FAIL [{name}]: {e}")
            n_fail += 1

        if (i + 1) % 25 == 0 or (i + 1) == len(samples):
            elapsed = time.time() - t0
            avg = elapsed / (i + 1)
            remain = avg * (len(samples) - i - 1)
            print(f"  [{i+1}/{len(samples)}] avg={avg:.2f}s ETA={remain/60:.1f}min")

    print(f"\nDone. ok={n_ok} fail={n_fail}")
    if consistency_scores:
        cs = np.array(consistency_scores)
        print(f"Consistency score: mean={cs.mean():.3f} min={cs.min():.3f} max={cs.max():.3f}")
        print(f"  >0.9: {(cs>0.9).sum()}/{len(cs)}  ({(cs>0.9).mean()*100:.0f}%)")
        print(f"  >0.8: {(cs>0.8).sum()}/{len(cs)}")

    # Verify output sizes
    out_sample = sorted(out_dirs["depth"].glob("*.png"))
    if out_sample:
        img = Image.open(out_sample[0])
        print(f"\nOutput depth size: {img.size} ({img.mode})")

    print(f"Output dir: {args.out_dir}")


if __name__ == "__main__":
    main()
