#!/usr/bin/env python3
"""
Step 3: Run FaceLift inference (cropped face -> 3D Gaussian Splat).

Matches facelift_pipeline.ipynb Cells 6-8.
Calls FaceLift/inference.py via subprocess with real-time stdout streaming.

Usage:
    python scripts/batch_inference.py
    python scripts/batch_inference.py --batch-size 50
"""

import argparse
import os
import subprocess
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
    parser = argparse.ArgumentParser(description="Batch FaceLift inference")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Number of images per batch (default: 100)")
    args = parser.parse_args()

    config = load_config()
    processed_dir = Path(config["paths"]["dataset_cropped"]).resolve()
    splat_dir = Path(config["paths"]["splat_output"]).resolve()
    splat_dir.mkdir(parents=True, exist_ok=True)
    facelift_repo = Path(config["paths"]["facelift_repo"]).resolve()

    # --- Check FaceLift model availability ---
    mvdiff_ckpt = facelift_repo / "checkpoints" / "mvdiffusion" / "pipeckpts"
    gslrm_ckpt = facelift_repo / "checkpoints" / "gslrm" / "ckpt_0000000000021125.pt"
    facelift_available = mvdiff_ckpt.exists() and gslrm_ckpt.exists()

    print(f"FaceLift repo:    {facelift_repo}")
    print(f"Models available: {facelift_available}")

    if not facelift_available:
        print("ERROR: FaceLift checkpoints not found.")
        print(f"  Expected: {mvdiff_ckpt}")
        print(f"  Expected: {gslrm_ckpt}")
        print("  Download FaceLift models first (see FaceLift README).")
        sys.exit(1)

    # --- Compute pending images ---
    image_exts = {".jpg", ".jpeg", ".png"}
    input_images = sorted([f for f in processed_dir.iterdir()
                           if f.suffix.lower() in image_exts])
    pending = [f for f in input_images
               if not (splat_dir / f.stem / "gaussians.ply").exists()]

    print(f"Cropped images: {len(input_images)}")
    print(f"Already done:   {len(input_images) - len(pending)}")
    print(f"Pending:        {len(pending)}")

    if not pending:
        print("All cropped images already have splats.")
        return

    # --- Run FaceLift inference via subprocess ---
    inf_cfg = config["inference"]
    inference_script = str(facelift_repo / "inference.py")

    cmd = [
        sys.executable, "-u", inference_script,  # -u: unbuffered stdout
        "--input_dir", str(processed_dir),
        "--output_dir", str(splat_dir),
        "--auto_crop",
        "--guidance_scale_2D", str(inf_cfg.get("guidance_scale_2D", 3.0)),
        "--step_2D", str(inf_cfg.get("step_2D", 50)),
    ]

    print(f"\nInput:  {processed_dir}")
    print(f"Output: {splat_dir}")
    print(f"Pending: {len(pending)} images")
    print(f"CMD: {' '.join(cmd)}")
    print("(Loading mvdiffusion + gslrm checkpoints -- takes ~1-2 min)\n", flush=True)

    env = os.environ.copy()
    env["SKIP_TURNTABLE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        cmd,
        cwd=str(facelift_repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        bufsize=1,
    )

    for line in process.stdout:
        print(line, end="", flush=True)

    process.wait()

    if process.returncode != 0:
        print(f"\nWARNING: Return code {process.returncode}")

    ply_count = len(list(splat_dir.glob("**/gaussians.ply")))
    print(f"\nTotal Gaussian Splats: {ply_count}")
    print(f"\nNext step: python scripts/export_depth.py")


if __name__ == "__main__":
    main()
