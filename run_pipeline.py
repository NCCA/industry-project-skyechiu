#!/usr/bin/env python3
"""
FaceLift Pipeline - Run complete workflow with one command.

Split into three stages:
    Stage A (1-4):  Download -> Crop -> FaceLift Inference -> Export initial depth
    [Manual]        Run high-quality re-rendering in FaceLift/render_improve.ipynb
    Stage B (7-9):  Postprocess -> Reorganize -> Make LR/HR pairs

Usage:
    python run_pipeline.py                       # Run all steps (1-9, skip 5/6 optional)
    python run_pipeline.py --start-from 3        # Start from step 3
    python run_pipeline.py --only 4              # Run only step 4
    python run_pipeline.py --start-from 3 --stop-at 4   # Only run Stage A back half
    python run_pipeline.py --start-from 7        # After render_improve, run Stage B
    python run_pipeline.py --dry-run             # Print commands without executing

Default --stop-at is 4 to avoid skipping the render_improve manual step.
To run Stage B, explicitly specify --start-from 7.
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("pipeline")

PROJECT_ROOT = Path(__file__).resolve().parent

STEPS = [
    {
        "num": 1,
        "name": "Download Dataset",
        "script": "scripts/download_dataset.py",
        "description": "Download face dataset from Kaggle",
    },
    {
        "num": 2,
        "name": "Prepare Images",
        "script": "scripts/prepare_images.py",
        "description": "Face detection + crop + resize to 512x512",
    },
    {
        "num": 3,
        "name": "FaceLift Inference",
        "script": "scripts/batch_inference.py",
        "description": "Batch run FaceLift (2D -> 3D Gaussian Splat)",
    },
    {
        "num": 4,
        "name": "Export Depth Maps",
        "script": "scripts/export_depth.py",
        "description": "Render depth maps and RGB from Gaussian Splat",
    },
    {
        "num": 5,
        "name": "Build Dataset",
        "script": "scripts/build_dataset.py",
        "description": "Verify rendered outputs and print dataset stats",
    },
    {
        "num": 6,
        "name": "Verify",
        "script": "scripts/verify_splat.py",
        "description": "Verify Gaussian Splat file integrity",
    },
    {
        "num": 7,
        "name": "Postprocess Maps",
        "script": "scripts/run_postprocess.py",
        "description": "Landmark alignment / nose-tip normalization / normal smoothing / hole-filling / consistency check (must run after render_improve.ipynb)",
    },
    {
        "num": 8,
        "name": "Reorganize Dataset",
        "script": "scripts/reorganize_dataset.py",
        "description": "90/10 train/val split (seed=42)",
    },
    {
        "num": 9,
        "name": "Make LR/HR Pairs",
        "script": "scripts/make_lr_hr_pairs.py",
        "description": "Generate 256 LR (8bit+16bit) / 1024 HR uint16 pairs",
    },
]

# Manual step notification: render_improve.ipynb must be run manually after step 4 and before step 7
MANUAL_STEP_AFTER = 4  # Prompt user for manual action after completing this step


def run_step(step: dict, extra_args: list[str] = None, dry_run: bool = False) -> bool:
    """Run a single step."""
    script = PROJECT_ROOT / step["script"]

    if not script.exists():
        logger.error(f"Script not found: {script}")
        return False

    cmd = [sys.executable, str(script)] + (extra_args or [])

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"  Step {step['num']}: {step['name']}")
    logger.info(f"  {step['description']}")
    logger.info(f"  Command: {' '.join(cmd)}")
    logger.info("=" * 70)

    if dry_run:
        logger.info("  [DRY RUN] Skipped")
        return True

    start = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - start

    if result.returncode != 0:
        logger.error(f"  Step {step['num']} FAILED (exit code {result.returncode})")
        return False

    logger.info(f"  Step {step['num']} completed in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="FaceLift Pipeline - Run all steps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  Stage A (Automatic):
    1. Download Dataset      - Download face dataset from Kaggle
    2. Prepare Images        - Face detection + crop + resize
    3. FaceLift Inference    - 2D -> 3D Gaussian Splat
    4. Export Depth Maps     - Gaussian Splat -> Depth + RGB (initial)

  [Manual]  Run high-quality re-rendering in FaceLift/render_improve.ipynb

  Optional:
    5. Build Dataset         - Legacy paired dataset (skip if desired)
    6. Verify                - Verify Splat file integrity (skip if desired)

  Stage B (Trigger manually after render_improve with --start-from 7):
    7. Postprocess Maps      - Alignment/normalization/smoothing/hole-fill/consistency check
    8. Reorganize Dataset    - 90/10 train/val split
    9. Make LR/HR Pairs      - 256 LR + 1024 HR pairs
        """,
    )
    parser.add_argument(
        "--start-from", type=int, default=1,
        help="Start from step N (default: 1)",
    )
    parser.add_argument(
        "--stop-at", type=int, default=4,
        help="Stop at step N (default: 4 - stops before render_improve manual step)",
    )
    parser.add_argument(
        "--skip", type=int, nargs="*", default=[5, 6],
        help="Skip these step numbers (default: 5 6 - legacy build/verify)",
    )
    parser.add_argument(
        "--only", type=int, default=None,
        help="Run only step N",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Limit number of images (passed to step 1 & 2)",
    )
    args = parser.parse_args()

    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║     FaceLift Pipeline Runner             ║")
    logger.info("║     2D Face → 3D Splat → Depth + RGB    ║")
    logger.info("╚══════════════════════════════════════════╝")

    # Determine which steps to run
    if args.only:
        steps_to_run = [s for s in STEPS if s["num"] == args.only]
    else:
        steps_to_run = [
            s for s in STEPS
            if args.start_from <= s["num"] <= args.stop_at
            and s["num"] not in args.skip
        ]

    if not steps_to_run:
        logger.error("No valid steps to run!")
        sys.exit(1)

    total_start = time.time()
    failed = []

    for step in steps_to_run:
        extra = []
        if args.max_images and step["num"] in [1, 2]:
            extra.extend(["--max-images", str(args.max_images)])

        ok = run_step(step, extra_args=extra, dry_run=args.dry_run)
        if not ok:
            failed.append(step["num"])
            logger.error(f"Pipeline stopped at step {step['num']}")
            break

        # Prompt for manual render_improve after step 4
        if step["num"] == MANUAL_STEP_AFTER and not args.dry_run:
            next_auto = [s for s in steps_to_run if s["num"] > MANUAL_STEP_AFTER]
            if next_auto:
                logger.warning("")
                logger.warning("!" * 70)
                logger.warning("  Manual step: Please run high-quality re-rendering in FaceLift/render_improve.ipynb")
                logger.warning("  After completion, resume with: python run_pipeline.py --start-from 7")
                logger.warning("!" * 70)
                logger.warning("")
                reply = input("  Has render_improve completed? [y/N]: ").strip().lower()
                if reply != "y":
                    logger.info("  Stopping here. Resume later with --start-from 7")
                    break

    total_elapsed = time.time() - total_start

    logger.info("")
    logger.info("=" * 70)
    if failed:
        logger.error(f"Pipeline FAILED at step(s): {failed}")
        logger.info(f"Fix the issue and re-run with: --start-from {failed[0]}")
    else:
        logger.info("Pipeline completed successfully!")
    logger.info(f"Total time: {total_elapsed / 60:.1f} minutes")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
