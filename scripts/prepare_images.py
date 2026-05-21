#!/usr/bin/env python3
"""
Step 2: Background removal + face crop + resize.

Matches facelift_pipeline.ipynb Cell 5.
Uses rembg for background removal and FaceLift's crop_face for alignment.

Usage:
    python scripts/prepare_images.py
    python scripts/prepare_images.py --max-images 20
"""

import argparse
import gc
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "pipeline_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Preprocess face images (rembg + crop)")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-input-side", type=int, default=1024,
                        help="Downsize huge images before rembg to save RAM")
    args = parser.parse_args()

    config = load_config()
    raw_dir = Path(config["paths"]["dataset_raw"]).resolve()
    processed_dir = Path(config["paths"]["dataset_cropped"]).resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)

    TARGET_SIZE = config["preprocessing"]["target_size"]  # 512
    BG_COLOR = (255, 255, 255)
    MAX_INPUT_SIDE = args.max_input_side

    # --- Redirect rembg model cache (avoid filling C: drive) ---
    rembg_home = (PROJECT_ROOT / ".rembg_models").resolve()
    rembg_home.mkdir(parents=True, exist_ok=True)
    os.environ["U2NET_HOME"] = str(rembg_home)
    os.environ["REMBG_HOME"] = str(rembg_home)
    print(f"rembg model cache: {rembg_home}")

    # --- Import FaceLift face utilities ---
    facelift_dir = Path(config["paths"]["facelift_repo"]).resolve()
    if str(facelift_dir) not in sys.path:
        sys.path.insert(0, str(facelift_dir))

    from rembg import new_session, remove
    from utils_folder.face_utils import crop_face, FACE_DETECTOR, FACE_SIZE, FACE_CENTER
    from tqdm.auto import tqdm

    # --- Load rembg model (try multiple in quality order) ---
    preferred_models = ["birefnet-portrait", "birefnet-general", "u2net_human_seg", "u2net"]
    rembg_session = None
    rembg_model_name = None
    for m in preferred_models:
        try:
            print(f"Loading rembg model: {m} ...")
            rembg_session = new_session(m)
            rembg_model_name = m
            print(f"  -> using {m}")
            break
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {e}")
    if rembg_session is None:
        raise RuntimeError("No rembg model could be loaded.")

    # --- Helper functions (from notebook) ---
    def _resize_if_huge(pil_img, max_side=MAX_INPUT_SIDE):
        w, h = pil_img.size
        if max(w, h) > max_side:
            s = max_side / max(w, h)
            return pil_img.resize((int(w * s), int(h * s)), Image.LANCZOS)
        return pil_img

    def preprocess_one(img_path, max_side=MAX_INPUT_SIDE):
        pil_in = Image.open(img_path).convert("RGB")
        pil_in = _resize_if_huge(pil_in, max_side)
        rgba = remove(pil_in, session=rembg_session)
        if rgba.mode != "RGBA":
            rgba = rgba.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, BG_COLOR + (255,))
        composited = Image.alpha_composite(bg, rgba).convert("RGB")
        arr = np.array(composited)
        cropped, _ = crop_face(arr, FACE_DETECTOR, FACE_SIZE, FACE_CENTER, TARGET_SIZE, BG_COLOR)
        return cropped

    # --- Enumerate raw images ---
    supported = set(config["preprocessing"]["supported_formats"])
    raw_images = sorted([f for f in raw_dir.iterdir() if f.suffix.lower() in supported])
    max_img = args.max_images or config["kaggle"].get("max_images")
    if max_img:
        raw_images = raw_images[:max_img]

    # Skip already-processed
    existing_stems = {p.stem for p in processed_dir.glob("*.png")}
    pending = [f for f in raw_images if f"{f.stem}_face" not in existing_stems]

    print(f"Raw images:        {len(raw_images)}")
    print(f"Already processed: {len(existing_stems)}")
    print(f"Pending:           {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    # --- Process ---
    success, fail = 0, 0
    error_log = []
    consecutive_fatal = 0
    pbar = tqdm(pending, desc=f"{rembg_model_name}+align")

    for i, img_path in enumerate(pbar):
        out_path = processed_dir / f"{img_path.stem}_face.png"
        try:
            # Try full resolution, fall back to smaller on OOM
            try:
                out = preprocess_one(img_path)
            except MemoryError:
                gc.collect()
                pbar.write(f"[OOM retry @ 768] {img_path.name}")
                try:
                    out = preprocess_one(img_path, max_side=768)
                except MemoryError:
                    gc.collect()
                    pbar.write(f"[OOM retry @ 512] {img_path.name}")
                    out = preprocess_one(img_path, max_side=512)

            out.save(out_path)
            if not out_path.exists() or out_path.stat().st_size == 0:
                raise IOError(f"File missing or empty after save: {out_path}")
            success += 1
            consecutive_fatal = 0

        except Exception as e:
            fail += 1
            err = f"{img_path.name}: {type(e).__name__}: {e}"
            error_log.append(err)
            pbar.write(f"[FAIL {fail}] {err}")
            if isinstance(e, OSError) and ("No space" in str(e) or "Errno 28" in str(e)):
                consecutive_fatal += 1
                if consecutive_fatal >= 3:
                    pbar.write("\n!! disk full -- aborting to avoid losing progress.")
                    break
            else:
                consecutive_fatal = 0

        if (i + 1) % 50 == 0:
            pbar.write(f"  checkpoint @ {i+1}/{len(pending)}: success={success} fail={fail}")
            gc.collect()

    print(f"\nDone! Success: {success}, Failed: {fail}")

    if error_log:
        log_path = processed_dir.parent / "preprocess_errors.log"
        log_path.write_text("\n".join(error_log), encoding="utf-8")
        print(f"Error details: {log_path}")

    # Verify all outputs are RGB
    for f in processed_dir.glob("*.png"):
        img = Image.open(f)
        if img.mode != "RGB":
            img.convert("RGB").save(f)
    total = len(list(processed_dir.glob("*.png")))
    print(f"All {total} images verified as RGB.")
    print(f"\nNext step: python scripts/batch_inference.py")


if __name__ == "__main__":
    main()
