"""
Compute pseudo-GT surface normals from RGB images using the pretrained Omnidata v2
monocular normal estimator (EPFL-VILAB/omnidata_tools).

Motivation
----------
Our 3DGS-rendered normals (data/dataset/{split}/normal/) suffer from per-splat
aliasing — pits/bumps on skin — which makes them a bad supervision signal for a
normal-aware depth SR loss. DN-Splatter (Turkulainen et al., WACV 2025) showed
that using a pretrained monocular normal estimator as pseudo-GT gives a much
cleaner signal, and the downstream geometry task actually improves.

This script runs Omnidata-v2 (DPT-Hybrid, 384×384 input) on every RGB in
data/dataset/{train,val}/image/ and saves the result as a 3-channel uint8 PNG
at data/dataset/{train,val}/normal_omnidata/<name>.png, using the same
[0,255] ↔ [-1,+1] packing convention as our existing normal/ files:
    r = (nx + 1) / 2 * 255
    g = (ny + 1) / 2 * 255
    b = (nz + 1) / 2 * 255

Output resolution matches the HR depth (1024×1024) — Omnidata predicts at
384×384, we bilinear-upsample + renormalize to unit length per-pixel.

Usage
-----
    # One-time model download (~500 MB) into ./omnidata_ckpts/
    python scripts/compute_omnidata_normals.py --download

    # Run on full dataset
    python scripts/compute_omnidata_normals.py --data_root data/dataset

    # Restrict to a subset / skip existing
    python scripts/compute_omnidata_normals.py --splits train --limit 20

Dependencies
------------
    pip install torch torchvision einops timm
The Omnidata pretrained weights file (omnidata_dpt_normal_v2.ckpt) is fetched
from the official release (~500 MB). The DPT-Hybrid arch is rebuilt locally
from a trimmed copy of the reference code bundled in scripts/_omnidata_dpt.py,
so we don't need to clone the full omnidata_tools repo.

Compatibility note
------------------
If scripts/_omnidata_dpt.py does not exist yet, run with --stub_arch to get a
ready-to-edit stub — the stub contains the exact import list and class shell
you need to paste from
    https://github.com/EPFL-VILAB/omnidata/blob/main/omnidata_tools/torch/modules/midas/dpt_depth.py
    https://github.com/EPFL-VILAB/omnidata/blob/main/omnidata_tools/torch/modules/midas/blocks.py
into that single file. This is a one-time setup step.

Timing
------
4070 Laptop @ 384×384 DPT-Hybrid, fp16, batch=4: ~0.25 s/image
1288 samples → ~5-6 min end-to-end.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
CKPT_URL = (
    "https://omnidata.vision/assets/pretrained_models/"
    "omnidata_dpt_normal_v2.ckpt"
)
CKPT_LOCAL = Path("omnidata_ckpts/omnidata_dpt_normal_v2.ckpt")

OMNI_INPUT = 384          # Omnidata DPT-Hybrid native input size
HR_OUT = 1024             # we upsample to match HR depth
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)
BATCH = 4


# -----------------------------------------------------------------------------
# Model loader
# -----------------------------------------------------------------------------
def _ensure_checkpoint():
    """Download the Omnidata v2 normal checkpoint if not already cached."""
    if CKPT_LOCAL.exists() and CKPT_LOCAL.stat().st_size > 100 * 1024 * 1024:
        return
    CKPT_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Omnidata v2 normal checkpoint (~500 MB) -> {CKPT_LOCAL}")
    urllib.request.urlretrieve(CKPT_URL, CKPT_LOCAL)
    print("Done.")


def load_omnidata_model(device: str = "cuda"):
    """Build DPT-Hybrid and load the Omnidata v2 normal weights."""
    _ensure_checkpoint()
    try:
        from _omnidata_dpt import DPTDepthModel  # local copy
    except ModuleNotFoundError as e:
        raise SystemExit(
            "scripts/_omnidata_dpt.py not found.\n"
            "Run with --stub_arch to generate a stub, then paste the official\n"
            "Omnidata DPT-Hybrid arch into it (see top-of-file comment for URLs)."
        ) from e

    model = DPTDepthModel(
        backbone="vitb_rn50_384",
        non_negative=False,          # normals are signed
        invert=False,
        enable_attention_hooks=False,
        num_channels=3,              # 3 channels for normals, not 1 for depth
    )
    sd = torch.load(CKPT_LOCAL, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    # The checkpoint often has a "model." prefix — strip it if present
    sd = {k.replace("model.", "", 1) if k.startswith("model.") else k: v
          for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[loader] missing={len(missing)}  unexpected={len(unexpected)}")
    model.eval().to(device)
    return model


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------
def _load_rgb_tensor(path: Path) -> torch.Tensor:
    """Load an RGB image and return a (3, OMNI_INPUT, OMNI_INPUT) float tensor
    normalized with MEAN/STD."""
    img = Image.open(path).convert("RGB")
    img = img.resize((OMNI_INPUT, OMNI_INPUT), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0          # (H,W,3)
    arr = (arr - np.asarray(MEAN)) / np.asarray(STD)
    t = torch.from_numpy(arr.transpose(2, 0, 1).copy())       # (3,H,W)
    return t


def _save_normal_png(n_chw: np.ndarray, out_path: Path, size: int = HR_OUT):
    """n_chw: (3, H, W) float in [-1, 1].  Save as RGB PNG at `size`×`size`."""
    n = n_chw
    # Upsample to HR
    if n.shape[-1] != size:
        n_t = torch.from_numpy(n).unsqueeze(0)                 # (1,3,H,W)
        n_t = F.interpolate(n_t, size=(size, size), mode="bilinear",
                             align_corners=False)
        n = n_t[0].numpy()
    # Renormalize to unit length per pixel (bilinear breaks unit length)
    norm = np.linalg.norm(n, axis=0, keepdims=True)
    norm = np.clip(norm, 1e-6, None)
    n = n / norm
    # Pack to uint8
    rgb = ((n + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    rgb = rgb.transpose(1, 2, 0)                               # (H,W,3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # PIL uses RGB order
    Image.fromarray(rgb, mode="RGB").save(out_path)


# -----------------------------------------------------------------------------
# Batch inference
# -----------------------------------------------------------------------------
def _split_files(data_root: Path, splits) -> list[tuple[Path, Path]]:
    """Return list of (rgb_in, normal_out) for all splits."""
    pairs = []
    for split in splits:
        img_dir = data_root / split / "image"
        out_dir = data_root / split / "normal_omnidata"
        if not img_dir.exists():
            print(f"[skip] no image dir at {img_dir}")
            continue
        for p in sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg")):
            pairs.append((p, out_dir / (p.stem + ".png")))
    return pairs


@torch.inference_mode()
def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_omnidata_model(device)

    pairs = _split_files(args.data_root, args.splits)
    if args.limit:
        pairs = pairs[: args.limit]

    if not args.force:
        pairs = [(a, b) for (a, b) in pairs if not b.exists()]

    if not pairs:
        print("Nothing to do (all outputs exist, pass --force to redo).")
        return

    print(f"Running Omnidata-v2 on {len(pairs)} images "
          f"(device={device}, batch={BATCH}, out={HR_OUT}×{HR_OUT}).")

    t0 = time.time()
    done = 0
    for i in range(0, len(pairs), BATCH):
        chunk = pairs[i : i + BATCH]
        xs = torch.stack([_load_rgb_tensor(p) for (p, _) in chunk]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                             enabled=device == "cuda"):
            pred = model(xs)                                   # (B, 3, H, W)
        # Omnidata normals are already in [-1, 1] roughly; clip to be safe
        pred = pred.float().clamp(-1.0, 1.0).cpu().numpy()
        for (_, out_path), n in zip(chunk, pred):
            _save_normal_png(n, out_path, size=HR_OUT)
        done += len(chunk)
        if done % 40 == 0 or done == len(pairs):
            dt = time.time() - t0
            print(f"  [{done}/{len(pairs)}]  {dt:.1f}s  "
                  f"({done / dt:.1f} im/s)")

    print(f"Done. Output: data/dataset/{{split}}/normal_omnidata/")


# -----------------------------------------------------------------------------
# Stub generator (one-time)
# -----------------------------------------------------------------------------
STUB_TEXT = '''"""
Local trimmed copy of the Omnidata-v2 DPT-Hybrid architecture.

Paste the contents of the following two files from
https://github.com/EPFL-VILAB/omnidata into this file (trimmed to just what
DPTDepthModel needs — Transformer blocks, FeatureFusionBlock, Interpolate, etc.):

  omnidata_tools/torch/modules/midas/dpt_depth.py
  omnidata_tools/torch/modules/midas/blocks.py

Keep the class name `DPTDepthModel` unchanged — compute_omnidata_normals.py
imports it by name.

This stub exists so we don't have to `pip install omnidata_tools` (which pulls
heavy unrelated deps). 5-minute one-time setup.
"""

raise NotImplementedError(
    "Paste the Omnidata DPT-Hybrid arch into this file — see module docstring."
)
'''


def _emit_stub():
    out = Path(__file__).parent / "_omnidata_dpt.py"
    if out.exists():
        print(f"[stub] {out} already exists, not overwriting.")
        return
    out.write_text(STUB_TEXT, encoding="utf-8")
    print(f"[stub] wrote {out} — see its docstring for the next step.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=Path("data/dataset"), type=Path)
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--limit", type=int, default=0,
                   help="Cap number of images (0 = no cap). Useful for smoke test.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing outputs.")
    p.add_argument("--download", action="store_true",
                   help="Only download the Omnidata checkpoint, then exit.")
    p.add_argument("--stub_arch", action="store_true",
                   help="Emit scripts/_omnidata_dpt.py stub and exit.")
    args = p.parse_args()

    if args.stub_arch:
        _emit_stub()
        return
    if args.download:
        _ensure_checkpoint()
        return
    run(args)


if __name__ == "__main__":
    main()
