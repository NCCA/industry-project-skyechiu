"""
Compute pseudo-GT surface normals from RGB images using DSINE
(Bae & Davison, "Rethinking Inductive Biases for Surface Normal Estimation",
CVPR 2024). Loaded via torch.hub — no manual repo clone needed.

Why this, not the 3DGS-rendered normals
---------------------------------------
Our data/dataset/{split}/normal/ comes from the FaceLift 3DGS renderer and
inherits the well-known per-splat aliasing: pits/bumps on skin where splats
don't agree on their local frame. This is a terrible supervision signal for a
normal-aware depth SR loss — the Sobel-from-predicted-depth normals are
smoother than it. DN-Splatter (WACV 2025) and several 2024 3DGS papers
sidestep this by replacing rendered normals with a pretrained monocular
normal estimator's output. DSINE is the simplest / highest-quality option
currently (better than Omnidata-v2 on face regions in the paper's own eval).

What this script does
---------------------
For every RGB in data/dataset/{train,val}/image/:
  1. Load image, resize to a DSINE-friendly size (default 480 on long side),
  2. Run DSINE once,
  3. Upsample the predicted normal map to 1024×1024, renormalize per-pixel,
  4. Save as 3-channel uint8 PNG at
        data/dataset/{split}/normal_dsine/<name>.png
     using the same [0,255] ↔ [-1,+1] packing our existing normal/ files use:
         r = (nx + 1) / 2 * 255
         g = (ny + 1) / 2 * 255
         b = (nz + 1) / 2 * 255

Camera / intrinsics note
------------------------
DSINE takes an optional camera intrinsics tensor `intrins` of shape (B, 3, 3)
for best quality. We do not have calibrated intrinsics for the FaceLift
renders; DSINE falls back to a default FoV if `intrins=None`, which is fine
for close-range face crops. If you later want to tighten this, plug in the
FaceLift render's actual intrinsics (they live in the notebook around the
get_camera_at_yaw helper).

Usage
-----
    # Smoke test on 20 images
    python scripts/compute_dsine_normals.py --limit 20

    # Full run (both splits)
    python scripts/compute_dsine_normals.py

    # Redo a split
    python scripts/compute_dsine_normals.py --splits train --force

Timing
------
4070 Laptop, fp16 AMP, 480×480 input, batch=4: ~0.15 s/image
1288 images → ~3-4 min end-to-end.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
DSINE_INPUT = 480          # DSINE default input long side
HR_OUT = 1024              # match HR depth resolution
BATCH = 4


# -----------------------------------------------------------------------------
# Model loader  (manual — torch.hub support in DSINE repo is broken as of 2026)
# -----------------------------------------------------------------------------
def _ensure_dsine_repo() -> Path:
    """Return path to cached DSINE repo, downloading if needed."""
    hub_dir = Path(torch.hub.get_dir()) / "baegwangbin_DSINE_main"
    if not hub_dir.exists():
        torch.hub.download_url_to_file(
            "https://github.com/baegwangbin/DSINE/zipball/main",
            str(hub_dir.parent / "main.zip"),
        )
        import zipfile
        with zipfile.ZipFile(hub_dir.parent / "main.zip") as zf:
            zf.extractall(hub_dir.parent)
    return hub_dir


def _ensure_checkpoint() -> Path:
    """Return path to DSINE checkpoint, downloading if needed."""
    ckpt = Path(torch.hub.get_dir()) / "checkpoints" / "dsine.pt"
    if not ckpt.exists():
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(
            "https://huggingface.co/camenduru/DSINE/resolve/main/dsine.pt",
            str(ckpt),
        )
    return ckpt


def _make_dsine_args():
    """Default config for the released DSINE v0.2 checkpoint."""
    from types import SimpleNamespace
    return SimpleNamespace(
        NNET_encoder_B=5,
        NNET_decoder_NF=2048,
        NNET_decoder_BN=False,
        NNET_decoder_down=8,
        NNET_learned_upsampling=True,
        NRN_prop_ps=5,
        NRN_num_iter_train=5,
        NRN_num_iter_test=5,
        NRN_ray_relu=True,
        NNET_output_dim=3,
        NNET_feature_dim=64,
        NNET_hidden_dim=64,
    )


def _axis_angle_to_matrix(axis_angle):
    """Minimal Rodrigues impl — replaces utils.rotation.axis_angle_to_matrix
    that the DSINE repo fails to import."""
    import math
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)  # (..., 1)
    axis = axis_angle / (angle + 1e-8)
    cos, sin = torch.cos(angle), torch.sin(angle)
    # (..., 3)
    ax, ay, az = axis.unbind(-1)
    zero = torch.zeros_like(ax)
    K = torch.stack([
        zero, -az,  ay,
         az, zero, -ax,
        -ay,  ax, zero,
    ], dim=-1).reshape(*axis_angle.shape[:-1], 3, 3)
    I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    # Broadcast I
    I = I.expand_as(K)
    R = I + sin.unsqueeze(-1) * K + (1 - cos.unsqueeze(-1)) * (K @ K)
    return R


def load_dsine(device: str = "cuda"):
    """Load DSINE v0.2 manually, bypassing broken torch.hub entry point."""
    import sys
    repo = _ensure_dsine_repo()
    ckpt = _ensure_checkpoint()

    # Add repo root to sys.path so `models.*` and `utils.*` resolve
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    # Patch missing utils.rotation before importing the model
    import types
    rotation_mod = types.ModuleType("utils.rotation")
    rotation_mod.axis_angle_to_matrix = _axis_angle_to_matrix
    sys.modules["utils.rotation"] = rotation_mod
    # Ensure parent `utils` package exists in sys.modules
    if "utils" not in sys.modules:
        utils_pkg = types.ModuleType("utils")
        utils_pkg.__path__ = [str(repo / "utils")]
        sys.modules["utils"] = utils_pkg
    sys.modules["utils"].rotation = rotation_mod

    # Now import the model class
    from models.dsine.v02 import DSINE_v02

    args = _make_dsine_args()
    model = DSINE_v02(args)
    sd = torch.load(str(ckpt), map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(sd, strict=True)
    model.eval().to(device)
    model.pixel_coords = model.pixel_coords.to(device)
    return model


# -----------------------------------------------------------------------------
# Intrinsics helper
# -----------------------------------------------------------------------------
def _default_intrins(H: int, W: int, device: str = "cuda",
                     fov: float = 60.0) -> torch.Tensor:
    """Pinhole intrinsics from a default FoV (degrees). Shape (3,3)."""
    import math
    f = 0.5 * W / math.tan(0.5 * fov * math.pi / 180.0)
    K = torch.tensor([
        [f,   0.0, W / 2.0],
        [0.0, f,   H / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32, device=device)
    return K


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------
# DSINE uses ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_rgb_tensor(path: Path) -> torch.Tensor:
    """Load RGB, resize to DSINE_INPUT (square), normalize, return (3,H,W)."""
    img = Image.open(path).convert("RGB")
    img = img.resize((DSINE_INPUT, DSINE_INPUT), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    t = torch.from_numpy(arr.transpose(2, 0, 1).copy())
    return t


def _save_normal_png(n_chw: np.ndarray, out_path: Path, size: int = HR_OUT):
    """n_chw: (3, h, w) float normals in [-1, 1]. Upsample to (size,size),
    renormalize per pixel, pack to RGB uint8."""
    if n_chw.shape[-1] != size:
        n_t = torch.from_numpy(n_chw).unsqueeze(0)
        n_t = F.interpolate(n_t, size=(size, size), mode="bilinear",
                             align_corners=False)
        n_chw = n_t[0].numpy()
    # Renormalize to unit length
    norm = np.linalg.norm(n_chw, axis=0, keepdims=True)
    norm = np.clip(norm, 1e-6, None)
    n_chw = n_chw / norm
    rgb = ((n_chw + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    rgb = rgb.transpose(1, 2, 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(out_path)


# -----------------------------------------------------------------------------
# Batch inference
# -----------------------------------------------------------------------------
def _split_files(data_root: Path, splits) -> list[tuple[Path, Path]]:
    pairs = []
    for split in splits:
        img_dir = data_root / split / "image"
        out_dir = data_root / split / "normal_dsine"
        if not img_dir.exists():
            print(f"[skip] no image dir at {img_dir}")
            continue
        for p in sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg")):
            pairs.append((p, out_dir / (p.stem + ".png")))
    return pairs


@torch.inference_mode()
def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading DSINE (device={device}) ...")
    model = load_dsine(device)

    pairs = _split_files(args.data_root, args.splits)
    if args.limit:
        pairs = pairs[: args.limit]
    if not args.force:
        pairs = [(a, b) for (a, b) in pairs if not b.exists()]

    if not pairs:
        print("Nothing to do (all outputs exist, pass --force to redo).")
        return

    print(f"Running DSINE on {len(pairs)} images "
          f"(batch={BATCH}, out={HR_OUT}×{HR_OUT}).")

    t0 = time.time()
    done = 0
    for i in range(0, len(pairs), BATCH):
        chunk = pairs[i : i + BATCH]
        xs = torch.stack([_load_rgb_tensor(p) for (p, _) in chunk]).to(device)
        # DSINE needs (B,3,3) intrinsics — use default FoV=60 for face crops
        B_cur = xs.shape[0]
        intrins = _default_intrins(DSINE_INPUT, DSINE_INPUT, device).unsqueeze(0).expand(B_cur, -1, -1).clone()
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                             enabled=device == "cuda"):
            # DSINE returns a list of pyramid outputs; last one is the final
            # full-resolution normal. Shape: (B, 3, H, W), values in [-1, 1].
            pred = model(xs, intrins=intrins)
            if isinstance(pred, (list, tuple)):
                pred = pred[-1]
        pred = pred.float().clamp(-1.0, 1.0).cpu().numpy()
        for (_, out_path), n in zip(chunk, pred):
            _save_normal_png(n, out_path, size=HR_OUT)
        done += len(chunk)
        if done % 40 == 0 or done == len(pairs):
            dt = time.time() - t0
            print(f"  [{done}/{len(pairs)}]  {dt:.1f}s  "
                  f"({done / dt:.1f} im/s)")

    print(f"Done. Output: data/dataset/{{split}}/normal_dsine/")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=Path("data/dataset"), type=Path)
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing outputs.")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
