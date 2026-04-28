"""
Zero-shot evaluation of DORNet (CVPR 2025) pretrained on NYU-v2.

Tests whether SOTA depth SR trained on standard bicubic degradation
generalizes to our 3DGS rendering degradation. Expected: it won't,
which supports contribution (b) — rendering degradation is distinct.

Usage:
    python scripts/eval_dornet_zeroshot.py
    python scripts/eval_dornet_zeroshot.py --resolution 512  # if OOM at 1024
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ============================================================
# Patch mmcv deformable conv → torchvision (no mmcv install needed)
# ============================================================
def _patch_mmcv_deform_conv():
    """Replace mmcv.ops.modulated_deform_conv2d with torchvision's version."""
    import types
    try:
        from torchvision.ops import deform_conv2d as tv_deform_conv2d
    except ImportError:
        raise RuntimeError("torchvision.ops.deform_conv2d not available. "
                           "Need torchvision >= 0.9")

    def modulated_deform_conv2d(input, offset, mask, weight, bias,
                                 stride, padding, dilation, groups,
                                 deformable_groups):
        """Wrapper: mmcv API → torchvision API."""
        # torchvision.ops.deform_conv2d doesn't expose groups directly,
        # but for batch=1 inference, groups=1 works.
        # For batch>1 with per-sample weights, we loop over batch.
        b_in = input.shape[0]  # always 1 due to DCN_layer_rgb reshape
        return tv_deform_conv2d(
            input, offset, weight,
            bias=bias,
            stride=(stride, stride) if isinstance(stride, int) else stride,
            padding=(padding, padding) if isinstance(padding, int) else padding,
            dilation=(dilation, dilation) if isinstance(dilation, int) else dilation,
            mask=mask,
        )

    # Create a fake mmcv.ops module
    mmcv_ops = types.ModuleType('mmcv.ops')
    mmcv_ops.modulated_deform_conv2d = modulated_deform_conv2d
    mmcv_mod = types.ModuleType('mmcv')
    mmcv_mod.ops = mmcv_ops
    sys.modules['mmcv'] = mmcv_mod
    sys.modules['mmcv.ops'] = mmcv_ops


# Patch before importing DORNet
_patch_mmcv_deform_conv()


# ============================================================
# Dataset (reuse our standard depth SR dataset)
# ============================================================
class DORNetValDataset(torch.utils.data.Dataset):
    """Val dataset for DORNet zero-shot: returns (rgb, lr_depth_upsampled, hr_depth, mask)."""

    def __init__(self, root: Path, split: str, lr_kind: str = "8bit",
                 resolution: int = 1024):
        self.split_dir = root / split
        self.hr_dir = self.split_dir / "depth"
        self.lr_dir = self.split_dir / f"depth_lr_{lr_kind}"
        self.rgb_dir = self.split_dir / "image"
        self.mask_dir = self.split_dir / "mask"
        self.resolution = resolution

        self.samples = sorted(
            p.stem for p in self.hr_dir.glob("*.png")
            if (self.lr_dir / p.name).exists()
            and (self.rgb_dir / p.name).exists()
        )
        if not self.samples:
            raise RuntimeError(f"No samples in {self.split_dir}")

        self.has_mask = self.mask_dir.exists() and any(self.mask_dir.glob("*.png"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name = self.samples[idx]
        R = self.resolution

        # HR depth
        with Image.open(self.hr_dir / f"{name}.png") as img:
            hr = np.array(img, dtype=np.float32)
        hr = hr / 65535.0 if hr.max() > 255 else hr / 255.0
        if hr.shape[0] != R:
            hr = cv2.resize(hr.astype(np.float32), (R, R), interpolation=cv2.INTER_AREA)
        hr_t = torch.from_numpy(hr).unsqueeze(0)

        # LR depth → bicubic upsample to HR size (same as NYU dataloader)
        with Image.open(self.lr_dir / f"{name}.png") as img:
            lr = np.array(img, dtype=np.float32)
        lr = lr / 255.0 if lr.max() <= 255 else lr / 65535.0
        lr_up = cv2.resize(lr.astype(np.float32), (R, R), interpolation=cv2.INTER_CUBIC)
        lr_t = torch.from_numpy(lr_up).unsqueeze(0)

        # RGB guidance
        with Image.open(self.rgb_dir / f"{name}.png") as img:
            rgb = np.array(img.convert("RGB").resize((R, R), Image.BICUBIC),
                          dtype=np.float32) / 255.0
        rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1))

        # Mask
        if self.has_mask:
            mp = self.mask_dir / f"{name}.png"
            if mp.exists():
                with Image.open(mp) as img:
                    m = np.array(img.convert("L").resize((R, R), Image.NEAREST),
                                dtype=np.float32) / 255.0
            else:
                m = np.ones((R, R), dtype=np.float32)
        else:
            m = np.ones((R, R), dtype=np.float32)
        m_t = torch.from_numpy(m).unsqueeze(0)

        return rgb_t, lr_t, hr_t, m_t


# ============================================================
# Evaluation
# ============================================================
def masked_l1(pred, target, mask, eps=1e-6):
    diff = (pred - target).abs() * mask
    return diff.sum() / mask.sum().clamp(min=eps)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Import DORNet
    dornet_dir = Path(args.dornet_dir)
    if str(dornet_dir) not in sys.path:
        sys.path.insert(0, str(dornet_dir))

    from net.dornet import Net as DORNet

    # Load model
    model = DORNet(tiny_model=args.tiny_model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"DORNet({'tiny' if args.tiny_model else 'full'}): {n_params/1e6:.2f}M params")

    # Load pretrained weights
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        return

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Handle different checkpoint formats
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']

    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded checkpoint: {ckpt_path} (strict)")
    except RuntimeError as e:
        print(f"Strict load failed: {e}")
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint: {ckpt_path} (non-strict)")

    model.eval()

    # Dataset
    data_root = Path(args.data_root)
    val_ds = DORNetValDataset(data_root, "val", lr_kind=args.lr_input,
                              resolution=args.resolution)
    print(f"Val samples: {len(val_ds)}, resolution: {args.resolution}")

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # Run inference
    val_l1 = 0.0
    val_mse = 0.0
    n_samples = 0
    t0 = time.time()

    with torch.no_grad():
        for i, (rgb, lr_up, hr, mask) in enumerate(val_loader):
            rgb = rgb.to(device)
            lr_up = lr_up.to(device)
            hr = hr.to(device)
            mask = mask.to(device)

            try:
                with torch.cuda.amp.autocast(enabled=args.amp):
                    # DORNet eval mode returns just the output (no aux_loss)
                    out = model(x_query=lr_up, rgb=rgb)
                    if isinstance(out, tuple):
                        out = out[0]
                    out = out.clamp(0, 1)

                l1 = masked_l1(out.float(), hr, mask).item()
                mse = ((out.float() - hr) ** 2 * mask).sum() / mask.sum().clamp(min=1e-6)

                val_l1 += l1
                val_mse += mse.item()
                n_samples += 1

                if (i + 1) % 20 == 0:
                    avg_l1 = val_l1 / n_samples
                    print(f"  [{i+1}/{len(val_ds)}] running_l1={avg_l1:.5f}")

            except torch.cuda.OutOfMemoryError:
                print(f"  OOM at sample {i}, skipping")
                torch.cuda.empty_cache()
                continue

    elapsed = time.time() - t0
    avg_l1 = val_l1 / max(n_samples, 1)
    avg_mse = val_mse / max(n_samples, 1)
    avg_psnr = -10 * np.log10(avg_mse + 1e-12)

    print(f"\n{'='*60}")
    print(f"DORNet zero-shot evaluation (NYU_X4 → 3DGS depth)")
    print(f"  Checkpoint: {ckpt_path.name}")
    print(f"  Resolution: {args.resolution}")
    print(f"  Samples evaluated: {n_samples}/{len(val_ds)}")
    print(f"  val_l1  = {avg_l1:.5f}")
    print(f"  val_MSE = {avg_mse:.6f}")
    print(f"  val_PSNR = {avg_psnr:.2f} dB")
    print(f"  Time: {elapsed:.0f}s ({elapsed/max(n_samples,1):.1f}s/sample)")
    print(f"{'='*60}")

    # Compare with our baselines
    print(f"\nComparison:")
    print(f"  UNet (ours, trained on 3DGS):   val_l1 = 0.00228")
    print(f"  SRResNet (trained on 3DGS):     val_l1 = 0.00228")
    print(f"  EDSR (trained on 3DGS):         val_l1 = 0.00241")
    print(f"  DORNet (NYU pretrained):         val_l1 = {avg_l1:.5f}")
    print(f"  SwinIR (trained on 3DGS):       val_l1 = 0.086")

    # Save results
    result_dir = Path(args.output_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "method": "DORNet_zeroshot_NYU",
        "checkpoint": str(ckpt_path),
        "resolution": args.resolution,
        "n_samples": n_samples,
        "val_l1": avg_l1,
        "val_mse": avg_mse,
        "val_psnr": avg_psnr,
        "time_s": elapsed,
    }
    out_path = result_dir / "dornet_zeroshot_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/dataset", type=str)
    p.add_argument("--dornet_dir", default="external/DORNet", type=str)
    p.add_argument("--checkpoint", default="external/DORNet/checkpoints/NYU_X4.pth",
                   type=str)
    p.add_argument("--lr_input", default="8bit", type=str)
    p.add_argument("--resolution", default=1024, type=int,
                   help="Eval resolution (1024 or 512 if OOM)")
    p.add_argument("--tiny_model", action="store_true",
                   help="Use tiny DORNet (n_feats=24). Must match checkpoint.")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--output_dir", default="eval", type=str)
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
