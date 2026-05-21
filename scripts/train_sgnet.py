"""
Train SGNet (AAAI 2024) baseline for depth SR with RGB guidance.

SGNet: "Structure Guided Network for Depth Map Super-resolution"
Paper: arXiv 2312.05799
GitHub: https://github.com/yanzq95/SGNet

SGNet takes (RGB_guidance, LR_depth) -> HR_depth with ×4 internal upsample.
Patch-based training (256×256 HR patches, 64×64 LR) to fit 8GB VRAM.
Full-res 1024×1024 inference for eval (256 LR → 1024 HR, same as other baselines).

Usage:
    python scripts/train_sgnet.py --epochs 100 --batch_size 4 --amp
    python scripts/train_sgnet.py --epochs 100 --batch_size 2 --amp  # if OOM
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Dataset: LR depth + RGB guidance + HR depth + mask
# ============================================================

class SGNetDataset(Dataset):
    """
    Dataset for SGNet: returns (rgb, lr_depth, hr_depth, mask).

    Training: random 256×256 HR patches (LR = 64×64 for ×4).
    Validation: full 1024×1024 resolution (LR = 256×256).
    """

    def __init__(self, root: Path, split: str, lr_kind: str = "8bit",
                 train: bool = True, patch_size: int = 256, scale: int = 4,
                 val_hr_size: int = 512):
        self.split_dir = root / split
        self.hr_dir = self.split_dir / "depth"
        self.lr_dir = self.split_dir / f"depth_lr_{lr_kind}"
        self.rgb_dir = self.split_dir / "image"
        self.mask_dir = self.split_dir / "mask"
        self.train = train
        self.patch_size = patch_size
        self.scale = scale
        self.val_hr_size = val_hr_size  # resize val to this for VRAM

        self.samples = sorted(
            p.stem for p in self.hr_dir.glob("*.png")
            if (self.lr_dir / p.name).exists()
            and (self.rgb_dir / p.name).exists()
        )
        if not self.samples:
            raise RuntimeError(f"No samples found in {self.split_dir}")

        self.has_mask = self.mask_dir.exists() and any(self.mask_dir.glob("*.png"))
        mode = f"patch={patch_size}" if train else f"val@{val_hr_size}"
        print(f"  [{split}] {len(self.samples)} samples, {mode}")

    def __len__(self):
        return len(self.samples)

    def _load_depth(self, path):
        with Image.open(path) as img:
            arr = np.array(img, dtype=np.float32)
        return arr / 65535.0 if arr.max() > 255 else arr / 255.0

    def __getitem__(self, idx):
        name = self.samples[idx]

        # HR depth (1024×1024)
        hr = self._load_depth(self.hr_dir / f"{name}.png")
        # LR depth (256×256)
        lr = self._load_depth(self.lr_dir / f"{name}.png")
        # RGB (1024×1024)
        with Image.open(self.rgb_dir / f"{name}.png") as img:
            rgb = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
        # Mask (1024×1024)
        if self.has_mask:
            mp = self.mask_dir / f"{name}.png"
            if mp.exists():
                with Image.open(mp) as img:
                    mask = (np.array(img.convert("L")) > 127).astype(np.float32)
            else:
                mask = np.ones(hr.shape[:2], dtype=np.float32)
        else:
            mask = np.ones(hr.shape[:2], dtype=np.float32)

        if self.train:
            # Random crop: HR patch_size × patch_size, LR patch_size/scale × patch_size/scale
            ps = self.patch_size
            lps = ps // self.scale
            H, W = hr.shape[:2]
            # Align crop to scale factor
            ty = np.random.randint(0, H - ps + 1)
            tx = np.random.randint(0, W - ps + 1)
            ty = (ty // self.scale) * self.scale
            tx = (tx // self.scale) * self.scale

            hr = hr[ty:ty+ps, tx:tx+ps]
            rgb = rgb[ty:ty+ps, tx:tx+ps]
            mask = mask[ty:ty+ps, tx:tx+ps]
            lr_ty, lr_tx = ty // self.scale, tx // self.scale
            lr = lr[lr_ty:lr_ty+lps, lr_tx:lr_tx+lps]

            # Random flip
            if np.random.random() < 0.5:
                hr = hr[:, ::-1].copy()
                lr = lr[:, ::-1].copy()
                rgb = rgb[:, ::-1].copy()
                mask = mask[:, ::-1].copy()
            if np.random.random() < 0.5:
                hr = hr[::-1].copy()
                lr = lr[::-1].copy()
                rgb = rgb[::-1].copy()
                mask = mask[::-1].copy()
        else:
            # Val: center-crop to patch_size (same as train) to avoid
            # NaN at larger resolutions with untrained / early weights.
            ps = self.patch_size
            lps = ps // self.scale
            H, W = hr.shape[:2]
            ty = (H - ps) // 2
            tx = (W - ps) // 2
            ty = (ty // self.scale) * self.scale
            tx = (tx // self.scale) * self.scale

            hr = hr[ty:ty+ps, tx:tx+ps]
            rgb = rgb[ty:ty+ps, tx:tx+ps]
            mask = mask[ty:ty+ps, tx:tx+ps]
            lr_ty, lr_tx = ty // self.scale, tx // self.scale
            lr = lr[lr_ty:lr_ty+lps, lr_tx:lr_tx+lps]

        hr_t = torch.from_numpy(np.ascontiguousarray(hr)).unsqueeze(0)
        lr_t = torch.from_numpy(np.ascontiguousarray(lr)).unsqueeze(0)
        rgb_t = torch.from_numpy(np.ascontiguousarray(rgb).transpose(2, 0, 1))
        mask_t = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)

        return rgb_t, lr_t, hr_t, mask_t


# ============================================================
# Loss (same as our UNet for fair comparison)
# ============================================================

def _masked_l1(pred, target, mask, eps=1e-6):
    diff = (pred - target).abs() * mask
    return diff.sum() / mask.sum().clamp(min=eps)


def gradient_loss(pred, target, mask=None, eps=1e-6):
    dy_p = pred[..., 1:, :] - pred[..., :-1, :]
    dx_p = pred[..., :, 1:] - pred[..., :, :-1]
    dy_t = target[..., 1:, :] - target[..., :-1, :]
    dx_t = target[..., :, 1:] - target[..., :, :-1]
    if mask is None:
        return F.l1_loss(dy_p, dy_t) + F.l1_loss(dx_p, dx_t)
    my = mask[..., 1:, :] * mask[..., :-1, :]
    mx = mask[..., :, 1:] * mask[..., :, :-1]
    ly = ((dy_p - dy_t).abs() * my).sum() / my.sum().clamp(min=eps)
    lx = ((dx_p - dx_t).abs() * mx).sum() / mx.sum().clamp(min=eps)
    return ly + lx


def depth_loss(pred, target, mask=None, w_grad=0.5):
    if mask is None:
        return F.l1_loss(pred, target) + w_grad * gradient_loss(pred, target)
    return _masked_l1(pred, target, mask) + w_grad * gradient_loss(pred, target, mask)


# ============================================================
# Training
# ============================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Import SGNet
    sgnet_dir = Path(args.sgnet_dir)
    if str(sgnet_dir) not in sys.path:
        sys.path.insert(0, str(sgnet_dir))
    from models.SGNet import SGNet
    from models.common import get_Fre, Get_gradient_nopadding_d

    # Dataset
    data_root = Path(args.data_root)
    train_ds = SGNetDataset(data_root, "train", lr_kind=args.lr_input,
                            train=True, patch_size=args.patch_size, scale=4)
    val_ds = SGNetDataset(data_root, "val", lr_kind=args.lr_input,
                          train=False, scale=4, val_hr_size=args.val_size)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    print(f"LR input: depth_lr_{args.lr_input}")
    print(f"Patch size: {args.patch_size} (LR={args.patch_size//4})")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=0, pin_memory=True)

    # Model
    model = SGNet(num_feats=args.num_feats, kernel_size=3, scale=4).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SGNet(num_feats={args.num_feats}, scale=4): {n_params/1e6:.2f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = (device.type == "cuda") and args.amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"AMP: {use_amp}")
    print(f"Gradient accumulation: {args.grad_accum} (effective batch={args.batch_size * args.grad_accum})")

    ckpt_dir = Path(args.checkpoints)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "train_log.json"
    history = []
    best_val_l1 = float("inf")
    start_epoch = 1

    # Resume from checkpoint if available (prefer last.pt over best.pt)
    if args.resume:
        resume_path = None
        if (ckpt_dir / "last.pt").exists():
            resume_path = ckpt_dir / "last.pt"
        elif (ckpt_dir / "best.pt").exists():
            resume_path = ckpt_dir / "best.pt"

        if resume_path:
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            start_epoch = ckpt["epoch"] + 1
            best_val_l1 = ckpt.get("val_l1", float("inf"))
            # Also check best.pt for the actual best val_l1
            if (ckpt_dir / "best.pt").exists() and resume_path != ckpt_dir / "best.pt":
                try:
                    best_ckpt = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
                    best_val_l1 = min(best_val_l1, best_ckpt.get("val_l1", float("inf")))
                except Exception:
                    pass
            # Advance scheduler to correct position
            for _ in range(start_epoch - 1):
                scheduler.step()
            # Load existing history
            if log_path.exists():
                try:
                    history = json.loads(log_path.read_text())
                except Exception:
                    history = []
            print(f"Resumed from {resume_path.name} epoch {ckpt['epoch']} (best_val_l1={best_val_l1:.5f})")

    print(f"\nTraining epochs {start_epoch}-{args.epochs}")
    print(f"Loss: L1 + 0.5*gradient (same as UNet for fair comparison)")
    print(f"Checkpoints: {ckpt_dir}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, (rgb, lr_in, hr_target, mask) in enumerate(train_loader):
            rgb = rgb.to(device, non_blocking=True)
            lr_in = lr_in.to(device, non_blocking=True)
            hr_target = hr_target.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                # SGNet forward: (guidance_rgb, lr_depth) -> (out, out_grad)
                out, out_grad = model((rgb, lr_in))
                out = out.clamp(0, 1)
                dl = depth_loss(out, hr_target, mask=mask)
                loss = dl / args.grad_accum

            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            train_loss += loss.item() * args.grad_accum

        if (step + 1) % args.grad_accum != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        train_loss /= len(train_ds)

        # Val (at reduced resolution to fit VRAM)
        model.eval()
        val_loss = 0.0
        val_l1 = 0.0
        with torch.no_grad():
            for rgb, lr_in, hr_target, mask in val_loader:
                rgb = rgb.to(device, non_blocking=True)
                lr_in = lr_in.to(device, non_blocking=True)
                hr_target = hr_target.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    out, _ = model((rgb, lr_in))
                    out = out.clamp(0, 1)
                    dl = depth_loss(out, hr_target, mask=mask)

                val_loss += dl.item()
                val_l1 += _masked_l1(out.float(), hr_target, mask).item()

        val_loss /= len(val_ds)
        val_l1 /= len(val_ds)
        scheduler.step()

        elapsed = time.time() - t0
        record = {
            "epoch": epoch, "train_loss": float(train_loss),
            "val_loss": float(val_loss), "val_l1": float(val_l1),
            "lr": optimizer.param_groups[0]["lr"], "time_s": elapsed,
        }
        history.append(record)
        print(f"[{epoch:03d}/{args.epochs}] train={train_loss:.5f} val={val_loss:.5f} "
              f"val_l1={val_l1:.5f} lr={record['lr']:.2e} time={elapsed:.0f}s")

        # Atomic save helper (write .tmp then rename — crash-safe)
        def _safe_save(obj, path):
            tmp = str(path) + ".tmp"
            torch.save(obj, tmp)
            os.replace(tmp, path)

        # Save log (atomic)
        tmp_log = str(log_path) + ".tmp"
        with open(tmp_log, "w") as f:
            json.dump(history, f, indent=2)
        os.replace(tmp_log, log_path)

        # Save last.pt every epoch (crash recovery)
        _safe_save({
            "model": model.state_dict(), "epoch": epoch,
            "val_loss": val_loss, "val_l1": val_l1,
            "args": vars(args),
        }, ckpt_dir / "last.pt")

        # Best by val_l1
        if val_l1 < best_val_l1:
            best_val_l1 = val_l1
            _safe_save({
                "model": model.state_dict(), "epoch": epoch,
                "val_loss": val_loss, "val_l1": val_l1,
                "args": vars(args),
            }, ckpt_dir / "best.pt")
            print(f"    -> saved best.pt (val_l1={val_l1:.5f})")

        # Periodic VRAM cleanup to prevent memory fragmentation crash
        if epoch % 10 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"\nDone. Best val_l1: {best_val_l1:.5f}")
    print(f"Checkpoints: {ckpt_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/dataset", type=str)
    p.add_argument("--sgnet_dir", default="external/SGNet", type=str)
    p.add_argument("--lr_input", default="8bit", type=str)
    p.add_argument("--epochs", default=100, type=int)
    p.add_argument("--batch_size", default=4, type=int)
    p.add_argument("--grad_accum", default=2, type=int)
    p.add_argument("--lr", default=1e-4, type=float)
    p.add_argument("--patch_size", default=256, type=int,
                   help="HR patch size for training (LR = patch_size/4)")
    p.add_argument("--val_size", default=512, type=int,
                   help="HR resolution for validation (512 to fit 8GB VRAM)")
    p.add_argument("--num_feats", default=40, type=int,
                   help="SGNet hidden features (40 = original paper default)")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--resume", action="store_true",
                   help="Resume from best.pt in checkpoint dir")
    p.add_argument("--checkpoints", default="checkpoints/baseline_sgnet", type=str)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
