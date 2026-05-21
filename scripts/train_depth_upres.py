"""
Depth Up-Res baseline trainer (UNet, single-channel depth only).

Aligned with Callum's direction:
    Input  : 256x256 depth (8-bit, simulating phone ToF / iPhone Pro depth)
    Output : 1024x1024 depth (16-bit, high precision)
    Goal   : 4x spatial up-res + bit depth recovery (8 -> 16 bit)

This is intentionally depth-only (no RGB, no normal, no opacity) so the
model focuses on the up-res + bit depth recovery problem first.
RGB-guided / multi-modal extensions can come later as v2.

Usage:
    python scripts/train_depth_upres.py
    python scripts/train_depth_upres.py --epochs 100 --batch_size 8 --lr 1e-4
    python scripts/train_depth_upres.py --lr_input 8bit  # or 16bit
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# cv2 on Windows fragments the CPU heap when it repeatedly imread()s 16-bit
# PNGs inside a long training loop (the 2-4 MB allocations end up scattered,
# and after ~1 epoch the allocator can't find a contiguous 2 MB block).
# Disabling cv2's internal thread pool + using PIL for image loading in
# the Dataset (see __getitem__) avoids the leak.
cv2.setNumThreads(0)


# ============================================================
# Dataset
# ============================================================

class DepthUpResDataset(Dataset):
    """
    Returns (lr_depth, hr_depth, mask_hr) triples.

        lr_depth: (1, 256, 256)   float32 in [0,1]
        hr_depth: (1, 1024, 1024) float32 in [0,1]
        mask_hr : (1, 1024, 1024) float32 in {0,1}   -- foreground mask

    The mask is built from data/cropped_faces in render_improve.ipynb
    (cell 6.1) and stored at {split}/mask/{name}.png. If the mask dir
    is missing the dataset falls back to an all-ones mask so training
    still works without it, but a warning is printed once.
    """

    _warned_no_mask = False
    _warned_no_normal = False

    def __init__(self, root: Path, split: str, lr_kind: str = "8bit",
                 load_normal: bool = False, smooth_normal: bool = False,
                 normal_dir_name: str = "normal"):
        assert lr_kind in ("8bit", "16bit")
        self.split_dir = root / split
        self.hr_dir = self.split_dir / "depth"
        self.lr_dir = self.split_dir / f"depth_lr_{lr_kind}"
        self.mask_dir = self.split_dir / "mask"
        # normal_dir_name picks the subfolder the GT normal PNGs live in:
        #   "normal"          -> 3DGS-rendered (noisy, per-splat aliasing)
        #   "normal_dsine"    -> pseudo-GT from DSINE (CVPR 2024) — clean
        #   "normal_omnidata" -> pseudo-GT from Omnidata-v2 — clean
        self.normal_dir = self.split_dir / normal_dir_name
        self.load_normal = load_normal
        self.smooth_normal = smooth_normal
        if not self.hr_dir.exists() or not self.lr_dir.exists():
            raise FileNotFoundError(
                f"Missing {self.hr_dir} or {self.lr_dir}. "
                f"Run reorganize_dataset.py + make_lr_hr_pairs.py first."
            )
        self.has_mask = self.mask_dir.exists() and any(self.mask_dir.glob("*.png"))
        if not self.has_mask and not DepthUpResDataset._warned_no_mask:
            print(f"WARN: no masks found at {self.mask_dir}, falling back to all-ones. "
                  f"Run render_improve.ipynb cell 6.1 to generate them.")
            DepthUpResDataset._warned_no_mask = True
        self.has_normal = (
            load_normal
            and self.normal_dir.exists()
            and any(self.normal_dir.glob("*.png"))
        )
        if load_normal and not self.has_normal and not DepthUpResDataset._warned_no_normal:
            print(f"WARN: no normals found at {self.normal_dir}, falling back to zeros. "
                  f"Normal-aware loss will not contribute.")
            DepthUpResDataset._warned_no_normal = True

        self.samples = sorted(p.stem for p in self.hr_dir.glob("*.png"))
        if not self.samples:
            raise RuntimeError(f"No samples in {self.hr_dir}")
        self.lr_kind = lr_kind

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name = self.samples[idx]
        # HR: 1024x1024, uint16 — use PIL (cv2 on Windows fragments heap).
        # NOTE: PIL opens 16-bit PNGs as mode 'I' (int32) or 'I;16' (uint16)
        # depending on library version. np.array(img) then returns int32 for
        # 'I' mode, which breaks a naive dtype==uint16 check. Use the observed
        # max value to decide the normalization scale (robust across both modes).
        with Image.open(self.hr_dir / f"{name}.png") as img:
            hr = np.asarray(img)
        hr_max = float(hr.max()) if hr.size else 0.0
        # If values exceed 255, the file is 16-bit depth regardless of the
        # numpy dtype we were handed.
        if hr_max > 255.0 or hr.dtype == np.uint16:
            hr_f = hr.astype(np.float32) / 65535.0
        else:
            hr_f = hr.astype(np.float32) / 255.0

        # LR: 256x256 — same heuristic, keeps the 8bit vs 16bit branch consistent.
        with Image.open(self.lr_dir / f"{name}.png") as img:
            lr = np.asarray(img)
        lr_max = float(lr.max()) if lr.size else 0.0
        if self.lr_kind == "16bit" or lr_max > 255.0:
            lr_f = lr.astype(np.float32) / 65535.0
        else:
            lr_f = lr.astype(np.float32) / 255.0

        # HR foreground mask (non-white region from cropped_faces)
        if self.has_mask:
            mp = self.mask_dir / f"{name}.png"
            # Pre-check exists() so missing-mask samples fall back silently.
            # 48-ish samples in postprocessed/ have no cropped_faces source so
            # make_face_masks.py couldn't generate a mask for them.
            if mp.exists():
                with Image.open(mp) as img:
                    m = np.array(img.convert("L"))
            else:
                m = None
            if m is None:
                m_f = np.ones_like(hr_f, dtype=np.float32)
            else:
                if m.shape[:2] != hr_f.shape[:2]:
                    # NEAREST resize via PIL (binary mask safe)
                    with Image.fromarray(m).resize(
                        (hr_f.shape[1], hr_f.shape[0]), Image.NEAREST
                    ) as img2:
                        m = np.array(img2)
                m_f = (m > 127).astype(np.float32)
        else:
            m_f = np.ones_like(hr_f, dtype=np.float32)

        hr_t = torch.from_numpy(hr_f).unsqueeze(0)  # (1, 1024, 1024)
        lr_t = torch.from_numpy(lr_f).unsqueeze(0)  # (1, 256, 256)
        m_t  = torch.from_numpy(m_f).unsqueeze(0)   # (1, 1024, 1024)

        if self.load_normal:
            # GT normal stored as RGB PNG encoding camera-space normal via
            # (n * 0.5 + 0.5) → [0, 255]. Decode with (rgb / 255) * 2 - 1.
            # Shape: (3, H, W), unit vectors per pixel (approximately).
            if self.has_normal:
                np_path = self.normal_dir / f"{name}.png"
                if np_path.exists():
                    with Image.open(np_path) as img:
                        n = np.asarray(img.convert("RGB"))
                    # Optional: bilateral filter on the uint8 normal map BEFORE
                    # decoding. Bilateral is edge-preserving — smooths flat
                    # skin regions (pores, splat-blend noise) while keeping
                    # facial feature edges (nose bridge, eye sockets, lips).
                    # d=11, sigmaColor=40, sigmaSpace=25 is aggressive enough
                    # to remove per-splat stippling but preserves ~eyebrow-
                    # width detail.
                    if self.smooth_normal:
                        # Two passes of stronger bilateral - Single weak pass barely affects
                        # 3DGS stippling; two strong passes remove skin pore noise while
                        # preserving feature edges (nose bridge, eye sockets, etc).
                        # Parameter selection: d=15 = ±7 pixel kernel, σColor=75 treats
                        # normal variations < 75/255 ≈ 0.29 as "similar", while true
                        # feature edge normal jumps typically > 100/255.
                        for _ in range(2):
                            n = cv2.bilateralFilter(n, d=15,
                                                    sigmaColor=75.0,
                                                    sigmaSpace=75.0)
                    n_f = n.astype(np.float32) / 255.0 * 2.0 - 1.0   # [-1, 1]
                    if n_f.shape[:2] != hr_f.shape[:2]:
                        # Nearest resize is wrong for vectors; use bilinear via PIL,
                        # then renormalize. For our data sizes already match.
                        pass
                    # Re-normalize to unit vectors (8-bit quantization + bilateral
                    # may have denormalized).
                    norm = np.linalg.norm(n_f, axis=-1, keepdims=True) + 1e-8
                    n_f = n_f / norm
                    n_t = torch.from_numpy(n_f).permute(2, 0, 1)   # (3, H, W)
                else:
                    n_t = torch.zeros((3, hr_f.shape[0], hr_f.shape[1]), dtype=torch.float32)
            else:
                n_t = torch.zeros((3, hr_f.shape[0], hr_f.shape[1]), dtype=torch.float32)
            return lr_t, hr_t, m_t, n_t

        return lr_t, hr_t, m_t


# ============================================================
# UNet (single-channel in/out, with 4x final upsample)
# ============================================================

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DepthUpResUNet(nn.Module):
    """
    Input  : (B, 1, 256, 256)   single-channel low-res depth
    Output : (B, 1, 1024, 1024) single-channel high-res depth (sigmoid -> [0,1])
             If predict_normal=True, also returns (B, 3, 1024, 1024) unit normals.

    Architecture:
        1) Pre-upsample LR (256 -> 1024) via bicubic
        2) Shared UNet encoder-decoder extracts features
        3) Depth head: 1-ch residual on bicubic (as before)
        4) Normal head (optional): independent 3-ch output, F.normalize per pixel
    Strategy: 'bicubic + residual' is more stable than 'learned upsample'.
    The separate normal head follows GeoNet (CVPR 2018) / NDDepth (ICCV 2023):
    normal is predicted directly, NOT derived from depth gradients, so the
    normal loss gradient does NOT interfere with depth head training.
    """

    def __init__(self, base_ch=32, predict_normal=False):
        super().__init__()
        c = base_ch
        self.predict_normal = predict_normal

        # Encoder
        self.enc1 = conv_block(1, c)         # 1024
        self.enc2 = conv_block(c, c * 2)     # 512
        self.enc3 = conv_block(c * 2, c * 4) # 256
        self.enc4 = conv_block(c * 4, c * 8) # 128

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = conv_block(c * 8, c * 16)

        # Decoder (transposed conv upsamples)
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = conv_block(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = conv_block(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = conv_block(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = conv_block(c * 2, c)

        # Depth head: predict residual on top of bicubic upsample
        self.out_conv = nn.Conv2d(c, 1, 1)

        # Normal head (independent, following GeoNet/NDDepth pattern)
        if predict_normal:
            self.normal_head = nn.Sequential(
                nn.Conv2d(c, c, 3, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, 3, 1),
            )

    def forward(self, lr):
        # Pre-upsample LR (256 -> 1024) as base prediction
        bicubic = F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False)
        bicubic = bicubic.clamp(0, 1)

        # Shared encoder-decoder
        e1 = self.enc1(bicubic)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))

        # Depth output (unchanged)
        residual = torch.tanh(self.out_conv(d1)) * 0.5  # bounded refinement
        depth_out = (bicubic + residual).clamp(0, 1)

        if self.predict_normal:
            # Normal output: independent head, unit-normalized per pixel
            normal_raw = self.normal_head(d1)  # (B, 3, H, W)
            normal_out = F.normalize(normal_raw, dim=1, eps=1e-6)
            return depth_out, normal_out

        return depth_out


# ============================================================
# Loss: L1 + gradient (preserves edges, important for depth)
# ============================================================

def _masked_l1(pred, target, mask, eps=1e-6):
    """Mean absolute error over pixels where mask==1."""
    diff = (pred - target).abs() * mask
    denom = mask.sum().clamp(min=eps)
    return diff.sum() / denom


def gradient_loss(pred, target, mask=None, eps=1e-6):
    dy_p = pred[..., 1:, :] - pred[..., :-1, :]
    dx_p = pred[..., :, 1:] - pred[..., :, :-1]
    dy_t = target[..., 1:, :] - target[..., :-1, :]
    dx_t = target[..., :, 1:] - target[..., :, :-1]
    if mask is None:
        return F.l1_loss(dy_p, dy_t) + F.l1_loss(dx_p, dx_t)
    # A gradient pixel is valid only if BOTH neighbours are in-mask.
    my = mask[..., 1:, :] * mask[..., :-1, :]
    mx = mask[..., :, 1:] * mask[..., :, :-1]
    ly = ((dy_p - dy_t).abs() * my).sum() / my.sum().clamp(min=eps)
    lx = ((dx_p - dx_t).abs() * mx).sum() / mx.sum().clamp(min=eps)
    return ly + lx


def depth_loss(pred, target, mask=None, w_grad=0.5):
    if mask is None:
        return F.l1_loss(pred, target) + w_grad * gradient_loss(pred, target)
    return _masked_l1(pred, target, mask) + w_grad * gradient_loss(pred, target, mask)


def normals_from_depth_torch(depth, fov_deg=50.0):
    """Derive unit-length surface normals from a depth map via finite differences.

    Arguments:
        depth: (B, 1, H, W) float tensor, values in [0, 1] (relative camera z).
        fov_deg: horizontal FoV of the rendering camera. Uses FaceLift's default 50.
    Returns:
        (B, 3, H, W) normals in camera space. Convention matches FaceLift's render
        (camera-space, normalized, with the sign chosen so z < 0 = toward camera,
        which after re-encoding with n*0.5+0.5 maps to the stored GT normal PNG).
    """
    B, _, H, W = depth.shape
    # Finite differences; pad the right / bottom row so shape stays (H, W).
    dz_dx = F.pad(depth[:, :, :, 1:] - depth[:, :, :, :-1], (0, 1, 0, 0))
    dz_dy = F.pad(depth[:, :, 1:, :] - depth[:, :, :-1, :], (0, 0, 0, 1))

    # Camera intrinsics (square pixels, centered principal point).
    fx = W / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
    fy = fx

    # Surface normal from finite differences. Sign convention calibrated by
    # brute-forcing all 8 (sx, sy, sz) flips on 10 samples against the stored
    # GT normal PNGs — (+1, +1, +1) wins with mean cos_sim = +0.77 (39° avg
    # error). The previous (-1, -1, -1) landed at the antipode (-0.77) which
    # is why the first normal-aware run failed to converge.
    nx = dz_dx * fx
    ny = dz_dy * fy
    nz = torch.ones_like(nx)

    normal = torch.cat([nx, ny, nz], dim=1)  # (B, 3, H, W)
    return F.normalize(normal, dim=1, eps=1e-6)


def normal_loss_cos(pred_depth, gt_normal, mask=None, fov_deg=50.0, eps=1e-6):
    """1 - cosine(pred_normal, gt_normal), optionally masked.
    LEGACY: derives normals from predicted depth via finite differences.
    Use normal_loss_cos_direct() with a separate normal head instead."""
    pred_normal = normals_from_depth_torch(pred_depth, fov_deg=fov_deg)   # (B,3,H,W)
    cos = (pred_normal * gt_normal).sum(dim=1)
    loss_map = 1.0 - cos   # in [0, 2]
    if mask is None:
        return loss_map.mean()
    m = mask[:, 0]
    return (loss_map * m).sum() / m.sum().clamp(min=eps)


def normal_loss_cos_direct(pred_normal, gt_normal, mask=None, eps=1e-6):
    """1 - cosine(pred_normal, gt_normal), using directly predicted normals.
    pred_normal: (B, 3, H, W) from the model's normal head (already unit-normalized).
    gt_normal:   (B, 3, H, W) from DSINE pseudo-GT.
    This does NOT derive normals from depth, so gradients flow only to the
    normal head and do not interfere with depth prediction."""
    cos = (pred_normal * gt_normal).sum(dim=1)  # (B, H, W)
    loss_map = 1.0 - cos   # in [0, 2]
    if mask is None:
        return loss_map.mean()
    m = mask[:, 0]
    return (loss_map * m).sum() / m.sum().clamp(min=eps)


# ============================================================
# Training loop
# ============================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    use_normal_loss = getattr(args, "normal_loss_weight", 0.0) > 0
    smooth_n = getattr(args, "smooth_normal", False)
    normal_dir = getattr(args, "normal_dir_name", "normal")
    train_ds = DepthUpResDataset(args.dataset, "train", args.lr_input,
                                 load_normal=use_normal_loss,
                                 smooth_normal=smooth_n,
                                 normal_dir_name=normal_dir)
    val_ds = DepthUpResDataset(args.dataset, "val", args.lr_input,
                               load_normal=use_normal_loss,
                               smooth_normal=smooth_n,
                               normal_dir_name=normal_dir)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    print(f"LR input kind: depth_lr_{args.lr_input}")
    if use_normal_loss:
        print(f"Normal-aware loss: w = {args.normal_loss_weight} "
              f"(cosine sim between pred-depth normals and GT normals) "
              f"from dataset/{{split}}/{normal_dir}/")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = DepthUpResUNet(base_ch=args.base_ch,
                           predict_normal=use_normal_loss).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: DepthUpResUNet(base_ch={args.base_ch}, "
          f"predict_normal={use_normal_loss}), params: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Mixed precision (FP16 autocast) - critical for fitting 1024x1024 in 8GB VRAM
    use_amp = (device.type == "cuda") and args.amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"Mixed precision (AMP): {use_amp}")
    print(f"Gradient accumulation: {args.grad_accum} steps "
          f"(effective batch = {args.batch_size * args.grad_accum})")

    ckpt_dir = Path(args.checkpoints)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "train_log.json"
    history = []
    best_val = float("inf")

    print(f"\nTraining for {args.epochs} epochs (batch={args.batch_size}, lr={args.lr})")
    print(f"Loss: L1 + 0.5 * gradient")

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        t0 = time.time()
        train_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            if use_normal_loss:
                lr_in, hr_target, mask, normal_gt = batch
                normal_gt = normal_gt.to(device, non_blocking=True)
            else:
                lr_in, hr_target, mask = batch
                normal_gt = None
            lr_in = lr_in.to(device, non_blocking=True)
            hr_target = hr_target.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                if use_normal_loss:
                    pred, pred_normal = model(lr_in)
                else:
                    pred = model(lr_in)
                dl = depth_loss(pred, hr_target, mask=mask)
                if use_normal_loss:
                    nl = normal_loss_cos_direct(pred_normal.float(),
                                               normal_gt, mask=mask)
                    step_loss = dl + args.normal_loss_weight * nl
                else:
                    step_loss = dl
                loss = step_loss / args.grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            train_loss += loss.item() * lr_in.size(0) * args.grad_accum
        # Flush remaining gradients
        if (step + 1) % args.grad_accum != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        train_loss /= len(train_ds)

        # --- Val ---
        model.eval()
        val_loss = 0.0
        val_l1 = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if use_normal_loss:
                    lr_in, hr_target, mask, normal_gt = batch
                    normal_gt = normal_gt.to(device, non_blocking=True)
                else:
                    lr_in, hr_target, mask = batch
                    normal_gt = None
                lr_in = lr_in.to(device, non_blocking=True)
                hr_target = hr_target.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    if use_normal_loss:
                        pred, pred_normal = model(lr_in)
                    else:
                        pred = model(lr_in)
                    dl = depth_loss(pred, hr_target, mask=mask)
                    if use_normal_loss:
                        nl = normal_loss_cos_direct(pred_normal.float(),
                                                   normal_gt, mask=mask)
                        vl = dl + args.normal_loss_weight * nl
                    else:
                        vl = dl
                val_loss += vl.item() * lr_in.size(0)
                # Masked val L1: only score the face region (matches training objective)
                val_l1 += _masked_l1(pred.float(), hr_target, mask).item() * lr_in.size(0)
        val_loss /= len(val_ds)
        val_l1 /= len(val_ds)

        scheduler.step()
        elapsed = time.time() - t0
        record = {
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_l1": val_l1, "lr": optimizer.param_groups[0]["lr"], "time_s": elapsed,
        }
        history.append(record)
        print(f"[{epoch:03d}/{args.epochs}] train={train_loss:.5f} val={val_loss:.5f} "
              f"val_l1={val_l1:.5f} lr={record['lr']:.2e} time={elapsed:.1f}s")

        # Save log + best checkpoint
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)
        # Select best checkpoint by val_l1 (pure depth error), NOT val_loss.
        # When normal loss is active, val_loss includes normal term and may
        # decrease while depth accuracy degrades. val_l1 is always honest.
        if val_l1 < best_val:
            best_val = val_l1
            torch.save({
                "model": model.state_dict(), "epoch": epoch,
                "val_loss": val_loss, "val_l1": val_l1, "args": vars(args),
            }, ckpt_dir / "best.pt")
            print(f"    -> saved best.pt (val_l1={val_l1:.5f})")

    # Save final
    torch.save({
        "model": model.state_dict(), "epoch": args.epochs,
        "val_loss": val_loss, "args": vars(args),
    }, ckpt_dir / "last.pt")
    print(f"\nDone. Best val_l1: {best_val:.5f}")
    print(f"Checkpoints: {ckpt_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/dataset", type=Path)
    p.add_argument("--checkpoints", default="checkpoints/depth_upres", type=Path)
    p.add_argument("--lr_input", default="8bit", choices=["8bit", "16bit"],
                   help="Which LR variant to use as input")
    p.add_argument("--epochs", default=100, type=int)
    p.add_argument("--batch_size", default=2, type=int,
                   help="Per-step batch size (use 1-2 for 8GB VRAM at 1024x1024)")
    p.add_argument("--grad_accum", default=4, type=int,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    p.add_argument("--lr", default=1e-4, type=float)
    p.add_argument("--num_workers", default=0, type=int,
                   help="DataLoader worker processes. Default 0 avoids a\n"
                        "Windows cv2+multiprocessing OOM; set to 2-4 on Linux.")
    p.add_argument("--base_ch", default=32, type=int,
                   help="UNet base channel count (24=light, 32=default, 64=heavy)")
    p.add_argument("--amp", action="store_true", default=True,
                   help="Use FP16 mixed precision (default on for CUDA)")
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--normal_loss_weight", default=0.0, type=float,
                   help="Weight on cosine loss between pred-depth normals and GT "
                        "normals (loaded from dataset/{split}/normal/*.png). "
                        "0 disables (default, same as before). 0.1-0.2 is a "
                        "typical ablation value.")
    p.add_argument("--smooth_normal", action="store_true", default=False,
                   help="Apply an edge-preserving bilateral filter to GT normal "
                        "at load time. Removes 3DGS per-splat stippling on flat "
                        "skin regions while preserving facial-feature edges.")
    p.add_argument("--normal_dir_name", default="normal", type=str,
                   help="Subfolder under dataset/{split}/ holding GT normal PNGs. "
                        "'normal' = 3DGS-rendered (noisy). "
                        "'normal_dsine' = pseudo-GT from DSINE (CVPR 2024, clean). "
                        "'normal_omnidata' = pseudo-GT from Omnidata-v2. "
                        "Pre-compute with scripts/compute_dsine_normals.py.")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
                                       