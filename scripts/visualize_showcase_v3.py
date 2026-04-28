#!/usr/bin/env python3
"""
Visualize WHY F-score differs: threshold binary maps + 3D surface relief.

Key insight: 2D depth images look identical, but back-projected surfaces differ.
This script shows:
  Row 1: Depth (inferno) — "looks the same"
  Row 2: Binary threshold map (green=within 1e-3, red=outside) — F-score source
  Row 3: 3D surface mesh rendering with shading — tiny bumps become visible

Usage:
  D:\zmm\miniconda3\envs\facelift\python.exe scripts/visualize_showcase_v3.py --device cuda
"""
import os, sys, argparse
import numpy as np
import cv2; cv2.setNumThreads(0)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'dataset')
CKPT = os.path.join(ROOT, 'checkpoints')
if not os.path.isdir(CKPT):
    CKPT = os.path.join(os.path.dirname(ROOT), 'checkpoints')
OUT = os.path.join(ROOT, 'eval', 'showcase_v3')

TOP_SAMPLES = [
    'ffhq_44764_face',
    'ffhq_48538_face',
    'ffhq_49905_face',
]

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
})

def load_depth(path):
    img = np.asarray(Image.open(path)).astype(np.float32)
    return img / 65535.0 if img.max() > 255 else img / 255.0

def run_model(model, lr, device):
    import torch
    with torch.no_grad():
        x = torch.from_numpy(lr).float().unsqueeze(0).unsqueeze(0).to(device)
        out = model(x)
        if isinstance(out, tuple): out = out[0]
        return out.squeeze().cpu().numpy().clip(0, 1)

def load_models(device):
    import torch, torch.nn as nn, torch.nn.functional as F
    models = {}
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    try:
        from train_depth_upres import DepthUpResUNet
        for name, ckdir, pn in [
            ('UNet', 'depth_upres_8bit_ch32', False),
            ('UNet+DSINE', 'depth_upres_8bit_ch32_normal_w050_dsine', True),
        ]:
            p = os.path.join(CKPT, ckdir, 'best.pt')
            if os.path.exists(p):
                ck = torch.load(p, map_location=device, weights_only=False)
                m = DepthUpResUNet(base_ch=32, predict_normal=pn)
                m.load_state_dict(ck['model'], strict=False)
                models[name] = m.to(device).eval()
                print(f'  loaded {name}')
    except Exception as e:
        print(f'  UNet load failed: {e}')

    class ResBlock(nn.Module):
        def __init__(self, nf, rs=0.1):
            super().__init__()
            self.body = nn.Sequential(nn.Conv2d(nf,nf,3,padding=1), nn.ReLU(True), nn.Conv2d(nf,nf,3,padding=1))
            self.rs = rs
        def forward(self, x): return x + self.body(x) * self.rs
    class EDSR_Light(nn.Module):
        def __init__(self, nf=64, nb=16):
            super().__init__()
            self.head = nn.Conv2d(1, nf, 3, padding=1)
            self.body = nn.Sequential(*[ResBlock(nf) for _ in range(nb)])
            self.body_tail = nn.Conv2d(nf, nf, 3, padding=1)
            self.up = nn.Sequential(nn.Conv2d(nf, nf*4, 3, padding=1), nn.PixelShuffle(2),
                                    nn.Conv2d(nf, nf*4, 3, padding=1), nn.PixelShuffle(2))
            self.tail = nn.Conv2d(nf, 1, 3, padding=1)
        def forward(self, x):
            h = self.head(x); h = h + self.body_tail(self.body(h)); h = self.up(h)
            bic = F.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False).clamp(0,1)
            return (bic + self.tail(h) * 0.5).clamp(0,1)
    class ResBlockBN(nn.Module):
        def __init__(self, nf):
            super().__init__()
            self.body = nn.Sequential(nn.Conv2d(nf,nf,3,padding=1), nn.BatchNorm2d(nf), nn.PReLU(),
                                      nn.Conv2d(nf,nf,3,padding=1), nn.BatchNorm2d(nf))
        def forward(self, x): return x + self.body(x)
    class SRResNetM(nn.Module):
        def __init__(self, nf=64, nb=16):
            super().__init__()
            self.head = nn.Sequential(nn.Conv2d(1, nf, 9, padding=4), nn.PReLU())
            self.body = nn.Sequential(*[ResBlockBN(nf) for _ in range(nb)])
            self.body_tail = nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1), nn.BatchNorm2d(nf))
            self.up = nn.Sequential(nn.Conv2d(nf, nf*4, 3, padding=1), nn.PixelShuffle(2), nn.PReLU(),
                                    nn.Conv2d(nf, nf*4, 3, padding=1), nn.PixelShuffle(2), nn.PReLU())
            self.tail = nn.Conv2d(nf, 1, 9, padding=4)
        def forward(self, x):
            h = self.head(x); h = h + self.body_tail(self.body(h)); h = self.up(h)
            bic = F.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False).clamp(0,1)
            return (bic + self.tail(h) * 0.5).clamp(0,1)
    for cls, name, ckdir in [(EDSR_Light, 'EDSR', 'baseline_edsr'),
                              (SRResNetM, 'SRResNet', 'baseline_srresnet')]:
        p = os.path.join(CKPT, ckdir, 'best.pt')
        if os.path.exists(p):
            try:
                ck = torch.load(p, map_location=device, weights_only=False)
                m = cls(); m.load_state_dict(ck['model'], strict=False)
                models[name] = m.to(device).eval()
                print(f'  loaded {name}')
            except Exception as e:
                print(f'  {name} failed: {e}')

    # SGNet (RGB-guided, trained with num_feats=24)
    sgnet_dir = os.path.join(ROOT, 'external', 'SGNet')
    if not os.path.isdir(sgnet_dir):
        sgnet_dir = os.path.join(os.path.dirname(ROOT), 'external', 'SGNet')
    sgnet_ck = os.path.join(CKPT, 'baseline_sgnet', 'best.pt')
    if os.path.exists(sgnet_ck) and os.path.isdir(sgnet_dir):
        try:
            sys.path.insert(0, sgnet_dir)
            from models.SGNet import SGNet as SGNetCls
            ck = torch.load(sgnet_ck, map_location=device, weights_only=False)
            nf = ck.get('args', {}).get('num_feats', 24) if isinstance(ck.get('args'), dict) else 24
            m = SGNetCls(num_feats=nf, kernel_size=3, scale=4)
            m.load_state_dict(ck['model'], strict=False)
            models['SGNet'] = m.to(device).eval()
            print(f'  loaded SGNet (num_feats={nf})')
        except Exception as e:
            print(f'  SGNet failed: {e}')

    return models


def shaded_surface(depth_crop, light_dir=(0.3, 0.3, 1.0)):
    """Simple Lambertian shading from depth. 
    Tiny depth bumps become visible under oblique lighting."""
    dy = np.gradient(depth_crop, axis=0)
    dx = np.gradient(depth_crop, axis=1)
    scale = 200.0  # exaggerate surface relief
    nx, ny, nz = -dx*scale, -dy*scale, np.ones_like(depth_crop)
    norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-8
    nx, ny, nz = nx/norm, ny/norm, nz/norm
    lx, ly, lz = light_dir
    ln = np.sqrt(lx**2 + ly**2 + lz**2)
    lx, ly, lz = lx/ln, ly/ln, lz/ln
    shade = np.clip(nx*lx + ny*ly + nz*lz, 0, 1)
    return shade


def make_vis(sname, models, device, split='val'):
    hr = load_depth(os.path.join(DATA, split, 'depth', f'{sname}.png'))
    lr = load_depth(os.path.join(DATA, split, 'depth_lr_8bit', f'{sname}.png'))
    mask_path = os.path.join(DATA, split, 'mask', f'{sname}.png')
    mask = None
    if os.path.exists(mask_path):
        mask = (np.asarray(Image.open(mask_path)) > 127).astype(np.float32)
    
    bic = cv2.resize(lr.astype(np.float32), (1024,1024), interpolation=cv2.INTER_CUBIC).clip(0,1)

    preds = [('Bicubic', bic)]
    for name in ['EDSR', 'SRResNet', 'SGNet', 'UNet', 'UNet+DSINE']:
        if name in models:
            if name == 'SGNet':
                import torch
                rgb_path = os.path.join(DATA, split, 'image', f'{sname}.png')
                if os.path.exists(rgb_path):
                    rgb = np.array(Image.open(rgb_path).convert('RGB')).astype(np.float32) / 255.0
                    rgb_t = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).float().to(device)
                    lr_t = torch.from_numpy(lr).float().unsqueeze(0).unsqueeze(0).to(device)
                    with torch.no_grad():
                        out = models['SGNet'](lr_t, rgb_t)
                    preds.append(('SGNet', out.squeeze().cpu().numpy().clip(0, 1)))
            else:
                preds.append((name, run_model(models[name], lr, device)))
    preds.append(('GT', hr))
    n = len(preds)

    fg = hr[hr > 0.3]
    vmin, vmax = float(np.percentile(fg, 1)), float(np.percentile(fg, 99))

    # Crop - nose bridge + eye socket area (most geometric detail)
    cy, cx, cs = 490, 500, 110
    crop = (slice(cy-cs, cy+cs), slice(cx-cs, cx+cs))

    # F-score threshold
    threshold = 1e-3

    # ─── Figure: 4 rows ───
    fig = plt.figure(figsize=(n * 2.5, 11))
    
    # Custom green/red colormap for threshold map
    pass_fail_cmap = ListedColormap(['#d32f2f', '#43a047'])  # red=fail, green=pass

    for col, (name, pred) in enumerate(preds):
        is_gt = (name == 'GT')
        fw = 'bold' if name in ('UNet', 'UNet+DSINE', 'GT') else 'normal'
        clr = '#1565C0' if 'UNet' in name else ('green' if is_gt else 'black')

        # Row 1: Depth (inferno)
        ax1 = fig.add_subplot(4, n, col+1)
        d_norm = np.clip((pred - vmin) / (vmax - vmin + 1e-8), 0, 1)
        ax1.imshow(d_norm[crop], cmap='inferno', vmin=0, vmax=1)
        ax1.set_title(name, fontsize=10, fontweight=fw, color=clr)
        ax1.axis('off')
        if col == 0:
            ax1.set_ylabel('Depth', fontsize=9, rotation=90, labelpad=8)
            ax1.yaxis.set_visible(True); ax1.tick_params(left=False, labelleft=False)

        # Row 2: Binary threshold map (green = |err| < 1e-3, red = outside)
        ax2 = fig.add_subplot(4, n, n + col + 1)
        if is_gt:
            # All green (reference)
            ax2.imshow(np.ones_like(hr[crop]), cmap=pass_fail_cmap, vmin=0, vmax=1)
            ax2.text(0.5, 0.5, '100%', transform=ax2.transAxes,
                    ha='center', va='center', fontsize=14, color='white', fontweight='bold')
        else:
            err = np.abs(pred - hr)
            within = (err < threshold).astype(float)
            ax2.imshow(within[crop], cmap=pass_fail_cmap, vmin=0, vmax=1)
            # Compute F-score percentage for this crop
            if mask is not None:
                mask_crop = mask[crop]
                pct = within[crop][mask_crop > 0.5].mean() * 100
            else:
                pct = within[crop].mean() * 100
            ax2.text(0.5, 0.05, f'{pct:.1f}%', transform=ax2.transAxes,
                    ha='center', va='bottom', fontsize=11, color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.7))
        ax2.axis('off')
        if col == 0:
            ax2.set_ylabel(f'|err| < {threshold}', fontsize=9, rotation=90, labelpad=8)
            ax2.yaxis.set_visible(True); ax2.tick_params(left=False, labelleft=False)

        # Row 3: Shaded surface (Lambertian lighting reveals bumps)
        ax3 = fig.add_subplot(4, n, 2*n + col + 1)
        shade = shaded_surface(pred[crop])
        ax3.imshow(shade, cmap='gray', vmin=0.3, vmax=1.0)
        ax3.axis('off')
        if col == 0:
            ax3.set_ylabel('Surface shading', fontsize=9, rotation=90, labelpad=8)
            ax3.yaxis.set_visible(True); ax3.tick_params(left=False, labelleft=False)

        # Row 4: Shaded DIFFERENCE from GT (amplified surface bumps)
        ax4 = fig.add_subplot(4, n, 3*n + col + 1)
        if