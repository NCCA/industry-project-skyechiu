#!/usr/bin/env python3
"""
Generate visual comparison for top showcase samples.
For each sample: depth crop (inferno) + Sobel gradient (hot) + |error| (magma)
across all methods side by side. Pick the most dramatic one by eye.

Usage:
  D:\zmm\miniconda3\envs\facelift\python.exe scripts/visualize_top_showcase.py --device cuda
"""
import os, sys, argparse
import numpy as np
import cv2; cv2.setNumThreads(0)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import sobel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'dataset')
CKPT = os.path.join(ROOT, 'checkpoints')
if not os.path.isdir(CKPT):
    CKPT = os.path.join(os.path.dirname(ROOT), 'checkpoints')
OUT = os.path.join(ROOT, 'eval', 'showcase_vis')

# Top samples from ranking (val split)
TOP_SAMPLES = [
    'ffhq_44764_face',
    'ffhq_48538_face',
    'ffhq_49905_face',
    'ffhq_51411_face',
    'ffhq_19151_face',
    'ffhq_30316_face',
    'ffhq_00712_face',
]

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
})

def load_depth(path):
    img = np.asarray(Image.open(path)).astype(np.float32)
    return img / 65535.0 if img.max() > 255 else img / 255.0

def sobel_mag(d):
    return np.sqrt(sobel(d, axis=1)**2 + sobel(d, axis=0)**2)

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


def make_vis(sname, models, device, split='val'):
    hr = load_depth(os.path.join(DATA, split, 'depth', f'{sname}.png'))
    lr = load_depth(os.path.join(DATA, split, 'depth_lr_8bit', f'{sname}.png'))
    bic = cv2.resize(lr.astype(np.float32), (1024, 1024), interpolation=cv2.INTER_CUBIC).clip(0, 1)

    fg = hr[hr > 0.3]
    vmin, vmax = float(np.percentile(fg, 1)), float(np.percentile(fg, 99))

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

    # Multiple tight crops on smooth regions where checkerboard is most visible
    crops = {
        'Forehead': (350, 512, 50),   # (cy, cx, half-size) — smooth skin
        'Cheek':    (520, 380, 50),   # left cheek
        'Nose':     (480, 512, 50),   # nose bridge
    }
    # Pick the crop with highest gradient variance ratio (EDSR vs UNet)
    best_crop_name = 'Forehead'
    best_ratio = 0
    bic_pred = bic
    for cname, (cy, cx, cs) in crops.items():
        c = (slice(cy-cs, cy+cs), slice(cx-cs, cx+cs))
        # Check which crop shows most difference
        sg_bic = sobel_mag(bic_pred)[c]
        sg_hr = sobel_mag(hr)[c]
        ratio = np.abs(sg_bic - sg_hr).mean()
        if ratio > best_ratio:
            best_ratio = ratio
            best_crop_name = cname

    cy, cx, cs = crops[best_crop_name]
    crop = (slice(cy-cs, cy+cs), slice(cx-cs, cx+cs))

    # 3 rows: depth / sobel / error×3