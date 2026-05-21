#!/usr/bin/env python3
"""
Find the best showcase sample for Fig 1 hero figure.
Goal: sample where OUR method looks best and PixelShuffle methods look worst.

Metric: per-sample Sobel-gradient SSIM against GT.
  gap = gradient_SSIM(UNet, GT) - gradient_SSIM(SGNet, GT)
  Larger gap = more dramatic visual difference in gradient domain.

Usage (on Windows with GPU):
  D:\zmm\miniconda3\envs\facelift\python.exe scripts/find_best_showcase.py --device cuda --top_k 10

Output: prints ranked list + saves eval/showcase_ranking.csv
"""
import os, sys, argparse, json
import numpy as np
import cv2; cv2.setNumThreads(0)
from PIL import Image
from scipy.ndimage import sobel
from skimage.metrics import structural_similarity as ssim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'dataset')
CKPT = os.path.join(ROOT, 'checkpoints')
if not os.path.isdir(CKPT):
    CKPT = os.path.join(os.path.dirname(ROOT), 'checkpoints')

def load_depth(path):
    img = np.asarray(Image.open(path)).astype(np.float32)
    return img / 65535.0 if img.max() > 255 else img / 255.0

def sobel_mag(d):
    return np.sqrt(sobel(d, axis=1)**2 + sobel(d, axis=0)**2)

def gradient_ssim(pred, gt, mask=None):
    """SSIM in Sobel gradient domain within mask region."""
    g_pred = sobel_mag(pred)
    g_gt = sobel_mag(gt)
    if mask is not None:
        # Crop to mask bounding box for speed
        ys, xs = np.where(mask > 0)
        if len(ys) == 0: return 0.0
        y0, y1 = ys.min(), ys.max()+1
        x0, x1 = xs.min(), xs.max()+1
        g_pred = g_pred[y0:y1, x0:x1]
        g_gt = g_gt[y0:y1, x0:x1]
    drange = max(g_gt.max() - g_gt.min(), 1e-8)
    return ssim(g_pred, g_gt, data_range=drange)

def gradient_l1(pred, gt, mask=None):
    """L1 in Sobel gradient domain."""
    g_pred = sobel_mag(pred)
    g_gt = sobel_mag(gt)
    if mask is not None:
        return np.abs(g_pred[mask > 0] - g_gt[mask > 0]).mean()
    return np.abs(g_pred - g_gt).mean()

def pixel_psnr(pred, gt, mask=None):
    if mask is not None:
        mse = ((pred[mask > 0] - gt[mask > 0]) ** 2).mean()
    else:
        mse = ((pred - gt) ** 2).mean()
    if mse < 1e-12: return 60.0
    return 10 * np.log10(1.0 / mse)

def run_model(model, lr_depth, device):
    import torch, torch.nn.functional as F
    with torch.no_grad():
        x = torch.from_numpy(lr_depth).float().unsqueeze(0).unsqueeze(0).to(device)
        out = model(x)
        if isinstance(out, tuple): out = out[0]
        return out.squeeze().cpu().numpy().clip(0, 1)

def load_all_models(device):
    import torch, torch.nn as nn, torch.nn.functional as F
    models = {}
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))

    # UNet variants
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

    # EDSR
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

    # SGNet
    sgnet_dir = os.path.join(ROOT, 'external', 'SGNet')
    if not os.path.isdir(sgnet_dir):
        sgnet_dir = os.path.join(os.path.dirname(ROOT), 'external', 'SGNet')
    sgnet_ck = os.path.join(CKPT, 'baseline_sgnet', 'best.pt')
    print(f'  SGNet dir: {sgnet_dir} (exists={os.path.isdir(sgnet_dir)})')
    print(f'  SGNet ckpt: {sgnet_ck} (exists={os.path.exists(sgnet_ck)})')
    if os.path.exists(sgnet_ck) and os.path.isdir(sgnet_dir):
        try:
            sys.path.insert(0, sgnet_dir)
            from models.SGNet import SGNet as SGNetCls
            ck = torch.load(sgnet_ck, map_location=device, weights_only=False)
            nf = ck.get('args', {}).get('num_feats', 24) if isinstance(ck.get('args'), dict) else 24
            m = SGNetCls(num_feats=nf, kernel_size=3, scale=4)
            m.load_state_dict(ck['model'], strict=False)
            models['SGNet'] = m.to(device).eval()
            print(f'  loaded SGNet')
        except Exception as e:
            print(f'  SGNet failed: {e}')
            import traceback; traceback.print_exc()

    return models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--top_k', type=int, default=15)
    parser.add_argument('--limit', type=int, default=0, help='0 = all samples')
    args = parser.parse_args()

    import torch

    print('Loading models...')
    models = load_all_models(args.device)
    print(f'Models ready: {list(models.keys())}')

    if 'UNet' not in models:
        print('ERROR: UNet not loaded. Cannot rank.')
        return

    # Gather samples
    depth_dir = os.path.join(DATA, args.split, 'depth')
    samples = sorted([f.replace('.png', '') for f in os.listdir(depth_dir) if f.endswith('.png')])
    if args.limit > 0:
        samples = samples[:args.limit]
    print(f'\nScanning {len(samples)} {args.split} samples...\n')

    results = []
    methods = list(models.keys()) + ['Bicubic']

    for idx, sname in enumerate(samples):
        hr_path = os.path.join(DATA, args.split, 'depth', f'{sname}.png')
        lr_path = os.path.join(DATA, args.split, 'depth_lr_8bit', f'{sname}.png')
        mask_path = os.path.join(DATA, args.split, 'mask', f'{sname}.png')

        if not os.path.exists(lr_path):
            continue

        hr = load_depth(hr_path)
        lr = load_depth(lr_path)
        mask = None
        if os.path.exists(mask_path):
            mask = (np.asarray(Image.open(mask_path)) > 127).astype(np.uint8)

        # Bicubic
        bic = cv2.resize(lr.astype(np.float32), (1024, 1024), interpolation=cv2.INTER_CUBIC).clip(0, 1)

        row = {'sample': sname}

        # Compute metrics for each method
        preds = {'Bicubic': bic}
        for mname, model in models.items():
            try:
                if mname == 'SGNet':
                    # SGNet needs RGB guide
                    rgb_path = os.path.join(DATA, args.split, 'image', f'{sname}.png')
                    if os.path.exists(rgb_path):
                        rgb = np.array(Image.open(rgb_path).convert('RGB')).astype(np.float32) / 255.0
                        rgb_t = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).float().to(args.device)
                        lr_t = torch.from_numpy(lr).float().unsqueeze(0).unsqueeze(0).to(args.device)
                        with torch.no_grad():
                            out = model(lr_t, rgb_t)
                        preds[mname] = out.squeeze().cpu().numpy().clip(0, 1)
                    else:
                        continue
                else:
                    preds[mname] = run_model(model, lr, args.device)
            except Exception as e:
                print(f'  {mname} inference failed on {sname}: {e}')
                continue

        for mname, pred in preds.items():
            row[f'{mname}_gSSIM'] = gradient_ssim(pred, hr, mask)
            row[f'{mname}_gL1'] = gradient_l1(pred, hr, mask)
            row[f'{mname}_PSNR'] = pixel_psnr(pred, hr, mask)

        # Compute gaps
        unet_gssim = row.get('UNet_gSSIM', 0)
        for other in ['SGNet', 'SRResNet', 'EDSR']:
            other_gssim = row.get(f'{other}_gSSIM', unet_gssim)
            row[f'gap_UNet_vs_{other}'] = unet_gssim - other_gssim

        # Overall showcase score: average gap over all PixelShuffle methods
        gaps = [row.get(f'gap_UNet_vs_{m}', 0) for m in ['SGNet', 'SRResNet', 'EDSR'] if f'{m}_gSSIM' in row]
        row['showcase_score'] = np.mean(gaps) if gaps else 0

        results.append(row)

        if (idx + 1) % 20 == 0 or idx == len(samples) - 1:
            print(f'  [{idx+1}/{len(samples)}] {sname}: showcase_score={row["showcase_score"]:.4f}')

    # Sort by showcase score (higher = better for us)
    results.sort(key=lambda r: r['showcase_score'], reverse=True)

    # Print top-k
    print(f'\n{"="*80}')
    print(f'TOP {args.top_k} SHOWCASE SAMPLES (UNet gradient SSIM >> PixelShuffle gradient SSIM)')
    print(f'{"="*80}')
    for i, r in enumerate(results[:args.top_k]):
        print(f'\n#{i+1}: {r["sample"]}  (showcase_score = {r["showcase_score"]:.4f})')
        for m in methods:
            gssim = r.get(f'{m}_gSSIM', None)
            psnr = r.get(f'{m}_PSNR', None)
            if gssim is not None:
                print(f'    {m:15s}  gSSIM={gssim:.4f}  PSNR={psnr:.2f} dB')

    