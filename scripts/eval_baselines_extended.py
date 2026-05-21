"""
Extended baseline evaluation: zone-aware metrics + mesh quality for EDSR/SRResNet.

Fills the two gaps in the current eval tables:
  1. multimetric_zoneaware.csv — missing EDSR, SRResNet
  2. mesh_quality.csv — missing EDSR, SRResNet, SGNet

Also produces:
  - Per-pixel error maps (sanity check before trusting zone-aware numbers)
  - Depth range alignment report (catches normalization mismatches before mesh eval)

Usage:
    python scripts/eval_baselines_extended.py                    # full run
    python scripts/eval_baselines_extended.py --sanity-only      # error maps only
    python scripts/eval_baselines_extended.py --skip-mesh        # zone-aware only
    python scripts/eval_baselines_extended.py --mesh-limit 10    # quick mesh smoke test
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

cv2.setNumThreads(0)

# ---------------------------------------------------------------------------
# Model definitions (must match training code in baselines.ipynb exactly)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """EDSR residual block (no BN)."""
    def __init__(self, nf, res_scale=0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(nf, nf, 3, padding=1))
        self.res_scale = res_scale
    def forward(self, x):
        return x + self.body(x) * self.res_scale

class EDSR_Light(nn.Module):
    """EDSR-light: 16 blocks, 64 channels, ~1.5M params."""
    def __init__(self, nf=64, nb=16):
        super().__init__()
        self.head = nn.Conv2d(1, nf, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(nf) for _ in range(nb)])
        self.body_tail = nn.Conv2d(nf, nf, 3, padding=1)
        self.up = nn.Sequential(
            nn.Conv2d(nf, nf * 4, 3, padding=1), nn.PixelShuffle(2),
            nn.Conv2d(nf, nf * 4, 3, padding=1), nn.PixelShuffle(2))
        self.tail = nn.Conv2d(nf, 1, 3, padding=1)
    def forward(self, x):
        h = self.head(x)
        h = h + self.body_tail(self.body(h))
        h = self.up(h)
        bic = F.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False).clamp(0, 1)
        return (bic + self.tail(h) * 0.5).clamp(0, 1)

class ResBlockBN(nn.Module):
    """SRResNet residual block (with BN)."""
    def __init__(self, nf):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, 3, padding=1), nn.BatchNorm2d(nf), nn.PReLU(),
            nn.Conv2d(nf, nf, 3, padding=1), nn.BatchNorm2d(nf))
    def forward(self, x):
        return x + self.body(x)

class SRResNet(nn.Module):
    """SRResNet: 16 blocks, 64 channels, ~1.5M params."""
    def __init__(self, nf=64, nb=16):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(1, nf, 9, padding=4), nn.PReLU())
        self.body = nn.Sequential(*[ResBlockBN(nf) for _ in range(nb)])
        self.body_tail = nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1), nn.BatchNorm2d(nf))
        self.up = nn.Sequential(
            nn.Conv2d(nf, nf * 4, 3, padding=1), nn.PixelShuffle(2), nn.PReLU(),
            nn.Conv2d(nf, nf * 4, 3, padding=1), nn.PixelShuffle(2), nn.PReLU())
        self.tail = nn.Conv2d(nf, 1, 9, padding=4)
    def forward(self, x):
        h = self.head(x)
        h = h + self.body_tail(self.body(h))
        h = self.up(h)
        bic = F.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False).clamp(0, 1)
        return (bic + self.tail(h) * 0.5).clamp(0, 1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics_masked(pred_np, gt_np, mask_np):
    """L1 / RMSE / PSNR / SSIM within mask > 0.5 region."""
    from skimage.metrics import structural_similarity as ssim_fn
    m = mask_np > 0.5
    if m.sum() < 100:
        m = np.ones_like(mask_np, dtype=bool)
    p, g = pred_np[m], gt_np[m]
    l1 = np.abs(p - g).mean()
    mse = ((p - g) ** 2).mean()
    rmse = np.sqrt(mse)
    psnr = -10 * np.log10(mse + 1e-12)
    # SSIM on bounding-box crop of mask
    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    ss = ssim_fn(gt_np[y0:y1, x0:x1], pred_np[y0:y1, x0:x1], data_range=1.0)
    return {'L1': l1, 'RMSE': rmse, 'PSNR': psnr, 'SSIM': ss}


# ---------------------------------------------------------------------------
# Mesh quality (Chamfer / Hausdorff / F-score)
# ---------------------------------------------------------------------------

def depth_to_pointcloud(depth_np, mask_np, fx=500.0, subsample=2):
    """Back-project depth to 3D point cloud. fx ~ 1024 * 0.5 / tan(hfov/2)."""
    H, W = depth_np.shape
    cx, cy = W / 2, H / 2
    ys, xs = np.where(mask_np > 0.5)
    # Subsample
    if subsample > 1:
        idx = np.arange(0, len(ys), subsample)
        ys, xs = ys[idx], xs[idx]
    z = depth_np[ys, xs]
    valid = z > 1e-6
    ys, xs, z = ys[valid], xs[valid], z[valid]
    # "close=1" convention: invert so that closer objects have smaller z in world
    z_world = 1.0 - z  # map [0,1] → [1,0], close=1 becomes close=0 in world
    x_world = (xs - cx) / fx * z_world
    y_world = (ys - cy) / fx * z_world
    return np.stack([x_world, y_world, z_world], axis=1).astype(np.float32)


def chamfer_hausdorff_fscore(pc_pred, pc_gt, f_thresholds=(0.001, 0.0005)):
    """Compute Chamfer-L1, Chamfer-L2, Hausdorff, F-score at given thresholds."""
    from scipy.spatial import cKDTree
    tree_gt = cKDTree(pc_gt)
    tree_pred = cKDTree(pc_pred)
    d_pred2gt, _ = tree_gt.query(pc_pred)
    d_gt2pred, _ = tree_pred.query(pc_gt)
    chamfer_l1 = (d_pred2gt.mean() + d_gt2pred.mean()) / 2
    chamfer_l2 = (np.mean(d_pred2gt ** 2) + np.mean(d_gt2pred ** 2)) / 2
    hausdorff = max(d_pred2gt.max(), d_gt2pred.max())
    result = {'chamfer_l1': chamfer_l1, 'chamfer_l2': chamfer_l2, 'hausdorff': hausdorff}
    for tau in f_thresholds:
        prec = (d_pred2gt < tau).mean()
        rec = (d_gt2pred < tau).mean()
        f = 2 * prec * rec / (prec + rec + 1e-12)
        result[f'f_score_{str(tau)[2:]}'] = f
    return result


def compute_smoothness(depth_np, mask_np):
    """Surface smoothness: mean Laplacian magnitude inside mask."""
    lap = cv2.Laplacian(depth_np.astype(np.float64), cv2.CV_64F, ksize=3)
    m = mask_np > 0.5
    if m.sum() < 100:
        return 0.0
    return np.abs(lap[m]).mean()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_val_data(data_root):
    """Load all val sample paths."""
    val_dir = Path(data_root) / 'val'
    hr_dir = val_dir / 'depth'
    lr_dir = val_dir / 'depth_lr_8bit'
    mask_dir = val_dir / 'mask'
    rgb_dir = val_dir / 'image'  # RGB guidance for SGNet
    stems = sorted(p.stem for p in hr_dir.glob('*.png'))
    return stems, hr_dir, lr_dir, mask_dir, rgb_dir


def load_model(name, ckpt_path, device):
    """Load a baseline model by name."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if name == 'EDSR-light':
        model = EDSR_Light(nf=64, nb=16)
    elif name == 'SRResNet':
        model = SRResNet(nf=64, nb=16)
    elif name == 'SGNet':
        import sys
        sys.path.insert(0, str(Path('external/SGNet')))
        from models.SGNet import SGNet as SGNetModel
        model = SGNetModel(num_feats=24, kernel_size=3, scale=4)
    else:
        raise ValueError(f"Unknown model: {name}")
    model.load_state_dict(ckpt['model'])
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded {name}: {n_params:.2f}M params, epoch {ckpt.get('epoch', '?')}")
    return model


def run_inference(model, lr_np, device, rgb_np=None, is_sgnet=False):
    """Run single-image inference. Returns (H, W) float32 in [0, 1].
    For SGNet: rgb_np is (H, W, 3) float32 in [0, 1], lr_np is (H, W) float32.
    """
    lr_t = torch.from_numpy(lr_np).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        if is_sgnet and rgb_np is not None:
            # SGNet has FFT ops — must NOT use autocast (FP16 breaks FFT)
            rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0).to(device)
            out = model((rgb_t, lr_t))
        else:
            with torch.cuda.amp.autocast():
                out = model(lr_t)
        if isinstance(out, tuple):
            out = out[0]
    return out[0, 0].float().cpu().numpy().clip(0, 1)


def sanity_check_error_maps(models_dict, stems, hr_dir, lr_dir, mask_dir, device, out_dir,
                            rgb_dir=None, sgnet_names=None, n_samples=3):
    """
    Visualize per-pixel error maps for face vs background regions.
    Quick sanity check: if face-region error >> background error for SRResNet
    but not for UNet, zone-aware eval will favor us.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    picks = [stems[0], stems[len(stems) // 2], stems[-1]][:n_samples]

    fig, axes = plt.subplots(len(picks), len(models_dict) + 1, figsize=(4 * (len(models_dict) + 1), 4 * len(picks)))
    if len(picks) == 1:
        axes = axes[np.newaxis, :]

    for row, stem in enumerate(picks):
        hr_pil = Image.open(hr_dir / f'{stem}.png')
        hr = np.asarray(hr_pil).astype(np.float32)
        hr = hr / 65535.0 if hr.max() > 255 else hr / 255.0

        lr_pil = Image.open(lr_dir / f'{stem}.png')
        lr = np.asarray(lr_pil).astype(np.float32)
        lr = lr / 255.0 if lr.max() <= 255 else lr / 65535.0

        mp = mask_dir / f'{stem}.png'
        mask = np.array(Image.open(mp).convert('L')).astype(np.float32) / 255.0 if mp.exists() else np.ones_like(hr)

        # Column 0: GT + mask overlay
        axes[row, 0].imshow(hr, cmap='gray', vmin=0, vmax=1)
        axes[row, 0].contour(mask, levels=[0.5], colors='red', linewidths=0.8)
        axes[row, 0].set_title(f'GT + mask\n{stem}', fontsize=8)
        axes[row, 0].axis('off')

        # Load RGB for SGNet if needed
        rgb_np = None
        if sgnet_names and rgb_dir is not None:
            rgb_path = rgb_dir / f'{stem}.png'
            if rgb_path.exists():
                rgb_np = np.array(Image.open(rgb_path).convert('RGB')).astype(np.float32) / 255.0

        for col, (mname, model) in enumerate(models_dict.items(), 1):
            is_sg = sgnet_names and mname in sgnet_names
            pred = run_inference(model, lr, device, rgb_np=rgb_np, is_sgnet=is_sg)
            err = np.abs(pred - hr)

            # Compute face vs background error
            m = mask > 0.5
            face_err = err[m].mean() if m.sum() > 0 else 0
            bg_err = err[~m].mean() if (~m).sum() > 0 else 0

            axes[row, col].imshow(err, cmap='inferno', vmin=0, vmax=0.02)
            axes[row, col].set_title(f'{mname}\nface={face_err:.5f} bg={bg_err:.5f}', fontsize=8)
            axes[row, col].axis('off')

    plt.tight_layout()
    save_path = out_dir / 'sanity_error_maps.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved error maps: {save_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_root', type=str, default='data/dataset')
    parser.add_argument('--sanity-only', action='store_true',
                        help='Only produce error maps, skip full eval')
    parser.add_argument('--skip-mesh', action='store_true',
                        help='Skip mesh quality eval (zone-aware only)')
    parser.add_argument('--mesh-limit', type=int, default=0,
                        help='Limit mesh eval to first N samples (0=all)')
    parser.add_argument('--mesh-subsample', type=int, default=2,
                        help='Point cloud subsample factor')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    stems, hr_dir, lr_dir, mask_dir, rgb_dir = load_val_data(args.data_root)
    print(f"Val samples: {len(stems)}")

    out_dir = Path('eval')
    out_dir.mkdir(exist_ok=True)

    # Discover baseline checkpoints
    baselines = {}
    sgnet_names = set()  # track which baselines need RGB guidance
    edsr_ckpt = Path('checkpoints/baseline_edsr/best.pt')
    srresnet_ckpt = Path('checkpoints/baseline_srresnet/best.pt')
    sgnet_ckpt = Path('checkpoints/baseline_sgnet/best.pt')
    if edsr_ckpt.exists():
        baselines['EDSR-light'] = load_model('EDSR-light', edsr_ckpt, device)
    if srresnet_ckpt.exists():
        baselines['SRResNet'] = load_model('SRResNet', srresnet_ckpt, device)
    if sgnet_ckpt.exists():
        baselines['SGNet'] = load_model('SGNet', sgnet_ckpt, device)
        sgnet_names.add('SGNet')

    if not baselines:
        print("ERROR: No baseline checkpoints found.")
        return

    # Also load UNet variants for comparison in error maps
    import sys
    sys.path.insert(0, 'scripts')
    from train_depth_upres import DepthUpResUNet

    unet_models = {}
    for tag, ckpt_path, predict_normal in [
        ('UNet-8bit', 'checkpoints/depth_upres_8bit_ch32/best.pt', False),
        ('UNet-8bit+DSINE', 'checkpoints/depth_upres_8bit_ch32_normal_w050_dsine/best.pt', True),
    ]:
        p = Path(ckpt_path)
        if p.exists():
            ckpt = torch.load(p, map_location=device, weights_only=False)
            m = DepthUpResUNet(base_ch=32, predict_normal=predict_normal).to(device).eval()
            m.load_state_dict(ckpt['model'], strict=False)
            unet_models[tag] = m
            print(f"  Loaded {tag} (epoch {ckpt.get('epoch', '?')})")

    # ---- Step 0: Sanity check error maps ----
    print("\n--- Sanity check: per-pixel error maps ---")
    all_models = {**unet_models, **baselines}
    sanity_check_error_maps(all_models, stems, hr_dir, lr_dir, mask_dir, device, out_dir,
                            rgb_dir=rgb_dir, sgnet_names=sgnet_names)

    if args.sanity_only:
        print("\n--sanity-only: stopping here.")
        return

    # ---- Step 1: Depth range alignment report ----
    print("\n--- Depth range alignment (3 samples) ---")
    for stem in [stems[0], stems[len(stems) // 2], stems[-1]]:
        hr_pil = Image.open(hr_dir / f'{stem}.png')
        hr = np.asarray(hr_pil).astype(np.float32)
        hr = hr / 65535.0 if hr.max() > 255 else hr / 255.0

        lr_pil = Image.open(lr_dir / f'{stem}.png')
        lr = np.asarray(lr_pil).astype(np.float32)
        lr = lr / 255.0 if lr.max() <= 255 else lr / 65535.0

        # Load RGB for SGNet
        rgb_np = None
        rgb_path = rgb_dir / f'{stem}.png'
        if rgb_path.exists():
            rgb_np = np.array(Image.open(rgb_path).convert('RGB')).astype(np.float32) / 255.0

        ranges = [f"GT=[{hr.min():.4f},{hr.max():.4f}]"]
        for mname, model in all_models.items():
            is_sg = mname in sgnet_names
            pred = run_inference(model, lr, device, rgb_np=rgb_np, is_sgnet=is_sg)
            ranges.append(f"{mname}=[{pred.min():.4f},{pred.max():.4f}]")
        print(f"  {stem}: {', '.join(ranges)}")

    # ---- Step 2: Zone-aware multi-metric eval ----
    print("\n--- Zone-aware eval (L1 / RMSE / PSNR / SSIM) ---")
    results = {name: [] for name in baselines}
    t0 = time.time()
    for i, stem in enumerate(stems):
        hr_pil = Image.open(hr_dir / f'{stem}.png')
        hr = np.asarray(hr_pil).astype(np.float32)
        hr = hr / 65535.0 if hr.max() > 255 else hr / 255.0

        lr_pil = Image.open(lr_dir / f'{stem}.png')
        lr = np.asarray(lr_pil).astype(np.float32)
        lr = lr / 255.0 if lr.max() <= 255 else lr / 65535.0

        mp = mask_dir / f'{stem}.png'
        mask = np.array(Image.open(mp).convert('L')).astype(np.float32) / 255.0 if mp.exists() else np.ones_like(hr)

        # Load RGB for SGNet
        rgb_np = None
        if sgnet_names:
            rgb_path = rgb_dir / f'{stem}.png'
            if rgb_path.exists():
                rgb_np = np.array(Image.open(rgb_path).convert('RGB')).astype(np.float32) / 255.0

        for mname, model in baselines.items():
            is_sg = mname in sgnet_names
            pred = run_inference(model, lr, device, rgb_np=rgb_np, is_sgnet=is_sg)
            metrics = compute_metrics_masked(pred, hr, mask)
            results[mname].append(metrics)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(stems)} samples...")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s")

    # Average metrics
    print("\n  Zone-aware results:")
    new_rows = []
    for mname, metrics_list in results.items():
        avg = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
        avg['method'] = mname
        new_rows.append(avg)
        print(f"    {mname}: L1={avg['L1']:.6f}  PSNR={avg['PSNR']:.2f}  SSIM={avg['SSIM']:.4f}")

    # Merge into existing multimetric_zoneaware.csv
    import pandas as pd
    csv_path = out_dir / 'multimetric_zoneaware.csv'
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        # Remove old entries for these methods if present
        df_existing = df_existing[~df_existing['method'].isin([r['method'] for r in new_rows])]
        df_new = pd.DataFrame(new_rows)
        df_merged = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_merged = pd.DataFrame(new_rows)
    df_merged.to_csv(csv_path, index=False)
    print(f"  Updated {csv_path} ({len(df_merged)} rows)")

    # ---- Step 3: Mesh quality eval ----
    if args.skip_mesh:
        print("\n--skip-mesh: skipping mesh quality eval.")
        for m in list(baselines.values()) + list(unet_models.values()):
            del m
        torch.cuda.empty_cache()
        return

    print("\n--- Mesh quality eval (Chamfer / Hausdorff / F-score) ---")
    limit = args.mesh_limit if args.mesh_limit > 0 else len(stems)
    eval_stems = stems[:limit]
    print(f"  Evaluating {len(eval_stems)} samples (subsample={args.mesh_subsample})")

    mesh_results = {name: [] for name in baselines}
    t0 = time.time()
    for i, stem in enumerate(eval_stems):
        hr_pil = Image.open(hr_dir / f'{stem}.png')
        hr = np.asarray(hr_pil).astype(np.float32)
        hr = hr / 65535.0 if hr.max() > 255 else hr / 255.0

        lr_pil = Image.open(lr_dir / f'{stem}.png')
        lr = np.asarray(lr_pil).astype(np.float32)
        lr = lr / 255.0 if lr.max() <= 255 else lr / 65535.0

        mp = mask_dir / f'{stem}.png'
        mask = np.array(Image.open(mp).convert('L')).astype(np.float32) / 255.0 if mp.exists() else np.ones_like(hr)

        # GT point cloud
        pc_gt = depth_to_pointcloud(hr, mask, subsample=args.mesh_subsample)
        if len(pc_gt) < 100:
            continue

        # Load RGB for SGNet
        rgb_np = None
        if sgnet_names:
            rgb_path = rgb_dir / f'{stem}.png'
            if rgb_path.exists():
                rgb_np = np.array(Image.open(rgb_path).convert('RGB')).astype(np.float32) / 255.0

        for mname, model in baselines.items():
            is_sg = mname in sgnet_names
            pred = run_inference(model, lr, device, rgb_np=rgb_np, is_sgnet=is_sg)
            pc_pred = depth_to_pointcloud(pred, mask, subsample=args.mesh_subsample)
            if len(pc_pred) < 100:
                continue
            metrics = chamfer_hausdorff_fscore(pc_pred, pc_gt,
                                               f_thresholds=(0.001, 0.0005))
            metrics['smoothness'] = compute_smoothness(pred, mask)
            mesh_results[mname].append(metrics)

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(eval_stems)} samples...")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s")

    # Average mesh metrics
    print("\n  Mesh quality results:")
    mesh_rows = []
    for mname, metrics_list in mesh_results.items():
        if not metrics_list:
            continue
        avg = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
        avg['method'] = mname
        mesh_rows.append(avg)
        print(f"    {mname}: Chamfer-L1={avg['chamfer_l1']:.6f}  "
              f"HD={avg['hausdorff']:.5f}  "
              f"F@0.001={avg['f_score_001']:.4f}  "
              f"F@0.0005={avg['f_score_0005']:.4f}")

    # Merge into existing mesh_quality.csv
    mesh_csv = out_dir / 'mesh_quality.csv'
    if mesh_csv.exists():
        df_existing = pd.read_csv(mesh_csv)
        df_existing = df_existing[~df_existing['method'].isin([r['method'] for r in mesh_rows])]
        df_new = pd.DataFrame(mesh_rows)
        df_merged = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_merged = pd.DataFrame(mesh_rows)
    df_merged.to_csv(mesh_csv, index=False)
    print(f"  Updated {mesh_csv} ({len(df_merged)} rows)")

    for m in list(baselines.values()) + list(unet_models.values()):
        del m
    torch.cuda.empty_cache()
    print("\nAll done.")


if __name__ == '__main__':
    main()
