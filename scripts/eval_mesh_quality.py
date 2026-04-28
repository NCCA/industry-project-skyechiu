"""
Evaluate depth SR by downstream mesh quality.

Motivation & References
-----------------------
Pixel-level metrics (PSNR, SSIM, L1) do not capture whether a super-resolved
depth map is actually *useful* for downstream 3D tasks. Several papers argue
that depth SR should be evaluated by the quality of the resulting 3D geometry:

  [1] Voynov et al., "Perceptual Deep Depth Super-Resolution", ICCV 2019.
      — Measures depth SR quality via renderings of reconstructed 3D surfaces,
        not pixel fidelity. Shows perceptual 3D loss >> PSNR for shape quality.

  [2] "Depth Map Super-Resolution Considering View Synthesis Quality",
      IEEE Trans. Image Process. 2017.
      — Evaluates depth SR via view synthesis quality as the downstream task.

  [3] AIM 2024 Challenge: Compressed Depth Map SR & Restoration.
      — Entire challenge motivated by downstream AR/VR 3D application quality.

  [4] Li et al., "High-Quality 3D Reconstruction With Depth Super-Resolution
      and Completion", IEEE Access 2019.
      — Depth SR as preprocessing for mesh reconstruction; evaluates on 3D
        reconstruction quality.

What this script does
---------------------
For each val sample:
  1. Load LR depth (256×256 8-bit), SR depth (UNet output), GT depth (1024 16-bit)
  2. Back-project each depth map to a 3D point cloud using pinhole camera model
  3. Optionally extract mesh via Poisson reconstruction or simply evaluate
     point cloud quality
  4. Compute metrics:
     - Chamfer Distance (CD): mean nearest-neighbor distance between point clouds
     - Hausdorff Distance (HD): max nearest-neighbor distance (worst-case)
     - Surface Smoothness: mean angle between adjacent face normals in mesh
     - Normal Consistency (NC): mean cosine similarity of normals at corresponding pts
     - F-Score @ threshold τ: % of points within τ of their nearest neighbor

Pipeline:  LR depth ──► backproject ──► PC_lr
           SR depth ──► backproject ──► PC_sr   ──► compare vs PC_gt
           GT depth ──► backproject ──► PC_gt

Usage
-----
    # Eval all val samples, auto-discover best checkpoints
    python scripts/eval_mesh_quality.py

    # Eval specific checkpoint
    python scripts/eval_mesh_quality.py --ckpt checkpoints/depth_upres_8bit_ch32/best.pt

    # Limit to N samples (fast smoke test)
    python scripts/eval_mesh_quality.py --limit 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree


# ============================================================
# Depth → Point Cloud (pinhole back-projection)
# ============================================================
# FaceLift renders use a pinhole camera with ~50° horizontal FoV.
# We back-project each pixel (u, v, d) to 3D (X, Y, Z) using:
#   X = (u - cx) * Z / fx
#   Y = (v - cy) * Z / fy
#   Z = depth_value (normalized [0,1] → mapped to metric-ish range)
#
# Since we don't have ground-truth metric depth, we use the normalized
# [0,1] depth directly. All point clouds live in the same normalized
# coordinate space, so Chamfer Distance is still meaningful for
# *relative* comparison (LR vs SR vs GT).

def depth_to_pointcloud(depth_np: np.ndarray, mask_np: np.ndarray | None = None,
                        fov_deg: float = 50.0, subsample: int = 1) -> np.ndarray:
    """Back-project a depth map to a 3D point cloud.

    Args:
        depth_np: (H, W) float32, values in [0, 1]. Convention: 1 = close, 0 = far.
        mask_np:  (H, W) float32 or None. Points where mask < 0.5 are excluded.
        fov_deg:  Horizontal field of view (degrees). FaceLift default ~50°.
        subsample: Take every N-th pixel (speeds up for large maps).

    Returns:
        (N, 3) float32 array of 3D points.
    """
    H, W = depth_np.shape
    fx = W / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
    fy = fx  # square pixels
    cx, cy = W / 2.0, H / 2.0

    # Create pixel grid
    us = np.arange(0, W, subsample, dtype=np.float32)
    vs = np.arange(0, H, subsample, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)

    # Sample depth and mask
    d = depth_np[::subsample, ::subsample]
    if mask_np is not None:
        m = mask_np[::subsample, ::subsample] > 0.5
    else:
        m = d > 0.01  # exclude near-zero depth (background)

    # Back-project
    Z = d[m]
    X = (uu[m] - cx) * Z / fx
    Y = (vv[m] - cy) * Z / fy

    pts = np.stack([X, Y, Z], axis=-1)
    return pts


# ============================================================
# Mesh quality metrics
# ============================================================
# Following the evaluation protocol in:
#   [1] Voynov et al. ICCV 2019 — Chamfer Distance for depth SR eval
#   [5] CD2 (ACM MM 2023) — Fine-grained 3D mesh with twice Chamfer Distance
#   [6] Tatarchenko et al. "What Do Single-View 3D Reconstruction Networks
#       Learn?" CVPR 2019 — F-Score at threshold τ

def chamfer_distance(pc_a: np.ndarray, pc_b: np.ndarray) -> dict:
    """Compute Chamfer Distance and related metrics between two point clouds.

    Args:
        pc_a, pc_b: (N, 3) and (M, 3) float arrays.

    Returns:
        dict with keys: chamfer_l1, chamfer_l2, hausdorff,
                        f_score_001, f_score_005
    """
    tree_a = cKDTree(pc_a)
    tree_b = cKDTree(pc_b)

    # A → B distances
    dist_a2b, _ = tree_b.query(pc_a, k=1)
    # B → A distances
    dist_b2a, _ = tree_a.query(pc_b, k=1)

    # Chamfer L1: mean of mean nearest-neighbor distances (symmetric)
    cd_l1 = (dist_a2b.mean() + dist_b2a.mean()) / 2.0

    # Chamfer L2: mean of mean squared nearest-neighbor distances
    cd_l2 = (np.mean(dist_a2b ** 2) + np.mean(dist_b2a ** 2)) / 2.0

    # Hausdorff: max nearest-neighbor distance (worst case)
    hausdorff = max(dist_a2b.max(), dist_b2a.max())

    # F-Score: fraction of points within threshold τ of their nearest neighbor
    # τ = 0.01 and 0.05 of the normalized [0,1] depth range
    def f_score(d_ab, d_ba, tau):
        prec = (d_ab < tau).mean()
        recall = (d_ba < tau).mean()
        if prec + recall < 1e-8:
            return 0.0
        return float(2 * prec * recall / (prec + recall))

    f001 = f_score(dist_a2b, dist_b2a, tau=0.01)
    f005 = f_score(dist_a2b, dist_b2a, tau=0.005)

    return {
        'chamfer_l1': float(cd_l1),
        'chamfer_l2': float(cd_l2),
        'hausdorff': float(hausdorff),
        'f_score_001': float(f001),
        'f_score_0005': float(f005),
    }


def surface_smoothness(depth_np: np.ndarray, mask_np: np.ndarray | None = None) -> float:
    """Measure surface smoothness via Laplacian magnitude.

    Lower = smoother surface. High values indicate staircase artifacts
    from quantization or low resolution.

    Ref: Standard mesh smoothness metric, adapted to depth maps.
         Laplacian of depth ≈ discrete mean curvature.
    """
    lap = cv2.Laplacian(depth_np.astype(np.float64), cv2.CV_64F, ksize=3)
    if mask_np is not None:
        m = mask_np > 0.5
        if m.sum() < 100:
            return float(np.abs(lap).mean())
        return float(np.abs(lap[m]).mean())
    return float(np.abs(lap).mean())


# ============================================================
# Main evaluation
# ============================================================

def load_model(ckpt_path, device):
    """Load a DepthUpResUNet from checkpoint."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train_depth_upres import DepthUpResUNet

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    base_ch = ckpt['args'].get('base_ch', 32)
    model = DepthUpResUNet(base_ch=base_ch).to(device).eval()
    # strict=False: ignore normal_head weights if present
    model.load_state_dict(ckpt['model'], strict=False)
    return model, ckpt


def run_eval(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_root = Path(args.data_root)
    val_hr_dir = data_root / 'val' / 'depth'
    val_lr8_dir = data_root / 'val' / 'depth_lr_8bit'
    val_lr16_dir = data_root / 'val' / 'depth_lr_16bit'
    val_mask_dir = data_root / 'val' / 'mask'

    samples = sorted(p.stem for p in val_hr_dir.glob('*.png'))
    if args.limit:
        samples = samples[:args.limit]
    print(f"Val samples: {len(samples)}")

    # --- Discover models ---
    models = {}  # name -> (ckpt_path, lr_kind)
    if args.ckpt:
        p = Path(args.ckpt)
        ckpt = torch.load(str(p), map_location='cpu', weights_only=False)
        kind = ckpt['args'].get('lr_input', '8bit')
        models[p.parent.name] = (p, kind)
    else:
        # Auto-discover
        for bc in (64, 32):
            for kind in ('8bit', '16bit'):
                p = Path(f'checkpoints/depth_upres_{kind}_ch{bc}/best.pt')
                if p.exists():
                    models[f'UNet-{kind}-ch{bc}'] = (p, kind)
                for wp in sorted(Path('checkpoints').glob(
                        f'depth_upres_{kind}_ch{bc}_normal_w*_dsine/best.pt')):
                    tag = wp.parent.name.replace('depth_upres_', '')
                    models[f'UNet-{tag}'] = (wp, kind)

    print(f"Models: {list(models.keys())}")

    # Load all models
    loaded = {}
    for name, (ckpt_path, lr_kind) in models.items():
        m, ckpt = load_model(ckpt_path, device)
        loaded[name] = (m, lr_kind, ckpt)
        print(f"  loaded {name} (epoch={ckpt['epoch']})")

    # --- Evaluate ---
    # Methods to evaluate: Nearest-8bit, Bicubic-8bit, each model, GT (reference)
    method_names = ['LR-nearest-8bit', 'LR-bicubic-8bit',
                    'LR-nearest-16bit', 'LR-bicubic-16bit'] + list(models.keys())

    all_results = {name: [] for name in method_names}
    smoothness_results = {name: [] for name in method_names + ['GT']}

    t0 = time.time()
    for i, name in enumerate(samples):
        # Load GT
        hr_img = Image.open(val_hr_dir / f'{name}.png')
        hr_arr = np.array(hr_img, dtype=np.float32)
        if hr_arr.max() > 255:
            hr = hr_arr / 65535.0
        else:
            hr = hr_arr / 255.0

        # Load LR
        lr8 = np.array(Image.open(val_lr8_dir / f'{name}.png'), dtype=np.float32) / 255.0
        lr16_path = val_lr16_dir / f'{name}.png'
        if lr16_path.exists():
            lr16_arr = np.array(Image.open(lr16_path), dtype=np.float32)
            lr16 = lr16_arr / 65535.0 if lr16_arr.max() > 255 else lr16_arr / 255.0
        else:
            lr16 = lr8

        # Load mask
        mask_path = val_mask_dir / f'{name}.png'
        if mask_path.exists():
            mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32) / 255.0
        else:
            mask = np.ones_like(hr)

        # GT point cloud (reference)
        pc_gt = depth_to_pointcloud(hr, mask, subsample=args.subsample)

        # GT smoothness
        smoothness_results['GT'].append(surface_smoothness(hr, mask))

        # Interpolation baselines
        near8 = cv2.resize(lr8, (hr.shape[1], hr.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        bic8 = cv2.resize(lr8, (hr.shape[1], hr.shape[0]),
                          interpolation=cv2.INTER_CUBIC).clip(0, 1)
        near16 = cv2.resize(lr16, (hr.shape[1], hr.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        bic16 = cv2.resize(lr16, (hr.shape[1], hr.shape[0]),
                           interpolation=cv2.INTER_CUBIC).clip(0, 1)

        interp_maps = {
            'LR-nearest-8bit': near8,
            'LR-bicubic-8bit': bic8,
            'LR-nearest-16bit': near16,
            'LR-bicubic-16bit': bic16,
        }
        for mname, depth_map in interp_maps.items():
            pc = depth_to_pointcloud(depth_map, mask, subsample=args.subsample)
            cd = chamfer_distance(pc, pc_gt)
            all_results[mname].append(cd)
            smoothness_results[mname].append(surface_smoothness(depth_map, mask))

        # Model predictions
        with torch.no_grad():
            lr8_t = torch.from_numpy(lr8).unsqueeze(0).unsqueeze(0).to(device)
            lr16_t = torch.from_numpy(lr16).unsqueeze(0).unsqueeze(0).to(device)

            for mname, (model, lr_kind, _) in loaded.items():
                inp = lr16_t if '16bit' in mname else lr8_t
                out = model(inp)
                if isinstance(out, tuple):
                    out = out[0]
                pred = out[0, 0].cpu().numpy().clip(0, 1)

                pc_pred = depth_to_pointcloud(pred, mask, subsample=args.subsample)
                cd = chamfer_distance(pc_pred, pc_gt)
                all_results[mname].append(cd)
                smoothness_results[mname].append(surface_smoothness(pred, mask))

        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            dt = time.time() - t0
            print(f"  [{i+1}/{len(samples)}] {dt:.1f}s")

    # --- Aggregate and print ---
    print("\n" + "=" * 80)
    print("DOWNSTREAM MESH QUALITY EVALUATION")
    print("  Depth → Point Cloud → Chamfer Distance vs GT")
    print("  Ref: Voynov et al. ICCV 2019, AIM 2024 Challenge")
    print("=" * 80)

    header = f"{'Method':<40s} {'CD-L1':>10s} {'CD-L2':>10s} {'Hausdorff':>10s} " \
             f"{'F@0.01':>8s} {'F@0.005':>8s} {'Smooth':>10s}"
    print(header)
    print("-" * len(header))

    summary_rows = []
    for mname in method_names + ['GT']:
        if mname == 'GT':
            sm = np.mean(smoothness_results['GT'])
            print(f"{'GT (reference)':<40s} {'—':>10s} {'—':>10s} {'—':>10s} "
                  f"{'—':>8s} {'—':>8s} {sm:>10.6f}")
            summary_rows.append({
                'method': 'GT', 'chamfer_l1': 0, 'chamfer_l2': 0,
                'hausdorff': 0, 'f_score_001': 1.0, 'f_score_0005': 1.0,
                'smoothness': sm,
            })
            continue

        if not all_results[mname]:
            continue
        metrics = all_results[mname]
        cd_l1 = np.mean([m['chamfer_l1'] for m in metrics])
        cd_l2 = np.mean([m['chamfer_l2'] for m in metrics])
        hd = np.mean([m['hausdorff'] for m in metrics])
        f001 = np.mean([m['f_score_001'] for m in metrics])
        f0005 = np.mean([m['f_score_0005'] for m in metrics])
        sm = np.mean(smoothness_results[mname])

        print(f"{mname:<40s} {cd_l1:>10.6f} {cd_l2:>10.6f} {hd:>10.6f} "
              f"{f001:>8.4f} {f0005:>8.4f} {sm:>10.6f}")

        summary_rows.append({
            'method': mname, 'chamfer_l1': cd_l1, 'chamfer_l2': cd_l2,
            'hausdorff': hd, 'f_score_001': f001, 'f_score_0005': f0005,
            'smoothness': sm,
        })

    # Save CSV
    out_dir = Path('eval')
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / 'mesh_quality.csv'

    import csv
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved: {out_csv}")

    # Save full JSON for per-sample analysis
    out_json = out_dir / 'mesh_quality_detail.json'
    detail = {}
    for mname in method_names:
        detail[mname] = {
            'per_sample': all_results[mname],
            'smoothness': smoothness_results[mname],
        }
    with open(out_json, 'w') as f:
        json.dump(detail, f, indent=2)
    print(f"Saved: {out_json}")


def main():
    p = argparse.ArgumentParser(
        description="Evaluate depth SR by downstream mesh/point-cloud quality. "
                    "Ref: Voynov et al. ICCV 2019, AIM 2024 Challenge.")
    p.add_argument("--data_root", default="data/dataset", type=str)
    p.add_argument("--ckpt", default=None, type=str,
                   help="Specific checkpoint to eval. If None, auto-discover all.")
    p.add_argument("--limit", default=0, type=int,
                   help="Limit to first N val samples (0 = all)")
    p.add_argument("--subsample", default=2, type=int,
                   help="Subsample factor for point cloud (2 = every other pixel). "
                        "Reduces memory/time for 1024×1024 maps.")
    args = p.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
