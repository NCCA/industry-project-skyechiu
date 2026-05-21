#!/usr/bin/env python3
"""
Step 4: Render depth / RGB / normal / opacity maps from Gaussian Splats.

Matches render_improve.ipynb Cells 1-2.
Uses FaceLift's gslrm CUDA rasterizer to render all four map types.
Saves depth as 16-bit PNG (mode I;16), others as 8-bit.

Requires:
    - FaceLift repo on PYTHONPATH (with gslrm compiled)
    - CUDA GPU

Usage:
    python scripts/export_depth.py
    python scripts/export_depth.py --resolution 1024
    python scripts/export_depth.py --force   # re-render everything
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "pipeline_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Camera utilities (from render_improve.ipynb Cell 1)
# ---------------------------------------------------------------------------

def get_frontal_camera(hfov, w, h, radius, elevation=0, device="cuda:0"):
    """Frontal-view camera (FaceLift convention: azimuth=270 = front face)."""
    fx = w / (2 * np.tan(np.deg2rad(hfov) / 2.0))
    fy = fx
    cx, cy = w / 2.0, h / 2.0
    fxfycxcy = np.array([fx, fy, cx, cy], dtype=np.float32)

    azim = np.deg2rad(270)
    elev = np.deg2rad(elevation)
    z = radius * np.sin(elev)
    base = radius * np.cos(elev)
    x = base * np.cos(azim)
    y = base * np.sin(azim)
    cam_pos = np.array([x, y, z])

    up_vector = np.array([0, 0, 1])
    forward = -cam_pos / np.linalg.norm(cam_pos)
    right = np.cross(forward, up_vector)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    R = np.stack((right, -up, forward), axis=1)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :4] = np.concatenate((R, cam_pos[:, None]), axis=1)

    fxfycxcy_t = torch.from_numpy(fxfycxcy).float().to(device)
    c2w_t = torch.from_numpy(c2w).float().to(device)
    return fxfycxcy_t, c2w_t


# ---------------------------------------------------------------------------
# Rendering (from render_improve.ipynb Cell 1)
# ---------------------------------------------------------------------------

def render_with_custom_colors(pc, height, width, c2w, fxfycxcy, colors_precomp,
                              bg_color=(0.0, 0.0, 0.0)):
    """Render with custom per-Gaussian colors via CUDA rasterizer."""
    from gslrm.model.gaussians_renderer import (
        GaussianRasterizationSettings, GaussianRasterizer, Camera,
    )
    viewpoint_camera = Camera(C2W=c2w, fxfycxcy=fxfycxcy, h=height, w=width)
    bg = torch.tensor(list(bg_color), dtype=torch.float32, device=c2w.device)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.h),
        image_width=int(viewpoint_camera.w),
        tanfovx=viewpoint_camera.tanfovX,
        tanfovy=viewpoint_camera.tanfovY,
        bg=bg,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=0,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    rendered_image, _ = rasterizer(
        means3D=pc.get_xyz,
        means2D=torch.zeros_like(pc.get_xyz[:, :2], requires_grad=False),
        shs=None,
        colors_precomp=colors_precomp,
        opacities=pc.get_opacity,
        scales=pc.get_scaling,
        rotations=pc.get_rotation,
        cov3D_precomp=None,
    )
    return rendered_image  # (3, H, W)


def render_all_maps_gpu(ply_path, fxfycxcy, c2w, width, height, device):
    """Render RGB + Depth + Normal + Opacity from a Gaussian Splat .ply file."""
    from gslrm.model.gaussians_renderer import (
        GaussianModel as GS_GaussianModel,
        Camera, render_opencv_cam,
    )

    pc = GS_GaussianModel(sh_degree=3)
    pc.load_ply(str(ply_path))
    pc = pc.to(device)

    with torch.no_grad():
        # 1) RGB via standard SH rendering
        result_rgb = render_opencv_cam(pc, height, width, c2w, fxfycxcy,
                                       bg_color=(1.0, 1.0, 1.0))
        rgb = result_rgb["render"].detach().cpu().numpy()
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)

        # 2) Opacity: render all-white with black bg
        n_pts = pc.get_xyz.shape[0]
        white = torch.ones(n_pts, 3, dtype=torch.float32, device=device)
        opacity_render = render_with_custom_colors(
            pc, height, width, c2w, fxfycxcy, white, bg_color=(0.0, 0.0, 0.0))
        opacity_map = opacity_render[0].detach().cpu().numpy()
        opacity_out = (opacity_map.clip(0, 1) * 255).astype(np.uint8)

        # 3) Depth: encode camera-space Z as grayscale
        viewpoint_camera = Camera(C2W=c2w, fxfycxcy=fxfycxcy, h=height, w=width)
        w2c = viewpoint_camera.world_view_transform.T
        xyz_world = pc.get_xyz
        xyz_h = torch.cat([xyz_world, torch.ones(n_pts, 1, device=device)], dim=1)
        xyz_cam = (w2c @ xyz_h.T).T[:, :3]
        depths = xyz_cam[:, 2]

        valid_mask = depths > 0.1
        if valid_mask.any():
            d_min = depths[valid_mask].min()
            d_max = depths[valid_mask].max()
            if d_max > d_min:
                depth_norm = 1.0 - (depths - d_min) / (d_max - d_min)
            else:
                depth_norm = torch.zeros_like(depths)
        else:
            depth_norm = torch.zeros_like(depths)
        depth_norm = depth_norm.clamp(0, 1)

        depth_colors = depth_norm.unsqueeze(1).expand(-1, 3)
        depth_render = render_with_custom_colors(
            pc, height, width, c2w, fxfycxcy, depth_colors, bg_color=(0.0, 0.0, 0.0))
        depth_map = depth_render[0].detach().cpu().numpy().clip(0, 1)

        # 4) Normal: encode camera-space normals from Gaussian min-scale axis
        rotations = pc.get_rotation
        scales = pc.get_scaling

        q = rotations / (rotations.norm(dim=1, keepdim=True) + 1e-8)
        w_q, x_q, y_q, z_q = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R_mat = torch.stack([
            1 - 2*(y_q*y_q + z_q*z_q), 2*(x_q*y_q - w_q*z_q), 2*(x_q*z_q + w_q*y_q),
            2*(x_q*y_q + w_q*z_q), 1 - 2*(x_q*x_q + z_q*z_q), 2*(y_q*z_q - w_q*x_q),
            2*(x_q*z_q - w_q*y_q), 2*(y_q*z_q + w_q*x_q), 1 - 2*(x_q*x_q + y_q*y_q),
        ], dim=-1).reshape(-1, 3, 3)

        min_axis = scales.argmin(dim=1)
        normals_world = torch.zeros(n_pts, 3, dtype=torch.float32, device=device)
        for ax in range(3):
            mask = (min_axis == ax)
            if mask.any():
                normals_world[mask] = R_mat[mask, :, ax]

        R_cam = w2c[:3, :3]
        normals_cam = (R_cam @ normals_world.T).T
        flip = normals_cam[:, 2] > 0
        normals_cam[flip] *= -1
        n_norm = normals_cam.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normals_cam = normals_cam / n_norm
        normal_colors = normals_cam * 0.5 + 0.5

        normal_render = render_with_custom_colors(
            pc, height, width, c2w, fxfycxcy, normal_colors, bg_color=(0.5, 0.5, 0.5))
        normal_map = normal_render.detach().cpu().numpy().transpose(1, 2, 0)
        normal_rgb = (normal_map.clip(0, 1) * 255).astype(np.uint8)

    del pc
    torch.cuda.empty_cache()

    return {
        "rgb": rgb,
        "depth": depth_map,
        "normal": normal_rgb,
        "opacity": opacity_out,
    }


def main():
    parser = argparse.ArgumentParser(description="Render maps from Gaussian Splats")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Re-render everything (wipe existing outputs)")
    args = parser.parse_args()

    config = load_config()
    depth_cfg = config["depth_export"]

    splat_dir = Path(config["paths"]["splat_output"]).resolve()
    depth_dir = Path(config["paths"]["depth_output"]).resolve()
    normal_dir = Path(config["paths"]["normal_output"]).resolve()
    opacity_dir = Path(config["paths"]["opacity_output"]).resolve()
    rgb_dir = Path(config["paths"]["rgb_output"]).resolve()

    render_res = args.resolution or depth_cfg["render_resolution"]
    cam_dist = depth_cfg["camera_distance"]
    fov = depth_cfg["fov"]

    # Add FaceLift repo to path
    facelift_repo = Path(config["paths"]["facelift_repo"]).resolve()
    if str(facelift_repo) not in sys.path:
        sys.path.insert(0, str(facelift_repo))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Resolution: {render_res}, FOV: {fov}, Camera dist: {cam_dist}")

    # Setup output dirs
    for d in [depth_dir, normal_dir, opacity_dir, rgb_dir]:
        if args.force and d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # Setup camera
    fxfycxcy, c2w = get_frontal_camera(fov, render_res, render_res, cam_dist,
                                        device=str(device))

    # Find splat files
    splat_folders = sorted([
        d for d in splat_dir.iterdir()
        if d.is_dir() and (d / "gaussians.ply").exists()
    ])
    n_splats = len(splat_folders)
    print(f"Splat models: {n_splats}")

    if n_splats == 0:
        print(f"No splats found in {splat_dir}")
        print("Run scripts/batch_inference.py first.")
        sys.exit(1)

    # Render loop with per-sample skip
    t0 = time.time()
    n_done, n_skip = 0, 0

    for i, folder in enumerate(splat_folders):
        name = folder.name
        ply_path = folder / "gaussians.ply"

        targets = {
            "rgb": rgb_dir / f"{name}.png",
            "depth": depth_dir / f"{name}.png",
            "normal": normal_dir / f"{name}.png",
            "opacity": opacity_dir / f"{name}.png",
        }

        # Skip if all outputs exist and are non-empty
        if not args.force and all(
            p.exists() and p.stat().st_size > 0 for p in targets.values()
        ):
            n_skip += 1
            continue

        try:
            maps = render_all_maps_gpu(ply_path, fxfycxcy, c2w,
                                        render_res, render_res, device)

            Image.fromarray(maps["rgb"]).save(targets["rgb"])

            depth_u16 = (maps["depth"].clip(0, 1) * 65535).astype(np.uint16)
            Image.fromarray(depth_u16, mode="I;16").save(targets["depth"])

            Image.fromarray(maps["normal"]).save(targets["normal"])
            Image.fromarray(maps["opacity"], mode="L").save(targets["opacity"])
            n_done += 1

            if n_done % 10 == 0 or (i + 1) == n_splats:
                elapsed = time.time() - t0
                avg = elapsed / max(n_done, 1)
                remain = avg * (n_splats - i - 1)
                print(f"[{i+1}/{n_splats}] {name} | rendered={n_done} "
                      f"skipped={n_skip} | ETA={remain/60:.1f}min")

        except Exception as e:
            print(f"ERROR [{name}]: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - t0
    print(f"\nDone. Rendered {n_done}, skipped {n_skip} in {elapsed/60:.1f} min")
    print(f"  RGB:     {len(list(rgb_dir.glob('*.png')))}")
    print(f"  Depth:   {len(list(depth_dir.glob('*.png')))}")
    print(f"  Normal:  {len(list(normal_dir.glob('*.png')))}")
    print(f"  Opacity: {len(list(opacity_dir.glob('*.png')))}")
    print(f"\nNext step: python scripts/postprocess_maps.py  (via run_postprocess.py)")


if __name__ == "__main__":
    main()
