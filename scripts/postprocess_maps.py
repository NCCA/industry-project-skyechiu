"""
Post-processing pipeline for FaceLift rendered maps.

Step 4.5: Landmark Alignment + Depth Renormalization + Normal Smoothing + Artifact Cleanup
          + Depth-Normal Consistency Check

Uses OpenCV's Haar cascade for face detection and feature-based alignment.
No external model downloads required.
"""

import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple, Dict
import json
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# 1. Facial Landmark Alignment (OpenCV-based)
# ============================================================

class LandmarkAligner:
    """
    Detect face region and key features in original RGB and rendered RGB,
    compute similarity transform, apply to depth/normal/opacity.

    Uses OpenCV's Haar cascade for face detection and template matching
    for eye/nose localization - no external model downloads needed.
    """

    def __init__(self):
        # Load OpenCV's built-in Haar cascades
        cv2_data = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(
            cv2_data + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2_data + "haarcascade_eye.xml"
        )

    def detect_landmarks(self, img_rgb: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect face keypoints: eyes center + nose tip + face center + chin.
        Returns (N, 2) array of anchor points, or None if detection fails.
        """
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # Detect face
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )

        if len(faces) == 0:
            # Fallback: use opacity/brightness-based centroid
            return self._fallback_keypoints(gray)

        # Take largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        fx, fy, fw, fh = face

        # Face center
        face_cx = fx + fw / 2.0
        face_cy = fy + fh / 2.0

        # Detect eyes within face region
        face_roi = gray[fy:fy + fh, fx:fx + fw]
        eyes = self.eye_cascade.detectMultiScale(
            face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
        )

        # Build keypoints
        points = []

        if len(eyes) >= 2:
            # Sort by x to get left eye, right eye
            eyes_sorted = sorted(eyes, key=lambda e: e[0])
            for eye in eyes_sorted[:2]:
                ex, ey, ew, eh = eye
                points.append([fx + ex + ew / 2.0, fy + ey + eh / 2.0])
        else:
            # Estimate eye positions from face bbox
            points.append([face_cx - fw * 0.18, face_cy - fh * 0.12])
            points.append([face_cx + fw * 0.18, face_cy - fh * 0.12])

        # Nose tip estimate (center of face, slightly below middle)
        nose_x = face_cx
        nose_y = face_cy + fh * 0.08
        points.append([nose_x, nose_y])

        # Face center
        points.append([face_cx, face_cy])

        # Chin
        points.append([face_cx, fy + fh * 0.95])

        # Forehead
        points.append([face_cx, fy + fh * 0.05])

        return np.array(points, dtype=np.float32)

    def _fallback_keypoints(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Fallback keypoint detection using brightness centroid.
        Used when Haar cascade fails (e.g., on rendered images).
        """
        h, w = gray.shape

        # Threshold to find bright region (face area)
        _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Largest contour
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 1000:
            return None

        # Bounding box
        bx, by, bw, bh = cv2.boundingRect(cnt)
        cx = bx + bw / 2.0
        cy = by + bh / 2.0

        # Estimate keypoints from bounding box
        points = [
            [cx - bw * 0.18, cy - bh * 0.12],  # left eye
            [cx + bw * 0.18, cy - bh * 0.12],  # right eye
            [cx, cy + bh * 0.08],               # nose
            [cx, cy],                            # center
            [cx, by + bh * 0.95],               # chin
            [cx, by + bh * 0.05],               # forehead
        ]
        return np.array(points, dtype=np.float32)

    def compute_alignment(
        self, src_points: np.ndarray, dst_points: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Compute similarity transform (scale + rotation + translation)
        from src (rendered) to dst (original).
        """
        if src_points is None or dst_points is None:
            return None
        if len(src_points) < 3 or len(dst_points) < 3:
            return None

        # Use min of available points
        n = min(len(src_points), len(dst_points))
        M, inliers = cv2.estimateAffinePartial2D(
            src_points[:n], dst_points[:n],
            method=cv2.RANSAC, ransacReprojThreshold=8.0
        )
        return M

    def apply_affine(
        self,
        img: np.ndarray,
        M: np.ndarray,
        output_size: Tuple[int, int],
        interpolation=cv2.INTER_LINEAR,
        border_value=0,
    ) -> np.ndarray:
        """Apply affine transform to an image."""
        h, w = output_size
        if img.ndim == 2:
            return cv2.warpAffine(
                img, M, (w, h),
                flags=interpolation,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=float(border_value),
            )
        else:
            if isinstance(border_value, (int, float)):
                border_value = tuple([border_value] * img.shape[2])
            return cv2.warpAffine(
                img, M, (w, h),
                flags=interpolation,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=border_value,
            )

    def close(self):
        pass  # No resources to clean up for Haar cascades

    @staticmethod
    def refine_alignment_ecc(
        src_gray: np.ndarray,
        dst_gray: np.ndarray,
        M_init: np.ndarray,
        max_iters: int = 50,
        eps: float = 1e-4,
    ) -> Optional[np.ndarray]:
        """
        Sub-pixel refinement of an affine transform using ECC (Enhanced
        Correlation Coefficient). Takes an initial coarse transform from
        landmark matching and polishes it using full image content.

        Returns refined 2x3 affine matrix, or M_init if ECC fails.
        """
        try:
            warp_matrix = M_init.astype(np.float32).copy()
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                max_iters, eps,
            )
            # Pre-blur slightly to make ECC more stable
            s = cv2.GaussianBlur(src_gray, (5, 5), 1.0)
            d = cv2.GaussianBlur(dst_gray, (5, 5), 1.0)
            cc, warp_matrix = cv2.findTransformECC(
                d, s, warp_matrix,
                motionType=cv2.MOTION_AFFINE,
                criteria=criteria,
                inputMask=None,
                gaussFiltSize=5,
            )
            return warp_matrix
        except cv2.error:
            return M_init


# ============================================================
# 2. Depth Normalization (nose-tip relative)
# ============================================================

def normalize_depth_nose_relative(
    depth_f32: np.ndarray,
    opacity: np.ndarray,
    nose_landmark: Optional[Tuple[float, float]] = None,
    opacity_threshold: float = 0.3,
) -> np.ndarray:
    """
    Re-normalize depth: nose tip = 1.0 (closest), fixed range based on face geometry.

    Args:
        depth_f32: float32 depth in [0, 1] from rendering
        opacity: uint8 opacity map
        nose_landmark: (x, y) pixel coords of nose tip
        opacity_threshold: mask threshold

    Returns:
        Renormalized depth in [0, 1], float32. 1.0 = closest (nose), 0.0 = furthest/bg
    """
    mask = opacity > (opacity_threshold * 255) if opacity.max() > 1 else opacity > opacity_threshold

    if not mask.any():
        return depth_f32

    fg_vals = depth_f32[mask]

    # Global "closest" anchor: use 99.9 percentile of face to avoid single-pixel
    # outliers but still capture the actual nose tip. This is the value we map -> 1.0.
    face_top = float(np.percentile(fg_vals, 99.9))

    # Find nose depth value (should be the closest = highest value if already inverted)
    if nose_landmark is not None:
        nx, ny = int(round(nose_landmark[0])), int(round(nose_landmark[1]))
        h, w = depth_f32.shape[:2]
        nx = np.clip(nx, 0, w - 1)
        ny = np.clip(ny, 0, h - 1)
        # Sample a small patch around nose for robustness
        r = 8
        y0, y1 = max(0, ny - r), min(h, ny + r + 1)
        x0, x1 = max(0, nx - r), min(w, nx + r + 1)
        nose_patch = depth_f32[y0:y1, x0:x1]
        nose_mask_patch = mask[y0:y1, x0:x1]
        if nose_mask_patch.any():
            # Anchor to whichever is closer: nose patch max OR global face_top.
            # Using face_top guarantees no face pixel exceeds nose_depth, so the
            # nose tip can never be clipped flat.
            nose_depth = max(float(nose_patch[nose_mask_patch].max()), face_top)
        else:
            nose_depth = face_top
    else:
        nose_depth = face_top

    # Quantization range. High end == nose_depth (== face_top) so closest pixel
    # lands exactly at 1.0 with NO clipping. Low end uses 1st percentile.
    p_low = float(np.percentile(fg_vals, 1))
    p_high = nose_depth
    d_range = p_high - p_low
    if d_range < 1e-6:
        d_range = 1.0

    # Normalize: nose = 1.0, further away = lower
    result = np.zeros_like(depth_f32)
    result[mask] = 1.0 - (nose_depth - depth_f32[mask]) / d_range
    result = np.clip(result, 0, 1)
    result[~mask] = 0.0

    return result


# ============================================================
# 3a. Normal Smoothing (RGB-guided Joint Bilateral Filter)
# ============================================================

def smooth_normals(
    normal_rgb: np.ndarray,
    opacity: np.ndarray,
    guide_rgb: Optional[np.ndarray] = None,
    d: int = 9,
    sigma_color: float = 30.0,
    sigma_space: float = 30.0,
    opacity_threshold: float = 0.3,
) -> np.ndarray:
    """
    Apply joint (RGB-guided) bilateral filter to normal map.

    If a guide image (the original RGB photo) is provided, edges are
    preserved according to the photo's geometric structure rather than
    the noisy normal map itself - much better for retaining nose ridge,
    eye sockets, lip contours, etc.

    Falls back to plain bilateral filter if guide is None or
    ximgproc is unavailable.
    """
    mask = opacity > (opacity_threshold * 255) if opacity.max() > 1 else opacity > opacity_threshold

    smoothed = None
    if guide_rgb is not None:
        try:
            # Joint bilateral via ximgproc (preferred)
            from cv2 import ximgproc
            guide_bgr = cv2.cvtColor(guide_rgb, cv2.COLOR_RGB2BGR)
            normal_bgr = cv2.cvtColor(normal_rgb, cv2.COLOR_RGB2BGR)
            smoothed_bgr = ximgproc.jointBilateralFilter(
                guide_bgr, normal_bgr, d, sigma_color, sigma_space
            )
            smoothed = cv2.cvtColor(smoothed_bgr, cv2.COLOR_BGR2RGB)
        except (ImportError, AttributeError, cv2.error):
            # Fallback: guided filter using guide as luminance edge mask
            try:
                from cv2 import ximgproc
                guide_gray = cv2.cvtColor(guide_rgb, cv2.COLOR_RGB2GRAY)
                smoothed = np.zeros_like(normal_rgb)
                for c in range(3):
                    smoothed[..., c] = ximgproc.guidedFilter(
                        guide_gray, normal_rgb[..., c], radius=d, eps=400
                    )
            except (ImportError, AttributeError, cv2.error):
                smoothed = None

    if smoothed is None:
        # Final fallback: plain bilateral
        smoothed = cv2.bilateralFilter(normal_rgb, d, sigma_color, sigma_space)

    # Renormalize: bilateral filtering breaks unit-length property of normals.
    # Decode RGB[0,255] -> [-1,1], renormalize, re-encode.
    n = smoothed.astype(np.float32) / 255.0 * 2.0 - 1.0
    norm = np.linalg.norm(n, axis=2, keepdims=True)
    n = n / (norm + 1e-8)
    smoothed = np.clip((n + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)

    result = normal_rgb.copy()
    result[mask] = smoothed[mask]
    # Restore background to neutral gray (128,128,128)
    result[~mask] = 128
    return result


# ============================================================
# 3b. Normal recomputation from depth gradient
# ============================================================

def normals_from_depth(
    depth_f32: np.ndarray,
    opacity: np.ndarray,
    fx: float = 512.0 / (2.0 * np.tan(np.deg2rad(50.0) / 2.0)),
    fy: float = None,
    opacity_threshold: float = 0.3,
    smooth_sigma: float = 1.5,
) -> np.ndarray:
    """
    Compute geometric normals analytically from depth map using cross
    product of tangent vectors. This produces a normal map that is
    physically consistent with depth (by construction) and far smoother
    than the splat-derived normals.

    Args:
        depth_f32: depth in [0, 1], 1.0 = closest (already inverted)
        opacity: foreground mask (uint8 or float)
        fx, fy: focal length in pixels (defaults match FaceLift fov=50, w=512)
        smooth_sigma: gaussian smoothing applied to depth before gradient

    Returns:
        normal map encoded as uint8 RGB in [0, 255], where (0.5, 0.5, 0.5)
        is the neutral normal (pointing toward camera)
    """
    if fy is None:
        fy = fx

    h, w = depth_f32.shape[:2]
    mask = opacity > (opacity_threshold * 255) if opacity.max() > 1 else opacity > opacity_threshold

    # Convert normalized depth back to a pseudo-metric for gradient calc.
    # We want "closer = smaller Z" (camera convention), so invert.
    z = (1.0 - depth_f32) * 0.5  # arbitrary scale; only relative matters

    # Smooth depth slightly to suppress 3DGS speckle before gradient
    if smooth_sigma > 0:
        ksize = int(2 * round(3 * smooth_sigma) + 1)
        z_smooth = cv2.GaussianBlur(z, (ksize, ksize), smooth_sigma)
    else:
        z_smooth = z

    # Compute gradients with Scharr (more accurate than Sobel for normals)
    dz_dx = cv2.Scharr(z_smooth, cv2.CV_32F, 1, 0) / 32.0
    dz_dy = cv2.Scharr(z_smooth, cv2.CV_32F, 0, 1) / 32.0

    # Build normal vector at each pixel:
    # n = normalize( (-dz/dx * fx, -dz/dy * fy, 1) )
    nx = -dz_dx * fx
    ny = -dz_dy * fy
    nz = np.ones_like(z)

    norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2) + 1e-8
    nx /= norm
    ny /= norm
    nz /= norm

    # Encode to RGB [0, 255]
    normal_rgb = np.zeros((h, w, 3), dtype=np.float32)
    normal_rgb[..., 0] = nx
    normal_rgb[..., 1] = ny
    normal_rgb[..., 2] = nz
    normal_rgb = (normal_rgb * 0.5 + 0.5) * 255.0
    normal_rgb = np.clip(normal_rgb, 0, 255).astype(np.uint8)

    # Set background to neutral (128, 128, 128)
    normal_rgb[~mask] = 128

    return normal_rgb


# ============================================================
# 3c. Hole filling (depth + opacity)
# ============================================================

def fill_holes(
    depth_f32: np.ndarray,
    opacity: np.ndarray,
    max_hole_area: int = 5000,
    opacity_threshold: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fill ALL interior holes in opacity (and propagate to depth).

    A "hole" is a background-classified pixel that is enclosed inside the
    main foreground component (cannot reach image border via background).
    We find them via flood-fill from the corner, then fill depth via inpaint
    (small holes) or nearest-neighbor + smoothing (large holes).
    """
    if opacity.max() <= 1:
        binary_fg = (opacity > opacity_threshold).astype(np.uint8)
    else:
        binary_fg = (opacity > int(opacity_threshold * 255)).astype(np.uint8)

    # Flood fill from border to identify "true background"
    h, w = binary_fg.shape
    flood = binary_fg.copy()
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 2)

    # Holes = pixels classified as background but NOT reached by flood
    holes = (binary_fg == 0) & (flood != 2)

    if not holes.any():
        return depth_f32, opacity

    # Separate small vs large holes
    holes_u8 = holes.astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(holes_u8, connectivity=8)
    small_mask = np.zeros_like(holes, dtype=bool)
    large_mask = np.zeros_like(holes, dtype=bool)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area <= max_hole_area:
            small_mask |= (labels == lbl)
        else:
            large_mask |= (labels == lbl)

    depth_filled = depth_f32.copy()

    # Small holes: TELEA inpaint (good for smooth interpolation)
    if small_mask.any():
        depth_u8 = (depth_f32.clip(0, 1) * 255).astype(np.uint8)
        inpainted = cv2.inpaint(depth_u8, small_mask.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
        depth_filled[small_mask] = inpainted[small_mask].astype(np.float32) / 255.0

    # Large holes: nearest-neighbor distance transform fill (more stable for big regions)
    if large_mask.any():
        # distanceTransform with labels gives us the index of the nearest non-hole pixel
        non_hole = (~large_mask).astype(np.uint8)
        _, lbl_map = cv2.distanceTransformWithLabels(
            1 - non_hole, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
        )
        # Build coord lookup for each label
        ys, xs = np.where(non_hole > 0)
        flat_labels = lbl_map[ys, xs]
        order = np.argsort(flat_labels)
        flat_labels = flat_labels[order]
        ys = ys[order]; xs = xs[order]
        # Fill each large-hole pixel with the value at the nearest source
        hy, hx = np.where(large_mask)
        src_lbls = lbl_map[hy, hx]
        idx = np.searchsorted(flat_labels, src_lbls)
        idx = np.clip(idx, 0, len(ys) - 1)
        depth_filled[hy, hx] = depth_f32[ys[idx], xs[idx]]
        # Smooth the filled large region a bit to avoid blocky edges
        if large_mask.sum() > 0:
            blur = cv2.GaussianBlur(depth_filled, (0, 0), sigmaX=2.0)
            depth_filled[large_mask] = blur[large_mask]

    fill_mask = small_mask | large_mask

    # Patch opacity to mark these as foreground
    opacity_filled = opacity.copy()
    if opacity.max() <= 1:
        opacity_filled[fill_mask] = 1.0
    else:
        opacity_filled[fill_mask] = 255

    return depth_filled, opacity_filled


# ============================================================
# 4. Artifact Cleanup (Opacity-based)
# ============================================================

def cleanup_artifacts(
    depth_f32: np.ndarray,
    normal_rgb: np.ndarray,
    opacity: np.ndarray,
    opacity_threshold: float = 0.3,
    min_component_area: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove floating Gaussian splat artifacts using opacity mask + connected components.
    Keeps only the largest connected region + any region above min_component_area.
    """
    # Binary mask from opacity
    if opacity.max() <= 1:
        binary = (opacity > opacity_threshold).astype(np.uint8) * 255
    else:
        thresh = int(opacity_threshold * 255)
        _, binary = cv2.threshold(opacity, thresh, 255, cv2.THRESH_BINARY)

    # Morphological close to fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Connected components - keep only the largest
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1:
        clean_mask = np.zeros_like(binary, dtype=bool)
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = np.argmax(areas) + 1
        clean_mask = labels == largest_label

        # Also keep components above min area threshold
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_component_area:
                clean_mask |= (labels == lbl)

    # Apply clean mask
    depth_clean = depth_f32.copy()
    depth_clean[~clean_mask] = 0.0

    normal_clean = normal_rgb.copy()
    normal_clean[~clean_mask] = 128  # neutral normal (0.5 encoded)

    opacity_clean = opacity.copy()
    opacity_clean[~clean_mask] = 0

    return depth_clean, normal_clean, opacity_clean


# ============================================================
# 5. Depth-Normal Consistency Check
# ============================================================

def check_depth_normal_consistency(
    depth_f32: np.ndarray,
    normal_rgb: np.ndarray,
    opacity: np.ndarray,
    opacity_threshold: float = 0.3,
) -> Tuple[float, np.ndarray]:
    """
    Check physical consistency between depth gradient and surface normal.

    Principle: the depth map's spatial gradient direction should align with
    the normal map's x/y components. If normal points left but depth shows
    a rightward bulge, the sample has "hallucinated" geometry.

    Returns:
        consistency_score: float in [0, 1], 1.0 = perfectly consistent
        error_map: per-pixel inconsistency map
    """
    mask = opacity > (opacity_threshold * 255) if opacity.max() > 1 else opacity > opacity_threshold

    if not mask.any():
        return 0.0, np.zeros_like(depth_f32)

    # Compute depth gradients (Sobel)
    dz_dx = cv2.Sobel(depth_f32, cv2.CV_32F, 1, 0, ksize=3)
    dz_dy = cv2.Sobel(depth_f32, cv2.CV_32F, 0, 1, ksize=3)

    # Decode normals from RGB [0,255] -> [-1,1]
    normals = normal_rgb.astype(np.float32) / 255.0 * 2.0 - 1.0
    nx = normals[:, :, 0]
    ny = normals[:, :, 1]

    # Compute direction alignment
    grad_mag = np.sqrt(dz_dx ** 2 + dz_dy ** 2) + 1e-8
    normal_xy_mag = np.sqrt(nx ** 2 + ny ** 2) + 1e-8

    # Note: depth_norm uses "close=1" convention (inverted from world Z),
    # so positive depth gradient means surface getting closer in that direction,
    # and the normal's xy components point in the SAME direction as the gradient.
    grad_dir_x = dz_dx / grad_mag
    grad_dir_y = dz_dy / grad_mag
    norm_dir_x = nx / normal_xy_mag
    norm_dir_y = ny / normal_xy_mag

    # Dot product = cosine similarity
    alignment = grad_dir_x * norm_dir_x + grad_dir_y * norm_dir_y
    alignment = np.clip(alignment, -1, 1)

    # Error: 0 = perfect, 1 = opposite direction
    error_map = (1.0 - alignment) / 2.0
    error_map[~mask] = 0.0

    # Only evaluate where both gradient and normal have significant magnitude
    significant = mask & (grad_mag > 0.005) & (normal_xy_mag > 0.05)
    if significant.any():
        consistency = 1.0 - np.mean(error_map[significant])
    else:
        consistency = 0.5  # neutral if no significant gradients

    return float(consistency), error_map


# ============================================================
# Main Post-Processing Pipeline
# ============================================================

def postprocess_single(
    orig_rgb_path: str,
    render_rgb_path: str,
    depth_path: str,
    normal_path: str,
    opacity_path: str,
    aligner: LandmarkAligner,
    config: dict,
) -> Dict[str, any]:
    """
    Post-process a single sample through the enhanced pipeline:
    1. Landmark alignment (Haar) + ECC sub-pixel refinement
    2. Artifact cleanup (opacity-based connected components)
    3. Hole filling (interior holes in depth/opacity)
    4. Depth renormalization (nose-relative)
    5. Normal recomputation from depth gradient (replaces noisy splat normals)
    6. Joint bilateral filter on normal using original RGB as guide
    7. Depth-normal consistency check
    """
    # Load images
    orig_rgb = np.array(Image.open(orig_rgb_path).convert("RGB"))
    render_rgb = np.array(Image.open(render_rgb_path).convert("RGB"))

    depth_img = Image.open(depth_path)
    depth_raw = np.array(depth_img, dtype=np.float32)
    if depth_raw.max() > 1.0:
        depth_raw = depth_raw / 65535.0

    normal_rgb_in = np.array(Image.open(normal_path).convert("RGB"))

    opacity_img = Image.open(opacity_path)
    if opacity_img.mode != "L":
        opacity = np.array(opacity_img.convert("L"))
    else:
        opacity = np.array(opacity_img)

    # Use the RENDERED maps as the reference resolution (they are the output
    # we want to keep at full quality). Upsample the original RGB guide to
    # match if necessary — needed when cropped_faces is 512 but renders are 1024.
    h, w = render_rgb.shape[:2]
    if orig_rgb.shape[:2] != (h, w):
        orig_rgb = cv2.resize(orig_rgb, (w, h), interpolation=cv2.INTER_CUBIC)
    result = {
        "aligned": False,
        "ecc_refined": False,
        "consistency_score": 0.0,
        "alignment_error": None,
    }

    # --- Step 1: Landmark Alignment (coarse) ---
    orig_lm = aligner.detect_landmarks(orig_rgb)
    render_lm = aligner.detect_landmarks(render_rgb)

    M = None
    nose_pt_aligned = None

    if orig_lm is not None and render_lm is not None:
        M = aligner.compute_alignment(render_lm, orig_lm)

    # --- Step 1b: ECC Sub-pixel Refinement ---
    if M is not None and config.get("use_ecc_refine", True):
        try:
            orig_gray = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY)
            render_gray = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2GRAY)
            M_refined = LandmarkAligner.refine_alignment_ecc(
                render_gray, orig_gray, M,
                max_iters=config.get("ecc_iters", 50),
                eps=config.get("ecc_eps", 1e-4),
            )
            if M_refined is not None and not np.allclose(M_refined, M):
                M = M_refined
                result["ecc_refined"] = True
        except Exception:
            pass

    if M is not None:
        # Compute alignment error
        n = min(len(render_lm), len(orig_lm))
        transformed_pts = cv2.transform(
            render_lm[:n].reshape(1, -1, 2), M
        ).reshape(-1, 2)
        align_err = np.mean(np.linalg.norm(transformed_pts - orig_lm[:n], axis=1))
        result["alignment_error"] = float(align_err)

        # Apply affine to all maps
        render_rgb = aligner.apply_affine(render_rgb, M, (h, w), border_value=255)
        depth_raw = aligner.apply_affine(depth_raw, M, (h, w), border_value=0)
        normal_rgb_in = aligner.apply_affine(normal_rgb_in, M, (h, w), border_value=128)
        opacity = aligner.apply_affine(opacity, M, (h, w), border_value=0)

        result["aligned"] = True

        if orig_lm is not None and len(orig_lm) > 2:
            nose_pt_aligned = (orig_lm[2][0], orig_lm[2][1])
    else:
        if render_lm is not None and len(render_lm) > 2:
            nose_pt_aligned = (render_lm[2][0], render_lm[2][1])

    # --- Step 2: Artifact Cleanup ---
    depth_clean, normal_clean, opacity_clean = cleanup_artifacts(
        depth_raw, normal_rgb_in, opacity,
        opacity_threshold=config.get("opacity_threshold", 0.3),
        min_component_area=config.get("min_component_area", 100),
    )

    # --- Step 3: Hole Filling (interior holes only) ---
    if config.get("fill_holes", True):
        depth_clean, opacity_clean = fill_holes(
            depth_clean, opacity_clean,
            max_hole_area=config.get("max_hole_area", 5000),
            opacity_threshold=config.get("opacity_threshold", 0.3),
        )

    # --- Step 4: Depth Renormalization (nose-relative) ---
    depth_norm = normalize_depth_nose_relative(
        depth_clean, opacity_clean,
        nose_landmark=nose_pt_aligned,
    )

    # --- Step 5: Normal Recomputation from Depth ---
    # This is the BIG quality improvement: replace noisy splat-derived
    # normals with geometrically consistent normals from depth gradients.
    if config.get("normals_from_depth", True):
        # FaceLift: hfov=50, w=512 -> fx ~= 548
        fov = config.get("camera_hfov", 50.0)
        fx = w / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
        normal_geo = normals_from_depth(
            depth_norm, opacity_clean,
            fx=fx, fy=fx,
            smooth_sigma=config.get("normal_depth_smooth", 1.5),
        )
    else:
        normal_geo = normal_clean

    # --- Step 6: Joint Bilateral Filter (RGB-guided) ---
    # Use the original photo as guide for edge-aware smoothing
    normal_smooth = smooth_normals(
        normal_geo, opacity_clean,
        guide_rgb=orig_rgb,  # KEY: use original photo as edge guide
        d=config.get("bilateral_d", 9),
        sigma_color=config.get("bilateral_sigma_color", 30.0),
        sigma_space=config.get("bilateral_sigma_space", 30.0),
    )

    # --- Step 7: Depth-Normal Consistency Check ---
    consistency, error_map = check_depth_normal_consistency(
        depth_norm, normal_smooth, opacity_clean
    )
    result["consistency_score"] = consistency

    # Package
    result["render_rgb"] = render_rgb
    result["depth"] = depth_norm
    result["normal"] = normal_smooth
    result["opacity"] = opacity_clean
    result["error_map"] = error_map

    return result


def save_processed(result: dict, name: str, output_dirs: dict):
    """Save post-processed maps to output directories."""
    Image.fromarray(result["render_rgb"]).save(
        Path(output_dirs["rgb"]) / f"{name}.png"
    )

    depth_u16 = (result["depth"].clip(0, 1) * 65535).astype(np.uint16)
    Image.fromarray(depth_u16, mode="I;16").save(
        Path(output_dirs["depth"]) / f"{name}.png"
    )

    Image.fromarray(result["normal"]).save(
        Path(output_dirs["normal"]) / f"{name}.png"
    )

    Image.fromarray(result["opacity"], mode="L").save(
        Path(output_dirs["opacity"]) / f"{name}.png"
    )
