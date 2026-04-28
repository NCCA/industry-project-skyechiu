"""
Filter raw_faces: remove side faces, incomplete faces, cluttered backgrounds.

Uses mediapipe FaceMesh for landmark detection (468 points).
1. Side face check: nose-to-eye horizontal distance ratio
2. Incomplete face: landmarks near image edges
3. Background: border pixel variance
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

_base = "/sessions/epic-zen-shannon/mnt/facelift_pipeline/data"
if not os.path.exists(_base):
    _base = r"D:\zmm\facelift_pipeline\data"

RAW_DIR = os.path.join(_base, "raw_faces")

# Thresholds
FRONTAL_THRESHOLD = 0.55
BACKGROUND_THRESHOLD = 70.0
BORDER_WIDTH = 40
EDGE_MARGIN = 0.05  # landmarks within 5% of edge = cropped

# Load mediapipe
import mediapipe as mp
try:
    # New API (mediapipe >= 0.10.8)
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    # Download model if needed
    import urllib.request
    model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
    if not os.path.exists(model_path):
        print("Downloading face_landmarker model...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.5
    )
    face_mesh = vision.FaceLandmarker.create_from_options(options)
    USE_NEW_API = True
    print("MediaPipe FaceLandmarker loaded (new API)")
except Exception:
    # Old API fallback
    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
        )
        USE_NEW_API = False
        print("MediaPipe FaceMesh loaded (legacy API)")
    except Exception as e:
        print("ERROR: Cannot load mediapipe: " + str(e))
        sys.exit(1)

# Key landmark indices
# 33 = left eye outer, 263 = right eye outer, 1 = nose tip
# 10 = forehead top, 152 = chin bottom
# 234 = left cheek, 454 = right cheek
LEFT_EYE = 33
RIGHT_EYE = 263
NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152
LEFT_EDGE = 234
RIGHT_EDGE = 454


def analyze_face(img_path):
    """
    Returns (has_face, is_frontal, is_complete, frontal_ratio, reason)
    """
    try:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return False, False, False, 0.0, "cannot read"

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        if USE_NEW_API:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            results = face_mesh.detect(mp_image)
            if not results.face_landmarks or len(results.face_landmarks) == 0:
                return False, False, False, 0.0, "no face"
            lm = results.face_landmarks[0]
        else:
            results = face_mesh.process(img_rgb)
            if not results.multi_face_landmarks:
                return False, False, False, 0.0, "no face"
            lm = results.multi_face_landmarks[0].landmark

        # Get key points (normalized 0-1)
        left_eye = lm[LEFT_EYE]
        right_eye = lm[RIGHT_EYE]
        nose = lm[NOSE_TIP]
        forehead = lm[FOREHEAD]
        chin = lm[CHIN]
        left_edge = lm[LEFT_EDGE]
        right_edge = lm[RIGHT_EDGE]

        # === Check 1: Frontal ===
        d_left = abs(nose.x - left_eye.x)
        d_right = abs(nose.x - right_eye.x)
        if max(d_left, d_right) < 1e-3:
            return True, False, False, 0.0, "side face"
        frontal_ratio = min(d_left, d_right) / max(d_left, d_right)
        is_frontal = frontal_ratio > FRONTAL_THRESHOLD

        # === Check 2: Complete face ===
        # Check if key landmarks are too close to image edges
        margin = EDGE_MARGIN
        is_complete = True
        crop_reason = "ok"

        # Forehead near top edge
        if forehead.y < margin:
            is_complete = False
            crop_reason = "top cropped (forehead)"
        # Chin near bottom edge
        elif chin.y > 1.0 - margin:
            is_complete = False
            crop_reason = "bottom cropped (chin)"
        # Left face edge near left image edge
        elif left_edge.x < margin:
            is_complete = False
            crop_reason = "left cropped"
        # Right face edge near right image edge
        elif right_edge.x > 1.0 - margin:
            is_complete = False
            crop_reason = "right cropped"

        if not is_frontal:
            return True, False, is_complete, frontal_ratio, "side face (ratio=" + str(round(frontal_ratio, 2)) + ")"
        if not is_complete:
            return True, True, False, frontal_ratio, crop_reason

        return True, True, True, frontal_ratio, "ok"

    except Exception as e:
        return False, False, False, 0.0, "error: " + str(e)


def get_bg_score(img_path):
    """Border pixel std dev. Lower = cleaner."""
    try:
        img = Image.open(img_path).convert('RGB')
        arr = np.array(img)
        h, w = arr.shape[:2]
        bw = min(BORDER_WIDTH, h // 5, w // 5)
        if bw < 5:
            return 999.0

        top = arr[:bw, :, :]
        bottom = arr[h-bw:, :, :]
        left = arr[bw:h-bw, :bw, :]
        right = arr[bw:h-bw, w-bw:, :]

        border = np.concatenate([
            top.reshape(-1, 3), bottom.reshape(-1, 3),
            left.reshape(-1, 3), right.reshape(-1, 3)
        ], axis=0)

        return float(np.mean(np.std(border.astype(np.float32), axis=0)))
    except Exception:
        return 999.0


def main():
    files = sorted(os.listdir(RAW_DIR))
    total = len(files)
    print("Total images: " + str(total))

    removed_side = 0
    removed_bg = 0
    removed_noface = 0
    removed_cropped = 0
    kept = 0

    for i, fname in enumerate(files):
        fpath = os.path.join(RAW_DIR, fname)
        if not os.path.isfile(fpath):
            continue

        ext = os.path.splitext(fname)[1].lower()
        if ext not in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
            continue

        # Check face with mediapipe
        has_face, is_frontal, is_complete, ratio, reason = analyze_face(fpath)

        if not has_face:
            removed_noface += 1
            os.remove(fpath)
            continue

        if not is_frontal:
            removed_side += 1
            os.remove(fpath)
            continue

        if not is_complete:
            removed_cropped += 1
            os.remove(fpath)
            continue

        # Check background
        bg_score = get_bg_score(fpath)
        if bg_score > BACKGROUND_THRESHOLD:
            removed_bg += 1
            os.remove(fpath)
            continue

        kept += 1

        if (i + 1) % 100 == 0:
            print("  [" + str(i+1) + "/" + str(total) + "] kept=" + str(kept) + " side=" + str(removed_side) + " crop=" + str(removed_cropped) + " bg=" + str(removed_bg) + " noface=" + str(removed_noface))

    remaining = len([f for f in os.listdir(RAW_DIR) if os.path.splitext(f)[1].lower() in ('.jpg','.jpeg','.png','.bmp','.webp')])
    print("\n=== Done ===")
    print("Removed side faces: " + str(removed_side))
    print("Removed cropped/incomplete: " + str(removed_cropped))
    print("Removed bad background: " + str(removed_bg))
    print("Removed no face detected: " + str(removed_noface))
    print("Kept: " + str(kept))
    print("Remaining in raw_faces: " + str(remaining))


if __name__ == "__main__":
    main()
