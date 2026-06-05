"""Magnitude-blended motion detection: perspective mag at center, fisheye at edge.

Pipeline:
  1. Compute Farneback in fisheye → mag_f, mag_ratio_f
  2. Undistort → Farneback in perspective → mag_p
  3. Reproject mag_p back to fisheye → mag_p_reproj
  4. Radial blend: mag = w(r)·mag_p_reproj + (1-w(r))·mag_f
  5. Compute mag_ratio, run V12 seed-expand + CCA on blended magnitude
"""

import cv2
import numpy as np
from core.calib import RadialPoly
from pathlib import Path

CALIB_DIR = (Path(__file__).resolve().parents[2] / "fisheyes_v2" /
             "fisheye_motion_tracking" / "data" / "homework2" / "calibration_data")

# ═══════════════════════════════════════════════════════════════════
# Radial blend weight map — precomputed once
# ═══════════════════════════════════════════════════════════════════
_BLEND_MAP = None  # (h, w) float32


def _get_blend_map(h, w):
    """Lazy-init radial blend weight: w=1 at center, w=0 at edge."""
    global _BLEND_MAP
    if _BLEND_MAP is not None and _BLEND_MAP.shape == (h, w):
        return _BLEND_MAP

    cx, cy = w / 2, h / 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / np.sqrt(cx * cx + cy * cy)  # 0..1

    # Smooth step: 1.0 at r<0.4, 0.0 at r>0.7, cosine ramp between
    ramp = np.clip((0.7 - r) / 0.3, 0.0, 1.0)
    _BLEND_MAP = (0.5 - 0.5 * np.cos(np.pi * ramp)).astype(np.float32)
    return _BLEND_MAP


# ═══════════════════════════════════════════════════════════════════
# Camera cache (all frames share same calibration)
# ═══════════════════════════════════════════════════════════════════
_CAMERA = None


def _get_camera(fov_scale=1.0):
    global _CAMERA
    if _CAMERA is None:
        json_path = CALIB_DIR / "00000_FV.json"
        _CAMERA = RadialPoly(str(json_path), fov_scale=fov_scale)
    return _CAMERA


# ═══════════════════════════════════════════════════════════════════
def detect_motion_blend(prev_gray, curr_gray, morph_ksize=7):
    """Magnitude-blended motion detection.

    Uses perspective-domain Farneback at image center where distortion
    is mild, falls back to fisheye-domain Farneback at edges where
    perspective remap stretching would break Farneback.
    """
    pg_f = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    cg_f = cv2.GaussianBlur(curr_gray, (5, 5), 0)
    h, w = pg_f.shape

    diff = cv2.absdiff(cg_f, pg_f)
    p95_diff = np.percentile(diff, 95)
    global_diff_mean = np.mean(diff)

    # ── 1. Fisheye-domain Farneback ─────────────────────────────
    flow_f = cv2.calcOpticalFlowFarneback(
        pg_f, cg_f, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag_f = np.linalg.norm(flow_f, axis=2)

    # ── 2. Perspective-domain Farneback ─────────────────────────
    camera = _get_camera(fov_scale=1.0)
    pg_p = camera.undistort(pg_f)
    cg_p = camera.undistort(cg_f)

    flow_p = cv2.calcOpticalFlowFarneback(
        pg_p, cg_p, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag_p = np.linalg.norm(flow_p, axis=2)

    # ── 3. Reproject perspective mag back to fisheye ────────────
    mag_p_reproj = camera.reproject_to_fisheye(mag_p)

    # ── 4. Radial blend ─────────────────────────────────────────
    w = _get_blend_map(h, w)
    mag = w * mag_p_reproj + (1.0 - w) * mag_f

    # ── 5. Ratio + rest of V12 ──────────────────────────────────
    mag_mean = cv2.GaussianBlur(mag, (41, 41), 15)
    mag_ratio = mag / (mag_mean + 0.5)
    global_mag_mean = np.mean(mag)

    # ── V1 fallback ─────────────────────────────────────────────
    if p95_diff > 120 or p95_diff < 20:
        if p95_diff > 120:
            st, ft, ed = 45, 2.0, 12
        else:
            st, ft, ed = 20, 2.5, 10
        _, seed_fb = cv2.threshold(diff, st, 255, cv2.THRESH_BINARY)
        _, cand_fb = cv2.threshold(mag, ft, 255, cv2.THRESH_BINARY)
        cand_fb = cand_fb.astype(np.uint8)
        d_fb = cv2.distanceTransform((seed_fb == 0).astype(np.uint8), cv2.DIST_L2, 5)
        exp_fb = cand_fb.copy(); exp_fb[d_fb > ed] = 0
        mask = seed_fb | exp_fb
        mask = _cca(mask, diff, mag, seed_fb, global_diff_mean, global_mag_mean)
        k = np.ones((morph_ksize, morph_ksize), np.uint8)
        return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)

    # ── 5-bin params ────────────────────────────────────────────
    if p95_diff > 100:
        st, rt, ed = 22, 1.3, 12; use_abs, ft = True, 2.0
    elif p95_diff > 70:
        st, rt, ed = 20, 1.5, 10; use_abs, ft = True, 1.5
    elif p95_diff > 50:
        st, rt, ed = 18, 1.8, 10; use_abs = False
    elif p95_diff > 30:
        st, rt, ed = 16, 2.0, 10; use_abs = False
    else:
        st, rt, ed = 14, 2.2, 4; use_abs = False

    if use_abs:
        candidate = ((mag > ft) & (mag_ratio > rt)).astype(np.uint8) * 255
    else:
        candidate = (mag_ratio > rt).astype(np.uint8) * 255

    # ── Seed + expand + CCA ─────────────────────────────────────
    _, seed_raw = cv2.threshold(diff, st, 255, cv2.THRESH_BINARY)
    nk = np.ones((7, 7), np.uint8)
    nc = cv2.filter2D((seed_raw > 0).astype(np.float32), -1, nk)
    mn = 3 if p95_diff > 80 else 4
    seed = seed_raw.copy(); seed[nc < mn] = 0
    if seed.max() == 0: seed = seed_raw

    dist = cv2.distanceTransform((seed == 0).astype(np.uint8), cv2.DIST_L2, 5)
    expanded = candidate.copy(); expanded[dist > ed] = 0
    mask = seed | expanded
    mask = _cca(mask, diff, mag, seed, global_diff_mean, global_mag_mean)

    k = np.ones((morph_ksize, morph_ksize), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)


def _cca(mask, diff, mag, seed, gdm, gmm):
    """V12 CCA filter."""
    nl, lb, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, nl):
        ar = st[i, cv2.CC_STAT_AREA]
        cw = st[i, cv2.CC_STAT_WIDTH]; ch = st[i, cv2.CC_STAT_HEIGHT]
        if ar < 100: mask[lb == i] = 0; continue
        cp = (lb == i)
        md, mm = np.mean(diff[cp]), np.mean(mag[cp])
        if md/(gdm+1e-6) < 1.5 or mm/(gmm+1e-6) < 1.3: mask[lb == i] = 0; continue
        if ar > 200:
            si = np.sum((seed > 0) & cp)
            if si/ar < 0.03: mask[lb == i] = 0; continue
        if cw > 0 and ch > 0:
            bba = cw*ch; fr = ar/bba; ar_ = max(cw,ch)/max(min(cw,ch),1)
            cb = (lb == i).astype(np.uint8)
            cts, _ = cv2.findContours(cb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sol = 1.0
            if cts:
                hull = cv2.convexHull(np.vstack(cts))
                ha = cv2.contourArea(hull); sol = ar/ha if ha > 0 else 1.0
            if (ar_ > 8 and sol < 0.4) or (100 <= ar <= 5000 and fr < 0.12 and sol < 0.4):
                mask[lb == i] = 0
    return mask
