"""Angular-Flow-Blended motion detection.

Converts both fisheye and perspective Farneback flow to angular magnitude
before blending, then runs V12 pipeline on the blended angular signal.
"""

import cv2
import numpy as np
from core.calib import RadialPoly
from core.angular import fisheye_angular_mag, perspective_angular_mag
from pathlib import Path

CALIB_DIR = (Path(__file__).resolve().parents[2] / "fisheyes_v2" /
             "fisheye_motion_tracking" / "data" / "homework2" / "calibration_data")

# ── Radial blend weight ──────────────────────────────────────────
_BLEND_MAP = None


def _get_blend_map(h, w):
    global _BLEND_MAP
    if _BLEND_MAP is not None and _BLEND_MAP.shape == (h, w):
        return _BLEND_MAP
    cx, cy = w / 2, h / 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / np.sqrt(cx * cx + cy * cy)
    ramp = np.clip((0.7 - r) / 0.3, 0.0, 1.0)
    _BLEND_MAP = (0.5 - 0.5 * np.cos(np.pi * ramp)).astype(np.float32)
    return _BLEND_MAP


_CAMERA = None


def _get_camera():
    global _CAMERA
    if _CAMERA is None:
        _CAMERA = RadialPoly(str(CALIB_DIR / "00000_FV.json"), fov_scale=1.0)
    return _CAMERA


# ═══════════════════════════════════════════════════════════════════
def detect_motion_angular(prev_gray, curr_gray, morph_ksize=7):
    """Angular-flow blended motion detection.

    Pipeline:
      1. Fisheye Farneback → ang_mag_f
      2. Undistort → Perspective Farneback → ang_mag_p → reproject to fisheye
      3. Blend: ang_mag = w(r)*ang_mag_p + (1-w)*ang_mag_f
      4. V12 seed-expand + CCA on ang_mag
    """
    pg_f = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    cg_f = cv2.GaussianBlur(curr_gray, (5, 5), 0)
    h, w = pg_f.shape

    diff = cv2.absdiff(cg_f, pg_f)
    p95_diff = np.percentile(diff, 95)
    global_diff_mean = np.mean(diff)

    # ── 1. Fisheye angular flow ─────────────────────────────────
    flow_f = cv2.calcOpticalFlowFarneback(
        pg_f, cg_f, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    amag_f = fisheye_angular_mag(flow_f)

    # ── 2. Perspective angular flow ─────────────────────────────
    cam = _get_camera()
    pg_p = cam.undistort(pg_f)
    cg_p = cam.undistort(cg_f)
    flow_p = cv2.calcOpticalFlowFarneback(
        pg_p, cg_p, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    amag_p = perspective_angular_mag(flow_p)
    amag_p_reproj = cam.reproject_to_fisheye(amag_p)

    # ── 3. Blend ────────────────────────────────────────────────
    w_map = _get_blend_map(h, w)
    mag = w_map * amag_p_reproj + (1.0 - w_map) * amag_f

    # ── 4. Ratio + V12 pipeline ─────────────────────────────────
    # Angular mag is in radians (typical: 0.001-0.05 rad/frame).
    # Scale thresholds: V12's pixel thresholds ÷ k1 ≈ angular thresholds.
    # mag_thresh=2.0 px → 2.0/340 ≈ 0.006 rad
    # mag_thresh=1.5 px → 1.5/340 ≈ 0.0045 rad
    # We keep V12's threshold logic but rescale to angular domain.
    SCALE = 1.0 / 340.0  # pixel → radian approximate scale

    mag_mean = cv2.GaussianBlur(mag, (41, 41), 15)
    mag_ratio = mag / (mag_mean + 1e-6)
    global_mag_mean = np.mean(mag)

    # ── V1 fallback ─────────────────────────────────────────────
    if p95_diff > 120 or p95_diff < 20:
        if p95_diff > 120:
            st, ft, ed = 45, 2.0 * SCALE, 12
        else:
            st, ft, ed = 20, 2.5 * SCALE, 10
        _, seed_fb = cv2.threshold(diff, st, 255, cv2.THRESH_BINARY)
        _, cand_fb = cv2.threshold(mag, ft, 255, cv2.THRESH_BINARY)
        cand_fb = cand_fb.astype(np.uint8)
        d_fb = cv2.distanceTransform((seed_fb == 0).astype(np.uint8), cv2.DIST_L2, 5)
        exp_fb = cand_fb.copy(); exp_fb[d_fb > ed] = 0
        mask = seed_fb | exp_fb
        mask = _cca(mask, diff, mag, seed_fb, global_diff_mean, global_mag_mean)
        k = np.ones((morph_ksize, morph_ksize), np.uint8)
        return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)

    # ── 5-bin params (rescaled to angular domain) ───────────────
    if p95_diff > 100:
        st, rt, ed = 22, 1.3, 12; use_abs, ft = True, 2.0 * SCALE
    elif p95_diff > 70:
        st, rt, ed = 20, 1.5, 10; use_abs, ft = True, 1.5 * SCALE
    elif p95_diff > 50:
        st, rt, ed = 18, 1.6, 10; use_abs = False
    elif p95_diff > 30:
        st, rt, ed = 16, 1.8, 10; use_abs = False
    else:
        st, rt, ed = 14, 2.0, 4; use_abs = False

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
        if md / (gdm + 1e-6) < 1.5 or mm / (gmm + 1e-6) < 1.3:
            mask[lb == i] = 0; continue
        if ar > 200:
            si = np.sum((seed > 0) & cp)
            if si / ar < 0.03: mask[lb == i] = 0; continue
        if cw > 0 and ch > 0:
            bba = cw * ch; fr = ar / bba
            ar_ = max(cw, ch) / max(min(cw, ch), 1)
            cb = (lb == i).astype(np.uint8)
            cts, _ = cv2.findContours(cb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sol = 1.0
            if cts:
                hull = cv2.convexHull(np.vstack(cts))
                ha = cv2.contourArea(hull); sol = ar / ha if ha > 0 else 1.0
            if (ar_ > 8 and sol < 0.4) or \
               (100 <= ar <= 5000 and fr < 0.12 and sol < 0.4):
                mask[lb == i] = 0
    return mask
