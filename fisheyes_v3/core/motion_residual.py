"""Motion detection using residual flow (actual - expected ego-flow).

Uses GroundProjector to subtract predicted ego-motion from Farneback flow,
then runs V12-style seed-expand on the residual flow magnitude.
"""

import cv2
import numpy as np
from core.ground import get_projector


def detect_motion_residual(prev_gray, curr_gray, morph_ksize=7):
    """Motion detection with residual flow (background subtraction via ego-motion).

    Pipeline:
      1. Compute Farneback flow
      2. Estimate ego-speed from road-region pixels
      3. residual_flow = actual_flow - expected_ego_flow
      4. Run seed-expand on residual flow magnitude (V12 logic)
    """
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)

    diff = cv2.absdiff(curr_gray, prev_gray)
    p95_diff = np.percentile(diff, 95)
    global_diff_mean = np.mean(diff)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    # ── Residual flow: actual − median road flow ────────────────
    # Road mask from geometric projection identifies background pixels.
    # Median flow on road ≈ ego-motion. Subtract to isolate independent motion.
    proj = get_projector()
    road = proj.is_road
    if road.sum() > 1000:
        bg_fx = np.median(flow[road, 0])
        bg_fy = np.median(flow[road, 1])
    else:
        bg_fx, bg_fy = 0.0, 0.0

    res_x = flow[..., 0] - bg_fx
    res_y = flow[..., 1] - bg_fy
    mag = np.sqrt(res_x * res_x + res_y * res_y)

    mag_mean = cv2.GaussianBlur(mag, (41, 41), 15)
    mag_ratio = mag / (mag_mean + 0.5)
    global_mag_mean = np.mean(mag)

    # ── V1 fallback ────────────────────────────────────────────
    if p95_diff > 120 or p95_diff < 20:
        if p95_diff > 120:
            st, ft, ed = 45, 2.0, 12
        else:
            st, ft, ed = 20, 2.5, 10

        _, seed_fb = cv2.threshold(diff, st, 255, cv2.THRESH_BINARY)
        _, cand_fb = cv2.threshold(mag, ft, 255, cv2.THRESH_BINARY)
        cand_fb = cand_fb.astype(np.uint8)
        d_fb = cv2.distanceTransform((seed_fb == 0).astype(np.uint8), cv2.DIST_L2, 5)
        exp_fb = cand_fb.copy()
        exp_fb[d_fb > ed] = 0
        mask = seed_fb | exp_fb
        mask = _cca_filter(mask, diff, mag, seed_fb, global_diff_mean, global_mag_mean)
        k = np.ones((morph_ksize, morph_ksize), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    # ── 5-bin parameters ───────────────────────────────────────
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

    # ── Candidate from residual flow ───────────────────────────
    if use_abs:
        candidate = ((mag > ft) & (mag_ratio > rt)).astype(np.uint8) * 255
    else:
        candidate = (mag_ratio > rt).astype(np.uint8) * 255

    # ── Spatial seed filtering ─────────────────────────────────
    _, seed_raw = cv2.threshold(diff, st, 255, cv2.THRESH_BINARY)
    nk = np.ones((7, 7), np.uint8)
    nc = cv2.filter2D((seed_raw > 0).astype(np.float32), -1, nk)
    mn = 3 if p95_diff > 80 else 4
    seed = seed_raw.copy()
    seed[nc < mn] = 0
    if seed.max() == 0:
        seed = seed_raw

    # ── Distance expansion ─────────────────────────────────────
    dist = cv2.distanceTransform((seed == 0).astype(np.uint8), cv2.DIST_L2, 5)
    expanded = candidate.copy()
    expanded[dist > ed] = 0

    mask = seed | expanded
    mask = _cca_filter(mask, diff, mag, seed, global_diff_mean, global_mag_mean)

    k = np.ones((morph_ksize, morph_ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def _cca_filter(mask, diff, mag, seed, global_diff_mean, global_mag_mean):
    """Three-level CCA noise filter (V12 P0)."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]

        if area < 100:
            mask[labels == i] = 0
            continue

        comp = (labels == i)

        md = np.mean(diff[comp])
        mm = np.mean(mag[comp])
        if (md / (global_diff_mean + 1e-6) < 1.5 or
                mm / (global_mag_mean + 1e-6) < 1.3):
            mask[labels == i] = 0
            continue

        if area > 200:
            si = np.sum((seed > 0) & comp)
            if si / area < 0.03:
                mask[labels == i] = 0
                continue

        if cw > 0 and ch > 0:
            bba = cw * ch
            fr = area / bba
            ar = max(cw, ch) / max(min(cw, ch), 1)
            cb = (labels == i).astype(np.uint8)
            cts, _ = cv2.findContours(cb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sol = 1.0
            if cts:
                hull = cv2.convexHull(np.vstack(cts))
                ha = cv2.contourArea(hull)
                sol = area / ha if ha > 0 else 1.0
            if (ar > 8 and sol < 0.4) or \
               (100 <= area <= 5000 and fr < 0.12 and sol < 0.4):
                mask[labels == i] = 0

    return mask
