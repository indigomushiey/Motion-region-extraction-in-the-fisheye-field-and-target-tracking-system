"""Hybrid Flow motion detection: perspective mag at center, fisheye at edge.

Pipeline:
  1. Compute Farneback in fisheye → mag_f, mag_ratio_f
  2. Undistort → Farneback in perspective → mag_p
  3. Reproject mag_p back to fisheye → mag_p_reproj
  4. Hybrid blend: mag = w(r)·mag_p_reproj + (1-w(r))·mag_f
  5. Compute mag_ratio, run V12 seed-expand + CCA on hybrid magnitude

Parameter overrides (set HYBRID_PARAMS dict before calling detect_motion_hybrid):
  r_start, r_end : blend transition zone [default: 0.4, 0.7]
  ratio_delta_A..G : ratio_thresh adjustment per p95 bin [default: 0.0]
"""

HYBRID_PARAMS = {}

import cv2
import numpy as np
from core.calib import RadialPoly
from pathlib import Path

CALIB_DIR = (Path(__file__).resolve().parents[1] /
             "data" / "homework2" / "calibration_data")

# ═══════════════════════════════════════════════════════════════════
# Hybrid blend weight map — precomputed once
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

    rs = HYBRID_PARAMS.get("r_start", 0.35)
    re = HYBRID_PARAMS.get("r_end", 0.65)
    ramp = np.clip((re - r) / (re - rs + 1e-6), 0.0, 1.0)
    _BLEND_MAP = (0.5 - 0.5 * np.cos(np.pi * ramp)).astype(np.float32)
    return _BLEND_MAP


# ═══════════════════════════════════════════════════════════════════
# Camera cache (all frames share same calibration)
# ═══════════════════════════════════════════════════════════════════
_GROUND_MASK = None  # (h, w) bool: True = pixel sees ground plane


def _get_ground_mask(h, w):
    """Lazy-init ground plane mask from camera extrinsic.

    Camera: height=0.66m, forward=3.75m, pitch≈0.6° downward.
    Uses the radial_poly model to compute ray directions and
    intersect with the ground plane (Y = -0.66m in camera coords).
    """
    global _GROUND_MASK
    if _GROUND_MASK is not None and _GROUND_MASK.shape == (h, w):
        return _GROUND_MASK

    # Camera intrinsic (from calib)
    k1, k2, k3, k4 = 339.75, -31.99, 48.28, -7.21
    cx, cy = w / 2, h / 2
    pitch_rad = np.deg2rad(0.6)  # downward pitch

    yy, xx = np.ogrid[:h, :w]
    r_d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    phi = np.arctan2(yy - cy, xx - cx)

    # Newton solve for theta (same as calib.py inverse maps)
    theta = r_d / k1  # initial guess
    for _ in range(5):
        t2 = theta * theta
        t3 = t2 * theta
        t5 = t3 * t2
        t7 = t5 * t2
        f_val = k1 * theta + k2 * t3 + k3 * t5 + k4 * t7 - r_d
        f_deriv = k1 + 3 * k2 * t2 + 5 * k3 * t2 * t2 + 7 * k4 * t2 * t3
        theta = theta - f_val / (f_deriv + 1e-10)

    # Ray direction in camera coords (optical axis = Z)
    sin_t = np.sin(np.clip(theta, 0, np.pi / 2.5))
    cos_t = np.cos(np.clip(theta, 0, np.pi / 2.5))
    ray_x = sin_t * np.cos(phi)
    ray_y = -sin_t * np.sin(phi)  # negate: image y↓ but world y↑
    ray_z = cos_t

    # Apply pitch rotation around X axis (downward)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    ray_y_rot = ray_y * cp - ray_z * sp
    ray_z_rot = ray_y * sp + ray_z * cp

    # Ground intersection: ray_y_rot < 0 (downward) AND ray_z_rot > 0 (forward)
    _GROUND_MASK = (ray_y_rot < -0.02) & (ray_z_rot > 0)
    return _GROUND_MASK


_CAMERA = None


def _get_camera(fov_scale=1.0):
    global _CAMERA
    if _CAMERA is None:
        json_path = CALIB_DIR / "00000_FV.json"
        _CAMERA = RadialPoly(str(json_path), fov_scale=fov_scale)
    return _CAMERA


# ═══════════════════════════════════════════════════════════════════
def detect_motion_hybrid(prev_gray, curr_gray, morph_ksize=7):
    """Hybrid Flow motion detection.

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

    # ── 4. Hybrid blend ─────────────────────────────────────────
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

    # ── 5-bin params (with HYBRID_PARAMS ratio overrides) ──────
    if p95_diff > 100:
        st, rt, ed = 22, 1.3, 12; use_abs, ft = True, 2.0
        rt += HYBRID_PARAMS.get("ratio_delta_A", 0.0)
    elif p95_diff > 70:
        st, rt, ed = 20, 1.5, 10; use_abs, ft = True, 1.5
        rt += HYBRID_PARAMS.get("ratio_delta_B", 0.0)
    elif p95_diff > 50:
        st, rt, ed = 18, 1.8, 10; use_abs = False
        rt += HYBRID_PARAMS.get("ratio_delta_C", 0.0)
    elif p95_diff > 30:
        st, rt, ed = 16, 2.0, 10; use_abs = False
        rt += HYBRID_PARAMS.get("ratio_delta_D", 0.0)
    else:
        st, rt, ed = 14, 2.2, 4; use_abs = False
        rt += HYBRID_PARAMS.get("ratio_delta_E", 0.0)

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


def temporal_smooth(mask, prev_mask, alpha=0.7):
    """Exponential moving average for temporal consistency.

    Smooths flickering between consecutive frames. Single-frame noise blips
    that appear and disappear are suppressed.
    Args:
        mask: current raw mask (0/255 uint8)
        prev_mask: smoothed mask from previous frame (float [0,255])
        alpha: weight for current frame (0.7 = 70% current, 30% history)
    Returns:
        smoothed mask (float [0,255]) and binary mask (0/255 uint8)
    """
    cur_float = mask.astype(np.float32)
    if prev_mask is None:
        smooth = cur_float
    else:
        smooth = alpha * cur_float + (1.0 - alpha) * prev_mask.astype(np.float32)
    binary = (smooth > 127).astype(np.uint8) * 255
    return smooth, binary


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

            # L4 (experimental): Canny edge density on original image
            # Disabled by default — needs curr_gray passed to _cca to work
            # reliably.  Using diff image doesn't separate buildings from
            # vehicles well enough.
            _ed_max = HYBRID_PARAMS.get("edge_density_max", None)
            if _ed_max is not None and 2000 <= ar <= 50000:
                left, top = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP]
                roi_gray = diff[top:top+ch, left:left+cw]
                edges = cv2.Canny(roi_gray, 40, 120)
                edge_density = edges.sum() / (255 * ar + 1e-6)
                if edge_density > _ed_max:
                    mask[lb == i] = 0
                    continue

            cts, _ = cv2.findContours(cb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sol = 1.0
            if cts:
                hull = cv2.convexHull(np.vstack(cts))
                ha = cv2.contourArea(hull); sol = ar/ha if ha > 0 else 1.0
            if (ar_ > 8 and sol < 0.4) or (100 <= ar <= 5000 and fr < 0.12 and sol < 0.4):
                mask[lb == i] = 0
    return mask


# ═══════════════════════════════════════════════════════════════════
# V2: Adaptive expand + spatial suppression
# ═══════════════════════════════════════════════════════════════════
def detect_motion_hybrid_v2(prev_gray, curr_gray, morph_ksize=7):
    """Hybrid Flow V2: smaller local window + mag×diff saliency + direction check.

    Three key improvements over V1:
      1. mag_ratio window 41→21 (sigma 15→7): less over-amplification in uniform regions
      2. mag×diff joint saliency: requires BOTH elevated flow AND elevated diff
      3. Flow direction consistency enabled for all main pipeline bins
    """
    pg_f = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    cg_f = cv2.GaussianBlur(curr_gray, (5, 5), 0)
    img_h, img_w = pg_f.shape

    diff = cv2.absdiff(cg_f, pg_f)
    diff_f = diff.astype(np.float32)
    p95_diff = np.percentile(diff, 95)
    global_diff_mean = np.mean(diff_f)

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

    # ── 3. Reproject + blend ────────────────────────────────────
    mag_p_reproj = camera.reproject_to_fisheye(mag_p)
    blend_w = _get_blend_map(img_h, img_w)
    mag = blend_w * mag_p_reproj + (1.0 - blend_w) * mag_f

    # ── 4. Joint saliency: mag × diff with local normalization ──
    # Window 21×21 (was 41×41) — less over-amplification in sky/road
    WS, SIG = 21, 7
    saliency = (mag + 0.1) * (diff_f + 0.1)
    saliency_mean = cv2.GaussianBlur(saliency, (WS, WS), SIG)
    saliency_ratio = saliency / (saliency_mean + 0.5)
    global_mag_mean = np.mean(mag)

    # ── 5. V1 fallback — EXACT V12 ──────────────────────────────
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

    # ── 6. 5-bin params (saliency = mag×diff, rt ≈ mag_ratio²) ─
    if p95_diff > 100:
        st, rt, ed = 22, 2.0, 10; use_abs, ft = True, 3.0
    elif p95_diff > 70:
        st, rt, ed = 20, 2.5, 8; use_abs, ft = True, 2.5
    elif p95_diff > 50:
        st, rt, ed = 18, 3.0, 8; use_abs = False
    elif p95_diff > 30:
        st, rt, ed = 16, 4.0, 6; use_abs = False
    else:
        st, rt, ed = 14, 5.0, 4; use_abs = False

    if use_abs:
        candidate = ((mag > ft) & (saliency_ratio > rt)).astype(np.uint8) * 255
    else:
        # Absolute floor: mag must be > 0.5 to prevent saliency amplification
        # in uniform regions (sky/road) where local mean → 0 inflates ratio
        candidate = ((mag > 0.5) & (saliency_ratio > rt)).astype(np.uint8) * 255

    # ── 7. Seed (spatial consistency filter) ────────────────────
    _, seed_raw = cv2.threshold(diff, st, 255, cv2.THRESH_BINARY)
    nk = np.ones((7, 7), np.uint8)
    nc = cv2.filter2D((seed_raw > 0).astype(np.float32), -1, nk)
    mn = 3 if p95_diff > 80 else 4
    seed = seed_raw.copy(); seed[nc < mn] = 0
    if seed.max() == 0: seed = seed_raw

    # ── 8. Weak-signal early exit ───────────────────────────────
    seed_px = np.count_nonzero(seed)
    if seed_px / (img_h * img_w) < 0.0003:
        return np.zeros((img_h, img_w), dtype=np.uint8)

    # ── 9. Constrained iterative dilation (replaces distance-transform) ─
    # Only expands along CONNECTED candidate paths, not arbitrary distance.
    # This prevents noise seeds from reaching disconnected noise candidates.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    current = seed.copy()
    for _ in range(int(ed)):
        dilated = cv2.dilate(current, kernel)
        dilated &= candidate  # constrain to candidate mask
        current |= dilated
    expanded = current & ~seed

    # ── 10. Direction consistency ────────────────────────────────
    seed_bool = (seed > 0)
    if seed_bool.sum() > 20:
        flow_angle = np.arctan2(flow_f[..., 1], flow_f[..., 0])
        seed_angle_map = np.zeros_like(flow_angle)
        seed_angle_map[seed_bool] = flow_angle[seed_bool]
        seed_weight = np.zeros_like(flow_angle)
        seed_weight[seed_bool] = 1.0

        ks = int(ed) * 2 + 1
        seed_angle_blurred = cv2.GaussianBlur(seed_angle_map, (ks, ks), int(ed) / 2.0)
        seed_weight_blurred = cv2.GaussianBlur(seed_weight, (ks, ks), int(ed) / 2.0)
        valid_weight = seed_weight_blurred > 0.02
        seed_angle_blurred[valid_weight] /= seed_weight_blurred[valid_weight]

        angle_diff = np.abs(flow_angle - seed_angle_blurred)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        direction_ok = (angle_diff < np.pi / 4) | ~valid_weight
        expanded[~direction_ok] = 0

    # ── 11. Merge + CCA + morphology ────────────────────────────
    mask = seed | expanded
    mask = _cca_v2(mask, diff, mag, seed, global_diff_mean, global_mag_mean)

    k = np.ones((morph_ksize, morph_ksize), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)


def _cca_v2(mask, diff, mag, seed, gdm, gmm):
    """CCA filter with tighter signal verification than _cca."""
    nl, lb, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, nl):
        ar = st[i, cv2.CC_STAT_AREA]
        cw = st[i, cv2.CC_STAT_WIDTH]; ch = st[i, cv2.CC_STAT_HEIGHT]
        if ar < 120: mask[lb == i] = 0; continue  # was 100
        cp = (lb == i)
        md, mm = np.mean(diff[cp]), np.mean(mag[cp])
        if md/(gdm+1e-6) < 1.8 or mm/(gmm+1e-6) < 1.5: mask[lb == i] = 0; continue  # was 1.5/1.3
        if ar > 200:
            si = np.sum((seed > 0) & cp)
            if si/ar < 0.04: mask[lb == i] = 0; continue  # was 0.03
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


# ═══════════════════════════════════════════════════════════════════
# V3: Structure separation + gradient-flow alignment
# ═══════════════════════════════════════════════════════════════════
def detect_motion_hybrid_v3(prev_gray, curr_gray, morph_ksize=7):
    """Hybrid Flow V3: texture suppression + normal-flow constraint.

    Two key improvements over V2:
      1. Bilateral filter before Farneback removes texture-driven false flow
         on building facades, tree leaves, and road markings.
      2. Normal-flow constraint penalizes flow tangential to strong edges
         (unreliable per optical-flow aperture problem).

    The V2 pipeline (mag×diff saliency, seed expand, CCA) is otherwise
    unchanged — we attack the problem at the signal source, not the
    post-processing.
    """
    pg_f = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    cg_f = cv2.GaussianBlur(curr_gray, (5, 5), 0)
    img_h, img_w = pg_f.shape

    # ── 0. Structure separation (bilateral filter) ─────────────
    # d=9, sigmaColor=50, sigmaSpace=10: smooth texture, keep edges
    pg_struct = cv2.bilateralFilter(pg_f, 9, 50, 10)
    cg_struct = cv2.bilateralFilter(cg_f, 9, 50, 10)

    # Frame diff on structure images (suppresses texture noise)
    diff = cv2.absdiff(cg_struct, pg_struct)
    diff_f = diff.astype(np.float32)
    p95_diff = np.percentile(diff, 95)
    global_diff_mean = np.mean(diff_f)

    # Gradient on ORIGINAL image (real edges, not smoothed)
    grad_x = cv2.Sobel(cg_f, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(cg_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2) + 1e-8

    # ── 1. Fisheye-domain Farneback on STRUCTURE ──────────────
    flow_f = cv2.calcOpticalFlowFarneback(
        pg_struct, cg_struct, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag_f = np.linalg.norm(flow_f, axis=2)

    # ── 2. Perspective-domain Farneback on STRUCTURE ──────────
    camera = _get_camera(fov_scale=1.0)
    pg_p = camera.undistort(pg_struct)
    cg_p = camera.undistort(cg_struct)
    flow_p = cv2.calcOpticalFlowFarneback(
        pg_p, cg_p, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag_p = np.linalg.norm(flow_p, axis=2)
    mag_p_reproj = camera.reproject_to_fisheye(mag_p)

    # ── 3. Gradient-flow alignment (normal-flow constraint) ───
    # Project flow onto gradient direction → reliably-measurable component
    flow_proj = np.abs(flow_f[..., 0] * grad_x + flow_f[..., 1] * grad_y) / grad_mag
    # Alignment ratio: cos(angle between flow and gradient)
    align = flow_proj / (mag_f + 0.01)  # [0, 1], low = tangential / unreliable

    # Gate: strong gradient → use normal flow; weak gradient → keep full flow
    edge_weight = np.clip(grad_mag / 25.0, 0.0, 1.0)
    mag_f_eff = (1.0 - edge_weight) * mag_f + edge_weight * flow_proj

    # Suppression mask for tangential flow on strong edges
    # Flow <40% aligned with gradient on a strong edge → suppress saliency
    suppress = (grad_mag > 15) & (align < 0.4)

    # ── 4. Hybrid blend ───────────────────────────────────────
    blend_w = _get_blend_map(img_h, img_w)
    mag = blend_w * mag_p_reproj + (1.0 - blend_w) * mag_f_eff

    # ── 5. Joint saliency: mag × diff ─────────────────────────
    WS, SIG = 21, 7
    saliency = (mag + 0.1) * (diff_f + 0.1)
    # Penalize tangentially-flowing edge pixels
    saliency[suppress] *= 0.25
    saliency_mean = cv2.GaussianBlur(saliency, (WS, WS), SIG)
    saliency_ratio = saliency / (saliency_mean + 0.5)
    global_mag_mean = np.mean(mag)

    # ── 6. V1 fallback ────────────────────────────────────────
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
        mask = _cca_v2(mask, diff, mag, seed_fb, global_diff_mean, global_mag_mean)
        k = np.ones((morph_ksize, morph_ksize), np.uint8)
        return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)

    # ── 7. 5-bin params ───────────────────────────────────────
    if p95_diff > 100:
        st, rt, ed = 22, 2.0, 10; use_abs, ft = True, 3.0
    elif p95_diff > 70:
        st, rt, ed = 20, 2.5, 8; use_abs, ft = True, 2.5
    elif p95_diff > 50:
        st, rt, ed = 18, 3.0, 8; use_abs = False
    elif p95_diff > 30:
        st, rt, ed = 16, 4.0, 6; use_abs = False
    else:
        st, rt, ed = 14, 5.0, 4; use_abs = False

    if use_abs:
        candidate = ((mag > ft) & (saliency_ratio > rt)).astype(np.uint8) * 255
    else:
        candidate = ((mag > 0.5) & (saliency_ratio > rt)).astype(np.uint8) * 255

    # ── 8. Seed (spatial consistency filter) ──────────────────
    _, seed_raw = cv2.threshold(diff, st, 255, cv2.THRESH_BINARY)
    nk = np.ones((7, 7), np.uint8)
    nc = cv2.filter2D((seed_raw > 0).astype(np.float32), -1, nk)
    mn = 3 if p95_diff > 80 else 4
    seed = seed_raw.copy(); seed[nc < mn] = 0
    if seed.max() == 0: seed = seed_raw

    # ── 9. Weak-signal early exit ─────────────────────────────
    seed_px = np.count_nonzero(seed)
    if seed_px / (img_h * img_w) < 0.0003:
        return np.zeros((img_h, img_w), dtype=np.uint8)

    # ── 10. Constrained iterative dilation ────────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    current = seed.copy()
    for _ in range(int(ed)):
        dilated = cv2.dilate(current, kernel)
        dilated &= candidate
        current |= dilated
    expanded = current & ~seed

    # ── 11. Direction consistency ─────────────────────────────
    seed_bool = (seed > 0)
    if seed_bool.sum() > 20:
        flow_angle = np.arctan2(flow_f[..., 1], flow_f[..., 0])
        seed_angle_map = np.zeros_like(flow_angle)
        seed_angle_map[seed_bool] = flow_angle[seed_bool]
        seed_weight = np.zeros_like(flow_angle)
        seed_weight[seed_bool] = 1.0
        ks = int(ed) * 2 + 1
        seed_angle_blurred = cv2.GaussianBlur(seed_angle_map, (ks, ks), int(ed) / 2.0)
        seed_weight_blurred = cv2.GaussianBlur(seed_weight, (ks, ks), int(ed) / 2.0)
        valid_weight = seed_weight_blurred > 0.02
        seed_angle_blurred[valid_weight] /= seed_weight_blurred[valid_weight]
        angle_diff = np.abs(flow_angle - seed_angle_blurred)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        direction_ok = (angle_diff < np.pi / 4) | ~valid_weight
        expanded[~direction_ok] = 0

    # ── 12. Merge + CCA + morphology ──────────────────────────
    mask = seed | expanded
    mask = _cca_v2(mask, diff, mag, seed, global_diff_mean, global_mag_mean)

    k = np.ones((morph_ksize, morph_ksize), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)


# ═══════════════════════════════════════════════════════════════════
# V4: V3 + per-frame adaptive saliency threshold (robust z-score)
# ═══════════════════════════════════════════════════════════════════
def detect_motion_hybrid_v4(prev_gray, curr_gray, morph_ksize=7):
    """Hybrid Flow V4: V3 + per-frame adaptive threshold via robust z-score.

    Replaces the fixed 5-bin saliency_ratio threshold (rt=2.0~5.0) with
    a per-frame z-score: candidate = (saliency_ratio - median) / IQR > z_thresh.

    This naturally adapts:
      - Low-motion frames: small IQR -> subtle motion peaks still pass
      - High-motion frames: large IQR -> only extreme outliers pass

    Also adapts seed threshold and expansion distance per-frame.
    Keeps V3's structure separation + normal-flow constraint.
    """
    pg_f = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    cg_f = cv2.GaussianBlur(curr_gray, (5, 5), 0)
    img_h, img_w = pg_f.shape

    # ── 0. Structure separation ────────────────────────────────
    pg_struct = cv2.bilateralFilter(pg_f, 9, 50, 10)
    cg_struct = cv2.bilateralFilter(cg_f, 9, 50, 10)
    diff = cv2.absdiff(cg_struct, pg_struct)
    diff_f = diff.astype(np.float32)
    p95_diff = np.percentile(diff, 95)
    global_diff_mean = np.mean(diff_f)

    # Gradient on original
    grad_x = cv2.Sobel(cg_f, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(cg_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2) + 1e-8

    # ── 1. Fisheye Farneback on STRUCTURE ─────────────────────
    flow_f = cv2.calcOpticalFlowFarneback(
        pg_struct, cg_struct, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag_f = np.linalg.norm(flow_f, axis=2)

    # ── 2. Perspective Farneback on STRUCTURE ─────────────────
    camera = _get_camera(fov_scale=1.0)
    pg_p = camera.undistort(pg_struct)
    cg_p = camera.undistort(cg_struct)
    flow_p = cv2.calcOpticalFlowFarneback(
        pg_p, cg_p, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag_p = np.linalg.norm(flow_p, axis=2)
    mag_p_reproj = camera.reproject_to_fisheye(mag_p)

    # ── 3. Normal-flow constraint ─────────────────────────────
    flow_proj = np.abs(flow_f[..., 0] * grad_x + flow_f[..., 1] * grad_y) / grad_mag
    align = flow_proj / (mag_f + 0.01)
    edge_weight = np.clip(grad_mag / 25.0, 0.0, 1.0)
    mag_f_eff = (1.0 - edge_weight) * mag_f + edge_weight * flow_proj
    suppress = (grad_mag > 15) & (align < 0.4)

    # ── 4. Hybrid blend ──────────────────────────────────────
    blend_w = _get_blend_map(img_h, img_w)
    mag = blend_w * mag_p_reproj + (1.0 - blend_w) * mag_f_eff

    # ── 5. Joint saliency ────────────────────────────────────
    WS, SIG = 21, 7
    saliency = (mag + 0.1) * (diff_f + 0.1)
    saliency[suppress] *= 0.25
    saliency_mean = cv2.GaussianBlur(saliency, (WS, WS), SIG)
    saliency_ratio = saliency / (saliency_mean + 0.5)
    global_mag_mean = np.mean(mag)

    # ── 6. V1 fallback (extreme frames) ──────────────────────
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
        mask = _cca_v2(mask, diff, mag, seed_fb, global_diff_mean, global_mag_mean)
        k = np.ones((morph_ksize, morph_ksize), np.uint8)
        return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)

    # ── 7. Per-frame ADAPTIVE threshold (robust z-score) ─────
    # Compute statistics from meaningful saliency values only
    sr_valid = saliency_ratio[saliency_ratio > 0.2]
    if len(sr_valid) > 500:
        sr_median = np.median(sr_valid)
        sr_iqr = np.percentile(sr_valid, 75) - np.percentile(sr_valid, 25)
        sr_iqr = max(sr_iqr, 0.5)
    else:
        sr_median, sr_iqr = 1.0, 1.0

    # Z-score: how many IQRs above the frame's own median
    sr_z = (saliency_ratio - sr_median) / sr_iqr

    # Z-threshold modulated by P95_diff (more motion -> lower threshold)
    if p95_diff > 100:   z_thresh = 1.8
    elif p95_diff > 70:  z_thresh = 2.2
    elif p95_diff > 50:  z_thresh = 2.6
    elif p95_diff > 30:  z_thresh = 3.0
    else:                z_thresh = 3.5

    candidate = ((sr_z > z_thresh) & (mag > 0.5)).astype(np.uint8) * 255

    # ── 8. Adaptive seed ─────────────────────────────────────
    diff_valid = diff_f[diff_f > 5]
    if len(diff_valid) > 500:
        diff_median = np.median(diff_valid)
        diff_iqr = np.percentile(diff_valid, 75) - np.percentile(diff_valid, 25)
        diff_iqr = max(diff_iqr, 5)
    else:
        diff_median, diff_iqr = 10, 10

    diff_z = (diff_f - diff_median) / diff_iqr
    if p95_diff > 100:   diff_z_thresh = 2.0
    elif p95_diff > 70:  diff_z_thresh = 2.5
    elif p95_diff > 50:  diff_z_thresh = 3.0
    elif p95_diff > 30:  diff_z_thresh = 3.5
    else:                diff_z_thresh = 4.0

    _, seed_raw = cv2.threshold(
        (diff_z * 255).clip(0, 255).astype(np.uint8),
        int(diff_z_thresh * 25.5), 255, cv2.THRESH_BINARY)

    # Spatial consistency filter
    nk = np.ones((7, 7), np.uint8)
    nc = cv2.filter2D((seed_raw > 0).astype(np.float32), -1, nk)
    mn = 3 if p95_diff > 80 else 4
    seed = seed_raw.copy(); seed[nc < mn] = 0
    if seed.max() == 0: seed = seed_raw

    # ── 9. Adaptive expansion distance ───────────────────────
    if p95_diff > 100:   ed = 10
    elif p95_diff > 70:  ed = 8
    elif p95_diff > 50:  ed = 6
    elif p95_diff > 30:  ed = 5
    else:                ed = 4

    # ── 10. Weak-signal early exit ───────────────────────────
    seed_px = np.count_nonzero(seed)
    if seed_px / (img_h * img_w) < 0.0003:
        return np.zeros((img_h, img_w), dtype=np.uint8)

    # ── 11. Constrained iterative dilation ───────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    current = seed.copy()
    for _ in range(int(ed)):
        dilated = cv2.dilate(current, kernel)
        dilated &= candidate
        current |= dilated
    expanded = current & ~seed

    # ── 12. Direction consistency ────────────────────────────
    seed_bool = (seed > 0)
    if seed_bool.sum() > 20:
        flow_angle = np.arctan2(flow_f[..., 1], flow_f[..., 0])
        seed_angle_map = np.zeros_like(flow_angle)
        seed_angle_map[seed_bool] = flow_angle[seed_bool]
        seed_weight = np.zeros_like(flow_angle)
        seed_weight[seed_bool] = 1.0
        ks = int(ed) * 2 + 1
        seed_angle_blurred = cv2.GaussianBlur(seed_angle_map, (ks, ks), int(ed) / 2.0)
        seed_weight_blurred = cv2.GaussianBlur(seed_weight, (ks, ks), int(ed) / 2.0)
        valid_weight = seed_weight_blurred > 0.02
        seed_angle_blurred[valid_weight] /= seed_weight_blurred[valid_weight]
        angle_diff = np.abs(flow_angle - seed_angle_blurred)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        direction_ok = (angle_diff < np.pi / 4) | ~valid_weight
        expanded[~direction_ok] = 0

    # ── 13. Merge + CCA + morphology ─────────────────────────
    mask = seed | expanded
    mask = _cca_v2(mask, diff, mag, seed, global_diff_mean, global_mag_mean)
    k = np.ones((morph_ksize, morph_ksize), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)


# ═══════════════════════════════════════════════════════════════════
# V5: Per-frame percentile + local peak detection
# ═══════════════════════════════════════════════════════════════════
def detect_motion_hybrid_v5(prev_gray, curr_gray, morph_ksize=7):
    """Hybrid Flow V5: percentile threshold + local peak constraint.

    Two problems with V3/V4 fixed:
      1. Fixed/z-score thresholds leave low-motion frames empty
         → per-frame percentile ensures every frame gets candidates
      2. Global thresholding picks noise pixels far from real motion
         → local peak constraint: only keep pixels near a local saliency max

    The candidate condition:
      saliency_ratio > P_N (frame-level percentile, N=90-97)
      AND saliency_ratio > local_max * 0.55  (within 55% of nearest local peak)

    Keeps V3's structure separation + normal-flow constraint.
    """
    pg_f = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    cg_f = cv2.GaussianBlur(curr_gray, (5, 5), 0)
    img_h, img_w = pg_f.shape

    # ── 0. Structure separation ────────────────────────────────
    pg_struct = cv2.bilateralFilter(pg_f, 9, 50, 10)
    cg_struct = cv2.bilateralFilter(cg_f, 9, 50, 10)
    diff = cv2.absdiff(cg_struct, pg_struct)
    diff_f = diff.astype(np.float32)
    p95_diff = np.percentile(diff, 95)
    global_diff_mean = np.mean(diff_f)

    # Gradient on original
    grad_x = cv2.Sobel(cg_f, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(cg_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2) + 1e-8

    # ── 1-4. Farneback + normal flow + blend ──────────────────
    flow_f = cv2.calcOpticalFlowFarneback(
        pg_struct, cg_struct, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag_f = np.linalg.norm(flow_f, axis=2)

    camera = _get_camera(fov_scale=1.0)
    pg_p = camera.undistort(pg_struct)
    cg_p = camera.undistort(cg_struct)
    flow_p = cv2.calcOpticalFlowFarneback(
        pg_p, cg_p, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag_p = np.linalg.norm(flow_p, axis=2)
    mag_p_reproj = camera.reproject_to_fisheye(mag_p)

    flow_proj = np.abs(flow_f[..., 0] * grad_x + flow_f[..., 1] * grad_y) / grad_mag
    align = flow_proj / (mag_f + 0.01)
    edge_weight = np.clip(grad_mag / 25.0, 0.0, 1.0)
    mag_f_eff = (1.0 - edge_weight) * mag_f + edge_weight * flow_proj
    suppress = (grad_mag > 15) & (align < 0.4)

    blend_w = _get_blend_map(img_h, img_w)
    mag = blend_w * mag_p_reproj + (1.0 - blend_w) * mag_f_eff

    # ── 5. Joint saliency ────────────────────────────────────
    WS, SIG = 15, 5  # smaller window for sharper local adaptation
    saliency = (mag + 0.1) * (diff_f + 0.1)
    saliency[suppress] *= 0.25
    saliency_mean = cv2.GaussianBlur(saliency, (WS, WS), SIG)
    saliency_ratio = saliency / (saliency_mean + 0.5)
    global_mag_mean = np.mean(mag)

    # ── 6. V1 fallback ──────────────────────────────────────
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
        mask = _cca_v2(mask, diff, mag, seed_fb, global_diff_mean, global_mag_mean)
        k = np.ones((morph_ksize, morph_ksize), np.uint8)
        return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)

    # ── 7. Pure LOCAL PEAK detection ─────────────────────────
    # Eliminates both fixed thresholds AND per-frame percentiles.
    # A pixel passes if it's close to its local saliency_ratio peak.
    # This naturally adapts: low-motion regions have low peaks that
    # still pass (if above absolute floor); high-motion regions have
    # high peaks that set a stricter local bar.
    peak_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    local_max_sr = cv2.dilate(saliency_ratio, peak_kernel)
    peak_factor = 0.28  # must be >= 28% of local max

    # Candidate: near a local peak AND above absolute minimum
    candidate = (
        (saliency_ratio > local_max_sr * peak_factor) &
        (saliency_ratio > 1.5) &
        (mag > 0.5)
    ).astype(np.uint8) * 255

    # ── 8. Seeds also via local peak (on diff) ───────────────
    diff_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    local_max_diff = cv2.dilate(diff_f, diff_kernel)
    diff_peak_factor = 0.30
    _, seed_raw = cv2.threshold(
        ((diff_f > local_max_diff * diff_peak_factor) & (diff_f > 10)).astype(np.uint8) * 255,
        128, 255, cv2.THRESH_BINARY)

    # Spatial consistency
    nk = np.ones((7, 7), np.uint8)
    nc = cv2.filter2D((seed_raw > 0).astype(np.float32), -1, nk)
    mn = 3 if p95_diff > 80 else 4
    seed = seed_raw.copy(); seed[nc < mn] = 0
    if seed.max() == 0: seed = seed_raw

    # ── 9. Adaptive expansion distance ───────────────────────
    if p95_diff > 100:   ed = 10
    elif p95_diff > 70:  ed = 8
    elif p95_diff > 50:  ed = 6
    elif p95_diff > 30:  ed = 5
    else:                ed = 4

    # ── 10. Weak-signal early exit ───────────────────────────
    seed_px = np.count_nonzero(seed)
    if seed_px / (img_h * img_w) < 0.0003:
        return np.zeros((img_h, img_w), dtype=np.uint8)

    # ── 11. Constrained iterative dilation ───────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    current = seed.copy()
    for _ in range(int(ed)):
        dilated = cv2.dilate(current, kernel)
        dilated &= candidate
        current |= dilated
    expanded = current & ~seed

    # ── 12. Direction consistency ────────────────────────────
    seed_bool = (seed > 0)
    if seed_bool.sum() > 20:
        flow_angle = np.arctan2(flow_f[..., 1], flow_f[..., 0])
        seed_angle_map = np.zeros_like(flow_angle)
        seed_angle_map[seed_bool] = flow_angle[seed_bool]
        seed_weight = np.zeros_like(flow_angle)
        seed_weight[seed_bool] = 1.0
        ks = int(ed) * 2 + 1
        seed_angle_blurred = cv2.GaussianBlur(seed_angle_map, (ks, ks), int(ed) / 2.0)
        seed_weight_blurred = cv2.GaussianBlur(seed_weight, (ks, ks), int(ed) / 2.0)
        valid_weight = seed_weight_blurred > 0.02
        seed_angle_blurred[valid_weight] /= seed_weight_blurred[valid_weight]
        angle_diff = np.abs(flow_angle - seed_angle_blurred)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        direction_ok = (angle_diff < np.pi / 4) | ~valid_weight
        expanded[~direction_ok] = 0

    # ── 13. Merge + CCA + morphology ─────────────────────────
    mask = seed | expanded
    mask = _cca_v2(mask, diff, mag, seed, global_diff_mean, global_mag_mean)
    k = np.ones((morph_ksize, morph_ksize), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)


# ═══════════════════════════════════════════════════════════════════
# V6: Tiered edge-correlation constraint (hard filter)
# ═══════════════════════════════════════════════════════════════════
def detect_motion_hybrid_v6(prev_gray, curr_gray, morph_ksize=7):
    """Hybrid Flow V6: tiered gradient-flow alignment filter.

    V3's soft suppression (saliency*=0.25) is replaced with a tiered system:
      - HARD KILL: strong edge + flow >75% tangential → candidate = 0
      - SOFT SUPPRESS: medium edge + flow >55% tangential → saliency *= 0.35
      - EDGE CONFIDENCE: weights saliency by alignment on all edges

    Gradient is computed on the structure image (bilateral filtered) so we
    detect object-level edges, not texture edges.

    Also: smaller saliency window (15→15) for sharper local adaptation.
    """
    pg_f = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    cg_f = cv2.GaussianBlur(curr_gray, (5, 5), 0)
    img_h, img_w = pg_f.shape

    # ── 0. Structure separation ────────────────────────────────
    pg_struct = cv2.bilateralFilter(pg_f, 9, 60, 10)
    cg_struct = cv2.bilateralFilter(cg_f, 9, 60, 10)
    diff = cv2.absdiff(cg_struct, pg_struct)
    diff_f = diff.astype(np.float32)
    p95_diff = np.percentile(diff, 95)
    global_diff_mean = np.mean(diff_f)

    # Gradient on STRUCTURE image (object edges, not texture)
    grad_x = cv2.Sobel(cg_struct, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(cg_struct, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2) + 1e-8

    # ── 1. Fisheye Farneback on STRUCTURE ─────────────────────
    flow_f = cv2.calcOpticalFlowFarneback(
        pg_struct, cg_struct, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag_f = np.linalg.norm(flow_f, axis=2)

    # ── 2. Perspective Farneback on STRUCTURE ─────────────────
    camera = _get_camera(fov_scale=1.0)
    pg_p = camera.undistort(pg_struct)
    cg_p = camera.undistort(cg_struct)
    flow_p = cv2.calcOpticalFlowFarneback(
        pg_p, cg_p, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag_p = np.linalg.norm(flow_p, axis=2)
    mag_p_reproj = camera.reproject_to_fisheye(mag_p)

    # ── 3. Gradient-flow alignment ────────────────────────────
    flow_proj = np.abs(flow_f[..., 0] * grad_x + flow_f[..., 1] * grad_y) / grad_mag
    align = flow_proj / (mag_f + 0.01)  # [0,1], low = tangential = unreliable

    # ── TIER 1: HARD KILL ─────────────────────────────────────
    # Strong edge (>25) + flow >75% tangential → certain noise
    hard_kill = (grad_mag > 25) & (align < 0.25)

    # ── TIER 2: SOFT SUPPRESS ─────────────────────────────────
    # Medium edge (>12) + flow >55% tangential → suspicious
    soft_suppress = (grad_mag > 12) & (align < 0.45) & ~hard_kill

    # ── TIER 3: Normal flow weighting ─────────────────────────
    # On edges, weight flow magnitude by alignment quality
    edge_weight = np.clip(grad_mag / 20.0, 0.0, 1.0)
    mag_f_eff = (1.0 - edge_weight) * mag_f + edge_weight * flow_proj

    # ── 4. Hybrid blend ──────────────────────────────────────
    blend_w = _get_blend_map(img_h, img_w)
    mag = blend_w * mag_p_reproj + (1.0 - blend_w) * mag_f_eff

    # ── 5. Joint saliency with edge confidence ────────────────
    WS, SIG = 15, 5
    # Edge confidence: how trustworthy is the flow at this pixel
    edge_confidence = np.ones_like(grad_mag)
    on_edge = grad_mag > 10
    # On edges, require alignment; off edges, full trust
    edge_confidence[on_edge] = np.clip(align[on_edge] * 2.0, 0.2, 1.0)

    saliency = (mag + 0.1) * (diff_f + 0.1) * edge_confidence
    # Apply tier-2 soft suppression
    saliency[soft_suppress] *= 0.35
    # Apply tier-1 hard kill
    saliency[hard_kill] = 0.0

    saliency_mean = cv2.GaussianBlur(saliency, (WS, WS), SIG)
    saliency_ratio = saliency / (saliency_mean + 0.5)
    global_mag_mean = np.mean(mag)

    # ── 6. V1 fallback ──────────────────────────────────────
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
        mask = _cca_v2(mask, diff, mag, seed_fb, global_diff_mean, global_mag_mean)
        k = np.ones((morph_ksize, morph_ksize), np.uint8)
        return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)

    # ── 7. 5-bin params (V3's approach, kept stable) ──────────
    if p95_diff > 100:
        st, rt, ed = 22, 2.0, 10; use_abs, ft = True, 3.0
    elif p95_diff > 70:
        st, rt, ed = 20, 2.5, 8; use_abs, ft = True, 2.5
    elif p95_diff > 50:
        st, rt, ed = 18, 3.0, 8; use_abs = False
    elif p95_diff > 30:
        st, rt, ed = 16, 4.0, 6; use_abs = False
    else:
        st, rt, ed = 14, 5.0, 4; use_abs = False

    if use_abs:
        candidate = ((mag > ft) & (saliency_ratio > rt)).astype(np.uint8) * 255
    else:
        candidate = ((mag > 0.5) & (saliency_ratio > rt)).astype(np.uint8) * 255

    # Apply hard_kill to candidate mask too
    candidate[hard_kill] = 0

    # ── 8. Seed ───────────────────────────────────────────────
    _, seed_raw = cv2.threshold(diff, st, 255, cv2.THRESH_BINARY)
    nk = np.ones((7, 7), np.uint8)
    nc = cv2.filter2D((seed_raw > 0).astype(np.float32), -1, nk)
    mn = 3 if p95_diff > 80 else 4
    seed = seed_raw.copy(); seed[nc < mn] = 0
    if seed.max() == 0: seed = seed_raw
    # Also hard-kill seeds on unreliable edges
    seed[hard_kill] = 0

    # ── 9. Weak-signal early exit ─────────────────────────────
    seed_px = np.count_nonzero(seed)
    if seed_px / (img_h * img_w) < 0.0003:
        return np.zeros((img_h, img_w), dtype=np.uint8)

    # ── 10. Constrained iterative dilation ────────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    current = seed.copy()
    for _ in range(int(ed)):
        dilated = cv2.dilate(current, kernel)
        dilated &= candidate
        current |= dilated
    expanded = current & ~seed

    # ── 11. Direction consistency ─────────────────────────────
    seed_bool = (seed > 0)
    if seed_bool.sum() > 20:
        flow_angle = np.arctan2(flow_f[..., 1], flow_f[..., 0])
        seed_angle_map = np.zeros_like(flow_angle)
        seed_angle_map[seed_bool] = flow_angle[seed_bool]
        seed_weight = np.zeros_like(flow_angle)
        seed_weight[seed_bool] = 1.0
        ks = int(ed) * 2 + 1
        seed_angle_blurred = cv2.GaussianBlur(seed_angle_map, (ks, ks), int(ed) / 2.0)
        seed_weight_blurred = cv2.GaussianBlur(seed_weight, (ks, ks), int(ed) / 2.0)
        valid_weight = seed_weight_blurred > 0.02
        seed_angle_blurred[valid_weight] /= seed_weight_blurred[valid_weight]
        angle_diff = np.abs(flow_angle - seed_angle_blurred)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        direction_ok = (angle_diff < np.pi / 4) | ~valid_weight
        expanded[~direction_ok] = 0

    # ── 12. Merge + CCA + morphology ──────────────────────────
    mask = seed | expanded
    mask = _cca_v2(mask, diff, mag, seed, global_diff_mean, global_mag_mean)
    k = np.ones((morph_ksize, morph_ksize), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)


# ═══════════════════════════════════════════════════════════════════
# V7: Edge differencing (orthogonal signal dimension)
# ═══════════════════════════════════════════════════════════════════
def detect_motion_hybrid_v7(prev_gray, curr_gray, morph_ksize=7):
    """Hybrid Flow V7: V3 + edge-position differencing.

    Core insight: a building's Canny edges stay in the same place between
    two frames; a car's edges move. By comparing edge positions directly,
    we get a signal that is orthogonal to optical flow magnitude.

    The edge-movement weight is multiplied into the joint saliency, acting
    as a spatial attention mechanism that suppresses static-scene regions.
    """
    pg_f = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    cg_f = cv2.GaussianBlur(curr_gray, (5, 5), 0)
    img_h, img_w = pg_f.shape

    # Structure separation
    pg_struct = cv2.bilateralFilter(pg_f, 9, 50, 10)
    cg_struct = cv2.bilateralFilter(cg_f, 9, 50, 10)
    diff = cv2.absdiff(cg_struct, pg_struct)
    diff_f = diff.astype(np.float32)
    p95_diff = np.percentile(diff, 95)
    global_diff_mean = np.mean(diff_f)

    # Gradient on structure image
    grad_x = cv2.Sobel(cg_struct, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(cg_struct, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2) + 1e-8

    # ═══════════════════════════════════════════════════════════
    # EDGE DIFFERENCING (new orthogonal signal)
    # ═══════════════════════════════════════════════════════════
    # Adaptive Canny thresholds from per-frame gradient statistics
    gm_valid = grad_mag[grad_mag > 1.0]
    gm_median = float(np.median(gm_valid)) if len(gm_valid) > 100 else 20.0
    lo = max(15.0, gm_median * 0.4)
    hi = max(40.0, gm_median * 1.2)

    edges_prev = cv2.Canny(pg_struct, lo, hi)
    edges_curr = cv2.Canny(cg_struct, lo, hi)

    # Distance to nearest prev/curr edge
    dist_to_prev = cv2.distanceTransform(
        (edges_prev == 0).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_curr = cv2.distanceTransform(
        (edges_curr == 0).astype(np.uint8), cv2.DIST_L2, 3)

    # Moved edge: curr edge far from any prev edge
    moved_edge = np.zeros((img_h, img_w), dtype=np.float32)
    moved_edge[(edges_curr > 0) & (dist_to_prev > 3.0)] = 1.0

    # Disappeared edge: prev edge far from any curr edge
    disappeared_edge = np.zeros((img_h, img_w), dtype=np.float32)
    disappeared_edge[(edges_prev > 0) & (dist_to_curr > 3.0)] = 1.0

    # Edge movement density map
    edge_movement = cv2.GaussianBlur(moved_edge + disappeared_edge, (31, 31), 9)
    em_max = edge_movement.max()
    if em_max > 0.001:
        edge_movement /= em_max

    # Static edge density (edges that stayed put)
    static_edge = np.zeros((img_h, img_w), dtype=np.float32)
    static_edge[(edges_curr > 0) & (dist_to_prev < 2.0)] = 1.0
    static_density = cv2.GaussianBlur(static_edge, (31, 31), 9)
    if static_density.max() > 0.001:
        static_density /= static_density.max()

    # Combined edge weight
    edge_weight = 0.5 + 2.5 * edge_movement - 0.4 * static_density
    edge_weight = np.clip(edge_weight, 0.15, 1.5)

    # ═══════════════════════════════════════════════════════════
    # Farneback on structure (same as V3)
    # ═══════════════════════════════════════════════════════════
    flow_f = cv2.calcOpticalFlowFarneback(
        pg_struct, cg_struct, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag_f = np.linalg.norm(flow_f, axis=2)

    camera = _get_camera(fov_scale=1.0)
    pg_p = camera.undistort(pg_struct)
    cg_p = camera.undistort(cg_struct)
    flow_p = cv2.calcOpticalFlowFarneback(
        pg_p, cg_p, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag_p = np.linalg.norm(flow_p, axis=2)
    mag_p_reproj = camera.reproject_to_fisheye(mag_p)

    # Normal-flow constraint
    flow_proj = np.abs(flow_f[..., 0] * grad_x + flow_f[..., 1] * grad_y) / grad_mag
    align = flow_proj / (mag_f + 0.01)
    edge_w = np.clip(grad_mag / 25.0, 0.0, 1.0)
    mag_f_eff = (1.0 - edge_w) * mag_f + edge_w * flow_proj
    suppress_tangent = (grad_mag > 15) & (align < 0.4)

    # Hybrid blend
    blend_w = _get_blend_map(img_h, img_w)
    mag = blend_w * mag_p_reproj + (1.0 - blend_w) * mag_f_eff

    # Joint saliency x EDGE WEIGHT
    WS, SIG = 21, 7
    saliency = (mag + 0.1) * (diff_f + 0.1)
    saliency *= edge_weight  # <-- orthogonal signal!
    saliency[suppress_tangent] *= 0.25
    saliency_mean = cv2.GaussianBlur(saliency, (WS, WS), SIG)
    saliency_ratio = saliency / (saliency_mean + 0.5)
    global_mag_mean = np.mean(mag)

    # V1 fallback
    if p95_diff > 120 or p95_diff < 20:
        sst = 45 if p95_diff > 120 else 20
        sft, sed = (2.0, 12) if p95_diff > 120 else (2.5, 10)
        _, sseed = cv2.threshold(diff, sst, 255, cv2.THRESH_BINARY)
        _, scand = cv2.threshold(mag, sft, 255, cv2.THRESH_BINARY)
        scand = scand.astype(np.uint8)
        sd = cv2.distanceTransform((sseed == 0).astype(np.uint8), cv2.DIST_L2, 5)
        sexp = scand.copy(); sexp[sd > sed] = 0
        smask = sseed | sexp
        smask = _cca_v2(smask, diff, mag, sseed, global_diff_mean, global_mag_mean)
        k = np.ones((morph_ksize, morph_ksize), np.uint8)
        return cv2.morphologyEx(cv2.morphologyEx(smask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)

    # 5-bin params
    if p95_diff > 100:
        st, rt, ed = 22, 2.0, 10; use_abs, ft = True, 3.0
    elif p95_diff > 70:
        st, rt, ed = 20, 2.5, 8; use_abs, ft = True, 2.5
    elif p95_diff > 50:
        st, rt, ed = 18, 3.0, 8; use_abs = False
    elif p95_diff > 30:
        st, rt, ed = 16, 4.0, 6; use_abs = False
    else:
        st, rt, ed = 14, 5.0, 4; use_abs = False

    if use_abs:
        candidate = ((mag > ft) & (saliency_ratio > rt)).astype(np.uint8) * 255
    else:
        candidate = ((mag > 0.5) & (saliency_ratio > rt)).astype(np.uint8) * 255

    # Seed
    _, seed_raw = cv2.threshold(diff, st, 255, cv2.THRESH_BINARY)
    nk = np.ones((7, 7), np.uint8)
    nc = cv2.filter2D((seed_raw > 0).astype(np.float32), -1, nk)
    mn = 3 if p95_diff > 80 else 4
    seed = seed_raw.copy(); seed[nc < mn] = 0
    if seed.max() == 0: seed = seed_raw

    # Weak-signal early exit
    seed_px = np.count_nonzero(seed)
    if seed_px / (img_h * img_w) < 0.0003:
        return np.zeros((img_h, img_w), dtype=np.uint8)

    # Constrained iterative dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    current = seed.copy()
    for _ in range(int(ed)):
        dilated = cv2.dilate(current, kernel)
        dilated &= candidate
        current |= dilated
    expanded = current & ~seed

    # Direction consistency
    seed_bool = (seed > 0)
    if seed_bool.sum() > 20:
        flow_angle = np.arctan2(flow_f[..., 1], flow_f[..., 0])
        seed_angle_map = np.zeros_like(flow_angle)
        seed_angle_map[seed_bool] = flow_angle[seed_bool]
        seed_weight = np.zeros_like(flow_angle)
        seed_weight[seed_bool] = 1.0
        ks = int(ed) * 2 + 1
        seed_angle_blurred = cv2.GaussianBlur(seed_angle_map, (ks, ks), int(ed) / 2.0)
        seed_weight_blurred = cv2.GaussianBlur(seed_weight, (ks, ks), int(ed) / 2.0)
        valid_weight = seed_weight_blurred > 0.02
        seed_angle_blurred[valid_weight] /= seed_weight_blurred[valid_weight]
        angle_diff = np.abs(flow_angle - seed_angle_blurred)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        direction_ok = (angle_diff < np.pi / 4) | ~valid_weight
        expanded[~direction_ok] = 0

    # Merge + CCA + morphology
    mask = seed | expanded
    mask = _cca_v2(mask, diff, mag, seed, global_diff_mean, global_mag_mean)
    k = np.ones((morph_ksize, morph_ksize), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)

# ═══════════════════════════════════════════════════════════════════
# V8: CLAHE contrast enhancement + dual-threshold Canny edge diff
# ═══════════════════════════════════════════════════════════════════
def detect_motion_hybrid_v8(prev_gray, curr_gray, morph_ksize=7):
    """Hybrid Flow V8: V7 + CLAHE for dark-target recovery.

    Dark objects (low contrast) are invisible to Farneback and Canny.
    CLAHE (Contrast Limited Adaptive Histogram Equalization) locally
    normalizes contrast, making dark-region edges detectable.

    Also: dual-threshold Canny (low + high) for edge differencing,
    capturing both low-contrast and normal-contrast edges.
    """
    pg_f = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    cg_f = cv2.GaussianBlur(curr_gray, (5, 5), 0)
    img_h, img_w = pg_f.shape

    # ── 0a. Structure separation ──────────────────────────────
    pg_struct = cv2.bilateralFilter(pg_f, 9, 50, 10)
    cg_struct = cv2.bilateralFilter(cg_f, 9, 50, 10)

    # ── 0b. CLAHE contrast enhancement ────────────────────────
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    pg_enhanced = clahe.apply(pg_struct)
    cg_enhanced = clahe.apply(cg_struct)

    # Diff and gradient on ENHANCED images
    diff = cv2.absdiff(cg_enhanced, pg_enhanced)
    diff_f = diff.astype(np.float32)
    p95_diff = np.percentile(diff, 95)
    global_diff_mean = np.mean(diff_f)

    grad_x = cv2.Sobel(cg_enhanced, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(cg_enhanced, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2) + 1e-8

    # ── 0c. DUAL-THRESHOLD Canny edge differencing ────────────
    gm_valid = grad_mag[grad_mag > 1.0]
    gm_median = float(np.median(gm_valid)) if len(gm_valid) > 100 else 20.0

    # Low threshold: catch dark-object edges
    edges_prev_lo = cv2.Canny(pg_enhanced, max(8, gm_median * 0.2), max(25, gm_median * 0.6))
    edges_curr_lo = cv2.Canny(cg_enhanced, max(8, gm_median * 0.2), max(25, gm_median * 0.6))

    # High threshold: catch normal edges (as in V7)
    edges_prev_hi = cv2.Canny(pg_enhanced, max(15, gm_median * 0.4), max(40, gm_median * 1.2))
    edges_curr_hi = cv2.Canny(cg_enhanced, max(15, gm_median * 0.4), max(40, gm_median * 1.2))

    # Union of both thresholds
    edges_prev = edges_prev_lo | edges_prev_hi
    edges_curr = edges_curr_lo | edges_curr_hi

    dist_to_prev = cv2.distanceTransform(
        (edges_prev == 0).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_curr = cv2.distanceTransform(
        (edges_curr == 0).astype(np.uint8), cv2.DIST_L2, 3)

    moved_edge = np.zeros((img_h, img_w), dtype=np.float32)
    moved_edge[(edges_curr > 0) & (dist_to_prev > 3.0)] = 1.0

    disappeared_edge = np.zeros((img_h, img_w), dtype=np.float32)
    disappeared_edge[(edges_prev > 0) & (dist_to_curr > 3.0)] = 1.0

    edge_movement = cv2.GaussianBlur(moved_edge + disappeared_edge, (31, 31), 9)
    em_max = edge_movement.max()
    if em_max > 0.001: edge_movement /= em_max

    static_edge = np.zeros((img_h, img_w), dtype=np.float32)
    static_edge[(edges_curr > 0) & (dist_to_prev < 2.0)] = 1.0
    static_density = cv2.GaussianBlur(static_edge, (31, 31), 9)
    if static_density.max() > 0.001: static_density /= static_density.max()

    edge_weight = 0.5 + 2.5 * edge_movement - 0.4 * static_density
    edge_weight = np.clip(edge_weight, 0.15, 1.5)

    # ── 1. Fisheye Farneback on ENHANCED ──────────────────────
    flow_f = cv2.calcOpticalFlowFarneback(
        pg_enhanced, cg_enhanced, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag_f = np.linalg.norm(flow_f, axis=2)

    # ── 2. Perspective Farneback on ENHANCED ──────────────────
    camera = _get_camera(fov_scale=1.0)
    pg_p = camera.undistort(pg_enhanced)
    cg_p = camera.undistort(cg_enhanced)
    flow_p = cv2.calcOpticalFlowFarneback(
        pg_p, cg_p, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag_p = np.linalg.norm(flow_p, axis=2)
    mag_p_reproj = camera.reproject_to_fisheye(mag_p)

    # ── 3. Normal-flow constraint ─────────────────────────────
    flow_proj = np.abs(flow_f[..., 0] * grad_x + flow_f[..., 1] * grad_y) / grad_mag
    align = flow_proj / (mag_f + 0.01)
    edge_w = np.clip(grad_mag / 25.0, 0.0, 1.0)
    mag_f_eff = (1.0 - edge_w) * mag_f + edge_w * flow_proj
    suppress_tangent = (grad_mag > 15) & (align < 0.4)

    # ── 4. Hybrid blend ──────────────────────────────────────
    blend_w = _get_blend_map(img_h, img_w)
    mag = blend_w * mag_p_reproj + (1.0 - blend_w) * mag_f_eff

    # ── 5. Saliency x EDGE WEIGHT ────────────────────────────
    WS, SIG = 21, 7
    saliency = (mag + 0.1) * (diff_f + 0.1)
    saliency *= edge_weight
    saliency[suppress_tangent] *= 0.25
    saliency_mean = cv2.GaussianBlur(saliency, (WS, WS), SIG)
    saliency_ratio = saliency / (saliency_mean + 0.5)
    global_mag_mean = np.mean(mag)

    # ── 6. V1 fallback ──────────────────────────────────────
    if p95_diff > 120 or p95_diff < 20:
        sst = 45 if p95_diff > 120 else 20
        sft, sed = (2.0, 12) if p95_diff > 120 else (2.5, 10)
        _, sseed = cv2.threshold(diff, sst, 255, cv2.THRESH_BINARY)
        _, scand = cv2.threshold(mag, sft, 255, cv2.THRESH_BINARY)
        scand = scand.astype(np.uint8)
        sd = cv2.distanceTransform((sseed == 0).astype(np.uint8), cv2.DIST_L2, 5)
        sexp = scand.copy(); sexp[sd > sed] = 0
        smask = sseed | sexp
        smask = _cca_v2(smask, diff, mag, sseed, global_diff_mean, global_mag_mean)
        k = np.ones((morph_ksize, morph_ksize), np.uint8)
        return cv2.morphologyEx(cv2.morphologyEx(smask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)

    # ── 7. 5-bin params ─────────────────────────────────────
    if p95_diff > 100:
        st, rt, ed = 22, 2.0, 10; use_abs, ft = True, 3.0
    elif p95_diff > 70:
        st, rt, ed = 20, 2.5, 8; use_abs, ft = True, 2.5
    elif p95_diff > 50:
        st, rt, ed = 18, 3.0, 8; use_abs = False
    elif p95_diff > 30:
        st, rt, ed = 16, 4.0, 6; use_abs = False
    else:
        st, rt, ed = 14, 5.0, 4; use_abs = False

    if use_abs:
        candidate = ((mag > ft) & (saliency_ratio > rt)).astype(np.uint8) * 255
    else:
        candidate = ((mag > 0.5) & (saliency_ratio > rt)).astype(np.uint8) * 255

    # Seed on enhanced diff
    _, seed_raw = cv2.threshold(diff, st, 255, cv2.THRESH_BINARY)
    nk = np.ones((7, 7), np.uint8)
    nc = cv2.filter2D((seed_raw > 0).astype(np.float32), -1, nk)
    mn = 3 if p95_diff > 80 else 4
    seed = seed_raw.copy(); seed[nc < mn] = 0
    if seed.max() == 0: seed = seed_raw

    seed_px = np.count_nonzero(seed)
    if seed_px / (img_h * img_w) < 0.0003:
        return np.zeros((img_h, img_w), dtype=np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    current = seed.copy()
    for _ in range(int(ed)):
        dilated = cv2.dilate(current, kernel)
        dilated &= candidate
        current |= dilated
    expanded = current & ~seed

    seed_bool = (seed > 0)
    if seed_bool.sum() > 20:
        flow_angle = np.arctan2(flow_f[..., 1], flow_f[..., 0])
        seed_angle_map = np.zeros_like(flow_angle)
        seed_angle_map[seed_bool] = flow_angle[seed_bool]
        seed_weight = np.zeros_like(flow_angle)
        seed_weight[seed_bool] = 1.0
        ks = int(ed) * 2 + 1
        seed_angle_blurred = cv2.GaussianBlur(seed_angle_map, (ks, ks), int(ed) / 2.0)
        seed_weight_blurred = cv2.GaussianBlur(seed_weight, (ks, ks), int(ed) / 2.0)
        valid_weight = seed_weight_blurred > 0.02
        seed_angle_blurred[valid_weight] /= seed_weight_blurred[valid_weight]
        angle_diff = np.abs(flow_angle - seed_angle_blurred)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        direction_ok = (angle_diff < np.pi / 4) | ~valid_weight
        expanded[~direction_ok] = 0

    mask = seed | expanded
    mask = _cca_v2(mask, diff, mag, seed, global_diff_mean, global_mag_mean)
    k = np.ones((morph_ksize, morph_ksize), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)

# ═══════════════════════════════════════════════════════════════════
# V8b: CLAHE for seed generation only (surgical dark-target fix)
# ═══════════════════════════════════════════════════════════════════
def detect_motion_hybrid_v8b(prev_gray, curr_gray, morph_ksize=7):
    """Hybrid Flow V8b: V7 + CLAHE-enhanced seeds for dark targets.

    Dark targets fail because diff values in dark regions are too low to
    pass the seed threshold. CLAHE boosts contrast locally, making dark
    region diffs detectable.

    Crucially, CLAHE is ONLY used for seed generation. Farneback and edge
    differencing still use the original structure images, avoiding the
    noise amplification that sank V8.
    """
    pg_f = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    cg_f = cv2.GaussianBlur(curr_gray, (5, 5), 0)
    img_h, img_w = pg_f.shape

    # ── 0. Structure separation ──────────────────────────────
    pg_struct = cv2.bilateralFilter(pg_f, 9, 50, 10)
    cg_struct = cv2.bilateralFilter(cg_f, 9, 50, 10)

    # Diff on original structure
    diff = cv2.absdiff(cg_struct, pg_struct)
    diff_f = diff.astype(np.float32)
    p95_diff = np.percentile(diff, 95)
    global_diff_mean = np.mean(diff_f)

    # ── CLAHE-enhanced images for SEED ONLY ──────────────────
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    pg_enhanced = clahe.apply(pg_struct)
    cg_enhanced = clahe.apply(cg_struct)
    diff_enhanced = cv2.absdiff(cg_enhanced, pg_enhanced)

    # Gradient on original structure
    grad_x = cv2.Sobel(cg_struct, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(cg_struct, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2) + 1e-8

    # ── Edge differencing (same as V7) ──────────────────────
    gm_valid = grad_mag[grad_mag > 1.0]
    gm_median = float(np.median(gm_valid)) if len(gm_valid) > 100 else 20.0
    lo = max(15.0, gm_median * 0.4)
    hi = max(40.0, gm_median * 1.2)

    edges_prev = cv2.Canny(pg_struct, lo, hi)
    edges_curr = cv2.Canny(cg_struct, lo, hi)

    dist_to_prev = cv2.distanceTransform(
        (edges_prev == 0).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_curr = cv2.distanceTransform(
        (edges_curr == 0).astype(np.uint8), cv2.DIST_L2, 3)

    moved_edge = np.zeros((img_h, img_w), dtype=np.float32)
    moved_edge[(edges_curr > 0) & (dist_to_prev > 3.0)] = 1.0
    disappeared_edge = np.zeros((img_h, img_w), dtype=np.float32)
    disappeared_edge[(edges_prev > 0) & (dist_to_curr > 3.0)] = 1.0

    edge_movement = cv2.GaussianBlur(moved_edge + disappeared_edge, (31, 31), 9)
    em_max = edge_movement.max()
    if em_max > 0.001: edge_movement /= em_max

    static_edge = np.zeros((img_h, img_w), dtype=np.float32)
    static_edge[(edges_curr > 0) & (dist_to_prev < 2.0)] = 1.0
    static_density = cv2.GaussianBlur(static_edge, (31, 31), 9)
    if static_density.max() > 0.001: static_density /= static_density.max()

    edge_weight = 0.5 + 2.5 * edge_movement - 0.4 * static_density
    edge_weight = np.clip(edge_weight, 0.15, 1.5)

    # ── Farneback on ORIGINAL structure (V7) ────────────────
    flow_f = cv2.calcOpticalFlowFarneback(
        pg_struct, cg_struct, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag_f = np.linalg.norm(flow_f, axis=2)

    camera = _get_camera(fov_scale=1.0)
    pg_p = camera.undistort(pg_struct)
    cg_p = camera.undistort(cg_struct)
    flow_p = cv2.calcOpticalFlowFarneback(
        pg_p, cg_p, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag_p = np.linalg.norm(flow_p, axis=2)
    mag_p_reproj = camera.reproject_to_fisheye(mag_p)

    # Normal-flow constraint
    flow_proj = np.abs(flow_f[..., 0] * grad_x + flow_f[..., 1] * grad_y) / grad_mag
    align = flow_proj / (mag_f + 0.01)
    edge_w = np.clip(grad_mag / 25.0, 0.0, 1.0)
    mag_f_eff = (1.0 - edge_w) * mag_f + edge_w * flow_proj
    suppress_tangent = (grad_mag > 15) & (align < 0.4)

    blend_w = _get_blend_map(img_h, img_w)
    mag = blend_w * mag_p_reproj + (1.0 - blend_w) * mag_f_eff

    # Saliency x edge weight (same as V7)
    WS, SIG = 21, 7
    saliency = (mag + 0.1) * (diff_f + 0.1)
    saliency *= edge_weight
    saliency[suppress_tangent] *= 0.25
    saliency_mean = cv2.GaussianBlur(saliency, (WS, WS), SIG)
    saliency_ratio = saliency / (saliency_mean + 0.5)
    global_mag_mean = np.mean(mag)

    # V1 fallback
    if p95_diff > 120 or p95_diff < 20:
        sst = 45 if p95_diff > 120 else 20
        sft, sed = (2.0, 12) if p95_diff > 120 else (2.5, 10)
        _, sseed = cv2.threshold(diff_enhanced, sst, 255, cv2.THRESH_BINARY)
        _, scand = cv2.threshold(mag, sft, 255, cv2.THRESH_BINARY)
        scand = scand.astype(np.uint8)
        sd = cv2.distanceTransform((sseed == 0).astype(np.uint8), cv2.DIST_L2, 5)
        sexp = scand.copy(); sexp[sd > sed] = 0
        smask = sseed | sexp
        smask = _cca_v2(smask, diff, mag, sseed, global_diff_mean, global_mag_mean)
        k = np.ones((morph_ksize, morph_ksize), np.uint8)
        return cv2.morphologyEx(cv2.morphologyEx(smask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)

    # 5-bin params
    if p95_diff > 100:
        st, rt, ed = 22, 2.0, 10; use_abs, ft = True, 3.0
    elif p95_diff > 70:
        st, rt, ed = 20, 2.5, 8; use_abs, ft = True, 2.5
    elif p95_diff > 50:
        st, rt, ed = 18, 3.0, 8; use_abs = False
    elif p95_diff > 30:
        st, rt, ed = 16, 4.0, 6; use_abs = False
    else:
        st, rt, ed = 14, 5.0, 4; use_abs = False

    if use_abs:
        candidate = ((mag > ft) & (saliency_ratio > rt)).astype(np.uint8) * 255
    else:
        candidate = ((mag > 0.5) & (saliency_ratio > rt)).astype(np.uint8) * 255

    # ── SEED on CLAHE-ENHANCED diff ──────────────────────────
    _, seed_raw = cv2.threshold(diff_enhanced, st, 255, cv2.THRESH_BINARY)
    nk = np.ones((7, 7), np.uint8)
    nc = cv2.filter2D((seed_raw > 0).astype(np.float32), -1, nk)
    mn = 3 if p95_diff > 80 else 4
    seed = seed_raw.copy(); seed[nc < mn] = 0
    if seed.max() == 0: seed = seed_raw

    seed_px = np.count_nonzero(seed)
    if seed_px / (img_h * img_w) < 0.0003:
        return np.zeros((img_h, img_w), dtype=np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    current = seed.copy()
    for _ in range(int(ed)):
        dilated = cv2.dilate(current, kernel)
        dilated &= candidate
        current |= dilated
    expanded = current & ~seed

    seed_bool = (seed > 0)
    if seed_bool.sum() > 20:
        flow_angle = np.arctan2(flow_f[..., 1], flow_f[..., 0])
        seed_angle_map = np.zeros_like(flow_angle)
        seed_angle_map[seed_bool] = flow_angle[seed_bool]
        seed_weight = np.zeros_like(flow_angle)
        seed_weight[seed_bool] = 1.0
        ks = int(ed) * 2 + 1
        seed_angle_blurred = cv2.GaussianBlur(seed_angle_map, (ks, ks), int(ed) / 2.0)
        seed_weight_blurred = cv2.GaussianBlur(seed_weight, (ks, ks), int(ed) / 2.0)
        valid_weight = seed_weight_blurred > 0.02
        seed_angle_blurred[valid_weight] /= seed_weight_blurred[valid_weight]
        angle_diff = np.abs(flow_angle - seed_angle_blurred)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        direction_ok = (angle_diff < np.pi / 4) | ~valid_weight
        expanded[~direction_ok] = 0

    mask = seed | expanded
    mask = _cca_v2(mask, diff, mag, seed, global_diff_mean, global_mag_mean)
    k = np.ones((morph_ksize, morph_ksize), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)

# ═══════════════════════════════════════════════════════════════════
# V9: V7 + local contrast normalization + seed direction filter
# ═══════════════════════════════════════════════════════════════════
def detect_motion_hybrid_v9(prev_gray, curr_gray, morph_ksize=7):
    """Hybrid Flow V9: V7 + two brightness-invariant improvements.

    1. Local contrast normalization: gradient / local_brightness amplifies
       dark-region gradients without changing the image domain. This makes
       the normal-flow constraint equally effective in dark and bright areas.

    2. Seed direction filter: removes seeds whose 7x7 neighborhood has
       high flow-direction variance (noise) vs consistent direction (motion).
       Uses circular variance on flow angles.
    """
    pg_f = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    cg_f = cv2.GaussianBlur(curr_gray, (5, 5), 0)
    img_h, img_w = pg_f.shape

    # ── 0a. Structure separation ──────────────────────────────
    pg_struct = cv2.bilateralFilter(pg_f, 9, 50, 10)
    cg_struct = cv2.bilateralFilter(cg_f, 9, 50, 10)
    diff = cv2.absdiff(cg_struct, pg_struct)
    diff_f = diff.astype(np.float32)
    p95_diff = np.percentile(diff, 95)
    global_diff_mean = np.mean(diff_f)

    # Local brightness for contrast normalization
    local_brightness = cv2.GaussianBlur(cg_struct.astype(np.float32), (21, 21), 7)
    brightness_norm = local_brightness + 0.5

    # Gradient on structure image
    grad_x = cv2.Sobel(cg_struct, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(cg_struct, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2) + 1e-8

    # ── 0b. LOCAL CONTRAST NORMALIZATION ──────────────────────
    # Dark regions get proportionally stronger gradient signal
    grad_mag_norm = grad_mag / brightness_norm

    # ── 0c. Edge differencing (V7's approach, uses raw gradient) ─
    gm_valid_raw = grad_mag[grad_mag > 1.0]
    gm_median_raw = float(np.median(gm_valid_raw)) if len(gm_valid_raw) > 100 else 20.0
    lo = max(15.0, gm_median_raw * 0.4)
    hi = max(40.0, gm_median_raw * 1.2)

    edges_prev = cv2.Canny(pg_struct, lo, hi)
    edges_curr = cv2.Canny(cg_struct, lo, hi)

    dist_to_prev = cv2.distanceTransform(
        (edges_prev == 0).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_curr = cv2.distanceTransform(
        (edges_curr == 0).astype(np.uint8), cv2.DIST_L2, 3)

    moved_edge = np.zeros((img_h, img_w), dtype=np.float32)
    moved_edge[(edges_curr > 0) & (dist_to_prev > 3.0)] = 1.0
    disappeared_edge = np.zeros((img_h, img_w), dtype=np.float32)
    disappeared_edge[(edges_prev > 0) & (dist_to_curr > 3.0)] = 1.0

    edge_movement = cv2.GaussianBlur(moved_edge + disappeared_edge, (31, 31), 9)
    em_max = edge_movement.max()
    if em_max > 0.001: edge_movement /= em_max

    static_edge = np.zeros((img_h, img_w), dtype=np.float32)
    static_edge[(edges_curr > 0) & (dist_to_prev < 2.0)] = 1.0
    static_density = cv2.GaussianBlur(static_edge, (31, 31), 9)
    if static_density.max() > 0.001: static_density /= static_density.max()

    edge_weight = 0.5 + 2.5 * edge_movement - 0.4 * static_density
    edge_weight = np.clip(edge_weight, 0.15, 1.5)

    # ── 1. Fisheye Farneback on STRUCTURE ─────────────────────
    flow_f = cv2.calcOpticalFlowFarneback(
        pg_struct, cg_struct, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag_f = np.linalg.norm(flow_f, axis=2)

    # ── 2. Perspective Farneback ──────────────────────────────
    camera = _get_camera(fov_scale=1.0)
    pg_p = camera.undistort(pg_struct)
    cg_p = camera.undistort(cg_struct)
    flow_p = cv2.calcOpticalFlowFarneback(
        pg_p, cg_p, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag_p = np.linalg.norm(flow_p, axis=2)
    mag_p_reproj = camera.reproject_to_fisheye(mag_p)

    # ── 3. Normal-flow constraint (uses NORMALIZED gradient) ──
    flow_proj = np.abs(flow_f[..., 0] * grad_x + flow_f[..., 1] * grad_y) / grad_mag
    align = flow_proj / (mag_f + 0.01)

    # Edge weight now uses normalized gradient → dark edges get full weight
    edge_w = np.clip(grad_mag_norm / 0.08, 0.0, 1.0)
    mag_f_eff = (1.0 - edge_w) * mag_f + edge_w * flow_proj

    suppress_tangent = (grad_mag_norm > 0.04) & (align < 0.4)

    # ── 4. Hybrid blend ──────────────────────────────────────
    blend_w = _get_blend_map(img_h, img_w)
    mag = blend_w * mag_p_reproj + (1.0 - blend_w) * mag_f_eff

    # ── 5. Saliency x edge weight ────────────────────────────
    WS, SIG = 21, 7
    saliency = (mag + 0.1) * (diff_f + 0.1)
    saliency *= edge_weight
    saliency[suppress_tangent] *= 0.25
    saliency_mean = cv2.GaussianBlur(saliency, (WS, WS), SIG)
    saliency_ratio = saliency / (saliency_mean + 0.5)
    global_mag_mean = np.mean(mag)

    # ── 6. V1 fallback ──────────────────────────────────────
    if p95_diff > 120 or p95_diff < 20:
        sst = 45 if p95_diff > 120 else 20
        sft, sed = (2.0, 12) if p95_diff > 120 else (2.5, 10)
        _, sseed = cv2.threshold(diff, sst, 255, cv2.THRESH_BINARY)
        _, scand = cv2.threshold(mag, sft, 255, cv2.THRESH_BINARY)
        scand = scand.astype(np.uint8)
        sd = cv2.distanceTransform((sseed == 0).astype(np.uint8), cv2.DIST_L2, 5)
        sexp = scand.copy(); sexp[sd > sed] = 0
        smask = sseed | sexp
        smask = _cca_v2(smask, diff, mag, sseed, global_diff_mean, global_mag_mean)
        k = np.ones((morph_ksize, morph_ksize), np.uint8)
        return cv2.morphologyEx(cv2.morphologyEx(smask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)

    # ── 7. 5-bin params ─────────────────────────────────────
    if p95_diff > 100:
        st, rt, ed = 22, 2.0, 10; use_abs, ft = True, 3.0
    elif p95_diff > 70:
        st, rt, ed = 20, 2.5, 8; use_abs, ft = True, 2.5
    elif p95_diff > 50:
        st, rt, ed = 18, 3.0, 8; use_abs = False
    elif p95_diff > 30:
        st, rt, ed = 16, 4.0, 6; use_abs = False
    else:
        st, rt, ed = 14, 5.0, 4; use_abs = False

    if use_abs:
        candidate = ((mag > ft) & (saliency_ratio > rt)).astype(np.uint8) * 255
    else:
        candidate = ((mag > 0.5) & (saliency_ratio > rt)).astype(np.uint8) * 255

    # ── 8. Seed generation ──────────────────────────────────
    _, seed_raw = cv2.threshold(diff, st, 255, cv2.THRESH_BINARY)

    # Spatial consistency filter
    nk = np.ones((7, 7), np.uint8)
    nc = cv2.filter2D((seed_raw > 0).astype(np.float32), -1, nk)
    mn = 3 if p95_diff > 80 else 4
    seed = seed_raw.copy(); seed[nc < mn] = 0
    if seed.max() == 0: seed = seed_raw

    # ── 8b. SEED DIRECTION VARIANCE FILTER ────────────────────
    # Compute circular variance of flow directions in 7x7 window.
    # R=1 = all same direction; R=0 = completely random.
    seed_bool_init = (seed > 0)
    if seed_bool_init.sum() > 10:
        flow_angle = np.arctan2(flow_f[..., 1], flow_f[..., 0])
        cos_angle = np.cos(flow_angle)
        sin_angle = np.sin(flow_angle)
        cos_local = cv2.blur(cos_angle, (7, 7))
        sin_local = cv2.blur(sin_angle, (7, 7))
        R = np.sqrt(cos_local ** 2 + sin_local ** 2)  # [0, 1] direction consistency
        # Require moderate direction consistency (R > 0.55)
        seed_dir_ok = R > 0.55
        seed[seed_bool_init & ~seed_dir_ok] = 0
        if seed.max() == 0:
            seed = seed_raw  # fallback to unfiltered

    # ── 9. Weak-signal early exit ────────────────────────────
    seed_px = np.count_nonzero(seed)
    if seed_px / (img_h * img_w) < 0.0003:
        return np.zeros((img_h, img_w), dtype=np.uint8)

    # ── 10. Constrained iterative dilation ───────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    current = seed.copy()
    for _ in range(int(ed)):
        dilated = cv2.dilate(current, kernel)
        dilated &= candidate
        current |= dilated
    expanded = current & ~seed

    # ── 11. Direction consistency (same as V7) ───────────────
    seed_bool = (seed > 0)
    if seed_bool.sum() > 20:
        flow_angle = np.arctan2(flow_f[..., 1], flow_f[..., 0])
        seed_angle_map = np.zeros_like(flow_angle)
        seed_angle_map[seed_bool] = flow_angle[seed_bool]
        seed_weight = np.zeros_like(flow_angle)
        seed_weight[seed_bool] = 1.0
        ks = int(ed) * 2 + 1
        seed_angle_blurred = cv2.GaussianBlur(seed_angle_map, (ks, ks), int(ed) / 2.0)
        seed_weight_blurred = cv2.GaussianBlur(seed_weight, (ks, ks), int(ed) / 2.0)
        valid_weight = seed_weight_blurred > 0.02
        seed_angle_blurred[valid_weight] /= seed_weight_blurred[valid_weight]
        angle_diff = np.abs(flow_angle - seed_angle_blurred)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        direction_ok = (angle_diff < np.pi / 4) | ~valid_weight
        expanded[~direction_ok] = 0

    # ── 12. Merge + CCA + morphology ─────────────────────────
    mask = seed | expanded
    mask = _cca_v2(mask, diff, mag, seed, global_diff_mean, global_mag_mean)
    k = np.ones((morph_ksize, morph_ksize), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)

# ═══════════════════════════════════════════════════════════════════
# V10: V7 + multi-scale saliency consensus + seed direction filter
# ═══════════════════════════════════════════════════════════════════
def detect_motion_hybrid_v10(prev_gray, curr_gray, morph_ksize=7):
    """Hybrid Flow V10: V7 + multi-scale consensus + seed filter
       + brightness-adaptive thresholds + ground plane mask."""
    pg_f = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    cg_f = cv2.GaussianBlur(curr_gray, (5, 5), 0)
    img_h, img_w = pg_f.shape

    # ── 0a. Structure separation ──────────────────────────────
    pg_struct = cv2.bilateralFilter(pg_f, 9, 50, 10)
    cg_struct = cv2.bilateralFilter(cg_f, 9, 50, 10)
    diff = cv2.absdiff(cg_struct, pg_struct)
    diff_f = diff.astype(np.float32)
    p95_diff = np.percentile(diff, 95)
    global_diff_mean = np.mean(diff_f)

    # Local brightness for adaptive thresholds (dark regions get lower thresholds)
    local_brightness = cv2.GaussianBlur(cg_struct.astype(np.float32), (51, 51), 20)
    darkness = np.clip(1.0 - local_brightness / 50.0, 0.0, 0.55)

    grad_x = cv2.Sobel(cg_struct, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(cg_struct, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2) + 1e-8

    # ── 0b. Edge differencing (V7) ────────────────────────────
    gm_valid = grad_mag[grad_mag > 1.0]
    gm_median = float(np.median(gm_valid)) if len(gm_valid) > 100 else 20.0
    lo, hi = max(15.0, gm_median * 0.4), max(40.0, gm_median * 1.2)
    edges_prev = cv2.Canny(pg_struct, lo, hi)
    edges_curr = cv2.Canny(cg_struct, lo, hi)

    dist_to_prev = cv2.distanceTransform((edges_prev == 0).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_curr = cv2.distanceTransform((edges_curr == 0).astype(np.uint8), cv2.DIST_L2, 3)

    moved_edge = np.zeros((img_h, img_w), dtype=np.float32)
    moved_edge[(edges_curr > 0) & (dist_to_prev > 3.0)] = 1.0
    disappeared_edge = np.zeros((img_h, img_w), dtype=np.float32)
    disappeared_edge[(edges_prev > 0) & (dist_to_curr > 3.0)] = 1.0

    edge_movement = cv2.GaussianBlur(moved_edge + disappeared_edge, (31, 31), 9)
    em_max = edge_movement.max()
    if em_max > 0.001: edge_movement /= em_max

    static_edge = np.zeros((img_h, img_w), dtype=np.float32)
    static_edge[(edges_curr > 0) & (dist_to_prev < 2.0)] = 1.0
    static_density = cv2.GaussianBlur(static_edge, (31, 31), 9)
    if static_density.max() > 0.001: static_density /= static_density.max()

    edge_weight = 0.5 + 2.5 * edge_movement - 0.4 * static_density
    edge_weight = np.clip(edge_weight, 0.15, 1.5)

    # ── 1-4. Farneback + normal flow + blend (V7) ────────────
    flow_f = cv2.calcOpticalFlowFarneback(pg_struct, cg_struct, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag_f = np.linalg.norm(flow_f, axis=2)

    camera = _get_camera(fov_scale=1.0)
    pg_p = camera.undistort(pg_struct)
    cg_p = camera.undistort(cg_struct)
    flow_p = cv2.calcOpticalFlowFarneback(pg_p, cg_p, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag_p = np.linalg.norm(flow_p, axis=2)
    mag_p_reproj = camera.reproject_to_fisheye(mag_p)

    flow_proj = np.abs(flow_f[..., 0] * grad_x + flow_f[..., 1] * grad_y) / grad_mag
    align = flow_proj / (mag_f + 0.01)
    edge_w = np.clip(grad_mag / 25.0, 0.0, 1.0)
    mag_f_eff = (1.0 - edge_w) * mag_f + edge_w * flow_proj
    suppress_tangent = (grad_mag > 15) & (align < 0.4)

    blend_w = _get_blend_map(img_h, img_w)
    mag = blend_w * mag_p_reproj + (1.0 - blend_w) * mag_f_eff

    # ── 5. MULTI-SCALE saliency with edge weight ──────────────
    saliency = (mag + 0.1) * (diff_f + 0.1)
    saliency *= edge_weight
    saliency[suppress_tangent] *= 0.25

    # Three scales
    saliency_ratios = []
    for ws, sg in [(11, 3), (21, 7), (41, 15)]:
        sm = cv2.GaussianBlur(saliency, (ws, ws), sg)
        saliency_ratios.append(saliency / (sm + 0.5))

    # Multi-scale consensus: take MINIMUM across scales
    # A pixel only gets a high value if it's high at ALL scales
    saliency_ratio = np.minimum(np.minimum(saliency_ratios[0], saliency_ratios[1]), saliency_ratios[2])
    global_mag_mean = np.mean(mag)

    # ── 6. V1 fallback ──────────────────────────────────────
    if p95_diff > 120 or p95_diff < 20:
        sst = 45 if p95_diff > 120 else 20
        sft, sed = (2.0, 12) if p95_diff > 120 else (2.5, 10)
        _, sseed = cv2.threshold(diff, sst, 255, cv2.THRESH_BINARY)
        _, scand = cv2.threshold(mag, sft, 255, cv2.THRESH_BINARY)
        scand = scand.astype(np.uint8)
        sd = cv2.distanceTransform((sseed == 0).astype(np.uint8), cv2.DIST_L2, 5)
        sexp = scand.copy(); sexp[sd > sed] = 0
        smask = sseed | sexp
        smask = _cca_v2(smask, diff, mag, sseed, global_diff_mean, global_mag_mean)
        k = np.ones((morph_ksize, morph_ksize), np.uint8)
        return cv2.morphologyEx(cv2.morphologyEx(smask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)

    # ── 7. 5-bin params ─────────────────────────────────────
    if p95_diff > 100:
        st, rt, ed = 22, 2.0, 10; use_abs, ft = True, 3.0
    elif p95_diff > 70:
        st, rt, ed = 20, 2.5, 8; use_abs, ft = True, 2.5
    elif p95_diff > 50:
        st, rt, ed = 18, 3.0, 8; use_abs = False
    elif p95_diff > 30:
        st, rt, ed = 16, 4.0, 6; use_abs = False
    else:
        st, rt, ed = 14, 5.0, 4; use_abs = False

    # Adaptive thresholds: lower in dark regions
    rt_map = rt * (1.0 - darkness * 0.35)
    st_map = st * (1.0 - darkness * 0.45)

    # Ground plane mask: raise threshold on road surface (camera extrinsic)
    ground_mask = _get_ground_mask(img_h, img_w)
    rt_map[ground_mask] *= 1.25  # requires 25% higher saliency on ground
    st_map[ground_mask] *= 1.15  # slightly higher seed threshold too

    if use_abs:
        candidate = ((mag > ft) & (saliency_ratio > rt_map)).astype(np.uint8) * 255
    else:
        candidate = ((mag > 0.5) & (saliency_ratio > rt_map)).astype(np.uint8) * 255

    # ── 8. Seed + direction filter ───────────────────────────
    seed_raw = (diff_f > st_map).astype(np.uint8) * 255
    nk = np.ones((7, 7), np.uint8)
    nc = cv2.filter2D((seed_raw > 0).astype(np.float32), -1, nk)
    mn = 3 if p95_diff > 80 else 4
    seed = seed_raw.copy(); seed[nc < mn] = 0
    if seed.max() == 0: seed = seed_raw

    # Seed direction filter (from V9, slightly relaxed)
    seed_init = (seed > 0)
    if seed_init.sum() > 10:
        flow_angle = np.arctan2(flow_f[..., 1], flow_f[..., 0])
        cos_a = np.cos(flow_angle); sin_a = np.sin(flow_angle)
        cos_local = cv2.blur(cos_a, (7, 7)); sin_local = cv2.blur(sin_a, (7, 7))
        R = np.sqrt(cos_local ** 2 + sin_local ** 2)
        seed[seed_init & (R < 0.5)] = 0
        if seed.max() == 0: seed = seed_raw

    # ── 9. Weak-signal early exit ────────────────────────────
    seed_px = np.count_nonzero(seed)
    if seed_px / (img_h * img_w) < 0.0003:
        return np.zeros((img_h, img_w), dtype=np.uint8)

    # ── 10-12. Dilation + direction + CCA (V7) ───────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    current = seed.copy()
    for _ in range(int(ed)):
        dilated = cv2.dilate(current, kernel)
        dilated &= candidate
        current |= dilated
    expanded = current & ~seed

    seed_bool = (seed > 0)
    if seed_bool.sum() > 20:
        flow_angle = np.arctan2(flow_f[..., 1], flow_f[..., 0])
        seed_angle_map = np.zeros_like(flow_angle)
        seed_angle_map[seed_bool] = flow_angle[seed_bool]
        seed_weight = np.zeros_like(flow_angle)
        seed_weight[seed_bool] = 1.0
        ks = int(ed) * 2 + 1
        seed_angle_blurred = cv2.GaussianBlur(seed_angle_map, (ks, ks), int(ed) / 2.0)
        seed_weight_blurred = cv2.GaussianBlur(seed_weight, (ks, ks), int(ed) / 2.0)
        valid_weight = seed_weight_blurred > 0.02
        seed_angle_blurred[valid_weight] /= seed_weight_blurred[valid_weight]
        angle_diff = np.abs(flow_angle - seed_angle_blurred)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        direction_ok = (angle_diff < np.pi / 4) | ~valid_weight
        expanded[~direction_ok] = 0

    mask = seed | expanded
    mask = _cca_v2(mask, diff, mag, seed, global_diff_mean, global_mag_mean)
    k = np.ones((morph_ksize, morph_ksize), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)

# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
# # ═══════════════════════════════════════════════════════════════════
# # ═══════════════════════════════════════════════════════════════════
# V11: Watershed expansion (replaces iterative dilation in V10)
# ═══════════════════════════════════════════════════════════════════
def detect_motion_hybrid_v11(prev_gray, curr_gray, morph_ksize=5):
    """Hybrid Flow V11: V10 + watershed expansion.

    Replaces constrained iterative dilation with watershed segmentation.
    Seeds are markers, -saliency is the terrain. Watershed naturally
    expands seeds to fill the motion region until saliency drops at the
    boundary — more precise than fixed-distance dilation.
    """
    pg_f = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    cg_f = cv2.GaussianBlur(curr_gray, (5, 5), 0)
    img_h, img_w = pg_f.shape

    # ── 0-5. V10 signal preprocessing (identical) ─────────────
    pg_struct = cv2.bilateralFilter(pg_f, 9, 50, 10)
    cg_struct = cv2.bilateralFilter(cg_f, 9, 50, 10)
    diff = cv2.absdiff(cg_struct, pg_struct)
    diff_f = diff.astype(np.float32)
    p95_diff = np.percentile(diff, 95)
    global_diff_mean = np.mean(diff_f)

    grad_x = cv2.Sobel(cg_struct, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(cg_struct, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2) + 1e-8

    gm_valid = grad_mag[grad_mag > 1.0]
    gm_median = float(np.median(gm_valid)) if len(gm_valid) > 100 else 20.0
    lo, hi = max(15.0, gm_median*0.4), max(40.0, gm_median*1.2)
    edges_prev = cv2.Canny(pg_struct, lo, hi)
    edges_curr = cv2.Canny(cg_struct, lo, hi)
    dist_prev = cv2.distanceTransform((edges_prev==0).astype(np.uint8), cv2.DIST_L2, 3)
    dist_curr = cv2.distanceTransform((edges_curr==0).astype(np.uint8), cv2.DIST_L2, 3)

    moved_edge = np.zeros((img_h, img_w), dtype=np.float32)
    moved_edge[(edges_curr>0)&(dist_prev>3)]=1.0
    disappeared_edge = np.zeros((img_h, img_w), dtype=np.float32)
    disappeared_edge[(edges_prev>0)&(dist_curr>3)]=1.0
    edge_movement = cv2.GaussianBlur(moved_edge+disappeared_edge, (31,31), 9)
    if edge_movement.max()>0.001: edge_movement/=edge_movement.max()
    static_edge = np.zeros((img_h, img_w), dtype=np.float32)
    static_edge[(edges_curr>0)&(dist_prev<2)]=1.0
    static_density = cv2.GaussianBlur(static_edge, (31,31), 9)
    if static_density.max()>0.001: static_density/=static_density.max()
    edge_weight = np.clip(0.5+2.5*edge_movement-0.4*static_density, 0.15, 1.5)

    flow_f = cv2.calcOpticalFlowFarneback(pg_struct, cg_struct, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag_f = np.linalg.norm(flow_f, axis=2)
    camera = _get_camera(fov_scale=1.0)
    pg_p=camera.undistort(pg_struct); cg_p=camera.undistort(cg_struct)
    flow_p=cv2.calcOpticalFlowFarneback(pg_p, cg_p, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag_p=np.linalg.norm(flow_p, axis=2); mag_p_reproj=camera.reproject_to_fisheye(mag_p)

    flow_proj=np.abs(flow_f[...,0]*grad_x+flow_f[...,1]*grad_y)/grad_mag
    align=flow_proj/(mag_f+0.01); edge_w=np.clip(grad_mag/25.0,0.0,1.0)
    mag_f_eff=(1.0-edge_w)*mag_f+edge_w*flow_proj
    suppress_t=(grad_mag>15)&(align<0.4)
    blend_w=_get_blend_map(img_h, img_w)
    mag=blend_w*mag_p_reproj+(1.0-blend_w)*mag_f_eff

    saliency=(mag+0.1)*(diff_f+0.1); saliency*=edge_weight; saliency[suppress_t]*=0.25
    ratios=[]
    for ws,sg in [(11,3),(21,7),(41,15)]:
        ratios.append(saliency/(cv2.GaussianBlur(saliency,(ws,ws),sg)+0.5))
    saliency_ratio=np.minimum(np.minimum(ratios[0],ratios[1]),ratios[2])

    # ── 6-8. Seeds + watershed (replaces dilation) ────────────
    if p95_diff > 100: st,rt,ed=22,2.0,10
    elif p95_diff > 70: st,rt,ed=20,2.5,8
    elif p95_diff > 50: st,rt,ed=18,3.0,8
    elif p95_diff > 30: st,rt,ed=16,4.0,6
    else: st,rt,ed=14,5.0,4

    _,seed_raw=cv2.threshold(diff,st,255,cv2.THRESH_BINARY)
    nk=np.ones((7,7),np.uint8); nc=cv2.filter2D((seed_raw>0).astype(np.float32),-1,nk)
    mn=3 if p95_diff>80 else 4; seed=seed_raw.copy(); seed[nc<mn]=0
    if seed.max()==0: seed=seed_raw

    # Seed direction filter
    seed_init=(seed>0)
    if seed_init.sum()>10:
        fa=np.arctan2(flow_f[...,1],flow_f[...,0])
        ca=np.cos(fa); sa=np.sin(fa)
        cl=cv2.blur(ca,(7,7)); sl=cv2.blur(sa,(7,7))
        R=np.sqrt(cl**2+sl**2); seed[seed_init&(R<0.5)]=0
        if seed.max()==0: seed=seed_raw

    seed_px=np.count_nonzero(seed)
    if seed_px/(img_h*img_w)<0.0003:
        return np.zeros((img_h,img_w),dtype=np.uint8)

    # ── WATERSHED expansion ──────────────────────────────────
    nl,lb=cv2.connectedComponents(seed,connectivity=8)
    if nl<2: return seed

    # Candidate mask from V10-style threshold
    if p95_diff>100: rt2=2.0
    elif p95_diff>70: rt2=2.5
    elif p95_diff>50: rt2=3.0
    elif p95_diff>30: rt2=4.0
    else: rt2=5.0
    candidate_v10 = (saliency_ratio>rt2).astype(np.uint8)*255

    # Terrain: 255 (barrier) outside candidate, -saliency inside
    terrain = np.full((img_h,img_w),255,dtype=np.uint8)
    inside = candidate_v10>0
    terrain[inside] = (255-(saliency_ratio[inside]*20).clip(0,255)).astype(np.uint8)

    # Markers: seeds get their CC labels, everything else = 0 (unknown)
    markers = np.zeros((img_h,img_w),dtype=np.int32)
    markers[seed>0] = lb[seed>0]

    cv2.watershed(cv2.cvtColor(terrain,cv2.COLOR_GRAY2BGR), markers)
    mask = (markers > 0).astype(np.uint8)*255

    # Direction consistency on expanded area
    seed_bool=(seed>0)
    if seed_bool.sum()>20:
        fa2=np.arctan2(flow_f[...,1],flow_f[...,0])
        sam=np.zeros_like(fa2); sam[seed_bool]=fa2[seed_bool]
        sw=np.zeros_like(fa2); sw[seed_bool]=1.0
        ks2=int(ed)*2+1
        sab=cv2.GaussianBlur(sam,(ks2,ks2),int(ed)/2.0)
        swb=cv2.GaussianBlur(sw,(ks2,ks2),int(ed)/2.0)
        vw=swb>0.02; sab[vw]/=swb[vw]
        ad=np.abs(fa2-sab); ad=np.minimum(ad,2*np.pi-ad)
        dok=(ad<np.pi/4)|~vw
        mask[~dok]=0

    k=np.ones((morph_ksize,morph_ksize),np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(mask,cv2.MORPH_OPEN,k),cv2.MORPH_CLOSE,k)
