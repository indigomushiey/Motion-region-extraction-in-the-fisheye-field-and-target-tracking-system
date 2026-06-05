import cv2
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "homework2"
CURRENT_DIR = DATA_DIR / "rgb_images"
PREVIOUS_DIR = DATA_DIR / "previous_images"
OUTPUT_DIR = PROJECT_ROOT / "output" / "masks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def preprocess(gray, blur_ksize=5, use_clahe=False):
    """Apply Gaussian blur and optional CLAHE normalization."""
    gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    return gray


def detect_motion_diff(prev_gray, curr_gray, diff_thresh=40, morph_ksize=5, use_clahe=False):
    """Frame-difference based motion detection.

    Returns binary mask (uint8, 0/255).
    """
    prev_gray = preprocess(prev_gray, use_clahe=use_clahe)
    curr_gray = preprocess(curr_gray, use_clahe=use_clahe)

    diff = cv2.absdiff(curr_gray, prev_gray)
    _, mask = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)

    k = np.ones((morph_ksize, morph_ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def detect_motion_flow(prev_gray, curr_gray, mag_thresh=3.0, morph_ksize=9):
    """Farneback optical flow based motion detection.

    Returns binary mask (uint8, 0/255).
    """
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag = np.linalg.norm(flow, axis=2)
    _, mask = cv2.threshold(mag, mag_thresh, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)

    k = np.ones((morph_ksize, morph_ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def detect_motion_hybrid(prev_gray, curr_gray, diff_thresh=50, mag_thresh=3.0, morph_ksize=7):
    """Combine frame difference and optical flow (AND logic) to reduce false positives.

    Returns binary mask (uint8, 0/255).
    """
    diff_mask = detect_motion_diff(prev_gray, curr_gray, diff_thresh, morph_ksize=1)
    flow_mask = detect_motion_flow(prev_gray, curr_gray, mag_thresh, morph_ksize=1)

    mask = diff_mask & flow_mask

    k = np.ones((morph_ksize, morph_ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def detect_motion_adaptive(prev_gray, curr_gray, center_thresh=40, edge_thresh=80,
                           edge_flow_thresh=2.0, morph_ksize=7):
    """Progressive threshold: lower at center (less distortion), higher at edge (more noise).

    At edge regions, also requires optical flow magnitude verification to suppress
    stretched-pixel false positives from the remap process.

    Returns binary mask (uint8, 0/255).
    """
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)

    h, w = prev_gray.shape
    cx, cy = w / 2, h / 2
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_r = np.sqrt(cx ** 2 + cy ** 2)

    # progressive threshold: center → edge_thresh at max radius
    th_map = center_thresh + (dist / max_r) * (edge_thresh - center_thresh)
    th_map = th_map.astype(np.float32)

    diff = cv2.absdiff(curr_gray, prev_gray)
    mask = (diff > th_map).astype(np.uint8) * 255

    # at edge regions, additionally verify with optical flow
    if edge_flow_thresh > 0:
        in_edge = (dist > 0.5 * max_r).astype(np.uint8) * 255
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag = np.linalg.norm(flow, axis=2)
        flow_ok = (mag > edge_flow_thresh).astype(np.uint8) * 255
        mask = np.where(in_edge, mask & flow_ok, mask)

    k = np.ones((morph_ksize, morph_ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def detect_motion_seed_expand(prev_gray, curr_gray, seed_thresh=45, flow_thresh=2.0,
                              expand_dist=12, morph_ksize=7):
    """Seed-and-expand: diff mask as trusted seeds, flow-based expansion in neighborhood.

    diff40 provides high-precision seeds. Optical flow captures the full motion extent
    but with many false positives. Distance transform keeps only flow pixels near a
    seed, suppressing isolated noise.

    Returns binary mask (uint8, 0/255).
    """
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)

    # seed: frame difference (high precision)
    diff = cv2.absdiff(curr_gray, prev_gray)
    _, seed = cv2.threshold(diff, seed_thresh, 255, cv2.THRESH_BINARY)

    # candidate: optical flow (high recall)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag = np.linalg.norm(flow, axis=2)
    _, candidate = cv2.threshold(mag, flow_thresh, 255, cv2.THRESH_BINARY)
    candidate = candidate.astype(np.uint8)

    # distance from each pixel to the nearest seed pixel
    dist = cv2.distanceTransform((seed == 0).astype(np.uint8), cv2.DIST_L2, 5)

    # keep candidate pixels within expand_dist of a seed
    expanded = candidate.copy()
    expanded[dist > expand_dist] = 0

    # merge seed + expanded
    mask = seed | expanded

    k = np.ones((morph_ksize, morph_ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


# ================================================================
# Improved seed-expand variants
# ================================================================

def detect_motion_seed_expand_v2(prev_gray, curr_gray,
                                  seed_percentile=95, flow_percentile=90,
                                  expand_dist=10, morph_ksize=7,
                                  min_seed_ratio=0.0005):
    """Seed-expand V2: adaptive percentile thresholds + seed density fallback.

    Key improvements over V1:
      - seed_thresh = percentile(diff, seed_percentile) — adapts per-frame
      - flow_thresh = percentile(mag, flow_percentile) — adapts per-frame
      - If seeds too sparse (< min_seed_ratio of pixels), lowers percentile & expand_dist
    """
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)

    total_pixels = prev_gray.size

    # --- adaptive seed threshold ---
    diff = cv2.absdiff(curr_gray, prev_gray)
    seed_thresh = np.percentile(diff, seed_percentile)
    seed_thresh = max(seed_thresh, 10)   # floor: avoid degenerate threshold
    _, seed = cv2.threshold(diff, seed_thresh, 255, cv2.THRESH_BINARY)

    # --- adaptive flow threshold ---
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag = np.linalg.norm(flow, axis=2)
    flow_thresh = np.percentile(mag, flow_percentile)
    flow_thresh = max(flow_thresh, 1.0)
    _, candidate = cv2.threshold(mag, flow_thresh, 255, cv2.THRESH_BINARY)
    candidate = candidate.astype(np.uint8)

    # --- seed density feedback ---
    seed_count = int(np.sum(seed > 0))
    actual_expand = expand_dist
    actual_flow_pct = flow_percentile

    if seed_count < total_pixels * min_seed_ratio:
        # Too few seeds → broaden seed capture, tighten expansion
        fallback_pct = max(seed_percentile - 10, 80)
        seed_thresh2 = np.percentile(diff, fallback_pct)
        seed_thresh2 = max(seed_thresh2, 8)
        _, seed = cv2.threshold(diff, seed_thresh2, 255, cv2.THRESH_BINARY)
        actual_expand = max(expand_dist - 4, 4)
        actual_flow_pct = max(flow_percentile - 5, 85)
        flow_thresh2 = np.percentile(mag, actual_flow_pct)
        flow_thresh2 = max(flow_thresh2, 1.0)
        _, candidate = cv2.threshold(mag, flow_thresh2, 255, cv2.THRESH_BINARY)
        candidate = candidate.astype(np.uint8)

    # --- distance-constrained expansion ---
    dist = cv2.distanceTransform((seed == 0).astype(np.uint8), cv2.DIST_L2, 5)
    expanded = candidate.copy()
    expanded[dist > actual_expand] = 0

    mask = seed | expanded

    k = np.ones((morph_ksize, morph_ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def detect_motion_seed_expand_v3(prev_gray, curr_gray,
                                  seed_percentile=95, flow_percentile=90,
                                  expand_dist=10, morph_ksize=7,
                                  angle_thresh=np.pi/4):
    """Seed-expand V3: V2 + flow direction consistency check.

    Additional improvement: candidate pixels whose flow direction differs
    from the nearest seed's flow direction by more than angle_thresh are
    filtered out, suppressing noise in textured-but-static regions.
    """
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)
    total_pixels = prev_gray.size

    # --- adaptive seed ---
    diff = cv2.absdiff(curr_gray, prev_gray)
    seed_thresh = np.percentile(diff, seed_percentile)
    seed_thresh = max(seed_thresh, 10)
    _, seed = cv2.threshold(diff, seed_thresh, 255, cv2.THRESH_BINARY)

    # --- adaptive flow ---
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag = np.linalg.norm(flow, axis=2)
    flow_thresh = np.percentile(mag, flow_percentile)
    flow_thresh = max(flow_thresh, 1.0)
    _, candidate = cv2.threshold(mag, flow_thresh, 255, cv2.THRESH_BINARY)
    candidate = candidate.astype(np.uint8)

    # --- seed density fallback ---
    seed_count = int(np.sum(seed > 0))
    actual_expand = expand_dist
    if seed_count < total_pixels * 0.0005:
        fallback_pct = max(seed_percentile - 10, 80)
        seed_thresh2 = np.percentile(diff, fallback_pct)
        seed_thresh2 = max(seed_thresh2, 8)
        _, seed = cv2.threshold(diff, seed_thresh2, 255, cv2.THRESH_BINARY)
        actual_expand = max(expand_dist - 4, 4)

    # --- distance-constrained expansion ---
    dist = cv2.distanceTransform((seed == 0).astype(np.uint8), cv2.DIST_L2, 5)
    expanded = candidate.copy()
    expanded[dist > actual_expand] = 0

    # --- flow direction consistency ---
    flow_angle = np.arctan2(flow[..., 1], flow[..., 0])
    # Compute mean flow angle of seed pixels; fallback = 0
    seed_mask_bool = (seed > 0)
    if seed_mask_bool.sum() > 10:
        # For each candidate pixel, compute angle diff to nearest seed
        # Use distance transform labels to find nearest seed for each pixel
        _, labels = cv2.distanceTransformWithLabels(
            (seed == 0).astype(np.uint8), cv2.DIST_L2, 5,
            labelType=cv2.DIST_LABEL_PIXEL
        )
        # labels contains the index of the nearest zero-pixel (background)
        # We need nearest seed (foreground). Invert: find nearest bg to each bg → then map
        # Simpler approach: compute seed angle map, then blur-expand it
        seed_angle_map = np.zeros_like(flow_angle)
        seed_angle_map[seed_mask_bool] = flow_angle[seed_mask_bool]
        # Expand seed angles via distance-propagated blur
        kernel_size = actual_expand * 2 + 1
        seed_angle_blurred = cv2.GaussianBlur(seed_angle_map, (kernel_size, kernel_size), actual_expand / 2)
        # Also blur a weight map (1 where seeds exist)
        seed_weight = np.zeros_like(flow_angle)
        seed_weight[seed_mask_bool] = 1.0
        seed_weight_blurred = cv2.GaussianBlur(seed_weight, (kernel_size, kernel_size), actual_expand / 2)
        # Normalized angle estimate
        valid = seed_weight_blurred > 0.01
        seed_angle_blurred[valid] /= seed_weight_blurred[valid]

        # Angle difference
        angle_diff = np.abs(flow_angle - seed_angle_blurred)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        direction_ok = (angle_diff < angle_thresh) | ~valid | (seed_weight_blurred < 0.01)
        expanded[~direction_ok] = 0

    mask = seed | expanded

    k = np.ones((morph_ksize, morph_ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def detect_motion_seed_expand_v4(prev_gray, curr_gray,
                                  seed_thresh=30, flow_thresh=2.5,
                                  expand_dist=7, morph_ksize=7,
                                  angle_thresh=np.pi/4):
    """Seed-expand V4: lower seed threshold + tight expand + direction consistency.

    Strategy:
      - seed_thresh=30 (lower than V1's 45 → more seeds, better recall)
      - flow_thresh=2.5 (slightly higher than V1's 2.0 → less noise)
      - expand_dist=7 (tighter than V1's 12 → less leakage)
      - direction consistency: candidate must have similar flow direction to nearby seeds
    """
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)

    # --- seed: generous frame difference ---
    diff = cv2.absdiff(curr_gray, prev_gray)
    _, seed = cv2.threshold(diff, seed_thresh, 255, cv2.THRESH_BINARY)

    # --- candidate: moderate flow threshold ---
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag = np.linalg.norm(flow, axis=2)
    _, candidate = cv2.threshold(mag, flow_thresh, 255, cv2.THRESH_BINARY)
    candidate = candidate.astype(np.uint8)

    # --- distance-constrained expansion ---
    dist = cv2.distanceTransform((seed == 0).astype(np.uint8), cv2.DIST_L2, 5)
    expanded = candidate.copy()
    expanded[dist > expand_dist] = 0

    # --- flow direction consistency ---
    seed_bool = (seed > 0)
    if seed_bool.sum() > 10:
        flow_angle = np.arctan2(flow[..., 1], flow[..., 0])
        # Build seed angle map, blur to propagate direction to nearby pixels
        seed_angle_map = np.zeros_like(flow_angle)
        seed_angle_map[seed_bool] = flow_angle[seed_bool]
        seed_weight = np.zeros_like(flow_angle)
        seed_weight[seed_bool] = 1.0

        ks = expand_dist * 2 + 1
        seed_angle_blurred = cv2.GaussianBlur(seed_angle_map, (ks, ks), expand_dist / 2.0)
        seed_weight_blurred = cv2.GaussianBlur(seed_weight, (ks, ks), expand_dist / 2.0)
        valid_weight = seed_weight_blurred > 0.02
        seed_angle_blurred[valid_weight] /= seed_weight_blurred[valid_weight]

        angle_diff = np.abs(flow_angle - seed_angle_blurred)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        direction_ok = (angle_diff < angle_thresh) | ~valid_weight
        expanded[~direction_ok] = 0

    mask = seed | expanded

    k = np.ones((morph_ksize, morph_ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def detect_motion_seed_expand_v5(prev_gray, curr_gray,
                                  seed_thresh=30, ratio_thresh=2.0,
                                  expand_dist=10, morph_ksize=7):
    """Seed-expand V5: diff seeds + relative flow ratio candidate (key innovation).

    Core insight: raw flow magnitude is uniformly high across the image due to
    sensor noise + Farneback artifacts (~50% pixels have mag > 2.0).
    Instead, we use the RATIO of flow magnitude to its local mean:
      ratio(x,y) = mag(x,y) / mean_mag(local_window)
    This highlights pixels that move MORE than their surroundings.

    Strategy:
      - seed_thresh=30 (moderate diff threshold → decent seed coverage)
      - ratio_thresh=2.0 (candidate = pixels moving 2x faster than local avg)
      - expand_dist=10 (moderate expansion from seeds)
    """
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)

    # --- seed: moderate frame difference ---
    diff = cv2.absdiff(curr_gray, prev_gray)
    _, seed = cv2.threshold(diff, seed_thresh, 255, cv2.THRESH_BINARY)

    # --- candidate: relative flow magnitude ---
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag = np.linalg.norm(flow, axis=2)
    # Local mean of flow magnitude (large window to capture background level)
    mag_mean = cv2.GaussianBlur(mag, (41, 41), 15)
    mag_ratio = mag / (mag_mean + 0.5)  # +0.5 avoids division by zero
    _, candidate = cv2.threshold(mag_ratio, ratio_thresh, 255, cv2.THRESH_BINARY)
    candidate = candidate.astype(np.uint8)

    # --- distance-constrained expansion ---
    dist = cv2.distanceTransform((seed == 0).astype(np.uint8), cv2.DIST_L2, 5)
    expanded = candidate.copy()
    expanded[dist > expand_dist] = 0

    mask = seed | expanded

    k = np.ones((morph_ksize, morph_ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def detect_motion_seed_expand_v6(prev_gray, curr_gray,
                                  seed_thresh=35, flow_thresh=3.5,
                                  ratio_thresh=1.2, expand_dist=8,
                                  morph_ksize=7):
    """Seed-expand V6: combined absolute + relative flow filtering + tighter expansion.

    Key idea: candidate = (mag > flow_thresh) AND (mag_ratio > ratio_thresh)
    This requires pixels to have BOTH sufficient absolute motion AND be moving
    faster than their local surroundings — eliminating uniform sensor noise.

    Vs V1: seed_thresh 45→35 (more seeds), flow_thresh 2.0→3.5 (less noise),
           expand_dist 12→8 (tighter), plus ratio check.
    """
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)

    # --- seed ---
    diff = cv2.absdiff(curr_gray, prev_gray)
    _, seed = cv2.threshold(diff, seed_thresh, 255, cv2.THRESH_BINARY)

    # --- candidate: dual filter (absolute + relative) ---
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag = np.linalg.norm(flow, axis=2)
    mag_mean = cv2.GaussianBlur(mag, (41, 41), 15)
    mag_ratio = mag / (mag_mean + 0.5)

    # BOTH conditions must hold
    candidate = ((mag > flow_thresh) & (mag_ratio > ratio_thresh)).astype(np.uint8) * 255

    # --- distance-constrained expansion ---
    dist = cv2.distanceTransform((seed == 0).astype(np.uint8), cv2.DIST_L2, 5)
    expanded = candidate.copy()
    expanded[dist > expand_dist] = 0

    mask = seed | expanded

    k = np.ones((morph_ksize, morph_ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def detect_motion_seed_expand_v7(prev_gray, curr_gray,
                                  seed_thresh=20, flow_thresh=3.5,
                                  expand_dist=8, min_seed_neighbors=3,
                                  morph_ksize=7):
    """Seed-expand V7: low-threshold seeds + spatial consistency filter.

    Key innovation: use a LOW seed threshold (20) to capture all motion,
    then filter seeds by spatial density — isolated seeds are discarded.
    Only seeds with >= min_seed_neighbors in a local window survive.
    This eliminates scattered noise while keeping dense motion regions.

    Then: flow-based expansion with moderate threshold + tight distance.
    """
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)

    # --- seed: low threshold to capture all motion ---
    diff = cv2.absdiff(curr_gray, prev_gray)
    _, seed_raw = cv2.threshold(diff, seed_thresh, 255, cv2.THRESH_BINARY)

    # --- spatial consistency: count neighbors in window ---
    neighbor_kernel = np.ones((7, 7), np.uint8)
    neighbor_count = cv2.filter2D(
        (seed_raw > 0).astype(np.float32), -1, neighbor_kernel
    )
    # A seed must have at least N neighbors (including itself in the count)
    seed = seed_raw.copy()
    seed[neighbor_count < min_seed_neighbors] = 0

    # If filtering removed all seeds, fall back to raw seeds
    if seed.max() == 0:
        seed = seed_raw

    # --- candidate: moderate flow threshold ---
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag = np.linalg.norm(flow, axis=2)
    _, candidate = cv2.threshold(mag, flow_thresh, 255, cv2.THRESH_BINARY)
    candidate = candidate.astype(np.uint8)

    # --- distance-constrained expansion ---
    dist = cv2.distanceTransform((seed == 0).astype(np.uint8), cv2.DIST_L2, 5)
    expanded = candidate.copy()
    expanded[dist > expand_dist] = 0

    mask = seed | expanded

    k = np.ones((morph_ksize, morph_ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def detect_motion_seed_expand_v8(prev_gray, curr_gray,
                                  scales=(1.0, 0.5, 0.25),
                                  seed_thresh=35, flow_thresh=2.5,
                                  expand_dist=10, morph_ksize=7,
                                  vote_thresh=2):
    """Seed-expand V8: multi-scale seed expansion with majority voting.

    Core idea (方案5):
      - Build an image pyramid at multiple scales
      - Run seed_expand independently at each scale
      - Resize all masks back to original resolution
      - Fuse via majority voting (pixel must be detected in >= vote_thresh scales)

    Why it works:
      - Small scales: downsampling acts as low-pass filter, smoothing noise.
        Large motion structures are clearer → high recall.
      - Original scale: preserves fine details and boundaries → high precision.
      - Voting suppresses isolated false positives that appear in only one scale.

    Scale-adapted parameters:
      - flow_thresh scales with resolution (physical motion = fewer pixels at small scale)
      - expand_dist scales with resolution (same physical distance = fewer pixels)
    """
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)

    h, w = prev_gray.shape
    masks = []

    for s in scales:
        if s < 1.0:
            h2, w2 = int(h * s), int(w * s)
            p_small = cv2.resize(prev_gray, (w2, h2))
            c_small = cv2.resize(curr_gray, (w2, h2))
        else:
            p_small, c_small = prev_gray, curr_gray
            h2, w2 = h, w

        # --- scale-adapted parameters ---
        s_expand_dist = max(3, int(expand_dist * s))
        s_flow_thresh = flow_thresh * s       # physical motion scales with resolution
        # At smaller scales, noise is smoothed → can use slightly lower seed_thresh
        s_seed_thresh = max(15, int(seed_thresh * (0.7 + 0.3 * s)))

        # --- seed ---
        diff = cv2.absdiff(c_small, p_small)
        _, seed = cv2.threshold(diff, s_seed_thresh, 255, cv2.THRESH_BINARY)

        # --- candidate: optical flow ---
        flow = cv2.calcOpticalFlowFarneback(
            p_small, c_small, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag = np.linalg.norm(flow, axis=2)
        _, candidate = cv2.threshold(mag, s_flow_thresh, 255, cv2.THRESH_BINARY)
        candidate = candidate.astype(np.uint8)

        # --- distance-constrained expansion ---
        dist = cv2.distanceTransform((seed == 0).astype(np.uint8), cv2.DIST_L2, 5)
        expanded = candidate.copy()
        expanded[dist > s_expand_dist] = 0

        mask_small = seed | expanded

        # --- morphology at this scale ---
        k = np.ones((morph_ksize, morph_ksize), np.uint8)
        mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_OPEN, k)
        mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, k)

        # --- resize back to original resolution ---
        if s < 1.0:
            mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            mask = mask_small

        masks.append(mask)

    # --- multi-scale majority voting ---
    vote_sum = np.zeros((h, w), dtype=np.float32)
    for m in masks:
        vote_sum += (m > 0).astype(np.float32)

    # At least vote_thresh scales must agree
    final = (vote_sum >= vote_thresh).astype(np.uint8) * 255

    # --- final morphology ---
    k = np.ones((morph_ksize, morph_ksize), np.uint8)
    final = cv2.morphologyEx(final, cv2.MORPH_OPEN, k)
    final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, k)
    return final


def detect_motion_seed_expand_v10(prev_gray, curr_gray, morph_ksize=7):
    """Seed-expand V10: adaptive ratio-based expansion with spatial consistency.

    Combines the best innovations from V1-V9:
      - V5's flow RATIO as candidate (suppresses uniform sensor noise)
      - V7's spatial seed consistency (allows lower seed thresholds safely)
      - V3's direction consistency (suppresses edge-distortion false positives)
      - V9's frame-adaptive parameter selection

    Core algorithm:
      1. Analyze diff stats → classify frame as strong/medium/weak motion
      2. Seed = low-threshold diff + spatial neighbor count filter
      3. Candidate = flow magnitude ratio > adaptive_ratio_thresh
         (+ optional absolute flow check for strong-motion frames)
      4. Direction consistency check (flow direction must match nearby seeds)
      5. Distance-constrained expansion from seeds
      6. Morphological cleanup

    Why ratio beats absolute flow:
      - Raw flow mag is uniformly high (sensor noise + Farneback artifacts)
      - mag/mag_local_mean > 1.5 means "moving faster than surroundings"
      - This intrinsically suppresses uniform background noise
    """
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)

    # ── frame analysis ──────────────────────────────────────────
    diff = cv2.absdiff(curr_gray, prev_gray)
    p95_diff = np.percentile(diff, 95)
    p50_diff = np.percentile(diff, 50)

    # ── compute flow once ───────────────────────────────────────
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag = np.linalg.norm(flow, axis=2)
    mag_mean = cv2.GaussianBlur(mag, (41, 41), 15)
    mag_ratio = mag / (mag_mean + 0.5)

    # ── frame-adaptive parameters ───────────────────────────────
    if p95_diff > 120:
        # Very strong motion (e.g. 00013): aggressive
        seed_thresh = 22
        ratio_thresh = 1.3
        expand_dist = 12
        use_abs_flow = True
        flow_thresh = 2.0
        use_direction = False  # direction less reliable with large motion
    elif p95_diff > 80:
        # Strong motion
        seed_thresh = 20
        ratio_thresh = 1.5
        expand_dist = 10
        use_abs_flow = True
        flow_thresh = 2.5
        use_direction = False
    elif p95_diff > 55:
        # Medium-strong motion (e.g. 00039)
        seed_thresh = 18
        ratio_thresh = 1.6
        expand_dist = 8
        use_abs_flow = False
        use_direction = True
    elif p95_diff > 35:
        # Medium motion (e.g. 00041)
        seed_thresh = 16
        ratio_thresh = 1.8
        expand_dist = 6
        use_abs_flow = False
        use_direction = True
    else:
        # Weak motion
        seed_thresh = 14
        ratio_thresh = 2.2
        expand_dist = 4
        use_abs_flow = False
        use_direction = True

    # ── seed: low-threshold diff + spatial consistency ──────────
    _, seed_raw = cv2.threshold(diff, seed_thresh, 255, cv2.THRESH_BINARY)

    # Spatial consistency: seed must have neighbors in 7x7 window
    neighbor_kernel = np.ones((7, 7), np.uint8)
    neighbor_count = cv2.filter2D(
        (seed_raw > 0).astype(np.float32), -1, neighbor_kernel
    )
    min_neighbors = 3 if p95_diff > 80 else 4
    seed = seed_raw.copy()
    seed[neighbor_count < min_neighbors] = 0
    if seed.max() == 0:
        seed = seed_raw  # fallback

    # ── candidate: flow ratio ───────────────────────────────────
    if use_abs_flow:
        candidate = ((mag > flow_thresh) & (mag_ratio > ratio_thresh)).astype(np.uint8) * 255
    else:
        _, candidate = cv2.threshold(mag_ratio, ratio_thresh, 255, cv2.THRESH_BINARY)
        candidate = candidate.astype(np.uint8)

    # ── distance-constrained expansion ──────────────────────────
    dist = cv2.distanceTransform((seed == 0).astype(np.uint8), cv2.DIST_L2, 5)
    expanded = candidate.copy()
    expanded[dist > expand_dist] = 0

    # ── direction consistency (selective) ───────────────────────
    if use_direction:
        seed_bool = (seed > 0)
        if seed_bool.sum() > 20:
            flow_angle = np.arctan2(flow[..., 1], flow[..., 0])
            seed_angle_map = np.zeros_like(flow_angle)
            seed_angle_map[seed_bool] = flow_angle[seed_bool]
            seed_weight = np.zeros_like(flow_angle)
            seed_weight[seed_bool] = 1.0

            ks = expand_dist * 2 + 1
            seed_angle_blurred = cv2.GaussianBlur(seed_angle_map, (ks, ks), expand_dist / 2.0)
            seed_weight_blurred = cv2.GaussianBlur(seed_weight, (ks, ks), expand_dist / 2.0)
            valid_weight = seed_weight_blurred > 0.02
            seed_angle_blurred[valid_weight] /= seed_weight_blurred[valid_weight]

            angle_diff = np.abs(flow_angle - seed_angle_blurred)
            angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
            direction_ok = (angle_diff < np.pi / 4) | ~valid_weight
            expanded[~direction_ok] = 0

    # ── merge ───────────────────────────────────────────────────
    mask = seed | expanded

    k = np.ones((morph_ksize, morph_ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def detect_motion_seed_expand_v11(prev_gray, curr_gray, morph_ksize=7):
    """Seed-expand V11: region-adaptive ratio + strong-motion fallback + CCA post-process.

    Combines all validated improvements from V1-V10 plus three new optimizations:

    P0a — Strong-motion fallback:
      When p95_diff > 120, use pure V1 strategy (abs flow, wide expansion).
      Prevents V10's over-constraint on frames like 00013 (V1 F1=0.678 vs V10=0.572).

    P0b — Refined frame-adaptive parameters:
      5 bins by p95_diff, with V10's rough boundaries refined for smoother transitions.

    P1 — Region-adaptive ratio threshold:
      Instead of a global ratio_thresh, apply a radial gradient:
        ratio_thresh(r) = ratio_thresh_base * (1.0 + edge_bonus * norm_dist^2)
      Center: base threshold (less noise). Edge: base + bonus (more noise).
      Quadratic falloff focuses the penalty on the outermost regions.

    P2 — Connected-component post-processing:
      Replace fixed-kernel morphology with CCA:
        1. Remove small isolated components (< min_component_area pixels)
        2. Fill each component with convex hull (fills gaps from ratio filtering)
        3. Merge only very close components (small dilate + re-label)
        4. Light 3x3 morphological cleanup
    """
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)

    # ── Phase 1: frame analysis ────────────────────────────────────
    diff = cv2.absdiff(curr_gray, prev_gray)
    p95_diff = np.percentile(diff, 95)
    h, w = prev_gray.shape

    # ── Phase 2: strong-motion fallback (P0a) ──────────────────────
    if p95_diff > 120:
        # Pure V1 strategy — abs flow beats ratio for strong motion
        seed_thresh = 45
        _, seed = cv2.threshold(diff, seed_thresh, 255, cv2.THRESH_BINARY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag = np.linalg.norm(flow, axis=2)
        _, candidate = cv2.threshold(mag, 2.0, 255, cv2.THRESH_BINARY)
        candidate = candidate.astype(np.uint8)

        dist = cv2.distanceTransform((seed == 0).astype(np.uint8), cv2.DIST_L2, 5)
        expanded = candidate.copy()
        expanded[dist > 12] = 0

        mask = seed | expanded
        k = np.ones((morph_ksize, morph_ksize), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    # ── Phase 3: compute flow once ──────────────────────────────────
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag = np.linalg.norm(flow, axis=2)
    mag_mean = cv2.GaussianBlur(mag, (41, 41), 15)
    mag_ratio = mag / (mag_mean + 0.5)

    # ── Phase 4: frame-adaptive parameters (P0b) ────────────────────
    if p95_diff > 100:
        seed_thresh = 22
        ratio_thresh_base = 1.3
        expand_dist = 12
        use_abs_flow = True
        flow_thresh = 2.0
        use_direction = False
    elif p95_diff > 70:
        seed_thresh = 20
        ratio_thresh_base = 1.5
        expand_dist = 10
        use_abs_flow = True
        flow_thresh = 2.5
        use_direction = False
    elif p95_diff > 50:
        seed_thresh = 18
        ratio_thresh_base = 1.6
        expand_dist = 8
        use_abs_flow = False
        use_direction = True
    elif p95_diff > 30:
        seed_thresh = 16
        ratio_thresh_base = 1.8
        expand_dist = 6
        use_abs_flow = False
        use_direction = True
    else:
        seed_thresh = 14
        ratio_thresh_base = 2.2
        expand_dist = 4
        use_abs_flow = False
        use_direction = True

    # ── Phase 5: region-adaptive ratio threshold (P1) ───────────────
    cx, cy = w / 2, h / 2
    yy, xx = np.ogrid[:h, :w]
    dist_map = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_r = np.sqrt(cx ** 2 + cy ** 2)
    norm_dist = dist_map / max_r  # 0 at center, 1 at corners

    # Edge bonus: ratio_thresh increases toward the edge where distortion is worse.
    # Use quadratic falloff — penalty concentrated at the outermost 50% of radius.
    edge_bonus = 0.35  # max 35% increase at extreme edge
    radius_mult = 1.0 + edge_bonus * (norm_dist ** 2)
    ratio_thresh_map = ratio_thresh_base * radius_mult

    # Build candidate from per-pixel ratio threshold
    if use_abs_flow:
        candidate = ((mag > flow_thresh) & (mag_ratio > ratio_thresh_map)).astype(np.uint8) * 255
    else:
        candidate = (mag_ratio > ratio_thresh_map).astype(np.uint8) * 255

    # ── Phase 6: spatial seed filtering ─────────────────────────────
    _, seed_raw = cv2.threshold(diff, seed_thresh, 255, cv2.THRESH_BINARY)

    neighbor_kernel = np.ones((7, 7), np.uint8)
    neighbor_count = cv2.filter2D(
        (seed_raw > 0).astype(np.float32), -1, neighbor_kernel
    )
    min_neighbors = 3 if p95_diff > 80 else 4
    seed = seed_raw.copy()
    seed[neighbor_count < min_neighbors] = 0
    if seed.max() == 0:
        seed = seed_raw  # fallback

    # ── Phase 7: distance-constrained expansion ─────────────────────
    dist = cv2.distanceTransform((seed == 0).astype(np.uint8), cv2.DIST_L2, 5)
    expanded = candidate.copy()
    expanded[dist > expand_dist] = 0

    # ── Phase 8: direction consistency (selective) ──────────────────
    if use_direction:
        seed_bool = (seed > 0)
        if seed_bool.sum() > 20:
            flow_angle = np.arctan2(flow[..., 1], flow[..., 0])
            seed_angle_map = np.zeros_like(flow_angle)
            seed_angle_map[seed_bool] = flow_angle[seed_bool]
            seed_weight = np.zeros_like(flow_angle)
            seed_weight[seed_bool] = 1.0

            ks = expand_dist * 2 + 1
            seed_angle_blurred = cv2.GaussianBlur(seed_angle_map, (ks, ks), expand_dist / 2.0)
            seed_weight_blurred = cv2.GaussianBlur(seed_weight, (ks, ks), expand_dist / 2.0)
            valid_weight = seed_weight_blurred > 0.02
            seed_angle_blurred[valid_weight] /= seed_weight_blurred[valid_weight]

            angle_diff = np.abs(flow_angle - seed_angle_blurred)
            angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
            direction_ok = (angle_diff < np.pi / 4) | ~valid_weight
            expanded[~direction_ok] = 0

    # ── Phase 9: merge seed + expanded ──────────────────────────────
    mask = seed | expanded

    # ── Phase 10: CCA post-processing (P2) ──────────────────────────
    # Remove small isolated noise components, then apply standard morphology.
    # Convex hull and aggressive merging were tested but found to over-expand
    # masks on fisheye images (from 15% → 51% coverage on 00039).
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_component_area = 50
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_component_area:
            mask[labels == i] = 0

    # Standard morphology (same as V10) after CCA cleanup
    k = np.ones((morph_ksize, morph_ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def detect_motion_seed_expand_v9(prev_gray, curr_gray,
                                  morph_ksize=7):
    """Seed-expand V9: frame-adaptive strategy selection.

    Problem: V1 works well for strong-motion frames (00013 F1=0.67)
    but terribly for weak-motion frames (00003 F1=0.003). The issue is
    that flow_thresh=2.0 captures 50%+ of pixels as noise.

    Solution: use diff signal strength to choose strategy.
      - Strong motion (P95_diff > 60): aggressive flow expansion (like V1)
      - Medium motion (P95_diff 30-60): moderate flow + tighter expansion
      - Weak motion (P95_diff < 30): diff-only, no flow expansion
    """
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)

    diff = cv2.absdiff(curr_gray, prev_gray)
    p95_diff = np.percentile(diff, 95)

    if p95_diff > 60:
        # Strong motion: V1-like aggressive expansion
        seed_thresh = 40
        flow_thresh = 2.5
        expand_dist = 12
        use_flow = True
    elif p95_diff > 30:
        # Medium motion: moderate expansion
        seed_thresh = 35
        flow_thresh = 3.5
        expand_dist = 8
        use_flow = True
    else:
        # Weak motion: diff-only, conservative
        seed_thresh = 25
        expand_dist = 0
        use_flow = False

    # --- seed ---
    _, seed = cv2.threshold(diff, seed_thresh, 255, cv2.THRESH_BINARY)

    if use_flow:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag = np.linalg.norm(flow, axis=2)
        _, candidate = cv2.threshold(mag, flow_thresh, 255, cv2.THRESH_BINARY)
        candidate = candidate.astype(np.uint8)

        dist = cv2.distanceTransform((seed == 0).astype(np.uint8), cv2.DIST_L2, 5)
        expanded = candidate.copy()
        expanded[dist > expand_dist] = 0
        mask = seed | expanded
    else:
        mask = seed

    k = np.ones((morph_ksize, morph_ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


# ================================================================
# Main processing entry
# ================================================================

def process_frame(frame_id, method="diff", **kwargs):
    """Process a single frame pair and save the motion mask.

    Args:
        frame_id: e.g. '00004'
        method: 'diff' | 'flow' | 'hybrid' | 'adaptive' | 'seed_expand'
    """
    curr_path = CURRENT_DIR / f"{frame_id}_FV.png"
    prev_path = PREVIOUS_DIR / f"{frame_id}_FV_prev.png"

    curr = cv2.imread(str(curr_path))
    prev = cv2.imread(str(prev_path))

    if curr is None:
        print(f"[ERROR] Cannot load current frame: {frame_id}")
        return None
    if prev is None:
        print(f"[ERROR] Cannot load previous frame: {frame_id}")
        return None

    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    if method == "flow":
        mask = detect_motion_flow(prev_gray, curr_gray, **kwargs)
    elif method == "hybrid":
        mask = detect_motion_hybrid(prev_gray, curr_gray, **kwargs)
    elif method == "adaptive":
        mask = detect_motion_adaptive(prev_gray, curr_gray, **kwargs)
    elif method == "seed_expand":
        mask = detect_motion_seed_expand(prev_gray, curr_gray, **kwargs)
    elif method == "seed_expand_v2":
        mask = detect_motion_seed_expand_v2(prev_gray, curr_gray, **kwargs)
    elif method == "seed_expand_v3":
        mask = detect_motion_seed_expand_v3(prev_gray, curr_gray, **kwargs)
    elif method == "seed_expand_v4":
        mask = detect_motion_seed_expand_v4(prev_gray, curr_gray, **kwargs)
    elif method == "seed_expand_v5":
        mask = detect_motion_seed_expand_v5(prev_gray, curr_gray, **kwargs)
    elif method == "seed_expand_v6":
        mask = detect_motion_seed_expand_v6(prev_gray, curr_gray, **kwargs)
    elif method == "seed_expand_v7":
        mask = detect_motion_seed_expand_v7(prev_gray, curr_gray, **kwargs)
    elif method == "seed_expand_v8":
        mask = detect_motion_seed_expand_v8(prev_gray, curr_gray, **kwargs)
    elif method == "seed_expand_v9":
        mask = detect_motion_seed_expand_v9(prev_gray, curr_gray, **kwargs)
    elif method == "seed_expand_v10":
        mask = detect_motion_seed_expand_v10(prev_gray, curr_gray, **kwargs)
    elif method == "seed_expand_v11":
        mask = detect_motion_seed_expand_v11(prev_gray, curr_gray, **kwargs)
    else:
        mask = detect_motion_diff(prev_gray, curr_gray, **kwargs)

    save_path = OUTPUT_DIR / f"{frame_id}_mask.png"
    cv2.imwrite(str(save_path), mask)

    return mask


if __name__ == "__main__":
    frame_id = "00004"
    mask = process_frame(frame_id)
    if mask is not None:
        cv2.imshow("motion mask", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
