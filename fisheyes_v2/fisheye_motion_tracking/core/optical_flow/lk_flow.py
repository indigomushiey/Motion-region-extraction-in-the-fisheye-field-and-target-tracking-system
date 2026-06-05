import cv2
import numpy as np


def detect_corners_in_mask(gray, mask, max_corners=200, quality_level=0.3, min_distance=7):
    """Extract Shi-Tomasi corners within a binary motion mask."""
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        mask=mask,
        blockSize=7,
    )
    if corners is None:
        return np.empty((0, 2), dtype=np.float32)
    return corners.reshape(-1, 2)


def track_points_lk(prev_gray, curr_gray, prev_pts, win_size=(15, 15), max_level=2):
    """Track feature points from prev_gray to curr_gray using LK optical flow.

    Returns:
        curr_pts: tracked point positions in current frame
        status: 1 if track succeeded, 0 if lost
        error: tracking error per point
    """
    if len(prev_pts) == 0:
        return np.empty((0, 2), dtype=np.float32), None, None

    lk_params = dict(
        winSize=win_size,
        maxLevel=max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    curr_pts, status, error = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_pts, None, **lk_params
    )
    return curr_pts, status, error


def filter_valid_tracks(prev_pts, curr_pts, status, error, max_error=50.0):
    """Keep only points that were successfully tracked with low error."""
    if status is None or len(prev_pts) == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
        )
    status = status.reshape(-1)
    error = error.reshape(-1)
    valid = (status == 1) & (error < max_error)
    return prev_pts[valid], curr_pts[valid]


def compute_displacements(prev_pts, curr_pts):
    """Compute displacement vectors and statistics for tracked points.

    Returns dict with:
        vectors: (N, 2) displacement vectors (dx, dy)
        magnitudes: (N,) displacement magnitudes
        angles: (N,) displacement angles in radians
        mean_vec: (2,) mean displacement
        mean_magnitude: float
        mean_angle: float
        std_magnitude: float
    """
    if len(prev_pts) == 0:
        return {
            "vectors": np.empty((0, 2), dtype=np.float32),
            "magnitudes": np.empty((0,), dtype=np.float32),
            "angles": np.empty((0,), dtype=np.float32),
            "mean_vec": np.array([0.0, 0.0]),
            "mean_magnitude": 0.0,
            "mean_angle": 0.0,
            "std_magnitude": 0.0,
        }

    vecs = curr_pts - prev_pts
    mags = np.linalg.norm(vecs, axis=1)
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])

    return {
        "vectors": vecs,
        "magnitudes": mags,
        "angles": angles,
        "mean_vec": np.mean(vecs, axis=0),
        "mean_magnitude": np.mean(mags),
        "mean_angle": np.arctan2(np.mean(vecs[:, 1]), np.mean(vecs[:, 0])),
        "std_magnitude": np.std(mags),
    }


def compute_flow_for_regions(
    prev_gray, curr_gray, regions, motion_mask,
    max_corners_per_region=100, quality_level=0.3, min_distance=7,
    lk_win_size=(15, 15), max_track_error=50.0,
):
    """Run LK optical flow on each motion region independently.

    Args:
        prev_gray, curr_gray: grayscale frame pair
        regions: list of dicts from contour_detection.detect_motion_regions()
        motion_mask: binary motion mask (used to constrain corner detection)

    Returns:
        list of region flow results, each containing:
            region_idx, bbox, center, area,
            prev_pts, curr_pts, displacements (dict from compute_displacements)
    """
    results = []

    for idx, region in enumerate(regions):
        x, y, w, h = region["bbox"]

        # constrain corner detection to this region's bounding box
        region_mask = np.zeros_like(motion_mask)
        region_mask[y : y + h, x : x + w] = motion_mask[y : y + h, x : x + w]

        prev_pts = detect_corners_in_mask(
            prev_gray, region_mask,
            max_corners=max_corners_per_region,
            quality_level=quality_level,
            min_distance=min_distance,
        )

        if len(prev_pts) == 0:
            continue

        curr_pts, status, error = track_points_lk(
            prev_gray, curr_gray, prev_pts, win_size=lk_win_size
        )
        prev_valid, curr_valid = filter_valid_tracks(
            prev_pts, curr_pts, status, error, max_error=max_track_error
        )

        if len(prev_valid) < 3:
            continue

        disp = compute_displacements(prev_valid, curr_valid)

        results.append({
            "region_idx": idx,
            "bbox": region["bbox"],
            "center": region["center"],
            "area": region["area"],
            "prev_pts": prev_valid,
            "curr_pts": curr_valid,
            "displacements": disp,
        })

    return results