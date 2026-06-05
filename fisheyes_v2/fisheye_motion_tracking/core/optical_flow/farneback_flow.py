import cv2
import numpy as np


def compute_farneback_flow(
    prev_gray, curr_gray,
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
):
    """Compute dense optical flow using Farneback method.

    Returns:
        flow: (H, W, 2) array of (dx, dy) vectors
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags,
    )
    return flow


def flow_to_hsv(flow, mag_max=None):
    """Convert optical flow to HSV color wheel visualization.

    Hue = flow direction, Saturation = 255 (fixed), Value = flow magnitude.
    Returns BGR image for display/writing with cv2.
    """
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    if mag_max is None:
        mag_max = np.percentile(mag, 95)

    mag = np.clip(mag, 0, mag_max)
    mag = mag / (mag_max + 1e-8) * 255

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = mag.astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def flow_magnitude_mask(flow, threshold=2.0):
    """Extract binary mask of pixels with motion magnitude above threshold."""
    mag = np.linalg.norm(flow, axis=2)
    return (mag > threshold).astype(np.uint8) * 255


def flow_magnitude(flow):
    """Return per-pixel flow magnitude."""
    return np.linalg.norm(flow, axis=2)


def flow_direction(flow):
    """Return per-pixel flow direction in radians."""
    return np.arctan2(flow[..., 1], flow[..., 0])


def compute_flow_stats(flow, mask=None):
    """Compute mean magnitude and dominant direction of flow, optionally within mask.

    Returns:
        mean_magnitude, std_magnitude, mean_angle, dominant_angle
    """
    mag = flow_magnitude(flow)
    ang = flow_direction(flow)

    if mask is not None and mask.any():
        mag = mag[mask > 0]
        ang = ang[mask > 0]

    if len(mag) == 0:
        return 0.0, 0.0, 0.0, 0.0

    # dominant direction via histogram
    hist, bins = np.histogram(ang, bins=36, range=(-np.pi, np.pi))
    dominant_angle = bins[np.argmax(hist)]

    return np.mean(mag), np.std(mag), np.mean(ang), dominant_angle