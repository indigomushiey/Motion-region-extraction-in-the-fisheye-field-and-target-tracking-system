"""Angular Flow: convert pixel flow to angular displacement (rad/frame).

Precomputes per-pixel 3D ray directions from radial_poly calibration.
At runtime, converts Farneback flow vectors to angular magnitude:
  ang_mag = arccos(ray(u,v) · ray(u+fx, v+fy))

Physical meaning: actual angular change on the unit sphere, independent of
fisheye distortion. Same angular change → same ang_mag everywhere in image.
"""

import json
import numpy as np
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
CALIB_DIR = DATA_ROOT / "homework2" / "calibration_data"

_RAY_MAP = None  # (h, w, 3) unit vectors


def _build_ray_map(calib_path=None):
    """Precompute unit ray directions for every fisheye pixel."""
    global _RAY_MAP
    if _RAY_MAP is not None:
        return _RAY_MAP

    if calib_path is None:
        calib_path = CALIB_DIR / "00000_FV.json"

    with open(calib_path, 'r') as f:
        intr = json.load(f)["intrinsic"]

    w, h = int(intr["width"]), int(intr["height"])
    cx = w / 2 + intr["cx_offset"]
    cy = h / 2 + intr["cy_offset"]
    k1, k2, k3, k4 = intr["k1"], intr["k2"], intr["k3"], intr["k4"]

    yy, xx = np.mgrid[:h, :w]
    dx = xx - cx; dy = yy - cy
    rd = np.sqrt(dx * dx + dy * dy)

    # Newton solve for theta
    theta = np.clip(rd / k1, 0, np.pi / 2 - 0.01)
    for _ in range(4):
        t2 = theta * theta; t3 = t2 * theta; t5 = t3 * t2; t7 = t5 * t2
        fv = k1 * theta + k2 * t3 + k3 * t5 + k4 * t7 - rd
        fd = k1 + 3 * k2 * t2 + 5 * k3 * t2 * t2 + 7 * k4 * t2 * t3
        theta = theta - fv / (fd + 1e-10)
    theta = np.clip(theta, 0, np.pi / 2 - 0.01)

    sin_t, cos_t = np.sin(theta), np.cos(theta)
    safe_r = np.where(rd > 0.5, rd, 1.0)

    # Camera ray: optical axis = z, right = x, down = y
    rx = sin_t * dx / safe_r
    ry = sin_t * dy / safe_r
    rz = cos_t

    # Normalize
    norm = np.sqrt(rx * rx + ry * ry + rz * rz)
    _RAY_MAP = np.stack([rx / norm, ry / norm, rz / norm], axis=-1).astype(np.float32)
    return _RAY_MAP


def fisheye_angular_mag(flow):
    """Convert fisheye-domain flow vectors to angular magnitude.

    Args:
        flow: (h, w, 2) Farneback flow in fisheye pixels

    Returns:
        ang_mag: (h, w) angular displacement in radians per frame
    """
    ray_map = _build_ray_map()
    h, w = flow.shape[:2]

    # Source pixel coordinates
    yy, xx = np.mgrid[:h, :w]
    # Destination (with flow)
    x2 = xx + flow[..., 0]
    y2 = yy + flow[..., 1]

    # Clip to image bounds
    x2 = np.clip(x2, 0, w - 1)
    y2 = np.clip(y2, 0, h - 1)

    # Bilinear interpolation of ray_map at (x2, y2)
    x0 = np.floor(x2).astype(int)
    y0 = np.floor(y2).astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = (x2 - x0).astype(np.float32)
    wy = (y2 - y0).astype(np.float32)

    r00 = ray_map[y0, x0]
    r10 = ray_map[y0, x1]
    r01 = ray_map[y1, x0]
    r11 = ray_map[y1, x1]

    # Interpolate
    r_top = r00 + (r10 - r00) * wx[..., None]
    r_bot = r01 + (r11 - r01) * wx[..., None]
    r_dst = r_top + (r_bot - r_top) * wy[..., None]

    # Normalize interpolated rays
    nd = np.sqrt((r_dst ** 2).sum(axis=-1))
    r_dst = r_dst / (nd[..., None] + 1e-10)

    # Dot product with source rays
    r_src = ray_map
    dot = (r_src * r_dst).sum(axis=-1)
    dot = np.clip(dot, -1.0, 1.0)

    ang = np.arccos(dot)
    return ang.astype(np.float32)


def perspective_angular_mag(flow, focal_length=339.749):
    """Convert perspective-domain flow to angular magnitude.

    In perspective images, angular displacement = atan2(pixel_displacement, f).
    """
    mag = np.linalg.norm(flow, axis=2)
    return np.arctan(mag / focal_length).astype(np.float32)
