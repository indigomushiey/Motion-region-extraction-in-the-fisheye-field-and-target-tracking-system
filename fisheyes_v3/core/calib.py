"""Fast vectorized radial_poly undistort + inverse reprojection for Route B.

Replaces the slow nested-loop undistort in v2 with numpy-vectorized ops.
Provides both forward (fisheye → perspective) and inverse (perspective → fisheye)
remap tables for image undistortion and mask reprojection.
"""

import json
import numpy as np
import cv2
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
CALIB_DIR = DATA_ROOT / "homework2" / "calibration_data"


class RadialPoly:
    """Radial polynomial fisheye model with precomputed remap tables."""

    def __init__(self, json_path, output_size=None, fov_scale=1.0):
        with open(json_path, 'r') as f:
            data = json.load(f)
        intr = data["intrinsic"]

        self.w = int(intr["width"])
        self.h = int(intr["height"])
        self.cx = self.w / 2 + intr["cx_offset"]
        self.cy = self.h / 2 + intr["cy_offset"]
        self.k1 = intr["k1"]
        self.k2 = intr["k2"]
        self.k3 = intr["k3"]
        self.k4 = intr["k4"]

        # Output perspective image size
        self.out_w = output_size[0] if output_size else self.w
        self.out_h = output_size[1] if output_size else self.h
        self.f = self.k1 * fov_scale

        # Precompute remap tables once
        self._map_x_fwd = None
        self._map_y_fwd = None
        self._map_x_inv = None
        self._map_y_inv = None

    # ── Forward: perspective → fisheye (undistort) ──────────────
    def _build_forward_maps(self):
        """Build remap tables for fisheye→perspective (what undistort uses)."""
        yy, xx = np.mgrid[:self.out_h, :self.out_w]

        xn = (xx - self.cx) / self.f
        yn = (yy - self.cy) / self.f
        r_u = np.sqrt(xn * xn + yn * yn)

        theta = np.arctan(r_u)
        t2 = theta * theta
        t3 = t2 * theta
        t5 = t3 * t2
        t7 = t5 * t2

        r_d = self.k1 * theta + self.k2 * t3 + self.k3 * t5 + self.k4 * t7

        # Avoid division by zero
        safe = np.where(r_u > 1e-8, r_u, 1.0)
        scale = np.where(r_u > 1e-8, r_d / safe, 1.0)

        self._map_x_fwd = (self.cx + xn * scale).astype(np.float32)
        self._map_y_fwd = (self.cy + yn * scale).astype(np.float32)

    def undistort(self, image):
        """Undistort a fisheye image to perspective view."""
        if self._map_x_fwd is None:
            self._build_forward_maps()
        return cv2.remap(image, self._map_x_fwd, self._map_y_fwd,
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # ── Inverse: fisheye → perspective (reproject mask) ─────────
    def _build_inverse_maps(self):
        """Build remap tables for perspective→fisheye (mask reprojection).

        For each fisheye pixel (xf, yf), compute where it maps to in the
        perspective output.  Uses Newton's method to invert radial_poly.
        """
        yy, xx = np.mgrid[:self.h, :self.w]
        xn = (xx - self.cx) / self.k1  # initial scale for Newton
        yn = (yy - self.cy) / self.k1
        r_d = np.sqrt(xn * xn + yn * yn)  # distorted radius / k1

        # Newton: solve f(theta) = r_d where f = k1*theta + k2*theta³ + k3*theta⁵ + k4*theta⁷
        # Use k1*theta as initial guess → theta0 = r_d / k1_ratio where k1_ratio=1
        # Actually r_d here is already divided by k1. Let me recompute.
        # r_d_actual = sqrt((xf-cx)² + (yf-cy)²)
        # We need theta such that: k1*theta + k2*theta³ + k3*theta⁵ + k4*theta⁷ = r_d_actual

        r_d_actual = np.sqrt((xx - self.cx) ** 2 + (yy - self.cy) ** 2)

        # Newton initial guess: theta ≈ r_d / k1 (linear term dominates)
        theta = r_d_actual / self.k1

        for _ in range(5):
            t2 = theta * theta
            t3 = t2 * theta
            t5 = t3 * t2
            t7 = t5 * t2

            f_val = self.k1 * theta + self.k2 * t3 + self.k3 * t5 + self.k4 * t7 - r_d_actual
            f_deriv = self.k1 + 3 * self.k2 * t2 + 5 * self.k3 * t2 * t2 + 7 * self.k4 * t2 * t3

            theta = theta - f_val / (f_deriv + 1e-10)

        # Back to undistorted radius: r_u = tan(theta)
        r_u = np.tan(np.clip(theta, 0, np.pi / 2 - 0.01))

        # Scale xn, yn to perspective coordinates
        safe_r = np.where(r_d_actual > 0.5, r_d_actual, 1.0)
        scale = np.where(r_d_actual > 0.5, (r_u / safe_r) * self.f, self.f)

        self._map_x_inv = (self.cx + (xx - self.cx) * scale).astype(np.float32)
        self._map_y_inv = (self.cy + (yy - self.cy) * scale).astype(np.float32)

    def reproject_to_fisheye(self, persp_mask):
        """Project a perspective-domain mask back to fisheye coordinates."""
        if self._map_x_inv is None:
            self._build_inverse_maps()
        return cv2.remap(persp_mask, self._map_x_inv, self._map_y_inv,
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def load_camera(frame_id, fov_scale=1.0):
    """Load RadialPoly for a given frame."""
    json_path = CALIB_DIR / f"{frame_id}_FV.json"
    return RadialPoly(json_path, fov_scale=fov_scale)
