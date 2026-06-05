"""Ground-plane projection & residual flow using calibration parameters.

Precomputes per-pixel:
  - is_road: whether the pixel looks at the ground plane (z=0)
  - ground_dist: physical distance along ground to the intersection point
  - ray_vehicle: 3D ray direction in vehicle coordinates

At runtime:
  - residual_flow = actual_flow - expected_ego_flow(ground_dist, ego_speed)
"""

import json
import numpy as np
import cv2
from pathlib import Path

V2_DATA = Path(__file__).resolve().parents[2] / "fisheyes_v2" / "fisheye_motion_tracking" / "data"
CALIB_DIR = V2_DATA / "homework2" / "calibration_data"

# ── Quaternion rotation ─────────────────────────────────────────
def _quat_rotate(q, v):
    """Rotate vector(s) v by quaternion q = (w, x, y, z). v shape (..., 3)."""
    w, x, y, z = q
    r = np.array([x, y, z])
    t = 2 * np.cross(r, v)
    return v + w * t + np.cross(r, t)


class GroundProjector:
    """Precompute road mask + ground distances for a radial_poly fisheye camera."""

    def __init__(self, calib_path):
        with open(calib_path, 'r') as f:
            data = json.load(f)

        intr = data["intrinsic"]
        extr = data["extrinsic"]

        self.w = int(intr["width"])
        self.h = int(intr["height"])
        self.cx = self.w / 2 + intr["cx_offset"]
        self.cy = self.h / 2 + intr["cy_offset"]
        self.k1 = intr["k1"]; self.k2 = intr["k2"]
        self.k3 = intr["k3"]; self.k4 = intr["k4"]

        # Extrinsic: camera position in vehicle coords
        self.tx = extr["translation"][0]   # forward
        self.ty = extr["translation"][1]   # lateral
        self.tz = extr["translation"][2]   # height above ground
        self.q = (extr["quaternion"][0], extr["quaternion"][1],
                  extr["quaternion"][2], extr["quaternion"][3])

        # Precompute
        self._precompute_rays()
        self._precompute_ground()

    def _precompute_rays(self):
        """Compute 3D ray direction (vehicle coords) for every pixel."""
        yy, xx = np.mgrid[:self.h, :self.w]
        dx = xx - self.cx
        dy = yy - self.cy
        rd = np.sqrt(dx*dx + dy*dy)

        # Newton solve: f(θ) = k1*θ + k2*θ³ + k3*θ⁵ + k4*θ⁷ - rd = 0
        theta = np.clip(rd / self.k1, 0, np.pi/2 - 0.01)
        for _ in range(4):
            t2 = theta * theta
            t3 = t2 * theta; t5 = t3 * t2; t7 = t5 * t2
            f_val = self.k1*theta + self.k2*t3 + self.k3*t5 + self.k4*t7 - rd
            f_der = self.k1 + 3*self.k2*t2 + 5*self.k3*t2*t2 + 7*self.k4*t2*t3
            theta = theta - f_val / (f_der + 1e-10)
        theta = np.clip(theta, 0, np.pi/2 - 0.01)

        sin_t = np.sin(theta); cos_t = np.cos(theta)
        safe_r = np.where(rd > 0.5, rd, 1.0)

        # Camera ray: x=right, y=down, z=forward
        cam_x = sin_t * dx / safe_r
        cam_y = sin_t * dy / safe_r
        cam_z = cos_t

        cam_ray = np.stack([cam_x, cam_y, cam_z], axis=-1)  # (h, w, 3)

        # Rotate to vehicle coords
        self.ray = _quat_rotate(self.q, cam_ray)  # (h, w, 3)

    def _precompute_ground(self):
        """Find ground-plane intersection for each pixel."""
        # Ray in vehicle coords: start at (tx, ty, tz), direction = self.ray
        # Ground plane: z = 0
        # Intersection: tz + λ * ray_z = 0  →  λ = -tz / ray_z
        ray_z = self.ray[..., 2]
        ray_x = self.ray[..., 0]
        ray_y = self.ray[..., 1]

        # Looking downward = negative z component (in vehicle coords, z is up)
        looks_down = ray_z < -0.005

        lam = np.where(looks_down, -self.tz / ray_z, np.inf)
        gx = self.tx + lam * ray_x
        gy = self.ty + lam * ray_y
        dist = np.sqrt(gx*gx + gy*gy)  # horizontal distance on ground

        # Road: ground intersection between 2m and 80m
        self.is_road = looks_down & (dist > 2.0) & (dist < 80.0)
        self.ground_dist = np.where(self.is_road, dist, np.inf)

        # Expected flow direction for ego-motion (forward, v=1 for direction only)
        # A static ground point at (gx, gy) moves to (gx - v*dt, gy) relative to vehicle
        # Project both to image, difference = expected flow
        self._precompute_ego_direction()

    def _precompute_ego_direction(self):
        """Precompute unit expected flow field for forward ego-motion (v=1 m/s).

        For each ground pixel, compute how it moves in image space when the
        vehicle moves forward by 1 meter. This is a per-pixel direction + scale
        that gets multiplied by the actual ego-speed estimate at runtime.
        """
        # World positions of ground points (at time t)
        ray_z = self.ray[..., 2]; ray_x = self.ray[..., 0]; ray_y = self.ray[..., 1]
        lam = np.where(self.is_road, -self.tz / ray_z, 0.0)
        gx = self.tx + lam * ray_x
        gy = self.ty + lam * ray_y
        gz = np.zeros_like(gx)  # on ground plane

        # After vehicle moves forward by 1m: point shifts backward in vehicle frame
        gx_new = gx - 1.0
        gy_new = gy

        # Project old and new world points to fisheye image
        u_old, v_old = self._world_to_pixel(gx, gy, gz)
        u_new, v_new = self._world_to_pixel(gx_new, gy_new, gz)

        flow_x = u_new - u_old
        flow_y = v_new - v_old

        # Mask invalid (outside image) projections
        valid = (self.is_road &
                 (u_old >= 0) & (u_old < self.w) &
                 (v_old >= 0) & (v_old < self.h) &
                 (u_new >= 0) & (u_new < self.w) &
                 (v_new >= 0) & (v_new < self.h))

        self.ego_flow_x = np.where(valid, flow_x.astype(np.float32), 0.0)
        self.ego_flow_y = np.where(valid, flow_y.astype(np.float32), 0.0)
        self.ego_flow_valid = valid

    def _world_to_pixel(self, wx, wy, wz):
        """Project world point (vehicle coords) back to fisheye pixel.

        Inverse of _precompute_rays + _precompute_ground.
        """
        # World → camera coords (inverse rotation = conjugate quaternion)
        qc = (self.q[0], -self.q[1], -self.q[2], -self.q[3])
        # Point relative to camera
        dx = wx - self.tx; dy = wy - self.ty; dz = wz - self.tz
        world_vec = np.stack([dx, dy, dz], axis=-1)
        cam_vec = _quat_rotate(qc, world_vec)

        cx = cam_vec[..., 0]; cy = cam_vec[..., 1]; cz = cam_vec[..., 2]

        # Camera ray → incident angle
        r_u = np.sqrt(cx*cx + cy*cy) / (cz + 1e-10)
        theta = np.arctan(r_u)

        # Radial poly: θ → rd
        t2 = theta * theta; t3 = t2 * theta; t5 = t3 * t2; t7 = t5 * t2
        rd = self.k1*theta + self.k2*t3 + self.k3*t5 + self.k4*t7

        # Pixel coordinates
        safe_ru = np.where(r_u > 1e-6, r_u, 1.0)
        u = self.cx + cx / safe_ru * rd
        v = self.cy + cy / safe_ru * rd

        return u, v

    def compute_residual_flow(self, actual_flow, ego_speed=None):
        """Compute residual flow = actual_flow - expected_ego_flow.

        If ego_speed is None, estimate from median road-pixel flow.
        """
        if ego_speed is None:
            # Estimate speed from road pixels: median flow in ego direction
            road_actual = actual_flow[self.ego_flow_valid]
            road_expected = np.stack([self.ego_flow_x[self.ego_flow_valid],
                                       self.ego_flow_y[self.ego_flow_valid]], axis=-1)
            # Project actual onto expected direction for speed estimation
            # ego_speed = median(actual ⋅ expected / |expected|²)
            exp_norm2 = (road_expected ** 2).sum(axis=-1) + 1e-6
            proj = (road_actual * road_expected).sum(axis=-1) / exp_norm2
            ego_speed = float(np.median(proj[exp_norm2 > 0.1]))
            ego_speed = max(0.1, min(ego_speed, 10.0))  # clamp

        # Expected flow
        exp_fx = self.ego_flow_x * ego_speed
        exp_fy = self.ego_flow_y * ego_speed

        # Residual (only valid where we have ego-flow estimates)
        res_x = actual_flow[..., 0] - exp_fx
        res_y = actual_flow[..., 1] - exp_fy
        res_mag = np.sqrt(res_x*res_x + res_y*res_y)

        # Where ego-flow is not valid, fall back to raw flow
        raw_mag = np.linalg.norm(actual_flow, axis=2)
        res_mag = np.where(self.ego_flow_valid, res_mag, raw_mag)

        return res_mag, ego_speed


# Singleton — all frames share same calibration
_projector = None


def get_projector():
    global _projector
    if _projector is None:
        calib_path = CALIB_DIR / "00000_FV.json"
        _projector = GroundProjector(str(calib_path))
    return _projector
