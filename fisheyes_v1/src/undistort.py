"""
去畸变/校正模块 — 实现 radial_poly → 透视/柱面/等距矩形投影的重映射。
"""

import numpy as np
import cv2
from radial_poly import theta_from_r, r_theta


def compute_undistort_maps_perspective(k1, k2, k3, k4, cx, cy, img_w, img_h,
                                       out_w, out_h, focal, balance=0.5):
    """预计算透视投影去畸变映射表。

    透视投影: r_new = focal * tan(θ)

    Args:
        k1-k4: radial_poly 系数
        cx, cy: 原始图像主点
        img_w, img_h: 原始图像尺寸
        out_w, out_h: 输出图像尺寸
        focal: 透视投影焦距
        balance: FOV 保留权衡参数

    Returns:
        map_x, map_y: (out_h, out_w) 映射表，用于 cv2.remap
    """
    map_x = np.zeros((out_h, out_w), dtype=np.float32)
    map_y = np.zeros((out_h, out_w), dtype=np.float32)

    out_cx, out_cy = out_w / 2.0, out_h / 2.0

    for v in range(out_h):
        for u in range(out_w):
            # 校正图像坐标 → 归一化方向
            dx = (u - out_cx) / focal
            dy = (v - out_cy) / focal
            r_new = np.sqrt(dx**2 + dy**2)

            theta = np.arctan(r_new)  # 透视投影反函数

            if theta < 1e-10:
                src_u, src_v = cx, cy
            else:
                r_img = r_theta(theta, k1, k2, k3, k4)
                cos_phi = dx / r_new if r_new > 1e-10 else 1.0
                sin_phi = dy / r_new if r_new > 1e-10 else 0.0
                src_u = cx + r_img * cos_phi
                src_v = cy + r_img * sin_phi

            map_x[v, u] = src_u
            map_y[v, u] = src_v

    return map_x, map_y


def compute_undistort_maps_perspective_fast(k1, k2, k3, k4, cx, cy, img_w, img_h,
                                            out_w, out_h, focal):
    """向量化版本：预计算透视投影去畸变映射表 (更快)。"""
    # 校正图像坐标网格
    u_grid, v_grid = np.meshgrid(np.arange(out_w), np.arange(out_h))
    out_cx, out_cy = out_w / 2.0, out_h / 2.0

    dx = (u_grid - out_cx) / focal
    dy = (v_grid - out_cy) / focal
    r_new = np.sqrt(dx**2 + dy**2)

    theta = np.arctan(r_new)

    # 逐像素反求 (当前为保持精度仍用循环，可优化为 LUT)
    map_x = np.zeros_like(r_new, dtype=np.float32)
    map_y = np.zeros_like(r_new, dtype=np.float32)

    r_img = r_theta(theta, k1, k2, k3, k4)

    mask = r_new > 1e-10
    cos_phi = np.ones_like(r_new)
    sin_phi = np.zeros_like(r_new)
    cos_phi[mask] = dx[mask] / r_new[mask]
    sin_phi[mask] = dy[mask] / r_new[mask]

    map_x = cx + r_img * cos_phi
    map_y = cy + r_img * sin_phi

    # 边界检查：标记超出原始图像范围的点为 -1
    out_of_bounds = (map_x < 0) | (map_x >= img_w) | (map_y < 0) | (map_y >= img_h)
    map_x[out_of_bounds] = -1
    map_y[out_of_bounds] = -1

    return map_x.astype(np.float32), map_y.astype(np.float32)


def compute_undistort_maps_cylindrical(k1, k2, k3, k4, cx, cy, img_w, img_h,
                                       out_w, out_h, focal):
    """预计算柱面投影去畸变映射表。

    柱面投影: x_new = focal * φ (水平角度), y_new = focal * tan(θ_y)
    """
    map_x = np.zeros((out_h, out_w), dtype=np.float32)
    map_y = np.zeros((out_h, out_w), dtype=np.float32)

    out_cx, out_cy = out_w / 2.0, out_h / 2.0

    for v in range(out_h):
        for u in range(out_w):
            phi = (u - out_cx) / focal
            z_cyl = (v - out_cy) / focal

            # 从圆柱坐标反算 3D 方向
            x_3d = np.sin(phi)
            y_3d = z_cyl
            z_3d = np.cos(phi)

            r_3d = np.sqrt(x_3d**2 + y_3d**2)
            theta = np.arctan2(r_3d, z_3d)

            if theta < 1e-10:
                src_u, src_v = cx, cy
            else:
                r_img = r_theta(theta, k1, k2, k3, k4)
                cos_phi_src = x_3d / r_3d if r_3d > 1e-10 else 1.0
                sin_phi_src = y_3d / r_3d if r_3d > 1e-10 else 0.0
                src_u = cx + r_img * cos_phi_src
                src_v = cy + r_img * sin_phi_src

            map_x[v, u] = src_u
            map_y[v, u] = src_v

    return map_x, map_y


def compute_undistort_maps_equirectangular(k1, k2, k3, k4, cx, cy, img_w, img_h,
                                           out_w, out_h):
    """预计算等距矩形投影 (Equirectangular) 去畸变映射表。

    经纬度展开: x ∝ φ, y ∝ θ
    """
    map_x = np.zeros((out_h, out_w), dtype=np.float32)
    map_y = np.zeros((out_h, out_w), dtype=np.float32)

    for v in range(out_h):
        theta = (v + 0.5) / out_h * np.pi / 1.8  # 覆盖约 100° 垂直
        for u in range(out_w):
            phi = (u + 0.5) / out_w * 2.0 * np.pi

            r_img = r_theta(theta, k1, k2, k3, k4)
            src_u = cx + r_img * np.cos(phi)
            src_v = cy + r_img * np.sin(phi)

            map_x[v, u] = src_u
            map_y[v, u] = src_v

    return map_x, map_y


class Undistorter:
    """鱼眼图像去畸变器。"""

    def __init__(self, k1, k2, k3, k4, cx, cy, img_w, img_h):
        self.k1, self.k2 = k1, k2
        self.k3, self.k4 = k3, k4
        self.cx, self.cy = cx, cy
        self.img_w, self.img_h = img_w, img_h
        self._maps = {}  # 缓存映射表

    def get_maps(self, projection="perspective", out_size=None, focal=None):
        """获取指定投影的 remap 映射表 (带缓存)。"""
        out_w, out_h = out_size or (self.img_w, self.img_h)
        focal = focal or out_w / 2.0

        key = (projection, out_w, out_h, focal)
        if key not in self._maps:
            if projection == "perspective":
                mx, my = compute_undistort_maps_perspective_fast(
                    self.k1, self.k2, self.k3, self.k4,
                    self.cx, self.cy, self.img_w, self.img_h,
                    out_w, out_h, focal
                )
            elif projection == "cylindrical":
                mx, my = compute_undistort_maps_cylindrical(
                    self.k1, self.k2, self.k3, self.k4,
                    self.cx, self.cy, self.img_w, self.img_h,
                    out_w, out_h, focal
                )
            elif projection == "equirectangular":
                mx, my = compute_undistort_maps_equirectangular(
                    self.k1, self.k2, self.k3, self.k4,
                    self.cx, self.cy, self.img_w, self.img_h,
                    out_w, out_h
                )
            else:
                raise ValueError(f"Unknown projection: {projection}")
            self._maps[key] = (mx, my)
        return self._maps[key]

    def undistort(self, image, projection="perspective", out_size=None, focal=None):
        """对图像进行去畸变。

        Returns:
            undistorted: 去畸变后图像
            valid_mask: 有效像素掩膜
        """
        map_x, map_y = self.get_maps(projection, out_size, focal)
        out_h, out_w = map_x.shape[:2]

        undistorted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        valid_mask = (map_x >= 0) & (map_y >= 0) & (map_x < self.img_w) & (map_y < self.img_h)
        valid_mask = valid_mask.astype(np.uint8) * 255

        return undistorted, valid_mask

    def warp_mask_to_fisheye(self, mask_undistorted, projection="perspective",
                              out_size=None, focal=None):
        """将校正域的掩膜反向映射回鱼眼域 (向量化实现)。

        用于在鱼眼图像坐标系中评估校正域结果。
        使用 numpy.bincount 替代逐像素循环，加速 ~100×。
        """
        map_x, map_y = self.get_maps(projection, out_size, focal)
        out_h, out_w = map_x.shape[:2]

        # 展平所有数组
        sx = map_x.ravel()
        sy = map_y.ravel()
        mask_flat = mask_undistorted.ravel().astype(np.float64)

        # 筛选有效源坐标
        valid = (sx >= 0) & (sx < self.img_w) & (sy >= 0) & (sy < self.img_h)
        sx_v = sx[valid].astype(np.int32)
        sy_v = sy[valid].astype(np.int32)
        mask_v = mask_flat[valid]

        if len(sx_v) == 0:
            return np.zeros((self.img_h, self.img_w), dtype=np.uint8)

        # 用 raveled 一维索引做 bincount 累加
        flat_idx = sy_v * self.img_w + sx_v
        total_size = self.img_h * self.img_w

        acc = np.bincount(flat_idx, weights=mask_v, minlength=total_size)
        cnt = np.bincount(flat_idx, minlength=total_size)

        # 重塑回 2D
        acc_2d = acc.reshape(self.img_h, self.img_w)
        cnt_2d = cnt.reshape(self.img_h, self.img_w)

        # 归一化: 多个校正像素映射到同一鱼眼像素时取平均
        valid_w = cnt_2d > 0
        acc_2d[valid_w] /= cnt_2d[valid_w]

        return (acc_2d > 128).astype(np.uint8) * 255
