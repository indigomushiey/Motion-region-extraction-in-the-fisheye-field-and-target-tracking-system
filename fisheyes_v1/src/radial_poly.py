"""
WoodScape radial_poly 相机模型 — 正向投影与反向投影。

投影模型: r(θ) = k1·θ + k2·θ³ + k3·θ⁵ + k4·θ⁷
其中 θ = arctan(sqrt(X²+Y²)/Z) 为入射角，r 为像点到主点的距离。
"""

import numpy as np
from scipy.optimize import newton


def r_theta(theta, k1, k2, k3, k4):
    """正向: θ → r (像高)"""
    t2 = theta * theta
    t4 = t2 * t2
    t6 = t4 * t2
    return k1 * theta + k2 * theta * t2 + k3 * theta * t4 + k4 * theta * t6


def dr_dtheta(theta, k1, k2, k3, k4):
    """r(θ) 的导数 dr/dθ，用于 Newton 迭代和尺度因子计算。"""
    t2 = theta * theta
    t4 = t2 * t2
    t6 = t4 * t2
    return k1 + 3 * k2 * t2 + 5 * k3 * t4 + 7 * k4 * t6


def theta_from_r(r, k1, k2, k3, k4):
    """反向: r → θ，使用 Newton 迭代求解。

    Args:
        r: 像点到主点的距离
        k1-k4: radial_poly 系数

    Returns:
        θ 入射角 (弧度)
    """
    if r <= 0:
        return 0.0
    # 初始猜测: 假设线性 r ≈ k1*θ
    theta0 = r / k1
    try:
        theta = newton(
            lambda t: r_theta(t, k1, k2, k3, k4) - r,
            theta0,
            fprime=lambda t: dr_dtheta(t, k1, k2, k3, k4),
            maxiter=100, tol=1e-10
        )
    except RuntimeError:
        # Newton 不收敛时回退到二分法
        theta = _theta_bisect(r, k1, k2, k3, k4)
    if theta < 0:
        theta = 0.0
    return theta


def _theta_bisect(r, k1, k2, k3, k4, lo=0.0, hi=np.pi / 2.0 + 0.5):
    """二分法求 θ (Newton 失败时的后备方案)。"""
    for _ in range(80):
        mid = (lo + hi) / 2.0
        val = r_theta(mid, k1, k2, k3, k4)
        if abs(val - r) < 1e-8:
            return mid
        if val < r:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def project_3d_to_2d(points_3d, k1, k2, k3, k4, cx, cy):
    """正向投影: 3D 相机坐标 → 2D 畸变像素坐标。

    Args:
        points_3d: (N, 3) 相机坐标系下的 3D 点
        k1-k4: radial_poly 系数
        cx, cy: 主点坐标

    Returns:
        (N, 2) 畸变图像坐标
    """
    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    r_3d = np.sqrt(x**2 + y**2)
    # 避免除零
    mask = r_3d > 1e-10
    theta = np.zeros_like(r_3d)
    theta[mask] = np.arctan2(r_3d[mask], z[mask])
    # 对 z<0 的点 (后方)，设 θ = π/2
    theta[~mask] = 0.0

    r_img = r_theta(theta, k1, k2, k3, k4)
    cos_phi = np.ones_like(x)
    sin_phi = np.zeros_like(x)
    cos_phi[mask] = x[mask] / r_3d[mask]
    sin_phi[mask] = y[mask] / r_3d[mask]

    u = cx + r_img * cos_phi
    v = cy + r_img * sin_phi
    return np.stack([u, v], axis=1)


def undistort_points(points_2d, k1, k2, k3, k4, cx, cy, new_focal=None):
    """对畸变图像点去畸变，返回透视投影坐标或归一化方向。

    步骤:
    1. 从畸变坐标计算径向距离 r
    2. 反求入射角 θ = f⁻¹(r)
    3. 用透视投影 r_new = f_new * tan(θ) 计算去畸变坐标

    Args:
        points_2d: (N, 2) 畸变图像坐标
        k1-k4: radial_poly 系数
        cx, cy: 主点
        new_focal: 新焦距, None 则返回归一化方向角

    Returns:
        (N, 2) 去畸变坐标 或 归一化方向
    """
    dx = points_2d[:, 0] - cx
    dy = points_2d[:, 1] - cy
    r = np.sqrt(dx**2 + dy**2)

    # 逐点反求 θ
    theta = np.array([theta_from_r(ri, k1, k2, k3, k4) for ri in r])

    if new_focal is None:
        # 返回归一化方向 (θ, φ)
        phi = np.arctan2(dy, dx)
        return np.stack([theta, phi], axis=1)

    # 透视投影: r_new = f * tan(θ)
    r_new = new_focal * np.tan(theta)
    # 保持方向角 φ
    cos_phi = np.ones_like(r)
    sin_phi = np.zeros_like(r)
    mask = r > 1e-10
    cos_phi[mask] = dx[mask] / r[mask]
    sin_phi[mask] = dy[mask] / r[mask]

    u_new = r_new * cos_phi
    v_new = r_new * sin_phi
    return np.stack([u_new, v_new], axis=1)


def scale_factor_at_pixel(u, v, k1, k2, k3, k4, cx, cy):
    """计算像素 (u,v) 处的局部尺度因子 dr/dθ。

    用于鱼眼域的自适应阈值: 边缘区域单位像素对应更小的物理角度变化。

    Returns:
        dθ/dr 的近似值 (rad/px)，值越小表示该位置越被"拉伸"
    """
    dx = u - cx
    dy = v - cy
    r = np.sqrt(dx**2 + dy**2)
    if r < 1e-6:
        theta = 0.0
    else:
        theta = theta_from_r(r, k1, k2, k3, k4)
    dr = dr_dtheta(theta, k1, k2, k3, k4)
    if dr < 1e-10:
        return 1.0
    return 1.0 / dr  # dθ/dr


def build_theta_lut(max_r, n_bins=2000, k1=None, k2=None, k3=None, k4=None):
    """构建 r → θ 的查找表用于加速。

    Returns:
        r_bins: (n_bins,) 离散半径
        theta_lut: (n_bins,) 对应的 θ
    """
    if k1 is None:
        from config import K1, K2, K3, K4
        k1, k2, k3, k4 = K1, K2, K3, K4
    r_bins = np.linspace(0, max_r, n_bins)
    theta_lut = np.array([theta_from_r(r, k1, k2, k3, k4) for r in r_bins])
    return r_bins, theta_lut


def theta_from_r_lut(r, r_bins, theta_lut):
    """使用 LUT 快速查找 θ。"""
    idx = np.searchsorted(r_bins, r)
    idx = np.clip(idx, 1, len(r_bins) - 1)
    # 线性插值
    r0, r1 = r_bins[idx - 1], r_bins[idx]
    t0, t1 = theta_lut[idx - 1], theta_lut[idx]
    frac = (r - r0) / (r1 - r0 + 1e-10)
    return t0 + frac * (t1 - t0)


def validate_model(k1=None, k2=None, k3=None, k4=None, cx=None, cy=None):
    """验证模型: 正向投影 → 反向投影 的一致性。

    Returns:
        dict: 包含最大误差等信息
    """
    if k1 is None:
        from config import K1, K2, K3, K4, CX, CY
        k1, k2, k3, k4 = K1, K2, K3, K4
        cx, cy = CX, CY

    # 生成测试点 (在相机前方半球)
    np.random.seed(42)
    theta_test = np.random.uniform(0, np.pi / 2.2, 500)
    phi_test = np.random.uniform(0, 2 * np.pi, 500)
    x = np.sin(theta_test) * np.cos(phi_test)
    y = np.sin(theta_test) * np.sin(phi_test)
    z = np.cos(theta_test)
    pts_3d = np.stack([x, y, z], axis=1)

    # 正向 → 畸变坐标
    pts_2d = project_3d_to_2d(pts_3d, k1, k2, k3, k4, cx, cy)

    # 反向 → 归一化方向
    recovered = undistort_points(pts_2d, k1, k2, k3, k4, cx, cy, new_focal=None)
    theta_rec = recovered[:, 0]

    errors = np.abs(theta_rec - theta_test)
    return {
        "max_error_rad": np.max(errors),
        "mean_error_rad": np.mean(errors),
        "max_error_deg": np.degrees(np.max(errors)),
        "mean_error_deg": np.degrees(np.mean(errors)),
        "passed": np.max(errors) < 1e-4,
    }
