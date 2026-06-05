"""
运动区域提取模块 — 帧差法、稠密光流、稀疏光流、背景建模、帧间对齐。
"""

import cv2
import numpy as np


# ============================================================
# 帧间对齐 — 补偿相机自运动
# ============================================================

def align_frames(prev_gray, curr_gray, transform_type="affine",
                 max_features=2000, min_inliers=10, ransac_thresh=3.0):
    """用特征匹配 + RANSAC 估计帧间背景运动，对齐两帧以补偿自运动。

    核心思路：
      1. ORB 提取两帧特征点并匹配
      2. RANSAC 估计主导运动（背景/ego-motion）的变换矩阵
      3. 将 prev 帧按此变换 warp 到 curr 帧坐标系
      4. 对齐后，静止背景像素重合，独立运动目标产生残差

    Args:
        prev_gray, curr_gray: 前后帧灰度图 (uint8)
        transform_type: "affine" (4DOF, 推荐) | "homography" (8DOF)
        max_features: ORB 最大特征点数
        min_inliers: RANSAC 最少内点数，不足则判定对齐失败
        ransac_thresh: RANSAC 重投影误差阈值 (px)

    Returns:
        dict: {
            'aligned':       对齐后的 prev 图像 (与 curr 同尺寸),
            'transform':     3×3 变换矩阵,
            'inliers':       内点数,
            'total_matches': 总匹配数,
            'success':       对齐是否成功,
        }
    """
    h, w = curr_gray.shape

    # 1. ORB 特征检测
    orb = cv2.ORB_create(nfeatures=max_features, scaleFactor=1.2, nlevels=8)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)

    if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
        return _alignment_failed(prev_gray)

    # 2. 暴力匹配 + 按距离排序取前 200
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 10:
        return _alignment_failed(prev_gray, len(matches))

    matches = sorted(matches, key=lambda m: m.distance)[:200]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 3. RANSAC 估计变换
    if transform_type == "affine":
        M23, inlier_mask = cv2.estimateAffine2D(
            pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh
        )
        M = np.eye(3, dtype=np.float32) if M23 is None else np.vstack([M23, [0, 0, 1]])
    else:
        M, inlier_mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)

    if M is None:
        return _alignment_failed(prev_gray, len(matches))

    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    if inliers < min_inliers:
        return _alignment_failed(prev_gray, len(matches), M, inliers)

    # 4. Warp prev → curr 坐标系
    aligned = cv2.warpPerspective(prev_gray, M, (w, h),
                                   flags=cv2.INTER_LINEAR)

    return {
        "aligned": aligned,
        "transform": M,
        "inliers": inliers,
        "total_matches": len(matches),
        "success": True,
    }


def _alignment_failed(prev_gray, total_matches=0,
                       transform=None, inliers=0):
    """构造对齐失败的返回。"""
    if transform is None:
        transform = np.eye(3, dtype=np.float32)
    return {
        "aligned": prev_gray,
        "transform": transform,
        "inliers": inliers,
        "total_matches": total_matches,
        "success": False,
    }


def frame_difference(prev_gray, curr_gray, threshold=15, blur_ksize=(5, 5)):
    """帧差法提取运动区域。

    Args:
        prev_gray, curr_gray: 前后帧灰度图
        threshold: 灰度差阈值
        blur_ksize: 预处理高斯模糊核大小

    Returns:
        motion_mask: 二值运动掩膜 (255=运动)
    """
    prev_blur = cv2.GaussianBlur(prev_gray, blur_ksize, 0)
    curr_blur = cv2.GaussianBlur(curr_gray, blur_ksize, 0)

    diff = cv2.absdiff(curr_blur, prev_blur)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return mask


def three_frame_difference(prev_gray, curr_gray, next_gray, threshold=15, blur_ksize=(5, 5)):
    """三帧差分法 — 取两个差分的交集以消除双影。"""
    diff1 = frame_difference(prev_gray, curr_gray, threshold, blur_ksize)
    diff2 = frame_difference(curr_gray, next_gray, threshold, blur_ksize)
    return cv2.bitwise_and(diff1, diff2)


def dense_optical_flow_farneback(prev_gray, curr_gray, params=None,
                                  compensate_ego_motion=False,
                                  cx=None, cy=None):
    """Farneback 稠密光流 + 运动区域提取。

    Args:
        prev_gray, curr_gray: 前后帧灰度图
        params: Farneback 参数，可含 mag_threshold, mag_percentile,
                ego_model ("median" | "radial")
        compensate_ego_motion: 是否补偿相机自运动
        cx, cy: 主点坐标 (径向模型需要)

    Returns:
        dict: {
            'flow': (H,W,2) 光流场,
            'magnitude': (H,W) 光流幅度,
            'angle': (H,W) 光流角度,
            'motion_mask': 二值运动掩膜,
            'residual_flow': 残差光流,
            'residual_magnitude': 残差幅度,
        }
    """
    if params is None:
        from config import FARNEBACK_PARAMS, FLOW_MAG_THRESHOLD, FLOW_MAG_PERCENTILE, CX, CY, EGO_MODEL
        params = dict(FARNEBACK_PARAMS)
        mag_th = FLOW_MAG_THRESHOLD
        mag_pct = FLOW_MAG_PERCENTILE
        ego_model = EGO_MODEL
        if cx is None: cx = CX
        if cy is None: cy = CY
    else:
        params = dict(params)
        mag_th = params.pop("mag_threshold", 5.0)
        mag_pct = params.pop("mag_percentile", 0)
        ego_model = params.pop("ego_model", "radial")
        if cx is None:
            from config import CX as _CX; cx = _CX
        if cy is None:
            from config import CY as _CY; cy = _CY

    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **params)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    result = {
        "flow": flow,
        "magnitude": mag,
        "angle": ang,
    }

    if compensate_ego_motion:
        residual_flow = _compensate_ego_motion(flow, cx, cy, model=ego_model)
        residual_mag = np.sqrt(
            residual_flow[..., 0]**2 + residual_flow[..., 1]**2
        )
        if mag_pct > 0:
            eff_th = np.percentile(residual_mag, mag_pct)
        else:
            eff_th = mag_th
        _, mask = cv2.threshold(residual_mag, eff_th, 255, cv2.THRESH_BINARY)
        result["residual_flow"] = residual_flow
        result["residual_magnitude"] = residual_mag
        result["motion_mask"] = mask.astype(np.uint8)
        result["effective_threshold"] = eff_th
    else:
        if mag_pct > 0:
            eff_th = np.percentile(mag, mag_pct)
        else:
            eff_th = mag_th
        _, mask = cv2.threshold(mag, eff_th, 255, cv2.THRESH_BINARY)
        result["motion_mask"] = mask.astype(np.uint8)
        result["effective_threshold"] = eff_th

    return result


def _compensate_ego_motion(flow, cx, cy, model="radial"):
    """补偿相机自运动光流。

    支持三种模型:
      "median"  — 全局中位数减法
      "radial"  — 径向展开模型: 期望光流 ∝ (x-cx, y-cy)
      "local"   — 局部中位数减法: 大核中值滤波捕获空间变化的背景光流
    """
    h, w = flow.shape[:2]

    if model == "median":
        return flow - np.median(flow, axis=(0, 1))

    if model == "local":
        # 大核均值滤波: 背景光流平滑变化, 运动目标=光流场中的"边缘"
        # 核足够大时局部均值≈局部背景光流 (运动目标占比小)
        from config import IMG_W, IMG_H
        ks = max(31, min(IMG_W, IMG_H) // 15)
        if ks % 2 == 0: ks += 1
        kernel = np.ones((ks, ks), np.float32) / (ks * ks)
        local_x = cv2.filter2D(flow[..., 0], -1, kernel)
        local_y = cv2.filter2D(flow[..., 1], -1, kernel)
        residual = flow.copy()
        residual[..., 0] -= local_x
        residual[..., 1] -= local_y
        return residual

    # ---- 径向展开模型 ----
    y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    dx = x_grid - cx
    dy = y_grid - cy
    r = np.sqrt(dx**2 + dy**2)

    flow_radial = (flow[..., 0] * dx + flow[..., 1] * dy) / (r + 1e-8)

    r_max = np.sqrt(max(cx, w - cx)**2 + max(cy, h - cy)**2)
    center = r < 0.6 * r_max
    valid = center & (r > 30)
    if valid.sum() > 1000:
        k = np.median(flow_radial[valid] / r[valid])
    else:
        k = 0.0

    expected_x = k * dx
    expected_y = k * dy

    residual = flow.copy()
    residual[..., 0] -= expected_x
    residual[..., 1] -= expected_y

    return residual


def sparse_optical_flow_lk(prev_gray, curr_gray, prev_pts=None,
                           feature_params=None, lk_params=None):
    """Lucas-Kanade 稀疏光流 + 特征点跟踪。

    Args:
        prev_gray, curr_gray: 前后帧灰度图
        prev_pts: 预先指定的特征点 (N×1×2), None 则自动检测
        feature_params: Shi-Tomasi 参数
        lk_params: LK 参数

    Returns:
        dict: {
            'prev_pts': 前一帧特征点,
            'next_pts': 跟踪到的当前帧特征点,
            'displacements': 各点位移量,
            'status': 跟踪成功标志,
            'motion_points': 运动幅度大于阈值的点
        }
    """
    if feature_params is None:
        from config import SHI_TOMASI_PARAMS
        feature_params = SHI_TOMASI_PARAMS
    if lk_params is None:
        from config import LK_PARAMS
        lk_params = LK_PARAMS
    from config import LK_TRACK_MIN_DISPLACEMENT
    min_disp = LK_TRACK_MIN_DISPLACEMENT

    if prev_pts is None:
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    if prev_pts is None or len(prev_pts) == 0:
        return {
            "prev_pts": np.array([]),
            "next_pts": np.array([]),
            "displacements": np.array([]),
            "status": np.array([]),
            "motion_points": np.array([]),
        }

    next_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_pts, None, **lk_params
    )

    # 筛选成功跟踪的点
    status = status.flatten()
    prev_pts_good = prev_pts[status == 1]
    next_pts_good = next_pts[status == 1]

    if len(prev_pts_good) == 0:
        return {
            "prev_pts": np.array([]),
            "next_pts": np.array([]),
            "displacements": np.array([]),
            "status": status,
            "motion_points": np.array([]),
        }

    # 计算位移
    displacements = np.linalg.norm(next_pts_good - prev_pts_good, axis=1)
    motion_mask = displacements > min_disp

    return {
        "prev_pts": prev_pts_good,
        "next_pts": next_pts_good,
        "displacements": displacements,
        "status": status,
        "motion_points": next_pts_good[motion_mask],
        "prev_motion_points": prev_pts_good[motion_mask],
    }


def background_subtraction_mog2(frames, history=100, var_threshold=20):
    """MOG2 背景建模 (适用于连续序列)。

    Args:
        frames: 帧列表 [frame0, frame1, ...]
        history: 背景模型历史帧数
        var_threshold: 方差阈值

    Returns:
        fg_masks: 每帧前景掩膜列表
    """
    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=history, varThreshold=var_threshold, detectShadows=False
    )
    masks = []
    for frame in frames:
        fg = mog2.apply(frame)
        masks.append(fg)
    return masks


def background_subtraction_knn(frames, history=100, dist2_threshold=400.0):
    """KNN 背景建模 (适用于连续序列)。"""
    knn = cv2.createBackgroundSubtractorKNN(
        history=history, dist2Threshold=dist2_threshold, detectShadows=False
    )
    masks = []
    for frame in frames:
        fg = knn.apply(frame)
        masks.append(fg)
    return masks


def extract_motion(prev_gray, curr_gray, method="frame_diff",
                   compensate_ego_motion=False, **kwargs):
    """统一运动提取接口。

    Args:
        prev_gray, curr_gray: 前后帧灰度图
        method: "frame_diff" | "optical_flow" | "sparse_optical_flow"
        compensate_ego_motion: 是否补偿相机自运动
        **kwargs: 传递给具体方法的参数

    Returns:
        dict: 包含 'motion_mask' 和具体方法的结果
    """
    if method == "frame_diff":
        threshold = kwargs.get("threshold", 15)
        blur_ksize = kwargs.get("blur_ksize", (5, 5))
        mask = frame_difference(prev_gray, curr_gray, threshold, blur_ksize)
        return {"motion_mask": mask}

    elif method == "optical_flow":
        params = kwargs.get("farneback_params", None)
        result = dense_optical_flow_farneback(
            prev_gray, curr_gray, params,
            compensate_ego_motion=compensate_ego_motion
        )
        return result

    elif method == "sparse_optical_flow":
        result = sparse_optical_flow_lk(prev_gray, curr_gray, **kwargs)
        return result

    else:
        raise ValueError(f"Unknown motion extraction method: {method}")
