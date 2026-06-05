"""
后处理模块 — 形态学操作、连通域分析、区域过滤、基于光流幅度的验证。
"""

import cv2
import numpy as np


def postprocess_mask_enhanced(motion_mask, residual_magnitude, median_ksize=5,
                               open_ksize=(3, 3), close_ksize=(7, 7),
                               min_area=50, min_solidity=0.3,
                               min_flow_ratio=1.5):
    """增强版运动掩膜后处理 — 结合光流残差幅度验证候选区域。

    在标准后处理基础上增加:
      1. 区域紧致度过滤 (solidity = area / convex_hull_area)
      2. 区域内平均残差光流幅度验证 (> min_flow_ratio * 背景中位数)

    Args:
        motion_mask: 输入二值掩膜
        residual_magnitude: 残差光流幅度图 (H,W), 与 motion_mask 同尺寸
        median_ksize: 中值滤波核
        open_ksize, close_ksize: 形态学核
        min_area: 最小连通域面积
        min_solidity: 最小紧致度 (0-1)
        min_flow_ratio: 区域内平均残差幅度 / 全局背景中位数 的最小比值

    Returns:
        dict: 同 postprocess_mask
    """
    if motion_mask.max() <= 1:
        motion_mask = (motion_mask * 255).astype(np.uint8)

    # 1. 中值滤波
    mask = cv2.medianBlur(motion_mask, median_ksize)

    # 2. 形态学
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, open_ksize)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_ksize)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # 3. 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    # 4. 全局背景光流中位数 (用于对比)
    bg_flow = np.median(residual_magnitude)

    valid_mask = np.zeros_like(mask)
    bboxes, centroid_list, area_list = [], [], []

    for i in range(1, num_labels):
        area = stats[i, 4]
        if area < min_area:
            continue

        # 区域内像素
        region_mask = (labels == i)
        region_flow = residual_magnitude[region_mask]

        # 紧致度验证
        hull = cv2.convexHull(
            np.column_stack(np.where(region_mask))[:, ::-1].astype(np.int32)
        )
        if hull is not None and len(hull) > 2:
            hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1e-8)
        else:
            solidity = 1.0
        if solidity < min_solidity:
            continue  # 松散、不规则形状 → 可能是噪声

        # 光流幅度验证: 区域内的平均残差应显著高于背景
        median_region_flow = np.median(region_flow)
        if bg_flow > 0 and median_region_flow / (bg_flow + 1e-8) < min_flow_ratio:
            continue  # 区域内光流与背景无异 → 假阳性

        valid_mask[labels == i] = 255
        bboxes.append((stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3]))
        centroid_list.append((centroids[i][0], centroids[i][1]))
        area_list.append(area)

    return {
        "mask": valid_mask,
        "num_regions": len(bboxes),
        "bboxes": bboxes,
        "centroids": centroid_list,
        "areas": area_list,
    }



def postprocess_mask(motion_mask, median_ksize=5, open_ksize=(3, 3),
                     close_ksize=(7, 7), min_area=50):
    """运动掩膜后处理流水线。

    1. 中值滤波去椒盐噪声
    2. 开运算去孤立噪点
    3. 闭运算填充孔洞
    4. 连通域分析 + 面积过滤

    Args:
        motion_mask: 输入二值掩膜 (0/255)
        median_ksize: 中值滤波核大小
        open_ksize: 开运算核大小
        close_ksize: 闭运算核大小
        min_area: 最小连通域面积 (px²)

    Returns:
        dict: {
            'mask': 后处理后的掩膜,
            'num_regions': 有效区域数,
            'bboxes': 边界框列表 [(x,y,w,h), ...],
            'centroids': 质心列表 [(cx,cy), ...],
            'areas': 面积列表,
        }
    """
    if motion_mask.max() <= 1:
        motion_mask = (motion_mask * 255).astype(np.uint8)

    # 中值滤波
    mask = cv2.medianBlur(motion_mask, median_ksize)

    # 形态学操作
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, open_ksize)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_ksize)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    # stats[i] = [x, y, w, h, area]

    valid_mask = np.zeros_like(mask)
    bboxes = []
    centroid_list = []
    area_list = []

    for i in range(1, num_labels):
        area = stats[i, 4]  # CC_STAT_AREA
        if area >= min_area:
            valid_mask[labels == i] = 255
            x = stats[i, 0]
            y = stats[i, 1]
            w = stats[i, 2]
            h = stats[i, 3]
            bboxes.append((x, y, w, h))
            centroid_list.append((centroids[i][0], centroids[i][1]))
            area_list.append(area)

    return {
        "mask": valid_mask,
        "num_regions": len(bboxes),
        "bboxes": bboxes,
        "centroids": centroid_list,
        "areas": area_list,
    }


def adaptive_postprocess_fisheye(motion_mask, cx, cy, median_ksize=5,
                                  open_ksize=(3, 3), close_ksize=(7, 7),
                                  min_area_center=50, min_area_edge=30,
                                  r_max=None):
    """鱼眼域自适应后处理 — 边缘区域使用更宽松的面积阈值。

    Args:
        motion_mask: 输入二值掩膜
        cx, cy: 主点坐标
        min_area_center: 中心区域最小面积
        min_area_edge: 边缘区域最小面积
        r_max: 最大半径

    Returns:
        同 postprocess_mask
    """
    if r_max is None:
        h, w = motion_mask.shape
        r_max = np.sqrt(max(cx, w - cx) ** 2 + max(cy, h - cy) ** 2)

    if motion_mask.max() <= 1:
        motion_mask = (motion_mask * 255).astype(np.uint8)

    # 滤波与形态学 (全局)
    mask = cv2.medianBlur(motion_mask, median_ksize)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, open_ksize)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_ksize)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    # 计算每个连通域中心的径向距离
    h, w = mask.shape
    y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    valid_mask = np.zeros_like(mask)
    bboxes, centroid_list, area_list = [], [], []

    for i in range(1, num_labels):
        area = stats[i, 4]  # CC_STAT_AREA
        cx_i, cy_i = centroids[i]

        # 判断区域
        r_i = np.sqrt((cx_i - cx) ** 2 + (cy_i - cy) ** 2)
        if r_i < 0.3 * r_max:
            min_a = min_area_center
        elif r_i < 0.7 * r_max:
            min_a = (min_area_center + min_area_edge) / 2.0
        else:
            min_a = min_area_edge

        if area >= min_a:
            valid_mask[labels == i] = 255
            x = stats[i, 0]
            y = stats[i, 1]
            w_i = stats[i, 2]
            h_i = stats[i, 3]
            bboxes.append((x, y, w_i, h_i))
            centroid_list.append((cx_i, cy_i))
            area_list.append(area)

    return {
        "mask": valid_mask,
        "num_regions": len(bboxes),
        "bboxes": bboxes,
        "centroids": centroid_list,
        "areas": area_list,
    }
