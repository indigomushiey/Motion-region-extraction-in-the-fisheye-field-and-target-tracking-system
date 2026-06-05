"""
评估模块 — IoU / Precision / Recall / F1-Score, 分区域评估。
"""

import numpy as np
import cv2


def compute_metrics(pred_mask, gt_mask):
    """计算运动区域提取的逐像素评估指标。

    Args:
        pred_mask: 预测二值掩膜 (0/255 或 0/1)
        gt_mask: 真值二值掩膜 (0/255 或 0/1)

    Returns:
        dict: {IoU, Precision, Recall, F1, TP, FP, FN, TN}
    """
    if pred_mask.max() > 1:
        pred_bin = (pred_mask > 127).astype(np.uint8)
    else:
        pred_bin = (pred_mask > 0).astype(np.uint8)

    if gt_mask.max() > 1:
        gt_bin = (gt_mask > 127).astype(np.uint8)
    else:
        gt_bin = (gt_mask > 0).astype(np.uint8)

    TP = np.sum((pred_bin == 1) & (gt_bin == 1))
    FP = np.sum((pred_bin == 1) & (gt_bin == 0))
    FN = np.sum((pred_bin == 0) & (gt_bin == 1))
    TN = np.sum((pred_bin == 0) & (gt_bin == 0))

    eps = 1e-8
    iou = TP / (TP + FP + FN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TP": int(TP),
        "FP": int(FP),
        "FN": int(FN),
        "TN": int(TN),
    }


def compute_region_metrics(pred_mask, gt_mask, cx, cy, ratios=None, r_max=None):
    """分区域 (中心/过渡/边缘) 计算评估指标。

    Args:
        pred_mask, gt_mask: 预测和真值掩膜
        cx, cy: 主点坐标
        ratios: [center_ratio, transition_ratio], 默认 [0.3, 0.7]
        r_max: 最大半径, None 则自动计算

    Returns:
        dict: {
            'center': {...}, 'transition': {...}, 'edge': {...}, 'overall': {...}
        }
    """
    if ratios is None:
        ratios = [0.3, 0.7]

    h, w = pred_mask.shape[:2]
    if r_max is None:
        r_max = np.sqrt(max(cx, w - cx) ** 2 + max(cy, h - cy) ** 2)

    y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    r = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)

    regions = {
        "center": r < ratios[0] * r_max,
        "transition": (r >= ratios[0] * r_max) & (r < ratios[1] * r_max),
        "edge": r >= ratios[1] * r_max,
    }

    result = {}
    for name, region_mask in regions.items():
        if pred_mask.max() > 1:
            pred_region = pred_mask.copy()
            pred_region[~region_mask] = 0
        else:
            pred_region = pred_mask.copy()
            pred_region[~region_mask] = 0

        if gt_mask.max() > 1:
            gt_region = gt_mask.copy()
            gt_region[~region_mask] = 0
        else:
            gt_region = gt_mask.copy()
            gt_region[~region_mask] = 0

        result[name] = compute_metrics(pred_region, gt_region)

    result["overall"] = compute_metrics(pred_mask, gt_mask)
    return result


def compute_instance_metrics(pred_bboxes, gt_mask, iou_threshold=0.5):
    """实例级评估 — 基于预测边界框与 GT 连通域的匹配。

    Args:
        pred_bboxes: [(x,y,w,h), ...]
        gt_mask: 真值掩膜 (多类别或二值)
        iou_threshold: 匹配 IoU 阈值

    Returns:
        dict: {TP, FP, FN, Precision, Recall, F1}
    """
    # 提取 GT 连通域
    if gt_mask.max() <= 1:
        gt_mask = (gt_mask * 255).astype(np.uint8)
    num_gt, gt_labels, gt_stats, _ = cv2.connectedComponentsWithStats(gt_mask)
    gt_bboxes = []
    for i in range(1, num_gt):
        x = gt_stats[i, 0]
        y = gt_stats[i, 1]
        w = gt_stats[i, 2]
        h = gt_stats[i, 3]
        gt_bboxes.append((x, y, w, h))

    if len(pred_bboxes) == 0:
        return {"TP": 0, "FP": 0, "FN": len(gt_bboxes),
                "Precision": 0.0, "Recall": 0.0, "F1": 0.0}
    if len(gt_bboxes) == 0:
        return {"TP": 0, "FP": len(pred_bboxes), "FN": 0,
                "Precision": 0.0, "Recall": 0.0, "F1": 0.0}

    # 计算 IoU 矩阵
    iou_mat = np.zeros((len(pred_bboxes), len(gt_bboxes)))
    for i, pb in enumerate(pred_bboxes):
        for j, gb in enumerate(gt_bboxes):
            iou_mat[i, j] = _box_iou(pb, gb)

    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(-iou_mat)

    tp = 0
    matched_gt = set()
    for r, c in zip(row_ind, col_ind):
        if iou_mat[r, c] >= iou_threshold:
            tp += 1
            matched_gt.add(c)

    fp = len(pred_bboxes) - tp
    fn = len(gt_bboxes) - len(matched_gt)

    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {"TP": tp, "FP": fp, "FN": fn,
            "Precision": precision, "Recall": recall, "F1": f1}


def _box_iou(b1, b2):
    """两个边界框的 IoU。"""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1, area2 = w1 * h1, w2 * h2
    union = area1 + area2 - inter
    return inter / (union + 1e-8)
