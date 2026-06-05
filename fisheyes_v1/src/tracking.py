"""
目标跟踪模块 — Kalman Filter + 匈牙利匹配 (简化 SORT), 稀疏光流点跟踪。
"""

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


# ============================================================
# Kalman Filter + 匈牙利匹配 (简化 SORT)
# ============================================================

class KalmanBoxTracker:
    """单目标 Kalman 跟踪器。

    状态: [cx, cy, area, aspect_ratio, vx, vy, v_area] (7维)
    观测: [cx, cy, area, aspect_ratio] (4维)
    """
    count = 0

    def __init__(self, bbox):
        """初始化跟踪器。

        Args:
            bbox: (x, y, w, h) 边界框
        """
        self.kf = cv2.KalmanFilter(7, 4, 0)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(7, dtype=np.float32)

        x, y, w, h = bbox
        cx, cy = x + w / 2.0, y + h / 2.0
        area = w * h
        aspect = w / float(h + 1e-8)
        self.kf.statePost = np.array([cx, cy, area, aspect, 0, 0, 0], dtype=np.float32)

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 1
        self.misses = 0
        self.age = 1
        self.history = [(cx, cy)]

    def predict(self):
        """预测下一帧状态。"""
        pred = self.kf.predict()
        cx, cy = pred[0], pred[1]
        self.age += 1
        return np.array([cx, cy])

    def update(self, bbox):
        """用观测更新状态。"""
        x, y, w, h = bbox
        cx, cy = x + w / 2.0, y + h / 2.0
        area = w * h
        aspect = w / float(h + 1e-8)
        self.kf.correct(np.array([cx, cy, area, aspect], dtype=np.float32))
        self.hits += 1
        self.misses = 0
        self.history.append((cx, cy))

    def get_state(self):
        """获取当前估计状态。"""
        s = self.kf.statePost
        area = max(float(s[2]), 1.0)
        w = max(int(np.sqrt(area)), 1)
        h = max(int(area / w), 1)
        return int(s[0]), int(s[1]), w, h


def iou(bbox1, bbox2):
    """计算两个边界框的 IoU。"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    area1, area2 = w1 * h1, w2 * h2
    union = area1 + area2 - inter
    return inter / (union + 1e-8)


def hungarian_match(detections, trackers, iou_threshold=0.3):
    """匈牙利算法匹配检测与跟踪器。

    Args:
        detections: 检测框列表 [(x,y,w,h), ...]
        trackers: KalmanBoxTracker 列表

    Returns:
        matched: [(det_idx, trk_idx), ...]
        unmatched_det: [det_idx, ...]
        unmatched_trk: [trk_idx, ...]
    """
    if len(detections) == 0:
        return [], [], list(range(len(trackers)))
    if len(trackers) == 0:
        return [], list(range(len(detections))), []

    cost = np.zeros((len(detections), len(trackers)))
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            trk_bbox = trk.get_state()
            cost[d, t] = 1.0 - iou(det, trk_bbox)

    row_ind, col_ind = linear_sum_assignment(cost)

    matched, unmatched_det, unmatched_trk = [], [], []
    for d in range(len(detections)):
        if d not in row_ind:
            unmatched_det.append(d)
    for t in range(len(trackers)):
        if t not in col_ind:
            unmatched_trk.append(t)
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < (1.0 - iou_threshold):
            matched.append((r, c))
        else:
            unmatched_det.append(r)
            unmatched_trk.append(c)

    return matched, unmatched_det, unmatched_trk


class SimpleSORT:
    """简化 SORT 多目标跟踪器。"""

    def __init__(self, iou_threshold=0.3, max_missing=3, min_hits=2):
        self.trackers = []
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self.min_hits = min_hits
        self.frame_count = 0

    def update(self, detections):
        """用新检测更新跟踪器。

        Args:
            detections: [(x,y,w,h), ...]

        Returns:
            active_tracks: [(trk_id, (cx,cy,w,h)), ...]
        """
        self.frame_count += 1

        # 获取所有跟踪器的预测
        for trk in self.trackers:
            trk.predict()

        # 匈牙利匹配
        matched, unmatched_det, unmatched_trk = hungarian_match(
            detections, self.trackers, self.iou_threshold
        )

        # 更新匹配的跟踪器
        for d_idx, t_idx in matched:
            self.trackers[t_idx].update(detections[d_idx])

        # 为新检测创建跟踪器
        for d_idx in unmatched_det:
            trk = KalmanBoxTracker(detections[d_idx])
            self.trackers.append(trk)

        # 更新未匹配跟踪器的丢失计数
        for t_idx in unmatched_trk:
            self.trackers[t_idx].misses += 1

        # 删除丢失的跟踪器
        self.trackers = [t for t in self.trackers if t.misses <= self.max_missing]

        # 返回活跃轨迹 (hits >= min_hits)
        active = []
        for trk in self.trackers:
            if trk.hits >= self.min_hits:
                state = trk.get_state()
                active.append((trk.id, state))
        return active

    def get_trail(self, tracker_id):
        """获取指定跟踪器的历史轨迹。"""
        for trk in self.trackers:
            if trk.id == tracker_id:
                return trk.history
        return []


# ============================================================
# 稀疏光流点跟踪 + 聚类
# ============================================================

def sparse_optical_flow_tracking(prev_gray, curr_gray, prev_pts=None,
                                 feature_params=None, lk_params=None):
    """基于稀疏光流的特征点跟踪。

    是对 motion_extraction.sparse_optical_flow_lk 的封装，
    额外返回运动点聚类结果。

    Returns:
        dict: 含 motion_points, clusters 等
    """
    from motion_extraction import sparse_optical_flow_lk

    result = sparse_optical_flow_lk(prev_gray, curr_gray, prev_pts,
                                    feature_params, lk_params)

    # 对运动点做 DBSCAN 聚类 (可选)
    motion_pts = result.get("motion_points", np.array([]))
    clusters = []
    if len(motion_pts) > 3:
        # 使用 XMeans 替代 — 简化为 DBSCAN
        from sklearn.cluster import DBSCAN
        pts = motion_pts.reshape(-1, 2)
        clustering = DBSCAN(eps=30, min_samples=3).fit(pts)
        labels = clustering.labels_
        for lbl in set(labels):
            if lbl >= 0:
                cluster_pts = pts[labels == lbl]
                clusters.append({
                    "points": cluster_pts,
                    "center": cluster_pts.mean(axis=0),
                    "bbox": (
                        int(cluster_pts[:, 0].min()),
                        int(cluster_pts[:, 1].min()),
                        int(cluster_pts[:, 0].max() - cluster_pts[:, 0].min()),
                        int(cluster_pts[:, 1].max() - cluster_pts[:, 1].min()),
                    ),
                })

    result["clusters"] = clusters
    return result
