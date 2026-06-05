import numpy as np


def bbox_iou(a, b):
    """Compute IoU between two bounding boxes (x, y, w, h)."""
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = a[2] * a[3]
    area_b = b[2] * b[3]
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


def bbox_center_distance(a, b):
    """Euclidean distance between bbox centers."""
    cx_a = a[0] + a[2] / 2
    cy_a = a[1] + a[3] / 2
    cx_b = b[0] + b[2] / 2
    cy_b = b[1] + b[3] / 2
    return np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)


def match_detections_to_tracks(detections, tracks, iou_threshold=0.3):
    """Match new detections to existing tracks via IoU-based Hungarian assignment.

    Args:
        detections: list of detection dicts, each with 'bbox' key
        tracks: list of track dicts, each with 'bbox' key (predicted position)
        iou_threshold: minimum IoU for a valid match

    Returns:
        matches: list of (det_idx, track_idx) pairs
        unmatched_dets: set of detection indices
        unmatched_tracks: set of track indices
    """
    if len(detections) == 0 or len(tracks) == 0:
        return [], set(range(len(detections))), set(range(len(tracks)))

    n_det = len(detections)
    n_trk = len(tracks)

    cost = np.zeros((n_det, n_trk), dtype=np.float32)
    for i, det in enumerate(detections):
        for j, trk in enumerate(tracks):
            cost[i, j] = 1.0 - bbox_iou(det["bbox"], trk["bbox"])

    # greedy assignment (simple, no scipy dependency for Hungarian)
    det_indices = list(range(n_det))
    trk_indices = list(range(n_trk))

    # sort by cost to approximate optimal matching
    pairs = [(cost[i, j], i, j) for i in det_indices for j in trk_indices]
    pairs.sort()

    matched_dets = set()
    matched_trks = set()
    matches = []

    for _, di, ti in pairs:
        if di in matched_dets or ti in matched_trks:
            continue
        if cost[di, ti] > (1.0 - iou_threshold):
            continue
        matches.append((di, ti))
        matched_dets.add(di)
        matched_trks.add(ti)

    unmatched_dets = set(det_indices) - matched_dets
    unmatched_tracks = set(trk_indices) - matched_trks

    return matches, unmatched_dets, unmatched_tracks


class TrackManager:
    """Manages track lifecycle: creation, updating, and deletion."""

    def __init__(self, max_missed=5, min_hits=2, max_tracks=50):
        self.next_id = 0
        self.max_missed = max_missed
        self.min_hits = min_hits
        self.max_tracks = max_tracks
        self.tracks = {}  # track_id -> track dict

    def create_track(self, bbox, center, area, feature_pts=None):
        """Create a new track from a detection.

        Track dict:
            id, bbox, center, area,
            age, hits, missed,
            history (list of positions),
            status: 'tentative' | 'active' | 'lost' | 'deleted'
            feature_pts: tracked feature points (optional)
        """
        if len(self.tracks) >= self.max_tracks:
            return None

        tid = self.next_id
        self.next_id += 1

        self.tracks[tid] = {
            "id": tid,
            "bbox": bbox,
            "center": center,
            "area": area,
            "age": 0,
            "hits": 1,
            "missed": 0,
            "history": [center],
            "status": "tentative",
            "feature_pts": feature_pts,
        }
        return tid

    def update_track(self, tid, bbox, center, area, feature_pts=None):
        """Update an existing track with a new detection."""
        t = self.tracks[tid]
        t["bbox"] = bbox
        t["center"] = center
        t["area"] = area
        t["age"] += 1
        t["hits"] += 1
        t["missed"] = 0
        t["history"].append(center)
        if feature_pts is not None:
            t["feature_pts"] = feature_pts
        if t["hits"] >= self.min_hits:
            t["status"] = "active"
        if len(t["history"]) > 30:
            t["history"].pop(0)

    def mark_missed(self, tid):
        """Mark a track as missed in this frame."""
        t = self.tracks[tid]
        t["age"] += 1
        t["missed"] += 1
        if t["missed"] > self.max_missed:
            t["status"] = "deleted"
        elif t["missed"] > 1:
            t["status"] = "lost"

    def delete_track(self, tid):
        self.tracks.pop(tid, None)

    def cleanup(self):
        """Remove deleted tracks."""
        deleted = [tid for tid, t in self.tracks.items() if t["status"] == "deleted"]
        for tid in deleted:
            del self.tracks[tid]

    def active_tracks(self):
        return {tid: t for tid, t in self.tracks.items()
                if t["status"] in ("active", "tentative", "lost")}

    def get_track_list(self):
        """Return list of active track dicts for matching."""
        return [t for t in self.tracks.values()
                if t["status"] in ("active", "tentative", "lost")]
