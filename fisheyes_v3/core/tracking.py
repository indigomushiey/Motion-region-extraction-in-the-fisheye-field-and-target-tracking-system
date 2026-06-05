"""Contour-based multi-object tracking via IoU matching across consecutive frames.

Assigns persistent IDs to motion components and draws labeled bounding boxes.
"""

import cv2
import numpy as np


class ContourTracker:
    """Simple IoU-based contour tracker for motion mask components."""

    def __init__(self, min_area=200, iou_thresh=0.3, max_lost=3):
        self.min_area = min_area
        self.iou_thresh = iou_thresh
        self.max_lost = max_lost
        self.tracks = []  # list of {"id": int, "bbox": (x,y,w,h), "lost": int, "color": tuple}

    def _next_id(self):
        ids = [t["id"] for t in self.tracks]
        return max(ids) + 1 if ids else 1

    def _bbox_iou(self, a, b):
        """Intersection over Union for two (x, y, w, h) boxes."""
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix = max(ax, bx); iy = max(ay, by)
        iw = min(ax + aw, bx + bw) - ix
        ih = min(ay + ah, by + bh) - iy
        if iw <= 0 or ih <= 0:
            return 0.0
        inter = iw * ih
        area_a = aw * ah; area_b = bw * bh
        return inter / (area_a + area_b - inter + 1e-6)

    def update(self, mask, frame_idx=None):
        """Extract contours, match to existing tracks, return drawable annotations.

        Args:
            mask: binary motion mask (0/255 uint8)
            frame_idx: optional frame number for display

        Returns:
            list of {"id": int, "bbox": (x,y,w,h), "center": (cx,cy), "color": (b,g,r)}
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract bounding boxes for valid contours
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append((x, y, w, h))

        # Mark all existing tracks as unmatched
        matched = [False] * len(detections)
        for t in self.tracks:
            t["_matched"] = False

        # IoU matching (greedy)
        for ti, t in enumerate(self.tracks):
            best_iou = self.iou_thresh
            best_di = -1
            for di, d in enumerate(detections):
                if matched[di]:
                    continue
                iou = self._bbox_iou(t["bbox"], d)
                if iou > best_iou:
                    best_iou = iou
                    best_di = di
            if best_di >= 0:
                t["bbox"] = detections[best_di]
                t["lost"] = 0
                t["_matched"] = True
                matched[best_di] = True

        # Create new tracks for unmatched detections
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
                  (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)]
        for di, d in enumerate(detections):
            if not matched[di]:
                tid = self._next_id()
                self.tracks.append({
                    "id": tid,
                    "bbox": d,
                    "lost": 0,
                    "_matched": True,
                    "color": colors[(tid - 1) % len(colors)],
                })

        # Age lost tracks
        for t in self.tracks:
            if not t.get("_matched", False):
                t["lost"] += 1

        # Remove long-lost tracks
        self.tracks = [t for t in self.tracks if t["lost"] <= self.max_lost]

        # Build result list
        result = []
        for t in self.tracks:
            if t["lost"] == 0:  # only active tracks
                x, y, w, h = t["bbox"]
                result.append({
                    "id": t["id"],
                    "bbox": (x, y, w, h),
                    "center": (x + w // 2, y + h // 2),
                    "color": t["color"],
                })
        return result


def draw_tracks(image, tracks, thickness=2):
    """Draw tracked objects with bounding boxes and IDs on an image."""
    out = image.copy()
    for t in tracks:
        x, y, w, h = t["bbox"]
        color = tuple(int(c) for c in t["color"])
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        label = f"ID:{t['id']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x, y - th - 4), (x + tw + 4, y), color, -1)
        cv2.putText(out, label, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1)
    return out
