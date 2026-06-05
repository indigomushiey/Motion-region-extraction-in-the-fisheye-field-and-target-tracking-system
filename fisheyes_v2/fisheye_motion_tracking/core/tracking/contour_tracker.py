import numpy as np

from core.tracking.kalman_filter import KalmanTracker
from core.tracking.tracker_utils import TrackManager, match_detections_to_tracks


class ContourTracker:
    """Short-term target tracker integrating Kalman filter + contour detection + optical flow.

    Pipeline per frame:
        1. Predict all existing Kalman tracks to current frame
        2. Match predicted positions to new detections (IoU)
        3. Update matched tracks, create new tracks for unmatched detections
        4. Predict unmatched tracks (handle short occlusions)
        5. Clean up stale tracks
    """

    def __init__(
        self,
        dt=1.0,
        process_noise=1e-2,
        measurement_noise=1e-1,
        iou_threshold=0.3,
        max_missed=5,
        min_hits=2,
    ):
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.iou_threshold = iou_threshold

        self.manager = TrackManager(max_missed=max_missed, min_hits=min_hits)
        self.kalman_filters = {}  # track_id -> KalmanTracker

    def update(self, regions, flow_results=None):
        """Update tracker with new detections and optional optical flow.

        Args:
            regions: list of dicts from detect_motion_regions()
                     each with 'bbox', 'center', 'area'
            flow_results: optional list from compute_flow_for_regions()
                          used to refine velocity estimates

        Returns:
            tracked: list of active track dicts after this frame
        """
        detections = self._prepare_detections(regions, flow_results)

        # predict all existing tracks
        for tid, kf in self.kalman_filters.items():
            px, py = kf.predict()
            if tid in self.manager.tracks:
                self.manager.tracks[tid]["predicted_center"] = (px, py)
                # update track bbox to predicted position (shift bbox to new center)
                old_bbox = self.manager.tracks[tid]["bbox"]
                self.manager.tracks[tid]["bbox"] = (
                    int(px - old_bbox[2] / 2),
                    int(py - old_bbox[3] / 2),
                    old_bbox[2],
                    old_bbox[3],
                )

        # match detections to tracks
        active_tracks = self.manager.get_track_list()
        matches, unmatched_dets, unmatched_trks = match_detections_to_tracks(
            detections, active_tracks, iou_threshold=self.iou_threshold
        )

        # update matched tracks
        for det_idx, trk_idx in matches:
            det = detections[det_idx]
            trk = active_tracks[trk_idx]
            tid = trk["id"]

            if flow_results is not None and det_idx < len(flow_results):
                flow = flow_results[det_idx]
            else:
                flow = None

            self._update_matched_track(tid, det, flow)

        # create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            self._create_track_from_detection(det)

        # handle unmatched tracks (predict only)
        for trk_idx in unmatched_trks:
            trk = active_tracks[trk_idx]
            tid = trk["id"]
            self.manager.mark_missed(tid)
            if tid in self.kalman_filters:
                self.kalman_filters[tid].update_missing()

        # delete tracks that are gone
        for tid in list(self.manager.tracks.keys()):
            if self.manager.tracks[tid]["status"] == "deleted":
                self.manager.delete_track(tid)
                self.kalman_filters.pop(tid, None)

        return self.manager.active_tracks()

    def _prepare_detections(self, regions, flow_results):
        """Convert regions to detection dicts, optionally refined by flow."""
        detections = []
        for i, region in enumerate(regions):
            center = region["center"]

            # refine center with mean flow displacement if available
            if flow_results is not None:
                for fr in flow_results:
                    if fr["region_idx"] == i:
                        disp = fr["displacements"]["mean_vec"]
                        center = (center[0] + disp[0], center[1] + disp[1])
                        break

            detections.append({
                "bbox": region["bbox"],
                "center": center,
                "area": region["area"],
                "region_idx": i,
            })
        return detections

    def _update_matched_track(self, tid, det, flow):
        """Update Kalman filter and track manager for a matched detection."""
        cx, cy = det["center"]

        # update kalman
        if tid in self.kalman_filters:
            self.kalman_filters[tid].update(cx, cy)
        else:
            self.kalman_filters[tid] = KalmanTracker(
                cx, cy, dt=self.dt,
                process_noise=self.process_noise,
                measurement_noise=self.measurement_noise,
            )

        self.manager.update_track(
            tid, det["bbox"], (cx, cy), det["area"],
            feature_pts=flow.get("prev_pts") if flow else None,
        )

    def _create_track_from_detection(self, det):
        """Create a new Kalman tracker and track entry."""
        cx, cy = det["center"]

        kf = KalmanTracker(
            cx, cy, dt=self.dt,
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise,
        )

        tid = self.manager.create_track(
            det["bbox"], (cx, cy), det["area"]
        )
        if tid is not None:
            self.kalman_filters[tid] = kf

    def get_trajectories(self):
        """Return trajectory history for all tracks."""
        result = {}
        for tid, t in self.manager.tracks.items():
            if t["status"] in ("active", "lost"):
                result[tid] = {
                    "id": tid,
                    "history": t["history"],
                    "status": t["status"],
                    "age": t["age"],
                }
        return result
