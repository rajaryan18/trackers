# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import supervision as sv

from trackers.core.base import BaseTracker
from trackers.core.deepsort.association import (
    DeepSORTDetection,
    NearestNeighborDistanceMetric,
    gate_cost_matrix,
    matching_cascade,
    min_cost_matching,
)
from trackers.core.deepsort.kalman import DeepSORTKalmanFilter
from trackers.core.deepsort.track import DeepSORTTrack, TrackState


class DeepSORTTracker(BaseTracker):
    """DeepSORT (Simple Online and Realtime Tracking with a Deep Association Metric)
    extends SORT by incorporating appearance information. It uses a deep learning
    model to extract features from bounding boxes, which are then used to match
    detections to tracks. This significantly reduces identity switches and improves
    tracking robustness in the presence of occlusions and complex motion.

    DeepSORT employs a Kalman filter for motion prediction and a matching cascade
    algorithm that prioritizes tracks with more recent updates. It also uses a
    Mahalanobis distance metric to gate the association process, ensuring that
    detections are only matched to tracks that are motion-compatible.

    Args:
        max_age: `int` specifying the maximum number of consecutive misses before
            a track is deleted. Default: 30.
        n_init: `int` specifying the number of consecutive detections before a
            track is confirmed. Default: 3.
        matching_threshold: `float` specifying the matching threshold for
            appearance-based association. Default: 0.2.
        max_iou_distance: `float` specifying the maximum IoU distance allowed for
            matching. Default: 0.7.
        max_cosine_distance: `float` specifying the maximum cosine distance
            allowed for appearance-based matching. Default: 0.2.
        nn_budget: `int` specifying the maximum number of appearance features
            to store per track. Default: 100.
    """

    tracker_id = "deepsort"

    def __init__(
        self,
        lost_track_buffer: int = 70,
        track_activation_threshold: float = 0.2,
        minimum_consecutive_frames: int = 2,
        minimum_iou_threshold: float = 0.3,
        matching_threshold: float = 0.2,
        nn_budget: int = 100,
        max_cosine_distance: float = 0.2,
        high_conf_det_threshold: float = 0.5,
        max_age: int | None = None,
        n_init: int | None = None,
        max_iou_distance: int | None = None,
    ) -> None:
        self.max_age = max_age if max_age is not None else lost_track_buffer
        self.n_init = n_init if n_init is not None else minimum_consecutive_frames
        self.max_iou_distance = (
            max_iou_distance
            if max_iou_distance is not None
            else (1.0 - minimum_iou_threshold)
        )
        self.track_activation_threshold = track_activation_threshold
        self.high_conf_det_threshold = high_conf_det_threshold
        self.max_cosine_distance = max_cosine_distance

        self.kf = DeepSORTKalmanFilter()
        self.tracks: list[DeepSORTTrack] = []
        self._next_id = 1

        self.metric = NearestNeighborDistanceMetric(
            "cosine", self.max_cosine_distance, nn_budget
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Process new detections and assign track IDs.

        Args:
            detections: Current frame detections with xyxy, confidence, class_id.
                The `detections.data` dictionary MUST contain 'features' key with
                appearance embeddings for each detection if Re-ID is used.

        Returns:
            Same detections enriched with tracker_id attribute for each box.
        """
        # Get features
        features = detections.data.get("features", None)
        if features is None:
            features = [np.zeros(128) for _ in range(len(detections))]

        # Split detections by confidence
        confidences = detections.confidence
        high_mask = confidences >= self.high_conf_det_threshold
        
        # indices in the original sv.Detections
        high_indices = np.where(high_mask)[0]
        low_indices = np.where(~high_mask)[0]
        
        # Prepare internal Detection objects for high and low confidence
        high_detections = []
        for i in high_indices:
            xyxy = detections.xyxy[i]
            tlwh = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
            high_detections.append(
                DeepSORTDetection(tlwh, confidences[i], features[i])
            )
            
        low_detections = []
        for i in low_indices:
            xyxy = detections.xyxy[i]
            tlwh = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
            low_detections.append(
                DeepSORTDetection(tlwh, confidences[i], features[i])
            )

        # Run Kalman filter prediction step on all tracks.
        for track in self.tracks:
            track.predict(self.kf)

        # Stage 1: Match high-confidence detections
        matches_high, unmatched_tracks_stage1, unmatched_high_idx = self._match(
            high_detections
        )
        
        # Stage 2: Match remaining confirmed tracks with low-confidence detections
        confirmed_unmatched = [
            i for i in unmatched_tracks_stage1 if self.tracks[i].is_confirmed()
        ]
        
        matches_low, unmatched_confirmed_low, unmatched_low_idx = min_cost_matching(
            self._iou_distance,
            self.max_iou_distance,
            self.tracks,
            low_detections,
            confirmed_unmatched,
            list(range(len(low_detections)))
        )
        
        # Final matches and unmatched
        # matches_low already contains (track_idx_absolute, det_idx_relative_to_low_detections)
        matches_stage2 = matches_low
        
        # Re-calculate unmatched tracks
        matched_track_indices = set(k for k, _ in matches_high) | set(k for k, _ in matches_stage2)
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_track_indices]

        # Update matching tracks
        for track_idx, det_local_idx in matches_high:
            self.tracks[track_idx].update(self.kf, high_detections[det_local_idx])
            
        for track_idx, det_local_idx in matches_stage2:
            self.tracks[track_idx].update(self.kf, low_detections[det_local_idx])

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # Initiate new tracks from unmatched high-confidence detections
        for det_local_idx in unmatched_high_idx:
            orig_idx = high_indices[det_local_idx]
            if confidences[orig_idx] >= self.track_activation_threshold:
                self._initiate_track(high_detections[det_local_idx])

        # Update distance metric with new samples from high-conf matches only
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features_to_fit, targets_to_fit = [], []
        # Only matches with high-confidence detections are used for appearance updating to avoid drift
        for track_idx, det_local_idx in matches_high:
            track = self.tracks[track_idx]
            if track.is_confirmed():
                features_to_fit.append(high_detections[det_local_idx].feature)
                targets_to_fit.append(track.track_id)
        
        if features_to_fit:
            self.metric.partial_fit(
                np.asarray(features_to_fit),
                np.asarray(targets_to_fit),
                active_targets,
            )

        # Build return sv.Detections
        tracker_ids = np.full(len(detections), -1, dtype=int)
        
        # Map high-conf matches
        for track_idx, det_local_idx in matches_high:
            track = self.tracks[track_idx]
            if track.is_confirmed():
                tracker_ids[high_indices[det_local_idx]] = track.track_id
                
        # Map low-conf matches
        for track_idx, det_local_idx in matches_stage2:
            track = self.tracks[track_idx]
            if track.is_confirmed():
                tracker_ids[low_indices[det_local_idx]] = track.track_id

        detections.tracker_id = tracker_ids

        # Prune deleted tracks at the very end to avoid index errors during update
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        return detections

    def _match(self, detections: list[DeepSORTDetection]):
        def gated_metric(tracks, detections, track_indices, det_indices):
            features = np.array([detections[i].feature for i in det_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            
            has_real_features = np.any(features != 0)
            
            if has_real_features:
                cost_matrix = self.metric.distance(features, targets)
                max_dist = self.max_cosine_distance
            else:
                # Use Mahalanobis distance as cost
                cost_matrix = np.zeros((len(track_indices), len(det_indices)))
                measurements = np.asarray([detections[i].to_xyah() for i in det_indices])
                for i, track_idx in enumerate(track_indices):
                    cost_matrix[i, :] = self.kf.gating_distance(
                        tracks[track_idx].mean, tracks[track_idx].covariance, measurements
                    )
                # Gating threshold for 4 degrees of freedom is 9.4877.
                # Increasing to 15.0 to handle camera motion more gracefully.
                max_dist = 15.0
            
            cost_matrix = gate_cost_matrix(
                self.kf,
                cost_matrix,
                tracks,
                detections,
                track_indices,
                det_indices,
            )
            return cost_matrix, max_dist

        # Split tracks into confirmed and unconfirmed.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance/motion cascade.
        det_indices = list(range(len(detections)))
        
        # Custom matching cascade logic to handle dynamic max_dist
        unmatched_detections = det_indices
        matches_a = []
        for age in range(1, self.max_age + 1):
            if len(unmatched_detections) == 0:
                break
            track_indices_l = [k for k in confirmed_tracks if self.tracks[k].time_since_update == age]
            if len(track_indices_l) == 0:
                continue
            
            # Get cost matrix and its corresponding max_dist
            cost_matrix, max_dist = gated_metric(self.tracks, detections, track_indices_l, unmatched_detections)
            
            matches_l, _, unmatched_detections = min_cost_matching(
                lambda _t, _d, _ti, _di: cost_matrix, 
                max_dist, 
                self.tracks, 
                detections, 
                track_indices_l, 
                unmatched_detections
            )
            matches_a += matches_l
        
        unmatched_tracks_a = list(set(confirmed_tracks) - set(k for k, _ in matches_a))

        # Associate remaining tracks together with unconfirmed tracks using IoU.
        # DeepSORT original: only age=1 tracks are matched by IoU.
        # Improvement: allow age <= 3 for IoU fallback to handle short occlusions better.
        has_features = any(np.any(d.feature != 0) for d in detections)
        
        if has_features:
            iou_track_candidates = unconfirmed_tracks + [
                k for k in unmatched_tracks_a if self.tracks[k].time_since_update <= 3
            ]
            unmatched_tracks_a = [
                k for k in unmatched_tracks_a if self.tracks[k].time_since_update > 3
            ]
        else:
            # Motion-only mode: all unmatched tracks are candidates for IoU matching
            iou_track_candidates = unconfirmed_tracks + unmatched_tracks_a
            unmatched_tracks_a = []

        matches_iou, unmatched_tracks_iou, unmatched_detections = min_cost_matching(
            self._iou_distance,
            0.7, # Tighter IoU threshold (standard DeepSORT)
            self.tracks,
            detections,
            iou_track_candidates,
            unmatched_detections,
        )

        matches = matches_a + matches_iou
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_iou))
        return matches, unmatched_tracks, unmatched_detections

    def _iou_distance(self, tracks, detections, track_indices, det_indices):
        """Compute IoU distance between tracks and detections."""
        if len(track_indices) == 0 or len(det_indices) == 0:
            return np.empty((len(track_indices), len(det_indices)))

        track_boxes = np.array([tracks[i].to_tlbr() for i in track_indices])
        det_boxes = np.array([detections[i].to_tlbr() for i in det_indices])
        
        # Calculate IoU
        iou_matrix = sv.box_iou_batch(track_boxes, det_boxes)
        return 1.0 - iou_matrix

    def _initiate_track(self, detection: DeepSORTDetection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(
            DeepSORTTrack(
                mean,
                covariance,
                self._next_id,
                self.n_init,
                self.max_age,
                detection.feature,
            )
        )
        self._next_id += 1

    def reset(self) -> None:
        """Clear all internal tracking state."""
        self.tracks = []
        self._next_id = 1
        self.metric.samples = {}
