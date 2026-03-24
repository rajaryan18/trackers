# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import numpy as np
from scipy.optimize import linear_sum_assignment


class DeepSORTDetection:
    """
    This class represents a single target detection of a (maybe multi-class)
    object detector.

    Parameters
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : float
        Detector confidence score.
    feature : ndarray
        A feature vector that describes the object appearance in this detection.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : float
        Detector confidence score.
    feature : ndarray
        A feature vector that describes the object appearance in this detection.
    """

    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        # Avoid division by zero
        if ret[3] > 0:
            ret[2] /= ret[3]
        else:
            ret[2] = 0
        return ret


class NearestNeighborDistanceMetric:
    """
    A nearest neighbor distance metric that, for each target, returns the
    closest distance to any sample that has been observed so far.

    Parameters
    ----------
    metric : str
        Either "cosine" or "euclidean".
    matching_threshold: float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
    budget : int | None
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.

    Attributes
    ----------
    samples : dict
        A dictionary that maps from target_id to a list of samples that have
        been observed so far.
    """

    def __init__(self, metric, matching_threshold, budget=None):
        if metric == "cosine":
            self._metric = self._cosine_distance
        elif metric == "euclidean":
            self._metric = self._nn_euclidean_distance
        else:
            raise ValueError("Invalid metric; must be either 'cosine' or 'euclidean'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        """Update the distance metric with new data.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : ndarray
            An integer array of associated target identities.
        active_targets : list
            A list of targets that are currently present in the scene.
        """
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget :]
        self.samples = {k: self.samples[k] for k in active_targets if k in self.samples}

    def distance(self, features, targets):
        """Compute distance between features and targets.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : list
            A list of targets to match the given features against.

        Returns
        -------
        ndarray
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.
        """
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            if target not in self.samples or len(self.samples[target]) == 0:
                # If no samples available for this target, assign max distance
                cost_matrix[i, :] = self.matching_threshold + 1e-5
            else:
                cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix

    def _nn_euclidean_distance(self, x, y):
        """Helper function for nearest neighbor distance (Euclidean)."""
        distances = np.linalg.norm(np.asarray(x)[:, np.newaxis] - np.asarray(y), axis=2)
        return np.min(distances, axis=0)

    def _cosine_distance(self, x, y):
        """Helper function for nearest neighbor distance (cosine)."""
        x = np.asarray(x)
        y = np.asarray(y)
        # Normalize features
        x_norm = np.linalg.norm(x, axis=1, keepdims=True)
        y_norm = np.linalg.norm(y, axis=1, keepdims=True)
        
        # Avoid division by zero for zero features
        x = np.divide(x, x_norm, out=np.zeros_like(x), where=x_norm > 0)
        y = np.divide(y, y_norm, out=np.zeros_like(y), where=y_norm > 0)
        
        # Similarity = dot product
        similarities = np.dot(x, y.T)
        if similarities.size == 0:
            return np.array([])
        # Distance = 1 - similarity
        return np.min(1.0 - similarities, axis=0)


def min_cost_matching(
    distance_metric, max_distance, tracks, detections, track_indices=None, det_indices=None
):
    """Solve linear assignment problem.

    Parameters
    ----------
    distance_metric : Callable[[List[Track], List[Detection], List[int], List[int]], ndarray]
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : list[Track]
        A list of active tracks.
    detections : list[Detection]
        A list of detections at the current time step.
    track_indices : list[int]
        List of track indices that should be matched. Defaults to all `tracks`.
    det_indices : list[int]
        List of detection indices that should be matched. Defaults to all
        `detections`.

    Returns
    -------
    (list[(int, int)], list[int], list[int])
        Returns a tuple with the following three entries:
        * A list of matched (track_index, detection_index) pairs.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if det_indices is None:
        det_indices = np.arange(len(detections))

    if len(det_indices) == 0 or len(track_indices) == 0:
        return [], list(track_indices), list(det_indices)

    cost_matrix = distance_metric(tracks, detections, track_indices, det_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    
    # Handle NaN or Inf values by replacing them with a very large cost
    cost_matrix[~np.isfinite(cost_matrix)] = max_distance + 1e5

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, row in enumerate(row_indices):
        track_idx = track_indices[row]
        det_idx = det_indices[col_indices[col]]
        if cost_matrix[row, col_indices[col]] <= max_distance:
            matches.append((track_idx, det_idx))
        else:
            unmatched_tracks.append(track_idx)

    for i, track_idx in enumerate(track_indices):
        if i not in row_indices:
            unmatched_tracks.append(track_idx)

    for i, det_idx in enumerate(det_indices):
        if i not in col_indices:
            unmatched_detections.append(det_idx)

    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(
    distance_metric, max_distance, max_age, tracks, detections, track_indices=None, det_indices=None
):
    """Run matching cascade.

    Parameters
    ----------
    distance_metric : Callable
        The distance metric.
    max_distance : float
        Gating threshold.
    max_age : int
        The maximum track age.
    tracks : list[Track]
        A list of active tracks.
    detections : list[Detection]
        A list of detections at the current time step.
    track_indices : list[int]
        List of track indices that should be matched. Defaults to all `tracks`.
    det_indices : list[int]
        List of detection indices that should be matched. Defaults to all
        `detections`.

    Returns
    -------
    (list[(int, int)], list[int], list[int])
        Returns a tuple with the following three entries:
        * A list of matched (track_index, detection_index) pairs.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.
    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if det_indices is None:
        det_indices = list(range(len(detections)))

    unmatched_detections = det_indices
    matches = []
    for age in range(1, max_age + 1):
        if len(unmatched_detections) == 0:
            break

        track_indices_l = [
            k for k in track_indices if tracks[k].time_since_update == age
        ]
        if len(track_indices_l) == 0:
            continue

        matches_l, _, unmatched_detections = min_cost_matching(
            distance_metric, max_distance, tracks, detections, track_indices_l, unmatched_detections
        )
        matches += matches_l

    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(
    kf, cost_matrix, tracks, detections, track_indices, det_indices, gated_cost=1e5, only_position=False, gating_threshold=9.4877
):
    """Invalidate entries in the cost matrix based on the Mahalanobis distance.

    Parameters
    ----------
    kf : kalman_filter.KalmanFilter
        The Kalman filter.
    cost_matrix : ndarray
        An NxM dimensional cost matrix, where N is the number of tracks and M
        is the number of detections.
    tracks : list[Track]
        A list of active tracks.
    detections : list[Detection]
        A list of detections at the current time step.
    track_indices : list[int]
        The track indices that correspond to the rows in the cost matrix.
    det_indices : list[int]
        The detection indices that correspond to the columns in the cost matrix.
    gated_cost : float
        The cost assigned to invalid matches.
    only_position : bool
        If True, only the center position is used to compute the Mahalanobis
        distance.
    """
    gating_threshold = 9.4877  # Chi-square 0.95 quantile for 4 degrees of freedom
    measurements = np.asarray([detections[i].to_xyah() for i in det_indices])
    for i, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position
        )
        cost_matrix[i, gating_distance > gating_threshold] = gated_cost
    return cost_matrix
