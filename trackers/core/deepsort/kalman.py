# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import numpy as np
import scipy.linalg


class DeepSORTKalmanFilter:
    """
    A Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space
        x, y, a, h, vx, vy, va, vh
    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion is followed by a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).
    """

    def __init__(self):
        ndim, dt = 4, 1.0

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in [0, 1, 3]:  # x, y, h have velocity
            self._motion_mat[i, i + ndim] = dt
        # Aspect ratio (index 2) does not have a velocity component in our model
        # which keeps detections more stable in MOT sequences.
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a heuristic that has been proven to work well
        # for bounding box tracking.
        # Heuristic weights for noise relative to bounding box height.
        self._std_weight_position = 1.0 / 40.0 
        self._std_weight_velocity = 1.0 / 140.0
        # Multiplier to increase noise when moving fast (Camera Motion compensation lite)
        self._velocity_noise_factor = 0.1

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Args:
            measurement (ndarray): Bounding box coordinates (x, y, a, h).

        Returns:
            (ndarray, ndarray): Returns the mean vector (8 dimensional) and
                covariance matrix (8x8 dimensional) of the new track.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2, # Aspect ratio
            2 * self._std_weight_position * measurement[3], # Height
            10 * self._std_weight_velocity * measurement[3], # VX
            10 * self._std_weight_velocity * measurement[3], # VY
            1e-10, # VA
            10 * self._std_weight_velocity * measurement[3], # VH
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Args:
            mean (ndarray): The 8 dimensional mean vector of the object state
                at the previous time step.
            covariance (ndarray): The 8x8 dimensional covariance matrix of the
                object state at the previous time step.

        Returns:
            (ndarray, ndarray): Returns the mean vector and covariance matrix
                of the predicted state.
        """
        # Scale process noise by velocity to handle camera motion and acceleration
        # If the object/camera is moving fast, we increase the uncertainty.
        vel_mag = np.linalg.norm(mean[4:6])
        adaptive_factor = 1.0 + self._velocity_noise_factor * vel_mag
        
        std_pos = [
            self._std_weight_position * mean[3] * adaptive_factor,
            self._std_weight_position * mean[3] * adaptive_factor,
            1e-4, # Aspect ratio noise
            self._std_weight_position * mean[3] * adaptive_factor,
        ]
        std_vel = [
            self._std_weight_velocity * mean[3] * adaptive_factor,
            self._std_weight_velocity * mean[3] * adaptive_factor,
            1e-5,
            self._std_weight_velocity * mean[3] * adaptive_factor,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = (
            np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T))
            + motion_cov
        )

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Args:
            mean (ndarray): The state's mean vector (8 dimensional).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).

        Returns:
            (ndarray, ndarray): Returns the projected mean and covariance matrix
                of the given state estimate.
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1, # Aspect ratio measurement noise
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T)
        )
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Args:
            mean (ndarray): The predicted state's mean vector (8 dimensional).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).
            measurement (ndarray): The 4 dimensional measurement vector
                (x, y, a, h), where (x, y) is the center position, a the aspect
                ratio, and h the height of the bounding box.

        Returns:
            (ndarray, ndarray): Returns the measurement-corrected state
                distribution.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )
        return new_mean, new_covariance

    def gating_distance(
        self, mean, covariance, measurements, only_position=False, metric="mahalanobis"
    ):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from chi-square distribution
        with 4 degrees of freedom for the full state or 2 degrees of freedom for
        position only.

        Args:
            mean (ndarray): Mean vector of the state distribution (8 dimensional).
            covariance (ndarray): Covariance matrix of the state distribution
                (8x8 dimensional).
            measurements (ndarray): An Nx4 dimensional matrix of N measurements,
                each in format (x, y, a, h).
            only_position (bool): If True, distance is only computed over
                position center (x, y).
            metric (str): The distance metric to use. Currently only
                'mahalanobis' is supported.

        Returns:
            ndarray: Returns an array of length N, where the i-th element contains
                the squared Mahalanobis distance between (mean, covariance) and
                `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == "mahalanobis":
            cholesky_factor = scipy.linalg.cho_factor(covariance, lower=True)
            z = scipy.linalg.cho_solve(cholesky_factor, d.T)
            squared_maha = np.sum(d * z.T, axis=1)
            return squared_maha
        else:
            raise ValueError("invalid distance metric")
