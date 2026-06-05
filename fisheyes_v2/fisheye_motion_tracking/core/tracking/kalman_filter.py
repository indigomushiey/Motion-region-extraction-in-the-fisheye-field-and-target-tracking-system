import cv2
import numpy as np


class KalmanTracker:
    """Kalman filter for 2D target tracking with constant velocity model.

    State: [x, y, vx, vy]
    Measurement: [x, y]
    """

    def __init__(self, init_x, init_y, dt=1.0, process_noise=1e-2, measurement_noise=1e-1):
        self.dt = dt
        self.kf = cv2.KalmanFilter(4, 2, 0)

        # state transition (constant velocity)
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float32)

        # measurement matrix
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # process noise covariance
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise

        # measurement noise covariance
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

        # error covariance posterior
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        # initial state
        self.kf.statePost = np.array([[init_x], [init_y], [0], [0]], dtype=np.float32)

        self.history = [(init_x, init_y)]
        self.missed_frames = 0

    def predict(self):
        """Predict next state and return predicted position (x, y)."""
        predicted = self.kf.predict()
        return float(predicted[0]), float(predicted[1])

    def update(self, x, y):
        """Update filter with new measurement and return corrected state."""
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        corrected = self.kf.correct(measurement)
        self.history.append((float(corrected[0]), float(corrected[1])))
        self.missed_frames = 0
        return float(corrected[0]), float(corrected[1])

    def update_missing(self):
        """Handle a frame where this target was not detected."""
        self.missed_frames += 1
        self.predict()
        self.history.append((float(self.kf.statePost[0]), float(self.kf.statePost[1])))

    @property
    def state(self):
        s = self.kf.statePost
        return float(s[0]), float(s[1]), float(s[2]), float(s[3])

    @property
    def position(self):
        return float(self.kf.statePost[0]), float(self.kf.statePost[1])

    @property
    def velocity(self):
        return float(self.kf.statePost[2]), float(self.kf.statePost[3])

    @property
    def trajectory(self):
        return self.history.copy()
