import numpy as np


class BallKalmanTracker:
    """Kalman filter tuned for football ball centres.

    The state uses a constant-acceleration model:
    [x, y, dx, dy, ddx, ddy]. This is intentionally separate from player
    tracking because the ball can accelerate and reverse direction much faster
    than athletes in broadcast footage.
    """

    def __init__(self, max_lost_frames=8, process_noise_scale=50.0, measurement_noise_scale=5.0):
        self.dt = 1.0
        self.F = np.asarray(
            [
                [1, 0, self.dt, 0, 0.5 * self.dt**2, 0],
                [0, 1, 0, self.dt, 0, 0.5 * self.dt**2],
                [0, 0, 1, 0, self.dt, 0],
                [0, 0, 0, 1, 0, self.dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.H = np.asarray(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        self.Q = np.eye(6, dtype=np.float32) * float(process_noise_scale)
        self.R = np.eye(2, dtype=np.float32) * float(measurement_noise_scale)
        self.P = np.eye(6, dtype=np.float32) * 100.0
        self.x = None
        self.lost_frames = 0
        self.max_lost_frames = int(max_lost_frames)

    def init(self, center_xy):
        self.x = np.asarray(
            [float(center_xy[0]), float(center_xy[1]), 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        self.lost_frames = 0

    def predict(self):
        if self.x is None:
            return None
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.lost_frames += 1
        return self.x[:2].astype(float).tolist()

    def update(self, center_xy):
        if self.x is None:
            self.init(center_xy)
            return self.x[:2].astype(float).tolist()
        z = np.asarray([float(center_xy[0]), float(center_xy[1])], dtype=np.float32)
        residual = z - self.H @ self.x
        innovation = self.H @ self.P @ self.H.T + self.R
        gain = self.P @ self.H.T @ np.linalg.inv(innovation)
        self.x = self.x + gain @ residual
        identity = np.eye(6, dtype=np.float32)
        self.P = (identity - gain @ self.H) @ self.P
        self.lost_frames = 0
        return self.x[:2].astype(float).tolist()

    def is_valid(self):
        return self.x is not None and self.lost_frames <= self.max_lost_frames

    def reset(self):
        self.x = None
        self.lost_frames = 0
        self.P = np.eye(6, dtype=np.float32) * 100.0
