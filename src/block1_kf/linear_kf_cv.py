import numpy as np


class LinearKFCV2D:
    """
    Linear Kalman Filter for a 2D constant-velocity target model.

    State:
        x = [px, py, vx, vy]^T

    Measurement:
        z = [px, py]^T
    """

    def __init__(self, dt, sigma_a, sigma_pos, x0=None, P0=None):
        """
        Args:
            dt: Sampling time [s]
            sigma_a: Process acceleration standard deviation [m/s^2]
            sigma_pos: Position measurement standard deviation [m]
            x0: Initial state estimate, shape (4,) or (4, 1)
            P0: Initial covariance, shape (4, 4)
        """
        if dt <= 0:
            raise ValueError("dt must be positive.")
        if sigma_a < 0:
            raise ValueError("sigma_a must be non-negative.")
        if sigma_pos < 0:
            raise ValueError("sigma_pos must be non-negative.")

        self.dt = float(dt)

        # State transition matrix for constant-velocity motion
        self.F = np.array([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=float)

        # Measurement matrix: only position is observed
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=float)

        # Discrete white-acceleration process noise model
        q = sigma_a ** 2
        dt2 = self.dt ** 2
        dt3 = self.dt ** 3
        dt4 = self.dt ** 4

        self.Q = q * np.array([
            [dt4 / 4.0, 0.0,       dt3 / 2.0, 0.0],
            [0.0,       dt4 / 4.0, 0.0,       dt3 / 2.0],
            [dt3 / 2.0, 0.0,       dt2,       0.0],
            [0.0,       dt3 / 2.0, 0.0,       dt2]
        ], dtype=float)

        # Measurement noise covariance
        self.R = (sigma_pos ** 2) * np.eye(2, dtype=float)

        # Initial state estimate
        if x0 is None:
            self.x = np.zeros((4, 1), dtype=float)
        else:
            self.x = self._as_column_vector(x0, 4)

        # Initial covariance
        if P0 is None:
            self.P = np.diag([400.0, 400.0, 10000.0, 10000.0]).astype(float)
        else:
            P0 = np.asarray(P0, dtype=float)
            if P0.shape != (4, 4):
                raise ValueError("P0 must have shape (4, 4).")
            self.P = P0

        self.I = np.eye(4, dtype=float)

    @staticmethod
    def _as_column_vector(x, n):
        """
        Convert input into a column vector of shape (n, 1).
        """
        x = np.asarray(x, dtype=float)
        if x.shape == (n,):
            return x.reshape(n, 1)
        if x.shape == (n, 1):
            return x
        raise ValueError(f"Input must have shape ({n},) or ({n}, 1).")

    def predict(self):
        """
        Perform the time update step.

        Returns:
            x_pred: Predicted state estimate, shape (4, 1)
            P_pred: Predicted covariance, shape (4, 4)
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy(), self.P.copy()

    def update(self, z):
        """
        Perform the measurement update step.

        Args:
            z: Measurement vector [px, py], shape (2,) or (2, 1)

        Returns:
            x_upd: Updated state estimate, shape (4, 1)
            P_upd: Updated covariance, shape (4, 4)
        """
        z = self._as_column_vector(z, 2)

        # Innovation
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Joseph-form covariance update for better numerical stability
        I_KH = self.I - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return self.x.copy(), self.P.copy()

    def step(self, z):
        """
        Run one complete Kalman filter cycle.

        Args:
            z: Measurement vector [px, py]

        Returns:
            x: Updated state estimate, shape (4, 1)
            P: Updated covariance, shape (4, 4)
        """
        self.predict()
        return self.update(z)