import numpy as np


def wrap_angle(angle):
    """
    Wrap angle to the interval [-pi, pi).
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class ExtendedKFRadar2D:
    """
    Extended Kalman Filter for 2D target tracking with radar measurements.

    State:
        x = [px, py, vx, vy]^T

    Measurement:
        z = [range, bearing]^T
    """

    def __init__(self, dt, sigma_a, sigma_r, sigma_b, radar_pos=(0.0, 0.0), x0=None, P0=None):
        """
        Args:
            dt: Sampling time [s]
            sigma_a: Process acceleration standard deviation [m/s^2]
            sigma_r: Range measurement noise standard deviation [m]
            sigma_b: Bearing measurement noise standard deviation [rad]
            radar_pos: Radar position (sx, sy) [m]
            x0: Initial state estimate, shape (4,) or (4, 1)
            P0: Initial covariance, shape (4, 4)
        """
        if dt <= 0:
            raise ValueError("dt must be positive.")
        if sigma_a < 0:
            raise ValueError("sigma_a must be non-negative.")
        if sigma_r < 0:
            raise ValueError("sigma_r must be non-negative.")
        if sigma_b < 0:
            raise ValueError("sigma_b must be non-negative.")

        self.dt = float(dt)
        self.radar_pos = radar_pos

        # Constant-velocity state transition matrix
        self.F = np.array([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=float)

        # Discrete white-acceleration process noise
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

        self.R = np.diag([sigma_r ** 2, sigma_b ** 2]).astype(float)

        if x0 is None:
            self.x = np.zeros((4, 1), dtype=float)
        else:
            self.x = self._as_column_vector(x0, 4)

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
        x = np.asarray(x, dtype=float)
        if x.shape == (n,):
            return x.reshape(n, 1)
        if x.shape == (n, 1):
            return x
        raise ValueError(f"Input must have shape ({n},) or ({n}, 1).")

    def h(self, x):
        """
        Nonlinear radar measurement function.

        Args:
            x: State vector with shape (4, 1)

        Returns:
            z_pred: Predicted measurement [range, bearing]^T, shape (2, 1)
        """
        sx, sy = self.radar_pos
        px, py = x[0, 0], x[1, 0]

        dx = px - sx
        dy = py - sy

        r = np.sqrt(dx**2 + dy**2)
        b = np.arctan2(dy, dx)

        return np.array([[r], [b]], dtype=float)

    def jacobian_h(self, x):
        """
        Jacobian of the radar measurement model h(x).

        Args:
            x: State vector with shape (4, 1)

        Returns:
            H_j: Jacobian matrix with shape (2, 4)
        """
        sx, sy = self.radar_pos
        px, py = x[0, 0], x[1, 0]

        dx = px - sx
        dy = py - sy

        r2 = dx**2 + dy**2
        r = np.sqrt(r2)

        eps = 1e-9
        r2 = max(r2, eps)
        r = max(r, eps)

        H_j = np.array([
            [dx / r,        dy / r,        0.0, 0.0],
            [-dy / r2,      dx / r2,       0.0, 0.0]
        ], dtype=float)

        return H_j

    def predict(self):
        """
        Perform the time update step.
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy(), self.P.copy()

    def update(self, z):
        """
        Perform the measurement update step.

        Args:
            z: Radar measurement [range, bearing], shape (2,) or (2, 1)
        """
        z = self._as_column_vector(z, 2)

        z_pred = self.h(self.x)
        H_j = self.jacobian_h(self.x)

        # Innovation
        y = z - z_pred
        y[1, 0] = wrap_angle(y[1, 0])

        # Innovation covariance
        S = H_j @ self.P @ H_j.T + self.R

        # Kalman gain
        K = self.P @ H_j.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Joseph-form covariance update
        I_KH = self.I - K @ H_j
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return self.x.copy(), self.P.copy()

    def step(self, z):
        """
        Run one complete EKF cycle.
        """
        self.predict()
        return self.update(z)