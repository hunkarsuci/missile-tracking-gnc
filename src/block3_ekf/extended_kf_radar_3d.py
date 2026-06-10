import numpy as np

from src.block3_ekf.simulate_radar_measurements import wrap_angle


class ExtendedKFRadar3D:
    """
    Extended Kalman Filter for 3D target tracking with radar measurements.

    State:
        x = [px, py, pz, vx, vy, vz]^T

    Measurement:
        z = [range, azimuth, elevation]^T
    """

    def __init__(
        self,
        dt,
        sigma_a,
        sigma_r,
        sigma_az,
        sigma_el,
        radar_pos=(0.0, 0.0, 0.0),
        x0=None,
        P0=None,
    ):
        if dt <= 0:
            raise ValueError("dt must be positive.")
        if sigma_a < 0:
            raise ValueError("sigma_a must be non-negative.")
        if sigma_r < 0:
            raise ValueError("sigma_r must be non-negative.")
        if sigma_az < 0:
            raise ValueError("sigma_az must be non-negative.")
        if sigma_el < 0:
            raise ValueError("sigma_el must be non-negative.")

        self.dt = float(dt)
        self.radar_pos = radar_pos

        self.F = np.eye(6, dtype=float)
        self.F[0, 3] = self.dt
        self.F[1, 4] = self.dt
        self.F[2, 5] = self.dt

        q = sigma_a**2
        dt2 = self.dt**2
        dt3 = self.dt**3
        dt4 = self.dt**4
        q_axis = np.array([[dt4 / 4.0, dt3 / 2.0], [dt3 / 2.0, dt2]], dtype=float)

        self.Q = np.zeros((6, 6), dtype=float)
        for pos_idx, vel_idx in ((0, 3), (1, 4), (2, 5)):
            idx = np.ix_([pos_idx, vel_idx], [pos_idx, vel_idx])
            self.Q[idx] = q * q_axis

        self.R = np.diag([sigma_r**2, sigma_az**2, sigma_el**2]).astype(float)
        self.x = self._as_column_vector(x0, 6) if x0 is not None else np.zeros((6, 1))

        if P0 is None:
            self.P = np.diag([900.0, 900.0, 900.0, 40000.0, 40000.0, 40000.0]).astype(float)
        else:
            P0 = np.asarray(P0, dtype=float)
            if P0.shape != (6, 6):
                raise ValueError("P0 must have shape (6, 6).")
            self.P = P0

        self.I = np.eye(6, dtype=float)

    @staticmethod
    def _as_column_vector(x, n):
        x = np.asarray(x, dtype=float)
        if x.shape == (n,):
            return x.reshape(n, 1)
        if x.shape == (n, 1):
            return x
        raise ValueError(f"Input must have shape ({n},) or ({n}, 1).")

    def h(self, x):
        sx, sy, sz = self.radar_pos
        px, py, pz = x[0, 0], x[1, 0], x[2, 0]

        dx = px - sx
        dy = py - sy
        dz = pz - sz
        horizontal_range = np.sqrt(dx**2 + dy**2)
        range_3d = np.sqrt(horizontal_range**2 + dz**2)
        azimuth = np.arctan2(dy, dx)
        elevation = np.arctan2(dz, horizontal_range)

        return np.array([[range_3d], [azimuth], [elevation]], dtype=float)

    def jacobian_h(self, x):
        sx, sy, sz = self.radar_pos
        px, py, pz = x[0, 0], x[1, 0], x[2, 0]

        dx = px - sx
        dy = py - sy
        dz = pz - sz

        eps = 1e-9
        rho2 = max(dx**2 + dy**2, eps)
        rho = np.sqrt(rho2)
        r2 = max(rho2 + dz**2, eps)
        r = np.sqrt(r2)

        H = np.zeros((3, 6), dtype=float)
        H[0, 0] = dx / r
        H[0, 1] = dy / r
        H[0, 2] = dz / r

        H[1, 0] = -dy / rho2
        H[1, 1] = dx / rho2

        H[2, 0] = -dx * dz / (rho * r2)
        H[2, 1] = -dy * dz / (rho * r2)
        H[2, 2] = rho / r2

        return H

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy(), self.P.copy()

    def update(self, z):
        z = self._as_column_vector(z, 3)
        z_pred = self.h(self.x)
        H = self.jacobian_h(self.x)

        innovation = z - z_pred
        innovation[1, 0] = wrap_angle(innovation[1, 0])
        innovation[2, 0] = wrap_angle(innovation[2, 0])

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ innovation
        I_KH = self.I - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return self.x.copy(), self.P.copy()

    def step(self, z):
        self.predict()
        return self.update(z)
