from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

# Add repository root to the Python path so the script can import from src/
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.block0_modeling.simulate_cv_target import simulate_target_cv, measure_position
from src.block1_kf.linear_kf_cv import LinearKFCV2D


def compute_rmse(truth, estimate):
    """
    Compute root-mean-square error component-wise.

    Args:
        truth: Ground-truth array with shape (N, M)
        estimate: Estimated array with shape (N, M)

    Returns:
        rmse: RMSE vector with shape (M,)
    """
    truth = np.asarray(truth, dtype=float)
    estimate = np.asarray(estimate, dtype=float)

    if truth.shape != estimate.shape:
        raise ValueError("truth and estimate must have the same shape.")

    return np.sqrt(np.mean((truth - estimate) ** 2, axis=0))

def estimate_initial_velocity_from_measurements(z, dt, n_init=8):
    """
    Estimate initial velocity using a linear fit over the first n_init measurements.

    Args:
        z: Position measurements with shape (N, 2)
        dt: Sampling time [s]
        n_init: Number of initial samples used for regression

    Returns:
        vx0, vy0: Initial velocity estimates [m/s]
    """
    if len(z) < 2:
        raise ValueError("At least two measurements are required.")
    if n_init < 2:
        raise ValueError("n_init must be at least 2.")

    n = min(n_init, len(z))
    t_fit = np.arange(n) * dt

    # Fit x(t) = ax * t + bx
    coeff_x = np.polyfit(t_fit, z[:n, 0], deg=1)
    vx0 = coeff_x[0]

    # Fit y(t) = ay * t + by
    coeff_y = np.polyfit(t_fit, z[:n, 1], deg=1)
    vy0 = coeff_y[0]

    return vx0, vy0

def main():
    # Fix the random seed for repeatable results
    np.random.seed(0)

    dt = 0.1
    steps = 50
    sigma_pos = 20.0
    sigma_a = 5.0

    # Initial true state: [px, py, vx, vy]
    x0_true = np.array([0.0, 0.0, 300.0, 40.0], dtype=float)

    # Use the existing block0 truth and measurement generators
    truth, _, t = simulate_target_cv(x0_true, dt, steps)
    z = measure_position(truth, sigma_pos=sigma_pos)

    # Initialize the filter from the first position measurement
    # Velocity is unknown initially, so start from zero and let the filter learn it
    # Initialize position from the first measurement
    # Initialize position from the first measurement
    px0, py0 = z[0]

    # Estimate initial velocity from multiple early measurements
    vx0, vy0 = estimate_initial_velocity_from_measurements(z, dt, n_init=8)

    x0_est = np.array([px0, py0, vx0, vy0], dtype=float)

    P0 = np.diag([400.0, 400.0, 10000.0, 10000.0])

    kf = LinearKFCV2D(
        dt=dt,
        sigma_a=sigma_a,
        sigma_pos=sigma_pos,
        x0=x0_est,
        P0=P0
    )

    x_est_hist = np.zeros((len(z), 4), dtype=float)
    x_est_hist[0] = np.nan

    for k in range(1, len(z)):
        x_est, _ = kf.step(z[k])
        x_est_hist[k] = x_est.ravel()

    # Compute RMSE for position and velocity
    pos_rmse = compute_rmse(truth[:, :2], x_est_hist[:, :2])
    vel_rmse = compute_rmse(truth[:, 2:], x_est_hist[:, 2:])

    print("Position RMSE [px, py] (m):", pos_rmse)
    print("Velocity RMSE [vx, vy] (m/s):", vel_rmse)
    print()

    for k in range(5):
        print(f"t[{k}]        = {t[k]:.2f} s")
        print(f"truth[{k}]    = {truth[k]}")
        print(f"meas[{k}]     = {z[k]}")
        print(f"x_est[{k}]    = {x_est_hist[k]}")
        print()

    # 2D trajectory plot
    plt.figure(figsize=(8, 6))
    plt.plot(truth[:, 0], truth[:, 1], label="True trajectory", linewidth=2)
    plt.scatter(z[:, 0], z[:, 1], label="Measurements", s=18, alpha=0.7)
    plt.plot(x_est_hist[:, 0], x_est_hist[:, 1], label="KF estimate", linewidth=2)
    plt.xlabel("x position [m]")
    plt.ylabel("y position [m]")
    plt.title("2D Constant-Velocity Tracking with Linear Kalman Filter")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    # Position components versus time
    plt.figure(figsize=(10, 6))
    plt.plot(t, truth[:, 0], label="True x", linewidth=2)
    plt.plot(t, truth[:, 1], label="True y", linewidth=2)
    plt.scatter(t, z[:, 0], label="Measured x", s=14, alpha=0.6)
    plt.scatter(t, z[:, 1], label="Measured y", s=14, alpha=0.6)
    plt.plot(t, x_est_hist[:, 0], label="Estimated x", linewidth=2)
    plt.plot(t, x_est_hist[:, 1], label="Estimated y", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("Position Tracking Performance")
    plt.legend()
    plt.grid(True)

    # Velocity components versus time
    plt.figure(figsize=(10, 6))
    plt.plot(t, truth[:, 2], label="True vx", linewidth=2)
    plt.plot(t, truth[:, 3], label="True vy", linewidth=2)
    plt.plot(t, x_est_hist[:, 2], label="Estimated vx", linewidth=2)
    plt.plot(t, x_est_hist[:, 3], label="Estimated vy", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.title("Velocity Estimation Performance")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()