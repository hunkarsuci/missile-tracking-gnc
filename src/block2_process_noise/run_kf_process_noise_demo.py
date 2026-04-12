from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

# Add repository root to the Python path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.block0_modeling.simulate_cv_target import measure_position
from src.block1_kf.linear_kf_cv import LinearKFCV2D
from src.block2_process_noise.simulate_maneuver_target import simulate_target_maneuvering


def compute_rmse(truth, estimate):
    """
    Compute component-wise RMSE.
    """
    truth = np.asarray(truth, dtype=float)
    estimate = np.asarray(estimate, dtype=float)

    if truth.shape != estimate.shape:
        raise ValueError("truth and estimate must have the same shape.")

    return np.sqrt(np.mean((truth - estimate) ** 2, axis=0))

def estimate_initial_velocity_from_measurements(z, dt, n_init=16):
    """
    Estimate initial velocity from the first n_init position measurements
    using a linear fit.
    """
    if len(z) < 2:
        raise ValueError("At least two measurements are required.")
    if n_init < 2:
        raise ValueError("n_init must be at least 2.")

    n = min(n_init, len(z))
    t_fit = np.arange(n) * dt

    coeff_x = np.polyfit(t_fit, z[:n, 0], deg=1)
    coeff_y = np.polyfit(t_fit, z[:n, 1], deg=1)

    vx0 = coeff_x[0]
    vy0 = coeff_y[0]

    return vx0, vy0


def run_filter(z, dt, sigma_pos, sigma_a, n_init=8):
    """
    Run the constant-velocity KF on a measurement sequence.
    """
    px0, py0 = z[0]
    vx0, vy0 = estimate_initial_velocity_from_measurements(z, dt, n_init=n_init)

    x0_est = np.array([px0, py0, vx0, vy0], dtype=float)
    P0 = np.diag([400.0, 400.0, 10000.0, 10000.0])

    kf = LinearKFCV2D(
        dt=dt,
        sigma_a=sigma_a,
        sigma_pos=sigma_pos,
        x0=x0_est,
        P0=P0
    )

    x_est_hist = np.full((len(z), 4), np.nan, dtype=float)

    for k in range(1, len(z)):
        x_est, _ = kf.step(z[k])
        x_est_hist[k] = x_est.ravel()

    return x_est_hist


def main():
    np.random.seed(0)

    dt = 0.05
    steps = 100
    sigma_pos = 20.0

    x0_true = np.array([0.0, 0.0, 300.0, 40.0], dtype=float)

    truth, t, acc_hist = simulate_target_maneuvering(x0_true, dt, steps)
    z = measure_position(truth, sigma_pos=sigma_pos)

    sigma_a_values = [1.0, 5.0, 15.0]
    estimates = {}

    for sigma_a in sigma_a_values:
        estimates[sigma_a] = run_filter(
            z=z,
            dt=dt,
            sigma_pos=sigma_pos,
            sigma_a=sigma_a,
            n_init=16
        )

    for sigma_a in sigma_a_values:
        est = estimates[sigma_a]
        valid = ~np.isnan(est[:, 0])
        pos_rmse = compute_rmse(truth[valid, :2], est[valid, :2])
        vel_rmse = compute_rmse(truth[valid, 2:], est[valid, 2:])
        print(f"sigma_a = {sigma_a:.1f}")
        print("  Position RMSE [px, py] (m):", pos_rmse)
        print("  Velocity RMSE [vx, vy] (m/s):", vel_rmse)
        print()

    # 2D trajectory
    plt.figure(figsize=(9, 7))
    plt.plot(truth[:, 0], truth[:, 1], label="True trajectory", linewidth=2)
    plt.scatter(z[:, 0], z[:, 1], label="Measurements", s=18, alpha=0.5)

    for sigma_a in sigma_a_values:
        est = estimates[sigma_a]
        plt.plot(est[:, 0], est[:, 1], label=f"KF estimate (sigma_a={sigma_a})", linewidth=2)

    plt.xlabel("x position [m]")
    plt.ylabel("y position [m]")
    plt.title("Maneuvering Target Tracking with Different Process Noise Levels")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    # x position vs time
    plt.figure(figsize=(10, 6))
    plt.plot(t, truth[:, 0], label="True x", linewidth=2)
    plt.scatter(t, z[:, 0], label="Measured x", s=14, alpha=0.5)

    for sigma_a in sigma_a_values:
        est = estimates[sigma_a]
        plt.plot(t, est[:, 0], label=f"Estimated x (sigma_a={sigma_a})", linewidth=2)

    plt.xlabel("Time [s]")
    plt.ylabel("x position [m]")
    plt.title("X Position Tracking vs Process Noise")
    plt.legend()
    plt.grid(True)

    # y position vs time
    plt.figure(figsize=(10, 6))
    plt.plot(t, truth[:, 1], label="True y", linewidth=2)
    plt.scatter(t, z[:, 1], label="Measured y", s=14, alpha=0.5)

    for sigma_a in sigma_a_values:
        est = estimates[sigma_a]
        plt.plot(t, est[:, 1], label=f"Estimated y (sigma_a={sigma_a})", linewidth=2)

    plt.xlabel("Time [s]")
    plt.ylabel("y position [m]")
    plt.title("Y Position Tracking vs Process Noise")
    plt.legend()
    plt.grid(True)

    # vx velocity vs time
    plt.figure(figsize=(10, 6))
    plt.plot(t, truth[:, 2], label="True vx", linewidth=2)

    for sigma_a in sigma_a_values:
        est = estimates[sigma_a]
        plt.plot(t, est[:, 2], label=f"Estimated vx (sigma_a={sigma_a})", linewidth=2)

    plt.xlabel("Time [s]")
    plt.ylabel("x velocity [m/s]")
    plt.title("X Velocity Tracking vs Process Noise")
    plt.legend()
    plt.grid(True)

    # vy velocity vs time
    plt.figure(figsize=(10, 6))
    plt.plot(t, truth[:, 3], label="True vy", linewidth=2)

    for sigma_a in sigma_a_values:
        est = estimates[sigma_a]
        plt.plot(t, est[:, 3], label=f"Estimated vy (sigma_a={sigma_a})", linewidth=2)

    plt.xlabel("Time [s]")
    plt.ylabel("y velocity [m/s]")
    plt.title("Y Velocity Tracking vs Process Noise")
    plt.legend()
    plt.grid(True)

    # Applied acceleration profile
    plt.figure(figsize=(10, 5))
    plt.plot(t, acc_hist[:, 0], label="ax", linewidth=2)
    plt.plot(t, acc_hist[:, 1], label="ay", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [m/s^2]")
    plt.title("Applied Target Acceleration Profile")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()