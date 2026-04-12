from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

# Add repository root to the Python path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.block2_process_noise.simulate_maneuver_target import simulate_target_maneuvering
from src.block3_ekf.simulate_radar_measurements import (
    measure_radar,
    radar_to_cartesian,
)
from src.block3_ekf.extended_kf_radar import ExtendedKFRadar2D


def compute_rmse(truth, estimate):
    """
    Compute component-wise RMSE.
    """
    truth = np.asarray(truth, dtype=float)
    estimate = np.asarray(estimate, dtype=float)

    if truth.shape != estimate.shape:
        raise ValueError("truth and estimate must have the same shape.")

    return np.sqrt(np.mean((truth - estimate) ** 2, axis=0))


def estimate_initial_state_from_radar(z_radar, dt, radar_pos=(0.0, 0.0), n_init=16):
    """
    Estimate an initial Cartesian state from the first n_init radar measurements.

    The estimated state is referenced to the last sample in the initialization window,
    i.e. time index k0 = n_init - 1.

    Args:
        z_radar: Radar measurements with shape (N, 2), [range, bearing]
        dt: Sampling time [s]
        radar_pos: Radar position (sx, sy) [m]
        n_init: Number of initial samples used for regression

    Returns:
        x0_est: Initial state estimate [px, py, vx, vy] at time index k0
        k0: Initialization index
    """
    if len(z_radar) < 2:
        raise ValueError("At least two radar measurements are required.")
    if n_init < 2:
        raise ValueError("n_init must be at least 2.")
    if len(z_radar) < n_init:
        raise ValueError("Number of radar measurements must be at least n_init.")

    n = n_init
    k0 = n - 1
    t_fit = np.arange(n) * dt

    pos_cart = np.zeros((n, 2), dtype=float)
    for k in range(n):
        r_k, b_k = z_radar[k]
        pos_cart[k, 0], pos_cart[k, 1] = radar_to_cartesian(
            r_k, b_k, radar_pos=radar_pos
        )

    # Fit x(t) and y(t) with first-order polynomials
    coeff_x = np.polyfit(t_fit, pos_cart[:, 0], deg=1)
    coeff_y = np.polyfit(t_fit, pos_cart[:, 1], deg=1)

    # Evaluate the fitted position at the end of the initialization window
    t0 = t_fit[k0]
    px0 = np.polyval(coeff_x, t0)
    py0 = np.polyval(coeff_y, t0)

    vx0 = coeff_x[0]
    vy0 = coeff_y[0]

    x0_est = np.array([px0, py0, vx0, vy0], dtype=float)
    return x0_est, k0


def main():
    np.random.seed(0)

    dt = 0.05
    steps = 100

    sigma_a = 5.0
    sigma_r = 15.0
    sigma_b_deg = 1.0
    sigma_b = np.deg2rad(sigma_b_deg)

    radar_pos = (0.0, 0.0)

    # Important:
    # Do not start the target at the radar origin for range-bearing measurements.
    x0_true = np.array([2000.0, 500.0, 300.0, 40.0], dtype=float)

    truth, t, acc_hist = simulate_target_maneuvering(x0_true, dt, steps)
    z_radar = measure_radar(
        truth_states=truth,
        sigma_r=sigma_r,
        sigma_b=sigma_b,
        radar_pos=radar_pos
    )

    # Convert noisy radar measurements to Cartesian only for visualization
    z_cart = np.zeros((len(z_radar), 2), dtype=float)
    for k in range(len(z_radar)):
        z_cart[k, 0], z_cart[k, 1] = radar_to_cartesian(
            z_radar[k, 0], z_radar[k, 1], radar_pos=radar_pos
        )

    n_init = 16

    x0_est, k0 = estimate_initial_state_from_radar(
        z_radar=z_radar,
        dt=dt,
        radar_pos=radar_pos,
        n_init=n_init
    )

    P0 = np.diag([400.0, 400.0, 10000.0, 10000.0])

    ekf = ExtendedKFRadar2D(
        dt=dt,
        sigma_a=sigma_a,
        sigma_r=sigma_r,
        sigma_b=sigma_b,
        radar_pos=radar_pos,
        x0=x0_est,
        P0=P0
    )

    x_est_hist = np.full((len(z_radar), 4), np.nan, dtype=float)

    # Store the initialized state at the end of the initialization window
    x_est_hist[k0] = x0_est

    # Start recursive EKF updates after the initialization window
    for k in range(k0 + 1, len(z_radar)):
        x_est, _ = ekf.step(z_radar[k])
        x_est_hist[k] = x_est.ravel()

    valid = ~np.isnan(x_est_hist[:, 0])
    pos_rmse = compute_rmse(truth[valid, :2], x_est_hist[valid, :2])
    vel_rmse = compute_rmse(truth[valid, 2:], x_est_hist[valid, 2:])

    print("Radar EKF results")
    print(f"sigma_r = {sigma_r:.1f} m")
    print(f"sigma_b = {sigma_b_deg:.2f} deg")
    print(f"sigma_a = {sigma_a:.1f} m/s^2")
    print("Position RMSE [px, py] (m):", pos_rmse)
    print("Velocity RMSE [vx, vy] (m/s):", vel_rmse)
    print()

    # 2D trajectory plot
    plt.figure(figsize=(9, 7))
    plt.plot(truth[:, 0], truth[:, 1], label="True trajectory", linewidth=2)
    plt.scatter(z_cart[:, 0], z_cart[:, 1], label="Radar measurements (converted to Cartesian)", s=16, alpha=0.5)
    plt.plot(x_est_hist[:, 0], x_est_hist[:, 1], label="EKF estimate", linewidth=2)
    plt.scatter(radar_pos[0], radar_pos[1], label="Radar", marker="x", s=100)
    plt.xlabel("x position [m]")
    plt.ylabel("y position [m]")
    plt.title("2D Radar Tracking with EKF")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    # x position vs time
    plt.figure(figsize=(10, 6))
    plt.plot(t, truth[:, 0], label="True x", linewidth=2)
    plt.scatter(t, z_cart[:, 0], label="Measured x (converted)", s=14, alpha=0.4)
    plt.plot(t, x_est_hist[:, 0], label="Estimated x", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("x position [m]")
    plt.title("X Position: Truth vs Radar-EKF Estimate")
    plt.legend()
    plt.grid(True)

    # y position vs time
    plt.figure(figsize=(10, 6))
    plt.plot(t, truth[:, 1], label="True y", linewidth=2)
    plt.scatter(t, z_cart[:, 1], label="Measured y (converted)", s=14, alpha=0.4)
    plt.plot(t, x_est_hist[:, 1], label="Estimated y", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("y position [m]")
    plt.title("Y Position: Truth vs Radar-EKF Estimate")
    plt.legend()
    plt.grid(True)

    # vx velocity vs time
    plt.figure(figsize=(10, 6))
    plt.plot(t, truth[:, 2], label="True vx", linewidth=2)
    plt.plot(t, x_est_hist[:, 2], label="Estimated vx", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("x velocity [m/s]")
    plt.title("X Velocity: Truth vs Radar-EKF Estimate")
    plt.legend()
    plt.grid(True)

    # vy velocity vs time
    plt.figure(figsize=(10, 6))
    plt.plot(t, truth[:, 3], label="True vy", linewidth=2)
    plt.plot(t, x_est_hist[:, 3], label="Estimated vy", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("y velocity [m/s]")
    plt.title("Y Velocity: Truth vs Radar-EKF Estimate")
    plt.legend()
    plt.grid(True)

    # Range and bearing measurements
    plt.figure(figsize=(10, 6))
    plt.plot(t, z_radar[:, 0], label="Measured range", linewidth=1.5)
    plt.xlabel("Time [s]")
    plt.ylabel("Range [m]")
    plt.title("Radar Range Measurements")
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 6))
    plt.plot(t, np.rad2deg(z_radar[:, 1]), label="Measured bearing", linewidth=1.5)
    plt.xlabel("Time [s]")
    plt.ylabel("Bearing [deg]")
    plt.title("Radar Bearing Measurements")
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