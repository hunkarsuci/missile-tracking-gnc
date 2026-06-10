from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.block3_ekf.extended_kf_radar_3d import ExtendedKFRadar3D
from src.block3_ekf.simulate_radar_measurements_3d import (
    measure_radar_3d,
    radar_to_cartesian_3d,
)


def altitude_profile(t):
    """
    Smooth 3D demo altitude profile.

    Sequence:
        0-8 s: climb from 0 km to 10 km
        8-11 s: descend from 10 km to 8 km
        11-13 s: climb from 8 km to 9 km
        13+ s: hold near 9 km while the target continues horizontally
    """
    segments = (
        (0.0, 8.0, 0.0, 10000.0),
        (8.0, 11.0, 10000.0, 8000.0),
        (11.0, 13.0, 8000.0, 9000.0),
    )

    for t0, t1, z0, z1 in segments:
        if t <= t1:
            duration = t1 - t0
            u = np.clip((t - t0) / duration, 0.0, 1.0)
            smooth = u**2 * (3.0 - 2.0 * u)
            smooth_dot = 6.0 * u * (1.0 - u) / duration
            altitude = z0 + (z1 - z0) * smooth
            climb_rate = (z1 - z0) * smooth_dot
            return altitude, climb_rate

    return 9000.0, 0.0


def evasive_acceleration_3d(t):
    """
    Lateral maneuver profile used by the 3D demo.

    The profile intentionally changes direction several times so the animation
    shows a curved target path instead of a mostly straight flyout.
    """
    ax = 18.0 * np.sin(0.95 * t) + 8.0 * np.sin(2.1 * t)
    ay = 42.0 * np.sin(0.72 * t + 0.4) - 30.0 * np.sin(1.55 * t)
    az = 0.0

    if 3.2 <= t < 5.0:
        ay += 28.0
    elif 5.0 <= t < 6.6:
        ay -= 38.0

    if 6.0 <= t < 7.4:
        ax -= 22.0

    return np.array([ax, ay, az], dtype=float)


def simulate_evasive_target_3d(x0, dt, steps):
    """
    Simulate a target with stronger x/y/z maneuvering for visualization.
    """
    x = np.asarray(x0, dtype=float).copy()
    if x.shape != (6,):
        raise ValueError("x0 must have shape (6,).")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    transition = np.eye(6, dtype=float)
    transition[0, 3] = dt
    transition[1, 4] = dt
    transition[2, 5] = dt

    control = np.zeros((6, 3), dtype=float)
    control[0, 0] = 0.5 * dt**2
    control[1, 1] = 0.5 * dt**2
    control[2, 2] = 0.5 * dt**2
    control[3, 0] = dt
    control[4, 1] = dt
    control[5, 2] = dt

    t = np.arange(steps + 1) * dt
    x[2], x[5] = altitude_profile(0.0)
    truth = [x.copy()]
    acceleration = [evasive_acceleration_3d(0.0)]

    for k in range(steps):
        a_k = evasive_acceleration_3d(t[k])
        x = transition @ x + control @ a_k
        x[2], x[5] = altitude_profile(t[k + 1])
        truth.append(x.copy())
        acceleration.append(a_k.copy())

    return np.array(truth), t, np.array(acceleration)


def estimate_initial_state_from_radar_3d(z_radar, dt, radar_pos=(0.0, 0.0, 0.0), n_init=18):
    """
    Estimate an initial 3D Cartesian state from early radar measurements.
    """
    if len(z_radar) < n_init:
        raise ValueError("Number of radar measurements must be at least n_init.")
    if n_init < 2:
        raise ValueError("n_init must be at least 2.")

    k0 = n_init - 1
    t_fit = np.arange(n_init) * dt
    pos_cart = np.zeros((n_init, 3), dtype=float)

    for k in range(n_init):
        pos_cart[k] = radar_to_cartesian_3d(
            z_radar[k, 0],
            z_radar[k, 1],
            z_radar[k, 2],
            radar_pos=radar_pos,
        )

    coeffs = [np.polyfit(t_fit, pos_cart[:, axis], deg=1) for axis in range(3)]
    t0 = t_fit[k0]
    position = np.array([np.polyval(coeff, t0) for coeff in coeffs], dtype=float)
    velocity = np.array([coeff[0] for coeff in coeffs], dtype=float)

    return np.concatenate((position, velocity)), k0


def build_tracking_scenario(seed=7):
    """
    Build a repeatable radar-EKF tracking scenario for 3D visualization.

    The underlying filter is a 3D radar EKF with state
    [px, py, pz, vx, vy, vz].
    """
    np.random.seed(seed)

    dt = 0.05
    steps = 320
    sigma_a = 85.0
    sigma_r = 10.0
    sigma_az = np.deg2rad(0.28)
    sigma_el = np.deg2rad(0.22)
    radar_pos = (0.0, 0.0, 0.0)
    x0_true = np.array([1600.0, -700.0, 0.0, 285.0, 155.0, 0.0], dtype=float)

    truth, t, _ = simulate_evasive_target_3d(x0_true, dt, steps)
    z_radar = measure_radar_3d(
        truth_states=truth,
        sigma_r=sigma_r,
        sigma_az=sigma_az,
        sigma_el=sigma_el,
        radar_pos=radar_pos,
    )

    z_cart = np.zeros((len(z_radar), 3), dtype=float)
    for k, (range_k, azimuth_k, elevation_k) in enumerate(z_radar):
        z_cart[k] = radar_to_cartesian_3d(
            range_k,
            azimuth_k,
            elevation_k,
            radar_pos=radar_pos,
        )

    x0_est, k0 = estimate_initial_state_from_radar_3d(
        z_radar=z_radar,
        dt=dt,
        radar_pos=radar_pos,
        n_init=18,
    )

    ekf = ExtendedKFRadar3D(
        dt=dt,
        sigma_a=sigma_a,
        sigma_r=sigma_r,
        sigma_az=sigma_az,
        sigma_el=sigma_el,
        radar_pos=radar_pos,
        x0=x0_est,
        P0=np.diag([400.0, 400.0, 400.0, 90000.0, 90000.0, 90000.0]),
    )

    x_est_hist = np.full((len(z_radar), 6), np.nan, dtype=float)
    x_est_hist[k0] = x0_est
    for k in range(k0 + 1, len(z_radar)):
        x_est, _ = ekf.step(z_radar[k])
        x_est_hist[k] = x_est.ravel()

    return {
        "t": t,
        "truth": truth[:, :3],
        "measurements": z_cart,
        "estimate": x_est_hist[:, :3],
        "radar": np.array(radar_pos, dtype=float),
    }


def set_equal_3d_axes(ax, points):
    valid = points[~np.isnan(points).any(axis=1)]
    mins = valid.min(axis=0)
    maxs = valid.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = max(maxs - mins) * 0.55

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(max(0.0, centers[2] - radius), centers[2] + radius)


def create_animation(save_path=None, show=True):
    scenario = build_tracking_scenario()
    t = scenario["t"]
    truth = scenario["truth"]
    measurements = scenario["measurements"]
    estimate = scenario["estimate"]
    radar = scenario["radar"]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#f7f7f4")
    ax.set_facecolor("#f7f7f4")

    all_points = np.vstack((truth, measurements, estimate, radar.reshape(1, 3)))
    set_equal_3d_axes(ax, all_points)

    ax.set_xlabel("x position [m]")
    ax.set_ylabel("y position [m]")
    ax.set_zlabel("altitude [m]")
    ax.set_title("3D Radar-EKF Tracking: Evasive Target Maneuver")
    ax.view_init(elev=25, azim=-46)
    ax.grid(True, alpha=0.25)

    truth_line, = ax.plot([], [], [], color="#1f77b4", linewidth=2.4, label="Truth")
    estimate_line, = ax.plot([], [], [], color="#d62728", linewidth=2.2, label="EKF estimate")
    meas_scatter = ax.scatter([], [], [], color="#555555", s=12, alpha=0.3, label="Radar measurements")
    target_marker = ax.scatter([], [], [], color="#1f77b4", s=58)
    estimate_marker = ax.scatter([], [], [], color="#d62728", s=48)
    ax.scatter([radar[0]], [radar[1]], [radar[2]], color="#111111", marker="^", s=90, label="Radar")
    time_text = ax.text2D(0.03, 0.94, "", transform=ax.transAxes)
    ax.legend(loc="upper left")

    def update(frame):
        end = frame + 1

        truth_line.set_data(truth[:end, 0], truth[:end, 1])
        truth_line.set_3d_properties(truth[:end, 2])

        estimate_line.set_data(estimate[:end, 0], estimate[:end, 1])
        estimate_line.set_3d_properties(estimate[:end, 2])

        stride = max(1, end // 45)
        meas = measurements[:end:stride]
        meas_scatter._offsets3d = (meas[:, 0], meas[:, 1], meas[:, 2])

        target_marker._offsets3d = (
            [truth[frame, 0]],
            [truth[frame, 1]],
            [truth[frame, 2]],
        )

        if np.isnan(estimate[frame]).any():
            estimate_marker._offsets3d = ([], [], [])
        else:
            estimate_marker._offsets3d = (
                [estimate[frame, 0]],
                [estimate[frame, 1]],
                [estimate[frame, 2]],
            )

        time_text.set_text(f"t = {t[frame]:.2f} s")
        return truth_line, estimate_line, meas_scatter, target_marker, estimate_marker, time_text

    if not save_path and not show:
        update(0)
        fig.canvas.draw()
        plt.close(fig)
        return None

    animation = FuncAnimation(
        fig,
        update,
        frames=len(t),
        interval=45,
        blit=False,
        repeat=True,
    )

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        animation.save(save_path, writer="pillow", fps=24)
        print(f"Saved animation to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return animation


def parse_args():
    parser = argparse.ArgumentParser(description="Animate a 3D radar-EKF tracking scenario.")
    parser.add_argument("--save", type=Path, help="Optional GIF output path, for example assets/tracking_3d.gif.")
    parser.add_argument("--no-show", action="store_true", help="Create or save the animation without opening a window.")
    return parser.parse_args()


def main():
    args = parse_args()
    create_animation(save_path=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
