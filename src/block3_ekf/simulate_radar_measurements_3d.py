import numpy as np

from src.block3_ekf.simulate_radar_measurements import wrap_angle


def cartesian_to_radar_3d(px, py, pz, radar_pos=(0.0, 0.0, 0.0)):
    """
    Convert a 3D Cartesian position to radar range, azimuth, and elevation.
    """
    sx, sy, sz = radar_pos
    dx = px - sx
    dy = py - sy
    dz = pz - sz

    horizontal_range = np.sqrt(dx**2 + dy**2)
    range_3d = np.sqrt(horizontal_range**2 + dz**2)
    azimuth = np.arctan2(dy, dx)
    elevation = np.arctan2(dz, horizontal_range)

    return range_3d, azimuth, elevation


def radar_to_cartesian_3d(range_3d, azimuth, elevation, radar_pos=(0.0, 0.0, 0.0)):
    """
    Convert radar range, azimuth, and elevation to a 3D Cartesian position.
    """
    sx, sy, sz = radar_pos
    horizontal_range = range_3d * np.cos(elevation)

    px = sx + horizontal_range * np.cos(azimuth)
    py = sy + horizontal_range * np.sin(azimuth)
    pz = sz + range_3d * np.sin(elevation)

    return px, py, pz


def measure_radar_3d(
    truth_states,
    sigma_r,
    sigma_az,
    sigma_el,
    radar_pos=(0.0, 0.0, 0.0),
):
    """
    Generate noisy 3D radar measurements from true target states.

    State shape:
        truth_states[:, 0:3] = [px, py, pz]

    Measurement:
        z = [range, azimuth, elevation]
    """
    truth_states = np.asarray(truth_states, dtype=float)

    if truth_states.ndim != 2 or truth_states.shape[1] < 3:
        raise ValueError("truth_states must have shape (N, M) with M >= 3.")
    if sigma_r < 0:
        raise ValueError("sigma_r must be non-negative.")
    if sigma_az < 0:
        raise ValueError("sigma_az must be non-negative.")
    if sigma_el < 0:
        raise ValueError("sigma_el must be non-negative.")

    measurements = np.zeros((len(truth_states), 3), dtype=float)

    for k, state in enumerate(truth_states):
        range_true, az_true, el_true = cartesian_to_radar_3d(
            state[0],
            state[1],
            state[2],
            radar_pos=radar_pos,
        )

        measurements[k] = np.array([
            range_true + np.random.randn() * sigma_r,
            wrap_angle(az_true + np.random.randn() * sigma_az),
            wrap_angle(el_true + np.random.randn() * sigma_el),
        ])

    return measurements
