import numpy as np


def wrap_angle(angle):
    """
    Wrap angle to the interval [-pi, pi).
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def cartesian_to_radar(px, py, radar_pos=(0.0, 0.0)):
    """
    Convert Cartesian position to radar range-bearing measurement.

    Args:
        px: Target x position [m]
        py: Target y position [m]
        radar_pos: Radar position (sx, sy) [m]

    Returns:
        r: Range [m]
        b: Bearing [rad]
    """
    sx, sy = radar_pos
    dx = px - sx
    dy = py - sy

    r = np.sqrt(dx**2 + dy**2)
    b = np.arctan2(dy, dx)

    return r, b


def radar_to_cartesian(r, b, radar_pos=(0.0, 0.0)):
    """
    Convert radar range-bearing measurement to Cartesian position.

    Args:
        r: Range [m]
        b: Bearing [rad]
        radar_pos: Radar position (sx, sy) [m]

    Returns:
        px, py: Cartesian position [m]
    """
    sx, sy = radar_pos
    px = sx + r * np.cos(b)
    py = sy + r * np.sin(b)
    return px, py


def measure_radar(truth_states, sigma_r, sigma_b, radar_pos=(0.0, 0.0)):
    """
    Generate noisy radar measurements from true target states.

    Measurement model:
        z = [range, bearing] + noise

    Args:
        truth_states: True state history with shape (N, 4)
        sigma_r: Range noise standard deviation [m]
        sigma_b: Bearing noise standard deviation [rad]
        radar_pos: Radar position (sx, sy) [m]

    Returns:
        z: Noisy radar measurements with shape (N, 2)
           z[:, 0] = range
           z[:, 1] = bearing
    """
    truth_states = np.asarray(truth_states, dtype=float)

    if truth_states.ndim != 2 or truth_states.shape[1] != 4:
        raise ValueError("truth_states must have shape (N, 4).")
    if sigma_r < 0:
        raise ValueError("sigma_r must be non-negative.")
    if sigma_b < 0:
        raise ValueError("sigma_b must be non-negative.")

    n = len(truth_states)
    z = np.zeros((n, 2), dtype=float)

    for k in range(n):
        px, py = truth_states[k, 0], truth_states[k, 1]
        r_true, b_true = cartesian_to_radar(px, py, radar_pos=radar_pos)

        r_meas = r_true + np.random.randn() * sigma_r
        b_meas = wrap_angle(b_true + np.random.randn() * sigma_b)

        z[k] = np.array([r_meas, b_meas], dtype=float)

    return z