import numpy as np


def acceleration_profile(t):
    """
    Define a piecewise-constant acceleration profile.

    Args:
        t: Time [s]

    Returns:
        a: Acceleration vector [ax, ay] in m/s^2
    """
    if t < 1.5:
        return np.array([0.0, 0.0], dtype=float)
    elif t < 3.0:
        return np.array([0.0, 8.0], dtype=float)
    elif t < 4.0:
        return np.array([-6.0, 0.0], dtype=float)
    else:
        return np.array([0.0, 0.0], dtype=float)


def simulate_target_maneuvering(x0, dt, steps):
    """
    Simulate a 2D target with piecewise-constant acceleration.

    State definition:
        x = [px, py, vx, vy]

    Dynamics:
        x[k+1] = F x[k] + B a[k]

    Args:
        x0: Initial state vector with shape (4,)
        dt: Sampling time [s]
        steps: Number of propagation steps

    Returns:
        truth: True state history with shape (steps + 1, 4)
        t: Time vector with shape (steps + 1,)
        acc_hist: Applied acceleration history with shape (steps + 1, 2)
    """
    x0 = np.asarray(x0, dtype=float)

    if x0.shape != (4,):
        raise ValueError("x0 must have shape (4,).")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    F = np.array([
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=float)

    B = np.array([
        [0.5 * dt**2, 0.0],
        [0.0, 0.5 * dt**2],
        [dt, 0.0],
        [0.0, dt]
    ], dtype=float)

    x = x0.copy()
    truth = [x.copy()]
    acc_hist = [acceleration_profile(0.0)]
    t = np.arange(steps + 1) * dt

    for k in range(steps):
        a_k = acceleration_profile(t[k])
        x = F @ x + B @ a_k
        truth.append(x.copy())
        acc_hist.append(a_k.copy())

    return np.array(truth), t, np.array(acc_hist)