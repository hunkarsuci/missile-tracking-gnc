import numpy as np
import matplotlib.pyplot as plt


def simulate_target_cv(x0, dt, steps):
    """
    Simulate a 2D constant-velocity target motion model.

    State definition:
        x = [px, py, vx, vy]

    Args:
        x0: Initial state vector with shape (4,)
        dt: Sampling time in seconds
        steps: Number of propagation steps

    Returns:
        truth: Array of true states with shape (steps + 1, 4)
               Includes the initial state at index 0.
        F: State transition matrix with shape (4, 4)
        t: Time vector with shape (steps + 1,)
    """
    x0 = np.asarray(x0, dtype=float)

    if x0.shape != (4,):
        raise ValueError("x0 must be a 1D array with shape (4,).")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    # Constant-velocity state transition matrix
    F = np.array([
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=float)

    x = x0.copy()
    truth = [x.copy()]

    for _ in range(steps):
        # Propagate the state one step forward
        x = F @ x
        truth.append(x.copy())

    truth = np.array(truth)
    t = np.arange(steps + 1) * dt

    return truth, F, t


def measure_position(truth_states, sigma_pos):
    """
    Generate noisy position measurements from the true states.

    Measurement model:
        z = [px, py] + noise

    Args:
        truth_states: True state history with shape (N, 4)
        sigma_pos: Standard deviation of position measurement noise in meters

    Returns:
        z: Noisy position measurements with shape (N, 2)
    """
    truth_states = np.asarray(truth_states, dtype=float)

    if truth_states.ndim != 2 or truth_states.shape[1] != 4:
        raise ValueError("truth_states must have shape (N, 4).")
    if sigma_pos < 0:
        raise ValueError("sigma_pos must be non-negative.")

    # Generate zero-mean Gaussian noise on position channels
    noise = np.random.randn(len(truth_states), 2) * sigma_pos

    # Extract [px, py] and add measurement noise
    z = truth_states[:, :2] + noise

    return z


def plot_results(t, truth, z):
    """
    Plot the true trajectory and noisy measurements.

    Args:
        t: Time vector with shape (N,)
        truth: True state history with shape (N, 4)
        z: Noisy position measurements with shape (N, 2)
    """
    # 2D trajectory plot
    plt.figure(figsize=(8, 6))
    plt.plot(truth[:, 0], truth[:, 1], label="True trajectory", linewidth=2)
    plt.scatter(z[:, 0], z[:, 1], label="Noisy measurements", s=18, alpha=0.7)
    plt.xlabel("x position [m]")
    plt.ylabel("y position [m]")
    plt.title("2D Target Motion: Truth vs Measurements")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    # Position components versus time
    plt.figure(figsize=(10, 6))
    plt.plot(t, truth[:, 0], label="True x", linewidth=2)
    plt.plot(t, truth[:, 1], label="True y", linewidth=2)
    plt.scatter(t, z[:, 0], label="Measured x", s=14, alpha=0.7)
    plt.scatter(t, z[:, 1], label="Measured y", s=14, alpha=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("Position Components vs Time")
    plt.legend()
    plt.grid(True)

    plt.show()


def main():
    # Fix the random seed for reproducibility
    np.random.seed(0)

    dt = 0.1
    steps = 50
    sigma_pos = 20.0

    # Initial state: [px, py, vx, vy]
    x0 = np.array([0.0, 0.0, 300.0, 40.0], dtype=float)

    truth, F, t = simulate_target_cv(x0, dt, steps)
    z = measure_position(truth, sigma_pos=sigma_pos)

    print("State transition matrix F:")
    print(F)
    print()

    print("truth shape:", truth.shape)
    print("measurement shape:", z.shape)
    print("time shape:", t.shape)
    print()

    for k in range(5):
        print(f"t[{k}]     = {t[k]:.2f} s")
        print(f"truth[{k}] = {truth[k]}")
        print(f"meas[{k}]  = {z[k]}")
        print()

    plot_results(t, truth, z)


if __name__ == "__main__":
    main()