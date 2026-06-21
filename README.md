# Target Tracking and GNC Simulation with KF/EKF

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C)](https://matplotlib.org/)
[![Tests](https://github.com/HunkarSuci/missile-tracking-gnc/actions/workflows/tests.yml/badge.svg)](https://github.com/HunkarSuci/missile-tracking-gnc/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Project Status](https://img.shields.io/badge/Status-In%20Development-yellow)](#roadmap)

End-to-end missile target tracking and guidance simulations using motion models, noisy sensor measurements, Kalman filtering, 2D/3D radar EKF estimation, and professional visualization.
=======
A Python-based target tracking and estimation project focused on **Guidance, Navigation, and Control (GNC)** fundamentals, Kalman filtering, radar measurement modeling, and maneuvering target simulation.

The repository is organized as an incremental engineering study. Each block introduces one additional concept, starting from simple target motion modeling and progressing toward nonlinear radar-based tracking with an Extended Kalman Filter (EKF).

> This project is intended for educational, research, and portfolio purposes. It is not an operational weapon system, real-world guidance implementation, or safety-critical control system.

The repository is organized as an incremental engineering build: start from physical system modeling, add estimators and maneuver models, then move toward closed-loop guidance and real-time implementation.

## Demo

![3D radar-EKF tracking animation](assets/tracking_3d.gif)

## System Architecture

![Missile tracking and guidance system architecture](assets/system_architecture.svg)
=======
## Overview

This project demonstrates the core estimation pipeline used in many aerospace, robotics, radar, and autonomous-systems applications:

1. Generate target ground truth
2. Simulate noisy sensor measurements
3. Estimate target state using Kalman filtering
4. Study the effect of process-noise tuning during maneuvers
5. Track a target using nonlinear radar range-bearing measurements with an EKF

The implemented simulations focus on 2D target tracking with position, velocity, and radar measurements.

---


## Implemented Blocks

### Block 0 - Constant-Velocity Target Model

Implements a 2D constant-velocity target motion model.

```text
State: x = [px, py, vx, vy]^T
```

where:

- `px`, `py`: target position components
- `vx`, `vy`: target velocity components

The block also generates noisy Cartesian position measurements:

```text
z = [px, py] + measurement_noise
```

This block establishes the ground-truth trajectory and basic sensor model used by later simulations.

---

### Block 1 - Linear Kalman Filter for CV Tracking

Implements a discrete-time linear Kalman Filter for a 2D constant-velocity model.

```text
State:       x = [px, py, vx, vy]^T
Measurement: z = [px, py]^T
```

The filter includes:

- State prediction using a constant-velocity transition matrix
- Position-only measurement update
- Discrete white-acceleration process-noise model
- Measurement-noise covariance
- Joseph-form covariance update for numerical stability
- RMSE calculation for tracking-performance evaluation

This block demonstrates how noisy position measurements can be fused over time to estimate both position and velocity.

---

### Block 2 - Maneuvering Target and Process-Noise Tuning

Extends the target model with piecewise-constant acceleration maneuvers.

```text
x[k+1] = F x[k] + B a[k]
```

where:

- `F`: constant-velocity state-transition matrix
- `B`: acceleration input matrix
- `a[k]`: applied target acceleration

The same constant-velocity Kalman Filter is tested with different process-noise values.

This block shows the tradeoff between:

- Low process noise: smoother estimates but slower response to maneuvers
- High process noise: faster maneuver response but noisier estimates

---

### Block 3 - Radar Extended Kalman Filter

Implements an Extended Kalman Filter for nonlinear radar measurements.

```text
State:       x = [px, py, vx, vy]^T
Measurement: z = [range, bearing]^T
```

The nonlinear radar measurement model is:

```text
range   = sqrt((px - sx)^2 + (py - sy)^2)
bearing = atan2(py - sy, px - sx)
```

where `(sx, sy)` is the radar position.

The EKF implementation includes:

- Nonlinear range-bearing measurement function
- Analytical measurement Jacobian
- Angle wrapping for bearing residuals
- Prediction and correction steps
- Joseph-form covariance update
- Radar-to-Cartesian conversion for visualization
- RMSE evaluation for position and velocity estimates

This block demonstrates nonlinear sensor fusion using radar-style measurements.


## Features

- Constant-velocity target truth model and noisy position measurements
- Linear Kalman Filter for Cartesian position and velocity tracking
- Maneuvering target simulation with process noise
- Radar range-bearing measurement model
- Extended Kalman Filter for nonlinear radar tracking
- 3D radar EKF demo with range, azimuth, elevation, a 0-to-10 km climb, an 8 km dip, a 9 km recovery, and horizontal continuation
- Animated tracking visualization for GitHub demos and presentations

## Create an environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run the test suite:

```bash
pytest -q
```

Run the existing 2D demos:

```bash
python src/block0_modeling/simulate_cv_target.py
python src/block1_kf/run_kf_cv_demo.py
python src/block3_ekf/run_ekf_radar_demo.py
```

Run the 3D animation:

```bash
python src/block4_visualization/animate_3d_tracking.py
```

Save the 3D animation as a GIF:

```bash
python src/block4_visualization/animate_3d_tracking.py --save assets/tracking_3d.gif --no-show
```

## Continuous Integration

This repository uses GitHub Actions via `.github/workflows/tests.yml` to run the project test workflow. The test status badge at the top of this README links to the latest workflow runs.


## Repository Structure

```text
src/
  block0_modeling/
    simulate_cv_target.py             # Target truth model + noisy measurements
  block1_kf/
    linear_kf_cv.py                   # Linear Kalman Filter implementation
    run_kf_cv_demo.py                 # 2D KF tracking demo
  block2_process_noise/
    simulate_maneuver_target.py       # Maneuvering target model
    run_kf_process_noise_demo.py      # Process-noise tuning demo
  block3_ekf/
    extended_kf_radar.py              # Radar EKF implementation
    extended_kf_radar_3d.py           # 3D radar EKF implementation
    simulate_radar_measurements.py    # Range-bearing measurement model
    simulate_radar_measurements_3d.py # Range-azimuth-elevation model
    run_ekf_radar_demo.py             # 2D radar-EKF demo
  block4_visualization/
    animate_3d_tracking.py            # 3D tracking animation
tests/
  test_models.py                       # Motion-model and measurement tests
  test_filters.py                      # KF, EKF, and radar geometry tests
  test_visualization.py                # 3D scenario and animation smoke tests
assets/
  system_architecture.svg              # Repository architecture diagram
  tracking_3d.gif                      # GitHub README animation
.github/workflows/
  tests.yml                            # Pytest CI workflow
```

## Roadmap

- Proportional Navigation guidance
- Missile dynamics and autopilot effects
- Closed-loop interceptor-target engagement simulation
- Real-time C++ implementation
- Formatting and linting checks

## License

This project is licensed under the [MIT License](LICENSE).


````md
## Project Structure

```text
missile-tracking-gnc/
│
├── src/
│   ├── block0_modeling/
│   │   └── simulate_cv_target.py
│   │
│   ├── block1_kf/
│   │   ├── linear_kf_cv.py
│   │   └── run_kf_cv_demo.py
│   │
│   ├── block2_process_noise/
│   │   ├── simulate_maneuver_target.py
│   │   └── run_kf_process_noise_demo.py
│   │
│   └── block3_ekf/
│       ├── extended_kf_radar.py
│       ├── simulate_radar_measurements.py
│       └── run_ekf_radar_demo.py
│
├── README.md
├── LICENSE
└── .gitignore
```

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/hunkarsuci/missile-tracking-gnc.git
cd missile-tracking-gnc
```

Install dependencies:

```bash
pip install numpy matplotlib
```

Run the individual simulation blocks:

```bash
python src/block0_modeling/simulate_cv_target.py
python src/block1_kf/run_kf_cv_demo.py
python src/block2_process_noise/run_kf_process_noise_demo.py
python src/block3_ekf/run_ekf_radar_demo.py
```

---

## Simulation Outputs

The demo scripts generate time-domain and 2D trajectory plots, including:

- True target trajectory
- Noisy Cartesian measurements
- Kalman Filter position and velocity estimates
- Maneuvering target response under different process-noise assumptions
- Radar range and bearing measurements
- EKF-estimated trajectory
- Position and velocity tracking performance
- Applied acceleration profile

The scripts also print RMSE values for position and velocity estimation performance.

---

## Technical Concepts Demonstrated

This project demonstrates practical knowledge of:

- Target motion modeling
- Discrete-time state-space systems
- Kalman filtering
- Extended Kalman filtering
- Sensor fusion
- Radar range-bearing measurements
- Process-noise modeling
- Maneuvering-target tracking
- Numerical simulation
- RMSE-based performance evaluation
- Python scientific computing
- Modular simulation design

---

## Current Scope

Implemented:

- 2D constant-velocity target simulation
- Noisy Cartesian position measurements
- Linear Kalman Filter for position/velocity estimation
- Maneuvering target model with piecewise acceleration
- Process-noise sensitivity study
- Radar range-bearing measurement simulation
- Extended Kalman Filter for nonlinear radar tracking
- 2D and time-domain visualization

Not currently implemented:

- Closed-loop missile-target engagement simulation
- Proportional Navigation guidance law
- Missile autopilot or actuator dynamics
- 3D engagement geometry
- Real-time C++ implementation
- Hardware-in-the-loop or safety-critical deployment

---

## Roadmap

Possible future extensions:

- Add Proportional Navigation guidance as a separate educational simulation block
- Add simple interceptor point-mass dynamics
- Extend the target and interceptor models to 3D
- Compare KF, EKF, and UKF tracking performance
- Add Monte Carlo analysis for estimator consistency
- Save generated plots under a `figures/` directory
- Add a `requirements.txt` file
- Add unit tests for model and filter components
- Add a C++ implementation of selected estimation blocks

---

## Design Notes

The project is intentionally block-based. This makes each concept easier to inspect, test, and extend:

- Block 0 creates the basic truth and measurement model.
- Block 1 introduces the linear Kalman Filter.
- Block 2 studies how process noise affects tracking during maneuvers.
- Block 3 introduces nonlinear radar measurements and EKF estimation.

This structure makes the repository useful both as a learning project and as a portfolio demonstration for aerospace, robotics, estimation, and GNC-related roles.

---

## Dependencies

The project uses:

- Python
- NumPy
- Matplotlib

---

## License

This project is released under the MIT License.

---

## Disclaimer

This repository is for educational and portfolio purposes only.

It does not provide an operational missile guidance system and must not be used for real-world weapon development, targeting, safety-critical control, or deployment.
