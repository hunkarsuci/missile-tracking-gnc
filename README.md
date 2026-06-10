# Missile Tracking and Guidance Algorithms

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C)](https://matplotlib.org/)
[![Tests](https://github.com/HunkarSuci/missile-tracking-gnc/actions/workflows/tests.yml/badge.svg)](https://github.com/HunkarSuci/missile-tracking-gnc/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Project Status](https://img.shields.io/badge/Status-In%20Development-yellow)](#roadmap)

End-to-end missile target tracking and guidance simulations using motion models, noisy sensor measurements, Kalman filtering, 2D/3D radar EKF estimation, and professional visualization.

The repository is organized as an incremental engineering build: start from physical system modeling, add estimators and maneuver models, then move toward closed-loop guidance and real-time implementation.

## Demo

![3D radar-EKF tracking animation](assets/tracking_3d.gif)

## System Architecture

![Missile tracking and guidance system architecture](assets/system_architecture.svg)

## Features

- Constant-velocity target truth model and noisy position measurements
- Linear Kalman Filter for Cartesian position and velocity tracking
- Maneuvering target simulation with process noise
- Radar range-bearing measurement model
- Extended Kalman Filter for nonlinear radar tracking
- 3D radar EKF demo with range, azimuth, elevation, a 0-to-10 km climb, an 8 km dip, a 9 km recovery, and horizontal continuation
- Animated tracking visualization for GitHub demos and presentations

## Quick Start

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
