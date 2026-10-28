# Radar Target Tracking with KF and EKF

[![Tests](https://github.com/hunkarsuci/missile-tracking-gnc/actions/workflows/tests.yml/badge.svg)](https://github.com/hunkarsuci/missile-tracking-gnc/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Python study of target-motion models and state estimation from noisy Cartesian and radar measurements. It progresses from a two-dimensional constant-velocity Kalman filter to nonlinear radar EKFs in two and three dimensions. The scenarios use synthetic measurements; this repository does not implement an interceptor, guidance law, or autopilot.

![Synthetic 3D radar tracking scenario](assets/tracking_3d.gif)

## What is implemented

| Block | Model and measurement | What to inspect |
| --- | --- | --- |
| 0 | 2D constant-velocity truth with noisy Cartesian positions | Measurement scatter relative to truth |
| 1 | Linear KF estimating position and velocity | Prediction, correction, covariance, and RMSE |
| 2 | Piecewise target acceleration with varied process noise | Lag versus noise sensitivity during maneuvers |
| 3 | 2D range/bearing radar EKF | Analytical Jacobian, wrapped bearing innovation, Joseph covariance update |
| 3D demo | Six-state radar EKF with range, azimuth, and elevation | Synthetic climb and lateral maneuvers, initialization from early measurements |
| 4 | Matplotlib 3D animation | Truth, measurements, and estimate from the same scenario |

The 3D filter uses `[px, py, pz, vx, vy, vz]`. The animation's truth path is a prescribed synthetic scenario: a climb to 10 km, descent to 8 km, and recovery to 9 km alongside lateral maneuvers. It is a tracking illustration, not a flight-dynamics model. See [the system diagram](assets/system_architecture.svg) and the implementations in [`src/`](src/).

## Reproduce

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pytest -q
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

Run the estimator examples individually:

```bash
python src/block0_modeling/simulate_cv_target.py
python src/block1_kf/run_kf_cv_demo.py
python src/block2_process_noise/run_kf_process_noise_demo.py
python src/block3_ekf/run_ekf_radar_demo.py
```

Run the 3D scenario interactively, or save its animation without opening a window:

```bash
python src/block4_visualization/animate_3d_tracking.py
python src/block4_visualization/animate_3d_tracking.py --save assets/tracking_3d.gif --no-show
```

The fixed-seed 3D scenario is assembled in [`build_tracking_scenario`](src/block4_visualization/animate_3d_tracking.py). The early radar measurements initialize position and velocity; the EKF starts updating after that window. The 2D demos print tracking errors and produce plots. The [GitHub Actions workflow](.github/workflows/tests.yml) runs the test suite.

## Model and verification boundaries

The filters use a constant-velocity transition model and acceleration process noise. The 2D radar update wraps bearing residuals and uses a Joseph-form covariance update. The 3D radar update wraps angular residuals and uses the same covariance form. The [tests](tests/) exercise motion models, measurement geometry, filter behavior, and visualization smoke paths.

The results depend on synthetic noise, assumed covariance, initial state, and maneuver profile. The repository does not establish estimator consistency across Monte Carlo trials or validate against flight/radar recordings. The 3D Jacobian regularizes near-zero horizontal range; behavior close to the radar axis needs separate evaluation. There is no closed-loop guidance, actuator model, real-time C++ implementation, or hardware validation.

## Next engineering steps

- Add seeded multi-run RMSE and NIS/NEES studies with stated acceptance criteria.
- Compare the analytical 3D measurement Jacobian against finite differences away from singular geometry.
- Add sensor dropout and outlier cases with explicit gating behavior.
- Keep any future guidance simulation separate from the estimator benchmark.

MIT licensed; see [LICENSE](LICENSE). This is an educational estimation project and is not suitable for operational targeting or safety-critical use.
