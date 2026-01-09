# Missile Tracking and Guidance Algorithms

End-to-end missile target tracking and guidance simulation using Kalman filtering, EKF, maneuver models, and proportional navigation.

This repository is built step-by-step, starting from physical system modeling and progressing toward a full closed-loop missile guidance simulation via Python and real-time C++ implementation.

---

## Scope

The project covers:

- Target motion modeling (CV / CA)
- Noisy sensor measurements
- Linear Kalman Filter (KF)
- Extended Kalman Filter (EKF) for radar
- Maneuvering target tracking
- Proportional Navigation (PN) guidance
- Missile dynamics and autopilot effects
- 2D/3D visualization
- Real-time C++ implementation

Each block is implemented incrementally and committed separately.

---

## Repository Structure

```text
src/
  block0_modeling/
    simulate_cv_target.py    # Target truth model + noisy measurements
  block1_kf/                 # Linear Kalman Filter (to be added)
  block2_process_noise/      # Maneuver models (to be added)
  block3_ekf/                # Radar EKF (to be added)
