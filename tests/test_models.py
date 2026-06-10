import numpy as np
import pytest

from src.block0_modeling.simulate_cv_target import measure_position, simulate_target_cv
from src.block2_process_noise.simulate_maneuver_target import (
    acceleration_profile,
    simulate_target_maneuvering,
)


def test_constant_velocity_model_propagates_expected_state():
    truth, transition, t = simulate_target_cv(
        x0=np.array([0.0, 10.0, 3.0, -2.0]),
        dt=0.5,
        steps=4,
    )

    assert truth.shape == (5, 4)
    assert transition.shape == (4, 4)
    np.testing.assert_allclose(t, [0.0, 0.5, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(truth[-1], [6.0, 6.0, 3.0, -2.0])


def test_measure_position_is_exact_when_noise_is_zero():
    truth = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

    measurements = measure_position(truth, sigma_pos=0.0)

    np.testing.assert_allclose(measurements, truth[:, :2])


def test_model_validation_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="dt must be positive"):
        simulate_target_cv(np.zeros(4), dt=0.0, steps=1)

    with pytest.raises(ValueError, match="x0 must have shape"):
        simulate_target_maneuvering(np.zeros(3), dt=0.1, steps=1)


def test_maneuvering_target_applies_piecewise_acceleration():
    assert np.array_equal(acceleration_profile(0.5), np.array([0.0, 0.0]))
    assert np.array_equal(acceleration_profile(2.0), np.array([0.0, 8.0]))
    assert np.array_equal(acceleration_profile(3.5), np.array([-6.0, 0.0]))

    truth, t, acc_hist = simulate_target_maneuvering(
        x0=np.array([0.0, 0.0, 10.0, 0.0]),
        dt=0.5,
        steps=4,
    )

    assert truth.shape == (5, 4)
    assert acc_hist.shape == (5, 2)
    np.testing.assert_allclose(t, [0.0, 0.5, 1.0, 1.5, 2.0])
