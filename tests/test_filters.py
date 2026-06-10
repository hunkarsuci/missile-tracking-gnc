import numpy as np
import pytest

from src.block1_kf.linear_kf_cv import LinearKFCV2D
from src.block3_ekf.extended_kf_radar import ExtendedKFRadar2D, wrap_angle
from src.block3_ekf.extended_kf_radar_3d import ExtendedKFRadar3D
from src.block3_ekf.simulate_radar_measurements import (
    cartesian_to_radar,
    radar_to_cartesian,
)
from src.block3_ekf.simulate_radar_measurements_3d import (
    cartesian_to_radar_3d,
    radar_to_cartesian_3d,
)


def test_linear_kf_predicts_constant_velocity_state():
    kf = LinearKFCV2D(
        dt=0.5,
        sigma_a=0.0,
        sigma_pos=1.0,
        x0=np.array([0.0, 2.0, 4.0, -2.0]),
        P0=np.eye(4),
    )

    x_pred, p_pred = kf.predict()

    np.testing.assert_allclose(x_pred.ravel(), [2.0, 1.0, 4.0, -2.0])
    assert p_pred.shape == (4, 4)
    np.testing.assert_allclose(p_pred, p_pred.T)


def test_linear_kf_update_moves_toward_measurement():
    kf = LinearKFCV2D(
        dt=1.0,
        sigma_a=0.0,
        sigma_pos=1.0,
        x0=np.zeros(4),
        P0=np.diag([100.0, 100.0, 10.0, 10.0]),
    )

    before_distance = np.linalg.norm(kf.x[:2].ravel() - np.array([10.0, -5.0]))
    x_upd, p_upd = kf.update(np.array([10.0, -5.0]))
    after_distance = np.linalg.norm(x_upd[:2].ravel() - np.array([10.0, -5.0]))

    assert after_distance < before_distance
    np.testing.assert_allclose(p_upd, p_upd.T, atol=1e-12)
    assert np.all(np.diag(p_upd) >= 0.0)


def test_radar_cartesian_round_trip():
    r, bearing = cartesian_to_radar(3.0, 4.0, radar_pos=(1.0, 1.0))
    px, py = radar_to_cartesian(r, bearing, radar_pos=(1.0, 1.0))

    np.testing.assert_allclose([px, py], [3.0, 4.0])


def test_radar_cartesian_3d_round_trip():
    measurement = cartesian_to_radar_3d(3.0, 4.0, 12.0, radar_pos=(1.0, 1.0, 2.0))
    position = radar_to_cartesian_3d(*measurement, radar_pos=(1.0, 1.0, 2.0))

    np.testing.assert_allclose(position, [3.0, 4.0, 12.0])


def test_angle_wrapping_range():
    wrapped = wrap_angle(3.0 * np.pi)

    assert -np.pi <= wrapped < np.pi
    np.testing.assert_allclose(wrapped, -np.pi)


def test_radar_ekf_measurement_model_and_jacobian_shapes():
    ekf = ExtendedKFRadar2D(
        dt=0.1,
        sigma_a=1.0,
        sigma_r=5.0,
        sigma_b=np.deg2rad(1.0),
        radar_pos=(0.0, 0.0),
        x0=np.array([3.0, 4.0, 1.0, 0.0]),
        P0=np.eye(4),
    )

    z_pred = ekf.h(ekf.x)
    jacobian = ekf.jacobian_h(ekf.x)

    np.testing.assert_allclose(z_pred.ravel(), [5.0, np.arctan2(4.0, 3.0)])
    assert jacobian.shape == (2, 4)


def test_radar_ekf_3d_measurement_model_and_jacobian_shapes():
    ekf = ExtendedKFRadar3D(
        dt=0.1,
        sigma_a=1.0,
        sigma_r=5.0,
        sigma_az=np.deg2rad(1.0),
        sigma_el=np.deg2rad(1.0),
        radar_pos=(0.0, 0.0, 0.0),
        x0=np.array([3.0, 4.0, 12.0, 1.0, 0.0, 2.0]),
        P0=np.eye(6),
    )

    z_pred = ekf.h(ekf.x)
    jacobian = ekf.jacobian_h(ekf.x)

    np.testing.assert_allclose(z_pred[0, 0], 13.0)
    np.testing.assert_allclose(z_pred[1, 0], np.arctan2(4.0, 3.0))
    np.testing.assert_allclose(z_pred[2, 0], np.arctan2(12.0, 5.0))
    assert jacobian.shape == (3, 6)


def test_filter_validation_rejects_bad_covariance():
    with pytest.raises(ValueError, match="P0 must have shape"):
        LinearKFCV2D(dt=0.1, sigma_a=1.0, sigma_pos=1.0, P0=np.eye(3))
