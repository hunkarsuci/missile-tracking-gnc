import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from src.block4_visualization.animate_3d_tracking import (
    build_tracking_scenario,
    create_animation,
)


def test_tracking_scenario_has_consistent_3d_shapes():
    scenario = build_tracking_scenario(seed=11)

    assert scenario["truth"].shape[1] == 3
    assert scenario["measurements"].shape == scenario["truth"].shape
    assert scenario["estimate"].shape == scenario["truth"].shape
    assert scenario["radar"].shape == (3,)
    assert np.isfinite(scenario["truth"]).all()
    assert np.isfinite(scenario["measurements"]).all()
    assert np.isnan(scenario["estimate"][:10, 0]).any()
    assert np.isfinite(scenario["estimate"][-1]).all()


def test_tracking_scenario_has_visible_lateral_and_altitude_motion():
    scenario = build_tracking_scenario(seed=11)
    t = scenario["t"]
    truth = scenario["truth"]

    x_span = np.ptp(truth[:, 0])
    y_span = np.ptp(truth[:, 1])
    z_span = np.ptp(truth[:, 2])
    drop_idx = np.argmin(np.abs(t - 11.0))
    final_climb_idx = np.argmin(np.abs(t - 13.0))
    horizontal_start = truth[final_climb_idx, :2]
    horizontal_end = truth[-1, :2]
    horizontal_extension = np.linalg.norm(horizontal_end - horizontal_start)

    assert y_span > 0.35 * x_span
    assert abs(truth[0, 2]) < 1e-9
    assert np.max(truth[:, 2]) == pytest.approx(10000.0, abs=30.0)
    assert truth[drop_idx, 2] == pytest.approx(8000.0, abs=30.0)
    assert truth[final_climb_idx, 2] == pytest.approx(9000.0, abs=30.0)
    assert truth[-1, 2] == pytest.approx(9000.0, abs=30.0)
    assert z_span > 9900.0
    assert horizontal_extension == pytest.approx(1000.0, abs=180.0)


def test_tracking_scenario_climb_is_curved_not_linear():
    scenario = build_tracking_scenario(seed=11)
    truth = scenario["truth"]
    midpoint = len(truth) // 2
    linear_midpoint_altitude = 0.5 * (truth[0, 2] + truth[-1, 2])

    assert truth[midpoint, 2] > linear_midpoint_altitude + 750.0


def test_3d_ekf_tracks_final_curve_closely():
    scenario = build_tracking_scenario(seed=11)
    truth = scenario["truth"]
    estimate = scenario["estimate"]
    valid = ~np.isnan(estimate).any(axis=1)
    error = np.linalg.norm(estimate[valid] - truth[valid], axis=1)
    final_curve_error = error[int(len(error) * 0.75):]

    assert final_curve_error.mean() < 90.0
    assert error[-1] < 120.0


def test_animation_smoke_test_without_display():
    assert create_animation(show=False) is None
