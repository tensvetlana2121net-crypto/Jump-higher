import numpy as np
import pytest

from jumpbot.cv.metrics import (
    axial_rotation_metrics,
    ballistic_height,
    flight_height,
    rotation_metrics,
    scale_from_height,
    unwrap_angular_velocity,
    vertical_velocity,
)


def test_flight_height_for_half_second() -> None:
    assert flight_height(0.5) == pytest.approx(0.3064578125)


def test_vertical_velocity_linear_signal() -> None:
    signal = np.arange(6, dtype=float) * 0.1
    assert np.allclose(vertical_velocity(signal, fps=10), 1.0)


def test_scale_uses_median_height() -> None:
    assert scale_from_height(1.8, np.array([899.0, 900.0, 901.0])) == pytest.approx(0.002)


def test_invalid_flight_time() -> None:
    with pytest.raises(ValueError):
        flight_height(0)


def test_ballistic_height_recovers_parabolic_apex() -> None:
    fps = 60.0
    time = np.arange(31) / fps
    position = 2.5 * time - 0.5 * 9.80665 * time**2

    assert ballistic_height(position, fps) == pytest.approx(2.5**2 / (2 * 9.80665), rel=0.01)


def test_rotation_metrics_detect_full_turn() -> None:
    fps = 60.0
    orientation = np.linspace(0, 2 * np.pi, 61)
    velocity = unwrap_angular_velocity(orientation, fps)

    degrees, turns, direction, speed = rotation_metrics(orientation, velocity, 0, 60)

    assert degrees == pytest.approx(360)
    assert turns == pytest.approx(1)
    assert direction == "clockwise"
    assert speed == pytest.approx(360, rel=0.05)


def test_rotation_metrics_ignores_small_postural_change() -> None:
    orientation = np.deg2rad(np.linspace(0, 20, 20))
    velocity = unwrap_angular_velocity(orientation, fps=60)

    assert rotation_metrics(orientation, velocity, 0, 19) == (None, None, None, None)


def test_rotation_metrics_detects_rotation_when_landing_angle_matches_takeoff() -> None:
    fps = 60.0
    orientation = np.deg2rad(
        np.concatenate((np.linspace(0, 180, 31), np.linspace(180, 0, 31)[1:]))
    )
    velocity = unwrap_angular_velocity(orientation, fps)

    degrees, turns, direction, speed = rotation_metrics(orientation, velocity, 0, 60)

    assert degrees == pytest.approx(180)
    assert turns == pytest.approx(0.5)
    assert direction in {"clockwise", "counterclockwise"}
    assert speed is not None


def test_rotation_metrics_tolerates_short_direction_reversal() -> None:
    fps = 60.0
    degrees_signal = np.concatenate(
        (np.linspace(0, 190, 30), np.linspace(190, 170, 5)[1:], np.linspace(170, 360, 30)[1:])
    )
    orientation = np.deg2rad(degrees_signal)
    velocity = unwrap_angular_velocity(orientation, fps)

    degrees, turns, direction, _ = rotation_metrics(
        orientation, velocity, 0, len(orientation) - 1
    )

    assert degrees == pytest.approx(380, rel=0.02)
    assert turns == pytest.approx(380 / 360, rel=0.02)
    assert direction == "clockwise"


def test_axial_rotation_counts_two_turns_from_projected_width() -> None:
    fps = 60.0
    width = np.cos(np.linspace(0, 4 * np.pi, 61))

    degrees, turns, speed = axial_rotation_metrics(width, fps, 0, 60)

    assert degrees == pytest.approx(720, rel=0.08)
    assert turns == pytest.approx(2, rel=0.08)
    assert speed is not None
