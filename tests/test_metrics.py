import numpy as np
import pytest

from jumpbot.cv.metrics import (
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
