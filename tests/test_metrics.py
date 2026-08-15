import numpy as np
import pytest

from jumpbot.cv.metrics import flight_height, scale_from_height, vertical_velocity


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
