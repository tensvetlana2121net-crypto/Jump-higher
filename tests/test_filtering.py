import numpy as np

from jumpbot.cv.filtering import hampel_filter, interpolate_short_gaps, smooth


def test_interpolates_short_gap() -> None:
    result = interpolate_short_gaps(np.array([0.0, np.nan, 2.0]), max_gap=1)
    assert np.allclose(result, [0, 1, 2])


def test_leaves_long_gap_for_rejection() -> None:
    result = interpolate_short_gaps(np.array([0.0, np.nan, np.nan, 3.0]), max_gap=1)
    assert np.isnan(result[1:3]).all()


def test_hampel_removes_spike() -> None:
    result = hampel_filter(np.array([1.0, 1.1, 1.0, 10.0, 1.0, 0.9, 1.0]))
    assert result[3] < 2


def test_smooth_keeps_length() -> None:
    assert len(smooth(np.arange(10, dtype=float), window=5)) == 10
