import pytest

from jumpbot.cv.pose import inference_stride


@pytest.mark.parametrize(
    ("fps", "expected"),
    [(24.0, 1), (30.0, 1), (50.0, 2), (59.94, 2), (60.0, 2), (120.0, 4)],
)
def test_inference_stride_targets_about_30_fps(fps: float, expected: int) -> None:
    assert inference_stride(fps) == expected


def test_inference_stride_rejects_invalid_fps() -> None:
    with pytest.raises(ValueError):
        inference_stride(0)
