from pathlib import Path

import pytest

from jumpbot.cv import pipeline


def test_retries_rtmpose_without_roi_after_landing_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []
    expected = object()

    def fake_analysis(
        video_path: Path,
        athlete_height_m: float | None,
        pose_backend: str,
        tracking_roi_enabled: bool,
        camera_stabilization_enabled: bool,
    ) -> object:
        calls.append(tracking_roi_enabled)
        if tracking_roi_enabled:
            raise ValueError("Landing was not detected")
        return expected

    monkeypatch.setattr(pipeline, "_analyze_jump_once", fake_analysis)

    result = pipeline.analyze_jump(Path("fall.mp4"), 1.70, "rtmpose", True, True)

    assert result is expected
    assert calls == [True, False]


def test_does_not_retry_non_tracking_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_analysis(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise ValueError("Athlete height is required")

    monkeypatch.setattr(pipeline, "_analyze_jump_once", fake_analysis)

    with pytest.raises(ValueError, match="Athlete height is required"):
        pipeline.analyze_jump(Path("fall.mp4"), None, "rtmpose", True, True)

    assert calls == 1


def test_does_not_retry_when_roi_is_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_analysis(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise ValueError("Landing was not detected")

    monkeypatch.setattr(pipeline, "_analyze_jump_once", fake_analysis)

    with pytest.raises(ValueError, match="Landing was not detected"):
        pipeline.analyze_jump(Path("fall.mp4"), 1.70, "rtmpose", False, True)

    assert calls == 1
