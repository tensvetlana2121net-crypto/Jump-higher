from jumpbot.cv.pipeline import _pose_visibility
from jumpbot.cv.types import FramePose, Landmark


def _landmark(visibility: float) -> Landmark:
    return Landmark(x_px=10.0, y_px=20.0, visibility=visibility)


def test_pose_visibility_is_driven_by_torso_not_hidden_extremities() -> None:
    points = {
        "head": _landmark(0.88),
        "left_shoulder": _landmark(0.91),
        "right_shoulder": _landmark(0.90),
        "left_hip": _landmark(0.89),
        "right_hip": _landmark(0.87),
        "left_knee": _landmark(0.84),
        "right_knee": _landmark(0.83),
        "left_ankle": _landmark(0.81),
        "right_ankle": _landmark(0.05),
        "left_foot": _landmark(0.01),
    }
    frames = [FramePose(frame=index, time_s=index / 30, points=points) for index in range(5)]

    assert _pose_visibility(frames) == 0.89


def test_pose_visibility_returns_zero_without_supported_landmarks() -> None:
    frame = FramePose(frame=0, time_s=0.0, points={"left_foot": _landmark(0.9)})

    assert _pose_visibility([frame]) == 0.0
