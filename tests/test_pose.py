from pathlib import Path

import numpy as np
import pytest

from jumpbot.cv.pose import (
    _CameraStabilizer,
    _infer_tracked_pose,
    _needs_full_frame_reacquire,
    _PersonTracker,
    _select_person,
    _tracking_roi,
    _transform_points,
    extract_pose,
)


def test_selects_person_with_best_mean_confidence() -> None:
    keypoints = np.zeros((2, 17, 2), dtype=float)
    keypoints[1, :, 0] = 42.0
    scores = np.vstack((np.full(17, 0.3), np.full(17, 0.9)))

    selected = _select_person(keypoints, scores)

    assert selected is not None
    assert np.all(selected[0][:, 0] == 42.0)
    assert np.all(selected[1] == 0.9)


def test_empty_pose_result_returns_none() -> None:
    assert _select_person(np.array([]), np.array([])) is None


def test_rejects_unknown_pose_backend() -> None:
    with pytest.raises(ValueError, match="Unknown pose backend"):
        extract_pose(Path("unused.mp4"), "unknown")


def test_tracker_keeps_same_athlete_when_bystander_has_higher_confidence() -> None:
    tracker = _PersonTracker()
    first = np.zeros((2, 26, 2), dtype=float)
    first[0, :, 0] = np.linspace(100, 130, 26)
    first[0, :, 1] = np.linspace(100, 300, 26)
    first[1, :, 0] = np.linspace(500, 530, 26)
    first[1, :, 1] = np.linspace(100, 300, 26)
    scores = np.vstack((np.full(26, 0.9), np.full(26, 0.8)))
    assert tracker.select(first, scores) is not None

    second = first.copy()
    second[0, :, 0] += 8
    second[1, :, 0] = np.linspace(125, 155, 26)
    second_scores = np.vstack((np.full(26, 0.75), np.full(26, 0.99)))

    selected = tracker.select(second, second_scores)

    assert selected is not None
    assert np.median(selected[0][:, 0]) < 130


def test_tracking_roi_follows_visible_athlete_and_stays_inside_frame() -> None:
    points = np.zeros((26, 2), dtype=float)
    points[:, 0] = np.linspace(10, 90, 26)
    points[:, 1] = np.linspace(20, 460, 26)
    scores = np.full(26, 0.9)

    roi = _tracking_roi((480, 640, 3), points, scores)

    assert roi is not None
    x1, y1, x2, y2 = roi
    assert x1 == 0
    assert y1 == 0
    assert 90 < x2 <= 640
    assert y2 == 480


def test_tracking_roi_requires_enough_visible_keypoints() -> None:
    points = np.ones((26, 2), dtype=float) * 100
    scores = np.zeros(26)
    scores[:5] = 0.9

    assert _tracking_roi((480, 640, 3), points, scores) is None


def test_transforms_pose_points_with_camera_affine_matrix() -> None:
    points = np.array([[100.0, 50.0], [25.0, 80.0]])
    transform = np.array([[1.0, 0.0, -12.0], [0.0, 1.0, 7.0]])

    transformed = _transform_points(points, transform)

    assert np.allclose(transformed, [[88.0, 57.0], [13.0, 87.0]])


def test_camera_stabilizer_compensates_background_translation() -> None:
    import cv2

    first = np.zeros((300, 400, 3), dtype=np.uint8)
    for x in range(20, 390, 40):
        for y in range(20, 290, 40):
            cv2.circle(first, (x, y), 3, (255, 255, 255), -1)
    second = cv2.warpAffine(
        first, np.array([[1.0, 0.0, 14.0], [0.0, 1.0, -6.0]]), (400, 300)
    )
    stabilizer = _CameraStabilizer()

    stabilizer.update(first, None)
    transform = stabilizer.update(second, None)

    assert transform[0, 2] == pytest.approx(-14.0, abs=1.0)
    assert transform[1, 2] == pytest.approx(6.0, abs=1.0)


def test_tracked_pose_skips_detector_and_restores_full_frame_coordinates() -> None:
    class PoseModel:
        def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            assert image.shape[:2] == (100, 80)
            return np.array([[[10.0, 20.0]]]), np.array([[0.9]])

    class Estimator:
        pose_model = PoseModel()

        def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            raise AssertionError("Full-frame detector must not run for a tracked crop")

    keypoints, scores = _infer_tracked_pose(
        Estimator(), np.zeros((300, 400, 3), dtype=np.uint8), (50, 70, 130, 170)
    )

    assert np.allclose(keypoints, [[[60.0, 90.0]]])
    assert np.allclose(scores, [[0.9]])


def test_reacquires_full_frame_when_cropped_torso_confidence_drops() -> None:
    points = np.full((1, 26, 2), 150.0)
    scores = np.full((1, 26), 0.9)
    scores[:, [5, 6, 11, 12]] = 0.2

    assert _needs_full_frame_reacquire(points, scores, (50, 50, 250, 350))


def test_keeps_crop_when_torso_is_confident_and_centered() -> None:
    points = np.full((1, 26, 2), 150.0)
    points[:, :, 1] = 200.0
    scores = np.full((1, 26), 0.9)

    assert not _needs_full_frame_reacquire(points, scores, (50, 50, 250, 350))


def test_reacquires_full_frame_when_torso_reaches_crop_edge() -> None:
    points = np.full((1, 26, 2), 150.0)
    points[:, :, 1] = 200.0
    points[:, 11, 1] = 342.0
    scores = np.full((1, 26), 0.9)

    assert _needs_full_frame_reacquire(points, scores, (50, 50, 250, 350))
