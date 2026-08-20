from pathlib import Path

import numpy as np
import pytest

from jumpbot.cv.pose import _PersonTracker, _select_person, extract_pose


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
