from pathlib import Path

import numpy as np
import pytest

from jumpbot.cv.pose import _select_person, extract_pose


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
