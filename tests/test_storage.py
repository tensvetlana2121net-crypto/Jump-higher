import uuid

import pytest

from jumpbot.services.storage import LocalStorage


def test_builds_safe_video_path(tmp_path: object) -> None:
    storage = LocalStorage(tmp_path)  # type: ignore[arg-type]
    path = storage.video_path(42, uuid.uuid4(), "jump.mp4")
    assert path.parent.name == "42"
    assert path.suffix == ".mp4"


def test_rejects_unknown_extension(tmp_path: object) -> None:
    storage = LocalStorage(tmp_path)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        storage.video_path(42, uuid.uuid4(), "payload.exe")
