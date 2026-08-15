import hashlib
import re
from pathlib import Path
from uuid import UUID

SAFE_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


class LocalStorage:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def video_path(self, user_id: int, jump_id: UUID, original_name: str) -> Path:
        suffix = Path(original_name).suffix.lower()
        if suffix not in SAFE_SUFFIXES:
            raise ValueError("Unsupported video format")
        safe_user = re.sub(r"[^0-9]", "", str(user_id))
        target_dir = (self.root / safe_user).resolve()
        if self.root not in target_dir.parents:
            raise ValueError("Unsafe storage path")
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / f"{jump_id}{suffix}"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
