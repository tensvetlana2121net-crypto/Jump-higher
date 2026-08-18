import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoMetadata:
    duration_s: float
    format_name: str
    video_codec: str


def validate_video_file(path: Path, max_size_bytes: int, max_duration_s: int) -> VideoMetadata:
    size = path.stat().st_size
    if size <= 0 or size > max_size_bytes:
        raise ValueError("Некорректный размер видео")

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,format_name:stream=codec_type,codec_name",
        "-of",
        "json",
        str(path),
    ]
    try:
        process = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
            shell=False,
        )
        payload = json.loads(process.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as exc:
        raise ValueError("Файл не является поддерживаемым видео") from exc

    try:
        duration = float(payload["format"]["duration"])
        format_name = str(payload["format"]["format_name"])
        video_stream = next(
            stream for stream in payload["streams"] if stream.get("codec_type") == "video"
        )
        codec = str(video_stream["codec_name"])
    except (KeyError, StopIteration, TypeError, ValueError) as exc:
        raise ValueError("В файле отсутствует корректный видеопоток") from exc

    allowed_formats = {"mov,mp4,m4a,3gp,3g2,mj2", "matroska,webm", "avi"}
    allowed_codecs = {"h264", "hevc", "vp8", "vp9", "av1", "mpeg4"}
    if format_name not in allowed_formats or codec not in allowed_codecs:
        raise ValueError("Неподдерживаемый формат или кодек видео")
    if duration <= 0 or duration > max_duration_s:
        raise ValueError(f"Продолжительность видео должна быть не более {max_duration_s} секунд")
    return VideoMetadata(duration_s=duration, format_name=format_name, video_codec=codec)
