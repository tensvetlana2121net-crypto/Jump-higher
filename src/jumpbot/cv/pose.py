from collections.abc import Callable
from pathlib import Path

import numpy as np

from jumpbot.cv.types import FramePose, Landmark

MEDIAPIPE_INDEXES = {
    "head": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot": 31,
    "right_foot": 32,
}

# Standard COCO-17 keypoint order used by RTMPose Body.
COCO_INDEXES = {
    "head": 0,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


def _open_video(video_path: Path):
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install JumpBot with the 'cv' extra") from exc
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Video cannot be opened")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        capture.release()
        raise ValueError("Video has invalid FPS metadata")
    return capture, fps, declared_frames


def _select_person(
    keypoints: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    points = np.asarray(keypoints)
    confidence = np.asarray(scores)
    if points.size == 0 or confidence.size == 0:
        return None
    if points.ndim == 2:
        points = points[None, ...]
    if confidence.ndim == 1:
        confidence = confidence[None, ...]
    person = int(np.argmax(np.mean(confidence, axis=1)))
    return points[person], confidence[person]


def extract_pose_rtmpose(video_path: Path) -> tuple[list[FramePose], float, int]:
    try:
        from rtmlib import Body
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("RTMPose is not installed; install JumpBot with the 'cv' extra") from exc

    capture, fps, declared_frames = _open_video(video_path)
    estimator = Body(mode="balanced", backend="onnxruntime", device="cpu")
    frames: list[FramePose] = []
    frame_index = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            selected = _select_person(*estimator(image))
            if selected is not None:
                keypoints, scores = selected
                points = {
                    name: Landmark(
                        x_px=float(keypoints[index, 0]),
                        y_px=float(keypoints[index, 1]),
                        visibility=float(scores[index]),
                    )
                    for name, index in COCO_INDEXES.items()
                }
                frames.append(FramePose(frame_index, frame_index / fps, points))
            frame_index += 1
    finally:
        capture.release()
    return frames, fps, declared_frames or frame_index


def extract_pose_mediapipe(video_path: Path) -> tuple[list[FramePose], float, int]:
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "MediaPipe is not installed; install JumpBot with the 'cv' extra"
        ) from exc

    capture, fps, declared_frames = _open_video(video_path)
    frames: list[FramePose] = []
    frame_index = 0
    try:
        with mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        ) as estimator:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                height, width = image.shape[:2]
                result = estimator.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if result.pose_landmarks:
                    points = {}
                    for name, index in MEDIAPIPE_INDEXES.items():
                        raw = result.pose_landmarks.landmark[index]
                        points[name] = Landmark(
                            x_px=float(raw.x * width),
                            y_px=float(raw.y * height),
                            visibility=float(raw.visibility),
                        )
                    frames.append(FramePose(frame_index, frame_index / fps, points))
                frame_index += 1
    finally:
        capture.release()
    return frames, fps, declared_frames or frame_index


POSE_BACKENDS: dict[str, Callable[[Path], tuple[list[FramePose], float, int]]] = {
    "rtmpose": extract_pose_rtmpose,
    "mediapipe": extract_pose_mediapipe,
}


def extract_pose(video_path: Path, backend: str = "rtmpose") -> tuple[list[FramePose], float, int]:
    try:
        extractor = POSE_BACKENDS[backend.lower()]
    except KeyError as exc:
        supported = ", ".join(sorted(POSE_BACKENDS))
        raise ValueError(f"Unknown pose backend '{backend}'; expected one of: {supported}") from exc
    return extractor(video_path)
