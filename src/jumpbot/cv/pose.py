from collections.abc import Callable
from dataclasses import dataclass
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

# Halpe-26 keypoint order used by RTMPose BodyWithFeet.
HALPE26_INDEXES = {
    **COCO_INDEXES,
    "left_foot": 20,
    "right_foot": 21,
    "left_small_toe": 22,
    "right_small_toe": 23,
    "left_heel": 24,
    "right_heel": 25,
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


def _person_geometry(points: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, float]:
    core = np.asarray(points)[5:17]
    visible = np.asarray(scores)[5:17] >= 0.35
    sample = core[visible] if np.count_nonzero(visible) >= 4 else core
    center = np.median(sample, axis=0)
    span = np.ptp(sample, axis=0)
    return center, max(float(np.hypot(*span)), 20.0)


@dataclass
class _PersonTracker:
    """Keep one athlete selected when other people cross the frame."""

    points: np.ndarray | None = None
    center: np.ndarray | None = None
    velocity: np.ndarray | None = None
    scale: float = 20.0
    missing_frames: int = 0

    def select(
        self, keypoints: np.ndarray, scores: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        candidates = np.asarray(keypoints)
        confidence = np.asarray(scores)
        if candidates.size == 0 or confidence.size == 0:
            self.missing_frames += 1
            return None
        if candidates.ndim == 2:
            candidates = candidates[None, ...]
        if confidence.ndim == 1:
            confidence = confidence[None, ...]

        geometries = [
            _person_geometry(candidate, candidate_scores)
            for candidate, candidate_scores in zip(candidates, confidence, strict=True)
        ]
        mean_scores = np.mean(confidence[:, :17], axis=1)
        if self.center is None:
            scales = np.array([geometry[1] for geometry in geometries])
            index = int(np.argmax(mean_scores + 0.25 * scales / np.max(scales)))
        else:
            predicted = self.center + (
                self.velocity if self.velocity is not None else np.zeros(2)
            ) * (self.missing_frames + 1)
            costs: list[float] = []
            for candidate, candidate_scores, (center, scale), confidence_score in zip(
                candidates, confidence, geometries, mean_scores, strict=True
            ):
                normalizer = max(self.scale, scale, 20.0)
                motion_cost = float(np.linalg.norm(center - predicted) / normalizer)
                scale_cost = abs(float(np.log(scale / self.scale)))
                pose_cost = 0.0
                if self.points is not None:
                    visible = (candidate_scores[:17] >= 0.35) & np.isfinite(
                        self.points[:17]
                    ).all(axis=1)
                    if np.count_nonzero(visible) >= 4:
                        current = candidate[:17][visible] - center
                        previous = self.points[:17][visible] - self.center
                        pose_cost = float(
                            np.median(np.linalg.norm(current - previous, axis=1)) / normalizer
                        )
                costs.append(
                    motion_cost + 0.35 * scale_cost + 0.25 * pose_cost - 0.2 * confidence_score
                )
            index = int(np.argmin(costs))
            if costs[index] > 2.5 and self.missing_frames < 12:
                self.missing_frames += 1
                return None

        selected_points = candidates[index]
        selected_scores = confidence[index]
        new_center, new_scale = geometries[index]
        if self.center is not None:
            observed_velocity = (new_center - self.center) / (self.missing_frames + 1)
            self.velocity = (
                observed_velocity
                if self.velocity is None
                else 0.65 * self.velocity + 0.35 * observed_velocity
            )
        self.points = selected_points.copy()
        self.center = new_center
        self.scale = new_scale
        self.missing_frames = 0
        return selected_points, selected_scores


def extract_pose_rtmpose(video_path: Path) -> tuple[list[FramePose], float, int]:
    try:
        from rtmlib import BodyWithFeet
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("RTMPose is not installed; install JumpBot with the 'cv' extra") from exc

    capture, fps, declared_frames = _open_video(video_path)
    estimator = BodyWithFeet(mode="balanced", backend="onnxruntime", device="cpu")
    tracker = _PersonTracker()
    frames: list[FramePose] = []
    frame_index = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            selected = tracker.select(*estimator(image))
            if selected is not None:
                keypoints, scores = selected
                points = {
                    name: Landmark(
                        x_px=float(keypoints[index, 0]),
                        y_px=float(keypoints[index, 1]),
                        visibility=float(scores[index]),
                    )
                    for name, index in HALPE26_INDEXES.items()
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
