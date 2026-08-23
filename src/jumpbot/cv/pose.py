import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np

from jumpbot.cv.types import FramePose, Landmark

logger = logging.getLogger(__name__)
TRACKING_REACQUIRE_FRAMES = 30

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


def inference_stride(fps: float, target_fps: float = 30.0) -> int:
    """Sample expensive pose inference near target FPS while preserving timing."""
    if fps <= 0 or target_fps <= 0:
        raise ValueError("FPS must be positive")
    return max(1, int(round(fps / target_fps)))


def _tracking_roi(
    image_shape: tuple[int, ...],
    points: np.ndarray,
    scores: np.ndarray,
    padding: float = 0.55,
) -> tuple[int, int, int, int] | None:
    """Build a padded virtual-camera crop around the tracked athlete."""
    height, width = image_shape[:2]
    visible = (np.asarray(scores) >= 0.35) & np.isfinite(points).all(axis=1)
    if np.count_nonzero(visible) < 6:
        return None
    athlete = np.asarray(points)[visible]
    left, top = np.min(athlete, axis=0)
    right, bottom = np.max(athlete, axis=0)
    body_width = max(float(right - left), width * 0.08)
    body_height = max(float(bottom - top), height * 0.15)
    margin_x = max(body_width * padding, width * 0.04)
    margin_y = max(body_height * padding, height * 0.06)
    x1 = max(0, int(left - margin_x))
    y1 = max(0, int(top - margin_y))
    x2 = min(width, int(right + margin_x) + 1)
    y2 = min(height, int(bottom + margin_y) + 1)
    if x2 - x1 < 64 or y2 - y1 < 96:
        return None
    return x1, y1, x2, y2


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    homogeneous = np.concatenate(
        (values, np.ones((*values.shape[:-1], 1), dtype=float)), axis=-1
    )
    return homogeneous @ np.asarray(transform, dtype=float).T


def _infer_tracked_pose(
    estimator: object,
    image: np.ndarray,
    crop_box: tuple[int, int, int, int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if crop_box is None:
        return estimator(image)  # type: ignore[operator]
    x1, y1, x2, y2 = crop_box
    crop = image[y1:y2, x1:x2]
    pose_model = estimator.pose_model  # type: ignore[attr-defined]
    keypoints, scores = pose_model(crop)
    if np.asarray(keypoints).size:
        keypoints = np.asarray(keypoints).copy()
        keypoints[..., 0] += x1
        keypoints[..., 1] += y1
    return keypoints, scores


@dataclass
class _CameraStabilizer:
    """Estimate camera pan/zoom from background features outside the athlete ROI."""

    previous_gray: np.ndarray | None = None
    cumulative: np.ndarray | None = None

    def update(
        self, image: np.ndarray, excluded_roi: tuple[int, int, int, int] | None
    ) -> np.ndarray:
        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        if self.previous_gray is None:
            self.previous_gray = gray
            self.cumulative = np.eye(3)
            return identity

        mask = np.full(self.previous_gray.shape, 255, dtype=np.uint8)
        if excluded_roi is not None:
            x1, y1, x2, y2 = excluded_roi
            mask[y1:y2, x1:x2] = 0
        previous_points = cv2.goodFeaturesToTrack(
            self.previous_gray,
            maxCorners=250,
            qualityLevel=0.01,
            minDistance=12,
            mask=mask,
        )
        delta = identity
        if previous_points is not None and len(previous_points) >= 12:
            current_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.previous_gray, gray, previous_points, None
            )
            if current_points is not None and status is not None:
                valid = status.ravel().astype(bool)
                if np.count_nonzero(valid) >= 10:
                    estimated, inliers = cv2.estimateAffinePartial2D(
                        current_points[valid],
                        previous_points[valid],
                        method=cv2.RANSAC,
                        ransacReprojThreshold=2.5,
                    )
                    if estimated is not None and inliers is not None:
                        inlier_ratio = float(np.mean(inliers))
                        scale = float(np.hypot(estimated[0, 0], estimated[0, 1]))
                        if inlier_ratio >= 0.45 and 0.85 <= scale <= 1.18:
                            delta = estimated

        delta_3x3 = np.vstack((delta, (0.0, 0.0, 1.0)))
        self.cumulative = self.cumulative @ delta_3x3
        self.previous_gray = gray
        return self.cumulative[:2]


@lru_cache(maxsize=1)
def _rtmpose_estimator():
    """Load RTMPose once per Celery worker process."""
    try:
        from rtmlib import BodyWithFeet
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "RTMPose is not installed; install JumpBot with the 'cv' extra"
        ) from exc
    return BodyWithFeet(mode="balanced", backend="onnxruntime", device="cpu")


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


def extract_pose_rtmpose(
    video_path: Path,
    tracking_roi_enabled: bool = True,
    camera_stabilization_enabled: bool = True,
) -> tuple[list[FramePose], float, int]:
    capture, fps, declared_frames = _open_video(video_path)
    estimator = _rtmpose_estimator()
    tracker = _PersonTracker()
    stabilizer = _CameraStabilizer()
    frames: list[FramePose] = []
    frame_index = 0
    sampled_index = 0
    stride = inference_stride(fps)
    last_selected: tuple[np.ndarray, np.ndarray] | None = None
    inference_seconds = 0.0
    started_at = perf_counter()
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if frame_index % stride:
                frame_index += 1
                continue
            crop_box = None
            if (
                tracking_roi_enabled
                and last_selected is not None
                and sampled_index % TRACKING_REACQUIRE_FRAMES
            ):
                crop_box = _tracking_roi(image.shape, *last_selected)
            camera_transform = (
                stabilizer.update(image, crop_box)
                if camera_stabilization_enabled
                else np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            )
            inference_started = perf_counter()
            keypoints, scores = _infer_tracked_pose(estimator, image, crop_box)
            inference_seconds += perf_counter() - inference_started
            selected = tracker.select(keypoints, scores)
            if selected is not None:
                keypoints, scores = selected
                last_selected = (keypoints.copy(), scores.copy())
                stabilized_keypoints = _transform_points(keypoints, camera_transform)
                points = {
                    name: Landmark(
                        x_px=float(stabilized_keypoints[index, 0]),
                        y_px=float(stabilized_keypoints[index, 1]),
                        visibility=float(scores[index]),
                    )
                    for name, index in HALPE26_INDEXES.items()
                }
                frames.append(FramePose(sampled_index, frame_index / fps, points))
            else:
                last_selected = None
            sampled_index += 1
            frame_index += 1
    finally:
        capture.release()
        logger.info(
            "Pose extraction finished video=%s sampled_frames=%d inference_seconds=%.3f "
            "total_seconds=%.3f",
            video_path.name,
            sampled_index,
            inference_seconds,
            perf_counter() - started_at,
        )
    effective_fps = fps / stride
    sampled_frame_count = (
        (declared_frames + stride - 1) // stride if declared_frames else sampled_index
    )
    return frames, effective_fps, sampled_frame_count


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


def extract_pose(
    video_path: Path,
    backend: str = "rtmpose",
    tracking_roi_enabled: bool = True,
    camera_stabilization_enabled: bool = True,
) -> tuple[list[FramePose], float, int]:
    if backend.lower() == "rtmpose":
        return extract_pose_rtmpose(
            video_path, tracking_roi_enabled, camera_stabilization_enabled
        )
    try:
        extractor = POSE_BACKENDS[backend.lower()]
    except KeyError as exc:
        supported = ", ".join(sorted(POSE_BACKENDS))
        raise ValueError(f"Unknown pose backend '{backend}'; expected one of: {supported}") from exc
    return extractor(video_path)
