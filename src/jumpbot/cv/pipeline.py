from pathlib import Path

import numpy as np

from jumpbot.cv.filtering import clean_trajectory
from jumpbot.cv.metrics import (
    flight_height,
    scale_from_height,
    unwrap_angular_velocity,
    vertical_velocity,
)
from jumpbot.cv.phases import detect_phases
from jumpbot.cv.pose import extract_pose
from jumpbot.cv.types import AnalysisResult, FramePose


def _midpoint(frames: list[FramePose], left: str, right: str, axis: str) -> np.ndarray:
    result = []
    for frame in frames:
        a, b = frame.points[left], frame.points[right]
        if min(a.visibility, b.visibility) < 0.6:
            result.append(np.nan)
        else:
            result.append((getattr(a, axis) + getattr(b, axis)) / 2)
    return np.asarray(result)


def analyze_jump(
    video_path: Path,
    athlete_height_m: float | None = None,
    pose_backend: str = "rtmpose",
) -> AnalysisResult:
    frames, fps, frame_count = extract_pose(video_path, pose_backend)
    if len(frames) < max(12, int(fps)):
        raise ValueError("Insufficient visible pose frames")
    if any(b.frame - a.frame > 5 for a, b in zip(frames, frames[1:], strict=False)):
        raise ValueError("Pose tracking contains a long gap")

    hip_x = clean_trajectory(_midpoint(frames, "left_hip", "right_hip", "x_px"))
    hip_y_px = clean_trajectory(_midpoint(frames, "left_hip", "right_hip", "y_px"))
    shoulder_x = clean_trajectory(_midpoint(frames, "left_shoulder", "right_shoulder", "x_px"))
    shoulder_y = clean_trajectory(_midpoint(frames, "left_shoulder", "right_shoulder", "y_px"))
    ankle_y = clean_trajectory(_midpoint(frames, "left_ankle", "right_ankle", "y_px"))
    if all("left_heel" in frame.points and "left_foot" in frame.points for frame in frames):
        foot_y = np.maximum(
            clean_trajectory(_midpoint(frames, "left_heel", "right_heel", "y_px")),
            clean_trajectory(_midpoint(frames, "left_foot", "right_foot", "y_px")),
        )
    else:
        foot_y = ankle_y.copy()

    scale = None
    if athlete_height_m:
        standing = min(len(frames) // 3, max(5, int(0.75 * fps)))
        head_y = np.asarray([frame.points["head"].y_px for frame in frames])
        apparent_height = ankle_y[:standing] - head_y[:standing]
        scale = scale_from_height(athlete_height_m, apparent_height)

    # Phase detection needs only a consistently scaled upward trajectory.
    hip_y_up = -hip_y_px * (scale or 0.001)
    phases = detect_phases(hip_y_up, foot_y, fps)
    flight_time = (phases.landing - phases.takeoff) / fps

    velocity = vertical_velocity(hip_y_up, fps)
    trunk_angle = np.arctan2(-(shoulder_y - hip_y_px), shoulder_x - hip_x)
    angular_velocity = unwrap_angular_velocity(trunk_angle, fps)

    visibility = np.mean(
        [min(point.visibility for point in frame.points.values()) for frame in frames]
    )
    flags: list[str] = []
    if fps < 50:
        flags.append("low_fps")
    if visibility < 0.7:
        flags.append("low_landmark_visibility")
    confidence = float(np.clip(0.55 * visibility + 0.25 * min(fps / 60, 1) + 0.2, 0, 1))

    displacement = None
    takeoff_velocity = None
    max_velocity = None
    if scale:
        standing_frames = max(3, min(phases.start, int(0.5 * fps)))
        baseline = float(np.median(hip_y_up[:standing_frames]))
        displacement = float(hip_y_up[phases.apex] - baseline)
        takeoff_velocity = float(velocity[phases.takeoff])
        max_velocity = float(np.max(velocity[phases.countermovement_bottom : phases.takeoff + 1]))

    return AnalysisResult(
        fps=fps,
        frame_count=frame_count,
        phases=phases,
        flight_time_s=flight_time,
        height_flight_m=flight_height(flight_time),
        height_displacement_m=displacement,
        takeoff_velocity_mps=takeoff_velocity,
        max_propulsion_velocity_mps=max_velocity,
        max_angular_velocity_dps=float(
            np.max(np.abs(angular_velocity[phases.start : phases.landing + 1]))
        ),
        confidence_score=confidence,
        quality_flags=flags,
    )
