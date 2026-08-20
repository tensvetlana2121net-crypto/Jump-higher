from pathlib import Path

import numpy as np

from jumpbot.cv.filtering import clean_trajectory, interpolate_short_gaps
from jumpbot.cv.metrics import (
    axial_rotation_metrics,
    ballistic_height,
    body_orientation,
    flight_height,
    rotation_metrics,
    scale_from_height,
    unwrap_angular_velocity,
    vertical_velocity,
)
from jumpbot.cv.phases import detect_phases
from jumpbot.cv.pose import extract_pose
from jumpbot.cv.types import AnalysisResult, FramePose


def _midpoint(
    frames: list[FramePose], left: str, right: str, axis: str, frame_count: int
) -> np.ndarray:
    result = np.full(frame_count, np.nan)
    for frame in frames:
        a, b = frame.points[left], frame.points[right]
        if min(a.visibility, b.visibility) >= 0.4:
            result[frame.frame] = (getattr(a, axis) + getattr(b, axis)) / 2
    return result


def _lowest_visible(
    frames: list[FramePose], names: tuple[str, ...], axis: str, frame_count: int
) -> np.ndarray:
    """Lowest visible point in image coordinates, used for ice contact."""
    result = np.full(frame_count, np.nan)
    for frame in frames:
        values = [
            getattr(frame.points[name], axis)
            for name in names
            if name in frame.points and frame.points[name].visibility >= 0.4
        ]
        if values:
            result[frame.frame] = max(values)
    return result


def _signed_width(
    frames: list[FramePose], left: str, right: str, frame_count: int
) -> np.ndarray:
    result = np.full(frame_count, np.nan)
    for frame in frames:
        left_point = frame.points[left]
        right_point = frame.points[right]
        if min(left_point.visibility, right_point.visibility) >= 0.4:
            result[frame.frame] = right_point.x_px - left_point.x_px
    return result


def _whole_body_com(frames: list[FramePose], axis: str, frame_count: int) -> np.ndarray:
    """Approximate whole-body COM using standard segment mass proportions."""
    result = np.full(frame_count, np.nan)
    segments = (
        ("head", "head", 0.081, 0.0),
        ("left_shoulder", "left_hip", 0.2485, 0.55),
        ("right_shoulder", "right_hip", 0.2485, 0.55),
        ("left_shoulder", "left_elbow", 0.028, 0.436),
        ("right_shoulder", "right_elbow", 0.028, 0.436),
        ("left_elbow", "left_wrist", 0.022, 0.682),
        ("right_elbow", "right_wrist", 0.022, 0.682),
        ("left_hip", "left_knee", 0.100, 0.433),
        ("right_hip", "right_knee", 0.100, 0.433),
        ("left_knee", "left_ankle", 0.061, 0.433),
        ("right_knee", "right_ankle", 0.061, 0.433),
    )
    for frame in frames:
        weighted = 0.0
        available_weight = 0.0
        for proximal_name, distal_name, weight, fraction in segments:
            proximal = frame.points.get(proximal_name)
            distal = frame.points.get(distal_name)
            if (
                proximal is None
                or distal is None
                or min(proximal.visibility, distal.visibility) < 0.35
            ):
                continue
            proximal_value = getattr(proximal, axis)
            distal_value = getattr(distal, axis)
            weighted += weight * (proximal_value + fraction * (distal_value - proximal_value))
            available_weight += weight
        # One hidden leg removes about 16% of the segment mass. Requiring 85%
        # therefore discarded otherwise useful skating frames during rotations.
        if available_weight >= 0.55:
            result[frame.frame] = weighted / available_weight
    return result


def _foot_orientation(frames: list[FramePose], frame_count: int) -> np.ndarray | None:
    """Directed heel-to-toe angle, combining both visible feet."""
    raw = np.full(frame_count, np.nan)
    for frame in frames:
        angles: list[float] = []
        for side in ("left", "right"):
            heel = frame.points.get(f"{side}_heel")
            toe = frame.points.get(f"{side}_foot")
            if heel is None or toe is None or min(heel.visibility, toe.visibility) < 0.5:
                continue
            dx = toe.x_px - heel.x_px
            dy_up = heel.y_px - toe.y_px
            if np.hypot(dx, dy_up) >= 4:
                angles.append(float(np.arctan2(dy_up, dx)))
        if angles:
            raw[frame.frame] = np.arctan2(
                np.mean(np.sin(angles)), np.mean(np.cos(angles))
            )
    valid = np.flatnonzero(np.isfinite(raw))
    if valid.size < 5:
        return None
    unwrapped = np.unwrap(raw[valid])
    interpolated = np.interp(np.arange(frame_count), valid, unwrapped)
    return clean_trajectory(interpolated)


def analyze_jump(
    video_path: Path,
    athlete_height_m: float | None = None,
    pose_backend: str = "rtmpose",
) -> AnalysisResult:
    frames, fps, frame_count = extract_pose(video_path, pose_backend)
    if len(frames) < max(12, int(fps)):
        raise ValueError("Insufficient visible pose frames")
    pose_gaps = [
        b.frame - a.frame - 1 for a, b in zip(frames, frames[1:], strict=False)
    ]
    longest_pose_gap = max(pose_gaps, default=0)
    max_pose_gap = max(5, int(0.75 * fps))
    if any(
        b.frame - a.frame > max_pose_gap + 1
        for a, b in zip(frames, frames[1:], strict=False)
    ):
        raise ValueError("Pose tracking contains a long gap")

    declared_frame_count = frame_count
    trajectory_frame_count = frames[-1].frame + 1
    hip_x = clean_trajectory(
        _midpoint(frames, "left_hip", "right_hip", "x_px", trajectory_frame_count),
        max_gap=max_pose_gap,
    )
    hip_y_px = clean_trajectory(
        _midpoint(frames, "left_hip", "right_hip", "y_px", trajectory_frame_count),
        max_gap=max_pose_gap,
    )
    shoulder_x = clean_trajectory(
        _midpoint(frames, "left_shoulder", "right_shoulder", "x_px", trajectory_frame_count),
        max_gap=max_pose_gap,
    )
    shoulder_y = clean_trajectory(
        _midpoint(frames, "left_shoulder", "right_shoulder", "y_px", trajectory_frame_count),
        max_gap=max_pose_gap,
    )
    shoulder_width = interpolate_short_gaps(
        _signed_width(
            frames, "left_shoulder", "right_shoulder", trajectory_frame_count
        ),
        max_gap=max_pose_gap,
    )
    hip_width = interpolate_short_gaps(
        _signed_width(frames, "left_hip", "right_hip", trajectory_frame_count),
        max_gap=max_pose_gap,
    )
    ankle_y = clean_trajectory(
        _lowest_visible(
            frames,
            ("left_ankle", "right_ankle"),
            "y_px",
            trajectory_frame_count,
        ),
        max_gap=max_pose_gap,
    )
    raw_com_y_px = _whole_body_com(frames, "y_px", trajectory_frame_count)
    com_fallback_used = bool(np.count_nonzero(~np.isfinite(raw_com_y_px)))
    valid_com = np.isfinite(raw_com_y_px)
    if np.count_nonzero(valid_com) < 3:
        raw_com_y_px = hip_y_px.copy()
        com_fallback_used = True
    elif not valid_com.all():
        hip_offset = float(np.median(raw_com_y_px[valid_com] - hip_y_px[valid_com]))
        raw_com_y_px[~valid_com] = hip_y_px[~valid_com] + hip_offset
    com_y_px = clean_trajectory(raw_com_y_px, max_gap=max_pose_gap)
    foot_angle = _foot_orientation(frames, trajectory_frame_count)
    foot_fallback_used = False
    if all("left_heel" in frame.points and "left_foot" in frame.points for frame in frames):
        try:
            foot_y = clean_trajectory(
                _lowest_visible(
                    frames,
                    ("left_heel", "right_heel", "left_foot", "right_foot"),
                    "y_px",
                    trajectory_frame_count,
                ),
                max_gap=max_pose_gap,
            )
        except ValueError:
            foot_y = ankle_y.copy()
            foot_fallback_used = True
    else:
        foot_y = ankle_y.copy()
        foot_fallback_used = True

    head_y = np.full(trajectory_frame_count, np.nan)
    for frame in frames:
        if frame.points["head"].visibility >= 0.4:
            head_y[frame.frame] = frame.points["head"].y_px
    head_y = clean_trajectory(head_y, max_gap=max_pose_gap)

    scale = None
    if athlete_height_m:
        standing = min(len(frames) // 3, max(5, int(0.75 * fps)))
        apparent_height = ankle_y[:standing] - head_y[:standing]
        scale = scale_from_height(athlete_height_m, apparent_height)
    else:
        apparent_height = ankle_y[: min(len(frames), max(5, int(0.75 * fps)))] - hip_y_px[
            : min(len(frames), max(5, int(0.75 * fps)))
        ]

    # Phase detection needs only a consistently scaled upward trajectory.
    com_y_up = -com_y_px * (scale or 0.001)
    body_height_px = float(np.nanmedian(apparent_height))
    phases = detect_phases(
        com_y_up,
        foot_y,
        fps,
        body_height_px=body_height_px,
    )
    flight_time = (phases.landing - phases.takeoff) / fps

    velocity = vertical_velocity(com_y_up, fps)
    trunk_angle = body_orientation(hip_x, hip_y_px, shoulder_x, shoulder_y)
    angular_velocity = unwrap_angular_velocity(trunk_angle, fps)
    rotation_degrees, rotation_turns, rotation_direction, rotation_speed = rotation_metrics(
        trunk_angle, angular_velocity, phases.takeoff, phases.landing
    )
    takeoff_foot_angle = None
    landing_foot_angle = None
    if foot_angle is not None:
        foot_velocity = unwrap_angular_velocity(foot_angle, fps)
        foot_rotation = rotation_metrics(
            foot_angle, foot_velocity, phases.takeoff, phases.landing
        )
        takeoff_foot_angle = float(
            np.rad2deg(
                np.arctan2(
                    np.sin(foot_angle[phases.takeoff]),
                    np.cos(foot_angle[phases.takeoff]),
                )
            )
        )
        landing_foot_angle = float(
            np.rad2deg(
                np.arctan2(
                    np.sin(foot_angle[phases.landing]),
                    np.cos(foot_angle[phases.landing]),
                )
            )
        )
        if foot_rotation[0] is not None:
            rotation_degrees, rotation_turns, rotation_direction, rotation_speed = foot_rotation
    shoulder_scale = max(float(np.percentile(np.abs(shoulder_width), 90)), 1.0)
    hip_scale = max(float(np.percentile(np.abs(hip_width), 90)), 1.0)
    axial_width = 0.65 * shoulder_width / shoulder_scale + 0.35 * hip_width / hip_scale
    axial_degrees, axial_turns, axial_speed = axial_rotation_metrics(
        axial_width, fps, phases.takeoff, phases.landing
    )
    if axial_degrees is not None and (
        rotation_degrees is None or axial_degrees > rotation_degrees
    ):
        rotation_degrees = axial_degrees
        rotation_turns = axial_turns
        rotation_speed = axial_speed
    trunk_length = np.hypot(shoulder_x - hip_x, shoulder_y - hip_y_px)
    inclination = np.rad2deg(np.arctan2(np.sin(trunk_angle), np.cos(trunk_angle)))
    takeoff_inclination = float(inclination[phases.takeoff])
    max_inclination = float(
        np.max(np.abs(inclination[phases.start : phases.takeoff + 1]))
    )

    visibility = np.mean(
        [min(point.visibility for point in frame.points.values()) for frame in frames]
    )
    flags: list[str] = []
    if fps < 50:
        flags.append("low_fps")
    if visibility < 0.7:
        flags.append("low_landmark_visibility")
    if longest_pose_gap > int(0.2 * fps):
        flags.append("interpolated_pose_gap")
    if com_fallback_used:
        flags.append("partial_com_fallback")
    if foot_fallback_used:
        flags.append("ankle_based_ground_contact")
    flight_trunk = trunk_length[phases.takeoff : phases.landing + 1]
    if np.percentile(flight_trunk, 10) < 0.55 * np.median(flight_trunk):
        flags.append("unstable_trunk_orientation")
    if rotation_speed is not None and rotation_speed > 3000:
        flags.append("implausible_rotation_speed")
        rotation_speed = 3000.0

    displacement = None
    trajectory_height = None
    fitted_height = None
    takeoff_velocity = None
    max_velocity = None
    if scale:
        standing_frames = max(3, min(phases.start, int(0.5 * fps)))
        baseline = float(np.median(com_y_up[:standing_frames]))
        displacement = float(com_y_up[phases.apex] - baseline)
        body_rises = []
        for y_px in (head_y, shoulder_y, hip_y_px):
            y_up = -y_px * scale
            rise = float(
                np.max(y_up[phases.takeoff : phases.landing + 1])
                - y_up[phases.takeoff]
            )
            if 0.02 <= rise <= 1.0:
                body_rises.append(rise)
        if len(body_rises) >= 2:
            trajectory_height = float(np.median(body_rises))
        else:
            trajectory_height = float(
                com_y_up[phases.apex] - com_y_up[phases.takeoff]
            )
        fitted_height = ballistic_height(
            com_y_up[phases.takeoff : phases.landing + 1], fps
        )
        takeoff_velocity = float(velocity[phases.takeoff])
        max_velocity = float(np.max(velocity[phases.countermovement_bottom : phases.takeoff + 1]))

    flight_height_m = flight_height(flight_time)
    height_candidates = [flight_height_m]
    height_candidates.extend(
        value
        for value in (trajectory_height, fitted_height)
        if value is not None and 0.02 <= value <= 2.0
    )
    jump_height = float(np.median(height_candidates))
    if len(height_candidates) > 1 and np.ptp(height_candidates) > 0.25 * jump_height:
        flags.append("inconsistent_height_estimates")
        jump_height = trajectory_height or fitted_height or flight_height_m
    if not scale:
        flags.append("height_requires_athlete_height")
    else:
        # The frame derivative is strongly attenuated by pose smoothing at
        # 30 FPS. Use the energy-equivalent vertical take-off speed so the
        # reported speed remains physically consistent with the measured rise.
        takeoff_velocity = float(np.sqrt(2.0 * 9.80665 * jump_height))

    penalty_flags = {
        "low_fps",
        "low_landmark_visibility",
        "interpolated_pose_gap",
        "partial_com_fallback",
        "unstable_trunk_orientation",
        "implausible_rotation_speed",
        "inconsistent_height_estimates",
    }
    penalty = 0.08 * len(penalty_flags.intersection(flags))
    confidence = float(
        np.clip(0.55 * visibility + 0.25 * min(fps / 60, 1) + 0.2 - penalty, 0, 1)
    )

    return AnalysisResult(
        fps=fps,
        frame_count=max(declared_frame_count, trajectory_frame_count),
        phases=phases,
        flight_time_s=flight_time,
        jump_height_m=jump_height,
        height_flight_m=flight_height_m,
        height_trajectory_m=trajectory_height,
        height_ballistic_m=fitted_height,
        height_displacement_m=displacement,
        takeoff_velocity_mps=takeoff_velocity,
        max_propulsion_velocity_mps=max_velocity,
        max_angular_velocity_dps=rotation_speed,
        rotation_degrees=rotation_degrees,
        rotation_turns=rotation_turns,
        rotation_direction=rotation_direction,
        takeoff_foot_angle_deg=takeoff_foot_angle,
        landing_foot_angle_deg=landing_foot_angle,
        takeoff_inclination_deg=takeoff_inclination,
        max_inclination_deg=max_inclination,
        confidence_score=confidence,
        quality_flags=flags,
    )
