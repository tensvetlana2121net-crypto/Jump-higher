import numpy as np

from jumpbot.cv.types import PhaseFrames


def _sustained_starts(mask: np.ndarray, frames: int = 3) -> np.ndarray:
    if len(mask) < frames:
        return np.array([], dtype=int)
    windows = np.convolve(mask.astype(int), np.ones(frames, dtype=int), mode="valid")
    sustained = np.flatnonzero(windows == frames)
    if sustained.size < 2:
        return sustained
    return sustained[np.r_[True, np.diff(sustained) > 1]]


def _torso_landing(
    trajectory: np.ndarray,
    takeoff: int,
    apex: int,
    minimum_flight: int,
) -> int | None:
    """Find landing from COM return when rotating feet are not reliably visible."""
    if apex <= takeoff or apex >= len(trajectory) - 3:
        return None
    takeoff_level = float(trajectory[takeoff])
    # A landing crouch normally brings COM to or below the take-off level.
    # Allow two centimetres for pose noise and perspective changes.
    returned = trajectory <= takeoff_level + 0.02
    search_start = max(apex + 1, takeoff + minimum_flight)
    starts = _sustained_starts(returned[search_start:], frames=3)
    if not starts.size:
        return None
    landing = search_start + int(starts[0])
    if float(trajectory[apex] - trajectory[landing]) < 0.03:
        return None
    return landing


def detect_phases(
    hip_y_m: np.ndarray,
    foot_y_px: np.ndarray,
    fps: float,
    floor_y_px: float | None = None,
    body_height_px: float | None = None,
    minimum_takeoff_frame: int = 0,
) -> PhaseFrames:
    """Detect a single countermovement jump using kinematics and floor distance.

    Metric hip coordinates point upward; image foot coordinates point downward.
    Thresholds are intentionally conservative and must be validated per camera protocol.
    """
    if len(hip_y_m) < max(12, int(fps)):
        raise ValueError("Video is too short for phase detection")

    velocity = np.gradient(hip_y_m, 1.0 / fps)
    baseline_count = min(len(hip_y_m) // 3, max(5, int(0.75 * fps)))
    baseline = float(np.median(hip_y_m[:baseline_count]))
    movement = np.abs(hip_y_m - baseline) > 0.015
    candidates = np.flatnonzero(movement)
    start = int(candidates[0]) if candidates.size else 0

    if floor_y_px is None:
        floor_y_px = float(np.nanpercentile(foot_y_px[:baseline_count], 90))
    clearance_px = max(4.0, 0.008 * body_height_px) if body_height_px else 4.0
    airborne = foot_y_px < floor_y_px - clearance_px

    # Require three consecutive airborne/contact frames to avoid single-frame noise.
    takeoff_candidates = _sustained_starts(airborne)
    if not takeoff_candidates.size:
        raise ValueError("Take-off was not detected")

    # Skaters may lift one foot repeatedly before the jump. Evaluate every
    # sustained airborne interval and choose the one with the largest COM rise,
    # then locate countermovement only in the second preceding that take-off.
    contact_starts = _sustained_starts(~airborne)
    minimum_flight = max(2, int(0.15 * fps))
    candidates: list[tuple[float, int, int]] = []
    for candidate in takeoff_candidates:
        if candidate < minimum_takeoff_frame:
            continue
        possible_landings = contact_starts[contact_starts > candidate + minimum_flight]
        if not possible_landings.size:
            tentative_apex = candidate + int(np.argmax(hip_y_m[candidate:]))
            post_apex = foot_y_px[tentative_apex:]
            if post_apex.size:
                landing_floor = float(np.nanpercentile(post_apex, 90))
                local_contact_starts = _sustained_starts(foot_y_px >= landing_floor - clearance_px)
                possible_landings = local_contact_starts[
                    local_contact_starts > max(tentative_apex, candidate + minimum_flight)
                ]
            if not possible_landings.size:
                torso_landing = _torso_landing(
                    hip_y_m,
                    int(candidate),
                    int(tentative_apex),
                    minimum_flight,
                )
                if torso_landing is not None:
                    possible_landings = np.array([torso_landing], dtype=int)
        if not possible_landings.size:
            continue
        candidate_landing = int(possible_landings[0])
        candidate_apex = int(candidate + np.argmax(hip_y_m[candidate : candidate_landing + 1]))
        if not candidate < candidate_apex < candidate_landing:
            continue
        rise = float(hip_y_m[candidate_apex] - hip_y_m[candidate])
        candidates.append((rise, int(candidate), candidate_landing))
    if not candidates:
        raise ValueError("Landing was not detected")
    _, takeoff, landing = max(candidates, key=lambda item: item[0])

    # The blade can touch the ice before the pose settles at the new perspective
    # floor level. Detect the first sharp deceleration of the descending foot;
    # otherwise a deep landing crouch is mistaken for extra flight time.
    provisional_apex = takeoff + int(np.argmax(hip_y_m[takeoff : landing + 1]))
    foot_steps = np.diff(foot_y_px)
    quiet_step_px = max(1.0, 0.003 * body_height_px) if body_height_px else 1.0
    for index in range(provisional_apex + 2, max(provisional_apex + 2, landing - 2)):
        descent = foot_y_px[index] - float(np.min(foot_y_px[provisional_apex:index]))
        recent_speed = foot_steps[max(provisional_apex, index - 5) : index]
        quiet_window = foot_steps[index : index + 3]
        if (
            descent >= 2 * clearance_px
            and recent_speed.size
            and float(np.max(recent_speed)) >= 2 * quiet_step_px
            and quiet_window.size == 3
            and float(np.mean(np.abs(quiet_window))) <= quiet_step_px
        ):
            landing = index
            break

    apex = takeoff + int(np.argmax(hip_y_m[takeoff : landing + 1]))
    ascent_frames = apex - takeoff
    descent_frames = landing - apex
    # A take-off detected much earlier than the time-reversed descent is
    # usually a gliding blade that RTMPose placed a few pixels above the ice.
    # Free-flight COM motion cannot spend substantially longer rising than
    # falling to a normal (usually lower) landing posture.
    if descent_frames >= 2 and ascent_frames > 1.25 * descent_frames:
        takeoff = max(0, apex - descent_frames)

    bottom_search_start = max(0, min(start, takeoff), takeoff - int(fps))
    bottom = bottom_search_start + int(np.argmin(hip_y_m[bottom_search_start : takeoff + 1]))

    apex = takeoff + int(np.argmax(hip_y_m[takeoff : landing + 1]))

    if not takeoff < apex < landing:
        raise ValueError("Detected flight phases are physically inconsistent")

    if velocity[takeoff] < -0.2:
        raise ValueError("Detected take-off conflicts with hip trajectory")
    return PhaseFrames(
        start=start,
        countermovement_bottom=bottom,
        takeoff=takeoff,
        apex=apex,
        landing=landing,
    )
