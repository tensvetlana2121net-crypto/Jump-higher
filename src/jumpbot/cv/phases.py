import numpy as np

from jumpbot.cv.types import PhaseFrames


def _sustained_starts(mask: np.ndarray, frames: int = 3) -> np.ndarray:
    if len(mask) < frames:
        return np.array([], dtype=int)
    windows = np.convolve(mask.astype(int), np.ones(frames, dtype=int), mode="valid")
    return np.flatnonzero(windows == frames)


def detect_phases(
    hip_y_m: np.ndarray,
    foot_y_px: np.ndarray,
    fps: float,
    floor_y_px: float | None = None,
    body_height_px: float | None = None,
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

    search_end = min(len(hip_y_m) - 1, start + int(2.5 * fps))
    bottom = start + int(np.argmin(hip_y_m[start : search_end + 1]))

    if floor_y_px is None:
        floor_y_px = float(np.nanpercentile(foot_y_px[:baseline_count], 90))
    clearance_px = max(4.0, 0.008 * body_height_px) if body_height_px else 4.0
    airborne = foot_y_px < floor_y_px - clearance_px

    # Require three consecutive airborne/contact frames to avoid single-frame noise.
    takeoff_candidates = _sustained_starts(airborne)
    takeoff_candidates = takeoff_candidates[takeoff_candidates > bottom]
    if not takeoff_candidates.size:
        raise ValueError("Take-off was not detected")
    takeoff = int(takeoff_candidates[0])

    contact = ~airborne
    landing_candidates = _sustained_starts(contact)
    landing_candidates = landing_candidates[
        landing_candidates > takeoff + max(2, int(0.15 * fps))
    ]
    if not landing_candidates.size:
        # On an ice rink the athlete moves across the frame, so perspective can
        # shift the apparent floor height between take-off and landing. Rebase
        # contact on the post-apex foot level instead of rejecting the jump.
        tentative_apex = takeoff + int(np.argmax(hip_y_m[takeoff:]))
        post_apex = foot_y_px[tentative_apex:]
        if post_apex.size:
            landing_floor = float(np.nanpercentile(post_apex, 90))
            local_contact = foot_y_px >= landing_floor - clearance_px
            landing_candidates = _sustained_starts(local_contact)
            landing_candidates = landing_candidates[
                landing_candidates > max(
                    tentative_apex, takeoff + max(2, int(0.15 * fps))
                )
            ]
        if not landing_candidates.size:
            raise ValueError("Landing was not detected")
    landing = int(landing_candidates[0])
    apex = takeoff + int(np.argmax(hip_y_m[takeoff : landing + 1]))

    if velocity[takeoff] < -0.2:
        raise ValueError("Detected take-off conflicts with hip trajectory")
    return PhaseFrames(
        start=start,
        countermovement_bottom=bottom,
        takeoff=takeoff,
        apex=apex,
        landing=landing,
    )
