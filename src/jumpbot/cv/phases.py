import numpy as np

from jumpbot.cv.types import PhaseFrames


def detect_phases(
    hip_y_m: np.ndarray,
    foot_y_px: np.ndarray,
    fps: float,
    floor_y_px: float | None = None,
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
    airborne = foot_y_px < floor_y_px - 4.0

    # Require three consecutive airborne/contact frames to avoid single-frame noise.
    run = np.convolve(airborne.astype(int), np.ones(3, dtype=int), mode="same") >= 3
    takeoff_candidates = np.flatnonzero(run & (np.arange(len(run)) > bottom))
    if not takeoff_candidates.size:
        raise ValueError("Take-off was not detected")
    takeoff = int(takeoff_candidates[0])

    contact = ~airborne
    contact_run = np.convolve(contact.astype(int), np.ones(3, dtype=int), mode="same") >= 3
    landing_candidates = np.flatnonzero(
        contact_run & (np.arange(len(contact_run)) > takeoff + max(2, int(0.15 * fps)))
    )
    if not landing_candidates.size:
        raise ValueError("Landing was not detected")
    landing = int(landing_candidates[0])
    apex = takeoff + int(np.argmax(hip_y_m[takeoff : landing + 1]))

    if velocity[takeoff] < -0.2:
        raise ValueError("Detected take-off conflicts with hip trajectory")
    return PhaseFrames(start=start, countermovement_bottom=bottom, takeoff=takeoff, apex=apex, landing=landing)
