import numpy as np
from scipy.signal import find_peaks, savgol_filter

GRAVITY_MPS2 = 9.80665


def flight_height(flight_time_s: float) -> float:
    if flight_time_s <= 0:
        raise ValueError("Flight time must be positive")
    return GRAVITY_MPS2 * flight_time_s**2 / 8.0


def vertical_velocity(position_m: np.ndarray, fps: float) -> np.ndarray:
    if fps <= 0:
        raise ValueError("FPS must be positive")
    if len(position_m) < 3:
        raise ValueError("At least three samples are required")
    return np.gradient(np.asarray(position_m, dtype=float), 1.0 / fps)


def unwrap_angular_velocity(angle_rad: np.ndarray, fps: float) -> np.ndarray:
    if fps <= 0:
        raise ValueError("FPS must be positive")
    unwrapped = np.unwrap(np.asarray(angle_rad, dtype=float))
    if unwrapped.size < 5:
        return np.rad2deg(np.gradient(unwrapped, 1.0 / fps))
    window = min(unwrapped.size if unwrapped.size % 2 else unwrapped.size - 1, 11)
    window = max(window, 5)
    polyorder = min(3, window - 2)
    return np.rad2deg(
        savgol_filter(
            unwrapped,
            window_length=window,
            polyorder=polyorder,
            deriv=1,
            delta=1.0 / fps,
            mode="interp",
        )
    )


def ballistic_height(position_m: np.ndarray, fps: float) -> float | None:
    """Estimate rise after take-off from a robust quadratic flight trajectory."""
    values = np.asarray(position_m, dtype=float)
    if fps <= 0 or values.size < 5 or not np.isfinite(values).all():
        return None
    time = np.arange(values.size, dtype=float) / fps
    coefficient, linear, intercept = np.polyfit(time, values, 2)
    if coefficient >= 0:
        return None
    apex_time = float(np.clip(-linear / (2 * coefficient), time[0], time[-1]))
    apex = coefficient * apex_time**2 + linear * apex_time + intercept
    height = float(apex - intercept)
    return height if height > 0 else None


def body_orientation(
    hip_x: np.ndarray,
    hip_y_px: np.ndarray,
    shoulder_x: np.ndarray,
    shoulder_y_px: np.ndarray,
) -> np.ndarray:
    """Directed trunk orientation, in radians, with zero representing upright."""
    return np.arctan2(shoulder_x - hip_x, hip_y_px - shoulder_y_px)


def rotation_metrics(
    orientation_rad: np.ndarray, angular_velocity_dps: np.ndarray, start: int, end: int
) -> tuple[float | None, float | None, str | None, float | None]:
    if end <= start:
        return None, None, None, None
    unwrapped_deg = np.rad2deg(np.unwrap(np.asarray(orientation_rad, dtype=float)))
    flight_angle = unwrapped_deg[start : end + 1]
    if flight_angle.size < 3 or not np.isfinite(flight_angle).all():
        return None, None, None, None

    # A jumper can land with nearly the same trunk angle as at take-off after
    # completing a rotation.  The old end-minus-start calculation therefore
    # reported zero.  Sum meaningful frame-to-frame movement in the dominant
    # direction instead.  A small dead band rejects pose jitter while retaining
    # genuine reversals and incomplete rotations.
    increments = np.diff(flight_angle)
    noise_floor = 0.75
    meaningful = increments[np.abs(increments) >= noise_floor]
    if meaningful.size == 0:
        return None, None, None, None
    clockwise = float(np.sum(meaningful[meaningful > 0]))
    counterclockwise = float(-np.sum(meaningful[meaningful < 0]))
    rotation = max(clockwise, counterclockwise)
    if rotation < 30:
        return None, None, None, None
    direction = "clockwise" if clockwise >= counterclockwise else "counterclockwise"
    speed = float(np.percentile(np.abs(angular_velocity_dps[start : end + 1]), 95))
    return rotation, rotation / 360.0, direction, speed


def axial_rotation_metrics(
    projected_width: np.ndarray, fps: float, start: int, end: int
) -> tuple[float | None, float | None, float | None]:
    """Estimate monotonic axial rotation from front/back width oscillations.

    Anatomical left/right shoulder ordering changes every half turn. The
    arccosine of normalized signed width forms a triangle wave whose travelled
    distance corresponds to rotation, including turns that finish in the same
    pose in which they started.
    """
    values = np.asarray(projected_width[start : end + 1], dtype=float)
    if fps <= 0 or values.size < 7 or not np.isfinite(values).all():
        return None, None, None
    scale = float(np.percentile(np.abs(values), 90))
    if scale <= 1e-6:
        return None, None, None
    normalized = np.clip(values / scale, -1.0, 1.0)
    window = min(values.size if values.size % 2 else values.size - 1, 7)
    smoothed = savgol_filter(normalized, window, min(2, window - 1), mode="interp")
    principal = np.rad2deg(np.arccos(np.clip(smoothed, -1.0, 1.0)))
    distance = max(2, int(0.08 * fps))
    peaks, _ = find_peaks(principal, prominence=25, distance=distance)
    troughs, _ = find_peaks(-principal, prominence=25, distance=distance)
    turning_points = np.unique(np.r_[0, peaks, troughs, len(principal) - 1])
    rotation = float(np.sum(np.abs(np.diff(principal[turning_points]))))
    if rotation < 45:
        return None, None, None
    cumulative = np.r_[0.0, np.cumsum(np.abs(np.diff(principal)))]
    speed = float(np.percentile(np.gradient(cumulative, 1.0 / fps), 95))
    return rotation, rotation / 360.0, speed


def scale_from_height(height_m: float, standing_height_px: np.ndarray) -> float:
    samples = standing_height_px[np.isfinite(standing_height_px) & (standing_height_px > 0)]
    if height_m <= 0 or samples.size < 3:
        raise ValueError("Invalid height calibration data")
    return float(height_m / np.median(samples))
