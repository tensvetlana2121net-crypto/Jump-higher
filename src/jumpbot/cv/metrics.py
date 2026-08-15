import numpy as np

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
    return np.rad2deg(np.gradient(np.unwrap(angle_rad), 1.0 / fps))


def scale_from_height(height_m: float, standing_height_px: np.ndarray) -> float:
    samples = standing_height_px[np.isfinite(standing_height_px) & (standing_height_px > 0)]
    if height_m <= 0 or samples.size < 3:
        raise ValueError("Invalid height calibration data")
    return float(height_m / np.median(samples))
