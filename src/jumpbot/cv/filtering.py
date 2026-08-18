import numpy as np


def interpolate_short_gaps(values: np.ndarray, max_gap: int = 5) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    valid = np.isfinite(result)
    if valid.sum() < 2:
        raise ValueError("At least two valid samples are required")

    indexes = np.arange(result.size)
    interpolated = np.interp(indexes, indexes[valid], result[valid])
    missing = ~valid
    starts = np.flatnonzero(missing & np.r_[True, ~missing[:-1]])
    ends = np.flatnonzero(missing & np.r_[~missing[1:], True])
    for start, end in zip(starts, ends, strict=True):
        if end - start + 1 <= max_gap:
            result[start : end + 1] = interpolated[start : end + 1]
    return result


def hampel_filter(values: np.ndarray, window: int = 7, sigma: float = 3.0) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    radius = max(1, window // 2)
    for index in range(result.size):
        left, right = max(0, index - radius), min(result.size, index + radius + 1)
        sample = result[left:right]
        median = np.nanmedian(sample)
        deviations = np.abs(sample - median)
        mad = np.nanmedian(deviations)
        if mad > 0:
            threshold = sigma * 1.4826 * mad
        else:
            nonzero = deviations[deviations > 0]
            threshold = sigma * np.nanmedian(nonzero) if nonzero.size else 0.0
        if np.isfinite(result[index]) and threshold > 0 and abs(result[index] - median) > threshold:
            result[index] = median
    return result


def smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    size = min(max(1, window), data.size)
    if size == 1:
        return data.copy()
    kernel = np.ones(size) / size
    left = size // 2
    right = size - 1 - left
    return np.convolve(np.pad(data, (left, right), mode="edge"), kernel, mode="valid")


def clean_trajectory(values: np.ndarray, max_gap: int = 5, window: int = 5) -> np.ndarray:
    interpolated = interpolate_short_gaps(values, max_gap=max_gap)
    if np.isnan(interpolated).any():
        raise ValueError("Trajectory contains a gap that is too long")
    return smooth(hampel_filter(interpolated), window=window)
