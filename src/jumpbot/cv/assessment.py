from collections.abc import Mapping, Sequence
from statistics import median
from typing import Any


def expected_airborne_turns(jump_type: str) -> float | None:
    """Return nominal airborne body turns from the user-declared jump label.

    Axel starts facing forward and therefore contains an additional half turn.
    This is a reference value, not an automated judging decision.
    """
    rotation_text, separator, jump_name = jump_type.partition("_")
    if not separator or jump_name not in {
        "axel",
        "loop",
        "salchow",
        "flip",
        "lutz",
        "toe_loop",
    }:
        return None
    try:
        rotations = int(rotation_text)
    except ValueError:
        return None
    if rotations not in {1, 2, 3, 4}:
        return None
    return rotations + 0.5 if jump_name == "axel" else float(rotations)


def assess_declared_rotation(metrics: Mapping[str, Any], jump_type: str) -> dict[str, Any]:
    """Conservatively compare measured and declared rotation.

    A monocular 2D estimate must never be represented as an official technical-panel
    call.  Low-confidence and low-frame-rate attempts remain explicitly inconclusive.
    """
    expected = expected_airborne_turns(jump_type)
    measured = metrics.get("rotation_turns")
    measured_turns = float(measured) if isinstance(measured, (int, float)) else None
    confidence = float(metrics.get("confidence_score") or 0.0)
    fps = float(metrics.get("fps") or 0.0)
    flags = set(metrics.get("quality_flags") or [])
    reliable = (
        expected is not None
        and measured_turns is not None
        and confidence >= 0.75
        and fps >= 50
        and "unstable_trunk_orientation" not in flags
        and "implausible_rotation_speed" not in flags
    )
    status = "inconclusive"
    deficit = None
    if reliable:
        assert expected is not None and measured_turns is not None
        deficit = max(0.0, expected - measured_turns)
        status = "possible_rotation_deficit" if deficit >= 0.25 else "within_2d_tolerance"
    return {
        "status": status,
        "declared_jump_type": jump_type,
        "expected_airborne_turns": expected,
        "measured_turns": measured_turns,
        "estimated_deficit_turns": deficit,
        "is_official_judgement": False,
        "limitations": "Monocular 2D coaching indicator; not an ISU technical-panel call.",
    }


def _number(payload: Mapping[str, Any], key: str) -> float | None:
    value = payload.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def compare_with_personal_baseline(
    current: Mapping[str, Any],
    previous: Sequence[Mapping[str, Any]],
    *,
    minimum_samples: int = 3,
    maximum_samples: int = 5,
) -> dict[str, Any] | None:
    """Compare an attempt with robust medians of recent like-for-like attempts."""
    usable = [
        payload
        for payload in previous[:maximum_samples]
        if float(payload.get("confidence_score") or 0.0) >= 0.65
    ]
    if len(usable) < minimum_samples:
        return None

    keys = ("jump_height_cm", "flight_time_s", "max_angular_velocity_dps")
    baselines: dict[str, float] = {}
    changes: dict[str, float] = {}
    for key in keys:
        samples = [value for payload in usable if (value := _number(payload, key)) is not None]
        current_value = _number(current, key)
        if len(samples) < minimum_samples or current_value is None:
            continue
        baseline = float(median(samples))
        if baseline <= 0:
            continue
        baselines[key] = baseline
        changes[key] = 100.0 * (current_value - baseline) / baseline

    signals: list[str] = []
    if float(current.get("confidence_score") or 0.0) >= 0.7:
        if changes.get("jump_height_cm", 0.0) <= -15.0:
            signals.append("possible_lower_jump_height")
        if changes.get("flight_time_s", 0.0) <= -12.0:
            signals.append("possible_shorter_flight")
        if changes.get("max_angular_velocity_dps", 0.0) <= -15.0:
            signals.append("possible_slower_rotation")

    return {
        "sample_size": len(usable),
        "baseline_method": "median_recent_same_jump_type",
        "baseline": {key: round(value, 4) for key, value in baselines.items()},
        "change_percent": {key: round(value, 1) for key, value in changes.items()},
        "signals": signals,
        "is_diagnostic": False,
    }
