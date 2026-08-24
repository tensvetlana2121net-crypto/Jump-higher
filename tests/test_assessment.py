import pytest

from jumpbot.cv.assessment import (
    assess_declared_rotation,
    compare_with_personal_baseline,
    expected_airborne_turns,
)


@pytest.mark.parametrize(
    ("jump_type", "expected"),
    [("2_axel", 2.5), ("3_lutz", 3.0), ("4_toe_loop", 4.0)],
)
def test_expected_airborne_turns(jump_type: str, expected: float) -> None:
    assert expected_airborne_turns(jump_type) == expected


def test_rotation_assessment_is_conservative() -> None:
    assessment = assess_declared_rotation(
        {
            "rotation_turns": 2.1,
            "confidence_score": 0.9,
            "fps": 60.0,
            "quality_flags": [],
        },
        "2_axel",
    )

    assert assessment["status"] == "possible_rotation_deficit"
    assert assessment["estimated_deficit_turns"] == pytest.approx(0.4)
    assert assessment["is_official_judgement"] is False


def test_rotation_assessment_stays_inconclusive_for_low_fps() -> None:
    assessment = assess_declared_rotation(
        {
            "rotation_turns": 1.2,
            "confidence_score": 0.95,
            "fps": 30.0,
            "quality_flags": [],
        },
        "2_lutz",
    )

    assert assessment["status"] == "inconclusive"
    assert assessment["estimated_deficit_turns"] is None


def test_personal_baseline_uses_median_and_marks_large_drop() -> None:
    previous = [
        {
            "jump_height_cm": height,
            "flight_time_s": 0.50,
            "max_angular_velocity_dps": 900.0,
            "confidence_score": 0.9,
        }
        for height in (39.0, 40.0, 41.0, 80.0)
    ]
    comparison = compare_with_personal_baseline(
        {
            "jump_height_cm": 32.0,
            "flight_time_s": 0.49,
            "max_angular_velocity_dps": 880.0,
            "confidence_score": 0.9,
        },
        previous,
    )

    assert comparison is not None
    assert comparison["baseline"]["jump_height_cm"] == pytest.approx(40.5)
    assert "possible_lower_jump_height" in comparison["signals"]


def test_personal_baseline_requires_three_good_attempts() -> None:
    assert (
        compare_with_personal_baseline(
            {"jump_height_cm": 30.0, "confidence_score": 0.9},
            [
                {"jump_height_cm": 31.0, "confidence_score": 0.9},
                {"jump_height_cm": 32.0, "confidence_score": 0.9},
            ],
        )
        is None
    )
