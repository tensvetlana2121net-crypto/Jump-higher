import numpy as np
import pytest

from jumpbot.cv.phases import detect_phases


def test_detects_takeoff_and_landing_at_start_of_sustained_runs() -> None:
    fps = 60.0
    hip = np.ones(90)
    hip[20:30] = np.linspace(1.0, 0.8, 10)
    hip[30:40] = np.linspace(0.8, 1.0, 10)
    flight_time = np.arange(30) / fps
    hip[40:70] = 1.0 + 2.4 * flight_time - 0.5 * 9.80665 * flight_time**2
    feet = np.full(90, 500.0)
    feet[40:70] = 480.0

    phases = detect_phases(hip, feet, fps, body_height_px=800)

    assert phases.takeoff == 40
    assert phases.landing == 70
    assert phases.takeoff < phases.apex < phases.landing


def test_detects_landing_when_ice_perspective_shifts_floor_level() -> None:
    fps = 60.0
    hip = np.ones(100)
    hip[20:30] = np.linspace(1.0, 0.8, 10)
    hip[30:40] = np.linspace(0.8, 1.0, 10)
    flight_time = np.arange(35) / fps
    hip[40:75] = 1.0 + 3.0 * flight_time - 0.5 * 9.80665 * flight_time**2
    hip[75:] = hip[74]
    feet = np.full(100, 500.0)
    feet[40:75] = 470.0
    feet[75:] = 485.0  # landing is higher in the image than the initial floor

    phases = detect_phases(hip, feet, fps, body_height_px=800)

    assert phases.takeoff == 40
    assert phases.landing == 75


def test_ignores_deeper_crouch_after_landing() -> None:
    fps = 30.0
    hip = np.ones(100)
    hip[25:35] = np.linspace(1.0, 0.85, 10)
    hip[35:45] = np.linspace(0.85, 1.0, 10)
    flight_time = np.arange(18) / fps
    hip[45:63] = 1.0 + 2.8 * flight_time - 0.5 * 9.80665 * flight_time**2
    hip[63:80] = np.linspace(0.95, 0.70, 17)  # deeper landing crouch
    hip[80:] = 0.70
    feet = np.full(100, 500.0)
    feet[45:63] = 475.0

    phases = detect_phases(hip, feet, fps, body_height_px=500)

    assert phases.takeoff == 45
    assert phases.landing == 63
    assert phases.countermovement_bottom < phases.takeoff


def test_uses_blade_deceleration_before_deep_crouch() -> None:
    fps = 30.0
    hip = np.ones(100)
    time = np.arange(44) / fps
    hip[20:64] = 1.0 + 2.0 * time - 0.5 * 3.0 * time**2
    hip[64:] = np.linspace(0.95, 0.70, 36)
    feet = np.full(100, 550.0)
    feet[20:52] = np.concatenate((np.linspace(540, 515, 12), np.linspace(515, 552, 20)))
    feet[52:55] = [552.0, 552.5, 553.0]
    feet[55:] = np.linspace(554, 580, 45)

    phases = detect_phases(hip, feet, fps, floor_y_px=550.0, body_height_px=500)

    assert phases.landing <= 54


def test_refines_implausibly_early_takeoff_from_flight_symmetry() -> None:
    fps = 30.0
    hip = np.ones(80)
    hip[20:41] = np.linspace(1.0, 1.3, 21)
    hip[41:49] = np.linspace(1.3, 1.5, 8)
    hip[49:57] = np.linspace(1.5, 1.25, 8)
    feet = np.full(80, 500.0)
    feet[36:57] = 480.0

    phases = detect_phases(hip, feet, fps, body_height_px=500)

    assert phases.apex == 48
    assert phases.landing == 57
    assert phases.takeoff == 39


def test_ignores_airborne_artifact_before_minimum_takeoff_frame() -> None:
    fps = 30.0
    hip = np.ones(100)
    hip[1:8] = np.linspace(1.0, 1.25, 7)
    hip[8:15] = np.linspace(1.25, 1.0, 7)
    hip[45:55] = np.linspace(1.0, 1.3, 10)
    hip[55:65] = np.linspace(1.3, 1.0, 10)
    feet = np.full(100, 500.0)
    feet[1:15] = 470.0
    feet[45:65] = 470.0

    phases = detect_phases(
        hip,
        feet,
        fps,
        body_height_px=500,
        minimum_takeoff_frame=10,
    )

    assert phases.takeoff >= 45
    assert phases.landing == 65


def test_rejects_flight_whose_apex_is_the_landing_frame() -> None:
    fps = 30.0
    hip = np.ones(60)
    hip[20:32] = np.linspace(1.0, 1.3, 12)
    feet = np.full(60, 500.0)
    feet[20:31] = 470.0

    with pytest.raises(ValueError, match="Landing was not detected"):
        detect_phases(hip, feet, fps, body_height_px=500)
