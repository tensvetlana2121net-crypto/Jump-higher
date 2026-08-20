import numpy as np

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
