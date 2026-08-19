from jumpbot.cv.types import AnalysisResult, PhaseFrames
from jumpbot.worker import _format_analysis_message


def test_analysis_message_contains_extended_metrics() -> None:
    result = AnalysisResult(
        fps=60.0,
        frame_count=120,
        phases=PhaseFrames(0, 20, 30, 45, 60),
        flight_time_s=0.5,
        height_flight_m=0.306,
        height_displacement_m=0.29,
        takeoff_velocity_mps=2.45,
        max_propulsion_velocity_mps=2.8,
        max_angular_velocity_dps=125.0,
        confidence_score=0.91,
    )

    message = _format_analysis_message(result)

    assert "30.6 см" in message
    assert "2.45 м/с" in message
    assert "2.80 м/с" in message
    assert "125.0 °/с" in message
    assert "91%" in message
