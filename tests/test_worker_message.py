import uuid

from jumpbot.cv.types import AnalysisResult, PhaseFrames
from jumpbot.worker import (
    _format_analysis_message,
    _result_card_png,
    _result_publication_keyboard,
)


def test_analysis_message_contains_extended_metrics() -> None:
    result = AnalysisResult(
        fps=60.0,
        frame_count=120,
        phases=PhaseFrames(0, 20, 30, 45, 60),
        flight_time_s=0.5,
        jump_height_m=0.30,
        height_flight_m=0.306,
        height_trajectory_m=0.29,
        height_ballistic_m=0.31,
        height_displacement_m=0.29,
        takeoff_velocity_mps=2.45,
        max_propulsion_velocity_mps=2.8,
        max_angular_velocity_dps=125.0,
        rotation_degrees=360.0,
        rotation_turns=1.0,
        rotation_direction="clockwise",
        takeoff_foot_angle_deg=18.0,
        landing_foot_angle_deg=-6.0,
        takeoff_inclination_deg=12.0,
        max_inclination_deg=18.0,
        confidence_score=0.91,
        jump_length_m=0.42,
    )

    message = _format_analysis_message(result)

    assert "30.0 см" in message
    assert "2.45 м/с" in message
    assert "2.80 м/с" in message
    assert "125.0 °/с" in message
    assert "21 об/мин" in message
    assert "360.0°" in message
    assert "1.00 оборота" in message
    assert "Направление вращения по одному 2D-видео" in message
    assert "Ориентация конька при отрыве относительно горизонта кадра: +18.0°" in message
    assert "Ориентация конька при приземлении относительно горизонта кадра: -6.0°" in message
    assert "91%" in message
    assert "42.0 см" in message

    card = _result_card_png(result)
    assert card.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(card) > 10_000


def test_result_publication_keyboard_contains_both_choices() -> None:
    jump_id = uuid.UUID("12345678-1234-5678-1234-567812345678")

    keyboard = _result_publication_keyboard(jump_id)

    buttons = [button for row in keyboard.inline_keyboard for button in row]
    assert [button.text for button in buttons] == [
        "Перенести в приложение",
        "Дождаться лучшего результата",
    ]
    assert buttons[0].callback_data == f"app:publish:{jump_id}"
    assert buttons[1].callback_data == f"app:keep:{jump_id}"
