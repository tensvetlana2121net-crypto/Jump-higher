import csv
import io

from jumpbot.db.models import JumpHistory


def jumps_to_csv(jumps: list[JumpHistory]) -> bytes:
    buffer = io.StringIO()
    fields = [
        "id",
        "created_at",
        "height_flight_cm",
        "height_displacement_cm",
        "flight_time_ms",
        "takeoff_velocity_mps",
        "max_angular_velocity_dps",
        "rotation_degrees",
        "rotation_turns",
        "takeoff_inclination_deg",
        "confidence_score",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for jump in jumps:
        row = {
            field: getattr(jump, field)
            for field in fields
            if field not in {"rotation_degrees", "rotation_turns", "takeoff_inclination_deg"}
        }
        metrics = jump.metric_data or {}
        row.update(
            {
                "rotation_degrees": metrics.get("rotation_degrees"),
                "rotation_turns": metrics.get("rotation_turns"),
                "takeoff_inclination_deg": metrics.get("takeoff_inclination_deg"),
            }
        )
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8-sig")
