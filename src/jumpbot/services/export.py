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
        "takeoff_velocity_mps",
        "confidence_score",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for jump in jumps:
        writer.writerow({field: getattr(jump, field) for field in fields})
    return buffer.getvalue().encode("utf-8-sig")
