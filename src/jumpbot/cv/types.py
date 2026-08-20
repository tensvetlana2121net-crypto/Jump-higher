from dataclasses import dataclass, field
from enum import StrEnum


class JumpPhase(StrEnum):
    STANDING = "standing"
    COUNTERMOVEMENT = "countermovement"
    PROPULSION = "propulsion"
    FLIGHT = "flight"
    LANDING = "landing"
    STABILIZATION = "stabilization"


@dataclass(frozen=True)
class Landmark:
    x_px: float
    y_px: float
    visibility: float


@dataclass(frozen=True)
class FramePose:
    frame: int
    time_s: float
    points: dict[str, Landmark]


@dataclass(frozen=True)
class PhaseFrames:
    start: int
    countermovement_bottom: int
    takeoff: int
    apex: int
    landing: int


@dataclass
class AnalysisResult:
    fps: float
    frame_count: int
    phases: PhaseFrames
    flight_time_s: float
    jump_height_m: float
    height_flight_m: float
    height_trajectory_m: float | None
    height_ballistic_m: float | None
    height_displacement_m: float | None
    takeoff_velocity_mps: float | None
    max_propulsion_velocity_mps: float | None
    max_angular_velocity_dps: float | None
    rotation_degrees: float | None
    rotation_turns: float | None
    rotation_direction: str | None
    takeoff_foot_angle_deg: float | None
    landing_foot_angle_deg: float | None
    takeoff_inclination_deg: float
    max_inclination_deg: float
    confidence_score: float
    quality_flags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "fps": self.fps,
            "frame_count": self.frame_count,
            "phases": {
                "start": self.phases.start,
                "countermovement_bottom": self.phases.countermovement_bottom,
                "takeoff": self.phases.takeoff,
                "apex": self.phases.apex,
                "landing": self.phases.landing,
            },
            "flight_time_s": self.flight_time_s,
            "jump_height_cm": self.jump_height_m * 100,
            "height_flight_cm": self.height_flight_m * 100,
            "height_trajectory_cm": (
                self.height_trajectory_m * 100 if self.height_trajectory_m is not None else None
            ),
            "height_ballistic_cm": (
                self.height_ballistic_m * 100 if self.height_ballistic_m is not None else None
            ),
            "height_displacement_cm": (
                self.height_displacement_m * 100 if self.height_displacement_m is not None else None
            ),
            "takeoff_velocity_mps": self.takeoff_velocity_mps,
            "max_propulsion_velocity_mps": self.max_propulsion_velocity_mps,
            "max_angular_velocity_dps": self.max_angular_velocity_dps,
            "rotation_frequency_rpm": (
                self.max_angular_velocity_dps / 6.0
                if self.max_angular_velocity_dps is not None
                else None
            ),
            "rotation_degrees": self.rotation_degrees,
            "rotation_turns": self.rotation_turns,
            "rotation_direction": self.rotation_direction,
            "takeoff_foot_angle_deg": self.takeoff_foot_angle_deg,
            "landing_foot_angle_deg": self.landing_foot_angle_deg,
            "takeoff_inclination_deg": self.takeoff_inclination_deg,
            "max_inclination_deg": self.max_inclination_deg,
            "confidence_score": self.confidence_score,
            "quality_flags": self.quality_flags,
            "algorithm_version": "0.4.0",
        }
