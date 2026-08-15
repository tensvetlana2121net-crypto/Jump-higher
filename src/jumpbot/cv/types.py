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
    height_flight_m: float
    height_displacement_m: float | None
    takeoff_velocity_mps: float | None
    max_propulsion_velocity_mps: float | None
    max_angular_velocity_dps: float | None
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
            "height_flight_cm": self.height_flight_m * 100,
            "height_displacement_cm": (
                self.height_displacement_m * 100
                if self.height_displacement_m is not None
                else None
            ),
            "takeoff_velocity_mps": self.takeoff_velocity_mps,
            "max_propulsion_velocity_mps": self.max_propulsion_velocity_mps,
            "max_angular_velocity_dps": self.max_angular_velocity_dps,
            "confidence_score": self.confidence_score,
            "quality_flags": self.quality_flags,
            "algorithm_version": "0.1.0",
        }
