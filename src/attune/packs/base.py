from __future__ import annotations

from dataclasses import dataclass, field

from attune.concordance_engine.safety import EscalationContract
from attune.packs.axes import Axis


@dataclass(frozen=True, slots=True)
class SignalSpec:
    key: str
    axis: Axis
    modality: str  # audio | vision | wearable | text | self_report


@dataclass(frozen=True, slots=True)
class Coupling:
    signals: tuple[str, ...]  # signal keys hypothesised to move together
    lag_days: int
    description: str  # will seed the reflection pass (build-plan step) and name the pattern


@dataclass(frozen=True, slots=True)
class BriefTemplate:
    name: str  # "Rotterdam" | "Cardiometabolic + mental-health risk"
    criteria: tuple[str, ...]  # sections the longitudinal memory maps onto


@dataclass(frozen=True, slots=True)
class Persona:
    register: str
    vocabulary_note: str
    voice: str = "verse"


@dataclass(frozen=True, slots=True)
class ConditionPack:
    name: str
    signals: tuple[SignalSpec, ...]
    couplings: tuple[Coupling, ...]
    brief: BriefTemplate
    persona: Persona
    escalation: EscalationContract
    axis_weights: dict[Axis, float] = field(default_factory=dict)

    @property
    def axis_of(self) -> dict[str, Axis]:
        return {s.key: s.axis for s in self.signals}
