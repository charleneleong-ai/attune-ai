from __future__ import annotations

from dataclasses import dataclass, field

from attune.concordance_engine.safety import EscalationContract
from attune.packs.axes import Axis


@dataclass(frozen=True, slots=True)
class SignalSpec:
    key: str
    axis: Axis
    modality: str  # audio | vision | wearable | text | self_report
    # Synthetic-patient generation params (consumed by attune.synth, not the engine):
    normal: float = 0.0  # baseline mean
    noise: float = 1.0  # day-to-day sd
    flare: float = 0.0  # signed deterioration applied during a planted flare
    cyclic: bool = False  # a phase counter (e.g. cycle day) rather than a noisy level


@dataclass(frozen=True, slots=True)
class Coupling:
    signals: tuple[str, ...]  # signal keys hypothesised to move together
    lag_days: int
    description: str  # will seed the reflection pass (build-plan step) and name the pattern


@dataclass(frozen=True, slots=True)
class Criterion:
    axis: Axis  # which axis's signals this clinician section summarises
    label: str  # clinician-facing section name


@dataclass(frozen=True, slots=True)
class BriefTemplate:
    name: str  # "Rotterdam" | "Cardiometabolic + mental-health risk"
    criteria: tuple[Criterion, ...]  # each maps an axis's signals onto a clinician section


@dataclass(frozen=True, slots=True)
class Persona:
    register: str
    vocabulary_note: str
    voice: str = "verse"


@dataclass(frozen=True, slots=True)
class CheckinItem:
    signal_key: str  # the Signal this turn elicits; must match a SignalSpec.key on the pack
    prompt: str  # the spoken question, in the pack's persona register
    source: str = "self_report"  # how this turn captures the value: self_report | audio | vision
    optional: bool = False  # optional "show me" visual / skippable turn — voice stays the floor


@dataclass(frozen=True, slots=True)
class ConditionPack:
    name: str
    signals: tuple[SignalSpec, ...]
    couplings: tuple[Coupling, ...]
    brief: BriefTemplate
    persona: Persona
    escalation: EscalationContract
    checkin: tuple[CheckinItem, ...] = ()  # the daily voice-first routine
    axis_weights: dict[Axis, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        declared = {s.key for s in self.signals}
        for item in self.checkin:
            if item.signal_key not in declared:
                raise ValueError(
                    f"pack '{self.name}': check-in item '{item.signal_key}' is not a declared signal"
                )

    @property
    def axis_of(self) -> dict[str, Axis]:
        return {s.key: s.axis for s in self.signals}
