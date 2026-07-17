from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Signal:
    key: str  # e.g. "hrv", "acne_score", "meal_gi", "mood_valence"
    axis: str  # opaque pack-owned label (see attune.packs.axes.Axis)
    value: float
    day: int  # day index in the patient's timeline
    source: str = "self_report"  # audio | vision | wearable | text | self_report
    note: str = ""


@dataclass
class Memory:
    signals: list[Signal] = field(default_factory=list)

    def add(self, signal: Signal) -> None:
        self.signals.append(signal)

    def series(self, key: str) -> list[Signal]:
        return sorted((s for s in self.signals if s.key == key), key=lambda s: s.day)

    def window(self, day: int, span: int) -> list[Signal]:
        lo = day - span + 1
        return [s for s in self.signals if lo <= s.day <= day]
