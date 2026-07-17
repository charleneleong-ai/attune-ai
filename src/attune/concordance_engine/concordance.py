from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from attune.concordance_engine.memory import Memory, Signal

MAD_TO_SIGMA = 0.6745  # scales median-absolute-deviation to a normal sigma
BASELINE_SPAN = 30  # days of personal history used as the baseline
Z_THRESHOLD = 1.5  # per-axis robust-z past which a signal counts as deviating


def robust_z(value: float, history: list[float]) -> float:
    if len(history) < 3:
        return 0.0
    med = median(history)
    mad = median([abs(x - med) for x in history]) or 1e-9
    return MAD_TO_SIGMA * (value - med) / mad


@dataclass(frozen=True, slots=True)
class AxisDeviation:
    axis: str
    z: float  # mean |robust z| across the axis's signals in the window


@dataclass(frozen=True, slots=True)
class ConcordanceFinding:
    day: int
    load: float  # weighted composite deviation
    axes: list[AxisDeviation]
    z_threshold: float = Z_THRESHOLD  # the threshold this finding was scored against

    @property
    def deviating_axes(self) -> list[str]:
        return [a.axis for a in self.axes if a.z >= self.z_threshold]

    @property
    def concordant(self) -> bool:  # >= 2 axes deviate together — specificity over one noisy channel
        return len(self.deviating_axes) >= 2


def baseline_history(series: list[Signal], before_day: int, span: int = BASELINE_SPAN) -> list[float]:
    return [s.value for s in series if s.day < before_day][-span:]


def deviations(
    mem: Memory, day: int, span: int, baseline_span: int, axis_of: dict[str, str]
) -> list[AxisDeviation]:
    cutoff = day - span + 1
    baselines: dict[str, list[float]] = {}  # per-key history, computed once even if the key recurs
    per_axis: dict[str, list[float]] = {}
    for s in mem.window(day, span):
        axis = axis_of.get(s.key)
        if axis is None:
            continue
        history = baselines.get(s.key)
        if history is None:
            history = baseline_history(mem.series(s.key), cutoff, baseline_span)
            baselines[s.key] = history
        per_axis.setdefault(axis, []).append(abs(robust_z(s.value, history)))
    return [AxisDeviation(axis, sum(zs) / len(zs)) for axis, zs in per_axis.items()]


def concordance(
    mem: Memory,
    day: int,
    axis_of: dict[str, str],
    *,
    span: int = 3,
    baseline_span: int = BASELINE_SPAN,
    z_threshold: float = Z_THRESHOLD,
    weights: dict[str, float] | None = None,
) -> ConcordanceFinding:
    weights = weights or {}
    devs = deviations(mem, day, span, baseline_span, axis_of)
    load = sum(weights.get(d.axis, 1.0) * d.z for d in devs)
    return ConcordanceFinding(day, load, devs, z_threshold)
