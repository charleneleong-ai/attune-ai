from __future__ import annotations

from dataclasses import dataclass

from attune.concordance_engine.concordance import baseline_history, robust_z
from attune.concordance_engine.engine import Engine
from attune.concordance_engine.memory import Memory


@dataclass(frozen=True, slots=True)
class MonitoringScores:
    recovery_capacity: float
    fatigue_risk: float
    anomaly_score: float
    medication_response: float
    visible_change: float
    mobility_change: float
    top_drivers: tuple[str, ...]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def modality_coverage(memory: Memory, *, day: int, span: int = 3) -> dict[str, int]:
    counts: dict[str, int] = {}
    for signal in memory.window(day, span):
        counts[signal.source] = counts.get(signal.source, 0) + 1
    return counts


def _latest_value(memory: Memory, key: str, day: int) -> float:
    series = [signal for signal in memory.series(key) if signal.day <= day]
    return series[-1].value if series else 0.0


def _signal_z(memory: Memory, key: str, day: int) -> float:
    series = memory.series(key)
    current = [signal for signal in series if signal.day <= day]
    if not current:
        return 0.0
    latest = current[-1]
    history = baseline_history(series, latest.day)
    return robust_z(latest.value, history)


def monitoring_scores(engine: Engine, *, day: int) -> MonitoringScores:
    finding = engine.reflect(day)
    load_score = _clip01(finding.load / 18.0)
    voice_fatigue = _clip01(_latest_value(engine.memory, "voice_fatigue", day))
    medication_tolerance = _clip01(
        _latest_value(engine.memory, "medication_tolerance", day)
    )
    visible_change = _clip01(_latest_value(engine.memory, "skin_wound_change", day))
    mobility_change = _clip01(_latest_value(engine.memory, "mobility_change", day))

    recovery_capacity = _clip01(1.0 - load_score - 0.25 * voice_fatigue)
    fatigue_risk = _clip01(load_score + 0.35 * voice_fatigue)
    anomaly_score = _clip01(load_score + 0.25 * visible_change + 0.2 * mobility_change)
    medication_response = _clip01(0.4 * medication_tolerance + 0.2 * anomaly_score)

    z_by_signal = {
        spec.key: abs(_signal_z(engine.memory, spec.key, day))
        for spec in engine.pack.signals
        if engine.memory.series(spec.key)
    }
    top_drivers = tuple(
        key.replace("_", " ")
        for key, _ in sorted(
            z_by_signal.items(), key=lambda item: item[1], reverse=True
        )[:4]
    )

    return MonitoringScores(
        recovery_capacity=recovery_capacity,
        fatigue_risk=fatigue_risk,
        anomaly_score=anomaly_score,
        medication_response=medication_response,
        visible_change=visible_change,
        mobility_change=mobility_change,
        top_drivers=top_drivers,
    )
