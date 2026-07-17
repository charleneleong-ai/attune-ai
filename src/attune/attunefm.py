from __future__ import annotations

from dataclasses import dataclass

from attune.concordance_engine.concordance import baseline_history, robust_z
from attune.concordance_engine.engine import Engine
from attune.concordance_engine.memory import Memory
from attune.datasets import DatasetStub, datasets_for_modality
from attune.packs.axes import Axis


SOURCE_MODALITIES = {
    "audio": "voice",
    "self_report": "context",
    "text": "context",
    "video": "video",
    "vision": "image",
    "wearable": "wearable",
}


@dataclass(frozen=True, slots=True)
class MonitoringScores:
    recovery_capacity: float
    fatigue_risk: float
    anomaly_score: float
    medication_response: float
    visible_change: float
    mobility_change: float
    top_drivers: tuple[str, ...]
    grounding_datasets: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MonitoringAnswer:
    headline: str
    stats: tuple[str, ...]
    interpretation: str
    recommendation: str


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def modality_coverage(memory: Memory, *, day: int, span: int = 3) -> dict[str, int]:
    counts: dict[str, int] = {}
    for signal in memory.window(day, span):
        counts[signal.source] = counts.get(signal.source, 0) + 1
    return counts


def dataset_grounding(engine: Engine) -> tuple[DatasetStub, ...]:
    modalities = {
        SOURCE_MODALITIES.get(signal.modality, signal.modality)
        for signal in engine.pack.signals
    }
    if any(signal.axis is Axis.METABOLIC for signal in engine.pack.signals):
        modalities.add("metabolic")

    by_name = {
        dataset.name: dataset
        for modality in modalities
        for dataset in datasets_for_modality(modality)
    }
    return tuple(by_name[name] for name in sorted(by_name))


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
    grounding_datasets = tuple(dataset.name for dataset in dataset_grounding(engine))

    return MonitoringScores(
        recovery_capacity=recovery_capacity,
        fatigue_risk=fatigue_risk,
        anomaly_score=anomaly_score,
        medication_response=medication_response,
        visible_change=visible_change,
        mobility_change=mobility_change,
        top_drivers=top_drivers,
        grounding_datasets=grounding_datasets,
    )


def _percent(value: float) -> str:
    return f"{round(100 * value):.0f}%"


def _risk_band(value: float) -> str:
    if value >= 0.7:
        return "high"
    if value >= 0.4:
        return "moderate"
    return "low"


def _suggested_next_step(axes: set[str], drivers: tuple[str, ...]) -> str:
    driver_text = ", ".join(drivers[:2]) or "the top driver"
    if "metabolic" in axes:
        return (
            "review the recent meal, sleep, and glucose pattern, ask a focused "
            f"follow-up on {driver_text}, and prepare a concise care-team summary "
            "if the pattern persists or symptoms worsen."
        )
    if "dermatological" in axes:
        return (
            "ask for a photo comparison and medication/symptom follow-up, track "
            f"{driver_text}, and prepare a concise care-team summary if the change persists."
        )
    if "behavioral" in axes and "pain" not in axes:
        return (
            "lighten the work block, add a recovery or posture check, and ask a "
            f"focused follow-up on {driver_text}."
        )
    if "pain" in axes or "cognitive" in axes:
        return (
            "reduce load where possible, ask a focused follow-up on pain, brain fog, "
            f"or {driver_text}, and prepare a concise care-team summary if the pattern persists."
        )
    return (
        "pause heavy exertion, check recovery and breathing symptoms, and prepare a "
        "concise care-team summary if the pattern persists or worsens."
    )


def monitoring_answer(engine: Engine, *, day: int) -> MonitoringAnswer:
    scores = monitoring_scores(engine, day=day)
    finding = engine.reflect(day)
    brief = engine.brief(day)
    drivers = ", ".join(scores.top_drivers[:3]) or "recent wearable and check-in drift"
    datasets = ", ".join(scores.grounding_datasets[:4])
    status = "multi-axis drift" if finding.concordant else "baseline-level variation"

    stats = (
        f"Recovery capacity {_percent(scores.recovery_capacity)}",
        f"fatigue risk {_percent(scores.fatigue_risk)} ({_risk_band(scores.fatigue_risk)})",
        f"anomaly score {_percent(scores.anomaly_score)} ({_risk_band(scores.anomaly_score)})",
        f"medication-response watch {_percent(scores.medication_response)}",
    )
    interpretation = (
        f"Today looks like {status}: load {finding.load:.1f} with "
        f"{len(finding.deviating_axes)} axes over threshold. The strongest contributors are {drivers}. "
        f"Grounding references include {datasets}."
    )
    recommendation = (
        f"{brief.recommendation}. Suggested demo answer: "
        f"{_suggested_next_step(set(finding.deviating_axes), scores.top_drivers)}"
    )

    return MonitoringAnswer(
        headline="Potential answer",
        stats=stats,
        interpretation=interpretation,
        recommendation=recommendation,
    )
