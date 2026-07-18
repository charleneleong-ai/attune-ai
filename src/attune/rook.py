"""Rook data-aggregator interface — mock now, live API later.

Rook (https://docs.tryrook.io) normalises wearable data (Garmin / Fitbit / Apple / Oura / ...)
into three health pillars: Physical, Sleep, Body. Our `wearable`-modality signals are exactly
that objective channel — the subjective voice/vision/self-report signals come from the check-in
layer, not a wearable, so they are deliberately not Rook.

This module emits a patient day as Rook-shaped payloads and ingests them back into `Signal`s.
The point is the interface: swap `to_rook_day` (the mock source) for the live Rook webhook and
nothing downstream changes.

Envelope shape follows Rook's documented structure: a per-pillar document with top-level
`version` / `data_structure` / `created_at`, the pillar nesting a `<pillar>_summaries` array of
daily summaries, and the ISO-8601 datetime standard `YYYY-MM-DDTHH:MM:SS.ssssss+TZ`. Field names
follow Rook's naming style; pin them against the API reference when wiring the real webhook.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from attune.concordance_engine.memory import Memory, Signal
from attune.packs.base import ConditionPack

ROOK_VERSION = 2
BASE_DATETIME = datetime(
    2026, 1, 1, tzinfo=UTC
)  # day index 0; keeps mock datetimes deterministic
WEARABLE_MODALITY = "wearable"
PILLAR_SHORT = {
    "physical_health": "physical",
    "sleep_health": "sleep",
    "body_health": "body",
}


@dataclass(frozen=True, slots=True)
class RookField:
    structure: str  # physical_health | sleep_health | body_health
    domain: str  # summary sub-object, e.g. heart_rate / oxygenation / glucose
    field: str  # Rook-style field name
    scale: float = 1.0  # signal value * scale = Rook value (e.g. hours -> seconds)


# Maps each objective signal onto its Rook home. Keys must stay in lockstep with the pack's
# `wearable` signals — `test_rook_mapping_covers_exactly_the_wearable_signals` enforces that.
ROOK_MAPPING: dict[str, RookField] = {
    "hrv": RookField("physical_health", "heart_rate", "hrv_rmssd_ms_number"),
    "resting_hr": RookField("physical_health", "heart_rate", "hr_resting_bpm_number"),
    "spo2": RookField("physical_health", "oxygenation", "saturation_percentage_number"),
    "sleep_hours": RookField(
        "sleep_health", "sleep_summary", "sleep_duration_seconds_int", scale=3600.0
    ),
    "glucose_variability": RookField(
        "body_health", "glucose", "glucose_variability_mg_dl_number"
    ),
}


def wearable_signal_keys(pack: ConditionPack) -> tuple[str, ...]:
    return tuple(
        spec.key for spec in pack.signals if spec.modality == WEARABLE_MODALITY
    )


def rook_datetime(day: int) -> str:
    # Rook standard: ISO-8601 with six-digit microseconds and timezone, e.g. ...T00:00:00.000000Z.
    return (BASE_DATETIME + timedelta(days=day)).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _summary(document: dict, structure: str) -> dict:
    return document[structure][f"{PILLAR_SHORT[structure]}_summaries"][0]


def _empty_document(structure: str, day: int, user_id: str) -> dict:
    when = rook_datetime(day)
    summary = {
        "metadata": {
            "datetime_string": when,
            "user_id_string": user_id,
            "sources_of_data_array": ["mock"],
        }
    }
    return {
        "version": ROOK_VERSION,
        "data_structure": structure,
        "created_at": when,
        structure: {f"{PILLAR_SHORT[structure]}_summaries": [summary]},
    }


def to_rook_day(
    memory: Memory, day: int, *, user_id: str = "mock-user"
) -> dict[str, dict]:
    """One Rook document per health pillar for the given day, keyed by data_structure."""
    day_values = {signal.key: signal.value for signal in memory.window(day, 1)}
    documents: dict[str, dict] = {}
    for key, mapping in ROOK_MAPPING.items():
        if key not in day_values:
            continue
        document = documents.setdefault(
            mapping.structure, _empty_document(mapping.structure, day, user_id)
        )
        section = _summary(document, mapping.structure).setdefault(mapping.domain, {})
        section[mapping.field] = round(day_values[key] * mapping.scale, 4)
    return documents


def signals_from_rook(
    documents: dict[str, dict], pack: ConditionPack, day: int
) -> list[Signal]:
    """Inverse of `to_rook_day`: recover typed Signals from Rook payloads."""
    axis_of = pack.axis_of
    signals = []
    for key, mapping in ROOK_MAPPING.items():
        document = documents.get(mapping.structure)
        if document is None:
            continue
        section = _summary(document, mapping.structure).get(mapping.domain, {})
        if mapping.field not in section:
            continue
        value = section[mapping.field] / mapping.scale
        signals.append(Signal(key, axis_of[key], value, day, source=WEARABLE_MODALITY))
    return signals


def ingest_daily_rook(
    memory: Memory, pack: ConditionPack, days: int, *, user_id: str = "mock-user"
) -> Memory:
    """Rebuild the wearable channel by round-tripping every day through Rook payloads.

    Subjective (check-in) signals pass through untouched; the wearable signals are re-derived from
    Rook documents — exactly what a live Rook webhook would deliver. In the mock the payloads are
    generated from `memory`, so the result matches the source: the training data is identical
    whether wearables come from the generator or from Rook, which is what makes the swap safe.
    """
    wearable = set(ROOK_MAPPING)
    rebuilt = Memory(
        [signal for signal in memory.signals if signal.key not in wearable]
    )
    for day in range(days):
        documents = to_rook_day(memory, day, user_id=user_id)
        for signal in signals_from_rook(documents, pack, day):
            rebuilt.add(signal)
    return rebuilt
