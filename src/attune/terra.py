"""Terra data-aggregator interface — mock now, live API later.

Terra (https://docs.tryterra.co) normalises wearable data (Garmin / Fitbit / Apple / Oura / ...)
into typed payloads (Daily, Sleep, Body, Activity, ...). Our `wearable`-modality signals are that
objective channel — the subjective voice/vision/self-report signals come from the check-in layer,
not a wearable, so they are deliberately not Terra.

This module emits a patient day as Terra-shaped webhook payloads and ingests them back into
`Signal`s. The point is the interface: swap `to_terra_day` (the mock source) for Terra's live
webhook and nothing downstream changes.

High fidelity: each payload carries Terra's real structure — the webhook envelope (`type` / `user`
/ `data`), a model object with `metadata`, the daily `summary` scalar AND the intraday `*_samples`
array Terra exposes for that signal (`heart_rate_data.detailed.hr_samples`, `oxygen_data.
saturation_samples`, `glucose_data.blood_glucose_samples`, ...). When the source has real intraday
readings (the generator's `intraday` mode), they ride in the sample array and `signals_from_terra`
recovers them, so intraday features are derivable from Terra alone; otherwise the daily value is
replicated across the timestamps. Field names and paths follow Terra's data-model version; pin them
against the API reference when wiring the webhook.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from attune.concordance_engine.memory import Memory, Signal
from attune.packs.base import ConditionPack

TERRA_VERSION = "2022_03_16"  # Terra data-model version tag carried on every payload
BASE_DATETIME = datetime(
    2026, 1, 1, tzinfo=UTC
)  # day index 0; keeps mock datetimes deterministic
WEARABLE_MODALITY = "wearable"
SAMPLES_PER_DAY = (
    6  # intraday points emitted per signal — enough to be a faithful *_samples array
)


@dataclass(frozen=True, slots=True)
class TerraField:
    data_type: str  # Terra payload type: daily | sleep | body
    summary_path: tuple[
        str, ...
    ]  # nested keys to the daily scalar within the model object
    sample_path: tuple[
        str, ...
    ]  # nested keys to the intraday sample array ("" -> no samples)
    sample_key: str  # value field inside each sample object (Terra's per-sample naming)
    # affine unit map: Terra value = signal value * scale + offset (hours -> seconds; a normalised
    # dysregulation index -> a physiological mg/dL range). Inverted on the way back.
    scale: float = 1.0
    offset: float = 0.0


# Maps each objective signal onto its Terra home. Keys must stay in lockstep with the pack's
# `wearable` signals — `test_terra_mapping_covers_exactly_the_wearable_signals` enforces that.
TERRA_MAPPING: dict[str, TerraField] = {
    "hrv": TerraField(
        "daily",
        ("heart_rate_data", "summary", "avg_hrv_rmssd"),
        ("heart_rate_data", "detailed", "hrv_samples_rmssd"),
        "hrv_rmssd",
    ),
    "resting_hr": TerraField(
        "daily",
        ("heart_rate_data", "summary", "resting_hr_bpm"),
        ("heart_rate_data", "detailed", "hr_samples"),
        "bpm",
    ),
    "spo2": TerraField(
        "daily",
        ("oxygen_data", "avg_saturation_percentage"),
        ("oxygen_data", "saturation_samples"),
        "percentage",
    ),
    "sleep_hours": TerraField(
        "sleep",
        ("sleep_durations_data", "asleep", "duration_asleep_state_seconds"),
        (),
        "",
        scale=3600.0,
    ),
    # Terra has no glucose-variability field, so we surface this dysregulation index (rises in
    # flares) as day-average glucose, affine-scaled into a physiological mg/dL range (~80-180).
    "glucose_variability": TerraField(
        "body",
        ("glucose_data", "day_avg_blood_glucose_mg_per_dL"),
        ("glucose_data", "blood_glucose_samples"),
        "blood_glucose_mg_per_dL",
        scale=130.0,
        offset=70.0,
    ),
}


def wearable_signal_keys(pack: ConditionPack) -> tuple[str, ...]:
    return tuple(
        spec.key for spec in pack.signals if spec.modality == WEARABLE_MODALITY
    )


def terra_datetime(day: int) -> str:
    return _iso(BASE_DATETIME + timedelta(days=day))


def terra_date(day: int) -> str:
    # the date-only form Terra's REST query params take (start_date / end_date)
    return (BASE_DATETIME + timedelta(days=day)).date().isoformat()


def terra_day(iso_date: str) -> int:
    # inverse of terra_date: an ISO date back to our day index
    return (date.fromisoformat(iso_date) - BASE_DATETIME.date()).days


def _iso(when: datetime) -> str:
    # Terra ISO-8601: full timestamp with microseconds + timezone, e.g. ...T00:00:00.000000Z.
    return when.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _set_path(obj: dict, path: tuple[str, ...], value: object) -> None:
    for key in path[:-1]:
        obj = obj.setdefault(key, {})
    obj[path[-1]] = value


def _get_path(obj: dict, path: tuple[str, ...]) -> object | None:
    for key in path:
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return obj


def _sample_array(values: tuple[float, ...], day: int, sample_key: str) -> list[dict]:
    base = BASE_DATETIME + timedelta(days=day)
    step = timedelta(hours=24 / len(values))
    return [
        {"timestamp": _iso(base + step * i), sample_key: value}
        for i, value in enumerate(values)
    ]


def terra_envelope(
    data_type: str, *, user_id: str, provider: str, data: list[dict]
) -> dict:
    """The Terra webhook envelope every payload shares: type + version + user + data array.

    One home for the envelope shape and the version tag, so ingest payloads and the serving
    prediction speak the exact same wrapper.
    """
    return {
        "type": data_type,
        "version": TERRA_VERSION,
        "user": {"user_id": user_id, "provider": provider, "reference_id": None},
        "data": data,
    }


def _payload(data_type: str, day: int, user_id: str) -> dict:
    return terra_envelope(
        data_type,
        user_id=user_id,
        provider="mock",
        data=[
            {
                "metadata": {
                    "start_time": terra_datetime(day),
                    "end_time": terra_datetime(day + 1),
                    "upload_type": 0,
                }
            }
        ],
    )


def _model(payload: dict) -> dict:
    return payload["data"][0]


def to_terra_day(
    memory: Memory, day: int, *, user_id: str = "mock-user"
) -> dict[str, dict]:
    """One Terra webhook payload per data type for the given day, keyed by `type`."""
    day_signals = {signal.key: signal for signal in memory.window(day, 1)}
    payloads: dict[str, dict] = {}
    for key, mapping in TERRA_MAPPING.items():
        signal = day_signals.get(key)
        if signal is None:
            continue
        model = _model(
            payloads.setdefault(
                mapping.data_type, _payload(mapping.data_type, day, user_id)
            )
        )
        _set_path(
            model,
            mapping.summary_path,
            round(signal.value * mapping.scale + mapping.offset, 4),
        )
        if mapping.sample_path:
            # carry the real intraday readings when present, else the daily value replicated
            raw = signal.samples or (signal.value,) * SAMPLES_PER_DAY
            samples = tuple(
                round(reading * mapping.scale + mapping.offset, 4) for reading in raw
            )
            _set_path(
                model,
                mapping.sample_path,
                _sample_array(samples, day, mapping.sample_key),
            )
    return payloads


def signals_from_terra(
    payloads: dict[str, dict], pack: ConditionPack, day: int
) -> list[Signal]:
    """Inverse of `to_terra_day`: recover typed Signals from the daily summary of each payload."""
    axis_of = pack.axis_of
    signals = []
    for key, mapping in TERRA_MAPPING.items():
        payload = payloads.get(mapping.data_type)
        if payload is None:
            continue
        model = _model(payload)
        value = _get_path(model, mapping.summary_path)
        if value is None:
            continue
        raw = _get_path(model, mapping.sample_path) if mapping.sample_path else None
        samples = (
            tuple(
                (point[mapping.sample_key] - mapping.offset) / mapping.scale
                for point in raw
            )
            if raw
            else ()
        )
        signals.append(
            Signal(
                key,
                axis_of[key],
                (value - mapping.offset) / mapping.scale,
                day,
                source=WEARABLE_MODALITY,
                samples=samples,
            )
        )
    return signals


def ingest_daily_terra(
    memory: Memory, pack: ConditionPack, days: int, *, user_id: str = "mock-user"
) -> Memory:
    """Rebuild the wearable channel by round-tripping every day through Terra payloads.

    Subjective (check-in) signals pass through untouched; the wearable signals are re-derived from
    Terra payloads — exactly what a live Terra webhook would deliver. In the mock the payloads are
    generated from `memory`, so the result matches the source: the training data is identical
    whether wearables come from the generator or from Terra, which is what makes the swap safe.
    """
    wearable = set(TERRA_MAPPING)
    rebuilt = Memory(
        [signal for signal in memory.signals if signal.key not in wearable]
    )
    for day in range(days):
        payloads = to_terra_day(memory, day, user_id=user_id)
        for signal in signals_from_terra(payloads, pack, day):
            rebuilt.add(signal)
    return rebuilt
