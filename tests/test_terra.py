import pytest

from attune.concordance_engine.engine import PACKS
from attune.synth import generate
from attune.terra import (
    SAMPLES_PER_DAY,
    TERRA_MAPPING,
    TERRA_VERSION,
    ingest_daily_terra,
    signals_from_terra,
    to_terra_day,
    wearable_signal_keys,
)

PACK = PACKS["attunefm"]


def _snapshot(memory):
    return sorted((s.key, s.day, s.source, round(s.value, 4)) for s in memory.signals)


def test_terra_mapping_covers_exactly_the_wearable_signals():
    # Terra is the objective wearable channel; every wearable signal must have a home, and nothing
    # subjective (voice/vision/self-report) should leak into it.
    assert set(TERRA_MAPPING) == set(wearable_signal_keys(PACK))


def test_terra_day_has_terra_envelope():
    memory = generate(PACK, days=90, profile="veteran")
    payloads = to_terra_day(memory, day=50, user_id="u-1")
    daily = payloads["daily"]
    assert daily["type"] == "daily"
    assert daily["version"] == TERRA_VERSION
    assert daily["user"]["user_id"] == "u-1"
    model = daily["data"][0]
    # Terra datetime standard: full ISO-8601 with microseconds + timezone
    when = model["metadata"]["start_time"]
    assert when.startswith("2026-") and "T" in when and when.endswith("Z")
    # two heart-rate signals share the daily HR summary
    summary = model["heart_rate_data"]["summary"]
    assert "avg_hrv_rmssd" in summary
    assert "resting_hr_bpm" in summary


def test_terra_day_carries_intraday_sample_arrays():
    # high fidelity: the payload exposes Terra's real *_samples arrays, not just daily scalars
    memory = generate(PACK, days=90, profile="veteran")
    payloads = to_terra_day(memory, day=50)
    detailed = payloads["daily"]["data"][0]["heart_rate_data"]["detailed"]
    assert len(detailed["hr_samples"]) == SAMPLES_PER_DAY
    assert all("timestamp" in s and "bpm" in s for s in detailed["hr_samples"])
    assert all("hrv_rmssd" in s for s in detailed["hrv_samples_rmssd"])
    glucose = payloads["body"]["data"][0]["glucose_data"]["blood_glucose_samples"]
    assert len(glucose) == SAMPLES_PER_DAY
    assert all("blood_glucose_mg_per_dL" in s for s in glucose)


def test_terra_roundtrip_preserves_wearable_signal_values():
    memory = generate(PACK, days=90, profile="firefighter")
    payloads = to_terra_day(memory, day=40)
    recovered = {
        signal.key: round(signal.value, 4)
        for signal in signals_from_terra(payloads, PACK, day=40)
    }
    original = {
        signal.key: round(signal.value, 4)
        for signal in memory.window(40, 1)
        if signal.key in TERRA_MAPPING
    }
    assert recovered == original
    assert all(
        signal.source == "wearable"
        for signal in signals_from_terra(payloads, PACK, day=40)
    )


def test_ingest_daily_terra_reproduces_the_source_memory():
    # sourcing the wearable channel through Terra must yield the same data as the generator,
    # so the mock->live swap can't change what the model trains or serves on
    memory = generate(PACK, days=90, profile="veteran")
    rebuilt = ingest_daily_terra(memory, PACK, days=90)
    assert _snapshot(rebuilt) == _snapshot(memory)


def test_terra_carries_and_recovers_real_intraday_samples():
    # with intraday generation on, the payload carries the *varied* readings (not a flat replica),
    # and signals_from_terra recovers them — so intraday features are derivable from Terra alone
    memory = generate(PACK, days=90, profile="veteran", intraday=True)
    source = next(s for s in memory.window(50, 1) if s.key == "resting_hr")
    payloads = to_terra_day(memory, 50)
    samples = payloads["daily"]["data"][0]["heart_rate_data"]["detailed"]["hr_samples"]
    assert (
        len({point["bpm"] for point in samples}) > 1
    )  # genuinely varied, not constant
    recovered = next(
        s for s in signals_from_terra(payloads, PACK, 50) if s.key == "resting_hr"
    )
    assert recovered.samples == pytest.approx(source.samples, abs=1e-3)
