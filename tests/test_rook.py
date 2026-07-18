from attune.concordance_engine.engine import PACKS
from attune.rook import (
    ROOK_MAPPING,
    ingest_daily_rook,
    signals_from_rook,
    to_rook_day,
    wearable_signal_keys,
)
from attune.synth import generate

PACK = PACKS["attunefm"]


def _snapshot(memory):
    return sorted((s.key, s.day, s.source, round(s.value, 4)) for s in memory.signals)


def test_rook_mapping_covers_exactly_the_wearable_signals():
    # Rook is the objective wearable channel; every wearable signal must have a home, and nothing
    # subjective (voice/vision/self-report) should leak into it.
    assert set(ROOK_MAPPING) == set(wearable_signal_keys(PACK))


def test_rook_day_has_rook_envelope():
    memory = generate(PACK, days=90, profile="veteran")
    documents = to_rook_day(memory, day=50, user_id="u-1")
    physical = documents["physical_health"]
    assert physical["version"] == 2
    assert physical["data_structure"] == "physical_health"
    # pillars nest a *_summaries array, per Rook's documented access pattern
    summary = physical["physical_health"]["physical_summaries"][0]
    assert summary["metadata"]["user_id_string"] == "u-1"
    # Rook datetime standard: full ISO-8601 with microseconds + timezone
    when = summary["metadata"]["datetime_string"]
    assert when.startswith("2026-") and "T" in when and when.endswith("Z")
    # two heart-rate signals share one Rook domain
    heart_rate = summary["heart_rate"]
    assert "hrv_rmssd_ms_number" in heart_rate
    assert "hr_resting_bpm_number" in heart_rate


def test_rook_roundtrip_preserves_wearable_signal_values():
    memory = generate(PACK, days=90, profile="firefighter")
    documents = to_rook_day(memory, day=40)
    recovered = {
        signal.key: round(signal.value, 4)
        for signal in signals_from_rook(documents, PACK, day=40)
    }
    original = {
        signal.key: round(signal.value, 4)
        for signal in memory.window(40, 1)
        if signal.key in ROOK_MAPPING
    }
    assert recovered == original
    assert all(
        signal.source == "wearable"
        for signal in signals_from_rook(documents, PACK, day=40)
    )


def test_ingest_daily_rook_reproduces_the_source_memory():
    # sourcing the wearable channel through Rook must yield the same data as the generator,
    # so the mock->live swap can't change what the model trains or serves on
    memory = generate(PACK, days=90, profile="veteran")
    rebuilt = ingest_daily_rook(memory, PACK, days=90)
    assert _snapshot(rebuilt) == _snapshot(memory)
