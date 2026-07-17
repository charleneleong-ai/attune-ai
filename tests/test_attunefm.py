from attune.attunefm import (
    dataset_grounding,
    monitoring_answer,
    monitoring_scores,
    modality_coverage,
)
from attune.concordance_engine.engine import Engine, PACKS
from attune.datasets import DEMO_DATASET_NAMES
from attune.packs.axes import Axis
from attune.synth import ATTUNEFM_PROFILES, flare_window, generate


def test_attunefm_pack_covers_occupational_chronic_and_visible_modalities():
    pack = PACKS["attunefm"]
    sources = {signal.modality for signal in pack.signals}
    axes = {signal.axis for signal in pack.signals}

    assert {"wearable", "audio", "vision", "video", "text", "self_report"}.issubset(
        sources
    )
    assert {
        Axis.PHYSIOLOGICAL,
        Axis.PSYCHOLOGICAL,
        Axis.METABOLIC,
        Axis.BEHAVIORAL,
        Axis.PAIN,
        Axis.COGNITIVE,
    }.issubset(axes)
    assert any("medication" in item.prompt.lower() for item in pack.checkin)


def test_modality_coverage_counts_recent_signals_by_source():
    pack = PACKS["attunefm"]
    memory = generate(pack, days=90, seed=4)
    day = flare_window(90).midpoint

    coverage = modality_coverage(memory, day=day, span=3)

    assert coverage["wearable"] > 0
    assert coverage["audio"] > 0
    assert coverage["vision"] > 0
    assert coverage["video"] > 0


def test_monitoring_scores_surface_recovery_anomaly_and_drivers():
    pack = PACKS["attunefm"]
    engine = Engine(pack, generate(pack, days=90, seed=5))
    day = flare_window(90).midpoint

    scores = monitoring_scores(engine, day=day)

    assert 0.0 <= scores.recovery_capacity <= 1.0
    assert scores.fatigue_risk > 0.5
    assert scores.anomaly_score > 0.5
    assert scores.visible_change > 0.0
    assert scores.mobility_change > 0.0
    assert len(scores.top_drivers) >= 2


def test_monitoring_scores_are_grounded_by_real_multimodal_datasets():
    pack = PACKS["attunefm"]
    engine = Engine(pack, generate(pack, days=90, seed=6))
    day = flare_window(90).midpoint

    datasets = dataset_grounding(engine)
    scores = monitoring_scores(engine, day=day)
    dataset_names = {dataset.name for dataset in datasets}

    assert {
        "BIDSleep",
        "WESAD",
        "CGMacros",
        "Bridge2AI-Voice",
        "DDI",
        "PAMAP2",
    }.issubset(dataset_names)
    assert scores.grounding_datasets == tuple(dataset.name for dataset in datasets)


def test_all_attunefm_profiles_include_current_demo_dataset_grounding():
    pack = PACKS["attunefm"]
    day = flare_window(90).midpoint

    for profile in ATTUNEFM_PROFILES:
        engine = Engine(pack, generate(pack, days=90, profile=profile))
        scores = monitoring_scores(engine, day=day)

        assert set(DEMO_DATASET_NAMES).issubset(scores.grounding_datasets)


def test_monitoring_answer_aligns_recommendation_with_scores_and_drivers():
    pack = PACKS["attunefm"]
    engine = Engine(pack, generate(pack, days=90, profile="firefighter_recovery"))
    day = flare_window(90).midpoint

    answer = monitoring_answer(engine, day=day)

    assert answer.headline == "Potential answer"
    assert any("Recovery capacity" in stat for stat in answer.stats)
    assert any("fatigue risk" in stat and "(high)" in stat for stat in answer.stats)
    assert "multi-axis drift" in answer.interpretation
    assert "axes over threshold" in answer.interpretation
    assert "Grounding references include" in answer.interpretation
    assert "ask focused follow-up" in answer.recommendation
    assert "care-team summary" in answer.recommendation


def test_veteran_monitoring_answer_surfaces_hidden_chronic_load():
    pack = PACKS["attunefm"]
    engine = Engine(pack, generate(pack, days=90, profile="veteran"))
    day = flare_window(90).midpoint

    answer = monitoring_answer(engine, day=day)

    assert "multi-axis drift" in answer.interpretation
    assert "pain interference" in answer.interpretation
    assert any("anomaly score" in stat and "(high)" in stat for stat in answer.stats)


def test_office_monitoring_answer_recommends_workload_and_posture_followup():
    pack = PACKS["attunefm"]
    engine = Engine(pack, generate(pack, days=90, profile="office"))
    day = flare_window(90).midpoint

    answer = monitoring_answer(engine, day=day)

    assert "lighten the work block" in answer.recommendation
    assert "posture" in answer.recommendation


def test_metabolic_pcos_monitoring_answer_recommends_metabolic_followup():
    pack = PACKS["attunefm"]
    engine = Engine(pack, generate(pack, days=90, profile="metabolic_pcos"))
    day = flare_window(90).midpoint

    answer = monitoring_answer(engine, day=day)

    assert "meal, sleep, and glucose pattern" in answer.recommendation
    assert "diet response" in answer.recommendation
