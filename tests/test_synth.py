import pytest

from attune.concordance_engine.engine import Engine
from attune.packs.attunefm import ATTUNEFM_PACK
from attune.packs.pcos import PCOS_PACK
from attune.packs.veteran import VETERAN_PACK
from attune.synth import ATTUNEFM_PROFILES, flare_window, generate, load_memory, save

WINDOW = flare_window(90)
FLARE_DAY = WINDOW.midpoint
CALM_DAY = 50


def test_generate_is_deterministic():
    assert (
        generate(VETERAN_PACK, seed=7).signals == generate(VETERAN_PACK, seed=7).signals
    )


@pytest.mark.parametrize("pack", [PCOS_PACK, VETERAN_PACK])
def test_planted_flare_is_concordant_and_calm_days_are_quiet(pack):
    eng = Engine(pack, generate(pack, days=90, seed=2))
    assert not eng.reflect(day=CALM_DAY).concordant
    assert eng.reflect(day=FLARE_DAY).concordant


def test_roundtrip_preserves_signals(tmp_path):
    mem = generate(PCOS_PACK, seed=3)
    path = tmp_path / "pcos.json"
    save(mem, path)
    assert load_memory(path).signals == mem.signals


def test_attunefm_profiles_create_distinct_demo_values():
    day = FLARE_DAY
    memories = {
        name: generate(ATTUNEFM_PACK, days=90, profile=name)
        for name in ATTUNEFM_PROFILES
    }

    office = {signal.key: signal.value for signal in memories["office"].window(day, 1)}
    firefighter = {
        signal.key: signal.value for signal in memories["firefighter"].window(day, 1)
    }
    firefighter_asthma = {
        signal.key: signal.value
        for signal in memories["firefighter_asthma"].window(day, 1)
    }
    firefighter_recovery = {
        signal.key: signal.value
        for signal in memories["firefighter_recovery"].window(day, 1)
    }
    firefighter_dormant = {
        signal.key: signal.value
        for signal in memories["firefighter_dormant"].window(day, 1)
    }
    veteran = {
        signal.key: signal.value for signal in memories["veteran"].window(day, 1)
    }
    autoimmune = {
        signal.key: signal.value for signal in memories["autoimmune"].window(day, 1)
    }
    metabolic = {
        signal.key: signal.value for signal in memories["metabolic_pcos"].window(day, 1)
    }

    assert office["work_burden"] > firefighter["work_burden"]
    assert office["cognitive_fog"] > firefighter["cognitive_fog"]
    assert firefighter["breathlessness_report"] > office["breathlessness_report"]
    assert (
        firefighter_asthma["breathlessness_report"]
        > firefighter["breathlessness_report"]
    )
    assert firefighter_asthma["spo2"] < firefighter["spo2"]
    assert (
        firefighter_asthma["medication_tolerance"] > firefighter["medication_tolerance"]
    )
    assert firefighter_recovery["voice_fatigue"] > firefighter["voice_fatigue"]
    assert firefighter_recovery["sleep_hours"] < firefighter["sleep_hours"]
    assert (
        firefighter_recovery["breathlessness_report"]
        < firefighter_asthma["breathlessness_report"]
    )
    assert (
        firefighter_dormant["breathlessness_report"]
        > autoimmune["breathlessness_report"]
    )
    assert firefighter_dormant["pain_interference"] > firefighter["pain_interference"]
    assert firefighter_dormant["skin_wound_change"] > firefighter["skin_wound_change"]
    assert veteran["pain_interference"] > office["pain_interference"]
    assert veteran["cognitive_fog"] > firefighter["cognitive_fog"]
    assert veteran["sleep_hours"] < firefighter["sleep_hours"]
    assert autoimmune["pain_interference"] > office["pain_interference"]
    assert autoimmune["skin_wound_change"] > firefighter["skin_wound_change"]
    assert metabolic["glucose_variability"] > office["glucose_variability"]
    assert metabolic["diet_response"] > firefighter["diet_response"]
    assert metabolic["food_photo_risk"] > autoimmune["food_photo_risk"]


def test_attunefm_profiles_stay_quiet_before_the_planted_flare():
    for name in ATTUNEFM_PROFILES:
        eng = Engine(ATTUNEFM_PACK, generate(ATTUNEFM_PACK, days=90, profile=name))

        assert not eng.reflect(day=WINDOW.onset - 4).concordant
        assert eng.reflect(day=FLARE_DAY).concordant


@pytest.mark.parametrize(
    ("profile", "expected_axes"),
    [
        ("office", {"behavioral", "cognitive"}),
        ("firefighter", {"physiological", "behavioral"}),
        (
            "firefighter_asthma",
            {"physiological", "psychological", "behavioral"},
        ),
        (
            "firefighter_recovery",
            {"physiological", "psychological", "behavioral", "pain", "cognitive"},
        ),
        (
            "firefighter_dormant",
            {"physiological", "psychological", "pain", "cognitive", "dermatological"},
        ),
        (
            "veteran",
            {"physiological", "psychological", "behavioral", "pain", "cognitive"},
        ),
        (
            "autoimmune",
            {"physiological", "psychological", "pain", "cognitive", "dermatological"},
        ),
        ("metabolic_pcos", {"physiological", "metabolic", "pain", "dermatological"}),
    ],
)
def test_attunefm_profile_flare_axes_are_profile_specific(profile, expected_axes):
    eng = Engine(ATTUNEFM_PACK, generate(ATTUNEFM_PACK, days=90, profile=profile))

    axes = set(eng.reflect(day=FLARE_DAY).deviating_axes)

    assert eng.reflect(day=FLARE_DAY).concordant
    assert axes == expected_axes


@pytest.mark.parametrize(
    ("profile", "expected_axes"),
    [
        ("office", {"behavioral", "cognitive"}),
        ("firefighter", {"physiological", "behavioral"}),
        (
            "firefighter_asthma",
            {"physiological", "psychological", "behavioral"},
        ),
        (
            "firefighter_recovery",
            {"physiological", "psychological", "behavioral", "pain", "cognitive"},
        ),
        (
            "firefighter_dormant",
            {"physiological", "psychological", "pain", "cognitive", "dermatological"},
        ),
        (
            "veteran",
            {"physiological", "psychological", "behavioral", "pain", "cognitive"},
        ),
        (
            "autoimmune",
            {"physiological", "psychological", "pain", "cognitive", "dermatological"},
        ),
        ("metabolic_pcos", {"physiological", "metabolic", "pain", "dermatological"}),
    ],
)
def test_attunefm_profile_display_window_has_no_unrelated_axes(profile, expected_axes):
    eng = Engine(ATTUNEFM_PACK, generate(ATTUNEFM_PACK, days=90, profile=profile))

    for day in range(WINDOW.onset, WINDOW.end + 2):
        axes = set(eng.reflect(day=day).deviating_axes)
        assert axes <= expected_axes


def test_office_profile_does_not_flag_unrelated_visible_or_metabolic_axes():
    eng = Engine(ATTUNEFM_PACK, generate(ATTUNEFM_PACK, days=90, profile="office"))

    axes = set(eng.reflect(day=FLARE_DAY).deviating_axes)

    assert "behavioral" in axes
    assert "cognitive" in axes
    assert "dermatological" not in axes
    assert "metabolic" not in axes
