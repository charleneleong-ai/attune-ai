from attune.concordance_engine.engine import Engine
from attune.demo import (
    attunefm_profile_names,
    channel_label,
    demo_checkin_answer,
    first_concordant_day,
)
from attune.packs.veteran import VETERAN_PACK
from attune.synth import flare_window, generate

WINDOW = flare_window(90)


def test_demo_finds_first_concordant_day_in_the_flare_window():
    eng = Engine(VETERAN_PACK, generate(VETERAN_PACK, days=90, seed=2))
    day = first_concordant_day(eng, WINDOW.onset - 4, WINDOW.end + 1)
    assert day is not None
    assert WINDOW.onset <= day <= WINDOW.end


def test_demo_preflare_window_is_clean():
    # the days shown just before onset must be quiet, or "caught as the drift begins" is a lie
    eng = Engine(VETERAN_PACK, generate(VETERAN_PACK, days=90, seed=2))
    assert first_concordant_day(eng, WINDOW.onset - 4, WINDOW.onset - 1) is None


def test_channel_label_keeps_image_and_video_distinct():
    assert channel_label("vision") == "photo"
    assert channel_label("video") == "video"
    assert channel_label("audio") == "voice"
    assert channel_label("self_report") == "voice"


def test_demo_checkin_answer_turns_scores_into_patient_language():
    assert "wiped out" in demo_checkin_answer("voice_fatigue", 1.01)
    assert "Pain is getting in the way" in demo_checkin_answer("pain_interference", 0.9)
    assert "foggy" in demo_checkin_answer("cognitive_fog", 0.7)
    assert "upload a photo" in demo_checkin_answer("skin_wound_change", 0.21)
    assert "movement video" in demo_checkin_answer("mobility_change", 0.7)


def test_attunefm_demo_profiles_cover_work_hazard_and_chronic_illness():
    assert attunefm_profile_names() == (
        "office",
        "firefighter",
        "firefighter_asthma",
        "firefighter_recovery",
        "firefighter_dormant",
        "veteran",
        "autoimmune",
        "metabolic_pcos",
    )
