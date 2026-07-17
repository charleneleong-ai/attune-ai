from attune.concordance_engine.brief import build_brief
from attune.packs.attunefm import ATTUNEFM_PACK
from attune.packs.pcos import PCOS_PACK
from attune.packs.veteran import VETERAN_PACK
from attune.reporting import render
from attune.synth import generate

FLARE_DAY = 80
CALM_DAY = 50


def test_brief_flags_the_concordant_axes_on_a_flare_day():
    brief = build_brief(
        VETERAN_PACK, generate(VETERAN_PACK, days=90, seed=2), FLARE_DAY
    )
    assert brief.concordant
    salient = {c.axis for c in brief.criteria if c.salient}
    assert {"physiological", "psychological"} <= salient
    assert "metabolic" not in salient  # uncoupled spine stays quiet


def test_brief_calm_day_recommends_monitoring():
    # a lone noisy signal may blip past threshold; the headline is that it is not concordant
    brief = build_brief(VETERAN_PACK, generate(VETERAN_PACK, days=90, seed=2), CALM_DAY)
    assert not brief.concordant
    assert "monitoring" in brief.recommendation


def test_render_maps_criteria_to_concrete_evidence():
    text = render(
        build_brief(PCOS_PACK, generate(PCOS_PACK, days=90, seed=1), FLARE_DAY)
    )
    assert "Rotterdam" in text
    assert "hyperandrogenism" in text  # a Rotterdam criterion label
    assert "acne_score" in text  # the concrete signal backing it


def test_attunefm_brief_flags_match_profile_concordance_axes():
    brief = build_brief(
        ATTUNEFM_PACK,
        generate(ATTUNEFM_PACK, days=90, profile="firefighter"),
        FLARE_DAY,
    )

    salient = {c.axis for c in brief.criteria if c.salient}

    assert salient == {"physiological", "behavioral"}


def test_attunefm_pcos_brief_does_not_flag_incidental_work_signal():
    brief = build_brief(
        ATTUNEFM_PACK,
        generate(ATTUNEFM_PACK, days=90, profile="metabolic_pcos"),
        FLARE_DAY,
    )

    salient = {c.axis for c in brief.criteria if c.salient}

    assert salient == {"physiological", "metabolic", "pain", "dermatological"}
