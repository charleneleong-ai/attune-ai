import pytest

from attune.concordance_engine.concordance import concordance, robust_z
from attune.concordance_engine.memory import Memory, Signal
from attune.concordance_engine.safety import Tier
from attune.packs.axes import Axis
from attune.packs.veteran import VETERAN_PACK

AXIS_OF = {
    "sleep_hours": Axis.PHYSIOLOGICAL,
    "hrv": Axis.PHYSIOLOGICAL,
    "voice_affect": Axis.PSYCHOLOGICAL,
}
BASELINE = {"sleep_hours": 7.5, "hrv": 60.0, "voice_affect": 0.6}


@pytest.fixture
def seeded():
    mem = Memory()
    for key, base in BASELINE.items():
        for day in range(30):
            mem.add(Signal(key, AXIS_OF[key], base, day))
    return mem


def deteriorate(mem, day, values):
    for key, value in values.items():
        mem.add(Signal(key, AXIS_OF[key], value, day))


def test_robust_z_flat_history_is_zero():
    assert robust_z(5.0, [5.0, 5.0, 5.0]) == 0.0


def test_concordant_mind_body_drop_fires(seeded):
    # both axes deteriorate together on day 30 → the specific, actionable signal
    deteriorate(seeded, 30, {"sleep_hours": 4.0, "hrv": 40.0, "voice_affect": 0.1})
    finding = concordance(seeded, 30, AXIS_OF, span=1, z_threshold=1.5)
    assert finding.concordant
    assert finding.load > 0


def test_single_axis_blip_does_not_fire(seeded):
    # physiology drops but affect holds → one axis only → not concordant (false-alarm guard)
    deteriorate(seeded, 30, {"sleep_hours": 4.0, "hrv": 40.0, "voice_affect": 0.6})
    finding = concordance(seeded, 30, AXIS_OF, span=1, z_threshold=1.5)
    assert not finding.concordant


def test_amber_promotes_to_red_on_crisis_language(seeded):
    from attune.concordance_engine.engine import Engine

    eng = Engine(VETERAN_PACK, seeded)
    deteriorate(seeded, 30, {"sleep_hours": 4.0, "hrv": 40.0, "voice_affect": 0.1})
    verdict = eng.assess("honestly I want to end it", day=30)
    assert verdict.tier is Tier.RED
    assert verdict.triggered_by == "deterministic"
