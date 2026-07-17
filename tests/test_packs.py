import pytest

from attune.concordance_engine.engine import PACKS, load
from attune.packs.axes import Axis
from attune.packs.base import ConditionPack


@pytest.mark.parametrize("name", ["pcos", "veteran"])
def test_pack_is_self_consistent(name):
    pack = PACKS[name]
    assert isinstance(pack, ConditionPack)
    # signal keys must be unique — a collision would silently vanish in the axis map
    assert len(pack.axis_of) == len(pack.signals)
    # a deterministic crisis floor must exist — safety cannot depend on the LLM alone
    assert pack.escalation.red_keywords


def test_veteran_couples_mind_and_body():
    axes = {s.axis for s in PACKS["veteran"].signals}
    assert Axis.PSYCHOLOGICAL in axes
    assert Axis.PHYSIOLOGICAL in axes


def test_pcos_is_metabolic_first():
    axes = {s.axis for s in PACKS["pcos"].signals}
    assert Axis.METABOLIC in axes
    assert Axis.DERMATOLOGICAL in axes  # vision-scored hyperandrogenism


def test_load_returns_engine_bound_to_pack():
    assert load("pcos").pack.name == "pcos"
