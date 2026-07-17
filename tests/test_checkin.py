import pytest

from attune.checkin import record_checkin
from attune.concordance_engine.safety import EscalationContract
from attune.packs.axes import Axis
from attune.packs.base import BriefTemplate, CheckinItem, ConditionPack, Persona, SignalSpec
from attune.packs.pcos import PCOS_PACK
from attune.packs.veteran import VETERAN_PACK


def test_pack_rejects_a_checkin_for_an_undeclared_signal():
    # the invariant is structural — a malformed pack fails at construction, not mid-check-in
    with pytest.raises(ValueError):
        ConditionPack(
            name="bad",
            signals=(SignalSpec("weight", Axis.METABOLIC, "wearable"),),
            couplings=(),
            brief=BriefTemplate("b", ()),
            persona=Persona("r", "v"),
            escalation=EscalationContract((), (), "", ""),
            checkin=(CheckinItem("nonexistent", "?"),),
        )


def test_record_checkin_types_and_sources_each_response():
    responses = {"sleep_hours": 4.0, "mood_valence": 3.0, "voice_affect": 0.2}
    by_key = {s.key: s for s in record_checkin(VETERAN_PACK, day=90, responses=responses)}
    assert by_key["sleep_hours"].source == "self_report"
    assert by_key["voice_affect"].source == "audio"
    assert all(s.day == 90 for s in by_key.values())


def test_optional_photo_turn_is_skipped_when_not_provided():
    # voice-only check-in still yields the spoken signals; the photo turns are optional
    signals = record_checkin(PCOS_PACK, day=90, responses={"mood_valence": 5.0, "voice_affect": 0.5})
    assert {s.key for s in signals} == {"mood_valence", "voice_affect"}


def test_missing_required_spoken_response_raises():
    with pytest.raises(KeyError):
        record_checkin(VETERAN_PACK, day=90, responses={"sleep_hours": 6.0})  # missing mood + affect
