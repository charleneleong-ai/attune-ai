from __future__ import annotations

from attune.concordance_engine.safety import BASE_RED_KEYWORDS, SAMARITANS, EscalationContract
from attune.packs.axes import Axis
from attune.packs.base import (
    BriefTemplate,
    CheckinItem,
    ConditionPack,
    Coupling,
    Criterion,
    Persona,
    SignalSpec,
)

# The deep-demo pack: metabolic-first, on eMed's home turf. Mood is present but minor —
# promoting it to a first-class axis + adding a physiological axis yields the veteran pack.
PCOS_PACK = ConditionPack(
    name="pcos",
    signals=(
        SignalSpec("cycle_day", Axis.CYCLE, "self_report", cyclic=True),
        SignalSpec("meal_gi", Axis.METABOLIC, "vision", normal=45, noise=8, flare=+30),
        SignalSpec("weight", Axis.METABOLIC, "wearable", normal=86, noise=0.4),
        SignalSpec("acne_score", Axis.DERMATOLOGICAL, "vision", normal=3, noise=1, flare=+4),
        SignalSpec("hirsutism_score", Axis.DERMATOLOGICAL, "vision", normal=4, noise=0.5),
        SignalSpec("mood_valence", Axis.PSYCHOLOGICAL, "audio", normal=0.6, noise=0.08),
        SignalSpec("voice_affect", Axis.PSYCHOLOGICAL, "audio", normal=0.6, noise=0.08),
    ),
    couplings=(
        Coupling(
            ("cycle_day", "meal_gi", "acne_score"),
            lag_days=2,
            description="luteal phase + high-GI meals precede acne and energy flares",
        ),
    ),
    brief=BriefTemplate(
        "Rotterdam",
        criteria=(
            Criterion(Axis.CYCLE, "ovulatory dysfunction — cycle irregularity"),
            Criterion(Axis.DERMATOLOGICAL, "clinical hyperandrogenism — acne / hirsutism (vision-scored)"),
            Criterion(Axis.METABOLIC, "metabolic context — weight / GI trend"),
        ),
    ),
    persona=Persona(
        register="warm, validating, plain-language",
        vocabulary_note="supportive; avoid clinical jargon",
    ),
    escalation=EscalationContract(
        red_keywords=BASE_RED_KEYWORDS,
        handoff_targets=("GP referral", SAMARITANS),
        amber_action="increase support + draft GP brief (androgen panel + pelvic ultrasound)",
        red_action="pause coaching, surface resources, warm handoff",
    ),
    checkin=(
        CheckinItem("mood_valence", "How's your energy and mood today?"),
        CheckinItem("voice_affect", "How are you feeling in yourself this week?", source="audio"),
        CheckinItem("acne_score", "Want to show me your skin? Totally optional.", source="vision", optional=True),
        CheckinItem("meal_gi", "Snap your last meal if you fancy it — no pressure.", source="vision", optional=True),
    ),
    axis_weights={Axis.DERMATOLOGICAL: 1.0, Axis.METABOLIC: 1.0, Axis.CYCLE: 0.5},
)
