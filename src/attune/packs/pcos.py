from __future__ import annotations

from attune.concordance_engine.safety import BASE_RED_KEYWORDS, SAMARITANS, EscalationContract
from attune.packs.axes import Axis
from attune.packs.base import BriefTemplate, ConditionPack, Coupling, Persona, SignalSpec

# The deep-demo pack: metabolic-first, on eMed's home turf. Mood is present but minor —
# promoting it to a first-class axis + adding a physiological axis yields the veteran pack.
PCOS_PACK = ConditionPack(
    name="pcos",
    signals=(
        SignalSpec("cycle_day", Axis.CYCLE, "self_report"),
        SignalSpec("meal_gi", Axis.METABOLIC, "vision"),
        SignalSpec("weight", Axis.METABOLIC, "wearable"),
        SignalSpec("acne_score", Axis.DERMATOLOGICAL, "vision"),
        SignalSpec("hirsutism_score", Axis.DERMATOLOGICAL, "vision"),
        SignalSpec("mood_valence", Axis.PSYCHOLOGICAL, "audio"),
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
            "ovulatory dysfunction (cycle irregularity)",
            "clinical hyperandrogenism (acne / hirsutism, vision-scored)",
            "metabolic context (weight / GI trend)",
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
    axis_weights={Axis.DERMATOLOGICAL: 1.0, Axis.METABOLIC: 1.0, Axis.CYCLE: 0.5},
)
