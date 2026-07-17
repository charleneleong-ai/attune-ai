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

# The generalization proof: PCOS's metabolic spine + a first-class psychological axis
# coupled to a physiological axis, and a higher-acuity escalation contract. Same engine.
VETERAN_PACK = ConditionPack(
    name="veteran",
    signals=(
        SignalSpec("hrv", Axis.PHYSIOLOGICAL, "wearable", normal=60, noise=4, flare=-18),
        SignalSpec("resting_hr", Axis.PHYSIOLOGICAL, "wearable", normal=58, noise=3),
        SignalSpec("sleep_hours", Axis.PHYSIOLOGICAL, "wearable", normal=7.5, noise=0.6, flare=-3.2),
        SignalSpec("weight", Axis.METABOLIC, "wearable", normal=88, noise=0.4),
        SignalSpec("voice_affect", Axis.PSYCHOLOGICAL, "audio", normal=0.65, noise=0.08, flare=-0.45),
        SignalSpec("mood_valence", Axis.PSYCHOLOGICAL, "audio", normal=0.6, noise=0.08),
        # skipped / short check-ins = withdrawal
        SignalSpec("engagement", Axis.BEHAVIORAL, "text", normal=1.0, noise=0.1),
    ),
    couplings=(
        Coupling(
            ("sleep_hours", "hrv", "voice_affect"),
            lag_days=2,
            description="poor sleep + HRV drop + flat affect precede a bad week by ~2 days",
        ),
    ),
    brief=BriefTemplate(
        "Cardiometabolic + mental-health risk",
        criteria=(
            Criterion(Axis.PHYSIOLOGICAL, "autonomic load — HRV / resting HR / sleep"),
            Criterion(Axis.PSYCHOLOGICAL, "psychological drift — voice affect / engagement"),
            Criterion(Axis.METABOLIC, "cardiometabolic spine — weight"),
        ),
    ),
    persona=Persona(
        register="peer, mission-framed, plain",
        vocabulary_note="no wellness vocabulary; direct and respectful",
    ),
    escalation=EscalationContract(
        red_keywords=(*BASE_RED_KEYWORDS, "no way out"),
        handoff_targets=("on-call clinician", SAMARITANS, "veterans' crisis line"),
        amber_action="increase support + draft clinician brief; propose grounding",
        red_action="pause management, de-escalation script, warm human handoff",
    ),
    checkin=(
        CheckinItem("sleep_hours", "How many hours did you actually get last night?"),
        CheckinItem("mood_valence", "Where's your head at today?"),
        CheckinItem("voice_affect", "How's the week landing on you?", source="audio"),
    ),
    axis_weights={Axis.PSYCHOLOGICAL: 1.5, Axis.PHYSIOLOGICAL: 1.5, Axis.METABOLIC: 1.0},
)
