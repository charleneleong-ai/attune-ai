from __future__ import annotations

from attune.concordance_engine.safety import BASE_RED_KEYWORDS, SAMARITANS, EscalationContract
from attune.packs.axes import Axis
from attune.packs.base import BriefTemplate, ConditionPack, Coupling, Persona, SignalSpec

# The generalization proof: PCOS's metabolic spine + a first-class psychological axis
# coupled to a physiological axis, and a higher-acuity escalation contract. Same engine.
VETERAN_PACK = ConditionPack(
    name="veteran",
    signals=(
        SignalSpec("hrv", Axis.PHYSIOLOGICAL, "wearable"),
        SignalSpec("resting_hr", Axis.PHYSIOLOGICAL, "wearable"),
        SignalSpec("sleep_hours", Axis.PHYSIOLOGICAL, "wearable"),
        SignalSpec("weight", Axis.METABOLIC, "wearable"),
        SignalSpec("voice_affect", Axis.PSYCHOLOGICAL, "audio"),
        SignalSpec("mood_valence", Axis.PSYCHOLOGICAL, "audio"),
        SignalSpec("engagement", Axis.BEHAVIORAL, "text"),  # skipped/short check-ins = withdrawal
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
            "autonomic load (HRV / resting HR / sleep trend)",
            "psychological drift (voice affect / engagement)",
            "cardiometabolic spine (weight / BP)",
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
    axis_weights={Axis.PSYCHOLOGICAL: 1.5, Axis.PHYSIOLOGICAL: 1.5, Axis.METABOLIC: 1.0},
)
