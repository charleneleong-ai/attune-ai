from __future__ import annotations

from attune.concordance_engine.safety import (
    BASE_RED_KEYWORDS,
    SAMARITANS,
    EscalationContract,
)
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


ATTUNEFM_PACK = ConditionPack(
    name="attunefm",
    signals=(
        SignalSpec(
            "hrv", Axis.PHYSIOLOGICAL, "wearable", normal=58, noise=4, flare=-18
        ),
        SignalSpec(
            "resting_hr", Axis.PHYSIOLOGICAL, "wearable", normal=62, noise=3, flare=+8
        ),
        SignalSpec(
            "sleep_hours",
            Axis.PHYSIOLOGICAL,
            "wearable",
            normal=7.4,
            noise=0.6,
            flare=-2.8,
        ),
        SignalSpec(
            "spo2", Axis.PHYSIOLOGICAL, "wearable", normal=97.5, noise=0.4, flare=-1.2
        ),
        SignalSpec(
            "glucose_variability",
            Axis.METABOLIC,
            "wearable",
            normal=0.25,
            noise=0.05,
            flare=+0.28,
        ),
        SignalSpec(
            "diet_response",
            Axis.METABOLIC,
            "self_report",
            normal=0.2,
            noise=0.06,
            flare=+0.35,
        ),
        SignalSpec(
            "medication_tolerance",
            Axis.PHYSIOLOGICAL,
            "self_report",
            normal=0.1,
            noise=0.04,
            flare=+0.45,
        ),
        SignalSpec(
            "voice_fatigue",
            Axis.PSYCHOLOGICAL,
            "audio",
            normal=0.18,
            noise=0.06,
            flare=+0.62,
        ),
        SignalSpec(
            "breathlessness_report",
            Axis.PHYSIOLOGICAL,
            "audio",
            normal=0.08,
            noise=0.04,
            flare=+0.32,
        ),
        SignalSpec(
            "work_burden", Axis.BEHAVIORAL, "text", normal=0.35, noise=0.08, flare=+0.45
        ),
        SignalSpec(
            "pain_interference",
            Axis.PAIN,
            "self_report",
            normal=0.12,
            noise=0.05,
            flare=+0.55,
        ),
        SignalSpec(
            "cognitive_fog",
            Axis.COGNITIVE,
            "self_report",
            normal=0.14,
            noise=0.05,
            flare=+0.48,
        ),
        SignalSpec(
            "engagement", Axis.BEHAVIORAL, "text", normal=0.9, noise=0.08, flare=-0.4
        ),
        SignalSpec(
            "skin_wound_change",
            Axis.DERMATOLOGICAL,
            "vision",
            normal=0.05,
            noise=0.03,
            flare=+0.52,
        ),
        SignalSpec(
            "food_photo_risk",
            Axis.METABOLIC,
            "vision",
            normal=0.25,
            noise=0.08,
            flare=+0.35,
        ),
        SignalSpec(
            "mobility_change",
            Axis.PHYSIOLOGICAL,
            "video",
            normal=0.08,
            noise=0.04,
            flare=+0.46,
        ),
        SignalSpec(
            "posture_strain",
            Axis.BEHAVIORAL,
            "video",
            normal=0.22,
            noise=0.07,
            flare=+0.38,
        ),
    ),
    couplings=(
        Coupling(
            ("sleep_hours", "hrv", "voice_fatigue", "work_burden", "cognitive_fog"),
            lag_days=2,
            description="workload and poor sleep precede recovery, fatigue, and brain-fog deterioration",
        ),
        Coupling(
            (
                "medication_tolerance",
                "resting_hr",
                "pain_interference",
                "skin_wound_change",
                "mobility_change",
            ),
            lag_days=1,
            description="medication or lifestyle changes align with pain, visible, and functional change",
        ),
    ),
    brief=BriefTemplate(
        "Occupational + chronic-health monitoring",
        criteria=(
            Criterion(Axis.PHYSIOLOGICAL, "recovery and autonomic load"),
            Criterion(Axis.PSYCHOLOGICAL, "voice and symptom burden"),
            Criterion(Axis.METABOLIC, "diet, glucose, and medication response"),
            Criterion(Axis.BEHAVIORAL, "work pattern and functional capacity"),
            Criterion(Axis.PAIN, "pain interference and flare burden"),
            Criterion(Axis.COGNITIVE, "brain fog, focus, and cognitive fatigue"),
            Criterion(Axis.DERMATOLOGICAL, "visible skin, wound, or swelling change"),
        ),
    ),
    persona=Persona(
        register="plain, nurse-like, careful, non-diagnostic",
        vocabulary_note="monitoring language; avoid diagnosis, treatment, or fitness-for-work claims",
    ),
    escalation=EscalationContract(
        red_keywords=(*BASE_RED_KEYWORDS, "chest pain", "can't breathe", "collapsed"),
        handoff_targets=("care team", "occupational-health clinician", SAMARITANS),
        amber_action="ask focused follow-up + draft monitoring summary for care team",
        red_action="pause coaching, de-escalate, and warm-handoff to urgent human support",
    ),
    checkin=(
        CheckinItem(
            "voice_fatigue", "How much energy do you have today?", source="audio"
        ),
        CheckinItem(
            "medication_tolerance",
            "Any medication changes or side effects since yesterday?",
        ),
        CheckinItem("work_burden", "How heavy does today's work or life load feel?"),
        CheckinItem(
            "pain_interference",
            "Is pain getting in the way of work, movement, or rest today?",
        ),
        CheckinItem(
            "cognitive_fog", "How clear or foggy does your thinking feel today?"
        ),
        CheckinItem(
            "skin_wound_change",
            "Want to show me any rash, swelling, wound, or skin change?",
            source="vision",
            optional=True,
        ),
        CheckinItem(
            "mobility_change",
            "Want to share a short movement or posture check video?",
            source="video",
            optional=True,
        ),
    ),
    axis_weights={
        Axis.PHYSIOLOGICAL: 1.5,
        Axis.PSYCHOLOGICAL: 1.2,
        Axis.METABOLIC: 1.1,
        Axis.BEHAVIORAL: 1.0,
        Axis.PAIN: 1.0,
        Axis.COGNITIVE: 0.9,
        Axis.DERMATOLOGICAL: 0.8,
    },
)
