"""Seeded synthetic patients — the demo fixture the reflection pass discovers live.

Generates a per-pack longitudinal history with a *planted concordant flare*: signals carrying a
nonzero `flare` deteriorate together over a short window, so `Engine.reflect()` fires
`concordant` inside the flare and stays quiet on calm days. Deterministic given a seed, so the
demo patient is reproducible (files are gitignored — regenerate with `python -m attune.synth`).
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import typer
from rich import print as rprint

from attune.concordance_engine.engine import PACKS
from attune.concordance_engine.memory import Memory, Signal
from attune.packs.base import ConditionPack, SignalSpec

CYCLE_DAYS = 28  # canonical cycle length for cyclic phase signals (e.g. cycle_day)


@dataclass(frozen=True, slots=True)
class FlareWindow:
    onset: int
    length: int

    @property
    def end(self) -> int:  # exclusive
        return self.onset + self.length

    @property
    def midpoint(self) -> int:
        return self.onset + self.length // 2


@dataclass(frozen=True, slots=True)
class PatientProfile:
    name: str
    label: str
    story: str
    seed: int
    offsets: dict[str, float]
    flare_multipliers: dict[str, float]


ATTUNEFM_PROFILES: dict[str, PatientProfile] = {
    "office": PatientProfile(
        name="office",
        label="office worker",
        story="Desk-based worker with meeting load, posture strain, sleep debt, and brain fog.",
        seed=31,
        offsets={
            "work_burden": 0.08,
            "posture_strain": 0.08,
            "engagement": -0.05,
            "cognitive_fog": 0.03,
        },
        flare_multipliers={
            "work_burden": 1.35,
            "posture_strain": 1.5,
            "cognitive_fog": 1.45,
            "sleep_hours": 1.15,
            "breathlessness_report": 0.45,
        },
    ),
    "firefighter": PatientProfile(
        name="firefighter",
        label="firefighter / occupational hazard",
        story="Responder profile with heat/exertion stress, breathlessness, mobility strain, and recovery load.",
        seed=41,
        offsets={
            "hrv": -4.0,
            "resting_hr": 3.0,
            "spo2": -0.25,
            "breathlessness_report": 0.05,
            "mobility_change": 0.04,
            "pain_interference": 0.04,
        },
        flare_multipliers={
            "hrv": 1.25,
            "resting_hr": 1.35,
            "spo2": 1.6,
            "breathlessness_report": 1.8,
            "mobility_change": 1.5,
            "posture_strain": 1.25,
        },
    ),
    "firefighter_asthma": PatientProfile(
        name="firefighter_asthma",
        label="firefighter / asthma",
        story="Responder profile where smoke, heat, and exertion trigger asthma-like breathlessness, lower SpO2, and recovery strain.",
        seed=43,
        offsets={
            "hrv": -3.5,
            "resting_hr": 2.5,
            "spo2": -0.45,
            "breathlessness_report": 0.08,
            "medication_tolerance": 0.04,
            "voice_fatigue": 0.03,
            "mobility_change": 0.03,
        },
        flare_multipliers={
            "hrv": 1.25,
            "resting_hr": 1.35,
            "spo2": 2.0,
            "breathlessness_report": 2.2,
            "medication_tolerance": 1.45,
            "voice_fatigue": 1.2,
            "mobility_change": 1.25,
            "posture_strain": 1.1,
        },
    ),
    "firefighter_recovery": PatientProfile(
        name="firefighter_recovery",
        label="firefighter / post-fire recovery",
        story="Post-fire recovery profile after smoke and heat exposure, with sleep debt, fatigue, autonomic load, and cautious return-to-duty signals.",
        seed=40,
        offsets={
            "hrv": -4.5,
            "resting_hr": 3.5,
            "sleep_hours": -0.3,
            "spo2": -0.25,
            "voice_fatigue": 0.04,
            "breathlessness_report": 0.04,
            "mobility_change": 0.04,
            "pain_interference": 0.04,
            "cognitive_fog": 0.02,
        },
        flare_multipliers={
            "hrv": 1.4,
            "resting_hr": 1.35,
            "sleep_hours": 1.25,
            "spo2": 1.25,
            "voice_fatigue": 1.3,
            "breathlessness_report": 1.1,
            "mobility_change": 1.35,
            "pain_interference": 1.25,
            "cognitive_fog": 1.1,
            "posture_strain": 1.2,
            "medication_tolerance": 0.9,
        },
    ),
    "firefighter_dormant": PatientProfile(
        name="firefighter_dormant",
        label="firefighter / dormant chronic illness",
        story="Responder profile where heat, smoke, sleep debt, and exertion uncover a usually quiet inflammatory condition.",
        seed=45,
        offsets={
            "hrv": -3.0,
            "resting_hr": 2.5,
            "spo2": -0.2,
            "breathlessness_report": 0.05,
            "pain_interference": 0.06,
            "skin_wound_change": 0.03,
            "voice_fatigue": 0.04,
            "cognitive_fog": 0.03,
            "medication_tolerance": 0.03,
        },
        flare_multipliers={
            "hrv": 1.2,
            "resting_hr": 1.3,
            "spo2": 1.55,
            "breathlessness_report": 1.65,
            "mobility_change": 1.35,
            "pain_interference": 1.55,
            "skin_wound_change": 1.25,
            "voice_fatigue": 1.2,
            "cognitive_fog": 1.15,
            "medication_tolerance": 1.2,
        },
    ),
    "veteran": PatientProfile(
        name="veteran",
        label="veteran / hidden chronic load",
        story="Veteran profile with sleep disruption, pain interference, cognitive fog, medication tolerance, and mobility strain.",
        seed=48,
        offsets={
            "hrv": -3.5,
            "resting_hr": 2.0,
            "sleep_hours": -0.45,
            "voice_fatigue": 0.05,
            "pain_interference": 0.08,
            "cognitive_fog": 0.06,
            "mobility_change": 0.05,
            "medication_tolerance": 0.04,
            "engagement": -0.04,
        },
        flare_multipliers={
            "hrv": 1.25,
            "resting_hr": 1.2,
            "sleep_hours": 1.3,
            "voice_fatigue": 1.35,
            "pain_interference": 1.55,
            "cognitive_fog": 1.45,
            "mobility_change": 1.35,
            "medication_tolerance": 1.25,
            "work_burden": 1.15,
            "breathlessness_report": 0.8,
        },
    ),
    "autoimmune": PatientProfile(
        name="autoimmune",
        label="autoimmune flare",
        story="Hidden chronic illness profile with pain, fatigue, skin change, medication tolerance, and brain fog.",
        seed=51,
        offsets={
            "pain_interference": 0.08,
            "voice_fatigue": 0.07,
            "cognitive_fog": 0.05,
            "skin_wound_change": 0.04,
            "medication_tolerance": 0.04,
        },
        flare_multipliers={
            "pain_interference": 1.7,
            "voice_fatigue": 1.25,
            "cognitive_fog": 1.35,
            "skin_wound_change": 1.55,
            "medication_tolerance": 1.35,
            "mobility_change": 1.2,
            "breathlessness_report": 0.7,
        },
    ),
    "metabolic_pcos": PatientProfile(
        name="metabolic_pcos",
        label="metabolic / PCOS",
        story="Metabolic disorder profile with glucose variability, diet response, sleep disruption, pain, and visible skin change.",
        seed=60,
        offsets={
            "glucose_variability": 0.08,
            "diet_response": 0.08,
            "food_photo_risk": 0.08,
            "sleep_hours": -0.25,
            "pain_interference": 0.03,
            "skin_wound_change": 0.03,
        },
        flare_multipliers={
            "glucose_variability": 1.75,
            "diet_response": 1.7,
            "food_photo_risk": 1.65,
            "sleep_hours": 1.2,
            "resting_hr": 1.15,
            "pain_interference": 1.2,
            "skin_wound_change": 1.25,
            "breathlessness_report": 0.55,
            "mobility_change": 0.8,
        },
    ),
}


def flare_window(days: int) -> FlareWindow:
    # A short flare near "today" so reflect() catches it live.
    return FlareWindow(days - 12, 5)


def sample(spec: SignalSpec, day: int, rng: random.Random) -> float:
    if spec.cyclic:
        return float(1 + day % CYCLE_DAYS)
    return spec.normal + rng.gauss(0, spec.noise)


def _resolve_profile(profile: str | PatientProfile | None) -> PatientProfile | None:
    if profile is None:
        return None
    if isinstance(profile, PatientProfile):
        return profile
    return ATTUNEFM_PROFILES[profile]


def generate(
    pack: ConditionPack,
    *,
    days: int = 90,
    seed: int = 0,
    profile: str | PatientProfile | None = None,
) -> Memory:
    patient = _resolve_profile(profile)
    seed = patient.seed if patient else seed
    rng = random.Random(seed)
    window = flare_window(days)
    mem = Memory()
    for spec in pack.signals:
        for day in range(days):
            value = sample(spec, day, rng)
            if patient:
                value += patient.offsets.get(spec.key, 0.0)
            if window.onset <= day < window.end:
                flare = spec.flare
                if patient:
                    multiplier = patient.flare_multipliers.get(spec.key)
                    if multiplier is None:
                        value = spec.normal + patient.offsets.get(spec.key, 0.0)
                        flare = 0.0
                    else:
                        flare *= multiplier
                value += flare  # 0.0 for signals that don't participate in the flare
            mem.add(
                Signal(spec.key, spec.axis, round(value, 2), day, source=spec.modality)
            )
    return mem


def save(mem: Memory, path: Path) -> None:
    path.write_text(json.dumps([asdict(s) for s in mem.signals], indent=2))


def load_memory(path: Path) -> Memory:
    return Memory([Signal(**row) for row in json.loads(path.read_text())])


def main(days: int = 90, out: str = "data") -> None:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (name, pack) in enumerate(PACKS.items()):
        mem = generate(pack, days=days, seed=i + 1)
        path = out_dir / f"{name}.json"
        save(mem, path)
        rprint(
            f"[green]seeded[/] {name}: {len(mem.signals)} signals / {days}d → {path}"
        )


def run() -> None:
    typer.run(main)


if __name__ == "__main__":
    run()
