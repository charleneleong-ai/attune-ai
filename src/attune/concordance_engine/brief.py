"""Clinician brief — maps the longitudinal memory onto a pack's BriefTemplate criteria.

Each criterion names an axis; the brief gathers that axis's signals, scores each against the
patient's own baseline (the same robust-z the concordance engine uses), and flags salient
deviations. Deterministic and offline — this is the async artifact that closes the loop with a
care team without an appointment. Presentation lives in attune.reporting, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from attune.concordance_engine.concordance import (
    baseline_history,
    concordance,
    robust_z,
)
from attune.concordance_engine.memory import Memory
from attune.packs.base import ConditionPack


@dataclass(frozen=True, slots=True)
class SignalEvidence:
    key: str
    latest: float
    baseline: float
    z: float  # signed robust z — direction matters to a clinician


@dataclass(frozen=True, slots=True)
class CriterionEvidence:
    axis: str
    label: str
    signals: list[SignalEvidence]
    salient: bool


@dataclass(frozen=True, slots=True)
class Brief:
    template: str
    day: int
    concordant: bool
    criteria: list[CriterionEvidence]
    recommendation: str
    handoff_targets: tuple[str, ...]


def signal_evidence(mem: Memory, key: str, day: int) -> SignalEvidence | None:
    upto = [s for s in mem.series(key) if s.day <= day]
    if not upto:
        return None
    latest = upto[-1].value
    past = baseline_history(upto, day)
    baseline = median(past) if past else latest
    return SignalEvidence(
        key, latest, round(baseline, 2), round(robust_z(latest, past), 2)
    )


def build_brief(pack: ConditionPack, mem: Memory, day: int) -> Brief:
    axis_of = pack.axis_of
    finding = concordance(mem, day, axis_of, weights=pack.axis_weights)
    deviating_axes = set(finding.deviating_axes)
    criteria: list[CriterionEvidence] = []
    for crit in pack.brief.criteria:
        keys = [k for k, axis in axis_of.items() if axis == crit.axis]
        evidence = [e for k in keys if (e := signal_evidence(mem, k, day)) is not None]
        criteria.append(
            CriterionEvidence(
                crit.axis,
                crit.label,
                evidence,
                salient=crit.axis in deviating_axes,
            )
        )
    recommendation = (
        pack.escalation.amber_action
        if finding.concordant
        else pack.escalation.green_action
    )
    return Brief(
        pack.brief.name,
        day,
        finding.concordant,
        criteria,
        recommendation,
        pack.escalation.handoff_targets,
    )
