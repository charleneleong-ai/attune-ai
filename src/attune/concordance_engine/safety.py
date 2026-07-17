from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

# Shared deterministic crisis floor — packs extend it rather than restating it, so the
# universal safety net cannot silently drift out of sync between conditions.
BASE_RED_KEYWORDS = ("hurt myself", "end it", "suicide", "kill myself")
SAMARITANS = "Samaritans 116 123"


class Tier(IntEnum):
    GREEN = 0  # within baseline — agent coaches autonomously
    AMBER = 1  # concordant deterioration — increase support + async clinician brief
    RED = 2  # acute risk — stop managing, de-escalate, warm human handoff


@dataclass(frozen=True, slots=True)
class EscalationContract:
    red_keywords: tuple[str, ...]  # deterministic crisis floor, independent of the LLM
    handoff_targets: tuple[str, ...]  # e.g. ("Samaritans 116 123", "on-call clinician")
    amber_action: str
    red_action: str
    green_action: str = "within personal baseline — continue monitoring"


@dataclass(frozen=True, slots=True)
class SafetyVerdict:
    tier: Tier
    reason: str
    triggered_by: str  # deterministic | classifier | concordance | none


def deterministic_scan(text: str, contract: EscalationContract) -> bool:
    low = text.lower()
    return any(kw in low for kw in contract.red_keywords)


def assess(
    text: str,
    contract: EscalationContract,
    *,
    classifier_flag: bool = False,
    amber: bool = False,
) -> SafetyVerdict:
    # Red fires on the UNION of the deterministic floor and the classifier — an LLM
    # miss cannot silently drop a crisis (fail-safe, not fail-open).
    if deterministic_scan(text, contract):
        return SafetyVerdict(Tier.RED, "crisis language matched deterministic floor", "deterministic")
    if classifier_flag:
        return SafetyVerdict(Tier.RED, "risk flagged by classifier", "classifier")
    if amber:
        return SafetyVerdict(Tier.AMBER, "concordant multi-axis deterioration", "concordance")
    return SafetyVerdict(Tier.GREEN, "within personal baseline", "none")
