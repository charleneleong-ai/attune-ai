from __future__ import annotations

from dataclasses import dataclass, field

from attune.concordance_engine.brief import Brief, build_brief
from attune.concordance_engine.concordance import ConcordanceFinding, concordance
from attune.concordance_engine.memory import Memory, Signal
from attune.concordance_engine.safety import SafetyVerdict, assess
from attune.packs.base import ConditionPack
from attune.packs.attunefm import ATTUNEFM_PACK
from attune.packs.pcos import PCOS_PACK
from attune.packs.veteran import VETERAN_PACK

PACKS: dict[str, ConditionPack] = {
    p.name: p for p in (PCOS_PACK, VETERAN_PACK, ATTUNEFM_PACK)
}


@dataclass
class Engine:
    pack: ConditionPack
    memory: Memory = field(default_factory=Memory)

    def ingest(self, signal: Signal) -> None:
        self.memory.add(signal)

    def reflect(self, day: int) -> ConcordanceFinding:
        return concordance(
            self.memory, day, self.pack.axis_of, weights=self.pack.axis_weights
        )

    def brief(self, day: int) -> Brief:
        return build_brief(self.pack, self.memory, day)

    def assess(
        self, text: str, day: int, *, classifier_flag: bool = False
    ) -> SafetyVerdict:
        finding = self.reflect(day)
        return assess(
            text,
            self.pack.escalation,
            classifier_flag=classifier_flag,
            amber=finding.concordant,
        )


def load(name: str) -> Engine:
    return Engine(PACKS[name])
