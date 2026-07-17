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


def flare_window(days: int) -> FlareWindow:
    return FlareWindow(days - 12, 5)  # a short flare near "today" so reflect() catches it live


def sample(spec: SignalSpec, day: int, rng: random.Random) -> float:
    if spec.cyclic:
        return float(1 + day % CYCLE_DAYS)
    return spec.normal + rng.gauss(0, spec.noise)


def generate(pack: ConditionPack, *, days: int = 90, seed: int = 0) -> Memory:
    rng = random.Random(seed)
    window = flare_window(days)
    mem = Memory()
    for spec in pack.signals:
        for day in range(days):
            value = sample(spec, day, rng)
            if window.onset <= day < window.end:
                value += spec.flare  # 0.0 for signals that don't participate in the flare
            mem.add(Signal(spec.key, spec.axis, round(value, 2), day, source=spec.modality))
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
        rprint(f"[green]seeded[/] {name}: {len(mem.signals)} signals / {days}d → {path}")


def run() -> None:
    typer.run(main)


if __name__ == "__main__":
    run()
