"""Demo runner — the offline narrative that stitches the engine into a story.

No API keys: replays a seeded patient through the same engine the live demo uses. The
voice/vision capture layer (build-plan step 3) plugs in by feeding live Signals in place of the
seed. Run: `python -m attune.demo [pcos|veteran]` (default: both, with the config hot-swap).
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from attune.checkin import record_checkin
from attune.concordance_engine.concordance import ConcordanceFinding
from attune.concordance_engine.engine import PACKS, Engine
from attune.concordance_engine.memory import Memory
from attune.concordance_engine.safety import Tier
from attune.packs.base import ConditionPack
from attune.reporting import render
from attune.synth import flare_window, generate

console = Console()

CALM_UTTERANCE = "rough week but I'm hanging in there"
CRISIS_UTTERANCE = "honestly some days I feel like I want to end it"
TIER_COLOR = {Tier.GREEN: "green", Tier.AMBER: "yellow", Tier.RED: "red"}


def channel_label(source: str) -> str:
    return {
        "audio": "voice",
        "self_report": "voice",
        "vision": "photo",
        "video": "video",
    }.get(source, source)


def first_concordant_day(eng: Engine, lo: int, hi: int) -> int | None:
    return next((day for day in range(lo, hi + 1) if eng.reflect(day).concordant), None)


def show_timeline(findings: list[ConcordanceFinding]) -> None:
    table = Table(
        "day", "load", "status", "axes over threshold", title="daily concordance"
    )
    for f in findings:
        status = "[red]concordant[/]" if f.concordant else "[green]ok[/]"
        table.add_row(str(f.day), f"{f.load:.1f}", status, ", ".join(f.deviating_axes))
    console.print(table)


def show_checkin(pack: ConditionPack, mem: Memory, day: int) -> None:
    # the patient's spoken answers are that day's actual values (voice fills these live at the venue)
    day_values = {s.key: s.value for s in mem.window(day, 1)}
    console.print(
        f"\n[bold]Daily voice check-in[/] — day {day} (voice-first, photos/videos optional):"
    )
    for item in pack.checkin:
        channel = channel_label(item.source)
        console.print(f'  [dim]agent:[/] "{item.prompt}"  [dim]({channel})[/]')
        console.print(
            f"    [italic]patient →[/] {item.signal_key} = {day_values.get(item.signal_key)}"
        )
    signals = record_checkin(pack, day, day_values)
    console.print(
        f"  [green]→ {len(signals)} signals captured[/] and scored against personal baseline"
    )


def show_escalation(eng: Engine, day: int) -> None:
    console.print("\n[bold]Safety tiering[/] — deterministic floor ∪ classifier:")
    for text in (CALM_UTTERANCE, CRISIS_UTTERANCE):
        verdict = eng.assess(text, day)
        console.print(f'  "[italic]{text}[/]"')
        console.print(
            f"    → [{TIER_COLOR[verdict.tier]}]{verdict.tier.name}[/] "
            f"({verdict.triggered_by}) — {verdict.reason}"
        )
        if verdict.tier is Tier.RED:
            console.print(
                f"    → warm handoff: {', '.join(eng.pack.escalation.handoff_targets)}"
            )


def run_pack(pack: ConditionPack, *, days: int = 90) -> None:
    seed = list(PACKS).index(pack.name) + 1  # the same patient the fixtures use
    eng = Engine(pack, generate(pack, days=days, seed=seed))
    window = flare_window(days)
    lo, hi = window.onset - 4, window.end + 1
    findings = [eng.reflect(day) for day in range(lo, hi + 1)]
    console.rule(f"[bold]{pack.name}[/] — {pack.persona.register}")
    show_timeline(findings)
    first = next((f.day for f in findings if f.concordant), None)
    console.print(
        f"[bold]Early warning:[/] first concordant on day {first} "
        f"(flare onset ~day {window.onset}) — caught as the drift begins."
    )
    show_checkin(pack, eng.memory, window.midpoint)
    console.print(
        Panel(
            Markdown(render(eng.brief(window.midpoint))),
            title="clinician brief",
            expand=False,
        )
    )
    show_escalation(eng, window.midpoint)


def main(
    pack: str | None = typer.Argument(None, help="pcos | veteran (default: both)"),
    days: int = 90,
) -> None:
    names = [pack] if pack else list(PACKS)
    for name in names:
        if name not in PACKS:
            console.print(f"[red]unknown pack '{name}' — choose from {list(PACKS)}[/]")
            raise typer.Exit(1)
    for i, name in enumerate(names):
        run_pack(PACKS[name], days=days)
        if i < len(names) - 1:
            console.rule("[dim]same engine — hot-swapping the condition pack[/]")
    if len(names) > 1:
        console.print(
            Panel(
                "One concordance engine. Two conditions. Config, not code.",
                title="the thesis",
            )
        )


def run() -> None:
    typer.run(main)


if __name__ == "__main__":
    run()
