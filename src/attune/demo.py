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

from attune.attunefm import monitoring_answer
from attune.checkin import record_checkin
from attune.concordance_engine.concordance import ConcordanceFinding
from attune.concordance_engine.engine import PACKS, Engine
from attune.concordance_engine.memory import Memory
from attune.concordance_engine.safety import Tier
from attune.packs.base import ConditionPack
from attune.reporting import render
from attune.synth import ATTUNEFM_PROFILES, PatientProfile, flare_window, generate

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


def attunefm_profile_names() -> tuple[str, ...]:
    return tuple(ATTUNEFM_PROFILES)


def first_concordant_day(eng: Engine, lo: int, hi: int) -> int | None:
    return next((day for day in range(lo, hi + 1) if eng.reflect(day).concordant), None)


def demo_checkin_answer(signal_key: str, value: float | None) -> str:
    if value is None:
        return "I do not have that reading yet."
    if signal_key == "voice_fatigue":
        return (
            "I am wiped out today; even a normal shift feels heavy."
            if value >= 0.7
            else "I am a little tired, but I can get through the day."
        )
    if signal_key == "medication_tolerance":
        return (
            "The new dose seems harder on me today; I feel off and more drained."
            if value >= 0.4
            else "No major side effects since yesterday."
        )
    if signal_key == "work_burden":
        return (
            "Work feels like too much today, and I am struggling to pace myself."
            if value >= 0.6
            else "The workload feels manageable."
        )
    if signal_key == "pain_interference":
        return (
            "Pain is getting in the way of moving, resting, and concentrating."
            if value >= 0.6
            else "Pain is present, but it is not stopping me much."
        )
    if signal_key == "cognitive_fog":
        return (
            "My thinking feels foggy; I am losing words and focus more than usual."
            if value >= 0.6
            else "My thinking feels mostly clear."
        )
    if signal_key == "skin_wound_change":
        return (
            "I would upload a photo because the rash or swelling looks more noticeable."
            if value >= 0.2
            else "Nothing visible looks meaningfully changed."
        )
    if signal_key == "mobility_change":
        return (
            "A short movement video would show I am stiffer and slower than usual."
            if value >= 0.5
            else "Movement looks close to my usual baseline."
        )
    return f"Reported value {value}."


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
        value = day_values.get(item.signal_key)
        console.print(f'  [dim]agent:[/] "{item.prompt}"  [dim]({channel})[/]')
        console.print(
            f'    [italic]patient →[/] "{demo_checkin_answer(item.signal_key, value)}"'
        )
        console.print(f"    [dim]parsed →[/] {item.signal_key} = {value}")
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


def show_monitoring_answer(eng: Engine, day: int) -> None:
    answer = monitoring_answer(eng, day=day)
    stats = "\n".join(f"- {stat}" for stat in answer.stats)
    console.print(
        Panel(
            Markdown(
                "\n".join(
                    (
                        f"## {answer.headline}",
                        "",
                        stats,
                        "",
                        f"**Interpretation:** {answer.interpretation}",
                        "",
                        f"**Recommendation:** {answer.recommendation}",
                    )
                )
            ),
            title="demo answer",
            expand=False,
        )
    )


def run_pack(
    pack: ConditionPack,
    *,
    days: int = 90,
    profile: PatientProfile | None = None,
) -> None:
    seed = list(PACKS).index(pack.name) + 1  # the same patient the fixtures use
    eng = Engine(pack, generate(pack, days=days, seed=seed, profile=profile))
    window = flare_window(days)
    lo, hi = window.onset - 4, window.end + 1
    findings = [eng.reflect(day) for day in range(lo, hi + 1)]
    title = pack.name if profile is None else f"{pack.name} / {profile.label}"
    console.rule(f"[bold]{title}[/] — {pack.persona.register}")
    if profile:
        console.print(f"[bold]Profile:[/] {profile.story}")
    show_timeline(findings)
    first = next((f.day for f in findings if f.concordant), None)
    console.print(
        f"[bold]Early warning:[/] first concordant on day {first} "
        f"(flare onset ~day {window.onset}) — caught as the drift begins."
    )
    show_checkin(pack, eng.memory, window.midpoint)
    if pack.name == "attunefm":
        show_monitoring_answer(eng, window.midpoint)
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
    profile: str | None = typer.Option(
        None,
        help="AttuneFM profile: office | firefighter | firefighter_asthma | firefighter_recovery | firefighter_dormant | veteran | autoimmune | metabolic_pcos (default: all AttuneFM profiles).",
    ),
) -> None:
    names = [pack] if pack else list(PACKS)
    for name in names:
        if name not in PACKS:
            console.print(f"[red]unknown pack '{name}' — choose from {list(PACKS)}[/]")
            raise typer.Exit(1)
    for i, name in enumerate(names):
        if name == "attunefm":
            if profile and profile not in ATTUNEFM_PROFILES:
                console.print(
                    f"[red]unknown AttuneFM profile '{profile}' — choose from {attunefm_profile_names()}[/]"
                )
                raise typer.Exit(1)
            profiles = (
                (ATTUNEFM_PROFILES[profile],)
                if profile
                else tuple(ATTUNEFM_PROFILES.values())
            )
            for j, patient in enumerate(profiles):
                run_pack(PACKS[name], days=days, profile=patient)
                if j < len(profiles) - 1:
                    console.rule("[dim]same pack — swapping the demo profile[/]")
        else:
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
