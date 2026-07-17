"""Presentation layer — renders a structured Brief to clinician-facing markdown.

Kept out of concordance_engine so the core emits the Brief dataclass and callers choose a format
(markdown here; a Realtime voice summary or a PDF could sit alongside).
"""

from __future__ import annotations

from attune.concordance_engine.brief import Brief


def render(brief: Brief) -> str:
    status = "concordant multi-axis deterioration" if brief.concordant else "within personal baseline"
    lines = [f"# {brief.template} — day {brief.day}", f"**Status:** {status}", ""]
    for c in brief.criteria:
        lines.append(f"## {c.label}{' — flagged' if c.salient else ''}")
        for e in c.signals:
            arrow = "up" if e.z > 0 else "down" if e.z < 0 else "flat"
            lines.append(f"- {e.key}: {e.latest} (baseline {e.baseline}, {arrow} z={e.z:+.1f})")
        lines.append("")
    lines.append(f"**Recommendation:** {brief.recommendation}")
    if brief.concordant:
        lines.append(f"**Handoff:** {', '.join(brief.handoff_targets)}")
    return "\n".join(lines)
