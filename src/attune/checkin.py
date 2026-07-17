"""Daily check-in — the voice-first routine that assesses the patient day to day.

Voice is the core channel: accessible to patients who aren't digitally native, and low-friction
enough to sustain daily (friction is the enemy of adherence). The routine is a short spoken
conversation with optional "show me" photo turns — voice stays the floor.

Offline this maps scripted responses to typed Signals. At the venue, Realtime speaks the prompts
and transcribes answers into `responses`, and GPT-4o vision scores the optional photos — the
downstream assessment is the same engine, unchanged.
"""

from __future__ import annotations

from attune.concordance_engine.memory import Signal
from attune.packs.base import ConditionPack


def record_checkin(pack: ConditionPack, day: int, responses: dict[str, float]) -> list[Signal]:
    axis_of = pack.axis_of
    signals = []
    for item in pack.checkin:
        if item.signal_key not in responses:
            if item.optional:
                continue
            raise KeyError(f"check-in missing required response: {item.signal_key}")
        signals.append(
            Signal(
                item.signal_key,
                axis_of[item.signal_key],
                responses[item.signal_key],
                day,
                source=item.source,
            )
        )
    return signals
