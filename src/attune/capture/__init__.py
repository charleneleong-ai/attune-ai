"""Perception adapters: turn raw audio / text / image into typed Signals.

Wire these to OpenAI Realtime (audio) and GPT-4o vision (image) at build time. Kept as thin
pure adapters so the engine and packs stay testable without network access or credits — the
contract is `-> Signal`, nothing else.
"""

from __future__ import annotations

from attune.concordance_engine.memory import Signal
from attune.packs.axes import Axis


def from_meal_photo(day: int, gi_estimate: float) -> Signal:
    # TODO: replace gi_estimate with a GPT-4o vision call on the photo.
    return Signal("meal_gi", Axis.METABOLIC, gi_estimate, day, source="vision")


def from_skin_photo(day: int, acne_score: float) -> Signal:
    # TODO: replace acne_score with a GPT-4o vision call scoring acne/hirsutism.
    return Signal("acne_score", Axis.DERMATOLOGICAL, acne_score, day, source="vision")


def from_voice_affect(day: int, valence: float) -> Signal:
    # TODO: derive valence from the Realtime transcript + tone.
    return Signal("voice_affect", Axis.PSYCHOLOGICAL, valence, day, source="audio")
