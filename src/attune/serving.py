"""Serving layer — a mock backend that ingests Rook data and returns AttuneFM predictions.

Closes the train -> serve loop: load a trained checkpoint, accept Rook-shaped wearable payloads
(the objective channel) plus check-in signals (the subjective channel) into a per-user memory,
and return a prediction — diagnosis (which profile) + forecast (episode onset within each horizon)
— emitted as a Rook-styled document so the whole system speaks one interface.

The linear + forecast heads are applied in pure Python from the checkpoint's weights, so serving
needs no torch and no training code. Swap `RookIngestSession.ingest_rook`'s source for the live
Rook webhook and the prediction path is unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import exp
from pathlib import Path

from attune.attunefm import series_by_day, window_features
from attune.concordance_engine.engine import PACKS
from attune.concordance_engine.memory import Memory, Signal
from attune.packs.base import ConditionPack
from attune.rook import ROOK_VERSION, rook_datetime, signals_from_rook

MODEL_NAME = "attunefm-lite"


@dataclass(frozen=True, slots=True)
class Prediction:
    profile: str  # most likely condition profile
    confidence: float
    profile_scores: dict[str, float]
    forecast: dict[int, float]  # horizon days -> episode-onset probability


@dataclass(frozen=True, slots=True)
class AttuneFMPredictor:
    pack: ConditionPack
    feature_window: int
    means: tuple[float, ...]
    scales: tuple[float, ...]
    labels: tuple[str, ...]
    weights: list[list[float]]
    bias: list[float]
    forecast_weights: list[list[float]]
    forecast_bias: list[float]
    forecast_horizons: tuple[int, ...]

    def predict(self, memory: Memory, day: int) -> Prediction:
        signal_keys = tuple(spec.key for spec in self.pack.signals)
        values = series_by_day(memory, signal_keys, day + 1)
        features = window_features(
            values, signal_keys, day=day, window=self.feature_window
        )
        x = [
            (value - m) / s
            for value, m, s in zip(features, self.means, self.scales, strict=True)
        ]
        scores = _softmax(
            [_linear(row, b, x) for row, b in zip(self.weights, self.bias, strict=True)]
        )
        best = max(range(len(self.labels)), key=scores.__getitem__)
        forecast = {
            horizon: _sigmoid(_linear(row, b, x))
            for horizon, row, b in zip(
                self.forecast_horizons,
                self.forecast_weights,
                self.forecast_bias,
                strict=True,
            )
        }
        return Prediction(
            profile=self.labels[best],
            confidence=scores[best],
            profile_scores=dict(zip(self.labels, scores, strict=True)),
            forecast=forecast,
        )


def load_predictor(checkpoint_path: Path) -> AttuneFMPredictor:
    data = json.loads(Path(checkpoint_path).read_text())
    return AttuneFMPredictor(
        pack=PACKS[data["config"]["pack"]],
        feature_window=int(data["config"]["feature_window"]),
        means=tuple(data["feature_mean"]),
        scales=tuple(data["feature_scale"]),
        labels=tuple(data["labels"]),
        weights=data["weights"],
        bias=data["bias"],
        forecast_weights=data["forecast_weights"],
        forecast_bias=data["forecast_bias"],
        forecast_horizons=tuple(data["forecast_horizons"]),
    )


def to_rook_prediction(
    prediction: Prediction, *, day: int, user_id: str = "mock-user"
) -> dict:
    """Emit a prediction as a Rook-styled document, so serving output matches the input interface."""
    when = rook_datetime(day)
    return {
        "version": ROOK_VERSION,
        "data_structure": "attunefm_prediction",
        "created_at": when,
        "attunefm_prediction": {
            "predictions": [
                {
                    "metadata": {
                        "datetime_string": when,
                        "user_id_string": user_id,
                        "model_string": MODEL_NAME,
                    },
                    "diagnosis": {
                        "predicted_profile_string": prediction.profile,
                        "confidence_number": round(prediction.confidence, 4),
                        "profile_scores_object": {
                            profile: round(score, 4)
                            for profile, score in prediction.profile_scores.items()
                        },
                    },
                    "forecast_events": [
                        {
                            "horizon_days_int": horizon,
                            "episode_probability_number": round(probability, 4),
                        }
                        for horizon, probability in prediction.forecast.items()
                    ],
                }
            ]
        },
    }


@dataclass
class RookIngestSession:
    """Mock backend: ingest a user's Rook + check-in stream, then predict in Rook format."""

    predictor: AttuneFMPredictor
    user_id: str = "mock-user"
    memory: Memory = field(default_factory=Memory)

    def ingest_rook(self, documents: dict[str, dict], day: int) -> None:
        for signal in signals_from_rook(documents, self.predictor.pack, day):
            self.memory.add(signal)

    def ingest_checkin(self, signals: list[Signal]) -> None:
        for signal in signals:
            self.memory.add(signal)

    def predict(self, day: int) -> dict:
        prediction = self.predictor.predict(self.memory, day)
        return to_rook_prediction(prediction, day=day, user_id=self.user_id)


def _linear(row: list[float], bias: float, x: list[float]) -> float:
    return sum(weight * value for weight, value in zip(row, x, strict=True)) + bias


def _softmax(logits: list[float]) -> list[float]:
    peak = max(logits)
    exponentials = [exp(value - peak) for value in logits]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))
