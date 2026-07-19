"""Serving layer — a mock backend that ingests Terra data and returns AttuneFM predictions.

Closes the train -> serve loop: load a trained checkpoint, accept Terra-shaped wearable payloads
(the objective channel) plus check-in signals (the subjective channel) into a per-user memory,
and return a prediction — diagnosis (which profile) + forecast (episode onset within each horizon)
— emitted as a Terra-styled payload so the whole system speaks one interface.

The linear + forecast heads are applied in pure Python from the checkpoint's weights, so serving
needs no torch and no training code. Swap `TerraIngestSession.ingest_terra`'s source for the live
Terra webhook and the prediction path is unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from attune.attunefm import (
    CheckpointModel,
    logits,
    predict_proba,
    series_by_day,
    sigmoid,
    standardize,
    window_features,
)
from attune.concordance_engine.engine import PACKS
from attune.concordance_engine.memory import Memory, Signal
from attune.packs.base import ConditionPack
from attune.terra import signals_from_terra, terra_datetime, terra_envelope

MODEL_NAME = "attunefm-lite"


@dataclass(frozen=True, slots=True)
class Prediction:
    profile_scores: dict[str, float]
    forecast: dict[int, float]  # horizon days -> episode-onset probability

    @property
    def profile(self) -> str:  # most likely condition profile
        return max(self.profile_scores, key=self.profile_scores.get)

    @property
    def confidence(self) -> float:
        return self.profile_scores[self.profile]


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
        x = standardize(features, self.means, self.scales)
        scores = predict_proba(x, self.weights, self.bias)
        forecast = {
            horizon: sigmoid(logit)
            for horizon, logit in zip(
                self.forecast_horizons,
                logits(x, self.forecast_weights, self.forecast_bias),
                strict=True,
            )
        }
        return Prediction(
            profile_scores=dict(zip(self.labels, scores, strict=True)),
            forecast=forecast,
        )


def load_predictor(checkpoint_path: Path) -> AttuneFMPredictor:
    model = CheckpointModel.from_dict(json.loads(Path(checkpoint_path).read_text()))
    return AttuneFMPredictor(
        pack=PACKS[model.pack],
        feature_window=model.feature_window,
        means=model.feature_mean,
        scales=model.feature_scale,
        labels=model.labels,
        weights=model.weights,
        bias=model.bias,
        forecast_weights=model.forecast_weights,
        forecast_bias=model.forecast_bias,
        forecast_horizons=model.forecast_horizons,
    )


def to_terra_prediction(
    prediction: Prediction, *, day: int, user_id: str = "mock-user"
) -> dict:
    """Emit a prediction as a Terra-styled payload, so serving output matches the input interface."""
    when = terra_datetime(day)
    return terra_envelope(
        "attunefm_prediction",
        user_id=user_id,
        provider=MODEL_NAME,
        data=[
            {
                "metadata": {
                    "start_time": when,
                    "end_time": when,
                    "upload_type": 0,
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
        ],
    )


@dataclass
class TerraIngestSession:
    """Mock backend: ingest a user's Terra + check-in stream, then predict in Terra format."""

    predictor: AttuneFMPredictor
    user_id: str = "mock-user"
    memory: Memory = field(default_factory=Memory)

    def ingest_terra(self, payloads: dict[str, dict], day: int) -> None:
        for signal in signals_from_terra(payloads, self.predictor.pack, day):
            self.memory.add(signal)

    def ingest_checkin(self, signals: list[Signal]) -> None:
        for signal in signals:
            self.memory.add(signal)

    def predict(self, day: int) -> dict:
        prediction = self.predictor.predict(self.memory, day)
        return to_terra_prediction(prediction, day=day, user_id=self.user_id)
