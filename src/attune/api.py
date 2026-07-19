"""Thin FastAPI wrapper over the Terra serving layer.

Endpoints:
- POST /ingest/terra    — a user's Terra wearable payloads for one day (objective channel)
- POST /ingest/checkin  — a user's check-in signals (subjective channel)
- GET  /predict/{user_id}?day=N — the current prediction, as a Terra-styled payload

`create_app(predictor)` builds an app around a loaded predictor (used in tests). For a real
server, point ATTUNE_CHECKPOINT at a training checkpoint and run:

    uvicorn attune.api:app_from_env --factory
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from attune.concordance_engine.memory import Signal
from attune.serving import AttuneFMPredictor, TerraIngestSession, load_predictor


class TerraIngestRequest(BaseModel):
    user_id: str = "mock-user"
    day: int
    payloads: dict[str, dict] = Field(
        description="Terra webhook payloads keyed by data type (daily / sleep / body)"
    )


class CheckinSignalIn(BaseModel):
    key: str
    value: float
    day: int
    source: str = "self_report"


class CheckinIngestRequest(BaseModel):
    user_id: str = "mock-user"
    signals: list[CheckinSignalIn]


def create_app(predictor: AttuneFMPredictor) -> FastAPI:
    app = FastAPI(title="AttuneFM serving", version="0.1.0")
    # In-memory, single-process, unbounded — one session per user_id for the mock demo.
    # A real deployment would back this with a store keyed by user and evict/persist sessions.
    sessions: dict[str, TerraIngestSession] = {}

    def session_for(user_id: str) -> TerraIngestSession:
        return sessions.setdefault(
            user_id, TerraIngestSession(predictor, user_id=user_id)
        )

    @app.post("/ingest/terra")
    def ingest_terra(request: TerraIngestRequest) -> dict:
        session_for(request.user_id).ingest_terra(request.payloads, request.day)
        return {"status": "ok", "user_id": request.user_id, "day": request.day}

    @app.post("/ingest/checkin")
    def ingest_checkin(request: CheckinIngestRequest) -> dict:
        axis_of = predictor.pack.axis_of
        unknown = [s.key for s in request.signals if s.key not in axis_of]
        if unknown:
            raise HTTPException(
                status_code=422, detail=f"unknown signal keys: {unknown}"
            )
        session_for(request.user_id).ingest_checkin(
            [
                Signal(s.key, axis_of[s.key], s.value, s.day, source=s.source)
                for s in request.signals
            ]
        )
        return {
            "status": "ok",
            "user_id": request.user_id,
            "ingested": len(request.signals),
        }

    @app.get("/predict/{user_id}")
    def predict(user_id: str, day: int) -> dict:
        session = sessions.get(user_id)
        if session is None:
            raise HTTPException(
                status_code=404, detail=f"no ingested data for user {user_id!r}"
            )
        return session.predict(day)

    return app


def app_from_env() -> FastAPI:
    checkpoint = os.environ.get("ATTUNE_CHECKPOINT")
    if not checkpoint:
        raise RuntimeError("set ATTUNE_CHECKPOINT to a training checkpoint JSON path")
    return create_app(load_predictor(Path(checkpoint)))
