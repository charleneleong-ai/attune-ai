"""Local Terra simulator — a fake Terra API to exercise the live client without a wearable.

Serves generator-produced patients at Terra's real `/v2/{daily,sleep,body}` endpoints, so the live
`TerraClient` can pull over HTTP and you can watch the whole ingest -> predict path run with no
device and no Terra account. `user_id` selects a synthetic profile; the date maps to a day index
(days since `BASE_DATETIME`). Run it with:

    uvicorn attune.terra_sim:app
    curl "http://localhost:8000/v2/daily?user_id=veteran&start_date=2026-02-20"
"""

from __future__ import annotations

from datetime import date

from fastapi import FastAPI, HTTPException

from attune.concordance_engine.engine import PACKS
from attune.synth import ATTUNEFM_PROFILES, generate
from attune.terra import BASE_DATETIME, TERRA_MAPPING, to_terra_day


def day_index(iso_date: str) -> int:
    return (date.fromisoformat(iso_date) - BASE_DATETIME.date()).days


def create_terra_sim(
    *, pack_name: str = "attunefm", days: int = 365, intraday: bool = True
) -> FastAPI:
    app = FastAPI(title="Terra simulator", version="0.1.0")
    pack = PACKS[pack_name]
    memories = {}  # profile -> generated timeline, built once per user

    def memory_for(user_id: str):
        if user_id not in ATTUNEFM_PROFILES:
            raise HTTPException(
                status_code=404,
                detail=f"unknown simulated user {user_id!r}; choose a profile: {sorted(ATTUNEFM_PROFILES)}",
            )
        if user_id not in memories:
            memories[user_id] = generate(
                pack, days=days, profile=user_id, intraday=intraday
            )
        return memories[user_id]

    def make_endpoint(data_type: str):
        def endpoint(user_id: str, start_date: str) -> dict:
            payloads = to_terra_day(
                memory_for(user_id), day_index(start_date), user_id=user_id
            )
            return payloads.get(
                data_type,
                {"type": data_type, "user": {"user_id": user_id}, "data": []},
            )

        return endpoint

    for data_type in sorted({field.data_type for field in TERRA_MAPPING.values()}):
        app.get(f"/v2/{data_type}")(make_endpoint(data_type))
    return app


app = create_terra_sim()
