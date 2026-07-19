"""Live Terra client — the objective channel behind `signals_from_terra`.

The mock (`to_terra_day`) and this client return the same payload shape, so swapping the source is
a one-line change and nothing downstream moves. Terra's REST API authenticates with a `dev-id` +
`x-api-key` header pair; each `/v2/{daily,sleep,body}` endpoint returns a payload whose `data` array
holds the model objects `TERRA_MAPPING` reads. No API key is configured in this repo, so the client
is live-ready and mock-tested against Terra's documented response shape, not proven against the
live service — a free Terra developer account flips it on with no code change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import httpx

from attune.concordance_engine.memory import Signal
from attune.packs.base import ConditionPack
from attune.terra import TERRA_MAPPING, signals_from_terra

TERRA_BASE_URL = "https://api.tryterra.co/v2"
# our internal payload types are exactly Terra's pull endpoints — derive them so they can't drift
TERRA_ENDPOINTS = tuple(sorted({field.data_type for field in TERRA_MAPPING.values()}))


@dataclass
class TerraClient:
    dev_id: str
    api_key: str
    base_url: str = TERRA_BASE_URL
    http: httpx.Client = field(default_factory=httpx.Client)

    def _get(self, endpoint: str, user_id: str, start_date: str, end_date: str) -> dict:
        response = self.http.get(
            f"{self.base_url}/{endpoint}",
            params={
                "user_id": user_id,
                "start_date": start_date,
                "end_date": end_date,
            },
            headers={"dev-id": self.dev_id, "x-api-key": self.api_key},
        )
        response.raise_for_status()
        return response.json()

    def fetch_day(self, user_id: str, date: str) -> dict[str, dict]:
        """One Terra payload per data type for a single date, keyed by type — the exact shape
        `signals_from_terra` consumes, so it drops in for the mock `to_terra_day`."""
        return {
            endpoint: self._get(endpoint, user_id, date, date)
            for endpoint in TERRA_ENDPOINTS
        }

    def signals_for(
        self, user_id: str, date: str, pack: ConditionPack, day: int
    ) -> list[Signal]:
        return signals_from_terra(self.fetch_day(user_id, date), pack, day)


def terra_client_from_env() -> TerraClient:
    dev_id = os.environ.get("TERRA_DEV_ID")
    api_key = os.environ.get("TERRA_API_KEY")
    if not (dev_id and api_key):
        raise RuntimeError(
            "set TERRA_DEV_ID and TERRA_API_KEY to use the live Terra client"
        )
    return TerraClient(dev_id=dev_id, api_key=api_key)
