import httpx
import pytest
from fastapi.testclient import TestClient

from attune.concordance_engine.engine import PACKS
from attune.synth import generate
from attune.terra import signals_from_terra, to_terra_day
from attune.terra_client import TerraClient
from attune.terra_sim import create_terra_sim, day_index

PACK = PACKS["attunefm"]


def _client_against_sim() -> TerraClient:
    # TestClient is a sync httpx.Client that serves the ASGI sim in-process — a real HTTP round-trip
    http = TestClient(create_terra_sim(days=90), base_url="http://sim")
    return TerraClient("dev", "key", base_url="http://sim/v2", http=http)


def test_day_index_maps_dates_to_offsets():
    assert day_index("2026-01-01") == 0
    assert day_index("2026-02-20") == 50


def test_live_client_pulls_simulated_patient_end_to_end():
    # the real client pulls the sim over HTTP; recovered signals match generating the same patient
    client = _client_against_sim()
    recovered = {
        s.key: round(s.value, 4)
        for s in client.signals_for("veteran", "2026-02-20", PACK, 50)
    }
    memory = generate(PACK, days=90, profile="veteran", intraday=True)
    direct = {
        s.key: round(s.value, 4)
        for s in signals_from_terra(
            to_terra_day(memory, 50, user_id="veteran"), PACK, 50
        )
    }
    assert recovered == direct
    assert recovered  # non-empty — the sim actually served data


def test_unknown_simulated_user_is_404():
    with pytest.raises(httpx.HTTPStatusError) as exc:
        _client_against_sim().fetch_day("nobody", "2026-02-20")
    assert exc.value.response.status_code == 404
