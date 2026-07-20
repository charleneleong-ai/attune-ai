import httpx
import pytest

from attune.concordance_engine.engine import PACKS
from attune.synth import generate
from attune.terra import signals_from_terra, to_terra_day
from attune.terra_client import (
    TERRA_ENDPOINTS,
    TerraClient,
    terra_client_from_env,
)

PACK = PACKS["attunefm"]


def _mock_client(handler, **kwargs) -> TerraClient:
    transport = httpx.MockTransport(handler)
    return TerraClient(
        "dev-1", "key-1", http=httpx.Client(transport=transport), **kwargs
    )


def test_endpoints_track_the_mapping():
    assert set(TERRA_ENDPOINTS) == {"body", "daily", "sleep"}


def test_client_recovers_the_same_signals_as_the_mock():
    # the live API's response shape == our mock's, so pulling through the client and recovering
    # signals must match calling signals_from_terra on to_terra_day directly — the swap is transparent
    memory = generate(PACK, days=90, profile="veteran", intraday=True)
    day = 50
    payloads = to_terra_day(memory, day)

    def handler(request: httpx.Request) -> httpx.Response:
        endpoint = request.url.path.rsplit("/", 1)[-1]
        return httpx.Response(200, json=payloads[endpoint])

    recovered = {
        s.key: round(s.value, 4)
        for s in _mock_client(handler).signals_for("u1", "2026-02-20", PACK, day)
    }
    direct = {s.key: round(s.value, 4) for s in signals_from_terra(payloads, PACK, day)}
    assert recovered == direct


def test_sends_auth_headers_and_date_params():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["dev-id"] = request.headers.get("dev-id")
        seen["x-api-key"] = request.headers.get("x-api-key")
        seen["start"] = request.url.params.get("start_date")
        return httpx.Response(200, json={"data": [{}]})

    _mock_client(handler).fetch_day("u1", "2026-02-20")
    assert seen == {"dev-id": "dev-1", "x-api-key": "key-1", "start": "2026-02-20"}


def test_raises_on_http_error():
    client = _mock_client(lambda request: httpx.Response(401, json={"message": "nope"}))
    with pytest.raises(httpx.HTTPStatusError):
        client.fetch_day("u1", "2026-02-20")


def test_from_env_requires_both_credentials(monkeypatch):
    monkeypatch.setenv("TERRA_DEV_ID", "dev-1")
    monkeypatch.delenv("TERRA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TERRA_DEV_ID and TERRA_API_KEY"):
        terra_client_from_env()


def test_from_env_honors_base_url_override(monkeypatch):
    monkeypatch.setenv("TERRA_DEV_ID", "dev-1")
    monkeypatch.setenv("TERRA_API_KEY", "key-1")
    monkeypatch.setenv("TERRA_BASE_URL", "http://localhost:8000/v2")
    assert terra_client_from_env().base_url == "http://localhost:8000/v2"
