import pytest
from fastapi.testclient import TestClient

from attune.api import create_app
from attune.concordance_engine.engine import PACKS
from attune.serving import load_predictor
from attune.synth import flare_window, generate
from attune.terra import TERRA_MAPPING, to_terra_day
from attune.terra_client import TerraClient
from attune.terra_sim import create_terra_sim
from attune.training import build_training_config, train_attunefm_lite

PACK = PACKS["attunefm"]
WEARABLE = set(TERRA_MAPPING)


@pytest.fixture(scope="module")
def predictor(tmp_path_factory):
    run = train_attunefm_lite(
        build_training_config(
            "smoke", output_dir=tmp_path_factory.mktemp("api"), epochs=60
        ),
        wandb_enabled=False,
    )
    return load_predictor(run.checkpoint_path)


@pytest.fixture(scope="module")
def client(predictor):
    return TestClient(create_app(predictor))


def _sim_backed_client(predictor) -> TestClient:
    # the app's Terra client points at the local simulator — a real HTTP pull, no device
    sim = TestClient(create_terra_sim(days=90))
    terra = TerraClient("dev", "key", base_url="http://sim/v2", http=sim)
    return TestClient(create_app(predictor, terra_client=terra))


def test_ingest_then_predict_returns_terra_document(client):
    memory = generate(PACK, days=90, profile="veteran")
    day = flare_window(90).midpoint
    for d in range(day + 1):
        terra = client.post(
            "/ingest/terra",
            json={"user_id": "u1", "day": d, "payloads": to_terra_day(memory, d)},
        )
        assert terra.status_code == 200
        checkins = [
            {"key": s.key, "value": s.value, "day": s.day, "source": s.source}
            for s in memory.window(d, 1)
            if s.key not in WEARABLE
        ]
        client.post("/ingest/checkin", json={"user_id": "u1", "signals": checkins})

    response = client.get("/predict/u1", params={"day": day})
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "attunefm_prediction"
    prediction = body["data"][0]
    assert prediction["diagnosis"]["predicted_profile_string"] == "veteran"
    assert {e["horizon_days_int"] for e in prediction["forecast_events"]} == {7, 30}


def test_predict_unknown_user_is_404(client):
    assert client.get("/predict/nobody", params={"day": 50}).status_code == 404


def test_ingest_checkin_rejects_unknown_signal(client):
    response = client.post(
        "/ingest/checkin",
        json={
            "user_id": "u2",
            "signals": [{"key": "not_a_signal", "value": 1.0, "day": 0}],
        },
    )
    assert response.status_code == 422


def test_live_terra_pull_then_predict(predictor):
    # full loop: pull the objective channel live from the (simulated) Terra feed + check-in, predict
    http = _sim_backed_client(predictor)
    memory = generate(PACK, days=90, profile="veteran")
    day = flare_window(90).midpoint

    pull = http.post(
        "/ingest/terra/live",
        json={"user_id": "veteran", "start_date": "2026-01-01", "days": 90},
    )
    assert pull.status_code == 200 and pull.json()["days_ingested"] == 90

    checkins = [
        {"key": s.key, "value": s.value, "day": s.day, "source": s.source}
        for s in memory.signals
        if s.key not in WEARABLE
    ]
    http.post("/ingest/checkin", json={"user_id": "veteran", "signals": checkins})

    response = http.get("/predict/veteran", params={"day": day})
    assert response.status_code == 200
    assert (
        response.json()["data"][0]["diagnosis"]["predicted_profile_string"] == "veteran"
    )


def test_live_pull_without_client_is_503(client):
    response = client.post(
        "/ingest/terra/live",
        json={"user_id": "veteran", "start_date": "2026-01-01", "days": 7},
    )
    assert response.status_code == 503
