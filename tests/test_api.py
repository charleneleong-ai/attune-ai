import pytest
from fastapi.testclient import TestClient

from attune.api import create_app
from attune.concordance_engine.engine import PACKS
from attune.rook import ROOK_MAPPING, to_rook_day
from attune.serving import load_predictor
from attune.synth import flare_window, generate
from attune.training import build_training_config, train_attunefm_lite

PACK = PACKS["attunefm"]
WEARABLE = set(ROOK_MAPPING)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    run = train_attunefm_lite(
        build_training_config(
            "smoke", output_dir=tmp_path_factory.mktemp("api"), epochs=60
        ),
        wandb_enabled=False,
    )
    return TestClient(create_app(load_predictor(run.checkpoint_path)))


def test_ingest_then_predict_returns_rook_document(client):
    memory = generate(PACK, days=90, profile="veteran")
    day = flare_window(90).midpoint
    for d in range(day + 1):
        rook = client.post(
            "/ingest/rook",
            json={"user_id": "u1", "day": d, "documents": to_rook_day(memory, d)},
        )
        assert rook.status_code == 200
        checkins = [
            {"key": s.key, "value": s.value, "day": s.day, "source": s.source}
            for s in memory.window(d, 1)
            if s.key not in WEARABLE
        ]
        client.post("/ingest/checkin", json={"user_id": "u1", "signals": checkins})

    response = client.get("/predict/u1", params={"day": day})
    assert response.status_code == 200
    body = response.json()
    assert body["data_structure"] == "attunefm_prediction"
    prediction = body["attunefm_prediction"]["predictions"][0]
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
