from attune.concordance_engine.engine import PACKS
from attune.serving import TerraIngestSession, load_predictor
from attune.synth import ATTUNEFM_PROFILES, flare_window, generate
from attune.terra import TERRA_MAPPING, TERRA_VERSION, to_terra_day
from attune.training import build_training_config, train_attunefm_lite

PACK = PACKS["attunefm"]
WEARABLE = set(TERRA_MAPPING)


def _serve_patient(predictor, profile: str, day: int) -> dict:
    # feed one patient's stream into the mock backend through the two channels, then predict
    memory = generate(PACK, days=90, profile=profile)
    session = TerraIngestSession(predictor, user_id=f"u-{profile}")
    for d in range(day + 1):
        session.ingest_terra(to_terra_day(memory, d), d)
        session.ingest_checkin(
            [s for s in memory.window(d, 1) if s.key not in WEARABLE]
        )
    return session.predict(day)


def test_serving_ingests_terra_and_predicts_in_terra_format(tmp_path):
    run = train_attunefm_lite(
        build_training_config("smoke", output_dir=tmp_path, epochs=60),
        wandb_enabled=False,
    )
    predictor = load_predictor(run.checkpoint_path)
    result = _serve_patient(predictor, "veteran", flare_window(90).midpoint)

    # output is a Terra-styled payload
    assert result["type"] == "attunefm_prediction"
    assert result["version"] == TERRA_VERSION
    assert result["user"]["provider"] == "attunefm-lite"
    prediction = result["data"][0]

    diagnosis = prediction["diagnosis"]
    assert diagnosis["predicted_profile_string"] == "veteran"  # correct on a flare day
    assert 0.0 < diagnosis["confidence_number"] <= 1.0
    assert set(diagnosis["profile_scores_object"]) == set(ATTUNEFM_PROFILES)

    forecast = prediction["forecast_events"]
    assert {event["horizon_days_int"] for event in forecast} == {7, 30}
    assert all(0.0 <= event["episode_probability_number"] <= 1.0 for event in forecast)
