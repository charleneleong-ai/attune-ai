import json
import os
import sys
from pathlib import Path

from attune.training import (
    build_training_config,
    build_training_plan,
    hardware_note,
    load_env,
    train_main,
    train_attunefm_lite,
)


def test_training_plan_validates_and_summarizes_public_datasets():
    plan = build_training_plan(
        pack="attunefm",
        datasets=("BIDSleep", "WESAD", "CGMacros", "Bridge2AI-Voice", "DDI", "PAMAP2"),
        accelerator="a100-80gb",
        max_hours=6,
    )

    assert plan.pack == "attunefm"
    assert plan.accelerator == "a100-80gb"
    assert plan.max_hours == 6
    assert plan.dataset_names == (
        "BIDSleep",
        "WESAD",
        "CGMacros",
        "Bridge2AI-Voice",
        "DDI",
        "PAMAP2",
    )
    assert {"wearable", "voice", "image", "video", "metabolic"}.issubset(
        plan.modalities
    )
    assert {"recovery", "stress", "diet_response", "voice_checkin"}.issubset(plan.heads)


def test_training_config_presets_separate_smoke_and_a100_targets():
    smoke = build_training_config("smoke")
    one_year = build_training_config("one_year")
    a100 = build_training_config("a100")

    # smoke: fast local CI sanity; one_year: the local default; a100: the scaled GPU target
    assert smoke.name == "smoke"
    assert smoke.accelerator == "cpu"
    assert one_year.name == "one_year"
    assert one_year.days == 365
    assert one_year.epochs > smoke.epochs
    assert a100.name == "a100"
    assert a100.accelerator == "a100-80gb"
    assert a100.days == one_year.days  # a full year of data on the GPU target too
    assert len(a100.train_seed_offsets) > len(one_year.train_seed_offsets)


def test_training_config_loads_from_yaml_path():
    config = build_training_config("configs/one_year.yaml")

    assert config.name == "one_year"
    assert config.days == 365
    assert config.output_dir == Path("runs/attunefm-one-year")


def test_hardware_note_does_not_imply_a100_without_runtime(monkeypatch):
    monkeypatch.setattr("attune.training.which", lambda _: None)

    assert "target config only" in hardware_note("a100-80gb")
    assert hardware_note("cpu") == "local CPU/default runtime"


def test_hardware_note_marks_current_trainer_cpu_only_on_a100(monkeypatch):
    monkeypatch.setattr("attune.training.which", lambda _: "/usr/bin/nvidia-smi")

    assert hardware_note("a100-80gb") == (
        "A100 host detected; current lightweight trainer is CPU-only"
    )


def test_training_config_rejects_unknown_preset():
    try:
        build_training_config("made_up")
    except ValueError as exc:
        assert "unknown training config or missing YAML: made_up" in str(exc)
    else:
        raise AssertionError("expected unknown training config validation to fail")


def test_load_env_sets_missing_values_without_overwriting(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "WANDB_PROJECT=attune-from-file",
                "WANDB_ENTITY=local-team",
            ]
        )
    )
    monkeypatch.setenv("WANDB_PROJECT", "already-set")

    load_env(env_file)

    assert os.environ["WANDB_PROJECT"] == "already-set"
    assert os.environ["WANDB_ENTITY"] == "local-team"
    monkeypatch.delenv("WANDB_ENTITY", raising=False)


def test_training_plan_rejects_unknown_dataset_names():
    try:
        build_training_plan(
            pack="attunefm",
            datasets=("BIDSleep", "MadeUpSet"),
            accelerator="a100-80gb",
            max_hours=6,
        )
    except ValueError as exc:
        assert "unknown datasets: MadeUpSet" in str(exc)
    else:
        raise AssertionError("expected unknown dataset validation to fail")


def test_train_attunefm_lite_fits_current_demo_profiles(tmp_path):
    config = build_training_config("smoke", output_dir=tmp_path, epochs=60)
    run = train_attunefm_lite(config, wandb_enabled=False)

    assert run.train_accuracy >= 0.70
    assert run.eval_accuracy >= 0.60
    assert run.checkpoint_path == tmp_path / "attunefm-lite-smoke-checkpoint.json"
    assert run.checkpoint_path.exists()
    assert (
        run.source_signal_path == tmp_path / "attunefm-lite-smoke-source-signals.jsonl"
    )
    assert run.source_signal_path.exists()
    assert run.checkin_path == tmp_path / "attunefm-lite-smoke-checkins.jsonl"
    assert run.checkin_path.exists()
    checkpoint = json.loads(run.checkpoint_path.read_text())
    assert checkpoint["config"]["accelerator"] == "cpu"
    assert checkpoint["artifacts"]["source_signal_records_path"] == str(
        run.source_signal_path
    )
    assert checkpoint["artifacts"]["checkin_records_path"] == str(run.checkin_path)
    assert checkpoint["metrics"]["eval_accuracy"] == run.eval_accuracy
    assert checkpoint["metrics"]["train_examples"] == len(run.train_examples)
    assert checkpoint["metrics"]["eval_examples"] == len(run.eval_examples)
    assert checkpoint["metrics"]["feature_columns"] == len(run.signal_keys)
    assert checkpoint["metrics"]["source_signal_records"] == 48960
    assert checkpoint["metrics"]["checkin_records"] == len(run.checkin_records)
    assert checkpoint["metrics"]["checkin_records"] == 20160
    assert checkpoint["metrics"]["checkin_captured_records"] == sum(
        record.answered for record in run.checkin_records
    )
    assert checkpoint["metrics"]["checkin_captured_records"] == 14457
    assert checkpoint["metrics"]["checkin_missing_records"] == sum(
        not record.answered for record in run.checkin_records
    )
    assert checkpoint["metrics"]["checkin_missing_records"] == 5703
    assert 0 < checkpoint["metrics"]["checkin_captured_records"] < 20160
    source_rows = [
        json.loads(line) for line in run.source_signal_path.read_text().splitlines()
    ]
    checkin_rows = [
        json.loads(line) for line in run.checkin_path.read_text().splitlines()
    ]
    assert len(source_rows) == checkpoint["metrics"]["source_signal_records"]
    assert len(checkin_rows) == checkpoint["metrics"]["checkin_records"]
    assert {"profile", "seed_offset", "day", "key", "axis", "source", "value"}.issubset(
        source_rows[0]
    )
    assert {
        "profile",
        "seed_offset",
        "day",
        "capture_modality",
        "answered",
        "missing_reason",
        "patient_response",
    }.issubset(checkin_rows[0])
    assert {row["capture_modality"] for row in checkin_rows} == {
        "voice",
        "photo",
        "video",
    }
    assert {row["answered"] for row in checkin_rows} == {True, False}
    assert {record.signal_key for record in run.checkin_records} >= {
        "voice_fatigue",
        "medication_tolerance",
        "skin_wound_change",
        "mobility_change",
    }
    assert {record.source for record in run.checkin_records} >= {
        "audio",
        "vision",
        "video",
    }
    assert min(record.day for record in run.checkin_records) == 0
    assert max(record.day for record in run.checkin_records) == config.days - 1
    assert any(not record.answered for record in run.checkin_records)
    assert all(
        record.value is None for record in run.checkin_records if not record.answered
    )
    assert all(
        record.missing_reason for record in run.checkin_records if not record.answered
    )
    assert {record.capture_modality for record in run.checkin_records} >= {
        "voice",
        "photo",
        "video",
    }
    assert all(
        record.patient_response
        for record in run.checkin_records
        if record.answered and record.capture_modality == "voice"
    )
    core_records = [record for record in run.checkin_records if not record.optional]
    optional_records = [record for record in run.checkin_records if record.optional]
    assert sum(record.answered for record in core_records) / len(core_records) > 0.80
    assert (
        sum(record.answered for record in optional_records) / len(optional_records)
        < 0.60
    )
    assert len(run.train_examples) > len(config.train_seed_offsets) * 8
    assert {example.period for example in run.train_examples} >= {
        "pre_flare",
        "flare_onset",
        "flare_peak",
        "recovery",
    }
    assert len({example.day for example in run.train_examples}) > 1
    assert {item.profile for item in run.evaluations} == {
        "office",
        "firefighter",
        "firefighter_asthma",
        "firefighter_recovery",
        "firefighter_dormant",
        "veteran",
        "autoimmune",
        "metabolic_pcos",
    }
    assert all(item.active_axes for item in run.evaluations)
    assert all(item.top_drivers for item in run.evaluations)


def test_one_year_checkins_cover_profiles_and_realistic_missing_days(tmp_path):
    config = build_training_config("one_year", output_dir=tmp_path, epochs=1)
    run = train_attunefm_lite(config, wandb_enabled=False)

    assert len(run.checkin_records) == 81760
    assert {record.profile for record in run.checkin_records} == {
        "office",
        "firefighter",
        "firefighter_asthma",
        "firefighter_recovery",
        "firefighter_dormant",
        "veteran",
        "autoimmune",
        "metabolic_pcos",
    }
    assert min(record.day for record in run.checkin_records) == 0
    assert max(record.day for record in run.checkin_records) == 364
    for profile in {record.profile for record in run.checkin_records}:
        profile_days = {
            record.day for record in run.checkin_records if record.profile == profile
        }
        assert len(profile_days) == 365

    captured = [record for record in run.checkin_records if record.answered]
    missing = [record for record in run.checkin_records if not record.answered]
    assert len(captured) == 58330
    assert len(missing) == 23430
    assert 0 < len(missing) < len(run.checkin_records)
    assert len(captured) + len(missing) == len(run.checkin_records)
    assert {"missed_day", "skipped_prompt", "optional_media_skipped"}.issubset(
        {record.missing_reason for record in missing}
    )
    assert {record.capture_modality for record in run.checkin_records} == {
        "voice",
        "photo",
        "video",
    }
    voice_responses = [
        record.patient_response
        for record in run.checkin_records
        if record.answered and record.capture_modality == "voice"
    ]
    assert any(
        "shift" in response or "work" in response for response in voice_responses
    )
    assert any(
        "flare" in response or "pain" in response for response in voice_responses
    )


def test_forecast_heads_learn_episode_onset_ahead_of_time(tmp_path):
    config = build_training_config("one_year", output_dir=tmp_path, epochs=300)
    run = train_attunefm_lite(config, wandb_enabled=False)

    assert tuple(run.forecast_metrics) == config.forecast_horizons
    for horizon, scores in run.forecast_metrics.items():
        # a relapsing episode within `horizon` days is genuinely predictable from the trailing
        # window — well above the 0.5 chance AUC, but not a saturated 1.0
        assert 0.65 < scores["auc"] < 0.95, (horizon, scores["auc"])
        assert 0.0 <= scores["base_rate"] <= 1.0
    checkpoint = json.loads(run.checkpoint_path.read_text())
    assert set(checkpoint["metrics"]["forecast"]) == {
        str(horizon) for horizon in config.forecast_horizons
    }


def test_train_attunefm_lite_logs_to_wandb_when_enabled(tmp_path, monkeypatch):
    events = []
    tables = []
    images = []
    plots = []

    class FakeRun:
        url = "https://wandb.local/run/123"

        def __enter__(self):
            events.append(("enter",))
            return self

        def __exit__(self, exc_type, exc, traceback):
            events.append(("exit", exc_type))

        def log(self, payload):
            events.append(("log", payload))

    class FakeWandb:
        class Table:
            def __init__(self, *, columns, data):
                self.columns = columns
                self.data = data
                tables.append(self)

        class Image:
            def __init__(self, path, *, caption):
                self.path = path
                self.caption = caption
                images.append(self)

        class plot:
            @staticmethod
            def bar(table, x, y, *, title):
                plots.append((table, x, y, title))
                return {"table": table, "x": x, "y": y, "title": title}

        def init(self, **kwargs):
            events.append(("init", kwargs))
            return FakeRun()

    monkeypatch.setitem(sys.modules, "wandb", FakeWandb())
    monkeypatch.setenv("WANDB_PROJECT", "attune-test")
    config = build_training_config("smoke", output_dir=tmp_path, epochs=10)

    run = train_attunefm_lite(config, wandb_enabled=True)

    assert run.wandb_url == "https://wandb.local/run/123"
    init_payload = next(payload for event, payload in events if event == "init")
    assert init_payload["project"] == "attune-test"
    assert init_payload["name"] == "attunefm-lite-smoke"
    logged = [event[1] for event in events if event[0] == "log"]
    assert {
        "train/accuracy",
        "eval/accuracy",
        "train/final_loss",
        "train/examples",
        "eval/examples",
        "train/source_signal_records",
        "train/checkin_records",
        "train/checkin_captured_records",
        "train/checkin_missing_records",
    }.issubset(logged[0])
    assert any("artifact/source_signal_records_path" in payload for payload in logged)
    assert any("artifact/checkin_records_path" in payload for payload in logged)
    assert any("profile/office/confidence" in payload for payload in logged)
    input_payload = next(
        payload for payload in logged if "training/input_examples" in payload
    )
    input_table = input_payload["training/input_examples"]
    assert input_table.columns[:4] == [
        "split",
        "example_index",
        "profile",
        "seed_offset",
    ]
    assert input_table.columns[4:6] == ["day", "period"]
    assert "hrv" in input_table.columns
    assert {row[0] for row in input_table.data} == {"train", "eval"}
    assert {row[2] for row in input_table.data} >= {"office", "firefighter"}
    assert {row[5] for row in input_table.data} >= {"pre_flare", "flare_peak"}
    assert (
        len(input_table.data)
        > (len(config.train_seed_offsets) + len(config.eval_seed_offsets)) * 8
    )
    checkin_payload = next(
        payload for payload in logged if "training/checkin_examples" in payload
    )
    checkin_table = checkin_payload["training/checkin_examples"]
    assert checkin_table.columns == [
        "profile",
        "seed_offset",
        "day",
        "signal_key",
        "source",
        "capture_modality",
        "optional",
        "answered",
        "value",
        "missing_reason",
        "patient_response",
        "prompt",
    ]
    assert len(checkin_table.data) <= 512
    assert {row[2] for row in checkin_table.data} >= {0, config.days - 1}
    assert {row[3] for row in checkin_table.data} >= {
        "voice_fatigue",
        "skin_wound_change",
        "mobility_change",
    }
    assert {row[5] for row in checkin_table.data} >= {"voice", "photo", "video"}
    assert {row[7] for row in checkin_table.data} == {True, False}
    visual_payload = next(
        payload for payload in logged if "plots/profile_confidence" in payload
    )
    assert visual_payload["plots/profile_confidence"]["title"] == "Profile confidence"
    assert (
        visual_payload["plots/input_split_counts"]["title"] == "Input examples by split"
    )
    assert (
        visual_payload["plots/forecast_auc"]["title"]
        == "Episode-forecast AUC by horizon"
    )
    assert (
        visual_payload["plots/eval_accuracy_by_period"]["title"]
        == "Diagnosis accuracy by day type"
    )
    heatmap = visual_payload["images/input_feature_heatmap"]
    assert heatmap.caption == "AttuneFM input feature z-score heatmap"
    assert Path(heatmap.path).read_bytes().startswith(b"\x89PNG")
    assert input_table in tables
    assert checkin_table in tables
    assert len(tables) == 6
    assert len(images) == 1
    assert len(plots) == 4


def test_train_attunefm_lite_logs_to_wandb_by_default(tmp_path, monkeypatch):
    events = []

    class FakeRun:
        url = "https://wandb.local/default"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def log(self, payload):
            events.append(payload)

    class FakeWandb:
        class Table:
            def __init__(self, *, columns, data):
                self.columns = columns
                self.data = data

        class Image:
            def __init__(self, path, *, caption):
                self.path = path
                self.caption = caption

        class plot:
            @staticmethod
            def bar(table, x, y, *, title):
                return {"table": table, "x": x, "y": y, "title": title}

        def init(self, **kwargs):
            events.append(kwargs)
            return FakeRun()

    monkeypatch.setitem(sys.modules, "wandb", FakeWandb())
    config = build_training_config("smoke", output_dir=tmp_path, epochs=10)

    run = train_attunefm_lite(config)

    assert run.wandb_url == "https://wandb.local/default"
    assert any("train/accuracy" in payload for payload in events)


def test_train_attunefm_lite_can_opt_out_of_default_wandb(tmp_path):
    config = build_training_config("smoke", output_dir=tmp_path, epochs=10)

    run = train_attunefm_lite(config, wandb_enabled=False)

    assert run.wandb_url is None


def test_train_cli_preserves_no_wandb_opt_out(tmp_path, monkeypatch):
    captured = {}

    def fake_train(config, *, wandb_enabled):
        captured["config"] = config
        captured["wandb_enabled"] = wandb_enabled
        return object()

    monkeypatch.setattr("attune.training.train_attunefm_lite", fake_train)
    monkeypatch.setattr("attune.training._render_run", lambda run: None)

    train_main(
        config="smoke",
        output_dir=tmp_path,
        epochs=1,
        accelerator=None,
        wandb=False,
    )

    assert captured["config"].output_dir == tmp_path
    assert captured["wandb_enabled"] is False
