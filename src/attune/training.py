from __future__ import annotations

import json
import os
import struct
import zlib
from binascii import crc32
from dataclasses import asdict, dataclass, replace
from importlib import import_module
from math import exp
from pathlib import Path
from shutil import which

import torch
import typer
import yaml
from rich.console import Console
from rich.table import Table

from attune.attunefm import (
    FEATURE_WINDOW,
    featurize_memory,
    monitoring_scores,
    series_by_day,
    window_feature_keys,
    window_features,
)
from attune.concordance_engine.concordance import BASELINE_SPAN
from attune.concordance_engine.engine import Engine, PACKS
from attune.concordance_engine.memory import Memory
from attune.datasets import DATASET_CATALOG, DEMO_DATASET_NAMES, DatasetStub
from attune.rook import ingest_daily_rook
from attune.synth import (
    ACTIVE_PERIODS,
    ATTUNEFM_PROFILES,
    PatientProfile,
    day_period,
    episode_onset_within,
    flare_window,
    flare_windows,
    generate,
)

console = Console()
CONFIG_DIR = Path("configs")
CHECKIN_LOG_LIMIT = 512
EXAMPLE_LOG_LIMIT = 400  # per split; an all-day dataset is thousands of rows
HEATMAP_ROW_LIMIT = 120  # per split; keeps the rendered PNG a readable height
MEMORY_CACHE: dict[tuple[str, int, int, str], Memory] = {}


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    pack: str
    dataset_names: tuple[str, ...]
    modalities: frozenset[str]
    heads: frozenset[str]
    accelerator: str
    max_hours: int


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    name: str
    pack: str
    dataset_names: tuple[str, ...]
    accelerator: str
    max_hours: int
    days: int
    epochs: int
    learning_rate: float
    train_seed_offsets: tuple[int, ...]
    eval_seed_offsets: tuple[int, ...]
    output_dir: Path
    weight_decay: float = (
        0.0  # L2 penalty; the linear head overfits 120 examples without it
    )
    feature_window: int = FEATURE_WINDOW  # trailing days summarised per signal
    forecast_horizons: tuple[int, ...] = (
        7,
        30,
    )  # days ahead for episode-onset forecasting
    source: str = (
        "generator"  # "generator" | "rook" — objective channel via the Rook interface
    )


@dataclass(frozen=True, slots=True)
class TrainingExample:
    profile: str
    seed_offset: int
    day: int
    period: str
    features: tuple[float, ...]
    forecast: tuple[
        int, ...
    ] = ()  # 1 per horizon: an episode onsets within that many days


@dataclass(frozen=True, slots=True)
class CheckinRecord:
    profile: str
    seed_offset: int
    day: int
    signal_key: str
    prompt: str
    source: str
    capture_modality: str
    optional: bool
    answered: bool
    value: float | None
    missing_reason: str
    patient_response: str


@dataclass(frozen=True, slots=True)
class ProfileEval:
    profile: str
    predicted_profile: str
    confidence: float
    active_axes: tuple[str, ...]
    top_drivers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrainingRun:
    config: TrainingConfig
    signal_keys: tuple[str, ...]
    train_examples: tuple[TrainingExample, ...]
    eval_examples: tuple[TrainingExample, ...]
    checkin_records: tuple[CheckinRecord, ...]
    train_accuracy: float
    eval_accuracy: float
    train_active_accuracy: float
    eval_active_accuracy: float
    eval_accuracy_by_period: dict[str, float]
    forecast_metrics: dict[int, dict[str, float]]
    final_loss: float
    checkpoint_path: Path
    source_signal_path: Path
    checkin_path: Path
    evaluations: tuple[ProfileEval, ...]
    hardware_note: str
    wandb_url: str | None = None


def _dataset_by_name() -> dict[str, DatasetStub]:
    return {dataset.name: dataset for dataset in DATASET_CATALOG}


def _default_datasets() -> tuple[str, ...]:
    return tuple(sorted(DEMO_DATASET_NAMES))


def _config_path(name: str) -> Path:
    path = Path(name)
    if path.suffix in {".yaml", ".yml"} or path.exists():
        return path
    return CONFIG_DIR / f"{name}.yaml"


def _tuple_ints(values: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(values, list | tuple):
        raise ValueError(f"training config field '{field}' must be a list")
    return tuple(int(value) for value in values)


def _tuple_strings(values: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(values, list | tuple):
        raise ValueError(f"training config field '{field}' must be a list")
    return tuple(str(value) for value in values)


def _load_training_config_file(name: str) -> TrainingConfig:
    path = _config_path(name)
    if not path.exists():
        raise ValueError(f"unknown training config or missing YAML: {name}")

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"training config '{path}' must contain a mapping")

    config_name = str(raw.get("name", path.stem))
    dataset_names = _tuple_strings(
        raw.get("dataset_names", _default_datasets()), field="dataset_names"
    )
    return TrainingConfig(
        name=config_name,
        pack=str(raw.get("pack", "attunefm")),
        dataset_names=dataset_names,
        accelerator=str(raw.get("accelerator", "cpu")),
        max_hours=int(raw.get("max_hours", 1)),
        days=int(raw.get("days", 90)),
        epochs=int(raw.get("epochs", 80)),
        learning_rate=float(raw.get("learning_rate", 0.08)),
        train_seed_offsets=_tuple_ints(
            raw.get("train_seed_offsets", (0,)), field="train_seed_offsets"
        ),
        eval_seed_offsets=_tuple_ints(
            raw.get("eval_seed_offsets", (101,)), field="eval_seed_offsets"
        ),
        output_dir=Path(raw.get("output_dir", f"runs/attunefm-{config_name}")),
        weight_decay=float(raw.get("weight_decay", 0.0)),
        feature_window=int(raw.get("feature_window", FEATURE_WINDOW)),
        forecast_horizons=_tuple_ints(
            raw.get("forecast_horizons", (7, 30)), field="forecast_horizons"
        ),
        source=str(raw.get("source", "generator")),
    )


def build_training_config(
    name: str = "smoke",
    *,
    output_dir: Path | str | None = None,
    epochs: int | None = None,
    accelerator: str | None = None,
    max_hours: int | None = None,
    datasets: tuple[str, ...] | None = None,
) -> TrainingConfig:
    config = _load_training_config_file(name)
    if datasets is not None:
        config = replace(config, dataset_names=datasets)
    if output_dir is not None:
        config = replace(config, output_dir=Path(output_dir))
    if epochs is not None:
        config = replace(config, epochs=epochs)
    if accelerator is not None:
        config = replace(config, accelerator=accelerator)
    if max_hours is not None:
        config = replace(config, max_hours=max_hours)
    return config


def build_training_plan(
    *,
    pack: str,
    datasets: tuple[str, ...],
    accelerator: str,
    max_hours: int,
) -> TrainingPlan:
    catalog = _dataset_by_name()
    unknown = tuple(name for name in datasets if name not in catalog)
    if unknown:
        raise ValueError(f"unknown datasets: {', '.join(unknown)}")

    selected = tuple(catalog[name] for name in datasets)
    modalities = frozenset(
        modality for dataset in selected for modality in dataset.modalities
    )
    heads = frozenset(head for dataset in selected for head in dataset.heads)
    return TrainingPlan(
        pack=pack,
        dataset_names=datasets,
        modalities=modalities,
        heads=heads,
        accelerator=accelerator,
        max_hours=max_hours,
    )


def load_env(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def hardware_note(accelerator: str) -> str:
    if "a100" not in accelerator.lower():
        return "local CPU/default runtime"
    if which("nvidia-smi") is None:
        return "A100 target config only; no local NVIDIA runtime detected"
    return "A100 host detected; current lightweight trainer is CPU-only"


def _profile_variant(profile: PatientProfile, seed_offset: int) -> PatientProfile:
    return replace(profile, seed=profile.seed + seed_offset)


def patient_memory(config: TrainingConfig, variant: PatientProfile) -> Memory:
    # generate() is a pure function of (pack, days, profile variant), but four consumers each
    # rebuilt the same timelines — ~69% of the calls were redundant.
    key = (config.pack, config.days, variant.seed, config.source)
    cached = MEMORY_CACHE.get(key)
    if cached is None:
        pack = PACKS[config.pack]
        cached = generate(pack, days=config.days, profile=variant)
        if config.source == "rook":
            # the objective (wearable) channel arrives via the Rook interface, not the generator
            cached = ingest_daily_rook(cached, pack, config.days)
        MEMORY_CACHE[key] = cached
    return cached


def _signal_keys(pack_name: str) -> tuple[str, ...]:
    return tuple(spec.key for spec in PACKS[pack_name].signals)


def _source_signal_records(config: TrainingConfig) -> int:
    seed_count = len(config.train_seed_offsets) + len(config.eval_seed_offsets)
    return (
        len(PACKS[config.pack].signals)
        * config.days
        * len(ATTUNEFM_PROFILES)
        * seed_count
    )


def _stable_percent(*parts: object) -> int:
    key = ":".join(str(part) for part in parts)
    return zlib.crc32(key.encode("utf-8")) % 100


def _capture_modality(source: str) -> str:
    return {
        "audio": "voice",
        "self_report": "voice",
        "vision": "photo",
        "video": "video",
    }.get(source, source)


def _missing_reason(
    profile: str, seed_offset: int, day: int, signal_key: str, *, optional: bool
) -> str:
    if _stable_percent(profile, seed_offset, day, "day") < 10:
        return "missed_day"
    if optional:
        if _stable_percent(profile, seed_offset, day, signal_key, "optional") < 55:
            return "optional_media_skipped"
        return ""
    if _stable_percent(profile, seed_offset, day, signal_key, "required") < 7:
        return "skipped_prompt"
    return ""


def _patient_response(
    profile: str, signal_key: str, value: float, *, capture_modality: str
) -> str:
    if capture_modality == "photo":
        return f"Photo shared for {signal_key.replace('_', ' ')}; change score {value:.2f}."
    if capture_modality == "video":
        return f"Movement video shared; mobility change feels around {value:.2f}."

    contexts = {
        "office": "Work feels mostly manageable, but the desk day is showing up.",
        "firefighter": "After the shift, smoke and heat load are still in my body.",
        "firefighter_asthma": "Breathing is the thing I am watching after the shift.",
        "firefighter_recovery": "Post-fire recovery is taking more out of me today.",
        "firefighter_dormant": "I am trying to catch the early warning signs before work.",
        "veteran": "Sleep, pain, and focus are all tied together today.",
        "autoimmune": "This feels like a flare day, with fatigue and pain more noticeable.",
        "metabolic_pcos": "Energy, cravings, and glucose swings feel more connected today.",
    }
    signal = signal_key.replace("_", " ")
    return (
        f"{contexts.get(profile, 'Here is my voice check-in.')} {signal}: {value:.2f}."
    )


def _checkin_records(config: TrainingConfig) -> tuple[CheckinRecord, ...]:
    records = []
    seed_offsets = (*config.train_seed_offsets, *config.eval_seed_offsets)
    pack = PACKS[config.pack]
    checkin_keys = tuple(item.signal_key for item in pack.checkin)
    for profile_name, profile in ATTUNEFM_PROFILES.items():
        for seed_offset in seed_offsets:
            memory = patient_memory(config, _profile_variant(profile, seed_offset))
            # One dense pass instead of re-scanning the memory for every (day, prompt) — this
            # loop was the single biggest cost in a run.
            values = series_by_day(memory, checkin_keys, config.days)
            for day in range(config.days):
                for item in pack.checkin:
                    missing_reason = _missing_reason(
                        profile_name,
                        seed_offset,
                        day,
                        item.signal_key,
                        optional=item.optional,
                    )
                    answered = missing_reason == ""
                    value = values[item.signal_key][day] if answered else None
                    capture_modality = _capture_modality(item.source)
                    records.append(
                        CheckinRecord(
                            profile=profile_name,
                            seed_offset=seed_offset,
                            day=day,
                            signal_key=item.signal_key,
                            prompt=item.prompt,
                            source=item.source,
                            capture_modality=capture_modality,
                            optional=item.optional,
                            answered=answered,
                            value=value,
                            missing_reason=missing_reason,
                            patient_response=(
                                _patient_response(
                                    profile_name,
                                    item.signal_key,
                                    value,
                                    capture_modality=capture_modality,
                                )
                                if value is not None
                                else ""
                            ),
                        )
                    )
    return tuple(records)


def _write_source_signal_records(config: TrainingConfig, path: Path) -> int:
    count = 0
    seed_offsets = (*config.train_seed_offsets, *config.eval_seed_offsets)
    with path.open("w") as file:
        for profile_name, profile in ATTUNEFM_PROFILES.items():
            for seed_offset in seed_offsets:
                memory = patient_memory(config, _profile_variant(profile, seed_offset))
                for signal in sorted(
                    memory.signals, key=lambda item: (item.day, item.key, item.source)
                ):
                    file.write(
                        json.dumps(
                            {
                                "profile": profile_name,
                                "seed_offset": seed_offset,
                                "day": signal.day,
                                "key": signal.key,
                                "axis": str(signal.axis),
                                "source": signal.source,
                                "value": signal.value,
                                "note": signal.note,
                            }
                        )
                        + "\n"
                    )
                    count += 1
    return count


def _write_checkin_records(path: Path, records: tuple[CheckinRecord, ...]) -> None:
    with path.open("w") as file:
        for record in records:
            file.write(json.dumps(asdict(record)) + "\n")


def _examples(
    config: TrainingConfig, seed_offsets: tuple[int, ...]
) -> tuple[TrainingExample, ...]:
    # Every day of the timeline, not a handful of landmark days: the generator produces a full
    # year per patient and sampling 5 days threw away ~99% of it.
    signal_keys = _signal_keys(config.pack)
    first_day = max(BASELINE_SPAN, config.feature_window)
    examples = []
    for profile_name, profile in ATTUNEFM_PROFILES.items():
        for seed_offset in seed_offsets:
            variant = _profile_variant(profile, seed_offset)
            memory = patient_memory(config, variant)
            # Episode timing is patient-specific, so day labels come from this patient's own
            # episodes rather than one shared window.
            windows = flare_windows(config.days, seed=variant.seed)
            values = series_by_day(memory, signal_keys, config.days)
            for day in range(first_day, config.days):
                examples.append(
                    TrainingExample(
                        profile=profile_name,
                        seed_offset=seed_offset,
                        day=day,
                        period=day_period(day, windows),
                        features=window_features(
                            values, signal_keys, day=day, window=config.feature_window
                        ),
                        forecast=episode_onset_within(
                            day, windows, config.forecast_horizons
                        ),
                    )
                )
    return tuple(examples)


def _mean_scale(
    examples: tuple[TrainingExample, ...],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    width = len(examples[0].features)
    means = tuple(
        sum(example.features[i] for example in examples) / len(examples)
        for i in range(width)
    )
    scales = []
    for i, mean in enumerate(means):
        variance = sum((example.features[i] - mean) ** 2 for example in examples) / len(
            examples
        )
        # Guard against divide-by-zero only. Flooring at 1.0 would leave every sub-unit-variance
        # signal (spo2, valence, breathlessness, voice_fatigue) effectively unstandardized, so the
        # head learns to ignore them — that alone cost ~half the single-day accuracy.
        scales.append(max(variance**0.5, 1e-6))
    return means, tuple(scales)


def _standardize(
    values: tuple[float, ...], means: tuple[float, ...], scales: tuple[float, ...]
) -> tuple[float, ...]:
    return tuple(
        (value - mean) / scale
        for value, mean, scale in zip(values, means, scales, strict=True)
    )


def _softmax(logits: tuple[float, ...]) -> tuple[float, ...]:
    shifted = tuple(logit - max(logits) for logit in logits)
    exps = tuple(exp(logit) for logit in shifted)
    total = sum(exps) or 1.0
    return tuple(value / total for value in exps)


def _predict_proba(
    features: tuple[float, ...],
    weights: list[list[float]],
    bias: list[float],
) -> tuple[float, ...]:
    logits = tuple(
        sum(weight * value for weight, value in zip(row, features, strict=True))
        + bias[i]
        for i, row in enumerate(weights)
    )
    return _softmax(logits)


def _select_device(accelerator: str) -> str:
    if "a100" in accelerator.lower() and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _train_linear_classifier(
    examples: tuple[TrainingExample, ...],
    labels: tuple[str, ...],
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float = 0.0,
    device: str = "cpu",
) -> tuple[list[list[float]], list[float], tuple[float, ...], tuple[float, ...], float]:
    # A linear-probe head (multinomial logistic regression) trained with Adam. Adam converges
    # reliably where hand-tuned SGD underfit; weight_decay is the L2 that keeps 272 params from
    # overfitting 120 examples. Weights are extracted back to plain lists so eval/checkpoint code
    # stays framework-free — the sequence encoder ("AttuneFM proper") swaps in behind this shape.
    means, scales = _mean_scale(examples)
    label_index = {label: i for i, label in enumerate(labels)}
    features = _feature_matrix(examples, means, scales, device)
    targets = torch.tensor(
        [label_index[example.profile] for example in examples],
        dtype=torch.long,
        device=device,
    )
    torch.manual_seed(0)
    head = torch.nn.Linear(features.shape[1], len(labels)).to(device)
    optimizer = torch.optim.Adam(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    final_loss = 0.0
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(head(features), targets)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())

    weights = head.weight.detach().cpu().tolist()
    bias = head.bias.detach().cpu().tolist()
    return weights, bias, means, scales, final_loss


def _accuracy(
    examples: tuple[TrainingExample, ...],
    labels: tuple[str, ...],
    weights: list[list[float]],
    bias: list[float],
    means: tuple[float, ...],
    scales: tuple[float, ...],
) -> float:
    correct = 0
    for example in examples:
        x = _standardize(example.features, means, scales)
        probs = _predict_proba(x, weights, bias)
        predicted = labels[max(range(len(labels)), key=probs.__getitem__)]
        correct += predicted == example.profile
    return correct / len(examples)


def _accuracy_by_period(
    examples: tuple[TrainingExample, ...],
    labels: tuple[str, ...],
    weights: list[list[float]],
    bias: list[float],
    means: tuple[float, ...],
    scales: tuple[float, ...],
) -> dict[str, float]:
    tally: dict[str, list[int]] = {}
    for example in examples:
        x = _standardize(example.features, means, scales)
        probs = _predict_proba(x, weights, bias)
        predicted = labels[max(range(len(labels)), key=probs.__getitem__)]
        bucket = tally.setdefault(example.period, [0, 0])
        bucket[0] += predicted == example.profile
        bucket[1] += 1
    return {period: correct / total for period, (correct, total) in tally.items()}


def _active_accuracy(
    examples: tuple[TrainingExample, ...],
    labels: tuple[str, ...],
    weights: list[list[float]],
    bias: list[float],
    means: tuple[float, ...],
    scales: tuple[float, ...],
    *,
    active: frozenset[str],
) -> float:
    subset = tuple(example for example in examples if example.period in active)
    return _accuracy(subset, labels, weights, bias, means, scales) if subset else 0.0


def _feature_matrix(
    examples: tuple[TrainingExample, ...],
    means: tuple[float, ...],
    scales: tuple[float, ...],
    device: str,
) -> torch.Tensor:
    return torch.tensor(
        [_standardize(example.features, means, scales) for example in examples],
        dtype=torch.float32,
        device=device,
    )


def _binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    positives = labels.sum().item()
    negatives = labels.numel() - positives
    if positives == 0 or negatives == 0:
        return 0.5  # undefined without both classes; report chance
    order = torch.argsort(scores)
    ranked = labels[order]
    ranks = torch.arange(1, labels.numel() + 1, dtype=torch.float32)
    rank_sum = ranks[ranked == 1].sum().item()
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _forecast_head(
    train_examples: tuple[TrainingExample, ...],
    eval_examples: tuple[TrainingExample, ...],
    horizons: tuple[int, ...],
    means: tuple[float, ...],
    scales: tuple[float, ...],
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: str,
) -> tuple[list[list[float]], list[float], dict[int, dict[str, float]]]:
    # One shared head with a sigmoid output per horizon (multi-label): "does an episode start
    # within h days?" Rare positives at short horizons, so weight the loss by class balance.
    features = _feature_matrix(train_examples, means, scales, device)
    targets = torch.tensor(
        [example.forecast for example in train_examples],
        dtype=torch.float32,
        device=device,
    )
    base_rate = targets.mean(dim=0).clamp(1e-6, 1 - 1e-6)
    torch.manual_seed(0)
    head = torch.nn.Linear(features.shape[1], len(horizons)).to(device)
    optimizer = torch.optim.Adam(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=(1 - base_rate) / base_rate)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss_fn(head(features), targets).backward()
        optimizer.step()

    eval_features = _feature_matrix(eval_examples, means, scales, device)
    eval_targets = torch.tensor(
        [example.forecast for example in eval_examples],
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        probabilities = torch.sigmoid(head(eval_features))
    metrics: dict[int, dict[str, float]] = {}
    for index, horizon in enumerate(horizons):
        scores = probabilities[:, index]
        labels = eval_targets[:, index]
        predicted = scores >= 0.5
        true_positive = (predicted & (labels == 1)).sum().item()
        false_positive = (predicted & (labels == 0)).sum().item()
        false_negative = ((~predicted) & (labels == 1)).sum().item()
        metrics[horizon] = {
            "base_rate": labels.mean().item(),
            "auc": _binary_auc(scores, labels),
            "precision": true_positive / max(true_positive + false_positive, 1),
            "recall": true_positive / max(true_positive + false_negative, 1),
        }
    return (
        head.weight.detach().cpu().tolist(),
        head.bias.detach().cpu().tolist(),
        metrics,
    )


def _profile_eval(
    config: TrainingConfig,
    labels: tuple[str, ...],
    weights: list[list[float]],
    bias: list[float],
    means: tuple[float, ...],
    scales: tuple[float, ...],
) -> tuple[ProfileEval, ...]:
    day = flare_window(config.days).midpoint
    signal_keys = _signal_keys(config.pack)
    evaluations = []
    for profile_name, profile in ATTUNEFM_PROFILES.items():
        memory = patient_memory(
            config, _profile_variant(profile, config.eval_seed_offsets[0])
        )
        engine = Engine(PACKS[config.pack], memory)
        features = featurize_memory(PACKS[config.pack], memory, day=day)
        values = series_by_day(memory, signal_keys, config.days)
        x = _standardize(
            window_features(values, signal_keys, day=day, window=config.feature_window),
            means,
            scales,
        )
        probs = _predict_proba(x, weights, bias)
        predicted_index = max(range(len(labels)), key=probs.__getitem__)
        active_axes = tuple(
            axis
            for axis, load in sorted(features.axis_loads.items())
            if load / 6.0 >= 0.25
        )
        scores = monitoring_scores(engine, day=day)
        evaluations.append(
            ProfileEval(
                profile=profile_name,
                predicted_profile=labels[predicted_index],
                confidence=probs[predicted_index],
                active_axes=active_axes,
                top_drivers=scores.top_drivers[:3],
            )
        )
    return tuple(evaluations)


def _wandb_config(config: TrainingConfig) -> dict[str, object]:
    return {
        **asdict(config),
        "output_dir": str(config.output_dir),
        "dataset_names": list(config.dataset_names),
        "train_seed_offsets": list(config.train_seed_offsets),
        "eval_seed_offsets": list(config.eval_seed_offsets),
    }


def _sample_examples(
    examples: tuple[TrainingExample, ...], *, limit: int = EXAMPLE_LOG_LIMIT
) -> tuple[TrainingExample, ...]:
    # An all-day dataset is thousands of rows and mostly baseline days. Keep every non-baseline
    # day (rare and the interesting ones) and evenly sample baseline to fill the budget, so the
    # logged table and heatmap stay small without ever dropping a flare.
    notable = tuple(item for item in examples if item.period != "baseline")
    baseline = tuple(item for item in examples if item.period == "baseline")
    if len(notable) >= limit:
        return _even_sample(notable, limit=limit)
    return (*notable, *_even_sample(baseline, limit=limit - len(notable)))


def _even_sample(items: tuple, *, limit: int) -> tuple:
    if limit <= 0:
        return ()
    if len(items) <= limit:
        return items
    step = (len(items) - 1) / (limit - 1) if limit > 1 else 0.0
    return tuple(items[round(index * step)] for index in range(limit))


def _wandb_examples_table(wandb, run: TrainingRun):
    columns = [
        "split",
        "example_index",
        "profile",
        "seed_offset",
        "day",
        "period",
        *run.signal_keys,
    ]
    rows = []
    for split, examples in (
        ("train", _sample_examples(run.train_examples)),
        ("eval", _sample_examples(run.eval_examples)),
    ):
        for index, example in enumerate(examples):
            rows.append(
                [
                    split,
                    index,
                    example.profile,
                    example.seed_offset,
                    example.day,
                    example.period,
                    *example.features,
                ]
            )
    return wandb.Table(columns=columns, data=rows)


def _wandb_profile_confidence_table(wandb, run: TrainingRun):
    return wandb.Table(
        columns=["profile", "predicted_profile", "confidence", "correct"],
        data=[
            [
                item.profile,
                item.predicted_profile,
                item.confidence,
                item.profile == item.predicted_profile,
            ]
            for item in run.evaluations
        ],
    )


def _wandb_input_split_table(wandb, run: TrainingRun):
    return wandb.Table(
        columns=["split", "examples"],
        data=[
            ["train", len(run.train_examples)],
            ["eval", len(run.eval_examples)],
        ],
    )


def _wandb_forecast_table(wandb, run: TrainingRun):
    return wandb.Table(
        columns=["horizon", "auc", "precision", "recall", "base_rate"],
        data=[
            [
                f"{horizon}d",
                scores["auc"],
                scores["precision"],
                scores["recall"],
                scores["base_rate"],
            ]
            for horizon, scores in sorted(run.forecast_metrics.items())
        ],
    )


def _wandb_period_accuracy_table(wandb, run: TrainingRun):
    return wandb.Table(
        columns=["period", "eval_accuracy"],
        data=[
            [period, accuracy]
            for period, accuracy in sorted(run.eval_accuracy_by_period.items())
        ],
    )


def _sample_checkin_records(
    records: tuple[CheckinRecord, ...], *, limit: int = CHECKIN_LOG_LIMIT
) -> tuple[CheckinRecord, ...]:
    if len(records) <= limit:
        return records
    step = (len(records) - 1) / (limit - 1)
    sampled = [records[round(index * step)] for index in range(limit)]
    required_predicates = (
        lambda record: record.capture_modality == "voice",
        lambda record: record.capture_modality == "photo",
        lambda record: record.capture_modality == "video",
        lambda record: record.answered,
        lambda record: not record.answered,
    )
    replace_index = max(0, limit - len(required_predicates))
    for predicate in required_predicates:
        if any(predicate(record) for record in sampled):
            continue
        replacement = next(record for record in records if predicate(record))
        sampled[replace_index] = replacement
        replace_index += 1
    return tuple(sampled)


def _wandb_checkin_table(wandb, run: TrainingRun):
    return wandb.Table(
        columns=[
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
        ],
        data=[
            [
                record.profile,
                record.seed_offset,
                record.day,
                record.signal_key,
                record.source,
                record.capture_modality,
                record.optional,
                record.answered,
                record.value,
                record.missing_reason,
                record.patient_response,
                record.prompt,
            ]
            for record in _sample_checkin_records(run.checkin_records)
        ],
    )


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def _write_rgb_png(path: Path, width: int, height: int, rows: list[bytes]) -> None:
    raw = b"".join(b"\x00" + row for row in rows)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(payload)


def _heat_color(value: float) -> bytes:
    clipped = max(-3.0, min(3.0, value)) / 3.0
    if clipped >= 0:
        return bytes((255, round(255 * (1 - clipped)), round(255 * (1 - clipped))))
    return bytes((round(255 * (1 + clipped)), round(255 * (1 + clipped)), 255))


def _write_feature_heatmap(run: TrainingRun) -> Path:
    cell_size = 8
    examples = (
        *_sample_examples(run.train_examples, limit=HEATMAP_ROW_LIMIT),
        *_sample_examples(run.eval_examples, limit=HEATMAP_ROW_LIMIT),
    )
    width = max(1, len(run.signal_keys) * cell_size)
    height = max(1, len(examples) * cell_size)
    rows = []
    for example in examples:
        row = b"".join(_heat_color(value) * cell_size for value in example.features)
        rows.extend([row] * cell_size)

    path = run.config.output_dir / f"attunefm-lite-{run.config.name}-input-heatmap.png"
    _write_rgb_png(path, width, height, rows)
    return path


def _log_wandb(run: TrainingRun) -> str | None:
    try:
        wandb = import_module("wandb")
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging requested but the wandb package is not installed. "
            "Install project dependencies with `uv sync` or pass `--no-wandb`."
        ) from exc

    project = os.environ.get("WANDB_PROJECT", "attune-ai")
    entity = os.environ.get("WANDB_ENTITY") or None
    mode = os.environ.get("WANDB_MODE") or None
    name = f"attunefm-lite-{run.config.name}"
    init_kwargs = {
        "project": project,
        "entity": entity,
        "mode": mode,
        "name": name,
        "config": _wandb_config(run.config),
        "tags": ("attunefm", run.config.name, run.config.accelerator),
    }
    init_kwargs = {
        key: value for key, value in init_kwargs.items() if value is not None
    }
    with wandb.init(**init_kwargs) as wandb_run:
        wandb_run.log(
            {
                "train/accuracy": run.train_accuracy,
                "eval/accuracy": run.eval_accuracy,
                "eval/active_accuracy": run.eval_active_accuracy,
                "train/active_accuracy": run.train_active_accuracy,
                "train/final_loss": run.final_loss,
                "train/epochs": run.config.epochs,
                "train/examples": len(run.train_examples),
                "eval/examples": len(run.eval_examples),
                "train/feature_columns": len(run.signal_keys),
                "train/source_signal_records": _source_signal_records(run.config),
                "train/checkin_records": len(run.checkin_records),
                "train/checkin_captured_records": sum(
                    record.answered for record in run.checkin_records
                ),
                "train/checkin_missing_records": sum(
                    not record.answered for record in run.checkin_records
                ),
                "train/day_types": len({e.period for e in run.train_examples}),
                **{
                    f"forecast/auc_{horizon}d": scores["auc"]
                    for horizon, scores in run.forecast_metrics.items()
                },
            }
        )
        wandb_run.log({"training/input_examples": _wandb_examples_table(wandb, run)})
        wandb_run.log({"training/checkin_examples": _wandb_checkin_table(wandb, run)})
        profile_table = _wandb_profile_confidence_table(wandb, run)
        split_table = _wandb_input_split_table(wandb, run)
        forecast_table = _wandb_forecast_table(wandb, run)
        period_table = _wandb_period_accuracy_table(wandb, run)
        heatmap_path = _write_feature_heatmap(run)
        wandb_run.log(
            {
                "plots/profile_confidence": wandb.plot.bar(
                    profile_table,
                    "profile",
                    "confidence",
                    title="Profile confidence",
                ),
                "plots/input_split_counts": wandb.plot.bar(
                    split_table,
                    "split",
                    "examples",
                    title="Input examples by split",
                ),
                "plots/forecast_auc": wandb.plot.bar(
                    forecast_table,
                    "horizon",
                    "auc",
                    title="Episode-forecast AUC by horizon",
                ),
                "plots/eval_accuracy_by_period": wandb.plot.bar(
                    period_table,
                    "period",
                    "eval_accuracy",
                    title="Diagnosis accuracy by day type",
                ),
                "images/input_feature_heatmap": wandb.Image(
                    str(heatmap_path),
                    caption="AttuneFM input feature z-score heatmap",
                ),
            }
        )
        for item in run.evaluations:
            wandb_run.log(
                {
                    f"profile/{item.profile}/confidence": item.confidence,
                    f"profile/{item.profile}/correct": item.profile
                    == item.predicted_profile,
                }
            )
        wandb_run.log(
            {
                "artifact/checkpoint_path": str(run.checkpoint_path),
                "artifact/source_signal_records_path": str(run.source_signal_path),
                "artifact/checkin_records_path": str(run.checkin_path),
            }
        )
        return getattr(wandb_run, "url", None)


def train_attunefm_lite(
    config: TrainingConfig | None = None, *, wandb_enabled: bool | None = None
) -> TrainingRun:
    config = config or build_training_config("one_year")
    load_env()
    build_training_plan(
        pack=config.pack,
        datasets=config.dataset_names,
        accelerator=config.accelerator,
        max_hours=config.max_hours,
    )
    labels = tuple(ATTUNEFM_PROFILES)
    train_examples = _examples(config, config.train_seed_offsets)
    eval_examples = _examples(config, config.eval_seed_offsets)
    checkin_records = _checkin_records(config)
    device = _select_device(config.accelerator)
    weights, bias, means, scales, final_loss = _train_linear_classifier(
        train_examples,
        labels,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=device,
    )
    train_accuracy = _accuracy(train_examples, labels, weights, bias, means, scales)
    eval_accuracy = _accuracy(eval_examples, labels, weights, bias, means, scales)
    active = ACTIVE_PERIODS
    train_active_accuracy = _active_accuracy(
        train_examples, labels, weights, bias, means, scales, active=active
    )
    eval_active_accuracy = _active_accuracy(
        eval_examples, labels, weights, bias, means, scales, active=active
    )
    eval_accuracy_by_period = _accuracy_by_period(
        eval_examples, labels, weights, bias, means, scales
    )
    forecast_weights, forecast_bias, forecast_metrics = _forecast_head(
        train_examples,
        eval_examples,
        config.forecast_horizons,
        means,
        scales,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=device,
    )
    evaluations = _profile_eval(config, labels, weights, bias, means, scales)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / f"attunefm-lite-{config.name}-checkpoint.json"
    source_signal_path = (
        config.output_dir / f"attunefm-lite-{config.name}-source-signals.jsonl"
    )
    checkin_path = config.output_dir / f"attunefm-lite-{config.name}-checkins.jsonl"
    source_signal_records = _write_source_signal_records(config, source_signal_path)
    _write_checkin_records(checkin_path, checkin_records)
    checkpoint = {
        "schema": "attunefm-lite-linear-v1",
        "config": {**asdict(config), "output_dir": str(config.output_dir)},
        "artifacts": {
            "checkpoint_path": str(checkpoint_path),
            "source_signal_records_path": str(source_signal_path),
            "checkin_records_path": str(checkin_path),
        },
        "day_types": sorted({example.period for example in train_examples}),
        "labels": labels,
        "signal_keys": window_feature_keys(PACKS[config.pack]),
        "feature_mean": means,
        "feature_scale": scales,
        "weights": weights,
        "bias": bias,
        "forecast_weights": forecast_weights,
        "forecast_bias": forecast_bias,
        "forecast_horizons": list(config.forecast_horizons),
        "metrics": {
            "train_accuracy": train_accuracy,
            "eval_accuracy": eval_accuracy,
            "train_active_accuracy": train_active_accuracy,
            "eval_active_accuracy": eval_active_accuracy,
            "eval_accuracy_by_period": eval_accuracy_by_period,
            "forecast": {
                str(horizon): scores for horizon, scores in forecast_metrics.items()
            },
            "final_loss": final_loss,
            "train_examples": len(train_examples),
            "eval_examples": len(eval_examples),
            "feature_columns": len(window_feature_keys(PACKS[config.pack])),
            "source_signal_records": source_signal_records,
            "checkin_records": len(checkin_records),
            "checkin_captured_records": sum(
                record.answered for record in checkin_records
            ),
            "checkin_missing_records": sum(
                not record.answered for record in checkin_records
            ),
        },
    }
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2))
    run = TrainingRun(
        config=config,
        signal_keys=window_feature_keys(PACKS[config.pack]),
        train_examples=train_examples,
        eval_examples=eval_examples,
        checkin_records=checkin_records,
        train_accuracy=train_accuracy,
        eval_accuracy=eval_accuracy,
        train_active_accuracy=train_active_accuracy,
        eval_active_accuracy=eval_active_accuracy,
        eval_accuracy_by_period=eval_accuracy_by_period,
        forecast_metrics=forecast_metrics,
        final_loss=final_loss,
        checkpoint_path=checkpoint_path,
        source_signal_path=source_signal_path,
        checkin_path=checkin_path,
        evaluations=evaluations,
        hardware_note=hardware_note(config.accelerator),
    )
    should_log_wandb = True if wandb_enabled is None else wandb_enabled
    if should_log_wandb:
        run = replace(run, wandb_url=_log_wandb(run))
    return run


def _render_plan(plan: TrainingPlan, *, config_name: str) -> None:
    console.rule(f"[bold]{plan.pack} training plan[/]")
    console.print(f"[bold]Config:[/] {config_name}")
    console.print(f"[bold]Accelerator:[/] {plan.accelerator}")
    console.print(f"[bold]Budget:[/] {plan.max_hours} hours")
    console.print(f"[bold]Datasets:[/] {', '.join(plan.dataset_names)}")
    console.print(f"[bold]Modalities:[/] {', '.join(sorted(plan.modalities))}")
    console.print(f"[bold]Heads:[/] {', '.join(sorted(plan.heads))}")

    table = Table("stage", "intent")
    table.add_row("1. ingest", "download/normalize public sources into typed Signals")
    table.add_row(
        "2. pretrain", "masked multimodal sequence modeling over personal windows"
    )
    table.add_row(
        "3. adapt",
        "attach task heads for recovery, anomaly, pain, cognitive, and visible change",
    )
    table.add_row(
        "4. evaluate",
        "hold out subjects and report calibration before any clinical claim",
    )
    console.print(table)


def _render_run(run: TrainingRun) -> None:
    console.rule(f"[bold]{run.config.pack} real training[/]")
    console.print(f"[bold]Config:[/] {run.config.name}")
    console.print(f"[bold]Accelerator target:[/] {run.config.accelerator}")
    console.print(f"[bold]Hardware:[/] {run.hardware_note}")
    console.print(f"[bold]Epochs:[/] {run.config.epochs}")
    console.print(
        "[bold]Sensor records:[/] "
        f"{_source_signal_records(run.config):,} generated signal rows"
    )
    console.print(
        f"[bold]Check-in turns:[/] {len(run.checkin_records):,} simulated daily prompts"
    )
    console.print(
        "[bold]Captured check-ins:[/] "
        f"{sum(record.answered for record in run.checkin_records):,} responses "
        f"/ {sum(not record.answered for record in run.checkin_records):,} missed"
    )
    period_count = len({example.period for example in run.train_examples})
    console.print(
        "[bold]Temporal examples:[/] "
        f"{len(run.train_examples):,} train / {len(run.eval_examples):,} eval "
        f"across {period_count} day types"
    )
    console.print(f"[bold]Feature columns:[/] {len(run.signal_keys)}")
    console.print(
        f"[bold]Diagnosis eval accuracy:[/] {run.eval_accuracy:.0%} "
        f"(active/drift days {run.eval_active_accuracy:.0%})"
    )
    console.print(f"[bold]Final loss:[/] {run.final_loss:.3f}")
    console.print(f"[bold]Checkpoint:[/] {run.checkpoint_path}")
    console.print(f"[bold]Source signal data:[/] {run.source_signal_path}")
    console.print(f"[bold]Check-in data:[/] {run.checkin_path}")
    if run.wandb_url:
        console.print(f"[bold]W&B:[/] {run.wandb_url}")

    eval_table = Table(
        "profile", "prediction", "confidence", "active axes", "top drivers"
    )
    for item in run.evaluations:
        eval_table.add_row(
            item.profile,
            item.predicted_profile,
            f"{item.confidence:.0%}",
            ", ".join(item.active_axes),
            ", ".join(item.top_drivers),
        )
    console.print(eval_table)

    forecast_table = Table(
        "episode forecast", "base rate", "AUC", "precision", "recall"
    )
    for horizon, scores in sorted(run.forecast_metrics.items()):
        forecast_table.add_row(
            f"within {horizon}d",
            f"{scores['base_rate']:.0%}",
            f"{scores['auc']:.3f}",
            f"{scores['precision']:.2f}",
            f"{scores['recall']:.2f}",
        )
    console.print(forecast_table)


def main(
    config: str = typer.Option(
        "one_year",
        help="Training config preset: smoke | one_year | a100.",
    ),
    pack: str | None = typer.Option(None, help="Condition pack override."),
    datasets: str = typer.Option(
        "",
        help="Optional comma-separated dataset override from attune.datasets.",
    ),
    accelerator: str | None = typer.Option(
        None, help="Target training accelerator override."
    ),
    max_hours: int | None = typer.Option(
        None, help="Maximum planning budget override."
    ),
) -> None:
    names = tuple(name.strip() for name in datasets.split(",") if name.strip()) or None
    try:
        training_config = build_training_config(
            config,
            datasets=names,
            accelerator=accelerator,
            max_hours=max_hours,
        )
        if pack is not None:
            training_config = replace(training_config, pack=pack)
        plan = build_training_plan(
            pack=training_config.pack,
            datasets=training_config.dataset_names,
            accelerator=training_config.accelerator,
            max_hours=training_config.max_hours,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    _render_plan(plan, config_name=training_config.name)


def train_main(
    config: str = typer.Option(
        "one_year",
        help="Training config preset: smoke | one_year | a100.",
    ),
    output_dir: Path | None = typer.Option(None, help="Checkpoint output directory."),
    epochs: int | None = typer.Option(
        None, help="Epoch override for local smoke tests."
    ),
    accelerator: str | None = typer.Option(None, help="Target accelerator override."),
    wandb: bool = typer.Option(
        True,
        "--wandb/--no-wandb",
        help="Log metrics to Weights & Biases.",
    ),
) -> None:
    try:
        training_config = build_training_config(
            config,
            output_dir=output_dir,
            epochs=epochs,
            accelerator=accelerator,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    _render_run(train_attunefm_lite(training_config, wandb_enabled=wandb))


def run() -> None:
    typer.run(main)


def run_train() -> None:
    typer.run(train_main)


if __name__ == "__main__":
    run()
