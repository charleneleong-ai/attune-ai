# Intraday features — high-fidelity forecast experiment

**Hypothesis.** Terra exposes intraday `*_samples` arrays, not just daily summaries. A blunted
circadian *amplitude* (the autonomic rhythm flattening as an episode approaches) is a real
prodromal marker the daily mean cannot carry. If we derive amplitude features from the intraday
samples and feed them to the forecast head, episode-onset forecasting should improve.

**Setup.** Baseline = `one_year` (68 daily features). Treatment = `one_year_intraday` — identical
config plus 10 intraday amplitude features (`amp_*`, `mean_amp_*` per wearable signal), derived from
the `heart_rate_data.detailed.*`, `oxygen_data.saturation_samples`, `glucose_data.
blood_glucose_samples` arrays that `to_terra_day` carries and `signals_from_terra` recovers.

**Per-head routing.** Intraday features encode episode *timing*, which is irrelevant to *which*
profile a patient has. So they go to the **forecast head only**; the diagnosis head keeps the daily
features and its own standardisation. This is the design that makes the added features free of
diagnosis cost.

## Result (eval, 8040 train / 2680 eval days)

| metric | baseline | +intraday | Δ |
|---|---|---|---|
| forecast AUC @ 7d | 0.765 | 0.948 | **+0.182** |
| forecast AUC @ 30d | 0.773 | 0.807 | **+0.034** |
| diagnosis accuracy | 0.982 | 0.982 | **±0.000** |

## Verdict

- **The pipeline works end-to-end.** Features derived from Terra intraday samples flow into the
  forecast head and lift it — a genuine demonstration of the high-fidelity path, not just carrying
  samples in the payload.
- **Per-head routing is free.** Diagnosis is byte-identical to baseline: the diagnosis head never
  sees the timing features, so the forecast gain costs nothing on the profile task.
- **The magnitude is not interpretable.** On synthetic data we control the prodromal signal, so the
  lift reflects *how cleanly we planted it*, not physiology. An earlier version aligned the blunting
  exactly to the 7-day window and hit **AUC 1.0000** — textbook label leakage; the graded, noisy,
  variable-lead prodrome here removes the degeneracy but the 7d lift is still large by construction.
  **Only a live Terra feed can give a real effect size.**

## Rigor notes

- **A/B confound caught + fixed.** Intraday sampling drew from the shared generator RNG, so turning
  it on perturbed the daily-value stream — the treatment was training on *different patients*. A
  dedicated `intraday_rng` makes the daily data byte-identical across on/off, so the only variable
  is the intraday features. (Before the fix, diagnosis appeared to drop 3.7pts; that was the
  different dataset, not the features.)

## Next move

- **Live-serving parity for intraday checkpoints** — `serving.predict` builds only the daily vector,
  so an intraday checkpoint's forecast head (wider input) can't be served yet; the predictor needs
  to derive amplitude features from the ingested Terra samples. `CheckpointModel` already carries the
  forecast head's own standardisation for this.
- **Validate on real Terra intraday data** before attaching any effect size to this.
