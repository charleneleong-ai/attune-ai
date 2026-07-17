# Attune

**A condition-agnostic concordance engine for at-home chronic condition management.**

Built for the eMed × OpenAI hackathon (London, 17–18 Jul 2026).

Attune is one engine that fuses multimodal signals (voice, text, image, wearable) into a
per-patient longitudinal memory, finds **concordance** — when signals move together — and uses
it to warn early and coach. A thin `ConditionPack` config turns the same engine into a
different product per condition.

---

## The pitch

We don't ship a PCOS app or a veteran app. We ship the **engine**, and each condition is a
config pack loaded on top. That's the innovation — and it speaks to what eMed literally is: a
*consumer healthcare platform provider*. We built the platform layer.

- **Deep demo — PCOS** (metabolic, eMed's home turf): cycle × high-GI meals × mood flares,
  vision-scored acne/hirsutism, a Rotterdam-mapped clinician brief.
- **Generalization proof — veterans/firefighters** (mind–body comorbidity): the *same*
  concordance engine, with mood promoted to a first-class **psychological** axis coupled to a
  **physiological** axis (HRV, sleep) — early-warning of a bad week ~2 days out, plus a
  higher-acuity crisis-escalation path.

The last thing the judges see is the mic-drop: *the same code, plus one axis, becomes either.*

## Why it's novel (couldn't exist 2 years ago)

Cheap voice-affect (Realtime) + zero-shot vision extraction (GPT-4o) + an LLM reasoner over a
typed longitudinal memory. No custom training — frozen foundation models + a real memory layer
+ a seeded synthetic patient. The moat is the **longitudinal multimodal memory**, not the voice.

## Architecture

```
capture/            audio · text · image  ──►  typed Signal
concordance_engine/
  memory.py         typed longitudinal Signal store + window/baseline queries
  concordance.py    personal baselines (robust z) + cross-axis concordance
  safety.py         tiered escalation with a DETERMINISTIC crisis floor
  brief.py          maps memory → BriefTemplate criteria (Rotterdam / cardiometabolic)
  engine.py         ties memory + pack together; ingest / reflect / assess / brief
packs/
  base.py           the ConditionPack abstraction (the swappable config surface)
  pcos.py           metabolic-first pack (deep demo)
  veteran.py        mind–body pack (generalization proof)
reporting.py        presentation layer — render a Brief to clinician markdown
synth.py            seeded synthetic patients with a planted concordant flare
```

**One engine, swappable packs.** Everything condition-specific lives in a `ConditionPack`:
signal→axis map, coupling hypotheses (seed the reflection pass), the clinical brief template,
the agent persona/register, and the escalation contract. Everything else is shared code.

### Concordance = specificity

Fire the early-warning when **≥2 axes deteriorate together** relative to the patient's *own*
baseline (robust median/MAD z-scores) — not when any single noisy channel blips. That's the
false-alarm guard and the reason the signal is actionable. `test_single_axis_blip_does_not_fire`
locks this behaviour.

### Safety (what wins the clinical judges)

Safety-critical decisions never rest on the LLM alone. Red tier fires on the **union** of a
deterministic keyword floor and the classifier — an LLM miss can't silently drop a crisis
(fail-safe, not fail-open). The agent never autonomously *treats* acute risk; it de-escalates,
surfaces resources, and does a **warm human handoff**.

| Tier | Trigger | Behaviour |
|---|---|---|
| Green | within baseline | coach autonomously |
| Amber | concordant multi-axis drift | more support + async clinician brief |
| Red | crisis language / affect | stop managing → de-escalate → warm handoff |

## Quickstart

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest            # concordance + safety + pack + brief + demo logic
uv run attune-seed       # write seeded patients to data/
uv run attune-demo       # narrate both packs end-to-end (or: attune-demo veteran)
```

```python
from attune import load, Signal, Axis

eng = load("veteran")                       # or "pcos"
eng.ingest(Signal("hrv", Axis.PHYSIOLOGICAL, 42.0, day=30, source="wearable"))
finding = eng.reflect(day=30)               # concordant? which axes?
verdict = eng.assess("rough night, no sleep", day=30)   # Green / Amber / Red
```

## Build plan (hackathon)

1. ✅ **Engine + packs** — memory, concordance, safety, PCOS + veteran packs.
2. ✅ **Seeded patients** — deterministic synthetic histories with a *planted* concordant flare
   the reflection pass discovers live; `attune-seed` writes them to `data/`.
3. **Capture** — wire `capture/` to Realtime (voice) + GPT-4o vision (skin/meal photos).
4. ✅ **Brief generator** — maps memory onto the pack's `BriefTemplate` (Rotterdam / cardiometabolic).
5. ✅ **Demo surface (offline)** — `attune-demo`: timeline → live concordant warning → clinician
   brief → AMBER/RED escalation → PCOS↔veteran hot-swap. Capture (step 3) swaps in live signals.

## Feasibility / credits

Everything is a frozen API call — OpenAI Realtime (voice), GPT-4o vision, embeddings
(memory retrieval), Runware (motivational visuals). No training. 2500 OpenAI credits cover a
demo many times over; Realtime is the only pricey call.
