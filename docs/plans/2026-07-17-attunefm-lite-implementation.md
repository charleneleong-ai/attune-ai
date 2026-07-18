# AttuneFM-lite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing Attune concordance engine with an AttuneFM-lite pack for occupational health, chronic illness monitoring, medication/lifestyle response, visible health evidence, mobility signals, and clinical summaries.

**Architecture:** The existing repo already has the right foundation spine: typed longitudinal memory, robust personal baselines, cross-axis concordance, condition packs, voice-first check-ins, safety tiering, and clinician briefs. AttuneFM-lite should be added as a broad multimodal `ConditionPack` plus a lightweight monitoring-score adapter over `Engine.reflect()`.

**Tech Stack:** Python 3.13, Typer, Rich, Pydantic, pytest, existing `attune.concordance_engine`.

**Architecture Diagram:** `docs/diagrams/attunefm-lite-architecture.mmd` is the source Mermaid diagram and is mirrored in `docs/specs/2026-07-17-attunefm-lite-design.md`.

## Global Constraints

- Save design specs under `docs/specs/`; save implementation plans under `docs/plans/`.
- Work on a feature branch, not `main`.
- Keep the prototype non-diagnostic: no treatment advice, medication adjustment, or employment fitness decision.
- Include wearables, voice, text, image, and video as first-class optional modalities.
- Preserve the existing PCOS and veteran demos.
- Use TDD for non-trivial behavior.

---

### Task 1: Add AttuneFM-lite Pack

**Files:**
- Create: `src/attune/packs/attunefm.py`
- Modify: `src/attune/concordance_engine/engine.py`
- Test: `tests/test_attunefm.py`

**Interfaces:**
- Produces: `ATTUNEFM_PACK: ConditionPack`
- Updates: `PACKS["attunefm"]`

- [x] **Step 1: Write failing pack test**

Add a test asserting `PACKS["attunefm"]` covers `wearable`, `audio`, `vision`, `video`, `text`, and `self_report`, and includes occupational/chronic-health check-in prompts.

- [x] **Step 2: Verify red**

Run: `uv run --extra dev pytest tests/test_attunefm.py -q`

Observed: FAIL with `ModuleNotFoundError: No module named 'attune.attunefm'`.

- [x] **Step 3: Implement pack**

Create `src/attune/packs/attunefm.py` with:
- physiological wearable signals: HRV, resting HR, sleep, SpO2
- metabolic signals: glucose variability, diet response, food photo risk
- medication/lifestyle signal: medication tolerance
- voice signals: fatigue and breathlessness
- text/work signals: work burden and engagement
- vision signals: skin/wound visible change
- video signals: mobility and posture strain

- [x] **Step 4: Register pack**

Add `ATTUNEFM_PACK` to `PACKS` in `src/attune/concordance_engine/engine.py`.

---

### Task 2: Add Monitoring Scores

**Files:**
- Create: `src/attune/attunefm.py`
- Test: `tests/test_attunefm.py`

**Interfaces:**
- Produces: `modality_coverage(memory: Memory, *, day: int, span: int = 3) -> dict[str, int]`
- Produces: `monitoring_scores(engine: Engine, *, day: int) -> MonitoringScores`

- [x] **Step 1: Write failing score tests**

Add tests for modality coverage and score outputs: recovery capacity, fatigue risk, anomaly score, medication response, visible change, mobility change, and top drivers.

- [x] **Step 2: Implement score adapter**

Build scores from existing concordance load, latest multimodal signal values, and per-signal robust z-scores.

- [x] **Step 3: Verify focused tests**

Run: `uv run --extra dev pytest tests/test_attunefm.py -q`

Expected: `3 passed`.

---

### Task 3: Make Image And Video Visible In Demo

**Files:**
- Modify: `src/attune/demo.py`
- Modify: `tests/test_demo.py`

**Interfaces:**
- Produces: `channel_label(source: str) -> str`

- [x] **Step 1: Write failing demo-label test**

Assert `vision -> photo`, `video -> video`, and `audio -> voice`.

- [x] **Step 2: Verify red**

Run: `uv run --extra dev pytest tests/test_demo.py::test_channel_label_keeps_image_and_video_distinct -q`

Observed: FAIL with `ImportError: cannot import name 'channel_label'`.

- [x] **Step 3: Implement helper**

Add `channel_label()` and use it in `show_checkin()`.

- [x] **Step 4: Verify green**

Run: `uv run --extra dev pytest tests/test_demo.py::test_channel_label_keeps_image_and_video_distinct -q`

Expected: `1 passed`.

---

### Task 4: Update Docs And Verify

**Files:**
- Modify: `README.md`
- Modify: `docs/specs/2026-07-17-attunefm-lite-design.md`
- Modify: `docs/plans/2026-07-17-attunefm-lite-implementation.md`

**Interfaces:**
- Produces: repo-accurate hackathon narrative and run commands.

- [x] **Step 1: Update README**

Document AttuneFM-lite as the third pack and add `uv run attune-demo attunefm`.

- [x] **Step 2: Align spec with actual architecture**

Mark the existing concordance engine as the MVP foundation layer and leave a trainable PyTorch fusion model as stretch.

- [x] **Step 3: Run full test suite**

Run: `uv run --extra dev pytest`

Expected: all tests pass.

- [x] **Step 4: Run AttuneFM demo**

Run: `uv run attune-demo attunefm`

Expected: output includes wearable, voice, photo, and video check-in channels plus an occupational/chronic-health clinician brief.

- [x] **Step 5: Review diff**

Run: `git diff -- README.md docs src tests`

Expected: changes are scoped to AttuneFM-lite docs, pack registration, score adapter, demo channel labels, and tests.

---

### Task 5: Add Project Commands And Training

**Files:**
- Create: `mise.toml`
- Create: `src/attune/training.py`
- Modify: `pyproject.toml`
- Test: `tests/test_training.py`

**Interfaces:**
- Produces: `attune-train-plan`
- Produces: `attune-train`
- Produces: default W&B logging with `--no-wandb` opt-out
- Produces: `mise run init`
- Produces: `mise run test`
- Produces: `mise run demo-attunefm`
- Produces: `mise run demo-attunefm-profile <profile>`
- Produces: `mise run train-attunefm-plan <config>`
- Produces: `mise run train-attunefm <config>`

- [x] **Step 1: Add command surface**

Add `mise.toml` tasks for project init, validation, full demo, AttuneFM-lite demo, a parameterized profile demo, parameterized training-plan configs, local smoke training, and the A100 training target over BIDSleep, WESAD, CGMacros, Bridge2AI-Voice, DDI, and PAMAP2.
Support targeted AttuneFM profile demos for office work, firefighter/occupational hazard, firefighter with asthma, firefighter post-fire recovery, firefighter with dormant chronic illness, veteran hidden chronic load, autoimmune flare, and metabolic/PCOS scenarios.

- [x] **Step 2: Add training-plan CLI**

Add `attune-train-plan` as a dry-run command that validates dataset names against `attune.datasets` and prints modalities, heads, and staged training intent.

- [x] **Step 3: Add real training CLI**

Add `attune-train` as a real local trainer that fits a lightweight multiclass classifier over deterministic AttuneFM profile memories and writes a JSON checkpoint with metrics. Training configs are flat YAML files under `configs/*.yaml`, including `debug`, `smoke`, `one_year`, `a100_train`, and `a100_full`. The generated profile timelines now include explicit daily check-in exchanges, so the local one-year config schedules 81,760 voice/photo/video prompts, captures 58,330 synthetic patient responses, and marks 23,430 realistic missed/skipped turns alongside 198,560 sensor-like signal rows. `a100_train` is the default one-year A100 target, with 397,120 sensor rows and 163,520 scheduled check-in turns.

- [x] **Step 4: Add optional experiment tracking**

Add local `.env` loading, a safe `.env.example`, and default-on W&B logging for training config, metrics, per-profile confidence, compact input-example and multimodal check-in exchange tables, plots, input-feature heatmap images, checkpoint path, full source-signal/check-in JSONL artifact paths, and run URL output. Real secrets stay in `.env`, which is gitignored; `--no-wandb` opts out for local runs.

- [x] **Step 5: Verify commands**

Run:
- `mise run demo-attunefm`
- `mise run demo-attunefm-profile firefighter_recovery`
- `mise run demo-attunefm-profile veteran`
- `mise run train-attunefm-plan smoke`
- `mise run train-attunefm smoke`
- `mise run train-attunefm-plan a100_train`
- `uv run --extra dev pytest`

Expected: the demo runs, the training plan prints the selected public datasets, the real smoke trainer writes a checkpoint, and the full test suite passes.
