# Follow-up: output-path policy — `output_path_token` default + `include_job_type_in_path`

## RESOLVED 2026-06-27 — `output_path_token` default fixed to `canonical_to_snake` (FZ 31e1d3f)

A second inconsistency surfaced while auditing the 18 declared `output_path_token` values against the
handler default: **the default itself (`step_type.lower()`) was the real bug.** It never snake-cased
multi-word step types, so 13 live Processing steps + 5 Training/Transform steps that relied on the
default shipped on concatenated-lowercase S3 paths (`tabularpreprocessing/`, `xgboosttraining/`)
while the 18 declared steps used snake_case. The fix:

- Changed the default token in **all three** synthesizing handlers (Processing/Training/Transform
  `get_outputs`) from `b.spec.step_type.lower()` to `canonical_to_snake(b.spec.step_type)` — the
  package's own PascalCase→snake util (`step_catalog/naming.py`), the same one mapping step types to
  builder/contract filenames, so acronyms (LightGBMMT, XGBoost) resolve correctly. Verified exact for
  all 31 step types.
- **Dropped 12 redundant `output_path_token` declarations** from `.step.yaml` (their value just
  restated `canonical_to_snake(step_type)`). Only the **6 genuine deviations** remain as data:
  `active_sampling`, `packaging`, and the shared `model_evaluation` / `model_inference` namespaces
  (XGBoost-eval and LightGBMMT-eval deliberately co-locate). `include_job_type_in_path: false` lines
  were all preserved — that is separate deviation data.
- This MOVED the generated S3 fallback path for 13 Processing + 5 Training/Transform steps (now
  snake_case). Internally self-consistent: downstream steps reference the upstream output *property*
  (Join/PropertyReference), so wiring follows automatically. External impact = a rename of the
  generated path + warm-start cache misses on re-runs. Gated by the resolved-edge-graph snapshot +
  full `tests/core/base` suite (522 passed) + a new convention-lock test class
  (`TestOutputTokenIsCanonicalSnakeByDefault` in `test_builder_templates.py`) that pins the
  canonical_to_snake default in all 3 handlers and verifies explicit deviations still win.

Net: the interface now declares only what DEVIATES from convention (6 tokens), and the convention is
derived — the true "declare the minimum" state.

---

## `include_job_type_in_path` — design vs implementation inconsistency (2 to fix)

> Analysis prompted by "how many `_get_outputs` exceptions are design choice vs implementation
> inconsistency?" — after the `_get_outputs` integration was folded into the interface
> (`output_path_token` + `include_job_type_in_path` on `ContractSection`, commit `3a76b457`).
> Method: cross-referenced each Processing builder's `include_job_type_in_path` decision against
> (a) whether its config actually has a `job_type` field, (b) whether its `.step.yaml` declares
> job-type `variants`, (c) whether its `scripts/<name>.py` reads `--job_type`.

## Verdict: 16 DESIGN, 2 INCONSISTENCY

The `include_job_type_in_path` variation is **mostly intentional design** — it is NOT arbitrary drift:

- **job_type IN the output path (genuine, 4):** `model_calibration`, `percentile_model_calibration`,
  `temporal_feature_engineering`, `temporal_sequence_normalization`. Config has `job_type`, the
  script reads `--job_type`, and 2 declare `.step.yaml` variants → they genuinely produce
  job-type-partitioned output (`…/<token>/<job_type>/<logical>`), the multi-node-DAG need.
- **flat output, no job_type segment (12):** `package`, `model_metrics_computation`, the model
  eval/inference steps, `tokenizer_training`, `temporal_split_preprocessing`,
  `active_sample_selection`, etc. Several have a `job_type` field and pass it as a *script argument*
  but deliberately do NOT put it in the *output path* — a legitimate, consistent choice (job_type
  affects script behavior, not output layout). One output location regardless of slice.

## The 2 implementation inconsistencies (latent bugs — NOT design)

These were **faithfully preserved** by the behavior-preserving migration (so no behavior changed),
but they are now visible as `.step.yaml` data and should be corrected as a deliberate, separate fix
(changing them MOVES the output S3 path, so it is a behavior change — gate with the edge-graph
snapshot + a real-session check, do NOT fold into the migration):

1. **`currency_conversion` — `include_job_type_in_path: true` but the config has NO `job_type` field.**
   `CurrencyConversionConfig` (and its full MRO: ProcessingStepConfigBase → BasePipelineConfig) has
   no `job_type` field (`job_type resolvable: False`). The pre-migration builder referenced
   `self.config.job_type` in BOTH the old `_get_outputs` (generated-destination branch) and
   `_get_job_arguments` — i.e. a latent `AttributeError` if those branches execute. It likely
   survived because job_type was supplied at runtime or the generated-destination branch was never
   hit. **Fix: `include_job_type_in_path: false`** (and audit `_get_job_arguments`, which is kept and
   still references `self.config.job_type`).

2. **`dummy_data_loading` — `include_job_type_in_path: true` but nothing uses the partition.**
   Has a `job_type` field, but its `scripts/dummy_data_loading.py` does NOT read `--job_type`, and
   there are no variants. So it stamps a `job_type` segment into the output path that no downstream
   consumer depends on — inconsistent with the step's own behavior. **Fix: `include_job_type_in_path:
   false`** (verify no existing pipeline reads the job-typed path first).

## Why this surfaced now (a benefit of the interface enhancement)

Before the fold-in, these decisions were buried in 19 hand-written `_get_outputs` methods — invisible
and un-auditable. Moving `output_path_token` + `include_job_type_in_path` into the `.step.yaml` made
the whole output-path policy a single reviewable column across all steps, which is exactly what
exposed the 2 inconsistencies. The interface-as-data refactor pays for itself here: it turns a
per-builder imperative quirk into auditable data.

## Action

- [x] **`currency_conversion` — FIXED (2026-06-28, commit `a5d6dc24`), with a CORRECTION to the
      diagnosis.** The deeper root cause (surfaced when implementing): the script declares
      `--job_type required=True` (argparse, `currency_conversion.py:482-486`) but
      `CurrencyConversionConfig` had **no `job_type` field**, so `get_job_arguments()` emitted nothing
      → the step would fail at container argparse (not just a path issue). The post-collapse
      `_job_type_arg()` uses `getattr(self,"job_type",None)`, so the old `AttributeError` was already
      gone; the real bug was the **missing field**. Fix: added `job_type: str = Field(default="training")`
      + validator (mirrors RiskTableMapping). The output path now correctly INCLUDES the segment
      (`.../currency_conversion/training/processed_data`) — so `include_job_type_in_path` is left at the
      default `true`, NOT set false (the original action was treating the symptom). job-args + env
      conformance gates green.
- [x] **`dummy_data_loading` — FIXED (2026-06-28, commit `a5d6dc24`).** Set
      `patterns.include_job_type_in_path: false`. Confirmed: the script never reads `--job_type`, no
      variants, not in the edge-graph snapshot DAG, and the config's `job_type` field is used by
      nothing. ⚠️ This MOVES the S3 output (`.../dummy_data_loading/<job_type>/<name>` →
      `.../dummy_data_loading/<name>`) — a behavior change, flagged in the `.step.yaml` comment;
      **still needs a real-session/prod-pipeline confirmation** that no live consumer reads the
      job-typed path before being relied on in prod.
- [x] Gated by the resolved-edge-graph snapshot (green) + the job-args/env conformance gates. The
      dummy_data_loading path move's **real-session check remains the one open confirmation** (can't
      run offline).
