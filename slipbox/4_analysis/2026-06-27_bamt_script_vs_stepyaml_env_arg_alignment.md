---
tags:
  - analysis
  - step_builder
  - environment_variables
  - job_arguments
  - script_contract_alignment
  - BuyerAbuseModsTemplate
keywords:
  - env_vars
  - expected_arguments
  - job_type
  - script os.environ
  - .step.yaml contract
  - BAMT dockers scripts
  - interface under-declaration
topics:
  - BAMT script vs cursus .step.yaml alignment
  - env-var / job-arg / channel conformance
language: python
date of note: 2026-06-27
---
# BAMT scripts vs cursus `.step.yaml`: env-var / job-arg / channel alignment

> Review requested 2026-06-27: go over the REAL scripts under BuyerAbuseModsTemplate (each project's
> `dockers/<x>/scripts` + `*/scripts`), extract what they actually consume — env vars (`os.environ`),
> job arguments (argparse `--flags`), input/output channels/paths — and check whether the cursus
> `.step.yaml` interface fields ALIGN, especially **env vars and job arguments**.
>
> Method: extracted the union surface across ALL BAMT script copies per step (438 script files →
> 37 steps that map 1:1 to a `.step.yaml`; many steps have 10–32 project copies, e.g. 32×
> `tabular_preprocessing`), then a 37-way dynamic-workflow fan-out (one agent per step) judged each
> against its `.step.yaml`, plus an independent deterministic oracle
> (`.tmp_align_oracle.json` — script-reads minus interface-declares per step, with the builder's env
> regime). The two agree on counts for every spot-checked step; the workflow adds per-var defaults
> and the rename/default-drift findings. Verdict tally: **ALIGNED 11 · UNDER-DECLARES 13 · BOTH
> (under+over) 7 · OVER-DECLARES 3** of 37. **HIGH 1 · medium 14 · low 14 · none 5.**

## Headline: the interface systematically UNDER-DECLARES; almost no runtime crashes

The drift is overwhelmingly one-directional — `.step.yaml` declares **fewer** env vars / args than the
scripts read. Crucially, **nearly every undeclared env var is read via `os.environ.get(NAME, default)`
with a safe in-code default**, so an unset env degrades silently rather than `KeyError`. So this is
**contract/documentation drift, not breakage** — with a handful of sharp exceptions below.

Two facts bound the risk:
1. **All 37 builders are config-delegate (post Strategy+Facade migration)** — env is built by
   `config.get_environment_variables()`/`environment_variables`, not solely the contract. So an
   undeclared var the config still injects is invisible drift, NOT a runtime hole. (An earlier
   synthesis pass mis-stated "15 contract-only builders" from the pre-migration shapes; the oracle
   confirms 36 config-delegate + 1 no-override SDK shell.)
2. **`os.environ.get(..., default)` everywhere** — unset → default, not crash.

## The 1 genuinely HIGH-severity finding

**`bedrock_processing` — required `--job_type` absent from the interface.** The script argparses
`--job_type` with `required=True` and branches on it for ALL core control flow (training → per-split
output naming; non-training → `{job_type}/` subdir convention; summary filename). The `.step.yaml`
`contract.arguments` declares only `batch-size`/`max-retries`, no `job_type`, no variants. A builder
constructing the command purely from the interface omits the required arg → the script exits non-zero
at argparse. This is the only step where the interface fails to supply a value the script HARD-requires.

## The most pervasive defect: `--job_type` declared NOWHERE (~23/37 steps)

Scripts argparse `--job_type` but `contract.arguments`/`expected_arguments` is `{}` for essentially
every processing step. It works today only because **every builder hardcodes `["--job_type",
job_type]` in `_get_job_arguments()`** (e.g. `builder_tabular_preprocessing_step.py:103`) — so the
CONTRACT is not the source of truth for this argument; the builder is. The `required=True` subset
(higher risk if a builder ever stops hardcoding it): bedrock_batch_processing, bedrock_processing,
lightgbm_model_eval/inference, lightgbmmt_model_eval, missing_value_imputation,
model_metrics_computation, pytorch_model_eval/inference, risk_table_mapping, temporal_feature_engineering,
temporal_sequence_normalization, temporal_split_preprocessing, tokenizer_training, xgboost_model_inference.

This is the env analog of the `output_path_token`/`source_dir` story: a per-step fact that lives in
imperative builder code instead of being declared interface data.

## Stale declarations — RENAME, do not just prune (a configured value is being IGNORED today)

These are the sharpest correctness issues after the HIGH one: the interface declares an env var the
script never reads because the **script reads a differently-named var for the same concept** — so any
operator who sets the declared name has their value silently dropped:

| Step | Interface declares (ignored) | Script actually reads | Action |
|---|---|---|---|
| feature_selection | `SELECTION_METHODS` | `FEATURE_SELECTION_METHODS` | rename |
| feature_selection | `TOP_K_FEATURES` | `N_FEATURES_TO_SELECT` | rename |
| missing_value_imputation | `IMPUTATION_STRATEGY` | `DEFAULT_NUMERICAL_STRATEGY` | rename |
| missing_value_imputation | `CATEGORICAL_STRATEGY` | `DEFAULT_CATEGORICAL_STRATEGY` | rename |
| temporal_sequence_normalization | `TIMESTAMP_FIELD` (**required**) | `TEMPORAL_FIELD` | rename + reconsider required |
| temporal_sequence_normalization | `MAX_SEQUENCE_LENGTH` | `SEQUENCE_LENGTH` | rename |
| active_sample_selection | `UNCERTAINTY_THRESHOLD` | `UNCERTAINTY_MODE` | rename |

## Truly stale (no script reads it) — prune candidates

`temporal_sequence_normalization` `NORMALIZATION_METHOD`; `lightgbm_training` `USE_NATIVE_CATEGORICAL`,
`REGION`; `pytorch_training` `USE_PRECOMPUTED_FEATURES`; **`risk_table_mapping` `LABEL_FIELD`
(required!) + `USE_PRECOMPUTED_RISK_TABLES`** (label comes from `hyperparams`, not env — a builder
enforcing `required` would demand an ignored var); **`pytorch_model_inference` `LABEL_FIELD`
(required!)** + `OUTPUT_FORMAT` + `JSON_ORIENT` (inference reads no label); `redshift_data_loading`
`AWS_DEFAULT_REGION`/`AWS_STS_REGIONAL_ENDPOINTS` (cosmetic — never read from env).

## Default-value DRIFT (declared default ≠ script default — would change behavior if interface drives)

- **`pytorch_training` `PREFETCH_FACTOR`: YAML `'None'` vs script `'2'`** — the literal string `'None'`
  would raise `ValueError` on the script's `int()` cast. Most dangerous drift found.
- `pytorch_training` `NUM_WORKERS_PER_RANK` `'0'`→`'4'`; `USE_PERSISTENT_WORKERS` `'false'`→`'true'`.
- **`tokenizer_training` `USE_SECURE_PYPI`: YAML `'true'` vs script `'false'`** — flips secure-vs-public
  PyPI behavior.

## Under-declared knob bundles (safe-defaulted; interface hides the tunable surface)

Whole families read-but-undeclared, all with in-code defaults (medium drift, no crash): bedrock_processing
(16 `BEDROCK_*`), bedrock_batch_processing (21), missing_value_imputation (13 `DEFAULT_*`/`*_CONSTANT_VALUE`),
model_metrics_computation (14: `SCORE_FIELD(S)`, recall/dollar/count, `COMPARISON_*`),
temporal_feature_engineering (10), temporal_sequence_normalization (12), model_wiki_generator (8:
`TEAM_ALIAS`/`CONTACT_EMAIL`/`CTI_CLASSIFICATION`/…), pseudo_label_merge (7), active_sample_selection (9),
dummy_data_loading (6).

## Cross-cutting patterns

- **`COMPARISON_METRICS` (`'all'`) + `COMPARISON_PLOTS` (`'true'`)** are under-declared in lockstep
  across the eval family (lightgbm/lightgbmmt/pytorch/xgboost `_model_eval` + model_metrics_computation)
  while sibling `COMPARISON_MODE`/`STATISTICAL_TESTS` ARE declared — fix once as a family.
- **`id_name`/`label_name` → `ID_FIELD`/`LABEL_FIELD` alias**: the config layer renames Pydantic fields
  `id_name`/`label_name` to the env vars `ID_FIELD`/`LABEL_FIELD` (e.g.
  `config_lightgbm_model_eval_step.py:204-205`), so those are config-provided even where the contract
  omits them. But several declare them **`required`** while the script reads them with a default — an
  un-enforced "required" (cosmetic overstatement).
- **`USE_SECURE_PYPI`** (pip/CodeArtifact toggle) is declared in some steps, not others — pick ONE
  policy (declare-everywhere or treat as framework var and exclude-everywhere).
- **`AWS_*`/`SM_*` framework vars handled correctly**: `AWS_STS_REGIONAL_ENDPOINTS` is WRITTEN by
  scripts (not read), `SM_CHANNEL_*`/`SM_MODEL_DIR`/`SM_CHECKPOINT_DIR` are SageMaker-injected — all
  correctly excluded from gap lists (only redshift's are cosmetically stale).
- **Channels/paths overwhelmingly ALIGNED.** Processing steps correctly have no `SM_CHANNEL_*`;
  training steps mount `/opt/ml/input/data` with `[train,val,test]` read by directory. **One genuine
  channel mismatch:** `lightgbmmt_model_eval` writes `/opt/ml/processing/output/plots`
  (`OUTPUT_PLOTS_DIR`, gated on `GENERATE_PLOTS`) but `contract.outputs` declares no `plots_output`.

## Recommendations (ordered)

1. **(HIGH) `bedrock_processing`:** declare `--job_type` as a required argument (or add `job_type`
   variants) — the only argparse-time failure.
2. **Project rule:** every script that argparses `--job_type` MUST represent it in the contract
   (`expected_arguments` or a `job_type` variants block). Apply across the ~23 steps, required-subset first.
3. **Fix the 7 RENAME stale declarations BEFORE any pruning** (table above) — these silently drop a
   configured value today.
4. Add `COMPARISON_METRICS`/`COMPARISON_PLOTS` to all eval steps as one coordinated change.
5. Back-fill the under-declared knob bundles into `contract.env_vars.optional` with their in-code defaults.
6. Prune the truly-unread stale vars — prioritize the misleading **`required`** ones (risk_table_mapping
   `LABEL_FIELD`, pytorch_model_inference `LABEL_FIELD`).
7. Add `lightgbmmt_model_eval` `plots_output` → `/opt/ml/processing/output/plots`.
8. Reconcile default-value drift (pytorch_training `PREFETCH_FACTOR`/`NUM_WORKERS_PER_RANK`/
   `USE_PERSISTENT_WORKERS`; tokenizer_training `USE_SECURE_PYPI`).
9. Decide ONE `USE_SECURE_PYPI` policy.
10. Re-classify un-enforced `required` env vars to optional (or add real enforcement) so the flag is honest.

## Relation to the env-vars single-source work

This directly feeds [../2_project_planning/2026-06-27_env_vars_config_single_source.md](../2_project_planning/2026-06-27_env_vars_config_single_source.md):
the conformance gate there checks config↔interface; THIS adds the **third party (the script)**, making
it the triangle `script reads = interface declares = config emits`. The rename + default-drift findings
are exactly what a blind "config single source" migration would have silently broken — they must be
fixed first. The package `steps/scripts` copies lead the BAMT vendored copies (env-read drift is
`CUR-only`, never `BAMT-only`), so aligning to the package scripts safely covers all projects.

## Provenance

37-way dynamic workflow (run `wf_90f98a56-e7b`, 38 agents, ~1.03M tokens) cross-validated against an
independent deterministic oracle. Per-step verdicts: `.tmp_align_verdicts.json`; oracle:
`.tmp_align_oracle.json`; script surface: `.tmp_bamt_script_surface.json` (union across all BAMT copies).
A first run mis-wired the step list (empty fan-out) and was re-run with the list embedded as a literal.
