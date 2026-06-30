---
tags:
  - project
  - planning
  - step_builder
  - environment_variables
  - config_interface_conformance
keywords:
  - _get_environment_variables
  - get_environment_variables
  - required_env_vars
  - config collector
  - conformance gate
  - env vars
topics:
  - environment variable ownership (interface declares, config provides)
  - config↔interface conformance
language: python
date of note: 2026-06-27
---
# Env vars: interface DECLARES, config PROVIDES, the two must not CONFLICT

> Model (user directives, 2026-06-27, three refinements):
> 1. "leave the env variables and job_type to be defined in config" — config is the VALUE source.
> 2. "you can keep env variables in interface, but config env setup should not be in conflict with it"
>    — the interface keeps the DECLARATION; config must conform to it.
> 3. "not all config fields are environment variables" — the config COLLECTOR
>    (`get_environment_variables()` / `environment_variables`) is the authoritative selector; the
>    conformance check inspects collector OUTPUT, never the raw config schema.
>
> Net: this is NOT a "remove the interface env duplicate" task. The interface declaration stays; the
> work is to make config's collector OUTPUT conform to the interface's declared env vars, gated.

## The two roles (kept separate)

| Concern | Owner | Surface |
|---|---|---|
| WHICH env vars exist (the declared contract + alignment surface) | **Step interface** | `.step.yaml` `env_vars.required` / `.optional` → `contract.required_env_vars` / `optional_env_vars` (also consumed by `validation/alignment` to check script `os.environ` reads) |
| The VALUES injected into the container | **Config** | `config.get_environment_variables()` (method, 18 configs) or `config.environment_variables` (property) — the collector, which selects a SUBSET of config fields |
| Composition into the container env dict | **One template method** (target) | `builder_base._get_environment_variables` = collector output + `config.env`; the 40 per-step overrides become deletable once conformant |

`job_type` is the same shape: a config field (drives multi-variant DAG resolution) whose value the
builder passes as `--job_type`; the interface's `expected_arguments` declaration is the names side.

## Conformance evidence (gate: `tests/core/base/test_env_vars_config_interface_conformance.py`)

The gate `model_construct`s each config (validity-free; sentinels fill required fields so the
collector runs; only KEY SETS are compared, so sentinel VALUES never cause a false conflict) and
compares the collector's emitted KEYS against the interface-declared keys (`required ∪ optional`):

- **CONFORMANT — 8**: collector ⊆ declared AND all required emitted (LabelRulesetGeneration,
  LightGBMTraining, ModelCalibration, PercentileModelCalibration, PyTorchTraining, TokenizerTraining,
  XGBoostModelInference, XGBoostTraining).
- **CONFLICT-A (undeclared) — 24**: the config collector EMITS keys the interface does NOT declare
  (`COMPARISON_METRICS`, `USE_SECURE_PYPI`, `CHUNK_SIZE`, …). So the duplication intuition INVERTS —
  the interface is an UNDER-declared subset; config is the richer source. Reconcile by completing the
  interface's `optional_env_vars` (sync interface ← what config/script actually use).
- **CONFLICT-B (required-not-emitted) — 2 (HARD errors, genuine misdeclarations):**
  - `RiskTableMapping`: interface requires `LABEL_FIELD`, but the config has NO `label_field` field
    and its collector never emits `LABEL_FIELD`. Either add the field to config or drop it from the
    interface's required set (check the script's `os.environ` reads to decide which).
  - `TemporalSequenceNormalization`: interface requires `TIMESTAMP_FIELD`, but config emits
    `TEMPORAL_FIELD` and the script reads `TEMPORAL_FIELD` (`DEFAULT_TEMPORAL_FIELD`). The interface
    declares the WRONG NAME — fix the `.step.yaml` to `TEMPORAL_FIELD`.
- **NO-COLLECTOR — 7**: config exposes no `get_environment_variables()`/`environment_variables`
  (CurrencyConversion, DummyTraining, Package, Payload, StratifiedSampling, TabularPreprocessing,
  TemporalSplitPreprocessing). These build env in the BUILDER (or only from the contract) today —
  they need a config collector added before config can be the value source.

## Worklist (to reach full conformance, then collapse the overrides)

1. **Fix the 2 CONFLICT-B misdeclarations** (independent bugs, do first):
   `TemporalSequenceNormalization` `TIMESTAMP_FIELD`→`TEMPORAL_FIELD` in the `.step.yaml`;
   `RiskTableMapping` `LABEL_FIELD` (decide via the script's reads: add config field, or de-require).
2. **Add a config collector to the 7 NO-COLLECTOR steps** (move builder-side / contract-only env
   assembly into `config.get_environment_variables()`), so config becomes the value source.
3. **Reconcile the 24 CONFLICT-A** by declaring the collector's keys in the interface
   `optional_env_vars` (interface = complete declaration of what's injected).
4. **Standardize the collector name** (`get_environment_variables()` everywhere; alias the
   `environment_variables` property + the `<named>_environment_variables` variants).
5. **Collapse to one template method** in `builder_base` (collector output + `config.env`); delete the
   40 overrides. Gate each deletion with the conformance gate + the resolved-edge-graph snapshot +
   full suite (same discipline as the `output_path_token` change).

After step 3 the gate flips to enforce direction A too (`collector ⊆ declared`), making "config must
not conflict with the interface" a hard CI invariant in BOTH directions.

## STATUS — COMPLETE (2026-06-28): 39/40 overrides collapsed, 1 genuine keep

All 5 worklist steps done; commits `0668d830` (5a) / `e44beef5` (5b) / `2128e595` (5c):
- **Step 1 (CONFLICT-B fix):** done earlier — baseline now `set()`.
- **Steps 4 + 5 (resolver + template):** `BasePipelineConfig.get_environment_variables(declared_names)`
  generic resolver (convention `NAME`→`self.name` + `_format_env_value` + `_env_overrides()` +
  `hyperparameters` fallback); `builder_base._get_environment_variables` is the ONE template
  (collector-dispatch: bespoke config method → bespoke property → inherited resolver; then interface
  defaults; then `config.env`; with missing-required/default-used diagnostics).
- **Step 2 (NO-COLLECTOR → config):** added `get_environment_variables()` to Package, Edx,
  TemporalSplit, CurrencyConversion configs; `_env_overrides()` to Payload; moved `CA_REPOSITORY_ARN`
  into the lightgbm/lightgbmmt training configs.
- **Step 3 (reconcile):** CONFORMANT 8 → 21; declared `CA_REPOSITORY_ARN` in the 2 training interfaces.
- **Collapsed 39 of 40** `_get_environment_variables` overrides (each byte-verified override==base or
  a config-backed improvement). 630 passed / 15 skipped; edited files ruff-clean.

**The 1 genuine keep — `BedrockBatchProcessing`** (verified against the real script, 2026-06-28):
its override computes `BEDROCK_BATCH_INPUT_S3_PATH`/`OUTPUT_S3_PATH` as
`Join(_get_base_output_path(), "bedrock-batch", "input"|"output")`. The script REQUIRES these
(`raise RuntimeError("No input/output S3 bucket configured for batch processing")` — AWS Bedrock
async batch reads input JSONL from S3 and writes results to S3), but the path is **builder/runtime
derived** (`_get_base_output_path()` — the pipeline's execution prefix, set by the assembler), NOT a
config field. The config's `bedrock_environment_variables` even emits `BEDROCK_BATCH_INPUT_S3_PATH: ""`
as a placeholder precisely because config cannot know it. So this is a legitimate builder-side
override (same class as `output_path_token`/base-output-path concerns) and stays — config genuinely
lacks the input. Marks the env single-source task COMPLETE.

## Option 1 (chosen): interface DRIVES the keys, config RESOLVES values — no per-step methods

Directive (2026-06-27): "if option 1 is used, we do not need to manually define the method to get
environment — we can use the step-interface fields to collect them in config, and in the builder call
config's get-environment." Confirmed feasible. The conflict exists only because TWO independent NAME
lists drift (the interface's `env_vars` and the implicit key set a collector returns). Option 1 makes
the interface name list the SINGLE driver, so config can never emit an undeclared key and a required
key it can't resolve fails loudly — drift becomes structurally impossible, not merely detected.

Shape (the ONE builder method + a generic config resolver):
```python
# builder_base._get_environment_variables — ONE method; all 40 overrides deleted
def _get_environment_variables(self):
    names = self.contract.required_env_vars + list(self.contract.optional_env_vars)
    env = self.config.get_environment_variables(names)        # interface NAMES -> config VALUES
    env = {**self.contract.optional_env_vars, **env}          # declared defaults for absent optionals
    if getattr(self.config, "env", None):
        env.update(self.config.env)
    missing = [n for n in self.contract.required_env_vars if n not in env]
    if missing:
        raise ValueError(f"{self.STEP_NAME}: required env vars not produced by config: {missing}")
    return env

# config base: generic typed resolver, driven by the interface names
def get_environment_variables(self, declared_names):
    overrides = self._env_overrides()                         # {} for most; small map where computed/aliased
    env = {}
    for name in declared_names:
        if name in overrides:
            env[name] = overrides[name]
        elif hasattr(self, name.lower()) and getattr(self, name.lower()) is not None:
            env[name] = _fmt(getattr(self, name.lower()))     # bool->lower, list->",".join, dict->json
    return env

def _env_overrides(self):   # base returns {}; only the ~10 compute/alias configs override
    return {}
```

### How far the GENERIC resolver gets (prototype measurement, realistic non-default values)

`_fmt` = the type formatting the hand-written collectors already do (`bool`→`str().lower()`,
`list`→`",".join`, `dict`→`json.dumps`, else `str`). Driving the resolver with each collector's own
key names:

- **154/190 declared env names (81%) resolve by pure convention** (`LABEL_FIELD`→`config.label_field`).
- **13 collectors are FULLY reproduced** by the generic resolver alone — ZERO override map needed
  (ActiveSampleSelection, DummyDataLoading, FeatureSelection, LabelRulesetExecution/Generation,
  LightGBM/LightGBMMT/XGBoost Training, ModelCalibration, PercentileModelCalibration, PseudoLabelMerge,
  RiskTableMapping, TokenizerTraining).
- **19 collectors are PARTIAL** — they need a SMALL `_env_overrides()` map, dominated by ONE recurring
  alias: **`ID_FIELD`←`config.id_name`, `LABEL_FIELD`←`config.label_name`** (the eval/inference family:
  LightGBM/LightGBMMT/PyTorch/XGBoost ModelEval/Inference + ModelMetricsComputation — ~13 of the 19 are
  JUST this one alias pair). The rest are a handful of genuine computes: Bedrock S3-path `Join`s +
  `AWS_DEFAULT_REGION` (4 keys), `MODEL_VERSION`←`pipeline_version`, `PREFETCH_FACTOR`, list-join
  cases (`VALUE_FIELDS`/`FEATURE_TYPES`/`MISSING_INDICATORS`/`INPUT_PLACEHOLDERS`).

So the "no manual method" vision holds: **per-step `_get_environment_variables` overrides → 0**;
per-config code → `{}` for ~24 configs and a 1–4 line `_env_overrides()` for ~10. The recurring
`id_name`/`label_name` alias is worth fixing at the BASE (one alias map covers ~13 steps), shrinking
the per-config maps further. NOTE: the 19 MISSING from the conformance pass (config produces nothing
for a declared name) are partly `model_construct`-default artifacts — re-measure each against a REAL
config during implementation; the genuine ones are the CONFLICT-B bugs + a few alias-needed fields.

### Scripts are the ground truth — and the package copy LEADS the project copies

The third party in the contract is the SCRIPT (`os.environ[...]` / `os.getenv`). Cross-checking the
script reads against interface-declared and config-emitted keys resolves which declarations are real
vs stale. Scripts exist in TWO places: the cursus package `src/cursus/steps/scripts/*.py` (41), and
each project's vendored `dockers/<x>/scripts/*.py` (BAMT: `mods_pipeline_adapter/dockers/xgboost_atoz/
scripts`, 14 overlapping). Diffing env reads between them:

- **Drift is one-directional — `CUR-only`, never `BAMT-only`.** The cursus package scripts read a
  SUPERSET; BAMT's docker scripts read a subset (older vendored snapshots lagging newer features like
  `ENABLE_TRUE_STREAMING`, `MAX_WORKERS`, multi-task `SCORE_FIELDS`/`TASK_LABEL_NAMES`). 3/14 identical
  (model_wiki_generator, xgboost_model_eval/inference), the rest CUR-superset.
- **Conclusion: the cursus package `steps/scripts` is the AUTHORITATIVE, LEADING copy** for env reads.
  Designing conformance against it is a safe superset that automatically covers the older project
  copies — no project script reads a var the package script doesn't.

Three-way cross-check (script-reads ⟷ interface-declares ⟷ config-emits), package scripts as truth:

- **HARD — script READS a var that NEITHER interface declares NOR config emits (5 steps):**
  ActiveSampleSelection (`BATCH_SIZE`,`METRIC`,`UNCERTAINTY_MODE`); ModelMetricsComputation
  (`SCORE_FIELD`/`SCORE_FIELDS`/`PREVIOUS_SCORE_FIELD*`/`TASK_LABEL_NAMES` — these ARE conditionally
  emitted by the collector, so partly a model_construct-default artifact; re-check live);
  ModelWikiGenerator (`MODEL_DESCRIPTION` — optional); PseudoLabelMerge (`TRAIN_RATIO`,`TEST_VAL_RATIO`);
  TemporalSplitPreprocessing (`JOB_TYPE`,`TARGETS`,`MAIN_TASK_INDEX`,… + `SM_*` framework vars). These
  are where the script could read an UNSET env at runtime → real risk; verify each against a live
  config (some are conditionally emitted; `SM_*` are SageMaker-injected, not domain env).
- **STALE — interface DECLARES a var the script does NOT read AND config does NOT emit (6 steps):**
  ActiveSampleSelection `UNCERTAINTY_THRESHOLD`; FeatureSelection `SELECTION_METHODS`/`TOP_K_FEATURES`;
  MissingValueImputation `IMPUTATION_STRATEGY`/`CATEGORICAL_STRATEGY`; Payload
  `CONTENT_TYPES`/`DEFAULT_*`/`FIELD_DEFAULTS`; RiskTableMapping `LABEL_FIELD`/`USE_PRECOMPUTED_RISK_TABLES`;
  TemporalSequenceNormalization `MAX_SEQUENCE_LENGTH`/`NORMALIZATION_METHOD`/`TIMESTAMP_FIELD`. These
  are over-declarations to PRUNE from the interface (the script proves they're unused) — e.g.
  RiskTableMapping `LABEL_FIELD` (CONFLICT-B) is confirmed stale: the package script does NOT read it.
  NOTE Payload's are emitted by the BUILDER today (Bucket E), not the config collector — so "config
  doesn't emit" reflects the not-yet-migrated builder, not a true absence; re-check after Bucket E moves.

This makes the reconciliation OBJECTIVE: align interface `env_vars` to exactly what the package script
reads; make config emit exactly that set. The conformance gate then enforces the triangle
(script = interface = config) instead of just the config↔interface pair.

### Prototype validation plan (the safe first increment)

Prove the interface-driven base method byte-matches current output on TWO representative steps before
touching the other 38: **RiskTableMapping** (pure-convention, full resolver — but first fix its
`LABEL_FIELD` CONFLICT-B) and **ModelMetricsComputation** (the `_env_overrides` alias case:
`ID_FIELD`/`LABEL_FIELD`). Byte-diff old `_get_environment_variables()` vs the Option-1 path on a real
config; only then roll out, deleting overrides batch-by-batch behind the conformance gate.

## Gate status

`test_env_vars_config_interface_conformance.py` is committed and green: it hard-fails on any NEW
CONFLICT-B beyond the 2 recorded, and prompts a baseline update if a recorded one is fixed. As the
worklist lands, the conformant set grows and the conflict lists shrink toward zero.
