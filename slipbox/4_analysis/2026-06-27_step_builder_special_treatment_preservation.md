---
tags:
  - analysis
  - cursus_core
  - step_builder
  - simplification
  - migration_safety
  - special_treatment
keywords:
  - per-builder special treatment
  - config fields to env vars
  - input output path construction
  - create_step assembly
  - source_dir path resolution
  - job arguments
  - migration risk register
  - MUST PRESERVE
topics:
  - step builder simplification
  - migration safety record
  - container contract preservation
language: python
date of note: 2026-06-27
status: active
---

# Per-Builder Special Treatment to Preserve Through the Step-Builder Simplification

> **Read this before you touch a single builder.** This note is the gating contract for the shell-conversion. Every behavior catalogued here is currently *load-bearing in production*. The simplification's goal is to collapse 44 builders into a thin shell + declarative interface; the failure mode is silently dropping a per-step quirk that a script downstream depends on. This document enumerates each quirk, where it lives (`file:line`), and the mechanism that must absorb it so the collapse is behavior-preserving rather than behavior-eroding.

## 1. Executive summary

### Why this note exists
The current builders work. They have a pre-destruction safety record: every one of these 44 step builders is wired into a shipping pipeline, and the per-step deviations below are not accidental noise — they encode real container contracts (env var names a script reads, S3 path tokens a downstream step globs, MIMS input ordering the SDK validates, framework containers a library is pip-installed into). The simplification *deletes the code that produces these behaviors* and re-derives them from a declarative interface plus a small number of shared handler methods. If the declarative layer cannot express a quirk, that quirk vanishes at runtime with no compile error and no test failure until a pipeline breaks in beta (or worse, prod). This note is the checklist that prevents that.

### The 5 construction verbs
Every builder, regardless of family, is some composition of these five verbs. The shell must reproduce each verb's per-step shape:

1. **Config fields → container env vars** — `_get_environment_variables()`. Base is contract-driven via `super()`; per-step merges/coercions layer on top.
2. **Input path construction** — `_get_inputs()`. The `spec.dependencies × contract.expected_input_paths` join, plus per-step injections, allowlists, ordering, and delegation.
3. **Output path construction** — `_get_outputs()`. Explicit-vs-generated `Join`, the path token, and whether `job_type` is a path segment.
4. **`create_step` assembly** — the per-verb SDK shape (ProcessingStep `code=`, ProcessingStep `run()->step_args`, TrainingStep `estimator=`, CreateModelStep `model=`, TransformStep `transformer=`, SDK-delegation steps), plus `setattr(_spec/_contract)`, caching divergence, error envelopes.
5. **source_dir / path resolution** — `get_script_path()` vs `effective_source_dir` vs `processing_source_dir`; the `code=` full-path vs `source_dir+entry_point` split; hybrid Lambda-safe resolution.

(A sixth cross-cutting concern, **job arguments** — `_get_job_arguments()` — is folded into verb 4 at runtime but documented separately in §2f because its hyphen/underscore and latent-hook variations are subtle.)

### What's at risk
The highest-consequence losses, called out in full in §4: **KMS + network config** (EdxUploading), **ordered MIMS ProcessingInput list** (Registration), **hardcoded `us-east-1` region** in three Model/Training builders (a *bug* that is nonetheless load-bearing because the environment "limitation" the comments cite is real), **3-way `model_data` resolution** (DummyTraining, XGBoostModel, PyTorchModel), **FullyReplicated broadcast** (RiskTableMapping), **computed S3-path env vars** (BedrockBatchProcessing), the **`calibration_config`/`prompt_configs`/`ruleset_configs` ship-in-source-dir input skips**, and a set of **latent/dead hooks** (`_get_job_arguments` on all four Training builders, `_get_metric_definitions`/`_create_profiler_config` on PyTorchTraining) that look deletable but are documented contracts someone may be about to wire up.

---

## 2. The cross-cutting mechanisms

### 2a. Config fields → container env vars

**How it works.** The base `StepBuilderBase._get_environment_variables()` (`builder_base.py:543-606`) is **contract-driven**: it iterates `self.contract.required_env_vars` and `self.contract.optional_env_vars` and pulls the corresponding config values. This base is the canonical "declarative" path — most env vars are already expressible as contract metadata. Per-builder overrides then do one of four things on top of the base. *(Verifier confirmed all four of TabularPreprocessing / StratifiedSampling / Payload / ModelWikiGenerator call `super()` first — none build env purely inline; the "inline-only" builder in this whole set is EdxUploading.)*

The four override shapes:

- **Pure delegation to base** — override is a no-op that just returns `super()`. TemporalSequenceNormalization (`builder_temporal_sequence_normalization_step.py:150-167`), TemporalFeatureEngineering (`...temporal_feature_engineering_step.py:163-179`), DummyTraining (`...dummy_training_step.py:139-152`). The override exists only to *document* which contract-declared env vars the script consumes. **On collapse these can disappear entirely — but the documentation value (the comment naming SEQUENCE_LENGTH/TEMPORAL_FIELD/etc.) must move into the YAML.**
- **`super()` + config-dict merge** — `super()` then `env_vars.update(self.config.<named_dict>)`. The *named attribute is load-bearing*: TemporalSplitPreprocessing uses `temporal_split_environment_variables` (`...temporal_split_preprocessing_step.py:151-152`); LabelRulesetExecution uses `execution_environment_variables` (`...label_ruleset_execution_step.py:151`, **not** `environment_variables` like its generation siblings); BedrockProcessing uses `bedrock_environment_variables` (`...bedrock_processing_step.py:298-299`); RiskTableMapping merges `config.environment_variables` **and then a second `config.env` overlay** (`...risk_table_mapping_step.py:165-170`, the only Processing-2B builder with the double merge); ActiveSampleSelection / PseudoLabelMerge call `config.get_environment_variables()` as a **method, not a property** (`...active_sample_selection_step.py:167-168`, `...pseudo_label_merge_step.py:191-192`).
- **`super()` + inline coercion** — the dangerous ones, because coercion logic lives *in the builder body* and is invisible to a declarative layer:
  - **UPPERCASE + list-join:** TabularPreprocessing loops `['categorical_columns','numerical_columns','text_columns','date_columns']`, uppercases each field name as the key and `','.join(...)`s the list value (`...tabular_preprocessing_step.py:144-153`). Payload sets `CONTENT_TYPES = ','.join(config.source_model_inference_content_types)` (`...payload_step.py:151-154`).
  - **`json.dumps`:** Payload `FIELD_DEFAULTS = json.dumps(config.field_defaults)` (guarded by truthiness, `json` imported inline, `...payload_step.py:165-168`). EdxUploading `EDX_MANIFEST_KEY_PARTS` and `EDX_OUTPUT_COLUMNS` both `json.dumps(...)` only if set (`...edx_uploading_step.py:90-99`).
  - **`str()` / bool-`.lower()` coercion:** StratifiedSampling — `TARGET_SAMPLE_SIZE`/`MIN_SAMPLES_PER_STRATUM`/`RANDOM_STATE`/`SAMPLING_MULTIPLIER` all `str(...)`, `ALLOW_REPLACEMENT = str(bool).lower()` (`...stratified_sampling_step.py:179-189`). ModelWikiGenerator — 13-key bulk dict with `INCLUDE_TECHNICAL_DETAILS = str(...).lower()` (`...model_wiki_generator_step.py:211-213`) and a **semantic rename** `MODEL_VERSION = config.pipeline_version` (`:203`). FeatureSelection — the **only** builder injecting `USE_SECURE_PYPI = str(config.use_secure_pypi).lower()` (`...feature_selection_step.py:200-201`). Package — `PIPELINE_NAME`/`REGION` plus a `[(model_type,MODEL_TYPE),(bucket,BUCKET_NAME),...]` tuple-loop with `str(getattr)` (`...package_step.py:142-156`).
  - **Computed PipelineVariable values:** BedrockBatchProcessing builds `BEDROCK_BATCH_INPUT_S3_PATH` / `BEDROCK_BATCH_OUTPUT_S3_PATH` as **`Join` objects** (S3 path tokens `'bedrock-batch'/'input'/'output'`, hyphenated), `.copy()`ing the config dict first to avoid mutation (`...bedrock_batch_processing_step.py:334-353`). **A generic handler that string-coerces every env value will destroy these — env *values* here are SageMaker pipeline variables, not strings.**
- **Config method/property as sole source (does NOT call `super()`):** TokenizerTraining uses `config.environment_variables` as the env *when present*, falling back to `super()` only otherwise (`...tokenizer_training_step.py:147-162`) — so base/contract vars appear **only in the fallback branch**. PercentileModelCalibration returns `config.get_environment_variables()` directly with **no `super()` and no fallback** (`...percentile_model_calibration_step.py:195-207`). ModelCalibration / ModelMetricsComputation / XGBoostModelEval / the 2B eval-inference family delegate to `config.get_environment_variables()` guarded by `hasattr`, `super()` only on the else branch. **Collapsing these to a uniform "`super()` then `.update(config.env)`" handler changes behavior:** for TokenizerTraining and PercentileModelCalibration the base vars would suddenly start appearing where today they are suppressed.

The Training builders (XGBoost/PyTorch/LightGBM/LightGBMMT) additionally merge `config.env` verbatim after the delegate (`...xgboost_training_step.py:163-164` and siblings). PyTorch/LightGBM/LightGBMMT inject `CA_REPOSITORY_ARN` when `config.use_secure_pypi` (`...pytorch_training_step.py:172-176`, `...lightgbm_training_step.py:185-189`, `...lightgbmmt_training_step.py:194-198`) — **XGBoostTraining does NOT** (it omits secure-PyPI handling entirely). The Model builders (XGBoostModel/PyTorchModel) merge `config.env` verbatim after `super()` (`...xgboost_model_step.py:161-162`, `...pytorch_model_step.py:163-164`). BatchTransform / the four SDK-delegation builders (Cradle/Redshift/DataUploading/Registration) set **no env at all** — env is owned upstream or by the SDK step.

**Deviation table — env var construction:**

| Builder | super()? | merge source | inline coercion |
|---|---|---|---|
| TabularPreprocessing | yes | `preprocessing_environment_variables` | UPPERCASE key + `','.join` over 4 column lists |
| TemporalSplitPreprocessing | yes | `temporal_split_environment_variables` | none |
| TemporalSequenceNormalization | yes (no-op) | none | none (doc-only override) |
| TemporalFeatureEngineering | yes (no-op) | none | none (doc-only override) |
| StratifiedSampling | yes | inline fields | `str()`, `str(bool).lower()`, conditionals |
| MissingValueImputation | yes | `config.environment_variables` **+** `config.env` | none (config property) |
| FeatureSelection | yes | `config.environment_variables` + `config.env` | `USE_SECURE_PYPI=str().lower()` (unique) |
| CurrencyConversion | yes | `config.environment_variables` + `config.env` | none |
| DummyDataLoading | yes | `config.get_environment_variables()` (method) | none; filtered debug log |
| BedrockPromptTemplateGeneration / LabelRulesetGeneration | yes | `config.environment_variables` | none |
| LabelRulesetExecution | yes | `execution_environment_variables` (distinct name) | none |
| ModelCalibration / ModelMetricsComputation | delegate (else super) | `config.get_environment_variables()` | none |
| Package | yes | inline | `str(getattr)`, hardcoded UPPERCASE keys |
| Payload | yes | inline | `','.join`, `str()`, `json.dumps` (inline import) |
| ModelWikiGenerator | yes | inline 13-key dict | `str().lower()`, `MODEL_VERSION`=`pipeline_version` rename |
| RiskTableMapping | yes | `config.environment_variables` **+** `config.env` (double) | none |
| TokenizerTraining | **only on fallback** | `config.environment_variables` replaces base | none |
| BedrockProcessing | yes | `bedrock_environment_variables` | none |
| BedrockBatchProcessing | yes | `bedrock_environment_variables.copy()` | **`Join` PipelineVariable values** (computed S3 paths) |
| ActiveSampleSelection / PseudoLabelMerge | yes | `config.get_environment_variables()` (method) | none |
| PercentileModelCalibration | **no super, no fallback** | `config.get_environment_variables()` sole source | none |
| XGBoost/PyTorch/LightGBM/LightGBMMT Training | delegate (else super) + `config.env` | config method | `CA_REPOSITORY_ARN` (all but XGBoost) |
| XGBoostModel / PyTorchModel | yes | `config.env` verbatim | none |
| BatchTransform / Cradle / Redshift / DataUploading / Registration | none | — | — |
| EdxUploading | **no super — fully inline** | inline | `json.dumps` ×2, region coalesce default |

**The 2B eval-inference family** (XGBoostModelEval/Inference, LightGBMModelEval/Inference, LightGBMMTModelEval/Inference, PyTorchModelEval/Inference) all delegate to `config.get_environment_variables()` with `super()` fallback; the LightGBM and LightGBMMT variants additionally `log_info` the env (`...lightgbm_model_eval_step.py:163` etc.) — a side-effect absent from the XGBoost and PyTorch twins. PyTorchModelEval/Inference docstrings *claim* they "add builder-specific environment variables" but the code adds none (`...pytorch_model_eval_step.py:151-153`) — a latent/aspirational comment.

---

### 2b. Input path construction

**How it works.** The generic path is a join: iterate `self.spec.dependencies.items()`; skip optional-and-absent; raise `ValueError` for required-and-absent; pull the container path from `self.contract.expected_input_paths[logical_name]` (raise if missing); build `ProcessingInput(input_name=logical_name, source=inputs[logical_name], destination=container_path)`. This is the canonical declarative join — fully derivable from `spec × contract`. Every quirk below is a *departure* from it.

**FullyReplicated (S3 distribution type).** **RiskTableMapping is the only builder** that sets `s3_data_distribution_type='FullyReplicated'` on every `ProcessingInput` (`...risk_table_mapping_step.py:221`). This is risk-table broadcast semantics — every instance gets the full table. **A generic join handler drops this silently and the step will mis-shard.**

**`direct_input_keys` allowlists (create_step kwarg injection).** Several builders inject direct kwargs into the `inputs` dict *before* the join runs, as a backward-compat allowlist of recognized logical names:
- TabularPreprocessing: `['DATA','METADATA','SIGNATURE']` (`:335`)
- TemporalSplitPreprocessing / TemporalSequenceNormalization: `['DATA','SIGNATURE']` — **no `METADATA`** (`:333`, `:347`)
- TemporalFeatureEngineering: `['normalized_sequences']` (`:359`) — ties it to the upstream normalization step
- StratifiedSampling: `['processed_data','DATA','METADATA','SIGNATURE']` (`:379`)
- MissingValueImputation: `['data_input','imputation_params_input','processed_data']` (`:382`)
- FeatureSelection: `['processed_data','selected_features']` (`:393`)
- CurrencyConversion: `['data_input']` (`:325`)
- RiskTableMapping: `['data_input','config_input','risk_tables']` (`:359-361`)

**Per-channel ship-in-source-dir skips.** Three builders deliberately **do not wire** a named channel as a `ProcessingInput` because the file ships inside the source dir instead:
- BedrockPromptTemplateGeneration: `prompt_configs` sourced from `config.resolved_prompt_configs_path`, wrapped in try/except re-raising "Ensure effective_source_dir is configured properly", then `continue` (`...bedrock_prompt_template_generation_step.py:216-240`).
- LabelRulesetGeneration: same pattern for `ruleset_configs` via `config.resolved_ruleset_configs_path` (`...label_ruleset_generation_step.py:208-232`).
- PercentileModelCalibration: `if logical_name == 'calibration_config'` → skip with log "calibration config loaded from script folder" (`...percentile_model_calibration_step.py:237-242`).

**Always-inject overrides.** Package **always** injects `inference_scripts_input` from `config.resolved_source_dir or config.source_dir or 'inference'`, *overriding any dependency-provided value*, then deletes it from the working set so the loop won't reprocess it; container path falls back to literal `/opt/ml/processing/input/script` if the contract lacks it (`...package_step.py:190-246`). Payload injects `custom_payload_input = config.custom_payload_path` (S3 or local) before the loop if set (`...payload_step.py:194-202`). DummyDataLoading **ignores the `inputs` arg entirely** and hardcodes a single `INPUT_DATA` channel from `config.get_data_source_uri()` and `contract.expected_input_paths['INPUT_DATA']` (`...dummy_data_loading_step.py:147-185`).

**Circular-reference detection.** ModelCalibration, ModelMetricsComputation, and ModelWikiGenerator each run `_detect_circular_references` over every input *before* the join, raising `ValueError` on a PipelineVariable cycle (recursive, id()-keyed visited set, skipping `'Get'` keys) (`...model_calibration_step.py:171-207`/`247-249`, and twins). These helper methods would be deleted on a naive collapse.

**Registration's ordered list.** Registration returns an **ordered `List[ProcessingInput]`** with MIMS-mandated positions: `PackagedModel` **required and first** (raise if missing, `...registration_step.py:158-184`), `GeneratedPayloadSamples` optional and second; both `s3_data_distribution_type='FullyReplicated'`, `s3_input_mode='File'`. The MIMS SDK validates exactly 1-or-2 inputs in this order. **Ordering is a hard contract.**

**DummyTraining's 3-tier model resolution.** Bespoke, not a join: model from `config.pretrained_model_path` (Tier1) → `inputs['model_artifacts_input']` (Tier2) → SOURCE-mode log-only no-input (Tier3); hyperparameters channel similar; **can return an empty list** in SOURCE mode (`...dummy_training_step.py:154-237`). Reads contract dests via `.get()` + truthiness, silently producing no input if a path is missing (does **not** raise like generic builders).

**BatchTransform's hardcoded dispatch.** Iterates `spec.dependencies` but dispatches by name: `model_name` → captures model, `processed_data` → captures input_data, else warns; returns a **`Tuple[TransformInput, model_name]`** not a dict (`...batch_transform_step.py:123-210`). EdxUploading / DataUploading resolve a single `input_data` logical name to `self._resolved_input_s3` (a Properties pipeline variable) with a config fallback, raising if neither (`...edx_uploading_step.py:103-127`, `...data_uploading_step.py:80-114`).

**Delegation's `[]`.** Cradle (`:200-214`), Redshift (`:85-87`), DataUploading (returns `[]` with a resolution side-effect, `:80-114`) return empty input lists — SOURCE/SDK-owned nodes. Registration returns the ordered list above; the Model builders return single-key `{'model_data': ...}` passthrough dicts.

---

### 2c. Output path construction

**How it works.** Iterate `self.spec.outputs.items()`; container path from `contract.expected_output_paths[logical_name]` (raise if missing); if the logical name is in the passed `outputs` dict use that explicit destination, **else generate** `Join(on='/', values=[base_output_path, <PATH_TOKEN>, (job_type?), logical_name])`. The two variables the shell must carry per step are **the path token** and **whether `job_type` is a path segment**.

**Path tokens and job_type-in-path (the critical per-step table):**

| Builder | path token | job_type in path? |
|---|---|---|
| TabularPreprocessing | `tabular_preprocessing` | **yes** |
| TemporalSplitPreprocessing | `temporal_split_preprocessing` | **NO** (the only one of its group to omit it) |
| TemporalSequenceNormalization | `temporal_sequence_normalization` | yes |
| TemporalFeatureEngineering | `temporal_feature_engineering` | yes |
| StratifiedSampling | `stratified_sampling` | yes |
| MissingValueImputation | `missing_value_imputation` | yes |
| FeatureSelection | `feature_selection` | yes |
| CurrencyConversion | `currency_conversion` | yes |
| DummyDataLoading | `dummy_data_loading` | yes (uses `config.job_type` even though validate doesn't require it) |
| BedrockPromptTemplateGeneration | `bedrock_prompt_template_generation` | NO |
| LabelRulesetGeneration | `label_ruleset_generation` | NO |
| LabelRulesetExecution | `label_ruleset_execution` | **yes** |
| ModelCalibration | `model_calibration` | yes |
| Package | `packaging` | NO |
| Payload | `payload` | NO |
| ModelMetricsComputation | `model_metrics_computation` | NO (even though job_type used in args) |
| ModelWikiGenerator | `model_wiki_generator` | NO |
| XGBoostModelEval | `model_evaluation` | NO |
| XGBoostModelInference | `model_inference` | NO |
| LightGBMModelEval | `model_evaluation` | NO |
| LightGBMModelInference | `model_inference` | NO |
| LightGBMMTModelEval | `model_evaluation` (shared, not MT-specific) | NO |
| LightGBMMTModelInference | `model_inference` | NO |
| PyTorchModelEval | `pytorch_model_evaluation` (unique prefix) | NO |
| PyTorchModelInference | `pytorch_model_inference` (unique prefix) | NO |
| RiskTableMapping | `risk_table_mapping` | **yes** |
| TokenizerTraining | `tokenizer_training` | NO |
| BedrockProcessing | `bedrock_processing` | NO |
| BedrockBatchProcessing | `bedrock_batch_processing` (underscores; env-var path uses `bedrock-batch` hyphens) | NO |
| ActiveSampleSelection | `active_sampling` (shorter than class name) | NO |
| PseudoLabelMerge | `pseudo_label_merge` | NO |
| PercentileModelCalibration | `percentile_model_calibration` | **yes** |
| XGBoostTraining | `xgboost_training` | NO (returns **string/Join**) |
| PyTorchTraining | `pytorch_training` | NO (string/Join) |
| LightGBMTraining | `lightgbm_training` | NO (string/Join) |
| LightGBMMTTraining | `lightgbmmt_training` | NO (string/Join; module-level `Join`, no local import) |
| XGBoostModel / PyTorchModel | — | returns **None** (CreateModelStep auto-provides ModelName) |
| BatchTransform | `spec.step_type.lower()` else `batch_transform` | **yes** (returns **str/Join**) |
| Cradle | — | returns **`{}`** (logs contract paths only) |
| Redshift | — | returns `{}` (logs) |
| DataUploading | — | returns `{}` (no log) |
| Registration | — | returns **None** |
| EdxUploading | — | returns **`[]`** (sink node) |

**Return-type divergence is itself a quirk.** Processing builders return `List[ProcessingOutput]`. Training builders return a **single string or `Join`** (and `create_step` always calls `_get_outputs({})` with an empty dict, so the explicit-output branch is **dead at runtime** and the default `Join` always wins — `...xgboost_training_step.py:444`). BatchTransform returns a **`str`**. Model builders return **`None`**. SDK-delegation returns **`{}`**, `None`, or `[]`. DummyTraining returns a **single-element list** with backward-compat dual logical-name lookup `outputs.get('model_output', outputs.get('model_input', default))` (`...dummy_training_step.py:267-269`).

---

### 2d. `create_step` assembly

**The per-verb SDK shapes:**

- **2A — ProcessingStep `code=`:** `ProcessingStep(name, processor, code=script_path, job_arguments=job_args, depends_on, cache_config)`. Used by all Processing-2A builders, BedrockBatchProcessing, ActiveSampleSelection, PseudoLabelMerge (the SKLearnProcessor / whole-path ones), and EdxUploading (with a self-built ScriptProcessor).
- **2B — ProcessingStep with `run()->step_args`:** `step_args = processor.run(code=..., inputs, outputs, arguments=job_args)`; then `ProcessingStep(name, step_args=step_args, depends_on, cache_config)`. Used by the 2B eval-inference family and the source-dir-splitting Processing-2B-rest builders (RiskTableMapping, TokenizerTraining, DummyTraining, BedrockProcessing, PercentileModelCalibration).
- **Training — `estimator=`:** `TrainingStep(name, estimator=..., inputs=training_inputs, depends_on, cache_config)`. No `code=`, no `step_args=`, no job_arguments (see latent hook §2f).
- **ModelCreation — `model=`:** `CreateModelStep(name, model=model, depends_on)` — model passed **directly, not as `step_args`** (explicit comment, `...xgboost_model_step.py:263`).
- **Transform — `transformer=`:** `TransformStep(name, transformer=..., inputs=transform_input, depends_on or [])`.
- **SDKDelegation:** `CradleDataLoadingStep(step_name, role, sagemaker_session)` / `RedshiftDataLoadingStep(...)` / `DataUploadingStep(..., input_s3_location=...)` / `MimsModelRegistrationProcessingStep(..., processing_input=<LIST>, performance_metadata_location, depends_on)`. **Dependencies via `step.add_depends_on(dependencies)`** for Cradle/Redshift/DataUploading, but **`depends_on=` ctor kwarg** for Registration.

**`setattr(_spec / _contract)` divergence:**
- Most builders: `setattr(step, '_spec', self.spec)` only, **no `_contract`**.
- Guard style varies: **unconditional** (TabularPreprocessing `:365`, LabelRulesetExecution `:374`, RiskTableMapping `:409`, DummyTraining `:382`) vs **`hasattr`-guarded** (DummyDataLoading `:334-335`, the 2B family, Bedrock, ActiveSampleSelection, PseudoLabelMerge, the Model/Transform builders).
- Cradle and Registration set **both** `_spec` and `_contract` (`...cradle_data_loading_step.py:291-294`, `...registration_step.py:292-295`).
- **EdxUploading sets neither** — the only builder that attaches no spec/contract to its step.

**Caching divergence:**
- Default `enable_caching=True` everywhere.
- **CreateModelStep does NOT support `cache_config`** — XGBoostModel/PyTorchModel log a warning and *drop* caching if enabled (`...xgboost_model_step.py:256-259`).
- **Cradle forces caching off** by mutating `step.cache_config.enable_caching = False` only when `not enable_caching` (`...cradle_data_loading_step.py:286-287`); Redshift does the same (`:126-127`).
- **DataUploading and Registration have NO caching handling at all.**
- BatchTransform supports caching normally.

**Error envelopes (try/except → re-raise as `ValueError('Failed to create <X> step: ...') from e`):**
- Present: MissingValueImputation (`:416-423`), FeatureSelection (`:427-433`), RiskTableMapping (`:413-418`), DummyTraining (`:386-391`), all four Training builders (XGBoost/PyTorch/LightGBM/LightGBMMT), XGBoostModel/PyTorchModel, Cradle, DataUploading, Registration.
- **PyTorchTraining uses `log_error`** in its envelope; the other three Training builders use `log_warning`.
- **Absent (bare construction):** TabularPreprocessing, the Temporal family, StratifiedSampling, CurrencyConversion, DummyDataLoading, TokenizerTraining, BedrockProcessing, BedrockBatchProcessing, ActiveSampleSelection, PseudoLabelMerge, PercentileModelCalibration, BatchTransform, Redshift, EdxUploading.
- The `extract_inputs_from_dependencies` call is itself wrapped in a try/except-that-only-warns in most 2B/Model builders (e.g. `...xgboost_model_eval_step.py:322-326`).

---

### 2e. source_dir / path resolution

This is the highest-variance verb. There are **five distinct modes**, and the docker container receives code differently in each:

1. **`code=` full path (no split).** `script_path = config.get_script_path()` → `ProcessingStep(code=script_path)` *(2A)* or `processor.run(code=script_path)` *(some 2B)*. The container gets a single script file. Used by: all Processing-2A builders; and — divergently — **LightGBMModelEval/Inference, BedrockBatchProcessing, ActiveSampleSelection, PseudoLabelMerge** pass the whole path as `code=` to `processor.run()` even though some use FrameworkProcessor. **LightGBMModelEval/Inference using `code=`-full-path despite FrameworkProcessor is an inconsistency vs the PyTorch/LightGBMMT siblings that split — preserve it or scripts with local imports break differently.**

2. **`source_dir + entry_point` split (hybrid Lambda-safe).** `script_path = config.get_script_path()` → `source_dir = Path(script_path).parent`, `entry_point = Path(script_path).name` → `processor.run(code=entry_point, source_dir=source_dir)`. The container gets the whole directory (enables local imports). Used by: the 2B FrameworkProcessor builders (LightGBMMTModelEval/Inference, PyTorchModelEval/Inference), RiskTableMapping, TokenizerTraining, DummyTraining, BedrockProcessing, PercentileModelCalibration. Comments cite "works in Lambda where dirs may not exist." `Path` is re-imported locally inside `create_step` in several (cosmetic; LightGBMMT uses module-level `Join`/`Path` instead — note the inconsistency).

3. **`effective_source_dir` on the estimator (Training).** `source_dir = config.effective_source_dir` passed with `entry_point = config.training_entry_point` to the estimator ctor (`...xgboost_training_step.py:127-132` and siblings). No `get_script_path()`, no split. Hybrid resolution lives in the config property.

4. **Model-SDK `source_dir + entry_point` pair.** XGBoostModel/PyTorchModel: `source_dir = config.effective_source_dir`, `entry_point = config.entry_point` passed to `XGBoostModel(...)`/`PyTorchModel(...)` (`...xgboost_model_step.py:134-141`). Both validated as required attrs.

5. **No source_dir at all.** BatchTransform (reuses an existing model by name), and all four SDK-delegation builders (Cradle/Redshift/DataUploading/Registration) — the SDK step owns the script. EdxUploading is special: it builds its own `ScriptProcessor` and resolves `code=` via a bespoke `_resolve_script_path()` 3-tier fallback — (1) `config.processing_source_dir / processing_entry_point` **only if it exists**, (2) `config.effective_source_dir / processing_entry_point` **without existence check**, (3) `Path(__file__).parent.parent/'scripts'/processing_entry_point` (`...edx_uploading_step.py:180-207`).

**source_dir as a ProcessingInput source (not processor source_dir).** Package ships the source dir *into the container as an input channel*: `inference_scripts_input` source is `config.resolved_source_dir or config.source_dir or 'inference'` (3-tier, `...package_step.py:190-194`) — distinct from its `code=` script path. The Bedrock/Ruleset/PercentileCalibration ship-in-source-dir input skips (§2b) similarly depend on the source dir being shipped, not on a wired channel.

---

### 2f. Job arguments

**How it works.** `_get_job_arguments()` returns the argv list the script receives. The dominant shape is `['--job_type', config.job_type]` — **sourced from config, not contract**, even when `contract.expected_arguments` is empty (verifier-confirmed: config takes precedence; the comment says so explicitly in StratifiedSampling `:336,341`).

**Hyphen vs underscore (read carefully — this is a per-step container contract):**
- `--job_type` (underscore): every Processing builder that emits job_type, XGBoostModelEval/Inference, the LightGBM/PyTorch eval-inference family, ModelCalibration, ModelMetricsComputation, RiskTableMapping, TokenizerTraining, PercentileModelCalibration, PseudoLabelMerge.
- **`--job-type` (HYPHEN):** **LabelRulesetExecution only** (`...label_ruleset_execution_step.py:303`). A script expecting `--job_type` will not see this.
- **Mixed style:** BedrockProcessing/BedrockBatchProcessing emit `['--job_type', X, '--batch-size', str(...), '--max-retries', str(...)]` — underscore on job_type, **hyphens** on the batch flags, ints `str()`-coerced (`...bedrock_processing_step.py:458-465`).
- Bedrock conditional flags: BedrockPromptTemplateGeneration appends `--include-examples` / `--generate-validation-schema` only when the config flag is true, always `['--template-version', config.template_version]` (`...bedrock_prompt_template_generation_step.py:355-382`).

**Returns `None` (no argv):** DummyDataLoading (`:266`), LabelRulesetGeneration (deliberate, env-only, `:347-361`), Package (`:346-355`), ModelWikiGenerator (`:377-387`), DummyTraining (`:292-301`). These scripts use hardcoded paths or env-only config.

**Config-overridable args:** Payload — if `config.processing_script_arguments` is set/truthy returns those, else `None` (`...payload_step.py:306-327`). This override path is easy to lose.

**Latent / unwired `_get_job_arguments` (the trap):** All four Training builders **define** `_get_job_arguments` returning `['--job_type', job_type]` or `None`, but it is **never called** — `TrainingStep` receives no `job_arguments` and the method has zero call sites (verifier-confirmed via grep: only the four `def` lines, no callers). Hyperparameters are embedded in the source dir instead. **Do not "clean up" this dead method without recording that it is a documented latent hook** — someone may intend to wire it. (TokenizerTraining defines *and calls* `_get_job_arguments` at `:345` because it builds a ProcessingStep, not a TrainingStep — not a counterexample.)

---

## 3. Per-builder quick-reference table

| Builder | env_vars deviation | input quirk | output token + jt-in-path | source_dir mode | job_args | other quirks |
|---|---|---|---|---|---|---|
| TabularPreprocessing | UPPER+`,`join 4 col lists | direct `[DATA,METADATA,SIGNATURE]` | `tabular_preprocessing` + **jt** | code= full | `--job_type` | unified spec; dup job_type allowlist |
| TemporalSplitPreprocessing | named dict merge | direct `[DATA,SIGNATURE]` (no METADATA) | `temporal_split_preprocessing` **no-jt** | code= full | `--job_type` | only group member omitting jt from path |
| TemporalSequenceNormalization | no-op (doc only) | direct `[DATA,SIGNATURE]` | `temporal_sequence_normalization` + jt | code= full | `--job_type` | per-job-type spec in `__init__`; log_info before super |
| TemporalFeatureEngineering | no-op (doc only) | direct `[normalized_sequences]` | `temporal_feature_engineering` + jt | code= full | `--job_type` | per-job-type spec; rich domain validate |
| StratifiedSampling | heavy inline `str`/`.lower` | direct `[processed_data,DATA,METADATA,SIGNATURE]` | `stratified_sampling` + jt | code= full | `--job_type` | TYPE_CHECKING imports; optimal-strategy warning |
| MissingValueImputation | `environment_variables`+`env` | direct `[data_input,imputation_params_input,processed_data]` | `missing_value_imputation` + jt | code= full | `--job_type` | **error envelope**; per-job-type spec; rich validate |
| FeatureSelection | + **`USE_SECURE_PYPI`** | direct `[processed_data,selected_features]` | `feature_selection` + jt | code= full | `--job_type` | **error envelope**; per-job-type spec |
| CurrencyConversion | config property + env | direct `[data_input]` | `currency_conversion` + jt | code= full | `--job_type` | validate delegated to config |
| DummyDataLoading | method merge; filtered log | **hardcoded INPUT_DATA from config** | `dummy_data_loading` + jt | code= full | **None** | fw default `1.2-1`; isinstance guard; no job_type req |
| BedrockPromptTemplateGeneration | config property | **`prompt_configs` ship-in-source** | `bedrock_prompt_template_generation` no-jt | code= full | conditional flags | heavy template validate; JSON-config probe |
| LabelRulesetGeneration | config property | **`ruleset_configs` ship-in-source** | `label_ruleset_generation` no-jt | code= full | **None** (env-only) | nested label_config validate |
| LabelRulesetExecution | **`execution_environment_variables`** | plain join | `label_ruleset_execution` + **jt** | code= full | **`--job-type` (hyphen)** | unconditional setattr; fw no fallback |
| ModelCalibration | delegate/super | **circular-ref detect** | `model_calibration` + jt | code= full | `--job_type` (underscore) | module spec; helpers; fw `1.0-1`; latent importlib |
| Package | inline `str(getattr)` UPPER | **always-inject `inference_scripts_input` override** | `packaging` no-jt | code= full **+ source_dir as input** | **None** | hardcoded `/opt/ml/processing/input/script` fallback |
| Payload | `,`join,`str`,`json.dumps` | inject `custom_payload_input` | `payload` no-jt | code= full | **config-overridable** | TYPE_CHECKING+runtime import dance |
| ModelMetricsComputation | delegate/super | circular-ref detect | `model_metrics_computation` no-jt | code= full | `--job_type` | module spec; helpers; warning-not-error validate |
| ModelWikiGenerator | 13-key inline; `MODEL_VERSION`=pipeline_version | circular-ref detect | `model_wiki_generator` no-jt | code= full | **None** | warnings-not-errors validate; cti allowlist |
| XGBoostModelEval | delegate/super | plain join | `model_evaluation` no-jt | **code= full (XGBoostProcessor)** | `--job_type` | XGBoostProcessor; `xgboost_framework_version` |
| XGBoostModelInference | delegate/super | plain join | `model_inference` no-jt | code= full (XGBoostProcessor) | `--job_type` | twin of eval; token differs |
| LightGBMModelEval/Inference | delegate/super **+log** | plain join | `model_evaluation`/`model_inference` no-jt | **code= full despite FrameworkProcessor** | `--job_type` | PyTorch container, LightGBM pip; **divergent from MT** |
| LightGBMMTModelEval/Inference | delegate/super +log | plain join | `model_evaluation`/`model_inference` no-jt | **source_dir split** | `--job_type` | multi-task validate (≥2 task_label_names) |
| PyTorchModelEval/Inference | delegate/super (latent doc) | plain join | **`pytorch_model_evaluation`/`pytorch_model_inference`** no-jt | source_dir split | `--job_type` | unique tokens; eval docstring mis-says SKLearn |
| RiskTableMapping | property + **`env` double** | **`FullyReplicated`**; direct `[data_input,config_input,risk_tables]` | `risk_table_mapping` + **jt** | source_dir split | `--job_type` | **error envelope**; unconditional setattr; dead imports |
| TokenizerTraining | **config replaces base** (no super in primary) | plain join | `tokenizer_training` no-jt | source_dir split | `--job_type` | PyTorch estimator (BPE); module spec |
| DummyTraining | super passthrough | **3-tier model resolution**; can return `[]` | `dummy_training`/output; **dual-name aliasing** | source_dir split | **None** | **error envelope**; `_get_processor`; `get_instance_type()` dispatch |
| BedrockProcessing | `bedrock_environment_variables` | plain join | `bedrock_processing` no-jt | source_dir split | `--job_type` + hyphen batch flags | PyTorch container for boto3≥1.35; heavy numeric validate |
| BedrockBatchProcessing | **computed `Join` S3-path env vals** | plain join | `bedrock_batch_processing` no-jt | **code= full (no split)** | mixed hyphen/underscore | ARN-format validate; cost summary log |
| ActiveSampleSelection | method merge | plain join | `active_sampling` no-jt | code= full (SKLearnProcessor) | `--job_type` **default `ssl_selection`** | label_field allowed `''` |
| PseudoLabelMerge | method merge | custom required-msg both channels | `pseudo_label_merge` no-jt | code= full (SKLearnProcessor) | `--job_type` | **`valid_job_types` NameError BUG** in validate |
| PercentileModelCalibration | **method sole source (no super)** | **`calibration_config` ship-in-source skip** | `percentile_model_calibration` + **jt** | source_dir split | `--job_type` | fw `1.2-1`; single/multi-task duality; dead SKLearnProcessor import |
| XGBoostTraining | delegate+`config.env`; **no CA_REPOSITORY_ARN** | input_path→train/val/test; **parts[5] channel parse**; `skip_hyperparameters_s3_uri` | `xgboost_training` no-jt (**string**) | effective_source_dir on estimator | **latent/dead** | XGBoost estimator; no region hardcode; dead imports |
| PyTorchTraining | delegate+`config.env`; **CA_REPOSITORY_ARN** | input_path channels; **logical_name-as-channel** (no parse) | `pytorch_training` no-jt (string) | effective_source_dir | **latent/dead** | **hardcoded `us-east-1`**; max_runtime_seconds; dead `_get_metric_definitions`/`_create_profiler_config`; [DEPENDS_ON] logging; log_error envelope |
| LightGBMTraining | delegate+`config.env`; CA_REPOSITORY_ARN | input_path; **parts[5] parse** | `lightgbm_training` no-jt (string) | effective_source_dir | **latent/dead** | **PyTorch estimator + LightGBM pip**; **hardcoded `us-east-1`**; max_runtime_seconds |
| LightGBMMTTraining | delegate+`config.env`; CA_REPOSITORY_ARN | input_path; parts[5] parse | `lightgbmmt_training` no-jt (string; module Join) | effective_source_dir | **latent/dead** | PyTorch+LightGBM pip; **hardcoded `us-east-1`**; **`max_run_seconds`** (field-name divergence) |
| XGBoostModel | super+`config.env` verbatim | **single-key `model_data` passthrough; 3-source resolve in create_step** | **None** | Model SDK source_dir+entry_point | — (none) | **hardcoded `us-east-1`**; XGBoostModel class; image_uri override; CreateModelStep caching-drop |
| PyTorchModel | super+`config.env` verbatim | single-key passthrough; 3-source resolve | None | Model SDK source_dir+entry_point | — | **hardcoded `us-east-1`**; eager Registry imports; CreateModelStep caching-drop |
| BatchTransform | **none** | hardcoded `model_name`/`processed_data` dispatch; **returns Tuple** | `step_type.lower()`/`batch_transform` + **jt** (**str**) | **none** | — | job-type-variant spec; caching supported; no envelope |
| CradleDataLoading | none | **`[]`** (source) | **`{}`** | none (SDK) | none | SDK step; both `_spec`+`_contract`; cache force-off; helpers; removed `_build_request` |
| RedshiftDataLoading | none | `[]` (source) | `{}` | none (SDK) | none | SDK step; no envelope; relies on base setting `self.contract`; cache force-off |
| DataUploading | none | `[]` + `_resolved_input_s3` side-effect | `{}` | none (SDK) | none (SDK builds `--input-path`) | Properties passthrough; **no caching**; envelope |
| Registration | none | **ordered MIMS list; PackagedModel first; FullyReplicated/File** | **None** | none (SDK) | none | module spec+contract; **region suffix on step name**; strict spec-contract align; both setattr |
| EdxUploading | **inline, no super; `json.dumps`×2** | single `input_data`; **hardcoded dest** | **`[]`** (sink) | bespoke `_resolve_script_path` 3-tier | none (env-only) | **own ScriptProcessor; hand-built ECR image URI from role ARN; KMS+network injection**; validate ×2; no setattr |

---

## 4. Migration risk register

For each special treatment: the **absorption mechanism** in the new shell, and a **MUST PRESERVE** flag where loss causes runtime breakage rather than cosmetic drift.

| # | Special treatment | Absorption mechanism | Flag |
|---|---|---|---|
| R1 | Contract-driven base env vars | Declarative — already the YAML contract `required_env_vars`/`optional_env_vars`; base handler reads it | — |
| R2 | Named config env-dict merge (`temporal_split_environment_variables`, `execution_environment_variables`, `bedrock_environment_variables`, double `config.env`) | **Declarative YAML knob**: `env_merge_source: <attr name>` per step | MUST PRESERVE the *attribute name* — wrong name = empty merge |
| R3 | UPPERCASE + `','.join` column lists (Tabular); `CONTENT_TYPES` join (Payload) | **Per-pattern handler method** `coerce_list_join(upper=True)` keyed off a YAML coercion spec | MUST PRESERVE |
| R4 | `json.dumps` env values (Payload FIELD_DEFAULTS; Edx ×2) | Per-pattern handler `coerce_json` | MUST PRESERVE |
| R5 | `str()` / `str(bool).lower()` coercion (Stratified, WikiGen, FeatureSelection USE_SECURE_PYPI) | Per-pattern handler `coerce_str` / `coerce_bool_lower`; declarative type tag per env key | MUST PRESERVE (script reads exact string form) |
| R6 | **Computed `Join` PipelineVariable env values** (BedrockBatchProcessing S3 paths) | **Per-step sidecar override** — env value is a pipeline variable, not coercible by a generic handler | **MUST PRESERVE — load-bearing** |
| R7 | `MODEL_VERSION` = `pipeline_version` rename (WikiGen) | Declarative key→source mapping in YAML | MUST PRESERVE |
| R8 | Config method-vs-property, and **no-super sole-source** (Tokenizer, PercentileCalibration) | **Per-step sidecar flag** `env_base: none\|super\|config_only`; collapsing to uniform super+update changes behavior | **MUST PRESERVE — load-bearing** (suppresses base vars today) |
| R9 | `CA_REPOSITORY_ARN` on PyTorch/LightGBM/LightGBMMT training (not XGBoost) | Declarative `secure_pypi: true` knob | MUST PRESERVE per-builder (XGBoost must stay off) |
| R10 | Generic `spec×contract` input join | Declarative — the default handler | — |
| R11 | **`FullyReplicated`** on RiskTableMapping inputs | **Declarative YAML knob** `s3_data_distribution_type` per input | **MUST PRESERVE — load-bearing** (broadcast semantics) |
| R12 | `direct_input_keys` allowlists (8 builders, varying lists incl. METADATA presence/absence) | Declarative `direct_input_keys: [...]` list per step | MUST PRESERVE (exact list, incl. METADATA in/out) |
| R13 | **Ship-in-source-dir input skips** (`prompt_configs`, `ruleset_configs`, `calibration_config`) | Declarative `skip_input: <logical_name>` + dependency on source_dir shipping | **MUST PRESERVE — load-bearing** (channel must NOT be wired) |
| R14 | Package always-inject `inference_scripts_input` override + delete-from-loop | **Per-step sidecar override** (3-tier source resolution + container fallback path) | **MUST PRESERVE — load-bearing** |
| R15 | DummyDataLoading hardcoded `INPUT_DATA` from `config.get_data_source_uri()` | Per-step sidecar override | **MUST PRESERVE — load-bearing** (ignores dependency inputs) |
| R16 | Circular-reference detection (ModelCalibration, ModelMetrics, WikiGen) | **Shared handler method** invoked when a YAML flag `detect_input_cycles: true` is set | MUST PRESERVE (latent safety check — see also latent hooks) |
| R17 | **Registration ordered MIMS list** (PackagedModel first, FullyReplicated/File, 1-or-2 only) | **Per-step sidecar override** — ordering + input_mode are MIMS contract | **MUST PRESERVE — load-bearing** |
| R18 | **DummyTraining 3-tier model_data resolution** (config > dependency > SOURCE empty) | Per-step sidecar override | **MUST PRESERVE — load-bearing** |
| R19 | **XGBoostModel/PyTorchModel 3-source model_data** (extract > inputs_raw > direct kwarg) | Per-pattern handler for the Model verb | **MUST PRESERVE — load-bearing** |
| R20 | BatchTransform Tuple return + hardcoded `model_name`/`processed_data` dispatch | Per-verb handler for Transform | MUST PRESERVE |
| R21 | Delegation `[]`/`{}`/`None` input/output shapes (Cradle/Redshift/DataUploading) | Per-verb SDKDelegation handler | — |
| R22 | Path token + job_type-in-path (per §2c table) | **Declarative YAML knobs** `output_path_token` + `job_type_in_path: bool` | MUST PRESERVE (downstream globs the path) |
| R23 | TemporalSplit omitting job_type from path (vs siblings) | Declarative `job_type_in_path: false` | **MUST PRESERVE — load-bearing** (silent path divergence) |
| R24 | PyTorch eval/inference unique tokens (`pytorch_model_evaluation`) | Declarative token | MUST PRESERVE |
| R25 | Return-type per verb (List/str/None/{}/[]/Tuple) | Per-verb handler | MUST PRESERVE |
| R26 | DummyTraining dual logical-name aliasing (`model_output`/`model_input`) | Per-step sidecar override (backward-compat) | MUST PRESERVE |
| R27 | Per-verb `create_step` SDK shape (code= / run() / estimator= / model= / transformer= / SDK ctor) | **Per-pattern handler method** (one per verb) | MUST PRESERVE |
| R28 | `setattr(_spec)` guard style; `_contract` on Cradle/Registration; EdxUploading neither | Per-verb handler with declarative `attach_contract: bool` | MUST PRESERVE (Cradle/Registration need `_contract`) |
| R29 | Caching divergence (CreateModelStep drop, Cradle/Redshift force-off, DataUploading/Registration none) | Per-verb handler caching policy | MUST PRESERVE (CreateModelStep crashes on cache_config) |
| R30 | Error-envelope presence + PyTorchTraining `log_error` vs `log_warning` | Per-verb handler; declarative `error_envelope: bool` | preserve (diagnostics) |
| R31 | Five source_dir modes (code= full / split / estimator effective / Model pair / none) | **Per-verb handler** keyed off `source_mode` YAML enum | MUST PRESERVE |
| R32 | LightGBMModelEval/Inference `code=`-full despite FrameworkProcessor (vs MT split) | Per-step sidecar `source_mode: code_full` | **MUST PRESERVE — load-bearing** (local-import behavior differs) |
| R33 | EdxUploading bespoke `_resolve_script_path` 3-tier + own ScriptProcessor | **Per-step sidecar override** | **MUST PRESERVE — load-bearing** |
| R34 | EdxUploading **hand-built ECR image URI from role ARN** (`role.split(':')[4]` + `sais_python_lib_docker_image`) | Per-step sidecar override | **MUST PRESERVE — load-bearing** |
| R35 | **EdxUploading KMS + network injection** (`KMS_ENCRYPTION_KEY_PARAM`, `PROCESSING_JOB_SHARED_NETWORK_CONFIG`) | Per-step sidecar override | **MUST PRESERVE — load-bearing (security/network)** |
| R36 | **Hardcoded `us-east-1`** in PyTorch/LightGBM/LightGBMMT Training image_uris.retrieve and XGBoostModel/PyTorchModel `_get_image_uri` | Per-step sidecar override; flag as a **known BUG** | **MUST PRESERVE — load-bearing BUG** (env "limitation" is real; fixing it during migration is out of scope and risky) |
| R37 | Framework-mismatch: LightGBM/LightGBMMT/Tokenizer/Bedrock on **PyTorch** container; LightGBM-eval on PyTorch via FrameworkProcessor | Declarative `processor_class` + `estimator_cls` knobs | **MUST PRESERVE — load-bearing** (library pip-installed into that container) |
| R38 | Framework-version fallback defaults (`1.2-1`, `1.0-1`; XGBoostModel none) | Declarative `framework_version_default` | preserve |
| R39 | `--job_type` underscore vs **`--job-type` hyphen** (LabelRulesetExecution); Bedrock mixed flags | Declarative `job_arg_style` + per-step flag spec | **MUST PRESERVE — load-bearing** (script argv parser is exact) |
| R40 | ActiveSampleSelection job_type **default `ssl_selection`** | Declarative `job_type_default` | MUST PRESERVE |
| R41 | `_get_job_arguments` returning `None` (DummyDataLoading, LabelRulesetGeneration, Package, WikiGen, DummyTraining) | Declarative `job_args: none` | MUST PRESERVE |
| R42 | Payload config-overridable args (`processing_script_arguments`) | Declarative `job_args_override_attr` | MUST PRESERVE |
| R43 | **Latent/dead `_get_job_arguments` on all four Training builders** (defined, never wired) | **Per-step sidecar `latent_hooks: [...]`** — keep as documented, do not delete | **MUST PRESERVE — latent hook** |
| R44 | PyTorchTraining dead `_get_metric_definitions` (4 regexes) + `_create_profiler_config` | Per-step `latent_hooks` | **MUST PRESERVE — latent hook** |
| R45 | LightGBMMTTraining **`max_run_seconds`** vs siblings' `max_runtime_seconds` | Declarative `max_run_attr` | MUST PRESERVE (field-name divergence) |
| R46 | Training input channel derivation: parts[5] path-parse (XGB/LightGBM/MT) vs logical-name-as-channel (PyTorch) | Per-verb handler with declarative `channel_mode` | **MUST PRESERVE — load-bearing** (channel names differ) |
| R47 | `skip_hyperparameters_s3_uri` allowlist skip (all Training) | Declarative `skip_input_if_config: <flag>` | MUST PRESERVE |
| R48 | **PseudoLabelMerge `valid_job_types` NameError BUG** in validate | Per-step sidecar; flag as **known BUG** — preserve behavior OR fix deliberately, do not silently alter | **MUST PRESERVE — latent BUG** (decide explicitly) |
| R49 | Rich `validate_configuration` domain allowlists/ranges (Bedrock numeric ranges, calibration methods, feature-selection method lists, etc.) | **Declarative validation block** in YAML OR retained per-step validator | MUST PRESERVE (deletion removes guardrails) |
| R50 | Spec load timing/variant (module-level unified vs `__init__` per-job-type vs job_type-parameterized) | Declarative `spec_variant: unified\|per_job_type` | MUST PRESERVE (per-job-type specs differ) |
| R51 | Registration **region suffix on step name**; step_name conventions | Per-verb handler | MUST PRESERVE (step name is an identity) |
| R52 | Dead imports / stale docstrings (mislabeled "SKLearn"/"Scikit-Learn" comments) | Cosmetic — safe to drop, but do not let stale comments mislead the YAML author | — |

**Decision rule for the engineer:** any row flagged **MUST PRESERVE** that you cannot express as a declarative YAML knob (R1–R5, R7, R9–R13, R22–R24, R39–R42, R45–R47, R49–R51) must be implemented as either a **per-pattern handler method** (shared across a verb family) or a **per-step sidecar override**. Rows R6, R8, R13–R19, R32–R37, R43–R44, R48 are explicitly **not declaratively expressible today** — they require sidecar overrides, and shipping the collapse without them is a regression. The hardcoded-region (R36) and PseudoLabelMerge (R48) entries are **bugs you must carry forward unchanged** unless the user explicitly authorizes fixing them in scope.

---

## Related

- Slipbox plan: [2026-06-26_step_builder_simplification_plan.md](../2_project_planning/2026-06-26_step_builder_simplification_plan.md) — the master plan this note gates.
- The FZ 31e design trail (the declarative-knob / per-pattern-handler / per-step-sidecar absorption taxonomy referenced throughout §4) lives in the **abuse_slipbox** vault, not in this repo's slipbox.