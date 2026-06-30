---
tags:
  - project
  - planning
  - step_builder
  - compute_construction
  - strategy_pattern
keywords:
  - _create_processor
  - _create_estimator
  - ComputeSpec
  - compute single source
  - declarative compute block
topics:
  - moving processor/estimator/model construction into the builder template
language: python
date of note: 2026-06-28
---
# Compute construction single-source — collapse `_create_processor` / `_create_estimator` into the template (FZ 31e1d3k)

> Directive (2026-06-28): "`_create_processor` and `_create_estimator` should be part of the standard
> builder template — all their fields are from config and can be fully constructed given config";
> "put the Class as part of steps.yaml so it is also surfaced to user". This is the COMPUTE axis of
> the per-shell residue collapse (after env / job-args / inputs / outputs). Same model: the
> `.step.yaml` declares WHICH SDK class + WHICH config fields (a `compute:` block); a generic
> `_create_compute()` on the builder template builds it from config. The compute CLASS is declared
> data, so it surfaces in `cursus steps patterns`.

## Why this is feasible (evidence)

Audited all 41 compute factories across the 45 builders. Every constructor ARG is already a config
field (`processing_framework_version`, instance type/count, volume, `role`, `session`, env) or a base
method (`_generate_job_name`, `_get_environment_variables`, `_get_base_output_path`). What varies
between steps is a tiny descriptor: the **SDK class** + which **framework/py version field** + a few
flags. The dominant group (19 `SKLearnProcessor` factories) is byte-identical (same md5). So the
factory body is boilerplate; the only per-step DATA is the descriptor.

`_generate_job_name` is already on `StepBuilderBase` (0 per-builder copies) — the generic resolver
can call it directly.

## Factory inventory (the 41 — class → declarative descriptor)

| Class | count | steps | descriptor |
|---|---|---|---|
| `SKLearnProcessor` | 19 | tabular_preprocessing, currency_conversion, stratified_sampling, feature_selection, missing_value_imputation, pseudo_label_merge, active_sample_selection, payload, temporal_*(3), package, dummy_data_loading, label_ruleset_generation/execution, bedrock_prompt_template_generation, model_calibration/metrics/wiki (`_get_processor`) | `kind: sklearn`, `framework_version_field` (`processing_framework_version` default; `framework_version` for some) |
| `FrameworkProcessor` | 11 | bedrock_processing, bedrock_batch_processing, risk_table_mapping, tokenizer_training, lightgbm/lightgbmmt model_eval+inference, pytorch_model_eval+inference, percentile_model_calibration, dummy_training | `kind: framework`, `sdk_class: PyTorch|SKLearn`, `framework_version_field`, `py_version_field` |
| `XGBoostProcessor` | 2 | xgboost_model_eval, xgboost_model_inference | `kind: xgboost`, `framework_version_field: xgboost_framework_version` |
| `ScriptProcessor` | 1 | edx_uploading | `kind: script`, `kms_network: true` (ECR-from-role image + `volume_kms_key` + `network_config` — the one genuinely special processor) |
| `XGBoost`/`PyTorch` estimator | 4 | xgboost_training, pytorch_training, lightgbm_training, lightgbmmt_training | `kind: estimator`, `sdk_class: XGBoost|PyTorch`, `retrieve_image: true` for the lightgbm PyTorch-container case (explicit `image_uris.retrieve`) |
| `PyTorchModel`/`XGBoostModel` | 2 | pytorch_model, xgboost_model | `kind: model`, `sdk_class: …` (threads `model_data`) |
| `Transformer` | 1 | batch_transform | `kind: transformer` (threads `model_name` + `output_path`) |

## Schema — `ComputeSpec`, a TOP-LEVEL `.step.yaml` section (DONE)

```yaml
# .step.yaml — compute is a TOP-LEVEL section (peer of registry/contract/spec), NOT under contract:
compute:
  kind: sklearn                 # sklearn|xgboost|framework|script|estimator|model|transformer
  framework_version_field: processing_framework_version
  py_version_field: py_version  # framework/estimator/model
  sdk_class: PyTorch            # framework estimator_cls / estimator / model class
  framework_name: xgboost       # model only: inference image framework for image_uris.retrieve
  instance_size_mode: large_or_small   # use_large_processing_instance ? large : small
  kms_network: false            # script (edx) only
  retrieve_image: false         # estimator: explicit image_uris.retrieve (lightgbm)
  lock_training_region: false   # estimator/model: toggleable SAIS us-east-1 image lock
```
`ComputeSpec` lives in `core/base/step_interface.py`; default `kind=None` ⇒ the step keeps its own
factory (opt-in per step, no behavior change until a step declares `compute:`).

**Promoted to top level (2026-06-28).** It was originally nested under `contract:`, but `compute`
describes the BUILDER's compute object, not the script contract — script-less steps (CreateModel /
Transform) carry a near-empty contract but a full compute, so nesting was a category error.
`StepInterface.compute` is now a top-level field (peer of `contract`/`spec`); `ContractSection.compute`
is retained as a BACK-COMPAT MIRROR, reconciled by `StepInterface._sync_and_align` (declare in one
place — they must agree if both set — and both get the resolved value), so the runtime
`b.contract.compute` read sites are untouched. All 41 `.step.yaml` compute blocks migrated;
`io_view`/conformance read the top-level. A conformance check asserts the mirror stays equal.

## Build plan (phased, byte-diff gated)

The compute object can't be byte-compared fully offline (the SDK ctor needs a real
`sagemaker_session`), but the existing `mock_session` (Mock with `boto_region_name='us-east-1'`,
`local_mode=False`) lets the processor construct — so the gate compares the constructed object's
identifying attributes (class, framework_version, py_version, instance_type/count, volume, image_uri).

1. **Foundation** (DONE): `ComputeSpec` model + `compute` field + a generic `_create_compute(self,
   output_path=None)` on `builder_base` that dispatches on `kind` and builds from config; the
   handlers' `make_compute` default calls `_create_compute` when the step declares `compute.kind`,
   else the existing `_create_processor`/`_create_estimator` hook (smooth transition).
2. **Phase A — SKLearnProcessor ×19** (the byte-identical bulk): declare `compute: {kind: sklearn,
   framework_version_field}`; delete the factories; byte-diff each via mock_session.
3. **Phase B — FrameworkProcessor ×11 + XGBoostProcessor ×2**: `kind: framework|xgboost` + `sdk_class`.
4. **Phase C — estimators ×4**: `kind: estimator`; handle `retrieve_image` (lightgbm) + the
   **hardcoded `region="us-east-1"` bug** (decide: fix to `config.aws_region` or carry forward — it
   is flagged in the safety record; the collapse is the natural place to fix it, gated).
5. **Phase D — models ×2 + transformer ×1**: `kind: model|transformer` (model_data/model_name threading).
6. **EdxUploading ScriptProcessor**: `kind: script, kms_network: true` — last, SDK-bound (static
   verification + mock-session where possible).

Genuine keeps expected: none anticipated — every factory is descriptor-reducible; the EdxUploading
KMS/network is a declared `kms_network` flag, not an override. If any factory has logic beyond the
descriptor (e.g. a computed image), it gets a narrow flag or stays as a documented keep.

## Surfacing (the directive's second half)

Because `compute.kind` + `sdk_class` are `.step.yaml` DATA, `cursus steps patterns` /
`steps.patterns` will show the compute class per step (extend `describe_step_patterns` to add a
`compute` axis: `handler` already shown; add `processor_class`/`estimator_class`). Faithfulness
gated the same way as the other axes (the view derives from the same `compute:` data the builder
reads).

## Gates

Per phase: mock-session byte-diff (constructed compute attrs == pre-collapse) + ruff F821 + the full
`tests/core/base` + `tests/steps` suite + the resolved-edge-graph snapshot (compute doesn't change
wiring, but run it). A new conformance test asserts every declared `compute.kind` resolves to a
buildable compute for a model_construct'd config.

## Toggleable region-lock pattern (clarified: SAIS restriction, NOT a bug)

The hardcoded `region="us-east-1"` in the training estimators' `image_uris.retrieve` is a **SAIS
platform restriction** (training images/jobs forced to us-east-1), not a bug. It is now an explicit,
TOGGLEABLE `ComputeSpec` pattern:
- `lock_training_region: true` + `locked_region: us-east-1` → the SAIS-locked behavior (the 3 PyTorch
  training steps declare this).
- `lock_training_region: false` → standard mode: region from `config.aws_region`.
A step switches between locked and standard mode via `.step.yaml`/config only — **no code change**.

## EdxUploading standardized

EdxUploading was collapsed (not kept): `compute: {kind: script, kms_network: true}` (the ECR-from-role
image + `volume_kms_key` + `network_config`). It was the ONLY processor omitting `base_job_name`
(auto-named); now the `script` kind sets `base_job_name=job_name` like every other processor — its job
naming is standardized to the fleet.

## Status — DONE (2026-06-28, commit `c23c2113`)

- [x] Foundation: `ComputeSpec` + `ContractSection.compute` + Pydantic validation (SDK-grounded).
- [x] `_create_compute()` resolver on builder_base + handler `make_compute` dispatch (override →
      compute.kind → error).
- [x] **Collapsed 37 processor + estimator factories** into `compute:` blocks (incl. EdxUploading
      script). `_create_compute` byte-verified == every factory baseline (mock-session for the 24
      offline-constructible; structural match for the 12 PyTorch-image ones). 1119 passed.
- [x] Region-lock toggleable pattern; EdxUploading standardized.
- [x] `steps patterns` compute axis: `describe_step_patterns` now emits a `compute` block
      (`kind` + `sdk_class` + `framework_version_field` + `lock_training_region`); CLI renders it and
      flags `⚠ custom override` for the 4 steps that keep a factory. The all-steps override-fidelity
      gate was extended to the 5 compute methods (`_create_processor`/`_get_processor`/
      `_create_estimator`/`_create_model`/`_create_transformer`) so the axis can't drift from source.
- [x] **Deferred-4 classified** (the user's question: non-standardized code vs genuine new pattern):
      | step | residue | verdict | why |
      |---|---|---|---|
      | DummyTraining | `_get_processor` | **non-standardized — collapsible** | `config.get_instance_type()` (no arg) == `effective_instance_type` == the `large_or_small` branch `_processing_instance_type` already computes. It's the `framework` kind (FrameworkProcessor + SKLearn) with no special arg. → `compute: {kind: framework, sdk_class: SKLearn, framework_version_field: processing_framework_version}`. |
      | XGBoostModel / PyTorchModel | `_create_model` + `_get_image_uri` | **genuine new pattern** (`kind: model`) | `make_compute` threads `model_data`; ctor needs `entry_point`/`source_dir`/`image_uri`; the image is `image_uris.retrieve(image_scope="inference")` with the toggleable region-lock. The two `_get_image_uri` are byte-identical modulo `framework="xgboost"/"pytorch"` → one `model` branch + `framework_name_field`. |
      | BatchTransform | `_create_transformer` | **genuine new pattern** (`kind: transformer`) | `make_compute` threads `model_name` + `output_path`; ctor is `Transformer(model_name, transform_instance_type/count, accept, assemble_with, output_path)` — a different field set + no image. |
- [x] **Collapsed the genuine patterns** (`kind: model`, `kind: transformer`) + DummyTraining
      (`kind: framework`). `_create_compute` gained a `model` branch (threads `model_data`, retrieves
      the INFERENCE image via `image_uris.retrieve(framework=framework_name, image_scope="inference")`
      with the toggleable region-lock) and a `transformer` branch (threads `model_name`/`output_path`,
      image-less `Transformer`). `ModelCreationHandler`/`TransformHandler` `make_compute` now dispatch
      to `_create_compute(model_data=…)` / `_create_compute(op, model_name=…)` when `compute.kind` is
      set (same resolution order as Processing/Training: override → compute.kind → error). New
      `ComputeSpec.framework_name` (model-only, the inference-image framework, distinct from the model
      `sdk_class`); `_resolve_sdk_class` extended with `PyTorchModel`/`XGBoostModel`. All 4 factories
      DELETED — the builders are pure `STEP_NAME` shells. Byte-diff vs captured baseline: XGBoostModel
      / BatchTransform / DummyTraining MATCH; PyTorchModel hits the same offline pytorch-2.1.2 SDK
      error on BOTH old and new paths (structural match — the known offline limitation).
      **Compute factory residue is now 0** across all 45 builders
      (`_create_processor`/`_get_processor`/`_create_estimator`/`_create_model`/`_create_transformer`/
      `_get_image_uri` = 0). 3094 passed. (DummyTraining still overrides `_get_inputs`/`_get_outputs`
      — a separate, non-compute residue.)
