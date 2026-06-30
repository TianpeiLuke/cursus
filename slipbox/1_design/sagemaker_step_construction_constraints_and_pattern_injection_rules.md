---
tags:
  - design
  - builder_templates
  - pattern_handler
  - sagemaker
  - step_construction
  - constraints
keywords:
  - step_args XOR object handle
  - ProcessingStep step_args processor
  - PatternHandler injection rules
  - PipelineSession
  - allowlist denylist
  - validate_step_args_input
topics:
  - pattern handler injection
  - sagemaker step constraints
  - factory shell construction
language: python
date of note: 2026-06-29
---

# SageMaker Step-Construction Constraints & cursus PatternHandler Injection Rules

**Status:** canonical reference. Read this before adding or changing any `PatternHandler` in `src/cursus/core/base/builder_templates.py`.

**Verified against:** SageMaker Python SDK v2.251.x (`sagemaker.workflow.steps`, `sagemaker.workflow.step_collections`, `sagemaker.workflow.pipeline_context`) and AWS docs *build-and-manage-steps.html*.

## Purpose

Every cursus `PatternHandler.build_step()` ends by calling a real SageMaker `Step` constructor. Those constructors enforce hard mutual-exclusion and provenance invariants in `__init__`. Unit tests in cursus mock the step classes, so violations pass CI and only surface at real pipeline execution in SAIS. This note states the SDK rules once, maps each cursus handler to the step type it builds, and gives an explicit allowlist / denylist of the kwarg combinations a handler may pass — so the next handler author does not reintroduce the bug we already hit.

## The Rule (universal step_args-XOR-object)

Every SageMaker pipeline step constructor accepts EITHER a pre-deferred `step_args` bundle OR a live compute object handle, and enforces an exclusive-or: provide EXACTLY ONE of `{step_args, <object-handle>}` — never both, never neither.

A handler MUST pass `step_args=` alone, or `<object-handle>=` alone. Passing both raises a `ValueError` at construction time; passing neither also raises.

## Per-Step-Type Constraint Table

| Step type | Object handle (the XOR partner of `step_args`) | `step_args` provenance call (deferred form) | Extra constraints | cursus handler | cursus mode |
|---|---|---|---|---|---|
| `ProcessingStep` | `processor` (`sagemaker.processing.Processor`) | `processor.run(...)` | `code` must be a valid S3 URI or local path, NOT a pipeline variable (raises if `is_pipeline_variable(code)`). In object form, `job_arguments=` goes to the step; in step_args form, arguments must go to `processor.run(arguments=...)`. `run(code=...)` validates the file exists on disk at construction. | `ProcessingHandler` | BOTH 2A (`code=`) and 2B (`step_args`) — the only either-mode handler |
| `TrainingStep` | `estimator` (`EstimatorBase`) | `estimator.fit(...)` | Profiler conflict warning if profiling enabled on the estimator while `step_args` also carries a `ProfilerConfig`. | `TrainingHandler` | object form only (`estimator=`) |
| `TransformStep` | `transformer` (`Transformer`) | `transformer.transform(...)` | If `transformer` is given, `inputs` can't be `None` ("Inputs can't be None when transformer is given."). | `TransformHandler` | object form only (`transformer=` + `inputs=`) |
| `CreateModelStep` | `model` (`Model` or `PipelineModel`) | `model.create(...)` (returns a dict) | Deprecated by the SDK in favor of `ModelStep` + `model.create()`. XOR uses `step_args is None ^ model is None`. CreateModelStep does NOT run `validate_step_args_input` — there is no step_args-provenance `ValueError` for it; the only CreateModelStep XOR error is the mutual-exclusion one. | `ModelCreationHandler` | object form only (`model=`) |
| `TuningStep` | `tuner` (`HyperparameterTuner`) | `tuner.fit(...)` | — | (none today) | n/a — no cursus handler yet |

Object-form construction for Processing/Training/Transform/CreateModel is DEPRECATED by the SDK (`DeprecationWarning: We are deprecating the instantiation of <Step> using "<object>". Instead, simply using "step_args".`). cursus currently relies on it for Training/Transform/CreateModel; treat migration to `step_args` as a future task, not a blocker.

## The Bug We Hit

**Where:** `ProcessingHandler.build_step` in `src/cursus/core/base/builder_templates.py`.

**What happened:** `processor=processor` lived in a shared `common` dict consumed by BOTH assembly branches. The 2B (`use_step_args`) branch called `ProcessingStep(step_args=step_args, processor=processor, ...)` — passing BOTH sides of the XOR. SageMaker raised:

> `ValueError: either step_args or processor need to be given, but not both.`

**Why CI missed it:** the unit tests mock `ProcessingStep`, so `__init__` never ran the XOR check. It failed only at real pipeline execution in SAIS (FZ `31e1d3j2`).

**The fix (commit `47bfc035`, AmazonCursus mainline):** `processor` was removed from `common`. The `common` dict now holds only `name` / `depends_on` / `cache_config` (lines 405-409). The 2B branch passes `step_args` ONLY — `ProcessingStep(step_args=step_args, **common)` (line 433). The 2A branch passes the processor explicitly — `ProcessingStep(processor=processor, inputs=..., outputs=..., code=..., job_arguments=..., **common)` (lines 436-443). The XOR rationale is documented inline at lines 399-404.

## Injection Allowlist

The complete set of legal kwarg combinations a handler may pass to each constructor. `name` / `depends_on` / `cache_config` are always-legal step metadata and are omitted from each row for brevity (in cursus these are the `common` dict).

- **`ProcessingStep` 2A (object form):** `processor` + `inputs` + `outputs` + `code` + `job_arguments`. NO `step_args`. (`builder_templates.py:436-443`)
- **`ProcessingStep` 2B (step_args form):** `step_args` ONLY (plus the always-legal metadata). NO `processor`, NO `inputs`/`outputs`/`code`/`job_arguments` on the step — those are arguments to `processor.run(code=, inputs=, outputs=, arguments=)`, which produces the `step_args`. (`builder_templates.py:431-433`)
- **`TrainingStep` (object form):** `estimator` + `inputs`. NO `step_args`. (`builder_templates.py:625-631`)
- **`TrainingStep` (step_args form, not used in cursus):** `step_args` from `estimator.fit(...)` ONLY.
- **`TransformStep` (object form):** `transformer` + `inputs` (required, must be non-None). NO `step_args`. (`builder_templates.py:870-873`)
- **`TransformStep` (step_args form, not used in cursus):** `step_args` from `transformer.transform(...)` ONLY.
- **`CreateModelStep` (object form):** `model` ONLY. NO `step_args`. (`builder_templates.py:719-723`; the `model=` kwarg is on `:721`)
- **`CreateModelStep` (step_args form, not used in cursus):** `step_args` from `model.create(...)` ONLY.
- **`TuningStep` (step_args form):** `step_args` from `tuner.fit(...)` ONLY. **(object form):** `tuner` ONLY.

## Injection Denylist

Each illegal combination with the exact SDK `ValueError` it raises at `__init__`.

- **`ProcessingStep(step_args=..., processor=...)`** → `either step_args or processor need to be given, but not both.` (this is the bug above.)
- **`ProcessingStep()` with neither `step_args` nor `processor`** → `either step_args or processor need to be given, but not both.`
- **`ProcessingStep` 2B passing `inputs`/`outputs`/`code`/`job_arguments` alongside `step_args`** → those args are redundant/conflicting; the step ignores or rejects them — route them through `processor.run(...)` instead.
- **`ProcessingStep(code=<pipeline variable>)`** → `code argument has to be a valid S3 URI or local file path rather than a pipeline variable` (`sagemaker.workflow.steps.ProcessingStep.__init__`, guarded by `if is_pipeline_variable(code)`).
- **`ProcessingStep(step_args=<not from processor.run()>)`** → `The step_args of ProcessingStep must be obtained from processor.run().` (SDK `validate_step_args_input`).
- **`TrainingStep(step_args=..., estimator=...)`** → `Either step_args or estimator need to be given.`
- **`TrainingStep()` with neither** → `Either step_args or estimator need to be given.`
- **`TrainingStep(step_args=<not from estimator.fit()>)`** → `The step_args of TrainingStep must be obtained from estimator.fit().`
- **`TransformStep(step_args=..., transformer=...)`** → `either step_args or transformer need to be given, but not both.`
- **`TransformStep(transformer=..., inputs=None)`** → `Inputs can't be None when transformer is given.`
- **`TransformStep(step_args=<not from transformer.transform()>)`** → `The step_args of TransformStep must be obtained from transformer.transform().`
- **`CreateModelStep(step_args=..., model=...)`** → `step_args and model are mutually exclusive. Either of them should be provided.`
- **`CreateModelStep()` with neither** → `step_args and model are mutually exclusive. Either of them should be provided.`
- **`TuningStep(step_args=..., tuner=...)`** → `either step_args or tuner need to be given, but not both.`
- **`TuningStep(step_args=<not from tuner.fit()>)`** → `The step_args of TuningStep must be obtained from tuner.fit().`

`SDKDelegationHandler` (`builder_templates.py:1052`) is outside this denylist: it constructs a SAIS `MODSPredefined*Step` directly (`step = SDKStepClass(name, role, session, ...)`), not a vanilla SageMaker `Step`, so the `step_args`/object XOR does not apply to it.

## PipelineSession Requirement (step_args mode only)

For `step_args` to DEFER (return arguments instead of starting a job), the compute object's `sagemaker_session` MUST be a `sagemaker.workflow.pipeline_context.PipelineSession`.

On a `PipelineSession`, `processor.run()` / `estimator.fit()` / `transformer.transform()` / `tuner.fit()` do NOT start a job — they emit a warning (`Running within a PipelineSession, there will be No Wait, No Logs, and No Job being started.`) and return a `_StepArguments` bundle suitable for `step_args=`.

On a regular `Session`, those same calls actually EXECUTE the job, reaching AWS — which fails offline or in the wrong environment, and is never what a pipeline build wants.

Therefore: any handler (or handler mode) that constructs `step_args` MUST build its compute object on a `PipelineSession`. cursus's object-form handlers (Training/Transform/CreateModel) do not call the deferred `.run()`/`.fit()`/`.transform()`, so they are exempt today — but the moment a handler is migrated to `step_args`, the `PipelineSession` requirement applies.

## Author Checklist

When adding or altering a `PatternHandler`, verify ALL of the following before merge:

1. The `build_step` path passes EXACTLY ONE of `{step_args, <object-handle>}` to the constructor — confirm against the table above for your step type.
2. No XOR partner ever lands in a shared dict (`common`-style) consumed by more than one assembly branch. Add the object handle / `step_args` per-branch, never shared. (This is precisely what the bug violated.)
3. If you use `step_args`, it is produced by the matching deferred call (`processor.run` / `estimator.fit` / `transformer.transform` / `tuner.fit` / `model.create`), and the compute object's session is a `PipelineSession`.
4. If you use object form, you satisfy the per-step extra constraints (Transform: non-None `inputs`; Processing: `code` is a literal path/S3 URI, not a pipeline variable; arguments routed to the right place).
5. You accept and, where appropriate, suppress/log the object-form `DeprecationWarning` rather than treating it as the long-term contract.
6. You added/updated a conformance test (below) that drives your handler through the REAL constructor.

## Recommended Conformance Test

Add a parametrized test that drives `build_step` through the REAL SageMaker step constructor — NOT a mock — for every `(handler, mode)` pair: `ProcessingHandler` 2A, `ProcessingHandler` 2B, `TrainingHandler`, `TransformHandler`, `ModelCreationHandler` (and `TuningHandler` if added). Build the compute objects on a `PipelineSession` so deferred `.run()`/`.fit()`/`.transform()` calls return `step_args` instead of starting jobs, and assert the returned object is the expected `Step` subclass. This is exactly the test that would have caught FZ `31e1d3j2`: the XOR check lives in `__init__`, so only an unmocked constructor exercises it. Keep the existing mocked unit tests for fast logic coverage; the conformance test is the guardrail for the SDK contract.

Note when grepping the SDK to verify the quoted messages: some are stored as two concatenated string literals, so a single-line grep for the full phrase will miss them. The TransformStep provenance message is one such case — the source holds `"The step_args of TransformStep must be obtained "` + `"from transformer.transform()."`, and the exact runtime string is `The step_args of TransformStep must be obtained from transformer.transform().`.
