# Task: Phase S3 C-SDK — Migrate the 5 SAIS-SDK-bound Builders to TemplateStepBuilder Shells

> **Owner: SAIS-equipped agent (Kiro).** This is the C-SDK tail of the Strategy + Facade builder
> migration (see [2026-06-27_strategy_facade_builder_implementation_plan.md](2026-06-27_strategy_facade_builder_implementation_plan.md),
> Phase S3). **40 of 45 builders are already migrated** to `TemplateStepBuilder` shells on
> `origin/mainline`. The remaining **5 cannot be done outside the SAIS sandbox** because they
> `import` the SAIS SDK / `mods_workflow_core` at module level, so they cannot be imported, run,
> or byte-diff-validated in a SDK-less environment. They are otherwise fully analyzed — this doc
> gives the exact per-builder recipe so the work is mechanical + gated.

## Why these 5 were deferred

Each imports a package absent locally, so the cursus test env raises `ModuleNotFoundError`:

| Builder file | Imports | sagemaker_step_type | Handler | Assembly |
|---|---|---|---|---|
| `builder_cradle_data_loading_step.py` | `secure_ai_sandbox_workflow_python_sdk.cradle_data_loading.cradle_data_loading_step` | `CradleDataLoading` | `SDKDelegationHandler` | (by type) |
| `builder_redshift_data_loading_step.py` | `secure_ai_sandbox_workflow_python_sdk.redshift_data_loading.redshift_data_loading_step` | `RedshiftDataLoading` | `SDKDelegationHandler` | (by type) |
| `builder_registration_step.py` | `secure_ai_sandbox_workflow_python_sdk.mims_model_registration.mims_model_registration_processing_step` | `MimsModelRegistrationProcessing` | `SDKDelegationHandler` | (by type) |
| `builder_data_uploading_step.py` | `secure_ai_sandbox_workflow_python_sdk.data_uploading.data_uploading_step` | `Processing` | `SDKDelegationHandler` | `STEP_ASSEMBLY="delegation"` |
| `builder_edx_uploading_step.py` | `mods_workflow_core.utils.constants` | `Processing` | `ProcessingHandler` | `code` (default — NOT delegation) |

## Background: how a shell works (the contract you're matching)

A migrated shell is a `TemplateStepBuilder` subclass that declares only `STEP_NAME` (+ optional
`STEP_ASSEMBLY`, `HANDLER_KNOBS`) and keeps the genuinely per-step methods. `__init__` and
`create_step` are inherited:
- `__init__` auto-loads the spec from `STEP_NAME` via `load_step_interface(STEP_NAME,
  job_type=getattr(config,"job_type",None))` and auto-binds the handler from the registry's
  `sagemaker_step_type` (+ `STEP_ASSEMBLY` for Processing).
- `create_step` delegates to `handler.build_step(self, **kwargs)` and makes `_attach_spec`
  non-bypassable.
- Per-step methods the shell keeps win over the handler's generic ones via the `_overrides` MRO
  check (so a deviating `_get_inputs`/`_get_outputs` is honored).

The 4 routing/preset facts are **already in the registry** (`src/cursus/core/base/builder_templates.py`,
`@register_strategy` decorations on `SDKDelegationHandler` ~lines 697-710) — you do NOT re-specify
them:
```
CradleDataLoading              preset_knobs={"input_mode":"none","caching_mode":"force_off_attr","log_output_paths":True}
RedshiftDataLoading            preset_knobs={"input_mode":"none","caching_mode":"force_off_attr","log_output_paths":True}
MimsModelRegistrationProcessing preset_knobs={"input_mode":"mims_ordered","depends_on_ctor":True,"outputs_return_none":True,"append_region":True, ...}
delegation (DataUploading)     preset_knobs={"input_mode":"resolve_s3"}
```

## THE ONE THING THAT REQUIRES THE SAIS ENV: the `sdk_step_class` knob

`SDKDelegationHandler.build_step` requires a **`sdk_step_class`** knob — the actual SAIS SDK `*Step`
class to instantiate (`builder_templates.py:683`, `:778-781` raise `NotImplementedError` without it).
The handler can't import it at registration time (that's the whole reason these are deferred). So
each SDK-delegation shell supplies it in `HANDLER_KNOBS`, importing the class at the shell module's
top (where the SAIS env makes it available):

```python
from secure_ai_sandbox_workflow_python_sdk.cradle_data_loading.cradle_data_loading_step import (
    CradleDataLoadingStep,
)
from ...core.base.builder_templates import TemplateStepBuilder

class CradleDataLoadingStepBuilder(TemplateStepBuilder):
    STEP_NAME = "CradleDataLoading"
    HANDLER_KNOBS = {"sdk_step_class": CradleDataLoadingStep}
    # keep per-step methods below (see per-builder notes)
```

The handler then builds `step = sdk_step_class(**ctor)` with the ctor shape the preset_knobs select
(`builder_templates.py:778-811`). Verify the handler's generated ctor matches the hand-written
builder's `create_step` ctor for your SDK version (it was re-homed from these builders and is gated
by `tests/core/base/test_sdk_delegation_handler.py`, but that test uses a recording fake — you have
the real SDK, so confirm against a real construction).

## Per-builder recipe

For ALL five: change the base class to `TemplateStepBuilder`, delete `__init__` and `create_step`,
keep the listed per-step methods, add `STEP_NAME` (+ `STEP_ASSEMBLY`/`HANDLER_KNOBS` as noted). Do
NOT re-add `validate_configuration` unless listed (the Pydantic config is the validation authority,
FZ 31e1d3e). Mirror an already-migrated reference shell:
`src/cursus/steps/builders/builder_batch_transform_step.py` (1:1 verb) or
`builder_tabular_preprocessing_step.py` (Processing/code).

### 1. `builder_cradle_data_loading_step.py`  → SDKDelegationHandler
- `STEP_NAME = "CradleDataLoading"`
- `HANDLER_KNOBS = {"sdk_step_class": CradleDataLoadingStep}`  (import it at module top)
- Keep: `get_output_location`, `get_step_outputs` (public helpers other code may call — verify
  callers via code search before assuming droppable). `_get_inputs`/`_get_outputs` here are the
  `input_mode="none"` shape (return `([], None)` / `None`) — the handler reproduces them, so they
  can be dropped, BUT keep them as overrides if their behavior deviates (it has Cradle-specific
  output-attribute attachment — check `create_step`'s post-construction `setattr`/output wiring and
  preserve it; if the handler doesn't reproduce that, keep a thin `create_step` override or move the
  output-attribute logic into a kept method).
- ⚠️ Cradle's `create_step` attaches extra output attributes to the returned step
  (`get_output_location`/`get_step_outputs` rely on them). **This is the highest-risk of the five —
  confirm the SDKDelegationHandler path preserves those attributes, or keep that wiring.**

### 2. `builder_redshift_data_loading_step.py`  → SDKDelegationHandler
- `STEP_NAME = "RedshiftDataLoading"`, `HANDLER_KNOBS = {"sdk_step_class": RedshiftDataLoadingStep}`
- Same `input_mode="none"` shape as Cradle. Simplest of the SDK set (no extra public helpers).

### 3. `builder_registration_step.py`  → SDKDelegationHandler (MimsModelRegistrationProcessing)
- `STEP_NAME = "Registration"`, `HANDLER_KNOBS = {"sdk_step_class": MimsModelRegistrationProcessingStep}`
- **Keep `validate_configuration`** — it is the ONE retained validate_configuration in the whole
  codebase (the runtime spec↔contract alignment check that reads `self.spec`/`self.contract`; FZ
  31e1d3e slimmed it to just that). Do not drop it.
- The preset knobs handle: `mims_ordered` inputs (PackagedModel-first ordering), region-suffixed
  step name (`append_region` → `step_name + "-" + region` — the eu-west-1 bug fix; NOT hardcoded
  us-east-1), `depends_on` via ctor, `outputs_return_none`, `performance_metadata_location`
  pass-through. Confirm the handler's `mims_ordered` input list matches the builder's `_get_inputs`
  ordered ProcessingInput list for your SDK.

### 4. `builder_data_uploading_step.py`  → SDKDelegationHandler via `STEP_ASSEMBLY="delegation"`
- `STEP_NAME = "DataUploading"`, **`STEP_ASSEMBLY = "delegation"`** (it's `Processing`-typed, so it
  needs the assembly discriminator to route to SDKDelegation instead of ProcessingHandler),
  `HANDLER_KNOBS = {"sdk_step_class": DataUploadingStep}`
- `input_mode="resolve_s3"`: resolves the `input_data` dependency (or the config
  `input_s3_location` fallback) and passes it as `input_s3_location=` to the SDK step ctor. Confirm.

### 5. `builder_edx_uploading_step.py`  → ProcessingHandler (code) — NOT delegation
- This one is a normal `Processing`/`code` step that merely *imports* `mods_workflow_core`
  (constants), so it's only SDK-bound at import time. It builds a vanilla `ProcessingStep(code=...)`,
  NOT a custom SDK step. Migrate it like the other Processing/code shells:
  - `STEP_NAME = "EdxUploading"`, no `STEP_ASSEMBLY` (default `code`).
  - `HANDLER_KNOBS = {"output_path_token": "<token its _get_outputs uses>", ...}` — derive the token
    from its `_get_outputs` (and `include_job_type_in_path` / `direct_input_keys` from its
    `create_step` like the other code builders). If you keep `_get_inputs`/`_get_outputs` as
    overrides (recommended/conservative), the token knob is belt-and-suspenders.
  - Keep `_create_processor` (it uses `ScriptProcessor` + ECR-from-role-ARN + KMS/network — the
    only common Processing-2A builder that sets KMS/network; preserve verbatim),
    `_get_environment_variables`, `_resolve_script_path`, `_get_inputs`, `_get_outputs`,
    `_get_job_arguments`.

## Gate (run for EACH builder before deleting its `create_step`)

The same gates the 40 local migrations used — now runnable because the SDK imports resolve:
1. `ruff check <file>` — clean (watch for orphaned imports; do NOT `ruff --fix` blindly if the file
   has multi-line `from ... import (...)` blocks — it can strip imports used by kept methods; check
   for `F821` after).
2. The shell imports + constructs, and `builder._handler` is the expected handler type
   (`SDKDelegationHandler` for 4, `ProcessingHandler` for edx).
3. **Real-session byte-diff**: build the step the OLD way (git-stash the shell) and the NEW way
   (shell) with the same config + a real SAIS session, and diff the produced SageMaker/SDK step —
   ctor args, attached `_spec`, output attributes (esp. Cradle's), region-suffixed name, ordered
   inputs (Mims). This is the C-SDK equivalent of the local resolved-edge-graph + parity gate, and
   is the thing only the SAIS env can run.
4. `tests/core/base/test_sdk_delegation_handler.py` stays green (it uses recording fakes — still a
   useful structural check).
5. `tests/registry/test_step_assembly_audit.py` stays green (it classifies the shell forms:
   DataUploading=`delegation`, EdxUploading=`code`, the 3 type-routed ones unaffected).
6. The full suite + discovery (`step_catalog`) finds each shell.

## Reference commits (the local migration pattern)

- 1:1 verb shell: `e658461f` (BatchTransform).
- Processing/code shells: `a2efea77` (6 builders) + `4913b054` (shell-with-overrides incl.
  `make_compute` knob for `_get_processor`).
- Processing/step_args shells: `98945452` (14 builders).
- The `_attach_spec` non-bypassable + `config.job_type` flow-through the shells rely on: `c7b22c47`,
  `9d010de0`.

## Definition of done

- All 5 builders are `TemplateStepBuilder` shells; their `create_step`/`__init__` deleted; the only
  retained `validate_configuration` is Registration's runtime spec-align check.
- Each passes the real-session byte-diff vs its pre-migration output (esp. Cradle output attributes,
  Mims region-suffix + ordered inputs, DataUploading resolve_s3).
- `step_assembly` audit + `test_sdk_delegation_handler` + full suite green; discovery finds all 5.
- Update the plan's Implementation Log + the "Progress: 40/45" line to 45/45.
