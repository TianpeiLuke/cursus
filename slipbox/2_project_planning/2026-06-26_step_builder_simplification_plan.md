---
tags:
  - project
  - planning
  - simplification_plan
  - step_builder
  - registry
  - template_builder
  - sagemaker_step_type_routing
keywords:
  - TemplateStepBuilder
  - sagemaker_step_type routing
  - resolve_handler
  - PatternHandler
  - registry fold-in step.yaml
  - build_registry_from_interfaces
  - STEP_BUILDER_BASE_NAMES discovery gate
  - step_assembly audit
  - phased dual-mode migration
  - get_step_names contract
topics:
  - step builder simplification plan
  - registry-to-interface fold-in
  - sagemaker_step_type construction routing
  - phased dual-mode builder migration
language: python
date of note: 2026-06-26
---
# Step Builder Simplification Plan

> **Status (2026-06-28): COMPLETE except the real-session e2e.** The final `step_names.yaml` drop
> SHIPPED (commit below) — the `.step.yaml` `registry:` blocks are now the sole registry source; the
> parity oracle is a golden snapshot. Only the real-session integration e2e byte-diff remains (needs a
> SAIS `PipelineSession`; all offline gates green).
> Phases 0–1 (discovery gate, registry loader + source-swap to interface-derived), Phase 0c (facade +
> 5 handlers + `resolve_handler`), and Phase 2..N (**all 45 builders are now `TemplateStepBuilder`
> shells; 0 define `create_step`**) all shipped to origin/mainline. The batched builder→shell
> migration + the post-migration residue collapse (env/job-args/inputs/outputs/compute) is owned by
> the [Strategy + Facade plan](2026-06-27_strategy_facade_builder_implementation_plan.md), whose
> **Phase S3 gate** (resolved-edge-graph snapshot + constituent byte-diff + the FZ-31e1d3d pre-S3
> prerequisites) **supersedes** this plan's Phase-2 "Per-batch gate".
>
> **What remains (by design):** only the **real-session e2e** byte-diff (needs a real
> `PipelineSession`; the constituent + edge-graph gates are green). The Final-Phase
> `step_names.yaml` drop is DONE (see the DoD + the Implementation Log entry below).
>
> Doc corrections vs shipped reality: the grammar shipped as **5 handler classes** (Processing 2A/2B
> fold into one `ProcessingHandler` — NI-1); `StepBuilderBase` now has **3** abstract methods
> (`validate_configuration` made a concrete no-op default under FZ 31e1d3e); and the Phase-1
> "delete `_create_step_builder_map`" item was **RETRACTED** (Implementation Log 2026-06-27: it has 2
> load-bearing call sites, only the single `pipeline_template_base.py` call was dead-and-removed).

## Overview

This document is the concrete implementation plan to collapse three currently-separate
things — the standalone `registry/step_names.yaml` table, and the **44 hand-written
`<Name>StepBuilder` classes** (each re-implementing the same constructor and `create_step`
boilerplate) — into **one per-step `.step.yaml`** (with a `registry:` block) plus **one
routed `TemplateStepBuilder` facade and six construction-verb handlers**.

The plan is engineered so the **compile→assemble→discover→build orchestration spine never
changes**: `dag_compiler`, `dynamic_template`, `pipeline_template_base`, and
`pipeline_assembler` are consumers of contracts this plan *preserves* — they traffic only
in `{CanonicalName → class}` maps and the fixed `create_step(inputs=, outputs=,
dependencies=, enable_caching=) -> Step` invariant. Risk is concentrated in **two Phase-0
prerequisites**, which is why they ship first, in isolation, with zero behavior change.

The design rationale (the *why*) lives in the slipbox analysis trail (FZ 31e1c..31e1f).
This document is the *what to touch, in what order, and what breaks if you skip a step* —
all citations are source-verified against the current `src/cursus`.

### End-state goals

- **#1** — Fold `step_names.yaml` into per-step `.step.yaml` `registry:` blocks, loaded via a new `build_registry_from_interfaces()`.
- **#2** — Replace the 44 hand-written `<Name>StepBuilder` classes with thin `TemplateStepBuilder` shells.
- **#3** — Route construction by `sagemaker_step_type` (NEVER by step name) inside `TemplateStepBuilder.__init__` via `resolve_handler(sagemaker_step_type, step_assembly)`.
- **#4** — A `SDKDelegationHandler` for the predefined/MODS steps (Cradle, Redshift, Mims, DataUploading) that take no spec-driven inputs.
- **#5** — Re-home the per-pattern `_get_inputs`/`_get_outputs` spec×contract message passing from the 44 builders into 6 handlers.

### Implementation Log — Discovered Issues

Findings made while executing the plan (newest first); each updates the plan above where it matters.

- **2026-06-28 — FINAL PHASE DONE: `step_names.yaml` DELETED; the `.step.yaml` `registry:` blocks are
  the sole registry source.** `step_names_base.STEP_NAMES = build_registry_from_interfaces()` with NO
  fallback (was `fallback=yaml_table`); removed the yaml file + `_load_step_names()`. The
  `interface_registry_loader` already derived 48/48 rows dict-identically from the interfaces alone
  (proven by `test_interface_registry_is_self_sufficient_without_fallback`), so the flip is data-neutral.
  **Oracle → golden snapshot:** `test_registry_interface_parity.py` now asserts the derived table equals
  a frozen `step_names_registry_snapshot.json` (re-baseline `CURSUS_UPDATE_REGISTRY_SNAPSHOT=1`) instead
  of comparing to the deleted yaml — drift still fails loudly. **Loud-fail preserved:**
  `build_registry_from_interfaces` raises if a `.step.yaml` lacks `registry.sagemaker_step_type` (no
  silent step drop — the failure-mode-#2 guard). `test_step_names_yaml.py` retitled to test the
  interface-derived `STEP_NAMES` (+ a `test_step_names_yaml_is_gone` regression guard). Verified: 48 rows
  intact, hybrid manager + `get_step_names()`/`get_config_step_registry()` facades unchanged, 2242
  passed, ruff clean. pyproject artifact comment updated (the `.step.yaml` files now carry the registry).
  **The "one invariant that MUST survive" held:** `get_step_names()` rows still carry
  `sagemaker_step_type` (+ config_class/builder_step_name). This is the point-of-no-return step — done.

- **2026-06-27 — Phase 2 PILOT DONE: TabularPreprocessing migrated to the shell form (−97 LOC, zero behavior change).**
  `builder_tabular_preprocessing_step.py` now extends `TemplateStepBuilder` and declares
  `STEP_NAME = "TabularPreprocessing"` + `HANDLER_KNOBS = {output_path_token, direct_input_keys}`.
  **Deleted** the duplicated `__init__` (spec now auto-loaded from STEP_NAME) and `create_step`
  (inherited; routes through `ProcessingHandler.build_step`). **Kept** the deviating per-step methods
  (`validate_configuration`, `_create_processor`, `_get_environment_variables`, `_get_inputs`,
  `_get_outputs`, `_get_job_arguments`) — the handler calls them via the `_overrides` MRO check.
  367→270 LOC (−97). **Verified:** the migration parity gate (6 tests) passes against the migrated
  builder; it's still a `StepBuilderBase` subclass; `StepCatalog.load_builder_class("TabularPreprocessing")`
  still discovers it (Phase-0a gate working); ruff-clean; 1425 tests pass (the only failures are
  pre-existing `tabular_preprocessing.py` *script* helper tests — `read_shard_wrapper` — confirmed
  identical with the migration stashed, unrelated to the builder).
  - **Facade enhancements this required (committed with the pilot):** `TemplateStepBuilder.__init__`
    now auto-loads the spec (`load_step_interface(STEP_NAME)`) and auto-binds the handler from the
    registry's `sagemaker_step_type` (`_auto_bind_handler` via `get_sagemaker_step_type`) + declared
    knobs; `_merge_inputs` reads `direct_input_keys` from the handler knobs. The audit test
    (`test_step_assembly_audit`) now recognizes the migrated shell form (verb from `STEP_ASSEMBLY`/
    knobs, default `code`, since there's no `create_step` body to scan).
  - **The proven migration recipe** (per builder, in its batch): (1) extend `TemplateStepBuilder`,
    add `STEP_NAME` + `HANDLER_KNOBS` (output_path_token, direct_input_keys, split_source_dir,
    use_step_args as needed); (2) delete `__init__` + `create_step`; (3) keep every deviating
    per-step method; (4) fix imports; (5) green the parity gate + discovery + ruff before commit.

- **2026-06-27 — `source_dir` is a PER-BUILDER knob within the 2B family (verified, user-flagged).**
  Read the actual `processor.run()` calls. The 2B `step_args` family splits into TWO sub-modes —
  and which one a builder uses is **not predictable from the verb**, so `split_source_dir` must be
  set per-builder:
  - **`run(code=script_path)` — full path, NO source_dir:** XGBoostModelEval (`:345`),
    XGBoostModelInference, LightGBMModelEval (`:354`), LightGBMModelInference.
  - **`run(code=entry_point, source_dir=source_dir)` — SPLIT (FrameworkProcessor.run supports it,
    enables local imports):** LightGBMMTModelEval/Inference, PyTorchModelEval/Inference (verified the
    explicit `source_dir=source_dir  # FrameworkProcessor.run() supports this`), RiskTableMapping,
    TokenizerTraining, **DummyTraining**, PercentileModelCalibration.
  - **DummyDataLoading is 2A** (`ProcessingStep(code=...)`, no run, no source_dir).
  **Consequence:** the byte-diff gate MUST cover the split-mode steps specifically — getting
  `split_source_dir` wrong silently breaks scripts with local imports (the container receives a
  single file instead of the dir). The LightGBM-eval-vs-LightGBMMT-eval divergence (same family,
  different mode) is the canonical trap. This refines the §2e "5 source_dir modes" — within 2B the
  split is a per-builder flag, confirmed against `run()` call sites.

- **2026-06-27 — Careful per-builder reading sharpens the SMOOTH-TRANSITION shell design (Processing-2A).**
  Read TabularPreprocessing, CurrencyConversion, Package `create_step` IN FULL (not just the audit
  summary). Finding: **the 2A `create_step` body is 100% mechanical/identical** across them — the
  same `inputs={}` merge (extract-from-deps → inputs_raw → `direct_input_keys`), `processor =
  _create_processor()`, `_get_inputs`/`_get_outputs`/`_get_job_arguments`, `code=get_script_path()`,
  `ProcessingStep(...)`, `setattr(_spec)`. The ONLY per-step deltas in `create_step` are: (1) the
  `direct_input_keys` list (`["DATA","METADATA","SIGNATURE"]` vs `["data_input"]` vs none), and (2)
  the `setattr` guard style (unconditional vs `hasattr`-guarded). **Audit CORRECTION:** Package's
  "always-inject `inference_scripts_input` override" is in **`_get_inputs`, NOT `create_step`** (the
  audit said create_step). So the smooth-transition shell is cleaner than feared: `create_step`
  collapses entirely into the handler; ALL per-step variation lives in the 5 overridable methods
  (`_get_inputs`/`_get_outputs`/`_get_environment_variables`/`_get_job_arguments`/`_create_processor`)
  + 2 trivial knobs. **Smooth-transition shell form (non-destructive first):** keep the per-step
  methods on the shell as `make_compute`/env/inputs hooks the handler calls — delete only the
  duplicated `__init__` + `create_step`. This is safer than deleting the I/O methods wholesale: a
  builder becomes `class XStepBuilder(TemplateStepBuilder)` with `STEP_NAME`, `PATTERN`/knobs, and
  retains its `_create_processor`/`_get_environment_variables`/`_get_inputs`-override only where it
  deviates from the handler default. The byte-diff gate (constituent-level, session-free) validates
  each before its `create_step` is removed.
- **2026-06-27 — Validation reality: full ProcessingStep/processor construction needs a real
  SageMaker session** (Mock trips `.arguments` jsonschema validation, both paths). So the unit gate
  proves session-INDEPENDENT parity (inputs/outputs/job-args/code/env); processor + full-step
  byte-diff are the integration e2e item. This means migration must keep `_create_processor`
  reachable as a hook (it can't be replaced by a session-free generic) — reinforcing the
  non-destructive shell form above.

- **2026-06-27 — Pre-destruction SAFETY RECORD written (gates all of Phase 2).**
  [`slipbox/4_analysis/2026-06-27_step_builder_special_treatment_preservation.md`](../4_analysis/2026-06-27_step_builder_special_treatment_preservation.md)
  documents every per-builder special treatment across all 44 builders with file:line — config→env
  (4 override shapes + coercion catalog incl. the `Join`-valued BedrockBatch env vars), input path
  construction (FullyReplicated, direct_input_keys allowlists, ship-in-source-dir skips, Registration's
  ordered MIMS list, DummyDataLoading's hardcoded channel), output path construction (the per-step
  token + job_type-in-path table — note TemporalSplit OMITS job_type), create_step per-verb shapes,
  the **5 source_dir/path-resolution modes**, and job arguments (`--job_type` vs `--job-type` hyphen,
  latent Training hooks). Plus a **52-entry migration risk register** flagging MUST-PRESERVE items and
  their absorption mechanism. All 3 high-stakes claims verifier-confirmed. **This is the gate: no
  builder body is deleted until its register rows are accounted for.** Known bugs to carry forward
  unchanged: hardcoded `us-east-1` (R36), PseudoLabelMerge NameError (R48).

- **2026-06-27 — Phase 2 GATE PROVEN: ProcessingHandler is byte-faithful to the hand-written builder.**
  Built the migration parity harness (`tests/core/base/test_builder_migration_parity.py`, plan item
  31e1d1b) and ran it on the Processing-2A pilot (TabularPreprocessing): the `ProcessingHandler`'s
  re-homed `get_inputs`/`get_outputs` produce `ProcessingInput`/`ProcessingOutput` lists
  **structurally identical** to `TabularPreprocessingStepBuilder._get_inputs`/`_get_outputs` — for
  provided inputs, the generated `Join` output destination, an explicit destination, and the
  required-missing raise. **This validates the re-homing approach before any builder body is
  deleted.** Two harness lessons: (1) the comparison must use `.expr` for SageMaker Pipeline
  variables (`Join` raises on `__str__`); (2) `TabularPreprocessingConfig` needs `source_dir` +
  `processing_entry_point` to construct. **Migration sequencing decision:** the actual
  shell-conversion (deleting builder bodies) is deferred to deliberate per-builder/-batch commits,
  each gated by extending this harness to the builder being migrated — the harness ships first as
  the safety net. `make_compute` (the per-step processor factory) still needs re-homing before a
  full `create_step` (not just get_inputs/get_outputs) can route — that is the next concrete
  Batch-A sub-task.

- **2026-06-27 — Phase 1 cleanup CORRECTED: do NOT delete `_create_step_builder_map`.** The plan
  said the method is "already dead — delete it." **Source review says that's wrong and dangerous:**
  `_create_step_builder_map` has **3 call sites**, and 2 are load-bearing — `dynamic_template.py:305`
  passes its result to `validate_dag_compatibility(builder_registry=...)`, and `dag_compiler.py:336`
  uses it for validation + error reporting. Only the **single** call at
  `pipeline_template_base.py:305` is dead (the result is computed then never passed to
  `PipelineAssembler` — no `step_builder_map=` kwarg). **Safe cleanup = remove only that one dead
  line** (and its comment mention), keeping the abstract method and the two real call sites intact.
  (Lesson: "dead code" claims from the design pass must be re-verified against all call sites before
  deletion — exactly the careful-validation discipline this work requires.)

- **2026-06-27 — Phase 1 (step 2) DONE: registry SOURCE SWAPPED to interface-derived.**
  `step_names_base.STEP_NAMES` now comes from `build_registry_from_interfaces(fallback=<yaml>)`
  (the package-wide live registry path), with `step_names.yaml` retained as the parity oracle +
  a try/except safety fallback (logs + reverts to yaml if interface derivation ever fails).
  Lazy import of the loader keeps `step_names_base` a dependency-free leaf (no circular import).
  **Test-oracle fix this forced:** `test_registry_interface_parity.py` previously used the
  module-level `STEP_NAMES` as its oracle — now that `STEP_NAMES` is *itself* loader-derived, that
  would compare the loader to itself (vacuously true). Repointed the oracle to the RAW yaml via
  `_load_step_names()` so parity stays meaningful. **Verified:** `STEP_NAMES` is interface-derived
  AND still `== ` the raw yaml (48 rows); **1483 tests green** across registry + step_catalog +
  core/base + core/compiler + core/assembler + CLI; live `get_step_names()` (48, all carry
  `sagemaker_step_type`), `get_config_step_registry()` (48), `get_builder_map()` (44) all unchanged;
  ruff-clean. This is the load-bearing flip — the package now reads its registry from the interface
  files. Fully reversible (revert one function). `step_names.yaml` is NOT yet dropped (that is the
  final phase, after the builder migration).

- **2026-06-26 — Phase 1 (step 1 of N) DONE: `registry:` blocks added to all 45 `.step.yaml`.**
  Inserted a `registry:` block (`sagemaker_step_type` + `description`, sourced from
  `step_names.yaml`, matched by `step_type`) right after `node_type:` in every interface file —
  purely additive (45 files, +135 lines; `git diff` confirms only `+` lines). **Verified:** all 45
  still parse; `build_registry_from_interfaces()` **with NO fallback** now reconstructs the table
  dict-equal to `step_names.yaml` (the self-sufficiency gate — strengthened the parity test with
  `test_interface_registry_is_self_sufficient_without_fallback`); `StepInterface.from_yaml`
  tolerates the new top-level key (it reads `contract`/`spec`, ignores `registry:`); the
  `get_sagemaker_step_type*` + `StepInfo` consumer families stay green (54 tests). This is the
  prerequisite that lets the registry source be repointed off `step_names.yaml` (next Phase-1
  steps) and eventually dropped — the interface files alone now carry the registry data.

- **2026-06-26 — Phase 0c DONE (landed not-yet-wired).** Shipped `core/base/builder_templates.py`:
  the `TemplateStepBuilder` facade (a `StepBuilderBase` subclass — verified via MRO, no circular
  import; keeps the 5-kwarg ctor; delegates `_get_inputs`/`_get_outputs`/`create_step` to a bound
  handler), the `PatternHandler` ABC, `resolve_handler(sagemaker_step_type, step_assembly)` with
  the routing table, and a **fully-implemented `ProcessingHandler`** (re-homes the spec×contract
  `get_inputs`/`get_outputs` joins verbatim from the builders; one class covers both 2A `code=` and
  2B `step_args` via `use_step_args`/`split_source_dir` knobs — NI-1). The Training/ModelCreation/
  Transform/SDKDelegation handlers are **scoped stubs that raise `NotImplementedError`** — they are
  completed in their Phase-2 byte-diff-gated batch (implementing them now, untested against real
  construction, would be guessing). `ProcessingHandler.build_step` requires a `make_compute` knob
  (the per-step processor factory) — that per-step wiring is the remaining Batch-A/C work.
  - **Decision recorded:** the facade does NOT sequence make_compute/inputs/outputs — `build_step`
    owns ordering (so Transform's "make_compute last" NI-3 and ModelCreation's caching-drop NI-4 are
    expressible). `resolve_handler` routes by `sagemaker_step_type` only (test asserts `DummyTraining`
    → Processing, never Training); `Base`/`Lambda` raise `NoBuilderError`.
  - 19 new tests (`tests/core/base/test_builder_templates.py`); module + test ruff-clean; **995
    tests across core/base + core/assembler + step_catalog stay green.**

- **2026-06-26 — Phase 0b DONE + an audit correction the executable test caught.** Shipped
  `registry/interface_registry_loader.py` (`build_registry_from_interfaces()`, reads
  `sagemaker_step_type`/`description` from a `.step.yaml` `registry:` block when present, else
  falls back to the current `step_names.yaml` row — a faithful drop-in today) + the CI parity
  oracle `tests/registry/test_registry_interface_parity.py` (4 tests: dict-equal to
  `step_names.yaml` across all 48 rows × 5 fields; every row carries `sagemaker_step_type`;
  `spec_type == step_name` for all; only `BatchTransform` breaks the config-class convention) +
  the enshrined `tests/registry/test_step_assembly_audit.py` (derives each verb from source).
  **Correction:** writing the audit as a source-deriving test caught that the workflow audit
  **mis-tagged `ModelMetricsComputation` and `ModelWikiGenerator` as `step_args`** — source
  shows both build `ProcessingStep(processor=, code=script_path)` (`:420/:425` and `:440/:445`),
  i.e. they are **`code`**. **Corrected split: 18 `code` + 16 `step_args` + 1 `delegation` = 35**
  (was 16/18/1). This is the value the Batch-C migration must use. (Lesson reinforced: enshrine
  the audit as an executable test, never trust the analysis output as ground truth.)

- **2026-06-26 — Phase 0b/0c analysis complete (workflow + 3/3 verified): `step_assembly` table locked + 7 handler-design findings.** The full verified `step_assembly` audit of the **35 concrete** Processing builders: **16 `code` + 18 `step_args` + 1 `delegation` (DataUploading)**. Verified traps: `DummyTraining` is `step_args` (NOT the `code` default); only `DataUploading` is delegation among Processing-typed rows. Seven design findings that update the handler plan:
  - **NI-1 — collapse Handlers A+B → one `ProcessingHandler`.** Processing-2A and Processing-2B share `get_inputs`/`get_outputs` *verbatim*; they differ only in `build_step` (`code=` vs `processor.run()→step_args=`) and a `split_source_dir` knob. So implement **one** `ProcessingHandler` with knobs `use_step_args: bool` + `split_source_dir: bool` → effectively **5 handler implementations, not 6**. (The 6 *verbs* stay; two map to one class.)
  - **NI-2 — `EdxUploading` is `code`+`sdk_dependent`.** It is 2A-assembled (uses `ScriptProcessor`, `code=`) yet imports `mods_workflow_core` at module level. So "sdk_dependent ⟺ delegation" is FALSE. Action: add `ScriptProcessor` to the Processing handler's `processor_class` knob, and make the **deferred-SDK-import guard apply to ANY `sdk_dependent` step regardless of verb** (not just delegation). The 5 sdk_dependent rows: EdxUploading (code), DataUploading/Registration/Cradle/Redshift (delegation).
  - **NI-3 — Transform breaks two facade invariants.** `get_inputs` returns a **2-tuple** `(TransformInput, model_name)`, and `make_compute` consumes both get_* results and runs **LAST**. So the facade must NOT impose a fixed make→inputs→outputs→build sequence — it hands merged inputs/outputs to `build_step` and lets the **handler orchestrate ordering**.
  - **NI-4 — ModelCreation INVERTS caching** (warns on `enable_caching=True`, passes no `cache_config`). Caching is therefore a per-verb `build_step` decision, never a facade-level rule: Processing/Training/Transform pass it; ModelCreation drops it; Cradle force-sets `False`; DataUploading/Registration ignore it.
  - **NI-5 — method-name drift + latent hooks.** Normalize `_get_processor`/`_create_processor` → one `make_compute` entrypoint. Preserve (do NOT "fix") two currently-unwired hooks: Training's `_get_job_arguments` and PyTorch's `_get_metric_definitions`/`_create_profiler_config` — wiring them is a behavior change outside Phase-0 scope.
  - **NI-6 — count correction: 35 concrete Processing builders, not 36** (the 36th `step_names.yaml` Processing row is the abstract `ProcessingStepBuilder` base — no builder file). All migration-list/dispatch artifacts use 35 concrete + 8 non-Processing rows.
  - **NI-7 — Registration's ordered `get_inputs` is a MODE of the SDKDelegation handler**, not a reach into the Processing handler: `SDKDelegationHandler.get_inputs` has 3 modes — empty (Cradle), resolve-and-stash (DataUploading), ordered-ProcessingInput-list (Registration, MIMS positional order is load-bearing).
  - **Also confirmed:** there are **no `_create_*` compute factories in `builder_base.py`** — every builder hand-writes its compute method, so handlers must *re-home* them (not call a base hook); `setattr(step, "_spec", self.spec)` is universal post-construction.

- **2026-06-26 — Phase 0a DONE + a discovered environment constraint.** The discovery base-name
  gate was widened (`STEP_BUILDER_BASE_NAMES = {"StepBuilderBase", "TemplateStepBuilder"}` in
  `builder_discovery.py`) with 5 new gate tests; the 44 builders discover byte-identically (107
  adjacent tests stay green; live `get_builder_map()` count unchanged at 44 before/after — the
  strict-superset proof holds). **Discovered:** the 4 MODS-predefined builders (Cradle, Redshift,
  Mims/Registration, DataUploading) **plus EdxUploading** already fail to load *in this local
  environment* — they import the SAIS SDK (`secure_ai_sandbox_workflow_python_sdk` /
  `mods_workflow_core`) at module level, which is not installed locally, so discovery's
  `_load_class_from_file` returns `None` for them (verified **pre-existing** — identical before
  and after the 0a change). **Plan impact:** the Phase-2 Batch-C **byte-diff gate cannot validate
  the SDKDelegation steps in a SAIS-SDK-less environment** — those shells must be verified either
  (a) in an environment with the SAIS SDK installed, or (b) by mocking the SDK step classes. The
  5 SDK-dependent builders should be their own migration sub-batch (C-SDK) with this caveat noted,
  and the local byte-diff harness must skip-with-warning (not silently pass) when the SDK is
  absent — otherwise a SDKDelegation regression hides behind an import error. (This is the same
  "make silent failures loud" discipline as the registry parity oracle.)

### The current builder hierarchy (what we are changing)

All 44 builders are **flat, single-level** children of `StepBuilderBase(ABC)`
(`core/base/builder_base.py:85`) — no intermediates, no mixins, no multiple inheritance.
`StepBuilderBase` declares 3 abstract methods (`_get_inputs` :761, `_get_outputs` :781,
`create_step` :1027) plus a concrete optional `validate_configuration` hook (:687, made a no-op
default under FZ 31e1d3e — was abstract) and ~26 concrete helpers.
The target adds **exactly one level**: `TemplateStepBuilder(StepBuilderBase)`, with the 44
builders becoming 2-line shells beneath it. (The 4 MODS-predefined builders stay flat
children too — their `MODSPredefinedProcessingStep` inheritance is on the SDK *step object*
they construct, not on the cursus builder.)

## The Two Failure Modes This Plan Designs Around

Both **degrade silently** rather than erroring — the sequencing exists to make them impossible:

1. **Invisible-builder outage.** `builder_discovery._inherits_from_step_builder_base`
   (`step_catalog/builder_discovery.py:368`) accepts a class only if a *direct* AST base is
   the literal name `StepBuilderBase` (the comparisons at :379 and :383). The moment a shell
   subclasses `TemplateStepBuilder`, this single-file direct-base check returns `False`, the
   shell vanishes from discovery, `load_builder_class` returns `None` (:218, logged DEBUG, not
   raised), and resolution returns nothing — surfacing far downstream as `RegistryError` in
   `dynamic_template._create_step_builder_map` (`core/compiler/dynamic_template.py:266`) or
   `UNKNOWN` builders in `dag_compiler.preview_resolution` (`core/compiler/dag_compiler.py:440`).
2. **Silent empty-registry.** `get_step_names()` is read at 7+ sites; `StepCatalog` has
   graceful-degradation (`step_catalog/step_catalog.py:1195-1204`) that masks a registry-load
   failure as an empty index. If `build_registry_from_interfaces()` returns a wrong-shaped dict
   (missing the `sagemaker_step_type`/`config_class`/`builder_step_name` keys, or a changed
   shape), consumers don't crash — they resolve nothing.

## Phase 0 — Prerequisites (additive, inert, reversible — land on mainline ahead of any migration)

### 0a. Widen the discovery base-name gate  *(do this first — highest severity, lowest effort)*

**File:** `step_catalog/builder_discovery.py:368` (`_inherits_from_step_builder_base`).

- **Change:** replace the two literal `"StepBuilderBase"` comparisons (:379, :383) with
  membership in a closed module-level constant
  `STEP_BUILDER_BASE_NAMES = {"StepBuilderBase", "TemplateStepBuilder"}`. The function still
  does single-file, direct-base AST matching — only the accepted *name set* widens.
- **Constraint:** `TemplateStepBuilder` must itself inherit `StepBuilderBase`, so the ABC
  remains the hierarchy root and `Type[StepBuilderBase]` still holds at
  `dynamic_template.py:103` and `pipeline_template_base.py:239`.
- **Optional hardening:** add a post-load `issubclass(cls, StepBuilderBase)` check in
  `_load_class_from_file` (:468-529) — the AST gate is a cheap pre-filter; the import-time
  check is the authoritative one and kills name false-positives.
- **Stays:** all scan/extract/import machinery (`_scan_builder_directory` :288,
  `_extract_builder_from_ast` :328, soft-fail `None` :218).
- **Verify:** the existing 44 builders discover **byte-identically** (widening is a strict
  superset → zero behavior change today). Add a discovery test that asserts a
  `TemplateStepBuilder` subclass IS discovered and a non-builder class is NOT.

### 0b. Add `build_registry_from_interfaces()` behind a CI parity oracle

**Files:** new loader (in `registry/` or delegated to from `registry/step_names.py`); CI test.

- **Change:** implement the loader walking each `.step.yaml` `registry:` block plus a ~3-row
  "extras" map (for steps with no interface file — `Base`, `Processing`, `HyperparameterPrep`),
  producing the **exact dict shape** `get_step_names()` returns today: per-step dict carrying
  `sagemaker_step_type`, `config_class`, `builder_step_name` (convention-derived values fine —
  the **keys must remain**). Derivation rules: `config_class = "<Name>Config"`,
  `builder_step_name = "<Name>StepBuilder"`, `spec_type = step_type` (dropped as a stored field —
  it is `== step_type` for all 48 rows), with a small `registry_overrides` tail (only
  `BatchTransform` among concrete steps breaks the `config_class` convention).
- **Parity oracle first:** keep `registry/step_names.yaml` in the tree; add a CI test asserting
  `build_registry_from_interfaces()` is **dict-equal** to the current `step_names.yaml`-derived
  `get_step_names()` for the read keys. Do NOT repoint any consumer yet.
- **`step_assembly` audit (do it here):** AST-scan all 36 `Processing` builders and tag each
  `code` / `step_args` / `delegation`, by inspecting the `create_step` body:
  - `code` → builds `ProcessingStep(..., code=script_path)` (e.g. `builder_tabular_preprocessing_step.py:354`)
  - `step_args` → `processor.run(...)` then `ProcessingStep(step_args=...)` (e.g. `builder_xgboost_model_eval_step.py:345`)
  - `delegation` → returns a non-`ProcessingStep` SDK class (`builder_data_uploading_step.py:140`)

  **Trap:** `DummyTraining` has `sagemaker_step_type: Processing` and uses `processor.run()`
  (`builder_dummy_training_step.py:365`) → `step_assembly: step_args`, NOT the `code` default.
  Getting this wrong silently mis-assembles it. This audit produces the `step_assembly` column
  and drives Phase 2 batching.
- **Stays:** `step_names.yaml` as the CI oracle until parity green; `get_step_names()` /
  `get_config_step_registry()` signatures.

### 0c. Land `TemplateStepBuilder` + 5 handler classes + `resolve_handler` (all unused)

> *As-shipped (NI-1): 6 construction **verbs** but **5 handler classes** — Processing-2A/2B fold
> into one `ProcessingHandler` with `use_step_args`/`split_source_dir` knobs.*

**Files:** new `core/base/` (or `steps/builders/`) module(s) for the facade, handlers, router.

- **`TemplateStepBuilder(StepBuilderBase)`** — must accept the **fixed 5-kwarg ctor**
  (`config, sagemaker_session, role, registry_manager, dependency_resolver`, matching
  `pipeline_assembler.py:190-196`), bind its handler via
  `resolve_handler(sagemaker_step_type, step_assembly)` inside `__init__`, and transparently
  expose `.spec`, `.contract`, `.config`, `_get_base_output_path`, `set_execution_prefix`, and
  `create_step(...) -> Step` delegating to the handler.
- **5 handler classes** (`ProcessingHandler` [covers 2A+2B via `use_step_args`/`split_source_dir`],
  `Training`, `ModelCreation`, `Transform`, `SDKDelegation`), each owning the re-homed
  `_get_inputs`/`_get_outputs` for its verb (#5).
  `SDKDelegationHandler` covers all 4 MODS-predefined steps (#4); "Registration" is
  SDKDelegation with a non-trivial ordered-input `get_inputs`, not a 7th handler.
- **`resolve_handler(sagemaker_step_type, step_assembly)`** — the dispatch table. 6 of the
  step types route to a handler by `sagemaker_step_type` alone (Training, CreateModel,
  Transform, Cradle, Redshift, Mims); `Processing` sub-discriminates by `step_assembly`
  (default `code`); `Base`/`Lambda` raise `NoBuilderError`. **Routes by registry type, NEVER
  by name** (test-enforced; `DummyTraining` routes as Processing).
- **Stays:** nothing wires to these yet — dead code on landing; the 44 builders remain live.
- **Critical test — facade fidelity:** each handler must surface `.spec`/`.contract`, or the
  assembler silently skips it (`_propagate_messages` guard at `pipeline_assembler.py:249`,
  `_generate_outputs` returns `{}` at :357). Unit-test each handler against the assembler's
  spec reads before any shell ships.

## Phase 1 — Repoint the registry source (only after 0b parity is green)

Keep `get_step_names()` / `get_config_step_registry()` / `BUILDER_STEP_NAMES` as **facades**
over the loader so call sites need no edits.

| File | Change | Stays | Risk |
|---|---|---|---|
| `step_catalog/step_catalog.py` | repoint `get_step_names` import (:1209, :1372, :1448) to the loader | `get_builder_for_config` :1504, `get_builder_for_step_type` :1520, `get_builder_map` :1594, `load_builder_class` :550; `sagemaker_step_type` keying in `get_builders_by_step_type` :1752 | silent empty index (:1195-1204) if dict shape wrong |
| `step_catalog/builder_discovery.py` | repoint `get_step_names` (:112, :139, :401) | scan/resolve logic; `step_type` field harmlessly absent (:144) | short/empty map |
| `step_catalog/mapping.py` | keep `get_config_step_registry()` a facade → **zero edits**; else repoint :235, :322 | `LEGACY_ALIASES` verbatim (:31-37); job-type fallback (:106-115); variant suffixing (:250-251) | reconcile `_fallback_config_to_step_type` (:279, incl. `CradleDataLoad→CradleDataLoading` :305) with the loader's convention — it becomes the *primary* path |
| `core/base/builder_base.py` | repoint the `STEP_NAMES` property load paths (:184, :201); `validate_configuration` made a concrete no-op default (FZ 31e1d3e) | the 3 abstract methods, `create_step` (:1027), 5-arg ctor (:270), `_get_environment_variables` (:543), `_get_job_arguments` (:607), `set_execution_prefix` (:640), `_get_base_output_path` (:663) | `_get_step_name` (:345) must stay **label-only** — nothing branches on it for dispatch |

**Subsystems requiring NO edits in Phase 1 (verified, the spine):**

- `core/compiler/dag_compiler.py` — `create_template` (:683), `compile` (:488), caller-hook
  (:216, :254), NVMe shim (:659-675) all unchanged. `preview_resolution` (:434) reads only
  `builder_class.__name__` for display — keeps working; the reported name may collapse toward a
  shared facade name. **Failure surface only.**
- `core/compiler/dynamic_template.py` — `get_builder_map` (:240) and `get_builder_for_config`
  (:256) traffic only in `{Name → class}`; the class being a shell is invisible. **Stays.**
- `core/assembler/pipeline_template_base.py` — `generate_pipeline` (:286) forwards `config_map`
  + `step_catalog`; assembler self-discovers. **Stays.** *Opportunistic cleanup:*
  `_create_step_builder_map` (abstract :239; call :305) is **already dead** — its result is
  computed but never passed to `PipelineAssembler` (no `step_builder_map` param). Deleting it is
  a behavior no-op.
- `core/assembler/pipeline_assembler.py` — **no `step_names`/`sagemaker_step_type` reads** (0
  grep hits). Resolves via `get_builder_for_config` (:145, :180), instantiates with the 5-kwarg
  contract (:190), calls `create_step` (:498, :721). **Two facade-fidelity items to verify (owned
  by 0c):** (a) `_generate_outputs` reads `builder.spec.step_type` (:368) — the *spec object's*
  field, so verify the YAML-loaded spec still populates `.step_type`; (b) a handler omitting
  `.spec` triggers the silent skip (:249/:357) — the 0b audit must confirm those steps need no
  spec. *Note:* the assembler DOES use a `RegistryManager` (:18, :122) — that is the
  **dependency-resolution** registry, NOT the step-names table; it is unaffected.

## Phase 2..N — Convert the 44 builders to shells, batched by handler

> **GATING PREREQUISITE — read first:** [2026-06-27_step_builder_special_treatment_preservation.md](../4_analysis/2026-06-27_step_builder_special_treatment_preservation.md)
> is the pre-destruction safety record. It documents every per-builder special treatment
> (config→env, I/O paths, create_step, source_dir/path-resolution, job args) with file:line and a
> 52-entry **migration risk register** marking each load-bearing item MUST-PRESERVE and its
> absorption mechanism (declarative YAML knob / per-pattern handler method / per-step sidecar).
> Do not collapse a builder until its rows in that register are accounted for. Known bugs to
> carry forward unchanged unless explicitly authorized: hardcoded `us-east-1` (R36), PseudoLabelMerge
> `valid_job_types` NameError (R48).

Replace each hand-written `<Name>StepBuilder` with a 2-line shell:
```python
class TabularPreprocessingStepBuilder(TemplateStepBuilder):
    STEP_NAME = "TabularPreprocessing"
```
Delete its `_get_inputs`/`_get_outputs`/`create_step` body (now in the handler). **Stays:** the
file location (`steps/builders/`), the `<Name>StepBuilder` class name (the discovery key), the
config class. Batch so each is independently verifiable against a held pipeline:

- **Batch A — 1:1 step types** (Transform, Cradle, Redshift, Mims): lowest risk; near-mechanical
  lift of each single builder's logic into its handler.
- **Batch B — 1-to-many** (Training ×4, CreateModel ×2 — many steps share one handler): migrate
  together so the handler is exercised by all members.
- **Batch C — Processing (36 rows), keyed by the 0b `step_assembly` audit**: largest/riskiest.
  Migrate `code` (2A) first, then `step_args` (2B), then the lone `delegation` (DataUploading).
  `DummyTraining` migrates in the `step_args` batch.

**Per-batch gate:** byte-diff the generated SageMaker step (`ProcessingStep`/`TrainingStep`/…
inputs/outputs/dependencies) produced by the routed shell against the legacy builder's output
for fixed inputs — **no drift**. A mismatch blocks that step, not the batch. Migration is
per-step reversible (dual-mode: the catalog/assembler can't tell a shell from a hand-written
builder).

## Final Phase — Drop `step_names.yaml`

Only after (i) 0b parity green, (ii) Phase-1 repoints live, (iii) all batches migrated + verified.

- **Change:** delete `registry/step_names.yaml`; the `.step.yaml` `registry:` blocks become the
  sole source; retire the parity oracle (or flip it to a golden snapshot).
- **The one invariant that MUST survive:** `get_step_names()` keeps returning per-step dicts
  carrying a **`sagemaker_step_type`** key (+ `config_class`/`builder_step_name`). Two consumer
  families break silently otherwise — the `StepInfo.registry_data[...]` family
  (`step_catalog.py:1752`, `step_catalog/models.py:51`) and the `get_sagemaker_step_type*` family
  (`registry/step_names.py:343-399`, used by `mcp/tools/catalog.py` and `mods/exe_doc/utils.py`).
- **Risk:** point of no return; the masking at `step_catalog.py:1195-1204` makes a regression
  silent. Keep a one-release window where the loader can fall back to a vendored snapshot if a
  `.step.yaml` block is malformed.

## Verification & Testing Strategy

Testing is **part of each phase, not a trailing step** — because both failure modes degrade
silently, every change must be guarded by a test that turns the silent failure into a loud one.
The table below maps each phase to the existing test files to **update** and the new tests to
**create**. Run the whole suite with the standard harness
(`brazil-build test`, output redirected per the workspace convention).

### Per-phase test work

| Phase | Update existing | Create new | What it must catch |
|---|---|---|---|
| **0a** discovery gate | `tests/step_catalog/test_builder_discovery.py` — add a case: a class whose direct base is `TemplateStepBuilder` IS discovered; a non-builder class is NOT; the 44 current builders still resolve. Re-run `tests/step_catalog/test_integration.py`, `test_expanded_discovery.py`, `test_dual_search_space.py` unchanged (must stay green = the strict-superset proof). | — (the widening is covered by updating the discovery test) | a `TemplateStepBuilder` shell going invisible (failure mode #1) |
| **0b** registry loader + audit | `tests/registry/test_step_names_yaml.py`, `tests/registry/test_step_names.py` — keep asserting the current `get_step_names()` shape (these become the parity baseline). | `tests/registry/test_registry_interface_parity.py` — assert `build_registry_from_interfaces()` is **dict-equal** to the `step_names.yaml`-derived `get_step_names()` for `sagemaker_step_type`/`config_class`/`builder_step_name`. `tests/registry/test_step_assembly_audit.py` — AST-scan each of the 36 Processing builders and assert its derived `step_assembly` (`code`/`step_args`/`delegation`) matches the registry tag; explicitly assert `DummyTraining == step_args`. | a wrong-shaped/short registry dict (failure mode #2); a mis-tagged `step_assembly` |
| **0c** facade + handlers + router | — | `tests/core/base/test_template_step_builder.py` — the facade accepts the 5-kwarg ctor, exposes `.spec`/`.contract`/`.config`/`_get_base_output_path`/`set_execution_prefix`, and delegates `create_step`/`_get_inputs`/`_get_outputs` to its bound handler. `tests/steps/builders/handlers/test_*_handler.py` (one per handler) — **facade-fidelity**: each handler surfaces `.spec`/`.contract` (so the assembler's `_propagate_messages` guard at `pipeline_assembler.py:249` and `_generate_outputs` at :357 are NOT silently triggered). `tests/core/base/test_resolve_handler.py` — routes by `sagemaker_step_type` (+`step_assembly`), raises `NoBuilderError` for `Base`/`Lambda`, and **never branches on the step name** (assert `DummyTraining` → Processing handler). | a handler dropping `.spec`; route-by-name regression |
| **1** registry repoint | `tests/step_catalog/test_step_catalog.py`, `tests/step_catalog/test_mapping.py`, `tests/step_catalog/test_builder_discovery.py`, `tests/core/base/test_builder_base.py`, `tests/registry/test_step_names.py` — must stay green with the source repointed behind the `get_step_names()`/`get_config_step_registry()` facades. Add a `mapping` case asserting `_fallback_config_to_step_type` uses the *same* convention as the loader. | — | a consumer that read a now-absent field; a divergent convention |
| **1** spine no-edit proof | `tests/core/compiler/test_dag_compiler.py`, `test_dynamic_template.py`, `tests/core/assembler/test_pipeline_assembler.py`, `test_pipeline_template_base.py` — run **unchanged**; their staying green is the evidence the spine needs no edits. Add an assembler assertion that the YAML-loaded spec still populates `builder.spec.step_type` (read at `pipeline_assembler.py:368`). | — | a spine assumption silently broken |
| **2..N** builder → shells | For each migrated builder, its existing per-builder test (where present) must still pass against the shell. The **universal builder validator** `tests/validation/builders/test_universal_test.py` must pass for every shell (it is the framework-level builder contract check). | `tests/core/integration/test_builder_migration_parity.py` — the **byte-diff gate**: for ≥1 representative pipeline per batch (A/B/C), assert the routed shell emits a SageMaker step structurally identical (inputs/outputs/dependencies) to the pre-migration baseline. | per-batch construction drift |
| **final** drop `step_names.yaml` | `tests/registry/test_step_names*.py` — repoint off the deleted file; keep the `get_step_names()`-carries-`sagemaker_step_type` assertion. Re-run the consumer tests for both families: `tests/step_catalog/test_models.py` (`StepInfo.sagemaker_step_type`), `tests/cli/test_catalog_cli.py`, and any `mcp`/`mods` exe-doc tests that read step types. | flip `test_registry_interface_parity.py` to a golden-snapshot assertion (oracle retired). | the `sagemaker_step_type` key silently dropped |

### The end-to-end gate (runs every phase from 0c on)

Extend `tests/core/integration/test_pipeline_execution_temp_dir_integration.py` (or add
`tests/core/integration/test_simplification_e2e.py`) with a full compile→assemble→discover→build
of a known multi-step pipeline containing **a Processing-2A step, a Training step, and a
MODS/SDKDelegation step**, asserting:
- the emitted `Pipeline` is structurally identical to the pre-migration baseline;
- `PipelineDAGCompiler.preview_resolution` and `validate_dag_compatibility` report **no
  `UNKNOWN`/unresolvable builders** (the canary for failure mode #1).

This single test is the highest-signal guard: it exercises every hierarchy hop and would catch a
silent discovery or registry regression that the unit tests individually miss.

### Coverage / regression discipline

- Run the **full suite green before each phase merges** — the spine tests (compiler/assembler)
  staying green *is* the containment proof; treat any new failure there as a design violation,
  not a test to patch.
- Keep the **parity oracle and byte-diff tests in CI for the whole migration window**; only retire
  them at the final phase (flip parity to a golden snapshot).
- Add a CI check that **fails loudly** if `get_step_names()` returns a dict missing the
  `sagemaker_step_type` key for any row — this converts the framework's silent
  graceful-degradation (`step_catalog.py:1195-1204`) into a hard error during migration.

## Definition of Done

- [x] **0a** `STEP_BUILDER_BASE_NAMES` widened in `_inherits_from_step_builder_base` (`builder_discovery.py:24`); all builders discover byte-identically; a test asserts `TemplateStepBuilder` subclasses ARE discovered.
- [x] **0b** `build_registry_from_interfaces()` exists (`registry/interface_registry_loader.py`); CI parity test (`test_registry_interface_parity.py`) dict-equal to `step_names.yaml`-derived `get_step_names()`; **green**.
- [x] **0b** `step_assembly` audit complete (`tests/registry/test_step_assembly_audit.py`, 18 code + 16 step_args + 1 delegation = 35); `DummyTraining → step_args` confirmed. (The audit was repointed to read `patterns.step_assembly` from the `.step.yaml` after FZ 31e1d3f1.)
- [x] **0c** `TemplateStepBuilder` is a `StepBuilderBase` subclass with the fixed 5-kwarg ctor, binds via `resolve_handler`, exposes `.spec`/`.contract`/`.config`/`_get_base_output_path`/`set_execution_prefix`/`create_step(...)`.
- [x] **0c** All 5 handler classes authored (6 verbs); each unit-tested against the assembler's `builder.spec` reads (:249, :357) — no handler silently drops `.spec` it needs. *(Shipped — see the Strategy+Facade plan's S1.)*
- [x] **0c** `resolve_handler` routes strictly by `sagemaker_step_type` (+`step_assembly` for Processing), never by name; `DummyTraining` routes as Processing (test-enforced). Routing now lives in `registry/strategy_registry.py` via `@register_strategy`.
- [x] **1** `get_step_names`/`get_config_step_registry`/`BUILDER_STEP_NAMES` are facades over the interface-derived loader (`step_names_base.STEP_NAMES` ← `build_registry_from_interfaces`); read sites verified; `LEGACY_ALIASES` preserved verbatim.
- [x] **1** Verified no edits needed in `dag_compiler`/`dynamic_template`/`pipeline_assembler`; the YAML spec still populates `.step_type`. (Spine tests stay green unchanged — the containment proof.)
- [x] **1** (Opportunistic) — **RETRACTED, then partially done.** `_create_step_builder_map` is NOT dead (2 load-bearing call sites: `dynamic_template.py` + `dag_compiler.py`); only the single dead `pipeline_template_base.py:305` call was removed (Implementation Log 2026-06-27). The abstract method stays.
- [x] **2..N** All **45** builders converted to discoverable shells, batched A/B/C(+C-SDK); each gated by the resolved-edge-graph snapshot + constituent byte-diff before its `create_step` was deleted. **0 builders define `create_step`; 45/45 extend `TemplateStepBuilder`.** (Owned + tracked in detail by the Strategy+Facade plan's Phase S3.)
- [x] **final** `step_names.yaml` deleted — **DONE (2026-06-28, commit `<final>`).** The per-step `.step.yaml` `registry:` blocks (+ the 3-row `_EXTRAS` map for the interface-less abstract steps) are now the SOLE source: `step_names_base.STEP_NAMES = build_registry_from_interfaces()` with NO fallback. The yaml + its `_load_step_names()` loader were removed. The parity oracle was flipped to a **golden snapshot** (`tests/registry/step_names_registry_snapshot.json`, re-baseline via `CURSUS_UPDATE_REGISTRY_SNAPSHOT=1`) — drift still fails loudly without comparing against a deleted file. Loud-fail preserved: `build_registry_from_interfaces` raises if any `.step.yaml` lacks `registry.sagemaker_step_type` (no silent drop). Verified: STEP_NAMES still 48 rows, dict-identical; 2242 passed; the hybrid manager (reads `STEP_NAMES` from the leaf) + `get_step_names()`/`get_config_step_registry()` facades unchanged.
- [ ] **end-to-end** — **NOT done (needs real session).** A full compile→assemble→discover→build byte-diff vs the pre-migration baseline with a real `PipelineSession`; the constituent byte-diffs + resolved-edge-graph snapshot + `preview_resolution` no-UNKNOWN checks are green offline, but the integration e2e is tracked for the SAIS env.
- [x] **tests** Per-phase test work shipped: `test_registry_interface_parity` (0b), `test_step_assembly_audit` (0b), `test_builder_templates`/handler-parity suites (`test_transform_handler_parity`/`test_model_creation_handler_parity`/`test_training_handler`/`test_sdk_delegation_handler`) (0c), `test_strategy_registry_consistency` (routing), `test_builder_migration_parity` + `test_migrated_shells_callable` (2..N), plus the post-migration conformance gates (`test_env_vars_config_interface_conformance`, `test_job_arguments_declaration_conformance`, `test_compute_spec`, `test_dependency_axis_conformance`, `test_patterns_section_conformance`). The universal builder validator (`tests/validation/builders/test_universal_test.py`) exists. *(Note: a SDK-env subset of universal-validator/migrated-shell tests skip-with-warning offline for the 4 SAIS builders — see the Strategy+Facade plan's Kiro §C.)*
- [x] **tests** Full suite stays green per merge; the spine tests stay green unchanged (containment proof); the loud-fail invariant `test_every_row_carries_sagemaker_step_type` (`test_registry_interface_parity.py:56`) is active. *(2 pre-existing exe_doc failures are unrelated to this plan, verified vs baseline.)*

## Cross-References

Design rationale and the full analysis trail (in the AmazonBuyerAbuseSlipboxAgent vault, FZ 31e
"Cursus Simplification"):

- FZ 31e1a — what registry info pushes into `.step.yaml` (the `registry:` block field spec) — backs **#1 / Phase 0b**.
- FZ 31e1c — `sagemaker_step_type` selects the construction verb (6 verbs; MODS steps share one) — backs **#3 / #4**.
- FZ 31e1d — the routed `TemplateStepBuilder` + `resolve_handler` design — backs **#2 / #3 / Phase 0c**.
- FZ 31e1d1 — resolving the 3 caveats (the `STEP_BUILDER_BASE_NAMES` gate, route-by-registry, the `step_assembly` audit) — backs **Phase 0a / 0b**.
- FZ 31e1d2 — the post-simplification two-hierarchy picture (builder vs SDK step object vs handler) — the class structure this plan builds.
- FZ 31e1e — the post-simplification end-to-end experience and flows.
- FZ 31e1f — the trail-side version of this change-list.

Related in-repo planning notes: [Simplify Pipeline Assembler](2025-07-09_simplify_pipeline_assembler.md),
[Pipeline Template Base Design](2025-07-09_pipeline_template_base_design.md),
[Specification-Driven Step Builder Plan](2025-07-07_specification_driven_step_builder_plan.md),
[Step Name Consistency Implementation Plan](2025-07-07_step_name_consistency_implementation_plan.md).
