---
tags:
  - project
  - planning
  - implementation_plan
  - strategy_pattern
  - facade_pattern
  - step_builder
  - template_builder
keywords:
  - Strategy pattern
  - Facade pattern
  - TemplateStepBuilder
  - PatternHandler
  - StrategyInfo
  - STRATEGY_REGISTRY
  - builder provider
  - registry-driven discovery
  - resolve_handler
  - strategies CLI MCP tool
  - dual-mode migration
topics:
  - step builder strategy + facade implementation
  - composition over inheritance
  - registry-driven builder discovery
language: python
date of note: 2026-06-27
---
# Strategy + Facade Pattern for the Builder Template — Implementation Plan

## Overview

This plan implements the **Strategy** and **Facade** design patterns for cursus step builders:
collapse the 44 hand-written `<Name>StepBuilder` classes into **one `TemplateStepBuilder` facade**
(Facade) that holds **injected strategy objects** — one per behavior axis — selected from a
self-describing **strategy library** (Strategy). It is the consolidated, source-grounded synthesis
of the design sub-arc developed this iteration (FZ 31e1d → 31e1d3c in the *Cursus Simplification*
trail) plus the foundations already shipped to `origin/mainline`.

It is the **companion** to [2026-06-26_step_builder_simplification_plan.md](2026-06-26_step_builder_simplification_plan.md)
(the registry-fold-in + hybrid-shell plan). That plan covers Phases 0–2 (discovery gate, registry
loader, the `TemplateStepBuilder` facade + 6 handlers, the registry source-swap, the pilot
migration). **This plan picks up where the hybrid leaves off** — formalizing the strategy library,
the introspection tool, and the optional pure-Strategy ("Design B") end-state — and is gated by the
pre-destruction safety record [2026-06-27_step_builder_special_treatment_preservation.md](../4_analysis/2026-06-27_step_builder_special_treatment_preservation.md).

### Already shipped (the foundation — do not re-do)

On `origin/mainline` as of 2026-06-27:
- **Facade exists**: `core/base/builder_templates.py` — `TemplateStepBuilder(StepBuilderBase)`,
  `PatternHandler` ABC, `resolve_handler(sagemaker_step_type, step_assembly)`, the **6 construction
  verbs** (Processing-2A/2B collapsed into one `ProcessingHandler` with `use_step_args`/
  `split_source_dir` knobs, Training/ModelCreation/Transform/SDKDelegation as stubs).
- **Discovery gate**: `STEP_BUILDER_BASE_NAMES = {"StepBuilderBase","TemplateStepBuilder"}` so shells
  are discovered (`builder_discovery.py`).
- **Registry is interface-derived**: `build_registry_from_interfaces()` + `registry:` blocks in all
  45 `.step.yaml`; `step_names_base.STEP_NAMES` now reads from it (parity-gated).
- **Pilot migrated**: `TabularPreprocessingStepBuilder` is a shell (−97 LOC), behavior-preserving,
  proven by the migration parity gate.
- **Safety record**: the 52-entry MUST-PRESERVE register documents every per-builder special
  treatment with file:line.

### The two patterns, precisely

- **Facade** — `TemplateStepBuilder` presents the unchanged `StepBuilderBase` interface the
  `PipelineAssembler` depends on (5-kwarg ctor, `create_step(**kwargs)->Step`, `.spec`/`.config`/
  `_get_base_output_path`/`set_execution_prefix`), hiding the strategy composition behind it.
- **Strategy** — each *behavior axis* (compute / inputs / outputs / env / job_args / construction)
  is an interchangeable, injected strategy object selected per step. The facade delegates each axis
  to its bound strategy; per-step variation becomes strategy *selection + declarative knobs*, not a
  subclass.

### End-state goals (synthesized)

- **G1** — A **strategy library**: ~low-20s strategy classes (≈5–7 per axis) covering all 44
  builders, with ~38–40/44 being *pure-knob over a shared strategy* per axis. (FZ 31e1d3b)
- **G2** — A **`STRATEGY_REGISTRY`** that is the single source both the runtime
  (`resolve_handler`/provider construction) and the introspection tool read — self-registering, no
  drift. (FZ 31e1d3b1)
- **G3** — The facade composes strategies (Facade + Strategy), preserving the assembler/catalog
  contract; per-step variation = strategy selection + `.step.yaml` knobs. (FZ 31e1d, 31e1d2)
- **G4** — An **introspection tool** (`cursus strategies` CLI + `strategies` MCP) so users/agents
  can enlist axes + strategies and query `for_step_type`. (FZ 31e1d3b1)
- **G5 (optional, gated)** — **Design B**: drop the per-step shell class entirely; discovery
  becomes a registry walk returning **builder providers**. Only as part of registry-auto-derivation,
  behind the FZ 31e2 CI closure check. (FZ 31e1d3, 31e1d3a, 31e1d3c)

## Plan status (2026-06-28) — CORE COMPLETE; only the OPTIONAL Design-B arm + the e2e gate remain

Every primary goal of this plan shipped to `origin/mainline`:
- **G1–G4 ✅** — strategy library (folded into 5 handlers + knobs), `STRATEGY_REGISTRY` single source,
  Facade composes strategies, `strategies` + `steps io` + `steps patterns` introspection tools.
- **S1 ✅ / S2 ✅ / S3 ✅** — all 45 builders are `TemplateStepBuilder` shells; **41/45 are PURE
  `STEP_NAME` shells** (4 SDK keep the code-only `sdk_step_class`; DummyTraining keeps 2 I/O methods;
  BedrockBatchProcessing keeps 1 env method).
- **Residue collapsed across EVERY axis** (env g, job-args h, inputs+outputs i, **compute k**) →
  per-step method residue is 1 env + 1 inputs + 1 outputs, all genuine keeps; **all compute factories
  0/45**.
- **G3 ✅ (knobs→YAML)** — the `patterns:` section (FZ 31e1d3f1); STEP_ASSEMBLY/HANDLER_KNOBS 0 on
  non-SDK builders; `output_path_token` removed (derived, FZ 31e1d3f1b); `us-east-1` is the
  toggleable `lock_training_region` pattern.
- **Beyond-plan additions** — a 3rd-party **dependency axis** (FZ 31e1d3l) and a **control-panel
  review** (FZ 31e1d3f) measuring the result against the define→align→read→steer principle.

**Remaining (both explicitly OPTIONAL/gated in the original plan, NOT regressions):**
- **S4 (Design B, classless)** — `get_builder_for_config` returns a provider; discovery becomes a
  registry walk. Gated behind FZ 31e2's triangle-closure CI check. Not pursued (the hybrid shell is
  the recommended end-state per 31e1d3 — A→B is a discovery swap, no wasted work).
- **e2e** — a full real-session compile→assemble→build byte-diff of a multi-step pipeline vs the
  pre-migration baseline (the constituent byte-diffs + the resolved-edge-graph snapshot are green;
  the integration e2e needs a real `PipelineSession`, tracked for the SAIS env).

## Implementation Log (progress)

Newest first. Tracks execution against the phased plan.

- **2026-06-28 — `output_path_token` REMOVED — derive S3 prefix from step name (FZ 31e1d3f1b). Commit
  `912daa88`, pushed.** Follow-up to the `patterns:` migration. Audit found `output_path_token`
  corresponds to the STEP NAME (25 unset, ~20 == the `canonical_to_snake(step_type)` default, the rest
  non-standard legacy values like `model_evaluation`/`packaging`). Per the directive that the
  deviations are non-standard, the field was REMOVED entirely (from `PatternsSection` +
  `ContractSection` + the 4 handler `KnobSpec`s + `as_knobs()`); the output S3 prefix is now DERIVED
  unconditionally as `canonical_to_snake(b.spec.step_type)` in all 3 synthesizing handlers. It is
  genuinely distinct from the contract container-path and the spec DAG reference, but its value is a
  pure function of the step name → a derivation, not a knob. For the formerly-deviating steps this
  standardizes their S3 output path (e.g. `model_evaluation/` → `xgboost_model_eval/`); no code/test
  depended on the old values. Conformance: `test_output_path_token_is_derived_not_declarable` +
  `test_output_prefix_derived_from_step_name`. Also fixed the surfaced ruff (F401 unused ClassVar;
  per-file E402 ignore for the builders `__init__` logger-before-guarded-imports pattern).

- **2026-06-28 — `patterns:` SECTION — per-step knob selections moved into the .step.yaml blueprint
  (FZ 31e1d3f1, closes G3). Commit `56006ee7`, pushed, −487 LOC / 81 files.** The directive: the step
  interface is the BLUEPRINT that wires pattern injection per axis, not hard-wired in the builder. A
  new top-level `PatternsSection` (peer of registry/compute/contract/spec) holds `step_assembly` /
  `output_path_token` / `include_job_type_in_path` / `direct_input_keys`; `_auto_bind_handler` reads it
  interface-first (class attrs back-compat fallback). `use_step_args` DROPPED (derived from
  `step_assembly`). 4 dead+build-BREAKING `make_compute: lambda b: b._get_processor()` knobs deleted
  (no `_get_processor` existed → AttributeError; the `compute:` block now drives them). **Result:
  STEP_ASSEMBLY 17→0, non-SDK HANDLER_KNOBS 39→0, 39/45 (now 41/45 after output_path_token removal)
  pure `STEP_NAME` shells.** The no-drift invariant became TRUE — `io_view` now reads
  `patterns.step_assembly`, the SAME field the build binds (closes the 31e1d3f Gap-1 introspection
  drift). Byte-diff: 41/45 effective bound knobs == baseline. Conformance:
  `test_patterns_section_conformance.py`.

- **2026-06-28 — 3rd-party DEPENDENCY AXIS declared in .step.yaml (FZ 31e1d3l). Commit `8c897621`,
  pushed.** Separates patterns that need a 3rd-party import (`mods_workflow_core` / SAIS SDK) from
  native sagemaker-only ones, as authored DATA. Grounded in a 42-agent adversarially-verified import
  sweep. Three descriptors: `ComputeSpec.requires` (auto-derived from `kms_network`),
  `RegistrySection.requires` (the 4 SDKDelegation steps; this also made `registry:` a REAL section —
  it was silently dropped before, which is why the patterns view showed `sagemaker_step_type=None`
  offline), `ContractSection.runtime_requires` (script in-container deps). Surfaced in
  `steps patterns` + a `dependencies` rollup; 6-check conformance gate
  `test_dependency_axis_conformance.py` re-derives each from the real import graph.

- **2026-06-28 — CONTROL-PANEL REVIEW (FZ 31e1d3f) + COMPUTE axis single-sourced incl.
  model/transformer (FZ 31e1d3k). Commits `c23c2113` (37 factories) + `7756aabd` (residue→0) +
  `00eac3c5` (compute promoted to a top-level section), pushed.** (a) **Compute:** all
  `_create_processor`/`_get_processor`/`_create_estimator`/`_create_model`/`_create_transformer`/
  `_get_image_uri` factories (was 41) collapsed into a declarative top-level `compute:` `ComputeSpec`
  (kind sklearn|xgboost|framework|script|estimator|model|transformer) + a single
  `builder_base._create_compute`; **compute factory residue now 0/45**. New `ComputeSpec` Pydantic
  validation against the SageMaker SDK surface; the hardcoded `us-east-1` became the toggleable
  `lock_training_region` pattern. (b) **Review:** a 29-agent adversarial review (10 confirmed / 13
  refuted gaps) measured the realized `.step.yaml`+facade against the user's control-panel principle
  (define→align→read→steer); its top gap (the STEP_ASSEMBLY split-brain) drove the FZ 31e1d3f1
  `patterns:` work above. Trail notes 31e1d3f/f1/k/l in abuse_slipbox.

- **2026-06-28 — `cursus steps patterns` / `steps.patterns` MCP shipped (FZ 31e1d3j). Commit
  `df7c0a44`, pushed.** A per-step view of the construction PATTERNS the TemplateStepBuilder composes
  (the 'plugins' a user sees when checking a step): bound `create_step` handler + env / job-arg /
  input / output patterns. **DERIVED** from data that already drives the build (registry binding +
  `.step.yaml` contract DATA the handlers read) — NO separate `patterns:` field, so the view cannot
  drift from behavior (the single-source-of-truth principle, vs. a parallel declaration that would
  need its own gate). Each axis carries `custom_override` where the builder still hand-overrides
  (the 4 genuine keeps). `io_view.describe_step_patterns()` (data) + CLI + MCP siblings to `steps io`;
  6 CLI tests; 1309 passed.

- **2026-06-28 — POST-S3 residue collapse: INPUTS + OUTPUTS axes single-sourced, RESIDUE LEDGER
  COMPLETE (FZ 31e1d3i). Commit `d00d5002`, pushed.** The last two residue axes. The 10 `_get_inputs`
  + 2 `_get_outputs` overrides decomposed into the standard spec×contract loop (already in
  ProcessingHandler) + small composable deviations, now per-step `.step.yaml` DATA — 4 NEW handler
  patterns so steps declare instead of override:
  - **`circular_ref_check`** (contract flag): run the PipelineVariable circular-ref guard first.
    `_detect_circular_references`/`_is_pipeline_variable` (byte-identical in 3 builders) moved to
    `builder_base`. [model_calibration, model_metrics_computation, model_wiki_generator]
  - **`input_source_overrides {logical: config_attr}`**: input SOURCE from a config attr/method
    (config is the value source), not the resolved dep. [bedrock_prompt_template_generation,
    label_ruleset_generation, dummy_data_loading, package]
  - **`skip_inputs [logical]`**: declared dependency the script loads internally (not mounted).
    [percentile_model_calibration `calibration_config`]
  - **`sink`** (contract flag): step produces no outputs → `get_outputs` returns `[]`. [edx_uploading]
  - Collapsed **8 `_get_inputs` + 1 `_get_outputs`** (byte-verified handler==override, temp-scrubbed;
    EdxUploading SDK-bound → static-only). Residue now `_get_inputs` 2, `_get_outputs` 1 — the 4 keeps
    (DummyTraining 3-tier + dual-name output, EdxUploading) are justified per-step deviations.
  - **Side-fix (review-flagged):** `PackageConfig.inference_scripts_source` had reimplemented a
    PARTIAL/buggy source-dir chain (`resolved_source_dir or source_dir or "inference"`) that ignored
    `processing_source_dir` — replaced with the canonical comprehensive `effective_source_dir`
    resolver. 1103 passed / 20 skipped; ruff F821-clean.

- **2026-06-28 — POST-S3 residue collapse: JOB-ARGUMENTS axis single-sourced (FZ 31e1d3h). Commits
  `4272892d` (collapse) + `58bad11d` (declarative interface). pushed.** The plan's stated "next axis"
  after env. Same config-is-single-source model.
  - **`BasePipelineConfig.get_job_arguments()`** (base default `None`) + **`_job_type_arg()`** helper
    for the dominant `["--job_type", config.job_type]` pattern (conditional-skip when unset; `flag`/
    `default` params cover the `--job-type` hyphen + the `getattr`-default variants).
    **`builder_base._get_job_arguments`** now delegates entirely to `config.get_job_arguments()` — the
    legacy `contract.expected_arguments` path removed (it held doc-strings, never values).
  - **Collapsed all 37** `_get_job_arguments` overrides → byte-verified IDENTICAL to a golden baseline
    of all 40 builders' output. 32 configs got a `get_job_arguments()` (28 via `_job_type_arg`
    incl. 4 conditional-training + getattr-default + hyphen; 4 bespoke: Bedrock batch/processing
    multi-arg, BedrockPrompt bool-flags, Payload passthrough); 5 return-None steps need no method.
  - **Declarative `job_arguments` in 31 `.step.yaml`** (`ContractSection.job_arguments: List[JobArgDecl]`,
    `{flag, source}`) — visibility/alignment/introspection ONLY; config still drives the values
    (mirrors `env_vars` declaring names). New conformance gate
    (`test_job_arguments_declaration_conformance.py`): declared flag set == config-emitted flag set.
  - Updated `test_builder_base` job-args tests to the config-source model; fixed F811
    (`new_serialize_config` dup import in `configs/utils.py`) + F401 cleanup. 1103 passed / 20 skipped.

- **2026-06-28 — POST-S3 residue collapse: ENV axis single-sourced (the `_get_environment_variables`
  residue ledger, FZ 31e1d3g). Commits `f975a29f`/`a124d130`/`0668d830`/`e44beef5`/`2128e595`/
  `f487e351`/`f80455a0`, pushed.** This is the first axis of the open "per-shell residue ledger" DoD
  item executed end-to-end. Full write-up:
  [2026-06-27_env_vars_config_single_source.md](2026-06-27_env_vars_config_single_source.md) +
  [../4_analysis/2026-06-27_bamt_script_vs_stepyaml_env_arg_alignment.md](../4_analysis/2026-06-27_bamt_script_vs_stepyaml_env_arg_alignment.md).
  - **Model:** the step interface DECLARES the env-var fields (`.step.yaml env_vars`); the config
    INSTANCE provides the VALUES; the two must not conflict. Implemented as **Option 1** — the
    interface name list drives the key set, config resolves each value, so drift is structurally
    impossible (config can't emit an undeclared key; a required key it can't resolve fails loudly).
  - **`BasePipelineConfig.get_environment_variables(declared_names)`** — generic resolver (convention
    `NAME`→`self.name` + `_format_env_value` type-formatting + `_env_overrides()` hook for
    computed/aliased + `hyperparameters` fallback). **`builder_base._get_environment_variables`** is
    now the ONE template (collector dispatch: bespoke config method → bespoke property → inherited
    resolver; then interface defaults; then `config.env`; missing-required/default-used diagnostics).
  - **Collapsed 39 of 40** `_get_environment_variables` overrides into the config-sourced base
    (each byte-verified `override == base` via a `model_construct` harness, or a config-backed
    improvement). NO-COLLECTOR steps (Package/Payload/Edx/TemporalSplit/CurrencyConversion) got config
    collectors; `CA_REPOSITORY_ARN` moved into the training configs. **1 genuine keep:
    `BedrockBatchProcessing`** — its `BEDROCK_BATCH_INPUT/OUTPUT_S3_PATH` are runtime/builder-derived
    (`Join(_get_base_output_path(), …)`, required by the script, NOT a config field), so it stays.
  - **Gates:** new config↔interface conformance gate (`test_env_vars_config_interface_conformance.py`,
    CONFORMANT 8→21, CONFLICT-B held at 0); 1100 passed / 20 skipped (core+steps+deps); ruff-clean.
  - **Side fixes:** preceded by a script↔interface alignment pass (rename/prune of stale `env_vars` +
    job-arg gaps across the `.step.yaml`, driven by the real BAMT scripts); 16 F541 fixes (11 were a
    latent bug — `job_type` validators used `{{v}}` so error messages never interpolated the value).
- **2026-06-27 — Phase S1, step 1/2 DONE: `strategy_registry` is live (commit `442813e9`, pushed).**
  Shipped `registry/strategy_registry.py` (the single source) — `StrategyInfo`/`KnobSpec`,
  `@register_strategy` (handlers self-register) + `register_no_builder`,
  `resolve_strategy`/`list_strategies`/`knobs_for`/`axes`, `NoBuilderError`, and a lazy guarded
  `ensure_strategies_loaded()` so it stays a dependency-free leaf (no circular import with
  `builder_templates`, verified). `builder_templates.resolve_handler` now **delegates to the
  registry** — the 4 hand-maintained routing dicts deleted, handlers self-register via decorators
  (`ProcessingHandler` under `step_assembly` code+step_args; the 4 verb handlers under their
  `sagemaker_step_type`/`delegation` with `implemented=False` until completed; Base/Lambda via
  `register_no_builder`). **Behavior-preserving** — routing parity verified (Processing
  code/step_args/delegation, the 1:1 types, Base→NoBuilderError, Processing-default=code). 7-test
  consistency gate (`test_strategy_registry_consistency.py`: covers-every-sagemaker_step_type,
  single-source grep-guard, knob coherence, the 2 axes). 32 S1+template+parity tests green; 1239
  across registry/core.base/step_catalog (only failure = the pre-existing `<0.1s` perf flake in
  `test_step_catalog`, untouched); new code ruff-clean.
- **2026-06-27 — Phase S1, step 2/2 DONE: all 4 stub handlers implemented, S1 COMPLETE.**
  Re-homed the construction logic from the live builders into the handlers, safest-first per the
  verified blueprint, one commit + push per handler:
  - **TransformHandler** (`b6378d90`): `get_inputs` → `(TransformInput, model_name)` tuple with
    model_name/processed_data dispatch + the two ValueErrors; `get_outputs` single str/Join;
    `build_step` ORDER inputs→outputs→make_compute LAST, singular `inputs=`, caching conditional.
  - **ModelCreationHandler** (`f1675bb2`): single-key `model_data` passthrough get_inputs;
    `get_outputs`→None; `build_step` DROPS caching (warn, no `cache_config`), `model=` direct,
    make_compute LAST, direct `model_data=` override.
  - **SDKDelegationHandler** (`f6118207`): 3 input modes (none / resolve_s3 / mims_ordered),
    `sdk_step_class` injection, region-suffix knob (the eu-west-1 bug fix), depends_on ctor-vs-method,
    force-off caching, MIMS PackagedModel-first ordering.
  - **TrainingHandler** (`5ba467ba`): channel-parse get_inputs (input_path → train/val/test fan-out,
    skip `hyperparameters_s3_uri` when configured, parts[5] channel naming); single-str get_outputs;
    `build_step` empty-inputs guard + outputs-before-compute + standard caching.
  Each gated by a session-independent parity suite in `tests/core/base/`; Training + ModelCreation
  assert **true byte-parity against the real builder** (get_inputs channels + get_outputs identical).
  Deleted the `_UnimplementedHandler` stub; `TestUnimplementedHandlers` → `TestAllHandlersImplemented`.
  Strategy registry now has **0 unimplemented construction handlers** (only Base/Lambda no-builder
  rows remain). 689 core/base + registry tests green; ruff clean.
- **2026-06-27 — Phase S2 DONE: strategy-introspection tool shipped (commit `6fc0516e`, pushed).**
  The strategy library is now queryable at runtime (FZ 31e1d3b1) via the twin CLI + MCP surface the
  project already uses for `catalog`. **Single-source discipline held:** added
  `axis_name_for_step_type()` to `strategy_registry` as the ONE routing rule — `resolve_handler` now
  calls it, so the tool's `for_step_type` resolves the exact (axis, name) the router binds (a test
  asserts this for every routable step type). Added JSON-safe renders `strategy_to_dict`/
  `knob_to_dict` (+ `_jsonsafe`: handler→class-name, callable defaults→name) and
  `find_strategies(name, axis?)`, all re-exported from `cursus.registry`.
  - **CLI** `cursus strategies` (`cli/strategies_cli.py`, registered under root): `axes` | `list
    [--axis]` | `show <name> [--axis]` | `for <step_type> [--step-assembly]` | `knobs --axis --name`,
    each `--format text|json`, nonzero exit on not-found / no-builder.
  - **MCP** `strategies` namespace (`mcp/tools/strategies.py`, registered in `mcp/registry.py`):
    `strategies.{list_axes,list,show,for_step_type,knobs}` returning `ToolResult` envelopes with
    `not_found`/`invalid_input` codes + remedies. `for_step_type` is the agent's "read the builder
    class" replacement under Design B.
  - **Tests:** `tests/cli/test_strategies_cli.py` (18) + `tests/mcp/test_strategies_tools.py` (11),
    incl. the router-parity guard. 718 core/base+registry+mcp + 109 cli green; new code ruff-clean.
- **2026-06-27 — Pre-S3 connection-mechanism preservation study + 3 fixes (FZ 31e1d3d/d1/d2).**
  Before starting destructive migration, audited whether the step-interface → builder → assembler
  **message-passing graph** (container↔step paths, step↔step `compatible_sources` matching,
  producer→consumer `property_path` transfer) survives the facade collapse. Method: a 44-agent
  workflow — 5-layer source map + 6-axis preservation analysis + 32 adversarial verdicts.
  **Finding: the graph is data-driven, never class-driven** — the resolver matches
  `normalize(provider_spec.step_type) in dep_spec.compatible_sources` (`dependency_resolver.py:486`)
  and transfers via `PropertyReference` keyed on `property_path` + the DAG node name, never a Python
  class — so the 44→1 collapse is structurally invisible. All 6 axes "with-care," none "at-risk";
  **2/32 risks confirmed real**, both about the YAML *transcription* that rides along with migration,
  not the collapse. All three actionable items now FIXED + pushed:
  - **job_type variant gap** (`9d010de0`): `TemplateStepBuilder.__init__` dropped `config.job_type`,
    so a variant-bearing step loaded the BASE spec instead of its job-typed variant — turning a
    required cross-step dep optional (the train→validation edge dissolves). `job_type` is the
    discriminator that lets one step type be N distinct inter-wired DAG nodes
    (`RiskTableMapping_Training/_Validation/_Testing`). Fix mirrors the legacy builders:
    `load_step_interface(STEP_NAME, job_type=getattr(config, "job_type", None))`. **Phase-S3 blocker
    for the 7 variant-bearing steps.** Test: `test_facade_job_type_variant.py` (3).
  - **resolved-edge-graph snapshot gate** (`c7b22c47`, FZ 31e1d3d1): `_sync_and_align` validates only
    intra-step alignment; nothing guarded inter-step wiring. New `test_resolved_edge_graph_snapshot.py`
    loads the REAL `.step.yaml` for a representative DAG, resolves the full graph, and freezes it
    (backbone-edge + full-snapshot assertions). Proven to catch a dropped-alias silent rewire
    (`XGBoostModel.model_data` → wrong producer). **This is the byte-diff gate (S3 DoD) applied to the
    connection graph — the per-step wiring gate for every shell migration.**
  - **_attach_spec non-bypassable** (`c7b22c47`, FZ 31e1d3d2): it is the sole writer of `step._spec`
    (the builder-driven resolver-enrichment path, Path B); a handler that forgot it broke Path B
    silently (assembler reads `builder.spec`, Path A — no symptom). Hoisted into the facade's
    `create_step`; `ForgetfulHandler` test locks the invariant.
  789 core/base+core/deps+registry tests green; ruff clean. Trail notes: 31e1d3d (+ d1/d2) in
  abuse_slipbox. Residual open (analysis-only, not blockers): d3 multi-output overlapping-alias
  mis-binding, d4 SDK-delegate `property_path`↔runtime-Properties, d5 output-destination token
  reconciliation (`step_type.lower()` vs `output_path_token` knob).
- **2026-06-27 — Phase S3 STARTED: BatchTransform migrated to a shell (commit `e658461f`, pushed).**
  First destructive migration — the 1:1 `Transform` verb (lowest risk, fully locally validatable;
  the other Batch-A steps Cradle/Redshift/Mims import the SAIS SDK → separate C-SDK batch).
  `BatchTransformStepBuilder` dropped `__init__`/`create_step`/`_get_inputs`/`_get_outputs` (all
  re-homed verbatim into `TransformHandler`, byte-identical per `test_transform_handler_parity.py`)
  and became a shell declaring `STEP_NAME="BatchTransform"`, **295 → 73 lines** (then dropped
  `validate_configuration` in `e448102f` as config-redundant → ~50 lines). Kept only **one**
  genuinely per-step method: `_create_transformer` (the handler's `make_compute`). **No HANDLER_KNOBS** — the output token
  `spec.step_type.lower()="batchtransform"` is identical both paths (the job-typed variant narrows
  `compatible_sources`, not `step_type`). Designed + adversarially verified by a 4-agent workflow
  (byte-parity / discovery+routing / must-preserve lenses — all "safe-to-apply", zero blockers).
  **All gates green:** ruff; discovery finds it + routes to `TransformHandler`;
  `spec.step_type=BatchTransform`; output token byte-identical; resolved-edge-graph snapshot +
  transform parity + **1377 core/base+deps+registry+step_catalog tests** pass. Dropped `__init__`
  guards (isinstance/job_type-hasattr) are by-design for shells (config typing enforced upstream by
  the registry). **Migrated: 2/45** (TabularPreprocessing pilot + BatchTransform).
- **2026-06-27 — `validate_configuration` cleanup: 43/44 overrides removed, 1 slimmed (FZ 31e1d3e).**
  Made `StepBuilderBase.validate_configuration` a concrete **no-op default** (was `@abstractmethod` —
  so the base now has 3 abstract methods: `_get_inputs`/`_get_outputs`/`create_step`). The Pydantic
  config is the validation authority (required fields + field/model validators + defaults enforced at
  construction, before the builder runs). A 45-agent audit classified all 44, then, per review, the
  non-redundant ones were NOT migrated as `min_length`/`ge` constraints (that re-introduces
  over-strict validation) — they were **dropped** because each was either redundant (required field
  already enforced) or **over-strict** (empty-string/range guard on an already-required field that
  fails downstream anyway; `processing_source_dir`-strict contradicts the config's
  `source_dir` fallback; `job_type` membership rejected valid single-word job_types). Commits
  `33f029b2` (11 redundant) → `cd0ec5ce` (8 job_type) → `658b874a` (7 source_dir) → `cdcade81`
  (15 over-strict + registration slimmed) → `a5536fc8` (untrack stray worktrees) → `1e0ece7f`
  (percentile + temporal — the alignment review found these "2 keeps" were ALSO fully config-covered,
  not genuine invariants). **1 remains:** `registration` (slimmed to its runtime spec↔contract
  alignment — the only check that reads `self.spec`/`self.contract`, which the config can't see).
  2001 tests pass; lint unchanged vs HEAD. This shrinks the per-step residue (31e1d3e) ahead of
  shell migration.
- **2026-06-27 — `cursus steps io` CLI + `steps.io` MCP + channels-into-`.step.yaml` (commit `e11c8be2`).**
  A second introspection tool (FZ 31e1d3d follow-up), the per-step **path / wiring view** the Facade
  hides: `steps/interfaces/io_view.py:describe_step_io` returns per-dependency `container_path` +
  training-channel fan-out and per-output `container_path` + `property_path` reference — the I-O
  complement to `catalog.step_spec` (see the "Introspection surface map" above). Shipped as the
  established twin: `cli/steps_cli.py` (`cursus steps io <name> [--job-type] [--format]`, registered
  in `cli/__init__.py`) + `mcp/tools/steps.py` (`steps.io`, registered in `mcp/registry.py`).
  **Also single-sourced the training channels into the `.step.yaml` (STEP-INTERFACE-AS-DATA, G3):**
  added `InputPort.channels` + `ContractSection.input_channels`; `TrainingHandler.channels_for` is
  now the ONE channel rule shared by `get_inputs` (runtime) and `io_view` (static); the 4 training
  interfaces declare `channels: [train, val, test]` (replacing the handler's hardcoded constant,
  kept only as a back-compat fallback). Tests: `tests/cli/test_steps_cli.py` + `tests/mcp/test_steps_tools.py`.
- **2026-06-27 — Phase S3 Batch B DONE: 6 builders → shells (commits `41641d65`/`be24796a`/`c6fa6ee9`/`5e85560f`).**
  The 1-to-many verbs, all locally validatable: **CreateModel ×2** (XGBoostModel, PyTorchModel →
  `ModelCreationHandler` — single-key `model_data` passthrough, `get_outputs`→None, caching dropped;
  kept `_create_model`/`_get_image_uri`/`_get_environment_variables`) and **Training ×4** (XGBoost,
  LightGBM, LightGBMMT, PyTorch → `TrainingHandler` — channel fan-out from the `.step.yaml`
  `channels:`, skip-hyperparameters, single output-path; kept `_create_estimator` +
  `_get_environment_variables` + `_get_job_arguments`, each with `HANDLER_KNOBS{output_path_token,
  direct_input_keys:[input_path]}`). Per-builder residue preserved verbatim (framework estimator
  class, `max_run` field-name differences, CA_REPOSITORY_ARN; PyTorch's dead-but-unwired
  `_get_metric_definitions`/`_create_profiler_config` kept as-is for behavior-identity). Each gated:
  ruff + handler-parity (`test_model_creation_handler_parity.py` / `test_training_handler.py` incl.
  true real-builder parity) + resolved-edge-graph snapshot + discovery/routing + full suite green.
  **Migrated: 8/45** (Batch B ✅ 6/6 + TabularPreprocessing pilot + BatchTransform).
- **2026-06-28 — Phase S3 C-SDK DONE: all 5 SAIS-SDK-bound builders → shells. MIGRATION COMPLETE: 45/45.**
  Executed in an SDK-equipped SAIS notebook environment (SageMaker m5.12xlarge, pytorch env, Python
  3.10 w/ `amzn-secure-ai-sandbox-workflow-python-sdk==1.0.233` + `amzn-mods-workflow-helper==1.0.233`
  + `amzn-mods-python-sdk==1.0.233` installed via `sais_environment/install.sh`, coral preserved via
  backup/restore). Each shell passes: ruff clean, imports + constructs, binds the correct handler
  type, `test_migrated_shells_callable` discovers all 45. Per-builder:
  - **CradleDataLoading** (71L): `SDKDelegationHandler`, `HANDLER_KNOBS={sdk_step_class:
    CradleDataLoadingStep}`. Keeps `get_output_location`/`get_step_outputs` (public API called by
    exe_doc generator and assembler introspection).
  - **RedshiftDataLoading** (19L): `SDKDelegationHandler`, simplest — no overrides at all.
  - **Registration** (19L): `SDKDelegationHandler`. `validate_configuration` was subsequently DROPPED
    (commit `13027c47`) — see the validation note below; the handler `mims_ordered` mode replicates the
    old `_get_inputs` (PackagedModel-first ordered `ProcessingInput` list) and `append_region` the
    region-suffixed step name.
  - **DataUploading** (21L): `SDKDelegationHandler` via `STEP_ASSEMBLY="delegation"` (Processing-typed
    but routes to SDK handler, not ProcessingHandler).
  - **EdxUploading** (105L): `ProcessingHandler` (code) — NOT delegation. The only Processing-2A
    builder that sets KMS/network. Keeps `_create_processor` + `_get_environment_variables` +
    `_get_inputs` + `_get_outputs` as MRO overrides. (`_resolve_script_path` was removed post-audit —
    see validation note: it was dead code, edx now resolves its script via `config.get_script_path()`
    like every other ScriptProcessor step.)
  Test updated: `test_migrated_shells_callable` count assertion 40→45, `SDKDelegationHandler` import
  added, 5 entries added to `_MIGRATED`. **232 tests green** (sdk_delegation_handler 48 + migrated
  shells 136 + builder_templates + step_assembly_audit + builder_discovery). Ruff clean all 5.
- **2026-06-27 — Phase S3 C-SDK VALIDATED (static + adversarial audit from the SDK-less mainline env).**
  Since the 5 builders can't be imported locally, validated by per-builder `git diff` vs pre-migration
  baseline `c12fc6eb` + import-free registry-routing checks + handler behavior-equivalence tracing +
  an adversarial refutation pass per builder. **Verdict: all 5 migrated correctly (45/45).** Full
  write-up: [2026-06-27_S3_C-SDK_5_builders_migration_audit.md](2026-06-27_S3_C-SDK_5_builders_migration_audit.md).
  Two findings:
  - **edx `_resolve_script_path` was dead code (FIXED here).** The migrated shell kept it but nothing
    called it — `ProcessingHandler` uses `config.get_script_path()`. edx was the only processing
    builder with a custom resolver. The feared `code=None` regression is provably unreachable
    (`validate_entry_point_paths` raises at config construction if both source dirs are unset, so the
    old tier-3 package fallback was dead code in the old builder too). Removed it to align edx with
    every other ScriptProcessor step — byte-equivalent, ruff clean, offline suite green.
  - **Registration `validate_configuration` drop (commit `13027c47`) is safe but mis-justified
    (RECORDED, no code change).** The message says "redundant with `_sync_and_align`", but the two run
    in OPPOSITE directions (dropped guard: `required spec deps ⊆ contract.inputs`; `_sync_and_align`:
    `contract.inputs ⊆ spec.deps`). The drop is defensible for a *different* reason — the old guard was
    over-strict for `path: null` contract inputs (would falsely fail BatchTransform/PyTorchModel/
    XGBoostModel) — and the loss is latent for Registration (both required deps have matching contract
    paths today). Finding recorded; protective value against future spec↔contract drift on SDK steps
    is gone.
- **PHASE S3 COMPLETE. ALL 45 BUILDERS ARE NOW `TemplateStepBuilder` SHELLS.**
  The builder simplification (FZ 31, Vector 3) has achieved its primary goal: per-step boilerplate
  collapsed into the facade + 6 handlers + declarative knobs.
- **2026-06-28 — REGISTRY DATA-SOURCE FLIP COMPLETE: `step_names.yaml` deleted.** The companion
  [2026-06-26 plan](2026-06-26_step_builder_simplification_plan.md)'s Final Phase shipped — the
  registry table now derives SOLELY from the `.step.yaml` `registry:` blocks (`STEP_NAMES =
  build_registry_from_interfaces()`, no fallback); the parity oracle became a golden snapshot. So
  BOTH the registry data AND the builder bodies now live in the `.step.yaml` files. Remaining open
  work is **Phase S4** (the Design B *classless-discovery* rewrite — `get_builder_for_config` returns
  a provider, AST scan → registry walk) — optional and gated by FZ 31e2's triangle-closure CI check.
  *(S4 is a discovery/instantiation rewrite, distinct from the now-done registry-data derivation.)*

## FOR KIRO (SAIS env) — additional checks + improvements on the 5 C-SDK builders

The static + adversarial audit (from the SDK-less mainline) cleared all 5, but some checks **only the
SAIS environment can run**, and the audit surfaced specific things to confirm/improve. Do these in the
SAIS notebook where the SDK imports resolve:

### A. Run the one gate the static audit could NOT — the real-session byte-diff (highest priority)
For EACH of the 5, build the step the OLD way (git-stash the shell to restore the pre-migration
`builder_*.py` at `c12fc6eb`) and the NEW way (current shell), with the **same config + a real SAIS
`PipelineSession`**, and diff the produced SageMaker/SDK step. Confirm equality of:
- **ctor args** passed to each SDK `*Step` (the handler re-homed these from the old `create_step` —
  confirm for your installed SDK version `1.0.233`, since ctor signatures can drift across SDK versions).
- **Cradle**: the returned step's output attributes — call `get_output_locations()` /
  `get_output_locations("DATA"|"METADATA"|"SIGNATURE")` on both and diff. The static audit argues the
  property-path branch removed from `get_output_location` was best-effort-with-fallthrough, but only a
  real Cradle step confirms the shapes match (lazy `Properties` ref vs concrete value).
- **Registration (Mims)**: the **region-suffixed step name** (`<name>-<region>`; confirm it uses
  `config.region`, NOT hardcoded `us-east-1`) AND the **ordered `processing_input` list**
  (PackagedModel first, GeneratedPayloadSamples second only when present) — diff element-by-element.
  Also confirm `performance_metadata_location` and `depends_on` pass through the ctor.
- **DataUploading**: `input_s3_location=` resolves identically (upstream `input_data` dep, else
  `config.input_s3_location` fallback).
- **EdxUploading**: the `ProcessingStep` `code=` path and the `_create_processor` `volume_kms_key` /
  `network_config` are byte-identical to the old builder (this one is a vanilla `ProcessingStep`, not
  an SDK step).

### B. Re-verify the items the audit could only reason about statically
- **EDX `_resolve_script_path` removal (done on mainline).** `git pull` so your SAIS tree has the
  removal, then confirm in a real session that `config.get_script_path()` resolves the edx script for
  the configs you actually run (esp. if any prod config sets `processing_source_dir`/`source_dir`
  unusually). If any real edx config genuinely needs the old package-bundled `cursus/steps/scripts/`
  fallback, flag it — the static audit proved that path was unreachable for *constructible* configs,
  but your prod configs are the ground truth.
- **CradleDataLoading helpers' callers.** Static grep found the only in-repo callers of
  `get_output_location`/`get_step_outputs` are TSA project scripts calling `step.get_output_locations()`
  directly. Confirm no SAIS-side / notebook / exe-doc code calls the *builder* helpers in a way that
  depended on the removed spec-driven property-path branch.

### C. Improvements to make (small, optional but recommended)
- **Fix commit `13027c47`'s rationale or restore a corrected guard.** Its message ("redundant with
  `_sync_and_align`") is directionally wrong (see the validation note above). Either (a) leave a
  corrected note, or (b) if you want SDK-delegated steps to keep spec↔contract drift protection,
  re-add a build-time guard that **skips deps whose contract path is `null`** (the over-strictness bug
  that justified removal) — e.g. only assert `required dep ∈ expected_input_paths` for deps the
  contract actually gives a non-null path. Gate with the full suite.
- **~~Add an SDK-env CI/skip marker for `test_migrated_shells_callable`.~~** ✅ **DONE (commit
  `04bc0616`, 2026-06-28).** Added `_SDK_STEPS` set + `_has_sais_sdk` probe (tries `import
  secure_ai_sandbox_workflow_python_sdk` + `import mods_workflow_core`); each of the 3 parametrized
  tests calls `pytest.skip(f"{step_name} requires SAIS SDK")` when the SDK is absent. In SAIS env:
  136 passed / 0 skipped. In non-SAIS env: 121 passed / 15 skipped (5 steps × 3 tests). The 45/45
  count assertion is unconditional (counts dict entries, not importable classes). Mainline green in
  both environments.
- **Confirm `test_sdk_delegation_handler.py`'s recording fakes still match the real `1.0.233` SDK
  ctor shapes** — it uses fakes, so a real-SDK ctor drift would pass the fake test but fail at runtime.
  The real-session byte-diff in (A) is what catches this.
## Architecture (target)

```
registry row (.step.yaml registry: block) ── sagemaker_step_type, step_assembly, knobs
        │
        ▼  resolve_handler + STRATEGY_REGISTRY lookup
TemplateStepBuilder  (FACADE — a StepBuilderBase; the assembler's contract surface)
        │  composes (STRATEGY, one per axis)
   ┌────┼─────────┬──────────┬──────────┬──────────┬───────────────┐
   ▼    ▼         ▼          ▼          ▼          ▼               ▼
 verb  compute  inputs    outputs     env       job_args     source_dir(=verb knob)
 (Pattern make_   get_      get_      get_env   get_job_
  Handler) compute inputs   outputs    vars      arguments
        │
        ▼  create_step(**kwargs) -> Step      (unchanged contract → PipelineAssembler)
```

A step = **one verb handler + one strategy per axis + knobs**. The two hierarchies stay distinct
(FZ 31e1d2): the cursus builder hierarchy (facade + optional thin shell) vs the SageMaker/SAIS step
*objects* a handler constructs; they meet only at the `step = ...(...)` call inside `build_step`.

## The strategy library (G1) — what to build

> **AS-SHIPPED (S1 outcome, alignment review 2026-06-27):** the per-axis strategy *classes* below
> were **NOT built as classes**, and **no `core/base/strategies/` dir was created**. The per-axis
> behavior was folded into the **5 `PatternHandler` verb classes + declarative `KnobSpec` knobs**,
> routed by the two axes `sagemaker_step_type` / `step_assembly` (see the S1 DoD + Implementation
> Log). This table is retained for the per-step quirk ↔ safety-record mapping it captures — it is
> **NOT a list of files to create.** The "Implement under `core/base/strategies/`" line below is
> superseded.

Per FZ 31e1d3b, the concrete classes per axis (the safety record gives the per-step quirks each must
absorb). ~~Implement under `core/base/strategies/` (or `steps/strategies/`):~~ (folded into the
handlers + knobs — see the AS-SHIPPED note above):

| Axis | Strategy classes (count) | Dominant pure-knob strategy |
|---|---|---|
| compute (`make_compute`) | SKLearnProcessor / XGBoostProcessor / FrameworkProcessor / FrameworkEstimator / FrameworkModel / Transformer / NoCompute (~7) | SKLearnProcessorStrategy (~18 builders) |
| inputs (`get_inputs`) | SpecContractJoin / TrainingChannel / ModelDataPassthrough / TieredModelArtifact / ConfigSourced / NoInputDelegation / TransformInput (~7) | SpecContractJoinInputStrategy (~28) |
| outputs (`get_outputs`) | SpecContractProcessingOutput / TrainingOutputPath / TransformOutputPath / NoOutputs (~4) | SpecContractProcessingOutputStrategy (~30) |
| env (`get_environment_variables`) | Contract / ConfigMerge / ConfigSoleSource / TrainingCaRepo / Coercing / ComputedJoin (~6) | ConfigMergeEnvStrategy (~15) |
| construction + source_dir + job_args | ProcessingHandler / TrainingHandler / ModelCreationHandler / TransformHandler / SDKDelegationHandler (5 verbs) + ~6 knobs | ProcessingHandler (33) |

**Knobs (no new class), examples:** `instance_type_mode`, `framework_version_field`, `processor_cls`,
`distribution_type` (FullyReplicated), `direct_input_keys`, `skip_logical_names`, `output_path_token`,
`job_type_in_path`, `source_attrs`/`merge_config_env`, `use_step_args`, `split_source_dir`,
`supports_cache`, `job_args_shape` (`jobtype_required`|`none`|`latent_training`|`config_overridable`),
`depends_on_mode`.

**Bug fixed for free:** the hardcoded `region='us-east-1'` in the training/model image-URI retrieval
collapses to one `region` param (default from session/config) inside `FrameworkEstimatorStrategy`/
`FrameworkModelStrategy`. (Flagged in the safety record R36 as a carry-forward bug — fixing it here
is an explicit, in-scope decision, not silent drift.)

**Genuine per-step residue — UPDATE (2026-06-28): nearly all collapsed.** The list below was the
S1-era prediction; almost every entry became declarative DATA, not a kept method:
- EdxUploading compute (ScriptProcessor + ECR-from-role + KMS/network) → `compute: {kind: script,
  kms_network: true}` (FZ 31e1d3k); EdxUploading is now a PURE shell.
- batch_transform inputs (`Tuple` return) → re-homed into `TransformHandler.get_inputs`; shell.
- the 3 Bedrock multi-flag job-args → config `get_job_arguments()` (FZ 31e1d3h); shells.
- registration/data_uploading ctor wiring → `SDKDelegationHandler` + the code-only `sdk_step_class`
  knob (the documented serialization exception).
- **The ONLY genuine method keeps left are DummyTraining's** `_get_inputs` (3-tier config>dep>SOURCE
  model resolution — no declarative knob expresses it yet) + `_get_outputs` (dual logical-name), and
  BedrockBatchProcessing's `_get_environment_variables` (runtime-S3 `Join`). All have MUST-PRESERVE
  rows in the safety record.

## The strategy registry + introspection tool (G2, G4)

> **AS-SHIPPED note (alignment review 2026-06-27):** the registry shipped at
> **`registry/strategy_registry.py`** (NOT `core/base/strategy_registry.py` as the body below
> says) — kept a dependency-free leaf to avoid a circular import with `core/base/builder_templates`
> (see S1 log). The introspection surface shipped as **TWO** tools, not one: `cursus strategies`
> (this section) **and** `cursus steps io` / `steps.io` MCP (the per-step I/O view — see the
> "Introspection surface map" below and the Implementation Log entry for commit `e11c8be2`).

- **`STRATEGY_REGISTRY`** (`registry/strategy_registry.py`): `{axis: [StrategyInfo]}` where
  `StrategyInfo = {axis, name, covers, parameters:[KnobSpec], summary, source_ref}`. Each strategy
  self-registers via `@register_strategy(axis=...)` (or a scanned `AXIS` attr — open question
  31e1d3b1a). `resolve_handler`/provider construction look strategies up *by axis+name* in this same
  registry. **Single source — the tool never re-lists.**
- **`cursus strategies` CLI** (`cli/strategies_cli.py`, registered in `cli/__main__.py` beside
  `catalog`/`registry`): `axes`, `list [--axis]`, `show <name>`, `for --sagemaker-step-type X
  [--step-assembly]`, `knobs --axis`. Mirror `catalog_cli.py` formatting.
- **`strategies` MCP tool** (`mcp/tools/strategies.py`, mirroring `mcp/tools/catalog.py`):
  `list_axes`, `list`, `show`, `for_step_type`, `knobs` — read-only, structured JSON.
- **`for_step_type` is the key call** for both audiences: returns the default strategy+knob binding
  per axis for a step type — the strategy-side analogue of `catalog.step_spec`, and the replacement
  for "read the builder class" under Design B.

### Introspection surface map (four shipped surfaces — which to call when)

The Facade hides per-step detail across FOUR complementary read surfaces (reconciled here so the
overlap isn't read as drift):

| Surface | Answers | Keyed by |
|---|---|---|
| `catalog.step_spec` / `cursus catalog show` | the declared I/O **contract / ports** (dependencies, outputs + `property_path`) | step_name |
| `strategies.for_step_type` / `cursus strategies for` | the **construction binding** — which handler/verb + knobs the facade selects | `sagemaker_step_type` (+ `step_assembly`) |
| `steps.io` / `cursus steps io` | the **path / wiring view** — per-dep `container_path` + training channel fan-out; per-output `container_path` + `property_path` reference | step_name (+ `job_type`) |
| `steps.patterns` / `cursus steps patterns` (FZ 31e1d3j) | the **per-axis construction patterns** ('plugins') for ONE step: bound `create_step` handler + env / job-arg / input / output patterns, each `custom_override`-flagged — DERIVED from the registry binding + `.step.yaml` contract DATA (no separate field, cannot drift) | step_name (+ `job_type`) |

`steps.patterns` is the step-name-keyed complement to `strategies.for_step_type` (type-keyed): the
latter answers "what does this step TYPE bind to + which knobs exist", the former resolves that for a
concrete step and folds in its per-step contract DATA (the active env/job-arg/input/output patterns)
and any genuine-keep overrides.

`property_path` deliberately appears in both `catalog.step_spec` and `steps.io` (steps.io adds
`container_path` + channels). `catalog.step_info` carries both `step_name` and `sagemaker_step_type`,
so it is the TYPE↔NAME pivot bridging `strategies.for_step_type` (type-keyed) and `steps.io`
(name-keyed).

## Phased plan

The phasing extends the shipped Phase 0–2. Each builder migration is **per-step reversible** and
**byte-diff-gated**; the safety record's MUST-PRESERVE rows gate every collapse.

### Phase S1 — Strategy library + registry (additive, no builder touched) — ✅ DONE (2026-06-27)
- Implement the strategy classes per axis (G1) + `STRATEGY_REGISTRY`/`@register_strategy` (G2).
- Wire `resolve_handler` + the handlers' `make_compute`/`get_inputs`/`get_outputs` defaults to look
  up strategies from the registry by axis+name+knobs (today the `ProcessingHandler` already does the
  generic join; this generalizes it to all axes/verbs).
- **Complete the 4 stub handlers** (Training/ModelCreation/Transform/SDKDelegation) from
  `builder_templates.py` using the strategy classes — currently `NotImplementedError`.
- **Tests:** per-strategy unit tests; extend the migration parity harness
  (`tests/core/base/test_builder_migration_parity.py`) to one representative builder per verb,
  proving the strategy produces identical `get_inputs`/`get_outputs`/processor-config/job-args to the
  hand-written builder (session-independent constituents — the full-step byte-diff is the e2e item).
- **Risk:** facade fidelity — a handler/strategy that drops `.spec` makes the assembler silently skip
  it (`pipeline_assembler.py:249/:357`). Unit-test every strategy against the assembler's spec reads.

### Phase S2 — Build the introspection tool (G4) — ✅ DONE (2026-06-27)
- `cursus strategies` CLI + `strategies` MCP tool reading `STRATEGY_REGISTRY`.
- **Tests:** `tests/cli/test_strategies_cli.py`, `tests/mcp/test_strategies_tools.py`; assert `axes`
  count, `for_step_type` returns one strategy per axis for each verb, `list --axis` filters,
  registry-tool consistency (the tool lists exactly what `STRATEGY_REGISTRY` holds).
- Lands independently of any builder migration — pure introspection over the library.

### Phase S3 — Migrate builders to shells, batched (the Facade adoption — destructive) — NEXT (gates in place)
Per the proven pilot recipe (TabularPreprocessing): extend `TemplateStepBuilder`, declare
`STEP_NAME` + `HANDLER_KNOBS` (+ **`STEP_ASSEMBLY`** for Processing `step_args`/`delegation` shells —
absent it routes as `code`; this is the same `STEP_ASSEMBLY` the 2026-06-26 plan documents and
`tests/registry/test_step_assembly_audit.py` enforces, so every Batch-C `step_args`/`delegation`
shell must set it), delete `__init__`+`create_step`, keep deviating per-step methods (or move them
to a per-step strategy), and **drop `validate_configuration`** unless it holds a runtime invariant
the Pydantic config can't (FZ 31e1d3e). Batched by verb, byte-diff-gated:
- **Batch A — 1:1 verbs:** Transform, Cradle, Redshift, Mims (lowest risk).
- **Batch B — 1-to-many:** Training ×4, CreateModel ×2.
- **Batch C — Processing (by step_assembly):** the 18 `code` then 16 `step_args` then 1 `delegation`
  (DataUploading). Drive the split off the enshrined `step_assembly` audit
  (`tests/registry/test_step_assembly_audit.py`).

  > **⚠️ source_dir is THREE patterns, not two — the load-bearing per-builder knob (verified by a
  > deterministic source audit of every builder's `.run()`/`ProcessingStep` call, 2026-06-27).** Get
  > `split_source_dir` wrong and the migrated step uploads the wrong files → container ImportError at
  > runtime (silent until execution). The handler reproduces all three EXACTLY (proven:
  > `split=False`→`run(code=/full/path.py)`; `split=True`→`run(code=name.py, source_dir=/full)`):
  >
  > | Pattern | `STEP_ASSEMBLY` | `split_source_dir` | builders |
  > |---|---|---|---|
  > | **2A code** | (none, default) | n/a (`ProcessingStep(code=full_path)`) | the 18 code builders |
  > | **2B step_args, NO split** | `step_args` | **False** | xgboost_model_eval, xgboost_model_inference, lightgbm_model_eval, lightgbm_model_inference, bedrock_batch_processing, active_sample_selection, pseudo_label_merge (7) |
  > | **2B step_args, SPLIT** | `step_args` | **True** | risk_table_mapping, tokenizer_training, dummy_training, lightgbmmt_model_eval, lightgbmmt_model_inference, pytorch_model_eval, pytorch_model_inference, bedrock_processing, percentile_model_calibration (9) |
  >
  > **Conservative recipe (used for FeatureSelection): keep each builder's own `_get_inputs`/
  > `_get_outputs` as MRO overrides** — then the kept overrides moot
  > `output_path_token`/`include_job_type_in_path`, shrinking the per-builder risk surface to
  > essentially the assembly choice.
  >
  > **UPDATE (2026-06-27, commit `f92d66cf`) — source_dir is now `.step.yaml` DATA, NOT a per-shell
  > knob.** `ContractSection.source_dir: bool` is the source of truth (the 9 split steps declare
  > `source_dir: true`); `ProcessingHandler` reads it as the `split_source_dir` switch. So a Batch-C
  > `step_args` shell sets `STEP_ASSEMBLY="step_args"` and does NOT need `split_source_dir` in
  > `HANDLER_KNOBS` — the source_dir decision lives in the interface, verified once. **Forward
  > standard (the "standardize the Processing call" decision):** ALL Processing steps use
  > `step_args` (retiring the deprecated 2A `code=` path); the two patterns are
  > `contract.source_dir: false` → `run(code=full_path)` vs `true` → `run(code=entry, source_dir=dir)`.
  > Converting the 18 current `code` builders to `step_args` is a behavioral change → gated by the
  > resolved-edge-graph snapshot + a real-session byte-diff (tracked as a post-faithful-migration
  > modernization, NOT folded into the byte-preserving shell migration).
  >
  > **Researched + confirmed (SDK 2.251.1 source + the SDK's own deprecation warning + AWS-expert
  > Sage Q&A, 2026-06-27) — what the 2A→2B change actually does:**
  > - **2A is SDK-deprecated:** `ProcessingStep.__init__` emits *"We are deprecating the
  >   instantiation of ProcessingStep using `processor`. Instead, simply use `step_args`."*
  > - **The behavioral delta is processor-class-specific.** `ScriptProcessor.run()` (→ SKLearn/
  >   XGBoost/Script — the 17 ScriptProcessor-family `code` builders) has **NO** `sourcedir.tar.gz`/
  >   `runproc.sh` packaging: it uploads the single script as `code` and runs it directly — so
  >   `ProcessingStep(code=script)` (2A) → `ScriptProcessor.run(code=script)→step_args` (2B) is a
  >   **near-equivalent path** (same single-script upload; the change just retires the deprecation
  >   warning). `ScriptProcessor.run` has **no `source_dir` param** (verified), so these stay
  >   `source_dir: false` — converting one to *split* would break.
  > - `FrameworkProcessor.run()` DOES pack (`_pack_and_upload_code` + `_create_and_upload_runproc`
  >   → `sourcedir.tar.gz` + `runproc.sh`). That packaging is the sensitive path and the source of
  >   the classic `/opt/ml/processing/input/code/ No such file` / `sourcedir.tar.gz Cannot open`
  >   failures — but our FrameworkProcessor steps are **already** on `step_args`, so the modernization
  >   does not touch them.
  > **Refined risk:** the 2A→2B modernization of the 17 ScriptProcessor builders is LOW risk
  >   (no packaging change), but still a code-path change → keep it gated by the edge-graph snapshot +
  >   a real-session byte-diff and out of the byte-preserving shell migration.
- **Batch C-SDK — the SAIS-SDK steps** (Cradle/Redshift/DataUploading/Registration/Edx): **cannot
  byte-diff locally** (module-level SAIS SDK import absent) — validate in an SDK-equipped env or with
  mocked SDK step classes; the local harness must **skip-with-warning, not silently pass**.
  > ⚠️ **NOT YET satisfied:** `test_migrated_shells_callable` currently FAILS hard on the 5 SDK builders
  >   in any non-SAIS env (15 failures, `No module named 'secure_ai_sandbox_workflow_python_sdk'` /
  >   `mods_workflow_core`) instead of skip-with-warning. **Action for Kiro** (SAIS follow-up §C): add
  >   `pytest.importorskip` / a `skip_unless_sais` marker so mainline is green while SAIS still exercises
  >   them.
- **Pre-S3 gates LANDED (2026-06-27, FZ 31e1d3d):** (a) `config.job_type` now flows through the
  facade (`9d010de0`) — unblocks the **7 variant-bearing steps** (risk_table_mapping,
  missing_value_imputation, feature_selection, batch_transform, cradle_data_loading,
  temporal_sequence_normalization, temporal_feature_engineering), which could not migrate without it;
  (b) `_attach_spec` is non-bypassable in the facade's `create_step` (`c7b22c47`); (c) the
  **resolved-edge-graph snapshot gate** (`tests/core/deps/test_resolved_edge_graph_snapshot.py`,
  `c7b22c47`) freezes the inter-step wiring.
- **Gate per builder:** the **resolved-edge-graph snapshot stays green** (catches a transcription
  drift that silently rewires) + the constituent byte-diff parity harness green + discovery finds it +
  ruff + the full suite stays green, **before** deleting its `create_step`. For variant-bearing
  steps, also assert the migrated shell resolves each job-typed variant (per
  `test_facade_job_type_variant.py`).

### Phase S4 (OPTIONAL, gated) — Design B: drop the shell class entirely
Only if/when registry-auto-derivation (the simplification plan's final phase) makes it nearly free.
Per FZ 31e1d3a/31e1d3c:
- `StepCatalog.get_builder_for_config` returns a **builder provider** (factory) instead of a `Type`.
- `PipelineAssembler._initialize_step_builders` calls `provider(...)` instead of `builder_cls(...)`
  with the **same 5 kwargs** — a class IS-A provider, so the edit is a no-op for class returns
  (dual-mode). **Do this call-site edit early/independently** — it's a behavior-identical
  generalization that de-risks B without committing.
- `builder_discovery` AST machinery → a **registry walk** producing providers, with the AST scan kept
  as a transition fallback so hybrid shells coexist.
- **Mandatory guard:** the FZ 31e2 **triangle-closure CI check** (`registry keys == providers ==
  _step_index keys`) lands BEFORE going registry-only — discovery soft-fails to `None` and
  `_build_index` masks errors (`step_catalog.py:1195-1204`), so a missing provider degrades silently
  without it.
- Re-point all `class.__name__` display readers to `step_name` (the 31e1d3c breakage register, 13
  rows, fixes each).

## Subsystem impact (what changes vs stays)

| Subsystem | Change | Stays |
|---|---|---|
| `core/base/builder_templates.py` | ✅ all 5 handlers via strategies; registry lookup; `__init__` passes `config.job_type` to `load_step_interface`; `create_step` makes `_attach_spec` non-bypassable; **`_auto_bind_handler` reads the `.step.yaml` `patterns:` blueprint interface-first** (FZ 31e1d3f1); handlers dispatch compute to `builder_base._create_compute` when `compute.kind` is set (FZ 31e1d3k); output prefix DERIVED as `canonical_to_snake(step_type)` (no token knob, FZ 31e1d3f1b) | facade ctor + `create_step` delegation + `_overrides` MRO check |
| `registry/strategy_registry.py` | ✅ `@register_strategy` + introspection helpers (`axis_name_for_step_type`/`strategy_to_dict`/`find_strategies`). **Shipped at `registry/` (a dependency-free leaf), NOT `core/base/strategies/`** — the per-axis class library was folded into the 5 handlers + knobs | — |
| `core/base/builder_base.py` | ✅ `validate_configuration` → concrete **no-op default** (FZ 31e1d3e). ✅ `_get_environment_variables` is now the ONE env template (config-sourced, FZ 31e1d3g) — dispatches to `config.get_environment_variables(declared_names)`, applies interface defaults + `config.env`, keeps diagnostics; 39/40 per-step overrides deleted | the 3 abstract methods (`_get_inputs`/`_get_outputs`/`create_step`) |
| `core/base/config_base.py` | ✅ `get_environment_variables(declared_names)` resolver + `_env_overrides()` + `_format_env_value` (FZ 31e1d3g). ✅ `get_job_arguments()` (base None) + `_job_type_arg()` helper (FZ 31e1d3h) — config is the single source for env values AND CLI args; the interface declares names/flags | the config model + `get_script_path`/`resolve_hybrid_path` |
| `core/base/step_interface.py` | ✅ TOP-LEVEL sections `registry` (sagemaker_step_type + `requires`), `compute` (`ComputeSpec`, FZ 31e1d3k), `patterns` (`PatternsSection`: step_assembly/include_job_type_in_path/direct_input_keys, FZ 31e1d3f1) + `ContractSection.{source_dir, include_job_type_in_path, env_vars, job_arguments, circular_ref_check, skip_inputs, input_source_overrides, sink, runtime_requires}` — per-step facts are `.step.yaml` DATA driving the handlers (FZ 31e1d3g/h/i/k/l). **`output_path_token` REMOVED** (FZ 31e1d3f1b — derived from step name); `job_arguments`/`env_vars` declarative (config provides values) | the StepInterface model + accessors |
| `core/base/builder_base.py` | ✅ `_get_environment_variables`/`_get_job_arguments` config-sourced templates (FZ 31e1d3g/h); `_detect_circular_references`/`_is_pipeline_variable` re-homed as shared input-safety helpers (FZ 31e1d3i) | the 3 abstract methods |
| `core/base/step_interface.py` | ✅ added `InputPort.channels` + `ContractSection.input_channels` — training channel layout is now `.step.yaml` DATA (G3), read by `TrainingHandler.channels_for` | the StepInterface model + accessors |
| `cli/strategies_cli.py` + `mcp/tools/strategies.py` | ✅ NEW — the **strategies** introspection tool (construction binding view) | the `catalog`/`registry` surfaces (sibling pattern) |
| `cli/steps_cli.py` + `mcp/tools/steps.py` + `steps/interfaces/io_view.py` | ✅ NEW — **steps io** (per-step path/wiring view) + **steps patterns** (FZ 31e1d3j: per-axis construction patterns — create_step/env/job-arg/input/output, `custom_override`-flagged, derived from registry binding + contract DATA) | catalog/strategies sibling pattern |
| `tests/core/deps/test_resolved_edge_graph_snapshot.py` | ✅ NEW — freezes the inter-step wiring graph (the S3 per-step wiring gate) | the resolver/assembler code (read-only gate) |
| `steps/builders/*` | become shells (S3); per-step residue → per-step strategy; `validate_configuration` overrides now OPTIONAL (43/44 removed, base no-op default) | file location, `<Name>StepBuilder` class name (discovery key) |
| `PipelineAssembler` | S4 only: `provider(...)` not `builder_cls(...)` (~3 lines) | propagation/outputs/instantiate; the 4-key `create_step` contract |
| `StepCatalog` / `mapping` | S4 only: `get_builder_for_config` returns a provider | name resolution + `LEGACY_ALIASES` + job-type fallback; `get_builders_by_step_type` registry filter |
| `builder_discovery` | S4 only: registry walk + AST fallback shim, then delete AST | (during S1–S3) unchanged — discovers the shells |

## Definition of Done

- [x] **S1** Strategy/handler logic implemented per verb (5 handlers); each unit-tested + parity-tested against the hand-written builder's spec reads. (Per-axis strategy *classes* G1 folded into the handlers + knobs; the registry is the routing single source.)
- [x] **S1** `strategy_registry` + `@register_strategy` landed; `resolve_handler`/handlers look strategies up from it; `test_strategy_registry_consistency.py` asserts the registry is the single source (no second list).
- [x] **S1** Migration parity harness covers a representative builder per verb — identical inputs/outputs vs hand-written (session-independent constituents); Training+ModelCreation assert true byte-parity against the real builder.
- [x] **S2** `cursus strategies` CLI + `strategies` MCP tool landed, reading the registry; `for_step_type` returns the binding for every verb + matches `resolve_handler`; CLI/MCP/registry consistency tested.
- [x] **Pre-S3 (FZ 31e1d3d)** Connection mechanism proven preserved (data-driven, not class-driven); the 3 actionable risks fixed — `config.job_type` flows through the facade, `_attach_spec` non-bypassable, resolved-edge-graph snapshot gate landed.
- [x] **Pre-S3 (FZ 31e1d3e)** `StepBuilderBase.validate_configuration` is a concrete no-op default (one fewer abstract method); the Pydantic config is the validation authority; 43/44 builder overrides removed, 1 slimmed (`registration`'s runtime spec↔contract check) — shrinks per-step shell residue ahead of S3.
- [x] **S2+ (FZ 31e1d3d follow-up)** `cursus steps io` CLI + `steps.io` MCP shipped (the per-step path/wiring introspection view); training channels relocated into the `.step.yaml` (`InputPort.channels`), single-sourced via `TrainingHandler.channels_for`. The introspection surface is now two tools (strategies + steps io); see the "Introspection surface map".
- [x] **S3** All **45** builders → shells, batched A/B/C(+C-SDK); each gated by the resolved-edge-graph snapshot + constituent byte-diff before its `create_step` was deleted; variant steps assert per-job_type resolution; SDK steps validated in an SDK env or skip-with-warning locally; every MUST-PRESERVE row from the safety record accounted for; the full suite green per batch. **45/45 — ALL builders migrated. ✅ COMPLETE.** Batch A ✅ 1/1 (BatchTransform) · Batch B ✅ 6/6 · Batch C ✅ 34/34 non-SDK · **C-SDK ✅ 5/5** (cradle/redshift/data_uploading/edx/registration) — migrated 2026-06-28 in SAIS SDK env. POST-MIGRATION (2026-06-28) the residue then collapsed further: the `make_compute: lambda b: b._get_processor()` knobs that some shells initially kept were DELETED (they were dead/broken — see FZ 31e1d3k/f1 log entries); compute is now the declarative `compute:` descriptor, and the assembly knobs are the `.step.yaml` `patterns:` block. **End state: 41/45 are PURE `STEP_NAME` shells** (4 SDK keep `sdk_step_class`; DummyTraining keeps 2 I/O methods; BedrockBatchProcessing keeps 1 env method). *(45 = concrete `builder_*.py`/`.step.yaml` count; "44" elsewhere is the legacy headline.)*
- [x] **S3** Per-shell residue ledger: for each migrated shell, record which retained methods are now-redundant copies of the handler (deletable behind the byte-diff gate) vs genuine keeps — so shadow-copies don't accumulate. **ALL AXES COLLAPSED INCL. COMPUTE** (2026-06-28): ENV (FZ 31e1d3g), JOB-ARGS (FZ 31e1d3h), INPUTS+OUTPUTS (FZ 31e1d3i), **COMPUTE (FZ 31e1d3k)**. **Final live residue counts across the 45 shells (re-verified 2026-06-28):** `validate_configuration` **0/45** ✅ · `_get_job_arguments` **0/45** ✅ · `_get_environment_variables` **1/45** ✅ (the 1 = BedrockBatchProcessing's runtime-S3 `Join`, a genuine keep) · `_get_inputs` **1/45** ✅ (keep = DummyTraining 3-tier model resolution — EdxUploading collapsed to a pure shell since the original count) · `_get_outputs` **1/45** ✅ (keep = DummyTraining dual logical-name) · **compute factories `_create_processor`/`_get_processor`/`_create_estimator`/`_create_model`/`_create_transformer`/`_get_image_uri` ALL 0/45** ✅ (FZ 31e1d3k — collapsed into the declarative top-level `compute:` `ComputeSpec` + `builder_base._create_compute`, incl. the model/transformer/dummy framework kinds). **Only 2 genuine method keeps remain (both DummyTraining); all else is declarative.** New handler patterns: `circular_ref_check`, `skip_inputs`, `input_source_overrides`, `sink` on `ContractSection`.
- [x] **S3/G3** Knobs→YAML source-swap (closes G3) — **DONE via a top-level `patterns:` section (FZ 31e1d3f1), commit `56006ee7`.** `STEP_ASSEMBLY` + the declarative `HANDLER_KNOBS` (`step_assembly`/`direct_input_keys`/`include_job_type_in_path`) moved into the `.step.yaml` `patterns:` block; `_auto_bind_handler` reads it interface-first (class attrs are a back-compat fallback). `use_step_args` DROPPED (derived from `step_assembly`). **`output_path_token` REMOVED entirely (FZ 31e1d3f1b, `912daa88`)** — it corresponds to the step name, so the output S3 prefix is DERIVED as `canonical_to_snake(step_type)`, not declared. **Live state: `STEP_ASSEMBLY` class attrs 0/45, non-SDK `HANDLER_KNOBS` 0/45, `output_path_token` declarations 0.** The only remaining `HANDLER_KNOBS` are the 4 SDKDelegation builders' code-only `sdk_step_class` (a SAIS class object, not serializable — the documented exception). Conformance: `test_patterns_section_conformance.py`. So 41/45 builders are PURE `STEP_NAME` shells (4 SDK keep sdk_step_class; DummyTraining keeps its 2 I/O methods; BedrockBatchProcessing keeps its 1 env method).
- [x] **S3** Bug decision recorded: hardcoded `us-east-1` made a **TOGGLEABLE `ComputeSpec.lock_training_region` pattern** (FZ 31e1d3k) — the SAIS training-image region lock is now opt-in declared data (`lock_training_region: true` + `locked_region`), switchable to standard mode (region from `config.aws_region`) via `.step.yaml`/config with NO code change. Not silently changed — made an explicit, declared, toggleable pattern.
- [ ] **S4 (if pursued)** `get_builder_for_config` returns a provider; assembler calls `provider(...)`; discovery is a registry walk + AST fallback; the FZ 31e2 triangle-closure CI check is green BEFORE any shell is deleted; all `class.__name__` readers re-pointed to `step_name`.
- [ ] **e2e** A full compile→assemble→discover→build of a multi-step pipeline (Processing + Training + a MODS/SDKDelegation step) produces a `Pipeline` identical to the pre-migration baseline; `preview_resolution`/`validate_dag_compatibility` show no `UNKNOWN`/unresolvable builders.

## Cross-References

Design trail (the *Cursus Simplification* trail, FZ 31, in the AmazonBuyerAbuseSlipboxAgent vault):
- FZ 31e1d — `TemplateStepBuilder` routed by `sagemaker_step_type` (the facade + 6 verbs).
- FZ 31e1d2 — the two-hierarchy picture (facade + composition vs the SDK step objects).
- FZ 31e1d3 — Strategy-injection vs the 2-line shell (the design fork; ship hybrid, B is the registry-arm).
- FZ 31e1d3a — refactoring PipelineAssembler + StepCatalog for Strategy (the provider seam).
- FZ 31e1d3b — the strategy library catalog (the class/knob enumeration G1 implements).
- FZ 31e1d3b1 — the introspection tool (G2/G4).
- FZ 31e1d3c — StepCatalog + builder_discovery deep review (the class-dependency map + 13-row breakage register for S4).
- FZ 31e1d3e — why the step still has a class (the shrinking-residue trajectory, empirical).
- FZ 31e1d3g/h/i/j/k/l — the residue-collapse axes shipped post-S3: env / job-args / inputs+outputs / `steps patterns` view / compute single-source / 3rd-party dependency axis.
- FZ 31e1d3f — control-panel review (define→align→read→steer); 31e1d3f1 — the `patterns:` section (closes G3); 31e1d3f1b — `output_path_token` removed (derived from step name).

In-repo:
- [2026-06-26_step_builder_simplification_plan.md](2026-06-26_step_builder_simplification_plan.md) — the registry-fold-in + hybrid-shell plan (Phases 0–2, already shipped); this plan is its Strategy/Facade continuation.
- [2026-06-27_env_vars_config_single_source.md](2026-06-27_env_vars_config_single_source.md) — the ENV-axis residue collapse (FZ 31e1d3g): interface declares env-var fields, config provides values; `_get_environment_variables` 40→1. The first residue axis executed; template for the `_get_job_arguments` collapse next.
- [2026-06-27_S3_C-SDK_5_builders_migration_audit.md](2026-06-27_S3_C-SDK_5_builders_migration_audit.md) — static+adversarial audit of the 5 SAIS-SDK-bound shell migrations (45/45 verdict).
- [../4_analysis/2026-06-27_bamt_script_vs_stepyaml_env_arg_alignment.md](../4_analysis/2026-06-27_bamt_script_vs_stepyaml_env_arg_alignment.md) — BAMT script ↔ `.step.yaml` env/job-arg/channel alignment review (drives the env + future job-arg reconciliation).
- [2026-06-27_step_builder_special_treatment_preservation.md](../4_analysis/2026-06-27_step_builder_special_treatment_preservation.md) — the 52-entry MUST-PRESERVE safety record that gates every builder collapse.
