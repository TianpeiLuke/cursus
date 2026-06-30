---
tags:
  - project
  - planning
  - step_builder
  - dependency_management
  - third_party
  - mods_workflow_core
  - sais_sdk
keywords:
  - dependency axis
  - mods_workflow_core
  - secure_ai_sandbox_workflow_python_sdk
  - requires
  - runtime_requires
  - native vs mods
topics:
  - declaring + surfacing each step pattern's 3rd-party package dependency in .step.yaml
language: python
date of note: 2026-06-28
---
# Third-party dependency axis — separate mods/SAIS-dependent patterns from native ones (FZ 31e1d3l)

> Directive (2026-06-28): "in your pattern you need to separate patterns that need to import from
> mods_workflow_core or sais python sdk from those do not: these indicate the dependency to a 3rd
> party package" — and "surface the separation of mods dependent patterns vs. native supported
> patterns in steps.yaml". So the dependency is DECLARED DATA in the `.step.yaml` (authored + visible),
> not just computed in the view — same single-source model as every other pattern axis, with a
> conformance gate that keeps the declaration equal to the actual imports.

## Why this axis exists

The Strategy+Facade migration made the step PATTERNS explicit (create_step / env / job-args / inputs /
outputs / compute). Orthogonal to *what* a pattern does is *what it costs to import*: a few patterns
couple to a heavyweight 3rd-party Amazon package, the rest are pure `sagemaker` + intra-cursus. That
coupling is operationally load-bearing — it decides whether a step's builder imports offline, whether
`import cursus` is safe in a SAIS-less env, and which Brazil deps a consumer must pull. It deserves to
be a first-class, declared axis.

## Audit (exhaustive, adversarially verified — 42-agent sweep, 2026-06-28)

Multi-modal sweep (by-import / by-builder / by-pattern-kind / by-config+script) + a completeness
critic, then every dependency claim adversarially verified against source. Findings:

- The ONLY 3rd-party packages a STEP pattern couples to are `mods_workflow_core` (build-time, lazy)
  and `secure_ai_sandbox_workflow_python_sdk` (build-time, hard module-level). `secure_ai_sandbox_python_lib`
  + the proxy-service models are RUNTIME-only (scripts / exe-doc), never builder-time.
- Compiler infra (`dag_compiler`, `single_node_compiler`, `nvme_security`) also touches
  `mods_workflow_core`/`mods_workflow_helper` but with try/except + working local fallback — optional
  and NOT a step pattern (out of scope for the per-step axis).

## Taxonomy (the tiers)

| tier | definition | members |
|---|---|---|
| **none** (native) | pure `sagemaker` + intra-cursus; builds + imports fully offline. The default. | all standard Processing/Training steps; compute kinds sklearn/xgboost/framework/estimator/model/transformer; the deferred-4 (DummyTraining, XGBoostModel, PyTorchModel, BatchTransform); SDKDelegationHandler itself (SDK class is INJECTED, the handler imports nothing) |
| **lazy-optional(mods_workflow_core)** | build-time soft dep, reached only on a specific code path; module import unaffected | `compute.kind=script` + `kms_network=true` → **EdxUploading** (lazy import of `KMS_ENCRYPTION_KEY_PARAM` / `PROCESSING_JOB_SHARED_NETWORK_CONFIG` in `builder_base._create_compute`; no fallback ⇒ fatal-on-path when that build runs, but nothing else). Compiler infra (dag/single_node/nvme) is the optional-with-fallback flavor — infra, not a step. |
| **hard(SAIS-SDK)** | build-time HARD module-level dep; importing the builder module is fatal-on-load if absent | the 4 SDKDelegation steps: **Registration**, **CradleDataLoading**, **DataUploading**, **RedshiftDataLoading** (each `from secure_ai_sandbox_workflow_python_sdk... import <Step>` at module top). Graceful degradation is EXTERNAL (steps/builders/__init__.py try/except → *_AVAILABLE flags; step_catalog AST-discovery try/except). |
| **runtime-script(SAIS-lib)** — *separate axis* | RUNTIME container-script dep; executes inside the SAIS Docker image at job time, NOT at pipeline construction; never affects offline import | `edx_uploading.py` + `redshift_data_loading.py` scripts → `secure_ai_sandbox_python_lib.session.Session`; `mods/exe_doc/cradle_helper.py` → proxy-service models (exe-doc generation, guarded). |

## Schema — declared in `.step.yaml` (DONE)

Three descriptors, build-time kept distinct from runtime:

1. **`compute.requires`** (`core/base/step_interface.py` `ComputeSpec`; `compute` is a top-level
   `.step.yaml` section as of 2026-06-28) — values `none` | `mods_workflow_core`. AUTO-DERIVED +
   validated as a CONSEQUENCE of `kms_network` (`requires=='mods_workflow_core'` IFF `kms_network`),
   so it can't drift and the `.step.yaml` may omit it. Declared explicitly on EdxUploading for
   visibility.
2. **`registry.requires`** (`core/base/step_interface.py` `RegistrySection` — a NEW section; the
   `registry:` YAML block was previously silently dropped) — values `none` |
   `secure_ai_sandbox_workflow_python_sdk`. Authored on the 4 SDKDelegation step YAMLs. Lives next to
   `sagemaker_step_type`/`step_assembly` because the SDK dep is a property of the create_step binding.
   Bonus: making `registry` a real section also fixed the view's `sagemaker_step_type`/`handler`
   resolution offline (it now falls back to the YAML `registry.sagemaker_step_type`).
3. **`contract.runtime_requires`** (`ContractSection`) — `List[str]` of packages the SCRIPT imports
   in-container (e.g. `[secure_ai_sandbox_python_lib]`). ORTHOGONAL to the two build-time descriptors;
   declared on EdxUploading + RedshiftDataLoading.

Value vocabulary: `none` | `mods_workflow_core` (compute, lazy) |
`secure_ai_sandbox_workflow_python_sdk` (create_step, hard) | runtime list members
`secure_ai_sandbox_python_lib` / `com.amazon.secureaisandboxproxyservice`.

## Surfacing (DONE)

`describe_step_patterns` folds `requires` into the relevant axis dict (`patterns.compute.requires`,
`patterns.create_step.requires`) AND adds a `dependencies` rollup:
`{build_time: {axis: pkg}, runtime: [...], native: bool}`. `cursus steps patterns <step>` renders a
`requires` block: `(none — native sagemaker only)` for the common case, else build-time (per axis,
annotated HARD module-level vs lazy/no-fallback) + runtime lines. Examples:
- `EdxUploading`  → build-time: compute → mods_workflow_core (lazy); runtime: secure_ai_sandbox_python_lib
- `Registration` → build-time: create_step → secure_ai_sandbox_workflow_python_sdk (HARD module-level)
- `TabularPreprocessing` / the deferred-4 → `(none — native sagemaker only)`

## Conformance gate (DONE) — `tests/core/base/test_dependency_axis_conformance.py`

Re-derives each declared `requires` from the ACTUAL import graph; fails on drift. Six checks:
1. `compute.requires == 'mods_workflow_core'` IFF `kms_network` (across all interfaces).
2. The only `from mods_workflow_core` import in `_create_compute` sits under `if spec.kms_network:`
   (AST) — catches a new unguarded mods import on another compute kind.
3. The set of builders declaring `registry.requires==SAIS-SDK` EXACTLY equals the set whose module
   does a module-level `from secure_ai_sandbox_workflow_python_sdk import ...` (catches the Redshift
   asymmetry: a SAIS import without the declaration, or vice-versa).
4. The 4 known SDKDelegation steps are declared (sanity, guards a no-op gate).
5. `contract.runtime_requires` equals the SAIS runtime packages the step's SCRIPT actually imports.
6. **Leak guard**: no builder MODULE imports a runtime-only SAIS lib at module level (keeps the
   runtime dep out of the build-time import surface).

## Deferred-4 dependency mapping (the user's earlier question, now answered on this axis too)

All four deferred-compute-residue steps are tier `none` — verified: their builders import only
`sagemaker.*`. So folding them into `_create_compute` later brings NO new dependency:
- DummyTraining (would-be `kind: framework`, SKLearn) → `requires: none`.
- XGBoostModel / PyTorchModel (`kind: model`) → `requires: none` (sagemaker `image_uris` only).
- BatchTransform (`kind: transformer`) → `requires: none`.
The ONLY compute-axis 3rd-party touch is `script`+`kms_network` (EdxUploading → mods_workflow_core);
the SAIS-SDK dep is exclusively on the create_step axis (the 4 SDKDelegation shells). The deferred
migration is dependency-clean.

## Status — DONE (2026-06-28)

- [x] `ComputeSpec.requires` (auto-derived + validated against `kms_network`).
- [x] `RegistrySection` (new section; `registry:` block no longer dropped) + `registry.requires`.
- [x] `ContractSection.runtime_requires`.
- [x] Declarations authored in the 4 SDK YAMLs + EdxUploading (build + runtime).
- [x] `describe_step_patterns` + CLI surface the dependency axis + a build-time/runtime rollup.
- [x] `sagemaker_step_type`/handler offline resolution fixed via the new registry section.
- [x] 6-check conformance gate (`test_dependency_axis_conformance.py`); full suite green
      (3085 passed; the 2 exe_doc + torch/xgboost failures are pre-existing, offline-env unrelated).
