---
tags:
  - analysis
  - brittleness
  - critical_review
  - technical_debt
  - cursus_package_wide
keywords:
  - silent error swallowing
  - drifting duplication
  - fragile string conventions
  - hidden mutable state
  - missing guards
  - single source of truth
  - naming.py
  - safe_cli_command
  - step_interface deep merge
topics:
  - code brittleness
  - systemic technical debt
  - remediation prioritization
language: python
date of note: 2026-06-26
status: completed
---

# Cursus Package-Wide Brittleness Review

## Overview

This is a package-wide brittleness review of `src/cursus/` — a hunt for code that is **correct today but will silently break, produce wrong results, or become a maintenance trap** under realistic change. It was produced by fanning out one finder per subsystem across all 15 subsystems, then **adversarially verifying every proposed finding against the actual code** (a second agent tried to refute or downgrade each one), then clustering the survivors into cross-cutting themes.

**Method**: 15 subsystem finders → per-subsystem adversarial verification → systemic-theme synthesis. 31 agents total. Findings that the verifier could not confirm against the code (e.g. a guard that actually existed, an unreachable path, or intentional container-script defensiveness) were dropped or downgraded — the numbers below are **post-verification**.

**Result**: **70 confirmed findings** across **15 subsystems** — **3 HIGH, 19 MEDIUM, 48 LOW**. They cluster into **5 systemic themes**. The two highest-value remediations are *adoption of infrastructure that already exists* (`naming.py` for naming, `_shared.safe_cli_command` for CLI exit codes), not new design.

> Brittleness ≠ a bug that fires today. Most LOW findings are latent (e.g. workspace-aware stale-state paths that don't fire because the default deployment has no workspaces). They are recorded so the *next* change doesn't trip over them. The HIGH findings, by contrast, already fire on shipped data or in CI.

### Confirmed findings by subsystem

| Subsystem | Total | 🔴 H | 🟠 M | 🟡 L |
|---|---:|---:|---:|---:|
| `cli` | 2 | 2 | 0 | 0 |
| `core_base` | 3 | 1 | 1 | 1 |
| `core_deps_utils` | 7 | 0 | 1 | 6 |
| `steps_scripts` | 7 | 0 | 2 | 5 |
| `core_config_fields` | 6 | 0 | 3 | 3 |
| `mcp` | 6 | 0 | 1 | 5 |
| `validation_builders_scripttest` | 5 | 0 | 2 | 3 |
| `api` | 5 | 0 | 1 | 4 |
| `pipeline_catalog_mods` | 5 | 0 | 2 | 3 |
| `processing` | 5 | 0 | 2 | 3 |
| `step_catalog` | 4 | 0 | 2 | 2 |
| `core_compiler_assembler` | 4 | 0 | 0 | 4 |
| `registry` | 4 | 0 | 2 | 2 |
| `steps_framework` | 4 | 0 | 0 | 4 |
| `validation_alignment` | 3 | 0 | 0 | 3 |
| **Total** | **70** | **3** | **19** | **48** |

### Confirmed findings by lens

| Lens | Count | What it means |
|---|---:|---|
| `silent_failure` | 20 | broad/bare `except` returning a sentinel indistinguishable from "absent" |
| `drifting_duplication` | 14 | same table/constant/rule copied across files; some **already diverged** |
| `fragile_string_convention` | 14 | name/path/import built by ad-hoc slicing/casing/substring; breaks on compound acronyms |
| `hidden_mutable_state` | 11 | singletons / class caches keyed once, never re-keyed or reset |
| `hardcoded_assumption` | 5 | hardcoded path/account/default that breaks across deployment modes |
| `missing_guard` | 5 | missing None/shape/init guard → cryptic error or invariant violation |
| `order_dependence` | 1 | logic silently depending on dict/insertion order |

### The three HIGH findings (fire today)

1. **`core_base/step_interface.py:374` — shallow variant merge breaks production builders.**
`from_yaml` merges variant overrides at the *section* level (`{**base_spec, **variant_spec}`), so a variant whose `spec.dependencies` lists a **subset** of the base **replaces the whole dict**, dropping the omitted ports. This fires on shipped data: `load_step_interface('RiskTableMapping', job_type='training')` and `'calibration'` both **raise `ValidationError`** (the dropped `hyperparameters_s3_uri` violates the contract↔spec `_sync_and_align` invariant), breaking `RiskTableMappingStepBuilder`'s variant path. `batch_transform.step.yaml`'s training variant hits the identical failure (drops `model_name`). → **deep/recursive merge + a test that loads every `.step.yaml` × every declared variant.**
2. **`cli/catalog_cli.py:115` (+16 sites) — swallowed exceptions exit 0.** All 17 inline
`except Exception` handlers log and fall through, returning `None`; Click ignores return values, so the process exits 0 even when the catalog lookup failed.
3. **`cli/compile_cli.py:363` — `return 1` never sets the exit code.** `compile_pipeline` returns
integer codes that Click discards, so DAG/config/validation/**upsert/start** failures all exit 0. Automation treats a failed SageMaker upsert/start as success — the most dangerous consequence in the review.

Both CLI HIGHs are fixed by adopting the **already-existing** `_shared.safe_cli_command` decorator (`_shared.py:56-87`, which correctly `raise SystemExit(1)` after logging) — the codebase already uses it in `dag_cli.py`; `catalog_cli`/`compile_cli` simply never adopted it.

## Systemic themes

This note synthesizes 70 adversarially-confirmed brittleness findings spanning all 15 reviewed subsystems (step_catalog, core_compiler_assembler, core_config_fields, core_base, core_deps_utils, registry, validation_alignment, validation_builders_scripttest, api, mcp, cli, pipeline_catalog_mods, processing, steps_framework, steps_scripts). Five systemic patterns recur across subsystem boundaries; each is far more tractable to fix at the pattern level than site-by-site.

---

### Theme 1: Silent error swallowing — broad `except` that hides failures behind benign-looking defaults

**The dominant pattern by count.** A `try/except Exception` (often a bare `except:`) catches everything, logs at most a WARNING, and returns a sentinel that is indistinguishable from a legitimate empty/missing result.

**Spans (subsystems / sites):**
- **cli** — `catalog_cli.py:115` (+16 more inline handlers) and `compile_cli.py:363` (+ all the `return 1` sites). **Both HIGH.** These don't just hide a message — they swallow the *exit code*. Click ignores function return values, so 17 catalog commands and every compile/upsert/start error path exit 0. CI gating on exit status treats real failures (and SageMaker upsert/start failures) as success.
- **validation_builders_scripttest** — `universal_test.py:504` bare `except:` defaults `step_type='Unknown'`, which makes the sagemaker-methods sub-check report green (medium).
- **core_compiler_assembler** — `dag_compiler.py:385` bare `except Exception:` collapses real catalog errors into `'UNKNOWN'` with no log (low).
- **registry** — `step_names.py:115/549` fallback hides UnifiedRegistryManager degradation; workspace introspection helpers return `[]/0/False` on any error (medium/low).
- **mcp** — `dag.py:257`, `catalog.py:120`, `server.py:84` swallow metadata/framework/SDK-import errors into `{}`/`None`/misleading hint (all low; best-effort enrichment, but no warning surfaced).
- **api** — `pipeline_dag_resolver.py:258/294`, `dag_config_factory.py:303` (low; intentional fallbacks but error-vs-empty conflation).
- **core_deps_utils** — `hybrid_path_resolution.py:223` outer catch logs at ERROR, returns None (low).
- **processing** — `custom_bpe_tokenize_processor.py:190`, `pipeline_iterable_datasets.py:450`, `categorical_validation_processor.py:148`, `dialogue_processor.py:114` (low–medium). The categorical validation one (medium) is the worst: a broken custom rule becomes a no-op that bypasses strict/filter semantics, letting bad rows through.
- **steps_scripts** — `dummy_data_loading.py:256` three bare `except: pass` in format sniffing (low).

**Shared root cause:** the codebase consistently conflates three distinct outcomes — "legitimately empty/absent," "expected degradation," and "genuine error" — into a single return sentinel, with logging too quiet (WARNING or nothing) to distinguish them. Compounded in the CLI by a framework-semantics mismatch: Click never converts Python return values to exit codes.

**Systemic fix:**
1. **CLI (highest leverage):** adopt the *already-existing* `_shared.safe_cli_command` decorator (`_shared.py:56-87`, which correctly `raise SystemExit(1)` after logging) across `catalog_cli` and `compile_cli`, replacing all inline handlers and `return 1`s. This single change fixes both HIGH findings and removes ~17+10 duplicated handlers.
2. **Everywhere else:** establish a convention that a swallowed exception always emits `logger.warning(..., exc_info=True)` (or appends a `warnings` entry on `ToolResult`), so "error" is never silently identical to "absent." Where the result feeds correctness (categorical validation strict mode), re-raise rather than swallow.

---

### Theme 2: Drifting duplicated tables / logic — the same constant or selection rule copied across files with no single source of truth

**Spans (subsystems / sites):**
- **step_catalog** — `JOB_SUFFIXES` (`step_catalog.py:1478`) vs `known_job_types` (`spec_discovery.py:699`) **already disagree** (`inference`/`evaluation` vs `model`), producing inconsistent variant classification (medium). `BASE_CONFIGS` (`step_catalog.py:1451`) duplicated in `universal_test.py:1342` (low).
- **api / core_config_fields** — `configuration_generator.py:110` MRO name-substring inheritance check vs `dag_config_factory.py` `issubclass()` — **already diverged into a live bug**: the substring literal `'BaseProcessingStepConfig'` never matches the real class `ProcessingStepConfigBase`, so the processing branch is dead and processing-specific base fields are dropped on the fallback path (medium).
- **pipeline_catalog_mods** — registration/payload/package selection-by-typename duplicated in `generator.py:423-435` and `registration_helper.py:226-233`, plus last-one-wins overwrite (medium x2 / low).
- **steps_scripts** — `CONTAINER_PATHS` duplicated across 21 scripts (`lightgbm/xgboost/pytorch_model_inference` byte-identical, medium); UNICODE quote maps duplicated in two Bedrock modules (low).
- **validation_alignment** — `VALIDATION_RULESETS` vs `STEP_TYPE_SPECIFIC_VALIDATION_RULES` (`step_type_specific_rules.py:35`) **already diverge** on `CradleDataLoading`/`MimsModelRegistrationProcessing` (low).
- **core_compiler_assembler** — `EXECUTION_S3_PREFIX` literal duplicated (`pipeline_assembler.py:202` vs `dag_compiler.py:36`) (low).
- **steps_framework** — `builders/__init__.py` and `configs/__init__.py` `__all__` drifted from `step_names.yaml` (7 missing builders) (low).
- **mcp** — `_require_config_exists` duplicated in `compile.py:31` vs `shared.py:162` (low); hardcoded namespace list in `registry.py:76` drifts from filesystem (low).

**Shared root cause:** no canonical home for cross-cutting constants (job types, abstract base-config set, container paths, parameter names, config-classification rules). Each consumer re-declares its own copy; some have already drifted, with `configuration_generator.py:110` being a *latent correctness bug today*, not just a future risk.

**Systemic fix:** create a small set of single-source constants/helpers and import them everywhere — e.g. canonical `JOB_TYPES` and `BASE_CONFIGS` in `naming.py` (or a `constants` module); a shared `classify_registration_configs(configs) -> (registration, payload, package)` helper; replace all name-substring/MRO inheritance checks with `issubclass(...)` against imported base classes. For genuinely-self-contained container scripts (CONTAINER_PATHS, quote maps) where importing is undesirable, add **CI contract-conformance tests** asserting each script's table matches its declared `.step.yaml`/contract, so drift is caught mechanically.

---

### Theme 3: Fragile string conventions — naming derived by ad-hoc slicing, casing, or substring matching instead of a centralized resolver

**Spans (subsystems / sites):**
- **step_catalog** — naive `"".join(word.capitalize() ...)` snake→Pascal at `step_catalog.py:1382` and `:1494` produces `PytorchTraining`/`Xgboost...` instead of canonical `PyTorchTraining`/`XGBoost...`, orphaning file_components for the pytorch/xgboost/lightgbm family into phantom snake_case index entries (medium). Offset-based `_extract_step_name` at `:1407` silently drops misnamed files (low).
- **core_base** — `builder_base.py:567` maps env var→config attr via `env_var.lower()` with no explicit mapping, silently dropping required env vars (medium). `config_base.py:406` regex Pascal→snake mangles compound acronyms (`XGBoostModelEval` → `xg_boost_model_eval`) but only on the unregistered-class fallback (low).
- **validation_alignment** — `dependency_validator.py:442/463` single-`_` split + naive capitalize in the fallback resolver (low x2).
- **api** — `dag_config_factory.py:240` case-insensitive substring match for config classes (low, Method-4 last resort).
- **validation_builders_scripttest** — `universal_test.py:1224` `replace('StepBuilder','')` removes all occurrences vs suffix-only helper (low).
- **steps_scripts** — `payload.py:66` substring model-type detection (`'trimodal' in model_class`) (low).
- **mcp** — hardcoded namespace strings (`registry.py:76`) (low; also a Theme-2 instance).

**Shared root cause:** name transformations (snake↔Pascal, suffix stripping, env-var mapping, class resolution) are reimplemented inline at each call site instead of routed through the registry or a single naming module. Compound acronyms (PyTorch, XGBoost, LightGBM) are the recurring failure trigger because every ad-hoc transform mishandles them.

**Systemic fix:** route **all** name conversions through `naming.py` (the `parts_to_pascal` helper already exists and produces the correct `PyTorchTraining`) and the registry as the single source of truth. For env-var resolution, add an explicit contract-level `env_var_mappings` dict and validate at builder init rather than guessing via `.lower()`. Replace substring class matching with exact/`issubclass` resolution (overlaps Theme 2).

---

### Theme 4: Hidden mutable state — process-lifetime singletons and class-level caches keyed once, never re-keyed or reset

**Spans (subsystems / sites):**
- **core_config_fields** — `get_unified_config_manager()` (`unified_config_manager.py:588/604`) caches first `workspace_dirs` and ignores later ones (medium, duplicate finding pair). `get_inheritance_aware_field_generator()` (`inheritance_aware_field_generator.py:466`) caches first `project_id` (medium). `ConfigClassStore._classes` shared class dict (`__init__.py:84`, low).
- **registry** — module-level `STEP_NAMES`/`CONFIG_STEP_REGISTRY`/etc. bound once at import (`step_names.py:211`); external importers capture by value and go stale after `set_workspace_context()` (medium). Unlocked lazy `_get_registry_manager` (`step_names.py:108`, low).
- **core_compiler_assembler** — `DynamicPipelineTemplate.CONFIG_CLASSES` class-level dict mutated by first instance (`dynamic_template.py:99`, low — benign due to complete-registry merge).
- **core_deps_utils** — module-global `_hybrid_resolution_metrics` (`hybrid_path_resolution.py:113`, low); `factory.py:40` thread-local components never cleared (low).
- **processing** — `categorical_label_processor.py:38` mutates label→id map per-row; under `num_workers>0` each forked worker diverges, corrupting labels (medium — only on the `strict=False` fallback config).
- **steps_framework** — `interfaces/__init__.py:39` interface `_cache` never invalidated (low).
- **steps_scripts** — `bedrock_batch_processing.py:1598` second-precision timestamps as implicit uniqueness key (low collision risk).

**Shared root cause:** the "get-or-create singleton" idiom is applied to objects whose identity actually depends on a parameter (`workspace_dirs`, `project_id`), and module-level snapshots are exported as if they were live, workspace-aware APIs when import-by-value freezes them. Blast radius is limited mainly because the package-only default deployment has no workspaces, so most stale-state paths don't fire today.

**Systemic fix:** key every cache by its real identity (`tuple(workspace_dirs)`, `(workspace_dirs, project_id)`), or drop the singleton and construct per request (these objects are cheap). For the registry, stop re-exporting captured dict objects — route workspace-aware reads through accessor functions or PEP 562 module-level `__getattr__`. For the label processor, require a pre-supplied label list with `strict=True` under multi-worker, or raise when `process()` would mutate inside a worker.

---

### Theme 5: Missing initialization / missing guards — attributes set only on a secondary path, or shapes/inputs unvalidated

**Spans (subsystems / sites):**
- **validation_builders_scripttest** — `universal_test.py:55` never sets `self.single_builder_mode` (only `from_builder_class` does); direct construction → AttributeError, and the bundled self-tests ERROR / raise TypeError on the legacy path (medium). `script_execution_registry.py:156` `self._node_specs` set only in `initialize_from_dependency_matcher` (low; prod path always inits first).
- **core_base** — `step_interface.py:374` shallow `{**base, **variant}` merge at the *section* level replaces whole `dependencies`/`outputs` dicts, dropping omitted ports. **HIGH** — exercised by shipped `risk_table_mapping.step.yaml` and `batch_transform.step.yaml`; `load_step_interface('RiskTableMapping', job_type='training')` raises ValidationError, breaking a production builder's variant path.
- **core_compiler_assembler** — `dynamic_template.py:388` `get_resolution_preview()` error branch omits the `resolutions` key, latent KeyError for external callers (low).
- **core_config_fields** — `step_catalog_aware_categorizer.py:103` `ProcessingStepConfigBase` import failure falls back to `object`, making `isinstance` universally True (low); `:498` `__non_serializable_` in-band marker collision (low).
- **core_deps_utils** — `dependency_resolver.py:106` returns `{}` for unknown vs no-dependency step (low, intentional).
- **steps_scripts** — `payload.py:835` `generate_csv_payload` no shape guard on `input_vars` (low, single internal caller).

**Shared root cause:** invariants are established on one construction/initialization path but not the others (refactors that moved init into `from_builder_class`), and merge/parse routines assume the happy-path input shape. The `step_interface.py:374` case is the standout: a shallow merge silently violates the spec/contract alignment invariant.

**Systemic fix:** initialize all instance attributes in `__init__` regardless of construction path (and fix/refresh the stale self-tests). For `step_interface.from_yaml`, implement a **deep/recursive merge** of nested spec sections (or require variants to restate the full dependency/output set) plus a test that loads every `.step.yaml` with each declared `job_type` variant and asserts construction succeeds. Use shape-consistent return dicts (always include the documented keys) and sentinel objects instead of in-band string markers.

---

### Risk Ranking (by aggregate severity-weighted exposure × likelihood to bite)

1. **Theme 1 — Silent error swallowing (HIGHEST).** Owns both standalone HIGH findings (`catalog_cli.py:115`, `compile_cli.py:363`), the largest site count (~15+), and the most dangerous real-world consequence: CI/automation believing a failed SageMaker compile/upsert/start succeeded. Likely to bite in any scripted/gated environment.
2. **Theme 5 — Missing init / missing guards.** Contains the one HIGH *correctness* bug that fires on shipped data (`step_interface.py:374` breaks `RiskTableMapping` and `BatchTransform` variant loading from a production builder), plus a medium that breaks the validation harness self-tests. High impact, narrower surface.
3. **Theme 2 — Drifting duplicated tables.** No HIGH sites, but it has the most *already-realized* drift, including a live wrong-result bug (`configuration_generator.py:110` drops processing fields) and several mediums. High maintenance-debt accrual rate.
4. **Theme 3 — Fragile string conventions.** Mostly medium/low and largely confined to metadata/fallback paths (loading still works via discovery components), but the pytorch/xgboost/lightgbm family is affected in `step_catalog` and env-var dropping in `builder_base.py:567` is a real medium.
5. **Theme 4 — Hidden mutable state.** Several mediums but blast radius is bounded today by the package-only default (no workspaces), so most stale-state paths are latent rather than active.

---

### Already Partially Addressed vs Untouched

**Partially addressed (groundwork landed, finish the migration):**
- **Theme 3** — the `naming.py` single-source-of-truth helper *already exists* (`parts_to_pascal(['pytorch','training']) -> 'PyTorchTraining'`), and the verdict notes confirm `_derive_step_name` already tries the registry first. The remaining work is mechanical: point the two `step_catalog` conversions (`:1382`, `:1494`) and the validation fallbacks at the existing helper rather than re-implementing the naive join.
- **Theme 1 (CLI)** — the correct pattern is *already implemented and in use elsewhere*: `_shared.safe_cli_command` (`_shared.py:56-87`) and `dag_cli.py:51-54/73/88` both `raise SystemExit(1)`. `catalog_cli` and `compile_cli` simply never adopted it — the fix is to migrate to existing infrastructure, not invent it.
- **Theme 1 (MCP)** — `_require_sdk()` (`server.py:33-41`) already does `raise RuntimeError(...) from exc`; `main()` at `:84` is an inconsistent regression of that same in-house pattern.

**Untouched (no canonical home or invariant exists yet):**
- **Theme 2** — no canonical constants module for `JOB_TYPES`/`BASE_CONFIGS`/CONTAINER_PATHS/parameter names, and no `is_abstract`/job-type source in the registry to key off; the duplicated registration-config classifier has no shared helper.
- **Theme 4** — no cache-keying convention and no `reset()`/clear hooks; module-level registry snapshots are still exported as if workspace-aware.
- **Theme 5** — `step_interface.from_yaml` still does a shallow section-level merge with no deep-merge and no per-variant load test; `single_builder_mode` init is still path-dependent.

---

### Recommended Remediation Order (best risk reduction per unit effort)

1. **CLI exit-code fix (Theme 1, both HIGH, low effort).** Migrate `catalog_cli.py` and `compile_cli.py` to the existing `safe_cli_command` decorator / `raise SystemExit(1)`. Removes ~27 duplicated handlers, fixes the dangerous "false success in CI for SageMaker upsert/start," and uses infrastructure that already exists. Highest payoff, smallest risk.
2. **`step_interface.from_yaml` deep merge (Theme 5, HIGH, medium effort).** Implement recursive merge of nested `spec.dependencies`/`outputs` plus a test loading every `.step.yaml` × every declared variant. Unblocks the already-broken `RiskTableMapping`/`BatchTransform` production variant paths. Self-contained, high impact.
3. **Finish the `naming.py` migration (Theme 3, medium severity, low effort).** Repoint `step_catalog.py:1382/1494` and the validation fallbacks at the existing `parts_to_pascal`. Cheap because the helper is already written and verified; restores file_components for the pytorch/xgboost/lightgbm family.
4. **Centralize the drifting constants + fix the inheritance check (Theme 2, includes a live bug, medium effort).** Promote `JOB_TYPES`/`BASE_CONFIGS` to `naming.py`/constants, reconcile the `step_catalog` vs `spec_discovery` membership difference, replace `configuration_generator.py:110` name-substring checks with `issubclass(...)` (fixes the dropped-processing-fields bug), and add CI conformance tests for the per-script CONTAINER_PATHS/quote maps that must stay self-contained.
5. **Establish swallowed-exception logging convention (rest of Theme 1, broad-but-shallow effort).** Add `exc_info=True` warnings / `ToolResult` warnings to the MCP, api, registry, and processing sites; re-raise in `categorical_validation_processor.py:148` strict mode. Lower per-site risk but large surface; do as a sweep once the convention is decided.
6. **Cache-keying + init hygiene (Themes 4 & 5 remainder, low individual risk).** Key `unified_config_manager`/`field_generator` singletons by their real identity, add `self.single_builder_mode = False` / `self._node_specs = {}` in `__init__`, and route registry workspace reads through accessors. Mostly latent today (package-only default), so lowest urgency — but cheap to land alongside related touches.

**Key cross-theme leverage:** `naming.py`/a constants module is the shared remediation surface for Themes 2 and 3, and `_shared.safe_cli_command` is the shared surface for the highest-risk slice of Theme 1 — both already exist, so the highest-value fixes are *adoption of existing infrastructure*, not new design.

## Related notes

- [[deployment_portability_analysis_step_catalog_import_failures]] — prior deployment-mode portability analysis (Theme 3/Theme 4 territory).
- [[2025-11-24_processing_step_path_resolution_cache_poisoning_fix]] — a previously-fixed instance of Theme 4 (hidden mutable state / cache poisoning) in path resolution.
- [[2025-09-19_importlib_usage_systemic_deployment_portability_analysis]] — systemic deployment-portability via importlib (overlaps Theme 4 registry snapshots).
- [[hybrid_registry_code_redundancy_analysis]] — registry-layer redundancy (Theme 2/Theme 4).
- [[validation_system_complexity_analysis]] — validation subsystem complexity (context for `validation_alignment` / `validation_builders_scripttest` findings).
- [[step_catalog_system_integration_analysis]] — step_catalog integration context.

## Appendix — all 70 confirmed findings by subsystem

Ordered by highest-severity-first within each subsystem; subsystems ordered by peak severity then finding count. Each finding carries its file:line, the realistic condition under which it bites, and the targeted fix — all **post-adversarial-verification**.

### `core_base` — 3 findings (🔴1 🟠1 🟡1)

- **🔴 HIGH · drift-dup** — Shallow merge of variant spec/contract overrides can silently drop or add nested keys
  - `src/cursus/core/base/step_interface.py:374`
  - **Bites when:** from_yaml merges variant overrides with `data['spec'] = {**(data.get('spec') or {}), **variant['spec']}` (line 374). The ** union is at the SECTION level (keys: dependencies/outputs/step_type), so when a variant's `spec.dependencies` lists a SUBSET of the base dependencies, the entire base `dependencies` dict is REPLACED, dropping the omitted ones. This is exercised by real, shipped interface files: src/cursus/steps/interfaces/risk_table_mapping.step.yaml base has deps {input_data, hyperparameters_s3_uri, model_artifacts_input} but every variant (training/validation/testing/calibration) lists only {input_data, model_artifacts_input}, dropping hyperparameters_s3_uri. Because the contract still declares hyperparameters_s3_uri as an input, the _sync_and_align model_validator then RAISES ValidationError ('Contract inputs missing from spec dependencies: {hyperparameters_s3_uri}'). Verified: load_step_interface('RiskTableMapping', job_type='training') and job_type='calibration' both raise ValidationError. RiskTableMappingStepBuilder (builder_risk_table_mapping_step.py:60) calls this with config.job_type, so the variant path is broken for a production builder. batch_transform.step.yaml's training variant hits the identical failure (drops model_name).
  - **Fix:** Implement a deep/recursive merge for the nested spec sections (dependencies/outputs) so a variant overrides individual ports rather than replacing the whole dict, OR require variant specs to restate the full dependency/output set. Add a test that loads every .step.yaml with each declared variant and asserts construction succeeds.
- **🟠 MED · fragile-string** — Environment variable name resolution uses fragile lowercase conversion
  - `src/cursus/core/base/builder_base.py:567`
  - **Bites when:** _get_environment_variables maps contract env var names to config attributes via env_var.lower() (lines 567, 587) with no explicit mapping. If a contract declares S3_BUCKET_NAME but the config attribute is `bucket` (not `s3_bucket_name`), the required var is silently dropped from the returned dict — the code only logs a log_warning (lines 580-582) and continues; it does NOT raise. The script then runs without the env var and fails downstream. The convention is genuinely brittle: it assumes a 1:1 UPPER<->lower attribute-name correspondence that the codebase does not enforce.
  - **Fix:** Add an explicit env_var->config_attr mapping (e.g. a contract-level env_var_mappings dict) and validate at builder init that every required_env_vars entry resolves to a real config attribute, raising instead of warning-and-skipping.
- **🟡 LOW · fragile-string** — PascalCase-to-snake_case regex produces wrong names for compound acronyms
  - `src/cursus/core/base/config_base.py:406`
  - **Bites when:** _derive_step_name's regex fallback (lines 406-409) mangles compound acronyms: verified XGBoostModelEval -> 'xg_boost_model_eval' and PyTorchModel -> 'py_torch_model'. (Note: the finder's claimed output 'x_g_boost_model_eval' is itself wrong; the actual output is 'xg_boost_model_eval' because the first regex groups consecutive capitals.) The fragility is real but heavily mitigated: this is Strategy 2, a fallback only reached when the class is NOT found in the registry (lines 379-391 try the registry first and return early on a hit). The module docstring (lines 370-372) explicitly acknowledges this and steers all real classes through the registry. So a wrong name only occurs for an unregistered config class, which would already be a misconfiguration.
  - **Fix:** Keep the registry as the source of truth (already done). Optionally drop the regex fallback entirely and raise a clear error for unregistered classes, or maintain an explicit acronym allowlist, rather than relying on regex.

### `cli` — 2 findings (🔴2 🟠0 🟡0)

- **🔴 HIGH · silent-failure** — Swallowed exceptions prevent exit code propagation in catalog_cli commands
  - `src/cursus/cli/catalog_cli.py:115`
  - **Bites when:** All 17 inline `except Exception as e:` blocks in catalog_cli (e.g. lines 115-117, 177-179, 263-265, 327-329, 380-382, 435-437, 471-473, 531-533, 574-576, 639-641, 693-695, 745-747, 785-787, 852-854, 914-916, 1009-1011, 1104-1106) echo the error and call logger.error but never re-raise. A Click command returns None on this path, and Click ignores command return values, so the process exits 0 even when the catalog lookup failed. CI/scripts gating on exit code treat real failures (broken catalog, missing step) as success.
  - **Fix:** catalog_cli should adopt the existing `safe_cli_command` decorator already defined in _shared.py:56-87 (which correctly re-raises SystemExit(1) after logging, and passes through SystemExit/click control-flow), instead of hand-rolling inline try/except per command. Replacing the 17 inline handlers with the decorator both fixes exit codes and removes duplication.
- **🔴 HIGH · silent-failure** — Click return values are ignored, exit codes silently 0
  - `src/cursus/cli/compile_cli.py:363`
  - **Bites when:** compile_pipeline returns integer codes (return 1 at lines 125, 138, 144, 158, 207, 266, 287, 307, 342, 368; return 0/1 at 202; return 0 at 363) but Click discards command return values, so the process exits 0 regardless. DAG-load failures, config-load failures, validation failures, compile failures, output-save failures, upsert failures, and execution-start failures all exit 0. Automated pipelines treating exit 0 as success will believe a failed compile/upsert/start succeeded.
  - **Fix:** Replace every error `return 1` with `raise SystemExit(1)` (and drop the no-op `return 0`), or wrap the command in the existing _shared.safe_cli_command decorator pattern. For validate-only, `return 0 if valid else 1` at line 202 should become `if not validation_result.is_valid: raise SystemExit(1)`.

### `core_deps_utils` — 7 findings (🔴0 🟠1 🟡6)

- **🟠 MED · fragile-string** — Path structure matching allows false positives for nested paths
  - `src/cursus/core/utils/generic_path_discovery.py:244`
  - **Bites when:** _matches_full_path (lines 244-258) only verifies the trailing path components equal expected_parts; for a bare folder name (len==1) it returns True unconditionally (line 246). Combined with _search_downward / _search_upward returning the FIRST match (lines 152-154, 199-201), if two sibling trees contain the same project_root_folder name, discovery returns whichever the traversal hits first, which may not be the user's intended project. This is the known cross-deployment path-resolution brittleness area.
  - **Fix:** When multiple candidates match, disambiguate by proximity to a reference point (cwd or package root) or warn that multiple matches were found. Prefer exact/closest match rather than first-found. Make _search_downward collect all matches rather than early-return.
- **🟡 LOW · missing-guard** — Dependency resolution returns empty dict instead of raising when consumer spec not found
  - `src/cursus/core/deps/dependency_resolver.py:106`
  - **Bites when:** resolve_step_dependencies returns {} when the consumer step has no registered specification (lines 106-108), conflating 'step unknown' with 'step has no dependencies'. A caller cannot distinguish the two from the return value alone, though a warning is logged.
  - **Fix:** Document the contract that {} means 'no resolvable dependencies (incl. unknown step)'. If callers need to distinguish, expose registry.get_specification() check separately. Raising would break resolve_all_dependencies (line 64) which deliberately tolerates missing specs, so do not raise here.
- **🟡 LOW · mutable-state** — Factory's thread-local components not cleared on error or context exit
  - `src/cursus/core/deps/factory.py:40`
  - **Bites when:** get_thread_components (lines 38-44) lazily creates and caches components in threading.local with no clearing API, so a pooled/reused thread keeps its first components (including a cached resolver and registry) across unrelated tasks. Stale step specs could persist within a reused thread.
  - **Fix:** Add a clear_thread_components() helper or prefer the existing dependency_resolution_context() context manager (lines 47-58) which already clears cache and contexts on exit. Document that thread-pool callers must reset between tasks.
- **🟡 LOW · silent-failure** — Silent fallback chain in hybrid path resolution swallows real file-not-found errors
  - `src/cursus/core/utils/hybrid_path_resolution.py:223`
  - **Bites when:** resolve_path wraps all five strategies in a broad `except Exception` (line 223) that logs at ERROR and returns None, so genuine OS errors (PermissionError, symlink loops) are indistinguishable from 'path not found'. Realistic but low-impact: each strategy only uses .exists()/.is_dir() which return False rather than raise on most permission cases, and a None return causes the caller to fall through to other resolution or fail loudly elsewhere — it does not produce a WRONG path silently.
  - **Fix:** Optionally log the full traceback (logger.exception) instead of just the message so transient FS errors are diagnosable. Re-raising is not warranted given the multi-strategy fallback design.
- **🟡 LOW · mutable-state** — Module-level global metrics state not reset between test runs or deployments
  - `src/cursus/core/utils/hybrid_path_resolution.py:113`
  - **Bites when:** _hybrid_resolution_metrics (line 113) is module-global and accumulates cumulative counts for process lifetime with no reset function exposed. This is purely observability/diagnostic data (success/failure rates, timings) — it does not affect path resolution results, so there is no functional brittleness or wrong-result risk.
  - **Fix:** If metrics are ever consumed by monitoring, add an optional clear_metrics() helper. Otherwise document as intentionally cumulative. Low priority.
- **🟡 LOW · silent-failure** — Config file parsing tolerates invalid JSON and silently continues discovery
  - `src/cursus/core/utils/project_discovery.py:124`
  - **Bites when:** _summarize_config_file (lines 120-147) catches a JSON parse failure and records it in summary.error (line 126), returning a ConfigSummary with node_count=0. A caller reading only node_count/nodes (and not .error) would mistake a corrupt config for an empty one.
  - **Fix:** No code change strictly needed — the error IS captured in summary.error and surfaced via to_dict() (lines 58-61). If stricter behavior is wanted, callers should check the error field; document that node_count=0 with a non-null error means parse failure, not empty config.
- **🟡 LOW · mutable-state** — Fallback registry manager initialization not guarded against concurrent access
  - `src/cursus/registry/step_names.py:108`
  - **Bites when:** _get_registry_manager (lines 111-124) lazily initializes the module-global _global_registry_manager with a check-then-set and no lock, so two threads racing on first use could each construct a manager and one wins. In practice the manager is read-mostly (created from static registry data) and last-writer-wins simply discards a redundant instance; divergent-state corruption is unlikely.
  - **Fix:** If multi-threaded first-use is a real concern, guard with a module-level threading.Lock using double-checked locking. Low priority given the manager is effectively immutable after construction.

### `steps_scripts` — 7 findings (🔴0 🟠2 🟡5)

- **🟠 MED · hardcoded** — CVE-like hardcoded ARN account IDs in CodeArtifact token retrieval
  - `src/cursus/steps/scripts/bedrock_batch_processing.py:52`
  - **Bites when:** _get_secure_pypi_access_token() hardcodes account 111111111111 in the AssumeRole ARN (line 52) and domainOwner 222222222222 (line 65, also embedded in the index URL at line 116). If these accounts change/decommission, secure-PyPI installs break with AssumeRole/access-denied. Important mitigations: this path is gated by USE_SECURE_PYPI which defaults to 'false' (line 27), so the default behavior uses public PyPI; and the role name is dynamically suffixed with the caller's own account (line 53). It is a maintainability/hardcoding concern, not a 'CVE' and not a credential leak.
  - **Fix:** Make the STS account and CodeArtifact domainOwner configurable via env vars with the current values as defaults, and document them in deployment config. Optionally fall back to public PyPI with a clear warning if AssumeRole fails, since secure PyPI is already opt-in.
- **🟠 MED · drift-dup** — Duplicated CONTAINER_PATHS across multiple inference scripts can drift independently
  - `src/cursus/steps/scripts/lightgbm_model_inference.py:359`
  - **Bites when:** CONTAINER_PATHS = {MODEL_DIR, EVAL_DATA_DIR, OUTPUT_EVAL_DIR} is byte-for-byte identical in lightgbm_model_inference.py (359), xgboost_model_inference.py (254), pytorch_model_inference.py (275), and is repeated (with per-step variations) in ~18 other scripts. There is no shared source of truth, so if the SageMaker processing contract changes for these inference steps, each script must be edited independently and one could drift to stale paths. Note: scripts are self-contained-by-design (each must run standalone in a container), so duplication is partly intentional, and each CONTAINER_PATHS aligns with its own per-step script contract — not a single global contract — which bounds the 'silent misrouting' risk.
  - **Fix:** If the self-contained constraint allows it, factor the shared inference paths into a single small constant module imported by the inference scripts; otherwise add a contract-conformance test that asserts each script's CONTAINER_PATHS matches its declared .step.yaml/script contract so drift is caught in CI.
- **🟡 LOW · drift-dup** — Duplicated UNICODE quote patterns in two Bedrock processing modules can drift
  - `src/cursus/steps/scripts/bedrock_batch_processing.py:234`
  - **Bites when:** UNICODE_DOUBLE_QUOTES / UNICODE_SINGLE_QUOTES (and GERMAN_OPEN_QUOTE_PATTERN) are duplicated verbatim in bedrock_batch_processing.py (lines 234-250) and bedrock_processing.py (lines 244-260). If one file's quote-normalization map is updated and the other is not, batch vs real-time response parsing could normalize quotes inconsistently. Both modules are self-contained container scripts by design, so duplication is partly intentional; the practical blast radius is cosmetic-to-moderate (parsing variance) rather than data loss.
  - **Fix:** If self-containment permits, extract the quote maps and GERMAN_OPEN_QUOTE_PATTERN into one shared constants module; otherwise add a test asserting the two modules' maps are equal to catch drift in CI.
- **🟡 LOW · mutable-state** — timestamp-based S3 key generation has collision risk under high concurrency
  - `src/cursus/steps/scripts/bedrock_batch_processing.py:1598`
  - **Bites when:** job_name = f'cursus-bedrock-batch-{timestamp}' (line 1597-1598) and the input S3 key input_{timestamp}.jsonl (lines 1398-1404) use only second-precision timestamps. If two batch jobs are launched within the same second AND share the same output_bucket/output_prefix, the output S3 prefix and job name could collide, risking overwrite or a duplicate-job-name error. In practice each SageMaker processing job typically owns a distinct framework-provided input_prefix/output_prefix, which scopes the keys and makes same-second collisions unlikely.
  - **Fix:** Append microsecond precision (`%Y%m%d-%H%M%S-%f`) or a short random/uuid suffix to job_name and the S3 key to eliminate same-second collisions regardless of prefix scoping.
- **🟡 LOW · silent-failure** — Bare except blocks swallow all exceptions including transient errors
  - `src/cursus/steps/scripts/dummy_data_loading.py:256`
  - **Bites when:** detect_file_format() uses three bare `except: pass` blocks (lines 256, 263, 270) when content-sniffing a file whose extension is unrecognized. These cannot distinguish 'not this format' from a transient read error, and the function returns 'unknown' which then leads read_data_file() to raise ValueError. Note: the PRIMARY detection path is extension-based (lines 243-249) and returns before any sniffing; content-sniffing only runs for files with no/unknown extension, which is uncommon in this pipeline.
  - **Fix:** Optionally narrow to `except (pd.errors.ParserError, OSError, ValueError)` and log the exception type so a transient AWS/permission error during sniffing is distinguishable from a genuine format mismatch. Low priority given extension-based detection is the primary path and failures already surface as a downstream ValueError.
- **🟡 LOW · fragile-string** — detect_model_type() has fragile substring matching for model_class without exact field validation
  - `src/cursus/steps/scripts/payload.py:66`
  - **Bites when:** detect_model_type() (lines 66-82) classifies via substring checks ('trimodal' in model_class, 'multimodal' in model_class) and key-presence checks. Compound/typo'd model_class strings could misclassify. The None-value concern is largely mitigated: downstream field_order construction (lines 444-470) only appends fields that are truthy (e.g. `if primary_text:`), so a present-but-None text key adds nothing to field_order and validate_payload_completeness() likewise only requires truthy fields.
  - **Fix:** Prefer explicit model_class membership (e.g. against a known set) over substring containment, and document the accepted model_class values. The None-validation part of the finding is unnecessary because field_order/validation already skip falsy fields.
- **🟡 LOW · missing-guard** — Missing validation of input_vars structure in generate_csv_payload() can cause KeyError downstream
  - `src/cursus/steps/scripts/payload.py:835`
  - **Bites when:** generate_csv_payload() branches on isinstance(input_vars, dict) (line 835) and otherwise does `{name: vtype for name, vtype in input_vars}` (line 838), which would raise ValueError/TypeError if input_vars is None or contains non-2-tuples. There is no entry-point guard. However this is an internal function with a single in-repo caller (line 1075) that passes the parsed var_type_list, so malformed input would be a programming error surfaced as an immediate, loud exception (not silent) — not WRONG RESULTS or DATA LOSS.
  - **Fix:** Add a lightweight entry guard (e.g. raise a descriptive ValueError if input_vars is not a dict or list of 2-element pairs) to improve the error message; functionally low priority since the single internal caller already supplies a well-formed var_type_list and bad input fails fast.

### `core_config_fields` — 6 findings (🔴0 🟠3 🟡3)

- **🟠 MED · mutable-state** — Field generator global singleton cache ignores project_id changes
  - `src/cursus/core/config_fields/inheritance_aware_field_generator.py:466`
  - **Bites when:** get_inheritance_aware_field_generator() caches the first InheritanceAwareFieldGenerator (capturing project_id and workspace_dirs at line 484) and reuses it; a later call with a different project_id returns the stale generator with the wrong project context.
  - **Fix:** Key the cache by (tuple(workspace_dirs), project_id), or drop the singleton and let callers construct per request (the generator is cheap).
- **🟠 MED · mutable-state** — Global singleton cache not reset between config lifecycle boundaries
  - `src/cursus/core/config_fields/unified_config_manager.py:588`
  - **Bites when:** get_unified_config_manager() caches the first UnifiedConfigManager instance (with its workspace_dirs) at module scope and never updates it; later calls with different workspace_dirs silently return the stale instance.
  - **Fix:** Key the cache by tuple(workspace_dirs) or expose a reset hook; or document that get_unified_config_manager() ignores workspace_dirs after first call and require explicit construction for workspace-scoped use (as the public APIs already do).
- **🟠 MED · order-dep** — Singleton cache ignores subsequent workspace_dirs parameters
  - `src/cursus/core/config_fields/unified_config_manager.py:604`
  - **Bites when:** Duplicate of finding #1: get_unified_config_manager() caches workspace_dirs from the first call; a first call with no workspace_dirs caches UnifiedConfigManager([]) and a later call with workspace_dirs returns the stale instance.
  - **Fix:** Same as finding #1: key the cache by tuple(workspace_dirs) or require direct construction for workspace-scoped contexts. Consolidate with finding #1.
- **🟡 LOW · mutable-state** — Fallback ConfigClassStore uses shared mutable class variable
  - `src/cursus/core/config_fields/__init__.py:84`
  - **Bites when:** The fallback ConfigClassStore._classes is a class-level dict shared process-wide; registrations persist and accumulate across the process lifetime.
  - **Fix:** Acceptable as-is; if isolation is needed, gate on always preferring the real adapter and optionally expose a clear() classmethod for test/reload scenarios.
- **🟡 LOW · fragile-string** — Non-serializable marker collision with legitimate field values
  - `src/cursus/core/config_fields/step_catalog_aware_categorizer.py:498`
  - **Bites when:** A field whose JSON-serialized form is a string literally beginning with '__non_serializable_' would be misread as the non-serializable placeholder and replaced with the raw first-config value.
  - **Fix:** Track non-serializable values with a sentinel object or a parallel set of field names rather than an in-band string prefix, eliminating any value-space collision.
- **🟡 LOW · silent-failure** — Processing base class fallback to object silently corrupts categorization
  - `src/cursus/core/config_fields/step_catalog_aware_categorizer.py:103`
  - **Bites when:** If ProcessingStepConfigBase import fails, processing_base_class becomes object, making isinstance(c, object) always True, so ALL configs are classified as processing and none as non-processing — corrupting the processing/non-processing split (and the is_cross_type derivation that depends on it).
  - **Fix:** On ImportError, leave processing_base_class as a sentinel that matches nothing (e.g. a private marker class) instead of object, or raise; this keeps the processing list empty rather than capturing everything.

### `mcp` — 6 findings (🔴0 🟠1 🟡5)

- **🟠 MED · fragile-string** — config.requirements tool schema inconsistent with compile/execdoc tools
  - `src/cursus/mcp/tools/config.py:308`
  - **Bites when:** config.requirements advertises only an inline `dag` property with `required=['dag']` and `additionalProperties=False` (config.py lines 317-322, using the local _DAG_SCHEMA at 277-305), but its handler calls `_build_dag(args)` = shared.resolve_dag (line 79), which is documented and coded to accept EITHER `dag_file` OR inline `dag` (shared.py 100-121). Because the schema both omits `dag_file` from properties and sets additionalProperties=False, passing `dag_file` to config.requirements fails registry._validate_args TWICE: 'missing required argument: dag' AND 'unknown argument: dag_file' (registry.py 147-154) — before the handler ever runs. So a capability the handler fully supports is unreachable through this tool, and it diverges from every compile.* tool, which spread DAG_INPUT_PROPS (advertising both `dag` and `dag_file`) and require only config_file.
  - **Fix:** Make config.requirements use the shared fragment like compile does: `from .shared import DAG_INPUT_PROPS`, set `properties={**DAG_INPUT_PROPS}` and drop the `required=['dag']` constraint (let resolve_dag enforce 'exactly one of dag/dag_file' at lines 112-121). Delete the local _DAG_SCHEMA (277-305). This re-aligns the advertised contract with the handler and with the rest of the DAG-taking tools.
- **🟡 LOW · fragile-string** — Namespace list in registry hardcoded instead of discovered from filesystem
  - `src/cursus/mcp/registry.py:76`
  - **Bites when:** _collect_tooldefs (registry.py 76-85) hardcodes `namespaces = ['catalog','dag','config','compile','validate','execdoc','pipeline_catalog','info']`. Adding a new tools/<ns>.py module that exposes TOOLS will NOT register unless the developer also appends the string here, so the registry can silently under-report the available tool set.
  - **Fix:** Optionally auto-discover with pkgutil over the tools package: `import pkgutil, cursus.mcp.tools as pkg; namespaces = [m.name for m in pkgutil.iter_modules(pkg.__path__) if m.name not in ('shared',)]`, keeping the existing try/except skip so an import failure in one namespace still logs-and-continues. Lower priority — if kept static, add a brief comment that new tool modules must be appended here and keep it in sync with tools/__init__.py's namespace docstring.
- **🟡 LOW · silent-failure** — Bare exception handler in server.py main() obscures missing SDK errors
  - `src/cursus/mcp/server.py:84`
  - **Bites when:** server.main() (lines 81-85) does `try: import anyio; from mcp.server.stdio import stdio_server except Exception: raise RuntimeError(_SDK_HINT)`. The bare `except Exception` (no `from exc`) catches and discards the original traceback, so a non-import failure (e.g. an internal error raised at import time of a broken installed mcp build) is replaced with the misleading 'install the mcp SDK' hint with no chaining.
  - **Fix:** Match the existing _require_sdk pattern: `except Exception as exc: raise RuntimeError(_SDK_HINT) from exc` (or narrow to `except (ImportError, ModuleNotFoundError) as exc`). Chaining `from exc` preserves the original traceback so a non-missing-SDK failure is still diagnosable.
- **🟡 LOW · silent-failure** — Silent exception in framework detection breaks agent reasoning without warning
  - `src/cursus/mcp/tools/catalog.py:120`
  - **Bites when:** catalog._step_info (lines 117-121) does `try: data['framework'] = catalog.detect_framework(step_name) except Exception: data['framework'] = None`. A genuine detection bug and a 'this step has no framework' both yield framework=None with no warning, so an agent cannot tell the field is null because detection failed vs. because none applies.
  - **Fix:** Surface the failure as a warning rather than swallowing silently: `except Exception as exc: data['framework'] = None; warnings.append(f"framework detection failed for {step_name}: {exc}")` and pass warnings to ToolResult.success. Keeps the field optional while telling the agent why it is null.
- **🟡 LOW · drift-dup** — Duplicate config file existence checker violates single-source-of-truth
  - `src/cursus/mcp/tools/compile.py:31`
  - **Bites when:** compile.py defines its own _require_config_exists() (lines 31-35) with logic identical to require_config_exists() in shared.py (lines 162-165) — both do `if not config_file or not os.path.exists(config_file): raise ToolError(... code="not_found")`. compile.py already imports DAG_INPUT_PROPS and CONFIG_INPUT_PROPS from shared.py (lines 19-23) but pointedly does NOT import the existing config-existence helper, instead re-implementing it and calling _require_config_exists 5 times (lines 46, 88, 112, 164, 186). If one copy is later enhanced (symlink handling, caching, better error text) the other drifts silently.
  - **Fix:** Delete _require_config_exists from compile.py (lines 31-35) and add `require_config_exists` to the existing `from .shared import (...)` block, then rename the 5 call sites. Zero behavior change; just collapses the duplicate.
- **🟡 LOW · silent-failure** — Silent exception swallows metadata extraction failures, masking real problems
  - `src/cursus/mcp/tools/dag.py:257`
  - **Bites when:** dag._deserialize (lines 254-258) does `try: file_meta = PipelineDAGReader.extract_metadata(path) except Exception: file_meta = {}`. A real extraction failure (corrupt embedded metadata block, permission quirk) is silently collapsed to empty {}, so the returned stats/metadata/created_at fields are blank with no warning telling the agent extraction actually failed vs. the file simply having no metadata.
  - **Fix:** Append a warning instead of fully swallowing: `except Exception as exc: warnings.append(f"could not read embedded DAG metadata: {exc}")` and pass warnings to ToolResult.success, so the agent can distinguish 'no metadata' from 'metadata read failed'. Optionally log at DEBUG.

### `validation_builders_scripttest` — 5 findings (🔴0 🟠2 🟡3)

- **🟠 MED · silent-failure** — Bare except swallows step_type resolution failure silently
  - `src/cursus/validation/builders/universal_test.py:504`
  - **Bites when:** Line 502-505 uses a bare `except:` that defaults step_type to 'Unknown'. With 'Unknown', _check_sagemaker_methods sets expected_methods=[] so `passed = len(expected_methods)==0 or ...` => True (verified). The sagemaker_methods sub-check therefore reports green even when get_sagemaker_step_type raised. The bare `except:` also swallows KeyboardInterrupt/SystemExit and logs nothing.
  - **Fix:** Replace `except:` with `except Exception as e:` and log at debug/warning. Note step_type='Unknown' is intentionally treated as non-critical here (the outer except at line 539 also returns passed=True deliberately), so no status change is required - the real fix is just narrowing the except and logging.
- **🟠 MED · missing-guard** — Missing initialization of single_builder_mode attribute in refactored UniversalStepBuilderTest
  - `src/cursus/validation/builders/universal_test.py:55`
  - **Bites when:** __init__ (lines 55-108) never sets self.single_builder_mode; it is assigned only in from_builder_class (line 1317). Accessing it on a directly-constructed instance raises AttributeError (verified). The bundled unittest test_refactored_initialization (line 1608) asserts tester.single_builder_mode on a normally-constructed instance and would ERROR; the same test's legacy path (line 1621) calls UniversalStepBuilderTest(MockBuilder) which now passes the builder class into the workspace_dirs slot and raises TypeError: 'ABCMeta object is not iterable' (verified) deep in StepCatalog init.
  - **Fix:** Add self.single_builder_mode = False in __init__ (e.g., near line 72). Separately, fix or remove the stale unittest test_refactored_initialization: the legacy line 1621 UniversalStepBuilderTest(MockBuilder, ...) no longer matches the refactored constructor signature and must use from_builder_class(MockBuilder).
- **🟡 LOW · fragile-string** — Script name extraction via string replacement ignores embedded 'StepBuilder' substrings
  - `src/cursus/validation/builders/universal_test.py:1224`
  - **Bites when:** Lines 1224 and 1240 use builder_name.replace('StepBuilder', '') which removes ALL occurrences, so a name like 'StepBuilderTraining' would become 'Training' or 'XStepBuilderStepBuilder' -> 'X' (verified). This differs from the suffix-only _extract_base_name (line 1250). The fragility is real in principle but cannot trigger with current data: all 48 registry builder_step_name values were verified to contain exactly one trailing 'StepBuilder' and none embedded.
  - **Fix:** Replace builder_name.replace('StepBuilder','') at lines 1224 and 1240 with the existing _extract_base_name() helper (suffix-only strip) for consistency and future-proofing against any embedded-substring builder names.
- **🟡 LOW · missing-guard** — Uninitialized _node_specs attribute accessed without guard
  - `src/cursus/validation/script_testing/script_execution_registry.py:156`
  - **Bites when:** __init__ (lines 44-65) never sets self._node_specs; it is assigned only in initialize_from_dependency_matcher (line 86). Calling get_node_config_for_resolver (line 156) or _get_expected_inputs (line 356) before initialize_from_dependency_matcher raises AttributeError. CONFIRMED empirically. BUT in every actual code path init always precedes access: api.py and script_dependency_matcher.py call registry.initialize_from_dependency_matcher() before collect_user_inputs_with_registry_coordination(), which is the sole live caller of get_node_config_for_resolver (input_collector.py:138). And _get_expected_inputs (line 356) is only reachable via sequential_state_update/_update_node_state/_apply_message_passing, which NO caller invokes (dead path).
  - **Fix:** Initialize self._node_specs = {} in __init__ (alongside self._state at lines 50-57) so the attribute always exists. This is the clean fix; guard checks are unnecessary once it is always present.
- **🟡 LOW · drift-dup** — Semantic message passing mapping hardcodes output->input name pairs that can drift
  - `src/cursus/validation/script_testing/script_execution_registry.py:388`
  - **Bites when:** _get_semantic_mapping hardcodes semantic_rules (lines 388-395) mapping output names to candidate input names. If contract/spec naming conventions change, this table is not updated. The 'fail silently' part is mild: _apply_message_passing always also emits the prefixed mapping (lines 336-337) as a guaranteed fallback, and direct-name matching (line 321) handles exact matches, so a stale semantic table degrades to (still-available) prefixed/direct mapping rather than breaking message passing.
  - **Fix:** If/when the sequential_state_update path is wired up, derive semantic rules from step contracts/specs (already loaded into _node_specs) rather than a static dict. Until then this is dormant code; low priority.

### `api` — 5 findings (🔴0 🟠1 🟡4)

- **🟠 MED · drift-dup** — Drifting duplication of inheritance checks across multiple files
  - `src/cursus/api/factory/configuration_generator.py:110`
  - **Bites when:** configuration_generator.py uses MRO name-substring matching while dag_config_factory.py uses issubclass(). The two implementations have ALREADY diverged into a latent bug: configuration_generator._inherits_from_processing_config (line 122-128) checks `'BaseProcessingStepConfig' in base_class.__name__`, but the real base class is named ProcessingStepConfigBase (steps/configs/config_processing_step_base.py:27). 'BaseProcessingStepConfig' is not a substring of 'ProcessingStepConfigBase' (different word order), so this method ALWAYS returns False for genuine processing configs. They then fall through to _inherits_from_base_config (which matches the literal 'BasePipelineConfig' correctly) and call _generate_with_base_inheritance, using base_config instead of base_processing_config — dropping processing-specific base fields.
  - **Fix:** Replace the MRO name-substring checks in configuration_generator.py (lines 122-128 and 145-151) with issubclass(config_class, ProcessingStepConfigBase) / issubclass(config_class, BasePipelineConfig), or consolidate both files onto a single shared helper. At minimum fix the literal: it should match 'ProcessingStepConfigBase', not 'BaseProcessingStepConfig'.
- **🟡 LOW · silent-failure** — Silent catch-all exception handlers swallow real discovery failures
  - `src/cursus/api/dag/pipeline_dag_resolver.py:258`
  - **Bites when:** _discover_step_contract (line 238-261) and _discover_step_contract_legacy (line 263-296) both catch all Exceptions and return None, conflating 'no contract exists for this step' with 'an error occurred while loading the contract'. A genuine load error is logged only at WARNING and produces the same None as a legitimately contract-less step.
  - **Fix:** Optional: split the bare 'No contract found' (return None) path from the 'error during load' path so the latter logs at ERROR and is distinguishable in telemetry. Not required for correctness since the generic fallback is intentional and logged.
- **🟡 LOW · hardcoded** — Version drift tolerance without schema migration path
  - `src/cursus/api/dag/pipeline_dag_serializer.py:21`
  - **Bites when:** SCHEMA_VERSION='1.0.0', _SUPPORTED_MAJOR_VERSIONS={'1'}. _validate_data (line 347-354) rejects mismatched major versions and tolerates a missing version for backward compatibility, with no migration/adapter layer for a future v2.
  - **Fix:** When (and only when) a major version bump is planned, add a normalize/migrate adapter before validation. Optionally tighten the missing-version branch to warn. No urgent change needed today.
- **🟡 LOW · fragile-string** — Case-insensitive substring matching for config class resolution is too loose
  - `src/cursus/api/factory/dag_config_factory.py:240`
  - **Bites when:** Method 4 (line 239-246) does 'if canonical_lower in class_name.lower()' substring match and returns the first hit, so a short canonical name could match an unintended longer class name, with result depending on dict iteration order.
  - **Fix:** Tighten Method 4 to require class_name.lower().startswith(canonical_lower) or equality after stripping 'Config', and log a warning if more than one class matches. Given Methods 1-3 cover all registered steps, impact is limited to unregistered-name guesses.
- **🟡 LOW · silent-failure** — Broad exception handler in config auto-discovery masks import/module errors
  - `src/cursus/api/factory/dag_config_factory.py:303`
  - **Bites when:** _get_available_config_classes (line 260-305) wraps the unified-manager call (line 267) and the ConfigAutoDiscovery fallback (line 285-301) in broad try/excepts; the inner failure logs at ERROR and returns {} (line 303-305). An empty dict means no config classes are available, so downstream mapping silently maps nothing.
  - **Fix:** Return None or raise ConfigurationError on discovery-system failure so callers can distinguish 'system failed' from 'genuinely no configs', rather than {}. Logging is already adequate.

### `pipeline_catalog_mods` — 5 findings (🔴0 🟠2 🟡3)

- **🟠 MED · drift-dup** — Registration config lookup duplicated and unsynchronized across modules
  - `src/cursus/mods/exe_doc/generator.py:424`
  - **Bites when:** The registration/payload/package selection-by-typename logic is genuinely duplicated: generator.py lines 423-435 selects registration_cfg/payload_cfg/package_cfg, and registration_helper.py lines 226-233 (_create_execution_doc_config) repeats the identical 'registration in name and not payload' / 'payload in name' / 'package in name' selection. Both must stay in sync if the config naming convention changes. Downgraded from high to medium: the duplication is real and maintenance-fragile, but the generator-side selection of payload_cfg/package_cfg is largely vestigial in the actual call (generator passes the configs to create_execution_doc_config_with_related_configs, which re-wraps them and re-runs the helper-side selection at 226-233), so behavior is governed by the helper copy. A divergence would cause maintenance bugs rather than an immediate runtime wrong-result.
  - **Fix:** Extract a single helper function (e.g., RegistrationHelper.classify_configs(configs)) returning (registration, payload, package) and call it from both generator.py and _create_execution_doc_config. Single source of truth.
- **🟠 MED · drift-dup** — Multiple registration configs overwrites previous config (last-one-wins)
  - `src/cursus/mods/exe_doc/registration_helper.py:229`
  - **Bites when:** In _create_execution_doc_config (lines 226-233) the loop assigns registration_cfg = cfg unconditionally for every config whose type name contains 'registration' (and not 'payload'). If two such configs exist, the last one in dict iteration order silently wins, with no warning. The same unconditional last-one-wins selection also exists in generator.py lines 423-435. This can select the wrong registration config. Downgraded from high to medium: a pipeline normally has exactly one registration config, so the multi-match condition is atypical, and dict iteration order in CPython 3.7+ is insertion-ordered (deterministic), reducing the 'unpredictable selection' risk the finder asserted.
  - **Fix:** If more than one matching config is encountered, log a warning naming both, and pick deterministically (e.g., by a documented priority) rather than silently keeping the last. Document the single-registration-config assumption.
- **🟡 LOW · fragile-string** — Hardcoded class name strings break under refactoring or aliasing
  - `src/cursus/mods/exe_doc/generator.py:345`
  - **Bites when:** _fill_cradle_configurations (line 345) and _fill_registration_configurations (line 407) locate their helper by iterating self.helpers and comparing helper.__class__.__name__ == "CradleDataLoadingHelper" / "RegistrationHelper". If the class were renamed, the lookup returns None and the block is skipped (cradle/registration configs never populated). Confirmed as fragile, but downgraded from high to low: the helpers are constructed directly in __init__ (self.cradle_helper, self.registration_helper at lines 64-65) and are the very objects placed into self.helpers. The string search is dead/redundant — a rename of the class would also rename the constructor call and break the build long before this string mismatch could silently fire. The realistic 'silently skipped' outcome requires renaming only the class name while leaving the hardcoded string, which the co-located construction makes unlikely.
  - **Fix:** Drop the name-based search entirely and use self.cradle_helper / self.registration_helper directly (they already exist). If a search is desired, use isinstance against the imported classes.
- **🟡 LOW · fragile-string** — Fragile 50% word overlap matching mis-resolves config names
  - `src/cursus/mods/exe_doc/generator.py:259`
  - **Bites when:** _names_match (lines 233-260) splits names into word sets and returns True when intersection >= 50% of the smaller set. This is permissive and could map a step to the wrong config on partial word overlap. Confirmed the heuristic is loose, but downgraded from medium to low: this runs only as a fallback-of-a-fallback inside _get_config_for_step (lines 226-229) — reached ONLY when config_resolver.resolve_config_for_step RAISES an exception AND there is no exact name match in self.configs. In the normal path the StepConfigResolverAdapter handles resolution and _names_match is never consulted. The mis-match also only causes a wrong config if a wrong config happens to clear the 50% bar before the right one in iteration order.
  - **Fix:** Require the config name (sans separators) to be a substring of the step name, or require the leading token to match. Log which config matched via _names_match so collisions are observable. Prefer relying on the step catalog resolver.
- **🟡 LOW · drift-dup** — Multiple payload/package configs overwritten (last-one-wins) in registration_helper
  - `src/cursus/mods/exe_doc/registration_helper.py:231`
  - **Bites when:** Same loop at lines 226-233 also assigns payload_cfg (line 231) and package_cfg (line 233) unconditionally, keeping only the last match. Real overwrite pattern. Downgraded from medium to low: in the live call path (generator._fill_registration_configurations -> create_execution_doc_config_with_related_configs, lines 343-350) the configs_dict passed in contains AT MOST one 'payload' and one 'package' entry (keyed 'registration'/'payload'/'package'), so the loop cannot encounter duplicates there. The duplicate risk only exists if _create_execution_doc_config is called with raw self.configs containing multiple payload/package configs, which is atypical. CPython dict order is deterministic, contradicting the 'order not stable across versions' claim.
  - **Fix:** Collect matches into lists and select deterministically or warn on multiples; document the one-payload/one-package assumption.

### `processing` — 5 findings (🔴0 🟠2 🟡3)

- **🟠 MED · mutable-state** — CategoricalLabelProcessor updates mutable state during processing without synchronization
  - `src/cursus/processing/categorical/categorical_label_processor.py:38`
  - **Bites when:** When a label processor runs in update-on-new / non-strict mode, it assigns new IDs by mutating instance state during per-row process() calls. Under DataLoader num_workers>0 the dataset (and processor) is forked per worker, so each worker learns label->id mappings independently and in different row orders, producing inconsistent IDs across workers for the same category and corrupting training labels. NOTE: the production training path uses strict=True (no mutation); divergence only occurs on the strict=False / update_on_new fallback.
  - **Fix:** For multi-worker use, require a pre-supplied label list and use strict=True (the existing safe path), and document that the update_on_new/strict=False mode is single-process fit-only. Alternatively raise if process() would mutate while running under a worker (torch.utils.data.get_worker_info() is not None).
- **🟠 MED · silent-failure** — Exception handling in categorical_validation_processor suppresses validation failures
  - `src/cursus/processing/categorical/categorical_validation_processor.py:148`
  - **Bites when:** When a user-supplied custom validation rule raises (e.g. KeyError/TypeError on a misconfigured column or dtype), the except logs at error level and continues without flagging or filtering any rows for that rule. In 'strict'/'filter' strategies the caller expects bad rows to raise or be removed; instead all rows pass that rule unchecked, allowing corrupted data through.
  - **Fix:** Re-raise by default (or only swallow when an explicit skip_broken_rules=True flag is set). At minimum, in 'strict' strategy a rule that errors should raise, since the caller asked to fail loudly.
- **🟡 LOW · silent-failure** — Bare except in __len__ silently returns 0 for shard loading errors
  - `src/cursus/processing/datasets/pipeline_iterable_datasets.py:450`
  - **Bites when:** `__len__` catches `except Exception` (not a bare except) when estimating length from the first shard and returns 0. A broken first shard makes len() report 0. But this is only a documented best-effort ESTIMATE for an IterableDataset; the same shard would raise loudly in `__iter__`/`_load_shard` during actual iteration, so training does not silently proceed on a broken dataset.
  - **Fix:** Log a warning before returning 0 (e.g. `logger.warning('Could not estimate dataset length from first shard: %s', e)`) so the degraded estimate is visible; keep the non-raising behavior since __len__ is optional for IterableDataset.
- **🟡 LOW · silent-failure** — Bare except block silently swallows conversion errors
  - `src/cursus/processing/text/custom_bpe_tokenize_processor.py:190`
  - **Bites when:** Lines 185-191 wrap `str(data)`/`.lower()` in a bare `except:` that returns empty string. A swallowed error would be silently treated as missing/empty text. In practice the try body only runs for non-dict, non-str inputs (numeric/None/NaN) and only calls str() and str.lower(), which essentially never raise, so the realistic blast radius is tiny.
  - **Fix:** Replace bare `except:` with `except (TypeError, ValueError) as e:` and log at debug/warning so a genuinely unexpected conversion failure is visible, even though the fallback behavior (empty text) is acceptable.
- **🟡 LOW · silent-failure** — Dialogue chunker truncates long messages silently without tracking
  - `src/cursus/processing/text/dialogue_processor.py:114`
  - **Bites when:** When a single message's token count exceeds max_tokens, it is truncated to max_tokens with no log or metric, so downstream cannot know information was dropped. This is intentional defensive behavior (prevents OOM from oversized sequences) but provides no observability into how much/how often data is lost.
  - **Fix:** Add a debug/warning log (and optionally a counter) when a message is truncated, including original vs kept token counts, so silent information loss is observable. Optionally gate with a track_truncations flag.

### `step_catalog` — 4 findings (🔴0 🟠2 🟡2)

- **🟠 MED · fragile-string** — Fragile snake_case to PascalCase conversion in two places skips COMPOUND_ACRONYMS
  - `src/cursus/step_catalog/step_catalog.py:1382`
  - **Bites when:** _resolve_to_canonical_name_for_indexing (line 1382) and _resolve_to_canonical_name (line 1494) both use `"".join(word.capitalize() for word in step_name.split("_"))`, which produces 'PytorchTraining'/'Xgboost...' instead of the canonical 'PyTorchTraining'/'XGBoost...'. CORRECTION to the finder: this does NOT make the step unreachable and does NOT break get_builder/contract/spec loading. _load_registry_data() pre-populates _step_index with the correct canonical key first, and load_builder_class/load_contract_class/load_spec_class delegate to the separate discovery components + YAML interface, not to _step_index.file_components. The real, narrower impact: when builder/config files like builder_pytorch_training_step.py are indexed, _resolve_to_canonical_name_for_indexing returns None (since 'PytorchTraining' is not in the registry which has 'PyTorchTraining'), so the file attaches to a phantom snake_case index entry 'pytorch_training' instead of merging into the canonical 'PyTorchTraining' StepInfo. Consequences: get_step_info('PyTorchTraining').file_components is empty for these frameworks, and find_step_by_component returns the snake_case name. list_available_steps still returns the correct PascalCase name (registry entry already present; the phantom snake_case entry is correctly dropped during _deduplicate_and_filter_concrete_steps). Affects the pytorch_*/xgboost_*/lightgbm_* family among the 47 builder + 48 config files.
  - **Fix:** Replace both naive conversions with the centralized helper: `from .naming import parts_to_pascal` then `pascal_candidate = parts_to_pascal(step_name.split('_'))`. This correctly maps pytorch->PyTorch, xgboost->XGBoost, etc., so file components attach to the canonical registry entry.
- **🟠 MED · drift-dup** — Hardcoded JOB_SUFFIXES list in _is_job_type_variant can drift from allowed job types
  - `src/cursus/step_catalog/step_catalog.py:1478`
  - **Bites when:** STRONGER than the finder stated: this is not merely a future drift risk — two job-type lists in the SAME subsystem ALREADY disagree. step_catalog.py:1478 JOB_SUFFIXES = ['_calibration','_testing','_training','_validation','_inference','_evaluation'] is used to filter variants out of list_available_steps; spec_discovery.py:699 known_job_types = ['training','validation','testing','calibration','model'] is used to classify variants. step_catalog has 'inference'+'evaluation' but not 'model'; spec_discovery has 'model' but not 'inference'/'evaluation'. So a '<base>_inference' spec is treated as a job-type variant (filtered) by list_available_steps but classified as 'default' (not a known job type) by spec_discovery, and a '<base>_model' file is the inverse. Impact is inconsistent variant detection/filtering, not data loss.
  - **Fix:** Define a single module-level canonical JOB_TYPES/JOB_SUFFIXES constant in one place (e.g., naming.py or a new constants module) and import it in both step_catalog._is_job_type_variant and spec_discovery._find_job_type_variants_in_dir so the two paths can never disagree. Reconcile the current membership difference ('model' vs 'inference'/'evaluation') as part of the fix.
- **🟡 LOW · drift-dup** — BASE_CONFIGS hardcoded list in _deduplicate_and_filter_concrete_steps may be incomplete
  - `src/cursus/step_catalog/step_catalog.py:1451`
  - **Bites when:** BASE_CONFIGS = {'Base','Processing'} at step_catalog.py:1451 is duplicated as {'Processing','Base'} in src/cursus/validation/builders/universal_test.py:1342. The two literals are currently identical in content, so there is no present-tense behavioral divergence; the risk is future drift if one is updated and the other isn't, or if a new abstract base config (e.g., a new ProcessingStepConfigBase-style entry) is added and not appended to both. Impact if it drifts: a base/abstract step could leak into list_available_steps() or be inconsistently filtered between the catalog and the validation harness.
  - **Fix:** Promote BASE_CONFIGS to a single shared module-level constant (e.g., in naming.py or a constants module) and import it in both step_catalog.py and validation/builders/universal_test.py so the exclusion set cannot drift.
- **🟡 LOW · fragile-string** — _extract_step_name assumes hardcoded filename conventions and will fail silently on edge cases
  - `src/cursus/step_catalog/step_catalog.py:1407`
  - **Bites when:** _extract_step_name (lines 1405-1426) uses fixed slice offsets (name[:-9] for _contract, name[8:-5] for builder_..._step, name[7:-5] for config_..._step) and returns None for any file not matching the exact prefix/suffix convention. The caller (line 1290-1301) does `if step_name:` and silently skips on None with no warning, so a misnamed component file (e.g. pytorch_training_builder.py missing the _step suffix, or a builder file lacking the builder_ prefix) is silently omitted from the index. Impact is limited: such a file simply isn't indexed; loading still works via the separate discovery components / YAML interface, and misnamed files are arguably an authoring error. No wrong results, only a missing-component silent skip.
  - **Fix:** When a file lives in a component directory but fails to match the expected pattern, emit a logger.warning instead of silently returning None, so naming typos are surfaced. Optionally centralize the prefix/suffix rules in naming.py rather than inline slice arithmetic.

### `registry` — 4 findings (🔴0 🟠2 🟡2)

- **🟠 MED · mutable-state** — Stale module-level variables never updated after workspace context changes
  - `src/cursus/registry/step_names.py:211`
  - **Bites when:** Module-level STEP_NAMES/CONFIG_STEP_REGISTRY/BUILDER_STEP_NAMES/SPEC_STEP_TYPES are bound once at import (lines 211-214). registry/__init__.py:27-30 and several modules (e.g. validation/builders/universal_test.py:27, builder_base.py:201, configs/utils.py:588) import these names BY VALUE. _refresh_module_variables() (582-587) rebinds the step_names module globals, but those external bindings still point at the original objects, so they remain stale after set_workspace_context(). The staleness is real; the cause is import-by-value semantics, NOT a scoping bug.
  - **Fix:** Stop exporting the module-level snapshots as the workspace-aware API. Either deprecate the module-level vars and route all workspace-aware reads through get_step_names()/get_config_step_registry()/etc., or implement module-level __getattr__ so `STEP_NAMES` resolves dynamically — and have registry/__init__.py expose accessor functions rather than re-exporting captured dict objects. Note: fixing _refresh_module_variables alone is insufficient because of import-by-value.
- **🟠 MED · silent-failure** — Silent fallback hides UnifiedRegistryManager initialization failure
  - `src/cursus/registry/step_names.py:115`
  - **Bites when:** If UnifiedRegistryManager() raises during __init__ for a reason OTHER than core-registry load failure (e.g. an import error inside manager.py / hybrid.models, or a bug in workspace discovery), step_names.py:120-123 catches it, logs a warning, and substitutes a FallbackManager that serves only core steps. Workspace-defined steps then become silently unavailable with no programmatic signal — callers cannot detect that the hybrid backend degraded.
  - **Fix:** Expose registry health: store the caught exception on a module-level _registry_init_error and add a public is_hybrid_active()/get_registry_health() function so callers can detect that workspace resolution silently degraded. Keep the fallback for resilience but make the degraded state queryable, and raise the log to error level when a fallback is used.
- **🟡 LOW · silent-failure** — Dead @property decorator on module-level function
  - `src/cursus/registry/step_names.py:172`
  - **Bites when:** @property at lines 172-175 decorates a module-level function. @property is a descriptor that only takes effect when assigned to a class attribute; at module scope it merely binds the name STEP_NAMES to a property object that does nothing. That property object is then immediately and unconditionally overwritten at line 211 by `STEP_NAMES = get_step_names()`, so it never has any observable effect at all — there is no 'dynamic vs static' divergence because the property object never survives import.
  - **Fix:** Delete lines 172-175 entirely. The intended 'dynamic STEP_NAMES' behavior cannot be achieved with @property at module scope; if dynamic access is desired, implement module-level __getattr__ (PEP 562). Otherwise document that STEP_NAMES is a static snapshot and steer callers to get_step_names().
- **🟡 LOW · silent-failure** — Silent exception swallowing in workspace utility functions
  - `src/cursus/registry/step_names.py:549`
  - **Bites when:** list_available_workspaces() (549-551), get_workspace_step_count() (561-563) and has_workspace_conflicts() (574-576) each catch Exception, log a warning, and return [] / 0 / False. A manager failure is therefore reported to callers as 'no workspaces' / 'zero steps' / 'no conflicts', indistinguishable from the legitimate empty case.
  - **Fix:** Acceptable as defensive helpers, but if precise health matters, return a richer result (e.g. raise on manager errors or return an object distinguishing error from empty). At minimum, log at error level rather than warning so failures are visible.

### `core_compiler_assembler` — 4 findings (🔴0 🟠0 🟡4)

- **🟡 LOW · hardcoded** — Hardcoded parameter name lookup without fallback breaks if dependency changes
  - `src/cursus/core/assembler/pipeline_assembler.py:202`
  - **Bites when:** pipeline_assembler.py:202 matches the execution-prefix parameter by the hardcoded literal `param.name == 'EXECUTION_S3_PREFIX'`, the same literal produced by dag_compiler.py:36 `ParameterString(name='EXECUTION_S3_PREFIX')`. The string is duplicated across files. If the constant's name drifts (or mods_workflow_core supplies a differently-named parameter), the match fails and set_execution_prefix is never called.
  - **Fix:** Define EXECUTION_S3_PREFIX (and the other parameter names) as a shared constant in one module and import it in both dag_compiler.py and pipeline_assembler.py, so the literal lives in exactly one place; optionally log at INFO when no execution-prefix parameter is matched so the fallback is observable.
- **🟡 LOW · silent-failure** — Bare exception handler swallows builder resolution errors silently
  - `src/cursus/core/compiler/dag_compiler.py:385`
  - **Bites when:** In preview_resolution(), lines 376-386 attempt step_catalog.get_builder_for_step_type(config_type) and catch any exception with a bare `except Exception:` that sets config_builder_map[config_type] = 'UNKNOWN' without logging the swallowed exception. A real error (catalog corruption, registration bug, runtime failure) is rendered indistinguishable from a legitimately missing builder, with no diagnostic trace.
  - **Fix:** Add `self.logger.warning(f"Builder lookup for {config_type} raised: {e}", exc_info=True)` inside the except block (capture e) before assigning 'UNKNOWN', so genuine errors are visible in logs while preview still completes.
- **🟡 LOW · mutable-state** — Class-level CONFIG_CLASSES mutated at runtime causes config mismatch across instances
  - `src/cursus/core/compiler/dynamic_template.py:99`
  - **Bites when:** DynamicPipelineTemplate.CONFIG_CLASSES is a class-level mutable dict (line 42). The first instance populates it via `cls.CONFIG_CLASSES = self._detect_config_classes()` (line 99), and the `if not cls.CONFIG_CLASSES` guard (line 97) means subsequent instances with a DIFFERENT config_path never re-detect — they inherit the first instance's detected class set. This is shared mutable class state, a genuine code smell.
  - **Fix:** Use an instance attribute (self.CONFIG_CLASSES set in __init__) or pass detected classes into _load_configs directly, rather than mutating the class object, to eliminate the shared-state smell even though it is currently benign due to the complete-registry merge.
- **🟡 LOW · drift-dup** — Resolution preview return value missing guaranteed 'resolutions' key
  - `src/cursus/core/compiler/dynamic_template.py:388`
  - **Bites when:** DynamicPipelineTemplate.get_resolution_preview() returns `{'nodes': ..., 'resolutions': {...}}` on the success path (lines 361-384) but `{'error': str(e)}` with no 'resolutions' (and no 'nodes') key on the exception path (line 388). A caller that unconditionally reads preview['resolutions'] would KeyError on the error branch.
  - **Fix:** In the except block at line 388 return a shape-consistent dict, e.g. `return {'nodes': len(self._dag.nodes), 'resolutions': {}, 'error': str(e)}`, so the 'resolutions' key is always present and the method's implicit contract holds.

### `steps_framework` — 4 findings (🔴0 🟠0 🟡4)

- **🟡 LOW · drift-dup** — Multiple builders missing from exports despite existing in registry and codebase
  - `src/cursus/steps/builders/__init__.py:104`
  - **Bites when:** Seven builder files have classes registered in step_names.yaml but never imported into builders/__init__.py: PseudoLabelMerge, TokenizerTraining, LightGBMMTModelEval, LightGBMMTModelInference, TemporalSplitPreprocessing, XGBoostModelInference, RedshiftDataLoading. This is genuine namespace drift. The same drift exists in configs/__init__.py (RedshiftDataLoadingConfig, PseudoLabelMergeConfig, etc. not imported). The consequence is NOT 'fails silently at runtime / cannot be discovered' — both builders and configs are discovered via AST directory scans — but the package namespace and __all__ are inconsistent with the registry, which is a real maintainability/drift hazard.
  - **Fix:** Add the missing imports and __all__ entries for namespace/registry parity (builders and configs), OR add an automated test asserting that every step in step_names.yaml has a matching __all__ entry. Treat as hygiene; it does not gate discovery.
- **🟡 LOW · missing-guard** — EdxUploadingStepBuilder conditionally imported but never added to __all__
  - `src/cursus/steps/builders/__init__.py:167`
  - **Bites when:** EdxUploadingStepBuilder is conditionally imported with EDX_UPLOADING_AVAILABLE (lines 54-61), but unlike the other three optional builders it is never appended to __all__. The 'add optional builders' block (lines 160-167) only handles CRADLE_DATA_LOADING_AVAILABLE, REGISTRATION_AVAILABLE, DATA_UPLOADING_AVAILABLE. So even when EdX imports successfully it is invisible to 'import *'. As with the other namespace findings, this does not block AST-based discovery, only star-import/namespace visibility.
  - **Fix:** Add 'if EDX_UPLOADING_AVAILABLE:\n __all__.append("EdxUploadingStepBuilder")' alongside the other three optional appends (after line 167).
- **🟡 LOW · hardcoded** — Hardcoded path using __file__ breaks in non-standard installations
  - `src/cursus/steps/builders/builder_edx_uploading_step.py:204`
  - **Bites when:** Lines 203-207 derive the script path as Path(__file__).parent.parent / 'scripts' / processing_entry_point. This assumes a normal on-disk package layout. In a zipimport/egg/frozen-binary deployment __file__ may point inside an archive or be unavailable, breaking script resolution. This is the fallback branch only — when config.effective_source_dir is set (lines 190-199) __file__ is not used. The path itself is correct for standard installs (builders/ -> parent.parent = steps/ -> scripts/, which exists).
  - **Fix:** Prefer importlib.resources.files('cursus.steps') / 'scripts' / entry_point to locate the packaged scripts directory robustly, or require the script path via config (effective_source_dir already covers the primary path). Low priority absent a zipimport/frozen deployment requirement.
- **🟡 LOW · mutable-state** — Module-level interface cache never cleared, causing stale interface data in long-running processes
  - `src/cursus/steps/interfaces/__init__.py:39`
  - **Bites when:** _cache (line 39) is populated on first load (line 134) and read on every subsequent call (lines 122-123) with no invalidation, TTL, or mtime check. If a .step.yaml were edited on disk in a long-running process, load_interface() would keep returning the stale parsed StepInterface. Realistically low-impact: .step.yaml files are packaged immutable resources shipped with the install, and pipeline compilation processes are short-lived, so on-disk hot-edits during a running process are not a normal scenario.
  - **Fix:** Add a clear_interface_cache() helper (and optionally a reload=True kwarg on load_interface) for tests and dev hot-reload. An mtime check is overkill given the files are packaged resources.

### `validation_alignment` — 3 findings (🔴0 🟠0 🟡3)

- **🟡 LOW · drift-dup** — Implicit type match between VALIDATION_RULESETS and unused STEP_TYPE_SPECIFIC_VALIDATION_RULES can diverge
  - `src/cursus/validation/alignment/config/step_type_specific_rules.py:35`
  - **Bites when:** STEP_TYPE_SPECIFIC_VALIDATION_RULES and VALIDATION_RULESETS are two independent dicts keyed by step type, with overlapping keys and both carrying a 'category' field, but no consistency check. They already diverge: VALIDATION_RULESETS has CradleDataLoading and MimsModelRegistrationProcessing, which STEP_TYPE_SPECIFIC lacks. They serve different purposes (level-enablement vs. required-methods), and missing keys are handled gracefully (None returns), so divergence degrades coverage rather than producing crashes.
  - **Fix:** Add a startup consistency check (in config/__init__.py) that warns when a step type present in VALIDATION_RULESETS is missing from STEP_TYPE_SPECIFIC_VALIDATION_RULES (or vice versa), or generate the shared category field from a single source. Low priority — current divergence does not cause wrong results, only reduced Level-4 method coverage for the two missing keys.
- **🟡 LOW · hardcoded** — Multiple implicit fallback chains for step type resolution converge to incorrect defaults
  - `src/cursus/validation/alignment/validators/dependency_validator.py:441`
  - **Bites when:** _get_canonical_step_name first calls the registry's get_canonical_name_from_file_name (line 433, the documented single source of truth). Only on ValueError does it fall through to string-munging with a hardcoded job_type_suffixes list (lines 445-454). A new job-type suffix not in this list would not be stripped in the fallback, yielding a wrong canonical base. But this is the final fallback, and the same hardcoded suffix list also exists in the primary registry function, so the fallback does not add fragility beyond the registry itself.
  - **Fix:** If the registry already encodes valid job types, have the fallback reuse the registry's suffix source rather than re-hardcoding the list, or remove the redundant fallback. Low priority since the primary registry path handles known job types.
- **🟡 LOW · fragile-string** — Overlapping fallback in spec type to canonical name breaks on underscores or hyphens in names
  - `src/cursus/validation/alignment/validators/dependency_validator.py:463`
  - **Bites when:** Line 442 splits on single underscore and line 463 naively capitalizes; hyphenated or oddly-separated names would parse incorrectly. This only manifests in the final fallback (reached after registry lookup fails), and spec/contract files use the underscore convention, so hyphenated inputs are not expected in practice.
  - **Fix:** If non-underscore separators ever become possible, use re.split(r'[_-]+', spec_file_name). Otherwise low priority — the registry path is tried first and the underscore convention is enforced by file naming.

