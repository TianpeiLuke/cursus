---
tags:
  - project
  - planning
  - step_builder
  - step_catalog
  - design_b
  - classless_discovery
keywords:
  - classless Design-B
  - provider returns
  - registry-walk discovery
  - lazy materialization
  - FZ 31e2 closure gate
  - config-class carve-out
  - failure-mode flip
topics:
  - phased rollout to delete the per-step builder shell classes
language: python
date of note: 2026-06-29
---
# Classless Design-B — Phased Rollout Plan (FZ 31e1d3 / 31e1d3g1)

> **SUPERSEDED (2026-06-29) by [`2026-06-29_factory_shell_migration_master_plan.md`](./2026-06-29_factory_shell_migration_master_plan.md).** This plan scoped only the step-catalog provider seam + a narrow validation chokepoint, and the FZ 31e1d3g3 audit later found it (a) prematurely marked Phase 2 done when only the lazy descriptor half shipped — the registry-walk *materializer* was never built — and (b) under-scoped validation (the checks already misfire on today's shells, so it is a 3-boundary redesign per FZ 31e1d3h, not a chokepoint patch). Read the master plan for the systematic all-risk sequence. This file is retained for the impact-matrix and the per-phase history.

This is the in-repo execution plan for the **classless** arm of the Design-B fork: delete the 45 per-step `<Name>StepBuilder` shell classes; make the step-catalog resolution methods return a PROVIDER CALLABLE instead of a `Type`; and switch discovery from an AST file-walk over `builder_*.py` to a registry-walk over the interface-derived `STEP_NAMES` (48 rows). The design rationale + comprehensive impact analysis live in the abuse_slipbox FZ trail: **FZ 31e1d3** (the fork), **31e1d3c** (the original StepCatalog register), **31e1d3g** (the realized facade/factory adaptation), **31e1d3g1** (the per-module-family impact analysis this plan executes). This document is the *what to touch, in what order, what's gated*.

## Verdict from the impact analysis

The blast radius is CONCENTRATED, not pervasive. step_catalog (the discovery rewrite) and the validation family (real-class contract testing) carry the weight; core is callable-satisfiable + two `__name__` guards; api is trivial; pipeline_catalog is verifiably untouched. The single most important property is NOT an API-shape change — it is the FAILURE-MODE FLIP from silent soft-None omission to a loud compile-time `ValueError` at `pipeline_assembler.py:183/190`, which is desirable but unsafe until the closure gate guarantees registry closure.

## Prerequisite — DONE: the FZ 31e2 triangle-closure gate (Tier-3a)

`tests/registry/test_registry_triangle_closure.py` shipped (commit `3051feed`). It asserts the four corners agree bidirectionally — registry `STEP_NAMES` ↔ `.step.yaml` interfaces ↔ discoverable builders ↔ importable config classes — and converts the silent soft-None drop into a loud CI failure. The 4 SDKDelegation builders are classified SDK-env-only (skip offline, assert present in the SAIS env); `LEGACY_ALIASES` tolerated in the reverse-orphan direction. It already caught + fixed a real registry-integrity bug (PyTorchModel/XGBoostModel config_class). **Design-B MUST NOT merge until this gate is green** — it is the silent-drop safety net the whole refactor balances on.

## Impact matrix (from 31e1d3g1, adversarially verified)

| module family | verdict | breaks-if-unchanged | effort |
|---|---|---|---|
| step_catalog | moderate (the real work) | 1 real: `builder_discovery.py:191/486/545` AST-walk → registry-walk + lazy load | medium-large |
| validation + cli | heavy (under-counted 4th family) | 4: `universal_test.py:292/294` issubclass; the 4 step-type validators' `inspect.getsource`/`__mro__`; `builder_reporter.py:303` `__name__`-derived report filename | medium (single chokepoint) |
| core (assembler/compiler/dynamic_template) | light | 0 hard — `pipeline_assembler.py:190` `builder_cls(**5kwargs)` has NO isinstance gate (callable-satisfiable); `dag_compiler.py` `__name__` is try/except-shielded; `validation.py` is key-only | small (annotations + 2 guards) |
| api (dag/factory) | light | 0 — truthiness existence checks only; `dag_config_factory` already on the function path | trivial |
| pipeline_catalog | none (verified) | 0 — DAG-data only, zero coupling | none |

## Phased plan

### Phase 0 — annotations + `__name__` display de-risk (no behavior change; ships independent of Design-B)
- Re-point the `class.__name__` DISPLAY readers to a canonical step-name source: `dag_compiler.py:~438` (already try/except→"UNKNOWN"; make it `getattr(builder_class, "__name__", "UNKNOWN")`), `dynamic_template.py:~260` debug log, `catalog_cli.py` display, and the LOAD-BEARING `builder_reporter.py:303` (report FILENAME is derived from `__name__` — must read the step name, not the class name).
- Add a public `step_name`/`STEP_NAME` surface so loggers/reporters stop reading `__class__.__name__`.
- Gate: full suite green; the resolved-edge-graph snapshot + the closure gate stay green. Pure no-behavior-change cleanup.

### Phase 1 — provider seam in step_catalog (dual-mode; class IS-A provider)
- Change the return contract of `load_builder_class` (`step_catalog.py:550`), `get_builder_for_config` (`:1504`), `get_builder_for_step_type` (`:1520`) from `Optional[Type]` to a provider callable. A class is ALREADY such a callable (the assembler's `builder_cls(**5kwargs)` at `pipeline_assembler.py:190` has no isinstance gate), so returning classes keeps everything working — dual-mode, behavior-identical.
- `mapping.py:52/84` delegate to `load_builder_class`, so they follow automatically; `resolve_legacy_aliases` is string-only and untouched.
- Gate behind the closure check (Phase-3a). Byte-diff: a built pipeline is identical with class-returns vs provider-returns.

### Phase 2 — discovery rewrite (the long pole)
- Switch the enumeration from the AST file-walk (`builder_discovery.py:159-189`, scans `builder_*.py` for `STEP_BUILDER_BASE_NAMES`) to a registry-walk over `STEP_NAMES`.
- Defer `_load_class_from_file` (`:486-545`, currently eager `getattr(module, class_name)`) into a lazy `provider.get_class()` with a load cache — so AST scanning is decoupled from importlib I/O. Bonus: the 4 SDK builders no longer need importlib at scan time, only at instantiation, so offline discovery stays clean.
- Keep the AST scan as a fallback shim for workspace-local builders (see the decision below).
- Gate: closure check green; resolved-edge-graph snapshot green; the `test_migrated_shells_callable` + universal-validator suites green.

### Phase 3 — validation family chokepoint (heavy, gated, can lag)
- Materialize the class ONCE at the single catalog-fetch chokepoints — `universal_test.py:1035 _get_builder_class_from_catalog` and `step_type_specific_validator.py:341 _get_builder_class` — via `provider.get_class()`. After that, the ~8 downstream Type-reads (`issubclass` at `universal_test.py:292/294`, the 4 step-type validators' `inspect.getsource`/`__mro__`, the `__name__` name-derivation) work unchanged.
- `universal_builder_rules.py:347` already guards `isinstance(builder_class, type)` and is unwired — no work.

## Carve-outs (cannot go classless — leave as real-class)
1. **Config-class discovery (permanent).** `build_complete_config_classes` (`step_catalog.py:382`) / `discover_config_classes` (`:362`) MUST keep returning live `Type` objects — `type_aware_config_serializer.py:358/372` calls `actual_class.model_fields` / `model_validate`, which a bare callable lacks. CRITICAL: real Design-B is BUILDERS-ONLY and never touches config discovery, so the carve-out is honored by simply not changing it.
2. **Universal builder validator (permanent).** It tests the builder's METHOD CONTRACT — you cannot `issubclass`/`getsource`/`__mro__` a function. Handled by the single-chokepoint materialization in Phase 3, not a classless rewrite.
3. **The 4 SDKDelegation builders.** Lazy materialization HELPS them (no scan-time importlib); they stay real classes, instantiated only at build.

## Behavior changes (observable, from 31e1d3g1)
- **Failure-mode flip (headline):** registry-walk makes a registered-but-unloadable step a loud `ValueError` at `pipeline_assembler.py:183` instead of a silent skip. Desirable; SAFE only behind the closure gate (done).
- **Discovery-semantics shift:** AST file-walk discovers ANY conforming `builder_*.py` incl. unregistered/workspace-local ones; registry-walk discovers only registered steps. GAIN: production isolation. REGRESSION: a dev can't drop a local builder and iterate without registering it. **DECISION (open, 31e1d3g1a):** ship an opt-in workspace AST-discovery flag (recommended) vs accept the dev-velocity regression.
- **`class.__name__` collapse** in logs/traces toward `TemplateStepBuilder` — cosmetic except `builder_reporter.py:303` (handled in Phase 0).

## Definition of done
- [x] **3a** FZ 31e2 closure gate green (prerequisite). Commit `3051feed`.
- [ ] **Phase 0** display-name + annotation de-risk; `step_name` surface; no behavior change.
- [ ] **Phase 1** provider seam (dual-mode); byte-diff identical pipeline.
- [ ] **Phase 2** registry-walk discovery + lazy materialization; closure + edge-snapshot green; workspace-discovery decision made.
- [ ] **Phase 3** validation chokepoint materialization; the 4 step-type validators + universal validator green.
- [ ] **carve-outs** config_discovery + universal validator left class-returning; verified untouched.
- [ ] **e2e** a full compile→assemble→build of a multi-step pipeline produces an identical `Pipeline`; `preview_resolution` shows no UNKNOWN.

## Recommendation
DO proceed — gated + phased, only with the closure gate green (it is). Phase 0 ships independently (zero behavior change). The honest cost is concentrated in the step_catalog discovery rewrite (Phase 2, the long pole) and the validation chokepoint (Phase 3). Everything else is annotations + truthiness. The refuted non-work (do NOT spend effort): `dag_compiler.py:438` (try/except-shielded), `pipeline_assembler.py:190` (callable-satisfiable), `get_builder_map` values (key-only consumer), `resolve_legacy_aliases` (string-only), `build_complete_config_classes` (builders-only never touches it), `universal_builder_rules.py:347` (guarded + dead).
