---
tags:
  - project
  - planning
  - step_builder
  - control_panel
  - patterns_section
keywords:
  - patterns section
  - HANDLER_KNOBS migration
  - step_assembly
  - output_path_token
  - direct_input_keys
  - use_step_args derived
  - blueprint-driven injection
topics:
  - making the .step.yaml the blueprint that wires pattern injection per axis
language: python
date of note: 2026-06-28
---
# `patterns:` section — move per-step knob selections into the .step.yaml blueprint (FZ 31e1d3f1)

> Directive (2026-06-28): "avoid hard-wiring each step's implementation by baking patterns into the
> builder template per step. Instead the step interface is the BLUEPRINT that guides the injection /
> combination of patterns along each axis — e.g. output_path_token / step_assembly / direct_input_keys
> / use_step_args added to the step interface, wiring the interface to control behavior."

This closes Gap 1 of the control-panel review (FZ 31e1d3f): the four routing/assembly facts still
lived in Python (`STEP_ASSEMBLY` class attr + `HANDLER_KNOBS` dict), so editing the `.step.yaml`
could NOT change them and the introspection view drifted. After this, the `.step.yaml` is the sole
blueprint; the builder shell is `STEP_NAME` only (for the 35 non-SDK steps).

> **ADDENDUM (2026-06-28, FZ 31e1d3f1b) — `output_path_token` REMOVED, not migrated.** The audit
> found `output_path_token` corresponds to the STEP NAME: 25 steps used the
> `canonical_to_snake(step_type)` default, 17 declared it == that default (redundant), and the 3
> "deviations" (`model_evaluation` / `model_inference` / `active_sampling` / `packaging`) were
> non-standard legacy values, not an intentional shared namespace. Per the directive that these are
> non-standard, ALL `output_path_token` declarations were deleted AND the field itself was removed
> from `PatternsSection` + `ContractSection` + the handler `KnobSpec`s — the output S3 prefix is now
> DERIVED unconditionally as `canonical_to_snake(step_type)` in all three synthesizing handlers
> (Processing/Training/Transform). So of the four facts the directive listed, only `step_assembly` +
> `direct_input_keys` became `patterns:` fields, `include_job_type_in_path` is a `patterns:` knob, and
> `output_path_token` is gone (derived). `use_step_args` was likewise dropped (derived from
> `step_assembly`). Conformance: `test_output_path_token_is_derived_not_declarable` +
> `test_output_prefix_derived_from_step_name`.

## A new top-level `patterns:` section (peer of registry / compute / contract / spec)

```yaml
patterns:
  step_assembly: step_args          # Processing sub-verb: code | step_args | delegation
  output_path_token: xgboost_model_eval   # path segment for synthesized output destinations
  direct_input_keys: [input_path]   # logical names passed straight through (not spec×contract joined)
  include_job_type_in_path: false   # whether config.job_type is a path segment (default true)
```
`patterns:` holds the per-axis STRATEGY-SELECTION knobs (how the handlers combine), distinct from
`contract:` (script-shaped I/O data), `compute:` (the SDK compute object), `registry:` (discovery +
3rd-party deps), and `spec:` (DAG wiring). A new `PatternsSection` Pydantic model validates it.

## Knob inventory + disposition (audited, all 45 builders)

| knob | count | disposition |
|---|---|---|
| `output_path_token` | 20 | → `patterns.output_path_token` (was knob-only or YAML-shadowed) |
| `use_step_args` | 16 | **DROPPED** — derived from `step_assembly` (the `step_args` strategy preset already sets `use_step_args: True`; all 16 set it redundantly on top of `STEP_ASSEMBLY="step_args"`) |
| `direct_input_keys` | 13 | → `patterns.direct_input_keys` |
| `include_job_type_in_path` | 8 | → `patterns.include_job_type_in_path` |
| `STEP_ASSEMBLY` (class attr) | 17 | → `patterns.step_assembly` |
| `make_compute` (lambda) | 4 | **DELETED as dead/broken** — `model_calibration`/`metrics`/`wiki`/`percentile` set `make_compute: lambda b: b._get_processor()` but NONE define `_get_processor` (0 defs) → AttributeError at build; the present knob also BLOCKS the `compute:` descriptor fallback. Removing it lets their declared `compute: {kind: sklearn/framework}` drive `_create_compute` (the correct path). A latent build bug fixed for free. |
| `sdk_step_class` (class ref) | 4 | **STAYS in Python** — cradle/data_uploading/redshift/registration inject a SAIS SDK *Step class object (imported at module top); genuinely code, not serializable, and these builders aren't pure shells anyway (e.g. cradle has `get_output_location`). The SDKDelegation steps are the documented exception. |

## Wiring: the interface drives injection, not the class attrs

`_auto_bind_handler` (builder_templates.py) currently reads `self.STEP_ASSEMBLY` + `self.HANDLER_KNOBS`
(class attrs). Change it to read from the loaded interface (`self.spec`, already set before the bind):
`step_assembly = iface.patterns.step_assembly`; `knobs = iface.patterns.as_knobs()`. The class attrs
become a back-compat FALLBACK (so the 4 SDK builders + any un-migrated step still work): prefer the
YAML `patterns:` value, else the class attr, else the strategy preset/default.

`use_step_args` is no longer a knob anyone sets — it comes only from the `step_assembly` strategy
preset (`code`→False, `step_args`→True), so it can never disagree with the routing verb.

The ProcessingHandler already reads `output_path_token`/`include_job_type_in_path` knob→contract→
default; the knob value now ORIGINATES from `patterns:` (via `as_knobs()`), so the YAML is the source.

## Result for the 35 non-SDK builders

```python
class XGBoostModelEvalStepBuilder(TemplateStepBuilder):
    STEP_NAME = "XGBoostModelEval"
```
…and everything else (`step_assembly`, the output token, direct input keys) is in
`xgboost_model_eval.step.yaml` `patterns:`. Editing the YAML now changes the build — verified by the
mutation probe + the resolved-edge-graph snapshot + a new conformance gate (no builder may carry a
migrated knob in Python).

## Gates

1. Byte-diff: for every migrated step, the built handler's effective `(step_assembly, output_path_token,
   include_job_type_in_path, direct_input_keys)` == the pre-migration value (model_construct harness).
2. `steps patterns` view now reads `patterns.step_assembly` → the no-drift invariant becomes TRUE
   (closes FZ 31e1d3f3): the view and `_auto_bind_handler` read the SAME field.
3. Conformance: assert no `builder_*_step.py` (except the 4 SDK) declares `STEP_ASSEMBLY` or a
   `HANDLER_KNOBS` key in the migrated set; assert every step's `patterns:` round-trips.
4. Full suite + the edge-graph snapshot green.

## Status — DONE (2026-06-28)
- [x] `PatternsSection` model (`step_assembly` / `output_path_token` / `include_job_type_in_path` /
      `direct_input_keys` + `as_knobs()`; `use_step_args` intentionally absent — derived) +
      `StepInterface.patterns` field + variant deep-merge in `from_yaml`.
- [x] `_auto_bind_handler` reads `patterns:` (interface-first: `patterns.step_assembly` ||
      `STEP_ASSEMBLY`; knobs = `HANDLER_KNOBS` UNDER `patterns.as_knobs()`), class attrs are a
      back-compat fallback (the 4 SDK builders keep code-only `sdk_step_class`).
- [x] Migrated all non-SDK builders' STEP_ASSEMBLY/HANDLER_KNOBS → `patterns:` in the `.step.yaml`;
      **DELETED the 4 dead+broken `make_compute: lambda b: b._get_processor()` knobs** (latent build
      bug — `_get_processor` didn't exist; `compute:` block now drives them); dropped `use_step_args`
      everywhere; deleted the 3 byte-dead circular-ref overrides + pytorch_training's 2 dead methods.
- [x] io_view reads `patterns.step_assembly` → the **no-drift invariant is now structurally true**
      (closes FZ 31e1d3f Gap 1/f3): `cursus steps patterns XGBoostModelEval` shows `assembly=step_args`,
      matching the build.
- [x] **Byte-diff verified**: 41 steps' effective bound `(handler, step_assembly, output_path_token,
      include_job_type_in_path, direct_input_keys)` == pre-migration baseline (4 SDK steps skip —
      offline). Conformance gate `test_patterns_section_conformance.py` (4 checks). Full suite 781
      passed; edge-graph snapshot + migration parity green; ruff clean (the 36 E402 are pre-existing
      in `__init__.py`).

**Result:** the 35 non-SDK builders are pure `STEP_NAME` shells; the `.step.yaml` is the blueprint
that wires pattern injection per axis. Editing `patterns:` steers the build with no Python change.
