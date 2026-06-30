# Audit: Phase S3 C-SDK — the 5 SAIS-SDK-bound builder migrations (Kiro/SAIS)

> **Verdict: all 5 migrated correctly (45/45 complete).** Static audit on 2026-06-27 of the 5
> builders Kiro migrated in the SAIS sandbox (commits `2ca79fac` + `13027c47`), since they cannot be
> imported in the SDK-less local env. Method: per-builder `git diff <pre> -- <file>` against the
> migration recipe ([2026-06-27_S3_C-SDK_remaining_5_builders_task.md](2026-06-27_S3_C-SDK_remaining_5_builders_task.md)),
> registry-routing checks (import-free), handler behavior-equivalence tracing, plus an adversarial
> refutation pass per builder (each builder-specific risk an agent actively tried to refute).
> Pre-migration baseline = `c12fc6eb` (parent of `2ca79fac`).

## Result table

| Builder | Binds | Verdict | Key check |
|---|---|---|---|
| `builder_cradle_data_loading_step.py` | SDKDelegationHandler | ✅ CLEAN | public helpers `get_output_location`/`get_step_outputs` kept; `_attach_spec` reproduces `_spec`/`_contract`; property_path wiring (the real output mechanism) untouched in `.step.yaml` + `property_reference.py` |
| `builder_redshift_data_loading_step.py` | SDKDelegationHandler | ✅ CLEAN | simplest; `input_mode="none"` ctor byte-identical to old `create_step` |
| `builder_data_uploading_step.py` | SDKDelegationHandler via `STEP_ASSEMBLY="delegation"` | ✅ CLEAN | correctly routes via the assembly discriminator (Processing-typed but NOT ProcessingHandler); `resolve_s3` input handling preserved |
| `builder_edx_uploading_step.py` | ProcessingHandler (`code`) | ✅ CLEAN (+1 alignment fix) | KMS/network `_create_processor` preserved verbatim + wired via `make_compute` default; **see Finding 1** |
| `builder_registration_step.py` | SDKDelegationHandler (`MimsModelRegistrationProcessing`) | ✅ CLEAN (+1 doc note) | region-suffix (`append_region`, the eu-west-1 fix) + `mims_ordered` PackagedModel-first inputs faithfully reproduced by preset knobs; **see Finding 2** |

Routing verified import-free: every `STEP_NAME` resolves to its expected handler via
`get_sagemaker_step_type` → `axis_name_for_step_type` → `resolve_handler` (incl. DataUploading's
`delegation` assembly and EdxUploading defaulting to `code` → ProcessingHandler, NOT SDKDelegation).
ruff clean on all 5.

## Finding 1 (FIXED) — edx `_resolve_script_path` was dead code; aligned edx with the other ScriptProcessor steps

The old edx `create_step` used `code=self._resolve_script_path()` (a bespoke 3-tier resolver with a
tier-3 fallback to the package-bundled `cursus/steps/scripts/edx_uploading.py`). The migrated shell
**keeps `_resolve_script_path` but nothing calls it** — `ProcessingHandler.build_step` uses
`code=b.config.get_script_path()` like every other ScriptProcessor step. edx was the ONLY processing
builder that ever defined a custom script resolver.

Adversarial check showed the feared `code=None` regression is **provably unreachable**:
`ProcessingStepConfigBase.validate_entry_point_paths` *raises at config construction* if both
`processing_source_dir` and `source_dir` are unset — so `effective_source_dir` is never None for a
valid `EdxUploadingConfig`, which is the only condition under which the old tier-3 package fallback
was ever reached. The fallback was dead code in the OLD builder too. Live reproduction confirmed
byte-identical `code=` for every constructible config.

**Fix applied (this repo, separate commit):** removed the dead `_resolve_script_path` from
`builder_edx_uploading_step.py` so edx resolves its script via `config.get_script_path()` exactly
like every other ScriptProcessor step. Pure dead-code removal — runtime no-op, ruff clean,
offline suite green.

## Finding 2 (RECORDED, no code change) — Registration `validate_configuration` drop is safe but mis-justified

Commit `13027c47` dropped Registration's `validate_configuration` with the message *"redundant with
StepInterface._sync_and_align"*. That justification is **directionally wrong**:

- **Dropped guard** checked: every *required spec dependency* appears in
  `contract.expected_input_paths`  →  direction `spec.required_deps ⊆ contract.inputs`.
- **`_sync_and_align`** (`step_interface.py:457`) checks: `contract.inputs ⊆ spec.dependencies`
  (and `contract.outputs ⊆ spec.outputs`)  →  the **reverse** direction. The companion
  `validate_contract_alignment` checks the same contract→spec direction. No validator in
  `step_interface.py` asserts a required spec dep must exist in the contract.

So the two invariants are **not** equivalent and neither subsumes the other.

- **For Registration the loss is LATENT, not a live break:** `registration.step.yaml` declares both
  required deps (`PackagedModel`, `GeneratedPayloadSamples`) with matching non-null contract input
  paths, so the old guard, `_sync_and_align`, and `validate_contract_alignment` all pass today.
- **The drop is nonetheless defensible** for a *different* reason than the commit gives: the old
  guard was over-strict for `path: null` contract inputs. `expected_input_paths`
  (`step_interface.py:188`) filters out ports whose contract path is null, so BatchTransform /
  PyTorchModel / XGBoostModel (required deps `model_name` / `model_data` with `path: null`) would
  have **falsely failed** the old guard while passing `_sync_and_align`. That over-strictness — not
  redundancy — is the legitimate package-wide reason to remove it.

**Net:** code is fine as-is (per decision); the protective value against future spec↔contract drift
on SDK-delegated steps is genuinely gone, and the commit message's stated reason is incorrect. No
restoration done (user decision: record-only). If a corrected guard is ever wanted, it must skip
deps whose contract path is null to avoid re-introducing the over-strictness bug.

## What only the SAIS env can still confirm (out of scope here)

The recipe's **real-session byte-diff** (build old-way vs new-way with a real SAIS session and diff
the produced SDK step — ctor args, region-suffixed name, Mims ordered inputs, Cradle output
attributes) is the one gate this static audit cannot run. Static + adversarial analysis found no
behavioral divergence, but the definitive construction parity check belongs to SAIS.
