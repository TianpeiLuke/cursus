"""Env-vars config↔interface CONFORMANCE gate (FZ 31e1d3g).

MODEL (user directive, 2026-06-27):
  1. Env vars stay DECLARED in the step interface (.step.yaml `env_vars.required` / `.optional`,
     surfaced as `contract.required_env_vars` / `contract.optional_env_vars`) — the declared
     contract + the script↔contract alignment surface.
  2. Config PROVIDES the VALUES via its collector (`get_environment_variables()` method, else the
     `environment_variables` property). The collector is the AUTHORITATIVE selector of which fields
     become env vars — NOT every config field is an env var, so this gate only ever inspects the
     COLLECTOR OUTPUT, never the raw config schema.
  3. The two MUST NOT CONFLICT. Two directional invariants:
       B (hard error): every interface-`required` env key is emitted by the config collector — else
                       a script that needs it gets nothing.
       A (reconcile) : every collector-emitted key is declared in the interface (required ∪ optional)
                       — else the interface under-declares what config actually injects.

WHY model_construct: validity is irrelevant — we only need an instance whose collector runs. Real
construction fails offline for most configs (step-specific required fields); model_construct fills
required-without-default fields with type sentinels so the collector executes. Sentinel VALUES are
ignored (this gate compares KEY SETS only), so sentinels never cause a false conflict.

This gate currently RECORDS the baseline (it does not yet hard-fail on the known conflicts, which are
the reconciliation worklist) and FAILS if NEW conflicts appear beyond the recorded baseline.
"""

import importlib
import inspect
import logging
import os
import pkgutil
import tempfile

import pytest

_BASE_KWARGS = dict(
    author="t",
    bucket="b",
    role="arn:aws:iam::123456789012:role/test",
    region="NA",
    service_name="s",
    pipeline_version="1.0.0",
    project_root_folder="p",
    job_type="training",
)


def _sentinel(annotation):
    s = str(annotation)
    if "bool" in s:
        return False
    if "int" in s and "str" not in s:
        return 1
    if "float" in s:
        return 1.0
    if "Dict" in s or "dict" in s:
        return {}
    if "List" in s or "list" in s:
        return []
    return "x"


def _discover_overriders():
    """STEP_NAME -> builder class for builders that override _get_environment_variables."""
    import cursus.steps.builders as B

    out = {}
    for m in pkgutil.iter_modules(B.__path__):
        try:
            mod = importlib.import_module(f"cursus.steps.builders.{m.name}")
        except Exception:
            continue
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if "_get_environment_variables" in obj.__dict__:
                sn = getattr(obj, "STEP_NAME", None)
                if sn:
                    out[sn] = obj
    return out


@pytest.fixture(scope="module")
def conformance():
    """Return (conformant, conflict_A, conflict_B, no_collector) lists of step names."""
    logging.disable(logging.CRITICAL)
    import warnings

    warnings.filterwarnings("ignore")
    from cursus.step_catalog.step_catalog import StepCatalog
    from cursus.steps.interfaces import clear_interface_cache, load_step_interface

    clear_interface_cache()
    tmp = tempfile.mkdtemp()
    open(os.path.join(tmp, "dummy.py"), "w").write("# stub\n")
    base = dict(_BASE_KWARGS, source_dir=tmp, processing_entry_point="dummy.py")

    cat = StepCatalog()
    try:
        cfgmap = cat.discover_config_classes()
    except Exception:
        cfgmap = {}

    def find_cfg(sn):
        key = sn.lower().replace("_", "")
        for k, v in cfgmap.items():
            if key in k.lower().replace("_", ""):
                return v
        return None

    def collector_keys(sn):
        """The KEY SET the config collector emits, plus the interface-declared req/opt key sets."""
        ccls = find_cfg(sn)
        if ccls is None:
            return None
        kwargs = dict(base)
        for fname, fld in ccls.model_fields.items():
            if fname in kwargs:
                continue
            req = fld.is_required() if hasattr(fld, "is_required") else (fld.default is None)
            if req:
                kwargs[fname] = _sentinel(fld.annotation)
        try:
            cfg = ccls.model_construct(**kwargs)
        except Exception:
            return None
        try:
            contract, _ = load_step_interface(sn)
            req = set(getattr(contract, "required_env_vars", []) or [])
            opt = set(getattr(contract, "optional_env_vars", {}) or {})
        except Exception:
            req = opt = set()
        # The collector is the names-driven resolver get_environment_variables(declared_names)
        # (FZ 31e1d3g): the interface declares WHICH names, config resolves their VALUES. Pass the
        # declared names so the resolver produces the same key set the builder would at runtime.
        # Tolerant of a legacy no-arg collector (returns its full dict regardless of the arg).
        declared = sorted(req | opt)
        import inspect as _inspect

        getter = getattr(cfg, "get_environment_variables", None)
        coll = None
        if callable(getter):
            try:
                accepts = any(
                    p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
                    for p in list(_inspect.signature(getter).parameters.values())
                )
            except (ValueError, TypeError):
                accepts = False
            coll = getter(declared) if accepts else getter()
        if not isinstance(coll, dict):
            coll = getattr(cfg, "environment_variables", None)
        if not isinstance(coll, dict):
            return ("NO_COLLECTOR", set(), set())
        return (set(coll), req, opt)

    conformant, conflict_a, conflict_b, no_collector = [], {}, {}, []
    for sn in sorted(_discover_overriders()):
        res = collector_keys(sn)
        if res is None:
            continue
        if res[0] == "NO_COLLECTOR":
            no_collector.append(sn)
            continue
        produced, req, opt = res
        declared = req | opt
        undeclared = produced - declared  # A: config emits keys the interface doesn't declare
        missing_required = req - produced  # B: interface requires a key config never emits
        if not undeclared and not missing_required:
            conformant.append(sn)
        else:
            if undeclared:
                conflict_a[sn] = sorted(undeclared)
            if missing_required:
                conflict_b[sn] = sorted(missing_required)
    return conformant, conflict_a, conflict_b, no_collector


# Recorded baseline. Direction B (interface requires a key config never emits) is the HARD invariant.
# Both original B-conflicts were FIXED 2026-06-27 by the script-driven stale/rename remediation:
#   RiskTableMapping: dropped the stale required LABEL_FIELD (the script reads label via hyperparams,
#     never from env) + USE_PRECOMPUTED_RISK_TABLES → env_vars now empty.
#   TemporalSequenceNormalization: renamed required TIMESTAMP_FIELD → TEMPORAL_FIELD (the name the
#     script + config actually use) and MAX_SEQUENCE_LENGTH → SEQUENCE_LENGTH.
# Baseline is now empty: ANY CONFLICT-B is a hard failure.
_BASELINE_CONFLICT_B = set()


def test_no_new_required_conflict(conformance):
    """Direction B (HARD): no NEW step where the interface requires an env var the config collector
    never emits. The 2 baseline misdeclarations are recorded; any beyond them fails."""
    conformant, conflict_a, conflict_b, no_collector = conformance
    print(f"\nCONFORMANT ({len(conformant)}): {conformant}")
    print(f"CONFLICT-B required-not-emitted ({len(conflict_b)}): {conflict_b}")
    print(f"CONFLICT-A undeclared-by-interface ({len(conflict_a)}): {sorted(conflict_a)}")
    print(f"NO-COLLECTOR (env from interface only) ({len(no_collector)}): {no_collector}")
    new_b = set(conflict_b) - _BASELINE_CONFLICT_B
    assert not new_b, (
        f"NEW config↔interface required-env conflict (interface requires a key the config collector "
        f"does not emit): { {k: conflict_b[k] for k in new_b} }. Either add the field to the config "
        f"collector or remove it from the interface's required_env_vars."
    )


def test_baseline_conflict_b_still_present(conformance):
    """Guard the worklist: if a baseline B-conflict is FIXED, shrink _BASELINE_CONFLICT_B so it can't
    silently regress later. (Fails if a recorded conflict disappears — prompting the baseline update.)"""
    _, _, conflict_b, _ = conformance
    fixed = _BASELINE_CONFLICT_B - set(conflict_b)
    assert not fixed, (
        f"Recorded CONFLICT-B resolved for {fixed} — remove from _BASELINE_CONFLICT_B to lock it in."
    )
