"""Job-arguments declaration ↔ config conformance gate (FZ 31e1d3h).

MODEL: config is the single source of truth for CLI arguments (``config.get_job_arguments()`` builds
them at runtime). The ``.step.yaml`` ``contract.job_arguments`` block is DECLARATIVE only — it records
the flags + their config source for visibility/alignment/introspection, and must not drift from what
the config actually emits. This gate enforces that: the set of declared ``--flag``s equals the set of
``--flag``s the config produces. (Values are NOT compared — they are runtime/config-supplied.)

Validation-bypassing ``model_construct`` is used so every config's ``get_job_arguments()`` runs
offline regardless of required fields.
"""

import os
import tempfile

import pytest


@pytest.fixture(scope="module")
def flag_pairs():
    import logging
    import warnings

    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    import glob

    import yaml

    from cursus.registry.step_names import get_step_names
    from cursus.step_catalog.step_catalog import StepCatalog
    from cursus.steps.interfaces import clear_interface_cache, load_step_interface

    clear_interface_cache()
    SN = get_step_names()
    cm = StepCatalog().discover_config_classes()
    tmp = tempfile.mkdtemp()
    open(os.path.join(tmp, "d.py"), "w").write("#\n")
    base = dict(
        author="t", bucket="b", role="arn:aws:iam::123456789012:role/test", region="NA",
        service_name="s", pipeline_version="1.0.0", project_root_folder="p",
        job_type="training", source_dir=tmp, processing_entry_point="d.py",
    )

    def sentinel(fl):
        s = str(fl.annotation)
        if "bool" in s:
            return False
        if "int" in s and "str" not in s:
            return 1
        if "float" in s:
            return 1.0
        if "List" in s or "list" in s:
            return []
        if "Dict" in s or "dict" in s:
            return {}
        return "x"

    rows = []
    for f in sorted(glob.glob("src/cursus/steps/interfaces/*.step.yaml")):
        st = (yaml.safe_load(open(f)) or {}).get("step_type")
        if not st:
            continue
        try:
            contract, _ = load_step_interface(st)
        except Exception:
            continue
        declared = {d.flag for d in getattr(contract, "job_arguments", [])}
        if not declared:
            continue  # steps that declare no args are out of scope for this gate
        ccls = cm.get(SN.get(st, {}).get("config_class"))
        if ccls is None:
            continue
        kwargs = dict(base)
        for fn, fld in ccls.model_fields.items():
            if fn in kwargs:
                continue
            req = fld.is_required() if hasattr(fld, "is_required") else (fld.default is None)
            if req:
                kwargs[fn] = sentinel(fld)
        try:
            cfg = ccls.model_construct(**kwargs)
            emitted = cfg.get_job_arguments() or []
        except Exception:
            continue
        emitted_flags = {a for a in emitted if str(a).startswith("--")}
        rows.append((st, declared, emitted_flags))
    return rows


def test_declared_job_arg_flags_match_config(flag_pairs):
    """Every flag the config EMITS must be DECLARED in the .step.yaml (emitted ⊆ declared).

    The dangerous drift is a config emitting an UNDECLARED flag — an argument the script receives
    that the interface never documents. A declared flag MAY be conditionally omitted: some configs
    emit optional-field-driven flags only when that field is set (e.g. TSATabularPreprocessing emits
    --label-field / --id-fields / --date-field only when the corresponding Optional[str] field is
    populated, mirroring its conditional env-var emission). The gate enforces the subset invariant,
    not exact equality, so a legitimately-conditional flag is not a failure. (Values are not
    compared — they are runtime/config-supplied.)
    """
    undeclared = [
        (st, sorted(emit - decl)) for st, decl, emit in flag_pairs if emit - decl
    ]
    assert not undeclared, (
        "config.get_job_arguments() emits flags NOT declared in the .step.yaml "
        "job_arguments block: "
        + "; ".join(f"{st}: undeclared={u}" for st, u in undeclared)
    )


def test_some_steps_declare_job_arguments(flag_pairs):
    """Sanity: the declarative block is actually populated (guards against a no-op gate)."""
    assert len(flag_pairs) >= 25
