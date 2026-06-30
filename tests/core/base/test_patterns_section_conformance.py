"""patterns: section migration conformance gate (FZ 31e1d3f1).

The per-axis strategy-selection facts (step_assembly, output_path_token, include_job_type_in_path,
direct_input_keys) are the .step.yaml `patterns:` BLUEPRINT — editing the YAML steers the build, not a
builder shell. This gate locks that in:

  1. No builder (except the 4 SDKDelegation builders) may declare STEP_ASSEMBLY or a migrated
     HANDLER_KNOBS key in Python — those facts belong in patterns:.
  2. use_step_args is never authored anywhere (derived from step_assembly).
  3. The patterns view's step_assembly == the value the build binds (no introspection drift).
"""

import ast
import glob
import os

import pytest

BUILDERS = "src/cursus/steps/builders"
# the 4 SDKDelegation builders legitimately keep a code-only sdk_step_class knob (a SAIS class object
# that can't be serialized to YAML) — the documented exception.
SDK_BUILDERS = {
    "builder_cradle_data_loading_step.py", "builder_redshift_data_loading_step.py",
    "builder_registration_step.py", "builder_data_uploading_step.py",
}
MIGRATED_KNOBS = {"output_path_token", "include_job_type_in_path", "direct_input_keys", "use_step_args"}


def _class_attrs(path):
    """Return (has_STEP_ASSEMBLY, set(HANDLER_KNOBS keys)) declared on the builder class."""
    tree = ast.parse(open(path).read())
    has_assembly = False
    knob_keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for b in node.body:
                if isinstance(b, ast.Assign):
                    for t in b.targets:
                        if not isinstance(t, ast.Name):
                            continue
                        if t.id == "STEP_ASSEMBLY":
                            has_assembly = True
                        if t.id == "HANDLER_KNOBS" and isinstance(b.value, ast.Dict):
                            for k in b.value.keys:
                                if isinstance(k, ast.Constant):
                                    knob_keys.add(k.value)
    return has_assembly, knob_keys


def test_no_builder_declares_step_assembly_in_python():
    """STEP_ASSEMBLY moved to patterns.step_assembly — no builder should keep the class attr."""
    offenders = [
        os.path.basename(f)
        for f in glob.glob(f"{BUILDERS}/builder_*_step.py")
        if _class_attrs(f)[0]
    ]
    assert not offenders, f"builders still declare STEP_ASSEMBLY in Python (move to patterns:): {offenders}"


def test_no_builder_declares_migrated_knobs():
    """The declarative knobs (output_path_token / include_job_type_in_path / direct_input_keys) +
    the derived use_step_args must not live in any builder's HANDLER_KNOBS — they belong in patterns:."""
    offenders = {}
    for f in glob.glob(f"{BUILDERS}/builder_*_step.py"):
        keys = _class_attrs(f)[1] & MIGRATED_KNOBS
        if keys:
            offenders[os.path.basename(f)] = sorted(keys)
    assert not offenders, f"builders still declare migrated knobs in Python (move to patterns:): {offenders}"


def test_only_sdk_builders_keep_handler_knobs():
    """After migration, the ONLY HANDLER_KNOBS left are the 4 SDKDelegation builders' code-only
    sdk_step_class. Any other builder with a non-empty HANDLER_KNOBS is a migration miss."""
    offenders = {}
    for f in glob.glob(f"{BUILDERS}/builder_*_step.py"):
        base = os.path.basename(f)
        keys = _class_attrs(f)[1]
        if not keys:
            continue
        if base in SDK_BUILDERS:
            assert keys == {"sdk_step_class"}, f"{base} SDK builder has unexpected knobs: {keys}"
        else:
            offenders[base] = sorted(keys)
    assert not offenders, f"non-SDK builders still carry HANDLER_KNOBS: {offenders}"


@pytest.fixture(scope="module")
def interfaces():
    import logging
    import warnings

    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    from cursus.steps.interfaces import clear_interface_cache, load_interface, list_available_interfaces

    clear_interface_cache()
    out = {}
    for stem in list_available_interfaces():
        try:
            out[stem] = load_interface(stem)
        except Exception:
            pass
    return out


def test_patterns_view_step_assembly_matches_blueprint(interfaces):
    """The introspection view reads patterns.step_assembly — the SAME field _auto_bind_handler binds
    — so the 'cannot drift' invariant (FZ 31e1d3f Gap 1) is now structurally true. Verify the view's
    reported step_assembly equals the interface's patterns.step_assembly for every step."""
    from cursus.steps.interfaces.io_view import describe_step_patterns

    bad = []
    for stem, iface in interfaces.items():
        try:
            view = describe_step_patterns(iface.spec.step_type)
        except Exception:
            continue
        expected = iface.patterns.step_assembly
        reported = view["patterns"]["create_step"]["step_assembly"]
        if expected != reported:
            bad.append((iface.spec.step_type, expected, reported))
    assert not bad, f"patterns view step_assembly drifted from patterns.step_assembly: {bad}"


def test_output_path_token_is_derived_not_declarable():
    """output_path_token corresponds to the step name (FZ 31e1d3f1b) — the S3 output prefix is
    ALWAYS ``canonical_to_snake(step_type)``, so the field was REMOVED entirely (not kept as an
    override). This gate locks that: neither PatternsSection nor ContractSection may re-introduce an
    output_path_token field, and no handler may read an output_path_token knob."""
    import ast

    from cursus.core.base.step_interface import ContractSection, PatternsSection

    assert "output_path_token" not in PatternsSection.model_fields, (
        "output_path_token re-introduced on PatternsSection — the S3 prefix is derived from the "
        "step name (canonical_to_snake), not declarable (FZ 31e1d3f1b)."
    )
    assert "output_path_token" not in ContractSection.model_fields, (
        "output_path_token re-introduced on ContractSection — it is derived, not declarable."
    )
    # no handler reads an output_path_token knob anymore
    src = open("src/cursus/core/base/builder_templates.py").read()
    assert 'knobs.get("output_path_token")' not in src and "KnobSpec(\"output_path_token\"" not in src, (
        "a handler still references an output_path_token knob — the token must be derived from "
        "canonical_to_snake(step_type) unconditionally."
    )
    # and the AST has no such KnobSpec
    assert "output_path_token" not in {
        n.value
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "KnobSpec"
        for n in node.args[:1]
        if isinstance(n, ast.Constant)
    }


def test_output_prefix_derived_from_step_name(interfaces):
    """Every non-sink step's output S3 prefix (in the patterns view) equals
    canonical_to_snake(step_type) — proving the derivation is live and uniform."""
    from cursus.step_catalog.naming import canonical_to_snake
    from cursus.steps.interfaces.io_view import describe_step_patterns

    bad = []
    for stem, iface in interfaces.items():
        st = iface.spec.step_type
        try:
            view = describe_step_patterns(st)
        except Exception:
            continue
        if view["patterns"]["outputs"].get("sink"):
            continue
        tok = view["patterns"]["outputs"]["output_path_token"]
        if tok != canonical_to_snake(st):
            bad.append((st, tok, canonical_to_snake(st)))
    assert not bad, f"output prefix not derived from step name: {bad}"
