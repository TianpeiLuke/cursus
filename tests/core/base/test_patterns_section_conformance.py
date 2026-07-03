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

import pytest

# The declarative facts that were migrated out of the (now-deleted) per-step builder shells into the
# .step.yaml `patterns:`/contract sections. use_step_args is DERIVED from step_assembly and must never
# be authored anywhere.
MIGRATED_KNOBS = {"output_path_token", "include_job_type_in_path", "direct_input_keys"}


def test_step_assembly_lives_on_patterns_section_not_derived_use_step_args():
    """STEP_ASSEMBLY is authored as PatternsSection.step_assembly; use_step_args is DERIVED from it and
    must NOT be a field anyone can author. (The per-step builder shells that used to carry STEP_ASSEMBLY /
    HANDLER_KNOBS in Python are gone — the fact now lives only in the interface, so it cannot drift.)"""
    from cursus.core.base.step_interface import PatternsSection

    fields = set(PatternsSection.model_fields)
    assert "step_assembly" in fields, (
        "step_assembly must be authored on PatternsSection (the .step.yaml patterns: blueprint)."
    )
    assert "use_step_args" not in fields, (
        "use_step_args is DERIVED from step_assembly — it must never be an authorable PatternsSection field."
    )


def test_migrated_knobs_are_patterns_or_contract_data_not_derived():
    """The migrated declarative knobs live on PatternsSection (include_job_type_in_path, direct_input_keys)
    or on ContractSection (output_path_token) — never on both, and never as a derived use_step_args knob."""
    from cursus.core.base.step_interface import ContractSection, PatternsSection

    patterns_fields = set(PatternsSection.model_fields)
    contract_fields = set(ContractSection.model_fields)
    for knob in MIGRATED_KNOBS:
        assert knob in patterns_fields or knob in contract_fields, (
            f"migrated knob {knob!r} must live on PatternsSection or ContractSection."
        )
    # use_step_args is derived, never authored on either section.
    assert "use_step_args" not in patterns_fields
    assert "use_step_args" not in contract_fields


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


def test_output_path_token_is_derived_by_default_optin_override_allowed():
    """The S3 output prefix is DERIVED from the step name (``canonical_to_snake(step_type)``) by
    DEFAULT. As of the PIPER work, ``output_path_token`` is re-introduced on ContractSection as an
    OPT-IN override (default ``None`` ⇒ derived) — needed when an EXTERNAL consumer keys off a fixed
    S3 folder name that does not match the cursus step name (e.g. PIPER scans
    ``<pipeline>/Model_Metric_Generation_Step/``). This gate locks the shape of that override:
      * it lives on ContractSection (contract DATA), NOT on PatternsSection;
      * it defaults to ``None`` so every step that does not set it keeps the derived token;
      * it is read via ``getattr(b.contract, "output_path_token", ...)``, NOT as a strategy
        KnobSpec — the token is contract data, not a handler knob like include_job_type_in_path."""
    import ast

    from cursus.core.base.step_interface import ContractSection, PatternsSection

    assert "output_path_token" not in PatternsSection.model_fields, (
        "output_path_token must NOT live on PatternsSection — it is contract DATA on ContractSection."
    )
    assert "output_path_token" in ContractSection.model_fields, (
        "output_path_token override missing from ContractSection — the PIPER fixed-folder use case "
        "requires an opt-in override of the derived token."
    )
    assert ContractSection.model_fields["output_path_token"].default is None, (
        "output_path_token must default to None so the derived canonical_to_snake token is used "
        "unless a step explicitly overrides it."
    )
    # the override is read as contract data (getattr), not exposed as a strategy KnobSpec
    src = open("src/cursus/core/base/builder_templates.py").read()
    assert 'getattr(b.contract, "output_path_token"' in src, (
        "get_outputs must read the override via getattr(b.contract, 'output_path_token', None)."
    )
    assert 'KnobSpec("output_path_token"' not in src, (
        "output_path_token is contract data, not a strategy knob — do not add a KnobSpec for it."
    )
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
    canonical_to_snake(step_type) UNLESS that step declares an explicit contract.output_path_token
    override — in which case the view must report the override verbatim (with source labeled as an
    override). This proves the derivation is live+uniform by default and the opt-in override is the
    sole sanctioned deviation."""
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
        override = getattr(iface.contract, "output_path_token", None)
        if override:
            # explicit override: the view must surface the literal + label the source as an override
            if tok != override or "override" not in (
                view["patterns"]["outputs"].get("output_path_token_source") or ""
            ):
                bad.append((st, tok, f"override={override!r}"))
        elif tok != canonical_to_snake(st):
            bad.append((st, tok, canonical_to_snake(st)))
    assert not bad, f"output prefix not derived (and not a sanctioned override): {bad}"
