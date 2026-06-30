"""
B3 RegistryBindingValidator TEETH-TEST (FZ 31e1d3g3 Phase D1 — mandatory gate).

This is the gate that proves B3 is NOT a false-green no-op. The single biggest residue risk of the
validation redesign is that the new config-coverage check silently passes everything (the very failure
mode the old getsource scans masked). So B3 must demonstrably FAIL on:
  (1) a broken CONFIG — a required field the bound handler reads is removed, AND
  (2) a broken REGISTRY BINDING — a step whose sagemaker_step_type has no routable handler;
and PASS on a live GREEN control. If any of these flips, the redesign has regressed to a no-op.
"""

import logging

from cursus.validation.alignment.validators.registry_binding_validator import (
    RegistryBindingValidator,
)

logging.disable(logging.CRITICAL)


def _errors(result):
    return [i for i in result["issues"] if i["level"] in ("CRITICAL", "ERROR")]


def test_green_control_live_shells_pass():
    """A live BatchTransform + XGBoostTraining shell must PASS B3 (handler binds, builder loads,
    config covers). Guards against B3 ERRORing on healthy steps."""
    v = RegistryBindingValidator()
    for step in ("BatchTransform", "XGBoostTraining"):
        result = v.validate_builder_config_alignment(step)
        assert result["status"] == "COMPLETED", f"{step}: {result}"
        assert not _errors(result), f"{step} should be clean: {_errors(result)}"


def test_teeth_broken_config_missing_required_field_fails(monkeypatch):
    """B3-3 must ERROR when the resolved config class lacks a field the bound handler requires.
    We make the handler require a field no config has, proving the coverage check has teeth (it is
    NOT a descriptor-only check that would be blind to handler-side reads)."""
    v = RegistryBindingValidator()

    # Inject a bogus required field into the bound handler's requires_config_fields for this run.
    from cursus.core.base.builder_templates import TransformHandler

    monkeypatch.setattr(
        TransformHandler,
        "requires_config_fields",
        ("job_type", "a_field_no_config_will_ever_have"),
        raising=False,
    )

    result = v.validate_builder_config_alignment("BatchTransform")
    errs = _errors(result)
    assert result["status"] == "ISSUES_FOUND"
    assert any("a_field_no_config_will_ever_have" in e["message"] for e in errs), errs


def test_teeth_unbindable_registry_row_fails(monkeypatch):
    """B3-1 must ERROR when a step's sagemaker_step_type has no routable handler (resolve_handler
    raises NoBuilderError). We force a routable step to report a bogus step type."""
    # Patch get_sagemaker_step_type as seen inside _check_handler_binds to return an unroutable type.
    from cursus.registry import step_names as sn

    monkeypatch.setattr(sn, "get_sagemaker_step_type", lambda *a, **k: "NoSuchSageMakerType")

    v = RegistryBindingValidator()
    result = v.validate_builder_config_alignment("XGBoostTraining")
    errs = _errors(result)
    assert result["status"] == "ISSUES_FOUND"
    assert any("No construction handler" in e["message"] for e in errs), errs


def test_lambda_and_base_skip_not_error():
    """Lambda/Base are no-builder rows that route into L4 by ruleset; B3 must SKIP them, not ERROR —
    so the teeth-test for unbindable rows (above) is about UNKNOWN types, not the by-design no-builder
    rows."""
    from cursus.registry import step_names as sn

    # Force a step to look like a Lambda row.
    orig = sn.get_sagemaker_step_type
    try:
        sn.get_sagemaker_step_type = lambda *a, **k: "Lambda"
        v = RegistryBindingValidator()
        result = v.validate_builder_config_alignment("XGBoostTraining")
        assert result["status"] == "SKIPPED", result
    finally:
        sn.get_sagemaker_step_type = orig
