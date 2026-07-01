"""
Tests for the cursus.mcp ``author.*`` namespace (FZ 31e1d3f5) — the agent step-authoring
guidance surface: ``author.checklist`` (the SOP as data), ``author.rules`` (restriction oracle
introspecting live enforcement objects), and ``author.preflight_step`` (the offline
constructibility proof = the CI merge gate).

These cover registration + phase tags, the live-introspection / no-drift property of author.rules,
the registry-derived routing of author.checklist, and that preflight composes the real gates
(including --all CI mode green, SDK skip-not-error, and a graceful failure envelope).
"""

import pytest

from cursus.mcp import call_tool
from cursus.mcp.registry import get_tool


class TestRegistration:
    @pytest.mark.parametrize(
        "name,tag",
        [
            ("author.checklist", "planner"),
            ("author.rules", "validator"),
            ("author.preflight_step", "validator"),
        ],
    )
    def test_registered_with_tag(self, name, tag):
        td = get_tool(name)
        assert td is not None
        assert tag in td.tags


class TestAuthorRules:
    @pytest.mark.parametrize(
        "topic", ["naming", "packaging", "sdk_carveout", "reuse_class", "closure"]
    )
    def test_every_topic_returns_rules(self, topic):
        r = call_tool("author.rules", {"topic": topic})
        assert r.ok
        assert r.data["topic"] == topic
        assert r.data["rules"]

    def test_unknown_topic_invalid_input(self):
        r = call_tool("author.rules", {"topic": "bogus"})
        assert not r.ok
        assert r.code == "invalid_input"

    def test_naming_introspects_live_pascal_pattern(self):
        """No-drift: the regex returned IS the live PASCAL_CASE_PATTERN, not a restated copy."""
        from cursus.registry.validation_utils import PASCAL_CASE_PATTERN

        r = call_tool("author.rules", {"topic": "naming"})
        assert (
            r.data["rules"]["step_name"]["pascal_case_regex"]
            == PASCAL_CASE_PATTERN.pattern
        )

    def test_naming_valid_step_types_match_pin(self):
        from cursus.core.base.step_interface import RegistrySection

        r = call_tool("author.rules", {"topic": "naming"})
        assert set(r.data["rules"]["sagemaker_step_type"]["valid_values"]) == set(
            RegistrySection._SAGEMAKER_STEP_TYPES
        )

    def test_sdk_carveout_keeps_the_two_requires_enums_distinct(self):
        """The registry SDK enum and the compute mods enum are DISTINCT — author.rules must not
        conflate them (a wrong reflection silently misleads the agent about the SDK carve-out)."""
        from cursus.core.base.step_interface import RegistrySection, ComputeSpec

        r = call_tool("author.rules", {"topic": "sdk_carveout"})
        assert set(r.data["rules"]["registry_requires"]["valid_values"]) == set(
            RegistrySection._REQUIRES
        )
        assert set(r.data["rules"]["compute_requires"]["valid_values"]) == set(
            ComputeSpec._REQUIRES
        )
        assert (
            "secure_ai_sandbox_workflow_python_sdk"
            in r.data["rules"]["registry_requires"]["valid_values"]
        )
        assert (
            "mods_workflow_core" in r.data["rules"]["compute_requires"]["valid_values"]
        )


class TestAuthorChecklist:
    def test_processing_step_args_routes_to_processing_handler(self):
        r = call_tool(
            "author.checklist",
            {"sagemaker_step_type": "Processing", "step_assembly": "step_args"},
        )
        assert r.ok
        assert r.data["routable"] is True
        assert r.data["bound_handler"]["handler"] == "ProcessingHandler"
        assert r.data["exemplar_step"] == "TabularPreprocessing"
        # the SOP names the existing tools in order
        tools = [s["tool"] for s in r.data["steps"]]
        assert "strategies.for_step_type" in tools[0]
        assert any("validate.step_interface" in t for t in tools)
        assert any("author.preflight_step" in t for t in tools)

    def test_training_routes_to_training_handler(self):
        r = call_tool("author.checklist", {"sagemaker_step_type": "Training"})
        assert r.ok
        assert r.data["bound_handler"]["handler"] == "TrainingHandler"
        assert r.data["exemplar_step"] == "XGBoostTraining"

    def test_invalid_step_type_rejected(self):
        r = call_tool("author.checklist", {"sagemaker_step_type": "Procesing"})
        assert not r.ok
        assert r.code == "invalid_input"

    def test_no_builder_type_warns_not_routable(self):
        r = call_tool("author.checklist", {"sagemaker_step_type": "Base"})
        assert r.ok
        assert r.data["routable"] is False
        assert r.warnings


class TestAuthorPreflightStep:
    def test_single_real_step_is_constructible(self):
        r = call_tool("author.preflight_step", {"step_name": "TabularPreprocessing"})
        assert r.ok
        assert r.data["constructible"] is True
        names = {g["name"] for g in r.data["gates"]}
        assert names == {
            "interface",
            "registry_parity",
            "registry_binding_b3",
            "routability",
        }

    def test_sdk_step_skips_not_errors(self):
        """An SDK-delegation step preflights clean offline (skip-not-error), not a hard fail."""
        r = call_tool("author.preflight_step", {"step_name": "CradleDataLoading"})
        assert r.ok
        assert r.data["constructible"] is True

    def test_all_mode_is_ci_green(self):
        """The CI merge gate: every step is constructible (interface + registry + B3 + routability)."""
        r = call_tool("author.preflight_step", {"all": True})
        assert r.ok, [
            g for g in (r.meta.get("details") or {}).get("gates", []) if not g["passed"]
        ]
        assert r.data["constructible"] is True
        # one interface + one registry gate, then B3+routability per canonical step
        assert (
            sum(1 for g in r.data["gates"] if g["name"] == "registry_binding_b3") >= 45
        )

    def test_missing_step_name_without_all_is_invalid_input(self):
        r = call_tool("author.preflight_step", {})
        assert not r.ok
        assert r.code == "invalid_input"

    def test_unknown_step_fails_with_remedy(self):
        r = call_tool("author.preflight_step", {"step_name": "NoSuchStepXYZ"})
        assert not r.ok
        assert r.code == "not_constructible"
        assert (r.remedy or {}).get("suggested_tools")


class TestAuthorConfigConstraints:
    """author.config_constraints surfaces a config's legal VALUES (not just types) — closes the
    Cat1/Cat3 empirical bug class where type-only listings hid the allowed enum sets."""

    def test_surfaces_allowed_values_and_required_no_default(self):
        r = call_tool(
            "author.config_constraints", {"step_name": "TabularPreprocessing"}
        )
        assert r.ok
        of = next(
            (c for c in r.data["constrained_fields"] if c["name"] == "output_format"),
            None,
        )
        assert of is not None
        assert set(of["allowed_values"]) == {"CSV", "TSV", "Parquet"}
        assert of["case_sensitive"] is False
        assert isinstance(r.data["required_no_default"], list)

    def test_unknown_step_not_found(self):
        r = call_tool("author.config_constraints", {"step_name": "NoSuchStepXYZ"})
        assert not r.ok
        assert r.code == "not_found"

    def test_missing_step_name_invalid_input(self):
        r = call_tool("author.config_constraints", {})
        assert not r.ok
        assert r.code == "invalid_input"


class TestAuthorPreflightConfig:
    """author.preflight_config model_validate's a config VALUE set — the value-level gate that
    catches the wrong-enum / missing-required bugs author.preflight_step (field-coverage) cannot."""

    def _base(self):
        return {
            "author": "x",
            "bucket": "b",
            "role": "r",
            "region": "NA",
            "service_name": "s",
            "pipeline_version": "1.0.0",
            "project_root_folder": "p",
            "job_type": "training",
            "source_dir": "dockers",
        }

    def test_valid_values_pass(self):
        vals = {**self._base(), "output_format": "Parquet"}
        r = call_tool(
            "author.preflight_config",
            {"step_name": "TabularPreprocessing", "values": vals},
        )
        assert r.ok
        assert r.data["valid"] is True

    def test_normalized_lowercase_enum_now_passes(self):
        """The Munged friction bug: 'parquet' was rejected; now normalized -> valid."""
        vals = {**self._base(), "output_format": "parquet"}
        r = call_tool(
            "author.preflight_config",
            {"step_name": "TabularPreprocessing", "values": vals},
        )
        assert r.ok

    def test_invalid_enum_caught_with_field_error(self):
        vals = {**self._base(), "output_format": "xml"}
        r = call_tool(
            "author.preflight_config",
            {"step_name": "TabularPreprocessing", "values": vals},
        )
        assert not r.ok
        assert r.code == "config_invalid"
        fields = {e["field"] for e in r.meta["details"]["errors"]}
        assert "output_format" in fields

    def test_missing_required_field_caught(self):
        vals = {k: v for k, v in self._base().items() if k != "author"}
        vals["output_format"] = "CSV"
        r = call_tool(
            "author.preflight_config",
            {"step_name": "TabularPreprocessing", "values": vals},
        )
        assert not r.ok
        assert any(e["field"] == "author" for e in r.meta["details"]["errors"])

    def test_non_dict_values_invalid_input(self):
        r = call_tool(
            "author.preflight_config",
            {"step_name": "TabularPreprocessing", "values": "x"},
        )
        assert not r.ok
        assert r.code == "invalid_input"


class TestAuthorCheckScript:
    """author.check_script runs both forward + reverse script<->contract alignment, and skips
    script-less / SDK-delegation steps."""

    def test_real_processing_step_passes(self):
        r = call_tool("author.check_script", {"step_name": "TabularPreprocessing"})
        assert r.ok
        assert r.data["status"] == "checked"
        assert r.data["passed"] is True

    def test_standardized_eval_step_passes(self):
        """After standardization the eval scripts read only canonical contract keys -> clean."""
        r = call_tool("author.check_script", {"step_name": "XGBoostModelEval"})
        assert r.ok
        assert r.data["passed"] is True

    def test_sdk_delegation_step_skipped(self):
        r = call_tool("author.check_script", {"step_name": "RedshiftDataLoading"})
        assert r.ok
        assert r.data["status"] == "skipped"

    def test_script_less_step_skipped(self):
        r = call_tool("author.check_script", {"step_name": "XGBoostModel"})
        assert r.ok
        assert r.data["status"] == "skipped"

    def test_training_step_no_false_arg_error(self):
        """Training steps declare --job_type in job_arguments but take args via JSON -> must not ERROR."""
        r = call_tool("author.check_script", {"step_name": "PyTorchTraining"})
        assert r.ok
        assert r.data["passed"] is True

    def test_missing_step_name_invalid_input(self):
        r = call_tool("author.check_script", {})
        assert not r.ok
        assert r.code == "invalid_input"
