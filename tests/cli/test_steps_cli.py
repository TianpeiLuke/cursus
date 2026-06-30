"""
Tests for the ``cursus steps io`` CLI — the per-step connection / I-O view.

Exercised through click's CliRunner against the real .step.yaml interfaces (deterministic, no
mocking). Covers the container-path view, the nested training-channel fan-out, the property-path
references, job_type variant resolution, and the not-found path.
"""

import json

import pytest
from click.testing import CliRunner

from cursus.cli.steps_cli import steps_cli


@pytest.fixture
def runner():
    return CliRunner()


class TestStepsIoText:
    def test_training_shows_paths_channels_and_refs(self, runner):
        result = runner.invoke(steps_cli, ["io", "XGBoostTraining"])
        assert result.exit_code == 0
        # container path + nested channels for the fan-out input
        assert "/opt/ml/input/data" in result.output
        assert "channels: train val test" in result.output
        # producer property-path reference
        assert "properties.ModelArtifacts.S3ModelArtifacts" in result.output

    def test_processing_step_has_no_channels(self, runner):
        result = runner.invoke(steps_cli, ["io", "TabularPreprocessing"])
        assert result.exit_code == 0
        assert "/opt/ml/processing/output" in result.output
        # non-Training steps do not show a channels fan-out line
        assert "channels:" not in result.output

    def test_unknown_step_exits_nonzero(self, runner):
        result = runner.invoke(steps_cli, ["io", "Nonexistent"])
        assert result.exit_code == 1
        assert "No interface found" in result.output


class TestStepsIoJson:
    def test_json_shape(self, runner):
        result = runner.invoke(steps_cli, ["io", "XGBoostTraining", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["step_name"] == "XGBoostTraining"
        assert data["sagemaker_step_type"] == "Training"
        ip = next(i for i in data["inputs"] if i["logical_name"] == "input_path")
        assert ip["container_path"] == "/opt/ml/input/data"
        assert ip["channels"] == ["train", "val", "test"]
        mo = next(o for o in data["outputs"] if o["logical_name"] == "model_output")
        assert mo["property_path"] == "properties.ModelArtifacts.S3ModelArtifacts"

    def test_job_type_variant_changes_required_flag(self, runner):
        """The validation variant makes model_artifacts_input required (vs optional in base)."""
        base = json.loads(
            runner.invoke(steps_cli, ["io", "RiskTableMapping", "--format", "json"]).output
        )
        val = json.loads(
            runner.invoke(
                steps_cli, ["io", "RiskTableMapping", "--job-type", "validation", "--format", "json"]
            ).output
        )

        def required(view, name):
            return next(i["required"] for i in view["inputs"] if i["logical_name"] == name)

        assert required(base, "model_artifacts_input") is False
        assert required(val, "model_artifacts_input") is True


class TestStepsPatterns:
    """The ``cursus steps patterns`` view — the per-axis construction patterns (FZ 31e1d3j),
    derived from the .step.yaml + registry binding (no separate patterns: field, cannot drift)."""

    def test_text_shows_all_five_axes(self, runner):
        result = runner.invoke(steps_cli, ["patterns", "TabularPreprocessing"])
        assert result.exit_code == 0, result.output
        for axis in ("create_step", "env_vars", "job_arguments", "inputs", "outputs"):
            assert axis in result.output
        # handler binding + the standard input loop are shown
        assert "ProcessingHandler" in result.output
        assert "standard spec" in result.output

    def test_json_shape_and_derived_handler(self, runner):
        out = runner.invoke(steps_cli, ["patterns", "XGBoostTraining", "--format", "json"]).output
        view = json.loads(out)
        p = view["patterns"]
        assert set(p) == {"create_step", "env_vars", "job_arguments", "inputs", "outputs", "compute"}
        assert p["create_step"]["handler"] == "TrainingHandler"
        # XGBoostTraining declares an estimator compute descriptor surfaced to the user.
        assert p["compute"]["kind"] == "estimator"
        assert p["compute"]["sdk_class"] == "XGBoost"

    def test_input_deviation_surfaced(self, runner):
        # PercentileModelCalibration declares skip_inputs -> shown as an input deviation.
        out = runner.invoke(
            steps_cli, ["patterns", "PercentileModelCalibration", "--format", "json"]
        ).output
        view = json.loads(out)
        assert "calibration_config" in str(view["patterns"]["inputs"]["deviations"])

    def test_sink_output_pattern(self, runner):
        out = runner.invoke(steps_cli, ["patterns", "EdxUploading", "--format", "json"]).output
        view = json.loads(out)
        assert view["patterns"]["outputs"]["sink"] is True

    def test_dependency_axis_native_step(self, runner):
        """A native (sagemaker-only) step reports no 3rd-party dependency on any axis (FZ 31e1d3l)."""
        out = runner.invoke(steps_cli, ["patterns", "TabularPreprocessing", "--format", "json"]).output
        view = json.loads(out)
        assert view["dependencies"]["native"] is True
        assert view["dependencies"]["build_time"] == {}
        assert view["dependencies"]["runtime"] == []
        assert view["patterns"]["compute"]["requires"] == "none"
        # text mode says so on one line
        text = runner.invoke(steps_cli, ["patterns", "TabularPreprocessing"]).output
        assert "requires      (none" in text

    def test_dependency_axis_compute_mods_and_runtime(self, runner):
        """EdxUploading: build-time compute dep on mods_workflow_core + a runtime script dep."""
        out = runner.invoke(steps_cli, ["patterns", "EdxUploading", "--format", "json"]).output
        view = json.loads(out)
        assert view["patterns"]["compute"]["requires"] == "mods_workflow_core"
        assert view["dependencies"]["build_time"] == {"compute": "mods_workflow_core"}
        assert "secure_ai_sandbox_python_lib" in view["dependencies"]["runtime"]
        assert view["dependencies"]["native"] is False

    def test_dependency_axis_create_step_sais_sdk(self, runner):
        """Registration: a HARD module-level SAIS-SDK dep on the create_step axis."""
        out = runner.invoke(steps_cli, ["patterns", "Registration", "--format", "json"]).output
        view = json.loads(out)
        assert view["patterns"]["create_step"]["requires"] == "secure_ai_sandbox_workflow_python_sdk"
        assert view["dependencies"]["build_time"]["create_step"] == "secure_ai_sandbox_workflow_python_sdk"
        text = runner.invoke(steps_cli, ["patterns", "Registration"]).output
        assert "HARD module-level" in text

    def test_bedrock_is_pure_shell_no_custom_override(self, runner):
        # FZ 31e1d3g3 Phase A3: BedrockBatchProcessing's _get_environment_variables override was
        # reduced to the declarative COMPUTED-S3-ENV pattern (contract.computed_env_paths), so it is
        # now a pure shell — the io_view must flag NO custom override on any axis (incl. env_vars).
        out = runner.invoke(
            steps_cli, ["patterns", "BedrockBatchProcessing", "--format", "json"]
        ).output
        view = json.loads(out)
        assert view["patterns"]["env_vars"]["custom_override"] is False
        assert all(
            axis.get("custom_override") is False
            for axis in view["patterns"].values()
            if isinstance(axis, dict)
        )

    def test_unknown_step_exits_nonzero(self, runner):
        result = runner.invoke(steps_cli, ["patterns", "Nonexistent"])
        assert result.exit_code != 0

    def test_sdk_bound_step_is_declarative_shell(self, runner):
        """FZ 31e1d3g3 Phase E: the per-step builder files are deleted, so Registration (a SAIS-bound
        SDK-delegation step) is now a SYNTHESIZED fileless shell. Offline (SAIS absent) its class
        can't load AND there is no source file to scan, so the view reports it as a declarative shell
        with no overrides — faithfully, not as a 'may be incomplete' failure."""
        out = runner.invoke(
            steps_cli, ["patterns", "Registration", "--format", "json"]
        ).output
        view = json.loads(out)
        assert "declarative shell" in (view.get("note") or "")
        assert all(
            axis.get("custom_override") is False
            for axis in view["patterns"].values()
            if isinstance(axis, dict)
        )

    def test_custom_override_matches_actual_builder_defs(self):
        """The whole point of the view — its custom_override flags must equal the methods each
        builder actually defines (ground truth = the builder source). Audited across all steps."""
        import glob
        import re

        from cursus.registry.step_names import get_step_names
        from cursus.steps.interfaces.io_view import describe_step_patterns

        # Each axis -> the builder method(s) whose presence in source means a custom override.
        # The 'compute' axis maps to ANY of the 5 compute-factory methods (FZ 31e1d3k).
        axis_methods = {
            "create_step": ["create_step"],
            "env_vars": ["_get_environment_variables"],
            "job_arguments": ["_get_job_arguments"],
            "inputs": ["_get_inputs"],
            "outputs": ["_get_outputs"],
            "compute": [
                "_create_processor", "_get_processor", "_create_estimator",
                "_create_model", "_create_transformer",
            ],
        }
        all_methods = [m for meths in axis_methods.values() for m in meths]
        truth = {}
        for f in glob.glob("src/cursus/steps/builders/builder_*_step.py"):
            src = open(f).read()
            m = re.search(r'STEP_NAME\s*=\s*"([^"]+)"', src)
            if not m:
                continue
            defined = {meth for meth in all_methods if re.search(rf"\n    def {meth}\b", src)}
            # collapse to the set of AXES the builder overrides (compute = any compute method)
            truth[m.group(1)] = {
                ax for ax, meths in axis_methods.items() if defined & set(meths)
            }
        mismatches = []
        for step in get_step_names():
            try:
                v = describe_step_patterns(step)
            except Exception:
                continue
            claimed = {
                ax for ax in v["patterns"] if v["patterns"][ax].get("custom_override")
            }
            if claimed != truth.get(step, set()):
                mismatches.append((step, sorted(claimed), sorted(truth.get(step, set()))))
        assert not mismatches, f"patterns view custom_override drifted from builder source: {mismatches}"


class TestRegisteredUnderRoot:
    def test_steps_registered(self):
        from cursus.cli import cli

        assert "steps" in cli.commands
