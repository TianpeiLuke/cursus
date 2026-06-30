"""
Tests for the cursus.mcp ``steps.io`` tool — the agent-facing per-step connection / I-O view.

End-to-end through call_tool against the real .step.yaml interfaces. Covers the structured shape,
the nested training channels (single-sourced from the YAML), the property-path references, job_type
variant resolution, and not-found handling.
"""

from cursus.mcp import call_tool


class TestStepsIo:
    def test_training_io_shape(self):
        r = call_tool("steps.io", {"step_name": "XGBoostTraining"})
        assert r.ok
        assert r.data["sagemaker_step_type"] == "Training"
        assert r.data["dependency_count"] == len(r.data["inputs"])
        assert r.data["output_count"] == len(r.data["outputs"])
        ip = next(i for i in r.data["inputs"] if i["logical_name"] == "input_path")
        assert ip["container_path"] == "/opt/ml/input/data"
        assert ip["channels"] == ["train", "val", "test"]
        mo = next(o for o in r.data["outputs"] if o["logical_name"] == "model_output")
        assert mo["property_path"] == "properties.ModelArtifacts.S3ModelArtifacts"

    def test_channels_come_from_yaml_not_hardcode(self):
        """The channels are read from contract.input_channels (the .step.yaml), the single source."""
        from cursus.steps.interfaces import load_interface

        iface = load_interface("XGBoostTraining")
        assert iface.contract.input_channels == {"input_path": ["train", "val", "test"]}

    def test_processing_step_has_no_channels(self):
        r = call_tool("steps.io", {"step_name": "TabularPreprocessing"})
        assert r.ok
        assert all("channels" not in i for i in r.data["inputs"])

    def test_job_type_variant_required_flip(self):
        base = call_tool("steps.io", {"step_name": "RiskTableMapping"})
        val = call_tool("steps.io", {"step_name": "RiskTableMapping", "job_type": "validation"})
        assert base.ok and val.ok

        def required(data, name):
            return next(i["required"] for i in data["inputs"] if i["logical_name"] == name)

        assert required(base.data, "model_artifacts_input") is False
        assert required(val.data, "model_artifacts_input") is True

    def test_unknown_step_not_found_with_remedy(self):
        r = call_tool("steps.io", {"step_name": "Nonexistent"})
        assert not r.ok
        assert r.code == "not_found"
        assert "catalog.list_steps" in (r.remedy or {}).get("suggested_tools", [])

    def test_missing_step_name_invalid_input(self):
        r = call_tool("steps.io", {})
        assert not r.ok
        assert r.code == "invalid_input"


class TestStepsPatternsDependencyAxis:
    """The dependency-axis rollup flows through the steps.patterns MCP tool (FZ 31e1d3l)."""

    def test_native_step_reports_no_dependency(self):
        r = call_tool("steps.patterns", {"step_name": "TabularPreprocessing"})
        assert r.ok
        assert r.data["dependencies"]["native"] is True
        assert r.data["patterns"]["compute"]["requires"] == "none"

    def test_sais_sdk_step_reports_build_time_create_step_dep(self):
        r = call_tool("steps.patterns", {"step_name": "Registration"})
        assert r.ok
        assert (
            r.data["dependencies"]["build_time"]["create_step"]
            == "secure_ai_sandbox_workflow_python_sdk"
        )

    def test_edx_reports_compute_mods_and_runtime_lib(self):
        r = call_tool("steps.patterns", {"step_name": "EdxUploading"})
        assert r.ok
        assert r.data["patterns"]["compute"]["requires"] == "mods_workflow_core"
        assert "secure_ai_sandbox_python_lib" in r.data["dependencies"]["runtime"]
