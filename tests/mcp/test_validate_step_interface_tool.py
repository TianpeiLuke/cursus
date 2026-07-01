"""
Tests for the cursus.mcp ``validate.step_interface`` tool — the agent-callable author-time
gate (FZ 31e1d3f5, the spine of the agent step-authoring surface).

The tool is a verbatim promotion of ``cursus validate step-interface`` into the ``validate.*``
MCP namespace, so the agent gate, the CLI gate, and the CI ``validate step-interface --all``
gate are LITERALLY one code path. These tests cover: registration + phase tag, the single-step
and ``--all`` (CI) modes, the in-band next_steps / remedy guidance, input guards, and — the
load-bearing one (open question 31e1d3f5b) — that the MCP tool and the CLI helper return
byte-identical results, proving the single-code-path claim.
"""

from cursus.mcp import call_tool
from cursus.mcp.registry import get_tool


class TestRegistration:
    def test_registered_with_validator_tag(self):
        td = get_tool("validate.step_interface")
        assert td is not None
        assert "validator" in td.tags

    def test_schema_accepts_step_name_all_job_type_only(self):
        td = get_tool("validate.step_interface")
        props = td.schema["properties"]
        assert set(props) == {"step_name", "all", "job_type"}
        assert td.schema["additionalProperties"] is False
        assert td.schema.get("required", []) == []


class TestSingleStep:
    def test_valid_step_passes_with_next_steps(self):
        r = call_tool("validate.step_interface", {"step_name": "TabularPreprocessing"})
        assert r.ok
        assert r.data["validated"] == 1
        assert r.data["errors"] == 0
        # author->validate->integrate loop: the clean result points at the constructibility gate.
        assert "validate.alignment" in [s["tool"] for s in r.next_steps]

    def test_result_shape_matches_cli_envelope(self):
        r = call_tool("validate.step_interface", {"step_name": "XGBoostTraining"})
        assert r.ok
        assert set(r.data) == {"validated", "errors", "warnings", "results"}
        one = r.data["results"][0]
        assert set(one) >= {"step", "job_type", "ok", "errors", "warnings"}

    def test_job_type_variant_resolves(self):
        r = call_tool(
            "validate.step_interface",
            {"step_name": "RiskTableMapping", "job_type": "validation"},
        )
        assert r.ok
        assert r.data["results"][0]["job_type"] == "validation"


class TestAllMode:
    def test_all_is_ci_green(self):
        """The CI gate: every shipped .step.yaml validates clean."""
        r = call_tool("validate.step_interface", {"all": True})
        assert r.ok
        assert r.data["errors"] == 0
        assert r.data["validated"] >= 45


class TestGuardsAndErrors:
    def test_missing_step_name_without_all_is_invalid_input(self):
        r = call_tool("validate.step_interface", {})
        assert not r.ok
        assert r.code == "invalid_input"

    def test_unknown_step_fails_with_remedy_not_crash(self):
        r = call_tool("validate.step_interface", {"step_name": "NoSuchStepXYZ"})
        assert not r.ok
        assert r.code == "validation_failed"
        tools = (r.remedy or {}).get("suggested_tools", [])
        assert "strategies.for_step_type" in tools

    def test_non_string_job_type_is_invalid_input(self):
        r = call_tool("validate.step_interface", {"step_name": "X", "job_type": 5})
        assert not r.ok
        assert r.code == "invalid_input"


class TestSingleCodePathParityWithCLI:
    """FZ 31e1d3f5b: the MCP tool and the CLI MUST share one validation code path.

    Both call ``_validate_one_interface`` — assert the per-step results are byte-identical,
    so green-local (CLI) == green-CI == green-agent (MCP) by construction.
    """

    def test_mcp_results_equal_cli_helper(self):
        from cursus.cli.validate_cli import _validate_one_interface

        for step in ("TabularPreprocessing", "XGBoostTraining", "ModelCalibration"):
            mcp = call_tool("validate.step_interface", {"step_name": step})
            cli = _validate_one_interface(step, None)
            assert mcp.data["results"][0] == cli
