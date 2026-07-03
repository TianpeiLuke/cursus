"""Tests for `cursus validate step-interface` (FZ 31e1d3f2).

Author-time validation of a `.step.yaml`: loads it through the production StepInterface.from_yaml
path + runs incompleteness checks (compatible_sources case-typos). Exercised through click's
CliRunner against the real interfaces.
"""

import json

import pytest
from click.testing import CliRunner

from cursus.cli.validate_cli import validate_cli


@pytest.fixture
def runner():
    return CliRunner()


class TestValidateStepInterface:
    def test_valid_step_passes(self, runner):
        r = runner.invoke(validate_cli, ["step-interface", "XGBoostTraining"])
        assert r.exit_code == 0, r.output
        assert "✅" in r.output
        assert "XGBoostTraining" in r.output

    def test_job_type_variant_resolves(self, runner):
        r = runner.invoke(
            validate_cli, ["step-interface", "RiskTableMapping", "--job-type", "validation"]
        )
        assert r.exit_code == 0, r.output
        assert "[validation]" in r.output

    def test_unknown_step_errors_nonzero(self, runner):
        r = runner.invoke(validate_cli, ["step-interface", "NoSuchStep"])
        assert r.exit_code == 1
        assert "not found" in r.output.lower()

    def test_requires_name_or_all(self, runner):
        r = runner.invoke(validate_cli, ["step-interface"])
        assert r.exit_code == 2

    def test_all_validates_every_interface_green(self, runner):
        import re

        r = runner.invoke(validate_cli, ["step-interface", "--all"])
        assert r.exit_code == 0, r.output
        # the whole fleet must be clean (no blocking errors); ~45+ steps
        assert "0 error(s)" in r.output
        # Parse the actual validated count rather than substring-matching a fixed prefix
        # (the old `"validated 4" in output` broke once the count wasn't 4X — e.g. 53).
        m = re.search(r"validated (\d+)", r.output)
        assert m, r.output
        assert int(m.group(1)) >= 40, r.output

    def test_json_shape(self, runner):
        r = runner.invoke(validate_cli, ["step-interface", "XGBoostTraining", "--format", "json"])
        assert r.exit_code == 0, r.output
        data = json.loads(r.output)
        assert data["validated"] == 1 and data["errors"] == 0
        assert data["results"][0]["step"] == "XGBoostTraining"
        assert data["results"][0]["ok"] is True

    def test_all_json_has_no_blocking_errors(self, runner):
        r = runner.invoke(validate_cli, ["step-interface", "--all", "--format", "json"])
        assert r.exit_code == 0, r.output
        data = json.loads(r.output)
        assert data["errors"] == 0, [x for x in data["results"] if not x["ok"]]
        # the compatible_sources case-typos were fixed (FZ 31e1d3f gap 3) — no warnings either
        assert data["warnings"] == 0, [
            (x["step"], x["warnings"]) for x in data["results"] if x["warnings"]
        ]
