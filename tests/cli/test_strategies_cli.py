"""
Tests for the ``cursus strategies`` CLI (the strategy-library introspection surface).

Exercised through click's ``CliRunner`` against the real strategy_registry — the registry is
deterministic, so no mocking is needed. Covers the five subcommands (axes / list / show / for /
knobs) in both text and JSON form plus the error/exit-code paths.
"""

import json

import pytest
from click.testing import CliRunner

from cursus.cli.strategies_cli import strategies_cli


@pytest.fixture
def runner():
    return CliRunner()


class TestAxes:
    def test_text(self, runner):
        result = runner.invoke(strategies_cli, ["axes"])
        assert result.exit_code == 0
        assert "sagemaker_step_type" in result.output
        assert "step_assembly" in result.output

    def test_json(self, runner):
        result = runner.invoke(strategies_cli, ["axes", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        axes = [a["axis"] for a in data]
        assert axes == ["sagemaker_step_type", "step_assembly"]
        assert all(a["strategy_count"] >= 1 for a in data)


class TestList:
    def test_text_lists_known_strategies(self, runner):
        result = runner.invoke(strategies_cli, ["list"])
        assert result.exit_code == 0
        for name in ("Training", "CreateModel", "Transform", "code", "step_args"):
            assert name in result.output

    def test_filter_by_axis_json(self, runner):
        result = runner.invoke(
            strategies_cli, ["list", "--axis", "step_assembly", "--format", "json"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert {s["name"] for s in data} == {"code", "step_args", "delegation"}
        assert all(s["axis"] == "step_assembly" for s in data)


class TestShow:
    def test_show_training_text(self, runner):
        result = runner.invoke(strategies_cli, ["show", "Training"])
        assert result.exit_code == 0
        assert "TrainingHandler" in result.output
        assert "direct_input_keys" in result.output

    def test_show_json(self, runner):
        result = runner.invoke(strategies_cli, ["show", "Training", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["handler"] == "TrainingHandler"
        # output_path_token is no longer a knob — the S3 prefix is derived from the step name.
        assert {k["name"] for k in data["knobs"]} >= {"direct_input_keys"}
        assert "output_path_token" not in {k["name"] for k in data["knobs"]}

    def test_show_unknown_exits_nonzero(self, runner):
        result = runner.invoke(strategies_cli, ["show", "Nonexistent"])
        assert result.exit_code == 1
        assert "No strategy named" in result.output


class TestForStepType:
    def test_training_text(self, runner):
        result = runner.invoke(strategies_cli, ["for", "Training"])
        assert result.exit_code == 0
        assert "TrainingHandler" in result.output
        assert "sagemaker_step_type = Training" in result.output

    def test_processing_step_args_json(self, runner):
        result = runner.invoke(
            strategies_cli,
            ["for", "Processing", "--step-assembly", "step_args", "--format", "json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["resolved_axis"] == "step_assembly"
        assert data["resolved_name"] == "step_args"
        assert data["strategy"]["preset_knobs"]["use_step_args"] is True

    def test_processing_defaults_to_code(self, runner):
        result = runner.invoke(strategies_cli, ["for", "Processing", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["resolved_name"] == "code"

    def test_base_binds_no_builder_exits_nonzero(self, runner):
        result = runner.invoke(strategies_cli, ["for", "Base"])
        assert result.exit_code == 1
        assert "binds no builder" in result.output


class TestKnobs:
    def test_text(self, runner):
        result = runner.invoke(
            strategies_cli, ["knobs", "--axis", "step_assembly", "--name", "code"]
        )
        assert result.exit_code == 0
        assert "use_step_args" in result.output
        assert "split_source_dir" in result.output

    def test_json(self, runner):
        result = runner.invoke(
            strategies_cli,
            ["knobs", "--axis", "sagemaker_step_type", "--name", "CreateModel", "--format", "json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert {k["name"] for k in data} == {"make_compute", "direct_input_keys"}

    def test_unknown_exits_nonzero(self, runner):
        result = runner.invoke(
            strategies_cli, ["knobs", "--axis", "sagemaker_step_type", "--name", "Nope"]
        )
        assert result.exit_code == 1


class TestRegisteredInRootCli:
    def test_strategies_is_registered_under_root(self):
        from cursus.cli import cli

        assert "strategies" in cli.commands
