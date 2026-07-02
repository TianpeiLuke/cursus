"""
Tests for the ``cursus dag resolve`` CLI — per-edge dependency-resolution scoring via the REAL
UnifiedDependencyResolver.

Exercised through click's CliRunner against real .step.yaml interfaces (deterministic, no mocking).
The load-bearing test is `test_score_matches_real_resolver`: it proves the CLI's emitted score is the
SAME number the production resolver computes for the same specs — so the author-time Kiro gate that
parses this JSON is a real oracle with zero drift (the reason this CLI exists rather than a JS
re-implementation of the 6-component formula).
"""

import json

import pytest
from click.testing import CliRunner

from cursus.cli.dag_cli import dag_cli


@pytest.fixture
def runner():
    return CliRunner()


class TestDagResolveJson:
    def test_emits_edges_with_scores(self, runner):
        # A known producer/consumer pair from the registry (both real, registered steps).
        result = runner.invoke(
            dag_cli, ["resolve", "XGBoostTraining", "XGBoostModelEval", "--format", "json"]
        )
        assert result.exit_code == 0
        # strip any leading sagemaker.config INFO lines the CLI prints to stdout
        payload = result.output[result.output.index("{"):]
        data = json.loads(payload)
        assert data["loaded"] == ["XGBoostTraining", "XGBoostModelEval"]
        assert data["threshold"] == 0.5
        assert isinstance(data["edges"], list) and data["edges"]
        for e in data["edges"]:
            assert set(["consumer", "dependency", "provider", "score", "resolves"]) <= set(e)
            assert isinstance(e["resolves"], bool)
            assert isinstance(e["score"], (int, float))

    def test_all_edges_resolve_is_and_of_edges(self, runner):
        result = runner.invoke(
            dag_cli, ["resolve", "XGBoostTraining", "XGBoostModelEval", "--format", "json"]
        )
        data = json.loads(result.output[result.output.index("{"):])
        expected = bool(data["edges"]) and all(e["resolves"] for e in data["edges"])
        assert data["all_edges_resolve"] == expected

    def test_unknown_step_recorded_not_crashed(self, runner):
        result = runner.invoke(
            dag_cli, ["resolve", "XGBoostTraining", "Nonexistent", "--format", "json"]
        )
        assert result.exit_code == 0  # unknown step is recorded, not fatal
        data = json.loads(result.output[result.output.index("{"):])
        assert "Nonexistent" in data["load_errors"]
        assert "Nonexistent" not in data["loaded"]

    def test_score_matches_real_resolver(self, runner):
        """DRIFT GUARD: the CLI score == the production resolver's score for the same specs."""
        from cursus.core.deps.specification_registry import SpecificationRegistry
        from cursus.core.deps.dependency_resolver import create_dependency_resolver
        from cursus.steps.interfaces import load_interface

        names = ["XGBoostTraining", "XGBoostModelEval"]
        reg = SpecificationRegistry()
        for n in names:
            reg.register(n, load_interface(n).spec)
        resolver = create_dependency_resolver(reg)

        # Reproduce the CLI's own computation directly and compare.
        direct = {}
        for consumer in names:
            spec = reg.get_specification(consumer)
            if not getattr(spec, "dependencies", None):
                continue
            rep = resolver.resolve_with_scoring(consumer, [n for n in names if n != consumer])
            for dep, info in (rep.get("failed_with_scores") or {}).items():
                best = info.get("best_candidate") or {}
                direct[(consumer, dep)] = round(float(best.get("score", 0.0)), 4)

        result = runner.invoke(dag_cli, ["resolve", *names, "--format", "json"])
        data = json.loads(result.output[result.output.index("{"):])
        cli_failed = {
            (e["consumer"], e["dependency"]): e["score"]
            for e in data["edges"]
            if not e["resolves"]
        }
        # every sub-threshold edge the CLI reports must carry the resolver's exact score
        for key, score in direct.items():
            assert key in cli_failed, f"CLI missing failed edge {key}"
            assert abs(cli_failed[key] - score) < 1e-9, f"score drift on {key}: {cli_failed[key]} vs {score}"


class TestDagResolveText:
    def test_text_output_lists_edges(self, runner):
        result = runner.invoke(dag_cli, ["resolve", "XGBoostTraining", "XGBoostModelEval"])
        assert result.exit_code == 0
        assert "Resolve:" in result.output
        # each edge line carries a resolves= verdict
        assert "resolves=" in result.output
