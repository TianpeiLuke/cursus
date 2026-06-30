"""
Tests for the cursus.mcp ``strategies.*`` namespace (the agent-facing strategy-library introspection
twin of ``cursus strategies``).

These run end-to-end through ``call_tool`` against the real assembled registry + the real
strategy_registry — no mocking needed (the tools only read the registry). The key consistency
guard: ``strategies.for_step_type`` resolves the SAME (axis, name) the runtime router binds, so the
tool can never drift from the actual builder routing.
"""

from cursus.mcp import call_tool


class TestListAxes:
    def test_returns_the_two_routing_axes(self):
        r = call_tool("strategies.list_axes", {})
        assert r.ok
        axes = [a["axis"] for a in r.data["axes"]]
        assert axes == ["sagemaker_step_type", "step_assembly"]
        assert r.data["count"] == 2
        # every axis carries at least one strategy
        assert all(a["strategy_count"] >= 1 for a in r.data["axes"])


class TestList:
    def test_lists_all_strategies(self):
        r = call_tool("strategies.list", {})
        assert r.ok
        assert r.data["count"] == len(r.data["strategies"])
        names = {s["name"] for s in r.data["strategies"]}
        # the implemented construction verbs are all present
        assert {"Training", "CreateModel", "Transform", "code", "step_args"} <= names

    def test_filter_by_axis(self):
        r = call_tool("strategies.list", {"axis": "step_assembly"})
        assert r.ok
        assert r.data["axis"] == "step_assembly"
        assert all(s["axis"] == "step_assembly" for s in r.data["strategies"])
        assert {s["name"] for s in r.data["strategies"]} == {"code", "step_args", "delegation"}


class TestShow:
    def test_show_training_full_descriptor(self):
        r = call_tool("strategies.show", {"name": "Training"})
        assert r.ok
        assert r.data["name"] == "Training"
        assert r.data["handler"] == "TrainingHandler"
        assert r.data["verb"] == "Training"
        assert r.data["routable"] is True
        assert r.data["implemented"] is True
        # knobs are fully expanded
        knob_names = {k["name"] for k in r.data["knobs"]}
        assert {"make_compute", "direct_input_keys"} <= knob_names
        assert "output_path_token" not in knob_names  # derived from step name, not a knob

    def test_show_unknown_gives_remedy(self):
        r = call_tool("strategies.show", {"name": "Nonexistent"})
        assert not r.ok
        assert r.code == "not_found"
        assert "strategies.list" in (r.remedy or {}).get("suggested_tools", [])

    def test_show_missing_name_is_invalid_input(self):
        r = call_tool("strategies.show", {})
        assert not r.ok
        assert r.code == "invalid_input"


class TestForStepType:
    def test_training_binds_training_handler(self):
        r = call_tool("strategies.for_step_type", {"sagemaker_step_type": "Training"})
        assert r.ok
        assert r.data["resolved_axis"] == "sagemaker_step_type"
        assert r.data["resolved_name"] == "Training"
        assert r.data["strategy"]["handler"] == "TrainingHandler"

    def test_processing_defaults_to_code_assembly(self):
        r = call_tool("strategies.for_step_type", {"sagemaker_step_type": "Processing"})
        assert r.ok
        # Processing with no step_assembly resolves to step_assembly=code
        assert r.data["resolved_axis"] == "step_assembly"
        assert r.data["resolved_name"] == "code"
        assert r.data["strategy"]["handler"] == "ProcessingHandler"
        assert r.data["strategy"]["preset_knobs"].get("use_step_args") is False

    def test_processing_step_args_binds_step_args_row(self):
        r = call_tool(
            "strategies.for_step_type",
            {"sagemaker_step_type": "Processing", "step_assembly": "step_args"},
        )
        assert r.ok
        assert r.data["resolved_name"] == "step_args"
        assert r.data["strategy"]["preset_knobs"].get("use_step_args") is True

    def test_base_type_binds_no_builder(self):
        r = call_tool("strategies.for_step_type", {"sagemaker_step_type": "Base"})
        assert not r.ok
        assert r.code == "not_found"
        assert "strategies.list" in (r.remedy or {}).get("suggested_tools", [])

    def test_matches_runtime_router_for_every_step_type(self):
        """The tool's resolution must equal what resolve_handler actually binds."""
        from cursus.core.base.builder_templates import resolve_handler
        from cursus.registry import strategy_registry as sr

        for info in sr.list_strategies(axis="sagemaker_step_type"):
            if not info.routable:
                continue
            r = call_tool("strategies.for_step_type", {"sagemaker_step_type": info.name})
            assert r.ok, info.name
            handler = resolve_handler(info.name)
            assert r.data["strategy"]["handler"] == type(handler).__name__


class TestKnobs:
    def test_knobs_for_processing_code(self):
        r = call_tool("strategies.knobs", {"axis": "step_assembly", "name": "code"})
        assert r.ok
        assert r.data["count"] == len(r.data["knobs"])
        names = {k["name"] for k in r.data["knobs"]}
        assert {"use_step_args", "split_source_dir", "include_job_type_in_path"} <= names
        assert "output_path_token" not in names  # derived from step name, not a knob

    def test_knobs_unknown_gives_remedy(self):
        r = call_tool("strategies.knobs", {"axis": "sagemaker_step_type", "name": "Nope"})
        assert not r.ok
        assert r.code == "not_found"

    def test_knobs_missing_args_invalid_input(self):
        r = call_tool("strategies.knobs", {"axis": "step_assembly"})
        assert not r.ok
        assert r.code == "invalid_input"
