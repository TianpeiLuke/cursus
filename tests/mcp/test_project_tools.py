"""
Tests for the cursus.mcp ``project.*`` namespace — the project scaffolder.

End-to-end through call_tool. ``project.init`` is deterministic and writes to a temp dir, so we
assert the actual files land, the three .py files compile, dag.json parses, project_root_folder is
filled, and the import prefix follows the target. ``project.bring_up`` returns the orchestrator
invocation. Input validation and the overwrite guard are covered too.
"""

import json
import py_compile
from pathlib import Path

import pytest

from cursus.mcp import call_tool


class TestProjectInit:
    def test_scaffolds_expected_tree(self, tmp_path):
        r = call_tool(
            "project.init",
            {
                "name": "secure_delivery",
                "framework": "xgboost",
                "target_dir": str(tmp_path),
            },
        )
        assert r.ok, r.error
        root = Path(r.data["target_path"])
        assert root.name == "secure_delivery_xgboost"
        # the fixed files + folder READMEs
        for rel in [
            "__init__.py",
            "run_pipeline.py",
            "secure_delivery_xgboost_pipeline.py",
            "generate_config.py",
            "pipeline_config/dag.json",
            "pipeline_config/README.md",
            "dockers/__init__.py",
            "dockers/README.md",
            "dockers/scripts/README.md",
            "dockers/processing/README.md",
            "dockers/hyperparams/README.md",
            "README.md",
        ]:
            assert (root / rel).exists(), f"missing scaffolded file: {rel}"

    def test_three_py_files_compile(self, tmp_path):
        r = call_tool(
            "project.init",
            {"name": "demo", "framework": "pytorch", "target_dir": str(tmp_path)},
        )
        assert r.ok, r.error
        root = Path(r.data["target_path"])
        for f in ["run_pipeline.py", "demo_pytorch_pipeline.py", "generate_config.py"]:
            py_compile.compile(
                str(root / f), doraise=True
            )  # raises PyCompileError on syntax error

    def test_dag_json_is_empty_valid_stub(self, tmp_path):
        r = call_tool(
            "project.init",
            {"name": "demo", "framework": "xgboost", "target_dir": str(tmp_path)},
        )
        root = Path(r.data["target_path"])
        dag = json.load(open(root / "pipeline_config" / "dag.json"))
        assert dag == {"dag": {"nodes": [], "edges": []}}

    def test_project_root_folder_is_filled(self, tmp_path):
        r = call_tool(
            "project.init",
            {
                "name": "abuse_polygraph",
                "framework": "xgboost",
                "target_dir": str(tmp_path),
            },
        )
        root = Path(r.data["target_path"])
        gc = (root / "generate_config.py").read_text()
        assert 'project_root_folder="abuse_polygraph_xgboost"' in gc

    def test_modstemplate_class_loads_dag_json(self, tmp_path):
        r = call_tool(
            "project.init",
            {
                "name": "secure_delivery",
                "framework": "xgboost",
                "target_dir": str(tmp_path),
            },
        )
        root = Path(r.data["target_path"])
        tpl = (root / "secure_delivery_xgboost_pipeline.py").read_text()
        assert "@MODSTemplate(" in tpl
        assert "class SecureDeliveryPipeline" in tpl
        assert "import_dag_from_json" in tpl  # loads the DAG, never inline

    def test_import_prefix_dev_vs_bamt(self, tmp_path):
        dev = call_tool(
            "project.init",
            {
                "name": "a",
                "framework": "xgboost",
                "target_dir": str(tmp_path / "projects"),
            },
        )
        bamt = call_tool(
            "project.init",
            {
                "name": "b",
                "framework": "xgboost",
                "target_dir": str(tmp_path / "src/buyer_abuse_mods_template"),
            },
        )
        assert (
            "from cursus.api.dag import import_dag_from_json"
            in (Path(dev.data["target_path"]) / "run_pipeline.py").read_text()
        )
        assert (
            "from buyer_abuse_mods_template.cursus.api.dag import import_dag_from_json"
            in (Path(bamt.data["target_path"]) / "run_pipeline.py").read_text()
        )

    def test_ledger_lists_action_items(self, tmp_path):
        r = call_tool(
            "project.init",
            {"name": "demo", "framework": "xgboost", "target_dir": str(tmp_path)},
        )
        readme = (Path(r.data["target_path"]) / "README.md").read_text()
        assert "Action items" in readme
        assert "/cursus-author-step" in readme
        assert "/cursus-configure-pipeline" in readme
        assert "What is already done" in readme

    def test_next_steps_present(self, tmp_path):
        r = call_tool(
            "project.init",
            {"name": "demo", "framework": "xgboost", "target_dir": str(tmp_path)},
        )
        names = [s.get("tool") for s in r.next_steps]
        assert "/cursus-configure-pipeline" in names

    def test_bad_framework_invalid_input(self, tmp_path):
        r = call_tool(
            "project.init",
            {"name": "demo", "framework": "sklearn", "target_dir": str(tmp_path)},
        )
        assert not r.ok
        assert r.code == "invalid_input"

    def test_missing_name_invalid_input(self, tmp_path):
        r = call_tool(
            "project.init", {"framework": "xgboost", "target_dir": str(tmp_path)}
        )
        assert not r.ok
        assert r.code == "invalid_input"

    def test_existing_target_guard(self, tmp_path):
        first = call_tool(
            "project.init",
            {"name": "demo", "framework": "xgboost", "target_dir": str(tmp_path)},
        )
        assert first.ok
        again = call_tool(
            "project.init",
            {"name": "demo", "framework": "xgboost", "target_dir": str(tmp_path)},
        )
        assert not again.ok
        assert again.code == "already_exists"
        # overwrite=true is allowed
        forced = call_tool(
            "project.init",
            {
                "name": "demo",
                "framework": "xgboost",
                "target_dir": str(tmp_path),
                "overwrite": True,
            },
        )
        assert forced.ok


class TestProjectBringUp:
    def test_returns_orchestrator_invocation(self):
        r = call_tool(
            "project.bring_up", {"name": "secure_delivery", "framework": "xgboost"}
        )
        assert r.ok, r.error
        assert r.data["workflow"] == "cursus-new-project"
        assert r.data["args"]["name"] == "secure_delivery"
        assert r.data["args"]["framework"] == "xgboost"
        assert r.data["args"]["dag_source"] == "catalog"

    def test_manual_dag_source(self):
        r = call_tool(
            "project.bring_up",
            {"name": "m", "framework": "pytorch", "dag_source": "manual"},
        )
        assert r.ok
        assert r.data["args"]["dag_source"] == "manual"

    def test_bad_dag_source_invalid_input(self):
        r = call_tool(
            "project.bring_up",
            {"name": "m", "framework": "pytorch", "dag_source": "bogus"},
        )
        assert not r.ok
        assert r.code == "invalid_input"


class TestProjectNamespaceRegistered:
    def test_tools_discovered(self):
        from cursus.mcp import list_tools

        names = {t.name for t in list_tools(namespace="project")}
        assert {"project.init", "project.bring_up"} <= names
