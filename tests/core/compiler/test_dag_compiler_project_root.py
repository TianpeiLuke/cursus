"""
Tests for PipelineDAGCompiler project-root (caller-hook) wiring.

Verifies the compiler resolves a project-root anchor (explicit or inferred from config_path)
and pushes it process-wide so config path resolution can use it as Strategy 0.
"""

import json
import tempfile
from pathlib import Path

import pytest

from cursus.core.compiler.dag_compiler import PipelineDAGCompiler
from cursus.core.utils.hybrid_path_resolution import set_project_root, get_project_root


@pytest.fixture(autouse=True)
def _clear_pushed_root():
    set_project_root(None)
    yield
    set_project_root(None)


def _write_config(dirpath: Path) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    cfg = dirpath / "config.json"
    cfg.write_text(json.dumps({"configuration": {"shared": {}, "specific": {}}}))
    return cfg


class TestResolveProjectRoot:
    """_resolve_project_root is a pure staticmethod — test its inference directly."""

    def test_explicit_wins(self):
        out = PipelineDAGCompiler._resolve_project_root("/abs/proj", "/x/pipeline_config/c.json")
        assert out == str(Path("/abs/proj"))

    def test_infer_from_pipeline_config_dir(self):
        with tempfile.TemporaryDirectory() as d:
            proj = Path(d) / "myproj"
            cfg = _write_config(proj / "pipeline_config")
            out = PipelineDAGCompiler._resolve_project_root(None, str(cfg))
            assert out == str(proj.resolve())

    def test_infer_from_versioned_subdir(self):
        with tempfile.TemporaryDirectory() as d:
            proj = Path(d) / "myproj"
            cfg = _write_config(proj / "pipeline_config" / "v2")
            out = PipelineDAGCompiler._resolve_project_root(None, str(cfg))
            assert out == str(proj.resolve())

    def test_infer_plain_dir(self):
        with tempfile.TemporaryDirectory() as d:
            proj = Path(d) / "flat"
            cfg = _write_config(proj)  # config.json directly in the project dir
            out = PipelineDAGCompiler._resolve_project_root(None, str(cfg))
            assert out == str(proj.resolve())


class TestCompilerPushesProjectRoot:
    def test_explicit_project_root_is_pushed(self):
        with tempfile.TemporaryDirectory() as d:
            proj = Path(d) / "myproj"
            cfg = _write_config(proj / "pipeline_config")
            PipelineDAGCompiler(config_path=str(cfg), project_root=str(proj))
            assert get_project_root() == str(proj.resolve())

    def test_inferred_project_root_is_pushed(self):
        with tempfile.TemporaryDirectory() as d:
            proj = Path(d) / "myproj"
            cfg = _write_config(proj / "pipeline_config")
            PipelineDAGCompiler(config_path=str(cfg))
            assert get_project_root() == str(proj.resolve())

    def test_compiler_stores_project_root_attr(self):
        with tempfile.TemporaryDirectory() as d:
            proj = Path(d) / "myproj"
            cfg = _write_config(proj / "pipeline_config")
            c = PipelineDAGCompiler(config_path=str(cfg), project_root=str(proj))
            assert c.project_root == str(proj.resolve())
