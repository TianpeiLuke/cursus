"""
Tests for the caller-hook (Strategy 0) project-root path resolution.

The caller hook lets a pipeline's entry point push its project folder so cursus resolves a
step's relative ``source_dir`` as ``project_root / source_dir`` — making both
``CURSUS_PROJECT_BASE`` and the ``project_root_folder`` config field optional. See
``slipbox/4_analysis/2026-06-26_non_invasive_project_registration_path_resolution_analysis.md``.
"""

import tempfile
from pathlib import Path

import pytest

from cursus.core.utils.hybrid_path_resolution import (
    HybridPathResolver,
    set_project_root,
    get_project_root,
)


@pytest.fixture
def project_tree():
    """A temp project: <proj>/dockers/scripts/ with a script file."""
    with tempfile.TemporaryDirectory() as d:
        proj = Path(d) / "munged_address_pytorch"
        scripts = proj / "dockers" / "scripts"
        scripts.mkdir(parents=True)
        (scripts / "train.py").write_text("# script")
        yield proj


@pytest.fixture(autouse=True)
def _clear_pushed_root():
    """Ensure the process-level pushed root never leaks across tests."""
    set_project_root(None)
    yield
    set_project_root(None)


class TestCallerHookResolver:
    def test_set_and_get_project_root(self, project_tree):
        set_project_root(str(project_tree))
        assert get_project_root() == str(project_tree)
        set_project_root(None)
        assert get_project_root() is None

    def test_resolves_relative_against_pushed_root(self, project_tree):
        set_project_root(str(project_tree))
        r = HybridPathResolver()
        # NOTE the empty project_root_folder — the hook does not need it.
        resolved = r.resolve_path("", "dockers/scripts")
        assert resolved == str(project_tree / "dockers" / "scripts")

    def test_resolves_file_relative_against_pushed_root(self, project_tree):
        set_project_root(str(project_tree))
        r = HybridPathResolver()
        resolved = r.resolve_path("", "dockers/scripts/train.py")
        assert resolved == str(project_tree / "dockers" / "scripts" / "train.py")

    def test_hook_ignores_wrong_project_root_folder(self, project_tree):
        """Strategy 0 anchors on the pushed root and ignores project_root_folder."""
        set_project_root(str(project_tree))
        r = HybridPathResolver()
        resolved = r.resolve_path("a_completely_wrong_name", "dockers/scripts")
        assert resolved == str(project_tree / "dockers" / "scripts")

    def test_no_pushed_root_does_not_use_strategy_0(self, project_tree):
        """With nothing pushed, Strategy 0 is inert (returns None for that strategy)."""
        set_project_root(None)
        r = HybridPathResolver()
        assert r._pushed_project_root_discovery("dockers/scripts") is None

    def test_pushed_root_nonexistent_target_returns_none(self, project_tree):
        set_project_root(str(project_tree))
        r = HybridPathResolver()
        assert r.resolve_path("", "dockers/does_not_exist") is None

    def test_pushed_root_path_that_does_not_exist(self):
        set_project_root("/no/such/dir/anywhere")
        r = HybridPathResolver()
        assert r._pushed_project_root_discovery("dockers") is None

    def test_strategy_0_takes_precedence_over_cwd(self, project_tree, monkeypatch):
        """Even when cwd would also resolve, the pushed root wins (it is tried first)."""
        # cwd = project_tree would let Strategy 2 resolve too; pushed root must win identically.
        monkeypatch.chdir(project_tree)
        set_project_root(str(project_tree))
        r = HybridPathResolver()
        resolved = r.resolve_path("", "dockers/scripts")
        assert resolved == str(project_tree / "dockers" / "scripts")
