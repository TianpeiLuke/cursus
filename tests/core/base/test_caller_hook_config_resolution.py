"""
End-to-end tests for the caller-hook anchor across EVERY config resolution path.

The caller hook (a project root pushed via ``set_project_root``) must flow through every
place a config resolves its docker ``source_dir``/``processing_source_dir``:

- ``BasePipelineConfig.effective_source_dir`` / ``resolved_source_dir`` / ``resolve_hybrid_path``
- ``ProcessingStepConfigBase.effective_source_dir`` (processing_source_dir, then source_dir)
- ``ProcessingStepConfigBase.resolved_processing_source_dir`` / ``get_resolved_script_path``

It must also make ``project_root_folder`` optional (resolve even when it's empty), take
precedence over ``CURSUS_PROJECT_BASE``, and not change behavior when nothing is pushed.
"""

import os
import tempfile
from pathlib import Path

import pytest

from cursus.core.base.config_base import BasePipelineConfig
from cursus.steps.configs.config_processing_step_base import ProcessingStepConfigBase
from cursus.core.utils.hybrid_path_resolution import set_project_root


_REQ = dict(
    author="t",
    bucket="b",
    role="r",
    region="NA",
    service_name="s",
    pipeline_version="1.0.0",
)


@pytest.fixture
def project_tree():
    """Temp project: <proj>/dockers/scripts/train.py."""
    with tempfile.TemporaryDirectory() as d:
        proj = Path(d) / "my_pipeline_project"
        scripts = proj / "dockers" / "scripts"
        scripts.mkdir(parents=True)
        (scripts / "train.py").write_text("# script")
        yield proj


@pytest.fixture(autouse=True)
def _isolate_resolution_state():
    """No pushed root and no CURSUS_PROJECT_BASE leak across tests."""
    set_project_root(None)
    saved = os.environ.pop("CURSUS_PROJECT_BASE", None)
    yield
    set_project_root(None)
    if saved is not None:
        os.environ["CURSUS_PROJECT_BASE"] = saved
    else:
        os.environ.pop("CURSUS_PROJECT_BASE", None)


# --- BasePipelineConfig paths --------------------------------------------------------


class TestBaseConfigResolution:
    def test_resolve_hybrid_path_uses_pushed_root(self, project_tree):
        set_project_root(str(project_tree))
        cfg = BasePipelineConfig(
            **_REQ, project_root_folder="proj", source_dir="dockers/scripts"
        )
        assert cfg.resolve_hybrid_path("dockers/scripts") == str(
            project_tree / "dockers" / "scripts"
        )

    def test_effective_source_dir_uses_pushed_root(self, project_tree):
        set_project_root(str(project_tree))
        cfg = BasePipelineConfig(
            **_REQ, project_root_folder="proj", source_dir="dockers/scripts"
        )
        assert cfg.effective_source_dir == str(project_tree / "dockers" / "scripts")

    def test_resolved_source_dir_uses_pushed_root(self, project_tree):
        set_project_root(str(project_tree))
        cfg = BasePipelineConfig(
            **_REQ, project_root_folder="proj", source_dir="dockers/scripts"
        )
        assert cfg.resolved_source_dir == str(project_tree / "dockers" / "scripts")

    def test_hook_makes_project_root_folder_optional(self, project_tree):
        """The headline win: resolution works with an EMPTY project_root_folder."""
        set_project_root(str(project_tree))
        cfg = BasePipelineConfig(
            **_REQ, project_root_folder="", source_dir="dockers/scripts"
        )
        assert cfg.effective_source_dir == str(project_tree / "dockers" / "scripts")

    def test_hook_ignores_wrong_project_root_folder(self, project_tree):
        set_project_root(str(project_tree))
        cfg = BasePipelineConfig(
            **_REQ, project_root_folder="totally_wrong_name", source_dir="dockers/scripts"
        )
        assert cfg.effective_source_dir == str(project_tree / "dockers" / "scripts")

    def test_no_push_and_no_prf_returns_relative_fallback(self, project_tree, monkeypatch):
        """No pushed root + empty project_root_folder + cwd elsewhere -> legacy passthrough."""
        set_project_root(None)
        monkeypatch.chdir(tempfile.gettempdir())
        cfg = BasePipelineConfig(
            **_REQ, project_root_folder="", source_dir="dockers/scripts"
        )
        # effective_source_dir falls back to the raw relative source_dir (unresolved).
        assert cfg.effective_source_dir == "dockers/scripts"


# --- ProcessingStepConfigBase paths --------------------------------------------------


class TestProcessingConfigResolution:
    def test_processing_effective_source_dir_uses_pushed_root(self, project_tree):
        set_project_root(str(project_tree))
        cfg = ProcessingStepConfigBase(
            **_REQ,
            project_root_folder="proj",
            processing_source_dir="dockers/scripts",
            processing_entry_point="train.py",
        )
        assert cfg.effective_source_dir == str(project_tree / "dockers" / "scripts")

    def test_processing_falls_back_to_source_dir(self, project_tree):
        """When processing_source_dir is absent, source_dir resolves via the hook."""
        set_project_root(str(project_tree))
        cfg = ProcessingStepConfigBase(
            **_REQ,
            project_root_folder="proj",
            source_dir="dockers/scripts",
            processing_entry_point="train.py",
        )
        assert cfg.effective_source_dir == str(project_tree / "dockers" / "scripts")

    def test_resolved_processing_source_dir_uses_pushed_root(self, project_tree):
        set_project_root(str(project_tree))
        cfg = ProcessingStepConfigBase(
            **_REQ,
            project_root_folder="proj",
            processing_source_dir="dockers/scripts",
            processing_entry_point="train.py",
        )
        assert cfg.resolved_processing_source_dir == str(
            project_tree / "dockers" / "scripts"
        )

    def test_get_resolved_script_path_uses_pushed_root(self, project_tree):
        set_project_root(str(project_tree))
        cfg = ProcessingStepConfigBase(
            **_REQ,
            project_root_folder="proj",
            processing_source_dir="dockers/scripts",
            processing_entry_point="train.py",
        )
        assert cfg.get_resolved_script_path() == str(
            project_tree / "dockers" / "scripts" / "train.py"
        )

    def test_processing_hook_optional_prf(self, project_tree):
        set_project_root(str(project_tree))
        cfg = ProcessingStepConfigBase(
            **_REQ,
            project_root_folder="",
            processing_source_dir="dockers/scripts",
            processing_entry_point="train.py",
        )
        assert cfg.effective_source_dir == str(project_tree / "dockers" / "scripts")


# --- Precedence over CURSUS_PROJECT_BASE ---------------------------------------------


class TestPrecedence:
    def test_pushed_root_wins_over_cursus_project_base(self, project_tree):
        """Caller hook (Strategy 0) takes precedence over the CURSUS_PROJECT_BASE env var."""
        # Set CURSUS_PROJECT_BASE to a DIFFERENT valid tree that also contains the path.
        with tempfile.TemporaryDirectory() as other:
            other_proj = Path(other) / "proj"
            (other_proj / "dockers" / "scripts").mkdir(parents=True)
            os.environ["CURSUS_PROJECT_BASE"] = str(other)

            set_project_root(str(project_tree))  # the hook points at project_tree
            cfg = BasePipelineConfig(
                **_REQ, project_root_folder="proj", source_dir="dockers/scripts"
            )
            # Must resolve against the PUSHED root, not CURSUS_PROJECT_BASE/proj.
            assert cfg.effective_source_dir == str(
                project_tree / "dockers" / "scripts"
            )

    def test_falls_through_to_cursus_project_base_when_no_push(self, project_tree):
        """With nothing pushed, CURSUS_PROJECT_BASE (Strategy 0b) still resolves."""
        with tempfile.TemporaryDirectory() as base:
            proj = Path(base) / "proj"
            (proj / "dockers" / "scripts").mkdir(parents=True)
            os.environ["CURSUS_PROJECT_BASE"] = str(base)
            set_project_root(None)
            cfg = BasePipelineConfig(
                **_REQ, project_root_folder="proj", source_dir="dockers/scripts"
            )
            assert cfg.effective_source_dir == str(proj / "dockers" / "scripts")
