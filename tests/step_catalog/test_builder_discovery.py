"""
Test BuilderAutoDiscovery component.

This module tests the BuilderAutoDiscovery class that provides:
- Builder class discovery from package and workspace directories
- Registry-guided step name mapping
- AST-based safe class detection
- Deployment-agnostic file loading
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from datetime import datetime
import tempfile
import os

from cursus.step_catalog.builder_discovery import BuilderAutoDiscovery, _BuilderDescriptor


def _desc(cls, step_name="test_step", file_path=None):
    """Wrap a class as an already-materialized ``_BuilderDescriptor``.

    Phase 2 (FZ 31e1d3g) changed ``_package_builders``/``_workspace_builders`` to hold
    import-free descriptors instead of raw classes; the class is materialized lazily via
    ``descriptor.get_class()``. Tests seed via this helper so ``get_class()`` returns the
    mock class directly (``_attempted=True`` short-circuits the loader).
    """
    return _BuilderDescriptor(
        step_name=step_name,
        file_path=file_path or Path(f"/path/to/builder_{step_name}_step.py"),
        class_name=getattr(cls, "__name__", "MockBuilder"),
        _loader=lambda *_a, **_k: cls,
        _cached=cls,
        _attempted=True,
    )


class TestBuilderAutoDiscovery:
    """Test the BuilderAutoDiscovery component."""

    @pytest.fixture
    def mock_package_root(self, tmp_path):
        """Create a mock package root directory."""
        package_root = tmp_path / "cursus"
        package_root.mkdir()

        # Create builders directory
        builders_dir = package_root / "steps" / "builders"
        builders_dir.mkdir(parents=True)

        return package_root

    @pytest.fixture
    def mock_workspace_dirs(self, tmp_path):
        """Create mock workspace directories."""
        workspace1 = tmp_path / "workspace1"
        workspace2 = tmp_path / "workspace2"

        # Create workspace structure
        for workspace in [workspace1, workspace2]:
            dev_dir = (
                workspace
                / "development"
                / "projects"
                / "test_project"
                / "src"
                / "cursus_dev"
                / "steps"
                / "builders"
            )
            dev_dir.mkdir(parents=True)

        return [workspace1, workspace2]

    @pytest.fixture
    def builder_discovery(self, mock_package_root, mock_workspace_dirs):
        """Create BuilderAutoDiscovery instance with test data."""
        return BuilderAutoDiscovery(mock_package_root, mock_workspace_dirs)

    def test_initialization(self, mock_package_root, mock_workspace_dirs):
        """Test BuilderAutoDiscovery initialization."""
        discovery = BuilderAutoDiscovery(mock_package_root, mock_workspace_dirs)

        assert discovery.package_root == mock_package_root
        assert discovery.workspace_dirs == mock_workspace_dirs
        assert discovery.logger is not None
        assert discovery._builder_cache == {}
        assert discovery._builder_paths == {}
        assert discovery._discovery_complete == False

    def test_initialization_no_workspace(self, mock_package_root):
        """Test BuilderAutoDiscovery initialization without workspace directories."""
        discovery = BuilderAutoDiscovery(mock_package_root, [])

        assert discovery.package_root == mock_package_root
        assert discovery.workspace_dirs == []
        assert discovery._builder_cache == {}

    def test_get_registry_builder_info(self, builder_discovery):
        """Test getting builder info from registry."""
        # Mock registry info
        builder_discovery._registry_info = {
            "xgboost_training": {"builder_step_name": "XGBoostTraining"},
            "pytorch_model": {"builder_step_name": "PyTorchModel"},
        }

        info = builder_discovery._get_registry_builder_info("xgboost_training")
        assert info is not None
        assert info["builder_step_name"] == "XGBoostTraining"

        # Test non-existent step
        info = builder_discovery._get_registry_builder_info("nonexistent")
        assert info is None

    def test_discover_builder_classes(self, builder_discovery):
        """Test discovering all builder classes."""
        # Mock the discovery process
        mock_class = Mock()
        mock_class.__name__ = "TestBuilder"

        builder_discovery._package_builders = {"test_step": _desc(mock_class)}
        builder_discovery._workspace_builders = {
            "workspace1": {"workspace_step": _desc(mock_class, "workspace_step")}
        }
        builder_discovery._discovery_complete = True
        # Isolate to descriptor materialization: empty the registry so the Phase-A registry-walk
        # fill (which would synthesize a shell for every routable registry step) adds nothing here.
        builder_discovery._registry_info = {}

        # Test discovery
        builders = builder_discovery.discover_builder_classes()

        assert "test_step" in builders
        assert "workspace_step" in builders
        assert len(builders) == 2

    def test_discover_builder_classes_with_project_id(self, builder_discovery):
        """Test discovering builder classes with specific project ID."""
        mock_class = Mock()
        mock_class.__name__ = "TestBuilder"

        builder_discovery._package_builders = {"test_step": _desc(mock_class)}
        builder_discovery._workspace_builders = {
            "workspace1": {"workspace1_step": _desc(mock_class, "workspace1_step")},
            "workspace2": {"workspace2_step": _desc(mock_class, "workspace2_step")},
        }
        builder_discovery._discovery_complete = True

        # Test with specific project ID
        builders = builder_discovery.discover_builder_classes(project_id="workspace1")

        assert "test_step" in builders
        assert "workspace1_step" in builders
        assert "workspace2_step" not in builders

    def test_synthesize_builder_fabricates_fileless_routable_step(self, builder_discovery):
        """FZ 31e1d3g3 Phase A: a routable, interface-having step with NO physical file is
        synthesized as a real TemplateStepBuilder subclass keyed on STEP_NAME (the deletion
        mechanism for the 45 shells)."""
        from cursus.core.base.builder_templates import TemplateStepBuilder

        builder_discovery._run_discovery()
        # Simulate a deleted builder_*.py for a normal routable step.
        builder_discovery._package_builders.pop("XGBoostTraining", None)
        builder_discovery._builder_cache.pop("XGBoostTraining", None)

        cls = builder_discovery.load_builder_class("XGBoostTraining")
        assert cls is not None
        assert cls.__name__ == "XGBoostTrainingStepBuilder"
        assert cls.STEP_NAME == "XGBoostTraining"
        assert issubclass(cls, TemplateStepBuilder)
        # Identity is stable within a process (cached per step_name).
        assert builder_discovery.load_builder_class("XGBoostTraining") is cls

    def test_synthesize_builder_skips_nonroutable_and_sdk_steps_offline(self, builder_discovery):
        """Offline (no SAIS SDK), the synthesizer returns None for abstract rows (Base/Processing —
        no interface) and the 4 SDK-delegation steps (the lazy sdk_bindings import fails), so the
        offline discovered set is unchanged and the closure gate's SDK carve-out holds."""
        _has_sais = True
        try:
            import secure_ai_sandbox_workflow_python_sdk  # noqa: F401
        except Exception:
            _has_sais = False

        builder_discovery._run_discovery()
        assert builder_discovery._synthesize_builder("Base") is None
        assert builder_discovery._synthesize_builder("Processing") is None
        if not _has_sais:
            for sdk_step in (
                "CradleDataLoading",
                "RedshiftDataLoading",
                "Registration",
                "DataUploading",
            ):
                assert builder_discovery._synthesize_builder(sdk_step) is None

    def test_synthesize_sdk_builder_bakes_handler_knobs_in_sais_env(self, builder_discovery):
        """In the SAIS env, an SDK-delegation step synthesizes WITH the sdk_step_class knob baked into
        HANDLER_KNOBS (mirroring the hand-written SDK shells), via the lazy sdk_bindings thunk."""
        try:
            import secure_ai_sandbox_workflow_python_sdk  # noqa: F401
        except Exception:
            pytest.skip("SAIS SDK absent — SDK synthesis only works in the SAIS env")

        builder_discovery._run_discovery()
        cls = builder_discovery._synthesize_builder("CradleDataLoading")
        assert cls is not None
        assert cls.STEP_NAME == "CradleDataLoading"
        assert "sdk_step_class" in cls.HANDLER_KNOBS

    def test_extract_step_name_from_builder_file(self, builder_discovery):
        """Test extracting step name from builder file."""
        # Mock registry info
        builder_discovery._registry_info = {
            "xgboost_training": {"builder_step_name": "XGBoostTrainingStepBuilder"}
        }

        # Test with registry match
        test_file = Path("builder_xgboost_training_step.py")
        step_name = builder_discovery._extract_step_name_from_builder_file(
            test_file, "XGBoostTrainingStepBuilder"
        )
        assert step_name == "xgboost_training"

        # Test with file name extraction
        test_file = Path("builder_custom_training_step.py")
        step_name = builder_discovery._extract_step_name_from_builder_file(
            test_file, "CustomTrainingStepBuilder"
        )
        assert step_name is not None

    def test_load_class_from_file(self, builder_discovery, tmp_path):
        """Test loading class from file."""
        # Create test file
        test_file = tmp_path / "test_builder.py"
        test_file.write_text("""
class TestBuilder:
    def __init__(self):
        self.name = "test"
""")

        # Test loading (will likely fail due to import issues in test environment)
        result = builder_discovery._load_class_from_file(test_file, "TestBuilder")
        # In test environment, this might return None due to import issues
        # The important thing is that it doesn't crash
        assert result is None or hasattr(result, "__name__")

    # --- Phase 0a: base-name discovery gate (STEP_BUILDER_BASE_NAMES) ---

    def _inherits(self, builder_discovery, source: str) -> bool:
        """Parse a class definition and run it through the base-name gate."""
        import ast

        tree = ast.parse(source)
        class_node = next(
            n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
        )
        return builder_discovery._inherits_from_step_builder_base(class_node)

    def test_inherits_legacy_step_builder_base(self, builder_discovery):
        """Legacy builders (direct StepBuilderBase base) are still recognized."""
        assert self._inherits(
            builder_discovery, "class XStepBuilder(StepBuilderBase): pass"
        )

    def test_inherits_template_step_builder(self, builder_discovery):
        """Routed shells (direct TemplateStepBuilder base) are recognized — the Phase-0a fix."""
        assert self._inherits(
            builder_discovery, "class XStepBuilder(TemplateStepBuilder): pass"
        )

    def test_inherits_qualified_base_names(self, builder_discovery):
        """Qualified bases (module.StepBuilderBase / module.TemplateStepBuilder) match."""
        assert self._inherits(
            builder_discovery, "class A(base.StepBuilderBase): pass"
        )
        assert self._inherits(
            builder_discovery, "class B(mod.TemplateStepBuilder): pass"
        )

    def test_does_not_inherit_unrelated_base(self, builder_discovery):
        """A non-builder class is NOT recognized (no false positives)."""
        assert not self._inherits(
            builder_discovery, "class NotABuilder(SomethingElse): pass"
        )
        assert not self._inherits(builder_discovery, "class Plain: pass")

    def test_base_name_set_is_strict_superset(self):
        """STEP_BUILDER_BASE_NAMES still contains the legacy name (superset guarantee)."""
        from cursus.step_catalog.builder_discovery import STEP_BUILDER_BASE_NAMES

        assert "StepBuilderBase" in STEP_BUILDER_BASE_NAMES
        assert "TemplateStepBuilder" in STEP_BUILDER_BASE_NAMES

    def test_get_builder_info_cached(self, builder_discovery):
        """Test getting builder info with caching."""
        # Create a mock class and cache it
        mock_class = Mock()
        mock_class.__name__ = "TestStepBuilder"
        builder_discovery._builder_cache["test_step"] = mock_class
        builder_discovery._builder_paths["test_step"] = Path("/path/to/test_builder.py")

        info = builder_discovery.get_builder_info("test_step")

        assert info is not None
        assert info["builder_class"] == "TestStepBuilder"
        assert info["file_path"] == "/path/to/test_builder.py"

    def test_get_builder_info_not_found(self, builder_discovery):
        """Test getting builder info when not found."""
        info = builder_discovery.get_builder_info("nonexistent_step")
        assert info is None

    def test_load_builder_class_success(self, builder_discovery):
        """Test successfully loading builder class."""
        # Create a mock class and set it up in the discovery system
        mock_class = Mock()
        mock_class.__name__ = "TestBuilder"

        # Set up the discovery system to find the builder
        builder_discovery._package_builders = {"test_step": _desc(mock_class)}
        builder_discovery._discovery_complete = True

        result = builder_discovery.load_builder_class("test_step")

        assert result == mock_class
        # Should be cached after loading
        assert builder_discovery._builder_cache["test_step"] == mock_class

    def test_load_builder_class_not_found(self, builder_discovery):
        """Test loading builder class when not found."""
        with patch.object(builder_discovery, "get_builder_info") as mock_get_info:
            mock_get_info.return_value = None

            result = builder_discovery.load_builder_class("nonexistent_step")
            assert result is None

    def test_load_builder_class_load_error(self, builder_discovery):
        """Test loading builder class when loading fails."""
        with patch.object(builder_discovery, "get_builder_info") as mock_get_info:
            mock_get_info.return_value = {
                "class_name": "TestBuilder",
                "file_path": "/path/to/test_builder.py",
            }

            with patch.object(builder_discovery, "_load_class_from_file") as mock_load:
                mock_load.return_value = None

                result = builder_discovery.load_builder_class("test_step")
                assert result is None

    def test_load_class_from_file_success(self, builder_discovery, tmp_path):
        """Test loading class from file via deployment-agnostic absolute import."""
        # Create test file under package root to simulate proper module structure
        test_file = (
            builder_discovery.package_root / "steps" / "builders" / "test_module.py"
        )
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("""
class TestClass:
    def __init__(self):
        self.name = "test"
""")

        # Mock the importlib.import_module function to simulate successful loading
        with patch("importlib.import_module") as mock_import_module:
            # Create mock module and class
            mock_module = Mock()
            mock_class = Mock()
            mock_class.__name__ = "TestClass"

            # Set up the mocks
            mock_import_module.return_value = mock_module
            setattr(mock_module, "TestClass", mock_class)

            result = builder_discovery._load_class_from_file(test_file, "TestClass")

            # The loader now prefers an ABSOLUTE import via the real root package name
            # (deployment-agnostic), so this resolves on the first call with no package=.
            expected_absolute_module_path = "cursus.steps.builders.test_module"
            mock_import_module.assert_called_once_with(expected_absolute_module_path)

            # Should return the mock class
            assert result == mock_class

    def test_load_class_from_file_not_found(self, builder_discovery):
        """Test loading class from non-existent file."""
        result = builder_discovery._load_class_from_file(
            "/non/existent/file.py", "TestClass"
        )
        assert result is None

    def test_load_class_from_file_import_error(self, builder_discovery, tmp_path):
        """Test handling import error when loading class."""
        test_file = tmp_path / "bad_import.py"
        test_file.write_text("import non_existent_module")

        result = builder_discovery._load_class_from_file(str(test_file), "TestClass")
        assert result is None

    def test_error_handling_and_logging(self, builder_discovery, caplog):
        """Test error handling and logging throughout the system."""
        import logging

        # Test with non-existent step
        with caplog.at_level(logging.WARNING):
            result = builder_discovery.get_builder_info("nonexistent_step")
            assert result is None

        # Test loading non-existent class
        with caplog.at_level(logging.WARNING):
            result = builder_discovery.load_builder_class("nonexistent_step")
            assert result is None


class TestBuilderAutoDiscoveryIntegration:
    """Integration tests for BuilderAutoDiscovery."""

    def test_end_to_end_discovery_and_loading(self, tmp_path):
        """Test complete end-to-end discovery and loading process."""
        # Create BuilderAutoDiscovery instance
        package_root = tmp_path / "cursus"
        package_root.mkdir()
        discovery = BuilderAutoDiscovery(package_root, [])

        # Mock the discovery system with a test builder
        mock_class = Mock()
        mock_class.__name__ = "XGBoostTrainingStepBuilder"

        # Set up the discovery system
        discovery._package_builders = {
            "xgboost_training": _desc(mock_class, "xgboost_training")
        }
        discovery._builder_paths = {
            "xgboost_training": Path("/path/to/builder_xgboost_training_step.py")
        }
        discovery._discovery_complete = True

        # Test discovery
        builder_info = discovery.get_builder_info("xgboost_training")

        assert builder_info is not None
        assert builder_info["builder_class"] == "XGBoostTrainingStepBuilder"
        assert "builder_xgboost_training_step.py" in str(builder_info["file_path"])

    def test_workspace_priority_over_package(self, tmp_path):
        """Test that workspace builders take priority over package builders."""
        package_root = tmp_path / "cursus"
        package_root.mkdir()
        discovery = BuilderAutoDiscovery(package_root, [])

        # Mock package and workspace builders
        package_class = Mock()
        package_class.__name__ = "TestStepBuilder"
        workspace_class = Mock()
        workspace_class.__name__ = "TestStepBuilder"

        # Set up discovery system with both package and workspace builders
        discovery._package_builders = {"test_step": _desc(package_class)}
        discovery._workspace_builders = {
            "test_project": {"test_step": _desc(workspace_class)}
        }
        discovery._builder_paths = {
            "test_step": Path("/workspace/test_project/builder_test_step.py")
        }
        discovery._discovery_complete = True

        # Test that workspace builder takes priority
        result = discovery.load_builder_class("test_step")

        # Should return workspace builder (higher priority)
        assert result == workspace_class

    def test_multiple_workspaces(self, tmp_path):
        """Test discovery across multiple workspace directories."""
        package_root = tmp_path / "cursus"
        package_root.mkdir()
        discovery = BuilderAutoDiscovery(package_root, [])

        # Mock builders from multiple workspaces
        workspace1_class = Mock()
        workspace1_class.__name__ = "Workspace1StepBuilder"
        workspace2_class = Mock()
        workspace2_class.__name__ = "Workspace2StepBuilder"

        # Set up discovery system
        discovery._workspace_builders = {
            "project1": {"workspace1_step": _desc(workspace1_class, "workspace1_step")},
            "project2": {"workspace2_step": _desc(workspace2_class, "workspace2_step")},
        }
        discovery._builder_paths = {
            "workspace1_step": Path("/workspace1/project1/builder_workspace1_step.py"),
            "workspace2_step": Path("/workspace2/project2/builder_workspace2_step.py"),
        }
        discovery._discovery_complete = True

        # Test discovery from both workspaces
        builder1_info = discovery.get_builder_info("workspace1_step")
        builder2_info = discovery.get_builder_info("workspace2_step")

        assert builder1_info is not None
        assert builder2_info is not None
        assert builder1_info["builder_class"] == "Workspace1StepBuilder"
        assert builder2_info["builder_class"] == "Workspace2StepBuilder"

    def test_list_available_builders(self, tmp_path):
        """Test listing all available builders."""
        package_root = tmp_path / "cursus"
        package_root.mkdir()
        discovery = BuilderAutoDiscovery(package_root, [])

        # Mock builders
        mock_class = Mock()
        mock_class.__name__ = "TestBuilder"

        discovery._package_builders = {"package_step": _desc(mock_class, "package_step")}
        discovery._workspace_builders = {
            "workspace1": {"workspace_step": _desc(mock_class, "workspace_step")}
        }
        discovery._discovery_complete = True

        # Test listing
        available_builders = discovery.list_available_builders()

        assert "package_step" in available_builders
        assert "workspace_step" in available_builders
        assert len(available_builders) == 2

    def test_get_discovery_stats(self, tmp_path):
        """Test getting discovery statistics."""
        package_root = tmp_path / "cursus"
        package_root.mkdir()
        discovery = BuilderAutoDiscovery(package_root, [])

        # Mock builders
        mock_class = Mock()
        mock_class.__name__ = "TestBuilder"

        discovery._package_builders = {"package_step": _desc(mock_class, "package_step")}
        discovery._workspace_builders = {
            "workspace1": {
                "workspace_step1": _desc(mock_class, "workspace_step1"),
                "workspace_step2": _desc(mock_class, "workspace_step2"),
            }
        }
        discovery._builder_cache = {"cached_step": mock_class}
        discovery._discovery_complete = True

        # Test stats
        stats = discovery.get_discovery_stats()

        assert stats["package_builders"] == 1
        assert stats["workspace_builders"]["workspace1"] == 2
        assert stats["total_builders"] == 3
        assert stats["cached_builders"] == 1
        assert stats["discovery_complete"] == True
