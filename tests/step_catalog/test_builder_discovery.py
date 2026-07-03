"""
Test BuilderAutoDiscovery component.

This module tests the BuilderAutoDiscovery class that provides:
- Builder class discovery from package and workspace directories
- Registry-guided step name mapping
- AST-based safe class detection
- Deployment-agnostic file loading
"""

import pytest
from unittest.mock import Mock, patch

from cursus.step_catalog.builder_discovery import BuilderAutoDiscovery


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
        assert discovery._synthesized_builders == {}
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
        """Test discovering all builder classes (synthesis-only, Phase E).

        File-based discovery was removed: builders are SYNTHESIZED by walking the registry
        interface. Drive the real "XGBoostTraining" registry step through synthesis and assert
        its synthesized class appears in the discovered map with the expected __name__.
        """
        builder_discovery._run_discovery()

        builders = builder_discovery.discover_builder_classes()

        assert "XGBoostTraining" in builders
        assert builders["XGBoostTraining"].__name__ == "XGBoostTrainingStepBuilder"
        # discover == synthesize for each step: identity matches load_builder_class.
        assert builders["XGBoostTraining"] is builder_discovery.load_builder_class(
            "XGBoostTraining"
        )

    def test_discover_builder_classes_with_project_id(self, builder_discovery):
        """Test discovering builder classes with a project ID (synthesis-only, Phase E).

        The project_id parameter is still accepted for signature compatibility, but discovery
        is now registry-wide synthesis (there is no per-workspace file-based filtering), so a
        routable registry step is synthesized regardless of project_id.
        """
        builder_discovery._run_discovery()

        builders = builder_discovery.discover_builder_classes(project_id="workspace1")

        assert "XGBoostTraining" in builders
        assert builders["XGBoostTraining"].__name__ == "XGBoostTrainingStepBuilder"

    def test_synthesize_builder_fabricates_fileless_routable_step(self, builder_discovery):
        """FZ 31e1d3g3 Phase A: a routable, interface-having step with NO physical file is
        synthesized as a real TemplateStepBuilder subclass keyed on STEP_NAME (the deletion
        mechanism for the 45 shells)."""
        from cursus.core.base.builder_templates import TemplateStepBuilder

        builder_discovery._run_discovery()
        # Phase E: there is no file-descriptor map anymore — every routable step is synthesized.
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

    # DELETED (Phase E): test_extract_step_name_from_builder_file, test_load_class_from_file,
    # and the base-name discovery gate tests (_inherits helper, test_inherits_legacy_step_builder_base,
    # test_inherits_template_step_builder, test_inherits_qualified_base_names,
    # test_does_not_inherit_unrelated_base, test_base_name_set_is_strict_superset) — these all
    # tested the removed file-based/AST builder-discovery path (_extract_step_name_from_builder_file,
    # _load_class_from_file, _inherits_from_step_builder_base, STEP_BUILDER_BASE_NAMES), which no
    # longer exists now that builders are synthesized from the registry interface.

    def test_get_builder_info_cached(self, builder_discovery):
        """Test getting builder info reads from the materialized-class cache (Phase E).

        File paths are no longer tracked (file-based discovery removed), so get_builder_info
        reports the class name/module from the loaded/cached class only.
        """
        # Create a mock class and cache it
        mock_class = Mock()
        mock_class.__name__ = "TestStepBuilder"
        builder_discovery._builder_cache["test_step"] = mock_class

        info = builder_discovery.get_builder_info("test_step")

        assert info is not None
        assert info["builder_class"] == "TestStepBuilder"

    def test_get_builder_info_not_found(self, builder_discovery):
        """Test getting builder info when not found."""
        info = builder_discovery.get_builder_info("nonexistent_step")
        assert info is None

    def test_load_builder_class_success(self, builder_discovery):
        """Test successfully loading builder class via synthesis (Phase E).

        Builders are synthesized from the registry interface — drive a real registry step
        ("XGBoostTraining") through load_builder_class and assert the synthesized class name,
        plus that it is cached after loading.
        """
        builder_discovery._run_discovery()

        result = builder_discovery.load_builder_class("XGBoostTraining")

        assert result is not None
        assert result.__name__ == "XGBoostTrainingStepBuilder"
        # Should be cached after loading (identity-stable per step).
        assert builder_discovery._builder_cache["XGBoostTraining"] is result

    def test_load_builder_class_not_found(self, builder_discovery):
        """Test loading builder class when not found."""
        with patch.object(builder_discovery, "get_builder_info") as mock_get_info:
            mock_get_info.return_value = None

            result = builder_discovery.load_builder_class("nonexistent_step")
            assert result is None

    # DELETED (Phase E): test_load_builder_class_load_error, test_load_class_from_file_success,
    # test_load_class_from_file_not_found, test_load_class_from_file_import_error — these tested the
    # removed file-based load path (_load_class_from_file and the get_builder_info/_load_class_from_file
    # error branch of load_builder_class). load_builder_class now synthesizes directly from the registry
    # interface, so there is no file-load step left to error on.

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
        """Test complete end-to-end discovery and loading via synthesis (Phase E).

        Builders are synthesized by walking the registry interface — no file-based seeding.
        Run discovery, then load a real registry step and confirm get_builder_info reports the
        synthesized class name.
        """
        # package_root is irrelevant to synthesis (registry drives it); use a bare temp dir.
        package_root = tmp_path / "cursus"
        package_root.mkdir()
        discovery = BuilderAutoDiscovery(package_root, [])

        discovery._run_discovery()

        # Load then introspect a real, routable registry step.
        cls = discovery.load_builder_class("XGBoostTraining")
        assert cls is not None
        assert cls.__name__ == "XGBoostTrainingStepBuilder"

        builder_info = discovery.get_builder_info("XGBoostTraining")
        assert builder_info is not None
        assert builder_info["builder_class"] == "XGBoostTrainingStepBuilder"

    # DELETED (Phase E): test_workspace_priority_over_package, test_multiple_workspaces — these
    # tested workspace-vs-package FILE-based builder discovery (_workspace_builders / _package_builders
    # / _builder_paths), a capability that was removed. Builders are now synthesized from the registry
    # interface, which has no workspace/package precedence dimension.

    def test_list_available_builders(self, tmp_path):
        """Test listing all available builders (synthesis-only, Phase E).

        list_available_builders now reflects the synthesized (discovered) set, so a real
        routable registry step must appear.
        """
        package_root = tmp_path / "cursus"
        package_root.mkdir()
        discovery = BuilderAutoDiscovery(package_root, [])

        available_builders = discovery.list_available_builders()

        assert "XGBoostTraining" in available_builders

    def test_get_discovery_stats(self, tmp_path):
        """Test getting discovery statistics (synthesis-only, Phase E)."""
        package_root = tmp_path / "cursus"
        package_root.mkdir()
        discovery = BuilderAutoDiscovery(package_root, [])

        discovery.discover_builder_classes()

        stats = discovery.get_discovery_stats()

        # Synthesized builders count reflects the registry-walk synthesis result.
        assert stats["synthesized_builders"] >= 1
        assert stats["total_builders"] == stats["synthesized_builders"]
        assert stats["discovery_complete"] == True
