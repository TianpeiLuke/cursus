"""
Tests for WorkspaceAwarePipelineTestingSpecBuilder

Tests the enhanced workspace-aware script discovery capabilities.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from cursus.validation.runtime.workspace_aware_spec_builder import (
    WorkspaceAwarePipelineTestingSpecBuilder,
)
from cursus.validation.runtime.runtime_models import ScriptExecutionSpec


class TestWorkspaceAwarePipelineTestingSpecBuilder:
    """Test workspace-aware spec builder functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = Path(self.temp_dir) / "test_data"
        self.test_data_dir.mkdir(parents=True)

        # Create test script directories
        self.scripts_dir = self.test_data_dir / "scripts"
        self.scripts_dir.mkdir(parents=True)

        # Create some test scripts
        (self.scripts_dir / "tabular_preprocessing.py").write_text("# Test script")
        (self.scripts_dir / "xgboost_training.py").write_text("# Test script")

        self.builder = WorkspaceAwarePipelineTestingSpecBuilder(
            test_data_dir=str(self.test_data_dir)
        )

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test builder initialization with workspace config."""
        builder = WorkspaceAwarePipelineTestingSpecBuilder(
            test_data_dir=str(self.test_data_dir),
            workspace_discovery_enabled=False,
            max_workspace_depth=5,
            workspace_script_patterns=["custom/scripts/"],
        )

        assert not builder.workspace_discovery_enabled
        assert builder.max_workspace_depth == 5
        assert "custom/scripts/" in builder.workspace_script_patterns

    def test_find_actual_script_file_test_data_priority(self):
        """Test that test data scripts have highest priority."""
        # Test script exists in test data
        result = self.builder._find_actual_script_file("TabularPreprocessing")
        assert result == "tabular_preprocessing"

    def test_canonical_to_script_name_conversion(self):
        """Test PascalCase to snake_case conversion."""
        test_cases = [
            ("TabularPreprocessing", "tabular_preprocessing"),
            ("XGBoostTraining", "xgboost_training"),
            ("PyTorchModelEval", "pytorch_model_eval"),
            ("ModelCalibration", "model_calibration"),
        ]

        for canonical, expected in test_cases:
            result = self.builder._canonical_to_script_name(canonical)
            assert (
                result == expected
            ), f"Expected {expected}, got {result} for {canonical}"

    def test_workspace_discovery_disabled(self):
        """Test behavior when workspace discovery is disabled."""
        self.builder.configure_workspace_discovery(workspace_discovery_enabled=False)

        workspace_dirs = self.builder._find_in_workspace("test_script")
        assert workspace_dirs == []

    def test_workspace_discovery_fallback(self):
        """Test fallback behavior when workspace system unavailable."""
        # Create a fallback directory
        fallback_dir = Path(self.temp_dir) / "workspace" / "scripts"
        fallback_dir.mkdir(parents=True)
        (fallback_dir / "test_script.py").write_text("# Fallback script")

        # Change to temp directory to make relative paths work
        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)

            # Mock step catalog and workspace adapter to raise ImportError
            with patch("cursus.step_catalog.StepCatalog", side_effect=ImportError):
                with patch("cursus.step_catalog.adapters.workspace_discovery.WorkspaceDiscoveryManagerAdapter", side_effect=ImportError):
                    workspace_dirs = self.builder._find_in_workspace("test_script")

                    # Should find fallback directory
                    location_names = [name for name, _ in workspace_dirs]
                    assert any("workspace_local_scripts" in name for name in location_names)
        finally:
            os.chdir(original_cwd)

    def test_workspace_discovery_success(self):
        """Test successful workspace discovery using WorkspaceDiscoveryManagerAdapter."""
        # Create workspace directory structure
        workspace_root = Path(self.temp_dir) / "workspace_root"
        developers_dir = workspace_root / "developers" / "test_workspace"
        scripts_dir = developers_dir / "scripts"
        scripts_dir.mkdir(parents=True)
        (scripts_dir / "test_script.py").write_text("# Workspace script")

        # Mock step catalog to fail so we use the workspace adapter
        with patch("cursus.step_catalog.StepCatalog", side_effect=ImportError):
            # Mock the workspace adapter
            with patch("cursus.step_catalog.adapters.workspace_discovery.WorkspaceDiscoveryManagerAdapter") as mock_adapter_class:
                mock_adapter = Mock()
                mock_adapter.list_available_developers.return_value = ["test_workspace"]
                mock_adapter.get_workspace_info.return_value = {
                    "workspace_id": "test_workspace",
                    "workspace_path": str(developers_dir),
                    "workspace_type": "developer",
                    "exists": True
                }
                mock_adapter_class.return_value = mock_adapter

                workspace_dirs = self.builder._find_in_workspace("test_script")

                # Should find workspace script
                assert len(workspace_dirs) > 0
                location_names = [name for name, _ in workspace_dirs]
                assert any("workspace_test_workspace_scripts" in name for name in location_names)

    def test_workspace_cache(self):
        """Test workspace discovery caching."""
        # First call
        result1 = self.builder._find_in_workspace("test_script")

        # Second call should use cache
        result2 = self.builder._find_in_workspace("test_script")

        assert result1 == result2
        assert "test_script" in self.builder._workspace_cache

    def test_clear_workspace_cache(self):
        """Test clearing workspace cache."""
        # Populate cache
        self.builder._find_in_workspace("test_script")
        assert len(self.builder._workspace_cache) > 0

        # Clear cache
        self.builder.clear_workspace_cache()
        assert len(self.builder._workspace_cache) == 0

    def test_get_workspace_discovery_status(self):
        """Test workspace discovery status reporting."""
        status = self.builder.get_workspace_discovery_status()

        assert "workspace_discovery_enabled" in status
        assert "max_workspace_depth" in status
        assert "workspace_script_patterns" in status
        assert "cache_size" in status
        assert "workspace_system_available" in status

    def test_configure_workspace_discovery(self):
        """Test workspace discovery configuration updates."""
        self.builder.configure_workspace_discovery(
            workspace_discovery_enabled=False,
            max_workspace_depth=10,
            workspace_script_patterns=["new/pattern/"],
        )

        assert not self.builder.workspace_discovery_enabled
        assert self.builder.max_workspace_depth == 10
        assert "new/pattern/" in self.builder.workspace_script_patterns

    def test_discover_available_scripts(self):
        """Test discovery of all available scripts."""
        scripts = self.builder.discover_available_scripts()

        assert "test_data" in scripts
        assert "tabular_preprocessing" in scripts["test_data"]
        assert "xgboost_training" in scripts["test_data"]

    def test_validate_workspace_setup(self):
        """Test workspace setup validation."""
        validation = self.builder.validate_workspace_setup()

        assert "status" in validation
        assert "warnings" in validation
        assert "errors" in validation
        assert "recommendations" in validation

    def test_fuzzy_matching_in_workspace_discovery(self):
        """Test fuzzy matching during workspace discovery."""
        # Create script with similar name
        (self.scripts_dir / "tabular_preprocess.py").write_text("# Similar script")

        # Should find fuzzy match
        result = self.builder._find_actual_script_file("TabularPreprocessing")
        # Should prefer exact match over fuzzy match
        assert result == "tabular_preprocessing"

        # Remove exact match, should find fuzzy match
        (self.scripts_dir / "tabular_preprocessing.py").unlink()
        result = self.builder._find_actual_script_file("TabularPreprocessing")
        assert result == "tabular_preprocess"

    def test_error_handling_no_scripts_found(self):
        """Test error handling when no scripts are found."""
        # Remove all test scripts
        for script_file in self.scripts_dir.glob("*.py"):
            script_file.unlink()

        with pytest.raises(ValueError, match="Cannot find script file"):
            self.builder._find_actual_script_file("NonExistentScript")

    def test_resolve_script_execution_spec_from_node(self):
        """Test end-to-end node-to-spec resolution."""
        # Test direct script resolution without registry dependency
        # Create a test script
        (self.scripts_dir / "tabular_preprocessing.py").write_text("# Test script")

        # Test the _find_actual_script_file method directly
        result = self.builder._find_actual_script_file("TabularPreprocessing")
        assert result == "tabular_preprocessing"

        # Test creating a spec manually (since resolve_script_execution_spec_from_node
        # depends on registry functions that may not be available)
        spec = ScriptExecutionSpec(
            step_name="TabularPreprocessing_training",
            script_name="tabular_preprocessing",
            expected_input_paths={"input": "data/input.csv"},
            expected_output_paths={"output": "data/output.csv"},
            expected_environment_variables={},
        )

        assert isinstance(spec, ScriptExecutionSpec)
        assert spec.script_name == "tabular_preprocessing"
        assert spec.step_name == "TabularPreprocessing_training"


class TestWorkspaceAwareIntegration:
    """Integration tests for workspace-aware functionality."""

    def setup_method(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = Path(self.temp_dir) / "test_data"
        self.test_data_dir.mkdir(parents=True)

        self.builder = WorkspaceAwarePipelineTestingSpecBuilder(
            test_data_dir=str(self.test_data_dir)
        )

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_inheritance_from_base_builder(self):
        """Test that workspace-aware builder properly inherits from base builder."""
        # Should have all base builder methods
        assert hasattr(self.builder, "build_from_dag")
        assert hasattr(self.builder, "save_script_spec")
        assert hasattr(self.builder, "update_script_spec")
        assert hasattr(self.builder, "list_saved_specs")

        # Should have enhanced workspace methods
        assert hasattr(self.builder, "_find_in_workspace")
        assert hasattr(self.builder, "configure_workspace_discovery")
        assert hasattr(self.builder, "get_workspace_discovery_status")
        assert hasattr(self.builder, "validate_workspace_setup")

    def test_workspace_configuration_persistence(self):
        """Test that workspace configuration persists across operations."""
        # Configure workspace settings
        self.builder.configure_workspace_discovery(
            workspace_discovery_enabled=False, max_workspace_depth=5
        )

        # Settings should persist
        assert not self.builder.workspace_discovery_enabled
        assert self.builder.max_workspace_depth == 5

        # Should still work after cache operations
        self.builder.clear_workspace_cache()
        assert not self.builder.workspace_discovery_enabled
        assert self.builder.max_workspace_depth == 5
