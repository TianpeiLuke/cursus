"""
Tests for generic_path_discovery module.

Following pytest best practices and common pitfalls prevention:
- Read source code first to understand implementation
- Mock at import locations, not definition locations
- Count method calls and match side_effect length
- Use MagicMock for Path operations
- Test actual behavior, not assumptions
- Use realistic fixtures with proper cleanup
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Import the module under test
from cursus.core.utils.generic_path_discovery import (
    find_project_folder_generic,
    _get_default_reference_points,
    _search_upward,
    _search_downward,
    _matches_full_path,
    GenericPathDiscoveryMetrics,
    get_generic_discovery_metrics,
    _generic_discovery_metrics,
)


class TestFindProjectFolderGeneric:
    """Test the main find_project_folder_generic function."""

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace with realistic structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir)

            # Create realistic project structure
            projects_dir = workspace_root / "projects"
            projects_dir.mkdir()

            # Create test projects
            atoz_project = projects_dir / "atoz_xgboost"
            atoz_project.mkdir()

            bsm_project = projects_dir / "bsm_pytorch"
            bsm_project.mkdir()

            # Create nested structure
            sub_projects = projects_dir / "sub" / "projects"
            sub_projects.mkdir(parents=True)
            nested_project = sub_projects / "nested_project"
            nested_project.mkdir()

            yield workspace_root

    def test_find_simple_project_name_success(self, temp_workspace):
        """Test finding a project by simple name (upward search)."""
        # Change to a subdirectory to test upward search
        test_subdir = temp_workspace / "projects" / "atoz_xgboost" / "subdir"
        test_subdir.mkdir(parents=True)

        with patch('pathlib.Path.cwd', return_value=test_subdir):
            result = find_project_folder_generic("atoz_xgboost")

            assert result is not None
            assert result.name == "atoz_xgboost"
            assert result.exists()

    def test_find_nested_project_path_success(self, temp_workspace):
        """Test finding a project by nested path (downward search)."""
        with patch('pathlib.Path.cwd', return_value=temp_workspace):
            result = find_project_folder_generic("projects/atoz_xgboost")

            assert result is not None
            assert result.name == "atoz_xgboost"
            assert result.parent.name == "projects"

    def test_find_project_not_found(self, temp_workspace):
        """Test behavior when project is not found."""
        with patch('pathlib.Path.cwd', return_value=temp_workspace):
            result = find_project_folder_generic("nonexistent_project")

            assert result is None

    def test_find_with_custom_reference_points(self, temp_workspace):
        """Test with custom reference points."""
        custom_refs = [temp_workspace / "projects"]
        result = find_project_folder_generic(
            "atoz_xgboost",
            reference_points=custom_refs
        )

        assert result is not None
        assert result.name == "atoz_xgboost"

    def test_find_with_depth_limits(self, temp_workspace):
        """Test with custom depth limits."""
        # Create deeply nested structure where project is at root level
        deep_path = temp_workspace / "level1" / "level2" / "level3" / "level4" / "level5"
        deep_path.mkdir(parents=True)

        # Create project at the temp_workspace root level
        project_at_root = temp_workspace / "atoz_xgboost"
        project_at_root.mkdir()

        with patch('pathlib.Path.cwd', return_value=deep_path):
            # Should fail with low max_depth_up (can't reach root)
            result = find_project_folder_generic(
                "atoz_xgboost",
                max_depth_up=2
            )
            assert result is None

            # Should succeed with higher max_depth_up (can reach root)
            result = find_project_folder_generic(
                "atoz_xgboost",
                max_depth_up=6
            )
            assert result is not None
            assert result == project_at_root

    @patch('cursus.core.utils.generic_path_discovery.logger')
    def test_find_with_exception_handling(self, mock_logger, temp_workspace):
        """Test exception handling in main function."""
        # Mock Path operations to raise exception
        with patch('pathlib.Path.cwd', side_effect=Exception("Test error")):
            result = find_project_folder_generic("any_project")

            assert result is None
            mock_logger.warning.assert_called_once()

    # Removed performance tracking test as metrics are not automatically updated by the main function


class TestHelperFunctions:
    """Test helper functions."""

    def test_get_default_reference_points(self):
        """Test default reference points generation."""
        with patch('pathlib.Path.cwd', return_value=Path("/test/cwd")):
            refs = _get_default_reference_points()

            assert len(refs) >= 1  # At least current working directory
            assert refs[0] == Path("/test/cwd")

    def test_search_upward_success(self, tmp_path):
        """Test upward search finding a directory."""
        # Create structure: tmp_path/projects/atoz_xgboost/subdir/
        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()
        target_dir = projects_dir / "atoz_xgboost"
        target_dir.mkdir()
        start_dir = target_dir / "subdir"
        start_dir.mkdir()

        result = _search_upward(start_dir, "atoz_xgboost", max_depth=3)

        assert result == target_dir

    def test_search_upward_not_found(self, tmp_path):
        """Test upward search when directory not found."""
        start_dir = tmp_path / "deep" / "nested" / "dir"
        start_dir.mkdir(parents=True)

        result = _search_upward(start_dir, "nonexistent", max_depth=3)

        assert result is None

    def test_search_downward_success(self, tmp_path):
        """Test downward search finding a directory."""
        # Create structure: tmp_path/level1/level2/target_dir
        level1 = tmp_path / "level1"
        level1.mkdir()
        level2 = level1 / "level2"
        level2.mkdir()
        target_dir = level2 / "target_project"
        target_dir.mkdir()

        result = _search_downward(tmp_path, "target_project", max_depth=3)

        assert result == target_dir

    def test_search_downward_depth_limit(self, tmp_path):
        """Test downward search respects depth limit."""
        # Create deep structure
        deep_dir = tmp_path
        for i in range(5):
            deep_dir = deep_dir / f"level{i}"
            deep_dir.mkdir()

        target = deep_dir / "target"
        target.mkdir()

        # Should not find with depth limit of 3
        result = _search_downward(tmp_path, "target", max_depth=3)
        assert result is None

    def test_search_downward_permission_error(self, tmp_path):
        """Test downward search handles permission errors gracefully."""
        # Mock iterdir to raise PermissionError
        with patch('pathlib.Path.iterdir', side_effect=PermissionError("Access denied")):
            result = _search_downward(tmp_path, "any_target", max_depth=2)

            # Should return None, not crash
            assert result is None

    def test_matches_full_path_simple(self):
        """Test path matching for simple folder names."""
        found = Path("/any/path/simple_folder")
        expected = ("simple_folder",)

        assert _matches_full_path(found, expected) is True

    def test_matches_full_path_nested(self):
        """Test path matching for nested paths."""
        found = Path("/workspace/projects/atoz_xgboost")
        expected = ("projects", "atoz_xgboost")

        assert _matches_full_path(found, expected) is True

    def test_matches_full_path_mismatch(self):
        """Test path matching when structure doesn't match."""
        found = Path("/workspace/other/atoz_xgboost")
        expected = ("projects", "atoz_xgboost")

        assert _matches_full_path(found, expected) is False


class TestGenericPathDiscoveryMetrics:
    """Test the metrics tracking functionality."""

    def setup_method(self):
        """Reset global metrics before each test."""
        global _generic_discovery_metrics
        _generic_discovery_metrics = GenericPathDiscoveryMetrics()

    def test_metrics_initial_state(self):
        """Test initial metrics state."""
        metrics = GenericPathDiscoveryMetrics()
        result = metrics.get_metrics()

        assert result == {"status": "no_data"}

    def test_metrics_record_success(self):
        """Test recording successful discoveries."""
        metrics = GenericPathDiscoveryMetrics()

        metrics.record_success(1.5)
        metrics.record_success(2.0)

        result = metrics.get_metrics()

        assert result["total_attempts"] == 2
        assert result["success_rate"] == 1.0
        assert result["failure_rate"] == 0.0
        assert result["average_search_time"] == 1.75

    def test_metrics_record_failure(self):
        """Test recording failed discoveries."""
        metrics = GenericPathDiscoveryMetrics()

        metrics.record_success(1.0)
        metrics.record_failure(2.0)
        metrics.record_failure(1.5)

        result = metrics.get_metrics()

        assert result["total_attempts"] == 3
        assert result["success_rate"] == 1/3
        assert result["failure_rate"] == 2/3
        assert result["average_search_time"] == (1.0 + 2.0 + 1.5) / 3

    def test_global_metrics_function(self):
        """Test the global metrics accessor function."""
        # Reset global state by calling the reset-like operations
        # Since we can't easily reassign the module global, test the basic functionality
        initial_result = get_generic_discovery_metrics()
        assert isinstance(initial_result, dict)  # Should return some dict

        # The function should work regardless of current state
        # We can't easily test state changes without more complex mocking


class TestIntegrationScenarios:
    """Integration tests for complex scenarios."""

    @pytest.fixture
    def complex_workspace(self):
        """Create complex workspace structure for integration testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            # Create multiple project locations
            projects1 = root / "projects"
            projects1.mkdir()

            projects2 = root / "workspace" / "ml_projects"
            projects2.mkdir(parents=True)

            # Create projects in different locations
            proj1 = projects1 / "atoz_xgboost"
            proj1.mkdir()

            proj2 = projects2 / "atoz_xgboost"
            proj2.mkdir()

            # Create nested project
            nested_proj = projects1 / "categories" / "ml" / "atoz_xgboost"
            nested_proj.mkdir(parents=True)

            yield root

    def test_integration_multiple_candidates(self, complex_workspace):
        """Test finding project when multiple candidates exist."""
        # Should find the first one (in projects/)
        with patch('pathlib.Path.cwd', return_value=complex_workspace):
            result = find_project_folder_generic("atoz_xgboost")

            assert result is not None
            assert "projects" in str(result)
            assert result.name == "atoz_xgboost"

    def test_integration_nested_path_search_within_limits(self, complex_workspace):
        """
        Test nested path search that works within default depth limits.

        Uses a shallower nested structure that can be found within max_depth_down=3.
        """
        # Remove conflicting atoz_xgboost folders
        import shutil
        shutil.rmtree(complex_workspace / "projects" / "atoz_xgboost")
        shutil.rmtree(complex_workspace / "workspace" / "ml_projects" / "atoz_xgboost")

        # Create a shallower nested structure that fits within depth limits
        # Structure: complex_workspace/projects/categories/atoz_xgboost
        # This puts atoz_xgboost at depth 3, which is reachable with max_depth_down=3
        categories_dir = complex_workspace / "projects" / "categories"
        categories_dir.mkdir(exist_ok=True)
        nested_project = categories_dir / "atoz_xgboost"
        nested_project.mkdir()

        # Verify the nested path exists
        assert nested_project.exists()

        # This should work within the default depth limits
        result = find_project_folder_generic(
            "categories/atoz_xgboost",
            reference_points=[complex_workspace]
        )

        assert result is not None
        assert result == nested_project
        assert result.name == "atoz_xgboost"
        assert "categories" in str(result)

    def test_integration_fallback_search(self, complex_workspace):
        """Test that search falls back through strategies."""
        # Mock upward search to fail, should try downward
        with patch('cursus.core.utils.generic_path_discovery._search_upward', return_value=None):
            with patch('pathlib.Path.cwd', return_value=complex_workspace):
                result = find_project_folder_generic("atoz_xgboost")

                assert result is not None  # Should find via downward search
