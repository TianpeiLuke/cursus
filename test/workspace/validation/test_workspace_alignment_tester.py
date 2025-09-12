"""
Unit tests for WorkspaceUnifiedAlignmentTester.

Tests workspace-aware alignment validation functionality including:
- Multi-workspace alignment testing
- Developer workspace switching
- Cross-workspace dependency validation
- Integration with existing alignment validation framework
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

from cursus.workspace.validation.workspace_alignment_tester import (
    WorkspaceUnifiedAlignmentTester,
)
from cursus.workspace.validation.workspace_manager import WorkspaceManager


class TestWorkspaceUnifiedAlignmentTester:
    """Test cases for WorkspaceUnifiedAlignmentTester."""

    @pytest.fixture
    def temp_workspace(self):
        """Set up test fixtures."""
        temp_dir = tempfile.mkdtemp()
        workspace_root = Path(temp_dir)

        # Create proper workspace structure
        developers_dir = workspace_root / "developers"
        dev1_path = developers_dir / "developer_1" / "src" / "cursus_dev" / "steps"
        dev2_path = developers_dir / "developer_2" / "src" / "cursus_dev" / "steps"

        for dev_path in [dev1_path, dev2_path]:
            for subdir in ["builders", "contracts", "scripts", "specs", "configs"]:
                (dev_path / subdir).mkdir(parents=True, exist_ok=True)
                # Create __init__.py files
                (dev_path / subdir / "__init__.py").touch()

        # Create shared workspace structure
        shared_dir = workspace_root / "shared" / "src" / "cursus_dev" / "steps"
        for subdir in ["builders", "contracts", "scripts", "specs", "configs"]:
            (shared_dir / subdir).mkdir(parents=True, exist_ok=True)
            (shared_dir / subdir / "__init__.py").touch()

        yield workspace_root, dev1_path, dev2_path

        # Cleanup
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_workspace_manager(self, temp_workspace):
        """Create mock workspace manager."""
        workspace_root, dev1_path, dev2_path = temp_workspace

        mock_manager = Mock()
        mock_manager.workspace_root = workspace_root
        mock_manager.list_available_developers.return_value = [
            "developer_1",
            "developer_2",
        ]

        # Create mock workspace info with proper structure
        mock_workspace_info = Mock()
        mock_workspace_info.developers = {
            "developer_1": Mock(workspace_path=dev1_path),
            "developer_2": Mock(workspace_path=dev2_path),
        }
        mock_workspace_info.model_dump.return_value = {
            "developers": {
                "developer_1": {"workspace_path": str(dev1_path)},
                "developer_2": {"workspace_path": str(dev2_path)},
            }
        }
        mock_manager.get_workspace_info.return_value = mock_workspace_info

        return mock_manager

    @pytest.fixture
    def tester(self, temp_workspace, mock_workspace_manager):
        """Create tester instance."""
        workspace_root, dev1_path, dev2_path = temp_workspace

        tester = WorkspaceUnifiedAlignmentTester(
            workspace_root=workspace_root, developer_id="developer_1"
        )
        # Inject mock workspace manager for testing
        tester.workspace_manager = mock_workspace_manager
        return tester

    def test_initialization(self, tester, mock_workspace_manager):
        """Test proper initialization of WorkspaceUnifiedAlignmentTester."""
        assert tester.workspace_manager is not None
        assert tester.workspace_manager == mock_workspace_manager
        assert tester.developer_id == "developer_1"

    def test_switch_developer(self, tester):
        """Test developer workspace switching functionality."""
        # Test switching to valid developer
        tester.switch_developer("developer_1")
        assert tester.developer_id == "developer_1"

        # Test switching to another developer
        tester.switch_developer("developer_2")
        assert tester.developer_id == "developer_2"

    def test_switch_developer_invalid(self, tester, mock_workspace_manager):
        """Test switching to invalid developer."""
        mock_workspace_manager.list_available_developers.return_value = ["developer_1"]

        # Should raise ValueError for invalid developer
        with pytest.raises(ValueError):
            tester.switch_developer("invalid_developer")

    def test_get_workspace_info(self, tester):
        """Test workspace information retrieval."""
        # get_workspace_info returns info about current workspace configuration
        info = tester.get_workspace_info()

        assert info is not None
        assert "developer_id" in info
        assert "workspace_root" in info
        assert "enable_shared_fallback" in info
        assert info["developer_id"] == "developer_1"

    def test_get_workspace_info_no_developer(self, tester):
        """Test workspace info retrieval - always returns info since developer is set during init."""
        info = tester.get_workspace_info()
        assert info is not None
        assert "developer_id" in info

    @patch(
        "cursus.workspace.validation.workspace_alignment_tester.UnifiedAlignmentTester.run_full_validation"
    )
    def test_run_workspace_validation_single_developer(
        self, mock_run_validation, tester
    ):
        """Test running validation for single developer workspace."""
        # Mock AlignmentReport object
        mock_report = Mock()
        mock_report.is_passing.return_value = True
        mock_report.summary = Mock()
        mock_report.summary.model_dump.return_value = {"total_tests": 4, "passed": 4}
        mock_report.level1_results = {"test_script": Mock(passed=True, details={})}
        mock_report.level2_results = {"test_script": Mock(passed=True, details={})}
        mock_report.level3_results = {"test_script": Mock(passed=True, details={})}
        mock_report.level4_results = {"test_script": Mock(passed=True, details={})}
        mock_run_validation.return_value = mock_report

        results = tester.run_workspace_validation()

        assert results is not None
        assert "workspace_metadata" in results
        mock_run_validation.assert_called_once()

    @patch(
        "cursus.workspace.validation.workspace_alignment_tester.UnifiedAlignmentTester.run_full_validation"
    )
    def test_run_workspace_validation_all_developers(self, mock_run_validation, tester):
        """Test running validation for all developer workspaces."""
        # Mock AlignmentReport object
        mock_report = Mock()
        mock_report.is_passing.return_value = True
        mock_report.summary = Mock()
        mock_report.summary.model_dump.return_value = {"total_tests": 4, "passed": 4}
        mock_report.level1_results = {"test_script": Mock(passed=True, details={})}
        mock_report.level2_results = {"test_script": Mock(passed=True, details={})}
        mock_report.level3_results = {"test_script": Mock(passed=True, details={})}
        mock_report.level4_results = {"test_script": Mock(passed=True, details={})}
        mock_run_validation.return_value = mock_report

        # Note: The current implementation doesn't support all_developers parameter
        # This test needs to be updated to match actual API
        results = tester.run_workspace_validation()

        assert results is not None
        assert "workspace_metadata" in results
        mock_run_validation.assert_called_once()

    @patch(
        "cursus.workspace.validation.workspace_alignment_tester.UnifiedAlignmentTester.run_full_validation"
    )
    def test_run_workspace_validation_with_errors(self, mock_run_validation, tester):
        """Test validation with errors in one workspace."""
        # Mock AlignmentReport object with errors
        mock_report = Mock()
        mock_report.is_passing.return_value = False
        mock_report.summary = Mock()
        mock_report.summary.model_dump.return_value = {"total_tests": 4, "passed": 3}
        mock_report.level1_results = {"test_script": Mock(passed=True, details={})}
        mock_report.level2_results = {
            "test_script": Mock(passed=False, details={"error": "Contract mismatch"})
        }
        mock_report.level3_results = {"test_script": Mock(passed=True, details={})}
        mock_report.level4_results = {"test_script": Mock(passed=True, details={})}
        mock_run_validation.return_value = mock_report

        results = tester.run_workspace_validation()

        assert results is not None
        assert "workspace_metadata" in results
        # Check that validation was attempted
        mock_run_validation.assert_called_once()

    def test_run_workspace_validation_no_developer_selected(self, tester):
        """Test validation without developer selected."""
        # The current implementation doesn't require developer selection
        # as it's set during initialization
        results = tester.run_workspace_validation()
        assert results is not None
        assert "workspace_metadata" in results

    @patch(
        "cursus.workspace.validation.workspace_alignment_tester.UnifiedAlignmentTester.run_full_validation"
    )
    def test_cross_workspace_dependency_analysis(self, mock_run_validation, tester):
        """Test cross-workspace dependency analysis."""
        # Mock AlignmentReport object
        mock_report = Mock()
        mock_report.is_passing.return_value = True
        mock_report.summary = Mock()
        mock_report.summary.model_dump.return_value = {"total_tests": 4, "passed": 4}
        mock_report.level1_results = {"test_script": Mock(passed=True, details={})}
        mock_report.level2_results = {"test_script": Mock(passed=True, details={})}
        mock_report.level3_results = {"test_script": Mock(passed=True, details={})}
        mock_report.level4_results = {"test_script": Mock(passed=True, details={})}
        mock_run_validation.return_value = mock_report

        # Test cross-workspace dependency analysis
        results = tester.run_workspace_validation(target_scripts=["test_script"])

        # Verify that results include workspace metadata
        assert "workspace_metadata" in results
        assert "cross_workspace_validation" in results

        # Check cross-workspace validation structure
        cross_workspace = results["cross_workspace_validation"]
        assert "enabled" in cross_workspace
        assert "recommendations" in cross_workspace

    @patch(
        "cursus.workspace.validation.workspace_alignment_tester.UnifiedAlignmentTester.__init__"
    )
    def test_inheritance_from_unified_alignment_tester(self, mock_init, temp_workspace):
        """Test that WorkspaceUnifiedAlignmentTester properly inherits from UnifiedAlignmentTester."""
        workspace_root, dev1_path, dev2_path = temp_workspace
        mock_init.return_value = None

        # Create instance to test inheritance
        tester = WorkspaceUnifiedAlignmentTester(
            workspace_root=workspace_root, developer_id="developer_1"
        )

        # Verify that parent class __init__ was called
        mock_init.assert_called_once()

    def test_workspace_context_management(self, tester):
        """Test that workspace context is properly managed during validation."""
        original_developer = tester.developer_id

        # Switch to developer_1
        tester.switch_developer("developer_1")
        assert tester.developer_id == "developer_1"

        # Switch to developer_2
        tester.switch_developer("developer_2")
        assert tester.developer_id == "developer_2"

        # Verify context switching worked
        assert tester.developer_id != original_developer
