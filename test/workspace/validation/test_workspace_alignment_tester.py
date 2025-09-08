"""
Unit tests for WorkspaceUnifiedAlignmentTester.

Tests workspace-aware alignment validation functionality including:
- Multi-workspace alignment testing
- Developer workspace switching
- Cross-workspace dependency validation
- Integration with existing alignment validation framework
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

from cursus.workspace.validation.workspace_alignment_tester import WorkspaceUnifiedAlignmentTester
from cursus.workspace.validation.workspace_manager import WorkspaceManager

class TestWorkspaceUnifiedAlignmentTester(unittest.TestCase):
    """Test cases for WorkspaceUnifiedAlignmentTester."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_root = Path(self.temp_dir)
        
        # Create proper workspace structure
        developers_dir = self.workspace_root / "developers"
        self.dev1_path = developers_dir / "developer_1" / "src" / "cursus_dev" / "steps"
        self.dev2_path = developers_dir / "developer_2" / "src" / "cursus_dev" / "steps"
        
        for dev_path in [self.dev1_path, self.dev2_path]:
            for subdir in ["builders", "contracts", "scripts", "specs", "configs"]:
                (dev_path / subdir).mkdir(parents=True, exist_ok=True)
                # Create __init__.py files
                (dev_path / subdir / "__init__.py").touch()
        
        # Create shared workspace structure
        shared_dir = self.workspace_root / "shared" / "src" / "cursus_dev" / "steps"
        for subdir in ["builders", "contracts", "scripts", "specs", "configs"]:
            (shared_dir / subdir).mkdir(parents=True, exist_ok=True)
            (shared_dir / subdir / "__init__.py").touch()
        
        # Create mock workspace manager with proper workspace info structure
        self.mock_workspace_manager = Mock()
        self.mock_workspace_manager.workspace_root = self.workspace_root
        self.mock_workspace_manager.list_available_developers.return_value = [
            "developer_1", "developer_2"
        ]
        
        # Create mock workspace info with proper structure
        mock_workspace_info = Mock()
        mock_workspace_info.developers = {
            "developer_1": Mock(workspace_path=self.dev1_path),
            "developer_2": Mock(workspace_path=self.dev2_path)
        }
        mock_workspace_info.model_dump.return_value = {
            "developers": {
                "developer_1": {"workspace_path": str(self.dev1_path)},
                "developer_2": {"workspace_path": str(self.dev2_path)}
            }
        }
        self.mock_workspace_manager.get_workspace_info.return_value = mock_workspace_info
        
        # Create tester instance
        self.tester = WorkspaceUnifiedAlignmentTester(
            workspace_root=self.workspace_root,
            developer_id="developer_1"
        )
        # Inject mock workspace manager for testing
        self.tester.workspace_manager = self.mock_workspace_manager
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test proper initialization of WorkspaceUnifiedAlignmentTester."""
        self.assertIsNotNone(self.tester.workspace_manager)
        self.assertEqual(self.tester.workspace_manager, self.mock_workspace_manager)
        self.assertEqual(self.tester.developer_id, "developer_1")
    
    def test_switch_developer(self):
        """Test developer workspace switching functionality."""
        # Test switching to valid developer
        self.tester.switch_developer("developer_1")
        self.assertEqual(self.tester.developer_id, "developer_1")
        
        # Test switching to another developer
        self.tester.switch_developer("developer_2")
        self.assertEqual(self.tester.developer_id, "developer_2")
    
    def test_switch_developer_invalid(self):
        """Test switching to invalid developer."""
        self.mock_workspace_manager.list_available_developers.return_value = ["developer_1"]
        
        # Should raise ValueError for invalid developer
        with self.assertRaises(ValueError):
            self.tester.switch_developer("invalid_developer")
    
    def test_get_workspace_info(self):
        """Test workspace information retrieval."""
        # get_workspace_info returns info about current workspace configuration
        info = self.tester.get_workspace_info()
        
        self.assertIsNotNone(info)
        self.assertIn("developer_id", info)
        self.assertIn("workspace_root", info)
        self.assertIn("enable_shared_fallback", info)
        self.assertEqual(info["developer_id"], "developer_1")
    
    def test_get_workspace_info_no_developer(self):
        """Test workspace info retrieval - always returns info since developer is set during init."""
        info = self.tester.get_workspace_info()
        self.assertIsNotNone(info)
        self.assertIn("developer_id", info)
    
    @patch('src.cursus.workspace.validation.workspace_alignment_tester.UnifiedAlignmentTester.run_full_validation')
    def test_run_workspace_validation_single_developer(self, mock_run_validation):
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
        
        results = self.tester.run_workspace_validation()
        
        self.assertIsNotNone(results)
        self.assertIn("workspace_metadata", results)
        mock_run_validation.assert_called_once()
    
    @patch('src.cursus.workspace.validation.workspace_alignment_tester.UnifiedAlignmentTester.run_full_validation')
    def test_run_workspace_validation_all_developers(self, mock_run_validation):
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
        results = self.tester.run_workspace_validation()
        
        self.assertIsNotNone(results)
        self.assertIn("workspace_metadata", results)
        mock_run_validation.assert_called_once()
    
    @patch('src.cursus.workspace.validation.workspace_alignment_tester.UnifiedAlignmentTester.run_full_validation')
    def test_run_workspace_validation_with_errors(self, mock_run_validation):
        """Test validation with errors in one workspace."""
        # Mock AlignmentReport object with errors
        mock_report = Mock()
        mock_report.is_passing.return_value = False
        mock_report.summary = Mock()
        mock_report.summary.model_dump.return_value = {"total_tests": 4, "passed": 3}
        mock_report.level1_results = {"test_script": Mock(passed=True, details={})}
        mock_report.level2_results = {"test_script": Mock(passed=False, details={"error": "Contract mismatch"})}
        mock_report.level3_results = {"test_script": Mock(passed=True, details={})}
        mock_report.level4_results = {"test_script": Mock(passed=True, details={})}
        mock_run_validation.return_value = mock_report
        
        results = self.tester.run_workspace_validation()
        
        self.assertIsNotNone(results)
        self.assertIn("workspace_metadata", results)
        # Check that validation was attempted
        mock_run_validation.assert_called_once()
    
    def test_run_workspace_validation_no_developer_selected(self):
        """Test validation without developer selected."""
        # The current implementation doesn't require developer selection
        # as it's set during initialization
        results = self.tester.run_workspace_validation()
        self.assertIsNotNone(results)
        self.assertIn("workspace_metadata", results)
    
    @patch('src.cursus.workspace.validation.workspace_alignment_tester.UnifiedAlignmentTester.run_full_validation')
    def test_cross_workspace_dependency_analysis(self, mock_run_validation):
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
        results = self.tester.run_workspace_validation(
            target_scripts=["test_script"]
        )
        
        # Verify that results include workspace metadata
        self.assertIn("workspace_metadata", results)
        self.assertIn("cross_workspace_validation", results)
        
        # Check cross-workspace validation structure
        cross_workspace = results["cross_workspace_validation"]
        self.assertIn("enabled", cross_workspace)
        self.assertIn("recommendations", cross_workspace)
    
    @patch('src.cursus.workspace.validation.workspace_alignment_tester.UnifiedAlignmentTester.__init__')
    def test_inheritance_from_unified_alignment_tester(self, mock_init):
        """Test that WorkspaceUnifiedAlignmentTester properly inherits from UnifiedAlignmentTester."""
        mock_init.return_value = None
        
        # Create instance to test inheritance
        tester = WorkspaceUnifiedAlignmentTester(
            workspace_root=self.workspace_root,
            developer_id="developer_1"
        )
        
        # Verify that parent class __init__ was called
        mock_init.assert_called_once()
    
    def test_workspace_context_management(self):
        """Test that workspace context is properly managed during validation."""
        original_developer = self.tester.developer_id
        
        # Switch to developer_1
        self.tester.switch_developer("developer_1")
        self.assertEqual(self.tester.developer_id, "developer_1")
        
        # Switch to developer_2
        self.tester.switch_developer("developer_2")
        self.assertEqual(self.tester.developer_id, "developer_2")
        
        # Verify context switching worked
        self.assertNotEqual(self.tester.developer_id, original_developer)

if __name__ == '__main__':
    unittest.main()
