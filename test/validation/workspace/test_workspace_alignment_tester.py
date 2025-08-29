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
import os
from pathlib import Path

from src.cursus.validation.workspace.workspace_alignment_tester import WorkspaceUnifiedAlignmentTester
from src.cursus.validation.workspace.workspace_manager import WorkspaceManager


class TestWorkspaceUnifiedAlignmentTester(unittest.TestCase):
    """Test cases for WorkspaceUnifiedAlignmentTester."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_root = Path(self.temp_dir)
        
        # Create mock workspace structure
        self.dev1_path = self.workspace_root / "developer_1"
        self.dev2_path = self.workspace_root / "developer_2"
        
        for dev_path in [self.dev1_path, self.dev2_path]:
            for subdir in ["builders", "contracts", "scripts", "specs", "configs"]:
                (dev_path / subdir).mkdir(parents=True, exist_ok=True)
        
        # Create mock workspace manager
        self.mock_workspace_manager = Mock()
        self.mock_workspace_manager.workspace_root = self.workspace_root
        self.mock_workspace_manager.list_available_developers.return_value = [
            "developer_1", "developer_2"
        ]
        
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
        self.assertIsNone(self.tester.current_developer)
    
    def test_switch_developer(self):
        """Test developer workspace switching functionality."""
        # Test switching to valid developer
        result = self.tester.switch_developer("developer_1")
        self.assertTrue(result)
        self.assertEqual(self.tester.current_developer, "developer_1")
        
        # Test switching to another developer
        result = self.tester.switch_developer("developer_2")
        self.assertTrue(result)
        self.assertEqual(self.tester.current_developer, "developer_2")
    
    def test_switch_developer_invalid(self):
        """Test switching to invalid developer."""
        self.mock_workspace_manager.list_available_developers.return_value = ["developer_1"]
        
        result = self.tester.switch_developer("invalid_developer")
        self.assertFalse(result)
        self.assertIsNone(self.tester.current_developer)
    
    def test_get_workspace_info(self):
        """Test workspace information retrieval."""
        self.tester.switch_developer("developer_1")
        
        # Mock workspace info
        mock_info = {
            "developer": "developer_1",
            "workspace_path": str(self.dev1_path),
            "components": ["builders", "contracts", "scripts", "specs", "configs"]
        }
        self.mock_workspace_manager.get_workspace_info.return_value = mock_info
        
        info = self.tester.get_workspace_info()
        self.assertEqual(info, mock_info)
        self.mock_workspace_manager.get_workspace_info.assert_called_once_with("developer_1")
    
    def test_get_workspace_info_no_developer(self):
        """Test workspace info retrieval without selected developer."""
        info = self.tester.get_workspace_info()
        self.assertIsNone(info)
    
    @patch('src.cursus.validation.workspace.workspace_alignment_tester.UnifiedAlignmentTester.run_all_tests')
    def test_run_workspace_validation_single_developer(self, mock_run_tests):
        """Test running validation for single developer workspace."""
        mock_run_tests.return_value = {
            "level1": {"passed": True, "errors": []},
            "level2": {"passed": True, "errors": []},
            "level3": {"passed": True, "errors": []},
            "level4": {"passed": True, "errors": []}
        }
        
        self.tester.switch_developer("developer_1")
        results = self.tester.run_workspace_validation()
        
        self.assertIsNotNone(results)
        self.assertIn("developer_1", results)
        self.assertEqual(len(results), 1)
        mock_run_tests.assert_called_once()
    
    @patch('src.cursus.validation.workspace.workspace_alignment_tester.UnifiedAlignmentTester.run_all_tests')
    def test_run_workspace_validation_all_developers(self, mock_run_tests):
        """Test running validation for all developer workspaces."""
        mock_run_tests.return_value = {
            "level1": {"passed": True, "errors": []},
            "level2": {"passed": True, "errors": []},
            "level3": {"passed": True, "errors": []},
            "level4": {"passed": True, "errors": []}
        }
        
        results = self.tester.run_workspace_validation(all_developers=True)
        
        self.assertIsNotNone(results)
        self.assertIn("developer_1", results)
        self.assertIn("developer_2", results)
        self.assertEqual(len(results), 2)
        # Should be called twice (once for each developer)
        self.assertEqual(mock_run_tests.call_count, 2)
    
    @patch('src.cursus.validation.workspace.workspace_alignment_tester.UnifiedAlignmentTester.run_all_tests')
    def test_run_workspace_validation_with_errors(self, mock_run_tests):
        """Test validation with errors in one workspace."""
        def side_effect():
            if self.tester.current_developer == "developer_1":
                return {
                    "level1": {"passed": True, "errors": []},
                    "level2": {"passed": False, "errors": ["Contract mismatch"]},
                    "level3": {"passed": True, "errors": []},
                    "level4": {"passed": True, "errors": []}
                }
            else:
                return {
                    "level1": {"passed": True, "errors": []},
                    "level2": {"passed": True, "errors": []},
                    "level3": {"passed": True, "errors": []},
                    "level4": {"passed": True, "errors": []}
                }
        
        mock_run_tests.side_effect = side_effect
        
        results = self.tester.run_workspace_validation(all_developers=True)
        
        self.assertIsNotNone(results)
        self.assertFalse(results["developer_1"]["level2"]["passed"])
        self.assertTrue(results["developer_2"]["level2"]["passed"])
    
    def test_run_workspace_validation_no_developer_selected(self):
        """Test validation without developer selected and all_developers=False."""
        results = self.tester.run_workspace_validation(all_developers=False)
        self.assertEqual(results, {})
    
    @patch('src.cursus.validation.workspace.workspace_alignment_tester.UnifiedAlignmentTester.run_all_tests')
    def test_cross_workspace_dependency_analysis(self, mock_run_tests):
        """Test cross-workspace dependency analysis."""
        mock_run_tests.return_value = {
            "level1": {"passed": True, "errors": []},
            "level2": {"passed": True, "errors": []},
            "level3": {"passed": True, "errors": [], "dependencies": ["step_a", "step_b"]},
            "level4": {"passed": True, "errors": []}
        }
        
        results = self.tester.run_workspace_validation(all_developers=True)
        
        # Verify that dependencies are tracked across workspaces
        self.assertIn("developer_1", results)
        self.assertIn("developer_2", results)
        
        # Check that level3 results include dependency information
        if "dependencies" in results["developer_1"]["level3"]:
            self.assertIsInstance(results["developer_1"]["level3"]["dependencies"], list)
    
    @patch('src.cursus.validation.workspace.workspace_alignment_tester.UnifiedAlignmentTester.__init__')
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
        original_developer = self.tester.current_developer
        
        # Switch to developer_1
        self.tester.switch_developer("developer_1")
        self.assertEqual(self.tester.current_developer, "developer_1")
        
        # Switch to developer_2
        self.tester.switch_developer("developer_2")
        self.assertEqual(self.tester.current_developer, "developer_2")
        
        # Verify context switching worked
        self.assertNotEqual(self.tester.current_developer, original_developer)


if __name__ == '__main__':
    unittest.main()
