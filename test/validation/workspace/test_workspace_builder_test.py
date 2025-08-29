"""
Unit tests for WorkspaceUniversalStepBuilderTest.

Tests workspace-aware step builder validation functionality including:
- Dynamic loading of builder classes from workspace directories
- Multi-workspace builder testing
- Workspace-specific error reporting
- Integration with existing builder validation framework
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import tempfile
import os
from pathlib import Path
import sys

from src.cursus.validation.workspace.workspace_builder_test import WorkspaceUniversalStepBuilderTest
from src.cursus.validation.workspace.workspace_manager import WorkspaceManager


class TestWorkspaceUniversalStepBuilderTest(unittest.TestCase):
    """Test cases for WorkspaceUniversalStepBuilderTest."""
    
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
        
        # Create mock module loader
        self.mock_module_loader = Mock()
        self.mock_workspace_manager.get_module_loader.return_value = self.mock_module_loader
        
        # Create tester instance
        self.tester = WorkspaceUniversalStepBuilderTest(
            workspace_root=self.workspace_root,
            developer_id="developer_1",
            builder_file_path="test_builder.py"
        )
        # Inject mock workspace manager for testing
        self.tester.workspace_manager = self.mock_workspace_manager
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test proper initialization of WorkspaceUniversalStepBuilderTest."""
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
            "builders": ["processing_step_builder", "createmodel_step_builder"]
        }
        self.mock_workspace_manager.get_workspace_info.return_value = mock_info
        
        info = self.tester.get_workspace_info()
        self.assertEqual(info, mock_info)
        self.mock_workspace_manager.get_workspace_info.assert_called_once_with("developer_1")
    
    def test_get_workspace_info_no_developer(self):
        """Test workspace info retrieval without selected developer."""
        info = self.tester.get_workspace_info()
        self.assertIsNone(info)
    
    def test_load_workspace_builder_classes(self):
        """Test loading builder classes from workspace."""
        self.tester.switch_developer("developer_1")
        
        # Mock builder files
        mock_file_resolver = Mock()
        mock_file_resolver.find_builder_files.return_value = [
            "processing_step_builder.py",
            "createmodel_step_builder.py"
        ]
        self.mock_workspace_manager.get_file_resolver.return_value = mock_file_resolver
        
        # Mock builder classes
        mock_builder_class1 = Mock()
        mock_builder_class1.__name__ = "ProcessingStepBuilder"
        mock_builder_class2 = Mock()
        mock_builder_class2.__name__ = "CreateModelStepBuilder"
        
        self.mock_module_loader.load_builder_class.side_effect = [
            mock_builder_class1, mock_builder_class2
        ]
        
        builder_classes = self.tester._load_workspace_builder_classes()
        
        self.assertEqual(len(builder_classes), 2)
        self.assertIn("ProcessingStepBuilder", [cls.__name__ for cls in builder_classes])
        self.assertIn("CreateModelStepBuilder", [cls.__name__ for cls in builder_classes])
    
    def test_load_workspace_builder_classes_no_developer(self):
        """Test loading builder classes without selected developer."""
        builder_classes = self.tester._load_workspace_builder_classes()
        self.assertEqual(builder_classes, [])
    
    def test_load_workspace_builder_classes_with_errors(self):
        """Test loading builder classes with import errors."""
        self.tester.switch_developer("developer_1")
        
        # Mock builder files
        mock_file_resolver = Mock()
        mock_file_resolver.find_builder_files.return_value = [
            "valid_builder.py",
            "invalid_builder.py"
        ]
        self.mock_workspace_manager.get_file_resolver.return_value = mock_file_resolver
        
        # Mock one successful load and one failure
        mock_builder_class = Mock()
        mock_builder_class.__name__ = "ValidBuilder"
        
        self.mock_module_loader.load_builder_class.side_effect = [
            mock_builder_class,
            ImportError("Failed to import invalid_builder")
        ]
        
        builder_classes = self.tester._load_workspace_builder_classes()
        
        # Should return only the successfully loaded class
        self.assertEqual(len(builder_classes), 1)
        self.assertEqual(builder_classes[0].__name__, "ValidBuilder")
    
    @patch('src.cursus.validation.workspace.workspace_builder_test.UniversalStepBuilderTest.test_builder_class')
    def test_run_workspace_builder_test_single_developer(self, mock_test_builder):
        """Test running builder tests for single developer workspace."""
        mock_test_builder.return_value = {
            "passed": True,
            "errors": [],
            "builder_name": "TestBuilder"
        }
        
        self.tester.switch_developer("developer_1")
        
        # Mock builder classes
        mock_builder_class = Mock()
        mock_builder_class.__name__ = "TestBuilder"
        
        with patch.object(self.tester, '_load_workspace_builder_classes', return_value=[mock_builder_class]):
            results = self.tester.run_workspace_builder_test()
        
        self.assertIsNotNone(results)
        self.assertIn("developer_1", results)
        self.assertEqual(len(results), 1)
        mock_test_builder.assert_called_once_with(mock_builder_class)
    
    @patch('src.cursus.validation.workspace.workspace_builder_test.UniversalStepBuilderTest.test_builder_class')
    def test_run_workspace_builder_test_all_developers(self, mock_test_builder):
        """Test running builder tests for all developer workspaces."""
        mock_test_builder.return_value = {
            "passed": True,
            "errors": [],
            "builder_name": "TestBuilder"
        }
        
        # Mock builder classes for each developer
        mock_builder_class = Mock()
        mock_builder_class.__name__ = "TestBuilder"
        
        with patch.object(self.tester, '_load_workspace_builder_classes', return_value=[mock_builder_class]):
            results = self.tester.run_workspace_builder_test(all_developers=True)
        
        self.assertIsNotNone(results)
        self.assertIn("developer_1", results)
        self.assertIn("developer_2", results)
        self.assertEqual(len(results), 2)
        # Should be called twice (once for each developer)
        self.assertEqual(mock_test_builder.call_count, 2)
    
    @patch('src.cursus.validation.workspace.workspace_builder_test.UniversalStepBuilderTest.test_builder_class')
    def test_run_workspace_builder_test_with_failures(self, mock_test_builder):
        """Test builder tests with failures in one workspace."""
        def side_effect(builder_class):
            if self.tester.current_developer == "developer_1":
                return {
                    "passed": False,
                    "errors": ["Builder validation failed"],
                    "builder_name": builder_class.__name__
                }
            else:
                return {
                    "passed": True,
                    "errors": [],
                    "builder_name": builder_class.__name__
                }
        
        mock_test_builder.side_effect = side_effect
        
        # Mock builder classes
        mock_builder_class = Mock()
        mock_builder_class.__name__ = "TestBuilder"
        
        with patch.object(self.tester, '_load_workspace_builder_classes', return_value=[mock_builder_class]):
            results = self.tester.run_workspace_builder_test(all_developers=True)
        
        self.assertIsNotNone(results)
        self.assertFalse(results["developer_1"]["TestBuilder"]["passed"])
        self.assertTrue(results["developer_2"]["TestBuilder"]["passed"])
    
    def test_run_workspace_builder_test_no_developer_selected(self):
        """Test builder tests without developer selected and all_developers=False."""
        results = self.tester.run_workspace_builder_test(all_developers=False)
        self.assertEqual(results, {})
    
    @patch('src.cursus.validation.workspace.workspace_builder_test.WorkspaceUniversalStepBuilderTest')
    def test_test_all_workspace_builders_class_method(self, mock_class):
        """Test the class method for testing all workspace builders."""
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        mock_instance.run_workspace_builder_test.return_value = {
            "developer_1": {"TestBuilder": {"passed": True, "errors": []}},
            "developer_2": {"TestBuilder": {"passed": True, "errors": []}}
        }
        
        results = WorkspaceUniversalStepBuilderTest.test_all_workspace_builders(
            workspace_root=self.workspace_root,
            developer_id="developer_1"
        )
        
        self.assertIsNotNone(results)
        mock_class.assert_called_once_with(
            workspace_root=self.workspace_root,
            developer_id="developer_1",
            builder_file_path=unittest.mock.ANY
        )
        mock_instance.run_workspace_builder_test.assert_called_once_with(test_config=None)
    
    def test_workspace_context_management(self):
        """Test that workspace context is properly managed during testing."""
        original_developer = self.tester.current_developer
        
        # Switch to developer_1
        self.tester.switch_developer("developer_1")
        self.assertEqual(self.tester.current_developer, "developer_1")
        
        # Switch to developer_2
        self.tester.switch_developer("developer_2")
        self.assertEqual(self.tester.current_developer, "developer_2")
        
        # Verify context switching worked
        self.assertNotEqual(self.tester.current_developer, original_developer)
    
    @patch('src.cursus.validation.workspace.workspace_builder_test.UniversalStepBuilderTest.__init__')
    def test_inheritance_from_universal_step_builder_test(self, mock_init):
        """Test that WorkspaceUniversalStepBuilderTest properly inherits from UniversalStepBuilderTest."""
        mock_init.return_value = None
        
        # Create instance to test inheritance
        tester = WorkspaceUniversalStepBuilderTest(
            workspace_root=self.workspace_root,
            developer_id="developer_1",
            builder_file_path="test_builder.py"
        )
        
        # Verify that parent class __init__ was called
        mock_init.assert_called_once()
    
    def test_error_handling_during_builder_loading(self):
        """Test error handling when builder loading fails."""
        self.tester.switch_developer("developer_1")
        
        # Mock file resolver to raise exception
        mock_file_resolver = Mock()
        mock_file_resolver.find_builder_files.side_effect = Exception("File system error")
        self.mock_workspace_manager.get_file_resolver.return_value = mock_file_resolver
        
        builder_classes = self.tester._load_workspace_builder_classes()
        
        # Should return empty list on error
        self.assertEqual(builder_classes, [])
    
    def test_multiple_builders_in_workspace(self):
        """Test handling multiple builder classes in a single workspace."""
        self.tester.switch_developer("developer_1")
        
        # Mock multiple builder files
        mock_file_resolver = Mock()
        mock_file_resolver.find_builder_files.return_value = [
            "processing_step_builder.py",
            "createmodel_step_builder.py",
            "registermodel_step_builder.py"
        ]
        self.mock_workspace_manager.get_file_resolver.return_value = mock_file_resolver
        
        # Mock multiple builder classes
        mock_builders = []
        for i, name in enumerate(["ProcessingStepBuilder", "CreateModelStepBuilder", "RegisterModelStepBuilder"]):
            mock_builder = Mock()
            mock_builder.__name__ = name
            mock_builders.append(mock_builder)
        
        self.mock_module_loader.load_builder_class.side_effect = mock_builders
        
        builder_classes = self.tester._load_workspace_builder_classes()
        
        self.assertEqual(len(builder_classes), 3)
        builder_names = [cls.__name__ for cls in builder_classes]
        self.assertIn("ProcessingStepBuilder", builder_names)
        self.assertIn("CreateModelStepBuilder", builder_names)
        self.assertIn("RegisterModelStepBuilder", builder_names)


if __name__ == '__main__':
    unittest.main()
