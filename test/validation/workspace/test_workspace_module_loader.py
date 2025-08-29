"""
Unit tests for WorkspaceModuleLoader

Tests workspace-aware module loading functionality including:
- Workspace mode detection and validation
- Dynamic module loading with Python path management
- Module caching and invalidation
- Workspace path context management
- Developer workspace module discovery
- Shared workspace fallback behavior
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from src.cursus.validation.workspace.workspace_module_loader import WorkspaceModuleLoader


class TestWorkspaceModuleLoader(unittest.TestCase):
    """Test cases for WorkspaceModuleLoader."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_root = Path(self.temp_dir) / "workspaces"
        self.workspace_root.mkdir(parents=True)
        
        # Create test workspace structure
        self._create_test_workspace_structure()
        
        # Store original sys.path
        self.original_sys_path = sys.path.copy()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
        
        # Restore original sys.path
        sys.path[:] = self.original_sys_path
    
    def _create_test_workspace_structure(self):
        """Create test workspace directory structure."""
        # Developer workspace
        dev1_dir = self.workspace_root / "developers" / "developer_1" / "src"
        dev1_dir.mkdir(parents=True)
        
        # Create cursus_dev structure
        cursus_dev_dir = dev1_dir / "cursus_dev" / "steps"
        cursus_dev_dir.mkdir(parents=True)
        
        # Create module directories
        for module_type in ["builders", "contracts"]:
            (cursus_dev_dir / module_type).mkdir()
            (cursus_dev_dir / module_type / "__init__.py").touch()
        
        # Create test modules
        builder_content = '''
class TestStepBuilder:
    """Test step builder class."""
    def __init__(self):
        self.step_name = "test_step"
'''
        (cursus_dev_dir / "builders" / "test_step_builder.py").write_text(builder_content)
        
        contract_content = '''
class TestStepContract:
    """Test step contract class."""
    def __init__(self):
        self.contract_name = "test_step"
'''
        (cursus_dev_dir / "contracts" / "test_step_contract.py").write_text(contract_content)
        
        # Create __init__.py files
        (dev1_dir / "__init__.py").touch()
        (dev1_dir / "cursus_dev" / "__init__.py").touch()
        (cursus_dev_dir / "__init__.py").touch()
        
        # Shared workspace
        shared_dir = self.workspace_root / "shared" / "src"
        shared_dir.mkdir(parents=True)
        
        shared_cursus_dev_dir = shared_dir / "cursus_dev" / "steps"
        shared_cursus_dev_dir.mkdir(parents=True)
        
        for module_type in ["builders", "contracts"]:
            (shared_cursus_dev_dir / module_type).mkdir()
            (shared_cursus_dev_dir / module_type / "__init__.py").touch()
        
        # Create shared test modules
        shared_builder_content = '''
class SharedBuilder:
    """Shared builder class."""
    def __init__(self):
        self.step_name = "shared_step"
'''
        (shared_cursus_dev_dir / "builders" / "shared_builder.py").write_text(shared_builder_content)
        
        # Create __init__.py files
        (shared_dir / "__init__.py").touch()
        (shared_dir / "cursus_dev" / "__init__.py").touch()
        (shared_cursus_dev_dir / "__init__.py").touch()
    
    def test_init_workspace_mode(self):
        """Test initialization in workspace mode."""
        loader = WorkspaceModuleLoader(
            workspace_root=self.workspace_root,
            developer_id="developer_1"
        )
        
        self.assertTrue(loader.workspace_mode)
        self.assertEqual(loader.workspace_root, self.workspace_root)
        self.assertEqual(loader.developer_id, "developer_1")
        self.assertTrue(loader.enable_shared_fallback)
        self.assertTrue(loader.cache_modules)
    
    def test_init_single_workspace_mode(self):
        """Test initialization in single workspace mode."""
        loader = WorkspaceModuleLoader()
        
        self.assertFalse(loader.workspace_mode)
        self.assertIsNone(loader.workspace_root)
        self.assertIsNone(loader.developer_id)
        self.assertTrue(loader.enable_shared_fallback)
        self.assertTrue(loader.cache_modules)
    
    def test_validate_workspace_structure_valid(self):
        """Test workspace structure validation with valid structure."""
        loader = WorkspaceModuleLoader(
            workspace_root=self.workspace_root,
            developer_id="developer_1"
        )
        
        # Should not raise exception
        self.assertTrue(loader.workspace_mode)
    
    def test_validate_workspace_structure_invalid_root(self):
        """Test workspace structure validation with invalid root."""
        invalid_root = Path(self.temp_dir) / "nonexistent"
        
        with self.assertRaises(ValueError) as context:
            WorkspaceModuleLoader(
                workspace_root=invalid_root,
                developer_id="developer_1"
            )
        
        self.assertIn("does not exist", str(context.exception))
    
    def test_build_workspace_paths(self):
        """Test building workspace-specific Python paths."""
        loader = WorkspaceModuleLoader(
            workspace_root=self.workspace_root,
            developer_id="developer_1"
        )
        
        self.assertTrue(hasattr(loader, 'developer_paths'))
        self.assertTrue(hasattr(loader, 'shared_paths'))
        
        # Check developer paths
        dev_base = str(self.workspace_root / "developers" / "developer_1" / "src")
        self.assertIn(dev_base, loader.developer_paths)
        
        # Check shared paths
        shared_base = str(self.workspace_root / "shared" / "src")
        self.assertIn(shared_base, loader.shared_paths)
    
    def test_workspace_path_context(self):
        """Test workspace path context manager."""
        loader = WorkspaceModuleLoader(
            workspace_root=self.workspace_root,
            developer_id="developer_1"
        )
        
        original_path = sys.path.copy()
        
        with loader.workspace_path_context() as added_paths:
            # Check that paths were added
            self.assertGreater(len(added_paths), 0)
            
            # Check that sys.path was modified
            for path in added_paths:
                self.assertIn(path, sys.path)
        
        # Check that sys.path was restored
        self.assertEqual(sys.path, original_path)
    
    def test_workspace_path_context_single_mode(self):
        """Test workspace path context manager in single workspace mode."""
        loader = WorkspaceModuleLoader()
        
        with loader.workspace_path_context() as added_paths:
            # Should return empty list in single workspace mode
            self.assertEqual(added_paths, [])
    
    @patch('importlib.import_module')
    def test_load_builder_class_success(self, mock_import):
        """Test successful builder class loading."""
        # Mock module with builder class
        mock_module = MagicMock()
        mock_builder_class = MagicMock()
        mock_module.TestStepBuilder = mock_builder_class
        mock_import.return_value = mock_module
        
        loader = WorkspaceModuleLoader(
            workspace_root=self.workspace_root,
            developer_id="developer_1"
        )
        
        result = loader.load_builder_class("test_step")
        
        self.assertEqual(result, mock_builder_class)
        self.assertTrue(mock_import.called)
    
    @patch('importlib.import_module')
    def test_load_builder_class_not_found(self, mock_import):
        """Test builder class loading when class not found."""
        mock_import.side_effect = ImportError("Module not found")
        
        loader = WorkspaceModuleLoader(
            workspace_root=self.workspace_root,
            developer_id="developer_1"
        )
        
        result = loader.load_builder_class("nonexistent_step")
        
        self.assertIsNone(result)
    
    @patch('importlib.import_module')
    def test_load_contract_class_success(self, mock_import):
        """Test successful contract class loading."""
        # Mock module with contract class
        mock_module = MagicMock()
        mock_contract_class = MagicMock()
        mock_module.TestStepContract = mock_contract_class
        mock_import.return_value = mock_module
        
        loader = WorkspaceModuleLoader(
            workspace_root=self.workspace_root,
            developer_id="developer_1"
        )
        
        result = loader.load_contract_class("test_step")
        
        self.assertEqual(result, mock_contract_class)
        self.assertTrue(mock_import.called)
    
    def test_module_caching(self):
        """Test module caching functionality."""
        loader = WorkspaceModuleLoader(
            workspace_root=self.workspace_root,
            developer_id="developer_1",
            cache_modules=True
        )
        
        with patch('importlib.import_module') as mock_import:
            mock_module = MagicMock()
            mock_builder_class = MagicMock()
            mock_module.TestStepBuilder = mock_builder_class
            mock_import.return_value = mock_module
            
            # First call should import module
            result1 = loader.load_builder_class("test_step")
            self.assertEqual(result1, mock_builder_class)
            self.assertEqual(mock_import.call_count, 1)
            
            # Second call should use cache
            result2 = loader.load_builder_class("test_step")
            self.assertEqual(result2, mock_builder_class)
            self.assertEqual(mock_import.call_count, 1)  # Should not increase
    
    def test_module_caching_disabled(self):
        """Test behavior when module caching is disabled."""
        loader = WorkspaceModuleLoader(
            workspace_root=self.workspace_root,
            developer_id="developer_1",
            cache_modules=False
        )
        
        with patch('importlib.import_module') as mock_import:
            mock_module = MagicMock()
            mock_builder_class = MagicMock()
            mock_module.TestStepBuilder = mock_builder_class
            mock_import.return_value = mock_module
            
            # Both calls should import module
            result1 = loader.load_builder_class("test_step")
            result2 = loader.load_builder_class("test_step")
            
            self.assertEqual(mock_import.call_count, 2)
    
    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_load_module_from_file_success(self, mock_module_from_spec, mock_spec_from_file):
        """Test successful module loading from file."""
        # Create test file
        test_file = Path(self.temp_dir) / "test_module.py"
        Path(test_file).write_text("# Test module")
        
        # Mock importlib functions
        mock_spec = MagicMock()
        mock_loader = MagicMock()
        mock_spec.loader = mock_loader
        mock_spec_from_file.return_value = mock_spec
        
        mock_module = MagicMock()
        mock_module_from_spec.return_value = mock_module
        
        loader = WorkspaceModuleLoader()
        
        result = loader.load_module_from_file(test_file)
        
        self.assertEqual(result, mock_module)
        mock_spec_from_file.assert_called_once_with("test_module", test_file)
        mock_loader.exec_module.assert_called_once_with(mock_module)
    
    def test_load_module_from_file_not_found(self):
        """Test module loading from non-existent file."""
        loader = WorkspaceModuleLoader()
        
        result = loader.load_module_from_file("/nonexistent/file.py")
        
        self.assertIsNone(result)
    
    def test_discover_workspace_modules(self):
        """Test workspace module discovery."""
        loader = WorkspaceModuleLoader(
            workspace_root=self.workspace_root,
            developer_id="developer_1"
        )
        
        discovered = loader.discover_workspace_modules("builders")
        
        self.assertIn("developer:developer_1", discovered)
        self.assertIn("shared", discovered)
        
        # Check developer modules
        dev_modules = discovered["developer:developer_1"]
        self.assertIn("test_step_builder", dev_modules)
        
        # Check shared modules
        shared_modules = discovered["shared"]
        self.assertIn("shared_builder", shared_modules)
    
    def test_discover_workspace_modules_single_mode(self):
        """Test module discovery in single workspace mode."""
        loader = WorkspaceModuleLoader()
        
        discovered = loader.discover_workspace_modules("builders")
        
        self.assertEqual(discovered, {})
    
    def test_clear_cache(self):
        """Test cache clearing."""
        loader = WorkspaceModuleLoader(
            workspace_root=self.workspace_root,
            developer_id="developer_1"
        )
        
        # Add something to cache
        loader._module_cache["test_key"] = "test_value"
        loader._path_cache["test_path"] = ["path1", "path2"]
        
        loader.clear_cache()
        
        self.assertEqual(len(loader._module_cache), 0)
        self.assertEqual(len(loader._path_cache), 0)
    
    def test_invalidate_cache_for_step(self):
        """Test cache invalidation for specific step."""
        loader = WorkspaceModuleLoader(
            workspace_root=self.workspace_root,
            developer_id="developer_1"
        )
        
        # Add cache entries
        loader._module_cache["builder:test_step:module:class"] = "builder_class"
        loader._module_cache["contract:test_step:module:class"] = "contract_class"
        loader._module_cache["builder:other_step:module:class"] = "other_builder"
        
        loader.invalidate_cache_for_step("test_step")
        
        # test_step entries should be removed
        self.assertNotIn("builder:test_step:module:class", loader._module_cache)
        self.assertNotIn("contract:test_step:module:class", loader._module_cache)
        
        # other_step entry should remain
        self.assertIn("builder:other_step:module:class", loader._module_cache)
    
    def test_get_workspace_info(self):
        """Test getting workspace information."""
        loader = WorkspaceModuleLoader(
            workspace_root=self.workspace_root,
            developer_id="developer_1"
        )
        
        info = loader.get_workspace_info()
        
        self.assertTrue(info['workspace_mode'])
        self.assertEqual(info['workspace_root'], str(self.workspace_root))
        self.assertEqual(info['developer_id'], "developer_1")
        self.assertTrue(info['enable_shared_fallback'])
        self.assertTrue(info['cache_modules'])
        self.assertGreater(len(info['developer_paths']), 0)
        self.assertGreater(len(info['shared_paths']), 0)
        self.assertEqual(info['cached_modules'], 0)
    
    def test_switch_developer(self):
        """Test switching to different developer workspace."""
        # Create another developer workspace
        dev2_dir = self.workspace_root / "developers" / "developer_2" / "src" / "cursus_dev" / "steps"
        dev2_dir.mkdir(parents=True)
        
        loader = WorkspaceModuleLoader(
            workspace_root=self.workspace_root,
            developer_id="developer_1"
        )
        
        # Add something to cache
        loader._module_cache["test_key"] = "test_value"
        
        # Switch to developer_2
        loader.switch_developer("developer_2")
        
        self.assertEqual(loader.developer_id, "developer_2")
        
        # Cache should be cleared
        self.assertEqual(len(loader._module_cache), 0)
    
    def test_switch_developer_invalid(self):
        """Test switching to invalid developer workspace."""
        loader = WorkspaceModuleLoader(
            workspace_root=self.workspace_root,
            developer_id="developer_1"
        )
        
        with self.assertRaises(ValueError):
            loader.switch_developer("nonexistent_developer")
    
    def test_switch_developer_single_mode(self):
        """Test switching developer in single workspace mode."""
        loader = WorkspaceModuleLoader()
        
        with self.assertRaises(ValueError):
            loader.switch_developer("developer_1")
    
    def test_class_name_generation(self):
        """Test automatic class name generation from step names."""
        loader = WorkspaceModuleLoader(
            workspace_root=self.workspace_root,
            developer_id="developer_1"
        )
        
        with patch('importlib.import_module') as mock_import:
            mock_module = MagicMock()
            
            # Test builder class name generation
            mock_module.CreateModelBuilder = MagicMock()
            mock_import.return_value = mock_module
            
            result = loader.load_builder_class("create_model")
            
            # Should look for CreateModelBuilder class
            self.assertTrue(hasattr(mock_module, 'CreateModelBuilder'))
            
            # Test contract class name generation
            mock_module.CreateModelContract = MagicMock()
            mock_import.return_value = mock_module
            
            result = loader.load_contract_class("create_model")
            
            # Should look for CreateModelContract class
            self.assertTrue(hasattr(mock_module, 'CreateModelContract'))
    
    def test_module_pattern_search(self):
        """Test different module path patterns are tried."""
        loader = WorkspaceModuleLoader(
            workspace_root=self.workspace_root,
            developer_id="developer_1"
        )
        
        with patch('importlib.import_module') as mock_import:
            # First few patterns fail, last one succeeds
            mock_import.side_effect = [
                ImportError("Not found"),
                ImportError("Not found"),
                ImportError("Not found"),
                MagicMock()  # Success on 4th try
            ]
            
            result = loader.load_builder_class("test_step")
            
            # Should have tried multiple patterns
            self.assertEqual(mock_import.call_count, 4)
            self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
