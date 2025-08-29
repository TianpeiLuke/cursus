"""
Unit tests for DeveloperWorkspaceFileResolver

Tests workspace-aware file resolution functionality including:
- Workspace mode detection and validation
- Developer workspace file discovery
- Shared workspace fallback behavior
- Backward compatibility with single workspace mode
- File resolution with different search patterns
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.cursus.validation.workspace.workspace_file_resolver import DeveloperWorkspaceFileResolver


class TestDeveloperWorkspaceFileResolver(unittest.TestCase):
    """Test cases for DeveloperWorkspaceFileResolver."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_root = Path(self.temp_dir) / "workspaces"
        self.workspace_root.mkdir(parents=True)
        
        # Create test workspace structure
        self._create_test_workspace_structure()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_test_workspace_structure(self):
        """Create test workspace directory structure."""
        # Developer workspace
        dev1_dir = self.workspace_root / "developers" / "developer_1" / "src" / "cursus_dev" / "steps"
        dev1_dir.mkdir(parents=True)
        
        # Create module directories
        for module_type in ["builders", "contracts", "specs", "scripts", "configs"]:
            (dev1_dir / module_type).mkdir()
            (dev1_dir / module_type / "__init__.py").touch()
        
        # Create test files
        (dev1_dir / "contracts" / "test_step_contract.py").write_text("# Test contract")
        (dev1_dir / "specs" / "test_step_spec.json").write_text('{"test": "spec"}')
        (dev1_dir / "builders" / "test_step_builder.py").write_text("# Test builder")
        (dev1_dir / "scripts" / "test_step_script.py").write_text("# Test script")
        (dev1_dir / "configs" / "test_step_config.json").write_text('{"test": "config"}')
        
        # Shared workspace
        shared_dir = self.workspace_root / "shared" / "src" / "cursus_dev" / "steps"
        shared_dir.mkdir(parents=True)
        
        for module_type in ["builders", "contracts", "specs", "scripts", "configs"]:
            (shared_dir / module_type).mkdir()
            (shared_dir / module_type / "__init__.py").touch()
        
        # Create shared test files
        (shared_dir / "contracts" / "shared_contract.py").write_text("# Shared contract")
        (shared_dir / "specs" / "shared_spec.json").write_text('{"shared": "spec"}')
        (shared_dir / "builders" / "shared_builder.py").write_text("# Shared builder")
    
    def test_init_workspace_mode(self):
        """Test initialization in workspace mode."""
        resolver = DeveloperWorkspaceFileResolver(
            workspace_root=self.workspace_root,
            developer_id="developer_1"
        )
        
        self.assertTrue(resolver.workspace_mode)
        self.assertEqual(resolver.workspace_root, self.workspace_root)
        self.assertEqual(resolver.developer_id, "developer_1")
        self.assertTrue(resolver.enable_shared_fallback)
    
    def test_init_single_workspace_mode(self):
        """Test initialization in single workspace mode."""
        resolver = DeveloperWorkspaceFileResolver()
        
        self.assertFalse(resolver.workspace_mode)
        self.assertIsNone(resolver.workspace_root)
        self.assertIsNone(resolver.developer_id)
        self.assertFalse(resolver.enable_shared_fallback)
    
    def test_validate_workspace_structure_valid(self):
        """Test workspace structure validation with valid structure."""
        resolver = DeveloperWorkspaceFileResolver(
            workspace_root=self.workspace_root,
            developer_id="developer_1"
        )
        
        # Should not raise exception
        self.assertTrue(resolver.workspace_mode)
    
    def test_validate_workspace_structure_invalid_root(self):
        """Test workspace structure validation with invalid root."""
        invalid_root = Path(self.temp_dir) / "nonexistent"
        
        with self.assertRaises(ValueError) as context:
            DeveloperWorkspaceFileResolver(
                workspace_root=invalid_root,
                developer_id="developer_1"
            )
        
        self.assertIn("does not exist", str(context.exception))
    
    def test_validate_workspace_structure_invalid_developer(self):
        """Test workspace structure validation with invalid developer."""
        with self.assertRaises(ValueError) as context:
            DeveloperWorkspaceFileResolver(
                workspace_root=self.workspace_root,
                developer_id="nonexistent_developer"
            )
        
        self.assertIn("does not exist", str(context.exception))
    
    def test_find_contract_file_developer_workspace(self):
        """Test finding contract file in developer workspace."""
        resolver = DeveloperWorkspaceFileResolver(
            workspace_root=self.workspace_root,
            developer_id="developer_1"
        )
        
        result = resolver.find_contract_file("test_step")
        self.assertIsNotNone(result)
        self.assertIn("test_step_contract.py", result)
        self.assertIn("developer_1", result)
    
    def test_find_spec_file_developer_workspace(self):
        """Test finding spec file in developer workspace."""
        resolver = DeveloperWorkspaceFileResolver(
            workspace_root=self.workspace_root,
            developer_id="developer_1"
        )
        
        result = resolver.find_spec_file("test_step")
        self.assertIsNotNone(result)
        self.assertIn("test_step_spec.json", result)
        self.assertIn("developer_1", result)
    
    def test_find_builder_file_developer_workspace(self):
        """Test finding builder file in developer workspace."""
        resolver = DeveloperWorkspaceFileResolver(
            workspace_root=self.workspace_root,
            developer_id="developer_1"
        )
        
        result = resolver.find_builder_file("test_step")
        self.assertIsNotNone(result)
        self.assertIn("test_step_builder.py", result)
        self.assertIn("developer_1", result)
    
    def test_find_script_file_developer_workspace(self):
        """Test finding script file in developer workspace."""
        resolver = DeveloperWorkspaceFileResolver(
            workspace_root=self.workspace_root,
            developer_id="developer_1"
        )
        
        result = resolver.find_script_file("test_step")
        self.assertIsNotNone(result)
        self.assertIn("test_step_script.py", result)
        self.assertIn("developer_1", result)
    
    def test_find_config_file_developer_workspace(self):
        """Test finding config file in developer workspace."""
        resolver = DeveloperWorkspaceFileResolver(
            workspace_root=self.workspace_root,
            developer_id="developer_1"
        )
        
        result = resolver.find_config_file("test_step")
        self.assertIsNotNone(result)
        self.assertIn("test_step_config.json", result)
        self.assertIn("developer_1", result)
    
    def test_shared_workspace_fallback(self):
        """Test fallback to shared workspace when file not found in developer workspace."""
        resolver = DeveloperWorkspaceFileResolver(
            workspace_root=self.workspace_root,
            developer_id="developer_1",
            enable_shared_fallback=True
        )
        
        # Look for file that only exists in shared workspace
        result = resolver.find_contract_file("shared")
        self.assertIsNotNone(result)
        self.assertIn("shared_contract.py", result)
        self.assertIn("shared", result)
    
    def test_shared_workspace_fallback_disabled(self):
        """Test behavior when shared workspace fallback is disabled."""
        resolver = DeveloperWorkspaceFileResolver(
            workspace_root=self.workspace_root,
            developer_id="developer_1",
            enable_shared_fallback=False
        )
        
        # Look for file that only exists in shared workspace
        result = resolver.find_contract_file("shared")
        self.assertIsNone(result)
    
    def test_get_workspace_info(self):
        """Test getting workspace information."""
        resolver = DeveloperWorkspaceFileResolver(
            workspace_root=self.workspace_root,
            developer_id="developer_1"
        )
        
        info = resolver.get_workspace_info()
        
        self.assertTrue(info['workspace_mode'])
        self.assertEqual(info['workspace_root'], str(self.workspace_root))
        self.assertEqual(info['developer_id'], "developer_1")
        self.assertTrue(info['enable_shared_fallback'])
        self.assertTrue(info['developer_workspace_exists'])
        self.assertTrue(info['shared_workspace_exists'])
    
    def test_list_available_developers(self):
        """Test listing available developers."""
        # Create another developer workspace
        dev2_dir = self.workspace_root / "developers" / "developer_2" / "src" / "cursus_dev" / "steps"
        dev2_dir.mkdir(parents=True)
        
        resolver = DeveloperWorkspaceFileResolver(
            workspace_root=self.workspace_root,
            developer_id="developer_1"
        )
        
        developers = resolver.list_available_developers()
        
        self.assertIn("developer_1", developers)
        self.assertIn("developer_2", developers)
        self.assertEqual(len(developers), 2)
    
    def test_switch_developer(self):
        """Test switching to different developer workspace."""
        # Create another developer workspace
        dev2_dir = self.workspace_root / "developers" / "developer_2" / "src" / "cursus_dev" / "steps"
        dev2_dir.mkdir(parents=True)
        (dev2_dir / "contracts").mkdir()
        (dev2_dir / "contracts" / "dev2_contract.py").write_text("# Dev2 contract")
        
        resolver = DeveloperWorkspaceFileResolver(
            workspace_root=self.workspace_root,
            developer_id="developer_1"
        )
        
        # Switch to developer_2
        resolver.switch_developer("developer_2")
        
        self.assertEqual(resolver.developer_id, "developer_2")
        
        # Should now find files in developer_2 workspace
        result = resolver.find_contract_file("dev2")
        self.assertIsNotNone(result)
        self.assertIn("developer_2", result)
    
    def test_switch_developer_invalid(self):
        """Test switching to invalid developer workspace."""
        resolver = DeveloperWorkspaceFileResolver(
            workspace_root=self.workspace_root,
            developer_id="developer_1"
        )
        
        with self.assertRaises(ValueError):
            resolver.switch_developer("nonexistent_developer")
    
    def test_single_workspace_mode_compatibility(self):
        """Test backward compatibility with single workspace mode."""
        # Mock parent class behavior
        with patch('src.cursus.validation.workspace.workspace_file_resolver.FlexibleFileResolver') as mock_parent:
            mock_instance = MagicMock()
            mock_parent.return_value = mock_instance
            mock_instance.find_contract_file.return_value = "/path/to/contract.py"
            
            resolver = DeveloperWorkspaceFileResolver()
            
            result = resolver.find_contract_file("test_step")
            
            # Should delegate to parent class
            mock_instance.find_contract_file.assert_called_once_with("test_step", None)
            self.assertEqual(result, "/path/to/contract.py")
    
    def test_file_not_found(self):
        """Test behavior when file is not found."""
        resolver = DeveloperWorkspaceFileResolver(
            workspace_root=self.workspace_root,
            developer_id="developer_1"
        )
        
        result = resolver.find_contract_file("nonexistent_step")
        self.assertIsNone(result)
    
    def test_build_workspace_paths(self):
        """Test building workspace-specific paths."""
        resolver = DeveloperWorkspaceFileResolver(
            workspace_root=self.workspace_root,
            developer_id="developer_1"
        )
        
        # Check that paths are set correctly
        self.assertTrue(hasattr(resolver, 'contracts_dir'))
        self.assertTrue(hasattr(resolver, 'specs_dir'))
        self.assertTrue(hasattr(resolver, 'builders_dir'))
        self.assertTrue(hasattr(resolver, 'scripts_dir'))
        self.assertTrue(hasattr(resolver, 'configs_dir'))
        
        # Check shared paths when fallback is enabled
        self.assertTrue(hasattr(resolver, 'shared_contracts_dir'))
        self.assertTrue(hasattr(resolver, 'shared_specs_dir'))
        self.assertTrue(hasattr(resolver, 'shared_builders_dir'))
        self.assertTrue(hasattr(resolver, 'shared_scripts_dir'))
        self.assertTrue(hasattr(resolver, 'shared_configs_dir'))


if __name__ == '__main__':
    unittest.main()
