"""
Unit tests for WorkspaceManager

Tests workspace management functionality including:
- Workspace discovery and validation
- Developer workspace creation and management
- Configuration management
- Integration with file resolver and module loader
- Workspace structure validation
"""

import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from src.cursus.workspace.validation.workspace_manager import (
    WorkspaceManager,
    WorkspaceConfig,
    DeveloperInfo,
    WorkspaceInfo
)


class TestWorkspaceManager(unittest.TestCase):
    """Test cases for WorkspaceManager."""
    
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
        # Developer workspaces
        for dev_id in ["developer_1", "developer_2"]:
            dev_dir = self.workspace_root / "developers" / dev_id / "src" / "cursus_dev" / "steps"
            dev_dir.mkdir(parents=True)
            
            # Create module directories
            for module_type in ["builders", "contracts", "specs", "scripts", "configs"]:
                (dev_dir / module_type).mkdir()
                (dev_dir / module_type / "__init__.py").touch()
            
            # Create test files
            (dev_dir / "builders" / f"{dev_id}_builder.py").write_text(f"# {dev_id} builder")
            (dev_dir / "contracts" / f"{dev_id}_contract.py").write_text(f"# {dev_id} contract")
        
        # Shared workspace
        shared_dir = self.workspace_root / "shared" / "src" / "cursus_dev" / "steps"
        shared_dir.mkdir(parents=True)
        
        for module_type in ["builders", "contracts", "specs", "scripts", "configs"]:
            (shared_dir / module_type).mkdir()
            (shared_dir / module_type / "__init__.py").touch()
        
        # Create shared test files
        (shared_dir / "builders" / "shared_builder.py").write_text("# Shared builder")
        (shared_dir / "contracts" / "shared_contract.py").write_text("# Shared contract")
    
    def test_init_with_workspace_root(self):
        """Test initialization with workspace root."""
        manager = WorkspaceManager(
            workspace_root=self.workspace_root,
            auto_discover=True
        )
        
        self.assertEqual(manager.workspace_root, self.workspace_root)
        self.assertIsNotNone(manager.workspace_info)
        self.assertEqual(manager.workspace_info.total_developers, 2)
    
    def test_init_without_workspace_root(self):
        """Test initialization without workspace root."""
        manager = WorkspaceManager(auto_discover=False)
        
        self.assertIsNone(manager.workspace_root)
        self.assertIsNone(manager.workspace_info)
    
    def test_init_with_config_file(self):
        """Test initialization with config file."""
        # Create config file
        config_file = Path(self.temp_dir) / "workspace.json"
        config_data = {
            "workspace_root": str(self.workspace_root),
            "developer_id": "developer_1",
            "enable_shared_fallback": True,
            "cache_modules": True
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        manager = WorkspaceManager(
            config_file=config_file,
            auto_discover=False
        )
        
        self.assertIsNotNone(manager.config)
        self.assertEqual(manager.config.developer_id, "developer_1")
    
    def test_discover_workspaces(self):
        """Test workspace discovery."""
        manager = WorkspaceManager(auto_discover=False)
        
        workspace_info = manager.discover_workspaces(self.workspace_root)
        
        self.assertEqual(workspace_info.workspace_root, str(self.workspace_root))
        self.assertTrue(workspace_info.has_shared)
        self.assertEqual(workspace_info.total_developers, 2)
        self.assertGreater(workspace_info.total_modules, 0)
        
        # Check developer info
        dev_ids = [dev.developer_id for dev in workspace_info.developers]
        self.assertIn("developer_1", dev_ids)
        self.assertIn("developer_2", dev_ids)
    
    def test_discover_workspaces_invalid_root(self):
        """Test workspace discovery with invalid root."""
        manager = WorkspaceManager(auto_discover=False)
        
        invalid_root = Path(self.temp_dir) / "nonexistent"
        
        with self.assertRaises(ValueError):
            manager.discover_workspaces(invalid_root)
    
    def test_discover_developers(self):
        """Test developer workspace discovery."""
        manager = WorkspaceManager(auto_discover=False)
        developers_dir = self.workspace_root / "developers"
        
        developers = manager._discover_developers(developers_dir)
        
        self.assertEqual(len(developers), 2)
        
        dev1 = next(dev for dev in developers if dev.developer_id == "developer_1")
        self.assertTrue(dev1.has_builders)
        self.assertTrue(dev1.has_contracts)
        self.assertGreater(dev1.module_count, 0)
    
    def test_validate_workspace_structure_valid(self):
        """Test workspace structure validation with valid structure."""
        manager = WorkspaceManager(auto_discover=False)
        
        is_valid, issues = manager.validate_workspace_structure(self.workspace_root)
        
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)
    
    def test_validate_workspace_structure_invalid_root(self):
        """Test workspace structure validation with invalid root."""
        manager = WorkspaceManager(auto_discover=False)
        
        invalid_root = Path(self.temp_dir) / "nonexistent"
        is_valid, issues = manager.validate_workspace_structure(invalid_root)
        
        self.assertFalse(is_valid)
        self.assertGreater(len(issues), 0)
        self.assertIn("does not exist", issues[0])
    
    def test_validate_workspace_structure_missing_directories(self):
        """Test workspace structure validation with missing directories."""
        # Create empty workspace root
        empty_root = Path(self.temp_dir) / "empty_workspace"
        empty_root.mkdir()
        
        manager = WorkspaceManager(auto_discover=False)
        
        is_valid, issues = manager.validate_workspace_structure(empty_root)
        
        self.assertFalse(is_valid)
        self.assertGreater(len(issues), 0)
        self.assertIn("developers", issues[0])
    
    def test_validate_workspace_structure_strict(self):
        """Test strict workspace structure validation."""
        # Create workspace with empty developer directory
        empty_workspace = Path(self.temp_dir) / "empty_dev_workspace"
        empty_workspace.mkdir()
        (empty_workspace / "developers").mkdir()
        (empty_workspace / "shared").mkdir()
        
        manager = WorkspaceManager(auto_discover=False)
        
        is_valid, issues = manager.validate_workspace_structure(empty_workspace, strict=True)
        
        self.assertFalse(is_valid)
        self.assertGreater(len(issues), 0)
    
    def test_create_developer_workspace(self):
        """Test creating new developer workspace."""
        new_workspace_root = Path(self.temp_dir) / "new_workspaces"
        manager = WorkspaceManager(auto_discover=False)
        
        dev_workspace = manager.create_developer_workspace(
            "new_developer",
            workspace_root=new_workspace_root,
            create_structure=True
        )
        
        self.assertTrue(dev_workspace.exists())
        
        # Check structure was created
        cursus_dev_dir = dev_workspace / "src" / "cursus_dev" / "steps"
        self.assertTrue(cursus_dev_dir.exists())
        
        # Check module directories
        for module_type in ["builders", "contracts", "specs", "scripts", "configs"]:
            module_dir = cursus_dev_dir / module_type
            self.assertTrue(module_dir.exists())
            self.assertTrue((module_dir / "__init__.py").exists())
    
    def test_create_developer_workspace_existing(self):
        """Test creating developer workspace that already exists."""
        manager = WorkspaceManager(workspace_root=self.workspace_root, auto_discover=False)
        
        with self.assertRaises(ValueError) as context:
            manager.create_developer_workspace("developer_1")
        
        self.assertIn("already exists", str(context.exception))
    
    def test_create_shared_workspace(self):
        """Test creating shared workspace."""
        new_workspace_root = Path(self.temp_dir) / "new_workspaces"
        manager = WorkspaceManager(auto_discover=False)
        
        shared_workspace = manager.create_shared_workspace(
            workspace_root=new_workspace_root,
            create_structure=True
        )
        
        self.assertTrue(shared_workspace.exists())
        
        # Check structure was created
        cursus_dev_dir = shared_workspace / "src" / "cursus_dev" / "steps"
        self.assertTrue(cursus_dev_dir.exists())
        
        # Check module directories
        for module_type in ["builders", "contracts", "specs", "scripts", "configs"]:
            module_dir = cursus_dev_dir / module_type
            self.assertTrue(module_dir.exists())
            self.assertTrue((module_dir / "__init__.py").exists())
    
    def test_get_file_resolver(self):
        """Test getting workspace-aware file resolver."""
        manager = WorkspaceManager(workspace_root=self.workspace_root, auto_discover=False)
        
        resolver = manager.get_file_resolver("developer_1")
        
        self.assertIsNotNone(resolver)
        self.assertEqual(resolver.developer_id, "developer_1")
        self.assertEqual(resolver.workspace_root, self.workspace_root)
    
    def test_get_file_resolver_no_workspace_root(self):
        """Test getting file resolver without workspace root."""
        manager = WorkspaceManager(auto_discover=False)
        
        with self.assertRaises(ValueError):
            manager.get_file_resolver("developer_1")
    
    def test_get_module_loader(self):
        """Test getting workspace-aware module loader."""
        manager = WorkspaceManager(workspace_root=self.workspace_root, auto_discover=False)
        
        loader = manager.get_module_loader("developer_1")
        
        self.assertIsNotNone(loader)
        self.assertEqual(loader.developer_id, "developer_1")
        self.assertEqual(loader.workspace_root, self.workspace_root)
    
    def test_get_module_loader_no_workspace_root(self):
        """Test getting module loader without workspace root."""
        manager = WorkspaceManager(auto_discover=False)
        
        with self.assertRaises(ValueError):
            manager.get_module_loader("developer_1")
    
    def test_load_config_json(self):
        """Test loading JSON configuration."""
        config_file = Path(self.temp_dir) / "workspace.json"
        config_data = {
            "workspace_root": str(self.workspace_root),
            "developer_id": "developer_1",
            "enable_shared_fallback": True,
            "cache_modules": False,
            "validation_settings": {"strict": True}
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        manager = WorkspaceManager(auto_discover=False)
        config = manager.load_config(config_file)
        
        self.assertEqual(config.workspace_root, str(self.workspace_root))
        self.assertEqual(config.developer_id, "developer_1")
        self.assertTrue(config.enable_shared_fallback)
        self.assertFalse(config.cache_modules)
        self.assertEqual(config.validation_settings["strict"], True)
    
    @patch('yaml.safe_load')
    def test_load_config_yaml(self, mock_yaml_load):
        """Test loading YAML configuration."""
        config_file = Path(self.temp_dir) / "workspace.yaml"
        config_data = {
            "workspace_root": str(self.workspace_root),
            "developer_id": "developer_1"
        }
        
        mock_yaml_load.return_value = config_data
        
        # Create empty file (content doesn't matter due to mock)
        config_file.touch()
        
        manager = WorkspaceManager(auto_discover=False)
        config = manager.load_config(config_file)
        
        self.assertEqual(config.workspace_root, str(self.workspace_root))
        self.assertEqual(config.developer_id, "developer_1")
        mock_yaml_load.assert_called_once()
    
    def test_load_config_nonexistent(self):
        """Test loading non-existent configuration file."""
        manager = WorkspaceManager(auto_discover=False)
        
        nonexistent_file = Path(self.temp_dir) / "nonexistent.json"
        
        with self.assertRaises(ValueError):
            manager.load_config(nonexistent_file)
    
    def test_save_config_json(self):
        """Test saving JSON configuration."""
        config_file = Path(self.temp_dir) / "workspace.json"
        config = WorkspaceConfig(
            workspace_root=str(self.workspace_root),
            developer_id="developer_1",
            enable_shared_fallback=True
        )
        
        manager = WorkspaceManager(auto_discover=False)
        manager.save_config(config_file, config)
        
        self.assertTrue(config_file.exists())
        
        # Verify saved content
        with open(config_file, 'r') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data["workspace_root"], str(self.workspace_root))
        self.assertEqual(saved_data["developer_id"], "developer_1")
        self.assertTrue(saved_data["enable_shared_fallback"])
    
    @patch('yaml.safe_dump')
    def test_save_config_yaml(self, mock_yaml_dump):
        """Test saving YAML configuration."""
        config_file = Path(self.temp_dir) / "workspace.yaml"
        config = WorkspaceConfig(
            workspace_root=str(self.workspace_root),
            developer_id="developer_1"
        )
        
        manager = WorkspaceManager(auto_discover=False)
        manager.save_config(config_file, config)
        
        mock_yaml_dump.assert_called_once()
    
    def test_save_config_no_config(self):
        """Test saving configuration without config object."""
        manager = WorkspaceManager(auto_discover=False)
        config_file = Path(self.temp_dir) / "workspace.json"
        
        with self.assertRaises(ValueError):
            manager.save_config(config_file)
    
    def test_get_workspace_summary(self):
        """Test getting workspace summary."""
        manager = WorkspaceManager(
            workspace_root=self.workspace_root,
            auto_discover=True
        )
        
        summary = manager.get_workspace_summary()
        
        self.assertEqual(summary["workspace_root"], str(self.workspace_root))
        self.assertTrue(summary["has_shared"])
        self.assertEqual(summary["total_developers"], 2)
        self.assertGreater(summary["total_modules"], 0)
        self.assertEqual(len(summary["developers"]), 2)
        
        # Check developer summary
        dev1_summary = next(
            dev for dev in summary["developers"] 
            if dev["developer_id"] == "developer_1"
        )
        self.assertTrue(dev1_summary["has_builders"])
        self.assertTrue(dev1_summary["has_contracts"])
        self.assertGreater(dev1_summary["module_count"], 0)
    
    def test_get_workspace_summary_no_workspace(self):
        """Test getting workspace summary without workspace configured."""
        manager = WorkspaceManager(auto_discover=False)
        
        summary = manager.get_workspace_summary()
        
        self.assertIn("error", summary)
    
    def test_workspace_config_dataclass(self):
        """Test WorkspaceConfig dataclass."""
        config = WorkspaceConfig(
            workspace_root="/path/to/workspace",
            developer_id="test_dev"
        )
        
        self.assertEqual(config.workspace_root, "/path/to/workspace")
        self.assertEqual(config.developer_id, "test_dev")
        self.assertTrue(config.enable_shared_fallback)
        self.assertTrue(config.cache_modules)
        self.assertFalse(config.auto_create_structure)
        self.assertEqual(config.validation_settings, {})
    
    def test_developer_info_dataclass(self):
        """Test DeveloperInfo dataclass."""
        dev_info = DeveloperInfo(
            developer_id="test_dev",
            workspace_path="/path/to/dev",
            has_builders=True,
            module_count=5
        )
        
        self.assertEqual(dev_info.developer_id, "test_dev")
        self.assertEqual(dev_info.workspace_path, "/path/to/dev")
        self.assertTrue(dev_info.has_builders)
        self.assertFalse(dev_info.has_contracts)  # Default value
        self.assertEqual(dev_info.module_count, 5)
    
    def test_workspace_info_dataclass(self):
        """Test WorkspaceInfo dataclass."""
        workspace_info = WorkspaceInfo(
            workspace_root="/path/to/workspace",
            has_shared=True,
            total_developers=2
        )
        
        self.assertEqual(workspace_info.workspace_root, "/path/to/workspace")
        self.assertTrue(workspace_info.has_shared)
        self.assertEqual(workspace_info.total_developers, 2)
        self.assertEqual(workspace_info.total_modules, 0)  # Default value
        self.assertEqual(len(workspace_info.developers), 0)  # Default empty list
    
    def test_config_with_default_developer(self):
        """Test using config with default developer for file resolver and module loader."""
        config = WorkspaceConfig(
            workspace_root=str(self.workspace_root),
            developer_id="developer_1"
        )
        
        manager = WorkspaceManager(workspace_root=self.workspace_root, auto_discover=False)
        manager.config = config
        
        # Should use config's developer_id when none specified
        resolver = manager.get_file_resolver()
        self.assertEqual(resolver.developer_id, "developer_1")
        
        loader = manager.get_module_loader()
        self.assertEqual(loader.developer_id, "developer_1")
    
    def test_config_override_developer(self):
        """Test overriding config's default developer."""
        config = WorkspaceConfig(
            workspace_root=str(self.workspace_root),
            developer_id="developer_1"
        )
        
        manager = WorkspaceManager(workspace_root=self.workspace_root, auto_discover=False)
        manager.config = config
        
        # Should use specified developer_id over config default
        resolver = manager.get_file_resolver("developer_2")
        self.assertEqual(resolver.developer_id, "developer_2")
        
        loader = manager.get_module_loader("developer_2")
        self.assertEqual(loader.developer_id, "developer_2")
    
    def test_workspace_config_detection(self):
        """Test automatic workspace config file detection."""
        # Create workspace config file
        config_file = self.workspace_root / "workspace.json"
        config_data = {"workspace_root": str(self.workspace_root)}
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        manager = WorkspaceManager(auto_discover=False)
        workspace_info = manager.discover_workspaces(self.workspace_root)
        
        self.assertEqual(workspace_info.config_file, str(config_file))


if __name__ == '__main__':
    unittest.main()
