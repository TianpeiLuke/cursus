"""
Unit tests for WorkspaceManager

Tests workspace management functionality including:
- Workspace discovery and validation
- Developer workspace creation and management
- Configuration management
- Integration with file resolver and module loader
- Workspace structure validation
"""

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from cursus.workspace.validation.workspace_manager import (
    WorkspaceManager,
    WorkspaceConfig,
    DeveloperInfo,
    WorkspaceInfo
)


class TestWorkspaceManager:
    """Test cases for WorkspaceManager."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Set up test fixtures."""
        temp_dir = tempfile.mkdtemp()
        workspace_root = Path(temp_dir) / "workspaces"
        workspace_root.mkdir(parents=True)
        
        # Create test workspace structure
        self._create_test_workspace_structure(workspace_root)
        
        yield temp_dir, workspace_root
        
        # Clean up test fixtures
        import shutil
        shutil.rmtree(temp_dir)
    
    def _create_test_workspace_structure(self, workspace_root):
        """Create test workspace directory structure."""
        # Developer workspaces
        for dev_id in ["developer_1", "developer_2"]:
            dev_dir = workspace_root / "developers" / dev_id / "src" / "cursus_dev" / "steps"
            dev_dir.mkdir(parents=True)
            
            # Create module directories
            for module_type in ["builders", "contracts", "specs", "scripts", "configs"]:
                (dev_dir / module_type).mkdir()
                (dev_dir / module_type / "__init__.py").touch()
            
            # Create test files
            (dev_dir / "builders" / f"{dev_id}_builder.py").write_text(f"# {dev_id} builder")
            (dev_dir / "contracts" / f"{dev_id}_contract.py").write_text(f"# {dev_id} contract")
        
        # Shared workspace
        shared_dir = workspace_root / "shared" / "src" / "cursus_dev" / "steps"
        shared_dir.mkdir(parents=True)
        
        for module_type in ["builders", "contracts", "specs", "scripts", "configs"]:
            (shared_dir / module_type).mkdir()
            (shared_dir / module_type / "__init__.py").touch()
        
        # Create shared test files
        (shared_dir / "builders" / "shared_builder.py").write_text("# Shared builder")
        (shared_dir / "contracts" / "shared_contract.py").write_text("# Shared contract")
    
    def test_init_with_workspace_root(self, temp_workspace):
        """Test initialization with workspace root."""
        temp_dir, workspace_root = temp_workspace
        
        manager = WorkspaceManager(
            workspace_root=workspace_root,
            auto_discover=True
        )
        
        assert manager.workspace_root == workspace_root
        assert manager.workspace_info is not None
        assert manager.workspace_info.total_developers == 2
    
    def test_init_without_workspace_root(self):
        """Test initialization without workspace root."""
        manager = WorkspaceManager(auto_discover=False)
        
        assert manager.workspace_root is None
        assert manager.workspace_info is None
    
    def test_init_with_config_file(self, temp_workspace):
        """Test initialization with config file."""
        temp_dir, workspace_root = temp_workspace
        
        # Create config file
        config_file = Path(temp_dir) / "workspace.json"
        config_data = {
            "workspace_root": str(workspace_root),
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
        
        assert manager.config is not None
        assert manager.config.developer_id == "developer_1"
    
    def test_discover_workspaces(self, temp_workspace):
        """Test workspace discovery."""
        temp_dir, workspace_root = temp_workspace
        
        manager = WorkspaceManager(auto_discover=False)
        
        workspace_info = manager.discover_workspaces(workspace_root)
        
        assert workspace_info.workspace_root == str(workspace_root)
        assert workspace_info.has_shared
        assert workspace_info.total_developers == 2
        assert workspace_info.total_modules > 0
        
        # Check developer info
        dev_ids = [dev.developer_id for dev in workspace_info.developers]
        assert "developer_1" in dev_ids
        assert "developer_2" in dev_ids
    
    def test_discover_workspaces_invalid_root(self, temp_workspace):
        """Test workspace discovery with invalid root."""
        temp_dir, workspace_root = temp_workspace
        
        manager = WorkspaceManager(auto_discover=False)
        
        invalid_root = Path(temp_dir) / "nonexistent"
        
        with pytest.raises(ValueError):
            manager.discover_workspaces(invalid_root)
    
    def test_discover_developers(self, temp_workspace):
        """Test developer workspace discovery."""
        temp_dir, workspace_root = temp_workspace
        
        manager = WorkspaceManager(auto_discover=False)
        developers_dir = workspace_root / "developers"
        
        developers = manager._discover_developers(developers_dir)
        
        assert len(developers) == 2
        
        dev1 = next(dev for dev in developers if dev.developer_id == "developer_1")
        assert dev1.has_builders
        assert dev1.has_contracts
        assert dev1.module_count > 0
    
    def test_validate_workspace_structure_valid(self, temp_workspace):
        """Test workspace structure validation with valid structure."""
        temp_dir, workspace_root = temp_workspace
        
        manager = WorkspaceManager(auto_discover=False)
        
        is_valid, issues = manager.validate_workspace_structure(workspace_root)
        
        assert is_valid
        assert len(issues) == 0
    
    def test_validate_workspace_structure_invalid_root(self, temp_workspace):
        """Test workspace structure validation with invalid root."""
        temp_dir, workspace_root = temp_workspace
        
        manager = WorkspaceManager(auto_discover=False)
        
        invalid_root = Path(temp_dir) / "nonexistent"
        is_valid, issues = manager.validate_workspace_structure(invalid_root)
        
        assert not is_valid
        assert len(issues) > 0
        assert "does not exist" in issues[0]
    
    def test_validate_workspace_structure_missing_directories(self, temp_workspace):
        """Test workspace structure validation with missing directories."""
        temp_dir, workspace_root = temp_workspace
        
        # Create empty workspace root
        empty_root = Path(temp_dir) / "empty_workspace"
        empty_root.mkdir()
        
        manager = WorkspaceManager(auto_discover=False)
        
        is_valid, issues = manager.validate_workspace_structure(empty_root)
        
        assert not is_valid
        assert len(issues) > 0
        assert "developers" in issues[0]
    
    def test_validate_workspace_structure_strict(self, temp_workspace):
        """Test strict workspace structure validation."""
        temp_dir, workspace_root = temp_workspace
        
        # Create workspace with empty developer directory
        empty_workspace = Path(temp_dir) / "empty_dev_workspace"
        empty_workspace.mkdir()
        (empty_workspace / "developers").mkdir()
        (empty_workspace / "shared").mkdir()
        
        manager = WorkspaceManager(auto_discover=False)
        
        is_valid, issues = manager.validate_workspace_structure(empty_workspace, strict=True)
        
        assert not is_valid
        assert len(issues) > 0
    
    def test_create_developer_workspace(self, temp_workspace):
        """Test creating new developer workspace."""
        temp_dir, workspace_root = temp_workspace
        
        new_workspace_root = Path(temp_dir) / "new_workspaces"
        manager = WorkspaceManager(auto_discover=False)
        
        dev_workspace = manager.create_developer_workspace(
            "new_developer",
            workspace_root=new_workspace_root,
            create_structure=True
        )
        
        assert dev_workspace.exists()
        
        # Check structure was created
        cursus_dev_dir = dev_workspace / "src" / "cursus_dev" / "steps"
        assert cursus_dev_dir.exists()
        
        # Check module directories
        for module_type in ["builders", "contracts", "specs", "scripts", "configs"]:
            module_dir = cursus_dev_dir / module_type
            assert module_dir.exists()
            assert (module_dir / "__init__.py").exists()
    
    def test_create_developer_workspace_existing(self, temp_workspace):
        """Test creating developer workspace that already exists."""
        temp_dir, workspace_root = temp_workspace
        
        manager = WorkspaceManager(workspace_root=workspace_root, auto_discover=False)
        
        with pytest.raises(ValueError) as exc_info:
            manager.create_developer_workspace("developer_1")
        
        assert "already exists" in str(exc_info.value)
    
    def test_create_shared_workspace(self, temp_workspace):
        """Test creating shared workspace."""
        temp_dir, workspace_root = temp_workspace
        
        new_workspace_root = Path(temp_dir) / "new_workspaces"
        manager = WorkspaceManager(auto_discover=False)
        
        shared_workspace = manager.create_shared_workspace(
            workspace_root=new_workspace_root,
            create_structure=True
        )
        
        assert shared_workspace.exists()
        
        # Check structure was created
        cursus_dev_dir = shared_workspace / "src" / "cursus_dev" / "steps"
        assert cursus_dev_dir.exists()
        
        # Check module directories
        for module_type in ["builders", "contracts", "specs", "scripts", "configs"]:
            module_dir = cursus_dev_dir / module_type
            assert module_dir.exists()
            assert (module_dir / "__init__.py").exists()
    
    def test_get_file_resolver(self, temp_workspace):
        """Test getting workspace-aware file resolver."""
        temp_dir, workspace_root = temp_workspace
        
        manager = WorkspaceManager(workspace_root=workspace_root, auto_discover=False)
        
        resolver = manager.get_file_resolver("developer_1")
        
        assert resolver is not None
        assert resolver.developer_id == "developer_1"
        assert resolver.workspace_root == workspace_root
    
    def test_get_file_resolver_no_workspace_root(self):
        """Test getting file resolver without workspace root."""
        manager = WorkspaceManager(auto_discover=False)
        
        with pytest.raises(ValueError):
            manager.get_file_resolver("developer_1")
    
    def test_get_module_loader(self, temp_workspace):
        """Test getting workspace-aware module loader."""
        temp_dir, workspace_root = temp_workspace
        
        manager = WorkspaceManager(workspace_root=workspace_root, auto_discover=False)
        
        loader = manager.get_module_loader("developer_1")
        
        assert loader is not None
        assert loader.developer_id == "developer_1"
        assert loader.workspace_root == workspace_root
    
    def test_get_module_loader_no_workspace_root(self):
        """Test getting module loader without workspace root."""
        manager = WorkspaceManager(auto_discover=False)
        
        with pytest.raises(ValueError):
            manager.get_module_loader("developer_1")
    
    def test_load_config_json(self, temp_workspace):
        """Test loading JSON configuration."""
        temp_dir, workspace_root = temp_workspace
        
        config_file = Path(temp_dir) / "workspace.json"
        config_data = {
            "workspace_root": str(workspace_root),
            "developer_id": "developer_1",
            "enable_shared_fallback": True,
            "cache_modules": False,
            "validation_settings": {"strict": True}
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        manager = WorkspaceManager(auto_discover=False)
        config = manager.load_config(config_file)
        
        assert config.workspace_root == str(workspace_root)
        assert config.developer_id == "developer_1"
        assert config.enable_shared_fallback
        assert not config.cache_modules
        assert config.validation_settings["strict"]
    
    @patch('yaml.safe_load')
    def test_load_config_yaml(self, mock_yaml_load, temp_workspace):
        """Test loading YAML configuration."""
        temp_dir, workspace_root = temp_workspace
        
        config_file = Path(temp_dir) / "workspace.yaml"
        config_data = {
            "workspace_root": str(workspace_root),
            "developer_id": "developer_1"
        }
        
        mock_yaml_load.return_value = config_data
        
        # Create empty file (content doesn't matter due to mock)
        config_file.touch()
        
        manager = WorkspaceManager(auto_discover=False)
        config = manager.load_config(config_file)
        
        assert config.workspace_root == str(workspace_root)
        assert config.developer_id == "developer_1"
        mock_yaml_load.assert_called_once()
    
    def test_load_config_nonexistent(self, temp_workspace):
        """Test loading non-existent configuration file."""
        temp_dir, workspace_root = temp_workspace
        
        manager = WorkspaceManager(auto_discover=False)
        
        nonexistent_file = Path(temp_dir) / "nonexistent.json"
        
        with pytest.raises(ValueError):
            manager.load_config(nonexistent_file)
    
    def test_save_config_json(self, temp_workspace):
        """Test saving JSON configuration."""
        temp_dir, workspace_root = temp_workspace
        
        config_file = Path(temp_dir) / "workspace.json"
        config = WorkspaceConfig(
            workspace_root=str(workspace_root),
            developer_id="developer_1",
            enable_shared_fallback=True
        )
        
        manager = WorkspaceManager(auto_discover=False)
        manager.save_config(config_file, config)
        
        assert config_file.exists()
        
        # Verify saved content
        with open(config_file, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data["workspace_root"] == str(workspace_root)
        assert saved_data["developer_id"] == "developer_1"
        assert saved_data["enable_shared_fallback"]
    
    @patch('yaml.safe_dump')
    def test_save_config_yaml(self, mock_yaml_dump, temp_workspace):
        """Test saving YAML configuration."""
        temp_dir, workspace_root = temp_workspace
        
        config_file = Path(temp_dir) / "workspace.yaml"
        config = WorkspaceConfig(
            workspace_root=str(workspace_root),
            developer_id="developer_1"
        )
        
        manager = WorkspaceManager(auto_discover=False)
        manager.save_config(config_file, config)
        
        mock_yaml_dump.assert_called_once()
    
    def test_save_config_no_config(self, temp_workspace):
        """Test saving configuration without config object."""
        temp_dir, workspace_root = temp_workspace
        
        manager = WorkspaceManager(auto_discover=False)
        config_file = Path(temp_dir) / "workspace.json"
        
        with pytest.raises(ValueError):
            manager.save_config(config_file)
    
    def test_get_workspace_summary(self, temp_workspace):
        """Test getting workspace summary."""
        temp_dir, workspace_root = temp_workspace
        
        manager = WorkspaceManager(
            workspace_root=workspace_root,
            auto_discover=True
        )
        
        summary = manager.get_workspace_summary()
        
        assert summary["workspace_root"] == str(workspace_root)
        assert summary["has_shared"]
        assert summary["total_developers"] == 2
        assert summary["total_modules"] > 0
        assert len(summary["developers"]) == 2
        
        # Check developer summary
        dev1_summary = next(
            dev for dev in summary["developers"] 
            if dev["developer_id"] == "developer_1"
        )
        assert dev1_summary["has_builders"]
        assert dev1_summary["has_contracts"]
        assert dev1_summary["module_count"] > 0
    
    def test_get_workspace_summary_no_workspace(self):
        """Test getting workspace summary without workspace configured."""
        manager = WorkspaceManager(auto_discover=False)
        
        summary = manager.get_workspace_summary()
        
        assert "error" in summary
    
    def test_workspace_config_dataclass(self):
        """Test WorkspaceConfig dataclass."""
        config = WorkspaceConfig(
            workspace_root="/path/to/workspace",
            developer_id="test_dev"
        )
        
        assert config.workspace_root == "/path/to/workspace"
        assert config.developer_id == "test_dev"
        assert config.enable_shared_fallback
        assert config.cache_modules
        assert not config.auto_create_structure
        assert config.validation_settings == {}
    
    def test_developer_info_dataclass(self):
        """Test DeveloperInfo dataclass."""
        dev_info = DeveloperInfo(
            developer_id="test_dev",
            workspace_path="/path/to/dev",
            has_builders=True,
            module_count=5
        )
        
        assert dev_info.developer_id == "test_dev"
        assert dev_info.workspace_path == "/path/to/dev"
        assert dev_info.has_builders
        assert not dev_info.has_contracts  # Default value
        assert dev_info.module_count == 5
    
    def test_workspace_info_dataclass(self):
        """Test WorkspaceInfo dataclass."""
        workspace_info = WorkspaceInfo(
            workspace_root="/path/to/workspace",
            has_shared=True,
            total_developers=2
        )
        
        assert workspace_info.workspace_root == "/path/to/workspace"
        assert workspace_info.has_shared
        assert workspace_info.total_developers == 2
        assert workspace_info.total_modules == 0  # Default value
        assert len(workspace_info.developers) == 0  # Default empty list
    
    def test_config_with_default_developer(self, temp_workspace):
        """Test using config with default developer for file resolver and module loader."""
        temp_dir, workspace_root = temp_workspace
        
        config = WorkspaceConfig(
            workspace_root=str(workspace_root),
            developer_id="developer_1"
        )
        
        manager = WorkspaceManager(workspace_root=workspace_root, auto_discover=False)
        manager.config = config
        
        # Should use config's developer_id when none specified
        resolver = manager.get_file_resolver()
        assert resolver.developer_id == "developer_1"
        
        loader = manager.get_module_loader()
        assert loader.developer_id == "developer_1"
    
    def test_config_override_developer(self, temp_workspace):
        """Test overriding config's default developer."""
        temp_dir, workspace_root = temp_workspace
        
        config = WorkspaceConfig(
            workspace_root=str(workspace_root),
            developer_id="developer_1"
        )
        
        manager = WorkspaceManager(workspace_root=workspace_root, auto_discover=False)
        manager.config = config
        
        # Should use specified developer_id over config default
        resolver = manager.get_file_resolver("developer_2")
        assert resolver.developer_id == "developer_2"
        
        loader = manager.get_module_loader("developer_2")
        assert loader.developer_id == "developer_2"
    
    def test_workspace_config_detection(self, temp_workspace):
        """Test automatic workspace config file detection."""
        temp_dir, workspace_root = temp_workspace
        
        # Create workspace config file
        config_file = workspace_root / "workspace.json"
        config_data = {"workspace_root": str(workspace_root)}
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        manager = WorkspaceManager(auto_discover=False)
        workspace_info = manager.discover_workspaces(workspace_root)
        
        assert workspace_info.config_file == str(config_file)
