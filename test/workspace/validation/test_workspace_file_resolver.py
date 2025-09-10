"""
Unit tests for DeveloperWorkspaceFileResolver

Tests workspace-aware file resolution functionality including:
- Workspace mode detection and validation
- Developer workspace file discovery
- Shared workspace fallback behavior
- Backward compatibility with single workspace mode
- File resolution with different search patterns
"""

import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from cursus.workspace.validation.workspace_file_resolver import DeveloperWorkspaceFileResolver


class TestDeveloperWorkspaceFileResolver:
    """Test cases for DeveloperWorkspaceFileResolver."""
    
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
        # Developer workspace
        dev1_dir = workspace_root / "developers" / "developer_1" / "src" / "cursus_dev" / "steps"
        dev1_dir.mkdir(parents=True)
        
        # Create module directories
        for module_type in ["builders", "contracts", "specs", "scripts", "configs"]:
            (dev1_dir / module_type).mkdir()
            (dev1_dir / module_type / "__init__.py").touch()
        
        # Create test files with correct patterns matching cursus/steps
        (dev1_dir / "contracts" / "test_step_contract.py").write_text("# Test contract")
        (dev1_dir / "specs" / "test_step_spec.py").write_text("# Test spec")
        (dev1_dir / "builders" / "builder_test_step_step.py").write_text("# Test builder")
        (dev1_dir / "scripts" / "test_step.py").write_text("# Test script")
        (dev1_dir / "configs" / "config_test_step_step.py").write_text("# Test config")
        
        # Shared workspace
        shared_dir = workspace_root / "shared" / "src" / "cursus_dev" / "steps"
        shared_dir.mkdir(parents=True)
        
        for module_type in ["builders", "contracts", "specs", "scripts", "configs"]:
            (shared_dir / module_type).mkdir()
            (shared_dir / module_type / "__init__.py").touch()
        
        # Create shared test files with correct patterns
        (shared_dir / "contracts" / "shared_contract.py").write_text("# Shared contract")
        (shared_dir / "specs" / "shared_spec.py").write_text("# Shared spec")
        (shared_dir / "builders" / "builder_shared_step.py").write_text("# Shared builder")
    
    def test_init_workspace_mode(self, temp_workspace):
        """Test initialization in workspace mode."""
        temp_dir, workspace_root = temp_workspace
        
        resolver = DeveloperWorkspaceFileResolver(
            workspace_root=workspace_root,
            developer_id="developer_1"
        )
        
        assert resolver.workspace_mode
        assert resolver.workspace_root == workspace_root
        assert resolver.developer_id == "developer_1"
        assert resolver.enable_shared_fallback
    
    def test_init_single_workspace_mode(self):
        """Test initialization in single workspace mode."""
        resolver = DeveloperWorkspaceFileResolver()
        
        assert not resolver.workspace_mode
        assert resolver.workspace_root is None
        assert resolver.developer_id is None
        assert not resolver.enable_shared_fallback
    
    def test_validate_workspace_structure_valid(self, temp_workspace):
        """Test workspace structure validation with valid structure."""
        temp_dir, workspace_root = temp_workspace
        
        resolver = DeveloperWorkspaceFileResolver(
            workspace_root=workspace_root,
            developer_id="developer_1"
        )
        
        # Should not raise exception
        assert resolver.workspace_mode
    
    def test_validate_workspace_structure_invalid_root(self, temp_workspace):
        """Test workspace structure validation with invalid root."""
        temp_dir, workspace_root = temp_workspace
        
        invalid_root = Path(temp_dir) / "nonexistent"
        
        with pytest.raises(ValueError) as exc_info:
            DeveloperWorkspaceFileResolver(
                workspace_root=invalid_root,
                developer_id="developer_1"
            )
        
        assert "does not exist" in str(exc_info.value)
    
    def test_validate_workspace_structure_invalid_developer(self, temp_workspace):
        """Test workspace structure validation with invalid developer."""
        temp_dir, workspace_root = temp_workspace
        
        with pytest.raises(ValueError) as exc_info:
            DeveloperWorkspaceFileResolver(
                workspace_root=workspace_root,
                developer_id="nonexistent_developer"
            )
        
        assert "does not exist" in str(exc_info.value)
    
    def test_find_contract_file_developer_workspace(self, temp_workspace):
        """Test finding contract file in developer workspace."""
        temp_dir, workspace_root = temp_workspace
        
        resolver = DeveloperWorkspaceFileResolver(
            workspace_root=workspace_root,
            developer_id="developer_1"
        )
        
        result = resolver.find_contract_file("test_step")
        assert result is not None
        assert "test_step_contract.py" in result
        assert "developer_1" in result
    
    def test_find_spec_file_developer_workspace(self, temp_workspace):
        """Test finding spec file in developer workspace."""
        temp_dir, workspace_root = temp_workspace
        
        resolver = DeveloperWorkspaceFileResolver(
            workspace_root=workspace_root,
            developer_id="developer_1"
        )
        
        result = resolver.find_spec_file("test_step")
        assert result is not None
        assert "test_step_spec.py" in result
        assert "developer_1" in result
    
    def test_find_builder_file_developer_workspace(self, temp_workspace):
        """Test finding builder file in developer workspace."""
        temp_dir, workspace_root = temp_workspace
        
        resolver = DeveloperWorkspaceFileResolver(
            workspace_root=workspace_root,
            developer_id="developer_1"
        )
        
        result = resolver.find_builder_file("test_step")
        assert result is not None
        assert "builder_test_step_step.py" in result
        assert "developer_1" in result
    
    def test_find_script_file_developer_workspace(self, temp_workspace):
        """Test finding script file in developer workspace."""
        temp_dir, workspace_root = temp_workspace
        
        resolver = DeveloperWorkspaceFileResolver(
            workspace_root=workspace_root,
            developer_id="developer_1"
        )
        
        result = resolver.find_script_file("test_step")
        assert result is not None
        assert "test_step.py" in result
        assert "developer_1" in result
    
    def test_find_config_file_developer_workspace(self, temp_workspace):
        """Test finding config file in developer workspace."""
        temp_dir, workspace_root = temp_workspace
        
        resolver = DeveloperWorkspaceFileResolver(
            workspace_root=workspace_root,
            developer_id="developer_1"
        )
        
        result = resolver.find_config_file("test_step")
        assert result is not None
        assert "config_test_step_step.py" in result
        assert "developer_1" in result
    
    def test_shared_workspace_fallback(self, temp_workspace):
        """Test fallback to shared workspace when file not found in developer workspace."""
        temp_dir, workspace_root = temp_workspace
        
        resolver = DeveloperWorkspaceFileResolver(
            workspace_root=workspace_root,
            developer_id="developer_1",
            enable_shared_fallback=True
        )
        
        # Look for file that only exists in shared workspace
        result = resolver.find_contract_file("shared")
        assert result is not None
        assert "shared_contract.py" in result
        assert "shared" in result
    
    def test_shared_workspace_fallback_disabled(self, temp_workspace):
        """Test behavior when shared workspace fallback is disabled."""
        temp_dir, workspace_root = temp_workspace
        
        resolver = DeveloperWorkspaceFileResolver(
            workspace_root=workspace_root,
            developer_id="developer_1",
            enable_shared_fallback=False
        )
        
        # Look for file that only exists in shared workspace
        result = resolver.find_contract_file("shared")
        assert result is None
    
    def test_get_workspace_info(self, temp_workspace):
        """Test getting workspace information."""
        temp_dir, workspace_root = temp_workspace
        
        resolver = DeveloperWorkspaceFileResolver(
            workspace_root=workspace_root,
            developer_id="developer_1"
        )
        
        info = resolver.get_workspace_info()
        
        assert info['workspace_mode']
        assert info['workspace_root'] == str(workspace_root)
        assert info['developer_id'] == "developer_1"
        assert info['enable_shared_fallback']
        assert info['developer_workspace_exists']
        assert info['shared_workspace_exists']
    
    def test_list_available_developers(self, temp_workspace):
        """Test listing available developers."""
        temp_dir, workspace_root = temp_workspace
        
        # Create another developer workspace
        dev2_dir = workspace_root / "developers" / "developer_2" / "src" / "cursus_dev" / "steps"
        dev2_dir.mkdir(parents=True)
        
        resolver = DeveloperWorkspaceFileResolver(
            workspace_root=workspace_root,
            developer_id="developer_1"
        )
        
        developers = resolver.list_available_developers()
        
        assert "developer_1" in developers
        assert "developer_2" in developers
        assert len(developers) == 2
    
    def test_switch_developer(self, temp_workspace):
        """Test switching to different developer workspace."""
        temp_dir, workspace_root = temp_workspace
        
        # Create another developer workspace
        dev2_dir = workspace_root / "developers" / "developer_2" / "src" / "cursus_dev" / "steps"
        dev2_dir.mkdir(parents=True)
        (dev2_dir / "contracts").mkdir()
        (dev2_dir / "contracts" / "dev2_contract.py").write_text("# Dev2 contract")
        
        resolver = DeveloperWorkspaceFileResolver(
            workspace_root=workspace_root,
            developer_id="developer_1"
        )
        
        # Switch to developer_2
        resolver.switch_developer("developer_2")
        
        assert resolver.developer_id == "developer_2"
        
        # Should now find files in developer_2 workspace
        result = resolver.find_contract_file("dev2")
        assert result is not None
        assert "developer_2" in result
    
    def test_switch_developer_invalid(self, temp_workspace):
        """Test switching to invalid developer workspace."""
        temp_dir, workspace_root = temp_workspace
        
        resolver = DeveloperWorkspaceFileResolver(
            workspace_root=workspace_root,
            developer_id="developer_1"
        )
        
        with pytest.raises(ValueError):
            resolver.switch_developer("nonexistent_developer")
    
    def test_single_workspace_mode_compatibility(self):
        """Test backward compatibility with single workspace mode."""
        # Test single workspace mode without mocking - just verify it works
        resolver = DeveloperWorkspaceFileResolver()
        
        # Verify it's in single workspace mode
        assert not resolver.workspace_mode
        assert resolver.workspace_root is None
        assert resolver.developer_id is None
        assert not resolver.enable_shared_fallback
        
        # Test that it can call parent methods without error
        # (The actual file won't be found, but the method should work)
        result = resolver.find_contract_file("nonexistent_step")
        assert result is None  # Expected since file doesn't exist
    
    def test_file_not_found(self, temp_workspace):
        """Test behavior when file is not found."""
        temp_dir, workspace_root = temp_workspace
        
        resolver = DeveloperWorkspaceFileResolver(
            workspace_root=workspace_root,
            developer_id="developer_1"
        )
        
        result = resolver.find_contract_file("nonexistent_step")
        assert result is None
    
    def test_build_workspace_paths(self, temp_workspace):
        """Test building workspace-specific paths."""
        temp_dir, workspace_root = temp_workspace
        
        resolver = DeveloperWorkspaceFileResolver(
            workspace_root=workspace_root,
            developer_id="developer_1"
        )
        
        # Check that paths are set correctly
        assert hasattr(resolver, 'contracts_dir')
        assert hasattr(resolver, 'specs_dir')
        assert hasattr(resolver, 'builders_dir')
        assert hasattr(resolver, 'scripts_dir')
        assert hasattr(resolver, 'configs_dir')
        
        # Check shared paths when fallback is enabled
        assert hasattr(resolver, 'shared_contracts_dir')
        assert hasattr(resolver, 'shared_specs_dir')
        assert hasattr(resolver, 'shared_builders_dir')
        assert hasattr(resolver, 'shared_scripts_dir')
        assert hasattr(resolver, 'shared_configs_dir')
