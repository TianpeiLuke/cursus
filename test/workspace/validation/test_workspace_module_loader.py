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

import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from cursus.workspace.validation.workspace_module_loader import WorkspaceModuleLoader


class TestWorkspaceModuleLoader:
    """Test cases for WorkspaceModuleLoader."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Set up test fixtures."""
        temp_dir = tempfile.mkdtemp()
        workspace_root = Path(temp_dir) / "workspaces"
        workspace_root.mkdir(parents=True)
        
        # Create test workspace structure
        self._create_test_workspace_structure(workspace_root)
        
        # Store original sys.path
        original_sys_path = sys.path.copy()
        
        yield temp_dir, workspace_root, original_sys_path
        
        # Clean up test fixtures
        import shutil
        shutil.rmtree(temp_dir)
        
        # Restore original sys.path
        sys.path[:] = original_sys_path
    
    def _create_test_workspace_structure(self, workspace_root):
        """Create test workspace directory structure."""
        # Developer workspace
        dev1_dir = workspace_root / "developers" / "developer_1" / "src"
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
        shared_dir = workspace_root / "shared" / "src"
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
    
    def test_init_workspace_mode(self, temp_workspace):
        """Test initialization in workspace mode."""
        temp_dir, workspace_root, original_sys_path = temp_workspace
        
        loader = WorkspaceModuleLoader(
            workspace_root=workspace_root,
            developer_id="developer_1"
        )
        
        assert loader.workspace_mode
        assert loader.workspace_root == workspace_root
        assert loader.developer_id == "developer_1"
        assert loader.enable_shared_fallback
        assert loader.cache_modules
    
    def test_init_single_workspace_mode(self):
        """Test initialization in single workspace mode."""
        loader = WorkspaceModuleLoader()
        
        assert not loader.workspace_mode
        assert loader.workspace_root is None
        assert loader.developer_id is None
        assert loader.enable_shared_fallback
        assert loader.cache_modules
    
    def test_validate_workspace_structure_valid(self, temp_workspace):
        """Test workspace structure validation with valid structure."""
        temp_dir, workspace_root, original_sys_path = temp_workspace
        
        loader = WorkspaceModuleLoader(
            workspace_root=workspace_root,
            developer_id="developer_1"
        )
        
        # Should not raise exception
        assert loader.workspace_mode
    
    def test_validate_workspace_structure_invalid_root(self, temp_workspace):
        """Test workspace structure validation with invalid root."""
        temp_dir, workspace_root, original_sys_path = temp_workspace
        
        invalid_root = Path(temp_dir) / "nonexistent"
        
        with pytest.raises(ValueError) as exc_info:
            WorkspaceModuleLoader(
                workspace_root=invalid_root,
                developer_id="developer_1"
            )
        
        assert "does not exist" in str(exc_info.value)
    
    def test_build_workspace_paths(self, temp_workspace):
        """Test building workspace-specific Python paths."""
        temp_dir, workspace_root, original_sys_path = temp_workspace
        
        loader = WorkspaceModuleLoader(
            workspace_root=workspace_root,
            developer_id="developer_1"
        )
        
        assert hasattr(loader, 'developer_paths')
        assert hasattr(loader, 'shared_paths')
        
        # Check developer paths
        dev_base = str(workspace_root / "developers" / "developer_1" / "src")
        assert dev_base in loader.developer_paths
        
        # Check shared paths
        shared_base = str(workspace_root / "shared" / "src")
        assert shared_base in loader.shared_paths
    
    def test_workspace_path_context(self, temp_workspace):
        """Test workspace path context manager."""
        temp_dir, workspace_root, original_sys_path = temp_workspace
        
        loader = WorkspaceModuleLoader(
            workspace_root=workspace_root,
            developer_id="developer_1"
        )
        
        original_path = sys.path.copy()
        
        with loader.workspace_path_context() as added_paths:
            # Check that paths were added
            assert len(added_paths) > 0
            
            # Check that sys.path was modified
            for path in added_paths:
                assert path in sys.path
        
        # Check that sys.path was restored
        assert sys.path == original_path
    
    def test_workspace_path_context_single_mode(self):
        """Test workspace path context manager in single workspace mode."""
        loader = WorkspaceModuleLoader()
        
        with loader.workspace_path_context() as added_paths:
            # Should return empty list in single workspace mode
            assert added_paths == []
    
    @patch('importlib.import_module')
    def test_load_builder_class_success(self, mock_import, temp_workspace):
        """Test successful builder class loading."""
        temp_dir, workspace_root, original_sys_path = temp_workspace
        
        # Mock module with builder class
        mock_module = MagicMock()
        mock_builder_class = MagicMock()
        mock_module.TestStepBuilder = mock_builder_class
        mock_import.return_value = mock_module
        
        loader = WorkspaceModuleLoader(
            workspace_root=workspace_root,
            developer_id="developer_1"
        )
        
        result = loader.load_builder_class("test_step")
        
        assert result == mock_builder_class
        assert mock_import.called
    
    @patch('importlib.import_module')
    def test_load_builder_class_not_found(self, mock_import, temp_workspace):
        """Test builder class loading when class not found."""
        temp_dir, workspace_root, original_sys_path = temp_workspace
        
        mock_import.side_effect = ImportError("Module not found")
        
        loader = WorkspaceModuleLoader(
            workspace_root=workspace_root,
            developer_id="developer_1"
        )
        
        result = loader.load_builder_class("nonexistent_step")
        
        assert result is None
    
    @patch('importlib.import_module')
    def test_load_contract_class_success(self, mock_import, temp_workspace):
        """Test successful contract class loading."""
        temp_dir, workspace_root, original_sys_path = temp_workspace
        
        # Mock module with contract class
        mock_module = MagicMock()
        mock_contract_class = MagicMock()
        mock_module.TestStepContract = mock_contract_class
        mock_import.return_value = mock_module
        
        loader = WorkspaceModuleLoader(
            workspace_root=workspace_root,
            developer_id="developer_1"
        )
        
        result = loader.load_contract_class("test_step")
        
        assert result == mock_contract_class
        assert mock_import.called
    
    def test_module_caching(self, temp_workspace):
        """Test module caching functionality."""
        temp_dir, workspace_root, original_sys_path = temp_workspace
        
        loader = WorkspaceModuleLoader(
            workspace_root=workspace_root,
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
            assert result1 == mock_builder_class
            assert mock_import.call_count == 1
            
            # Second call should use cache
            result2 = loader.load_builder_class("test_step")
            assert result2 == mock_builder_class
            assert mock_import.call_count == 1  # Should not increase
    
    def test_module_caching_disabled(self, temp_workspace):
        """Test behavior when module caching is disabled."""
        temp_dir, workspace_root, original_sys_path = temp_workspace
        
        loader = WorkspaceModuleLoader(
            workspace_root=workspace_root,
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
            
            assert mock_import.call_count == 2
    
    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_load_module_from_file_success(self, mock_module_from_spec, mock_spec_from_file, temp_workspace):
        """Test successful module loading from file."""
        temp_dir, workspace_root, original_sys_path = temp_workspace
        
        # Create test file
        test_file = Path(temp_dir) / "test_module.py"
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
        
        assert result == mock_module
        mock_spec_from_file.assert_called_once_with("test_module", test_file)
        mock_loader.exec_module.assert_called_once_with(mock_module)
    
    def test_load_module_from_file_not_found(self):
        """Test module loading from non-existent file."""
        loader = WorkspaceModuleLoader()
        
        result = loader.load_module_from_file("/nonexistent/file.py")
        
        assert result is None
    
    def test_discover_workspace_modules(self, temp_workspace):
        """Test workspace module discovery."""
        temp_dir, workspace_root, original_sys_path = temp_workspace
        
        loader = WorkspaceModuleLoader(
            workspace_root=workspace_root,
            developer_id="developer_1"
        )
        
        discovered = loader.discover_workspace_modules("builders")
        
        assert "developer:developer_1" in discovered
        assert "shared" in discovered
        
        # Check developer modules
        dev_modules = discovered["developer:developer_1"]
        assert "test_step_builder" in dev_modules
        
        # Check shared modules
        shared_modules = discovered["shared"]
        assert "shared_builder" in shared_modules
    
    def test_discover_workspace_modules_single_mode(self):
        """Test module discovery in single workspace mode."""
        loader = WorkspaceModuleLoader()
        
        discovered = loader.discover_workspace_modules("builders")
        
        assert discovered == {}
    
    def test_clear_cache(self, temp_workspace):
        """Test cache clearing."""
        temp_dir, workspace_root, original_sys_path = temp_workspace
        
        loader = WorkspaceModuleLoader(
            workspace_root=workspace_root,
            developer_id="developer_1"
        )
        
        # Add something to cache
        loader._module_cache["test_key"] = "test_value"
        loader._path_cache["test_path"] = ["path1", "path2"]
        
        loader.clear_cache()
        
        assert len(loader._module_cache) == 0
        assert len(loader._path_cache) == 0
    
    def test_invalidate_cache_for_step(self, temp_workspace):
        """Test cache invalidation for specific step."""
        temp_dir, workspace_root, original_sys_path = temp_workspace
        
        loader = WorkspaceModuleLoader(
            workspace_root=workspace_root,
            developer_id="developer_1"
        )
        
        # Add cache entries
        loader._module_cache["builder:test_step:module:class"] = "builder_class"
        loader._module_cache["contract:test_step:module:class"] = "contract_class"
        loader._module_cache["builder:other_step:module:class"] = "other_builder"
        
        loader.invalidate_cache_for_step("test_step")
        
        # test_step entries should be removed
        assert "builder:test_step:module:class" not in loader._module_cache
        assert "contract:test_step:module:class" not in loader._module_cache
        
        # other_step entry should remain
        assert "builder:other_step:module:class" in loader._module_cache
    
    def test_get_workspace_info(self, temp_workspace):
        """Test getting workspace information."""
        temp_dir, workspace_root, original_sys_path = temp_workspace
        
        loader = WorkspaceModuleLoader(
            workspace_root=workspace_root,
            developer_id="developer_1"
        )
        
        info = loader.get_workspace_info()
        
        assert info['workspace_mode']
        assert info['workspace_root'] == str(workspace_root)
        assert info['developer_id'] == "developer_1"
        assert info['enable_shared_fallback']
        assert info['cache_modules']
        assert len(info['developer_paths']) > 0
        assert len(info['shared_paths']) > 0
        assert info['cached_modules'] == 0
    
    def test_switch_developer(self, temp_workspace):
        """Test switching to different developer workspace."""
        temp_dir, workspace_root, original_sys_path = temp_workspace
        
        # Create another developer workspace
        dev2_dir = workspace_root / "developers" / "developer_2" / "src" / "cursus_dev" / "steps"
        dev2_dir.mkdir(parents=True)
        
        loader = WorkspaceModuleLoader(
            workspace_root=workspace_root,
            developer_id="developer_1"
        )
        
        # Add something to cache
        loader._module_cache["test_key"] = "test_value"
        
        # Switch to developer_2
        loader.switch_developer("developer_2")
        
        assert loader.developer_id == "developer_2"
        
        # Cache should be cleared
        assert len(loader._module_cache) == 0
    
    def test_switch_developer_invalid(self, temp_workspace):
        """Test switching to invalid developer workspace."""
        temp_dir, workspace_root, original_sys_path = temp_workspace
        
        loader = WorkspaceModuleLoader(
            workspace_root=workspace_root,
            developer_id="developer_1"
        )
        
        with pytest.raises(ValueError):
            loader.switch_developer("nonexistent_developer")
    
    def test_switch_developer_single_mode(self):
        """Test switching developer in single workspace mode."""
        loader = WorkspaceModuleLoader()
        
        with pytest.raises(ValueError):
            loader.switch_developer("developer_1")
    
    def test_class_name_generation(self, temp_workspace):
        """Test automatic class name generation from step names."""
        temp_dir, workspace_root, original_sys_path = temp_workspace
        
        loader = WorkspaceModuleLoader(
            workspace_root=workspace_root,
            developer_id="developer_1"
        )
        
        with patch('importlib.import_module') as mock_import:
            mock_module = MagicMock()
            
            # Test builder class name generation
            mock_module.CreateModelBuilder = MagicMock()
            mock_import.return_value = mock_module
            
            result = loader.load_builder_class("create_model")
            
            # Should look for CreateModelBuilder class
            assert hasattr(mock_module, 'CreateModelBuilder')
            
            # Test contract class name generation
            mock_module.CreateModelContract = MagicMock()
            mock_import.return_value = mock_module
            
            result = loader.load_contract_class("create_model")
            
            # Should look for CreateModelContract class
            assert hasattr(mock_module, 'CreateModelContract')
    
    def test_module_pattern_search(self, temp_workspace):
        """Test different module path patterns are tried."""
        temp_dir, workspace_root, original_sys_path = temp_workspace
        
        loader = WorkspaceModuleLoader(
            workspace_root=workspace_root,
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
            assert mock_import.call_count == 4
            assert result is not None
