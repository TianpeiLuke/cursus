"""
Unit tests for workspace component registry.

Tests the WorkspaceComponentRegistry for component discovery, caching, and management.
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.cursus.core.workspace.registry import WorkspaceComponentRegistry
from src.cursus.core.workspace.config import WorkspaceStepDefinition, WorkspacePipelineDefinition


class TestWorkspaceComponentRegistry:
    """Test cases for WorkspaceComponentRegistry."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir)
            
            # Create workspace structure
            dev1_path = workspace_path / "dev1"
            dev2_path = workspace_path / "dev2"
            
            for dev_path in [dev1_path, dev2_path]:
                dev_path.mkdir()
                (dev_path / "builders").mkdir()
                (dev_path / "configs").mkdir()
                (dev_path / "contracts").mkdir()
                (dev_path / "specs").mkdir()
                (dev_path / "scripts").mkdir()
                
                # Create sample files
                (dev_path / "builders" / "__init__.py").touch()
                (dev_path / "builders" / "test_builder.py").touch()
                (dev_path / "contracts" / "__init__.py").touch()
                (dev_path / "contracts" / "test_contract.py").touch()
                (dev_path / "specs" / "__init__.py").touch()
                (dev_path / "specs" / "test_spec.py").touch()
                (dev_path / "scripts" / "__init__.py").touch()
                (dev_path / "scripts" / "test_script.py").touch()
            
            yield str(workspace_path)
    
    @pytest.fixture
    def mock_workspace_manager(self):
        """Create a mock workspace manager."""
        mock_manager = Mock()
        
        # Mock workspace info
        mock_workspace_info = Mock()
        mock_workspace_info.developers = ['dev1', 'dev2']
        mock_manager.discover_workspaces.return_value = mock_workspace_info
        
        # Mock file resolver
        mock_file_resolver = Mock()
        mock_file_resolver.workspace_root = '/test/workspace'
        mock_manager.get_file_resolver.return_value = mock_file_resolver
        
        # Mock module loader
        mock_module_loader = Mock()
        mock_module_loader.discover_workspace_modules.return_value = {
            'test_step': ['/test/workspace/dev1/builders/test_builder.py']
        }
        mock_manager.get_module_loader.return_value = mock_module_loader
        
        return mock_manager
    
    def test_registry_initialization(self, temp_workspace):
        """Test registry initialization."""
        registry = WorkspaceComponentRegistry(temp_workspace)
        
        assert registry.workspace_root == temp_workspace
        assert registry.cache_expiry == 300  # 5 minutes
        assert registry._component_cache == {}
        assert registry._builder_cache == {}
        assert registry._config_cache == {}
        assert registry._cache_timestamp == {}
    
    @patch('src.cursus.core.workspace.registry.WorkspaceManager')
    def test_discover_components_all_developers(self, mock_workspace_manager_class, temp_workspace):
        """Test discovering components for all developers."""
        # Setup mock
        mock_manager = Mock()
        mock_workspace_info = Mock()
        mock_workspace_info.developers = ['dev1', 'dev2']
        mock_manager.discover_workspaces.return_value = mock_workspace_info
        
        # Mock file resolver and module loader
        mock_file_resolver = Mock()
        mock_file_resolver.workspace_root = temp_workspace
        mock_manager.get_file_resolver.return_value = mock_file_resolver
        
        mock_module_loader = Mock()
        mock_module_loader.discover_workspace_modules.return_value = {
            'test_step': [f'{temp_workspace}/dev1/builders/test_builder.py']
        }
        mock_manager.get_module_loader.return_value = mock_module_loader
        
        mock_workspace_manager_class.return_value = mock_manager
        
        registry = WorkspaceComponentRegistry(temp_workspace)
        
        # Mock the _discover_developer_components method to avoid complex setup
        with patch.object(registry, '_discover_developer_components') as mock_discover:
            components = registry.discover_components()
            
            assert 'builders' in components
            assert 'configs' in components
            assert 'contracts' in components
            assert 'specs' in components
            assert 'scripts' in components
            assert 'summary' in components
            
            # Should be called for each developer
            assert mock_discover.call_count == 2
    
    @patch('src.cursus.core.workspace.registry.WorkspaceManager')
    def test_discover_components_specific_developer(self, mock_workspace_manager_class, temp_workspace):
        """Test discovering components for a specific developer."""
        # Setup mock
        mock_manager = Mock()
        mock_workspace_info = Mock()
        mock_workspace_info.developers = ['dev1', 'dev2']
        mock_manager.discover_workspaces.return_value = mock_workspace_info
        mock_workspace_manager_class.return_value = mock_manager
        
        registry = WorkspaceComponentRegistry(temp_workspace)
        
        # Mock the _discover_developer_components method
        with patch.object(registry, '_discover_developer_components') as mock_discover:
            components = registry.discover_components(developer_id='dev1')
            
            # Should be called only for dev1
            mock_discover.assert_called_once_with('dev1', components)
    
    @patch('src.cursus.core.workspace.registry.WorkspaceManager')
    def test_discover_components_caching(self, mock_workspace_manager_class, temp_workspace):
        """Test component discovery caching."""
        mock_manager = Mock()
        mock_workspace_info = Mock()
        mock_workspace_info.developers = ['dev1']
        mock_manager.discover_workspaces.return_value = mock_workspace_info
        mock_workspace_manager_class.return_value = mock_manager
        
        registry = WorkspaceComponentRegistry(temp_workspace)
        
        with patch.object(registry, '_discover_developer_components'):
            # First call should discover components
            components1 = registry.discover_components()
            
            # Second call should use cache
            components2 = registry.discover_components()
            
            # Should return the same cached result
            assert components1 is components2
    
    def test_cache_expiry(self, temp_workspace):
        """Test cache expiry functionality."""
        registry = WorkspaceComponentRegistry(temp_workspace)
        registry.cache_expiry = 0.1  # 100ms for testing
        
        # Set cache entry
        cache_key = "test_key"
        registry._cache_timestamp[cache_key] = time.time()
        
        # Should be valid immediately
        assert registry._is_cache_valid(cache_key) is True
        
        # Wait for expiry
        time.sleep(0.2)
        
        # Should be expired
        assert registry._is_cache_valid(cache_key) is False
    
    @patch('src.cursus.core.workspace.registry.WorkspaceManager')
    def test_find_builder_class_specific_developer(self, mock_workspace_manager_class, temp_workspace):
        """Test finding builder class for specific developer."""
        mock_manager = Mock()
        mock_module_loader = Mock()
        
        # Mock builder class
        mock_builder_class = Mock()
        mock_builder_class.__name__ = 'TestBuilder'
        mock_module_loader.load_builder_class.return_value = mock_builder_class
        
        mock_manager.get_module_loader.return_value = mock_module_loader
        mock_workspace_manager_class.return_value = mock_manager
        
        registry = WorkspaceComponentRegistry(temp_workspace)
        
        result = registry.find_builder_class('test_step', 'dev1')
        
        assert result == mock_builder_class
        mock_manager.get_module_loader.assert_called_with('dev1')
        mock_module_loader.load_builder_class.assert_called_with('test_step')
    
    @patch('src.cursus.core.workspace.registry.WorkspaceManager')
    def test_find_builder_class_any_developer(self, mock_workspace_manager_class, temp_workspace):
        """Test finding builder class from any developer."""
        mock_manager = Mock()
        mock_workspace_info = Mock()
        mock_workspace_info.developers = ['dev1', 'dev2']
        mock_manager.discover_workspaces.return_value = mock_workspace_info
        
        # First developer returns None, second returns builder
        mock_module_loader1 = Mock()
        mock_module_loader1.load_builder_class.return_value = None
        
        mock_module_loader2 = Mock()
        mock_builder_class = Mock()
        mock_builder_class.__name__ = 'TestBuilder'
        mock_module_loader2.load_builder_class.return_value = mock_builder_class
        
        mock_manager.get_module_loader.side_effect = [mock_module_loader1, mock_module_loader2]
        mock_workspace_manager_class.return_value = mock_manager
        
        registry = WorkspaceComponentRegistry(temp_workspace)
        
        result = registry.find_builder_class('test_step')
        
        assert result == mock_builder_class
    
    @patch('src.cursus.core.workspace.registry.WorkspaceManager')
    @patch('src.cursus.core.workspace.registry.STEP_NAMES')
    def test_find_builder_class_core_fallback(self, mock_step_names, mock_workspace_manager_class, temp_workspace):
        """Test fallback to core registry for builder class."""
        # Setup workspace manager mock
        mock_manager = Mock()
        mock_workspace_info = Mock()
        mock_workspace_info.developers = ['dev1']
        mock_manager.discover_workspaces.return_value = mock_workspace_info
        
        mock_module_loader = Mock()
        mock_module_loader.load_builder_class.return_value = None  # Not found in workspace
        mock_manager.get_module_loader.return_value = mock_module_loader
        mock_workspace_manager_class.return_value = mock_manager
        
        # Setup STEP_NAMES mock
        mock_step_names.__iter__ = Mock(return_value=iter(['TestConfig']))
        mock_step_names.items.return_value = [
            ('TestConfig', {'step_name': 'test_step', 'step_type': 'TestType'})
        ]
        
        registry = WorkspaceComponentRegistry(temp_workspace)
        
        # Mock core registry
        mock_core_builder = Mock()
        mock_core_builder.__name__ = 'CoreBuilder'
        registry.core_registry.get_builder_for_step_type = Mock(return_value=mock_core_builder)
        
        result = registry.find_builder_class('test_step')
        
        assert result == mock_core_builder
        registry.core_registry.get_builder_for_step_type.assert_called_with('TestType')
    
    @patch('src.cursus.core.workspace.registry.WorkspaceManager')
    def test_find_config_class(self, mock_workspace_manager_class, temp_workspace):
        """Test finding config class."""
        mock_manager = Mock()
        mock_module_loader = Mock()
        
        # Mock config class
        mock_config_class = Mock()
        mock_config_class.__name__ = 'TestConfig'
        mock_module_loader.load_contract_class.return_value = mock_config_class
        
        mock_manager.get_module_loader.return_value = mock_module_loader
        mock_workspace_manager_class.return_value = mock_manager
        
        registry = WorkspaceComponentRegistry(temp_workspace)
        
        result = registry.find_config_class('test_step', 'dev1')
        
        assert result == mock_config_class
        mock_module_loader.load_contract_class.assert_called_with('test_step')
    
    @patch('src.cursus.core.workspace.registry.WorkspaceManager')
    def test_get_workspace_summary(self, mock_workspace_manager_class, temp_workspace):
        """Test getting workspace summary."""
        mock_manager = Mock()
        mock_workspace_manager_class.return_value = mock_manager
        
        registry = WorkspaceComponentRegistry(temp_workspace)
        
        # Mock discover_components
        mock_components = {
            'builders': {'dev1:step1': {}, 'dev2:step2': {}},
            'configs': {'dev1:step1': {}},
            'contracts': {},
            'specs': {'dev1:step1': {}},
            'scripts': {},
            'summary': {
                'total_components': 4,
                'developers': ['dev1', 'dev2'],
                'step_types': ['TestType']
            }
        }
        
        with patch.object(registry, 'discover_components', return_value=mock_components):
            summary = registry.get_workspace_summary()
            
            assert summary['workspace_root'] == temp_workspace
            assert summary['total_components'] == 4
            assert summary['developers'] == ['dev1', 'dev2']
            assert summary['step_types'] == ['TestType']
            assert summary['component_counts']['builders'] == 2
            assert summary['component_counts']['configs'] == 1
    
    def test_validate_component_availability_valid(self, temp_workspace):
        """Test component availability validation with valid components."""
        registry = WorkspaceComponentRegistry(temp_workspace)
        
        # Create test workspace config
        steps = [
            WorkspaceStepDefinition(
                step_name='test_step',
                developer_id='dev1',
                step_type='TestType',
                config_data={'param': 'value'},
                workspace_root=temp_workspace
            )
        ]
        
        workspace_config = WorkspacePipelineDefinition(
            pipeline_name='test_pipeline',
            workspace_root=temp_workspace,
            steps=steps
        )
        
        # Mock find methods to return valid components
        mock_builder = Mock()
        mock_builder.__name__ = 'TestBuilder'
        mock_config = Mock()
        mock_config.__name__ = 'TestConfig'
        
        with patch.object(registry, 'find_builder_class', return_value=mock_builder), \
             patch.object(registry, 'find_config_class', return_value=mock_config):
            
            result = registry.validate_component_availability(workspace_config)
            
            assert result['valid'] is True
            assert len(result['available_components']) == 2  # builder + config
            assert len(result['missing_components']) == 0
    
    def test_validate_component_availability_missing(self, temp_workspace):
        """Test component availability validation with missing components."""
        registry = WorkspaceComponentRegistry(temp_workspace)
        
        # Create test workspace config
        steps = [
            WorkspaceStepDefinition(
                step_name='missing_step',
                developer_id='dev1',
                step_type='TestType',
                config_data={'param': 'value'},
                workspace_root=temp_workspace
            )
        ]
        
        workspace_config = WorkspacePipelineDefinition(
            pipeline_name='test_pipeline',
            workspace_root=temp_workspace,
            steps=steps
        )
        
        # Mock find methods to return None (missing components)
        with patch.object(registry, 'find_builder_class', return_value=None), \
             patch.object(registry, 'find_config_class', return_value=None):
            
            result = registry.validate_component_availability(workspace_config)
            
            assert result['valid'] is False
            assert len(result['missing_components']) == 1
            assert result['missing_components'][0]['step_name'] == 'missing_step'
            assert result['missing_components'][0]['component_type'] == 'builder'
    
    def test_clear_cache(self, temp_workspace):
        """Test clearing all caches."""
        registry = WorkspaceComponentRegistry(temp_workspace)
        
        # Add some cache entries
        registry._component_cache['test'] = {'data': 'value'}
        registry._builder_cache['test'] = Mock()
        registry._config_cache['test'] = Mock()
        registry._cache_timestamp['test'] = time.time()
        
        # Clear cache
        registry.clear_cache()
        
        assert registry._component_cache == {}
        assert registry._builder_cache == {}
        assert registry._config_cache == {}
        assert registry._cache_timestamp == {}
    
    def test_discover_developer_components_error_handling(self, temp_workspace):
        """Test error handling in developer component discovery."""
        registry = WorkspaceComponentRegistry(temp_workspace)
        
        # Mock workspace manager to raise exception
        with patch.object(registry.workspace_manager, 'get_file_resolver', side_effect=Exception("Test error")):
            components = {
                'builders': {},
                'configs': {},
                'contracts': {},
                'specs': {},
                'scripts': {},
                'summary': {'step_types': set()}
            }
            
            # Should not raise exception, just log error
            registry._discover_developer_components('dev1', components)
            
            # Components should remain empty
            assert len(components['builders']) == 0
    
    @patch('src.cursus.core.workspace.registry.WorkspaceManager')
    def test_builder_class_caching(self, mock_workspace_manager_class, temp_workspace):
        """Test builder class caching functionality."""
        mock_manager = Mock()
        mock_module_loader = Mock()
        
        mock_builder_class = Mock()
        mock_builder_class.__name__ = 'TestBuilder'
        mock_module_loader.load_builder_class.return_value = mock_builder_class
        
        mock_manager.get_module_loader.return_value = mock_module_loader
        mock_workspace_manager_class.return_value = mock_manager
        
        registry = WorkspaceComponentRegistry(temp_workspace)
        
        # First call should load and cache
        result1 = registry.find_builder_class('test_step', 'dev1')
        
        # Second call should use cache
        result2 = registry.find_builder_class('test_step', 'dev1')
        
        assert result1 == mock_builder_class
        assert result2 == mock_builder_class
        assert result1 is result2  # Same cached object
        
        # Module loader should only be called once
        mock_module_loader.load_builder_class.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__])
