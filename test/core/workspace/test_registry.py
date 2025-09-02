"""
Unit tests for workspace component registry.

Tests the WorkspaceComponentRegistry for component discovery, caching, and management.
"""

import unittest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.cursus.core.workspace.registry import WorkspaceComponentRegistry
from src.cursus.core.workspace.config import WorkspaceStepDefinition, WorkspacePipelineDefinition


class TestWorkspaceComponentRegistry(unittest.TestCase):
    """Test cases for WorkspaceComponentRegistry."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_workspace = self.temp_dir
        
        # Create workspace structure
        workspace_path = Path(self.temp_workspace)
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
        
        # Create mock workspace manager
        self.mock_workspace_manager = Mock()
        
        # Mock workspace info
        mock_workspace_info = Mock()
        mock_workspace_info.developers = ['dev1', 'dev2']
        self.mock_workspace_manager.discover_workspaces.return_value = mock_workspace_info
        
        # Mock file resolver
        mock_file_resolver = Mock()
        mock_file_resolver.workspace_root = '/test/workspace'
        self.mock_workspace_manager.get_file_resolver.return_value = mock_file_resolver
        
        # Mock module loader
        mock_module_loader = Mock()
        mock_module_loader.discover_workspace_modules.return_value = {
            'test_step': ['/test/workspace/dev1/builders/test_builder.py']
        }
        self.mock_workspace_manager.get_module_loader.return_value = mock_module_loader
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.cursus.core.workspace.registry.WorkspaceDiscoveryManager')
    def test_registry_initialization_with_discovery_manager(self, mock_discovery_class):
        """Test registry initialization with discovery manager (Phase 2 optimization)."""
        mock_discovery = Mock()
        mock_discovery_class.return_value = mock_discovery
        
        registry = WorkspaceComponentRegistry(
            workspace_root=self.temp_workspace,
            discovery_manager=mock_discovery
        )
        
        self.assertEqual(registry.workspace_root, self.temp_workspace)
        self.assertEqual(registry.discovery_manager, mock_discovery)
        self.assertEqual(registry.cache_expiry, 300)  # 5 minutes
        self.assertEqual(registry._component_cache, {})
        self.assertEqual(registry._builder_cache, {})
        self.assertEqual(registry._config_cache, {})
        self.assertEqual(registry._cache_timestamp, {})
    
    @patch('src.cursus.core.workspace.registry.WorkspaceDiscoveryManager')
    def test_registry_initialization_without_discovery_manager(self, mock_discovery_class):
        """Test registry initialization without discovery manager."""
        mock_discovery = Mock()
        mock_discovery_class.return_value = mock_discovery
        
        registry = WorkspaceComponentRegistry(workspace_root=self.temp_workspace)
        
        self.assertEqual(registry.workspace_root, self.temp_workspace)
        self.assertEqual(registry.discovery_manager, mock_discovery)
        mock_discovery_class.assert_called_once_with(self.temp_workspace)
    
    @patch('src.cursus.core.workspace.registry.WorkspaceDiscoveryManager')
    def test_discover_components_all_developers(self, mock_discovery_class):
        """Test discovering components for all developers."""
        # Setup mock
        mock_discovery = Mock()
        mock_discovery.discover_workspace_components.return_value = {
            'builders': {'dev1:test_step': {}, 'dev2:test_step': {}},
            'configs': {'dev1:test_step': {}},
            'contracts': {},
            'specs': {'dev1:test_step': {}},
            'scripts': {},
            'summary': {
                'total_components': 4,
                'developers': ['dev1', 'dev2'],
                'step_types': ['TestType']
            }
        }
        mock_discovery_class.return_value = mock_discovery
        
        registry = WorkspaceComponentRegistry(self.temp_workspace)
        components = registry.discover_components()
        
        self.assertIn('builders', components)
        self.assertIn('configs', components)
        self.assertIn('contracts', components)
        self.assertIn('specs', components)
        self.assertIn('scripts', components)
        self.assertIn('summary', components)
        
        # Should use discovery manager
        mock_discovery.discover_workspace_components.assert_called_once()
    
    @patch('src.cursus.core.workspace.registry.WorkspaceDiscoveryManager')
    def test_discover_components_specific_developer(self, mock_discovery_class):
        """Test discovering components for a specific developer."""
        mock_discovery = Mock()
        mock_discovery.discover_workspace_components.return_value = {
            'builders': {'dev1:test_step': {}},
            'configs': {'dev1:test_step': {}},
            'contracts': {},
            'specs': {'dev1:test_step': {}},
            'scripts': {},
            'summary': {
                'total_components': 3,
                'developers': ['dev1'],
                'step_types': ['TestType']
            }
        }
        mock_discovery_class.return_value = mock_discovery
        
        registry = WorkspaceComponentRegistry(self.temp_workspace)
        components = registry.discover_components(developer_id='dev1')
        
        # Should call discovery manager with specific developer
        mock_discovery.discover_workspace_components.assert_called_once_with(developer_id='dev1')
    
    @patch('src.cursus.core.workspace.registry.WorkspaceDiscoveryManager')
    def test_discover_components_caching(self, mock_discovery_class):
        """Test component discovery caching."""
        mock_discovery = Mock()
        mock_components = {
            'builders': {'dev1:test_step': {}},
            'configs': {},
            'contracts': {},
            'specs': {},
            'scripts': {},
            'summary': {'total_components': 1, 'developers': ['dev1']}
        }
        mock_discovery.discover_workspace_components.return_value = mock_components
        mock_discovery_class.return_value = mock_discovery
        
        registry = WorkspaceComponentRegistry(self.temp_workspace)
        
        # First call should discover components
        components1 = registry.discover_components()
        
        # Second call should use cache
        components2 = registry.discover_components()
        
        # Should return the same cached result
        self.assertIs(components1, components2)
        
        # Discovery manager should only be called once
        mock_discovery.discover_workspace_components.assert_called_once()
    
    def test_cache_expiry(self):
        """Test cache expiry functionality."""
        with patch('src.cursus.core.workspace.registry.WorkspaceDiscoveryManager'):
            registry = WorkspaceComponentRegistry(self.temp_workspace)
            registry.cache_expiry = 0.1  # 100ms for testing
            
            # Set cache entry
            cache_key = "test_key"
            registry._cache_timestamp[cache_key] = time.time()
            
            # Should be valid immediately
            self.assertTrue(registry._is_cache_valid(cache_key))
            
            # Wait for expiry
            time.sleep(0.2)
            
            # Should be expired
            self.assertFalse(registry._is_cache_valid(cache_key))
    
    @patch('src.cursus.core.workspace.registry.WorkspaceDiscoveryManager')
    def test_find_builder_class_specific_developer(self, mock_discovery_class):
        """Test finding builder class for specific developer."""
        mock_discovery = Mock()
        
        # Mock builder class
        mock_builder_class = Mock()
        mock_builder_class.__name__ = 'TestBuilder'
        mock_discovery.find_component.return_value = mock_builder_class
        
        mock_discovery_class.return_value = mock_discovery
        
        registry = WorkspaceComponentRegistry(self.temp_workspace)
        result = registry.find_builder_class('test_step', 'dev1')
        
        self.assertEqual(result, mock_builder_class)
        mock_discovery.find_component.assert_called_with(
            'test_step', 'builder', developer_id='dev1'
        )
    
    @patch('src.cursus.core.workspace.registry.WorkspaceDiscoveryManager')
    def test_find_builder_class_any_developer(self, mock_discovery_class):
        """Test finding builder class from any developer."""
        mock_discovery = Mock()
        
        mock_builder_class = Mock()
        mock_builder_class.__name__ = 'TestBuilder'
        mock_discovery.find_component.return_value = mock_builder_class
        
        mock_discovery_class.return_value = mock_discovery
        
        registry = WorkspaceComponentRegistry(self.temp_workspace)
        result = registry.find_builder_class('test_step')
        
        self.assertEqual(result, mock_builder_class)
        mock_discovery.find_component.assert_called_with(
            'test_step', 'builder', developer_id=None
        )
    
    @patch('src.cursus.core.workspace.registry.WorkspaceDiscoveryManager')
    @patch('src.cursus.core.workspace.registry.STEP_NAMES')
    def test_find_builder_class_core_fallback(self, mock_step_names, mock_discovery_class):
        """Test fallback to core registry for builder class."""
        # Setup discovery manager mock
        mock_discovery = Mock()
        mock_discovery.find_component.return_value = None  # Not found in workspace
        mock_discovery_class.return_value = mock_discovery
        
        # Setup STEP_NAMES mock
        mock_step_names.__iter__ = Mock(return_value=iter(['TestConfig']))
        mock_step_names.items.return_value = [
            ('TestConfig', {'step_name': 'test_step', 'step_type': 'TestType'})
        ]
        
        registry = WorkspaceComponentRegistry(self.temp_workspace)
        
        # Mock core registry
        mock_core_builder = Mock()
        mock_core_builder.__name__ = 'CoreBuilder'
        registry.core_registry.get_builder_for_step_type = Mock(return_value=mock_core_builder)
        
        result = registry.find_builder_class('test_step')
        
        self.assertEqual(result, mock_core_builder)
        registry.core_registry.get_builder_for_step_type.assert_called_with('TestType')
    
    @patch('src.cursus.core.workspace.registry.WorkspaceDiscoveryManager')
    def test_find_config_class(self, mock_discovery_class):
        """Test finding config class."""
        mock_discovery = Mock()
        
        # Mock config class
        mock_config_class = Mock()
        mock_config_class.__name__ = 'TestConfig'
        mock_discovery.find_component.return_value = mock_config_class
        
        mock_discovery_class.return_value = mock_discovery
        
        registry = WorkspaceComponentRegistry(self.temp_workspace)
        result = registry.find_config_class('test_step', 'dev1')
        
        self.assertEqual(result, mock_config_class)
        mock_discovery.find_component.assert_called_with(
            'test_step', 'config', developer_id='dev1'
        )
    
    @patch('src.cursus.core.workspace.registry.WorkspaceDiscoveryManager')
    def test_get_workspace_summary(self, mock_discovery_class):
        """Test getting workspace summary."""
        mock_discovery = Mock()
        mock_discovery_class.return_value = mock_discovery
        
        registry = WorkspaceComponentRegistry(self.temp_workspace)
        
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
            
            self.assertEqual(summary['workspace_root'], self.temp_workspace)
            self.assertEqual(summary['total_components'], 4)
            self.assertEqual(summary['developers'], ['dev1', 'dev2'])
            self.assertEqual(summary['step_types'], ['TestType'])
            self.assertEqual(summary['component_counts']['builders'], 2)
            self.assertEqual(summary['component_counts']['configs'], 1)
    
    @patch('src.cursus.core.workspace.registry.WorkspaceDiscoveryManager')
    def test_validate_component_availability_valid(self, mock_discovery_class):
        """Test component availability validation with valid components."""
        mock_discovery = Mock()
        mock_discovery_class.return_value = mock_discovery
        
        registry = WorkspaceComponentRegistry(self.temp_workspace)
        
        # Create test workspace config
        steps = [
            WorkspaceStepDefinition(
                step_name='test_step',
                developer_id='dev1',
                step_type='TestType',
                config_data={'param': 'value'},
                workspace_root=self.temp_workspace
            )
        ]
        
        workspace_config = WorkspacePipelineDefinition(
            pipeline_name='test_pipeline',
            workspace_root=self.temp_workspace,
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
            
            self.assertTrue(result['valid'])
            self.assertEqual(len(result['available_components']), 2)  # builder + config
            self.assertEqual(len(result['missing_components']), 0)
    
    @patch('src.cursus.core.workspace.registry.WorkspaceDiscoveryManager')
    def test_validate_component_availability_missing(self, mock_discovery_class):
        """Test component availability validation with missing components."""
        mock_discovery = Mock()
        mock_discovery_class.return_value = mock_discovery
        
        registry = WorkspaceComponentRegistry(self.temp_workspace)
        
        # Create test workspace config
        steps = [
            WorkspaceStepDefinition(
                step_name='missing_step',
                developer_id='dev1',
                step_type='TestType',
                config_data={'param': 'value'},
                workspace_root=self.temp_workspace
            )
        ]
        
        workspace_config = WorkspacePipelineDefinition(
            pipeline_name='test_pipeline',
            workspace_root=self.temp_workspace,
            steps=steps
        )
        
        # Mock find methods to return None (missing components)
        with patch.object(registry, 'find_builder_class', return_value=None), \
             patch.object(registry, 'find_config_class', return_value=None):
            
            result = registry.validate_component_availability(workspace_config)
            
            self.assertFalse(result['valid'])
            self.assertEqual(len(result['missing_components']), 1)
            self.assertEqual(result['missing_components'][0]['step_name'], 'missing_step')
            self.assertEqual(result['missing_components'][0]['component_type'], 'builder')
    
    @patch('src.cursus.core.workspace.registry.WorkspaceDiscoveryManager')
    def test_clear_cache(self, mock_discovery_class):
        """Test clearing all caches."""
        mock_discovery = Mock()
        mock_discovery_class.return_value = mock_discovery
        
        registry = WorkspaceComponentRegistry(self.temp_workspace)
        
        # Add some cache entries
        registry._component_cache['test'] = {'data': 'value'}
        registry._builder_cache['test'] = Mock()
        registry._config_cache['test'] = Mock()
        registry._cache_timestamp['test'] = time.time()
        
        # Clear cache
        registry.clear_cache()
        
        self.assertEqual(registry._component_cache, {})
        self.assertEqual(registry._builder_cache, {})
        self.assertEqual(registry._config_cache, {})
        self.assertEqual(registry._cache_timestamp, {})
    
    @patch('src.cursus.core.workspace.registry.WorkspaceDiscoveryManager')
    def test_discover_developer_components_error_handling(self, mock_discovery_class):
        """Test error handling in developer component discovery."""
        mock_discovery = Mock()
        mock_discovery.discover_workspace_components.side_effect = Exception("Test error")
        mock_discovery_class.return_value = mock_discovery
        
        registry = WorkspaceComponentRegistry(self.temp_workspace)
        
        # Should not raise exception, should return empty components
        components = registry.discover_components()
        
        # Should return default empty structure
        self.assertIn('builders', components)
        self.assertIn('configs', components)
        self.assertIn('contracts', components)
        self.assertIn('specs', components)
        self.assertIn('scripts', components)
        self.assertIn('summary', components)
    
    @patch('src.cursus.core.workspace.registry.WorkspaceDiscoveryManager')
    def test_builder_class_caching(self, mock_discovery_class):
        """Test builder class caching functionality."""
        mock_discovery = Mock()
        
        mock_builder_class = Mock()
        mock_builder_class.__name__ = 'TestBuilder'
        mock_discovery.find_component.return_value = mock_builder_class
        
        mock_discovery_class.return_value = mock_discovery
        
        registry = WorkspaceComponentRegistry(self.temp_workspace)
        
        # First call should load and cache
        result1 = registry.find_builder_class('test_step', 'dev1')
        
        # Second call should use cache
        result2 = registry.find_builder_class('test_step', 'dev1')
        
        self.assertEqual(result1, mock_builder_class)
        self.assertEqual(result2, mock_builder_class)
        self.assertIs(result1, result2)  # Same cached object
        
        # Discovery manager should only be called once
        mock_discovery.find_component.assert_called_once()
    
    @patch('src.cursus.core.workspace.registry.WorkspaceDiscoveryManager')
    def test_optimized_component_discovery_with_caching(self, mock_discovery_class):
        """Test optimized component discovery using consolidated discovery manager (Phase 2)."""
        mock_discovery = Mock()
        
        # Mock cached discovery results
        cached_components = {
            'builders': {'dev1:test_step': {'cached': True}},
            'configs': {'dev1:test_step': {'cached': True}},
            'contracts': {},
            'specs': {},
            'scripts': {},
            'summary': {
                'total_components': 2,
                'developers': ['dev1'],
                'step_types': ['TestType'],
                'cached': True
            }
        }
        
        mock_discovery.get_cached_components.return_value = cached_components
        mock_discovery.discover_workspace_components.return_value = cached_components
        mock_discovery_class.return_value = mock_discovery
        
        registry = WorkspaceComponentRegistry(self.temp_workspace)
        
        # First call should use discovery manager's caching
        components1 = registry.discover_components()
        
        # Verify cached results are used
        self.assertTrue(components1['summary']['cached'])
        self.assertEqual(components1['summary']['total_components'], 2)
        
        # Second call should use registry's own cache
        components2 = registry.discover_components()
        
        # Should be the same cached object
        self.assertIs(components1, components2)


if __name__ == '__main__':
    unittest.main()
