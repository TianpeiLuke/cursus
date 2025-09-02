"""
Unit tests for WorkspaceDiscoveryManager.

This module provides comprehensive unit testing for the WorkspaceDiscoveryManager
and its component discovery capabilities.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from src.cursus.core.workspace.discovery import WorkspaceDiscoveryManager


class TestWorkspaceDiscoveryManager(unittest.TestCase):
    """Test suite for WorkspaceDiscoveryManager."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_workspace = str(Path(self.temp_dir) / "test_workspace")
        Path(self.temp_workspace).mkdir(parents=True, exist_ok=True)
        
        # Create mock workspace manager
        self.mock_workspace_manager = Mock()
        self.mock_workspace_manager.workspace_root = self.temp_workspace
        self.mock_workspace_manager.config_file = None
        self.mock_workspace_manager.auto_discover = True
    
    def tearDown(self):
        """Clean up after each test method."""
        if hasattr(self, 'temp_dir') and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_discovery_manager_initialization(self):
        """Test WorkspaceDiscoveryManager initialization."""
        discovery_manager = WorkspaceDiscoveryManager(self.mock_workspace_manager)
        
        self.assertIs(discovery_manager.workspace_manager, self.mock_workspace_manager)
        self.assertEqual(discovery_manager.workspace_root, self.mock_workspace_manager.workspace_root)
        self.assertEqual(discovery_manager.component_cache, {})
        self.assertEqual(discovery_manager.dependency_graph, {})
        self.assertEqual(discovery_manager.discovery_index, {})
    
    def test_discover_workspace_components_empty(self):
        """Test component discovery with empty workspace."""
        discovery_manager = WorkspaceDiscoveryManager(self.mock_workspace_manager)
        
        # Create empty workspace
        workspace_path = Path(self.temp_workspace) / "developer_1"
        workspace_path.mkdir(exist_ok=True)
        
        inventory = discovery_manager.discover_workspace_components("developer_1")
        
        self.assertEqual(inventory.workspace_id, "developer_1")
        self.assertEqual(len(inventory.builders), 0)
        self.assertEqual(len(inventory.configs), 0)
        self.assertEqual(len(inventory.scripts), 0)
    
    def test_discover_workspace_components_with_content(self):
        """Test component discovery with workspace content."""
        discovery_manager = WorkspaceDiscoveryManager(self.mock_workspace_manager)
        
        # Create workspace with components
        workspace_path = Path(self.temp_workspace) / "developer_1"
        workspace_path.mkdir(exist_ok=True)
        
        # Create builders
        builders_path = workspace_path / "src" / "cursus_dev" / "steps" / "builders"
        builders_path.mkdir(parents=True, exist_ok=True)
        (builders_path / "test_builder.py").write_text("class TestBuilder: pass")
        
        # Create configs
        configs_path = workspace_path / "src" / "cursus_dev" / "steps" / "configs"
        configs_path.mkdir(parents=True, exist_ok=True)
        (configs_path / "test_config.py").write_text("class TestConfig: pass")
        
        # Create scripts
        scripts_path = workspace_path / "src" / "cursus_dev" / "steps" / "scripts"
        scripts_path.mkdir(parents=True, exist_ok=True)
        (scripts_path / "test_script.py").write_text("def main(): pass")
        
        inventory = discovery_manager.discover_workspace_components("developer_1")
        
        self.assertEqual(inventory.workspace_id, "developer_1")
        self.assertGreater(len(inventory.builders), 0)
        self.assertGreater(len(inventory.configs), 0)
        self.assertGreater(len(inventory.scripts), 0)
    
    def test_build_dependency_graph_simple(self):
        """Test building dependency graph with simple dependencies."""
        discovery_manager = WorkspaceDiscoveryManager(self.mock_workspace_manager)
        
        # Mock component inventories
        mock_inventory_1 = Mock()
        mock_inventory_1.workspace_id = "developer_1"
        mock_inventory_1.builders = {"step1": {"dependencies": []}}
        
        mock_inventory_2 = Mock()
        mock_inventory_2.workspace_id = "developer_2"
        mock_inventory_2.builders = {"step2": {"dependencies": ["step1"]}}
        
        with patch.object(discovery_manager, 'discover_workspace_components', side_effect=[mock_inventory_1, mock_inventory_2]):
            graph = discovery_manager.build_dependency_graph(["developer_1", "developer_2"])
            
            self.assertIn("step1", graph.nodes)
            self.assertIn("step2", graph.nodes)
            self.assertTrue(graph.has_edge("step1", "step2"))
    
    def test_find_component_conflicts_none(self):
        """Test finding component conflicts with no conflicts."""
        discovery_manager = WorkspaceDiscoveryManager(self.mock_workspace_manager)
        
        # Mock inventories with unique components
        mock_inventory_1 = Mock()
        mock_inventory_1.workspace_id = "developer_1"
        mock_inventory_1.builders = {"step1": {"name": "step1"}}
        
        mock_inventory_2 = Mock()
        mock_inventory_2.workspace_id = "developer_2"
        mock_inventory_2.builders = {"step2": {"name": "step2"}}
        
        with patch.object(discovery_manager, 'discover_workspace_components', side_effect=[mock_inventory_1, mock_inventory_2]):
            conflicts = discovery_manager.find_component_conflicts(["developer_1", "developer_2"])
            
            self.assertEqual(len(conflicts), 0)
    
    def test_find_component_conflicts_found(self):
        """Test finding component conflicts with conflicts present."""
        discovery_manager = WorkspaceDiscoveryManager(self.mock_workspace_manager)
        
        # Mock inventories with conflicting components
        mock_inventory_1 = Mock()
        mock_inventory_1.workspace_id = "developer_1"
        mock_inventory_1.builders = {"duplicate_step": {"name": "duplicate_step", "version": "1.0"}}
        
        mock_inventory_2 = Mock()
        mock_inventory_2.workspace_id = "developer_2"
        mock_inventory_2.builders = {"duplicate_step": {"name": "duplicate_step", "version": "2.0"}}
        
        with patch.object(discovery_manager, 'discover_workspace_components', side_effect=[mock_inventory_1, mock_inventory_2]):
            conflicts = discovery_manager.find_component_conflicts(["developer_1", "developer_2"])
            
            self.assertGreater(len(conflicts), 0)
            self.assertTrue(any(c.component_name == "duplicate_step" for c in conflicts))
    
    def test_resolve_component_dependencies_simple(self):
        """Test resolving component dependencies."""
        discovery_manager = WorkspaceDiscoveryManager(self.mock_workspace_manager)
        
        # Set up dependency graph
        discovery_manager.dependency_graph = {
            "step1": [],
            "step2": ["step1"],
            "step3": ["step2"]
        }
        
        resolution = discovery_manager.resolve_component_dependencies("step3")
        
        self.assertEqual(resolution.component_id, "step3")
        self.assertTrue(resolution.valid)
        self.assertIn("step1", resolution.resolved_dependencies)
        self.assertIn("step2", resolution.resolved_dependencies)
    
    def test_resolve_component_dependencies_circular(self):
        """Test resolving component dependencies with circular dependency."""
        discovery_manager = WorkspaceDiscoveryManager(self.mock_workspace_manager)
        
        # Set up circular dependency graph
        discovery_manager.dependency_graph = {
            "step1": ["step2"],
            "step2": ["step1"]
        }
        
        resolution = discovery_manager.resolve_component_dependencies("step1")
        
        self.assertEqual(resolution.component_id, "step1")
        self.assertFalse(resolution.valid)
        self.assertIn("circular dependency", resolution.error.lower())
    
    def test_discover_components_with_caching(self):
        """Test component discovery with caching."""
        discovery_manager = WorkspaceDiscoveryManager(self.mock_workspace_manager)
        
        workspaces = ["developer_1", "developer_2"]
        
        # Mock the actual discovery method
        with patch.object(discovery_manager, 'discover_workspace_components') as mock_discover:
            mock_inventory = Mock()
            mock_inventory.workspace_id = "test"
            mock_inventory.builders = {}
            mock_discover.return_value = mock_inventory
            
            # First call should trigger discovery
            result1 = discovery_manager.discover_components(workspaces)
            
            # Second call should use cache
            result2 = discovery_manager.discover_components(workspaces)
            
            # Discovery should only be called once per workspace due to caching
            self.assertEqual(mock_discover.call_count, len(workspaces))
    
    def test_get_component_cache(self):
        """Test getting component cache."""
        discovery_manager = WorkspaceDiscoveryManager(self.mock_workspace_manager)
        
        # Add some test data to cache
        test_cache = {"developer_1": {"builders": {"step1": "info"}}}
        discovery_manager.component_cache = test_cache
        
        cache = discovery_manager.get_component_cache()
        
        self.assertEqual(cache, test_cache)
    
    def test_clear_discovery_cache(self):
        """Test clearing discovery cache."""
        discovery_manager = WorkspaceDiscoveryManager(self.mock_workspace_manager)
        
        # Add test data
        discovery_manager.component_cache = {"developer_1": {"data": "test"}}
        discovery_manager.dependency_graph = {"step1": ["step2"]}
        discovery_manager.discovery_index = {"step1": "developer_1"}
        
        # Clear cache
        discovery_manager.clear_discovery_cache()
        
        self.assertEqual(len(discovery_manager.component_cache), 0)
        self.assertEqual(len(discovery_manager.dependency_graph), 0)
        self.assertEqual(len(discovery_manager.discovery_index), 0)
    
    def test_refresh_component_discovery(self):
        """Test refreshing component discovery."""
        discovery_manager = WorkspaceDiscoveryManager(self.mock_workspace_manager)
        
        # Add old cache data
        discovery_manager.component_cache = {"developer_1": {"old": "data"}}
        
        with patch.object(discovery_manager, 'discover_workspace_components') as mock_discover:
            mock_inventory = Mock()
            mock_inventory.workspace_id = "developer_1"
            mock_inventory.builders = {"new": "data"}
            mock_discover.return_value = mock_inventory
            
            # Refresh discovery
            discovery_manager.refresh_component_discovery("developer_1")
            
            # Should have called discovery again
            mock_discover.assert_called_once_with("developer_1")
    
    def test_validate_pipeline_dependencies_valid(self):
        """Test pipeline dependency validation with valid dependencies."""
        discovery_manager = WorkspaceDiscoveryManager(self.mock_workspace_manager)
        
        mock_pipeline = Mock()
        mock_pipeline.steps = []
        
        # Mock dependency resolution to return valid
        with patch.object(discovery_manager, 'resolve_component_dependencies') as mock_resolve:
            mock_resolution = Mock()
            mock_resolution.valid = True
            mock_resolution.resolved_dependencies = []
            mock_resolve.return_value = mock_resolution
            
            result = discovery_manager.validate_pipeline_dependencies(mock_pipeline)
            
            self.assertTrue(result['valid'])
            self.assertEqual(len(result['errors']), 0)
    
    def test_validate_pipeline_dependencies_invalid(self):
        """Test pipeline dependency validation with invalid dependencies."""
        discovery_manager = WorkspaceDiscoveryManager(self.mock_workspace_manager)
        
        mock_step = Mock()
        mock_step.step_name = "invalid_step"
        
        mock_pipeline = Mock()
        mock_pipeline.steps = [mock_step]
        
        # Mock dependency resolution to return invalid
        with patch.object(discovery_manager, 'resolve_component_dependencies') as mock_resolve:
            mock_resolution = Mock()
            mock_resolution.valid = False
            mock_resolution.error = "Dependency not found"
            mock_resolve.return_value = mock_resolution
            
            result = discovery_manager.validate_pipeline_dependencies(mock_pipeline)
            
            self.assertFalse(result['valid'])
            self.assertGreater(len(result['errors']), 0)
    
    def test_resolve_cross_workspace_dependencies(self):
        """Test cross-workspace dependency resolution."""
        discovery_manager = WorkspaceDiscoveryManager(self.mock_workspace_manager)
        
        mock_pipeline = Mock()
        mock_pipeline.get_developers.return_value = ["developer_1", "developer_2"]
        mock_pipeline.steps = []
        
        # Mock component discovery
        with patch.object(discovery_manager, 'discover_components') as mock_discover:
            mock_result = Mock()
            mock_result.components = {"step1": {"workspace": "developer_1"}}
            mock_discover.return_value = mock_result
            
            result = discovery_manager.resolve_cross_workspace_dependencies(mock_pipeline)
            
            self.assertTrue(result['valid'])
            mock_discover.assert_called_once_with(["developer_1", "developer_2"])
    
    def test_get_discovery_statistics(self):
        """Test getting discovery statistics."""
        discovery_manager = WorkspaceDiscoveryManager(self.mock_workspace_manager)
        
        # Add test data
        discovery_manager.component_cache = {
            "developer_1": {"builders": {"step1": {}, "step2": {}}},
            "developer_2": {"builders": {"step3": {}}}
        }
        discovery_manager.dependency_graph = {"step1": [], "step2": ["step1"], "step3": []}
        
        stats = discovery_manager.get_discovery_statistics()
        
        self.assertEqual(stats.total_workspaces, 2)
        self.assertEqual(stats.total_components, 3)
        self.assertEqual(stats.total_dependencies, 1)
        self.assertEqual(len(stats.workspace_breakdown), 2)
    
    def test_get_summary(self):
        """Test getting discovery manager summary."""
        discovery_manager = WorkspaceDiscoveryManager(self.mock_workspace_manager)
        
        # Add test data
        discovery_manager.component_cache = {"developer_1": {"builders": {"step1": {}}}}
        discovery_manager.dependency_graph = {"step1": []}
        
        summary = discovery_manager.get_summary()
        
        self.assertIn('total_workspaces_discovered', summary)
        self.assertIn('total_components_cached', summary)
        self.assertIn('dependency_graph_size', summary)
        self.assertIn('workspace_root', summary)
        self.assertEqual(summary['total_workspaces_discovered'], 1)
        self.assertEqual(summary['total_components_cached'], 1)
    
    def test_validate_health(self):
        """Test discovery manager health validation."""
        discovery_manager = WorkspaceDiscoveryManager(self.mock_workspace_manager)
        
        health = discovery_manager.validate_health()
        
        self.assertIn('healthy', health)
        self.assertIn('discovery_system_functional', health)
        self.assertIn('cache_system_operational', health)
        self.assertIn('workspace_root_accessible', health)
        self.assertTrue(health['healthy'])
    
    def test_error_handling_invalid_workspace(self):
        """Test error handling for invalid workspace."""
        discovery_manager = WorkspaceDiscoveryManager(self.mock_workspace_manager)
        
        # Try to discover components in non-existent workspace
        inventory = discovery_manager.discover_workspace_components("non_existent_workspace")
        
        self.assertEqual(inventory.workspace_id, "non_existent_workspace")
        self.assertEqual(len(inventory.builders), 0)
        self.assertGreater(len(inventory.errors), 0)
    
    def test_concurrent_discovery_operations(self):
        """Test concurrent discovery operations."""
        discovery_manager = WorkspaceDiscoveryManager(self.mock_workspace_manager)
        
        # Create multiple workspaces
        workspaces = []
        for i in range(3):
            workspace_path = Path(self.temp_workspace) / f"developer_{i}"
            workspace_path.mkdir(exist_ok=True)
            workspaces.append(f"developer_{i}")
        
        # Discover components for all workspaces
        inventories = []
        for workspace in workspaces:
            inventory = discovery_manager.discover_workspace_components(workspace)
            inventories.append(inventory)
        
        # All should succeed
        self.assertEqual(len(inventories), 3)
        self.assertTrue(all(inv.workspace_id.startswith("developer_") for inv in inventories))


if __name__ == "__main__":
    unittest.main()
