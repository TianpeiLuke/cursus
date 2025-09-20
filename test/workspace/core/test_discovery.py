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

from cursus.workspace.core.discovery import (
    WorkspaceDiscoveryManager,
    ComponentInventory,
    DependencyGraph,
)


class TestWorkspaceDiscoveryManager(unittest.TestCase):
    """Test suite for WorkspaceDiscoveryManager."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_workspace = Path(self.temp_dir) / "test_workspace"
        self.temp_workspace.mkdir(parents=True, exist_ok=True)

        # Create mock workspace manager
        self.mock_workspace_manager = Mock()
        self.mock_workspace_manager.workspace_root = self.temp_workspace
        self.mock_workspace_manager.config = None

    def tearDown(self):
        """Clean up after each test method."""
        if hasattr(self, "temp_dir") and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_discovery_manager_initialization(self):
        """Test WorkspaceDiscoveryManager initialization."""
        # Updated for new adapter API - uses workspace_root instead of workspace_manager
        discovery_manager = WorkspaceDiscoveryManager(self.temp_workspace)

        # The adapter has different attributes than the old manager
        self.assertIsInstance(discovery_manager._component_cache, dict)
        self.assertIsInstance(discovery_manager._dependency_cache, dict)
        self.assertIsInstance(discovery_manager._cache_timestamp, dict)
        self.assertEqual(discovery_manager.cache_expiry, 300)
        # The adapter has a catalog instead of workspace_manager
        self.assertIsNotNone(discovery_manager.catalog)

    def test_discover_workspaces_empty(self):
        """Test workspace discovery with empty workspace root."""
        # Updated for new adapter API
        discovery_manager = WorkspaceDiscoveryManager(self.temp_workspace)

        result = discovery_manager.discover_workspaces(self.temp_workspace)

        self.assertIsInstance(result, dict)
        self.assertEqual(result["workspace_root"], str(self.temp_workspace))
        self.assertIsInstance(result["workspaces"], list)
        self.assertEqual(result["summary"]["total_workspaces"], 0)

    def test_discover_workspaces_with_content(self):
        """Test workspace discovery with workspace content."""
        # Updated for new adapter API
        discovery_manager = WorkspaceDiscoveryManager(self.temp_workspace)

        # Create developer workspace with components
        developers_dir = self.temp_workspace / "developers"
        developers_dir.mkdir(exist_ok=True)

        workspace_path = developers_dir / "developer_1"
        workspace_path.mkdir(exist_ok=True)

        # Create cursus_dev structure
        cursus_dev_path = workspace_path / "src" / "cursus_dev" / "steps"
        cursus_dev_path.mkdir(parents=True, exist_ok=True)

        # Create builders
        builders_path = cursus_dev_path / "builders"
        builders_path.mkdir(exist_ok=True)
        (builders_path / "test_builder.py").write_text("class TestBuilder: pass")

        result = discovery_manager.discover_workspaces(self.temp_workspace)

        self.assertIsInstance(result, dict)
        self.assertEqual(result["summary"]["total_workspaces"], 1)
        self.assertEqual(result["summary"]["total_developers"], 1)
        self.assertGreater(result["summary"]["total_components"], 0)

    def test_discover_components_empty(self):
        """Test component discovery with empty workspaces."""
        # Updated for new adapter API
        discovery_manager = WorkspaceDiscoveryManager(self.temp_workspace)

        result = discovery_manager.discover_components()

        self.assertIsInstance(result, dict)
        self.assertIn("builders", result)
        self.assertIn("configs", result)
        self.assertIn("summary", result)
        self.assertEqual(result["summary"]["total_components"], 0)

    def test_discover_components_with_workspace_ids(self):
        """Test component discovery with specific workspace IDs."""
        # Updated for new adapter API
        discovery_manager = WorkspaceDiscoveryManager(self.temp_workspace)

        # Create developer workspace
        developers_dir = self.temp_workspace / "developers"
        developers_dir.mkdir(exist_ok=True)

        workspace_path = developers_dir / "developer_1"
        workspace_path.mkdir(exist_ok=True)

        result = discovery_manager.discover_components(workspace_ids=["developer_1"])

        self.assertIsInstance(result, dict)
        self.assertIn("summary", result)

    def test_discover_components_with_developer_id(self):
        """Test component discovery with specific developer ID."""
        # Updated for new adapter API
        discovery_manager = WorkspaceDiscoveryManager(self.temp_workspace)

        result = discovery_manager.discover_components(developer_id="developer_1")

        self.assertIsInstance(result, dict)
        self.assertIn("summary", result)

    def test_resolve_cross_workspace_dependencies_simple(self):
        """Test resolving cross-workspace dependencies."""
        # Updated for new adapter API
        discovery_manager = WorkspaceDiscoveryManager(self.temp_workspace)

        pipeline_definition = {
            "steps": [
                {
                    "step_name": "step1",
                    "developer_id": "developer_1",
                    "dependencies": [],
                },
                {
                    "step_name": "step2",
                    "developer_id": "developer_2",
                    "dependencies": ["step1"],
                },
            ]
        }

        result = discovery_manager.resolve_cross_workspace_dependencies(
            pipeline_definition
        )

        self.assertIsInstance(result, dict)
        self.assertIn("resolved_dependencies", result)
        self.assertIn("dependency_graph", result)

    def test_resolve_cross_workspace_dependencies_circular(self):
        """Test resolving cross-workspace dependencies with circular dependency."""
        # Updated for new adapter API
        discovery_manager = WorkspaceDiscoveryManager(self.temp_workspace)

        pipeline_definition = {
            "steps": [
                {
                    "step_name": "step1",
                    "developer_id": "developer_1",
                    "dependencies": ["step2"],
                },
                {
                    "step_name": "step2",
                    "developer_id": "developer_2",
                    "dependencies": ["step1"],
                },
            ]
        }

        result = discovery_manager.resolve_cross_workspace_dependencies(
            pipeline_definition
        )

        self.assertIsInstance(result, dict)
        self.assertIn("dependency_graph", result)
        # Check that dependency graph was created with circular dependencies
        dep_graph = result["dependency_graph"]
        self.assertIn("nodes", dep_graph)
        self.assertIn("edges", dep_graph)
        # Should have detected the circular structure in the graph
        self.assertGreater(len(dep_graph["edges"]), 0)

    def test_get_file_resolver(self):
        """Test getting file resolver."""
        # Updated for new adapter API
        discovery_manager = WorkspaceDiscoveryManager(self.temp_workspace)

        # Create proper workspace structure
        developers_dir = self.temp_workspace / "developers"
        developers_dir.mkdir(exist_ok=True)
        dev_workspace = developers_dir / "developer_1"
        dev_workspace.mkdir(exist_ok=True)

        # Create the expected cursus_dev structure
        cursus_dev_path = dev_workspace / "src" / "cursus_dev" / "steps"
        cursus_dev_path.mkdir(parents=True, exist_ok=True)

        # Create component directories
        for component_dir in ["builders", "contracts", "specs", "scripts", "configs"]:
            (cursus_dev_path / component_dir).mkdir(exist_ok=True)

        file_resolver = discovery_manager.get_file_resolver("developer_1")

        self.assertIsNotNone(file_resolver)
        # Should be a DeveloperWorkspaceFileResolver instance
        self.assertEqual(file_resolver.workspace_root, self.temp_workspace)
        # Should have the expected directory attributes
        self.assertTrue(hasattr(file_resolver, "contracts_dir"))
        self.assertTrue(hasattr(file_resolver, "specs_dir"))
        self.assertTrue(hasattr(file_resolver, "builders_dir"))

    def test_get_module_loader(self):
        """Test getting module loader."""
        # Updated for new adapter API
        discovery_manager = WorkspaceDiscoveryManager(self.temp_workspace)

        # Create proper workspace structure
        developers_dir = self.temp_workspace / "developers"
        developers_dir.mkdir(exist_ok=True)
        (developers_dir / "developer_1").mkdir(exist_ok=True)

        module_loader = discovery_manager.get_module_loader("developer_1")

        self.assertIsNotNone(module_loader)
        # Should be a WorkspaceModuleLoader instance
        self.assertEqual(module_loader.workspace_root, self.temp_workspace)

    def test_list_available_developers(self):
        """Test listing available developers."""
        # Updated for new adapter API
        discovery_manager = WorkspaceDiscoveryManager(self.temp_workspace)

        # Create developer workspaces
        developers_dir = self.temp_workspace / "developers"
        developers_dir.mkdir(exist_ok=True)

        for dev_id in ["developer_1", "developer_2", "developer_3"]:
            (developers_dir / dev_id).mkdir(exist_ok=True)

        # Create shared workspace
        shared_dir = self.temp_workspace / "shared"
        shared_dir.mkdir(exist_ok=True)

        developers = discovery_manager.list_available_developers()

        self.assertIsInstance(developers, list)
        self.assertIn("developer_1", developers)
        self.assertIn("developer_2", developers)
        self.assertIn("developer_3", developers)
        self.assertIn("shared", developers)
        self.assertEqual(len(developers), 4)

    def test_get_workspace_info_specific(self):
        """Test getting workspace info for specific workspace."""
        # Updated for new adapter API
        discovery_manager = WorkspaceDiscoveryManager(self.temp_workspace)

        # Create developer workspace
        developers_dir = self.temp_workspace / "developers"
        developers_dir.mkdir(exist_ok=True)

        workspace_path = developers_dir / "developer_1"
        workspace_path.mkdir(exist_ok=True)

        info = discovery_manager.get_workspace_info(workspace_id="developer_1")

        self.assertIsInstance(info, dict)
        self.assertEqual(info["workspace_id"], "developer_1")
        self.assertEqual(info["workspace_type"], "developer")

    def test_get_workspace_info_all(self):
        """Test getting workspace info for all workspaces."""
        # Updated for new adapter API
        discovery_manager = WorkspaceDiscoveryManager(self.temp_workspace)

        # Create developer workspace
        developers_dir = self.temp_workspace / "developers"
        developers_dir.mkdir(exist_ok=True)
        (developers_dir / "developer_1").mkdir(exist_ok=True)

        info = discovery_manager.get_workspace_info()

        self.assertIsInstance(info, dict)
        self.assertIn("workspaces", info)
        self.assertIn("summary", info)

    def test_refresh_cache(self):
        """Test refreshing discovery cache."""
        # Updated for new adapter API
        discovery_manager = WorkspaceDiscoveryManager(self.temp_workspace)

        # Add test data to caches
        discovery_manager._component_cache["test"] = ComponentInventory()
        discovery_manager._dependency_cache["test"] = DependencyGraph()
        discovery_manager._cache_timestamp["test"] = 123456789

        # Refresh cache
        discovery_manager.refresh_cache()

        # All caches should be empty
        self.assertEqual(len(discovery_manager._component_cache), 0)
        self.assertEqual(len(discovery_manager._dependency_cache), 0)
        self.assertEqual(len(discovery_manager._cache_timestamp), 0)

    def test_get_discovery_summary(self):
        """Test getting discovery summary."""
        # Updated for new adapter API
        discovery_manager = WorkspaceDiscoveryManager(self.temp_workspace)

        # Add test data
        discovery_manager._component_cache["developer_1"] = ComponentInventory()
        discovery_manager._cache_timestamp["developer_1"] = 123456789

        summary = discovery_manager.get_discovery_summary()

        self.assertIsInstance(summary, dict)
        self.assertIn("cached_discoveries", summary)
        self.assertIn("cache_entries", summary)
        self.assertIn("available_developers", summary)
        self.assertEqual(summary["cached_discoveries"], 1)

    def test_get_statistics(self):
        """Test getting discovery statistics."""
        # Updated for new adapter API
        discovery_manager = WorkspaceDiscoveryManager(self.temp_workspace)

        stats = discovery_manager.get_statistics()

        self.assertIsInstance(stats, dict)
        self.assertIn("discovery_operations", stats)
        self.assertIn("component_summary", stats)
        self.assertIn("discovery_summary", stats)

    def test_component_inventory_creation(self):
        """Test ComponentInventory creation and usage."""
        inventory = ComponentInventory()

        self.assertIsInstance(inventory.builders, dict)
        self.assertIsInstance(inventory.configs, dict)
        self.assertIsInstance(inventory.summary, dict)
        self.assertEqual(inventory.summary["total_components"], 0)

        # Add a component
        component_info = {"developer_id": "developer_1", "step_type": "processing"}
        inventory.add_component("builders", "test_builder", component_info)

        self.assertEqual(inventory.summary["total_components"], 1)
        self.assertIn("developer_1", inventory.summary["developers"])
        self.assertIn("processing", inventory.summary["step_types"])

    def test_dependency_graph_creation(self):
        """Test DependencyGraph creation and usage."""
        graph = DependencyGraph()

        self.assertIsInstance(graph.nodes, set)
        self.assertIsInstance(graph.edges, list)
        self.assertEqual(len(graph.nodes), 0)

        # Add components and dependencies
        graph.add_component("step1", {"type": "builder"})
        graph.add_component("step2", {"type": "processor"})
        graph.add_dependency("step1", "step2")

        self.assertEqual(len(graph.nodes), 2)
        self.assertEqual(len(graph.edges), 1)
        self.assertIn("step1", graph.nodes)
        self.assertIn("step2", graph.nodes)

        # Test dependency queries
        deps = graph.get_dependencies("step1")
        self.assertIn("step2", deps)

        dependents = graph.get_dependents("step2")
        self.assertIn("step1", dependents)

    def test_dependency_graph_circular_detection(self):
        """Test circular dependency detection."""
        graph = DependencyGraph()

        # Create circular dependency
        graph.add_component("step1")
        graph.add_component("step2")
        graph.add_dependency("step1", "step2")
        graph.add_dependency("step2", "step1")

        self.assertTrue(graph.has_circular_dependencies())

        # Test non-circular graph
        graph2 = DependencyGraph()
        graph2.add_component("step1")
        graph2.add_component("step2")
        graph2.add_dependency("step1", "step2")

        self.assertFalse(graph2.has_circular_dependencies())

    def test_cache_validation(self):
        """Test cache validation logic."""
        # Updated for new adapter API
        discovery_manager = WorkspaceDiscoveryManager(self.temp_workspace)

        # Test invalid cache (not present)
        self.assertFalse(discovery_manager._is_cache_valid("nonexistent"))

        # Test valid cache (recent)
        import time

        discovery_manager._cache_timestamp["test"] = time.time()
        self.assertTrue(discovery_manager._is_cache_valid("test"))

        # Test expired cache
        discovery_manager._cache_timestamp["expired"] = (
            time.time() - 400
        )  # Older than cache_expiry
        self.assertFalse(discovery_manager._is_cache_valid("expired"))

    def test_error_handling_no_workspace_root(self):
        """Test error handling when no workspace root is configured."""
        # Updated for new adapter API - test with None workspace_root
        discovery_manager = WorkspaceDiscoveryManager(None)

        with self.assertRaises(ValueError):
            discovery_manager.get_file_resolver("developer_1")

        with self.assertRaises(ValueError):
            discovery_manager.get_module_loader("developer_1")

        # discover_components should handle this gracefully
        result = discovery_manager.discover_components()
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
