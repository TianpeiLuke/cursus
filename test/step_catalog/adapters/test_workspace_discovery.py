"""
Unit tests for step_catalog.adapters.workspace_discovery module.

Tests the WorkspaceDiscoveryManagerAdapter class that provides backward 
compatibility with legacy workspace discovery systems.
"""

import pytest
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional, List

from cursus.step_catalog.adapters.workspace_discovery import (
    WorkspaceDiscoveryManagerAdapter
)


class TestWorkspaceDiscoveryManagerAdapter:
    """Test WorkspaceDiscoveryManagerAdapter functionality."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace with structure for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir)
            
            # Create workspace structure
            dev_workspace = workspace_root / "dev1"
            dev_workspace.mkdir()
            
            # Create component directories
            (dev_workspace / "contracts").mkdir()
            (dev_workspace / "specs").mkdir()
            (dev_workspace / "builders").mkdir()
            (dev_workspace / "scripts").mkdir()
            (dev_workspace / "configs").mkdir()
            
            # Create some test files
            (dev_workspace / "contracts" / "test_step_contract.py").write_text("# Test contract")
            (dev_workspace / "specs" / "test_step_spec.py").write_text("# Test spec")
            (dev_workspace / "builders" / "builder_test_step.py").write_text("# Test builder")
            
            # Create shared workspace
            shared_workspace = workspace_root / "shared"
            shared_workspace.mkdir()
            (shared_workspace / "contracts").mkdir()
            (shared_workspace / "contracts" / "shared_contract.py").write_text("# Shared contract")
            
            yield workspace_root
    
    def test_init(self, temp_workspace):
        """Test WorkspaceDiscoveryManagerAdapter initialization."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog') as mock_catalog:
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            
            # Should initialize StepCatalog with workspace directories
            mock_catalog.assert_called_once()
            args, kwargs = mock_catalog.call_args
            assert 'workspace_dirs' in kwargs
            
            assert adapter.workspace_root == temp_workspace
            assert adapter.logger is not None
            assert isinstance(adapter._component_cache, dict)
            assert isinstance(adapter._dependency_cache, dict)
            assert isinstance(adapter._cache_timestamp, dict)
            assert adapter.cache_expiry == 300
    
    def test_discover_workspaces_success(self, temp_workspace):
        """Test successful workspace discovery."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.workspace_dirs = [temp_workspace / "dev1", temp_workspace / "shared"]
            mock_catalog_class.return_value = mock_catalog
            
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            result = adapter.discover_workspaces(temp_workspace)
            
            assert isinstance(result, dict)
            assert "workspace_root" in result
            assert "workspaces" in result
            assert "summary" in result
            
            assert result["workspace_root"] == str(temp_workspace)
            assert len(result["workspaces"]) == 2
            
            # Check summary
            summary = result["summary"]
            assert "total_workspaces" in summary
            assert "workspace_types" in summary
            assert "total_developers" in summary
            assert "total_components" in summary
    
    def test_discover_workspaces_error_handling(self, temp_workspace):
        """Test error handling in workspace discovery."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.workspace_dirs = []
            mock_catalog_class.side_effect = Exception("Test error")
            
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            result = adapter.discover_workspaces(temp_workspace)
            
            assert "error" in result
    
    def test_count_workspace_components(self, temp_workspace):
        """Test counting components in workspace."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog'):
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            
            dev_workspace = temp_workspace / "dev1"
            count = adapter._count_workspace_components(dev_workspace)
            
            # Should count the test files we created
            assert count >= 3  # At least contract, spec, and builder files
    
    def test_count_workspace_components_nonexistent(self, temp_workspace):
        """Test counting components in nonexistent workspace."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog'):
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            
            nonexistent = temp_workspace / "nonexistent"
            count = adapter._count_workspace_components(nonexistent)
            
            assert count == 0
    
    def test_discover_components_no_constraints(self, temp_workspace):
        """Test component discovery with no workspace constraints."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.list_available_steps.return_value = ["test_step"]
            
            mock_step_info = Mock()
            mock_step_info.workspace_id = "core"
            mock_step_info.step_name = "test_step"
            mock_step_info.file_components = {
                "script": Mock(path=Path("/path/to/script.py")),
                "contract": Mock(path=Path("/path/to/contract.py"))
            }
            mock_step_info.registry_data = {}
            mock_catalog.get_step_info.return_value = mock_step_info
            mock_catalog_class.return_value = mock_catalog
            
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            result = adapter.discover_components()
            
            assert isinstance(result, dict)
            assert "builders" in result
            assert "configs" in result
            assert "contracts" in result
            assert "specs" in result
            assert "scripts" in result
            assert "metadata" in result
            
            metadata = result["metadata"]
            assert "discovery_timestamp" in metadata
            assert "total_components" in metadata
            assert "workspaces_scanned" in metadata
            assert "component_counts" in metadata
    
    def test_discover_components_with_workspace_ids(self, temp_workspace):
        """Test component discovery with specific workspace IDs."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.list_available_steps.return_value = ["test_step"]
            
            mock_step_info = Mock()
            mock_step_info.workspace_id = "dev1"
            mock_step_info.step_name = "test_step"
            mock_step_info.file_components = {
                "contract": Mock(path=Path("/path/to/contract.py"))
            }
            mock_step_info.registry_data = {}
            mock_catalog.get_step_info.return_value = mock_step_info
            mock_catalog_class.return_value = mock_catalog
            
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            result = adapter.discover_components(workspace_ids=["dev1"])
            
            assert isinstance(result, dict)
            assert "metadata" in result
            assert "dev1" in result["metadata"]["workspaces_scanned"]
    
    def test_discover_components_with_developer_id(self, temp_workspace):
        """Test component discovery with developer ID."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.list_available_steps.return_value = []
            mock_catalog_class.return_value = mock_catalog
            
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            result = adapter.discover_components(developer_id="dev1")
            
            assert isinstance(result, dict)
            assert "metadata" in result
    
    def test_discover_components_no_workspace_root(self):
        """Test component discovery with no workspace root."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog'):
            adapter = WorkspaceDiscoveryManagerAdapter(Path("/nonexistent"))
            adapter.workspace_root = None
            
            result = adapter.discover_components()
            
            assert "error" in result
            assert "No workspace root configured" in result["error"]
    
    def test_discover_components_error_handling(self, temp_workspace):
        """Test error handling in component discovery."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.list_available_steps.side_effect = Exception("Test error")
            mock_catalog_class.return_value = mock_catalog
            
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            result = adapter.discover_components()
            
            assert "error" in result
    
    def test_discover_step_components(self, temp_workspace):
        """Test discovering components for a specific step."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog'):
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            
            # Create mock step info
            mock_step_info = Mock()
            mock_step_info.workspace_id = "dev1"
            mock_step_info.step_name = "test_step"
            mock_step_info.file_components = {
                "contract": Mock(path=Path("/path/to/contract.py")),
                "spec": Mock(path=Path("/path/to/spec.py"))
            }
            mock_step_info.registry_data = {"test": "data"}
            
            # Create inventory
            inventory = {
                "contracts": {},
                "specs": {},
                "builders": {},
                "configs": {},
                "scripts": {},
                "metadata": {
                    "component_counts": {
                        "contracts": 0,
                        "specs": 0,
                        "builders": 0,
                        "configs": 0,
                        "scripts": 0
                    }
                }
            }
            
            adapter._discover_step_components(mock_step_info, inventory)
            
            # Check that components were added
            assert "dev1" in inventory["contracts"]
            assert "dev1" in inventory["specs"]
            assert inventory["metadata"]["component_counts"]["contracts"] == 1
            assert inventory["metadata"]["component_counts"]["specs"] == 1
    
    def test_discover_step_components_error_handling(self, temp_workspace):
        """Test error handling in step component discovery."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog'):
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            
            # Create mock step info that will cause an error
            mock_step_info = Mock()
            mock_step_info.workspace_id = None  # This will cause an error
            mock_step_info.step_name = "test_step"
            
            inventory = {
                "contracts": {},
                "metadata": {"component_counts": {"contracts": 0}}
            }
            
            # Should handle error gracefully
            adapter._discover_step_components(mock_step_info, inventory)
            
            # Inventory should remain unchanged
            assert len(inventory["contracts"]) == 0
    
    def test_discover_filesystem_components(self, temp_workspace):
        """Test filesystem component discovery."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.package_root = temp_workspace
            mock_catalog.workspace_dirs = [temp_workspace / "dev1"]
            mock_catalog_class.return_value = mock_catalog
            
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            
            inventory = {
                "contracts": {},
                "specs": {},
                "builders": {},
                "configs": {},
                "scripts": {},
                "metadata": {
                    "component_counts": {
                        "contracts": 0,
                        "specs": 0,
                        "builders": 0,
                        "configs": 0,
                        "scripts": 0
                    }
                }
            }
            
            adapter._discover_filesystem_components(["dev1"], inventory)
            
            # Should find components in the filesystem
            assert inventory["metadata"]["component_counts"]["contracts"] > 0
    
    def test_discover_filesystem_components_core_workspace(self, temp_workspace):
        """Test filesystem component discovery for core workspace."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.package_root = temp_workspace
            mock_catalog.workspace_dirs = []
            mock_catalog_class.return_value = mock_catalog
            
            # Create core workspace structure
            core_workspace = temp_workspace / "steps"
            core_workspace.mkdir()
            (core_workspace / "contracts").mkdir()
            (core_workspace / "contracts" / "core_contract.py").write_text("# Core contract")
            
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            
            inventory = {
                "contracts": {},
                "metadata": {"component_counts": {"contracts": 0}}
            }
            
            adapter._discover_filesystem_components(["core"], inventory)
            
            # Should find core components
            assert inventory["metadata"]["component_counts"]["contracts"] > 0
    
    def test_find_workspace_path(self, temp_workspace):
        """Test finding workspace path."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.workspace_dirs = [temp_workspace / "dev1", temp_workspace / "shared"]
            mock_catalog_class.return_value = mock_catalog
            
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            
            # Test finding existing workspace
            path = adapter._find_workspace_path("dev1")
            assert path == temp_workspace / "dev1"
            
            # Test finding nonexistent workspace
            path = adapter._find_workspace_path("nonexistent")
            assert path is None
    
    def test_find_workspace_path_error_handling(self, temp_workspace):
        """Test error handling in workspace path finding."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.workspace_dirs = []
            mock_catalog_class.return_value = mock_catalog
            
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            
            # Mock workspace_dirs to cause an error
            adapter.catalog.workspace_dirs = Mock()
            adapter.catalog.workspace_dirs.__iter__.side_effect = Exception("Test error")
            
            path = adapter._find_workspace_path("dev1")
            assert path is None
    
    def test_scan_component_directory(self, temp_workspace):
        """Test scanning component directory."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog'):
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            
            contracts_dir = temp_workspace / "dev1" / "contracts"
            inventory = {
                "contracts": {},
                "metadata": {"component_counts": {"contracts": 0}}
            }
            
            adapter._scan_component_directory(contracts_dir, "contracts", "dev1", inventory)
            
            # Should find the test contract file
            assert "dev1" in inventory["contracts"]
            assert inventory["metadata"]["component_counts"]["contracts"] > 0
    
    def test_scan_component_directory_no_duplicates(self, temp_workspace):
        """Test that scanning doesn't create duplicates."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog'):
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            
            contracts_dir = temp_workspace / "dev1" / "contracts"
            inventory = {
                "contracts": {
                    "dev1": {
                        "dev1:test_step": {"existing": "component"}
                    }
                },
                "metadata": {"component_counts": {"contracts": 1}}
            }
            
            adapter._scan_component_directory(contracts_dir, "contracts", "dev1", inventory)
            
            # Should not create duplicates
            assert len(inventory["contracts"]["dev1"]) == 1
            assert inventory["metadata"]["component_counts"]["contracts"] == 1
    
    def test_scan_component_directory_error_handling(self, temp_workspace):
        """Test error handling in component directory scanning."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog'):
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            
            nonexistent_dir = temp_workspace / "nonexistent"
            inventory = {
                "contracts": {},
                "metadata": {"component_counts": {"contracts": 0}}
            }
            
            # Should handle nonexistent directory gracefully
            adapter._scan_component_directory(nonexistent_dir, "contracts", "dev1", inventory)
            
            assert len(inventory["contracts"]) == 0
    
    def test_extract_step_name_from_file(self, temp_workspace):
        """Test extracting step name from filename."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog'):
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            
            # Test different component types
            assert adapter._extract_step_name_from_file("test_step_contract.py", "contracts") == "test_step"
            assert adapter._extract_step_name_from_file("test_step_spec.py", "specs") == "test_step"
            assert adapter._extract_step_name_from_file("builder_test_step_step.py", "builders") == "test_step"
            assert adapter._extract_step_name_from_file("config_test_step_step.py", "configs") == "test_step"
            assert adapter._extract_step_name_from_file("test_script.py", "scripts") == "test_script"
            
            # Test invalid patterns
            assert adapter._extract_step_name_from_file("invalid.py", "contracts") is None
            assert adapter._extract_step_name_from_file("test.py", "builders") is None
    
    def test_extract_step_name_from_file_error_handling(self, temp_workspace):
        """Test error handling in step name extraction."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog'):
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            
            # Test with invalid input
            result = adapter._extract_step_name_from_file("", "contracts")
            assert result is None
    
    def test_get_file_resolver(self, temp_workspace):
        """Test getting file resolver."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog'):
            with patch('cursus.step_catalog.adapters.file_resolver.DeveloperWorkspaceFileResolverAdapter') as mock_resolver:
                adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
                
                resolver = adapter.get_file_resolver("dev1")
                
                mock_resolver.assert_called_once_with(temp_workspace, project_id="dev1")
                assert resolver == mock_resolver.return_value
    
    def test_get_file_resolver_no_workspace_root(self):
        """Test getting file resolver with no workspace root."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog'):
            adapter = WorkspaceDiscoveryManagerAdapter(Path("/nonexistent"))
            adapter.workspace_root = None
            
            with pytest.raises(ValueError, match="No workspace root configured"):
                adapter.get_file_resolver()
    
    def test_get_module_loader(self, temp_workspace):
        """Test getting module loader."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog'):
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            
            loader = adapter.get_module_loader("dev1")
            
            # Should return a mock loader
            assert loader is not None
            assert loader.workspace_root == temp_workspace
    
    def test_get_module_loader_no_workspace_root(self):
        """Test getting module loader with no workspace root."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog'):
            adapter = WorkspaceDiscoveryManagerAdapter(Path("/nonexistent"))
            adapter.workspace_root = None
            
            with pytest.raises(ValueError, match="No workspace root configured"):
                adapter.get_module_loader()
    
    def test_list_available_developers(self, temp_workspace):
        """Test listing available developers."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.workspace_dirs = [temp_workspace / "dev1", temp_workspace / "dev2", temp_workspace / "shared"]
            mock_catalog_class.return_value = mock_catalog
            
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            developers = adapter.list_available_developers()
            
            assert isinstance(developers, list)
            assert "dev1" in developers
            assert "dev2" in developers
            assert "shared" in developers
            assert developers == sorted(developers)  # Should be sorted
    
    def test_list_available_developers_error_handling(self, temp_workspace):
        """Test error handling in listing developers."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.workspace_dirs = []
            mock_catalog_class.return_value = mock_catalog
            
            # Mock workspace_dirs to cause an error
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            adapter.catalog.workspace_dirs = Mock()
            adapter.catalog.workspace_dirs.__iter__.side_effect = Exception("Test error")
            
            developers = adapter.list_available_developers()
            assert developers == []
    
    def test_get_workspace_info_specific_workspace(self, temp_workspace):
        """Test getting info for specific workspace."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.workspace_dirs = [temp_workspace / "dev1"]
            mock_catalog_class.return_value = mock_catalog
            
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            info = adapter.get_workspace_info(workspace_id="dev1")
            
            assert isinstance(info, dict)
            assert info["workspace_id"] == "dev1"
            assert info["exists"] is True
            assert "workspace_path" in info
            assert "workspace_type" in info
    
    def test_get_workspace_info_nonexistent_workspace(self, temp_workspace):
        """Test getting info for nonexistent workspace."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.workspace_dirs = []
            mock_catalog_class.return_value = mock_catalog
            
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            info = adapter.get_workspace_info(workspace_id="nonexistent")
            
            assert "error" in info
            assert "Workspace not found" in info["error"]
    
    def test_get_workspace_info_all_workspaces(self, temp_workspace):
        """Test getting info for all workspaces."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.workspace_dirs = [temp_workspace / "dev1"]
            mock_catalog_class.return_value = mock_catalog
            
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            
            with patch.object(adapter, 'discover_workspaces') as mock_discover:
                mock_discover.return_value = {"test": "data"}
                
                info = adapter.get_workspace_info()
                
                mock_discover.assert_called_once_with(temp_workspace)
                assert info == {"test": "data"}
    
    def test_get_workspace_info_error_handling(self, temp_workspace):
        """Test error handling in get_workspace_info."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog') as mock_catalog_class:
            mock_catalog_class.side_effect = Exception("Test error")
            
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            info = adapter.get_workspace_info()
            
            assert "error" in info
    
    def test_refresh_cache(self, temp_workspace):
        """Test cache refresh."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog'):
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            
            # Add some cache data
            adapter._component_cache["test"] = "data"
            adapter._dependency_cache["test"] = "data"
            adapter._cache_timestamp["test"] = time.time()
            
            adapter.refresh_cache()
            
            # Cache should be cleared
            assert len(adapter._component_cache) == 0
            assert len(adapter._dependency_cache) == 0
            assert len(adapter._cache_timestamp) == 0
    
    def test_get_discovery_summary(self, temp_workspace):
        """Test getting discovery summary."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.workspace_dirs = [temp_workspace / "dev1"]
            mock_catalog_class.return_value = mock_catalog
            
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            
            # Add some cache data
            adapter._component_cache["test"] = "data"
            adapter._cache_timestamp["test"] = time.time()
            
            summary = adapter.get_discovery_summary()
            
            assert isinstance(summary, dict)
            assert "cached_discoveries" in summary
            assert "cache_entries" in summary
            assert "last_discovery" in summary
            assert "available_developers" in summary
            
            assert summary["cached_discoveries"] == 1
            assert "test" in summary["cache_entries"]
    
    def test_get_discovery_summary_error_handling(self, temp_workspace):
        """Test error handling in get_discovery_summary."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog'):
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            
            # Mock list_available_developers to cause an error
            with patch.object(adapter, 'list_available_developers', side_effect=Exception("Test error")):
                summary = adapter.get_discovery_summary()
                
                assert "error" in summary
    
    def test_get_statistics(self, temp_workspace):
        """Test getting statistics."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.workspace_dirs = [temp_workspace / "dev1"]
            mock_catalog_class.return_value = mock_catalog
            
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            
            stats = adapter.get_statistics()
            
            assert isinstance(stats, dict)
            assert "discovery_operations" in stats
            assert "component_summary" in stats
            assert "discovery_summary" in stats
            
            assert "cached_discoveries" in stats["discovery_operations"]
            assert "available_workspaces" in stats["discovery_operations"]
    
    def test_get_statistics_error_handling(self, temp_workspace):
        """Test error handling in get_statistics."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog'):
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            
            # Mock get_discovery_summary to cause an error
            with patch.object(adapter, 'get_discovery_summary', side_effect=Exception("Test error")):
                stats = adapter.get_statistics()
                
                assert "error" in stats
    
    def test_is_cache_valid(self, temp_workspace):
        """Test cache validity checking."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog'):
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            
            # Test with no cache entry
            assert adapter._is_cache_valid("nonexistent") is False
            
            # Test with fresh cache entry
            adapter._cache_timestamp["fresh"] = time.time()
            assert adapter._is_cache_valid("fresh") is True
            
            # Test with expired cache entry
            adapter._cache_timestamp["expired"] = time.time() - 400  # Older than cache_expiry (300)
            assert adapter._is_cache_valid("expired") is False


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    @pytest.fixture
    def comprehensive_workspace(self):
        """Create comprehensive workspace for integration testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir)
            
            # Create multiple developer workspaces
            for dev_id in ["dev1", "dev2"]:
                dev_workspace = workspace_root / dev_id
                dev_workspace.mkdir()
                
                # Create component directories with files
                for component_type in ["contracts", "specs", "builders", "scripts", "configs"]:
                    component_dir = dev_workspace / component_type
                    component_dir.mkdir()
                    
                    if component_type == "contracts":
                        (component_dir / f"{dev_id}_step_contract.py").write_text(f"# {dev_id} contract")
                    elif component_type == "specs":
                        (component_dir / f"{dev_id}_step_spec.py").write_text(f"# {dev_id} spec")
                    elif component_type == "builders":
                        (component_dir / f"builder_{dev_id}_step_step.py").write_text(f"# {dev_id} builder")
                    elif component_type == "scripts":
                        (component_dir / f"{dev_id}_script.py").write_text(f"# {dev_id} script")
                    elif component_type == "configs":
                        (component_dir / f"config_{dev_id}_step_step.py").write_text(f"# {dev_id} config")
            
            # Create shared workspace
            shared_workspace = workspace_root / "shared"
            shared_workspace.mkdir()
            (shared_workspace / "contracts").mkdir()
            (shared_workspace / "contracts" / "shared_contract.py").write_text("# Shared contract")
            
            yield workspace_root
    
    def test_complete_workspace_discovery_workflow(self, comprehensive_workspace):
        """Test complete workspace discovery workflow."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.workspace_dirs = [
                comprehensive_workspace / "dev1",
                comprehensive_workspace / "dev2",
                comprehensive_workspace / "shared"
            ]
            mock_catalog.list_available_steps.return_value = ["dev1_step", "dev2_step"]
            
            # Mock step info for comprehensive testing
            mock_step_info1 = Mock()
            mock_step_info1.workspace_id = "dev1"
            mock_step_info1.step_name = "dev1_step"
            mock_step_info1.file_components = {
                "contract": Mock(path=Path("/path/to/dev1_contract.py")),
                "script": Mock(path=Path("/path/to/dev1_script.py"))
            }
            mock_step_info1.registry_data = {}
            
            mock_step_info2 = Mock()
            mock_step_info2.workspace_id = "dev2"
            mock_step_info2.step_name = "dev2_step"
            mock_step_info2.file_components = {
                "spec": Mock(path=Path("/path/to/dev2_spec.py"))
            }
            mock_step_info2.registry_data = {}
            
            mock_catalog.get_step_info.side_effect = [mock_step_info1, mock_step_info2]
            mock_catalog_class.return_value = mock_catalog
            
            adapter = WorkspaceDiscoveryManagerAdapter(comprehensive_workspace)
            
            # Test workspace discovery
            workspaces = adapter.discover_workspaces(comprehensive_workspace)
            assert len(workspaces["workspaces"]) == 3
            assert workspaces["summary"]["total_workspaces"] == 3
            
            # Test component discovery
            components = adapter.discover_components()
            assert "metadata" in components
            assert components["metadata"]["total_components"] > 0
            
            # Test developer listing
            developers = adapter.list_available_developers()
            assert "dev1" in developers
            assert "dev2" in developers
            assert "shared" in developers
    
    def test_workspace_discovery_with_filesystem_fallback(self, comprehensive_workspace):
        """Test workspace discovery with filesystem fallback."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.workspace_dirs = [comprehensive_workspace / "dev1"]
            mock_catalog.package_root = comprehensive_workspace
            mock_catalog.list_available_steps.return_value = []  # No catalog steps
            mock_catalog.get_step_info.return_value = None
            mock_catalog_class.return_value = mock_catalog
            
            adapter = WorkspaceDiscoveryManagerAdapter(comprehensive_workspace)
            
            # Should discover components via filesystem scanning
            components = adapter.discover_components(workspace_ids=["dev1"])
            
            assert "metadata" in components
            # Should find components via filesystem discovery
            assert components["metadata"]["total_components"] > 0


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_adapter_with_catalog_failure(self, temp_workspace):
        """Test adapter behavior when catalog fails."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog') as mock_catalog_class:
            mock_catalog_class.side_effect = Exception("Catalog initialization failed")
            
            # Should still initialize but methods will fail gracefully
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            
            # Methods should handle catalog failures gracefully
            result = adapter.discover_workspaces(temp_workspace)
            assert "error" in result
            
            result = adapter.discover_components()
            assert "error" in result
    
    def test_component_discovery_with_invalid_step_info(self, temp_workspace):
        """Test component discovery with invalid step info."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.list_available_steps.return_value = ["invalid_step"]
            mock_catalog.get_step_info.return_value = None  # Invalid step info
            mock_catalog_class.return_value = mock_catalog
            
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            result = adapter.discover_components()
            
            # Should handle invalid step info gracefully
            assert isinstance(result, dict)
            assert "metadata" in result
    
    def test_filesystem_discovery_with_permission_errors(self, temp_workspace):
        """Test filesystem discovery with permission errors."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.workspace_dirs = [temp_workspace / "restricted"]
            mock_catalog_class.return_value = mock_catalog
            
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            
            # Mock glob to raise permission error
            with patch('pathlib.Path.glob', side_effect=PermissionError("Access denied")):
                inventory = {
                    "contracts": {},
                    "metadata": {"component_counts": {"contracts": 0}}
                }
                
                # Should handle permission errors gracefully
                adapter._scan_component_directory(temp_workspace / "restricted", "contracts", "test", inventory)
                
                assert len(inventory["contracts"]) == 0
    
    def test_cache_operations_with_time_manipulation(self, temp_workspace):
        """Test cache operations with time manipulation."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog'):
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            
            # Test cache validity with mocked time
            with patch('time.time', return_value=1000):
                adapter._cache_timestamp["test"] = 1000
                assert adapter._is_cache_valid("test") is True
            
            with patch('time.time', return_value=1400):  # 400 seconds later
                assert adapter._is_cache_valid("test") is False
    
    def test_step_name_extraction_edge_cases(self, temp_workspace):
        """Test step name extraction with edge cases."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog'):
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            
            # Test with very short filenames
            assert adapter._extract_step_name_from_file("a.py", "scripts") == "a"
            assert adapter._extract_step_name_from_file(".py", "scripts") == ""
            
            # Test with multiple underscores
            assert adapter._extract_step_name_from_file("test_step_with_underscores_contract.py", "contracts") == "test_step_with_underscores"
            
            # Test with numbers
            assert adapter._extract_step_name_from_file("step123_contract.py", "contracts") == "step123"
    
    def test_workspace_info_with_complex_paths(self, temp_workspace):
        """Test workspace info with complex path scenarios."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            # Create complex path structure
            complex_path = temp_workspace / "complex" / "nested" / "workspace"
            complex_path.mkdir(parents=True)
            mock_catalog.workspace_dirs = [complex_path]
            mock_catalog_class.return_value = mock_catalog
            
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            
            # Test finding workspace with complex path
            path = adapter._find_workspace_path("workspace")
            assert path == complex_path
            
            # Test workspace info
            info = adapter.get_workspace_info(workspace_id="workspace")
            assert info["workspace_id"] == "workspace"
            assert info["exists"] is True


class TestPerformanceAndScalability:
    """Test performance and scalability scenarios."""
    
    @pytest.fixture
    def large_workspace(self):
        """Create large workspace for performance testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir)
            
            # Create many developer workspaces
            for i in range(10):
                dev_workspace = workspace_root / f"dev{i}"
                dev_workspace.mkdir()
                
                # Create component directories with multiple files
                for component_type in ["contracts", "specs", "builders"]:
                    component_dir = dev_workspace / component_type
                    component_dir.mkdir()
                    
                    # Create multiple files per component type
                    for j in range(5):
                        if component_type == "contracts":
                            (component_dir / f"step{j}_contract.py").write_text(f"# Contract {j}")
                        elif component_type == "specs":
                            (component_dir / f"step{j}_spec.py").write_text(f"# Spec {j}")
                        elif component_type == "builders":
                            (component_dir / f"builder_step{j}_step.py").write_text(f"# Builder {j}")
            
            yield workspace_root
    
    def test_large_workspace_discovery_performance(self, large_workspace):
        """Test performance with large workspace."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.workspace_dirs = [large_workspace / f"dev{i}" for i in range(10)]
            mock_catalog.list_available_steps.return_value = []  # Force filesystem discovery
            mock_catalog_class.return_value = mock_catalog
            
            adapter = WorkspaceDiscoveryManagerAdapter(large_workspace)
            
            # Test that discovery completes in reasonable time
            import time
            start_time = time.time()
            
            result = adapter.discover_workspaces(large_workspace)
            
            end_time = time.time()
            discovery_time = end_time - start_time
            
            # Should complete within reasonable time (adjust threshold as needed)
            assert discovery_time < 5.0  # 5 seconds threshold
            assert len(result["workspaces"]) == 10
    
    def test_component_discovery_with_many_files(self, large_workspace):
        """Test component discovery with many files."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.workspace_dirs = [large_workspace / "dev0"]
            mock_catalog.package_root = large_workspace
            mock_catalog.list_available_steps.return_value = []
            mock_catalog_class.return_value = mock_catalog
            
            adapter = WorkspaceDiscoveryManagerAdapter(large_workspace)
            
            result = adapter.discover_components(workspace_ids=["dev0"])
            
            # Should discover multiple components
            assert result["metadata"]["total_components"] >= 15  # 5 files × 3 component types
            assert result["metadata"]["component_counts"]["contracts"] == 5
            assert result["metadata"]["component_counts"]["specs"] == 5
            assert result["metadata"]["component_counts"]["builders"] == 5
    
    def test_cache_efficiency_with_repeated_operations(self, temp_workspace):
        """Test cache efficiency with repeated operations."""
        with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog'):
            adapter = WorkspaceDiscoveryManagerAdapter(temp_workspace)
            
            # Simulate repeated cache operations
            for i in range(100):
                cache_key = f"test_{i % 10}"  # Reuse keys to test cache behavior
                adapter._cache_timestamp[cache_key] = time.time()
                
                # Test cache validity
                is_valid = adapter._is_cache_valid(cache_key)
                assert is_valid is True
            
            # Test cache cleanup
            adapter.refresh_cache()
            assert len(adapter._cache_timestamp) == 0
