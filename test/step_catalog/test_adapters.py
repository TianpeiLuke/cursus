"""
Tests for backward compatibility adapters.

This module tests the legacy adapters that maintain compatibility with
the 16+ discovery systems being consolidated.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from cursus.step_catalog.adapters.contract_adapter import (
    ContractDiscoveryEngineAdapter,
    ContractDiscoveryManagerAdapter,
)
from cursus.step_catalog.adapters.file_resolver import (
    FlexibleFileResolverAdapter,
    DeveloperWorkspaceFileResolverAdapter,
    HybridFileResolverAdapter,
)
from cursus.step_catalog.adapters.workspace_discovery import (
    WorkspaceDiscoveryManagerAdapter,
)
from cursus.step_catalog.adapters.legacy_wrappers import (
    LegacyDiscoveryWrapper,
)
from cursus.step_catalog.models import StepInfo, FileMetadata
from datetime import datetime


@pytest.fixture
def mock_workspace_root():
    """Mock workspace root directory."""
    return Path("/mock/workspace")


@pytest.fixture
def mock_step_info():
    """Mock step info with file components."""
    return StepInfo(
        step_name="test_step",
        workspace_id="core",
        registry_data={"config_class": "TestConfig"},
        file_components={
            "script": FileMetadata(
                path=Path("/mock/script.py"),
                file_type="script",
                modified_time=datetime.now()
            ),
            "contract": FileMetadata(
                path=Path("/mock/contract.py"),
                file_type="contract",
                modified_time=datetime.now()
            ),
            "spec": FileMetadata(
                path=Path("/mock/spec.py"),
                file_type="spec",
                modified_time=datetime.now()
            ),
        }
    )


class TestContractDiscoveryEngineAdapter:
    """Test ContractDiscoveryEngineAdapter backward compatibility."""
    
    def test_discover_all_contracts(self, mock_workspace_root, mock_step_info):
        """Test discovering all contracts."""
        adapter = ContractDiscoveryEngineAdapter(mock_workspace_root)
        
        # Mock the catalog methods
        adapter.catalog.list_available_steps = Mock(return_value=["test_step", "other_step"])
        adapter.catalog.get_step_info = Mock(side_effect=lambda name: mock_step_info if name == "test_step" else None)
        
        contracts = adapter.discover_all_contracts()
        
        assert "test_step" in contracts
        assert len(contracts) == 1
        adapter.catalog.list_available_steps.assert_called_once()
    
    def test_discover_contracts_with_scripts(self, mock_workspace_root, mock_step_info):
        """Test discovering contracts that have scripts."""
        adapter = ContractDiscoveryEngineAdapter(mock_workspace_root)
        
        # Mock the built-in method directly instead of trying to mock catalog methods
        adapter.catalog.discover_contracts_with_scripts = Mock(return_value=["test_step"])
        
        contracts = adapter.discover_contracts_with_scripts()
        
        assert "test_step" in contracts
        assert len(contracts) == 1
    
    def test_extract_contract_reference_from_spec(self, mock_workspace_root, mock_step_info):
        """Test extracting contract reference from spec."""
        adapter = ContractDiscoveryEngineAdapter(mock_workspace_root)
        
        adapter.catalog.find_step_by_component = Mock(return_value="test_step")
        adapter.catalog.get_step_info = Mock(return_value=mock_step_info)
        
        result = adapter.extract_contract_reference_from_spec("/mock/spec.py")
        
        assert result == "test_step"
    
    def test_build_entry_point_mapping(self, mock_workspace_root, mock_step_info):
        """Test building entry point mapping."""
        adapter = ContractDiscoveryEngineAdapter(mock_workspace_root)
        
        adapter.catalog.list_available_steps = Mock(return_value=["test_step"])
        adapter.catalog.get_step_info = Mock(return_value=mock_step_info)
        
        mapping = adapter.build_entry_point_mapping()
        
        assert "test_step" in mapping
        assert mapping["test_step"] == str(mock_step_info.file_components["script"].path)


class TestContractDiscoveryManagerAdapter:
    """Test ContractDiscoveryManagerAdapter backward compatibility."""
    
    def test_discover_contract(self, mock_workspace_root, mock_step_info):
        """Test discovering contract for specific step."""
        adapter = ContractDiscoveryManagerAdapter(mock_workspace_root)
        
        # Mock both get_step_info and load_contract_class to match new architecture
        adapter.catalog.get_step_info = Mock(return_value=mock_step_info)
        adapter.catalog.load_contract_class = Mock(return_value=None)  # No actual contract loaded in test
        
        result = adapter.discover_contract("test_step")
        
        # With our new architecture, if no contract is loaded, it returns None
        # But the method should still try to get step info for backward compatibility path
        assert result is None
        adapter.catalog.load_contract_class.assert_called_once_with("test_step")
    
    def test_discover_contract_not_found(self, mock_workspace_root):
        """Test discovering contract when step not found."""
        adapter = ContractDiscoveryManagerAdapter(mock_workspace_root)
        
        adapter.catalog.get_step_info = Mock(return_value=None)
        
        result = adapter.discover_contract("nonexistent_step")
        
        assert result is None
    
    def test_get_contract_input_paths(self, mock_workspace_root):
        """Test getting contract input paths with correct signature."""
        adapter = ContractDiscoveryManagerAdapter(workspace_root=mock_workspace_root)
        
        # Create a mock contract object
        mock_contract = type('MockContract', (), {
            'expected_input_paths': {'input1': '/opt/ml/input/data', 'input2': '/opt/ml/input/config'}
        })()
        
        result = adapter.get_contract_input_paths(mock_contract, "test_step")
        
        assert isinstance(result, dict)
    
    def test_adapt_path_for_local_testing(self, mock_workspace_root):
        """Test path adaptation for local testing with correct signature."""
        adapter = ContractDiscoveryManagerAdapter(workspace_root=mock_workspace_root)
        
        from pathlib import Path
        result = adapter._adapt_path_for_local_testing("/opt/ml/input/data", Path("/test/base"), "input")
        
        assert isinstance(result, Path)
        assert "input" in str(result)


class TestFlexibleFileResolverAdapter:
    """Test FlexibleFileResolverAdapter backward compatibility."""
    
    def test_find_contract_file(self, mock_workspace_root, mock_step_info):
        """Test finding contract file."""
        adapter = FlexibleFileResolverAdapter(mock_workspace_root)
        
        adapter.catalog.get_step_info = Mock(return_value=mock_step_info)
        
        result = adapter.find_contract_file("test_step")
        
        assert str(result) == str(mock_step_info.file_components["contract"].path)
    
    def test_find_spec_file(self, mock_workspace_root, mock_step_info):
        """Test finding spec file."""
        adapter = FlexibleFileResolverAdapter(mock_workspace_root)
        
        adapter.catalog.get_step_info = Mock(return_value=mock_step_info)
        
        result = adapter.find_spec_file("test_step")
        
        assert result == mock_step_info.file_components["spec"].path
    
    def test_find_builder_file(self, mock_workspace_root, mock_step_info):
        """Test finding builder file when not present."""
        adapter = FlexibleFileResolverAdapter(mock_workspace_root)
        
        adapter.catalog.get_step_info = Mock(return_value=mock_step_info)
        
        result = adapter.find_builder_file("test_step")
        
        assert result is None  # No builder in mock_step_info
    
    def test_find_all_component_files(self, mock_workspace_root, mock_step_info):
        """Test finding all component files."""
        adapter = FlexibleFileResolverAdapter(mock_workspace_root)
        
        adapter.catalog.get_step_info = Mock(return_value=mock_step_info)
        
        result = adapter.find_all_component_files("test_step")
        
        assert "script" in result
        assert "contract" in result
        assert "spec" in result
        assert result["script"] == mock_step_info.file_components["script"].path


class TestDeveloperWorkspaceFileResolverAdapter:
    """Test DeveloperWorkspaceFileResolverAdapter backward compatibility."""
    
    @patch('cursus.step_catalog.adapters.file_resolver.DeveloperWorkspaceFileResolverAdapter._validate_workspace_structure')
    def test_workspace_aware_contract_discovery(self, mock_validate, mock_workspace_root, mock_step_info):
        """Test workspace-aware contract file discovery."""
        # Mock the validation to pass
        mock_validate.return_value = None
        
        # Create workspace-specific step info
        workspace_step_info = StepInfo(
            step_name="test_step",
            workspace_id="project_alpha",
            registry_data={},
            file_components={
                "contract": FileMetadata(
                    path=Path("/workspace/contract.py"),
                    file_type="contract",
                    modified_time=datetime.now()
                )
            }
        )
        
        adapter = DeveloperWorkspaceFileResolverAdapter(mock_workspace_root, "project_alpha")
        
        adapter.catalog.list_available_steps = Mock(return_value=["test_step"])
        adapter.catalog.get_step_info = Mock(return_value=workspace_step_info)
        
        result = adapter.find_contract_file("test_step")
        
        assert str(result) == str(workspace_step_info.file_components["contract"].path)
    
    @patch('cursus.step_catalog.adapters.file_resolver.DeveloperWorkspaceFileResolverAdapter._validate_workspace_structure')
    def test_fallback_to_core(self, mock_validate, mock_workspace_root, mock_step_info):
        """Test fallback to core when workspace step not found."""
        # Mock the validation to pass
        mock_validate.return_value = None
        
        adapter = DeveloperWorkspaceFileResolverAdapter(mock_workspace_root, "project_alpha")
        
        adapter.catalog.list_available_steps = Mock(return_value=[])  # No workspace steps
        adapter.catalog.get_step_info = Mock(return_value=mock_step_info)
        
        result = adapter.find_contract_file("test_step")
        
        assert str(result) == str(mock_step_info.file_components["contract"].path)


class TestWorkspaceDiscoveryManagerAdapter:
    """Test WorkspaceDiscoveryManagerAdapter backward compatibility."""
    
    def test_discover_workspaces(self, mock_workspace_root):
        """Test discovering available workspaces with correct signature."""
        adapter = WorkspaceDiscoveryManagerAdapter(mock_workspace_root)
        
        # The method requires workspace_root parameter
        result = adapter.discover_workspaces(mock_workspace_root)
        
        assert isinstance(result, dict)
        assert "workspace_root" in result
        assert "workspaces" in result
        assert "summary" in result
    
    def test_discover_components(self, mock_workspace_root):
        """Test discovering components in workspace."""
        adapter = WorkspaceDiscoveryManagerAdapter(mock_workspace_root)
        
        # The method returns an inventory dictionary, not a simple list
        components = adapter.discover_components(["core"])
        
        assert isinstance(components, dict)
        # The method returns an inventory structure with builders, scripts, etc.
        assert "builders" in components
        assert "scripts" in components
        assert "contracts" in components
    
    def test_resolve_cross_workspace_dependencies(self, mock_workspace_root, mock_step_info):
        """Test resolving cross-workspace dependencies with correct signature."""
        adapter = WorkspaceDiscoveryManagerAdapter(mock_workspace_root)
        
        # The method expects a pipeline definition dictionary
        pipeline_definition = {
            "steps": [
                {
                    "step_name": "test_step",
                    "workspace_id": "core",
                    "dependencies": []
                }
            ]
        }
        
        result = adapter.resolve_cross_workspace_dependencies(pipeline_definition)
        
        assert isinstance(result, dict)
        assert "pipeline_definition" in result
        assert "resolved_dependencies" in result
        assert "dependency_graph" in result


class TestHybridFileResolverAdapter:
    """Test HybridFileResolverAdapter backward compatibility."""
    
    def test_resolve_file_pattern(self, mock_workspace_root, mock_step_info):
        """Test resolving files matching pattern."""
        adapter = HybridFileResolverAdapter(mock_workspace_root)
        
        adapter.catalog.list_available_steps = Mock(return_value=["test_step", "other_step"])
        adapter.catalog.get_step_info = Mock(side_effect=lambda name: mock_step_info if name == "test_step" else None)
        
        results = adapter.resolve_file_pattern("test", "contract")
        
        assert len(results) == 1
        assert results[0] == mock_step_info.file_components["contract"].path


class TestLegacyDiscoveryWrapper:
    """Test LegacyDiscoveryWrapper integration."""
    
    def test_initialization(self, mock_workspace_root):
        """Test wrapper initialization with all adapters."""
        wrapper = LegacyDiscoveryWrapper(mock_workspace_root)
        
        assert wrapper.contract_discovery_engine is not None
        assert wrapper.contract_discovery_manager is not None
        assert wrapper.flexible_file_resolver is not None
        assert wrapper.workspace_discovery_manager is not None
        assert wrapper.hybrid_file_resolver is not None
        assert wrapper.catalog is not None
    
    def test_get_adapter(self, mock_workspace_root):
        """Test getting specific adapter by type."""
        wrapper = LegacyDiscoveryWrapper(mock_workspace_root)
        
        engine = wrapper.get_adapter('contract_discovery_engine')
        manager = wrapper.get_adapter('contract_discovery_manager')
        resolver = wrapper.get_adapter('flexible_file_resolver')
        
        assert isinstance(engine, ContractDiscoveryEngineAdapter)
        assert isinstance(manager, ContractDiscoveryManagerAdapter)
        assert isinstance(resolver, FlexibleFileResolverAdapter)
    
    def test_get_nonexistent_adapter(self, mock_workspace_root):
        """Test getting nonexistent adapter returns None."""
        wrapper = LegacyDiscoveryWrapper(mock_workspace_root)
        
        result = wrapper.get_adapter('nonexistent_adapter')
        
        assert result is None
    
    def test_get_unified_catalog(self, mock_workspace_root):
        """Test getting underlying unified catalog."""
        wrapper = LegacyDiscoveryWrapper(mock_workspace_root)
        
        catalog = wrapper.get_unified_catalog()
        
        assert catalog is wrapper.catalog


class TestAdapterErrorHandling:
    """Test error handling in adapters."""
    
    def test_contract_discovery_engine_error_handling(self, mock_workspace_root):
        """Test error handling in ContractDiscoveryEngineAdapter."""
        adapter = ContractDiscoveryEngineAdapter(mock_workspace_root)
        
        # Mock catalog to raise exception
        adapter.catalog.list_available_steps = Mock(side_effect=Exception("Test error"))
        
        contracts = adapter.discover_all_contracts()
        
        assert contracts == []  # Should return empty list on error
    
    def test_flexible_file_resolver_error_handling(self, mock_workspace_root):
        """Test error handling in FlexibleFileResolverAdapter."""
        adapter = FlexibleFileResolverAdapter(mock_workspace_root)
        
        # Mock catalog to raise exception
        adapter.catalog.get_step_info = Mock(side_effect=Exception("Test error"))
        
        result = adapter.find_contract_file("test_step")
        
        assert result is None  # Should return None on error
    
    def test_workspace_discovery_manager_error_handling(self, mock_workspace_root):
        """Test error handling in WorkspaceDiscoveryManagerAdapter."""
        adapter = WorkspaceDiscoveryManagerAdapter(mock_workspace_root)
        
        # Mock catalog to raise exception
        adapter.catalog.list_available_steps = Mock(side_effect=Exception("Test error"))
        
        components = adapter.discover_components("core")
        
        # The method returns an error dictionary, not an empty list
        assert isinstance(components, dict)
        assert "error" in components


class TestAdapterIntegration:
    """Integration tests for adapters with real StepCatalog."""
    
    def test_contract_discovery_engine_integration(self, mock_workspace_root):
        """Test ContractDiscoveryEngineAdapter integration."""
        adapter = ContractDiscoveryEngineAdapter(mock_workspace_root)
        
        # Mock the adapter's catalog directly instead of trying to patch the class
        adapter.catalog.list_available_steps = Mock(return_value=["test_step"])
        adapter.catalog.get_step_info = Mock(return_value=StepInfo(
            step_name="test_step",
            workspace_id="core",
            registry_data={},
            file_components={
                "contract": FileMetadata(
                    path=Path("/test/contract.py"),
                    file_type="contract",
                    modified_time=datetime.now()
                )
            }
        ))
        
        contracts = adapter.discover_all_contracts()
        
        assert "test_step" in contracts
        adapter.catalog.list_available_steps.assert_called()
        adapter.catalog.get_step_info.assert_called_with("test_step")
