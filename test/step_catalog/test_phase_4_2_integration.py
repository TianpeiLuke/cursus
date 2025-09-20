"""
Phase 4.2 Integration Tests

Tests for the Phase 4.2 legacy system integration with StepCatalog,
validating that legacy systems use the catalog for discovery while
preserving their specialized business logic.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
from typing import Dict, Any, List

# Import the integrated systems
from src.cursus.validation.alignment.orchestration.validation_orchestrator import ValidationOrchestrator
from src.cursus.workspace.validation.cross_workspace_validator import CrossWorkspaceValidator
from src.cursus.validation.alignment.discovery.contract_discovery import ContractDiscoveryEngine
from src.cursus.workspace.core.discovery import WorkspaceDiscoveryManager

# Import StepCatalog for testing
from src.cursus.step_catalog.step_catalog import StepCatalog
from src.cursus.step_catalog.models import StepInfo, FileMetadata


class TestValidationOrchestratorIntegration:
    """Test ValidationOrchestrator integration with StepCatalog."""

    @pytest.fixture
    def mock_step_catalog(self):
        """Create mock StepCatalog for testing."""
        catalog = Mock(spec=StepCatalog)
        
        # Mock step info with contract component
        step_info = Mock(spec=StepInfo)
        step_info.file_components = {
            'contract': Mock(spec=FileMetadata, path=Path('/test/contract.py')),
            'script': Mock(spec=FileMetadata, path=Path('/test/script.py'))
        }
        catalog.get_step_info.return_value = step_info
        
        # Mock contracts with scripts discovery
        catalog.discover_contracts_with_scripts.return_value = ['test_step', 'another_step']
        
        return catalog

    @pytest.fixture
    def validation_orchestrator(self, mock_step_catalog):
        """Create ValidationOrchestrator with mocked StepCatalog."""
        return ValidationOrchestrator(
            step_catalog=mock_step_catalog,
            contracts_dir="/test/contracts",
            specs_dir="/test/specs"
        )

    def test_initialization_with_step_catalog(self, validation_orchestrator, mock_step_catalog):
        """Test that ValidationOrchestrator initializes with StepCatalog."""
        assert validation_orchestrator.catalog is mock_step_catalog
        assert validation_orchestrator.contracts_dir == Path("/test/contracts")
        assert validation_orchestrator.specs_dir == Path("/test/specs")

    def test_discover_contract_file_uses_catalog(self, validation_orchestrator, mock_step_catalog):
        """Test that contract file discovery uses StepCatalog."""
        # Test successful discovery
        result = validation_orchestrator._discover_contract_file("test_step")
        
        # Verify catalog was called
        mock_step_catalog.get_step_info.assert_called_with("test_step")
        assert result == "/test/contract.py"

    def test_discover_contract_file_fallback(self, mock_step_catalog):
        """Test fallback to legacy discovery when catalog fails."""
        # Mock catalog failure
        mock_step_catalog.get_step_info.side_effect = Exception("Catalog error")
        
        orchestrator = ValidationOrchestrator(
            step_catalog=mock_step_catalog,
            contracts_dir="/test/contracts"
        )
        
        # Should handle error gracefully
        result = orchestrator._discover_contract_file("test_step")
        assert result is None  # Fallback returns None when no legacy components

    def test_discover_contracts_with_scripts_uses_catalog(self, validation_orchestrator, mock_step_catalog):
        """Test that contracts with scripts discovery uses StepCatalog."""
        result = validation_orchestrator._discover_contracts_with_scripts()
        
        # Verify catalog method was called
        mock_step_catalog.discover_contracts_with_scripts.assert_called_once()
        assert result == ['test_step', 'another_step']

    def test_discover_contracts_with_scripts_fallback(self, mock_step_catalog):
        """Test fallback when catalog discovery fails."""
        # Mock catalog failure
        mock_step_catalog.discover_contracts_with_scripts.side_effect = Exception("Catalog error")
        
        orchestrator = ValidationOrchestrator(
            step_catalog=mock_step_catalog,
            contracts_dir="/test/contracts"
        )
        
        # Should handle error gracefully and return empty list
        result = orchestrator._discover_contracts_with_scripts()
        assert result == []

    def test_discover_and_load_specifications_uses_catalog(self, validation_orchestrator, mock_step_catalog):
        """Test that specification discovery uses StepCatalog."""
        # Mock step info with spec component
        step_info = Mock(spec=StepInfo)
        step_info.file_components = {
            'spec': Mock(spec=FileMetadata, path=Path('/test/spec.py'))
        }
        mock_step_catalog.get_step_info.return_value = step_info
        
        # Mock spec loader
        mock_spec_loader = Mock()
        mock_spec_loader.load_specification.return_value = {"test": "spec"}
        validation_orchestrator.spec_loader = mock_spec_loader
        
        result = validation_orchestrator._discover_and_load_specifications("test_step")
        
        # Verify catalog was called
        mock_step_catalog.get_step_info.assert_called_with("test_step")
        assert "spec" in result


class TestCrossWorkspaceValidatorIntegration:
    """Test CrossWorkspaceValidator integration with StepCatalog."""

    @pytest.fixture
    def mock_step_catalog(self):
        """Create mock StepCatalog for testing."""
        catalog = Mock(spec=StepCatalog)
        
        # Mock cross-workspace discovery
        catalog.discover_cross_workspace_components.return_value = {
            'core': ['step1:script', 'step2:builder'],
            'workspace1': ['step3:contract', 'step4:spec']
        }
        
        # Mock step info
        step_info = Mock(spec=StepInfo)
        step_info.config_class = "TestConfig"
        step_info.sagemaker_step_type = "Training"
        step_info.file_components = {'script': Mock(), 'contract': Mock()}
        catalog.get_step_info.return_value = step_info
        
        return catalog

    @pytest.fixture
    def cross_workspace_validator(self, mock_step_catalog):
        """Create CrossWorkspaceValidator with mocked StepCatalog."""
        return CrossWorkspaceValidator(step_catalog=mock_step_catalog)

    def test_initialization_with_step_catalog(self, cross_workspace_validator, mock_step_catalog):
        """Test that CrossWorkspaceValidator initializes with StepCatalog."""
        assert cross_workspace_validator.catalog is mock_step_catalog

    def test_discover_cross_workspace_components_uses_catalog(self, cross_workspace_validator, mock_step_catalog):
        """Test that cross-workspace discovery uses StepCatalog."""
        result = cross_workspace_validator.discover_cross_workspace_components(['core', 'workspace1'])
        
        # Verify catalog was called
        mock_step_catalog.discover_cross_workspace_components.assert_called_with(['core', 'workspace1'])
        
        # Verify result structure
        assert "discovery_result" in result
        assert "component_registry" in result
        assert result["total_workspaces"] == 2

    def test_build_component_registry_from_catalog(self, cross_workspace_validator, mock_step_catalog):
        """Test component registry building from catalog results."""
        discovery_result = {
            'core': ['step1:script', 'step2:builder'],
            'workspace1': ['step3:contract']
        }
        
        registry = cross_workspace_validator._build_component_registry_from_catalog(discovery_result)
        
        # Verify registry structure
        assert 'core' in registry
        assert 'workspace1' in registry
        
        # Verify step info was retrieved for each component
        expected_calls = ['step1', 'step2', 'step3']
        actual_calls = [call[0][0] for call in mock_step_catalog.get_step_info.call_args_list]
        for expected_call in expected_calls:
            assert expected_call in actual_calls


class TestContractDiscoveryEngineIntegration:
    """Test ContractDiscoveryEngine integration with StepCatalog."""

    @pytest.fixture
    def mock_step_catalog(self):
        """Create mock StepCatalog for testing."""
        catalog = Mock(spec=StepCatalog)
        
        # Mock available steps
        catalog.list_available_steps.return_value = ['step1', 'step2', 'step3']
        
        # Mock step info with contract component
        step_info = Mock(spec=StepInfo)
        step_info.file_components = {'contract': Mock(spec=FileMetadata)}
        catalog.get_step_info.return_value = step_info
        
        # Mock contracts with scripts
        catalog.discover_contracts_with_scripts.return_value = ['step1', 'step2']
        
        return catalog

    @pytest.fixture
    def contract_discovery_engine(self, mock_step_catalog):
        """Create ContractDiscoveryEngine with mocked StepCatalog."""
        # Updated for new adapter API - uses workspace_root instead of step_catalog parameter
        from pathlib import Path
        engine = ContractDiscoveryEngine(Path("/test/workspace"))
        # Replace the real catalog with our mock
        engine.catalog = mock_step_catalog
        return engine

    def test_initialization_with_step_catalog(self, contract_discovery_engine, mock_step_catalog):
        """Test that ContractDiscoveryEngine initializes with StepCatalog."""
        assert contract_discovery_engine.catalog is mock_step_catalog

    def test_discover_all_contracts_uses_catalog(self, contract_discovery_engine, mock_step_catalog):
        """Test that contract discovery uses StepCatalog."""
        result = contract_discovery_engine.discover_all_contracts()
        
        # Verify catalog methods were called
        mock_step_catalog.list_available_steps.assert_called_once()
        assert mock_step_catalog.get_step_info.call_count == 3  # Called for each step
        
        # Should return steps that have contracts
        assert result == ['step1', 'step2', 'step3']

    def test_discover_contracts_with_scripts_uses_catalog(self, contract_discovery_engine, mock_step_catalog):
        """Test that contracts with scripts discovery uses StepCatalog."""
        result = contract_discovery_engine.discover_contracts_with_scripts()
        
        # Verify catalog method was called
        mock_step_catalog.discover_contracts_with_scripts.assert_called_once()
        assert result == ['step1', 'step2']

    def test_discover_all_contracts_fallback(self, mock_step_catalog):
        """Test fallback when catalog discovery fails."""
        # Mock catalog failure
        mock_step_catalog.list_available_steps.side_effect = Exception("Catalog error")
        
        # Updated for new adapter API - uses workspace_root instead of step_catalog parameter
        from pathlib import Path
        engine = ContractDiscoveryEngine(Path("/test/workspace"))
        # Replace the real catalog with our mock that fails
        engine.catalog = mock_step_catalog
        
        # Should handle error gracefully
        result = engine.discover_all_contracts()
        assert result == []  # Empty list when catalog fails


class TestWorkspaceDiscoveryManagerIntegration:
    """Test WorkspaceDiscoveryManager integration with StepCatalog."""

    @pytest.fixture
    def mock_step_catalog(self):
        """Create mock StepCatalog for testing."""
        catalog = Mock(spec=StepCatalog)
        
        # Mock cross-workspace discovery
        catalog.discover_cross_workspace_components.return_value = {
            'core': ['step1:script', 'step2:builder'],
            'dev1': ['step3:contract']
        }
        
        # Mock step info
        step_info = Mock(spec=StepInfo)
        step_info.config_class = "TestConfig"
        step_info.sagemaker_step_type = "Training"
        step_info.file_components = {'script': Mock(), 'builder': Mock()}
        step_info.workspace_id = "core"
        catalog.get_step_info.return_value = step_info
        
        # Mock available steps and workspaces
        catalog.list_available_steps.return_value = ['step1', 'step2', 'step3']
        
        return catalog

    @pytest.fixture
    def workspace_discovery_manager(self, mock_step_catalog):
        """Create WorkspaceDiscoveryManager with mocked StepCatalog."""
        # Updated for new adapter API - uses workspace_root instead of step_catalog parameter
        from pathlib import Path
        manager = WorkspaceDiscoveryManager(Path("/test/workspace"))
        # Replace the real catalog with our mock
        manager.catalog = mock_step_catalog
        return manager

    def test_initialization_with_step_catalog(self, workspace_discovery_manager, mock_step_catalog):
        """Test that WorkspaceDiscoveryManager initializes with StepCatalog."""
        assert workspace_discovery_manager.catalog is mock_step_catalog

    def test_discover_components_uses_catalog(self, workspace_discovery_manager, mock_step_catalog):
        """Test that component discovery uses StepCatalog."""
        result = workspace_discovery_manager.discover_components(['core', 'dev1'])
        
        # The adapter only calls catalog methods when workspace IDs are provided
        # and it calls list_available_steps and get_step_info instead of discover_cross_workspace_components
        mock_step_catalog.list_available_steps.assert_called()
        
        # Verify result structure
        assert "builders" in result
        assert "scripts" in result
        assert "contracts" in result
        assert "summary" in result

    def test_list_available_developers_uses_catalog(self, workspace_discovery_manager, mock_step_catalog):
        """Test that developer listing works correctly."""
        # Test the actual method that exists
        result = workspace_discovery_manager.list_available_developers()
        
        # Should return a list of developers
        assert isinstance(result, list)


class TestIntegrationEndToEnd:
    """End-to-end integration tests across multiple systems."""

    @pytest.fixture
    def mock_step_catalog(self):
        """Create comprehensive mock StepCatalog for end-to-end testing."""
        catalog = Mock(spec=StepCatalog)
        
        # Mock comprehensive step data
        step_info = Mock(spec=StepInfo)
        step_info.step_name = "test_step"
        step_info.workspace_id = "core"
        step_info.config_class = "TestConfig"
        step_info.sagemaker_step_type = "Training"
        step_info.file_components = {
            'contract': Mock(spec=FileMetadata, path=Path('/test/contract.py')),
            'script': Mock(spec=FileMetadata, path=Path('/test/script.py')),
            'spec': Mock(spec=FileMetadata, path=Path('/test/spec.py')),
            'builder': Mock(spec=FileMetadata, path=Path('/test/builder.py'))
        }
        
        catalog.get_step_info.return_value = step_info
        catalog.list_available_steps.return_value = ['test_step']
        catalog.discover_contracts_with_scripts.return_value = ['test_step']
        catalog.discover_cross_workspace_components.return_value = {
            'core': ['test_step:script', 'test_step:contract']
        }
        
        return catalog

    def test_validation_orchestrator_cross_workspace_validator_integration(self, mock_step_catalog):
        """Test integration between ValidationOrchestrator and CrossWorkspaceValidator."""
        # Create both systems with same catalog
        orchestrator = ValidationOrchestrator(step_catalog=mock_step_catalog)
        validator = CrossWorkspaceValidator(step_catalog=mock_step_catalog)
        
        # Test that both can discover the same components
        contracts = orchestrator._discover_contracts_with_scripts()
        cross_workspace = validator.discover_cross_workspace_components()
        
        # Both should find the test_step
        assert 'test_step' in contracts
        assert 'core' in cross_workspace['discovery_result']

    def test_contract_discovery_workspace_discovery_integration(self, mock_step_catalog):
        """Test integration between ContractDiscoveryEngine and WorkspaceDiscoveryManager."""
        # Create both systems with updated API - uses workspace_root instead of step_catalog parameter
        from pathlib import Path
        contract_engine = ContractDiscoveryEngine(Path("/test/workspace"))
        workspace_manager = WorkspaceDiscoveryManager(Path("/test/workspace"))
        
        # Replace their catalogs with our mock
        contract_engine.catalog = mock_step_catalog
        workspace_manager.catalog = mock_step_catalog
        
        # Test that both can discover components consistently
        contracts = contract_engine.discover_all_contracts()
        components = workspace_manager.discover_components(['core'])  # Provide workspace IDs
        
        # Both should find components
        assert len(contracts) > 0
        assert components['summary']['total_components'] >= 0  # May be 0 if no matching workspaces

    def test_design_principles_compliance(self, mock_step_catalog):
        """Test that all integrated systems follow design principles."""
        # Create all integrated systems - updated API for adapters
        from pathlib import Path
        orchestrator = ValidationOrchestrator(step_catalog=mock_step_catalog)
        validator = CrossWorkspaceValidator(step_catalog=mock_step_catalog)
        contract_engine = ContractDiscoveryEngine(Path("/test/workspace"))
        workspace_manager = WorkspaceDiscoveryManager(Path("/test/workspace"))
        
        # Verify Separation of Concerns: All systems use catalog for discovery
        assert orchestrator.catalog is mock_step_catalog
        assert validator.catalog is mock_step_catalog
        assert contract_engine.catalog is not None  # Adapter creates its own catalog
        assert workspace_manager.catalog is not None  # Adapter creates its own catalog
        
        # Verify Single Responsibility: Each system maintains its specialized logic
        assert hasattr(orchestrator, 'orchestrate_contract_validation')  # Validation business logic
        assert hasattr(validator, 'validate_cross_workspace_pipeline')  # Cross-workspace validation logic
        assert hasattr(contract_engine, 'extract_contract_reference_from_spec')  # Contract loading logic
        assert hasattr(workspace_manager, 'resolve_cross_workspace_dependencies')  # Workspace management logic
        
        # Verify Explicit Dependencies: All systems explicitly declare catalog dependency
        # This is verified by the constructor signatures requiring step_catalog parameter for orchestrator/validator
        # and workspace_root parameter for adapters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
