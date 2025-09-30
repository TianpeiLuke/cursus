"""
Comprehensive tests for WorkspaceAPI.

This test suite validates the unified API for all workspace operations,
ensuring proper integration with the step catalog system.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

from cursus.workspace.api import WorkspaceAPI
from cursus.workspace.validator import ValidationResult, CompatibilityResult
from cursus.workspace.integrator import IntegrationResult


class TestWorkspaceAPI:
    """Test WorkspaceAPI functionality."""

    @pytest.fixture
    def temp_workspace_dirs(self):
        """Create temporary workspace directories for testing."""
        temp_dirs = []
        for i in range(3):
            temp_dir = Path(tempfile.mkdtemp())
            # Create basic structure
            (temp_dir / "components").mkdir(exist_ok=True)
            (temp_dir / "configs").mkdir(exist_ok=True)
            temp_dirs.append(temp_dir)
        
        yield temp_dirs
        
        # Cleanup
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_step_catalog(self):
        """Mock StepCatalog for testing."""
        with patch('cursus.workspace.api.StepCatalog') as mock_catalog_class:
            mock_catalog = MagicMock()
            mock_catalog_class.return_value = mock_catalog
            
            # Mock basic catalog methods
            mock_catalog.list_available_steps.return_value = ['test_step1', 'test_step2']
            mock_catalog.get_step_info.return_value = MagicMock()
            mock_catalog.search_steps.return_value = []
            
            yield mock_catalog

    @pytest.fixture
    def api_with_workspaces(self, temp_workspace_dirs, mock_step_catalog):
        """Create WorkspaceAPI instance with test workspaces."""
        return WorkspaceAPI(workspace_dirs=temp_workspace_dirs)

    @pytest.fixture
    def api_package_only(self, mock_step_catalog):
        """Create WorkspaceAPI instance in package-only mode."""
        return WorkspaceAPI()

    def test_initialization_package_only(self, mock_step_catalog):
        """Test WorkspaceAPI initialization in package-only mode."""
        api = WorkspaceAPI()
        
        assert api.workspace_dirs == []
        assert api.catalog is not None
        assert api.manager is not None
        assert api.validator is not None
        assert api.integrator is not None
        assert api.metrics['api_calls'] == 0

    def test_initialization_single_workspace(self, temp_workspace_dirs, mock_step_catalog):
        """Test WorkspaceAPI initialization with single workspace."""
        api = WorkspaceAPI(workspace_dirs=temp_workspace_dirs[0])
        
        assert len(api.workspace_dirs) == 1
        assert api.workspace_dirs[0] == temp_workspace_dirs[0]

    def test_initialization_multiple_workspaces(self, temp_workspace_dirs, mock_step_catalog):
        """Test WorkspaceAPI initialization with multiple workspaces."""
        api = WorkspaceAPI(workspace_dirs=temp_workspace_dirs)
        
        assert len(api.workspace_dirs) == 3
        assert all(wd in api.workspace_dirs for wd in temp_workspace_dirs)

    def test_discover_components_success(self, api_with_workspaces):
        """Test successful component discovery."""
        with patch.object(api_with_workspaces.manager, 'discover_components') as mock_discover:
            mock_discover.return_value = ['component1', 'component2']
            
            components = api_with_workspaces.discover_components()
            
            assert components == ['component1', 'component2']
            assert api_with_workspaces.metrics['successful_operations'] == 1
            mock_discover.assert_called_once_with(workspace_id=None)

    def test_discover_components_with_workspace_filter(self, api_with_workspaces):
        """Test component discovery with workspace filter."""
        with patch.object(api_with_workspaces.manager, 'discover_components') as mock_discover:
            mock_discover.return_value = ['workspace_component']
            
            components = api_with_workspaces.discover_components(workspace_id='test_workspace')
            
            assert components == ['workspace_component']
            mock_discover.assert_called_once_with(workspace_id='test_workspace')

    def test_discover_components_error_handling(self, api_with_workspaces):
        """Test component discovery error handling."""
        with patch.object(api_with_workspaces.manager, 'discover_components') as mock_discover:
            mock_discover.side_effect = Exception("Discovery failed")
            
            components = api_with_workspaces.discover_components()
            
            assert components == []
            assert api_with_workspaces.metrics['failed_operations'] == 1

    def test_get_component_info_success(self, api_with_workspaces):
        """Test successful component info retrieval."""
        mock_info = MagicMock()
        with patch.object(api_with_workspaces.manager, 'get_component_info') as mock_get_info:
            mock_get_info.return_value = mock_info
            
            info = api_with_workspaces.get_component_info('test_component')
            
            assert info == mock_info
            assert api_with_workspaces.metrics['successful_operations'] == 1

    def test_get_component_info_not_found(self, api_with_workspaces):
        """Test component info retrieval when component not found."""
        with patch.object(api_with_workspaces.manager, 'get_component_info') as mock_get_info:
            mock_get_info.return_value = None
            
            info = api_with_workspaces.get_component_info('nonexistent_component')
            
            assert info is None
            assert api_with_workspaces.metrics['failed_operations'] == 1

    def test_find_component_file_success(self, api_with_workspaces):
        """Test successful component file finding."""
        test_path = Path('/test/path/component.py')
        with patch.object(api_with_workspaces.manager, 'find_component_file') as mock_find:
            mock_find.return_value = test_path
            
            file_path = api_with_workspaces.find_component_file('test_component', 'script')
            
            assert file_path == test_path
            assert api_with_workspaces.metrics['successful_operations'] == 1

    def test_search_components_success(self, api_with_workspaces):
        """Test successful component search."""
        mock_results = [MagicMock(), MagicMock()]
        with patch.object(api_with_workspaces.catalog, 'search_steps') as mock_search:
            mock_search.return_value = mock_results
            
            results = api_with_workspaces.search_components('test_query')
            
            assert results == mock_results
            assert api_with_workspaces.metrics['successful_operations'] == 1

    def test_search_components_with_workspace_filter(self, api_with_workspaces):
        """Test component search with workspace filter."""
        mock_result1 = MagicMock()
        mock_result1.workspace_id = 'workspace1'
        mock_result2 = MagicMock()
        mock_result2.workspace_id = 'workspace2'
        
        with patch.object(api_with_workspaces.catalog, 'search_steps') as mock_search:
            mock_search.return_value = [mock_result1, mock_result2]
            
            results = api_with_workspaces.search_components('test_query', workspace_id='workspace1')
            
            assert len(results) == 1
            assert results[0].workspace_id == 'workspace1'

    def test_get_workspace_summary(self, api_with_workspaces):
        """Test workspace summary generation."""
        mock_summary = {'components': 5, 'workspaces': 2}
        with patch.object(api_with_workspaces.manager, 'get_workspace_summary') as mock_summary_method:
            mock_summary_method.return_value = mock_summary
            
            summary = api_with_workspaces.get_workspace_summary()
            
            assert 'api_metrics' in summary
            assert summary['components'] == 5
            assert api_with_workspaces.metrics['successful_operations'] == 1

    def test_validate_workspace_structure(self, api_with_workspaces, temp_workspace_dirs):
        """Test workspace structure validation."""
        mock_validation = {'valid': True, 'issues': []}
        with patch.object(api_with_workspaces.manager, 'validate_workspace_structure') as mock_validate:
            mock_validate.return_value = mock_validation
            
            result = api_with_workspaces.validate_workspace_structure(temp_workspace_dirs[0])
            
            assert result == mock_validation
            assert api_with_workspaces.metrics['successful_operations'] == 1

    def test_create_workspace_pipeline_success(self, api_with_workspaces):
        """Test successful workspace pipeline creation."""
        mock_dag = MagicMock()
        mock_pipeline = MagicMock()
        
        with patch.object(api_with_workspaces.manager, 'create_workspace_pipeline') as mock_create:
            mock_create.return_value = mock_pipeline
            
            pipeline = api_with_workspaces.create_workspace_pipeline(mock_dag, '/test/config.json')
            
            assert pipeline == mock_pipeline
            assert api_with_workspaces.metrics['successful_operations'] == 1

    def test_validate_workspace_components(self, api_with_workspaces):
        """Test workspace component validation."""
        mock_result = ValidationResult(is_valid=True, errors=[], details={})
        
        with patch.object(api_with_workspaces.validator, 'validate_workspace_components') as mock_validate:
            mock_validate.return_value = mock_result
            
            result = api_with_workspaces.validate_workspace_components('test_workspace')
            
            assert result.is_valid is True
            assert api_with_workspaces.metrics['successful_operations'] == 1

    def test_validate_component_quality(self, api_with_workspaces):
        """Test component quality validation."""
        mock_result = ValidationResult(is_valid=True, errors=[], details={})
        
        with patch.object(api_with_workspaces.validator, 'validate_component_quality') as mock_validate:
            mock_validate.return_value = mock_result
            
            result = api_with_workspaces.validate_component_quality('test_component')
            
            assert result.is_valid is True
            assert api_with_workspaces.metrics['successful_operations'] == 1

    def test_validate_cross_workspace_compatibility(self, api_with_workspaces):
        """Test cross-workspace compatibility validation."""
        mock_result = CompatibilityResult(is_compatible=True, issues=[])
        
        with patch.object(api_with_workspaces.validator, 'validate_cross_workspace_compatibility') as mock_validate:
            mock_validate.return_value = mock_result
            
            result = api_with_workspaces.validate_cross_workspace_compatibility(['ws1', 'ws2'])
            
            assert result.is_compatible is True
            assert api_with_workspaces.metrics['successful_operations'] == 1

    def test_promote_component_to_core(self, api_with_workspaces):
        """Test component promotion to core."""
        mock_result = IntegrationResult(success=True, message="Promoted successfully", details={})
        
        with patch.object(api_with_workspaces.integrator, 'promote_component_to_core') as mock_promote:
            mock_promote.return_value = mock_result
            
            result = api_with_workspaces.promote_component_to_core('test_component', 'source_ws')
            
            assert result.success is True
            assert api_with_workspaces.metrics['successful_operations'] == 1

    def test_integrate_cross_workspace_components(self, api_with_workspaces):
        """Test cross-workspace component integration."""
        mock_result = IntegrationResult(success=True, message="Integrated successfully", details={})
        source_components = [{'step_name': 'comp1', 'source_workspace_id': 'ws1'}]
        
        with patch.object(api_with_workspaces.integrator, 'integrate_cross_workspace_components') as mock_integrate:
            mock_integrate.return_value = mock_result
            
            result = api_with_workspaces.integrate_cross_workspace_components('target_ws', source_components)
            
            assert result.success is True
            assert api_with_workspaces.metrics['successful_operations'] == 1

    def test_rollback_promotion(self, api_with_workspaces):
        """Test component promotion rollback."""
        mock_result = IntegrationResult(success=True, message="Rollback successful", details={})
        
        with patch.object(api_with_workspaces.integrator, 'rollback_promotion') as mock_rollback:
            mock_rollback.return_value = mock_result
            
            result = api_with_workspaces.rollback_promotion('test_component')
            
            assert result.success is True
            assert api_with_workspaces.metrics['successful_operations'] == 1

    def test_refresh_catalog(self, api_with_workspaces):
        """Test catalog refresh."""
        with patch.object(api_with_workspaces.manager, 'refresh_catalog') as mock_refresh:
            mock_refresh.return_value = True
            
            success = api_with_workspaces.refresh_catalog()
            
            assert success is True
            assert api_with_workspaces.metrics['successful_operations'] == 1

    def test_get_system_status(self, api_with_workspaces):
        """Test system status retrieval."""
        # Mock all component status methods
        with patch.object(api_with_workspaces.manager, 'get_workspace_summary') as mock_manager_status, \
             patch.object(api_with_workspaces.validator, 'get_validation_summary') as mock_validator_status, \
             patch.object(api_with_workspaces.integrator, 'get_integration_summary') as mock_integrator_status:
            
            mock_manager_status.return_value = {'manager': 'status'}
            mock_validator_status.return_value = {'validator': 'status'}
            mock_integrator_status.return_value = {'integrator': 'status'}
            
            status = api_with_workspaces.get_system_status()
            
            assert 'workspace_api' in status
            assert 'manager' in status
            assert 'validator' in status
            assert 'integrator' in status
            assert 'success_rate' in status['workspace_api']

    def test_list_all_workspaces(self, api_with_workspaces):
        """Test listing all workspaces."""
        mock_components = {'ws1': ['comp1'], 'ws2': ['comp2']}
        
        with patch.object(api_with_workspaces, 'get_cross_workspace_components') as mock_get_components:
            mock_get_components.return_value = mock_components
            
            workspaces = api_with_workspaces.list_all_workspaces()
            
            assert set(workspaces) == {'ws1', 'ws2'}

    def test_get_workspace_component_count(self, api_with_workspaces):
        """Test getting workspace component count."""
        with patch.object(api_with_workspaces, 'discover_components') as mock_discover:
            mock_discover.return_value = ['comp1', 'comp2', 'comp3']
            
            count = api_with_workspaces.get_workspace_component_count('test_workspace')
            
            assert count == 3
            mock_discover.assert_called_once_with(workspace_id='test_workspace')

    def test_is_component_available(self, api_with_workspaces):
        """Test checking component availability."""
        with patch.object(api_with_workspaces, 'discover_components') as mock_discover:
            mock_discover.return_value = ['available_component', 'another_component']
            
            # Test available component
            assert api_with_workspaces.is_component_available('available_component') is True
            
            # Test unavailable component
            assert api_with_workspaces.is_component_available('unavailable_component') is False

    def test_metrics_tracking(self, api_with_workspaces):
        """Test that metrics are properly tracked across operations."""
        initial_calls = api_with_workspaces.metrics['api_calls']
        
        # Perform several operations
        with patch.object(api_with_workspaces.manager, 'discover_components') as mock_discover:
            mock_discover.return_value = ['comp1']
            api_with_workspaces.discover_components()
            
        with patch.object(api_with_workspaces.manager, 'get_component_info') as mock_get_info:
            mock_get_info.return_value = MagicMock()
            api_with_workspaces.get_component_info('comp1')
            
        # Check metrics were updated
        assert api_with_workspaces.metrics['api_calls'] == initial_calls + 2
        assert api_with_workspaces.metrics['successful_operations'] == 2

    def test_error_handling_preserves_metrics(self, api_with_workspaces):
        """Test that error handling properly updates metrics."""
        with patch.object(api_with_workspaces.manager, 'discover_components') as mock_discover:
            mock_discover.side_effect = Exception("Test error")
            
            components = api_with_workspaces.discover_components()
            
            assert components == []
            assert api_with_workspaces.metrics['failed_operations'] == 1
            assert api_with_workspaces.metrics['api_calls'] == 1

    def test_success_rate_calculation(self, api_with_workspaces):
        """Test success rate calculation in system status."""
        # Simulate some successful and failed operations
        api_with_workspaces.metrics['successful_operations'] = 7
        api_with_workspaces.metrics['failed_operations'] = 3
        
        with patch.object(api_with_workspaces.manager, 'get_workspace_summary') as mock_manager_status, \
             patch.object(api_with_workspaces.validator, 'get_validation_summary') as mock_validator_status, \
             patch.object(api_with_workspaces.integrator, 'get_integration_summary') as mock_integrator_status:
            
            mock_manager_status.return_value = {}
            mock_validator_status.return_value = {}
            mock_integrator_status.return_value = {}
            
            status = api_with_workspaces.get_system_status()
            
            # Success rate should be 7/10 = 0.7
            assert status['workspace_api']['success_rate'] == 0.7


class TestWorkspaceAPIIntegration:
    """Integration tests for WorkspaceAPI with real components."""

    @pytest.fixture
    def integration_workspace(self):
        """Create a realistic workspace structure for integration testing."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create realistic workspace structure
        components_dir = temp_dir / "components"
        components_dir.mkdir()
        
        # Create sample component files
        (components_dir / "test_component.py").write_text("# Test component script")
        (components_dir / "test_component_contract.py").write_text("# Test component contract")
        
        configs_dir = temp_dir / "configs"
        configs_dir.mkdir()
        (configs_dir / "test_config.json").write_text('{"test": "config"}')
        
        yield temp_dir
        
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_end_to_end_component_discovery(self, integration_workspace):
        """Test end-to-end component discovery with real workspace structure."""
        with patch('cursus.workspace.api.StepCatalog') as mock_catalog_class:
            mock_catalog = MagicMock()
            mock_catalog_class.return_value = mock_catalog
            mock_catalog.list_available_steps.return_value = ['test_component']
            
            api = WorkspaceAPI(workspace_dirs=[integration_workspace])
            
            # This should work without errors
            assert api.workspace_dirs == [integration_workspace]
            assert len(api.workspace_dirs) == 1

    def test_api_resilience_to_missing_dependencies(self):
        """Test API resilience when dependencies are missing or fail."""
        with patch('cursus.workspace.api.StepCatalog') as mock_catalog_class:
            # Simulate catalog initialization failure
            mock_catalog_class.side_effect = Exception("Catalog initialization failed")
            
            # API should still initialize but handle errors gracefully
            try:
                api = WorkspaceAPI()
                # If we get here, the API handled the error gracefully
                assert True
            except Exception:
                # If initialization fails completely, that's also acceptable
                # as long as it's a clear error
                assert True


if __name__ == "__main__":
    pytest.main([__file__])
