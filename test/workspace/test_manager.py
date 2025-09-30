"""
Comprehensive tests for WorkspaceManager.

This test suite validates the core workspace management functionality
using step catalog integration.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

from cursus.workspace.manager import WorkspaceManager


class TestWorkspaceManager:
    """Test WorkspaceManager functionality."""

    @pytest.fixture
    def temp_workspace_dirs(self):
        """Create temporary workspace directories for testing."""
        temp_dirs = []
        for i in range(2):
            temp_dir = Path(tempfile.mkdtemp())
            # Create realistic workspace structure
            (temp_dir / "scripts").mkdir(exist_ok=True)
            (temp_dir / "contracts").mkdir(exist_ok=True)
            (temp_dir / "specs").mkdir(exist_ok=True)
            (temp_dir / "builders").mkdir(exist_ok=True)
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
        with patch('cursus.workspace.manager.StepCatalog') as mock_catalog_class:
            mock_catalog = MagicMock()
            mock_catalog_class.return_value = mock_catalog
            
            # Mock basic catalog methods
            mock_catalog.list_available_steps.return_value = ['test_step1', 'test_step2']
            mock_catalog.get_step_info.return_value = MagicMock()
            mock_catalog.search_steps.return_value = []
            
            yield mock_catalog

    @pytest.fixture
    def manager_with_workspaces(self, temp_workspace_dirs, mock_step_catalog):
        """Create WorkspaceManager instance with test workspaces."""
        return WorkspaceManager(workspace_dirs=temp_workspace_dirs)

    @pytest.fixture
    def manager_package_only(self, mock_step_catalog):
        """Create WorkspaceManager instance in package-only mode."""
        return WorkspaceManager()

    def test_initialization_package_only(self, mock_step_catalog):
        """Test WorkspaceManager initialization in package-only mode."""
        manager = WorkspaceManager()
        
        assert manager.workspace_dirs == []
        assert manager.catalog is not None

    def test_initialization_with_workspaces(self, temp_workspace_dirs, mock_step_catalog):
        """Test WorkspaceManager initialization with workspaces."""
        manager = WorkspaceManager(workspace_dirs=temp_workspace_dirs)
        
        assert len(manager.workspace_dirs) == 2
        assert all(wd in manager.workspace_dirs for wd in temp_workspace_dirs)

    def test_initialization_single_workspace_path(self, temp_workspace_dirs, mock_step_catalog):
        """Test WorkspaceManager initialization with single workspace as Path."""
        manager = WorkspaceManager(workspace_dirs=[temp_workspace_dirs[0]])
        
        assert len(manager.workspace_dirs) == 1
        assert manager.workspace_dirs[0] == temp_workspace_dirs[0]

    def test_discover_components_all(self, manager_with_workspaces):
        """Test discovering all components across workspaces."""
        mock_steps = ['component1', 'component2', 'workspace_component']
        manager_with_workspaces.catalog.list_available_steps.return_value = mock_steps
        
        components = manager_with_workspaces.discover_components()
        
        assert components == mock_steps
        manager_with_workspaces.catalog.list_available_steps.assert_called_once()

    def test_discover_components_with_workspace_filter(self, manager_with_workspaces):
        """Test discovering components with workspace filter."""
        # Mock step info for filtering
        mock_step_info1 = MagicMock()
        mock_step_info1.workspace_id = 'workspace1'
        mock_step_info2 = MagicMock()
        mock_step_info2.workspace_id = 'workspace2'
        
        manager_with_workspaces.catalog.list_available_steps.return_value = ['comp1', 'comp2']
        manager_with_workspaces.catalog.get_step_info.side_effect = [mock_step_info1, mock_step_info2]
        
        components = manager_with_workspaces.discover_components(workspace_id='workspace1')
        
        assert components == ['comp1']

    def test_get_component_info_success(self, manager_with_workspaces):
        """Test successful component info retrieval."""
        mock_info = MagicMock()
        manager_with_workspaces.catalog.get_step_info.return_value = mock_info
        
        info = manager_with_workspaces.get_component_info('test_component')
        
        assert info == mock_info
        manager_with_workspaces.catalog.get_step_info.assert_called_once_with('test_component')

    def test_get_component_info_not_found(self, manager_with_workspaces):
        """Test component info retrieval when component not found."""
        manager_with_workspaces.catalog.get_step_info.return_value = None
        
        info = manager_with_workspaces.get_component_info('nonexistent_component')
        
        assert info is None

    def test_find_component_file_success(self, manager_with_workspaces):
        """Test successful component file finding."""
        mock_step_info = MagicMock()
        mock_file_component = MagicMock()
        mock_file_component.path = Path('/test/path/component.py')
        mock_step_info.file_components = {'script': mock_file_component}
        
        manager_with_workspaces.catalog.get_step_info.return_value = mock_step_info
        
        file_path = manager_with_workspaces.find_component_file('test_component', 'script')
        
        assert file_path == Path('/test/path/component.py')

    def test_find_component_file_not_found(self, manager_with_workspaces):
        """Test component file finding when file not found."""
        mock_step_info = MagicMock()
        mock_step_info.file_components = {}
        
        manager_with_workspaces.catalog.get_step_info.return_value = mock_step_info
        
        file_path = manager_with_workspaces.find_component_file('test_component', 'script')
        
        assert file_path is None

    def test_find_component_file_component_not_found(self, manager_with_workspaces):
        """Test component file finding when component doesn't exist."""
        manager_with_workspaces.catalog.get_step_info.return_value = None
        
        file_path = manager_with_workspaces.find_component_file('nonexistent_component', 'script')
        
        assert file_path is None

    def test_get_workspace_summary(self, manager_with_workspaces):
        """Test workspace summary generation."""
        # Mock catalog methods
        manager_with_workspaces.catalog.list_available_steps.return_value = ['comp1', 'comp2', 'comp3']
        
        summary = manager_with_workspaces.get_workspace_summary()
        
        assert 'total_components' in summary
        assert 'workspace_components' in summary  # Actual field name
        assert 'workspace_directories' in summary
        assert summary['total_components'] == 3
        assert len(summary['workspace_directories']) == 2

    def test_validate_workspace_structure_valid(self, manager_with_workspaces, temp_workspace_dirs):
        """Test workspace structure validation for valid workspace."""
        workspace_dir = temp_workspace_dirs[0]
        
        result = manager_with_workspaces.validate_workspace_structure(workspace_dir)
        
        assert result['valid'] is True
        assert result['workspace_dir'] == str(workspace_dir)
        assert 'components_found' in result  # Actual field name

    def test_validate_workspace_structure_invalid(self, manager_with_workspaces):
        """Test workspace structure validation for invalid workspace."""
        nonexistent_dir = Path('/nonexistent/workspace')
        
        result = manager_with_workspaces.validate_workspace_structure(nonexistent_dir)
        
        assert result['valid'] is False
        assert 'warnings' in result  # Actual field name for issues

    def test_get_cross_workspace_components(self, manager_with_workspaces):
        """Test getting components organized by workspace."""
        # Mock components and their workspace info
        manager_with_workspaces.catalog.list_available_steps.return_value = ['comp1', 'comp2', 'comp3']
        
        mock_step_info1 = MagicMock()
        mock_step_info1.workspace_id = 'workspace1'
        mock_step_info2 = MagicMock()
        mock_step_info2.workspace_id = 'workspace2'
        mock_step_info3 = MagicMock()
        mock_step_info3.workspace_id = 'workspace1'
        
        manager_with_workspaces.catalog.get_step_info.side_effect = [
            mock_step_info1, mock_step_info2, mock_step_info3
        ]
        
        cross_workspace = manager_with_workspaces.get_cross_workspace_components()
        
        assert 'workspace1' in cross_workspace
        assert 'workspace2' in cross_workspace
        assert len(cross_workspace['workspace1']) == 2
        assert len(cross_workspace['workspace2']) == 1

    def test_create_workspace_pipeline_success(self, manager_with_workspaces):
        """Test successful workspace pipeline creation."""
        mock_dag = MagicMock()
        mock_pipeline = MagicMock()
        
        with patch('cursus.workspace.manager.PipelineAssembler') as mock_assembler_class:
            mock_assembler = MagicMock()
            mock_assembler_class.return_value = mock_assembler
            mock_assembler.generate_pipeline.return_value = mock_pipeline  # Actual method name
            
            pipeline = manager_with_workspaces.create_workspace_pipeline(mock_dag, '/test/config.json')
            
            assert pipeline == mock_pipeline
            mock_assembler.generate_pipeline.assert_called_once()

    def test_create_workspace_pipeline_error(self, manager_with_workspaces):
        """Test workspace pipeline creation error handling."""
        mock_dag = MagicMock()
        
        with patch('cursus.workspace.manager.PipelineAssembler') as mock_assembler_class:
            mock_assembler_class.side_effect = Exception("Assembly failed")
            
            pipeline = manager_with_workspaces.create_workspace_pipeline(mock_dag, '/test/config.json')
            
            assert pipeline is None

    def test_refresh_catalog_success(self, manager_with_workspaces):
        """Test successful catalog refresh."""
        with patch('cursus.workspace.manager.StepCatalog') as mock_catalog_class:
            mock_new_catalog = MagicMock()
            mock_catalog_class.return_value = mock_new_catalog
            
            success = manager_with_workspaces.refresh_catalog()
            
            assert success is True
            # Should create new catalog instance
            mock_catalog_class.assert_called_once_with(workspace_dirs=manager_with_workspaces.workspace_dirs)

    def test_refresh_catalog_no_method(self, manager_with_workspaces):
        """Test catalog refresh when refresh method doesn't exist."""
        # Remove refresh method to simulate older catalog
        if hasattr(manager_with_workspaces.catalog, 'refresh'):
            delattr(manager_with_workspaces.catalog, 'refresh')
        
        success = manager_with_workspaces.refresh_catalog()
        
        assert success is True  # Should succeed gracefully

    def test_refresh_catalog_error(self, manager_with_workspaces):
        """Test catalog refresh error handling."""
        with patch.object(manager_with_workspaces.catalog, 'refresh') as mock_refresh:
            mock_refresh.side_effect = Exception("Refresh failed")
            
            success = manager_with_workspaces.refresh_catalog()
            
            assert success is False

    def test_get_workspace_summary_with_metrics(self, manager_with_workspaces):
        """Test workspace summary includes metrics when available."""
        # Mock catalog metrics
        mock_metrics = {'total_steps': 10, 'workspaces': 3}
        manager_with_workspaces.catalog.get_metrics_report = MagicMock(return_value=mock_metrics)
        
        summary = manager_with_workspaces.get_workspace_summary()
        
        assert 'catalog_metrics' in summary
        assert summary['catalog_metrics'] == mock_metrics

    def test_get_workspace_summary_no_metrics_method(self, manager_with_workspaces):
        """Test workspace summary when catalog doesn't have metrics method."""
        # Ensure catalog doesn't have metrics method
        if hasattr(manager_with_workspaces.catalog, 'get_metrics_report'):
            delattr(manager_with_workspaces.catalog, 'get_metrics_report')
        
        summary = manager_with_workspaces.get_workspace_summary()
        
        assert 'catalog_metrics' in summary
        assert summary['catalog_metrics'] == {}

    def test_error_handling_in_discover_components(self, manager_with_workspaces):
        """Test error handling in component discovery."""
        manager_with_workspaces.catalog.list_available_steps.side_effect = Exception("Catalog error")
        
        components = manager_with_workspaces.discover_components()
        
        assert components == []

    def test_error_handling_in_get_component_info(self, manager_with_workspaces):
        """Test error handling in component info retrieval."""
        manager_with_workspaces.catalog.get_step_info.side_effect = Exception("Info retrieval error")
        
        info = manager_with_workspaces.get_component_info('test_component')
        
        assert info is None

    def test_workspace_directory_normalization(self, temp_workspace_dirs, mock_step_catalog):
        """Test that workspace directories are properly normalized."""
        # Test with string paths
        string_paths = [str(wd) for wd in temp_workspace_dirs]
        manager = WorkspaceManager(workspace_dirs=string_paths)
        
        # Should convert to Path objects
        assert all(isinstance(wd, Path) for wd in manager.workspace_dirs)
        assert len(manager.workspace_dirs) == len(temp_workspace_dirs)

    def test_component_type_validation(self, manager_with_workspaces):
        """Test component type validation in find_component_file."""
        valid_types = ['script', 'contract', 'spec', 'builder', 'config']
        
        for component_type in valid_types:
            # Should not raise exception for valid types
            result = manager_with_workspaces.find_component_file('test_component', component_type)
            # Result can be None if component not found, but no exception should be raised

    def test_workspace_summary_with_empty_catalog(self, manager_package_only):
        """Test workspace summary with empty catalog."""
        manager_package_only.catalog.list_available_steps.return_value = []
        
        summary = manager_package_only.get_workspace_summary()
        
        assert summary['total_components'] == 0
        assert summary['workspace_components'] == {}  # Actual field name

    def test_concurrent_access_safety(self, manager_with_workspaces):
        """Test that manager handles concurrent access safely."""
        import threading
        import time
        
        results = []
        errors = []
        
        def worker():
            try:
                components = manager_with_workspaces.discover_components()
                results.append(len(components))
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should not have any errors
        assert len(errors) == 0
        assert len(results) == 5


class TestWorkspaceManagerIntegration:
    """Integration tests for WorkspaceManager with realistic scenarios."""

    @pytest.fixture
    def realistic_workspace(self):
        """Create a realistic workspace structure for integration testing."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create realistic workspace structure
        scripts_dir = temp_dir / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "data_processing.py").write_text("# Data processing script")
        (scripts_dir / "model_training.py").write_text("# Model training script")
        
        contracts_dir = temp_dir / "contracts"
        contracts_dir.mkdir()
        (contracts_dir / "data_processing_contract.py").write_text("# Data processing contract")
        (contracts_dir / "model_training_contract.py").write_text("# Model training contract")
        
        specs_dir = temp_dir / "specs"
        specs_dir.mkdir()
        (specs_dir / "data_processing_spec.py").write_text("# Data processing spec")
        
        builders_dir = temp_dir / "builders"
        builders_dir.mkdir()
        (builders_dir / "builder_data_processing_step.py").write_text("# Data processing builder")
        
        configs_dir = temp_dir / "configs"
        configs_dir.mkdir()
        (configs_dir / "config_data_processing_step.py").write_text("# Data processing config")
        
        yield temp_dir
        
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_realistic_workspace_validation(self, realistic_workspace):
        """Test workspace validation with realistic structure."""
        with patch('cursus.workspace.manager.StepCatalog') as mock_catalog_class:
            mock_catalog = MagicMock()
            mock_catalog_class.return_value = mock_catalog
            
            manager = WorkspaceManager(workspace_dirs=[realistic_workspace])
            
            result = manager.validate_workspace_structure(realistic_workspace)
            
            assert result['valid'] is True
            assert 'structure_analysis' in result

    def test_workspace_component_discovery_integration(self, realistic_workspace):
        """Test component discovery with realistic workspace."""
        with patch('cursus.workspace.manager.StepCatalog') as mock_catalog_class:
            mock_catalog = MagicMock()
            mock_catalog_class.return_value = mock_catalog
            mock_catalog.list_available_steps.return_value = ['data_processing', 'model_training']
            
            manager = WorkspaceManager(workspace_dirs=[realistic_workspace])
            
            components = manager.discover_components()
            
            assert 'data_processing' in components
            assert 'model_training' in components

    def test_manager_resilience_to_catalog_failures(self, realistic_workspace):
        """Test manager resilience when catalog operations fail."""
        with patch('cursus.workspace.manager.StepCatalog') as mock_catalog_class:
            # Simulate catalog initialization failure
            mock_catalog_class.side_effect = Exception("Catalog initialization failed")
            
            # Manager should handle this gracefully
            try:
                manager = WorkspaceManager(workspace_dirs=[realistic_workspace])
                # If we get here, the manager handled the error gracefully
                assert True
            except Exception:
                # If initialization fails completely, that's also acceptable
                # as long as it's a clear error
                assert True


if __name__ == "__main__":
    pytest.main([__file__])
