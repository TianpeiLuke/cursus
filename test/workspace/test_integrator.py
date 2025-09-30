"""
Comprehensive tests for WorkspaceIntegrator.

This test suite validates the component integration and promotion functionality
for cross-workspace operations.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

from cursus.workspace.integrator import WorkspaceIntegrator, IntegrationResult


class TestWorkspaceIntegrator:
    """Test WorkspaceIntegrator functionality."""

    @pytest.fixture
    def mock_step_catalog(self):
        """Mock StepCatalog for testing."""
        mock_catalog = MagicMock()
        mock_catalog.list_available_steps.return_value = ['test_step1', 'test_step2']
        mock_catalog.get_step_info.return_value = MagicMock()
        return mock_catalog

    @pytest.fixture
    def integrator(self, mock_step_catalog):
        """Create WorkspaceIntegrator instance."""
        return WorkspaceIntegrator(mock_step_catalog)

    @pytest.fixture
    def temp_workspace_dirs(self):
        """Create temporary workspace directories for testing."""
        temp_dirs = []
        for i in range(2):
            temp_dir = Path(tempfile.mkdtemp())
            # Create realistic workspace structure
            for subdir in ['scripts', 'contracts', 'specs', 'builders', 'configs']:
                (temp_dir / subdir).mkdir(exist_ok=True)
            temp_dirs.append(temp_dir)
        
        yield temp_dirs
        
        # Cleanup
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def test_initialization(self, mock_step_catalog):
        """Test WorkspaceIntegrator initialization."""
        integrator = WorkspaceIntegrator(mock_step_catalog)
        
        assert integrator.catalog == mock_step_catalog
        assert integrator.metrics is not None

    def test_promote_component_to_core_success(self, integrator):
        """Test successful component promotion to core."""
        # Mock component exists in workspace
        mock_step_info = MagicMock()
        mock_step_info.step_name = 'test_component'
        mock_step_info.workspace_id = 'source_workspace'
        mock_step_info.file_components = {
            'builder': MagicMock(path=Path('/workspace/builders/test_component.py')),
            'config': MagicMock(path=Path('/workspace/configs/test_component.py'))
        }
        
        integrator.catalog.get_step_info.return_value = mock_step_info
        integrator.catalog.package_root = Path('/test/cursus')
        integrator.catalog.list_available_steps.return_value = []  # No conflicts in core
        
        with patch('shutil.copy2') as mock_copy, \
             patch('pathlib.Path.mkdir') as mock_mkdir, \
             patch('pathlib.Path.exists', return_value=True):
            
            result = integrator.promote_component_to_core(
                'test_component', 'source_workspace', dry_run=False
            )
            
            assert result.success is True
            assert 'successfully promoted' in result.message.lower()

    def test_promote_component_to_core_dry_run(self, integrator):
        """Test component promotion dry run."""
        # Mock component exists
        mock_step_info = MagicMock()
        mock_step_info.step_name = 'test_component'
        mock_step_info.workspace_id = 'source_workspace'
        mock_step_info.file_components = {
            'builder': MagicMock(path=Path('/workspace/builders/test_component.py')),
            'config': MagicMock(path=Path('/workspace/configs/test_component.py'))
        }
        
        integrator.catalog.get_step_info.return_value = mock_step_info
        integrator.catalog.list_available_steps.return_value = []  # No conflicts in core
        
        with patch('pathlib.Path.exists', return_value=True):
            result = integrator.promote_component_to_core(
                'test_component', 'source_workspace', dry_run=True
            )
        
        assert result.success is True
        assert 'dry run' in result.message.lower()

    def test_promote_component_to_core_not_found(self, integrator):
        """Test component promotion when component not found."""
        integrator.catalog.get_step_info.return_value = None
        
        result = integrator.promote_component_to_core(
            'nonexistent_component', 'source_workspace'
        )
        
        assert result.success is False
        assert 'not found' in result.message.lower()

    def test_promote_component_to_core_wrong_workspace(self, integrator):
        """Test component promotion from wrong workspace."""
        # Mock component exists but in different workspace
        mock_step_info = MagicMock()
        mock_step_info.workspace_id = 'different_workspace'
        
        integrator.catalog.get_step_info.return_value = mock_step_info
        
        result = integrator.promote_component_to_core(
            'test_component', 'source_workspace'
        )
        
        assert result.success is False
        assert 'not found in workspace' in result.message.lower()

    def test_promote_component_to_core_file_error(self, integrator):
        """Test component promotion with file operation error."""
        # Mock component exists
        mock_step_info = MagicMock()
        mock_step_info.workspace_id = 'source_workspace'
        mock_step_info.file_components = {
            'script': MagicMock(path=Path('/workspace/scripts/test_component.py'))
        }
        
        integrator.catalog.get_step_info.return_value = mock_step_info
        
        with patch('shutil.copy2') as mock_copy:
            mock_copy.side_effect = OSError("Permission denied")
            
            result = integrator.promote_component_to_core(
                'test_component', 'source_workspace', dry_run=False
            )
            
            assert result.success is False
            assert 'failed' in result.message.lower()

    def test_integrate_cross_workspace_components_success(self, integrator):
        """Test successful cross-workspace component integration."""
        source_components = [
            {'step_name': 'comp1', 'source_workspace_id': 'workspace1'},
            {'step_name': 'comp2', 'source_workspace_id': 'workspace2'}
        ]
        
        # Mock components exist
        mock_step_info1 = MagicMock()
        mock_step_info1.workspace_id = 'workspace1'
        mock_step_info1.file_components = {
            'script': MagicMock(path=Path('/ws1/scripts/comp1.py'))
        }
        
        mock_step_info2 = MagicMock()
        mock_step_info2.workspace_id = 'workspace2'
        mock_step_info2.file_components = {
            'script': MagicMock(path=Path('/ws2/scripts/comp2.py'))
        }
        
        integrator.catalog.get_step_info.side_effect = [mock_step_info1, mock_step_info2]
        
        with patch('shutil.copy2') as mock_copy, \
             patch('pathlib.Path.mkdir') as mock_mkdir:
            
            result = integrator.integrate_cross_workspace_components(
                'target_workspace', source_components
            )
            
            assert result.success is True
            assert 'Cross-workspace integration: 2/2 successful' in result.message

    def test_integrate_cross_workspace_components_partial_failure(self, integrator):
        """Test cross-workspace integration with partial failures."""
        source_components = [
            {'step_name': 'comp1', 'source_workspace_id': 'workspace1'},
            {'step_name': 'nonexistent', 'source_workspace_id': 'workspace2'}
        ]
        
        # Mock first component exists, second doesn't
        mock_step_info1 = MagicMock()
        mock_step_info1.workspace_id = 'workspace1'
        mock_step_info1.file_components = {
            'script': MagicMock(path=Path('/ws1/scripts/comp1.py'))
        }
        
        integrator.catalog.get_step_info.side_effect = [mock_step_info1, None]
        
        with patch('shutil.copy2') as mock_copy, \
             patch('pathlib.Path.mkdir') as mock_mkdir:
            
            result = integrator.integrate_cross_workspace_components(
                'target_workspace', source_components
            )
            
            assert result.success is False
            assert 'Cross-workspace integration: 1/2 successful' in result.message

    def test_integrate_cross_workspace_components_empty_list(self, integrator):
        """Test cross-workspace integration with empty component list."""
        result = integrator.integrate_cross_workspace_components('target_workspace', [])
        
        assert result.success is True
        assert 'Cross-workspace integration: 0/0 successful' in result.message

    def test_rollback_promotion_success(self, integrator):
        """Test successful promotion rollback."""
        # Mock component exists in core
        mock_step_info = MagicMock()
        mock_step_info.workspace_id = 'core'
        mock_step_info.step_name = 'test_component'
        mock_step_info.file_components = {
            'script': MagicMock(path=Path('src/cursus/steps/scripts/test_component.py')),
            'contract': MagicMock(path=Path('src/cursus/steps/contracts/test_component_contract.py'))
        }
        
        integrator.catalog.get_step_info.return_value = mock_step_info
        integrator.catalog.package_root = Path('src/cursus')
        
        with patch('pathlib.Path.unlink') as mock_unlink, \
             patch('pathlib.Path.exists', return_value=True):
            
            result = integrator.rollback_promotion('test_component')
            
            assert result.success is True
            assert 'rolled back' in result.message.lower()

    def test_rollback_promotion_not_found(self, integrator):
        """Test promotion rollback when no promotion found."""
        integrator.catalog.get_step_info.return_value = None
        
        result = integrator.rollback_promotion('nonexistent_component')
        
        assert result.success is False
        assert 'not found' in result.message.lower()

    def test_get_integration_summary(self, integrator):
        """Test integration summary generation."""
        summary = integrator.get_integration_summary()
        
        assert 'metrics' in summary
        assert 'promotion_success_rate' in summary

    def test_integration_result_model(self):
        """Test IntegrationResult model functionality."""
        result = IntegrationResult(
            success=True,
            message="Integration successful",
            details={'components': ['comp1', 'comp2']}
        )
        
        assert result.success is True
        assert result.message == "Integration successful"
        assert len(result.details['components']) == 2

    def test_concurrent_integration_safety(self, integrator):
        """Test that integrator handles concurrent operations safely."""
        import threading
        
        # Mock component exists
        mock_step_info = MagicMock()
        mock_step_info.workspace_id = 'source_workspace'
        mock_step_info.file_components = {
            'script': MagicMock(path=Path('/workspace/scripts/test_component.py'))
        }
        
        integrator.catalog.get_step_info.return_value = mock_step_info
        
        results = []
        errors = []
        
        def worker(component_name):
            try:
                result = integrator.promote_component_to_core(
                    component_name, 'source_workspace', dry_run=True
                )
                results.append(result.success)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=[f'component_{i}'])
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should not have any errors
        assert len(errors) == 0
        assert len(results) == 5


    def test_error_recovery_in_integration(self, integrator):
        """Test error recovery during integration operations."""
        source_components = [
            {'step_name': 'comp1', 'source_workspace_id': 'workspace1'},
            {'step_name': 'comp2', 'source_workspace_id': 'workspace2'}
        ]
        
        # Mock first component succeeds, second fails
        mock_step_info1 = MagicMock()
        mock_step_info1.workspace_id = 'workspace1'
        mock_step_info1.file_components = {
            'script': MagicMock(path=Path('/ws1/scripts/comp1.py'))
        }
        
        integrator.catalog.get_step_info.side_effect = [mock_step_info1, Exception("Catalog error")]
        
        with patch('shutil.copy2') as mock_copy, \
             patch('pathlib.Path.mkdir') as mock_mkdir:
            
            result = integrator.integrate_cross_workspace_components(
                'target_workspace', source_components
            )
            
            # Should handle catalog error gracefully
            assert result.success is False
            assert 'cross-workspace integration failed' in result.message.lower()


class TestWorkspaceIntegratorIntegration:
    """Integration tests for WorkspaceIntegrator with realistic scenarios."""

    @pytest.fixture
    def realistic_workspace_structure(self):
        """Create realistic workspace structure for integration testing."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create source workspace
        source_ws = temp_dir / "source_workspace"
        for subdir in ['scripts', 'contracts', 'specs', 'builders', 'configs']:
            (source_ws / subdir).mkdir(parents=True)
        
        # Create sample component files
        (source_ws / 'scripts' / 'data_processor.py').write_text('# Data processor script')
        (source_ws / 'contracts' / 'data_processor_contract.py').write_text('# Data processor contract')
        (source_ws / 'specs' / 'data_processor_spec.py').write_text('# Data processor spec')
        (source_ws / 'builders' / 'data_processor.py').write_text('# Data processor builder')
        (source_ws / 'configs' / 'data_processor.py').write_text('# Data processor config')
        
        # Create target core structure
        core_dir = temp_dir / "src" / "cursus" / "steps"
        for subdir in ['scripts', 'contracts', 'specs', 'builders', 'configs']:
            (core_dir / subdir).mkdir(parents=True)
        
        yield temp_dir, source_ws, core_dir
        
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_realistic_component_promotion(self, realistic_workspace_structure):
        """Test component promotion with realistic file structure."""
        temp_dir, source_ws, core_dir = realistic_workspace_structure
        
        # Mock step catalog
        mock_catalog = MagicMock()
        mock_step_info = MagicMock()
        mock_step_info.step_name = 'data_processor'
        mock_step_info.workspace_id = 'source_workspace'
        mock_step_info.file_components = {
            'builder': MagicMock(path=source_ws / 'builders' / 'data_processor.py'),
            'config': MagicMock(path=source_ws / 'configs' / 'data_processor.py'),
            'script': MagicMock(path=source_ws / 'scripts' / 'data_processor.py')
        }
        mock_catalog.get_step_info.return_value = mock_step_info
        mock_catalog.package_root = core_dir.parent  # /tmp/xxx/src/cursus -> /tmp/xxx/src
        mock_catalog.list_available_steps.return_value = []  # No conflicts in core
        
        integrator = WorkspaceIntegrator(mock_catalog)
        
        result = integrator.promote_component_to_core(
            'data_processor', 'source_workspace', dry_run=False
        )
        
        assert result.success is True
        
        # Verify files were copied (only the files that were actually in the mock)
        assert (core_dir / 'builders' / 'data_processor.py').exists()
        assert (core_dir / 'configs' / 'data_processor.py').exists()
        assert (core_dir / 'scripts' / 'data_processor.py').exists()

    def test_integrator_resilience_to_catalog_failures(self):
        """Test integrator resilience when catalog operations fail."""
        mock_catalog = MagicMock()
        mock_catalog.get_step_info.side_effect = Exception("Catalog failure")
        
        integrator = WorkspaceIntegrator(mock_catalog)
        
        # Should handle catalog failures gracefully
        result = integrator.promote_component_to_core('failing_component', 'workspace')
        
        assert result.success is False
        assert 'failed' in result.message.lower()


if __name__ == "__main__":
    pytest.main([__file__])
