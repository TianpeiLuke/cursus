"""
Comprehensive tests for WorkspaceValidator.

This test suite validates the workspace validation functionality
using existing validation frameworks.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

from cursus.workspace.validator import WorkspaceValidator, ValidationResult, CompatibilityResult


class TestWorkspaceValidator:
    """Test WorkspaceValidator functionality."""

    @pytest.fixture
    def mock_step_catalog(self):
        """Mock StepCatalog for testing."""
        mock_catalog = MagicMock()
        mock_catalog.list_available_steps.return_value = ['test_step1', 'test_step2']
        mock_catalog.get_step_info.return_value = MagicMock()
        return mock_catalog

    @pytest.fixture
    def validator(self, mock_step_catalog):
        """Create WorkspaceValidator instance."""
        return WorkspaceValidator(mock_step_catalog)

    @pytest.fixture
    def mock_unified_tester(self):
        """Mock UnifiedAlignmentTester for testing."""
        with patch('cursus.workspace.validator.UnifiedAlignmentTester') as mock_tester_class:
            mock_tester = MagicMock()
            mock_tester_class.return_value = mock_tester
            yield mock_tester

    def test_initialization(self, mock_step_catalog):
        """Test WorkspaceValidator initialization."""
        validator = WorkspaceValidator(mock_step_catalog)
        
        assert validator.catalog == mock_step_catalog
        assert validator.metrics == {
            'validations_performed': 0,
            'components_validated': 0,
            'compatibility_checks': 0
        }

    def test_validate_workspace_components_success(self, validator):
        """Test successful workspace component validation."""
        # Mock step catalog to return workspace components
        mock_step_info1 = MagicMock()
        mock_step_info1.step_name = 'comp1'
        mock_step_info1.workspace_id = 'test_workspace'
        mock_step_info1.file_components = {
            'builder': MagicMock(path=Path('/test/builder1.py')),
            'config': MagicMock(path=Path('/test/config1.py'))
        }
        
        mock_step_info2 = MagicMock()
        mock_step_info2.step_name = 'comp2'
        mock_step_info2.workspace_id = 'test_workspace'
        mock_step_info2.file_components = {
            'builder': MagicMock(path=Path('/test/builder2.py')),
            'config': MagicMock(path=Path('/test/config2.py'))
        }
        
        validator.catalog.list_available_steps.return_value = ['comp1', 'comp2']
        validator.catalog.get_step_info.side_effect = [mock_step_info1, mock_step_info2]
        
        # Mock file existence
        with patch('pathlib.Path.exists', return_value=True):
            result = validator.validate_workspace_components('test_workspace')
        
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.details['workspace_id'] == 'test_workspace'
        assert result.details['validated_components'] == 2

    def test_validate_workspace_components_with_failures(self, validator):
        """Test workspace component validation with failures."""
        # Mock step catalog with component that has missing files
        mock_step_info = MagicMock()
        mock_step_info.step_name = 'failing_comp'
        mock_step_info.workspace_id = 'test_workspace'
        mock_step_info.file_components = {
            'builder': MagicMock(path=Path('/nonexistent/builder.py')),
            'config': MagicMock(path=Path('/nonexistent/config.py'))
        }
        
        validator.catalog.list_available_steps.return_value = ['failing_comp']
        validator.catalog.get_step_info.return_value = mock_step_info
        
        # Mock file doesn't exist
        with patch('pathlib.Path.exists', return_value=False):
            result = validator.validate_workspace_components('test_workspace')
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert 'File not accessible' in result.errors[0]

    def test_validate_workspace_components_no_components(self, validator):
        """Test workspace validation when no components found."""
        validator.catalog.list_available_steps.return_value = []
        
        result = validator.validate_workspace_components('empty_workspace')
        
        assert result.is_valid is True  # Empty workspace is valid
        assert result.details['component_count'] == 0

    def test_validate_component_quality_success(self, validator):
        """Test successful component quality validation."""
        # Mock component exists with all required components
        mock_step_info = MagicMock()
        mock_step_info.step_name = 'test_component'
        mock_step_info.workspace_id = 'test_workspace'
        mock_step_info.file_components = {
            'builder': MagicMock(path=Path('/test/builder.py')),
            'config': MagicMock(path=Path('/test/config.py')),
            'contract': MagicMock(path=Path('/test/contract.py')),
            'spec': MagicMock(path=Path('/test/spec.py')),
            'script': MagicMock(path=Path('/test/script.py'))
        }
        validator.catalog.get_step_info.return_value = mock_step_info
        
        # Mock file existence
        with patch('pathlib.Path.exists', return_value=True):
            result = validator.validate_component_quality('test_component')
        
        assert result.is_valid is True
        assert result.details['quality_score'] == 100  # Perfect score with all components
        assert result.details['step_name'] == 'test_component'

    def test_validate_component_quality_not_found(self, validator, mock_unified_tester):
        """Test component quality validation when component not found."""
        validator.catalog.get_step_info.return_value = None
        
        result = validator.validate_component_quality('nonexistent_component')
        
        assert result.is_valid is False
        assert 'Component not found' in result.errors[0]

    def test_validate_component_quality_low_score(self, validator):
        """Test component quality validation with very low quality score."""
        # Mock component exists with missing files (causes quality issues)
        mock_step_info = MagicMock()
        mock_step_info.step_name = 'low_quality_component'
        mock_step_info.workspace_id = 'test_workspace'
        mock_step_info.file_components = {
            'script': MagicMock(path=Path('/test/script.py')),
            'builder': MagicMock(path=Path('/test/builder.py')),
            'config': MagicMock(path=Path('/test/config.py'))
        }
        validator.catalog.get_step_info.return_value = mock_step_info
        
        # Mock some files don't exist (causes validation errors)
        with patch('pathlib.Path.exists', return_value=False):
            result = validator.validate_component_quality('low_quality_component')
        
        assert result.is_valid is False  # Invalid due to missing files
        assert len(result.errors) > 0
        assert 'File not accessible' in result.errors[0]

    def test_validate_cross_workspace_compatibility_success(self, validator):
        """Test successful cross-workspace compatibility validation."""
        # Mock workspace components with different names (no conflicts)
        validator.catalog.list_available_steps.side_effect = [
            ['comp1'],  # workspace1 components
            ['comp2']   # workspace2 components
        ]
        
        mock_step_info1 = MagicMock()
        mock_step_info1.workspace_id = 'workspace1'
        mock_step_info1.registry_data = {'version': '1.0.0', 'framework': 'sklearn'}
        
        mock_step_info2 = MagicMock()
        mock_step_info2.workspace_id = 'workspace2'
        mock_step_info2.registry_data = {'version': '1.0.0', 'framework': 'sklearn'}
        
        validator.catalog.get_step_info.side_effect = [mock_step_info1, mock_step_info2]
        
        result = validator.validate_cross_workspace_compatibility(['workspace1', 'workspace2'])
        
        assert result.is_compatible is True
        assert len(result.issues) == 0

    def test_validate_cross_workspace_compatibility_version_conflict(self, validator):
        """Test cross-workspace compatibility with name conflicts."""
        # Mock workspace components with same names (name conflicts)
        validator.catalog.list_available_steps.side_effect = [
            ['comp1'],  # workspace1 components
            ['comp1']   # workspace2 components - same name = conflict
        ]
        
        mock_step_info1 = MagicMock()
        mock_step_info1.workspace_id = 'workspace1'
        mock_step_info1.registry_data = {'version': '1.0.0', 'framework': 'sklearn'}
        
        mock_step_info2 = MagicMock()
        mock_step_info2.workspace_id = 'workspace2'
        mock_step_info2.registry_data = {'version': '2.0.0', 'framework': 'sklearn'}
        
        validator.catalog.get_step_info.side_effect = [mock_step_info1, mock_step_info2]
        
        result = validator.validate_cross_workspace_compatibility(['workspace1', 'workspace2'])
        
        assert result.is_compatible is False
        assert len(result.issues) > 0
        assert 'component name conflict' in result.issues[0].lower()

    def test_validate_cross_workspace_compatibility_framework_conflict(self, validator):
        """Test cross-workspace compatibility with name conflicts (same as version conflict test)."""
        # Mock workspace components with same names (name conflicts)
        validator.catalog.list_available_steps.side_effect = [
            ['comp1'],  # workspace1 components
            ['comp1']   # workspace2 components - same name = conflict
        ]
        
        mock_step_info1 = MagicMock()
        mock_step_info1.workspace_id = 'workspace1'
        mock_step_info1.registry_data = {'version': '1.0.0', 'framework': 'sklearn'}
        
        mock_step_info2 = MagicMock()
        mock_step_info2.workspace_id = 'workspace2'
        mock_step_info2.registry_data = {'version': '1.0.0', 'framework': 'pytorch'}
        
        validator.catalog.get_step_info.side_effect = [mock_step_info1, mock_step_info2]
        
        result = validator.validate_cross_workspace_compatibility(['workspace1', 'workspace2'])
        
        assert result.is_compatible is False
        assert len(result.issues) > 0
        assert 'component name conflict' in result.issues[0].lower()

    def test_get_validation_summary(self, validator):
        """Test validation summary generation."""
        # Perform some validations to populate metrics
        validator.metrics['validations_performed'] = 2
        validator.metrics['components_validated'] = 5
        validator.metrics['compatibility_checks'] = 1
        
        summary = validator.get_validation_summary()
        
        assert 'metrics' in summary
        assert summary['metrics']['validations_performed'] == 2
        assert summary['metrics']['components_validated'] == 5
        assert summary['metrics']['compatibility_checks'] == 1

    def test_error_handling_in_workspace_validation(self, validator, mock_unified_tester):
        """Test error handling during workspace validation."""
        # Mock step catalog to raise exception
        validator.catalog.list_available_steps.side_effect = Exception("Catalog error")
        
        result = validator.validate_workspace_components('error_workspace')
        
        assert result.is_valid is False
        assert 'Validation failed' in result.errors[0]

    def test_error_handling_in_quality_validation(self, validator):
        """Test error handling during quality validation."""
        # Mock component exists but catalog access fails
        validator.catalog.get_step_info.side_effect = Exception("Catalog error")
        
        result = validator.validate_component_quality('error_component')
        
        assert result.is_valid is False
        assert 'Quality validation failed' in result.errors[0]

    def test_validation_with_missing_scoring_info(self, validator):
        """Test quality validation with component that has missing files."""
        # Mock component exists with files that don't exist
        mock_step_info = MagicMock()
        mock_step_info.step_name = 'minimal_component'
        mock_step_info.workspace_id = 'test_workspace'
        mock_step_info.file_components = {
            'script': MagicMock(path=Path('/test/script.py')),
            'builder': MagicMock(path=Path('/test/builder.py'))
        }
        validator.catalog.get_step_info.return_value = mock_step_info
        
        # Mock files don't exist (causes validation failure)
        with patch('pathlib.Path.exists', return_value=False):
            result = validator.validate_component_quality('minimal_component')
        
        assert result.is_valid is False  # Invalid due to missing files
        assert len(result.errors) > 0
        assert 'File not found' in result.errors[0]

    def test_compatibility_validation_with_missing_registry_data(self, validator):
        """Test compatibility validation when registry data is missing."""
        # Mock workspace components with different names (no conflicts)
        validator.catalog.list_available_steps.side_effect = [
            ['comp1'],  # workspace1 components
            ['comp2']   # workspace2 components - different names
        ]
        
        mock_step_info1 = MagicMock()
        mock_step_info1.workspace_id = 'workspace1'
        mock_step_info1.registry_data = {}
        
        mock_step_info2 = MagicMock()
        mock_step_info2.workspace_id = 'workspace2'
        mock_step_info2.registry_data = None
        
        validator.catalog.get_step_info.side_effect = [mock_step_info1, mock_step_info2]
        
        result = validator.validate_cross_workspace_compatibility(['workspace1', 'workspace2'])
        
        # Should be compatible when no name conflicts and registry data is missing
        assert result.is_compatible is True

    def test_validation_result_model(self):
        """Test ValidationResult model functionality."""
        result = ValidationResult(
            is_valid=True,
            errors=['Error 1', 'Error 2'],
            warnings=['Warning 1'],
            details={'key': 'value'}
        )
        
        assert result.is_valid is True
        assert len(result.errors) == 2
        assert len(result.warnings) == 1
        assert result.details['key'] == 'value'

    def test_compatibility_result_model(self):
        """Test CompatibilityResult model functionality."""
        result = CompatibilityResult(
            is_compatible=False,
            issues=['Issue 1', 'Issue 2'],
            compatibility_matrix={'workspace1': {'conflicts': 1}}
        )
        
        assert result.is_compatible is False
        assert len(result.issues) == 2
        assert 'workspace1' in result.compatibility_matrix

    def test_validation_with_empty_workspace_list(self, validator):
        """Test compatibility validation with empty workspace list."""
        result = validator.validate_cross_workspace_compatibility([])
        
        assert result.is_compatible is True  # Empty list is compatible
        assert len(result.issues) == 0

    def test_validation_with_single_workspace(self, validator):
        """Test compatibility validation with single workspace."""
        result = validator.validate_cross_workspace_compatibility(['single_workspace'])
        
        assert result.is_compatible is True  # Single workspace is always compatible
        assert len(result.issues) == 0

    def test_concurrent_validation_safety(self, validator, mock_unified_tester):
        """Test that validator handles concurrent validations safely."""
        import threading
        
        # Mock step catalog
        mock_step_info = MagicMock()
        mock_step_info.workspace_id = 'test_workspace'
        
        validator.catalog.list_available_steps.return_value = ['comp1']
        validator.catalog.get_step_info.return_value = mock_step_info
        
        mock_unified_tester.validate_specific_script.return_value = {
            'overall_status': 'PASSING', 'script_name': 'comp1'
        }
        
        results = []
        errors = []
        
        def worker():
            try:
                result = validator.validate_workspace_components('test_workspace')
                results.append(result.is_valid)
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
        assert all(results)  # All should be valid


class TestWorkspaceValidatorIntegration:
    """Integration tests for WorkspaceValidator with realistic scenarios."""

    @pytest.fixture
    def realistic_catalog(self):
        """Create a realistic step catalog mock."""
        mock_catalog = MagicMock()
        
        # Mock realistic step info
        mock_step_info1 = MagicMock()
        mock_step_info1.workspace_id = 'data_science'
        mock_step_info1.registry_data = {
            'version': '1.0.0',
            'framework': 'sklearn',
            'dependencies': ['pandas', 'numpy']
        }
        
        mock_step_info2 = MagicMock()
        mock_step_info2.workspace_id = 'ml_ops'
        mock_step_info2.registry_data = {
            'version': '1.0.0',
            'framework': 'sklearn',
            'dependencies': ['pandas', 'numpy', 'mlflow']
        }
        
        mock_catalog.list_available_steps.return_value = ['data_processor', 'model_trainer']
        mock_catalog.get_step_info.side_effect = [mock_step_info1, mock_step_info2]
        
        return mock_catalog

    def test_realistic_workspace_validation(self, realistic_catalog):
        """Test workspace validation with realistic components."""
        # Mock realistic step info with file components
        mock_step_info1 = MagicMock()
        mock_step_info1.step_name = 'data_processor'
        mock_step_info1.workspace_id = 'data_science'
        mock_step_info1.file_components = {
            'builder': MagicMock(path=Path('/data_science/builders/data_processor.py')),
            'config': MagicMock(path=Path('/data_science/configs/data_processor.py'))
        }
        
        mock_step_info2 = MagicMock()
        mock_step_info2.step_name = 'model_trainer'
        mock_step_info2.workspace_id = 'ml_ops'
        mock_step_info2.file_components = {
            'builder': MagicMock(path=Path('/ml_ops/builders/model_trainer.py')),
            'config': MagicMock(path=Path('/ml_ops/configs/model_trainer.py'))
        }
        
        realistic_catalog.get_step_info.side_effect = [mock_step_info1, mock_step_info2]
        
        validator = WorkspaceValidator(realistic_catalog)
        
        # Mock file existence
        with patch('pathlib.Path.exists', return_value=True):
            result = validator.validate_workspace_components('data_science')
        
        assert result.is_valid is True
        assert result.details['validated_components'] == 2  # Both components validated

    def test_realistic_compatibility_validation(self, realistic_catalog):
        """Test compatibility validation with realistic workspace data."""
        # Mock different components in each workspace (no conflicts)
        realistic_catalog.list_available_steps.side_effect = [
            ['data_processor'],  # data_science components
            ['model_trainer']    # ml_ops components
        ]
        
        validator = WorkspaceValidator(realistic_catalog)
        
        result = validator.validate_cross_workspace_compatibility(['data_science', 'ml_ops'])
        
        # Should be compatible (different component names)
        assert result.is_compatible is True

    def test_validator_resilience_to_catalog_failures(self):
        """Test validator resilience when catalog operations fail."""
        mock_catalog = MagicMock()
        mock_catalog.list_available_steps.side_effect = Exception("Catalog failure")
        
        validator = WorkspaceValidator(mock_catalog)
        
        # Should handle catalog failures gracefully
        result = validator.validate_workspace_components('failing_workspace')
        
        assert result.is_valid is False
        assert 'Validation failed' in result.errors[0]


if __name__ == "__main__":
    pytest.main([__file__])
