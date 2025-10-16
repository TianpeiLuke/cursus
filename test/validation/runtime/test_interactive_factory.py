"""
Test Interactive Runtime Testing Factory

Tests for the InteractiveRuntimeTestingFactory implementation,
validating DAG-guided script discovery, interactive configuration,
and end-to-end testing orchestration.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.cursus.validation.runtime.interactive_factory import InteractiveRuntimeTestingFactory
from src.cursus.validation.runtime.runtime_models import ScriptExecutionSpec
from src.cursus.api.dag.base_dag import PipelineDAG


class TestInteractiveRuntimeTestingFactory:
    """Test suite for InteractiveRuntimeTestingFactory."""
    
    @pytest.fixture
    def mock_dag(self):
        """Create a mock DAG for testing."""
        dag = Mock(spec=PipelineDAG)
        dag.nodes = ['data_preprocessing', 'model_training', 'model_evaluation']
        dag.name = 'test_pipeline'
        return dag
    
    @pytest.fixture
    def mock_script_spec(self):
        """Create a mock ScriptExecutionSpec for testing."""
        return ScriptExecutionSpec(
            script_name='data_preprocessing',
            step_name='data_preprocessing',
            script_path='scripts/data_preprocessing.py',
            input_paths={'raw_data': 'test/data/raw.csv'},
            output_paths={'processed_data': 'test/output/processed.csv'},
            environ_vars={'CURSUS_ENV': 'testing'},
            job_args={'job_type': 'testing'}
        )
    
    @patch('src.cursus.validation.runtime.interactive_factory.StepCatalog')
    @patch('src.cursus.validation.runtime.interactive_factory.PipelineTestingSpecBuilder')
    def test_factory_initialization(self, mock_spec_builder, mock_step_catalog, mock_dag):
        """Test factory initialization with DAG analysis."""
        # Setup mocks
        mock_spec_builder_instance = Mock()
        mock_spec_builder.return_value = mock_spec_builder_instance
        mock_spec_builder_instance._resolve_script_with_step_catalog_if_available.return_value = None
        mock_spec_builder_instance.resolve_script_execution_spec_from_node.side_effect = Exception("No script found")
        
        # Initialize factory
        factory = InteractiveRuntimeTestingFactory(mock_dag, "test/workspace")
        
        # Verify initialization
        assert factory.dag == mock_dag
        assert factory.workspace_dir == Path("test/workspace")
        assert len(factory.script_info_cache) == 3  # 3 nodes in mock DAG
        assert len(factory.pending_scripts) == 3  # All scripts need configuration
        assert len(factory.auto_configured_scripts) == 0  # No auto-configured scripts
    
    @patch('src.cursus.validation.runtime.interactive_factory.StepCatalog')
    @patch('src.cursus.validation.runtime.interactive_factory.PipelineTestingSpecBuilder')
    def test_script_discovery_with_existing_specs(self, mock_spec_builder, mock_step_catalog, mock_dag, mock_script_spec):
        """Test script discovery when specs can be resolved."""
        # Setup mocks
        mock_spec_builder_instance = Mock()
        mock_spec_builder.return_value = mock_spec_builder_instance
        mock_spec_builder_instance._resolve_script_with_step_catalog_if_available.return_value = mock_script_spec
        
        # Mock file existence for auto-configuration
        with patch('pathlib.Path.exists', return_value=True):
            factory = InteractiveRuntimeTestingFactory(mock_dag, "test/workspace")
        
        # Verify script discovery
        assert 'data_preprocessing' in factory.script_info_cache
        assert factory.script_info_cache['data_preprocessing']['script_name'] == 'data_preprocessing'
        assert factory.script_info_cache['data_preprocessing']['step_name'] == 'data_preprocessing'
        assert 'data_preprocessing' in factory.auto_configured_scripts  # Auto-configured due to existing files
    
    @patch('src.cursus.validation.runtime.interactive_factory.StepCatalog')
    @patch('src.cursus.validation.runtime.interactive_factory.PipelineTestingSpecBuilder')
    def test_get_scripts_requiring_testing(self, mock_spec_builder, mock_step_catalog, mock_dag):
        """Test getting all scripts that require testing."""
        # Setup mocks
        mock_spec_builder_instance = Mock()
        mock_spec_builder.return_value = mock_spec_builder_instance
        mock_spec_builder_instance._resolve_script_with_step_catalog_if_available.return_value = None
        mock_spec_builder_instance.resolve_script_execution_spec_from_node.side_effect = Exception("No script found")
        
        factory = InteractiveRuntimeTestingFactory(mock_dag, "test/workspace")
        
        # Test getting scripts requiring testing
        scripts = factory.get_scripts_requiring_testing()
        assert len(scripts) == 3
        assert 'data_preprocessing' in scripts
        assert 'model_training' in scripts
        assert 'model_evaluation' in scripts
    
    @patch('src.cursus.validation.runtime.interactive_factory.StepCatalog')
    @patch('src.cursus.validation.runtime.interactive_factory.PipelineTestingSpecBuilder')
    def test_get_script_testing_requirements(self, mock_spec_builder, mock_step_catalog, mock_dag):
        """Test getting testing requirements for a specific script."""
        # Setup mocks
        mock_spec_builder_instance = Mock()
        mock_spec_builder.return_value = mock_spec_builder_instance
        mock_spec_builder_instance._resolve_script_with_step_catalog_if_available.return_value = None
        mock_spec_builder_instance.resolve_script_execution_spec_from_node.side_effect = Exception("No script found")
        
        factory = InteractiveRuntimeTestingFactory(mock_dag, "test/workspace")
        
        # Test getting requirements
        requirements = factory.get_script_testing_requirements('data_preprocessing')
        
        assert requirements['script_name'] == 'data_preprocessing'
        assert requirements['step_name'] == 'data_preprocessing'
        assert len(requirements['expected_inputs']) == 1
        assert requirements['expected_inputs'][0]['name'] == 'data_input'
        assert len(requirements['expected_outputs']) == 1
        assert requirements['expected_outputs'][0]['name'] == 'data_output'
        assert not requirements['auto_configurable']  # Should be False for pending scripts
    
    @patch('src.cursus.validation.runtime.interactive_factory.StepCatalog')
    @patch('src.cursus.validation.runtime.interactive_factory.PipelineTestingSpecBuilder')
    def test_configure_script_testing_success(self, mock_spec_builder, mock_step_catalog, mock_dag):
        """Test successful script configuration."""
        # Setup mocks
        mock_spec_builder_instance = Mock()
        mock_spec_builder.return_value = mock_spec_builder_instance
        mock_spec_builder_instance._resolve_script_with_step_catalog_if_available.return_value = None
        mock_spec_builder_instance.resolve_script_execution_spec_from_node.side_effect = Exception("No script found")
        
        factory = InteractiveRuntimeTestingFactory(mock_dag, "test/workspace")
        
        # Mock file existence for validation
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value.st_size = 100  # Non-empty file
            
            # Configure script
            spec = factory.configure_script_testing(
                'data_preprocessing',
                expected_inputs={'data_input': 'test/input.csv'},
                expected_outputs={'data_output': 'test/output.csv'}
            )
        
        # Verify configuration
        assert isinstance(spec, ScriptExecutionSpec)
        assert spec.script_name == 'data_preprocessing'
        assert spec.input_paths == {'data_input': 'test/input.csv'}
        assert spec.output_paths == {'data_output': 'test/output.csv'}
        assert 'data_preprocessing' not in factory.pending_scripts  # Should be removed from pending
        assert 'data_preprocessing' in factory.script_specs  # Should be added to configured specs
    
    @patch('src.cursus.validation.runtime.interactive_factory.StepCatalog')
    @patch('src.cursus.validation.runtime.interactive_factory.PipelineTestingSpecBuilder')
    def test_configure_script_testing_validation_failure(self, mock_spec_builder, mock_step_catalog, mock_dag):
        """Test script configuration with validation failure."""
        # Setup mocks
        mock_spec_builder_instance = Mock()
        mock_spec_builder.return_value = mock_spec_builder_instance
        mock_spec_builder_instance._resolve_script_with_step_catalog_if_available.return_value = None
        mock_spec_builder_instance.resolve_script_execution_spec_from_node.side_effect = Exception("No script found")
        
        factory = InteractiveRuntimeTestingFactory(mock_dag, "test/workspace")
        
        # Mock file non-existence for validation failure
        with patch('pathlib.Path.exists', return_value=False):
            # Attempt to configure script with non-existent input file
            with pytest.raises(ValueError, match="Configuration validation failed"):
                factory.configure_script_testing(
                    'data_preprocessing',
                    expected_inputs={'data_input': 'test/nonexistent.csv'},
                    expected_outputs={'data_output': 'test/output.csv'}
                )
    
    @patch('src.cursus.validation.runtime.interactive_factory.StepCatalog')
    @patch('src.cursus.validation.runtime.interactive_factory.PipelineTestingSpecBuilder')
    def test_get_testing_factory_summary(self, mock_spec_builder, mock_step_catalog, mock_dag):
        """Test getting factory summary information."""
        # Setup mocks
        mock_spec_builder_instance = Mock()
        mock_spec_builder.return_value = mock_spec_builder_instance
        mock_spec_builder_instance._resolve_script_with_step_catalog_if_available.return_value = None
        mock_spec_builder_instance.resolve_script_execution_spec_from_node.side_effect = Exception("No script found")
        
        factory = InteractiveRuntimeTestingFactory(mock_dag, "test/workspace")
        
        # Get summary
        summary = factory.get_testing_factory_summary()
        
        # Verify summary
        assert summary['dag_name'] == 'test_pipeline'
        assert summary['total_scripts'] == 3
        assert summary['configured_scripts'] == 0
        assert summary['auto_configured_scripts'] == 0
        assert summary['manually_configured_scripts'] == 0
        assert summary['pending_scripts'] == 3
        assert not summary['ready_for_testing']
        assert summary['completion_percentage'] == 0.0
        assert len(summary['script_details']) == 3
    
    @patch('src.cursus.validation.runtime.interactive_factory.StepCatalog')
    @patch('src.cursus.validation.runtime.interactive_factory.PipelineTestingSpecBuilder')
    @patch('src.cursus.validation.runtime.interactive_factory.RuntimeTester')
    def test_execute_dag_guided_testing_with_pending_scripts(self, mock_runtime_tester, mock_spec_builder, mock_step_catalog, mock_dag):
        """Test execution failure when scripts are still pending."""
        # Setup mocks
        mock_spec_builder_instance = Mock()
        mock_spec_builder.return_value = mock_spec_builder_instance
        mock_spec_builder_instance._resolve_script_with_step_catalog_if_available.return_value = None
        mock_spec_builder_instance.resolve_script_execution_spec_from_node.side_effect = Exception("No script found")
        
        factory = InteractiveRuntimeTestingFactory(mock_dag, "test/workspace")
        
        # Attempt to execute testing with pending scripts
        with pytest.raises(ValueError, match="Cannot execute testing - missing configuration"):
            factory.execute_dag_guided_testing()
    
    @patch('src.cursus.validation.runtime.interactive_factory.StepCatalog')
    @patch('src.cursus.validation.runtime.interactive_factory.PipelineTestingSpecBuilder')
    def test_validate_configuration_preview(self, mock_spec_builder, mock_step_catalog, mock_dag):
        """Test configuration validation preview utility."""
        # Setup mocks
        mock_spec_builder_instance = Mock()
        mock_spec_builder.return_value = mock_spec_builder_instance
        mock_spec_builder_instance._resolve_script_with_step_catalog_if_available.return_value = None
        mock_spec_builder_instance.resolve_script_execution_spec_from_node.side_effect = Exception("No script found")
        
        factory = InteractiveRuntimeTestingFactory(mock_dag, "test/workspace")
        
        # Test validation preview with non-existent file
        with patch('pathlib.Path.exists', return_value=False):
            issues = factory.validate_configuration_preview(
                'data_preprocessing',
                {'data_input': 'test/nonexistent.csv'}
            )
            
            assert len(issues) == 1
            assert "Input file missing" in issues[0]
    
    @patch('src.cursus.validation.runtime.interactive_factory.StepCatalog')
    @patch('src.cursus.validation.runtime.interactive_factory.PipelineTestingSpecBuilder')
    def test_get_script_info(self, mock_spec_builder, mock_step_catalog, mock_dag):
        """Test getting basic script information."""
        # Setup mocks
        mock_spec_builder_instance = Mock()
        mock_spec_builder.return_value = mock_spec_builder_instance
        mock_spec_builder_instance._resolve_script_with_step_catalog_if_available.return_value = None
        mock_spec_builder_instance.resolve_script_execution_spec_from_node.side_effect = Exception("No script found")
        
        factory = InteractiveRuntimeTestingFactory(mock_dag, "test/workspace")
        
        # Get script info
        info = factory.get_script_info('data_preprocessing')
        
        assert info['script_name'] == 'data_preprocessing'
        assert info['step_name'] == 'data_preprocessing'
        assert info['script_path'] == 'scripts/data_preprocessing.py'
        assert 'data_input' in info['expected_inputs']
        assert 'data_output' in info['expected_outputs']
        assert not info['auto_configurable']


class TestInteractiveFactoryIntegration:
    """Integration tests for InteractiveRuntimeTestingFactory."""
    
    def test_factory_import(self):
        """Test that the factory can be imported from the runtime module."""
        from src.cursus.validation.runtime import InteractiveRuntimeTestingFactory
        assert InteractiveRuntimeTestingFactory is not None
    
    def test_factory_in_all_exports(self):
        """Test that the factory is included in __all__ exports."""
        from src.cursus.validation.runtime import __all__
        assert 'InteractiveRuntimeTestingFactory' in __all__
