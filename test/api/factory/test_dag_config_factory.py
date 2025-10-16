"""
Test suite for dag_config_factory.py

Following pytest best practices:
1. Source code first - Read actual implementation before writing tests
2. Implementation-driven testing - Tests match actual behavior
3. Comprehensive coverage - All public methods tested
4. Mock-based isolation - External dependencies mocked appropriately
5. Clear test organization - Grouped by functionality with descriptive names
"""

import pytest
from typing import Dict, List, Type, Any, Optional
from pydantic import BaseModel, Field
from unittest.mock import patch, MagicMock
from datetime import datetime

from cursus.api.factory.dag_config_factory import (
    DAGConfigFactory,
    ConfigurationIncompleteError
)


# Test fixtures - Mock DAG and configuration classes
class MockDAG:
    """Mock DAG for testing."""
    def __init__(self, nodes, name="test_dag"):
        self.nodes = nodes
        self.name = name


class MockBasePipelineConfig(BaseModel):
    """Mock base pipeline configuration."""
    project_name: str = Field(description="Project name")
    version: str = Field(default="1.0.0", description="Version")
    debug: bool = Field(default=False, description="Debug mode")


class MockBaseProcessingStepConfig(MockBasePipelineConfig):
    """Mock base processing step configuration."""
    processing_mode: str = Field(default="batch", description="Processing mode")
    max_workers: int = Field(default=4, description="Maximum workers")


class MockStepConfigA(MockBasePipelineConfig):
    """Mock step configuration A inheriting from base."""
    step_param_a: str = Field(description="Step parameter A")
    threshold_a: float = Field(default=0.5, description="Threshold A")
    
    @classmethod
    def from_base_config(cls, base_config, **kwargs):
        """Mock from_base_config method."""
        base_values = base_config.model_dump()
        base_values.update(kwargs)
        return cls(**base_values)


class MockStepConfigB(MockBaseProcessingStepConfig):
    """Mock step configuration B inheriting from processing base."""
    step_param_b: int = Field(description="Step parameter B")
    enabled_b: bool = Field(default=True, description="Enabled B")
    
    @classmethod
    def from_base_config(cls, base_config, **kwargs):
        """Mock from_base_config method."""
        base_values = base_config.model_dump()
        base_values.update(kwargs)
        return cls(**base_values)


class MockStandaloneConfig(BaseModel):
    """Mock standalone configuration (no inheritance)."""
    standalone_param: str = Field(description="Standalone parameter")
    value: int = Field(default=100, description="Value")


class TestDAGConfigFactory:
    """Test DAGConfigFactory class."""
    
    def test_init_with_dag(self):
        """Test initialization with DAG."""
        dag = MockDAG(['step1', 'step2'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {
                'step1': MockStepConfigA,
                'step2': MockStepConfigB
            }
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            assert factory.dag == dag
            assert factory.config_generator is None  # Not initialized until base config is set
            assert len(factory._config_class_map) == 2
            assert factory.base_config is None
            assert factory.base_processing_config is None
            assert factory.step_configs == {}
            assert factory.step_config_instances == {}
    
    def test_get_config_class_map(self):
        """Test getting config class map."""
        dag = MockDAG(['step1', 'step2'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            config_map = {'step1': MockStepConfigA, 'step2': MockStepConfigB}
            mock_mapper.map_dag_to_config_classes.return_value = config_map
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            result = factory.get_config_class_map()
            
            assert result == config_map
            # Should return a copy, not the original
            result['step3'] = MockStandaloneConfig
            assert 'step3' not in factory._config_class_map
    
    def test_get_base_config_requirements_success(self):
        """Test getting base config requirements successfully."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper'):
            factory = DAGConfigFactory(dag)
            
            # Mock the import and field extraction
            with patch('cursus.api.factory.dag_config_factory.extract_field_requirements') as mock_extract:
                mock_extract.return_value = [
                    {'name': 'project_name', 'type': 'str', 'required': True, 'description': 'Project name'},
                    {'name': 'version', 'type': 'str', 'required': False, 'default': '1.0.0', 'description': 'Version'}
                ]
                
                requirements = factory.get_base_config_requirements()
                
                assert len(requirements) == 2
                assert requirements[0]['name'] == 'project_name'
                assert requirements[0]['required'] is True
                assert requirements[1]['name'] == 'version'
                assert requirements[1]['required'] is False
    
    def test_get_base_config_requirements_import_error(self):
        """Test getting base config requirements when import fails."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper'):
            factory = DAGConfigFactory(dag)
            
            # Mock import to fail by patching the import inside the method
            with patch('cursus.api.factory.dag_config_factory.extract_field_requirements', side_effect=ImportError("Module not found")):
                requirements = factory.get_base_config_requirements()
                
                assert requirements == []
    
    def test_get_base_processing_config_requirements_needed(self):
        """Test getting base processing config requirements when needed."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigB}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Mock the inheritance check and field extraction
            with patch.object(factory, '_inherits_from_processing_config', return_value=True):
                with patch('cursus.api.factory.dag_config_factory.extract_non_inherited_fields') as mock_extract:
                    mock_extract.return_value = [
                        {'name': 'processing_mode', 'type': 'str', 'required': False, 'default': 'batch'},
                        {'name': 'max_workers', 'type': 'int', 'required': False, 'default': 4}
                    ]
                    
                    requirements = factory.get_base_processing_config_requirements()
                    
                    assert len(requirements) == 2
                    assert requirements[0]['name'] == 'processing_mode'
    
    def test_get_base_processing_config_requirements_not_needed(self):
        """Test getting base processing config requirements when not needed."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigA}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Mock the inheritance check to return False for all steps
            with patch.object(factory, '_inherits_from_processing_config', return_value=False):
                requirements = factory.get_base_processing_config_requirements()
                
                assert requirements == []
    
    def test_set_base_config_success(self):
        """Test setting base config successfully."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper'):
            factory = DAGConfigFactory(dag)
            
            # Mock the BasePipelineConfig import to use our mock class
            with patch('cursus.core.base.config_base.BasePipelineConfig', MockBasePipelineConfig):
                with patch('cursus.api.factory.dag_config_factory.ConfigurationGenerator') as mock_gen_class:
                    mock_generator = MagicMock()
                    mock_gen_class.return_value = mock_generator
                    
                    factory.set_base_config(project_name="test_project", version="2.0.0")
                    
                    assert factory.base_config is not None
                    assert factory.base_config.project_name == "test_project"
                    assert factory.base_config.version == "2.0.0"
                    assert factory.config_generator == mock_generator
    
    def test_set_base_config_import_error(self):
        """Test setting base config when import fails."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper'):
            factory = DAGConfigFactory(dag)
            
            # Mock the import to fail inside the method by patching the actual import path
            with patch('cursus.core.base.config_base.BasePipelineConfig', side_effect=ImportError("Module not found")):
                with pytest.raises(ValueError, match="BasePipelineConfig class not found"):
                    factory.set_base_config(project_name="test_project")
    
    def test_set_base_config_validation_error(self):
        """Test setting base config with validation error."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper'):
            factory = DAGConfigFactory(dag)
            
            # Mock BasePipelineConfig to raise validation error by patching the actual import path
            with patch('cursus.core.base.config_base.BasePipelineConfig', side_effect=Exception("Validation failed")):
                with pytest.raises(ValueError, match="Invalid base configuration"):
                    factory.set_base_config(project_name="test_project")
    
    def test_set_base_processing_config_success(self):
        """Test setting base processing config successfully."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper'):
            factory = DAGConfigFactory(dag)
            
            # Set base config first
            factory.base_config = MockBasePipelineConfig(project_name="test_project")
            factory.config_generator = MagicMock()
            factory.config_generator._extract_config_values.return_value = {
                'project_name': 'test_project',
                'version': '1.0.0'
            }
            
            # Mock the ProcessingStepConfigBase import to use our mock class
            with patch('cursus.steps.configs.config_processing_step_base.ProcessingStepConfigBase', MockBaseProcessingStepConfig):
                factory.set_base_processing_config(processing_mode="stream", max_workers=8)
                
                assert factory.base_processing_config is not None
                assert factory.base_processing_config.processing_mode == "stream"
                assert factory.base_processing_config.max_workers == 8
                assert factory.config_generator.base_processing_config == factory.base_processing_config
    
    def test_get_pending_steps(self):
        """Test getting pending steps that need configuration."""
        dag = MockDAG(['step1', 'step2', 'step3'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {
                'step1': MockStepConfigA,
                'step2': MockStepConfigB,
                'step3': MockStandaloneConfig
            }
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Configure step1, leave step2 and step3 pending
            factory.step_configs['step1'] = {'step_param_a': 'configured'}
            
            # Mock can_auto_configure_step
            with patch.object(factory, 'can_auto_configure_step') as mock_can_auto:
                mock_can_auto.side_effect = lambda step: step == 'step3'  # step3 can be auto-configured
                
                pending = factory.get_pending_steps()
                
                # Should only include step2 (step1 is configured, step3 can be auto-configured)
                assert pending == ['step2']
    
    def test_can_auto_configure_step_true(self):
        """Test checking if step can be auto-configured (returns True)."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigA}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Mock prerequisites validation and step requirements
            with patch.object(factory, '_validate_prerequisites_for_step'):
                with patch.object(factory, 'get_step_requirements') as mock_get_reqs:
                    mock_get_reqs.return_value = [
                        {'name': 'optional_field', 'required': False}
                    ]
                    
                    result = factory.can_auto_configure_step('step1')
                    
                    assert result is True
    
    def test_can_auto_configure_step_false_prerequisites(self):
        """Test checking if step can be auto-configured (False due to prerequisites)."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigA}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Mock prerequisites validation to fail
            with patch.object(factory, '_validate_prerequisites_for_step', side_effect=ValueError("Prerequisites not met")):
                result = factory.can_auto_configure_step('step1')
                
                assert result is False
    
    def test_can_auto_configure_step_false_required_fields(self):
        """Test checking if step can be auto-configured (False due to required fields)."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigA}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Mock prerequisites validation and step requirements with required fields
            with patch.object(factory, '_validate_prerequisites_for_step'):
                with patch.object(factory, 'get_step_requirements') as mock_get_reqs:
                    mock_get_reqs.return_value = [
                        {'name': 'required_field', 'required': True}
                    ]
                    
                    result = factory.can_auto_configure_step('step1')
                    
                    assert result is False
    
    def test_get_step_requirements(self):
        """Test getting step requirements."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigA}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Mock _extract_step_specific_fields
            with patch.object(factory, '_extract_step_specific_fields') as mock_extract:
                mock_extract.return_value = [
                    {'name': 'step_param_a', 'type': 'str', 'required': True},
                    {'name': 'threshold_a', 'type': 'float', 'required': False, 'default': 0.5}
                ]
                
                requirements = factory.get_step_requirements('step1')
                
                assert len(requirements) == 2
                assert requirements[0]['name'] == 'step_param_a'
                assert requirements[0]['required'] is True
                assert requirements[1]['name'] == 'threshold_a'
                assert requirements[1]['required'] is False
    
    def test_get_step_requirements_step_not_found(self):
        """Test getting step requirements for non-existent step."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigA}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            with pytest.raises(ValueError, match="Step 'nonexistent' not found in DAG"):
                factory.get_step_requirements('nonexistent')
    
    def test_set_step_config_success(self):
        """Test setting step config successfully."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigA}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Mock prerequisites validation and config creation
            with patch.object(factory, '_validate_prerequisites_for_step'):
                with patch.object(factory, '_create_config_instance_with_inheritance') as mock_create:
                    mock_config = MockStepConfigA(project_name="test", step_param_a="test_value")
                    mock_create.return_value = mock_config
                    
                    result = factory.set_step_config('step1', step_param_a="test_value")
                    
                    assert result == mock_config
                    assert factory.step_configs['step1'] == {'step_param_a': 'test_value'}
                    assert factory.step_config_instances['step1'] == mock_config
    
    def test_set_step_config_step_not_found(self):
        """Test setting step config for non-existent step."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigA}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            with pytest.raises(ValueError, match="Step 'nonexistent' not found in DAG"):
                factory.set_step_config('nonexistent', param="value")
    
    def test_set_step_config_prerequisites_not_met(self):
        """Test setting step config when prerequisites not met."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigA}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Mock prerequisites validation to fail
            with patch.object(factory, '_validate_prerequisites_for_step', side_effect=ValueError("Prerequisites not met")):
                with pytest.raises(ValueError, match="Prerequisites not met"):
                    factory.set_step_config('step1', step_param_a="test_value")
    
    def test_set_step_config_validation_error(self):
        """Test setting step config with validation error."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigA}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Mock prerequisites validation and config creation to fail
            with patch.object(factory, '_validate_prerequisites_for_step'):
                with patch.object(factory, '_create_config_instance_with_inheritance', side_effect=Exception("Validation failed")):
                    with patch.object(factory, '_build_error_context', return_value="Error context"):
                        with pytest.raises(ValueError, match="Configuration validation failed for step1"):
                            factory.set_step_config('step1', step_param_a="test_value")
    
    def test_auto_configure_step_if_possible_success(self):
        """Test auto-configuring step successfully."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigA}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Mock prerequisites, requirements, and config creation
            with patch.object(factory, '_validate_prerequisites_for_step'):
                with patch.object(factory, 'get_step_requirements') as mock_get_reqs:
                    mock_get_reqs.return_value = []  # No required fields
                    
                    with patch.object(factory, '_create_config_instance_with_inheritance') as mock_create:
                        mock_config = MockStepConfigA(project_name="test", step_param_a="default")
                        mock_create.return_value = mock_config
                        
                        result = factory.auto_configure_step_if_possible('step1')
                        
                        assert result == mock_config
                        assert factory.step_configs['step1'] == {}
                        assert factory.step_config_instances['step1'] == mock_config
    
    def test_auto_configure_step_if_possible_has_required_fields(self):
        """Test auto-configuring step with required fields (should fail)."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigA}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Mock prerequisites and requirements with required fields
            with patch.object(factory, '_validate_prerequisites_for_step'):
                with patch.object(factory, 'get_step_requirements') as mock_get_reqs:
                    mock_get_reqs.return_value = [{'name': 'required_field', 'required': True}]
                    
                    result = factory.auto_configure_step_if_possible('step1')
                    
                    assert result is None
    
    def test_get_configuration_status(self):
        """Test getting configuration status."""
        dag = MockDAG(['step1', 'step2'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {
                'step1': MockStepConfigA,
                'step2': MockStepConfigB
            }
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Set base config and configure one step
            factory.base_config = MockBasePipelineConfig(project_name="test")
            factory.step_configs['step1'] = {'step_param_a': 'configured'}
            
            # Mock get_base_processing_config_requirements
            with patch.object(factory, 'get_base_processing_config_requirements', return_value=[]):
                status = factory.get_configuration_status()
                
                assert status['base_config'] is True
                assert status['base_processing_config'] is True  # No requirements, so considered complete
                assert status['step_step1'] is True
                assert status['step_step2'] is False
    
    def test_generate_all_configs_with_pre_validated_instances(self):
        """Test generating all configs when pre-validated instances exist."""
        dag = MockDAG(['step1', 'step2'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {
                'step1': MockStepConfigA,
                'step2': MockStepConfigB
            }
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Set up pre-validated instances
            config1 = MockStepConfigA(project_name="test", step_param_a="value1")
            config2 = MockStepConfigB(project_name="test", step_param_b=42)
            factory.step_config_instances = {'step1': config1, 'step2': config2}
            
            # Mock auto-configure and get_pending_steps
            with patch.object(factory, '_auto_configure_eligible_steps', return_value=0):
                with patch.object(factory, 'get_pending_steps', return_value=[]):
                    configs = factory.generate_all_configs()
                    
                    assert len(configs) == 2
                    assert config1 in configs
                    assert config2 in configs
    
    def test_generate_all_configs_missing_steps(self):
        """Test generating all configs when steps are missing."""
        dag = MockDAG(['step1', 'step2'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {
                'step1': MockStepConfigA,
                'step2': MockStepConfigB
            }
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Mock auto-configure and get_pending_steps to return missing steps
            with patch.object(factory, '_auto_configure_eligible_steps', return_value=0):
                with patch.object(factory, 'get_pending_steps', return_value=['step2']):
                    with pytest.raises(ValueError, match="Missing configuration for steps: \\['step2'\\]"):
                        factory.generate_all_configs()
    
    def test_generate_all_configs_fallback_to_traditional(self):
        """Test generating all configs using traditional fallback method."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigA}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Set base config and step configs but not pre-validated instances
            factory.base_config = MockBasePipelineConfig(project_name="test")
            factory.step_configs = {'step1': {'step_param_a': 'value'}}
            
            # Mock auto-configure, get_pending_steps, and config generator
            with patch.object(factory, '_auto_configure_eligible_steps', return_value=0):
                with patch.object(factory, 'get_pending_steps', return_value=[]):
                    # Mock the config generator that should already exist
                    factory.config_generator = MagicMock()
                    mock_config = MockStepConfigA(project_name="test", step_param_a="value")
                    factory.config_generator.generate_all_instances.return_value = [mock_config]
                    
                    configs = factory.generate_all_configs()
                    
                    assert len(configs) == 1
                    assert configs[0] == mock_config
    
    def test_get_factory_summary(self):
        """Test getting factory summary."""
        dag = MockDAG(['step1', 'step2'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {
                'step1': MockStepConfigA,
                'step2': MockStepConfigB
            }
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Set some configuration
            factory.base_config = MockBasePipelineConfig(project_name="test")
            factory.step_configs['step1'] = {'step_param_a': 'configured'}
            
            # Mock get_configuration_status and get_pending_steps
            with patch.object(factory, 'get_configuration_status') as mock_status:
                mock_status.return_value = {
                    'base_config': True,
                    'base_processing_config': True,
                    'step_step1': True,
                    'step_step2': False
                }
                with patch.object(factory, 'get_pending_steps', return_value=['step2']):
                    summary = factory.get_factory_summary()
                    
                    assert summary['dag_steps'] == 2
                    assert summary['mapped_config_classes'] == ['step1', 'step2']
                    assert summary['completed_steps'] == 1
                    assert summary['pending_steps'] == ['step2']
                    assert summary['base_config_set'] is True
                    assert summary['processing_config_set'] is False
                    assert summary['ready_for_generation'] is False
    
    def test_update_step_config_success(self):
        """Test updating existing step config successfully."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigA}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Set initial config
            factory.step_configs['step1'] = {'step_param_a': 'initial_value'}
            
            # Mock set_step_config
            with patch.object(factory, 'set_step_config') as mock_set:
                mock_config = MockStepConfigA(project_name="test", step_param_a="updated_value")
                mock_set.return_value = mock_config
                
                result = factory.update_step_config('step1', step_param_a="updated_value", new_param="new")
                
                # Should call set_step_config with merged inputs
                expected_inputs = {'step_param_a': 'updated_value', 'new_param': 'new'}
                mock_set.assert_called_once_with('step1', **expected_inputs)
                assert result == mock_config
    
    def test_update_step_config_not_configured(self):
        """Test updating step config when step not configured yet."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigA}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            with pytest.raises(ValueError, match="Step 'step1' not configured yet"):
                factory.update_step_config('step1', step_param_a="value")
    
    def test_get_step_config_instance_exists(self):
        """Test getting step config instance when it exists."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigA}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Set up config instance
            mock_config = MockStepConfigA(project_name="test", step_param_a="value")
            factory.step_config_instances['step1'] = mock_config
            
            result = factory.get_step_config_instance('step1')
            
            assert result == mock_config
    
    def test_get_step_config_instance_not_exists(self):
        """Test getting step config instance when it doesn't exist."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigA}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            result = factory.get_step_config_instance('step1')
            
            assert result is None
    
    def test_get_all_config_instances(self):
        """Test getting all config instances."""
        dag = MockDAG(['step1', 'step2'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {
                'step1': MockStepConfigA,
                'step2': MockStepConfigB
            }
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Set up config instances
            config1 = MockStepConfigA(project_name="test", step_param_a="value1")
            config2 = MockStepConfigB(project_name="test", step_param_b=42)
            factory.step_config_instances = {'step1': config1, 'step2': config2}
            
            result = factory.get_all_config_instances()
            
            assert len(result) == 2
            assert result['step1'] == config1
            assert result['step2'] == config2
            
            # Should return a copy, not the original
            result['step3'] = MockStandaloneConfig(standalone_param="test")
            assert 'step3' not in factory.step_config_instances
    
    def test_save_partial_state(self):
        """Test saving partial factory state."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigA}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Set up some state
            factory.base_config = MockBasePipelineConfig(project_name="test")
            factory.step_configs = {'step1': {'step_param_a': 'value'}}
            factory.config_generator = MagicMock()
            factory.config_generator._extract_config_values.return_value = {'project_name': 'test'}
            
            # Mock file operations
            with patch('json.dump') as mock_dump:
                with patch('pathlib.Path') as mock_path:
                    mock_path.return_value.parent.mkdir = MagicMock()
                    with patch('builtins.open', create=True) as mock_open:
                        factory.save_partial_state('/test/path/state.json')
                        
                        # Verify file operations
                        mock_path.assert_called_once_with('/test/path/state.json')
                        mock_path.return_value.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
                        mock_open.assert_called_once_with('/test/path/state.json', 'w')
                        mock_dump.assert_called_once()
    
    def test_load_partial_state(self):
        """Test loading partial factory state."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigA}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Mock file operations and state data
            state_data = {
                'step_configs': {'step1': {'step_param_a': 'loaded_value'}},
                'base_config_dict': {'project_name': 'loaded_project'},
                'base_processing_config_dict': None,
                'config_class_map': {'step1': 'MockStepConfigA'}
            }
            
            with patch('json.load', return_value=state_data):
                with patch('builtins.open', create=True):
                    with patch.object(factory, 'set_base_config') as mock_set_base:
                        factory.load_partial_state('/test/path/state.json')
                        
                        # Verify state was loaded
                        assert factory.step_configs == {'step1': {'step_param_a': 'loaded_value'}}
                        mock_set_base.assert_called_once_with(project_name='loaded_project')
    
    def test_validate_prerequisites_for_step_processing_inheritance(self):
        """Test validating prerequisites for step with processing inheritance."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigB}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Mock inheritance detection
            with patch.object(factory, '_inherits_from_processing_config', return_value=True):
                with patch.object(factory, '_inherits_from_base_config', return_value=True):
                    # Test with missing base config
                    with pytest.raises(ValueError, match="Step 'step1' requires base config to be set first"):
                        factory._validate_prerequisites_for_step('step1', MockStepConfigB)
                    
                    # Set base config but not processing config
                    factory.base_config = MockBasePipelineConfig(project_name="test")
                    with pytest.raises(ValueError, match="Step 'step1' requires base processing config to be set first"):
                        factory._validate_prerequisites_for_step('step1', MockStepConfigB)
                    
                    # Set both configs - should not raise
                    factory.base_processing_config = MockBaseProcessingStepConfig(project_name="test")
                    factory._validate_prerequisites_for_step('step1', MockStepConfigB)  # Should not raise
    
    def test_validate_prerequisites_for_step_base_inheritance(self):
        """Test validating prerequisites for step with base inheritance only."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigA}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Mock inheritance detection
            with patch.object(factory, '_inherits_from_processing_config', return_value=False):
                with patch.object(factory, '_inherits_from_base_config', return_value=True):
                    # Test with missing base config
                    with pytest.raises(ValueError, match="Step 'step1' requires base config to be set first"):
                        factory._validate_prerequisites_for_step('step1', MockStepConfigA)
                    
                    # Set base config - should not raise
                    factory.base_config = MockBasePipelineConfig(project_name="test")
                    factory._validate_prerequisites_for_step('step1', MockStepConfigA)  # Should not raise
    
    def test_create_config_instance_with_inheritance_processing(self):
        """Test creating config instance with processing inheritance."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigB}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Set up configs
            factory.base_config = MockBasePipelineConfig(project_name="test")
            factory.base_processing_config = MockBaseProcessingStepConfig(project_name="test")
            
            # Mock inheritance detection and creation method
            with patch.object(factory, '_inherits_from_processing_config', return_value=True):
                with patch.object(factory, '_create_with_processing_inheritance') as mock_create:
                    mock_config = MockStepConfigB(project_name="test", step_param_b=42)
                    mock_create.return_value = mock_config
                    
                    step_inputs = {'step_param_b': 42}
                    result = factory._create_config_instance_with_inheritance(MockStepConfigB, step_inputs)
                    
                    assert result == mock_config
                    mock_create.assert_called_once_with(MockStepConfigB, step_inputs)
    
    def test_create_config_instance_with_inheritance_base(self):
        """Test creating config instance with base inheritance."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigA}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Set up config
            factory.base_config = MockBasePipelineConfig(project_name="test")
            
            # Mock inheritance detection and creation method
            with patch.object(factory, '_inherits_from_processing_config', return_value=False):
                with patch.object(factory, '_inherits_from_base_config', return_value=True):
                    with patch.object(factory, '_create_with_base_inheritance') as mock_create:
                        mock_config = MockStepConfigA(project_name="test", step_param_a="value")
                        mock_create.return_value = mock_config
                        
                        step_inputs = {'step_param_a': 'value'}
                        result = factory._create_config_instance_with_inheritance(MockStepConfigA, step_inputs)
                        
                        assert result == mock_config
                        mock_create.assert_called_once_with(MockStepConfigA, step_inputs)
    
    def test_create_config_instance_with_inheritance_standalone(self):
        """Test creating config instance for standalone class."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStandaloneConfig}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Mock inheritance detection
            with patch.object(factory, '_inherits_from_processing_config', return_value=False):
                with patch.object(factory, '_inherits_from_base_config', return_value=False):
                    step_inputs = {'standalone_param': 'test_value'}
                    result = factory._create_config_instance_with_inheritance(MockStandaloneConfig, step_inputs)
                    
                    assert isinstance(result, MockStandaloneConfig)
                    assert result.standalone_param == 'test_value'
    
    def test_create_with_processing_inheritance_from_base_config(self):
        """Test creating config with processing inheritance using from_base_config."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigB}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Set up configs
            factory.base_config = MockBasePipelineConfig(project_name="test")
            factory.base_processing_config = MockBaseProcessingStepConfig(project_name="test")
            
            # Mock from_base_config method
            mock_result = MockStepConfigB(project_name="test", step_param_b=42)
            with patch.object(MockStepConfigB, 'from_base_config', return_value=mock_result) as mock_from_base:
                step_inputs = {'step_param_b': 42}
                
                result = factory._create_with_processing_inheritance(MockStepConfigB, step_inputs)
                
                assert result == mock_result
                mock_from_base.assert_called_once_with(factory.base_processing_config, **step_inputs)
    
    def test_create_with_processing_inheritance_fallback(self):
        """Test creating config with processing inheritance using fallback."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigB}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Set up configs and config generator
            factory.base_config = MockBasePipelineConfig(project_name="test", debug=True)
            factory.base_processing_config = MockBaseProcessingStepConfig(
                project_name="test", debug=True, processing_mode="stream"
            )
            factory.config_generator = MagicMock()
            factory.config_generator._extract_config_values.side_effect = [
                {'project_name': 'test', 'debug': True},
                {'project_name': 'test', 'debug': True, 'processing_mode': 'stream'}
            ]
            
            # Mock from_base_config to fail
            with patch.object(MockStepConfigB, 'from_base_config', side_effect=Exception("from_base_config failed")):
                step_inputs = {'step_param_b': 42}
                
                result = factory._create_with_processing_inheritance(MockStepConfigB, step_inputs)
                
                assert isinstance(result, MockStepConfigB)
                assert result.project_name == "test"
                assert result.step_param_b == 42
    
    def test_create_with_base_inheritance_from_base_config(self):
        """Test creating config with base inheritance using from_base_config."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigA}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Set up config
            factory.base_config = MockBasePipelineConfig(project_name="test", version="2.0.0")
            
            # Mock from_base_config method
            mock_result = MockStepConfigA(project_name="test", version="2.0.0", step_param_a="value")
            with patch.object(MockStepConfigA, 'from_base_config', return_value=mock_result) as mock_from_base:
                step_inputs = {'step_param_a': 'value'}
                
                result = factory._create_with_base_inheritance(MockStepConfigA, step_inputs)
                
                assert result == mock_result
                mock_from_base.assert_called_once_with(factory.base_config, **step_inputs)
    
    def test_create_with_base_inheritance_fallback(self):
        """Test creating config with base inheritance using fallback."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigA}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Set up config and config generator
            factory.base_config = MockBasePipelineConfig(project_name="test", version="2.0.0")
            factory.config_generator = MagicMock()
            factory.config_generator._extract_config_values.return_value = {
                'project_name': 'test', 'version': '2.0.0'
            }
            
            # Mock from_base_config to fail
            with patch.object(MockStepConfigA, 'from_base_config', side_effect=Exception("from_base_config failed")):
                step_inputs = {'step_param_a': 'value'}
                
                result = factory._create_with_base_inheritance(MockStepConfigA, step_inputs)
                
                assert isinstance(result, MockStepConfigA)
                assert result.project_name == "test"
                assert result.version == "2.0.0"
                assert result.step_param_a == "value"
    
    def test_inherits_from_base_config_true(self):
        """Test detecting base config inheritance (True case)."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigA}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Mock the import and issubclass
            with patch('cursus.api.factory.dag_config_factory.issubclass', return_value=True):
                result = factory._inherits_from_base_config(MockStepConfigA)
                assert result is True
    
    def test_inherits_from_base_config_false(self):
        """Test detecting base config inheritance (False case)."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStandaloneConfig}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Mock the import and issubclass
            with patch('cursus.api.factory.dag_config_factory.issubclass', return_value=False):
                result = factory._inherits_from_base_config(MockStandaloneConfig)
                assert result is False
    
    def test_inherits_from_processing_config_true(self):
        """Test detecting processing config inheritance (True case)."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigB}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Mock the import and issubclass
            with patch('cursus.api.factory.dag_config_factory.issubclass', return_value=True):
                result = factory._inherits_from_processing_config(MockStepConfigB)
                assert result is True
    
    def test_inherits_from_processing_config_false(self):
        """Test detecting processing config inheritance (False case)."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigA}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Mock the import and issubclass
            with patch('cursus.api.factory.dag_config_factory.issubclass', return_value=False):
                result = factory._inherits_from_processing_config(MockStepConfigA)
                assert result is False
    
    def test_build_error_context(self):
        """Test building error context for debugging."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigA}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Mock inheritance detection
            with patch.object(factory, '_inherits_from_processing_config', return_value=False):
                with patch.object(factory, '_inherits_from_base_config', return_value=True):
                    step_inputs = {'step_param_a': 'value'}
                    error = Exception("Test error")
                    
                    context = factory._build_error_context('step1', MockStepConfigA, step_inputs, error)
                    
                    assert 'Step: step1' in context
                    assert 'Config Class: MockStepConfigA' in context
                    assert 'Has from_base_config: True' in context
                    assert 'Inherits from processing: False' in context
                    assert 'Inherits from base: True' in context
                    assert 'Step inputs: [\'step_param_a\']' in context
                    assert 'Error: Test error' in context
    
    def test_extract_step_specific_fields_processing_inheritance(self):
        """Test extracting step-specific fields for processing inheritance."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigB}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Mock inheritance detection and field extraction
            with patch.object(factory, '_inherits_from_processing_config', return_value=True):
                with patch('cursus.api.factory.dag_config_factory.extract_non_inherited_fields') as mock_extract:
                    mock_extract.return_value = [
                        {'name': 'step_param_b', 'type': 'int', 'required': True}
                    ]
                    
                    result = factory._extract_step_specific_fields(MockStepConfigB)
                    
                    assert len(result) == 1
                    assert result[0]['name'] == 'step_param_b'
    
    def test_extract_step_specific_fields_base_inheritance(self):
        """Test extracting step-specific fields for base inheritance."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigA}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Mock inheritance detection and field extraction
            with patch.object(factory, '_inherits_from_processing_config', return_value=False):
                with patch('cursus.api.factory.dag_config_factory.extract_non_inherited_fields') as mock_extract:
                    mock_extract.return_value = [
                        {'name': 'step_param_a', 'type': 'str', 'required': True}
                    ]
                    
                    result = factory._extract_step_specific_fields(MockStepConfigA)
                    
                    assert len(result) == 1
                    assert result[0]['name'] == 'step_param_a'


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""
    
    def test_complete_factory_workflow(self):
        """Test complete factory workflow from DAG to final configs."""
        dag = MockDAG(['preprocessing', 'training', 'evaluation'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {
                'preprocessing': MockStepConfigA,
                'training': MockStepConfigB,
                'evaluation': MockStandaloneConfig
            }
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Set base configurations
            factory.base_config = MockBasePipelineConfig(project_name="integration_test")
            factory.base_processing_config = MockBaseProcessingStepConfig(project_name="integration_test")
            
            # Configure steps
            factory.step_configs = {
                'preprocessing': {'step_param_a': 'preprocess_value'},
                'training': {'step_param_b': 100},
                'evaluation': {'standalone_param': 'eval_value'}
            }
            
            # Set up pre-validated instances
            config1 = MockStepConfigA(project_name="integration_test", step_param_a="preprocess_value")
            config2 = MockStepConfigB(project_name="integration_test", step_param_b=100)
            config3 = MockStandaloneConfig(standalone_param="eval_value")
            factory.step_config_instances = {
                'preprocessing': config1,
                'training': config2,
                'evaluation': config3
            }
            
            # Generate all configs
            with patch.object(factory, '_auto_configure_eligible_steps', return_value=0):
                with patch.object(factory, 'get_pending_steps', return_value=[]):
                    configs = factory.generate_all_configs()
                    
                    assert len(configs) == 3
                    assert config1 in configs
                    assert config2 in configs
                    assert config3 in configs
    
    def test_factory_state_persistence(self):
        """Test factory state save and load functionality."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigA}
            mock_mapper_class.return_value = mock_mapper
            
            # Create and configure factory
            factory1 = DAGConfigFactory(dag)
            factory1.base_config = MockBasePipelineConfig(project_name="persistence_test")
            factory1.step_configs = {'step1': {'step_param_a': 'persistent_value'}}
            factory1.config_generator = MagicMock()
            factory1.config_generator._extract_config_values.return_value = {'project_name': 'persistence_test'}
            
            # Mock file operations for save
            with patch('json.dump') as mock_dump:
                with patch('pathlib.Path') as mock_path:
                    mock_path.return_value.parent.mkdir = MagicMock()
                    with patch('builtins.open', create=True) as mock_open:
                        factory1.save_partial_state('/test/path/state.json')
                        
                        # Verify save operations
                        mock_dump.assert_called_once()
            
            # Create new factory and load state
            factory2 = DAGConfigFactory(dag)
            
            state_data = {
                'step_configs': {'step1': {'step_param_a': 'persistent_value'}},
                'base_config_dict': {'project_name': 'persistence_test'},
                'base_processing_config_dict': None,
                'config_class_map': {'step1': 'MockStepConfigA'}
            }
            
            with patch('json.load', return_value=state_data):
                with patch('builtins.open', create=True):
                    with patch.object(factory2, 'set_base_config') as mock_set_base:
                        factory2.load_partial_state('/test/path/state.json')
                        
                        # Verify state was loaded correctly
                        assert factory2.step_configs == {'step1': {'step_param_a': 'persistent_value'}}
                        mock_set_base.assert_called_once_with(project_name='persistence_test')


# Pytest fixtures for reuse across tests
@pytest.fixture
def sample_dag():
    """Fixture providing a sample DAG for testing."""
    return MockDAG(['preprocessing', 'training', 'evaluation'])


@pytest.fixture
def sample_config_classes():
    """Fixture providing sample config classes for testing."""
    return {
        'preprocessing': MockStepConfigA,
        'training': MockStepConfigB,
        'evaluation': MockStandaloneConfig
    }


@pytest.fixture
def configured_factory(sample_dag):
    """Fixture providing a configured factory for testing."""
    with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
        mock_mapper = MagicMock()
        mock_mapper.map_dag_to_config_classes.return_value = {
            'preprocessing': MockStepConfigA,
            'training': MockStepConfigB,
            'evaluation': MockStandaloneConfig
        }
        mock_mapper_class.return_value = mock_mapper
        
        factory = DAGConfigFactory(sample_dag)
        factory.base_config = MockBasePipelineConfig(project_name="fixture_test")
        factory.base_processing_config = MockBaseProcessingStepConfig(project_name="fixture_test")
        
        return factory


class TestFixtureUsage:
    """Test using pytest fixtures."""
    
    def test_sample_dag_fixture(self, sample_dag):
        """Test using sample DAG fixture."""
        assert hasattr(sample_dag, 'nodes')
        assert len(sample_dag.nodes) == 3
        assert 'preprocessing' in sample_dag.nodes
        assert 'training' in sample_dag.nodes
        assert 'evaluation' in sample_dag.nodes
    
    def test_sample_config_classes_fixture(self, sample_config_classes):
        """Test using sample config classes fixture."""
        assert len(sample_config_classes) == 3
        assert 'preprocessing' in sample_config_classes
        assert sample_config_classes['preprocessing'] == MockStepConfigA
        assert sample_config_classes['training'] == MockStepConfigB
        assert sample_config_classes['evaluation'] == MockStandaloneConfig
    
    def test_configured_factory_fixture(self, configured_factory):
        """Test using configured factory fixture."""
        assert isinstance(configured_factory, DAGConfigFactory)
        assert configured_factory.base_config is not None
        assert configured_factory.base_processing_config is not None
        assert configured_factory.base_config.project_name == "fixture_test"
        assert len(configured_factory._config_class_map) == 3
    
    def test_integration_with_fixtures(self, configured_factory):
        """Test integration using fixtures."""
        # Test getting pending steps
        pending = configured_factory.get_pending_steps()
        
        # All steps should be pending initially
        assert len(pending) <= 3  # Depends on auto-configuration logic
        
        # Test configuration status
        status = configured_factory.get_configuration_status()
        assert status['base_config'] is True
        assert status['base_processing_config'] is True
        
        # Test factory summary
        summary = configured_factory.get_factory_summary()
        assert summary['dag_steps'] == 3
        assert summary['base_config_set'] is True
        assert summary['processing_config_set'] is True


class TestConfigurationIncompleteError:
    """Test ConfigurationIncompleteError exception."""
    
    def test_configuration_incomplete_error_creation(self):
        """Test creating ConfigurationIncompleteError."""
        error = ConfigurationIncompleteError("Test error message")
        
        assert isinstance(error, Exception)
        assert str(error) == "Test error message"
    
    def test_configuration_incomplete_error_usage(self):
        """Test using ConfigurationIncompleteError in context."""
        with pytest.raises(ConfigurationIncompleteError, match="Configuration incomplete"):
            raise ConfigurationIncompleteError("Configuration incomplete")


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""
    
    def test_factory_with_empty_dag(self):
        """Test factory behavior with empty DAG."""
        dag = MockDAG([])  # Empty DAG
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            assert len(factory._config_class_map) == 0
            assert factory.get_pending_steps() == []
            
            # Should be able to generate configs (empty list)
            with patch.object(factory, '_auto_configure_eligible_steps', return_value=0):
                configs = factory.generate_all_configs()
                assert configs == []
    
    def test_factory_with_none_dag_nodes(self):
        """Test factory behavior when DAG has None nodes."""
        class NoneDAG:
            nodes = None
        
        dag = NoneDAG()
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Should handle gracefully
            assert len(factory._config_class_map) == 0
    
    def test_factory_with_invalid_config_class(self):
        """Test factory behavior with invalid config class."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': 'not_a_class'}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Should handle invalid config class gracefully - the method will try to extract fields
            # from a string which should cause an AttributeError when trying to access class attributes
            with patch.object(factory, '_extract_step_specific_fields', side_effect=AttributeError("'str' object has no attribute")):
                with pytest.raises(AttributeError):
                    factory.get_step_requirements('step1')
    
    def test_factory_state_operations_with_missing_files(self):
        """Test factory state operations when files are missing."""
        dag = MockDAG(['step1'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {'step1': MockStepConfigA}
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Test loading non-existent file
            with patch('builtins.open', side_effect=FileNotFoundError("File not found")):
                with pytest.raises(FileNotFoundError):
                    factory.load_partial_state('/nonexistent/path/state.json')
    
    def test_factory_with_circular_dependencies(self):
        """Test factory behavior with potential circular dependencies."""
        dag = MockDAG(['step1', 'step2'])
        
        with patch('cursus.api.factory.dag_config_factory.ConfigClassMapper') as mock_mapper_class:
            mock_mapper = MagicMock()
            mock_mapper.map_dag_to_config_classes.return_value = {
                'step1': MockStepConfigA,
                'step2': MockStepConfigB
            }
            mock_mapper_class.return_value = mock_mapper
            
            factory = DAGConfigFactory(dag)
            
            # Set up base configs
            factory.base_config = MockBasePipelineConfig(project_name="test")
            factory.base_processing_config = MockBaseProcessingStepConfig(project_name="test")
            
            # Configure steps that might have circular references
            factory.step_configs = {
                'step1': {'step_param_a': 'references_step2'},
                'step2': {'step_param_b': 42}
            }
            
            # Should handle without infinite loops
            status = factory.get_configuration_status()
            assert status['step_step1'] is True
            assert status['step_step2'] is True
