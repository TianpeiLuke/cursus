"""
Test suite for dag_config_factory.py

Behavior-focused tests that verify the public API of DAGConfigFactory
without testing internal implementation details.
"""

import pytest
from typing import Dict, List
from pydantic import BaseModel, Field
from unittest.mock import patch, MagicMock

from cursus.api.factory.dag_config_factory import (
    DAGConfigFactory,
    ConfigurationIncompleteError,
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


class MockStepConfigA(MockBasePipelineConfig):
    """Mock step configuration A."""
    step_param_a: str = Field(description="Step parameter A")
    
    @classmethod
    def from_base_config(cls, base_config, **kwargs):
        base_values = base_config.model_dump()
        base_values.update(kwargs)
        return cls(**base_values)


class MockStepConfigB(MockBasePipelineConfig):
    """Mock step configuration B."""
    step_param_b: int = Field(description="Step parameter B")
    
    @classmethod
    def from_base_config(cls, base_config, **kwargs):
        base_values = base_config.model_dump()
        base_values.update(kwargs)
        return cls(**base_values)


class TestDAGConfigFactoryInit:
    """Test DAGConfigFactory initialization."""
    
    def test_init_with_dag(self):
        """Test initialization with DAG."""
        dag = MockDAG(["step1", "step2"])
        
        with patch.object(
            DAGConfigFactory,
            "_map_dag_to_config_classes_robust",
            return_value={"step1": MockStepConfigA, "step2": MockStepConfigB}
        ):
            factory = DAGConfigFactory(dag)
            
            assert factory.dag == dag
            assert factory.config_generator is None
            assert len(factory._config_class_map) == 2
            assert factory.base_config is None
            assert factory.step_configs == {}


class TestDAGConfigFactoryConfigClassMapping:
    """Test config class mapping functionality."""
    
    def test_get_config_class_map(self):
        """Test getting config class map returns a copy."""
        dag = MockDAG(["step1"])
        config_map = {"step1": MockStepConfigA}
        
        with patch.object(
            DAGConfigFactory,
            "_map_dag_to_config_classes_robust",
            return_value=config_map
        ):
            factory = DAGConfigFactory(dag)
            result = factory.get_config_class_map()
            
            # Should return a copy
            assert result == config_map
            result["step2"] = MockStepConfigB
            assert "step2" not in factory._config_class_map


class TestDAGConfigFactoryBaseConfig:
    """Test base configuration management."""
    
    def test_set_base_config_success(self):
        """Test setting base config successfully."""
        dag = MockDAG(["step1"])
        
        with patch.object(
            DAGConfigFactory,
            "_map_dag_to_config_classes_robust",
            return_value={"step1": MockStepConfigA}
        ):
            factory = DAGConfigFactory(dag)
            
            with patch("cursus.core.base.config_base.BasePipelineConfig", MockBasePipelineConfig):
                with patch("cursus.api.factory.dag_config_factory.ConfigurationGenerator") as mock_gen:
                    factory.set_base_config(project_name="test_project", version="2.0.0")
                    
                    assert factory.base_config is not None
                    assert factory.base_config.project_name == "test_project"
                    assert factory.base_config.version == "2.0.0"
                    assert factory.config_generator is not None


class TestDAGConfigFactoryStepConfiguration:
    """Test step configuration functionality."""
    
    def test_set_step_config_success(self):
        """Test setting step config successfully."""
        dag = MockDAG(["step1"])
        
        with patch.object(
            DAGConfigFactory,
            "_map_dag_to_config_classes_robust",
            return_value={"step1": MockStepConfigA}
        ):
            factory = DAGConfigFactory(dag)
            factory.base_config = MockBasePipelineConfig(project_name="test")
            factory.config_generator = MagicMock()
            
            with patch.object(factory, "_validate_prerequisites_for_step"):
                with patch.object(factory, "_create_config_instance_with_inheritance") as mock_create:
                    mock_config = MockStepConfigA(project_name="test", step_param_a="value")
                    mock_create.return_value = mock_config
                    
                    result = factory.set_step_config("step1", step_param_a="value")
                    
                    assert result == mock_config
                    assert "step1" in factory.step_configs
                    assert "step1" in factory.step_config_instances
    
    def test_set_step_config_step_not_found(self):
        """Test setting config for non-existent step."""
        dag = MockDAG(["step1"])
        
        with patch.object(
            DAGConfigFactory,
            "_map_dag_to_config_classes_robust",
            return_value={"step1": MockStepConfigA}
        ):
            factory = DAGConfigFactory(dag)
            
            with pytest.raises(ValueError, match="Step 'nonexistent' not found"):
                factory.set_step_config("nonexistent", param="value")
    
    def test_get_step_config_instance(self):
        """Test getting step config instance."""
        dag = MockDAG(["step1"])
        
        with patch.object(
            DAGConfigFactory,
            "_map_dag_to_config_classes_robust",
            return_value={"step1": MockStepConfigA}
        ):
            factory = DAGConfigFactory(dag)
            mock_config = MockStepConfigA(project_name="test", step_param_a="value")
            factory.step_config_instances["step1"] = mock_config
            
            result = factory.get_step_config_instance("step1")
            assert result == mock_config
    
    def test_get_all_config_instances(self):
        """Test getting all config instances."""
        dag = MockDAG(["step1", "step2"])
        
        with patch.object(
            DAGConfigFactory,
            "_map_dag_to_config_classes_robust",
            return_value={"step1": MockStepConfigA, "step2": MockStepConfigB}
        ):
            factory = DAGConfigFactory(dag)
            config1 = MockStepConfigA(project_name="test", step_param_a="value1")
            config2 = MockStepConfigB(project_name="test", step_param_b=42)
            factory.step_config_instances = {"step1": config1, "step2": config2}
            
            result = factory.get_all_config_instances()
            
            assert len(result) == 2
            assert result["step1"] == config1
            assert result["step2"] == config2


class TestDAGConfigFactoryConfigGeneration:
    """Test configuration generation."""
    
    def test_generate_all_configs_with_pre_validated_instances(self):
        """Test generating configs when pre-validated instances exist."""
        dag = MockDAG(["step1", "step2"])
        
        with patch.object(
            DAGConfigFactory,
            "_map_dag_to_config_classes_robust",
            return_value={"step1": MockStepConfigA, "step2": MockStepConfigB}
        ):
            factory = DAGConfigFactory(dag)
            
            config1 = MockStepConfigA(project_name="test", step_param_a="value1")
            config2 = MockStepConfigB(project_name="test", step_param_b=42)
            factory.step_config_instances = {"step1": config1, "step2": config2}
            
            with patch.object(factory, "_auto_configure_eligible_steps", return_value=0):
                with patch.object(factory, "get_pending_steps", return_value=[]):
                    configs = factory.generate_all_configs()
                    
                    assert len(configs) == 2
                    assert config1 in configs
                    assert config2 in configs
    
    def test_generate_all_configs_missing_steps(self):
        """Test generating configs with missing steps."""
        dag = MockDAG(["step1", "step2"])
        
        with patch.object(
            DAGConfigFactory,
            "_map_dag_to_config_classes_robust",
            return_value={"step1": MockStepConfigA, "step2": MockStepConfigB}
        ):
            factory = DAGConfigFactory(dag)
            
            with patch.object(factory, "_auto_configure_eligible_steps", return_value=0):
                with patch.object(factory, "get_pending_steps", return_value=["step2"]):
                    with pytest.raises(ValueError, match="Missing configuration for steps"):
                        factory.generate_all_configs()


class TestDAGConfigFactoryStatus:
    """Test factory status and summary methods."""
    
    def test_get_configuration_status(self):
        """Test getting configuration status."""
        dag = MockDAG(["step1", "step2"])
        
        with patch.object(
            DAGConfigFactory,
            "_map_dag_to_config_classes_robust",
            return_value={"step1": MockStepConfigA, "step2": MockStepConfigB}
        ):
            factory = DAGConfigFactory(dag)
            factory.base_config = MockBasePipelineConfig(project_name="test")
            factory.step_configs["step1"] = {"step_param_a": "configured"}
            
            with patch.object(factory, "get_base_processing_config_requirements", return_value=[]):
                status = factory.get_configuration_status()
                
                assert status["base_config"] is True
                assert status["base_processing_config"] is True
                assert status["step_step1"] is True
                assert status["step_step2"] is False
    
    def test_get_factory_summary(self):
        """Test getting factory summary."""
        dag = MockDAG(["step1", "step2"])
        
        with patch.object(
            DAGConfigFactory,
            "_map_dag_to_config_classes_robust",
            return_value={"step1": MockStepConfigA, "step2": MockStepConfigB}
        ):
            factory = DAGConfigFactory(dag)
            factory.base_config = MockBasePipelineConfig(project_name="test")
            factory.step_configs["step1"] = {"step_param_a": "configured"}
            
            with patch.object(factory, "get_configuration_status") as mock_status:
                mock_status.return_value = {
                    "base_config": True,
                    "base_processing_config": True,
                    "step_step1": True,
                    "step_step2": False,
                }
                with patch.object(factory, "get_pending_steps", return_value=["step2"]):
                    summary = factory.get_factory_summary()
                    
                    assert summary["dag_steps"] == 2
                    assert summary["completed_steps"] == 1
                    assert summary["pending_steps"] == ["step2"]
                    assert summary["base_config_set"] is True


class TestConfigurationIncompleteError:
    """Test ConfigurationIncompleteError exception."""
    
    def test_error_creation(self):
        """Test creating ConfigurationIncompleteError."""
        error = ConfigurationIncompleteError("Test message")
        assert isinstance(error, Exception)
        assert str(error) == "Test message"
    
    def test_error_raising(self):
        """Test raising ConfigurationIncompleteError."""
        with pytest.raises(ConfigurationIncompleteError, match="Config incomplete"):
            raise ConfigurationIncompleteError("Config incomplete")


class TestDAGConfigFactoryEdgeCases:
    """Test edge cases and error handling."""
    
    def test_factory_with_empty_dag(self):
        """Test factory with empty DAG."""
        dag = MockDAG([])
        
        with patch.object(
            DAGConfigFactory,
            "_map_dag_to_config_classes_robust",
            return_value={}
        ):
            factory = DAGConfigFactory(dag)
            
            assert len(factory._config_class_map) == 0
            assert factory.get_pending_steps() == []
            
            with patch.object(factory, "_auto_configure_eligible_steps", return_value=0):
                configs = factory.generate_all_configs()
                assert configs == []
