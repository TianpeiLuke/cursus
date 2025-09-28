"""
Integration tests for PIPELINE_EXECUTION_TEMP_DIR functionality.

These tests verify the complete parameter flow from DAGCompiler through
DynamicPipelineTemplate, PipelineAssembler, and StepBuilders to ensure
that PIPELINE_EXECUTION_TEMP_DIR parameters are properly propagated
and used for output path generation.
"""

import pytest
import tempfile
import json
import os
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.functions import Join

from cursus.api.dag.base_dag import PipelineDAG
from cursus.core.compiler.dag_compiler import PipelineDAGCompiler
from cursus.core.compiler.dynamic_template import DynamicPipelineTemplate
from cursus.core.assembler.pipeline_assembler import PipelineAssembler
from cursus.core.base.config_base import BasePipelineConfig
from cursus.core.base.builder_base import StepBuilderBase


class MockConfig(BasePipelineConfig):
    """Mock configuration for integration testing."""
    
    def __init__(self):
        super().__init__(
            author="test_author",
            bucket="test-bucket",
            role="test-role",
            region="NA",
            service_name="test_service",
            pipeline_version="1.0.0",
            project_root_folder="cursus",
        )


class MockStepBuilder(StepBuilderBase):
    """Mock step builder for integration testing."""
    
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
    
    def validate_configuration(self):
        pass
    
    def _get_inputs(self, inputs):
        return inputs
    
    def _get_outputs(self, outputs):
        return outputs
    
    def create_step(self, **kwargs):
        mock_step = Mock()
        mock_step.name = "test_step"
        return mock_step

# Set the __name__ attribute for the class to avoid registry issues
MockStepBuilder.__name__ = "MockStepBuilder"


class TestPipelineExecutionTempDirIntegration:
    """Integration tests for PIPELINE_EXECUTION_TEMP_DIR functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        # Create test DAG
        self.dag = PipelineDAG()
        self.dag.add_node("data_loading")
        self.dag.add_node("training")
        self.dag.add_edge("data_loading", "training")
        
        # Create temporary config file
        self.config_content = {
            "Base": {
                "pipeline_name": "test-pipeline",
                "config_type": "BasePipelineConfig",
                "author": "test_author",
                "bucket": "test-bucket",
                "role": "test-role",
                "region": "NA",
                "service_name": "test_service",
                "pipeline_version": "1.0.0",
            },
            "data_loading": {
                "config_type": "MockConfig",
                "job_type": "training",
            },
            "training": {
                "config_type": "MockConfig",
                "job_type": "training",
            },
        }
        
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.json")
        
        with open(self.config_path, "w") as f:
            json.dump(self.config_content, f)
        
        yield
        
        # Cleanup
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_end_to_end_parameter_flow_with_custom_execution_prefix(self):
        """Test complete parameter flow from DAGCompiler to StepBuilders with custom execution prefix."""
        # Create custom execution parameter
        custom_execution_param = ParameterString(
            name="EXECUTION_S3_PREFIX", 
            default_value="s3://integration-test-bucket/custom-execution"
        )
        
        pipeline_params = [
            custom_execution_param,
            ParameterString(name="KMS_ENCRYPTION_KEY_PARAM", default_value="test-key"),
        ]
        
        with patch("cursus.core.compiler.dag_compiler.Path") as mock_path, \
             patch("cursus.core.compiler.dag_compiler.StepCatalog") as mock_catalog_class, \
             patch("cursus.core.compiler.dag_compiler.StepConfigResolver") as mock_resolver_class, \
             patch("cursus.steps.configs.utils.load_configs") as mock_load_configs, \
             patch("cursus.steps.configs.utils.detect_config_classes_from_json") as mock_detect_classes:
            
            # Setup mocks
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            
            # Mock config loading
            mock_configs = {
                "Base": MockConfig(),
                "data_loading": MockConfig(),
                "training": MockConfig(),
            }
            mock_load_configs.return_value = mock_configs
            mock_detect_classes.return_value = {"MockConfig": MockConfig}
            
            # Mock config resolver
            mock_resolver = MagicMock()
            mock_resolver.resolve_config_map.return_value = {
                "data_loading": mock_configs["data_loading"],
                "training": mock_configs["training"],
            }
            mock_resolver_class.return_value = mock_resolver
            
            # Mock step catalog
            mock_catalog = MagicMock()
            mock_catalog.get_builder_map.return_value = {"MockConfig": MockStepBuilder}
            mock_catalog_class.return_value = mock_catalog
            
            # Create compiler with custom parameters
            compiler = PipelineDAGCompiler(
                config_path=self.config_path,
                pipeline_parameters=pipeline_params,
            )
            
            # Verify compiler stored parameters
            assert compiler.pipeline_parameters == pipeline_params
            
            # Create template with skip_validation to avoid builder registry issues
            template = compiler.create_template(self.dag, skip_validation=True)
            
            # Verify template received parameters
            assert hasattr(template, '_stored_pipeline_parameters')
            assert template._stored_pipeline_parameters == pipeline_params
            
            # Verify template returns custom parameters
            template_params = template._get_pipeline_parameters()
            param_names = [p.name for p in template_params if hasattr(p, 'name')]
            assert "EXECUTION_S3_PREFIX" in param_names
            assert "KMS_ENCRYPTION_KEY_PARAM" in param_names

    def test_end_to_end_parameter_flow_with_assembler_integration(self):
        """Test parameter flow through PipelineAssembler to StepBuilders."""
        custom_execution_param = ParameterString(
            name="EXECUTION_S3_PREFIX", 
            default_value="s3://assembler-test-bucket/execution"
        )
        
        pipeline_params = [custom_execution_param]
        
        # Create config map
        config_map = {
            "data_loading": MockConfig(),
            "training": MockConfig(),
        }
        
        # Create assembler with pipeline parameters
        with patch("cursus.core.assembler.pipeline_assembler.StepCatalog") as mock_catalog:
            mock_catalog_instance = MagicMock()
            mock_catalog_instance.get_builder_for_config.return_value = MockStepBuilder
            mock_catalog_instance.load_builder_class.return_value = MockStepBuilder
            mock_catalog.return_value = mock_catalog_instance
            
            assembler = PipelineAssembler(
                dag=self.dag,
                config_map=config_map,
                pipeline_parameters=pipeline_params,
            )
        
        # Verify parameters were stored
        assert assembler.pipeline_parameters == pipeline_params
        
        # Verify execution prefix was set on all step builders
        for step_name, builder in assembler.step_builders.items():
            assert hasattr(builder, 'execution_prefix')
            assert builder.execution_prefix is not None
            assert builder.execution_prefix.name == "EXECUTION_S3_PREFIX"
            assert builder.execution_prefix.default_value == "s3://assembler-test-bucket/execution"

    def test_step_builder_base_output_path_resolution_with_execution_prefix(self):
        """Test that StepBuilderBase correctly resolves output paths with execution prefix."""
        config = MockConfig()
        builder = MockStepBuilder(config=config)
        
        # Test without execution prefix - should use config
        base_path = builder._get_base_output_path()
        assert base_path == config.pipeline_s3_loc
        
        # Test with execution prefix - should use parameter
        execution_param = ParameterString(
            name="EXECUTION_S3_PREFIX", 
            default_value="s3://param-test-bucket/execution"
        )
        builder.set_execution_prefix(execution_param)
        
        base_path = builder._get_base_output_path()
        assert base_path == execution_param
        assert base_path.name == "EXECUTION_S3_PREFIX"
        assert base_path.default_value == "s3://param-test-bucket/execution"

    def test_output_generation_with_join_pattern(self):
        """Test that output generation uses Join pattern for parameter compatibility."""
        custom_execution_param = ParameterString(
            name="EXECUTION_S3_PREFIX", 
            default_value="s3://join-test-bucket/execution"
        )
        
        config_map = {
            "data_loading": MockConfig(),
            "training": MockConfig(),
        }
        
        # Create assembler with pipeline parameters
        with patch("cursus.core.assembler.pipeline_assembler.StepCatalog") as mock_catalog:
            mock_catalog_instance = MagicMock()
            mock_catalog_instance.get_builder_for_config.return_value = MockStepBuilder
            mock_catalog_instance.load_builder_class.return_value = MockStepBuilder
            mock_catalog.return_value = mock_catalog_instance
            
            assembler = PipelineAssembler(
                dag=self.dag,
                config_map=config_map,
                pipeline_parameters=[custom_execution_param],
            )
        
        # Generate outputs for a step
        outputs = assembler._generate_outputs("data_loading")
        
        # Verify outputs use Join objects (not f-strings)
        for output_name, output_path in outputs.items():
            assert isinstance(output_path, Join), f"Output {output_name} should use Join, got {type(output_path)}"

    def test_backward_compatibility_without_parameters(self):
        """Test that existing behavior is preserved when no parameters provided."""
        config_map = {
            "data_loading": MockConfig(),
            "training": MockConfig(),
        }
        
        # Create assembler without pipeline parameters
        with patch("cursus.core.assembler.pipeline_assembler.StepCatalog") as mock_catalog:
            mock_catalog_instance = MagicMock()
            mock_catalog_instance.get_builder_for_config.return_value = MockStepBuilder
            mock_catalog_instance.load_builder_class.return_value = MockStepBuilder
            mock_catalog.return_value = mock_catalog_instance
            
            assembler = PipelineAssembler(
                dag=self.dag,
                config_map=config_map,
            )
        
        # Verify no execution prefix was set on step builders
        for step_name, builder in assembler.step_builders.items():
            assert hasattr(builder, 'execution_prefix')
            assert builder.execution_prefix is None
            
            # Verify fallback to config.pipeline_s3_loc
            base_path = builder._get_base_output_path()
            assert base_path == builder.config.pipeline_s3_loc

    def test_parameter_type_compatibility(self):
        """Test that both ParameterString and string types work correctly."""
        config = MockConfig()
        builder = MockStepBuilder(config=config)
        
        # Test with ParameterString
        param_string = ParameterString(
            name="EXECUTION_S3_PREFIX", 
            default_value="s3://param-string-test/execution"
        )
        builder.set_execution_prefix(param_string)
        assert builder._get_base_output_path() == param_string
        
        # Test with regular string
        string_path = "s3://string-test-bucket/execution"
        builder.set_execution_prefix(string_path)
        assert builder._get_base_output_path() == string_path
        
        # Test with None (fallback)
        builder.set_execution_prefix(None)
        assert builder._get_base_output_path() == config.pipeline_s3_loc

    def test_multiple_parameter_handling(self):
        """Test handling of multiple pipeline parameters."""
        execution_param = ParameterString(
            name="EXECUTION_S3_PREFIX", 
            default_value="s3://multi-param-test/execution"
        )
        kms_param = ParameterString(
            name="KMS_ENCRYPTION_KEY_PARAM", 
            default_value="test-kms-key"
        )
        security_param = ParameterString(
            name="SECURITY_GROUP_ID", 
            default_value="sg-12345"
        )
        
        pipeline_params = [execution_param, kms_param, security_param]
        
        config_map = {
            "data_loading": MockConfig(),
            "training": MockConfig(),
        }
        
        # Create assembler with pipeline parameters
        with patch("cursus.core.assembler.pipeline_assembler.StepCatalog") as mock_catalog:
            mock_catalog_instance = MagicMock()
            mock_catalog_instance.get_builder_for_config.return_value = MockStepBuilder
            mock_catalog_instance.load_builder_class.return_value = MockStepBuilder
            mock_catalog.return_value = mock_catalog_instance
            
            assembler = PipelineAssembler(
                dag=self.dag,
                config_map=config_map,
                pipeline_parameters=pipeline_params,
            )
        
        # Verify all parameters were stored
        assert len(assembler.pipeline_parameters) == 3
        param_names = [p.name for p in assembler.pipeline_parameters]
        assert "EXECUTION_S3_PREFIX" in param_names
        assert "KMS_ENCRYPTION_KEY_PARAM" in param_names
        assert "SECURITY_GROUP_ID" in param_names
        
        # Verify only EXECUTION_S3_PREFIX was passed to step builders
        for step_name, builder in assembler.step_builders.items():
            assert builder.execution_prefix == execution_param

    def test_parameter_extraction_logic(self):
        """Test the parameter extraction logic in PipelineAssembler."""
        # Test with EXECUTION_S3_PREFIX parameter
        execution_param = ParameterString(name="EXECUTION_S3_PREFIX", default_value="s3://test/execution")
        other_param = ParameterString(name="OTHER_PARAM", default_value="other-value")
        
        config_map = {"data_loading": MockConfig()}
        
        with patch("cursus.core.assembler.pipeline_assembler.StepCatalog") as mock_catalog:
            mock_catalog_instance = MagicMock()
            mock_catalog_instance.get_builder_for_config.return_value = MockStepBuilder
            mock_catalog_instance.load_builder_class.return_value = MockStepBuilder
            mock_catalog.return_value = mock_catalog_instance
            
            assembler = PipelineAssembler(
                dag=PipelineDAG(nodes=["data_loading"], edges=[]),
                config_map=config_map,
                pipeline_parameters=[execution_param, other_param],
            )
        
        # Should extract EXECUTION_S3_PREFIX and set it on builders
        builder = assembler.step_builders["data_loading"]
        assert builder.execution_prefix == execution_param
        
        # Test without EXECUTION_S3_PREFIX parameter
        with patch("cursus.core.assembler.pipeline_assembler.StepCatalog") as mock_catalog2:
            mock_catalog_instance2 = MagicMock()
            mock_catalog_instance2.get_builder_for_config.return_value = MockStepBuilder
            mock_catalog_instance2.load_builder_class.return_value = MockStepBuilder
            mock_catalog2.return_value = mock_catalog_instance2
            
            assembler2 = PipelineAssembler(
                dag=PipelineDAG(nodes=["data_loading"], edges=[]),
                config_map=config_map,
                pipeline_parameters=[other_param],  # No EXECUTION_S3_PREFIX
            )
        
        # Should not set execution prefix
        builder2 = assembler2.step_builders["data_loading"]
        assert builder2.execution_prefix is None

    def test_error_handling_with_invalid_parameters(self):
        """Test error handling with invalid parameter configurations."""
        config_map = {"data_loading": MockConfig()}
        
        # Test with empty parameter list - should work fine
        with patch("cursus.core.assembler.pipeline_assembler.StepCatalog") as mock_catalog:
            mock_catalog_instance = MagicMock()
            mock_catalog_instance.get_builder_for_config.return_value = MockStepBuilder
            mock_catalog_instance.load_builder_class.return_value = MockStepBuilder
            mock_catalog.return_value = mock_catalog_instance
            
            assembler = PipelineAssembler(
                dag=PipelineDAG(nodes=["data_loading"], edges=[]),
                config_map=config_map,
                pipeline_parameters=[],
            )
        
        builder = assembler.step_builders["data_loading"]
        assert builder.execution_prefix is None
        
        # Test with None parameter list - should work fine
        with patch("cursus.core.assembler.pipeline_assembler.StepCatalog") as mock_catalog2:
            mock_catalog_instance2 = MagicMock()
            mock_catalog_instance2.get_builder_for_config.return_value = MockStepBuilder
            mock_catalog_instance2.load_builder_class.return_value = MockStepBuilder
            mock_catalog2.return_value = mock_catalog_instance2
            
            assembler2 = PipelineAssembler(
                dag=PipelineDAG(nodes=["data_loading"], edges=[]),
                config_map=config_map,
                pipeline_parameters=None,
            )
        
        builder2 = assembler2.step_builders["data_loading"]
        assert builder2.execution_prefix is None

    def test_logging_integration(self):
        """Test that parameter operations are properly logged."""
        with patch("cursus.core.base.builder_base.logger") as mock_logger:
            config = MockConfig()
            builder = MockStepBuilder(config=config)
            
            execution_param = ParameterString(
                name="EXECUTION_S3_PREFIX", 
                default_value="s3://logging-test/execution"
            )
            
            # Set execution prefix - should log
            builder.set_execution_prefix(execution_param)
            mock_logger.debug.assert_called()
            
            # Get base output path - should log
            builder._get_base_output_path()
            mock_logger.info.assert_called()
