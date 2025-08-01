"""
Tests for RegistryManager integration with pipeline builders.
"""

import unittest
from src.pipeline_deps import RegistryManager, integrate_with_pipeline_builder
from src.pipeline_deps.specification_registry import SpecificationRegistry
from src.pipeline_deps.base_specifications import (
    StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType
)
from test.pipeline_deps.test_helpers import IsolatedTestCase, reset_all_global_state


class TestPipelineBuilderIntegration(IsolatedTestCase):
    """Test integration with pipeline builders."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.manager = RegistryManager()
        
        # Create test specifications
        output_spec = OutputSpec(
            logical_name="test_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['test'].S3Output.S3Uri",
            data_type="S3Uri"
        )
        
        self.test_spec = StepSpecification(
            step_type="TestStep",
            node_type=NodeType.SOURCE,
            dependencies=[],
            outputs=[output_spec]
        )
        
        # Create pipeline config class
        class PipelineConfig:
            def __init__(self, pipeline_name=None, environment=None):
                self.pipeline_name = pipeline_name
                self.environment = environment
                
        self.PipelineConfig = PipelineConfig
    
    def test_decorator_with_pipeline_name(self):
        """Test decorator integration with pipeline name in config."""
        test_spec = self.test_spec  # Capture the test_spec from the test class
        
        # Create a pipeline builder class
        @integrate_with_pipeline_builder
        class TestPipelineBuilder:
            def __init__(self, base_config):
                self.base_config = base_config
                self.test_spec = test_spec  # Store the test spec
                
            def build_pipeline(self):
                # Just register a step to verify registry works
                self.registry.register("test_step", self.test_spec)
                return self.registry.list_step_names()
        
        # Create builder with named pipeline
        config = self.PipelineConfig(pipeline_name="test_pipeline")
        builder = TestPipelineBuilder(config)
        
        # Verify registry was created with correct name
        self.assertTrue(hasattr(builder, "registry"))
        self.assertEqual(builder.registry.context_name, "test_pipeline")
        
        # Verify registry is usable
        pipeline_steps = builder.build_pipeline()
        self.assertIn("test_step", pipeline_steps)
    
    def test_decorator_with_no_pipeline_name(self):
        """Test decorator integration when no pipeline name is provided."""
        test_spec = self.test_spec  # Capture the test_spec from the test class
        
        @integrate_with_pipeline_builder
        class TestPipelineBuilder:
            def __init__(self, base_config):
                self.base_config = base_config
                self.test_spec = test_spec  # Store the test spec
                
            def build_pipeline(self):
                self.registry.register("test_step", self.test_spec)
                return self.registry.list_step_names()
        
        # Create builder without pipeline name
        config = self.PipelineConfig()  # No pipeline name
        builder = TestPipelineBuilder(config)
        
        # Verify default context name was used
        self.assertTrue(hasattr(builder, "registry"))
        self.assertEqual(builder.registry.context_name, "default_pipeline")
    
    def test_multiple_pipeline_builders_isolation(self):
        """Test isolation between multiple decorated pipeline builders."""
        @integrate_with_pipeline_builder
        class TrainingPipelineBuilder:
            def __init__(self, base_config):
                self.base_config = base_config
                
            def add_step(self, step_name, spec):
                self.registry.register(step_name, spec)
                
            def list_steps(self):
                return self.registry.list_step_names()
        
        @integrate_with_pipeline_builder
        class InferencePipelineBuilder:
            def __init__(self, base_config):
                self.base_config = base_config
                
            def add_step(self, step_name, spec):
                self.registry.register(step_name, spec)
                
            def list_steps(self):
                return self.registry.list_step_names()
        
        # Create builders for different pipelines
        train_config = self.PipelineConfig(pipeline_name="training")
        infer_config = self.PipelineConfig(pipeline_name="inference")
        
        train_builder = TrainingPipelineBuilder(train_config)
        infer_builder = InferencePipelineBuilder(infer_config)
        
        # Add steps to each builder
        train_builder.add_step("train_step", self.test_spec)
        infer_builder.add_step("inference_step", self.test_spec)
        
        # Verify steps are isolated
        self.assertIn("train_step", train_builder.list_steps())
        self.assertNotIn("inference_step", train_builder.list_steps())
        
        self.assertIn("inference_step", infer_builder.list_steps())
        self.assertNotIn("train_step", infer_builder.list_steps())
    
    def test_environment_based_registries(self):
        """Test environment-specific registries through the decorator."""
        test_spec = self.test_spec  # Capture for the builder class
        
        @integrate_with_pipeline_builder
        class EnvironmentPipelineBuilder:
            def __init__(self, base_config):
                self.base_config = base_config
                self.test_spec = test_spec  # Store the test spec
                
                # The decorator will create self.registry_manager and self.registry
                # But we can't override them here because they don't exist yet
                # We'll do that in setup_environment_registry method
                
            def setup_environment_registry(self):
                # Override the registry to include environment in context name
                env = self.base_config.environment if self.base_config.environment else "default"
                context_name = f"{self.base_config.pipeline_name}_{env}"
                self.registry = self.registry_manager.get_registry(context_name)
                return context_name
                
            def add_env_specific_step(self):
                env = self.base_config.environment
                if env == "dev":
                    self.registry.register("dev_step", self.test_spec)
                elif env == "prod":
                    self.registry.register("prod_step", self.test_spec)
                    
            def list_steps(self):
                return self.registry.list_step_names()
        
        # Create builders for different environments
        dev_config = self.PipelineConfig(pipeline_name="fraud", environment="dev")
        prod_config = self.PipelineConfig(pipeline_name="fraud", environment="prod")
        
        dev_builder = EnvironmentPipelineBuilder(dev_config)
        prod_builder = EnvironmentPipelineBuilder(prod_config)
        
        # Set up environment-specific registries
        dev_context = dev_builder.setup_environment_registry()
        prod_context = prod_builder.setup_environment_registry()
        
        dev_builder.add_env_specific_step()
        prod_builder.add_env_specific_step()
        
        # Verify environment-specific steps
        self.assertIn("dev_step", dev_builder.list_steps())
        self.assertNotIn("dev_step", prod_builder.list_steps())
        
        self.assertIn("prod_step", prod_builder.list_steps())
        self.assertNotIn("prod_step", dev_builder.list_steps())
        
        # Verify context names reflect environments
        self.assertEqual(dev_builder.registry.context_name, "fraud_dev")
        self.assertEqual(prod_builder.registry.context_name, "fraud_prod")


if __name__ == '__main__':
    unittest.main()
