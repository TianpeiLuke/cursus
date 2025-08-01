"""
Tests for various context usage patterns in RegistryManager.
"""

import unittest
from src.pipeline_deps import RegistryManager
from src.pipeline_deps.specification_registry import SpecificationRegistry
from src.pipeline_deps.base_specifications import (
    StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType
)
from test.pipeline_deps.test_helpers import IsolatedTestCase, reset_all_global_state


class TestMultiContextPatterns(IsolatedTestCase):
    """Test various context usage patterns."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.manager = RegistryManager()
        
        # Create test specification
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
    
    def test_environment_based_contexts(self):
        """Test environment-based context pattern."""
        # Create environment-specific registries
        dev_registry = self.manager.get_registry("development")
        staging_registry = self.manager.get_registry("staging")
        prod_registry = self.manager.get_registry("production")
        
        # Register environment-specific specs
        dev_spec = StepSpecification(
            step_type="FastDevStep",
            node_type=NodeType.SOURCE,
            dependencies=[],
            outputs=[OutputSpec(
                logical_name="dev_output",
                output_type=DependencyType.PROCESSING_OUTPUT,
                property_path="properties.Output",
                data_type="S3Uri"
            )]
        )
        
        prod_spec = StepSpecification(
            step_type="RobustProdStep",
            node_type=NodeType.SOURCE,
            dependencies=[],
            outputs=[OutputSpec(
                logical_name="prod_output",
                output_type=DependencyType.PROCESSING_OUTPUT,
                property_path="properties.Output",
                data_type="S3Uri"
            )]
        )
        
        dev_registry.register("data_loading", dev_spec)
        prod_registry.register("data_loading", prod_spec)
        staging_registry.register("data_loading", prod_spec)  # Staging uses prod spec
        
        # Verify environment-specific specs
        dev_step = dev_registry.get_specification("data_loading")
        prod_step = prod_registry.get_specification("data_loading")
        staging_step = staging_registry.get_specification("data_loading")
        
        self.assertEqual(dev_step.step_type, "FastDevStep")
        self.assertEqual(prod_step.step_type, "RobustProdStep")
        self.assertEqual(staging_step.step_type, "RobustProdStep")  # Same as prod
    
    def test_pipeline_type_contexts(self):
        """Test pipeline-type context pattern."""
        # Create pipeline-type registries
        training_registry = self.manager.get_registry("training_pipeline")
        inference_registry = self.manager.get_registry("inference_pipeline")
        batch_registry = self.manager.get_registry("batch_processing")
        
        # Register pipeline-type specs
        training_spec = StepSpecification(
            step_type="TrainingStep",
            node_type=NodeType.SOURCE,
            dependencies=[],
            outputs=[OutputSpec(
                logical_name="model_artifact",
                output_type=DependencyType.MODEL_ARTIFACTS,
                property_path="properties.ModelArtifacts.S3ModelArtifacts",
                data_type="S3Uri"
            )]
        )
        
        inference_spec = StepSpecification(
            step_type="InferenceStep",
            node_type=NodeType.SINK,
            dependencies=[DependencySpec(
                logical_name="model",
                dependency_type=DependencyType.MODEL_ARTIFACTS,
                required=True
            )],
            outputs=[]
        )
        
        batch_spec = StepSpecification(
            step_type="BatchProcessingStep",
            node_type=NodeType.SINK,
            dependencies=[DependencySpec(
                logical_name="model",
                dependency_type=DependencyType.MODEL_ARTIFACTS,
                required=True
            )],
            outputs=[]
        )
        
        training_registry.register("model_training", training_spec)
        inference_registry.register("model_inference", inference_spec)
        batch_registry.register("batch_transform", batch_spec)
        
        # Verify pipeline-specific specs
        self.assertEqual(training_registry.get_specification("model_training").step_type, "TrainingStep")
        self.assertEqual(inference_registry.get_specification("model_inference").step_type, "InferenceStep")
        self.assertEqual(batch_registry.get_specification("batch_transform").step_type, "BatchProcessingStep")
    
    def test_multitenant_contexts(self):
        """Test multi-tenant context pattern."""
        # Create tenant-specific registries
        tenant_a_registry = self.manager.get_registry("customer_a")
        tenant_b_registry = self.manager.get_registry("customer_b")
        
        # Register tenant-specific specs with customizations
        # Using step_type to store tenant-specific info instead of custom_parameters
        tenant_a_spec = StepSpecification(
            step_type="CustomizedProcessingStep_TenantA",  # Embed tenant info in step_type
            node_type=NodeType.SOURCE,
            dependencies=[],
            outputs=[OutputSpec(
                logical_name="tenant_a_output",
                output_type=DependencyType.PROCESSING_OUTPUT,
                property_path="properties.Output",
                data_type="S3Uri"
            )]
        )
        
        # Using step_type to store tenant-specific info
        tenant_b_spec = StepSpecification(
            step_type="CustomizedProcessingStep_TenantB",  # Embed tenant info in step_type
            node_type=NodeType.SOURCE,
            dependencies=[],
            outputs=[OutputSpec(
                logical_name="tenant_b_output",
                output_type=DependencyType.PROCESSING_OUTPUT,
                property_path="properties.Output",
                data_type="S3Uri"
            )]
        )
        
        tenant_a_registry.register("processing", tenant_a_spec)
        tenant_b_registry.register("processing", tenant_b_spec)
        
        # Verify tenant-specific customizations in step_type
        tenant_a_step_type = tenant_a_registry.get_specification("processing").step_type
        tenant_b_step_type = tenant_b_registry.get_specification("processing").step_type
        
        self.assertEqual(tenant_a_step_type, "CustomizedProcessingStep_TenantA")
        self.assertEqual(tenant_b_step_type, "CustomizedProcessingStep_TenantB")


if __name__ == '__main__':
    unittest.main()
