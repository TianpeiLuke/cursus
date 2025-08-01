"""
Test demonstrating how to initialize child config objects from parent config objects.

This test shows how to use the get_public_init_fields and from_base_config methods
to initialize a ProcessingStepConfigBase from a BasePipelineConfig without duplicating
field values.
"""

import unittest
from src.pipeline_steps.config_base import BasePipelineConfig
from src.pipeline_steps.config_processing_step_base import ProcessingStepConfigBase


class TestConfigInheritance(unittest.TestCase):
    def test_field_categorization(self):
        """Test that fields are properly categorized based on whether they're required."""
        # Create a base config with essential fields
        base_config = BasePipelineConfig(
            author="test-user",
            bucket="test-bucket",
            role="test-role",
            region="NA",
            service_name="TestService",
            pipeline_version="1.0.0",
        )
        
        # Get field categories
        categories = base_config.categorize_fields()
        
        # Verify essential fields are correctly categorized
        essential_fields = sorted(categories['essential'])
        self.assertIn("author", essential_fields)
        self.assertIn("bucket", essential_fields)
        self.assertIn("role", essential_fields)
        self.assertIn("region", essential_fields)
        self.assertIn("service_name", essential_fields)
        self.assertIn("pipeline_version", essential_fields)
        
        # Verify system fields are correctly categorized
        system_fields = sorted(categories['system'])
        self.assertIn("model_class", system_fields)
        self.assertIn("current_date", system_fields)
        self.assertIn("framework_version", system_fields)
        self.assertIn("py_version", system_fields)
        self.assertIn("source_dir", system_fields)
        
        # Verify derived fields are correctly categorized
        derived_fields = sorted(categories['derived'])
        self.assertIn("aws_region", derived_fields)
        self.assertIn("pipeline_name", derived_fields)
        self.assertIn("pipeline_description", derived_fields)
        self.assertIn("pipeline_s3_loc", derived_fields)
        
        # Test the string representation to ensure fields appear in the right categories
        str_repr = str(base_config)
        
        # Check that essential fields appear under Essential User Inputs
        self.assertIn("- Essential User Inputs -", str_repr)
        self.assertIn("author: test-user", str_repr)
        self.assertIn("bucket: test-bucket", str_repr)
        self.assertIn("role: test-role", str_repr)
        
        # Check that system fields appear under System Inputs
        self.assertIn("- System Inputs -", str_repr)
        self.assertIn("model_class: xgboost", str_repr)
        
        # Check that derived fields appear under Derived Fields
        self.assertIn("- Derived Fields -", str_repr)
        self.assertIn("pipeline_name:", str_repr)
        
    def test_config_inheritance(self):
        # Create a base pipeline config with essential fields
        base_config = BasePipelineConfig(
            author="lukexie",
            bucket="sagemaker-example-bucket",
            role="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
            region="NA",
            service_name="AtoZ",
            pipeline_version="0.1.0",
        )
        
        # Create a processing config from the base config, without duplicating fields
        processing_config = ProcessingStepConfigBase.from_base_config(
            base_config,
            # Add processing-specific fields
            processing_instance_count=2,
            processing_volume_size=100,
            use_large_processing_instance=True,
        )
        
        # Verify that parent fields are correctly inherited
        self.assertEqual(processing_config.author, base_config.author)
        self.assertEqual(processing_config.bucket, base_config.bucket)
        self.assertEqual(processing_config.role, base_config.role)
        self.assertEqual(processing_config.region, base_config.region)
        self.assertEqual(processing_config.service_name, base_config.service_name)
        self.assertEqual(processing_config.pipeline_version, base_config.pipeline_version)
        self.assertEqual(processing_config.model_class, base_config.model_class)
        
        # Verify that child-specific fields are set correctly
        self.assertEqual(processing_config.processing_instance_count, 2)
        self.assertEqual(processing_config.processing_volume_size, 100)
        self.assertTrue(processing_config.use_large_processing_instance)
        
        # Verify that derived fields are initialized correctly
        self.assertEqual(processing_config.aws_region, "us-east-1")  # Default for 'NA'
        self.assertEqual(processing_config.pipeline_name, f"{base_config.author}-{base_config.service_name}-{base_config.model_class}-{base_config.region}")
        
        # Test that child-specific derived fields are initialized
        self.assertEqual(processing_config.effective_instance_type, processing_config.processing_instance_type_large)
        
    def test_field_override(self):
        # Create a base pipeline config
        base_config = BasePipelineConfig(
            author="lukexie",
            bucket="sagemaker-example-bucket",
            role="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
            region="EU",  # Set region to EU
            service_name="AtoZ",
            pipeline_version="0.1.0",
        )
        
        # Create a processing config overriding a parent field
        processing_config = ProcessingStepConfigBase.from_base_config(
            base_config,
            region="FE",  # Override the region from EU to FE
        )
        
        # Verify the field was overridden
        self.assertEqual(processing_config.region, "FE")
        self.assertEqual(processing_config.aws_region, "us-west-2")  # FE region maps to us-west-2


if __name__ == "__main__":
    unittest.main()
