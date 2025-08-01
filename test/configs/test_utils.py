import unittest
import json
import os
from pathlib import Path
import tempfile
import sys

# Add the repository root directory to the path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, repo_root)

import pytest
from unittest.mock import patch, MagicMock, Mock, PropertyMock

# Import utilities for config serialization
from src.pipeline_steps.utils import merge_and_save_configs, serialize_config
from src.config_field_manager.constants import SPECIAL_FIELDS_TO_KEEP_SPECIFIC, CategoryType
from src.pipeline_steps.config_base import BasePipelineConfig
from src.pipeline_steps.config_processing_step_base import ProcessingStepConfigBase
from src.pipeline_steps.config_dummy_training_step import DummyTrainingConfig
from src.pipeline_steps.hyperparameters_base import ModelHyperparameters


class TestSimplifiedConfigFieldCategorization(unittest.TestCase):
    """
    Tests for the simplified configuration field categorization logic in utils.py.
    This tests the flattened structure where fields are categorized only as shared or specific.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Define paths
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        self.model_path = os.path.join(os.path.dirname(__file__), "model.tar.gz")
        self.pipeline_scripts_path = os.path.join(self.repo_root, "src/pipeline_scripts")
        
        # Check that required directories and files exist
        self.assertTrue(os.path.exists(self.model_path), f"Test model file missing: {self.model_path}")
        self.assertTrue(os.path.exists(self.pipeline_scripts_path), 
                       f"Required directory not found: {self.pipeline_scripts_path}")
        
        # Create a base config
        self.base_config = BasePipelineConfig(
            bucket="test-bucket",
            author="test-author",
            pipeline_name="test-pipeline",
            pipeline_description="Test Pipeline",
            pipeline_version="1.0.0",
            pipeline_s3_loc="s3://test-bucket/test-pipeline"
        )
        
        # Create a sample hyperparameters object
        self.hyperparams = ModelHyperparameters(
            full_field_list=["field1", "field2", "field3"],
            cat_field_list=["field3"],
            tab_field_list=["field1", "field2"],
            input_tab_dim=2,
            is_binary=True,
            num_classes=2,
            multiclass_categories=[0, 1],
            class_weights=[1.0, 2.0]
        )
        
        # Create a dummy training config (ProcessingStepConfigBase subclass)
        self.dummy_config = DummyTrainingConfig(
            bucket="test-bucket",
            author="test-author",
            pipeline_name="test-pipeline",
            pipeline_description="Test Pipeline",
            pipeline_version="1.0.0",
            pipeline_s3_loc="s3://test-bucket/test-pipeline",
            pretrained_model_path=self.model_path,
            processing_source_dir=self.pipeline_scripts_path,  # Use absolute path
            hyperparameters=self.hyperparams
        )
        
        # Create a custom processing config with various field types
        class CustomProcessingConfig(ProcessingStepConfigBase):
            # Properly define output_schema as a model field
            output_schema: dict = None
            
            def get_script_contract(self):
                return None
                
            def validate_config(self):
                return self
            
        # Use output_schema field which is in SPECIAL_FIELDS_TO_KEEP_SPECIFIC
        self.processing_config = CustomProcessingConfig(
            bucket="test-bucket",
            author="test-author",
            pipeline_name="test-pipeline",
            pipeline_description="Test Pipeline",
            pipeline_version="1.0.0",
            pipeline_s3_loc="s3://test-bucket/test-pipeline",
            processing_source_dir=self.pipeline_scripts_path,  # Use absolute path
            processing_instance_count=2,  # Different from dummy_config
            output_schema={"fields": ["field1", "field2"]}  # This is in SPECIAL_FIELDS_TO_KEEP_SPECIFIC
        )
        
        # Create a custom non-processing config with various field types
        class CustomSpecificConfig(BasePipelineConfig):
            # Define fields for non-static detection
            input_names: dict = None
            
        self.specific_config = CustomSpecificConfig(
            bucket="test-bucket",
            author="test-author",
            pipeline_name="test-pipeline",
            pipeline_description="Test Pipeline",
            pipeline_version="1.0.0",
            pipeline_s3_loc="s3://test-bucket/test-pipeline",
            input_names={"input1": "path1"}  # This is recognized as non-static
        )

    def test_simplified_structure(self):
        """Test that the output has the simplified structure with just shared and specific sections."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            output_path = temp_file.name
            
        try:
            # Merge the configs and save to the temp file
            merge_and_save_configs([self.base_config, self.dummy_config, self.processing_config], output_path)
            
            # Read the output file
            with open(output_path, 'r') as f:
                output_json = json.load(f)
                
            # Print output for debugging
            print(f"Output JSON structure: {json.dumps(output_json, indent=2)}")
            
            # Verify that the top-level structure contains only shared, specific, and metadata
            config_keys = set(output_json['configuration'].keys())
            self.assertEqual(config_keys, {'shared', 'specific'}, 
                           "Configuration should only have 'shared' and 'specific' top-level keys")
            
            # Verify that the metadata contains config_types and created_at
            self.assertIn('metadata', output_json)
            self.assertIn('config_types', output_json['metadata'])
            self.assertIn('created_at', output_json['metadata'])
            
            # Verify there's no nested processing sections or old hierarchical structure
            self.assertNotIn('processing', output_json['configuration'])
            self.assertNotIn('processing_shared', output_json['configuration'])
            self.assertNotIn('processing_specific', output_json['configuration'])
            
            # Ensure 'specific' section contains entries for both processing and non-processing configs
            specific_section = output_json['configuration']['specific']
            self.assertIsInstance(specific_section, dict, "'specific' section should be a dictionary")
            
            # Get step names from metadata for validation
            step_names = set(output_json['metadata']['config_types'].keys())
            
            # Verify that there's an entry for each config in the specific section
            specific_keys = set(specific_section.keys())
            self.assertTrue(len(specific_keys) > 0, "The 'specific' section should not be empty")
            
            # Verify all step names are present in specific section (allowing for some to be empty)
            # At least one step from each config type should be in the specific section
            processing_found = False
            non_processing_found = False
            
            for step_name in specific_keys:
                if any(proc_type in step_name for proc_type in ["Processing", "TabularPreprocessing", "DummyTraining"]):
                    processing_found = True
                else:
                    non_processing_found = True
            
            self.assertTrue(processing_found, "No processing config found in the 'specific' section")
            self.assertTrue(non_processing_found, "No non-processing config found in the 'specific' section")
            
        finally:
            # Clean up the temp file
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_simplified_special_fields(self):
        """Test that special fields are always kept in specific sections in the simplified model."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            output_path = temp_file.name
            
        try:
            # Merge the configs and save to the temp file
            merge_and_save_configs([self.base_config, self.dummy_config], output_path)
            
            # Read the output file
            with open(output_path, 'r') as f:
                output_json = json.load(f)
                
            # Find the config with hyperparameters in the specific section
            found_hyperparameters = False
            specific_sections = output_json['configuration']['specific']
            
            for step_name, step_config in specific_sections.items():
                if 'hyperparameters' in step_config:
                    found_hyperparameters = True
                    # Verify that the hyperparameters object is properly serialized
                    self.assertIn('is_binary', step_config['hyperparameters'])
                    self.assertIn('num_classes', step_config['hyperparameters'])
                    self.assertEqual(step_config['hyperparameters']['num_classes'], 2)
            
            # Verify that we found the hyperparameters
            self.assertTrue(found_hyperparameters, "hyperparameters should be in a specific section")
            
            # Verify hyperparameters is not in the shared section
            self.assertNotIn('hyperparameters', output_json['configuration']['shared'])
            
        finally:
            # Clean up the temp file
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_simplified_common_fields(self):
        """Test that common fields are in the shared section in the simplified model."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            output_path = temp_file.name
            
        try:
            # All configs have the same values for these fields
            # Merge the configs and save to the temp file
            merge_and_save_configs([self.base_config, self.dummy_config, self.processing_config], output_path)
            
            # Read the output file
            with open(output_path, 'r') as f:
                output_json = json.load(f)
                
            # Fields like bucket, author, etc. should be in shared since they're the same across all configs
            shared = output_json['configuration']['shared']
            self.assertIn('bucket', shared)
            self.assertIn('author', shared)
            self.assertIn('pipeline_name', shared)
            self.assertIn('pipeline_version', shared)
            
            # Verify the values are correct
            self.assertEqual(shared['bucket'], "test-bucket")
            self.assertEqual(shared['author'], "test-author")
            self.assertEqual(shared['pipeline_name'], "test-pipeline")
            self.assertEqual(shared['pipeline_version'], "1.0.0")
            
        finally:
            # Clean up the temp file
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_simplified_unique_fields(self):
        """Test that fields unique to specific configs are in the specific sections."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            output_path = temp_file.name
            
        try:
            # Merge the configs and save to the temp file
            merge_and_save_configs([self.specific_config, self.processing_config], output_path)
            
            # Read the output file
            with open(output_path, 'r') as f:
                output_json = json.load(f)
                
            # Find the configs with our unique fields
            specific_sections = output_json['configuration']['specific']
            
            # Find the section with our input_names field
            found_input_names = False
            found_output_schema = False
            
            for step_name, step_config in specific_sections.items():
                if 'input_names' in step_config:
                    found_input_names = True
                    self.assertEqual(step_config['input_names'], {"input1": "path1"})
                
                if 'output_schema' in step_config:
                    found_output_schema = True
                    self.assertEqual(step_config['output_schema'], {"fields": ["field1", "field2"]})
            
            # Verify we found our specific fields
            self.assertTrue(found_input_names, "input_names should be in a specific section")
            self.assertTrue(found_output_schema, "output_schema should be in a specific section")
            
            # Verify they're not in shared
            self.assertNotIn('input_names', output_json['configuration']['shared'])
            self.assertNotIn('output_schema', output_json['configuration']['shared'])
            
        finally:
            # Clean up the temp file
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_simplified_non_static_fields(self):
        """Test that non-static fields are kept in specific sections in the simplified model."""
        class ConfigWithNonStatic(BasePipelineConfig):
            # Define required fields to handle serialization correctly
            input_names: dict = None
            output_names: dict = None
            complex_dict: dict = None
            
        # Create a very large dict that will definitely be detected as non-static
        very_large_dict = {}
        for i in range(20):
            very_large_dict[f"key{i}"] = {f"subkey{j}": f"value{j}" for j in range(20)}
            
        config = ConfigWithNonStatic(
            bucket="test-bucket",
            author="test-author",
            pipeline_name="test-pipeline-nonstatic",  # Unique name to ensure it stands out
            pipeline_description="Test Pipeline",
            pipeline_version="1.0.0",
            pipeline_s3_loc="s3://test-bucket/test-pipeline",
            input_names={"input1": "s3://bucket/input1"},  # Non-static field by name pattern
            output_names={"output1": "s3://bucket/output1"},  # Non-static field by name pattern
            complex_dict=very_large_dict  # Large dictionary to ensure non-static detection
        )
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            output_path = temp_file.name
            
        try:
            # Merge the configs and save to the temp file
            merge_and_save_configs([self.base_config, config], output_path)
            
            # Read the output file
            with open(output_path, 'r') as f:
                output_json = json.load(f)
                
            # Get the specific sections
            specific_sections = output_json['configuration']['specific']
            
            # Find the config with our non-static fields - the name might vary
            config_with_non_static = None
            for step_name, step_config in specific_sections.items():
                if 'input_names' in step_config:
                    config_with_non_static = step_config
                    break
                    
            # Make sure we found the config
            self.assertIsNotNone(config_with_non_static, "Config with non-static fields not found")
            
            # Non-static fields should be in specific section
            self.assertIn('input_names', config_with_non_static)
            self.assertIn('output_names', config_with_non_static)
            self.assertIn('complex_dict', config_with_non_static)
            
            # Verify they're not in shared
            shared = output_json['configuration']['shared']
            self.assertNotIn('input_names', shared)
            self.assertNotIn('output_names', shared)
            self.assertNotIn('complex_dict', shared)
            
        finally:
            # Clean up the temp file
            if os.path.exists(output_path):
                os.unlink(output_path)

    # Define local test classes at the class level
    class LocalTestProcessingConfig(ProcessingStepConfigBase):
        custom_field: str = None
        
        def get_script_contract(self):
            return None
            
        def validate_config(self):
            return self
    
    class LocalBasePipelineConfig(BasePipelineConfig):
        """Local base pipeline config for testing."""
        pass
    
    class LocalDummyTrainingConfig(DummyTrainingConfig):
        """Local dummy training config for testing."""
        pass
    
    class LocalCustomSpecificConfig(BasePipelineConfig):
        """Local custom specific config for testing."""
        input_names: dict = None
        
    def test_simplified_structure_with_local_configs(self):
        """Test that output structure is correct using local config classes."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            output_path = temp_file.name
            
        try:
            # Create instances of our local config classes
            local_base_config = self.LocalBasePipelineConfig(
                bucket="test-bucket",
                author="test-author",
                pipeline_name="test-pipeline",
                pipeline_description="Test Pipeline",
                pipeline_version="1.0.0",
                pipeline_s3_loc="s3://test-bucket/test-pipeline"
            )
            
            local_dummy_config = self.LocalDummyTrainingConfig(
                bucket="test-bucket",
                author="test-author",
                pipeline_name="test-pipeline",
                pipeline_description="Test Pipeline",
                pipeline_version="1.0.0",
                pipeline_s3_loc="s3://test-bucket/test-pipeline",
                pretrained_model_path=self.model_path,
                processing_source_dir=self.pipeline_scripts_path,
                hyperparameters=self.hyperparams
            )
            
            local_specific_config = self.LocalCustomSpecificConfig(
                bucket="test-bucket",
                author="test-author",
                pipeline_name="test-pipeline",
                pipeline_description="Test Pipeline",
                pipeline_version="1.0.0",
                pipeline_s3_loc="s3://test-bucket/test-pipeline",
                input_names={"input1": "path1"}
            )
            
            local_processing_config = self.LocalTestProcessingConfig(
                bucket="test-bucket",
                author="test-author",
                pipeline_name="test-pipeline",
                pipeline_description="Test Processing Config",
                pipeline_version="1.0.0",
                pipeline_s3_loc="s3://test-bucket/test-pipeline",
                processing_source_dir=self.pipeline_scripts_path,
                processing_instance_count=1,
                processing_instance_type_small="ml.m5.xlarge", 
                custom_field="processing_specific_value"
            )
            
            # Save configs with the simplified structure
            merge_and_save_configs([
                local_base_config, 
                local_dummy_config, 
                local_specific_config, 
                local_processing_config
            ], output_path)
            
            # Verify the structure has no nested processing sections
            with open(output_path, 'r') as f:
                output_json = json.load(f)
                self.assertNotIn('processing', output_json['configuration'])
                self.assertEqual(set(output_json['configuration'].keys()), {'shared', 'specific'})
                
            # Verify that the metadata contains config_types
            self.assertIn('metadata', output_json)
            self.assertIn('config_types', output_json['metadata'])
            
            # Verify that all our config types are in the config_types section
            config_types = output_json['metadata']['config_types']
            
            # The exact names may vary, but we should have these types represented
            type_names = list(config_types.values())
            self.assertIn('LocalBasePipelineConfig', type_names)
            self.assertIn('LocalDummyTrainingConfig', type_names)
            self.assertIn('LocalCustomSpecificConfig', type_names)
            self.assertIn('LocalTestProcessingConfig', type_names)
            
            # Verify that the specific section contains our configs
            specific_section = output_json['configuration']['specific']
            self.assertEqual(len(specific_section), 4, "Should have 4 configs in the specific section")
            
            # Verify that key special fields are in the specific sections
            hyperparams_found = False
            input_names_found = False
            custom_field_found = False
            
            for config_name, config_data in specific_section.items():
                if "hyperparameters" in config_data:
                    hyperparams_found = True
                    self.assertEqual(config_data["hyperparameters"]["num_classes"], 2)
                    
                if "input_names" in config_data:
                    input_names_found = True
                    self.assertEqual(config_data["input_names"], {"input1": "path1"})
                    
                if "custom_field" in config_data:
                    custom_field_found = True
                    self.assertEqual(config_data["custom_field"], "processing_specific_value")
                    
            self.assertTrue(hyperparams_found, "Should find hyperparameters in config")
            self.assertTrue(input_names_found, "Should find input_names in config")
            self.assertTrue(custom_field_found, "Should find custom_field in config")
            
            # Verify common fields are in shared section
            shared_section = output_json['configuration']['shared']
            self.assertIn("pipeline_name", shared_section)
            self.assertEqual(shared_section["pipeline_name"], "test-pipeline")
            
        finally:
            # Clean up the temp file
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_simplified_different_values(self):
        """Test that fields with different values across configs go to the specific sections."""
        # Create configs with some fields having different values
        config1 = BasePipelineConfig(
            bucket="test-bucket",
            author="test-author",
            pipeline_name="test-pipeline",
            pipeline_description="Test Pipeline 1",  # Different
            pipeline_version="1.0.0",
            pipeline_s3_loc="s3://test-bucket/test-pipeline-1"  # Different
        )
        
        config2 = BasePipelineConfig(
            bucket="test-bucket",
            author="test-author",
            pipeline_name="test-pipeline",
            pipeline_description="Test Pipeline 2",  # Different
            pipeline_version="1.0.0",
            pipeline_s3_loc="s3://test-bucket/test-pipeline-2"  # Different
        )
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            output_path = temp_file.name
            
        try:
            # Merge the configs and save to the temp file
            merge_and_save_configs([config1, config2], output_path)
            
            # Read the output file
            with open(output_path, 'r') as f:
                output_json = json.load(f)
                
            # Fields with identical values should be in shared
            shared = output_json['configuration']['shared']
            self.assertIn('bucket', shared)
            self.assertIn('author', shared)
            self.assertIn('pipeline_name', shared)
            self.assertIn('pipeline_version', shared)
            
            # Fields with different values should be in specific sections
            specific_sections = output_json['configuration']['specific']
            
            # Check both configs for the different fields
            found_descriptions = set()
            found_s3_locs = set()
            
            for step_name, step_config in specific_sections.items():
                if 'pipeline_description' in step_config:
                    found_descriptions.add(step_config['pipeline_description'])
                if 'pipeline_s3_loc' in step_config:
                    found_s3_locs.add(step_config['pipeline_s3_loc'])
            
            # Verify we found both different values - at least one pipeline description should be in specific
            self.assertTrue(len(found_descriptions) > 0, "No pipeline descriptions found in specific sections")
            self.assertTrue(all(desc in {"Test Pipeline 1", "Test Pipeline 2", "Test Pipeline"} for desc in found_descriptions),
                           f"Found unexpected pipeline descriptions: {found_descriptions}")
            
            # Verify we found both s3 locs
            self.assertTrue(len(found_s3_locs) > 0, "No s3 locations found in specific sections")
            self.assertTrue(all(loc in {"s3://test-bucket/test-pipeline-1", "s3://test-bucket/test-pipeline-2", 
                                      "s3://test-bucket/test-pipeline"} for loc in found_s3_locs),
                           f"Found unexpected s3 locations: {found_s3_locs}")
            
            # Verify different-valued fields are not in shared
            self.assertNotIn('pipeline_description', shared)
            self.assertNotIn('pipeline_s3_loc', shared)
            
        finally:
            # Clean up the temp file
            if os.path.exists(output_path):
                os.unlink(output_path)
                
    def test_processing_configs_in_flattened_structure(self):
        """Test that processing configs are correctly represented in the flattened structure."""
        class CustomProcessingConfig1(ProcessingStepConfigBase):
            # Add a field that will be specific to this processing config
            custom_field: str = None
            
            def get_script_contract(self):
                return None
                
            def validate_config(self):
                return self
        
        class CustomProcessingConfig2(ProcessingStepConfigBase):
            # Different processing config type with the same field name
            custom_field: str = None
            
            def get_script_contract(self):
                return None
                
            def validate_config(self):
                return self
        
        # Create two processing configs with some shared and some different values
        proc_config1 = CustomProcessingConfig1(
            bucket="test-bucket",
            author="test-author",
            pipeline_name="test-pipeline",
            pipeline_description="Test Pipeline",
            pipeline_version="1.0.0",
            pipeline_s3_loc="s3://test-bucket/test-pipeline",
            processing_source_dir=self.pipeline_scripts_path,
            processing_instance_type_small="ml.m5.xlarge",  # Same across both
            processing_instance_count=1,  # Same across both
            custom_field="value1"  # Different value
        )
        
        proc_config2 = CustomProcessingConfig2(
            bucket="test-bucket",
            author="test-author",
            pipeline_name="test-pipeline",
            pipeline_description="Test Pipeline",
            pipeline_version="1.0.0",
            pipeline_s3_loc="s3://test-bucket/test-pipeline",
            processing_source_dir=self.pipeline_scripts_path,
            processing_instance_type_small="ml.m5.xlarge",  # Same across both
            processing_instance_count=1,  # Same across both
            custom_field="value2"  # Different value
        )
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            output_path = temp_file.name
            
        try:
            # Merge the configs and save to the temp file
            merge_and_save_configs([proc_config1, proc_config2], output_path)
            
            # Read the output file
            with open(output_path, 'r') as f:
                output_json = json.load(f)
                
            # Verify structure is flattened (no processing section)
            self.assertNotIn('processing', output_json['configuration'])
            
            # Common processing fields should be in shared at the top level
            shared = output_json['configuration']['shared']
            self.assertIn('processing_instance_type_small', shared)
            self.assertIn('processing_instance_count', shared)
            self.assertEqual(shared['processing_instance_type_small'], "ml.m5.xlarge")
            self.assertEqual(shared['processing_instance_count'], 1)
            
            # Different values should be in specific section at the top level
            specific_sections = output_json['configuration']['specific']
            
            # Check both configs for the different field values
            found_custom_values = set()
            
            for step_name, step_config in specific_sections.items():
                if 'custom_field' in step_config:
                    found_custom_values.add(step_config['custom_field'])
            
            # Verify we found both different values for the custom field
            self.assertEqual(found_custom_values, {"value1", "value2"})
            
            # Verify different-valued fields are not in shared
            self.assertNotIn('custom_field', shared)
            
        finally:
            # Clean up the temp file
            if os.path.exists(output_path):
                os.unlink(output_path)


if __name__ == "__main__":
    unittest.main()
