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
from src.pipeline_steps.utils_legacy import merge_and_save_configs_legacy as merge_and_save_configs
from src.pipeline_steps.utils_legacy import serialize_config, _serialize, _is_likely_static
from src.pipeline_steps.utils_legacy import SPECIAL_FIELDS_TO_KEEP_SPECIFIC
from src.pipeline_steps.config_base import BasePipelineConfig
from src.pipeline_steps.config_processing_step_base import ProcessingStepConfigBase
from src.pipeline_steps.config_dummy_training_step import DummyTrainingConfig
from src.pipeline_steps.hyperparameters_base import ModelHyperparameters


class TestConfigFieldCategorization(unittest.TestCase):
    """
    Tests for the configuration field categorization logic in utils.py.
    This tests the Decision Process Flow described in config_field_categorization.md.
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

    def test_1_special_fields_handling(self):
        """Test that special fields like hyperparameters are always kept in specific sections."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            output_path = temp_file.name
            
        try:
            # Merge the configs and save to the temp file
            merge_and_save_configs([self.base_config, self.dummy_config], output_path)
            
            # Read the output file
            with open(output_path, 'r') as f:
                output_json = json.load(f)
                
            # Verify special fields are in processing_specific section
            dummy_specific = output_json['configuration']['processing']['processing_specific']['DummyTraining']
            self.assertIn('hyperparameters', dummy_specific)
            
            # Verify hyperparameters is not in the shared or processing_shared sections
            self.assertNotIn('hyperparameters', output_json['configuration']['shared'])
            self.assertNotIn('hyperparameters', output_json['configuration']['processing']['processing_shared'])
            
        finally:
            # Clean up the temp file
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_2_cross_type_fields(self):
        """Test that cross-type fields (fields in both processing and non-processing configs) are handled correctly."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            output_path = temp_file.name
            
        try:
            # Merge the configs and save to the temp file
            merge_and_save_configs([self.specific_config, self.processing_config], output_path)
            
            # Read the output file
            with open(output_path, 'r') as f:
                output_json = json.load(f)
                
            # Print output for debugging
            print(f"Output JSON structure: {json.dumps(output_json, indent=2)}")
            
            # Fields like bucket, author, etc. appear in both config types and should be in shared
            shared = output_json['configuration']['shared']
            self.assertIn('bucket', shared)
            self.assertIn('author', shared)
            
            # Unique fields should be in their specific sections
            # Find the processing config in the output - the actual name might vary
            proc_specific_sections = output_json['configuration']['processing']['processing_specific']
            
            # Find the section with our special field (output_schema is in SPECIAL_FIELDS_TO_KEEP_SPECIFIC)
            found_output_schema = False
            for step_name, step_config in proc_specific_sections.items():
                if 'output_schema' in step_config:
                    found_output_schema = True
                    self.assertEqual(step_config['output_schema'], {"fields": ["field1", "field2"]})
            self.assertTrue(found_output_schema, "output_schema not found in any specific section")
            self.assertNotIn('output_schema', shared)
            
            # Find the specific config in the output
            specific_sections = output_json['configuration']['specific']
            
            # Find the section with our non-static field
            found_input_names = False
            for step_name, step_config in specific_sections.items():
                if 'input_names' in step_config:
                    found_input_names = True
                    self.assertEqual(step_config['input_names'], {"input1": "path1"})
            self.assertTrue(found_input_names, "input_names not found in any specific section")
            self.assertNotIn('input_names', shared)
            
        finally:
            # Clean up the temp file
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_3_processing_fields_categorization(self):
        """Test that processing fields are correctly categorized into processing_shared or processing_specific."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            output_path = temp_file.name
            
        try:
            # Create a second processing config with the same values for some fields
            # but different values for others
            class AnotherProcessingConfig(ProcessingStepConfigBase):
                # Properly define output_schema as a model field
                output_schema: dict = None
                
                def get_script_contract(self):
                    return None
                
                def validate_config(self):
                    return self
                    
            another_config = AnotherProcessingConfig(
                bucket="test-bucket",
                author="test-author",
                pipeline_name="test-pipeline",
                pipeline_description="Test Pipeline",
                pipeline_version="1.0.0",
                pipeline_s3_loc="s3://test-bucket/test-pipeline",
                processing_source_dir=self.pipeline_scripts_path,  # Use absolute path
                # Same for most processing fields
                processing_instance_type_small="ml.m5.xlarge",
                processing_framework_version="0.23-1",
                # But different for some
                processing_instance_count=3,  # Different from other configs
                # Use another field in SPECIAL_FIELDS_TO_KEEP_SPECIFIC
                output_schema={"fields": ["field3", "field4"]}
            )
            
            # Merge the configs and save to the temp file
            merge_and_save_configs([self.processing_config, another_config], output_path)
            
            # Read the output file
            with open(output_path, 'r') as f:
                output_json = json.load(f)
                
            # Read the output file and print for debugging
            with open(output_path, 'r') as f:
                output_json = json.load(f)
            print(f"Test 3 Output: {json.dumps(output_json, indent=2)}")
            
            # Fields with same values across all processing configs should be in processing_shared
            proc_shared = output_json['configuration']['processing']['processing_shared']
            # Check some fields that should definitely be shared (from the output logs we can see these are shared)
            self.assertIn('processing_framework_version', proc_shared)
            self.assertIn('processing_volume_size', proc_shared)
            
            # Get the specific sections
            proc_specific_sections = output_json['configuration']['processing']['processing_specific']
            self.assertEqual(len(proc_specific_sections), 2, "Should have two processing configs in specific sections")
            
            # Find the configs with our special fields
            config_with_output_schema1 = None
            config_with_output_schema2 = None
            
            for step_name, step_config in proc_specific_sections.items():
                if 'output_schema' in step_config:
                    if step_config['output_schema'].get('fields') == ["field1", "field2"]:
                        config_with_output_schema1 = step_config
                    elif step_config['output_schema'].get('fields') == ["field3", "field4"]:
                        config_with_output_schema2 = step_config
            
            # Check that we found both configs
            self.assertIsNotNone(config_with_output_schema1, "Config with first output_schema not found")
            self.assertIsNotNone(config_with_output_schema2, "Config with second output_schema not found")
            
            # Fields with different values should be in processing_specific
            self.assertIn('processing_instance_count', config_with_output_schema1)
            self.assertIn('processing_instance_count', config_with_output_schema2)
            
            # Check the actual values
            self.assertEqual(config_with_output_schema1['processing_instance_count'], 2)
            self.assertEqual(config_with_output_schema2['processing_instance_count'], 3)
            
        finally:
            # Clean up the temp file
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_4_non_static_fields(self):
        """Test that non-static fields are kept in specific sections."""
        class ConfigWithNonStatic(BasePipelineConfig):
            # Define required fields to handle serialization correctly
            input_names: dict = None
            output_names: dict = None
            output_schema: dict = None
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
            output_schema={"fields": ["field1", "field2"]},  # In SPECIAL_FIELDS_TO_KEEP_SPECIFIC
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
                
            # Print the output for debugging
            print(f"Test 4 Output: {json.dumps(output_json, indent=2)}")
            
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
            self.assertIn('output_schema', config_with_non_static)
            self.assertIn('complex_dict', config_with_non_static)
            
            # Verify they're not in shared
            shared = output_json['configuration']['shared']
            self.assertNotIn('input_names', shared)
            self.assertNotIn('output_names', shared)
            self.assertNotIn('output_schema', shared)
            self.assertNotIn('complex_dict', shared)
            
        finally:
            # Clean up the temp file
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_5_serialization_of_complex_types(self):
        """Test that complex types like Pydantic models are properly serialized."""
        # Test serialization of the hyperparameters object
        serialized = _serialize(self.hyperparams)
        
        # Verify it's a dictionary
        self.assertIsInstance(serialized, dict)
        
        # Verify key fields are present
        self.assertIn('full_field_list', serialized)
        self.assertIn('cat_field_list', serialized)
        self.assertIn('tab_field_list', serialized)
        self.assertIn('is_binary', serialized)
        self.assertIn('num_classes', serialized)
        
        # Verify values are correctly serialized
        self.assertEqual(serialized['is_binary'], True)
        self.assertEqual(serialized['num_classes'], 2)
        self.assertEqual(serialized['cat_field_list'], ['field3'])

    def test_6_static_field_detection(self):
        """Test the _is_likely_static function for detecting static vs. non-static fields."""
        # Static fields
        self.assertTrue(_is_likely_static('author', 'test-author'))
        self.assertTrue(_is_likely_static('bucket', 'test-bucket'))
        self.assertTrue(_is_likely_static('simple_field', 'simple_value'))
        self.assertTrue(_is_likely_static('simple_number', 42))
        self.assertTrue(_is_likely_static('simple_list', [1, 2, 3]))
        
        # Non-static fields by name pattern
        self.assertFalse(_is_likely_static('input_names', {'input1': 'path1'}))
        self.assertFalse(_is_likely_static('output_names', {'output1': 'path1'}))
        self.assertFalse(_is_likely_static('field_names', ['name1', 'name2']))
        
        # Test SPECIAL_FIELDS_TO_KEEP_SPECIFIC fields
        for special_field in SPECIAL_FIELDS_TO_KEEP_SPECIFIC:
            self.assertFalse(_is_likely_static(special_field, {'a': 1}), 
                          f"{special_field} should be detected as non-static")
        
        # Complex structures (lists and dicts) - need to be very large to be detected
        # since smaller ones are considered static by the implementation
        very_large_dict = {}
        for i in range(15):
            very_large_dict[f'key{i}'] = {f'nested{j}': f'value{j}' for j in range(15)}
        self.assertFalse(_is_likely_static('very_large_dict', very_large_dict))
        
        # Very long list
        very_long_list = list(range(20))
        self.assertFalse(_is_likely_static('very_long_list', very_long_list))
        
        # Pydantic models are always non-static
        self.assertFalse(_is_likely_static('hyperparameters', self.hyperparams))

    def test_7_special_field_recovery(self):
        """Test that special fields are properly recovered if they end up in shared."""
        # Create a custom config class with output_schema properly defined
        # Add job_type attribute for distinguishing configs of the same class
        class CustomSpecialFieldConfig(ProcessingStepConfigBase):
            # Properly define output_schema as a model field
            output_schema: dict = None
            # Add job_type attribute for distinguishing configs of the same class
            job_type: str = None
            
            def get_script_contract(self):
                return None
                
            def validate_config(self):
                return self
        
        # Create identical output_schema values for both configs
        output_schema = {"fields": ["field1", "field2"], "type": "schema"}
        
        # Create two configs with identical output_schema values but different job_types
        # This follows the design assumption that configs of the same class need distinguishing attributes
        with patch('pathlib.Path.exists', return_value=True):  # This ensures path validation passes
            custom_config1 = CustomSpecialFieldConfig(
                bucket="test-bucket",
                author="test-author",
                pipeline_name="test-config",
                pipeline_description="Test Config",
                pipeline_version="1.0.0",
                pipeline_s3_loc="s3://test-bucket/test-config",
                processing_source_dir=self.pipeline_scripts_path,
                output_schema=output_schema,  # Same output_schema for both configs
                job_type="training"  # Distinguishing attribute
            )
            
            custom_config2 = CustomSpecialFieldConfig(
                bucket="test-bucket",
                author="test-author",
                pipeline_name="test-config",
                pipeline_description="Test Config",
                pipeline_version="1.0.0",
                pipeline_s3_loc="s3://test-bucket/test-config",
                processing_source_dir=self.pipeline_scripts_path,
                output_schema=output_schema,  # Same output_schema for both configs
                job_type="inference"  # Different distinguishing attribute
            )
        
        # Override SPECIAL_FIELDS_TO_KEEP_SPECIFIC to ensure output_schema is included
        with patch('src.pipeline_steps.utils_legacy.SPECIAL_FIELDS_TO_KEEP_SPECIFIC', {'output_schema'}):
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
                output_path = temp_file.name
                
            try:
                # Without special recovery, identical output_schema values would go to shared
                # but our special field handling should move them back to processing_specific
                merge_and_save_configs([custom_config1, custom_config2], output_path)
                
                # Read the output file
                with open(output_path, 'r') as f:
                    output_json = json.load(f)
                    
                # Print output for debugging
                print(f"Test 7 Output: {json.dumps(output_json, indent=2)}")
                
                # Verify output_schema is not in shared
                shared = output_json['configuration']['shared']
                self.assertNotIn('output_schema', shared, 
                              "output_schema should not be in shared section")
                
                # Verify output_schema is in both processing_specific sections
                # Since we're using job_type to distinguish configs, we should check for the step names
                # that include the job_type
                specific_configs = output_json['configuration']['processing']['processing_specific']
                self.assertEqual(len(specific_configs), 2, 
                               "Should have two configs in processing_specific section")
                
                # Verify both step names exist with the correct format (including job_type)
                self.assertTrue(
                    any("training" in step_name for step_name in specific_configs.keys()),
                    "Should have a config with job_type=training"
                )
                self.assertTrue(
                    any("inference" in step_name for step_name in specific_configs.keys()),
                    "Should have a config with job_type=inference"
                )
                
                # Count configs with output_schema
                configs_with_output_schema = 0
                for step_name, step_config in specific_configs.items():
                    if 'output_schema' in step_config:
                        configs_with_output_schema += 1
                        # Verify the schema content is correct
                        self.assertEqual(step_config['output_schema'], output_schema)
                        
                # We should have output_schema in both configs
                self.assertEqual(configs_with_output_schema, 2, 
                               "Should have output_schema in both processing configs")
                    
            finally:
                # Clean up the temp file
                if os.path.exists(output_path):
                    os.unlink(output_path)


if __name__ == "__main__":
    unittest.main()
