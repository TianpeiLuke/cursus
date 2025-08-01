import unittest
import json
import os
import tempfile
import sys
from pathlib import Path

# Add the repository root directory to the path - more reliable method
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from unittest.mock import patch, MagicMock

# Import utilities for config serialization
from src.pipeline_steps.utils import merge_and_save_configs, serialize_config
from src.pipeline_steps.config_base import BasePipelineConfig
from src.pipeline_steps.hyperparameters_base import ModelHyperparameters
from src.config_field_manager.config_field_categorizer import ConfigFieldCategorizer
from src.config_field_manager.constants import SPECIAL_FIELDS_TO_KEEP_SPECIFIC, CategoryType


class TestBasicSerializationAndFields(unittest.TestCase):
    """Test basic serialization and field categorization functionality for configurations."""
    
    def test_special_fields_in_serialization(self):
        """Test that Pydantic models are properly serialized."""
        # Create a sample hyperparameters object
        hyperparams = ModelHyperparameters(
            full_field_list=["field1", "field2"],
            cat_field_list=["field2"],
            tab_field_list=["field1"],
            input_tab_dim=1,
            is_binary=True,
            num_classes=2,
            multiclass_categories=[0, 1],
            class_weights=[1.0, 2.0]
        )
        
        # Serialize it
        serialized = serialize_config(hyperparams)
        
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
        self.assertEqual(serialized['cat_field_list'], ['field2'])
        
    def test_static_field_detection(self):
        """Test the _is_likely_static function for detecting static vs. non-static fields."""
        # Create a sample hyperparameters object
        hyperparams = ModelHyperparameters(
            full_field_list=["field1", "field2"],
            cat_field_list=["field2"],
            tab_field_list=["field1"],
            input_tab_dim=1
        )
        
        # Create a categorizer to use its _is_likely_static method
        categorizer = ConfigFieldCategorizer([], None)
        
        # Static fields
        self.assertTrue(categorizer._is_likely_static('author', 'test-author'))
        self.assertTrue(categorizer._is_likely_static('bucket', 'test-bucket'))
        self.assertTrue(categorizer._is_likely_static('simple_field', 'simple_value'))
        self.assertTrue(categorizer._is_likely_static('simple_number', 42))
        self.assertTrue(categorizer._is_likely_static('simple_list', [1, 2, 3]))
        
        # Non-static fields by name pattern
        self.assertFalse(categorizer._is_likely_static('input_names', {'input1': 'path1'}))
        self.assertFalse(categorizer._is_likely_static('output_names', {'output1': 'path1'}))
        self.assertFalse(categorizer._is_likely_static('field_names', ['name1', 'name2']))
        
        # Very large dictionaries should be non-static
        big_dict = {}
        for i in range(10):
            big_dict[f"key{i}"] = {f"subkey{j}": f"value{j}" for j in range(10)}
        self.assertFalse(categorizer._is_likely_static('big_complex_dict', big_dict))
        
        # Long lists
        self.assertFalse(categorizer._is_likely_static('long_list', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]))
        
        # Non-static fields by type
        self.assertFalse(categorizer._is_likely_static('hyperparameters', hyperparams))
        
    def test_basic_merge_and_fields_in_shared(self):
        """Test that fields with identical values go to shared."""
        # Create two configs with identical fields
        config1 = BasePipelineConfig(
            bucket="test-bucket",
            author="test-author",
            pipeline_name="test-pipeline-1",
            pipeline_description="Test Pipeline"
        )
        
        config2 = BasePipelineConfig(
            bucket="test-bucket",
            author="test-author", 
            pipeline_name="test-pipeline-2",
            pipeline_description="Test Pipeline"
        )
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            output_path = temp_file.name
            
        try:
            # Merge the configs and save to the temp file
            merge_and_save_configs([config1, config2], output_path)
            
            # Read the output file
            with open(output_path, 'r') as f:
                output_json = json.load(f)
                
            # Print the JSON structure to help debug
            print(f"Output JSON: {json.dumps(output_json, indent=2)}")
                
            # Fields with identical values should be in shared
            shared = output_json['configuration']['shared']
            self.assertIn('bucket', shared)
            self.assertIn('author', shared)
            self.assertIn('pipeline_description', shared)
            
            # Pipeline name is different, so it should be in specific sections
            self.assertNotIn('pipeline_name', shared)
            
        finally:
            # Clean up the temp file
            if os.path.exists(output_path):
                os.unlink(output_path)
                
    def test_pydantic_model_serialization(self):
        """Test directly that our Pydantic model special field is serialized correctly."""
        # Create a sample hyperparameters model
        hyperparams = ModelHyperparameters(
            full_field_list=["field1", "field2"],
            cat_field_list=["field2"],
            tab_field_list=["field1"],
            input_tab_dim=1,
            is_binary=True,
            num_classes=2
        )
        
        # Create a categorizer to use its _is_likely_static method
        categorizer = ConfigFieldCategorizer([], None)
        
        # Now check that this complex type is properly identified as special
        self.assertFalse(categorizer._is_likely_static('hyperparameters', hyperparams),
                       "Pydantic models should be identified as non-static")
        
        # Verify serialize works properly
        serialized = serialize_config(hyperparams)
        self.assertIsInstance(serialized, dict)
        self.assertIn('full_field_list', serialized)
        
        # Create a categorizer instance
        categorizer = ConfigFieldCategorizer([], None)
        
        # Check that special fields are detected properly
        with patch('src.config_field_manager.constants.SPECIAL_FIELDS_TO_KEEP_SPECIFIC', {'hyperparameters'}):
            # Create a mock config with the hyperparameters field
            mock_config = MagicMock()
            mock_config.hyperparameters = hyperparams
            
            # The field should be identified as one to keep specific
            self.assertTrue(
                categorizer._is_special_field('hyperparameters', hyperparams, mock_config),
                "hyperparameters field should be identified as special"
            )


if __name__ == "__main__":
    unittest.main()
