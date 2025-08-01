"""
Tests for the field_sources functionality in utils.py
"""

import unittest
import sys
import os
import json
import tempfile
from pathlib import Path

# Add the repository root directory to the path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repo_root)

from src.pipeline_steps.utils import get_field_sources, merge_and_save_configs
from src.pipeline_steps.config_base import BasePipelineConfig
from src.pipeline_steps.config_processing_step_base import ProcessingStepConfigBase
from src.pipeline_steps.config_dummy_training_step import DummyTrainingConfig
from src.pipeline_steps.hyperparameters_base import ModelHyperparameters


class TestFieldSources(unittest.TestCase):
    """Test the field_sources functionality"""
    
    def setUp(self):
        """Set up test data"""
        # Create a base config
        self.base_config = BasePipelineConfig(
            pipeline_name="test-pipeline",
            pipeline_description="Test Description",
            pipeline_version="1.0.0",
            author="test-author",
            bucket="test-bucket",
            pipeline_s3_loc="s3://test-bucket/test",
            region="NA"
        )
        
        # Create a simple processing config
        class CustomProcessingConfig(ProcessingStepConfigBase):
            output_schema: dict = {"fields": ["field1", "field2"]}
            processing_instance_count: int = 2
            processing_instance_type_small: str = "ml.m5.2xlarge"
        
        # Create a processing config instance
        self.processing_config = CustomProcessingConfig(
            pipeline_name="test-pipeline",
            pipeline_description="Test Description",
            pipeline_version="1.0.0",
            author="test-author",
            bucket="test-bucket",
            pipeline_s3_loc="s3://test-bucket/test",
            region="NA",
            processing_source_dir=str(Path(__file__).parent.parent.parent / "src" / "pipeline_scripts")
        )
        
        # Create a dummy training config
        hyperparams = ModelHyperparameters(
            full_field_list=["field1", "field2", "field3"],
            cat_field_list=["field3"],
            tab_field_list=["field1", "field2"],
            input_tab_dim=2,
            is_binary=True,
            num_classes=2
        )
        
        # Find the model path in the test directory
        model_path = Path(__file__).parent / "model.tar.gz"
        if not model_path.exists():
            raise FileNotFoundError(f"Test model file not found: {model_path}")
        
        self.training_config = DummyTrainingConfig(
            pipeline_name="test-pipeline",
            pipeline_description="Test Description",
            pipeline_version="1.0.0",
            author="test-author",
            bucket="test-bucket",
            pipeline_s3_loc="s3://test-bucket/test",
            region="NA",
            processing_source_dir=str(Path(__file__).parent.parent.parent / "src" / "pipeline_scripts"),
            pretrained_model_path=str(model_path),
            hyperparameters=hyperparams
        )
    
    def test_field_sources_categories(self):
        """Test that field_sources has the expected categories"""
        # Print the serialized output for the training config to see what fields are present
        from src.pipeline_steps.utils import serialize_config
        serialized = serialize_config(self.training_config)
        print("SERIALIZED TRAINING CONFIG:")
        print(f"Fields: {list(serialized.keys())}")
        if 'hyperparameters' in serialized:
            print(f"Hyperparameters found in serialized output: {type(serialized['hyperparameters'])}")

        config_list = [self.base_config, self.processing_config, self.training_config]
        field_sources = get_field_sources(config_list)
        
        # Print debug info
        print("\nFIELD SOURCES DEBUG:")
        print(f"Categories: {list(field_sources.keys())}")
        print(f"'all' fields: {list(field_sources['all'].keys())}")
        print(f"'specific' fields: {list(field_sources['specific'].keys())}")
        print(f"'processing' fields: {list(field_sources['processing'].keys())}")
        
        # Verify categories exist
        self.assertIn('all', field_sources)
        self.assertIn('processing', field_sources)
        self.assertIn('specific', field_sources)
        
        # Verify some fields are categorized correctly
        # Base fields should be in 'all' and either 'processing' or 'specific'
        self.assertIn('pipeline_name', field_sources['all'])
        self.assertIn('author', field_sources['all'])
        
        # Processing specific fields
        self.assertIn('processing_instance_count', field_sources['processing'])
        self.assertIn('processing_source_dir', field_sources['processing'])
        
        # Hyperparameters should be in specific (as they only appear in the training config)
        self.assertIn('hyperparameters', field_sources['specific'])
        
    def test_field_sources_values(self):
        """Test that field_sources contains correct step names"""
        config_list = [self.base_config, self.processing_config, self.training_config]
        field_sources = get_field_sources(config_list)
        
        # Common fields should be in multiple configs
        self.assertGreaterEqual(len(field_sources['all']['pipeline_name']), 3)
        self.assertGreaterEqual(len(field_sources['all']['author']), 3)
        
        # Processing fields should only be in processing configs
        self.assertGreaterEqual(len(field_sources['processing']['processing_instance_count']), 2)
        
        # Each source should be a step name
        for category in field_sources:
            for field, sources in field_sources[category].items():
                for source in sources:
                    # Ensure each source is a string (step name)
                    self.assertIsInstance(source, str)
                    self.assertNotEqual(source, '')  # Not empty
                    self.assertNotEqual(source, 'unknown')  # Not unknown
    
    def test_field_sources_metadata(self):
        """Test that field_sources metadata is added to the JSON output."""
        # Create a temp file for the test
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        temp_file.close()
        
        try:
            # Merge and save configs
            config_list = [self.base_config, self.processing_config]
            result = merge_and_save_configs(config_list, temp_file.name)
            
            # Read the output file
            with open(temp_file.name, 'r') as f:
                data = json.load(f)
            
            # Check that the metadata section exists
            self.assertIn('metadata', data)
            
            # Check that field_sources is in the metadata
            self.assertIn('field_sources', data['metadata'])
            
            # Check that field_sources is a dictionary
            self.assertIsInstance(data['metadata']['field_sources'], dict)
            
            # Check that some expected fields are present
            field_sources = data['metadata']['field_sources']
            self.assertIn('pipeline_name', field_sources)
            self.assertIn('author', field_sources)
            
            # Check that processing-specific fields are included
            self.assertIn('processing_instance_count', field_sources)
            self.assertIn('output_schema', field_sources)
            
            # Check that sources are tracked correctly
            self.assertEqual(len(field_sources['pipeline_name']), 2)  # Both configs have this field
            self.assertEqual(len(field_sources['processing_instance_count']), 1)  # Only processing config has this
            
            print("Field sources metadata:", json.dumps(field_sources, indent=2))
            
        finally:
            # Clean up the temp file
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)


if __name__ == "__main__":
    unittest.main()
