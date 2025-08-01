"""
Unit tests for TabularPreprocessingConfig three-tier categorization implementation.

These tests verify that the TabularPreprocessingConfig properly categorizes fields
into the three tiers (Essential User Inputs, System Fields, and Derived Fields).
"""

import unittest
import sys
import os
from pathlib import Path

# Add the src directory to the path to allow importing modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline_steps.config_tabular_preprocessing_step import TabularPreprocessingConfig
from src.pipeline_steps.config_base import BasePipelineConfig


class TestTabularPreprocessingTiers(unittest.TestCase):
    """
    Tests for the three-tier field categorization in TabularPreprocessingConfig.
    """
    
    def setUp(self):
        """Set up test environment with a basic configuration."""
        # Use the actual script directory - no need for environment variable to skip validation
        self.script_dir = str(Path(__file__).parent.parent.parent / "src" / "pipeline_scripts")
        
        # Create basic configuration
        self.config = TabularPreprocessingConfig(
            # Base configuration fields
            author="test-author",
            bucket="test-bucket",
            role="test-role",
            region="NA",
            service_name="TestService",
            pipeline_version="1.0.0",
            source_dir=self.script_dir,
            
            # Essential user inputs (Tier 1)
            label_name="is_abuse"
            
            # System fields (Tier 2) use defaults
        )
    
    # No need for tearDown as we're not setting environment variables anymore
    
    def test_field_categorization(self):
        """Test that fields are correctly categorized into tiers."""
        # Get field categories
        categories = self.config.categorize_fields()
        
        # Test Tier 1 (Essential User Inputs)
        self.assertIn('label_name', categories['essential'])
        
        # Test Tier 2 (System Fields with Defaults)
        self.assertIn('processing_entry_point', categories['system'])
        self.assertIn('job_type', categories['system'])
        self.assertIn('train_ratio', categories['system'])
        self.assertIn('test_val_ratio', categories['system'])
        
        # Test Tier 3 (Derived Fields as properties)
        self.assertIn('full_script_path', categories['derived'])
    
    def test_derived_field_initialization(self):
        """Test that derived fields are properly initialized."""
        # Full script path should be constructed from effective source directory
        # and processing_entry_point
        expected_path = str(Path(self.config.effective_source_dir) / self.config.processing_entry_point)
        self.assertEqual(self.config.full_script_path, expected_path)
        
        # The script file should exist
        self.assertTrue(Path(self.config.full_script_path).exists())
        self.assertTrue(Path(self.config.full_script_path).is_file())
        
        # Check correct path to tabular_preprocess.py
        self.assertEqual(
            Path(self.config.full_script_path).name,
            "tabular_preprocess.py"
        )
    
    def test_serialization_includes_derived_fields(self):
        """Test that serialization includes derived fields."""
        data = self.config.model_dump()
        self.assertIn("full_script_path", data)
        self.assertEqual(data["full_script_path"], self.config.full_script_path)
        
    def test_public_init_fields(self):
        """Test that get_public_init_fields includes all necessary fields."""
        fields = self.config.get_public_init_fields()
        
        # Should include essential fields
        self.assertIn('label_name', fields)
        
        # Should include system fields
        self.assertIn('processing_entry_point', fields)
        self.assertIn('job_type', fields)
        self.assertIn('train_ratio', fields)
        self.assertIn('test_val_ratio', fields)
        
        # Should not include derived fields
        self.assertNotIn('full_script_path', fields)
    
    def test_tiers_in_inheritance(self):
        """Test that tiers are maintained when inheriting configuration."""
        # Create a subclass for testing
        class SubTabularConfig(TabularPreprocessingConfig):
            sub_field: str = "test"
        
        # Create instance of subclass
        sub_config = SubTabularConfig(
            # Base configuration fields
            author="sub-author",
            bucket="sub-bucket",
            role="sub-role",
            region="NA",
            service_name="SubService",
            pipeline_version="2.0.0",
            source_dir=self.script_dir,
            
            # Essential user inputs (Tier 1)
            label_name="sub_label"
        )
        
        # Check that tiers are maintained
        categories = sub_config.categorize_fields()
        
        # Tier 1 fields should be properly categorized
        self.assertIn('label_name', categories['essential'])
        
        # Tier 2 fields should be properly categorized
        self.assertIn('processing_entry_point', categories['system'])
        self.assertIn('job_type', categories['system'])
        self.assertIn('train_ratio', categories['system'])
        self.assertIn('test_val_ratio', categories['system'])
        
        # Tier 3 derived fields should be accessible
        self.assertIn('full_script_path', categories['derived'])
        self.assertTrue(sub_config.full_script_path.endswith('tabular_preprocess.py'))

    def test_from_base_config(self):
        """Test creating TabularPreprocessingConfig from a base config."""
        # Create a base config
        base_config = BasePipelineConfig(
            author="base-author",
            bucket="base-bucket",
            role="base-role",
            region="NA",
            service_name="BaseService",
            pipeline_version="1.0.0",
            source_dir=self.script_dir
        )
        
        # Create TabularPreprocessingConfig from base config
        tabular_config = TabularPreprocessingConfig.from_base_config(
            base_config=base_config,
            label_name="base_label",
            job_type="validation",  # Override default
            test_val_ratio=0.6      # Override default
        )
        
        # Verify base fields were inherited
        self.assertEqual(tabular_config.author, "base-author")
        self.assertEqual(tabular_config.bucket, "base-bucket")
        self.assertEqual(tabular_config.role, "base-role")
        self.assertEqual(tabular_config.region, "NA")
        self.assertEqual(tabular_config.service_name, "BaseService")
        self.assertEqual(tabular_config.pipeline_version, "1.0.0")
        self.assertEqual(tabular_config.source_dir, self.script_dir)
        
        # Verify new fields were set
        self.assertEqual(tabular_config.label_name, "base_label")
        self.assertEqual(tabular_config.job_type, "validation")  # Overridden value
        self.assertEqual(tabular_config.test_val_ratio, 0.6)     # Overridden value
        
        # Verify defaults for fields not specified
        self.assertEqual(tabular_config.train_ratio, 0.7)  # Default preserved
        self.assertEqual(tabular_config.processing_entry_point, "tabular_preprocess.py")  # Default preserved
        
        # Verify derived fields were properly initialized
        self.assertTrue(tabular_config.full_script_path.endswith('tabular_preprocess.py'))
        
        # Try creating with different overrides - use the existing script
        # instead of trying to create a custom one that doesn't exist
        custom_tabular_config = TabularPreprocessingConfig.from_base_config(
            base_config=base_config,
            label_name="custom_label",
            job_type="calibration",  # Override default
            train_ratio=0.8
        )
        
        # Verify overrides took effect
        self.assertEqual(custom_tabular_config.label_name, "custom_label")
        self.assertEqual(custom_tabular_config.job_type, "calibration")
        self.assertEqual(custom_tabular_config.train_ratio, 0.8)
        
        # Verify defaults for non-overridden fields
        self.assertEqual(custom_tabular_config.processing_entry_point, "tabular_preprocess.py")  # Default preserved
        self.assertEqual(custom_tabular_config.test_val_ratio, 0.5)   # Default preserved
        
        # Verify derived fields with script path
        self.assertTrue(custom_tabular_config.full_script_path.endswith('tabular_preprocess.py'))


if __name__ == '__main__':
    unittest.main()
