import unittest
import json
import os
from pathlib import Path
import tempfile
import sys
from typing import Any, Dict, List, Optional, Union, Set

# Add the repository root directory to the path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, repo_root)

import pytest
from unittest.mock import patch, MagicMock, Mock

# Import utilities for config serialization
from src.pipeline_steps.utils import merge_and_save_configs, load_configs, serialize_config
from src.config_field_manager import ConfigClassStore, build_complete_config_classes
from src.pipeline_steps.config_base import BasePipelineConfig
from src.pipeline_steps.hyperparameters_base import ModelHyperparameters

# Try to import the BSM hyperparameters class if available
try:
    from src.pipeline_steps.hyperparameters_bsm import BSMModelHyperparameters
    BSM_AVAILABLE = True
except ImportError:
    BSM_AVAILABLE = False
    # Create a mock class for testing if the real one is not available
    class BSMModelHyperparameters(ModelHyperparameters):
        lr_decay: float = 0.05
        adam_epsilon: float = 1e-08
        text_name: str = "dialogue"
        
# Import config_merger for saving configs
from src.config_field_manager.config_merger import ConfigMerger
from src.config_field_manager.type_aware_config_serializer import TypeAwareConfigSerializer

# Define simple test config classes for serialization testing
class TestBaseConfig(BasePipelineConfig):
    """Simple test config class that inherits from BasePipelineConfig."""
    pipeline_name: str
    pipeline_description: str
    pipeline_version: str
    bucket: str
    model_path: str = "default_model_path"  # Required field from validation
    hyperparameters: ModelHyperparameters

    # Optional fields with default values
    author: str = "test-author"
    job_type: Optional[str] = None
    step_name_override: Optional[str] = None
    
    def validate_config(self) -> Dict[str, Any]:
        """Basic validation function."""
        errors = {}
        required_fields = ["pipeline_name", "pipeline_description", "pipeline_version", 
                          "bucket", "model_path"]
                          
        for field in required_fields:
            if not hasattr(self, field) or getattr(self, field) is None:
                errors[field] = f"Field {field} is required"
                
        return errors

# Test config with specific job types, similar to Tabular preprocessing step
class TestProcessingConfig(TestBaseConfig):
    """Processing-specific config for testing."""
    input_path: str = "default_input_path"
    output_path: str = "default_output_path"
    # Add job_type field explicitly matching the tabular preprocessing step
    job_type: str = "tabular"  # Default job_type
    data_type: Optional[str] = None
    feature_columns: List[str] = []
    target_column: Optional[str] = None
    
    # Options for different preprocessing steps
    normalize_features: bool = False
    encoding_method: str = "one_hot"
    handle_missing: str = "median"
    
    def validate_config(self) -> Dict[str, Any]:
        """Extended validation for processing configs."""
        errors = super().validate_config()
        
        processing_required = ["input_path", "output_path", "job_type"]
        for field in processing_required:
            if not hasattr(self, field) or getattr(self, field) is None:
                errors[field] = f"Field {field} is required for processing"
                
        return errors

# Test config with training-specific fields
class TestTrainingConfig(TestBaseConfig):
    """Training-specific config for testing."""
    training_data_path: str = "default_training_data_path"
    validation_data_path: Optional[str] = None
    epochs: int = 10
    
    def validate_config(self) -> Dict[str, Any]:
        """Extended validation for training configs."""
        errors = super().validate_config()
        
        training_required = ["training_data_path", "epochs"]
        for field in training_required:
            if not hasattr(self, field) or getattr(self, field) is None:
                errors[field] = f"Field {field} is required for training"
                
        return errors


class TestTypeAwareDeserialization(unittest.TestCase):
    """
    Tests for the type-aware model serialization and deserialization.
    This tests the ability to correctly serialize and deserialize derived model classes.
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
        
        # Create a base hyperparameters object
        self.base_hyperparams = ModelHyperparameters(
            full_field_list=["field1", "field2", "field3"],
            cat_field_list=["field3"],
            tab_field_list=["field1", "field2"],
            input_tab_dim=2,
            is_binary=True,
            num_classes=2,
            multiclass_categories=[0, 1],
            class_weights=[1.0, 2.0]
        )
        
        # Skip BSM tests if the class is not available
        if not BSM_AVAILABLE:
            self.skipTest("BSMModelHyperparameters not available")
            
        # Create a derived BSM hyperparameters object with additional fields
        self.bsm_hyperparams = BSMModelHyperparameters(
            full_field_list=["field1", "field2", "field3"],
            cat_field_list=["field3"],
            tab_field_list=["field1", "field2"],
            input_tab_dim=2,
            is_binary=True,
            num_classes=2,
            multiclass_categories=[0, 1],
            class_weights=[1.0, 2.0],
            # BSM-specific fields
            lr_decay=0.05,
            adam_epsilon=1e-08,
            text_name="dialogue",
            chunk_trancate=True,
            max_total_chunks=3,
            tokenizer="bert-base-multilingual-uncased",
            max_sen_len=512
        )
        
        # Create test config objects with different hyperparameters types
        self.processing_config = TestProcessingConfig(
            pipeline_name="test-processing-pipeline",
            pipeline_description="Test Processing Pipeline",
            pipeline_version="1.0.0",
            bucket="test-bucket",
            hyperparameters=self.base_hyperparams,
            job_type="processing"
        )
        
        self.processing_config_raw = TestProcessingConfig(
            pipeline_name="test-processing-pipeline-raw",
            pipeline_description="Test Processing Pipeline Raw",
            pipeline_version="1.0.0",
            bucket="test-bucket",
            hyperparameters=self.base_hyperparams,
            job_type="raw"
        )
        
        self.training_config = TestTrainingConfig(
            pipeline_name="test-training-pipeline",
            pipeline_description="Test Training Pipeline", 
            pipeline_version="1.0.0",
            bucket="test-bucket",
            hyperparameters=self.base_hyperparams,
            job_type="training"
        )
        
        self.bsm_training_config = TestTrainingConfig(
            pipeline_name="test-bsm-pipeline",
            pipeline_description="Test BSM Pipeline",
            pipeline_version="1.0.0",
            bucket="test-bucket",
            hyperparameters=self.bsm_hyperparams,
            job_type="bsm"
        )
        
        self.override_config = TestProcessingConfig(
            pipeline_name="test-override-pipeline",
            pipeline_description="Test Override Pipeline",
            pipeline_version="1.0.0", 
            bucket="test-bucket",
            hyperparameters=self.base_hyperparams,
            job_type="custom",
            step_name_override="CustomStepName"
        )
        
        # Register our custom classes
        self.config_classes = build_complete_config_classes()
        self.config_classes.update({
            'TestBaseConfig': TestBaseConfig,
            'TestProcessingConfig': TestProcessingConfig,
            'TestTrainingConfig': TestTrainingConfig
        })
        
    def test_type_preservation(self):
        """Test that derived class types are preserved during serialization and deserialization."""
        # Use the serializer directly to test round-trip serialization
        from src.config_field_manager.type_aware_config_serializer import TypeAwareConfigSerializer, deserialize_config
        
        # Create serializer with complete config classes
        config_classes = build_complete_config_classes()
        serializer = TypeAwareConfigSerializer(config_classes=config_classes)
        
        # Test BSM hyperparameters serialization and deserialization
        serialized_bsm = serializer.serialize(self.bsm_hyperparams)
        print("BSM serialized:", serialized_bsm)
        self.assertIn('__model_type__', serialized_bsm)
        self.assertEqual(serialized_bsm['__model_type__'], 'BSMModelHyperparameters')
        
        # Deserialize back
        deserialized_bsm = serializer.deserialize(serialized_bsm)
        self.assertIsInstance(deserialized_bsm, BSMModelHyperparameters)
        self.assertTrue(hasattr(deserialized_bsm, 'lr_decay'))
        self.assertEqual(deserialized_bsm.lr_decay, 0.05)
        
        # Test BSM-specific fields
        self.assertTrue(hasattr(deserialized_bsm, 'text_name'))
        self.assertEqual(deserialized_bsm.text_name, 'dialogue')
        
        # Test that base hyperparameters class doesn't have BSM-specific fields
        serialized_base = serializer.serialize(self.base_hyperparams)
        deserialized_base = serializer.deserialize(serialized_base)
        self.assertIsInstance(deserialized_base, ModelHyperparameters)
        self.assertFalse(hasattr(deserialized_base, 'lr_decay'))
                
    def test_type_metadata_in_serialized_output(self):
        """Test that type metadata is included in the serialized output."""
        # Create a serializer and use it directly
        from src.config_field_manager.type_aware_config_serializer import TypeAwareConfigSerializer
        serializer = TypeAwareConfigSerializer()
        
        # Serialize a BSM hyperparameters object
        serialized = serializer.serialize(self.bsm_hyperparams)
        
        # Verify type metadata fields are present
        self.assertIn('__model_type__', serialized, "Type metadata field missing")
        self.assertEqual(serialized['__model_type__'], 'BSMModelHyperparameters', 
                        "Type metadata has incorrect value")
                        
        self.assertIn('__model_module__', serialized, "Module metadata field missing")
        self.assertTrue(serialized['__model_module__'].endswith('hyperparameters_bsm'), 
                       "Module metadata has incorrect value")
                       
        # Verify BSM-specific fields are present
        self.assertIn('lr_decay', serialized, "BSM-specific field missing in serialized output")
        self.assertEqual(serialized['lr_decay'], 0.05, "BSM-specific field has incorrect value")
                
    def test_config_types_format(self):
        """Test that config_types uses step names as keys when saved to file."""
        # Skip if we can't work with a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            try:
                # Create multiple configs with different job types
                config1 = TestTrainingConfig(
                    bucket="test-bucket",
                    author="test-author",
                    pipeline_name="test-pipeline-1",
                    pipeline_description="Test Pipeline 1",
                    pipeline_version="1.0.0",
                    hyperparameters=self.base_hyperparams,
                    job_type="training"
                )
                
                config2 = TestTrainingConfig(
                    bucket="test-bucket",
                    author="test-author",
                    pipeline_name="test-pipeline-2",
                    pipeline_description="Test Pipeline 2",
                    pipeline_version="1.0.0",
                    hyperparameters=self.base_hyperparams,
                    job_type="evaluation"
                )
                
                # Add our custom classes to the registry for merge_and_save_configs
                config_classes = {
                    'TestBaseConfig': TestBaseConfig,
                    'TestProcessingConfig': TestProcessingConfig,
                    'TestTrainingConfig': TestTrainingConfig
                }
                
                # Save configs to temporary file
                merger = ConfigMerger([config1, config2])
                merger.save(tmp.name)
                
                # Read the saved file directly to check the format
                with open(tmp.name, 'r') as f:
                    saved_data = json.load(f)
                
                # Verify the structure of config_types
                self.assertIn('metadata', saved_data)
                self.assertIn('config_types', saved_data['metadata'])
                
                config_types = saved_data['metadata']['config_types']
                
                # Keys should be step names with job types, not class names
                self.assertIn('TestTraining_training', config_types)
                self.assertIn('TestTraining_evaluation', config_types)
                
                # Values should be class names
                self.assertEqual(config_types['TestTraining_training'], 'TestTrainingConfig')
                self.assertEqual(config_types['TestTraining_evaluation'], 'TestTrainingConfig')
                
                # Load the configs back with our custom registry
                serializer = TypeAwareConfigSerializer(config_classes=config_classes)
                loaded_data = json.loads(json.dumps(saved_data))  # Deep copy
                
                # Get the specific configs section
                specific = loaded_data["configuration"]["specific"]
                
                # Verify the structure
                self.assertIn('TestTraining_training', specific)
                self.assertIn('TestTraining_evaluation', specific)
                
                # Verify the loaded data has the correct job types
                self.assertEqual(specific['TestTraining_training']['job_type'], 'training')
                self.assertEqual(specific['TestTraining_evaluation']['job_type'], 'evaluation')
                
            finally:
                # Clean up the temporary file
                os.unlink(tmp.name)
    
    def test_custom_config_with_hyperparameters(self):
        """Test serialization of custom config classes with different hyperparameters types."""
        # Create a serializer with our test classes
        from src.config_field_manager.type_aware_config_serializer import TypeAwareConfigSerializer
        
        # Add our custom classes to the registry
        config_classes = build_complete_config_classes()
        config_classes.update({
            'TestBaseConfig': TestBaseConfig,
            'TestProcessingConfig': TestProcessingConfig,
            'TestTrainingConfig': TestTrainingConfig
        })
        serializer = TypeAwareConfigSerializer(config_classes=config_classes)
        
        # Create test configs with different hyperparameters types
        basic_config = TestTrainingConfig(
            pipeline_name="test-basic-pipeline",
            pipeline_description="Test Basic Pipeline",
            pipeline_version="1.0.0",
            bucket="test-bucket",
            hyperparameters=self.base_hyperparams
        )
        
        bsm_config = TestTrainingConfig(
            pipeline_name="test-bsm-pipeline",
            pipeline_description="Test BSM Pipeline",
            pipeline_version="1.0.0",
            bucket="test-bucket",
            hyperparameters=self.bsm_hyperparams,
            job_type="bsm"
        )
        
        # Test serialization of basic config
        serialized_basic = serializer.serialize(basic_config)
        self.assertIn('hyperparameters', serialized_basic)
        self.assertIn('__model_type__', serialized_basic['hyperparameters'])
        self.assertEqual(serialized_basic['hyperparameters']['__model_type__'], 'ModelHyperparameters')
        
        # Test serialization of BSM config
        serialized_bsm = serializer.serialize(bsm_config)
        self.assertIn('hyperparameters', serialized_bsm)
        self.assertIn('__model_type__', serialized_bsm['hyperparameters'])
        self.assertEqual(serialized_bsm['hyperparameters']['__model_type__'], 'BSMModelHyperparameters')
        self.assertIn('lr_decay', serialized_bsm['hyperparameters'])
        self.assertEqual(serialized_bsm['hyperparameters']['lr_decay'], 0.05)
        
        # Test round-trip serialization/deserialization
        deserialized_basic = serializer.deserialize(serialized_basic)
        self.assertIsInstance(deserialized_basic, TestTrainingConfig)
        self.assertIsInstance(deserialized_basic.hyperparameters, ModelHyperparameters)
        self.assertFalse(hasattr(deserialized_basic.hyperparameters, 'lr_decay'))
        
        deserialized_bsm = serializer.deserialize(serialized_bsm)
        self.assertIsInstance(deserialized_bsm, TestTrainingConfig)
        self.assertIsInstance(deserialized_bsm.hyperparameters, BSMModelHyperparameters)
        self.assertTrue(hasattr(deserialized_bsm.hyperparameters, 'lr_decay'))
        self.assertEqual(deserialized_bsm.hyperparameters.lr_decay, 0.05)
        
    def test_config_types_format_with_custom_configs(self):
        """Test that config_types uses step names as keys when using custom configs."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            try:
                # Create configs with different job types
                processing_config = TestProcessingConfig(
                    pipeline_name="test-processing-pipeline",
                    pipeline_description="Test Processing Pipeline",
                    pipeline_version="1.0.0",
                    bucket="test-bucket",
                    hyperparameters=self.base_hyperparams,
                    job_type="processing"
                )
                
                training_config = TestTrainingConfig(
                    pipeline_name="test-training-pipeline",
                    pipeline_description="Test Training Pipeline", 
                    pipeline_version="1.0.0",
                    bucket="test-bucket",
                    hyperparameters=self.bsm_hyperparams,
                    job_type="training"
                )
                
                # Create a config with a step_name_override
                override_config = TestProcessingConfig(
                    pipeline_name="test-override-pipeline",
                    pipeline_description="Test Override Pipeline",
                    pipeline_version="1.0.0", 
                    bucket="test-bucket",
                    hyperparameters=self.base_hyperparams,
                    job_type="custom",
                    step_name_override="CustomStepName"
                )
                
                # Add our custom classes to the registry for merge_and_save_configs
                config_classes = build_complete_config_classes()
                config_classes.update({
                    'TestBaseConfig': TestBaseConfig,
                    'TestProcessingConfig': TestProcessingConfig,
                    'TestTrainingConfig': TestTrainingConfig
                })
                
                # Save configs to temporary file
                from src.config_field_manager.config_merger import ConfigMerger
                merger = ConfigMerger([processing_config, training_config, override_config])
                merger.save(tmp.name)
                
                # Read the saved file to check format
                with open(tmp.name, 'r') as f:
                    saved_data = json.load(f)
                
                # Verify config_types structure
                self.assertIn('metadata', saved_data)
                self.assertIn('config_types', saved_data['metadata'])
                
                config_types = saved_data['metadata']['config_types']
                print("Generated config_types:", config_types)
                
                # Keys should be step names (with job_type variants)
                self.assertIn('TestProcessing_processing', config_types)
                self.assertIn('TestTraining_training', config_types)
                self.assertIn('CustomStepName', config_types)  # Using step_name_override
                
                # Values should be class names
                self.assertEqual(config_types['TestProcessing_processing'], 'TestProcessingConfig')
                self.assertEqual(config_types['TestTraining_training'], 'TestTrainingConfig')
                self.assertEqual(config_types['CustomStepName'], 'TestProcessingConfig')
                
            finally:
                # Clean up
                os.unlink(tmp.name)
                
    def test_multiple_config_scenarios(self):
        """Test serialization and deserialization with multiple config scenarios.
        
        This test covers:
        1. Multiple different types of configs
        2. Two same class of configs with different job_type
        3. Multiple configs, one has a complex field as hyperparameters
        4. Multiple different types of configs, but some fields rely on default values
        """
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            try:
                # Create test configs for different scenarios
                
                # Scenario 1: Different types of configs (Processing vs Training)
                # Scenario 3: One config with complex BSM hyperparameters
                configs = [
                    # Processing config with basic hyperparameters
                    TestProcessingConfig(
                        pipeline_name="processing-pipeline",
                        pipeline_description="Processing Pipeline",
                        pipeline_version="1.0.0",
                        bucket="test-bucket",
                        hyperparameters=self.base_hyperparams,
                        job_type="processing",
                        feature_columns=["feature1", "feature2"],
                        target_column="target"
                    ),
                    
                    # Training config with BSM hyperparameters (complex field)
                    TestTrainingConfig(
                        pipeline_name="training-pipeline",
                        pipeline_description="Training Pipeline",
                        pipeline_version="1.0.0",
                        bucket="test-bucket", 
                        hyperparameters=self.bsm_hyperparams,  # Complex field
                        job_type="training",
                        epochs=20,
                        validation_data_path="/path/to/validation"
                    )
                ]
                
                # Scenario 2: Same class with different job_type
                configs.append(
                    TestProcessingConfig(
                        pipeline_name="processing-pipeline-raw",
                        pipeline_description="Processing Pipeline Raw",
                        pipeline_version="1.0.0",
                        bucket="test-bucket",
                        hyperparameters=self.base_hyperparams,
                        job_type="raw"  # Different job_type from first processing config
                    )
                )
                
                # Scenario 4: Config with fields using default values
                configs.append(
                    TestProcessingConfig(
                        pipeline_name="processing-pipeline-defaults",
                        pipeline_description="Processing Pipeline With Defaults",
                        pipeline_version="1.0.0",
                        bucket="test-bucket",
                        hyperparameters=self.base_hyperparams
                        # Not specifying job_type, input_path, output_path - using defaults
                    )
                )
                
                # Save all configs to a file
                merger = ConfigMerger(configs)
                merger.save(tmp.name)
                
                # Read the saved file and check structure
                with open(tmp.name, 'r') as f:
                    saved_data = json.load(f)
                
                # Verify the structure of config_types
                self.assertIn('metadata', saved_data)
                self.assertIn('config_types', saved_data['metadata'])
                
                config_types = saved_data['metadata']['config_types']
                print("Generated config_types for multiple scenarios:", config_types)
                
                # Verify step names are correctly generated with job_types
                self.assertIn('TestProcessing_processing', config_types)
                self.assertIn('TestProcessing_raw', config_types)
                self.assertIn('TestProcessing_tabular', config_types)  # Using default job_type
                self.assertIn('TestTraining_training', config_types)
                
                # Verify class names are preserved
                self.assertEqual(config_types['TestProcessing_processing'], 'TestProcessingConfig')
                self.assertEqual(config_types['TestProcessing_raw'], 'TestProcessingConfig')
                self.assertEqual(config_types['TestProcessing_tabular'], 'TestProcessingConfig')
                self.assertEqual(config_types['TestTraining_training'], 'TestTrainingConfig')
                
                # Verify configuration structure
                self.assertIn('configuration', saved_data)
                self.assertIn('shared', saved_data['configuration'])
                self.assertIn('specific', saved_data['configuration'])
                
                # Check for fields using default values in the tabular processing config
                specific = saved_data['configuration']['specific']
                self.assertIn('TestProcessing_tabular', specific)
                defaults_config = specific['TestProcessing_tabular']
                
                # Verify default fields are present
                self.assertEqual(defaults_config.get('job_type'), 'tabular')
                self.assertEqual(defaults_config.get('input_path'), 'default_input_path')
                self.assertEqual(defaults_config.get('output_path'), 'default_output_path')
                
            finally:
                # Clean up
                os.unlink(tmp.name)
    
    def test_fallback_behavior(self):
        """Test the fallback behavior when a derived class is not available."""
        # Use the serializer directly to test fallback behavior
        from src.config_field_manager.type_aware_config_serializer import TypeAwareConfigSerializer
        
        # Create serializer with limited config classes (no BSMModelHyperparameters)
        limited_config_classes = {
            'BasePipelineConfig': BasePipelineConfig,
            'TestBaseConfig': TestBaseConfig,
            'TestProcessingConfig': TestProcessingConfig, 
            'TestTrainingConfig': TestTrainingConfig,
            'ModelHyperparameters': ModelHyperparameters
            # BSMModelHyperparameters intentionally omitted
        }
        
        # Create serializer with limited classes
        serializer = TypeAwareConfigSerializer(config_classes=limited_config_classes)
        
        # Test direct serialization and deserialization of BSM hyperparameters
        serialized_bsm = serializer.serialize(self.bsm_hyperparams)
        self.assertIn('__model_type__', serialized_bsm)
        self.assertEqual(serialized_bsm['__model_type__'], 'BSMModelHyperparameters')
        
        # Deserialize with limited class registry - should fallback to ModelHyperparameters
        deserialized_bsm = serializer.deserialize(serialized_bsm)
        self.assertIsInstance(deserialized_bsm, ModelHyperparameters, 
                             "Should fallback to ModelHyperparameters")
        
        # Verify base fields are still present
        self.assertTrue(hasattr(deserialized_bsm, 'full_field_list'))
        self.assertListEqual(deserialized_bsm.full_field_list, ["field1", "field2", "field3"])
                

if __name__ == "__main__":
    unittest.main()
