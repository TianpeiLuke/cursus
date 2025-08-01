import unittest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the project root to the Python path to allow for absolute imports
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the config class to be tested
from src.pipeline_steps.config_training_step_xgboost import XGBoostTrainingConfig
from src.pipeline_steps.hyperparameters_xgboost import XGBoostModelHyperparameters


class TestXGBoostTrainingConfig(unittest.TestCase):
    def setUp(self):
        """Set up a minimal, valid configuration for each test."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a minimal valid hyperparameters object
        self.hyperparameters = XGBoostModelHyperparameters(
            full_field_list=['order_id', 'feature1', 'feature2', 'label'],
            tab_field_list=['feature1', 'feature2'],
            cat_field_list=[],
            label_name='label',
            id_name='order_id',
            multiclass_categories=['0', '1'],
            model_class='xgboost',
            num_round=100,
            max_depth=6
        )
        
        # Create a minimal valid config
        self.config_data = {
            "region": "NA",
            "author": "test-author",
            "bucket": "test-bucket",
            "role": "test-role",
            "service_name": "test-service",
            "pipeline_version": "0.1.0",
            "training_entry_point": "train_xgb.py",
            "source_dir": self.temp_dir,
            "training_instance_type": "ml.m5.large",
            "training_instance_count": 1,
            "training_volume_size": 30,
            "framework_version": "1.7-1",
            "py_version": "py3",
            "hyperparameters": self.hyperparameters
        }

    def tearDown(self):
        """Clean up the temporary directory after each test."""
        shutil.rmtree(self.temp_dir)

    def test_init_with_valid_config(self):
        """Test initialization with valid configuration."""
        config = XGBoostTrainingConfig(**self.config_data)
        
        # Verify basic attributes
        self.assertEqual(config.training_entry_point, "train_xgb.py")
        self.assertEqual(config.source_dir, self.temp_dir)
        self.assertEqual(config.training_instance_type, "ml.m5.large")
        self.assertEqual(config.training_instance_count, 1)
        self.assertEqual(config.training_volume_size, 30)
        self.assertEqual(config.framework_version, "1.7-1")
        self.assertEqual(config.py_version, "py3")
        self.assertEqual(config.hyperparameters, self.hyperparameters)
        
        # Verify derived fields
        expected_hyperparameter_file = f"{config.pipeline_s3_loc}/hyperparameters/{config.region}_hyperparameters.json"
        self.assertEqual(config.hyperparameter_file, expected_hyperparameter_file)

    def test_empty_training_entry_point_is_allowed(self):
        """Test that empty training_entry_point is allowed."""
        # Set training_entry_point to empty string
        modified_config = self.config_data.copy()
        modified_config["training_entry_point"] = ""
        
        # Should not raise an error
        config = XGBoostTrainingConfig(**modified_config)
        self.assertEqual(config.training_entry_point, "")

    def test_missing_source_dir_allowed(self):
        """Test that missing source_dir is allowed (it's optional)."""
        # Remove source_dir
        modified_config = self.config_data.copy()
        modified_config.pop("source_dir")
        
        # Should not raise an error
        config = XGBoostTrainingConfig(**modified_config)
        self.assertIsNone(config.source_dir)

    def test_missing_hyperparameters(self):
        """Test that missing hyperparameters raises ValidationError."""
        # Remove hyperparameters
        invalid_config = self.config_data.copy()
        invalid_config.pop("hyperparameters")
        
        # Should raise ValidationError
        with self.assertRaises(ValueError):
            XGBoostTrainingConfig(**invalid_config)

    def test_invalid_training_instance_type(self):
        """Test that invalid training_instance_type raises ValidationError."""
        # Set invalid training_instance_type
        invalid_config = self.config_data.copy()
        invalid_config["training_instance_type"] = "invalid-instance-type"
        
        # Should raise ValidationError
        with self.assertRaises(ValueError):
            XGBoostTrainingConfig(**invalid_config)

    def test_hyperparameter_validation_invalid_field_list(self):
        """Test that invalid field lists in hyperparameters raise ValidationError."""
        # Create hyperparameters with invalid field list
        invalid_hyperparameters = XGBoostModelHyperparameters(
            full_field_list=['order_id', 'feature1', 'feature2', 'label'],
            tab_field_list=['feature1', 'feature2', 'invalid_feature'],  # invalid feature not in full_field_list
            cat_field_list=[],
            label_name='label',
            id_name='order_id',
            multiclass_categories=['0', '1'],
            model_class='xgboost',
            num_round=100,
            max_depth=6
        )
        
        invalid_config = self.config_data.copy()
        invalid_config["hyperparameters"] = invalid_hyperparameters
        
        # Should raise ValidationError
        with self.assertRaises(ValueError):
            XGBoostTrainingConfig(**invalid_config)

    def test_hyperparameter_validation_invalid_label_name(self):
        """Test that invalid label_name in hyperparameters raises ValidationError."""
        # Create hyperparameters with invalid label_name
        invalid_hyperparameters = XGBoostModelHyperparameters(
            full_field_list=['order_id', 'feature1', 'feature2', 'label'],
            tab_field_list=['feature1', 'feature2'],
            cat_field_list=[],
            label_name='invalid_label',  # invalid label not in full_field_list
            id_name='order_id',
            multiclass_categories=['0', '1'],
            model_class='xgboost',
            num_round=100,
            max_depth=6
        )
        
        invalid_config = self.config_data.copy()
        invalid_config["hyperparameters"] = invalid_hyperparameters
        
        # Should raise ValidationError
        with self.assertRaises(ValueError):
            XGBoostTrainingConfig(**invalid_config)

    def test_hyperparameter_validation_invalid_id_name(self):
        """Test that invalid id_name in hyperparameters raises ValidationError."""
        # Create hyperparameters with invalid id_name
        invalid_hyperparameters = XGBoostModelHyperparameters(
            full_field_list=['order_id', 'feature1', 'feature2', 'label'],
            tab_field_list=['feature1', 'feature2'],
            cat_field_list=[],
            label_name='label',
            id_name='invalid_id',  # invalid id not in full_field_list
            multiclass_categories=['0', '1'],
            model_class='xgboost',
            num_round=100,
            max_depth=6
        )
        
        invalid_config = self.config_data.copy()
        invalid_config["hyperparameters"] = invalid_hyperparameters
        
        # Should raise ValidationError
        with self.assertRaises(ValueError):
            XGBoostTrainingConfig(**invalid_config)

    def test_default_values(self):
        """Test that default values are set correctly."""
        # Create a minimal config with only required fields
        minimal_config = {
            "region": "NA",
            "author": "test-author",
            "bucket": "test-bucket",
            "role": "test-role",
            "service_name": "test-service",
            "pipeline_version": "0.1.0",
            "training_entry_point": "train_xgb.py",
            "hyperparameters": self.hyperparameters
        }
        
        config = XGBoostTrainingConfig(**minimal_config)
        
        # Verify default values
        self.assertEqual(config.training_instance_type, "ml.m5.4xlarge")
        self.assertEqual(config.training_instance_count, 1)
        self.assertEqual(config.training_volume_size, 30)
        self.assertEqual(config.framework_version, "1.7-1")
        self.assertEqual(config.py_version, "py3")
        
        # Verify derived fields
        expected_hyperparameter_file = f"{config.pipeline_s3_loc}/hyperparameters/{config.region}_hyperparameters.json"
        self.assertEqual(config.hyperparameter_file, expected_hyperparameter_file)

    def test_to_hyperparameter_dict(self):
        """Test the to_hyperparameter_dict method."""
        config = XGBoostTrainingConfig(**self.config_data)
        hyperparam_dict = config.to_hyperparameter_dict()
        
        # Verify hyperparameter dictionary contains expected values
        # The actual values come from the hyperparameters.serialize_config() method
        # which is already tested in its own test file
        self.assertTrue(isinstance(hyperparam_dict, dict))
        
        # Check a few key values that should be present
        self.assertIn('model_class', hyperparam_dict)
        self.assertIn('num_round', hyperparam_dict)
        self.assertIn('max_depth', hyperparam_dict)
        self.assertIn('objective', hyperparam_dict)

    def test_get_public_init_fields(self):
        """Test the get_public_init_fields method."""
        config = XGBoostTrainingConfig(**self.config_data)
        init_fields = config.get_public_init_fields()
        
        # Verify init_fields contains expected keys
        expected_keys = [
            'author', 'bucket', 'role', 'region', 'service_name', 'pipeline_version',
            'model_class', 'current_date', 'framework_version', 'py_version', 'source_dir',
            'training_entry_point', 'training_instance_type', 'training_instance_count',
            'training_volume_size', 'hyperparameters'
        ]
        
        for key in expected_keys:
            self.assertIn(key, init_fields)
        
        # Verify values are correct
        self.assertEqual(init_fields['training_entry_point'], "train_xgb.py")
        self.assertEqual(init_fields['training_instance_type'], "ml.m5.large")
        self.assertEqual(init_fields['training_instance_count'], 1)
        self.assertEqual(init_fields['training_volume_size'], 30)
        self.assertEqual(init_fields['framework_version'], "1.7-1")
        self.assertEqual(init_fields['py_version'], "py3")
        self.assertEqual(init_fields['hyperparameters'], self.hyperparameters)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
