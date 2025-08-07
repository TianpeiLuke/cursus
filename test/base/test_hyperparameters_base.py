import unittest
from unittest.mock import Mock, patch
from typing import Dict, List, Any
import logging
import json

# Add the project root to the Python path to allow for absolute imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.cursus.core.base.hyperparameters_base import ModelHyperparameters


class TestModelHyperparameters(unittest.TestCase):
    """Test cases for ModelHyperparameters class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.valid_hyperparam_data = {
            # Essential User Inputs (Tier 1)
            "full_field_list": ["id", "feature1", "feature2", "category1", "label"],
            "cat_field_list": ["category1"],
            "tab_field_list": ["feature1", "feature2"],
            "id_name": "id",
            "label_name": "label",
            "multiclass_categories": ["class_a", "class_b", "class_c"],
            
            # System Inputs with Defaults (Tier 2) - optional overrides
            "model_class": "xgboost",
            "lr": 0.01,
            "batch_size": 16,
            "max_epochs": 5
        }
    
    def test_init_with_required_fields(self):
        """Test initialization with required fields only."""
        minimal_data = {
            "full_field_list": ["id", "feature1", "label"],
            "cat_field_list": [],
            "tab_field_list": ["feature1"],
            "id_name": "id",
            "label_name": "label",
            "multiclass_categories": ["class_a", "class_b"]
        }
        
        hyperparam = ModelHyperparameters(**minimal_data)
        
        # Verify required fields
        self.assertEqual(hyperparam.full_field_list, ["id", "feature1", "label"])
        self.assertEqual(hyperparam.cat_field_list, [])
        self.assertEqual(hyperparam.tab_field_list, ["feature1"])
        self.assertEqual(hyperparam.id_name, "id")
        self.assertEqual(hyperparam.label_name, "label")
        self.assertEqual(hyperparam.multiclass_categories, ["class_a", "class_b"])
        
        # Verify default values
        self.assertEqual(hyperparam.model_class, "base_model")
        self.assertEqual(hyperparam.device, -1)
        self.assertEqual(hyperparam.lr, 3e-05)
        self.assertEqual(hyperparam.batch_size, 2)
        self.assertEqual(hyperparam.max_epochs, 3)
        self.assertEqual(hyperparam.metric_choices, ['f1_score', 'auroc'])
        self.assertEqual(hyperparam.optimizer, 'SGD')
    
    def test_init_with_all_fields(self):
        """Test initialization with all fields."""
        hyperparam = ModelHyperparameters(**self.valid_hyperparam_data)
        
        # Verify essential fields
        self.assertEqual(hyperparam.full_field_list, ["id", "feature1", "feature2", "category1", "label"])
        self.assertEqual(hyperparam.multiclass_categories, ["class_a", "class_b", "class_c"])
        
        # Verify overridden system fields
        self.assertEqual(hyperparam.model_class, "xgboost")
        self.assertEqual(hyperparam.lr, 0.01)
        self.assertEqual(hyperparam.batch_size, 16)
        self.assertEqual(hyperparam.max_epochs, 5)
    
    def test_derived_properties(self):
        """Test derived properties are calculated correctly."""
        hyperparam = ModelHyperparameters(**self.valid_hyperparam_data)
        
        # Test input_tab_dim (length of tab_field_list)
        self.assertEqual(hyperparam.input_tab_dim, 2)
        
        # Test num_classes (length of multiclass_categories)
        self.assertEqual(hyperparam.num_classes, 3)
        
        # Test is_binary (False for 3 classes)
        self.assertFalse(hyperparam.is_binary)
    
    def test_binary_classification(self):
        """Test binary classification detection."""
        binary_data = self.valid_hyperparam_data.copy()
        binary_data["multiclass_categories"] = ["class_a", "class_b"]
        
        hyperparam = ModelHyperparameters(**binary_data)
        
        self.assertEqual(hyperparam.num_classes, 2)
        self.assertTrue(hyperparam.is_binary)
    
    def test_class_weights_default(self):
        """Test default class weights generation."""
        hyperparam = ModelHyperparameters(**self.valid_hyperparam_data)
        
        # Should default to [1.0] * num_classes
        expected_weights = [1.0, 1.0, 1.0]
        self.assertEqual(hyperparam.class_weights, expected_weights)
    
    def test_class_weights_custom(self):
        """Test custom class weights."""
        data_with_weights = self.valid_hyperparam_data.copy()
        data_with_weights["class_weights"] = [0.5, 1.0, 2.0]
        
        hyperparam = ModelHyperparameters(**data_with_weights)
        
        self.assertEqual(hyperparam.class_weights, [0.5, 1.0, 2.0])
    
    def test_class_weights_validation_error(self):
        """Test class weights validation with wrong length."""
        data_with_wrong_weights = self.valid_hyperparam_data.copy()
        data_with_wrong_weights["class_weights"] = [1.0, 2.0]  # Wrong length for 3 classes
        
        with self.assertRaises(ValueError) as context:
            ModelHyperparameters(**data_with_wrong_weights)
        
        self.assertIn("class_weights length", str(context.exception))
    
    def test_batch_size_validation(self):
        """Test batch size validation."""
        # Test valid batch size
        valid_data = self.valid_hyperparam_data.copy()
        valid_data["batch_size"] = 32
        hyperparam = ModelHyperparameters(**valid_data)
        self.assertEqual(hyperparam.batch_size, 32)
        
        # Test invalid batch size (too large)
        invalid_data = self.valid_hyperparam_data.copy()
        invalid_data["batch_size"] = 300
        
        with self.assertRaises(ValueError):
            ModelHyperparameters(**invalid_data)
        
        # Test invalid batch size (zero)
        invalid_data["batch_size"] = 0
        
        with self.assertRaises(ValueError):
            ModelHyperparameters(**invalid_data)
    
    def test_max_epochs_validation(self):
        """Test max epochs validation."""
        # Test valid max epochs
        valid_data = self.valid_hyperparam_data.copy()
        valid_data["max_epochs"] = 8
        hyperparam = ModelHyperparameters(**valid_data)
        self.assertEqual(hyperparam.max_epochs, 8)
        
        # Test invalid max epochs (too large)
        invalid_data = self.valid_hyperparam_data.copy()
        invalid_data["max_epochs"] = 15
        
        with self.assertRaises(ValueError):
            ModelHyperparameters(**invalid_data)
        
        # Test invalid max epochs (zero)
        invalid_data["max_epochs"] = 0
        
        with self.assertRaises(ValueError):
            ModelHyperparameters(**invalid_data)
    
    def test_categorize_fields(self):
        """Test field categorization."""
        hyperparam = ModelHyperparameters(**self.valid_hyperparam_data)
        categories = hyperparam.categorize_fields()
        
        # Check that all categories exist
        self.assertIn('essential', categories)
        self.assertIn('system', categories)
        self.assertIn('derived', categories)
        
        # Check essential fields (required, no defaults)
        essential_fields = set(categories['essential'])
        expected_essential = {
            'full_field_list', 'cat_field_list', 'tab_field_list',
            'id_name', 'label_name', 'multiclass_categories'
        }
        self.assertEqual(essential_fields, expected_essential)
        
        # Check system fields (have defaults)
        system_fields = set(categories['system'])
        expected_system = {
            'categorical_features_to_encode', 'model_class', 'device', 'header',
            'lr', 'batch_size', 'max_epochs', 'metric_choices', 'optimizer', 'class_weights'
        }
        self.assertEqual(system_fields, expected_system)
        
        # Check derived fields (properties)
        derived_fields = set(categories['derived'])
        expected_derived = {'input_tab_dim', 'num_classes', 'is_binary'}
        self.assertEqual(derived_fields, expected_derived)
    
    def test_get_public_init_fields(self):
        """Test getting public initialization fields."""
        hyperparam = ModelHyperparameters(**self.valid_hyperparam_data)
        init_fields = hyperparam.get_public_init_fields()
        
        # Should include all essential fields
        for field in ['full_field_list', 'cat_field_list', 'tab_field_list', 'id_name', 'label_name', 'multiclass_categories']:
            self.assertIn(field, init_fields)
        
        # Should include non-None system fields
        self.assertIn('model_class', init_fields)
        self.assertIn('lr', init_fields)
        self.assertIn('batch_size', init_fields)
        self.assertIn('max_epochs', init_fields)
        
        # Should not include derived fields
        self.assertNotIn('input_tab_dim', init_fields)
        self.assertNotIn('num_classes', init_fields)
        self.assertNotIn('is_binary', init_fields)
    
    def test_from_base_hyperparam(self):
        """Test creating hyperparameters from base hyperparameters."""
        base_hyperparam = ModelHyperparameters(**self.valid_hyperparam_data)
        
        # Create derived hyperparameters with additional fields
        derived_hyperparam = ModelHyperparameters.from_base_hyperparam(
            base_hyperparam,
            model_class="pytorch",
            lr=0.001
        )
        
        # Should inherit base fields
        self.assertEqual(derived_hyperparam.full_field_list, base_hyperparam.full_field_list)
        self.assertEqual(derived_hyperparam.multiclass_categories, base_hyperparam.multiclass_categories)
        
        # Should override with new values
        self.assertEqual(derived_hyperparam.model_class, "pytorch")
        self.assertEqual(derived_hyperparam.lr, 0.001)
        
        # Should maintain other base values
        self.assertEqual(derived_hyperparam.batch_size, 16)  # From base
    
    def test_get_config(self):
        """Test getting configuration dictionary."""
        hyperparam = ModelHyperparameters(**self.valid_hyperparam_data)
        config = hyperparam.get_config()
        
        # Should be a dictionary
        self.assertIsInstance(config, dict)
        
        # Should contain essential fields
        self.assertIn('full_field_list', config)
        self.assertIn('multiclass_categories', config)
        
        # Should contain system fields
        self.assertIn('model_class', config)
        self.assertIn('lr', config)
    
    def test_serialize_config(self):
        """Test configuration serialization for SageMaker."""
        hyperparam = ModelHyperparameters(**self.valid_hyperparam_data)
        serialized = hyperparam.serialize_config()
        
        # Should be a dictionary with string values
        self.assertIsInstance(serialized, dict)
        for key, value in serialized.items():
            self.assertIsInstance(value, str)
        
        # Should include derived fields
        self.assertIn('input_tab_dim', serialized)
        self.assertIn('num_classes', serialized)
        self.assertIn('is_binary', serialized)
        
        # Test deserialization of complex types
        self.assertEqual(json.loads(serialized['full_field_list']), hyperparam.full_field_list)
        self.assertEqual(json.loads(serialized['multiclass_categories']), hyperparam.multiclass_categories)
        self.assertEqual(json.loads(serialized['is_binary']), hyperparam.is_binary)
    
    def test_string_representation(self):
        """Test string representation."""
        hyperparam = ModelHyperparameters(**self.valid_hyperparam_data)
        str_repr = str(hyperparam)
        
        # Should contain class name
        self.assertIn('ModelHyperparameters', str_repr)
        
        # Should contain field categories
        self.assertIn('Essential User Inputs', str_repr)
        self.assertIn('System Inputs', str_repr)
        self.assertIn('Derived Fields', str_repr)
        
        # Should contain some field values
        self.assertIn('xgboost', str_repr)
        self.assertIn('0.01', str_repr)
    
    def test_print_hyperparam_method(self):
        """Test print_hyperparam method."""
        hyperparam = ModelHyperparameters(**self.valid_hyperparam_data)
        
        # Should not raise any exceptions
        try:
            hyperparam.print_hyperparam()
        except Exception as e:
            self.fail(f"print_hyperparam raised an exception: {e}")
    
    def test_derived_fields_caching(self):
        """Test that derived fields are cached."""
        hyperparam = ModelHyperparameters(**self.valid_hyperparam_data)
        
        # Access derived property multiple times
        first_access = hyperparam.input_tab_dim
        second_access = hyperparam.input_tab_dim
        
        # Should return the same value (testing caching behavior)
        self.assertEqual(first_access, second_access)
        self.assertEqual(first_access, 2)
        
        # Check that private attribute is set
        self.assertEqual(hyperparam._input_tab_dim, 2)
    
    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed."""
        data_with_extra = self.valid_hyperparam_data.copy()
        data_with_extra['extra_field'] = 'extra_value'
        
        # Should not raise an exception
        hyperparam = ModelHyperparameters(**data_with_extra)
        
        # Extra field should be accessible
        self.assertEqual(hyperparam.extra_field, 'extra_value')
    
    def test_validate_assignment(self):
        """Test that validate_assignment works."""
        hyperparam = ModelHyperparameters(**self.valid_hyperparam_data)
        
        # Should be able to modify fields
        hyperparam.lr = 0.001
        self.assertEqual(hyperparam.lr, 0.001)
        
        # Should validate on assignment
        with self.assertRaises(ValueError):
            hyperparam.batch_size = 0  # Invalid batch size
    
    def test_empty_field_lists(self):
        """Test with empty field lists."""
        data_with_empty_lists = {
            "full_field_list": ["id", "label"],
            "cat_field_list": [],
            "tab_field_list": [],
            "id_name": "id",
            "label_name": "label",
            "multiclass_categories": ["class_a", "class_b"]
        }
        
        hyperparam = ModelHyperparameters(**data_with_empty_lists)
        
        # Should handle empty lists correctly
        self.assertEqual(hyperparam.input_tab_dim, 0)
        self.assertEqual(hyperparam.num_classes, 2)
        self.assertTrue(hyperparam.is_binary)
    
    def test_single_class_error(self):
        """Test that single class raises appropriate error if needed."""
        single_class_data = self.valid_hyperparam_data.copy()
        single_class_data["multiclass_categories"] = ["single_class"]
        
        hyperparam = ModelHyperparameters(**single_class_data)
        
        # Should work but not be binary
        self.assertEqual(hyperparam.num_classes, 1)
        self.assertFalse(hyperparam.is_binary)


if __name__ == '__main__':
    # Set up logging for tests
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
