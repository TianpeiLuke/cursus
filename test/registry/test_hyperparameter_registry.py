"""Unit tests for the hyperparameter registry module."""

import unittest
from src.cursus.registry.hyperparameter_registry import (
    HYPERPARAMETER_REGISTRY,
    get_all_hyperparameter_classes,
    get_hyperparameter_class_by_model_type,
    get_module_path,
    get_all_hyperparameter_info,
    validate_hyperparameter_class
)


class TestHyperparameterRegistry(unittest.TestCase):
    """Test case for hyperparameter registry functions."""

    def test_hyperparameter_registry_structure(self):
        """Test that the HYPERPARAMETER_REGISTRY has the expected structure."""
        self.assertIsInstance(HYPERPARAMETER_REGISTRY, dict)
        self.assertGreater(len(HYPERPARAMETER_REGISTRY), 0)
        
        # Check that each entry has the required fields
        for class_name, info in HYPERPARAMETER_REGISTRY.items():
            self.assertIsInstance(class_name, str)
            self.assertIsInstance(info, dict)
            
            required_fields = ["class_name", "module_path", "model_type", "description"]
            for field in required_fields:
                self.assertIn(field, info, f"Missing field '{field}' in {class_name}")
                
            # Verify class_name matches the key
            self.assertEqual(info["class_name"], class_name)

    def test_get_all_hyperparameter_classes(self):
        """Test get_all_hyperparameter_classes function."""
        classes = get_all_hyperparameter_classes()
        
        self.assertIsInstance(classes, list)
        self.assertGreater(len(classes), 0)
        
        # Should match the keys in the registry
        expected_classes = list(HYPERPARAMETER_REGISTRY.keys())
        self.assertEqual(set(classes), set(expected_classes))

    def test_get_hyperparameter_class_by_model_type(self):
        """Test get_hyperparameter_class_by_model_type function."""
        # Test with known model types
        xgboost_class = get_hyperparameter_class_by_model_type("xgboost")
        self.assertEqual(xgboost_class, "XGBoostHyperparameters")
        
        pytorch_class = get_hyperparameter_class_by_model_type("pytorch")
        self.assertEqual(pytorch_class, "BSMModelHyperparameters")
        
        # Test with unknown model type
        unknown_class = get_hyperparameter_class_by_model_type("unknown_model")
        self.assertIsNone(unknown_class)
        
        # Test with None model type (base class)
        base_class = get_hyperparameter_class_by_model_type(None)
        self.assertEqual(base_class, "ModelHyperparameters")

    def test_get_module_path(self):
        """Test get_module_path function."""
        # Test with known class names
        for class_name, info in HYPERPARAMETER_REGISTRY.items():
            module_path = get_module_path(class_name)
            self.assertEqual(module_path, info["module_path"])
        
        # Test with unknown class name
        unknown_path = get_module_path("UnknownHyperparameters")
        self.assertIsNone(unknown_path)

    def test_get_all_hyperparameter_info(self):
        """Test get_all_hyperparameter_info function."""
        info = get_all_hyperparameter_info()
        
        self.assertIsInstance(info, dict)
        self.assertEqual(info, HYPERPARAMETER_REGISTRY)
        
        # Verify it's a copy, not the original
        self.assertIsNot(info, HYPERPARAMETER_REGISTRY)

    def test_validate_hyperparameter_class(self):
        """Test validate_hyperparameter_class function."""
        # Test with valid class names
        for class_name in HYPERPARAMETER_REGISTRY.keys():
            self.assertTrue(validate_hyperparameter_class(class_name))
        
        # Test with invalid class names
        invalid_names = ["InvalidClass", "NonExistentHyperparameters", ""]
        for invalid_name in invalid_names:
            self.assertFalse(validate_hyperparameter_class(invalid_name))

    def test_registry_contains_expected_classes(self):
        """Test that the registry contains expected hyperparameter classes."""
        expected_classes = [
            "ModelHyperparameters",
            "XGBoostHyperparameters", 
            "BSMModelHyperparameters"
        ]
        
        for expected_class in expected_classes:
            self.assertIn(expected_class, HYPERPARAMETER_REGISTRY)
            self.assertTrue(validate_hyperparameter_class(expected_class))

    def test_model_type_mappings(self):
        """Test that model type mappings are correct."""
        # Test specific model type mappings
        model_type_mappings = {
            "xgboost": "XGBoostHyperparameters",
            "pytorch": "BSMModelHyperparameters",
            None: "ModelHyperparameters"  # Base class
        }
        
        for model_type, expected_class in model_type_mappings.items():
            actual_class = get_hyperparameter_class_by_model_type(model_type)
            self.assertEqual(actual_class, expected_class)

    def test_module_paths_are_valid_format(self):
        """Test that module paths follow expected format."""
        for class_name, info in HYPERPARAMETER_REGISTRY.items():
            module_path = info["module_path"]
            
            # Should be a string
            self.assertIsInstance(module_path, str)
            
            # Should not be empty
            self.assertGreater(len(module_path), 0)
            
            # Should contain dots (package.module format)
            self.assertIn(".", module_path)

    def test_descriptions_are_present(self):
        """Test that all entries have non-empty descriptions."""
        for class_name, info in HYPERPARAMETER_REGISTRY.items():
            description = info["description"]
            
            self.assertIsInstance(description, str)
            self.assertGreater(len(description), 0)
            self.assertNotEqual(description.strip(), "")


if __name__ == '__main__':
    unittest.main()
