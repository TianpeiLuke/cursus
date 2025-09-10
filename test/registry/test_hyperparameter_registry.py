"""Pytest tests for the hyperparameter registry module."""

import pytest
from cursus.registry.hyperparameter_registry import (
    HYPERPARAMETER_REGISTRY,
    get_all_hyperparameter_classes,
    get_hyperparameter_class_by_model_type,
    get_module_path,
    get_all_hyperparameter_info,
    validate_hyperparameter_class
)


class TestHyperparameterRegistry:
    """Test case for hyperparameter registry functions."""

    def test_hyperparameter_registry_structure(self):
        """Test that the HYPERPARAMETER_REGISTRY has the expected structure."""
        assert isinstance(HYPERPARAMETER_REGISTRY, dict)
        assert len(HYPERPARAMETER_REGISTRY) > 0
        
        # Check that each entry has the required fields
        for class_name, info in HYPERPARAMETER_REGISTRY.items():
            assert isinstance(class_name, str)
            assert isinstance(info, dict)
            
            required_fields = ["class_name", "module_path", "model_type", "description"]
            for field in required_fields:
                assert field in info, f"Missing field '{field}' in {class_name}"
                
            # Verify class_name matches the key
            assert info["class_name"] == class_name

    def test_get_all_hyperparameter_classes(self):
        """Test get_all_hyperparameter_classes function."""
        classes = get_all_hyperparameter_classes()
        
        assert isinstance(classes, list)
        assert len(classes) > 0
        
        # Should match the keys in the registry
        expected_classes = list(HYPERPARAMETER_REGISTRY.keys())
        assert set(classes) == set(expected_classes)

    def test_get_hyperparameter_class_by_model_type(self):
        """Test get_hyperparameter_class_by_model_type function."""
        # Test with known model types
        xgboost_class = get_hyperparameter_class_by_model_type("xgboost")
        assert xgboost_class == "XGBoostHyperparameters"
        
        pytorch_class = get_hyperparameter_class_by_model_type("pytorch")
        assert pytorch_class == "BSMModelHyperparameters"
        
        # Test with unknown model type
        unknown_class = get_hyperparameter_class_by_model_type("unknown_model")
        assert unknown_class is None
        
        # Test with None model type (base class)
        base_class = get_hyperparameter_class_by_model_type(None)
        assert base_class == "ModelHyperparameters"

    def test_get_module_path(self):
        """Test get_module_path function."""
        # Test with known class names
        for class_name, info in HYPERPARAMETER_REGISTRY.items():
            module_path = get_module_path(class_name)
            assert module_path == info["module_path"]
        
        # Test with unknown class name
        unknown_path = get_module_path("UnknownHyperparameters")
        assert unknown_path is None

    def test_get_all_hyperparameter_info(self):
        """Test get_all_hyperparameter_info function."""
        info = get_all_hyperparameter_info()
        
        assert isinstance(info, dict)
        assert info == HYPERPARAMETER_REGISTRY
        
        # Verify it's a copy, not the original
        assert info is not HYPERPARAMETER_REGISTRY

    def test_validate_hyperparameter_class(self):
        """Test validate_hyperparameter_class function."""
        # Test with valid class names
        for class_name in HYPERPARAMETER_REGISTRY.keys():
            assert validate_hyperparameter_class(class_name)
        
        # Test with invalid class names
        invalid_names = ["InvalidClass", "NonExistentHyperparameters", ""]
        for invalid_name in invalid_names:
            assert not validate_hyperparameter_class(invalid_name)

    def test_registry_contains_expected_classes(self):
        """Test that the registry contains expected hyperparameter classes."""
        expected_classes = [
            "ModelHyperparameters",
            "XGBoostHyperparameters", 
            "BSMModelHyperparameters"
        ]
        
        for expected_class in expected_classes:
            assert expected_class in HYPERPARAMETER_REGISTRY
            assert validate_hyperparameter_class(expected_class)

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
            assert actual_class == expected_class

    def test_module_paths_are_valid_format(self):
        """Test that module paths follow expected format."""
        for class_name, info in HYPERPARAMETER_REGISTRY.items():
            module_path = info["module_path"]
            
            # Should be a string
            assert isinstance(module_path, str)
            
            # Should not be empty
            assert len(module_path) > 0
            
            # Should contain dots (package.module format)
            assert "." in module_path

    def test_descriptions_are_present(self):
        """Test that all entries have non-empty descriptions."""
        for class_name, info in HYPERPARAMETER_REGISTRY.items():
            description = info["description"]
            
            assert isinstance(description, str)
            assert len(description) > 0
            assert description.strip() != ""
