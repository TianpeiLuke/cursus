import pytest
from unittest.mock import Mock, patch
from typing import Dict, List, Any
import logging
import json

from cursus.core.base.hyperparameters_base import ModelHyperparameters


class TestModelHyperparameters:
    """Test cases for ModelHyperparameters class."""

    @pytest.fixture
    def valid_hyperparam_data(self):
        """Set up test fixtures."""
        return {
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
            "max_epochs": 5,
        }

    def test_init_with_required_fields(self):
        """Test initialization with required fields only."""
        minimal_data = {
            "full_field_list": ["id", "feature1", "label"],
            "cat_field_list": [],
            "tab_field_list": ["feature1"],
            "id_name": "id",
            "label_name": "label",
            "multiclass_categories": ["class_a", "class_b"],
        }

        hyperparam = ModelHyperparameters(**minimal_data)

        # Verify required fields
        assert hyperparam.full_field_list == ["id", "feature1", "label"]
        assert hyperparam.cat_field_list == []
        assert hyperparam.tab_field_list == ["feature1"]
        assert hyperparam.id_name == "id"
        assert hyperparam.label_name == "label"
        assert hyperparam.multiclass_categories == ["class_a", "class_b"]

        # Verify default values
        assert hyperparam.model_class == "base_model"
        assert hyperparam.device == -1
        assert hyperparam.lr == 3e-05
        assert hyperparam.batch_size == 2
        assert hyperparam.max_epochs == 3
        assert hyperparam.metric_choices == ["f1_score", "auroc"]
        assert hyperparam.optimizer == "SGD"

    def test_init_with_all_fields(self, valid_hyperparam_data):
        """Test initialization with all fields."""
        hyperparam = ModelHyperparameters(**valid_hyperparam_data)

        # Verify essential fields
        assert hyperparam.full_field_list == [
            "id",
            "feature1",
            "feature2",
            "category1",
            "label",
        ]
        assert hyperparam.multiclass_categories == ["class_a", "class_b", "class_c"]

        # Verify overridden system fields
        assert hyperparam.model_class == "xgboost"
        assert hyperparam.lr == 0.01
        assert hyperparam.batch_size == 16
        assert hyperparam.max_epochs == 5

    def test_derived_properties(self, valid_hyperparam_data):
        """Test derived properties are calculated correctly."""
        hyperparam = ModelHyperparameters(**valid_hyperparam_data)

        # Test input_tab_dim (length of tab_field_list)
        assert hyperparam.input_tab_dim == 2

        # Test num_classes (length of multiclass_categories)
        assert hyperparam.num_classes == 3

        # Test is_binary (False for 3 classes)
        assert not hyperparam.is_binary

    def test_binary_classification(self, valid_hyperparam_data):
        """Test binary classification detection."""
        binary_data = valid_hyperparam_data.copy()
        binary_data["multiclass_categories"] = ["class_a", "class_b"]

        hyperparam = ModelHyperparameters(**binary_data)

        assert hyperparam.num_classes == 2
        assert hyperparam.is_binary

    def test_class_weights_default(self, valid_hyperparam_data):
        """Test default class weights generation."""
        hyperparam = ModelHyperparameters(**valid_hyperparam_data)

        # Should default to [1.0] * num_classes
        expected_weights = [1.0, 1.0, 1.0]
        assert hyperparam.class_weights == expected_weights

    def test_class_weights_custom(self, valid_hyperparam_data):
        """Test custom class weights."""
        data_with_weights = valid_hyperparam_data.copy()
        data_with_weights["class_weights"] = [0.5, 1.0, 2.0]

        hyperparam = ModelHyperparameters(**data_with_weights)

        assert hyperparam.class_weights == [0.5, 1.0, 2.0]

    def test_class_weights_validation_error(self, valid_hyperparam_data):
        """Test class weights validation with wrong length."""
        data_with_wrong_weights = valid_hyperparam_data.copy()
        data_with_wrong_weights["class_weights"] = [
            1.0,
            2.0,
        ]  # Wrong length for 3 classes

        with pytest.raises(ValueError) as exc_info:
            ModelHyperparameters(**data_with_wrong_weights)

        assert "class_weights length" in str(exc_info.value)

    def test_batch_size_validation(self, valid_hyperparam_data):
        """Test batch size validation."""
        # Test valid batch size
        valid_data = valid_hyperparam_data.copy()
        valid_data["batch_size"] = 32
        hyperparam = ModelHyperparameters(**valid_data)
        assert hyperparam.batch_size == 32

        # Test invalid batch size (too large)
        invalid_data = valid_hyperparam_data.copy()
        invalid_data["batch_size"] = 300

        with pytest.raises(ValueError):
            ModelHyperparameters(**invalid_data)

        # Test invalid batch size (zero)
        invalid_data["batch_size"] = 0

        with pytest.raises(ValueError):
            ModelHyperparameters(**invalid_data)

    def test_max_epochs_validation(self, valid_hyperparam_data):
        """Test max epochs validation."""
        # Test valid max epochs
        valid_data = valid_hyperparam_data.copy()
        valid_data["max_epochs"] = 8
        hyperparam = ModelHyperparameters(**valid_data)
        assert hyperparam.max_epochs == 8

        # Test invalid max epochs (too large)
        invalid_data = valid_hyperparam_data.copy()
        invalid_data["max_epochs"] = 15

        with pytest.raises(ValueError):
            ModelHyperparameters(**invalid_data)

        # Test invalid max epochs (zero)
        invalid_data["max_epochs"] = 0

        with pytest.raises(ValueError):
            ModelHyperparameters(**invalid_data)

    def test_categorize_fields(self, valid_hyperparam_data):
        """Test field categorization."""
        hyperparam = ModelHyperparameters(**valid_hyperparam_data)
        categories = hyperparam.categorize_fields()

        # Check that all categories exist
        assert "essential" in categories
        assert "system" in categories
        assert "derived" in categories

        # Check essential fields (required, no defaults)
        essential_fields = set(categories["essential"])
        expected_essential = {
            "full_field_list",
            "cat_field_list",
            "tab_field_list",
            "id_name",
            "label_name",
            "multiclass_categories",
        }
        assert essential_fields == expected_essential

        # Check system fields (have defaults)
        system_fields = set(categories["system"])
        expected_system = {
            "categorical_features_to_encode",
            "model_class",
            "device",
            "header",
            "lr",
            "batch_size",
            "max_epochs",
            "metric_choices",
            "optimizer",
            "class_weights",
        }
        assert system_fields == expected_system

        # Check derived fields (properties)
        derived_fields = set(categories["derived"])
        expected_derived = {
            "input_tab_dim",
            "num_classes",
            "is_binary",
            "model_extra",
            "model_fields_set",
        }
        assert derived_fields == expected_derived

    def test_get_public_init_fields(self, valid_hyperparam_data):
        """Test getting public initialization fields."""
        hyperparam = ModelHyperparameters(**valid_hyperparam_data)
        init_fields = hyperparam.get_public_init_fields()

        # Should include all essential fields
        for field in [
            "full_field_list",
            "cat_field_list",
            "tab_field_list",
            "id_name",
            "label_name",
            "multiclass_categories",
        ]:
            assert field in init_fields

        # Should include non-None system fields
        assert "model_class" in init_fields
        assert "lr" in init_fields
        assert "batch_size" in init_fields
        assert "max_epochs" in init_fields

        # Should not include derived fields
        assert "input_tab_dim" not in init_fields
        assert "num_classes" not in init_fields
        assert "is_binary" not in init_fields

    def test_from_base_hyperparam(self, valid_hyperparam_data):
        """Test creating hyperparameters from base hyperparameters."""
        base_hyperparam = ModelHyperparameters(**valid_hyperparam_data)

        # Create derived hyperparameters with additional fields
        derived_hyperparam = ModelHyperparameters.from_base_hyperparam(
            base_hyperparam, model_class="pytorch", lr=0.001
        )

        # Should inherit base fields
        assert derived_hyperparam.full_field_list == base_hyperparam.full_field_list
        assert (
            derived_hyperparam.multiclass_categories
            == base_hyperparam.multiclass_categories
        )

        # Should override with new values
        assert derived_hyperparam.model_class == "pytorch"
        assert derived_hyperparam.lr == 0.001

        # Should maintain other base values
        assert derived_hyperparam.batch_size == 16  # From base

    def test_get_config(self, valid_hyperparam_data):
        """Test getting configuration dictionary."""
        hyperparam = ModelHyperparameters(**valid_hyperparam_data)
        config = hyperparam.get_config()

        # Should be a dictionary
        assert isinstance(config, dict)

        # Should contain essential fields
        assert "full_field_list" in config
        assert "multiclass_categories" in config

        # Should contain system fields
        assert "model_class" in config
        assert "lr" in config

    def test_serialize_config(self, valid_hyperparam_data):
        """Test configuration serialization for SageMaker."""
        hyperparam = ModelHyperparameters(**valid_hyperparam_data)
        serialized = hyperparam.serialize_config()

        # Should be a dictionary with string values
        assert isinstance(serialized, dict)
        for key, value in serialized.items():
            assert isinstance(value, str)

        # Should include derived fields
        assert "input_tab_dim" in serialized
        assert "num_classes" in serialized
        assert "is_binary" in serialized

        # Test deserialization of complex types
        assert json.loads(serialized["full_field_list"]) == hyperparam.full_field_list
        assert (
            json.loads(serialized["multiclass_categories"])
            == hyperparam.multiclass_categories
        )
        assert json.loads(serialized["is_binary"]) == hyperparam.is_binary

    def test_string_representation(self, valid_hyperparam_data):
        """Test string representation."""
        hyperparam = ModelHyperparameters(**valid_hyperparam_data)
        str_repr = str(hyperparam)

        # Should contain class name
        assert "ModelHyperparameters" in str_repr

        # Should contain field categories
        assert "Essential User Inputs" in str_repr
        assert "System Inputs" in str_repr
        assert "Derived Fields" in str_repr

        # Should contain some field values
        assert "xgboost" in str_repr
        assert "0.01" in str_repr

    def test_print_hyperparam_method(self, valid_hyperparam_data):
        """Test print_hyperparam method."""
        hyperparam = ModelHyperparameters(**valid_hyperparam_data)

        # Should not raise any exceptions
        hyperparam.print_hyperparam()

    def test_derived_fields_caching(self, valid_hyperparam_data):
        """Test that derived fields are cached."""
        hyperparam = ModelHyperparameters(**valid_hyperparam_data)

        # Access derived property multiple times
        first_access = hyperparam.input_tab_dim
        second_access = hyperparam.input_tab_dim

        # Should return the same value (testing caching behavior)
        assert first_access == second_access
        assert first_access == 2

        # Check that private attribute is set
        assert hyperparam._input_tab_dim == 2

    def test_extra_fields_allowed(self, valid_hyperparam_data):
        """Test that extra fields are allowed."""
        data_with_extra = valid_hyperparam_data.copy()
        data_with_extra["extra_field"] = "extra_value"

        # Should not raise an exception
        hyperparam = ModelHyperparameters(**data_with_extra)

        # Extra field should be accessible
        assert hyperparam.extra_field == "extra_value"

    def test_validate_assignment(self, valid_hyperparam_data):
        """Test that validate_assignment works."""
        hyperparam = ModelHyperparameters(**valid_hyperparam_data)

        # Should be able to modify fields
        hyperparam.lr = 0.001
        assert hyperparam.lr == 0.001

        # Should validate on assignment
        with pytest.raises(ValueError):
            hyperparam.batch_size = 0  # Invalid batch size

    def test_empty_field_lists(self):
        """Test with empty field lists."""
        data_with_empty_lists = {
            "full_field_list": ["id", "label"],
            "cat_field_list": [],
            "tab_field_list": [],
            "id_name": "id",
            "label_name": "label",
            "multiclass_categories": ["class_a", "class_b"],
        }

        hyperparam = ModelHyperparameters(**data_with_empty_lists)

        # Should handle empty lists correctly
        assert hyperparam.input_tab_dim == 0
        assert hyperparam.num_classes == 2
        assert hyperparam.is_binary

    def test_single_class_error(self, valid_hyperparam_data):
        """Test that single class raises appropriate error if needed."""
        single_class_data = valid_hyperparam_data.copy()
        single_class_data["multiclass_categories"] = ["single_class"]

        hyperparam = ModelHyperparameters(**single_class_data)

        # Should work but not be binary
        assert hyperparam.num_classes == 1
        assert not hyperparam.is_binary
