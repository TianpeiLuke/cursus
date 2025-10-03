"""
Test Validator Factory Module

Tests for priority-based validator creation and step-type-aware validator selection.
Tests validator registry management and factory configuration validation.
"""

import pytest
from unittest.mock import patch, MagicMock, Mock

from cursus.validation.alignment.validators.validator_factory import ValidatorFactory


class TestValidatorFactory:
    """Test ValidatorFactory functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.workspace_dirs = ["/test/workspace"]
        self.factory = ValidatorFactory(self.workspace_dirs)

    def test_factory_initialization(self):
        """Test ValidatorFactory initialization."""
        assert self.factory.workspace_dirs == self.workspace_dirs
        assert hasattr(self.factory, '_validator_registry')
        assert isinstance(self.factory._validator_registry, dict)

    @patch('cursus.validation.alignment.validators.validator_factory.get_universal_validation_rules')
    @patch('cursus.validation.alignment.validators.validator_factory.get_step_type_validation_rules')
    def test_factory_initialization_with_rules(self, mock_step_type_rules, mock_universal_rules):
        """Test factory initialization loads validation rules."""
        mock_universal_rules.return_value = {"universal": "rules"}
        mock_step_type_rules.return_value = {"step_type": "rules"}
        
        factory = ValidatorFactory(self.workspace_dirs)
        
        # Verify rules were loaded
        mock_universal_rules.assert_called_once()
        mock_step_type_rules.assert_called_once()
        assert hasattr(factory, 'universal_rules')
        assert hasattr(factory, 'step_type_rules')

    def test_get_validator_success(self):
        """Test successful validator retrieval."""
        # Mock a validator class
        mock_validator_class = Mock()
        mock_validator_instance = Mock()
        mock_validator_class.return_value = mock_validator_instance
        
        # Add to registry
        self.factory._validator_registry["TestValidator"] = mock_validator_class
        
        validator = self.factory.get_validator("TestValidator")
        
        # Verify validator was instantiated with workspace_dirs
        mock_validator_class.assert_called_once_with(self.workspace_dirs)
        assert validator == mock_validator_instance

    def test_get_validator_not_found(self):
        """Test validator retrieval for non-existent validator."""
        validator = self.factory.get_validator("NonExistentValidator")
        
        # Should return None for non-existent validator
        assert validator is None

    def test_get_validator_none_in_registry(self):
        """Test validator retrieval when registry has None value."""
        # Add None to registry (placeholder)
        self.factory._validator_registry["PlaceholderValidator"] = None
        
        validator = self.factory.get_validator("PlaceholderValidator")
        
        # Should return None for placeholder validators
        assert validator is None

    @patch('cursus.validation.alignment.validators.validator_factory.get_validation_ruleset')
    def test_get_validator_for_step_type_success(self, mock_get_ruleset):
        """Test successful step-type validator retrieval."""
        # Mock ruleset
        mock_ruleset = Mock()
        mock_ruleset.level_4_validator_class = "ProcessingStepBuilderValidator"
        mock_get_ruleset.return_value = mock_ruleset
        
        # Mock validator class
        mock_validator_class = Mock()
        mock_validator_instance = Mock()
        mock_validator_class.return_value = mock_validator_instance
        self.factory._validator_registry["ProcessingStepBuilderValidator"] = mock_validator_class
        
        validator = self.factory.get_validator_for_step_type("Processing")
        
        # Verify ruleset was retrieved
        mock_get_ruleset.assert_called_once_with("Processing")
        
        # Verify validator was instantiated
        mock_validator_class.assert_called_once_with(self.workspace_dirs)
        assert validator == mock_validator_instance

    @patch('cursus.validation.alignment.validators.validator_factory.get_validation_ruleset')
    def test_get_validator_for_step_type_no_ruleset(self, mock_get_ruleset):
        """Test step-type validator retrieval with no ruleset."""
        mock_get_ruleset.return_value = None
        
        validator = self.factory.get_validator_for_step_type("UnknownStepType")
        
        # Should return None when no ruleset found
        assert validator is None

    @patch('cursus.validation.alignment.validators.validator_factory.get_validation_ruleset')
    def test_get_validator_for_step_type_no_validator_class(self, mock_get_ruleset):
        """Test step-type validator retrieval with no validator class in ruleset."""
        # Mock ruleset without validator class
        mock_ruleset = Mock()
        mock_ruleset.level_4_validator_class = None
        mock_get_ruleset.return_value = mock_ruleset
        
        validator = self.factory.get_validator_for_step_type("Processing")
        
        # Should return None when no validator class specified
        assert validator is None

    def test_get_available_validators(self):
        """Test retrieval of available validators."""
        # Add some test validators to registry
        self.factory._validator_registry.update({
            "ProcessingStepBuilderValidator": Mock(),
            "TrainingStepBuilderValidator": Mock(),
            "CreateModelStepBuilderValidator": None,  # Placeholder
        })
        
        available = self.factory.get_available_validators()
        
        # Should return list of validator names
        assert isinstance(available, list)
        assert "ProcessingStepBuilderValidator" in available
        assert "TrainingStepBuilderValidator" in available
        assert "CreateModelStepBuilderValidator" in available

    def test_is_validator_available(self):
        """Test checking validator availability."""
        # Add test validator to registry
        self.factory._validator_registry["TestValidator"] = Mock()
        self.factory._validator_registry["PlaceholderValidator"] = None
        
        # Test available validator
        assert self.factory.is_validator_available("TestValidator") is True
        
        # Test placeholder validator (should still be considered available)
        assert self.factory.is_validator_available("PlaceholderValidator") is True
        
        # Test non-existent validator
        assert self.factory.is_validator_available("NonExistentValidator") is False

    def test_get_validator_registry_status(self):
        """Test getting validator registry status."""
        # Add test validators to registry
        mock_validator_class = Mock()
        self.factory._validator_registry.update({
            "ProcessingStepBuilderValidator": mock_validator_class,
            "TrainingStepBuilderValidator": mock_validator_class,
            "CreateModelStepBuilderValidator": None,  # Placeholder
            "TransformStepBuilderValidator": None,  # Placeholder
        })
        
        status = self.factory.get_validator_registry_status()
        
        # Verify status structure
        assert "total_validators" in status
        assert "implemented_validators" in status
        assert "placeholder_validators" in status
        assert "implementation_rate" in status
        
        # Verify counts - accept current registry size
        assert status["total_validators"] >= 4  # At least the ones we added
        assert status["implemented_validators"] >= 2  # At least the ones we added
        assert status["placeholder_validators"] >= 2  # At least the ones we added
        assert 0.0 <= status["implementation_rate"] <= 1.0  # Valid rate range

    @patch('cursus.validation.alignment.validators.validator_factory.get_validation_ruleset')
    def test_validate_step_with_priority_system(self, mock_get_ruleset):
        """Test step validation using priority-based validation system."""
        # Mock step type detection
        step_type = "Processing"
        
        # Mock ruleset
        mock_ruleset = Mock()
        mock_ruleset.level_4_validator_class = "ProcessingStepBuilderValidator"
        mock_get_ruleset.return_value = mock_ruleset
        
        # Mock validator
        mock_validator_class = Mock()
        mock_validator_instance = Mock()
        mock_validator_instance.validate_builder_config_alignment.return_value = {
            "passed": True,
            "issues": [],
            "priority_system": "universal_first_then_step_specific"
        }
        mock_validator_class.return_value = mock_validator_instance
        self.factory._validator_registry["ProcessingStepBuilderValidator"] = mock_validator_class
        
        # Mock step type detection
        with patch('cursus.validation.alignment.validators.validator_factory.get_sagemaker_step_type', return_value=step_type):
            result = self.factory.validate_step_with_priority_system("test_step")
        
        # Verify validation was called
        mock_validator_instance.validate_builder_config_alignment.assert_called_once_with("test_step")
        
        # Verify result
        assert result["passed"] is True
        assert "priority_system" in result

    @patch('cursus.validation.alignment.validators.validator_factory.get_sagemaker_step_type')
    @patch('cursus.validation.alignment.validators.validator_factory.get_validation_ruleset')
    @patch('cursus.validation.alignment.validators.validator_factory.is_step_type_excluded')
    def test_validate_step_excluded_type(self, mock_is_excluded, mock_get_ruleset, mock_get_step_type):
        """Test validation of excluded step type."""
        # Mock excluded step type
        mock_get_step_type.return_value = "Base"
        mock_is_excluded.return_value = True
        
        # Mock ruleset
        mock_ruleset = Mock()
        mock_ruleset.skip_reason = "Base configurations - no builder to validate"
        mock_get_ruleset.return_value = mock_ruleset
        
        result = self.factory.validate_step_with_priority_system("base_step")
        
        # Verify excluded step handling
        assert result["status"] == "EXCLUDED"
        assert result["reason"] == "Base configurations - no builder to validate"

    @patch('cursus.validation.alignment.validators.validator_factory.get_sagemaker_step_type')
    @patch('cursus.validation.alignment.validators.validator_factory.get_validation_ruleset')
    @patch('cursus.validation.alignment.validators.validator_factory.is_step_type_excluded')
    def test_validate_step_no_validator(self, mock_is_excluded, mock_get_ruleset, mock_get_step_type):
        """Test validation when no validator is available."""
        # Mock non-excluded step type
        mock_get_step_type.return_value = "UnknownType"
        mock_is_excluded.return_value = False
        mock_get_ruleset.return_value = None  # No ruleset
        
        result = self.factory.validate_step_with_priority_system("unknown_step")
        
        # Verify no validator handling
        assert result["status"] == "NO_VALIDATOR"
        assert "No validator available" in result["message"]

    def test_factory_configuration_validation(self):
        """Test factory configuration validation."""
        # Test with valid configuration
        validation_result = self.factory.validate_factory_configuration()
        
        # Should return validation status
        assert "valid" in validation_result
        assert "issues" in validation_result
        assert isinstance(validation_result["issues"], list)

    def test_factory_health_check(self):
        """Test factory health check functionality."""
        # Add test validators to registry
        mock_validator_class = Mock()
        self.factory._validator_registry.update({
            "ProcessingStepBuilderValidator": mock_validator_class,
            "TrainingStepBuilderValidator": None,  # Placeholder
        })
        
        health_status = self.factory.get_factory_health_status()
        
        # Verify health status structure
        assert "healthy" in health_status
        assert "registry_status" in health_status
        assert "workspace_dirs" in health_status
        assert "total_validators" in health_status["registry_status"]

    def test_multiple_workspace_dirs(self):
        """Test factory with multiple workspace directories."""
        multiple_dirs = ["/workspace1", "/workspace2", "/workspace3"]
        factory = ValidatorFactory(multiple_dirs)
        
        # Mock validator class
        mock_validator_class = Mock()
        mock_validator_instance = Mock()
        mock_validator_class.return_value = mock_validator_instance
        factory._validator_registry["TestValidator"] = mock_validator_class
        
        validator = factory.get_validator("TestValidator")
        
        # Verify validator was instantiated with all workspace dirs
        mock_validator_class.assert_called_once_with(multiple_dirs)
        assert validator == mock_validator_instance

    def test_factory_error_handling(self):
        """Test factory error handling."""
        # Mock validator class that raises exception during instantiation
        mock_validator_class = Mock()
        mock_validator_class.side_effect = Exception("Validator initialization failed")
        self.factory._validator_registry["ErrorValidator"] = mock_validator_class
        
        # Should handle exception gracefully
        validator = self.factory.get_validator("ErrorValidator")
        
        # Should return None when validator instantiation fails
        assert validator is None

    def test_factory_statistics(self):
        """Test factory statistics collection."""
        # Add various validators to registry
        self.factory._validator_registry.update({
            "ProcessingStepBuilderValidator": Mock(),
            "TrainingStepBuilderValidator": Mock(),
            "CreateModelStepBuilderValidator": None,
            "TransformStepBuilderValidator": None,
            "RegisterModelStepBuilderValidator": None,
        })
        
        stats = self.factory.get_factory_statistics()
        
        # Verify statistics structure
        assert "validator_counts" in stats
        assert "implementation_status" in stats
        assert "workspace_info" in stats
        
        # Verify counts - accept current registry size
        counts = stats["validator_counts"]
        assert counts["total"] >= 5  # At least the ones we added
        assert counts["implemented"] >= 2  # At least the ones we added
        assert counts["placeholder"] >= 3  # At least the ones we added
