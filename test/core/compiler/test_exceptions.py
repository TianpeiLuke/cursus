"""
Unit tests for the exceptions module.

This module tests the custom exception classes used throughout the Pipeline API
to ensure they provide clear, actionable error messages for users.
"""

import pytest
from cursus.core.compiler.exceptions import (
    PipelineAPIError,
    ConfigurationError,
    AmbiguityError,
    ValidationError,
    ResolutionError
)


class TestPipelineAPIExceptions:
    """Tests for the Pipeline API exception classes."""

    def test_pipeline_api_error_base(self):
        """Test the base PipelineAPIError exception."""
        error = PipelineAPIError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_configuration_error_basic(self):
        """Test ConfigurationError with basic message."""
        error = ConfigurationError("Configuration not found")
        assert str(error) == "Configuration not found"
        assert isinstance(error, PipelineAPIError)
        assert error.missing_configs == []
        assert error.available_configs == []

    def test_configuration_error_with_details(self):
        """Test ConfigurationError with missing and available configs."""
        missing_configs = ["data_loading", "preprocessing"]
        available_configs = ["training", "evaluation"]
        
        error = ConfigurationError(
            "Missing configurations",
            missing_configs=missing_configs,
            available_configs=available_configs
        )
        
        error_str = str(error)
        assert "Missing configurations" in error_str
        assert "Missing configurations: ['data_loading', 'preprocessing']" in error_str
        assert "Available configurations: ['training', 'evaluation']" in error_str
        
        assert error.missing_configs == missing_configs
        assert error.available_configs == available_configs

    def test_ambiguity_error_basic(self):
        """Test AmbiguityError with basic message."""
        error = AmbiguityError("Multiple configurations match")
        assert str(error) == "Multiple configurations match"
        assert isinstance(error, PipelineAPIError)
        assert error.node_name is None
        assert error.candidates == []

    def test_ambiguity_error_with_tuple_candidates(self):
        """Test AmbiguityError with tuple format candidates."""
        # Mock config objects
        class MockConfig:
            def __init__(self, job_type):
                self.job_type = job_type
        
        config1 = MockConfig("training")
        config2 = MockConfig("evaluation")
        
        candidates = [
            (config1, 0.85, "semantic"),
            (config2, 0.82, "pattern")
        ]
        
        error = AmbiguityError(
            "Ambiguous match for node",
            node_name="preprocessing",
            candidates=candidates
        )
        
        error_str = str(error)
        assert "Ambiguous match for node" in error_str
        assert "Candidates for node 'preprocessing':" in error_str
        assert "MockConfig (job_type='training', confidence=0.85)" in error_str
        assert "MockConfig (job_type='evaluation', confidence=0.82)" in error_str
        
        assert error.node_name == "preprocessing"
        assert error.candidates == candidates

    def test_ambiguity_error_with_dict_candidates(self):
        """Test AmbiguityError with dictionary format candidates."""
        candidates = [
            {
                "config_type": "XGBoostTrainingConfig",
                "confidence": 0.85,
                "job_type": "training"
            },
            {
                "config_type": "XGBoostModelEvalConfig", 
                "confidence": 0.82,
                "job_type": "evaluation"
            }
        ]
        
        error = AmbiguityError(
            "Multiple matches found",
            node_name="model_step",
            candidates=candidates
        )
        
        error_str = str(error)
        assert "Multiple matches found" in error_str
        assert "Candidates for node 'model_step':" in error_str
        assert "XGBoostTrainingConfig (job_type='training', confidence=0.85)" in error_str
        assert "XGBoostModelEvalConfig (job_type='evaluation', confidence=0.82)" in error_str

    def test_validation_error_basic(self):
        """Test ValidationError with basic message."""
        error = ValidationError("Validation failed")
        assert str(error) == "Validation failed"
        assert isinstance(error, PipelineAPIError)
        assert error.validation_errors == {}

    def test_validation_error_with_details(self):
        """Test ValidationError with detailed validation errors."""
        validation_errors = {
            "missing_fields": ["required_param", "output_path"],
            "invalid_values": ["batch_size must be positive", "learning_rate out of range"],
            "dependency_issues": ["input_data not found"]
        }
        
        error = ValidationError(
            "Configuration validation failed",
            validation_errors=validation_errors
        )
        
        error_str = str(error)
        assert "Configuration validation failed" in error_str
        assert "Validation errors:" in error_str
        assert "missing_fields:" in error_str
        assert "- required_param" in error_str
        assert "- output_path" in error_str
        assert "invalid_values:" in error_str
        assert "- batch_size must be positive" in error_str
        assert "- learning_rate out of range" in error_str
        assert "dependency_issues:" in error_str
        assert "- input_data not found" in error_str
        
        assert error.validation_errors == validation_errors

    def test_resolution_error_basic(self):
        """Test ResolutionError with basic message."""
        error = ResolutionError("Resolution failed")
        assert str(error) == "Resolution failed"
        assert isinstance(error, PipelineAPIError)
        assert error.failed_nodes == []
        assert error.suggestions == []

    def test_resolution_error_with_details(self):
        """Test ResolutionError with failed nodes and suggestions."""
        failed_nodes = ["data_loading", "preprocessing"]
        suggestions = [
            "Add configuration for data_loading node",
            "Check node naming conventions",
            "Ensure job_type attributes are set correctly"
        ]
        
        error = ResolutionError(
            "Failed to resolve DAG nodes",
            failed_nodes=failed_nodes,
            suggestions=suggestions
        )
        
        error_str = str(error)
        assert "Failed to resolve DAG nodes" in error_str
        assert "Failed to resolve nodes: ['data_loading', 'preprocessing']" in error_str
        assert "Suggestions:" in error_str
        assert "- Add configuration for data_loading node" in error_str
        assert "- Check node naming conventions" in error_str
        assert "- Ensure job_type attributes are set correctly" in error_str
        
        assert error.failed_nodes == failed_nodes
        assert error.suggestions == suggestions

    def test_exception_inheritance(self):
        """Test that all exceptions inherit from PipelineAPIError."""
        exceptions = [
            ConfigurationError("test"),
            AmbiguityError("test"),
            ValidationError("test"),
            ResolutionError("test")
        ]
        
        for exc in exceptions:
            assert isinstance(exc, PipelineAPIError)
            assert isinstance(exc, Exception)
