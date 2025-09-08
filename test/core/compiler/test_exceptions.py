"""
Unit tests for the exceptions module.

This module tests the custom exception classes used throughout the Pipeline API
to ensure they provide clear, actionable error messages for users.
"""

import unittest
from cursus.core.compiler.exceptions import (
    PipelineAPIError,
    ConfigurationError,
    AmbiguityError,
    ValidationError,
    ResolutionError
)

class TestPipelineAPIExceptions(unittest.TestCase):
    """Tests for the Pipeline API exception classes."""

    def test_pipeline_api_error_base(self):
        """Test the base PipelineAPIError exception."""
        error = PipelineAPIError("Test error message")
        self.assertEqual(str(error), "Test error message")
        self.assertIsInstance(error, Exception)

    def test_configuration_error_basic(self):
        """Test ConfigurationError with basic message."""
        error = ConfigurationError("Configuration not found")
        self.assertEqual(str(error), "Configuration not found")
        self.assertIsInstance(error, PipelineAPIError)
        self.assertEqual(error.missing_configs, [])
        self.assertEqual(error.available_configs, [])

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
        self.assertIn("Missing configurations", error_str)
        self.assertIn("Missing configurations: ['data_loading', 'preprocessing']", error_str)
        self.assertIn("Available configurations: ['training', 'evaluation']", error_str)
        
        self.assertEqual(error.missing_configs, missing_configs)
        self.assertEqual(error.available_configs, available_configs)

    def test_ambiguity_error_basic(self):
        """Test AmbiguityError with basic message."""
        error = AmbiguityError("Multiple configurations match")
        self.assertEqual(str(error), "Multiple configurations match")
        self.assertIsInstance(error, PipelineAPIError)
        self.assertIsNone(error.node_name)
        self.assertEqual(error.candidates, [])

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
        self.assertIn("Ambiguous match for node", error_str)
        self.assertIn("Candidates for node 'preprocessing':", error_str)
        self.assertIn("MockConfig (job_type='training', confidence=0.85)", error_str)
        self.assertIn("MockConfig (job_type='evaluation', confidence=0.82)", error_str)
        
        self.assertEqual(error.node_name, "preprocessing")
        self.assertEqual(error.candidates, candidates)

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
        self.assertIn("Multiple matches found", error_str)
        self.assertIn("Candidates for node 'model_step':", error_str)
        self.assertIn("XGBoostTrainingConfig (job_type='training', confidence=0.85)", error_str)
        self.assertIn("XGBoostModelEvalConfig (job_type='evaluation', confidence=0.82)", error_str)

    def test_validation_error_basic(self):
        """Test ValidationError with basic message."""
        error = ValidationError("Validation failed")
        self.assertEqual(str(error), "Validation failed")
        self.assertIsInstance(error, PipelineAPIError)
        self.assertEqual(error.validation_errors, {})

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
        self.assertIn("Configuration validation failed", error_str)
        self.assertIn("Validation errors:", error_str)
        self.assertIn("missing_fields:", error_str)
        self.assertIn("- required_param", error_str)
        self.assertIn("- output_path", error_str)
        self.assertIn("invalid_values:", error_str)
        self.assertIn("- batch_size must be positive", error_str)
        self.assertIn("- learning_rate out of range", error_str)
        self.assertIn("dependency_issues:", error_str)
        self.assertIn("- input_data not found", error_str)
        
        self.assertEqual(error.validation_errors, validation_errors)

    def test_resolution_error_basic(self):
        """Test ResolutionError with basic message."""
        error = ResolutionError("Resolution failed")
        self.assertEqual(str(error), "Resolution failed")
        self.assertIsInstance(error, PipelineAPIError)
        self.assertEqual(error.failed_nodes, [])
        self.assertEqual(error.suggestions, [])

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
        self.assertIn("Failed to resolve DAG nodes", error_str)
        self.assertIn("Failed to resolve nodes: ['data_loading', 'preprocessing']", error_str)
        self.assertIn("Suggestions:", error_str)
        self.assertIn("- Add configuration for data_loading node", error_str)
        self.assertIn("- Check node naming conventions", error_str)
        self.assertIn("- Ensure job_type attributes are set correctly", error_str)
        
        self.assertEqual(error.failed_nodes, failed_nodes)
        self.assertEqual(error.suggestions, suggestions)

    def test_exception_inheritance(self):
        """Test that all exceptions inherit from PipelineAPIError."""
        exceptions = [
            ConfigurationError("test"),
            AmbiguityError("test"),
            ValidationError("test"),
            ResolutionError("test")
        ]
        
        for exc in exceptions:
            self.assertIsInstance(exc, PipelineAPIError)
            self.assertIsInstance(exc, Exception)

if __name__ == '__main__':
    unittest.main()
