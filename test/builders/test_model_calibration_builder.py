#!/usr/bin/env python
"""Tests for the ModelCalibration step builder.

This module contains tests for the ModelCalibrationStepBuilder class, which builds
SageMaker ProcessingSteps for calibrating model prediction scores.
"""

import pytest
from unittest.mock import MagicMock, patch
import boto3
from botocore.stub import Stubber
from sagemaker.workflow.entities import PipelineVariable

from src.pipeline_steps.builder_model_calibration_step import ModelCalibrationStepBuilder
from src.pipeline_steps.config_model_calibration_step import ModelCalibrationConfig


class TestModelCalibrationBuilder:
    """Test cases for the ModelCalibration step builder."""

    @pytest.fixture
    def basic_config(self):
        """Create a basic configuration for tests."""
        return ModelCalibrationConfig(
            region="us-west-2",
            pipeline_s3_loc="s3://bucket/prefix/",
            calibration_method="isotonic",
            label_field="label",
            score_field="score",
            processing_image_uri="137112412989.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3"
        )
    
    @pytest.fixture
    def builder(self, basic_config):
        """Create a builder instance for tests."""
        return ModelCalibrationStepBuilder(config=basic_config)
    
    def test_init(self, basic_config):
        """Test initialization of the builder."""
        builder = ModelCalibrationStepBuilder(config=basic_config)
        assert builder.config == basic_config
        assert builder.spec is not None
        assert builder.contract is not None
        
    def test_init_with_wrong_config_type(self):
        """Test initialization with wrong config type."""
        with pytest.raises(ValueError, match="requires a ModelCalibrationConfig instance"):
            ModelCalibrationStepBuilder(config={"not": "a proper config"})
            
    def test_validate_configuration_success(self, builder):
        """Test successful configuration validation."""
        # Should not raise any exceptions
        builder.validate_configuration()
        
    def test_validate_configuration_missing_attribute(self, basic_config):
        """Test configuration validation with missing attribute."""
        basic_config.calibration_method = None
        builder = ModelCalibrationStepBuilder(config=basic_config)
        
        with pytest.raises(ValueError, match="missing required attribute"):
            builder.validate_configuration()
            
    def test_validate_configuration_invalid_method(self, basic_config):
        """Test configuration validation with invalid calibration method."""
        basic_config.calibration_method = "invalid_method"
        builder = ModelCalibrationStepBuilder(config=basic_config)
        
        with pytest.raises(ValueError, match="Invalid calibration method"):
            builder.validate_configuration()
            
    def test_validate_configuration_invalid_gam_splines(self, basic_config):
        """Test configuration validation with invalid gam_splines value."""
        basic_config.gam_splines = 0
        builder = ModelCalibrationStepBuilder(config=basic_config)
        
        with pytest.raises(ValueError, match="gam_splines must be > 0"):
            builder.validate_configuration()
            
    def test_validate_configuration_invalid_error_threshold(self, basic_config):
        """Test configuration validation with invalid error_threshold value."""
        basic_config.error_threshold = 1.5  # Should be between 0 and 1
        builder = ModelCalibrationStepBuilder(config=basic_config)
        
        with pytest.raises(ValueError, match="error_threshold must be between 0 and 1"):
            builder.validate_configuration()
    
    def test_circular_reference_detection(self, builder):
        """Test detection of circular references in PipelineVariable objects."""
        # Create circular reference in PipelineVariable
        var1 = PipelineVariable("var1")
        var2 = PipelineVariable("var2")
        
        # No circular reference
        assert not builder._detect_circular_references(var1)
        
        # Create circular reference by manipulating internals (for testing only)
        var1._dependencies = [var2]
        var2._dependencies = [var1]
        
        # Should detect circular reference
        assert builder._detect_circular_references(var1)
    
    def test_normalize_s3_uri_with_pipeline_variable(self, builder):
        """Test normalization of S3 URIs with PipelineVariables."""
        var = PipelineVariable("s3://bucket/path")
        result = builder._normalize_s3_uri(var)
        assert result is var
        
    def test_validate_s3_uri_with_invalid_uri(self, builder):
        """Test validation of S3 URIs with invalid URI."""
        with pytest.raises(ValueError, match="Invalid S3 URI"):
            builder._validate_s3_uri("http://not-s3")
    
    def test_complex_nested_pipeline_variables(self, builder):
        """Test handling of complex nested PipelineVariable structures."""
        var1 = PipelineVariable("base")
        var2 = PipelineVariable("prefix")
        var3 = PipelineVariable("path")
        
        # Create a complex nested structure
        nested_var = PipelineVariable(f"{var2}/{var3}")
        complex_var = PipelineVariable(f"{var1}/{nested_var}")
        
        # Should handle complex nesting without issues
        result = builder._normalize_s3_uri(complex_var)
        assert result is complex_var
        
        # No circular references
        assert not builder._detect_circular_references(complex_var)
            
    def test_get_environment_variables(self, builder):
        """Test generation of environment variables."""
        env_vars = builder._get_environment_variables()
        
        # Check calibration-specific variables
        assert "CALIBRATION_METHOD" in env_vars
        assert env_vars["CALIBRATION_METHOD"] == "isotonic"
        assert "LABEL_FIELD" in env_vars
        assert "SCORE_FIELD" in env_vars
        assert "MONOTONIC_CONSTRAINT" in env_vars
        assert "GAM_SPLINES" in env_vars
        assert "ERROR_THRESHOLD" in env_vars

    @patch("src.pipeline_steps.builder_model_calibration_step.ProcessingStep")
    def test_create_step(self, mock_processing_step, builder):
        """Test creation of the processing step."""
        # Setup mock inputs
        mock_inputs = {
            "evaluation_data": "s3://bucket/evaluation_data/",
            "model_artifacts": "s3://bucket/model/"
        }
        
        # Create step
        step = builder.create_step(inputs=mock_inputs)
        
        # Verify ProcessingStep was called
        mock_processing_step.assert_called_once()
        
        # Verify spec was attached
        assert hasattr(step, "_spec")
