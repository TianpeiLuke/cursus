"""
Unit tests for property_path_validator.py

Tests SageMaker Property Path Validator functionality including:
- Property path validation for different step types
- Step type resolution via registry integration
- Pattern matching for array access and wildcards
- Documentation and suggestion generation
- Error handling and edge cases
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from cursus.validation.alignment.property_path_validator import (
    SageMakerPropertyPathValidator,
    validate_property_paths
)


class TestSageMakerPropertyPathValidator:
    """Test SageMaker Property Path Validator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = SageMakerPropertyPathValidator()
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        assert self.validator.documentation_version == "v2.92.2"
        assert "sagemaker.readthedocs.io" in self.validator.documentation_url
        assert self.validator._property_path_cache == {}
    
    def test_validate_specification_property_paths_empty_spec(self):
        """Test validation with empty specification."""
        specification = {}
        issues = self.validator.validate_specification_property_paths(specification, "test_contract")
        
        # Should have step type resolution info
        assert len(issues) >= 1
        assert any(issue['category'] == 'step_type_resolution' for issue in issues)
    
    def test_validate_specification_property_paths_training_step(self):
        """Test validation for training step."""
        specification = {
            'step_type': 'training',
            'node_type': 'training',
            'outputs': [
                {
                    'logical_name': 'model_artifacts',
                    'property_path': 'properties.ModelArtifacts.S3ModelArtifacts'
                },
                {
                    'logical_name': 'invalid_output',
                    'property_path': 'properties.InvalidPath.NotExists'
                }
            ]
        }
        
        issues = self.validator.validate_specification_property_paths(specification, "training_contract")
        
        # Should have validation results for both outputs
        validation_issues = [issue for issue in issues if issue['category'] == 'property_path_validation']
        assert len(validation_issues) == 2
        
        # One should be valid, one invalid
        valid_issues = [issue for issue in validation_issues if issue['severity'] == 'INFO']
        error_issues = [issue for issue in validation_issues if issue['severity'] == 'ERROR']
        assert len(valid_issues) == 1
        assert len(error_issues) == 1
    
    def test_validate_specification_property_paths_processing_step(self):
        """Test validation for processing step."""
        specification = {
            'step_type': 'processing',
            'node_type': 'processing',
            'outputs': [
                {
                    'logical_name': 'output_data',
                    'property_path': 'properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri'
                }
            ]
        }
        
        issues = self.validator.validate_specification_property_paths(specification, "processing_contract")
        
        # Should validate successfully
        validation_issues = [issue for issue in issues if issue['category'] == 'property_path_validation']
        assert len(validation_issues) == 1
        assert validation_issues[0]['severity'] == 'INFO'
    
    def test_validate_specification_property_paths_unknown_step_type(self):
        """Test validation for unknown step type."""
        specification = {
            'step_type': 'unknown_step',
            'node_type': 'unknown',
            'outputs': [
                {
                    'logical_name': 'some_output',
                    'property_path': 'properties.SomeProperty'
                }
            ]
        }
        
        issues = self.validator.validate_specification_property_paths(specification, "unknown_contract")
        
        # Should skip validation for unknown step type
        skip_issues = [issue for issue in issues if 'skipped' in issue['message']]
        assert len(skip_issues) == 1
    
    @patch('cursus.validation.alignment.property_path_validator.STEP_REGISTRY_AVAILABLE', True)
    @patch('cursus.validation.alignment.property_path_validator.get_step_name_from_spec_type')
    @patch('cursus.validation.alignment.property_path_validator.get_sagemaker_step_type')
    def test_step_registry_integration(self, mock_get_sagemaker_type, mock_get_step_name):
        """Test step registry integration for step type resolution."""
        # Mock registry functions
        mock_get_step_name.return_value = "CurrencyConversion"
        mock_get_sagemaker_type.return_value = "Processing"
        
        specification = {
            'step_type': 'CurrencyConversion_Training',
            'outputs': []
        }
        
        issues = self.validator.validate_specification_property_paths(specification, "registry_test")
        
        # Should have step type resolution info
        resolution_issues = [issue for issue in issues if issue['category'] == 'step_type_resolution']
        assert len(resolution_issues) == 1
        assert 'resolved via registry' in resolution_issues[0]['message']
        assert resolution_issues[0]['details']['resolved_sagemaker_type'] == 'Processing'
        
        # Verify registry functions were called
        mock_get_step_name.assert_called_once_with('CurrencyConversion_Training')
        mock_get_sagemaker_type.assert_called_once_with('CurrencyConversion')
    
    @patch('cursus.validation.alignment.property_path_validator.STEP_REGISTRY_AVAILABLE', False)
    def test_step_registry_unavailable(self):
        """Test behavior when step registry is unavailable."""
        specification = {
            'step_type': 'SomeStep_Training',
            'outputs': []
        }
        
        issues = self.validator.validate_specification_property_paths(specification, "no_registry_test")
        
        # Should have warning about registry not available
        resolution_issues = [issue for issue in issues if issue['category'] == 'step_type_resolution']
        assert len(resolution_issues) == 1
        assert resolution_issues[0]['severity'] == 'WARNING'
        assert 'not available' in resolution_issues[0]['message']
    
    def test_get_valid_property_paths_for_step_type_training(self):
        """Test getting valid property paths for training step."""
        paths = self.validator._get_valid_property_paths_for_step_type('training', 'training')
        
        assert 'model_artifacts' in paths
        assert 'metrics' in paths
        assert 'job_info' in paths
        assert 'properties.ModelArtifacts.S3ModelArtifacts' in paths['model_artifacts']
        assert any('FinalMetricDataList' in path for path in paths['metrics'])
    
    def test_get_valid_property_paths_for_step_type_processing(self):
        """Test getting valid property paths for processing step."""
        paths = self.validator._get_valid_property_paths_for_step_type('processing', 'processing')
        
        assert 'outputs' in paths
        assert 'inputs' in paths
        assert 'job_info' in paths
        assert any('ProcessingOutputConfig' in path for path in paths['outputs'])
        assert any('ProcessingInputs' in path for path in paths['inputs'])
    
    def test_get_valid_property_paths_for_step_type_transform(self):
        """Test getting valid property paths for transform step."""
        paths = self.validator._get_valid_property_paths_for_step_type('transform', 'transform')
        
        assert 'outputs' in paths
        assert 'job_info' in paths
        assert 'inputs' in paths
        assert any('TransformOutput' in path for path in paths['outputs'])
    
    def test_get_valid_property_paths_for_step_type_tuning(self):
        """Test getting valid property paths for tuning step."""
        paths = self.validator._get_valid_property_paths_for_step_type('tuning', 'tuning')
        
        assert 'best_training_job' in paths
        assert 'training_job_summaries' in paths
        assert 'job_info' in paths
        assert any('BestTrainingJob' in path for path in paths['best_training_job'])
    
    def test_get_valid_property_paths_for_step_type_create_model(self):
        """Test getting valid property paths for create model step."""
        paths = self.validator._get_valid_property_paths_for_step_type('create_model', 'model')
        
        assert 'model_info' in paths
        assert 'primary_container' in paths
        assert 'containers' in paths
        assert any('ModelName' in path for path in paths['model_info'])
        assert any('PrimaryContainer' in path for path in paths['primary_container'])
    
    def test_get_valid_property_paths_for_step_type_lambda(self):
        """Test getting valid property paths for lambda step."""
        paths = self.validator._get_valid_property_paths_for_step_type('lambda', 'lambda')
        
        assert 'output_parameters' in paths
        assert 'OutputParameters[*]' in paths['output_parameters']
    
    def test_get_valid_property_paths_for_step_type_callback(self):
        """Test getting valid property paths for callback step."""
        paths = self.validator._get_valid_property_paths_for_step_type('callback', 'callback')
        
        assert 'output_parameters' in paths
        assert 'OutputParameters[*]' in paths['output_parameters']
    
    def test_get_valid_property_paths_for_step_type_quality_check(self):
        """Test getting valid property paths for quality check step."""
        paths = self.validator._get_valid_property_paths_for_step_type('quality_check', 'quality')
        
        assert 'baseline_constraints' in paths
        assert 'baseline_statistics' in paths
        assert 'drift_check' in paths
    
    def test_get_valid_property_paths_for_step_type_clarify(self):
        """Test getting valid property paths for clarify step."""
        paths = self.validator._get_valid_property_paths_for_step_type('clarify', 'clarify')
        
        assert 'baseline_constraints' in paths
        assert 'drift_check' in paths
    
    def test_get_valid_property_paths_for_step_type_emr(self):
        """Test getting valid property paths for EMR step."""
        paths = self.validator._get_valid_property_paths_for_step_type('emr', 'emr')
        
        assert 'cluster_info' in paths
        assert 'properties.ClusterId' in paths['cluster_info']
    
    def test_get_valid_property_paths_caching(self):
        """Test property path caching functionality."""
        # First call should populate cache
        paths1 = self.validator._get_valid_property_paths_for_step_type('training', 'training')
        
        # Second call should use cache
        paths2 = self.validator._get_valid_property_paths_for_step_type('training', 'training')
        
        assert paths1 == paths2
        assert 'training_training' in self.validator._property_path_cache
    
    def test_validate_single_property_path_exact_match(self):
        """Test single property path validation with exact match."""
        valid_paths = {
            'model_artifacts': ['properties.ModelArtifacts.S3ModelArtifacts']
        }
        
        result = self.validator._validate_single_property_path(
            'properties.ModelArtifacts.S3ModelArtifacts',
            'training',
            'training',
            valid_paths
        )
        
        assert result['valid'] is True
        assert result['match_type'] == 'exact'
    
    def test_validate_single_property_path_pattern_match(self):
        """Test single property path validation with pattern match."""
        valid_paths = {
            'metrics': ['properties.FinalMetricDataList[*].Value']
        }
        
        # Test indexed access
        result = self.validator._validate_single_property_path(
            'properties.FinalMetricDataList[0].Value',
            'training',
            'training',
            valid_paths
        )
        
        assert result['valid'] is True
        assert result['match_type'] == 'pattern'
        
        # Test named access
        result = self.validator._validate_single_property_path(
            'properties.FinalMetricDataList["accuracy"].Value',
            'training',
            'training',
            valid_paths
        )
        
        assert result['valid'] is True
        assert result['match_type'] == 'pattern'
    
    def test_validate_single_property_path_invalid(self):
        """Test single property path validation with invalid path."""
        valid_paths = {
            'model_artifacts': ['properties.ModelArtifacts.S3ModelArtifacts']
        }
        
        result = self.validator._validate_single_property_path(
            'properties.InvalidPath.NotExists',
            'training',
            'training',
            valid_paths
        )
        
        assert result['valid'] is False
        assert result['match_type'] == 'none'
        assert 'not valid' in result['error']
        assert len(result['suggestions']) > 0
    
    def test_matches_property_path_pattern_exact(self):
        """Test property path pattern matching for exact matches."""
        assert self.validator._matches_property_path_pattern(
            'properties.ModelArtifacts.S3ModelArtifacts',
            'properties.ModelArtifacts.S3ModelArtifacts'
        ) is True
    
    def test_matches_property_path_pattern_wildcard(self):
        """Test property path pattern matching for wildcards."""
        pattern = 'properties.FinalMetricDataList[*].Value'
        
        # Should match indexed access
        assert self.validator._matches_property_path_pattern(
            'properties.FinalMetricDataList[0].Value',
            pattern
        ) is True
        
        # Should match named access
        assert self.validator._matches_property_path_pattern(
            'properties.FinalMetricDataList["accuracy"].Value',
            pattern
        ) is True
        
        # Should match wildcard
        assert self.validator._matches_property_path_pattern(
            'properties.FinalMetricDataList[*].Value',
            pattern
        ) is True
        
        # Should not match different path
        assert self.validator._matches_property_path_pattern(
            'properties.DifferentPath[0].Value',
            pattern
        ) is False
    
    def test_matches_property_path_pattern_complex(self):
        """Test property path pattern matching for complex patterns."""
        pattern = 'properties.ProcessingOutputConfig.Outputs[*].S3Output.S3Uri'
        
        # Should match indexed access
        assert self.validator._matches_property_path_pattern(
            'properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri',
            pattern
        ) is True
        
        # Should match named access
        assert self.validator._matches_property_path_pattern(
            'properties.ProcessingOutputConfig.Outputs["output1"].S3Output.S3Uri',
            pattern
        ) is True
    
    def test_get_property_path_suggestions(self):
        """Test property path suggestion generation."""
        all_valid_paths = [
            'properties.ModelArtifacts.S3ModelArtifacts',
            'properties.OutputDataConfig.S3OutputPath',
            'properties.FinalMetricDataList[*].Value'
        ]
        
        suggestions = self.validator._get_property_path_suggestions(
            'properties.ModelArtifacts.InvalidPath',
            all_valid_paths
        )
        
        assert len(suggestions) > 0
        # Should suggest the most similar path first
        assert 'properties.ModelArtifacts.S3ModelArtifacts' in suggestions
    
    def test_calculate_path_similarity(self):
        """Test path similarity calculation."""
        # Identical paths
        similarity = self.validator._calculate_path_similarity(
            'properties.modelartifacts.s3modelartifacts',
            'properties.modelartifacts.s3modelartifacts'
        )
        assert similarity == 1.0
        
        # Similar paths
        similarity = self.validator._calculate_path_similarity(
            'properties.modelartifacts.invalidpath',
            'properties.modelartifacts.s3modelartifacts'
        )
        assert 0.5 < similarity < 1.0
        
        # Completely different paths
        similarity = self.validator._calculate_path_similarity(
            'properties.modelartifacts.s3modelartifacts',
            'outputparameters[0]'
        )
        assert similarity < 0.5
    
    def test_get_step_type_documentation(self):
        """Test getting step type documentation."""
        doc_info = self.validator.get_step_type_documentation('training', 'training')
        
        assert doc_info['step_type'] == 'training'
        assert doc_info['node_type'] == 'training'
        assert 'documentation_url' in doc_info
        assert 'valid_property_paths' in doc_info
        assert doc_info['total_valid_paths'] > 0
        assert len(doc_info['categories']) > 0
    
    def test_list_supported_step_types(self):
        """Test listing supported step types."""
        supported_types = self.validator.list_supported_step_types()
        
        assert len(supported_types) > 0
        
        # Check that all expected step types are present
        step_type_names = [step['step_type'] for step in supported_types]
        expected_types = ['training', 'processing', 'transform', 'tuning', 'create_model', 'lambda', 'callback']
        
        for expected_type in expected_types:
            assert expected_type in step_type_names
        
        # Check that each step type has documentation
        for step_info in supported_types:
            assert 'description' in step_info
            assert 'documentation_url' in step_info
            assert 'valid_property_paths' in step_info


class TestValidatePropertyPathsConvenienceFunction:
    """Test the convenience function for property path validation."""
    
    def test_validate_property_paths_function(self):
        """Test the convenience function."""
        specification = {
            'step_type': 'training',
            'outputs': [
                {
                    'logical_name': 'model_artifacts',
                    'property_path': 'properties.ModelArtifacts.S3ModelArtifacts'
                }
            ]
        }
        
        issues = validate_property_paths(specification, "test_contract")
        
        assert len(issues) > 0
        # Should have at least step type resolution and validation results
        categories = [issue['category'] for issue in issues]
        assert 'step_type_resolution' in categories


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = SageMakerPropertyPathValidator()
    
    def test_validate_specification_with_no_outputs(self):
        """Test validation with specification that has no outputs."""
        specification = {
            'step_type': 'training',
            'outputs': []
        }
        
        issues = self.validator.validate_specification_property_paths(specification, "no_outputs")
        
        # Should still have step type resolution and summary
        assert len(issues) >= 1
        summary_issues = [issue for issue in issues if 'summary' in issue['category']]
        assert len(summary_issues) == 1
        assert summary_issues[0]['details']['total_outputs'] == 0
    
    def test_validate_specification_with_outputs_no_property_paths(self):
        """Test validation with outputs that have no property paths."""
        specification = {
            'step_type': 'training',
            'outputs': [
                {
                    'logical_name': 'output1'
                    # No property_path field
                }
            ]
        }
        
        issues = self.validator.validate_specification_property_paths(specification, "no_paths")
        
        # Should have summary showing 0 outputs with property paths
        summary_issues = [issue for issue in issues if 'summary' in issue['category']]
        assert len(summary_issues) == 1
        assert summary_issues[0]['details']['outputs_with_property_paths'] == 0
    
    def test_validate_specification_with_empty_property_path(self):
        """Test validation with empty property path."""
        specification = {
            'step_type': 'training',
            'outputs': [
                {
                    'logical_name': 'output1',
                    'property_path': ''
                }
            ]
        }
        
        issues = self.validator.validate_specification_property_paths(specification, "empty_path")
        
        # Should not validate empty property paths
        validation_issues = [issue for issue in issues if issue['category'] == 'property_path_validation']
        assert len(validation_issues) == 0
    
    def test_validate_specification_with_malformed_data(self):
        """Test validation with malformed specification data."""
        specification = {
            'step_type': 'training',
            'outputs': [
                {
                    'logical_name': None,  # Malformed data
                    'property_path': 'properties.ModelArtifacts.S3ModelArtifacts'
                },
                None,  # Malformed output entry
                {
                    'logical_name': 'valid_output',
                    'property_path': 'properties.ModelArtifacts.S3ModelArtifacts'
                }
            ]
        }
        
        # Should handle malformed data gracefully
        issues = self.validator.validate_specification_property_paths(specification, "malformed")
        
        # Should still process valid entries
        validation_issues = [issue for issue in issues if issue['category'] == 'property_path_validation']
        assert len(validation_issues) >= 1
    
    @patch('cursus.validation.alignment.property_path_validator.get_step_name_from_spec_type')
    def test_step_registry_exception_handling(self, mock_get_step_name):
        """Test handling of exceptions in step registry integration."""
        # Mock registry function to raise exception
        mock_get_step_name.side_effect = Exception("Registry error")
        
        specification = {
            'step_type': 'SomeStep_Training',
            'outputs': []
        }
        
        issues = self.validator.validate_specification_property_paths(specification, "registry_error")
        
        # Should handle exception gracefully
        resolution_issues = [issue for issue in issues if issue['category'] == 'step_type_resolution']
        assert len(resolution_issues) == 1
        assert resolution_issues[0]['severity'] == 'WARNING'
        assert 'failed' in resolution_issues[0]['message']
    
    def test_pattern_matching_with_invalid_regex(self):
        """Test pattern matching with patterns that could cause regex errors."""
        # Test with patterns that might cause regex issues
        test_cases = [
            ('properties.Test[*].Value', 'properties.Test[0].Value'),
            ('properties.Test[*].Value', 'properties.Test["key"].Value'),
            ('properties.Test[*].Value', 'properties.Test[*].Value'),
        ]
        
        for pattern, test_path in test_cases:
            # Should not raise exceptions
            result = self.validator._matches_property_path_pattern(test_path, pattern)
            assert isinstance(result, bool)
    
    def test_similarity_calculation_with_empty_paths(self):
        """Test similarity calculation with empty or None paths."""
        # Should handle empty paths gracefully
        similarity = self.validator._calculate_path_similarity('', '')
        assert similarity == 0.0
        
        similarity = self.validator._calculate_path_similarity('properties.test', '')
        assert similarity == 0.0
    
    def test_property_path_suggestions_with_no_valid_paths(self):
        """Test suggestion generation with no valid paths."""
        suggestions = self.validator._get_property_path_suggestions('invalid.path', [])
        assert suggestions == []
    
    def test_validate_single_property_path_with_empty_valid_paths(self):
        """Test single property path validation with empty valid paths."""
        result = self.validator._validate_single_property_path(
            'properties.Test.Path',
            'training',
            'training',
            {}
        )
        
        assert result['valid'] is False
        assert result['suggestions'] == []
    
    def test_case_insensitive_step_type_matching(self):
        """Test case insensitive step type matching."""
        # Test with various case combinations
        test_cases = [
            ('TRAINING', 'TRAINING'),
            ('Training', 'training'),
            ('processing', 'PROCESSING'),
            ('Transform', 'transform')
        ]
        
        for step_type, node_type in test_cases:
            paths = self.validator._get_valid_property_paths_for_step_type(step_type, node_type)
            assert len(paths) > 0  # Should find valid paths regardless of case


if __name__ == '__main__':
    pytest.main([__file__])
