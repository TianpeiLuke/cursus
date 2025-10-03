"""
Test module for specification-dependency alignment validation.

Tests the core functionality of spec-dependency alignment validation,
including integration with DependencyValidator.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any, List

from cursus.validation.alignment.core.spec_dependency_alignment import SpecificationDependencyAlignmentTester
from cursus.validation.alignment.validators.dependency_validator import DependencyValidator


class TestSpecDependencyAlignment:
    """Test cases for SpecDependencyAlignment class."""

    @pytest.fixture
    def workspace_dirs(self):
        """Fixture providing workspace directories."""
        return ["/test/workspace1", "/test/workspace2"]

    @pytest.fixture
    def spec_dependency_alignment(self, workspace_dirs):
        """Fixture providing SpecificationDependencyAlignmentTester instance."""
        return SpecificationDependencyAlignmentTester(workspace_dirs=workspace_dirs)

    @pytest.fixture
    def sample_specification(self):
        """Fixture providing sample specification data."""
        return {
            "dependencies": [
                {
                    "logical_name": "training_data",
                    "type": "s3_path",
                    "source": "step_output",
                    "step_name": "preprocessing"
                },
                {
                    "logical_name": "validation_data", 
                    "type": "s3_path",
                    "source": "step_output",
                    "step_name": "data_split"
                },
                {
                    "logical_name": "hyperparameters",
                    "type": "json",
                    "source": "config",
                    "config_key": "model_params"
                }
            ],
            "outputs": [
                {
                    "logical_name": "model_artifacts",
                    "type": "s3_path",
                    "description": "Trained model artifacts"
                },
                {
                    "logical_name": "evaluation_metrics",
                    "type": "json",
                    "description": "Model evaluation results"
                }
            ]
        }

    @pytest.fixture
    def sample_dependency_info(self):
        """Fixture providing sample dependency resolution info."""
        return {
            "resolved_dependencies": {
                "training_data": {
                    "status": "resolved",
                    "source_step": "preprocessing",
                    "output_name": "processed_training_data",
                    "path": "/workspace/preprocessing/outputs/training.csv"
                },
                "validation_data": {
                    "status": "resolved", 
                    "source_step": "data_split",
                    "output_name": "validation_split",
                    "path": "/workspace/data_split/outputs/validation.csv"
                },
                "hyperparameters": {
                    "status": "resolved",
                    "source": "config",
                    "config_key": "model_params",
                    "value": {"learning_rate": 0.01, "epochs": 100}
                }
            },
            "unresolved_dependencies": [],
            "circular_dependencies": [],
            "missing_steps": []
        }

    def test_init_with_workspace_dirs(self, workspace_dirs):
        """Test SpecificationDependencyAlignmentTester initialization with workspace directories."""
        alignment = SpecificationDependencyAlignmentTester(workspace_dirs=workspace_dirs)
        assert alignment.workspace_dirs == workspace_dirs

    def test_init_without_workspace_dirs(self):
        """Test SpecificationDependencyAlignmentTester initialization without workspace directories."""
        alignment = SpecificationDependencyAlignmentTester()
        assert alignment.workspace_dirs == []

    @patch('cursus.validation.alignment.core.spec_dependency_alignment.StepCatalog')
    def test_step_catalog_initialization(self, mock_step_catalog, workspace_dirs):
        """Test that StepCatalog is properly initialized."""
        SpecificationDependencyAlignmentTester(workspace_dirs=workspace_dirs)
        mock_step_catalog.assert_called_once_with(workspace_dirs=workspace_dirs)

    def test_validate_specification_with_resolved_dependencies(self, spec_dependency_alignment, sample_specification, sample_dependency_info):
        """Test specification validation with all dependencies resolved."""
        spec_name = "test_spec"
        
        with patch.object(spec_dependency_alignment, '_load_specification') as mock_load_spec, \
             patch('cursus.validation.alignment.validators.dependency_validator.DependencyValidator') as mock_validator_class:
            
            # Setup mocks
            mock_load_spec.return_value = sample_specification
            
            mock_validator = Mock()
            mock_validator.resolve_dependencies.return_value = sample_dependency_info
            mock_validator.validate_dependency_resolution.return_value = []
            mock_validator_class.return_value = mock_validator
            
            # Execute validation
            result = spec_dependency_alignment.validate_specification(spec_name)
            
            # Verify results
            assert result["passed"] is True
            assert result["spec_name"] == spec_name
            assert len(result["issues"]) == 0
            assert result["dependency_resolution"] == sample_dependency_info
            
            # Verify DependencyValidator was called correctly
            mock_validator.resolve_dependencies.assert_called_once_with(sample_specification["dependencies"])
            mock_validator.validate_dependency_resolution.assert_called_once_with(sample_dependency_info)

    def test_validate_specification_with_unresolved_dependencies(self, spec_dependency_alignment, sample_specification):
        """Test specification validation with unresolved dependencies."""
        spec_name = "test_spec"
        
        # Mock dependency info with unresolved dependencies
        unresolved_dependency_info = {
            "resolved_dependencies": {
                "training_data": {
                    "status": "resolved",
                    "source_step": "preprocessing",
                    "output_name": "processed_training_data"
                }
            },
            "unresolved_dependencies": [
                {
                    "logical_name": "validation_data",
                    "reason": "source_step_not_found",
                    "missing_step": "data_split"
                },
                {
                    "logical_name": "hyperparameters",
                    "reason": "config_key_not_found",
                    "missing_config_key": "model_params"
                }
            ],
            "circular_dependencies": [],
            "missing_steps": ["data_split"]
        }
        
        validation_issues = [
            {
                "severity": "ERROR",
                "category": "unresolved_dependency",
                "message": "Dependency 'validation_data' cannot be resolved: source step 'data_split' not found",
                "details": {"logical_name": "validation_data", "missing_step": "data_split"},
                "recommendation": "Add step 'data_split' to the pipeline or update dependency source"
            },
            {
                "severity": "ERROR", 
                "category": "unresolved_dependency",
                "message": "Dependency 'hyperparameters' cannot be resolved: config key 'model_params' not found",
                "details": {"logical_name": "hyperparameters", "missing_config_key": "model_params"},
                "recommendation": "Add 'model_params' to configuration or update dependency config_key"
            }
        ]
        
        with patch.object(spec_dependency_alignment, '_load_specification') as mock_load_spec, \
             patch('cursus.validation.alignment.validators.dependency_validator.DependencyValidator') as mock_validator_class:
            
            # Setup mocks
            mock_load_spec.return_value = sample_specification
            
            mock_validator = Mock()
            mock_validator.resolve_dependencies.return_value = unresolved_dependency_info
            mock_validator.validate_dependency_resolution.return_value = validation_issues
            mock_validator_class.return_value = mock_validator
            
            # Execute validation
            result = spec_dependency_alignment.validate_specification(spec_name)
            
            # Verify results
            assert result["passed"] is False
            assert len(result["issues"]) == 2
            assert all(issue["severity"] == "ERROR" for issue in result["issues"])
            assert result["dependency_resolution"]["unresolved_dependencies"] == unresolved_dependency_info["unresolved_dependencies"]

    def test_validate_specification_with_circular_dependencies(self, spec_dependency_alignment, sample_specification):
        """Test specification validation with circular dependencies."""
        spec_name = "test_spec"
        
        # Mock dependency info with circular dependencies
        circular_dependency_info = {
            "resolved_dependencies": {},
            "unresolved_dependencies": [],
            "circular_dependencies": [
                {
                    "cycle": ["step_a", "step_b", "step_c", "step_a"],
                    "dependencies": [
                        {"step": "step_a", "depends_on": "step_b"},
                        {"step": "step_b", "depends_on": "step_c"},
                        {"step": "step_c", "depends_on": "step_a"}
                    ]
                }
            ],
            "missing_steps": []
        }
        
        validation_issues = [
            {
                "severity": "CRITICAL",
                "category": "circular_dependency",
                "message": "Circular dependency detected: step_a -> step_b -> step_c -> step_a",
                "details": {"cycle": ["step_a", "step_b", "step_c", "step_a"]},
                "recommendation": "Remove circular dependency by restructuring pipeline dependencies"
            }
        ]
        
        with patch.object(spec_dependency_alignment, '_load_specification') as mock_load_spec, \
             patch('cursus.validation.alignment.validators.dependency_validator.DependencyValidator') as mock_validator_class:
            
            # Setup mocks
            mock_load_spec.return_value = sample_specification
            
            mock_validator = Mock()
            mock_validator.resolve_dependencies.return_value = circular_dependency_info
            mock_validator.validate_dependency_resolution.return_value = validation_issues
            mock_validator_class.return_value = mock_validator
            
            # Execute validation
            result = spec_dependency_alignment.validate_specification(spec_name)
            
            # Verify results
            assert result["passed"] is False
            assert len(result["issues"]) == 1
            assert result["issues"][0]["severity"] == "CRITICAL"
            assert result["issues"][0]["category"] == "circular_dependency"

    def test_validate_specification_with_missing_specification(self, spec_dependency_alignment):
        """Test specification validation when specification cannot be loaded."""
        spec_name = "test_spec"
        
        with patch.object(spec_dependency_alignment, '_load_specification') as mock_load_spec:
            # Setup mock to return None
            mock_load_spec.return_value = None
            
            # Execute validation
            result = spec_dependency_alignment.validate_specification(spec_name)
            
            # Verify results indicate failure due to missing specification
            assert result["passed"] is False
            assert len(result["issues"]) > 0
            assert any("specification" in issue.get("message", "").lower() for issue in result["issues"])

    def test_validate_specification_with_malformed_specification(self, spec_dependency_alignment):
        """Test specification validation with malformed specification data."""
        spec_name = "test_spec"
        malformed_spec = {"invalid": "structure"}
        
        with patch.object(spec_dependency_alignment, '_load_specification') as mock_load_spec, \
             patch('cursus.validation.alignment.validators.dependency_validator.DependencyValidator') as mock_validator_class:
            
            # Setup mocks
            mock_load_spec.return_value = malformed_spec
            
            mock_validator = Mock()
            mock_validator.resolve_dependencies.return_value = {
                "resolved_dependencies": {},
                "unresolved_dependencies": [],
                "circular_dependencies": [],
                "missing_steps": []
            }
            mock_validator.validate_dependency_resolution.return_value = []
            mock_validator_class.return_value = mock_validator
            
            # Execute validation
            result = spec_dependency_alignment.validate_specification(spec_name)
            
            # Verify validator handles malformed data gracefully
            mock_validator.resolve_dependencies.assert_called_once()

    def test_validate_specification_error_handling(self, spec_dependency_alignment):
        """Test specification validation error handling."""
        spec_name = "test_spec"
        
        with patch.object(spec_dependency_alignment, '_load_specification') as mock_load_spec:
            # Setup mock to raise exception
            mock_load_spec.side_effect = Exception("Test error")
            
            # Execute validation and verify it handles errors gracefully
            result = spec_dependency_alignment.validate_specification(spec_name)
            
            # Should return a result indicating failure
            assert result["passed"] is False
            assert len(result["issues"]) > 0

    def test_validate_specification_result_structure(self, spec_dependency_alignment, sample_specification, sample_dependency_info):
        """Test that validate_specification returns properly structured results."""
        spec_name = "test_spec"
        
        with patch.object(spec_dependency_alignment, '_load_specification') as mock_load_spec, \
             patch('cursus.validation.alignment.validators.dependency_validator.DependencyValidator') as mock_validator_class:
            
            # Setup mocks
            mock_load_spec.return_value = sample_specification
            
            mock_validator = Mock()
            mock_validator.resolve_dependencies.return_value = sample_dependency_info
            mock_validator.validate_dependency_resolution.return_value = []
            mock_validator_class.return_value = mock_validator
            
            # Execute validation
            result = spec_dependency_alignment.validate_specification(spec_name)
            
            # Verify result structure
            required_keys = ["passed", "issues", "spec_name", "specification", "dependency_resolution"]
            for key in required_keys:
                assert key in result
            
            assert isinstance(result["passed"], bool)
            assert isinstance(result["issues"], list)
            assert isinstance(result["spec_name"], str)
            assert isinstance(result["specification"], dict)
            assert isinstance(result["dependency_resolution"], dict)

    def test_integration_with_dependency_validator(self, spec_dependency_alignment, sample_specification):
        """Test integration with DependencyValidator."""
        spec_name = "test_spec"
        
        with patch.object(spec_dependency_alignment, '_load_specification') as mock_load_spec:
            # Setup mocks
            mock_load_spec.return_value = sample_specification
            
            # Execute validation (using real DependencyValidator)
            result = spec_dependency_alignment.validate_specification(spec_name)
            
            # Verify basic structure
            assert "passed" in result
            assert "issues" in result
            assert "dependency_resolution" in result
            assert isinstance(result["issues"], list)

    def test_workspace_directory_propagation(self, workspace_dirs):
        """Test that workspace directories are properly propagated."""
        alignment = SpecificationDependencyAlignmentTester(workspace_dirs=workspace_dirs)
        assert alignment.workspace_dirs == workspace_dirs

    def test_validate_specification_with_complex_dependencies(self, spec_dependency_alignment):
        """Test specification validation with complex dependency structures."""
        spec_name = "complex_spec"
        
        # Complex specification with many dependencies
        complex_spec = {
            "dependencies": [
                {
                    "logical_name": f"dep_{i}",
                    "type": "s3_path",
                    "source": "step_output",
                    "step_name": f"step_{i}"
                } for i in range(20)
            ] + [
                {
                    "logical_name": f"config_{i}",
                    "type": "json",
                    "source": "config",
                    "config_key": f"param_{i}"
                } for i in range(10)
            ],
            "outputs": [
                {
                    "logical_name": f"output_{i}",
                    "type": "s3_path",
                    "description": f"Output {i}"
                } for i in range(15)
            ]
        }
        
        # Complex dependency resolution
        complex_dependency_info = {
            "resolved_dependencies": {
                f"dep_{i}": {
                    "status": "resolved",
                    "source_step": f"step_{i}",
                    "output_name": f"output_{i}"
                } for i in range(20)
            },
            "unresolved_dependencies": [],
            "circular_dependencies": [],
            "missing_steps": []
        }
        
        with patch.object(spec_dependency_alignment, '_load_specification') as mock_load_spec, \
             patch('cursus.validation.alignment.validators.dependency_validator.DependencyValidator') as mock_validator_class:
            
            # Setup mocks
            mock_load_spec.return_value = complex_spec
            
            mock_validator = Mock()
            mock_validator.resolve_dependencies.return_value = complex_dependency_info
            mock_validator.validate_dependency_resolution.return_value = []
            mock_validator_class.return_value = mock_validator
            
            # Execute validation
            result = spec_dependency_alignment.validate_specification(spec_name)
            
            # Should handle complex dependencies without issues
            assert "passed" in result
            assert result["dependency_resolution"] == complex_dependency_info

    def test_validate_specification_with_mixed_dependency_types(self, spec_dependency_alignment):
        """Test specification validation with mixed dependency types and sources."""
        spec_name = "mixed_spec"
        
        mixed_spec = {
            "dependencies": [
                {
                    "logical_name": "step_output_dep",
                    "type": "s3_path",
                    "source": "step_output",
                    "step_name": "preprocessing"
                },
                {
                    "logical_name": "config_dep",
                    "type": "json",
                    "source": "config",
                    "config_key": "model_params"
                },
                {
                    "logical_name": "external_dep",
                    "type": "s3_path",
                    "source": "external",
                    "path": "s3://bucket/external/data.csv"
                }
            ],
            "outputs": [
                {
                    "logical_name": "mixed_output",
                    "type": "s3_path"
                }
            ]
        }
        
        mixed_dependency_info = {
            "resolved_dependencies": {
                "step_output_dep": {
                    "status": "resolved",
                    "source_step": "preprocessing",
                    "output_name": "processed_data"
                },
                "config_dep": {
                    "status": "resolved",
                    "source": "config",
                    "config_key": "model_params",
                    "value": {"lr": 0.01}
                },
                "external_dep": {
                    "status": "resolved",
                    "source": "external",
                    "path": "s3://bucket/external/data.csv"
                }
            },
            "unresolved_dependencies": [],
            "circular_dependencies": [],
            "missing_steps": []
        }
        
        with patch.object(spec_dependency_alignment, '_load_specification') as mock_load_spec, \
             patch('cursus.validation.alignment.validators.dependency_validator.DependencyValidator') as mock_validator_class:
            
            # Setup mocks
            mock_load_spec.return_value = mixed_spec
            
            mock_validator = Mock()
            mock_validator.resolve_dependencies.return_value = mixed_dependency_info
            mock_validator.validate_dependency_resolution.return_value = []
            mock_validator_class.return_value = mock_validator
            
            # Execute validation
            result = spec_dependency_alignment.validate_specification(spec_name)
            
            # Should handle mixed dependency types correctly
            assert result["passed"] is True
            assert len(result["dependency_resolution"]["resolved_dependencies"]) == 3

    def test_validate_specification_performance_with_large_spec(self, spec_dependency_alignment):
        """Test performance with large specification."""
        spec_name = "large_spec"
        
        # Create large specification
        large_spec = {
            "dependencies": [
                {
                    "logical_name": f"large_dep_{i}",
                    "type": "s3_path",
                    "source": "step_output",
                    "step_name": f"large_step_{i}"
                } for i in range(100)
            ],
            "outputs": [
                {
                    "logical_name": f"large_output_{i}",
                    "type": "s3_path"
                } for i in range(50)
            ]
        }
        
        large_dependency_info = {
            "resolved_dependencies": {
                f"large_dep_{i}": {
                    "status": "resolved",
                    "source_step": f"large_step_{i}"
                } for i in range(100)
            },
            "unresolved_dependencies": [],
            "circular_dependencies": [],
            "missing_steps": []
        }
        
        with patch.object(spec_dependency_alignment, '_load_specification') as mock_load_spec, \
             patch('cursus.validation.alignment.validators.dependency_validator.DependencyValidator') as mock_validator_class:
            
            # Setup mocks
            mock_load_spec.return_value = large_spec
            
            mock_validator = Mock()
            mock_validator.resolve_dependencies.return_value = large_dependency_info
            mock_validator.validate_dependency_resolution.return_value = []
            mock_validator_class.return_value = mock_validator
            
            # Execute validation and verify it completes
            result = spec_dependency_alignment.validate_specification(spec_name)
            
            # Should complete successfully
            assert "passed" in result
            assert "dependency_resolution" in result
            assert len(result["dependency_resolution"]["resolved_dependencies"]) == 100
