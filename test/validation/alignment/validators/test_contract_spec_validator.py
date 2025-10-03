"""
Test Contract Spec Validator Module

Tests for restored contract-specification validation logic.
Tests logical name validation and input/output alignment validation.
"""

import pytest
from unittest.mock import patch, MagicMock

from cursus.validation.alignment.validators.contract_spec_validator import ConsolidatedContractSpecValidator


class TestConsolidatedContractSpecValidator:
    """Test ConsolidatedContractSpecValidator functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ConsolidatedContractSpecValidator()

    def test_validate_logical_names_success(self):
        """Test successful logical name validation."""
        contract = {
            "inputs": {
                "training_data": {"type": "S3Uri"},
                "validation_data": {"type": "S3Uri"}
            },
            "outputs": {
                "model": {"type": "S3Uri"},
                "metrics": {"type": "S3Uri"}
            }
        }
        
        specification = {
            "dependencies": [
                {"logical_name": "training_data", "type": "S3Uri"},
                {"logical_name": "validation_data", "type": "S3Uri"}
            ],
            "outputs": [
                {"logical_name": "model", "type": "S3Uri"},
                {"logical_name": "metrics", "type": "S3Uri"}
            ]
        }
        
        issues = self.validator.validate_logical_names(contract, specification, "test_contract")
        
        assert len(issues) == 0

    def test_validate_logical_names_missing_dependencies(self):
        """Test logical name validation with missing dependencies."""
        contract = {
            "inputs": {
                "training_data": {"type": "S3Uri"},
                "validation_data": {"type": "S3Uri"},
                "undeclared_input": {"type": "S3Uri"}  # Not in spec dependencies
            },
            "outputs": {
                "model": {"type": "S3Uri"}
            }
        }
        
        specification = {
            "dependencies": [
                {"logical_name": "training_data", "type": "S3Uri"},
                {"logical_name": "validation_data", "type": "S3Uri"}
            ],
            "outputs": [
                {"logical_name": "model", "type": "S3Uri"}
            ]
        }
        
        issues = self.validator.validate_logical_names(contract, specification, "test_contract")
        
        assert len(issues) == 1
        assert issues[0]["severity"] == "ERROR"
        assert issues[0]["category"] == "logical_names"
        assert "undeclared_input" in issues[0]["message"]
        assert "not declared as specification dependency" in issues[0]["message"]

    def test_validate_logical_names_missing_outputs(self):
        """Test logical name validation with missing outputs."""
        contract = {
            "inputs": {
                "training_data": {"type": "S3Uri"}
            },
            "outputs": {
                "model": {"type": "S3Uri"},
                "undeclared_output": {"type": "S3Uri"}  # Not in spec outputs
            }
        }
        
        specification = {
            "dependencies": [
                {"logical_name": "training_data", "type": "S3Uri"}
            ],
            "outputs": [
                {"logical_name": "model", "type": "S3Uri"}
            ]
        }
        
        issues = self.validator.validate_logical_names(contract, specification, "test_contract")
        
        assert len(issues) == 1
        assert issues[0]["severity"] == "ERROR"
        assert issues[0]["category"] == "logical_names"
        assert "undeclared_output" in issues[0]["message"]
        assert "not declared as specification output" in issues[0]["message"]

    def test_validate_logical_names_malformed_contract(self):
        """Test logical name validation with malformed contract data."""
        # Contract with non-dict inputs/outputs
        malformed_contract = {
            "inputs": "not_a_dict",
            "outputs": ["not", "a", "dict"]
        }
        
        specification = {
            "dependencies": [
                {"logical_name": "training_data", "type": "S3Uri"}
            ],
            "outputs": [
                {"logical_name": "model", "type": "S3Uri"}
            ]
        }
        
        # Should handle malformed data gracefully
        issues = self.validator.validate_input_output_alignment(malformed_contract, specification, "test_contract")

        # Should report missing dependencies/outputs since malformed data is treated as empty
        assert len(issues) == 2  # Missing training_data dependency and model output

    def test_validate_logical_names_malformed_specification(self):
        """Test logical name validation with malformed specification data."""
        contract = {
            "inputs": {
                "training_data": {"type": "S3Uri"}
            },
            "outputs": {
                "model": {"type": "S3Uri"}
            }
        }
        
        # Specification with malformed dependencies/outputs
        malformed_specification = {
            "dependencies": "not_a_list",
            "outputs": [
                {"missing_logical_name": "model"},  # Missing logical_name field
                "not_a_dict"  # Not a dict
            ]
        }
        
        # Should handle malformed data gracefully
        issues = self.validator.validate_logical_names(contract, malformed_specification, "test_contract")
        
        # Should report contract inputs/outputs not found in spec
        assert len(issues) == 2  # training_data and model not found in malformed spec

    def test_validate_input_output_alignment_success(self):
        """Test successful input/output alignment validation."""
        contract = {
            "inputs": {
                "training_data": {"type": "S3Uri"},
                "validation_data": {"type": "S3Uri"}
            },
            "outputs": {
                "model": {"type": "S3Uri"}
            }
        }
        
        specification = {
            "dependencies": [
                {"logical_name": "training_data", "type": "S3Uri"},
                {"logical_name": "validation_data", "type": "S3Uri"}
            ],
            "outputs": [
                {"logical_name": "model", "type": "S3Uri"}
            ]
        }
        
        issues = self.validator.validate_input_output_alignment(contract, specification, "test_contract")
        
        assert len(issues) == 0

    def test_validate_input_output_alignment_unmatched_dependencies(self):
        """Test input/output alignment with unmatched dependencies."""
        contract = {
            "inputs": {
                "training_data": {"type": "S3Uri"}
                # Missing validation_data input
            },
            "outputs": {
                "model": {"type": "S3Uri"}
            }
        }
        
        specification = {
            "dependencies": [
                {"logical_name": "training_data", "type": "S3Uri"},
                {"logical_name": "validation_data", "type": "S3Uri"}  # No corresponding contract input
            ],
            "outputs": [
                {"logical_name": "model", "type": "S3Uri"}
            ]
        }
        
        issues = self.validator.validate_input_output_alignment(contract, specification, "test_contract")
        
        assert len(issues) == 1
        assert issues[0]["severity"] == "WARNING"
        assert issues[0]["category"] == "input_output_alignment"
        assert "validation_data" in issues[0]["message"]
        assert "has no corresponding contract input" in issues[0]["message"]

    def test_validate_input_output_alignment_unmatched_outputs(self):
        """Test input/output alignment with unmatched outputs."""
        contract = {
            "inputs": {
                "training_data": {"type": "S3Uri"}
            },
            "outputs": {
                "model": {"type": "S3Uri"}
                # Missing metrics output
            }
        }
        
        specification = {
            "dependencies": [
                {"logical_name": "training_data", "type": "S3Uri"}
            ],
            "outputs": [
                {"logical_name": "model", "type": "S3Uri"},
                {"logical_name": "metrics", "type": "S3Uri"}  # No corresponding contract output
            ]
        }
        
        issues = self.validator.validate_input_output_alignment(contract, specification, "test_contract")
        
        assert len(issues) == 1
        assert issues[0]["severity"] == "WARNING"
        assert issues[0]["category"] == "input_output_alignment"
        assert "metrics" in issues[0]["message"]
        assert "has no corresponding contract output" in issues[0]["message"]

    def test_validate_input_output_alignment_malformed_data(self):
        """Test input/output alignment with malformed data."""
        # Contract with malformed data
        malformed_contract = {
            "inputs": None,
            "outputs": "not_a_dict"
        }
        
        # Specification with malformed data
        malformed_specification = {
            "dependencies": [
                {"logical_name": "training_data"},  # Valid dependency
                {"missing_logical_name": "invalid"},  # Invalid dependency
                "not_a_dict"  # Invalid dependency
            ],
            "outputs": None
        }
        
        # Should handle malformed data gracefully
        issues = self.validator.validate_input_output_alignment(malformed_contract, malformed_specification, "test_contract")
        
        # Should report unmatched dependency (training_data has no contract input)
        assert len(issues) == 1
        assert issues[0]["severity"] == "WARNING"
        assert "training_data" in issues[0]["message"]

    def test_comprehensive_validation_workflow(self):
        """Test comprehensive validation workflow with both methods."""
        contract = {
            "inputs": {
                "training_data": {"type": "S3Uri"},
                "extra_contract_input": {"type": "S3Uri"}  # Not in spec
            },
            "outputs": {
                "model": {"type": "S3Uri"},
                "extra_contract_output": {"type": "S3Uri"}  # Not in spec
            }
        }
        
        specification = {
            "dependencies": [
                {"logical_name": "training_data", "type": "S3Uri"},
                {"logical_name": "extra_spec_dependency", "type": "S3Uri"}  # Not in contract
            ],
            "outputs": [
                {"logical_name": "model", "type": "S3Uri"},
                {"logical_name": "extra_spec_output", "type": "S3Uri"}  # Not in contract
            ]
        }
        
        # Test logical names validation
        logical_issues = self.validator.validate_logical_names(contract, specification, "test_contract")
        
        # Should have 2 errors: extra_contract_input and extra_contract_output not in spec
        assert len(logical_issues) == 2
        logical_categories = [issue["category"] for issue in logical_issues]
        assert all(cat == "logical_names" for cat in logical_categories)
        logical_severities = [issue["severity"] for issue in logical_issues]
        assert all(sev == "ERROR" for sev in logical_severities)
        
        # Test input/output alignment validation
        alignment_issues = self.validator.validate_input_output_alignment(contract, specification, "test_contract")
        
        # Should have 2 warnings: extra_spec_dependency and extra_spec_output not in contract
        assert len(alignment_issues) == 2
        alignment_categories = [issue["category"] for issue in alignment_issues]
        assert all(cat == "input_output_alignment" for cat in alignment_categories)
        alignment_severities = [issue["severity"] for issue in alignment_issues]
        assert all(sev == "WARNING" for sev in alignment_severities)
        
        # Combined validation should have 4 issues total
        all_issues = logical_issues + alignment_issues
        assert len(all_issues) == 4

    def test_empty_contract_and_specification(self):
        """Test validation with empty contract and specification."""
        empty_contract = {
            "inputs": {},
            "outputs": {}
        }
        
        empty_specification = {
            "dependencies": [],
            "outputs": []
        }
        
        logical_issues = self.validator.validate_logical_names(empty_contract, empty_specification, "empty_contract")
        alignment_issues = self.validator.validate_input_output_alignment(empty_contract, empty_specification, "empty_contract")
        
        # Should have no issues with empty but valid structures
        assert len(logical_issues) == 0
        assert len(alignment_issues) == 0

    def test_validation_with_job_type_parameter(self):
        """Test logical name validation with optional job_type parameter."""
        contract = {
            "inputs": {
                "training_data": {"type": "S3Uri"}
            },
            "outputs": {
                "model": {"type": "S3Uri"}
            }
        }
        
        specification = {
            "dependencies": [
                {"logical_name": "training_data", "type": "S3Uri"}
            ],
            "outputs": [
                {"logical_name": "model", "type": "S3Uri"}
            ]
        }
        
        # Test with job_type parameter (should work the same)
        issues = self.validator.validate_logical_names(
            contract, specification, "test_contract", job_type="Training"
        )
        
        assert len(issues) == 0

    def test_detailed_issue_structure(self):
        """Test that validation issues have correct structure and details."""
        contract = {
            "inputs": {
                "undeclared_input": {"type": "S3Uri"}
            },
            "outputs": {
                "undeclared_output": {"type": "S3Uri"}
            }
        }
        
        specification = {
            "dependencies": [],
            "outputs": []
        }
        
        issues = self.validator.validate_logical_names(contract, specification, "test_contract")
        
        assert len(issues) == 2
        
        for issue in issues:
            # Check required fields
            assert "severity" in issue
            assert "category" in issue
            assert "message" in issue
            assert "details" in issue
            assert "recommendation" in issue
            
            # Check details structure
            assert "logical_name" in issue["details"]
            assert "contract" in issue["details"]
            assert issue["details"]["contract"] == "test_contract"
            
            # Check severity and category
            assert issue["severity"] == "ERROR"
            assert issue["category"] == "logical_names"
            
            # Check recommendation format
            assert issue["recommendation"].startswith("Add")
