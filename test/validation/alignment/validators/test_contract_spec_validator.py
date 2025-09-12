"""
Unit tests for cursus.validation.alignment.validators.contract_spec_validator module.

Tests the ContractSpecValidator class that handles core validation logic for
contract-specification alignment including data type validation, input/output
alignment, and logical name validation.
"""

import pytest
from typing import Dict, Any, List

from cursus.validation.alignment.validators.contract_spec_validator import (
    ContractSpecValidator,
)


@pytest.fixture
def validator():
    """Set up ContractSpecValidator fixture."""
    return ContractSpecValidator()


@pytest.fixture
def sample_contract():
    """Set up sample contract data fixture."""
    return {
        "entry_point": "model_training.py",
        "inputs": {
            "training_data": {"path": "/opt/ml/input/data/training"},
            "validation_data": {"path": "/opt/ml/input/data/validation"},
        },
        "outputs": {
            "model": {"path": "/opt/ml/model"},
            "metrics": {"path": "/opt/ml/output/metrics.json"},
        },
        "arguments": {
            "learning_rate": {"default": 0.01, "required": False},
            "max_depth": {"default": 6, "required": False},
        },
    }


@pytest.fixture
def sample_specification():
    """Set up sample specification data fixture."""
    return {
        "step_type": "ModelTraining",
        "node_type": "ProcessingJob",
        "dependencies": [
            {
                "logical_name": "training_data",
                "dependency_type": "InputData",
                "required": True,
                "compatible_sources": ["S3"],
                "data_type": "tabular",
                "description": "Training dataset",
            },
            {
                "logical_name": "validation_data",
                "dependency_type": "InputData",
                "required": False,
                "compatible_sources": ["S3"],
                "data_type": "tabular",
                "description": "Validation dataset",
            },
        ],
        "outputs": [
            {
                "logical_name": "model",
                "output_type": "Model",
                "property_path": "/opt/ml/model",
                "data_type": "model",
                "description": "Trained model",
            },
            {
                "logical_name": "metrics",
                "output_type": "Metrics",
                "property_path": "/opt/ml/output/metrics.json",
                "data_type": "json",
                "description": "Training metrics",
            },
        ],
    }


class TestContractSpecValidator:
    """Test cases for ContractSpecValidator class."""

    def test_validate_logical_names_perfect_match(
        self, validator, sample_contract, sample_specification
    ):
        """Test logical name validation when contract and spec match perfectly."""
        issues = validator.validate_logical_names(
            sample_contract, sample_specification, "model_training"
        )

        assert issues == []

    def test_validate_logical_names_missing_contract_input(
        self, validator, sample_contract, sample_specification
    ):
        """Test validation when contract is missing an input declared in spec."""
        # Remove training_data from contract inputs
        contract_missing_input = sample_contract.copy()
        contract_missing_input["inputs"] = {
            "validation_data": {"path": "/opt/ml/input/data/validation"}
        }

        issues = validator.validate_logical_names(
            contract_missing_input, sample_specification, "model_training"
        )

        assert (
            len(issues) == 0
        )  # This should not generate issues since spec deps should match contract inputs

    def test_validate_logical_names_extra_contract_input(
        self, validator, sample_contract, sample_specification
    ):
        """Test validation when contract has extra input not in spec."""
        # Add extra input to contract
        contract_extra_input = sample_contract.copy()
        contract_extra_input["inputs"] = {
            "training_data": {"path": "/opt/ml/input/data/training"},
            "validation_data": {"path": "/opt/ml/input/data/validation"},
            "extra_data": {"path": "/opt/ml/input/data/extra"},
        }

        issues = validator.validate_logical_names(
            contract_extra_input, sample_specification, "model_training"
        )

        # Should have one error for extra input not in spec
        assert len(issues) == 1
        assert issues[0]["severity"] == "ERROR"
        assert issues[0]["category"] == "logical_names"
        assert "extra_data" in issues[0]["message"]
        assert "not declared as specification dependency" in issues[0]["message"]

    def test_validate_logical_names_missing_contract_output(
        self, validator, sample_contract, sample_specification
    ):
        """Test validation when contract is missing an output declared in spec."""
        # Remove model from contract outputs
        contract_missing_output = sample_contract.copy()
        contract_missing_output["outputs"] = {
            "metrics": {"path": "/opt/ml/output/metrics.json"}
        }

        issues = validator.validate_logical_names(
            contract_missing_output, sample_specification, "model_training"
        )

        assert (
            len(issues) == 0
        )  # This should not generate issues since spec outputs should match contract outputs

    def test_validate_logical_names_extra_contract_output(
        self, validator, sample_contract, sample_specification
    ):
        """Test validation when contract has extra output not in spec."""
        # Add extra output to contract
        contract_extra_output = sample_contract.copy()
        contract_extra_output["outputs"] = {
            "model": {"path": "/opt/ml/model"},
            "metrics": {"path": "/opt/ml/output/metrics.json"},
            "extra_output": {"path": "/opt/ml/output/extra.json"},
        }

        issues = validator.validate_logical_names(
            contract_extra_output, sample_specification, "model_training"
        )

        # Should have one error for extra output not in spec
        assert len(issues) == 1
        assert issues[0]["severity"] == "ERROR"
        assert issues[0]["category"] == "logical_names"
        assert "extra_output" in issues[0]["message"]
        assert "not declared as specification output" in issues[0]["message"]

    def test_validate_logical_names_empty_contract(
        self, validator, sample_specification
    ):
        """Test validation with empty contract."""
        empty_contract = {"entry_point": "test.py", "inputs": {}, "outputs": {}}

        issues = validator.validate_logical_names(
            empty_contract, sample_specification, "test_contract"
        )

        # Should have no issues since empty contract has no extra inputs/outputs
        assert len(issues) == 0

    def test_validate_logical_names_empty_specification(
        self, validator, sample_contract
    ):
        """Test validation with empty specification."""
        empty_spec = {
            "step_type": "EmptyStep",
            "node_type": "ProcessingJob",
            "dependencies": [],
            "outputs": [],
        }

        issues = validator.validate_logical_names(
            sample_contract, empty_spec, "model_training"
        )

        # Should have errors for all contract inputs/outputs not in empty spec
        assert len(issues) == 4  # 2 inputs + 2 outputs

        # Check that all issues are about missing declarations
        for issue in issues:
            assert issue["severity"] == "ERROR"
            assert issue["category"] == "logical_names"
            assert "not declared as specification" in issue["message"]

    def test_validate_logical_names_with_job_type(
        self, validator, sample_contract, sample_specification
    ):
        """Test validation with job type parameter."""
        issues = validator.validate_logical_names(
            sample_contract, sample_specification, "model_training", job_type="training"
        )

        # Job type should not affect basic logical name validation
        assert issues == []

    def test_validate_logical_names_missing_logical_name_in_spec(
        self, validator, sample_contract
    ):
        """Test validation when spec dependencies/outputs are missing logical_name."""
        spec_missing_names = {
            "step_type": "TestStep",
            "node_type": "ProcessingJob",
            "dependencies": [
                {
                    "dependency_type": "InputData",
                    "required": True,
                    # Missing logical_name
                }
            ],
            "outputs": [
                {
                    "output_type": "Model",
                    "property_path": "/opt/ml/model",
                    # Missing logical_name
                }
            ],
        }

        issues = validator.validate_logical_names(
            sample_contract, spec_missing_names, "model_training"
        )

        # Should have errors for all contract inputs/outputs since spec has no logical names
        assert len(issues) == 4  # 2 inputs + 2 outputs

    def test_validate_data_types_basic(
        self, validator, sample_contract, sample_specification
    ):
        """Test basic data type validation."""
        issues = validator.validate_data_types(
            sample_contract, sample_specification, "model_training"
        )

        # Currently returns empty list as noted in the implementation
        assert issues == []

    def test_validate_data_types_empty_inputs(self, validator, sample_specification):
        """Test data type validation with empty inputs."""
        empty_contract = {"entry_point": "test.py", "inputs": {}, "outputs": {}}

        issues = validator.validate_data_types(
            empty_contract, sample_specification, "test_contract"
        )

        assert issues == []

    def test_validate_input_output_alignment_perfect_match(
        self, validator, sample_contract, sample_specification
    ):
        """Test input/output alignment when contract and spec match perfectly."""
        issues = validator.validate_input_output_alignment(
            sample_contract, sample_specification, "model_training"
        )

        assert issues == []

    def test_validate_input_output_alignment_unmatched_dependency(
        self, validator, sample_contract, sample_specification
    ):
        """Test alignment when spec has dependency without contract input."""
        # Add extra dependency to spec
        spec_extra_dep = sample_specification.copy()
        spec_extra_dep["dependencies"] = sample_specification["dependencies"] + [
            {
                "logical_name": "extra_dependency",
                "dependency_type": "InputData",
                "required": True,
                "compatible_sources": ["S3"],
                "data_type": "tabular",
                "description": "Extra dependency",
            }
        ]

        issues = validator.validate_input_output_alignment(
            sample_contract, spec_extra_dep, "model_training"
        )

        # Should have one warning for unmatched dependency
        assert len(issues) == 1
        assert issues[0]["severity"] == "WARNING"
        assert issues[0]["category"] == "input_output_alignment"
        assert "extra_dependency" in issues[0]["message"]
        assert "has no corresponding contract input" in issues[0]["message"]

    def test_validate_input_output_alignment_unmatched_output(
        self, validator, sample_contract, sample_specification
    ):
        """Test alignment when spec has output without contract output."""
        # Add extra output to spec
        spec_extra_output = sample_specification.copy()
        spec_extra_output["outputs"] = sample_specification["outputs"] + [
            {
                "logical_name": "extra_output",
                "output_type": "Data",
                "property_path": "/opt/ml/output/extra.json",
                "data_type": "json",
                "description": "Extra output",
            }
        ]

        issues = validator.validate_input_output_alignment(
            sample_contract, spec_extra_output, "model_training"
        )

        # Should have one warning for unmatched output
        assert len(issues) == 1
        assert issues[0]["severity"] == "WARNING"
        assert issues[0]["category"] == "input_output_alignment"
        assert "extra_output" in issues[0]["message"]
        assert "has no corresponding contract output" in issues[0]["message"]

    def test_validate_input_output_alignment_none_logical_names(
        self, validator, sample_contract
    ):
        """Test alignment when spec has None logical names."""
        spec_with_nones = {
            "step_type": "TestStep",
            "node_type": "ProcessingJob",
            "dependencies": [
                {
                    "logical_name": None,
                    "dependency_type": "InputData",
                    "required": True,
                },
                {
                    "logical_name": "valid_dep",
                    "dependency_type": "InputData",
                    "required": True,
                },
            ],
            "outputs": [
                {
                    "logical_name": None,
                    "output_type": "Model",
                    "property_path": "/opt/ml/model",
                },
                {
                    "logical_name": "valid_output",
                    "output_type": "Data",
                    "property_path": "/opt/ml/output/data.json",
                },
            ],
        }

        issues = validator.validate_input_output_alignment(
            sample_contract, spec_with_nones, "model_training"
        )

        # Should have warnings for valid_dep and valid_output, but not for None values
        assert len(issues) == 2
        for issue in issues:
            assert issue["severity"] == "WARNING"
            assert issue["category"] == "input_output_alignment"
            assert "valid_" in issue["message"]

    def test_validate_input_output_alignment_empty_spec(
        self, validator, sample_contract
    ):
        """Test alignment with empty specification."""
        empty_spec = {
            "step_type": "EmptyStep",
            "node_type": "ProcessingJob",
            "dependencies": [],
            "outputs": [],
        }

        issues = validator.validate_input_output_alignment(
            sample_contract, empty_spec, "model_training"
        )

        # Should have no issues since empty spec has no unmatched dependencies/outputs
        assert issues == []

    def test_validate_input_output_alignment_empty_contract(
        self, validator, sample_specification
    ):
        """Test alignment with empty contract."""
        empty_contract = {"entry_point": "test.py", "inputs": {}, "outputs": {}}

        issues = validator.validate_input_output_alignment(
            empty_contract, sample_specification, "test_contract"
        )

        # Should have warnings for all spec dependencies and outputs
        assert len(issues) == 4  # 2 dependencies + 2 outputs
        for issue in issues:
            assert issue["severity"] == "WARNING"
            assert issue["category"] == "input_output_alignment"
            assert "has no corresponding contract" in issue["message"]

    def test_validate_input_output_alignment_missing_keys(
        self, validator, sample_specification
    ):
        """Test alignment when contract is missing inputs/outputs keys."""
        incomplete_contract = {
            "entry_point": "test.py"
            # Missing inputs and outputs keys
        }

        issues = validator.validate_input_output_alignment(
            incomplete_contract, sample_specification, "test_contract"
        )

        # Should handle missing keys gracefully and report unmatched spec items
        assert len(issues) == 4  # 2 dependencies + 2 outputs
        for issue in issues:
            assert issue["severity"] == "WARNING"
            assert issue["category"] == "input_output_alignment"


class TestContractSpecValidatorEdgeCases:
    """Test cases for edge cases and error conditions."""

    def test_validate_with_malformed_contract(self, validator):
        """Test validation with malformed contract data."""
        malformed_contract = {
            "entry_point": "test.py",
            "inputs": None,  # Should be dict
            "outputs": "invalid",  # Should be dict
        }

        spec = {
            "step_type": "TestStep",
            "node_type": "ProcessingJob",
            "dependencies": [],
            "outputs": [],
        }

        # Should handle malformed data gracefully
        issues = validator.validate_logical_names(malformed_contract, spec, "test")
        assert isinstance(issues, list)

        issues = validator.validate_input_output_alignment(
            malformed_contract, spec, "test"
        )
        assert isinstance(issues, list)

    def test_validate_with_malformed_specification(self, validator):
        """Test validation with malformed specification data."""
        contract = {"entry_point": "test.py", "inputs": {}, "outputs": {}}

        malformed_spec = {
            "step_type": "TestStep",
            "node_type": "ProcessingJob",
            "dependencies": None,  # Should be list
            "outputs": "invalid",  # Should be list
        }

        # Should handle malformed data gracefully
        issues = validator.validate_logical_names(contract, malformed_spec, "test")
        assert isinstance(issues, list)

        issues = validator.validate_input_output_alignment(
            contract, malformed_spec, "test"
        )
        assert isinstance(issues, list)

    def test_validate_with_complex_logical_names(self, validator):
        """Test validation with complex logical names containing special characters."""
        contract = {
            "entry_point": "test.py",
            "inputs": {
                "input-with-dashes": {"path": "/path1"},
                "input_with_underscores": {"path": "/path2"},
                "input.with.dots": {"path": "/path3"},
            },
            "outputs": {"output-complex-name": {"path": "/output1"}},
        }

        spec = {
            "step_type": "TestStep",
            "node_type": "ProcessingJob",
            "dependencies": [
                {"logical_name": "input-with-dashes", "dependency_type": "InputData"},
                {
                    "logical_name": "input_with_underscores",
                    "dependency_type": "InputData",
                },
                {"logical_name": "input.with.dots", "dependency_type": "InputData"},
            ],
            "outputs": [{"logical_name": "output-complex-name", "output_type": "Data"}],
        }

        issues = validator.validate_logical_names(contract, spec, "test")
        assert issues == []

        issues = validator.validate_input_output_alignment(contract, spec, "test")
        assert issues == []

    def test_issue_format_consistency(self, validator):
        """Test that all validation methods return consistently formatted issues."""
        contract = {
            "entry_point": "test.py",
            "inputs": {"extra_input": {"path": "/path"}},
            "outputs": {"extra_output": {"path": "/path"}},
        }

        spec = {
            "step_type": "TestStep",
            "node_type": "ProcessingJob",
            "dependencies": [
                {"logical_name": "spec_dep", "dependency_type": "InputData"}
            ],
            "outputs": [{"logical_name": "spec_output", "output_type": "Data"}],
        }

        # Test all validation methods
        logical_issues = validator.validate_logical_names(contract, spec, "test")
        data_type_issues = validator.validate_data_types(contract, spec, "test")
        alignment_issues = validator.validate_input_output_alignment(
            contract, spec, "test"
        )

        all_issues = logical_issues + data_type_issues + alignment_issues

        # Check that all issues have required fields
        for issue in all_issues:
            assert "severity" in issue
            assert "category" in issue
            assert "message" in issue
            assert "details" in issue
            assert "recommendation" in issue

            # Check severity values are valid
            assert issue["severity"] in ["ERROR", "WARNING", "INFO"]

            # Check that details contain contract name
            assert "contract" in issue["details"]
            assert issue["details"]["contract"] == "test"


if __name__ == "__main__":
    pytest.main([__file__])
