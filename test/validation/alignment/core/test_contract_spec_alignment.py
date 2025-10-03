"""
Test module for contract-specification alignment validation.

Tests the core functionality of contract-spec alignment validation,
including integration with ConsolidatedContractSpecValidator.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any, List

from cursus.validation.alignment.core.contract_spec_alignment import ContractSpecAlignment
from cursus.validation.alignment.validators.contract_spec_validator import ConsolidatedContractSpecValidator


class TestContractSpecAlignment:
    """Test cases for ContractSpecAlignment class."""

    @pytest.fixture
    def workspace_dirs(self):
        """Fixture providing workspace directories."""
        return ["/test/workspace1", "/test/workspace2"]

    @pytest.fixture
    def contract_spec_alignment(self, workspace_dirs):
        """Fixture providing ContractSpecAlignment instance."""
        return ContractSpecAlignment(workspace_dirs=workspace_dirs)

    @pytest.fixture
    def sample_contract(self):
        """Fixture providing sample contract data."""
        return {
            "inputs": {
                "training_data": {"type": "s3_path"},
                "validation_data": {"type": "s3_path"}
            },
            "outputs": {
                "model_artifacts": {"type": "s3_path"},
                "evaluation_report": {"type": "s3_path"}
            }
        }

    @pytest.fixture
    def sample_specification(self):
        """Fixture providing sample specification data."""
        return {
            "dependencies": [
                {"logical_name": "training_data", "type": "s3_path"},
                {"logical_name": "validation_data", "type": "s3_path"}
            ],
            "outputs": [
                {"logical_name": "model_artifacts", "type": "s3_path"},
                {"logical_name": "evaluation_report", "type": "s3_path"}
            ]
        }

    def test_init_with_workspace_dirs(self, workspace_dirs):
        """Test ContractSpecAlignment initialization with workspace directories."""
        alignment = ContractSpecAlignment(workspace_dirs=workspace_dirs)
        assert alignment.workspace_dirs == workspace_dirs

    def test_init_without_workspace_dirs(self):
        """Test ContractSpecAlignment initialization without workspace directories."""
        alignment = ContractSpecAlignment()
        assert alignment.workspace_dirs == []

    @patch('cursus.validation.alignment.core.contract_spec_alignment.StepCatalog')
    def test_step_catalog_initialization(self, mock_step_catalog, workspace_dirs):
        """Test that StepCatalog is properly initialized."""
        ContractSpecAlignment(workspace_dirs=workspace_dirs)
        mock_step_catalog.assert_called_once_with(workspace_dirs=workspace_dirs)

    def test_validate_contract_with_valid_data(self, contract_spec_alignment, sample_contract, sample_specification):
        """Test contract validation with valid contract and specification data."""
        contract_name = "test_contract"
        
        with patch.object(contract_spec_alignment, '_load_contract') as mock_load_contract, \
             patch.object(contract_spec_alignment, '_load_unified_specification') as mock_load_spec, \
             patch('cursus.validation.alignment.validators.contract_spec_validator.ConsolidatedContractSpecValidator') as mock_validator_class:
            
            # Setup mocks
            mock_load_contract.return_value = sample_contract
            mock_load_spec.return_value = {"primary_spec": sample_specification}
            
            mock_validator = Mock()
            mock_validator.validate_logical_names.return_value = []
            mock_validator.validate_input_output_alignment.return_value = []
            mock_validator_class.return_value = mock_validator
            
            # Execute validation
            result = contract_spec_alignment.validate_contract(contract_name)
            
            # Verify results
            assert result["passed"] is True
            assert result["contract_name"] == contract_name
            assert len(result["issues"]) == 0
            
            # Verify validator was called correctly
            mock_validator.validate_logical_names.assert_called_once_with(
                sample_contract, sample_specification, contract_name
            )
            mock_validator.validate_input_output_alignment.assert_called_once_with(
                sample_contract, sample_specification, contract_name
            )

    def test_validate_contract_with_logical_name_issues(self, contract_spec_alignment, sample_contract, sample_specification):
        """Test contract validation when logical name validation finds issues."""
        contract_name = "test_contract"
        logical_issues = [
            {
                "severity": "ERROR",
                "category": "logical_names",
                "message": "Contract input missing_input not declared as specification dependency",
                "details": {"logical_name": "missing_input", "contract": contract_name},
                "recommendation": "Add missing_input to specification dependencies"
            }
        ]
        
        with patch.object(contract_spec_alignment, '_load_contract') as mock_load_contract, \
             patch.object(contract_spec_alignment, '_load_unified_specification') as mock_load_spec, \
             patch('cursus.validation.alignment.validators.contract_spec_validator.ConsolidatedContractSpecValidator') as mock_validator_class:
            
            # Setup mocks
            mock_load_contract.return_value = sample_contract
            mock_load_spec.return_value = {"primary_spec": sample_specification}
            
            mock_validator = Mock()
            mock_validator.validate_logical_names.return_value = logical_issues
            mock_validator.validate_input_output_alignment.return_value = []
            mock_validator_class.return_value = mock_validator
            
            # Execute validation
            result = contract_spec_alignment.validate_contract(contract_name)
            
            # Verify results
            assert result["passed"] is False
            assert len(result["issues"]) == 1
            assert result["issues"][0]["severity"] == "ERROR"
            assert result["issues"][0]["category"] == "logical_names"

    def test_validate_contract_with_io_alignment_issues(self, contract_spec_alignment, sample_contract, sample_specification):
        """Test contract validation when I/O alignment validation finds issues."""
        contract_name = "test_contract"
        io_issues = [
            {
                "severity": "WARNING",
                "category": "input_output_alignment",
                "message": "Specification dependency extra_dep has no corresponding contract input",
                "details": {"logical_name": "extra_dep", "contract": contract_name},
                "recommendation": "Add extra_dep to contract inputs or remove from specification dependencies"
            }
        ]
        
        with patch.object(contract_spec_alignment, '_load_contract') as mock_load_contract, \
             patch.object(contract_spec_alignment, '_load_unified_specification') as mock_load_spec, \
             patch('cursus.validation.alignment.validators.contract_spec_validator.ConsolidatedContractSpecValidator') as mock_validator_class:
            
            # Setup mocks
            mock_load_contract.return_value = sample_contract
            mock_load_spec.return_value = {"primary_spec": sample_specification}
            
            mock_validator = Mock()
            mock_validator.validate_logical_names.return_value = []
            mock_validator.validate_input_output_alignment.return_value = io_issues
            mock_validator_class.return_value = mock_validator
            
            # Execute validation
            result = contract_spec_alignment.validate_contract(contract_name)
            
            # Verify results
            assert result["passed"] is True  # Warnings don't fail validation
            assert len(result["issues"]) == 1
            assert result["issues"][0]["severity"] == "WARNING"
            assert result["issues"][0]["category"] == "input_output_alignment"

    def test_validate_contract_with_multiple_issues(self, contract_spec_alignment, sample_contract, sample_specification):
        """Test contract validation with both logical name and I/O alignment issues."""
        contract_name = "test_contract"
        logical_issues = [
            {
                "severity": "ERROR",
                "category": "logical_names",
                "message": "Contract input missing_input not declared as specification dependency"
            }
        ]
        io_issues = [
            {
                "severity": "WARNING",
                "category": "input_output_alignment",
                "message": "Specification dependency extra_dep has no corresponding contract input"
            }
        ]
        
        with patch.object(contract_spec_alignment, '_load_contract') as mock_load_contract, \
             patch.object(contract_spec_alignment, '_load_unified_specification') as mock_load_spec, \
             patch('cursus.validation.alignment.validators.contract_spec_validator.ConsolidatedContractSpecValidator') as mock_validator_class:
            
            # Setup mocks
            mock_load_contract.return_value = sample_contract
            mock_load_spec.return_value = {"primary_spec": sample_specification}
            
            mock_validator = Mock()
            mock_validator.validate_logical_names.return_value = logical_issues
            mock_validator.validate_input_output_alignment.return_value = io_issues
            mock_validator_class.return_value = mock_validator
            
            # Execute validation
            result = contract_spec_alignment.validate_contract(contract_name)
            
            # Verify results
            assert result["passed"] is False  # ERROR causes failure
            assert len(result["issues"]) == 2
            assert any(issue["category"] == "logical_names" for issue in result["issues"])
            assert any(issue["category"] == "input_output_alignment" for issue in result["issues"])

    def test_validate_contract_with_malformed_contract(self, contract_spec_alignment):
        """Test contract validation with malformed contract data."""
        contract_name = "test_contract"
        malformed_contract = {"invalid": "structure"}
        sample_spec = {"dependencies": [], "outputs": []}
        
        with patch.object(contract_spec_alignment, '_load_contract') as mock_load_contract, \
             patch.object(contract_spec_alignment, '_load_unified_specification') as mock_load_spec, \
             patch('cursus.validation.alignment.validators.contract_spec_validator.ConsolidatedContractSpecValidator') as mock_validator_class:
            
            # Setup mocks
            mock_load_contract.return_value = malformed_contract
            mock_load_spec.return_value = {"primary_spec": sample_spec}
            
            mock_validator = Mock()
            mock_validator.validate_logical_names.return_value = []
            mock_validator.validate_input_output_alignment.return_value = []
            mock_validator_class.return_value = mock_validator
            
            # Execute validation
            result = contract_spec_alignment.validate_contract(contract_name)
            
            # Verify validator handles malformed data gracefully
            mock_validator.validate_logical_names.assert_called_once()
            mock_validator.validate_input_output_alignment.assert_called_once()

    def test_validate_contract_with_missing_specification(self, contract_spec_alignment, sample_contract):
        """Test contract validation when specification cannot be loaded."""
        contract_name = "test_contract"
        
        with patch.object(contract_spec_alignment, '_load_contract') as mock_load_contract, \
             patch.object(contract_spec_alignment, '_load_unified_specification') as mock_load_spec:
            
            # Setup mocks
            mock_load_contract.return_value = sample_contract
            mock_load_spec.return_value = None
            
            # Execute validation
            result = contract_spec_alignment.validate_contract(contract_name)
            
            # Verify results indicate failure due to missing specification
            assert result["passed"] is False
            assert len(result["issues"]) > 0

    def test_validate_contract_error_handling(self, contract_spec_alignment):
        """Test contract validation error handling."""
        contract_name = "test_contract"
        
        with patch.object(contract_spec_alignment, '_load_contract') as mock_load_contract:
            # Setup mock to raise exception
            mock_load_contract.side_effect = Exception("Test error")
            
            # Execute validation and verify it handles errors gracefully
            result = contract_spec_alignment.validate_contract(contract_name)
            
            # Should return a result indicating failure
            assert result["passed"] is False
            assert len(result["issues"]) > 0

    def test_integration_with_consolidated_validator(self, contract_spec_alignment, sample_contract, sample_specification):
        """Test integration with ConsolidatedContractSpecValidator."""
        contract_name = "test_contract"
        
        with patch.object(contract_spec_alignment, '_load_contract') as mock_load_contract, \
             patch.object(contract_spec_alignment, '_load_unified_specification') as mock_load_spec:
            
            # Setup mocks
            mock_load_contract.return_value = sample_contract
            mock_load_spec.return_value = {"primary_spec": sample_specification}
            
            # Execute validation (using real ConsolidatedContractSpecValidator)
            result = contract_spec_alignment.validate_contract(contract_name)
            
            # Verify basic structure
            assert "passed" in result
            assert "issues" in result
            assert "contract_name" in result
            assert isinstance(result["issues"], list)

    def test_validate_contract_result_structure(self, contract_spec_alignment, sample_contract, sample_specification):
        """Test that validate_contract returns properly structured results."""
        contract_name = "test_contract"
        
        with patch.object(contract_spec_alignment, '_load_contract') as mock_load_contract, \
             patch.object(contract_spec_alignment, '_load_unified_specification') as mock_load_spec, \
             patch('cursus.validation.alignment.validators.contract_spec_validator.ConsolidatedContractSpecValidator') as mock_validator_class:
            
            # Setup mocks
            mock_load_contract.return_value = sample_contract
            mock_load_spec.return_value = {"primary_spec": sample_specification}
            
            mock_validator = Mock()
            mock_validator.validate_logical_names.return_value = []
            mock_validator.validate_input_output_alignment.return_value = []
            mock_validator_class.return_value = mock_validator
            
            # Execute validation
            result = contract_spec_alignment.validate_contract(contract_name)
            
            # Verify result structure
            required_keys = ["passed", "issues", "contract_name", "contract", "unified_specification"]
            for key in required_keys:
                assert key in result
            
            assert isinstance(result["passed"], bool)
            assert isinstance(result["issues"], list)
            assert isinstance(result["contract_name"], str)
            assert isinstance(result["contract"], dict)

    def test_workspace_directory_propagation(self, workspace_dirs):
        """Test that workspace directories are properly propagated."""
        alignment = ContractSpecAlignment(workspace_dirs=workspace_dirs)
        assert alignment.workspace_dirs == workspace_dirs

    def test_performance_with_large_contract(self, contract_spec_alignment):
        """Test performance with large contract and specification data."""
        # Create large contract and specification
        large_contract = {
            "inputs": {f"input_{i}": {"type": "s3_path"} for i in range(100)},
            "outputs": {f"output_{i}": {"type": "s3_path"} for i in range(100)}
        }
        large_spec = {
            "dependencies": [{"logical_name": f"input_{i}", "type": "s3_path"} for i in range(100)],
            "outputs": [{"logical_name": f"output_{i}", "type": "s3_path"} for i in range(100)]
        }
        
        with patch.object(contract_spec_alignment, '_load_contract') as mock_load_contract, \
             patch.object(contract_spec_alignment, '_load_unified_specification') as mock_load_spec:
            
            # Setup mocks
            mock_load_contract.return_value = large_contract
            mock_load_spec.return_value = {"primary_spec": large_spec}
            
            # Execute validation and verify it completes
            result = contract_spec_alignment.validate_contract("large_contract")
            
            # Should complete successfully
            assert "passed" in result
            assert "issues" in result
