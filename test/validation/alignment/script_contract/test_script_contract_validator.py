"""
Test suite for ScriptContractValidator with enhanced path validation logic.

Tests the three path validation scenarios:
1. Contract file path + Script uses file path → Direct match
2. Contract file path + Script uses directory path → Parent-child relationship  
3. Contract directory path + Script uses directory path → Direct match
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List

from cursus.validation.alignment.validators.script_contract_validator import (
    ScriptContractValidator,
)


class TestScriptContractValidator:
    """Test ScriptContractValidator with enhanced path validation logic."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ScriptContractValidator()

    def _create_mock_path_reference(
        self, path: str, context: str = "", construction_method: str = None
    ):
        """Create a mock path reference object."""
        mock_ref = MagicMock()
        mock_ref.path = path
        mock_ref.context = context
        mock_ref.construction_method = construction_method
        return mock_ref

    def _create_mock_file_operation(
        self, file_path: str, operation_type: str, method: str = None
    ):
        """Create a mock file operation object."""
        mock_op = MagicMock()
        mock_op.file_path = file_path
        mock_op.operation_type = operation_type
        mock_op.method = method
        return mock_op

    def test_scenario_1_direct_file_to_file_match(self):
        """Test Scenario 1: Contract file path + Script uses file path → Direct match."""
        # Contract declares a specific file path
        contract = {
            "inputs": {
                "config": {"path": "/opt/ml/input/data/config/hyperparameters.json"}
            },
            "outputs": {},
        }

        # Script uses the exact same file path
        analysis = {
            "path_references": [
                self._create_mock_path_reference(
                    "/opt/ml/input/data/config/hyperparameters.json"
                )
            ],
            "file_operations": [
                self._create_mock_file_operation(
                    "/opt/ml/input/data/config/hyperparameters.json", "read"
                )
            ],
        }

        issues = self.validator.validate_path_usage(analysis, contract, "test_script")

        # Should have no ERROR issues - direct match should work
        error_issues = [issue for issue in issues if issue["severity"] == "ERROR"]
        assert len(error_issues) == 0, f"Unexpected errors: {error_issues}"

    def test_scenario_2_parent_child_relationship(self):
        """Test Scenario 2: Contract file path + Script uses directory path → Parent-child relationship."""
        # Contract declares a specific file path
        contract = {
            "inputs": {
                "config": {"path": "/opt/ml/input/data/config/hyperparameters.json"}
            },
            "outputs": {},
        }

        # Script uses the parent directory and constructs the file path
        analysis = {
            "path_references": [
                self._create_mock_path_reference(
                    "/opt/ml/input/data/config",
                    context='os.path.join(config_dir, "hyperparameters.json")',
                    construction_method="os.path.join",
                )
            ],
            "file_operations": [],
        }

        issues = self.validator.validate_path_usage(analysis, contract, "test_script")

        # Should have an INFO message about correct parent-child usage
        info_issues = [
            issue
            for issue in issues
            if issue["severity"] == "INFO"
            and "correctly uses parent directory" in issue["message"]
        ]
        assert (
            len(info_issues) > 0
        ), "Should have INFO message about correct parent-child usage"

        # Should have no ERROR issues
        error_issues = [issue for issue in issues if issue["severity"] == "ERROR"]
        assert len(error_issues) == 0, f"Unexpected errors: {error_issues}"

    def test_scenario_3_direct_directory_to_directory_match(self):
        """Test Scenario 3: Contract directory path + Script uses directory path → Direct match."""
        # Contract declares a directory path
        contract = {
            "inputs": {"data_dir": {"path": "/opt/ml/input/data"}},
            "outputs": {},
        }

        # Script uses the same directory path
        analysis = {
            "path_references": [self._create_mock_path_reference("/opt/ml/input/data")],
            "file_operations": [],
        }

        issues = self.validator.validate_path_usage(analysis, contract, "test_script")

        # Should have no ERROR issues - direct directory match should work
        error_issues = [issue for issue in issues if issue["severity"] == "ERROR"]
        assert len(error_issues) == 0, f"Unexpected errors: {error_issues}"

    def test_undeclared_sagemaker_path_error(self):
        """Test that undeclared SageMaker paths generate ERROR issues."""
        contract = {"inputs": {}, "outputs": {}}

        # Script uses an undeclared SageMaker path
        analysis = {
            "path_references": [
                self._create_mock_path_reference(
                    "/opt/ml/processing/input/undeclared.csv"
                )
            ],
            "file_operations": [],
        }

        issues = self.validator.validate_path_usage(analysis, contract, "test_script")

        # Should have ERROR for undeclared SageMaker path
        error_issues = [
            issue
            for issue in issues
            if issue["severity"] == "ERROR"
            and "undeclared SageMaker path" in issue["message"]
        ]
        assert len(error_issues) > 0, "Should have ERROR for undeclared SageMaker path"

    def test_unused_contract_path_warning(self):
        """Test that unused contract paths generate WARNING issues."""
        contract = {
            "inputs": {"unused_input": {"path": "/opt/ml/input/data/unused.csv"}},
            "outputs": {},
        }

        # Script doesn't use the declared path
        analysis = {"path_references": [], "file_operations": []}

        issues = self.validator.validate_path_usage(analysis, contract, "test_script")

        # Should have WARNING for unused contract path
        warning_issues = [
            issue
            for issue in issues
            if issue["severity"] == "WARNING"
            and "not used in script" in issue["message"]
        ]
        assert len(warning_issues) > 0, "Should have WARNING for unused contract path"

    def test_is_file_path_detection(self):
        """Test the _is_file_path helper method."""
        # File paths (should return True)
        file_paths = [
            "/opt/ml/model/model.bst",
            "/opt/ml/input/data/config/hyperparameters.json",
            "/tmp/data.csv",
            "/opt/ml/output/metrics.json",
            "/path/to/file.tar.gz",
        ]

        for path in file_paths:
            assert self.validator._is_file_path(
                path
            ), f"{path} should be detected as file"

        # Directory paths (should return False)
        directory_paths = [
            "/opt/ml/input/data",
            "/opt/ml/model/",
            "/opt/ml/processing/input",
            "/tmp/directory",
            "/opt/ml/output/data/",
        ]

        for path in directory_paths:
            assert not self.validator._is_file_path(
                path
            ), f"{path} should be detected as directory"

    def test_script_constructs_file_path_detection(self):
        """Test the _script_constructs_file_path helper method."""
        # Analysis with os.path.join pattern
        analysis = {
            "path_references": [
                self._create_mock_path_reference(
                    "/opt/ml/input/data/config",
                    context='hparam_path = os.path.join(config_dir, "hyperparameters.json")',
                )
            ]
        }

        # Should detect that script constructs hyperparameters.json from config directory
        result = self.validator._script_constructs_file_path(
            analysis, "/opt/ml/input/data/config", "hyperparameters.json"
        )
        assert result, "Should detect file path construction"

        # Analysis without construction pattern
        analysis_no_construction = {
            "path_references": [
                self._create_mock_path_reference("/opt/ml/input/data/config")
            ]
        }

        result = self.validator._script_constructs_file_path(
            analysis_no_construction,
            "/opt/ml/input/data/config",
            "hyperparameters.json",
        )
        assert not result, "Should not detect file path construction without pattern"

    def test_resolve_logical_name_from_contract(self):
        """Test the _resolve_logical_name_from_contract helper method."""
        contract = {
            "inputs": {
                "training_data": {"path": "/opt/ml/input/data/train.csv"},
                "config": {"path": "/opt/ml/input/data/config/hyperparameters.json"},
            },
            "outputs": {"model": {"path": "/opt/ml/model"}},
        }

        # Should resolve logical names correctly
        assert (
            self.validator._resolve_logical_name_from_contract(
                "/opt/ml/input/data/train.csv", contract
            )
            == "training_data"
        )

        assert (
            self.validator._resolve_logical_name_from_contract(
                "/opt/ml/input/data/config/hyperparameters.json", contract
            )
            == "config"
        )

        assert (
            self.validator._resolve_logical_name_from_contract(
                "/opt/ml/model", contract
            )
            == "model"
        )

        # Should return None for paths not in contract
        assert (
            self.validator._resolve_logical_name_from_contract(
                "/opt/ml/unknown/path", contract
            )
            is None
        )

    def test_resolve_parent_logical_name_from_contract(self):
        """Test the _resolve_parent_logical_name_from_contract helper method."""
        contract = {
            "inputs": {
                "config": {"path": "/opt/ml/input/data/config/hyperparameters.json"}
            },
            "outputs": {"model_file": {"path": "/opt/ml/model/model.bst"}},
        }

        # Should resolve parent logical names correctly
        assert (
            self.validator._resolve_parent_logical_name_from_contract(
                "/opt/ml/input/data/config", contract
            )
            == "config"
        )

        assert (
            self.validator._resolve_parent_logical_name_from_contract(
                "/opt/ml/model", contract
            )
            == "model_file"
        )

        # Should return None for paths that are not parents of contract paths
        assert (
            self.validator._resolve_parent_logical_name_from_contract(
                "/opt/ml/unknown", contract
            )
            is None
        )

    def test_enhanced_file_operations_detection(self):
        """Test the _detect_file_operations_from_paths helper method."""
        contract_inputs = {"data": {"path": "/opt/ml/input/data.csv"}}

        contract_outputs = {"results": {"path": "/opt/ml/output/results.json"}}

        # Analysis with path references that match contract paths
        analysis = {
            "path_references": [
                self._create_mock_path_reference(
                    "/opt/ml/input/data.csv",
                    context='pd.read_csv("/opt/ml/input/data.csv")',
                ),
                self._create_mock_path_reference(
                    "/opt/ml/output/results.json",
                    context='json.dump(data, "/opt/ml/output/results.json")',
                ),
            ]
        }

        script_reads, script_writes = self.validator._detect_file_operations_from_paths(
            analysis, contract_inputs, contract_outputs
        )

        # Should detect read and write operations based on context and contract matching
        assert "/opt/ml/input/data.csv" in script_reads
        assert "/opt/ml/output/results.json" in script_writes

    def test_xgboost_training_scenario(self):
        """Test the specific XGBoost training scenario that was causing false positives."""
        # This is the exact scenario from the XGBoost training script
        contract = {
            "inputs": {
                "input_path": {"path": "/opt/ml/input/data"},
                "hyperparameters_s3_uri": {
                    "path": "/opt/ml/input/data/config/hyperparameters.json"
                },
            },
            "outputs": {
                "model_output": {"path": "/opt/ml/model"},
                "evaluation_output": {"path": "/opt/ml/output/data"},
            },
        }

        # Script uses directory path to construct file path (the problematic scenario)
        analysis = {
            "path_references": [
                self._create_mock_path_reference("/opt/ml/input/data"),
                self._create_mock_path_reference(
                    "/opt/ml/input/data/config",
                    context='hparam_path = os.path.join(hparam_path, "hyperparameters.json")',
                ),
                self._create_mock_path_reference("/opt/ml/model"),
                self._create_mock_path_reference("/opt/ml/output/data"),
            ],
            "file_operations": [],
        }

        issues = self.validator.validate_path_usage(
            analysis, contract, "xgboost_training"
        )

        # Should have INFO message about correct parent-child usage
        info_issues = [
            issue
            for issue in issues
            if issue["severity"] == "INFO"
            and "correctly uses parent directory" in issue["message"]
        ]
        assert (
            len(info_issues) > 0
        ), "Should have INFO message about correct parent-child usage"

        # Should have NO ERROR issues (this was the false positive we fixed)
        error_issues = [issue for issue in issues if issue["severity"] == "ERROR"]
        assert (
            len(error_issues) == 0
        ), f"Should have no ERROR issues, but got: {error_issues}"

        # Verify the specific INFO message content
        parent_child_issue = next(
            (
                issue
                for issue in info_issues
                if "correctly uses parent directory" in issue["message"]
            ),
            None,
        )
        assert parent_child_issue is not None
        assert "/opt/ml/input/data/config" in parent_child_issue["message"]
        assert (
            "/opt/ml/input/data/config/hyperparameters.json"
            in parent_child_issue["message"]
        )
        assert parent_child_issue["details"]["construction_method"] == "os.path.join"

    def test_step_type_specific_validation_training(self):
        """Test step type-specific validation for training steps."""
        analysis = {
            "step_type": "Training",
            "framework": "xgboost",
            "step_type_patterns": {},
        }

        contract = {"outputs": {"model_output": {"path": "/opt/ml/model"}}}

        issues = self.validator.validate_step_type_specific(
            analysis, contract, "training_script"
        )

        # Should have recommendations for training-specific paths
        training_issues = [
            issue
            for issue in issues
            if issue["category"] == "training_contract_validation"
        ]
        assert (
            len(training_issues) > 0
        ), "Should have training-specific validation issues"

    def test_step_type_specific_validation_processing(self):
        """Test step type-specific validation for processing steps."""
        analysis = {
            "step_type": "Processing",
            "framework": "pandas",
            "step_type_patterns": {},
        }

        contract = {"inputs": {}, "outputs": {}}

        issues = self.validator.validate_step_type_specific(
            analysis, contract, "processing_script"
        )

        # Should have recommendations for processing-specific paths
        processing_issues = [
            issue
            for issue in issues
            if issue["category"] == "processing_contract_validation"
        ]
        assert (
            len(processing_issues) > 0
        ), "Should have processing-specific validation issues"


if __name__ == "__main__":
    pytest.main([__file__])
