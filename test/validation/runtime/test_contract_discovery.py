"""
Pytest tests for contract discovery system

Tests the ContractDiscoveryManager and ContractDiscoveryResult classes
for intelligent discovery and loading of script contracts.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from cursus.validation.runtime.contract_discovery import (
    ContractDiscoveryManager,
    ContractDiscoveryResult,
)
from cursus.core.base.contract_base import ScriptContract


class TestContractDiscoveryResult:
    """Test ContractDiscoveryResult model"""

    def test_contract_discovery_result_creation(self):
        """Test creating a ContractDiscoveryResult"""
        mock_contract = Mock(spec=ScriptContract)
        result = ContractDiscoveryResult(
            contract=mock_contract,
            contract_name="test_contract",
            discovery_method="direct_import",
            error_message=None,
        )

        assert result.contract == mock_contract
        assert result.contract_name == "test_contract"
        assert result.discovery_method == "direct_import"
        assert result.error_message is None

    def test_contract_discovery_result_with_error(self):
        """Test creating a ContractDiscoveryResult with error"""
        result = ContractDiscoveryResult(
            contract=None,
            contract_name="not_found",
            discovery_method="none",
            error_message="Contract not found",
        )

        assert result.contract is None
        assert result.contract_name == "not_found"
        assert result.discovery_method == "none"
        assert result.error_message == "Contract not found"


class TestContractDiscoveryManager:
    """Test ContractDiscoveryManager class"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def discovery_manager(self, temp_dir):
        """Create ContractDiscoveryManager instance"""
        return ContractDiscoveryManager(test_data_dir=temp_dir)

    def test_contract_discovery_manager_initialization(self, temp_dir):
        """Test ContractDiscoveryManager initialization"""
        manager = ContractDiscoveryManager(test_data_dir=temp_dir)

        assert manager.test_data_dir == Path(temp_dir)
        assert isinstance(manager._contract_cache, dict)
        assert len(manager.contract_patterns) > 0
        assert len(manager.contract_module_paths) > 0

    def test_discover_contract_cache_hit(self, discovery_manager):
        """Test contract discovery with cache hit"""
        # Pre-populate cache
        mock_contract = Mock(spec=ScriptContract)
        cached_result = ContractDiscoveryResult(
            contract=mock_contract,
            contract_name="cached_contract",
            discovery_method="cached",
            error_message=None,
        )
        discovery_manager._contract_cache["test_script:None"] = cached_result

        result = discovery_manager.discover_contract("test_script")

        assert result == cached_result
        assert result.discovery_method == "cached"

    @patch("importlib.import_module")
    def test_discover_by_direct_import_success(self, mock_import, discovery_manager):
        """Test successful direct import discovery"""
        # Create mock contract
        mock_contract = Mock(spec=ScriptContract)
        mock_contract.entry_point = "test_script.py"

        # Create mock module
        mock_module = Mock()
        mock_module.TEST_SCRIPT_CONTRACT = mock_contract
        mock_import.return_value = mock_module

        result = discovery_manager._discover_by_direct_import("test_script")

        assert result.contract == mock_contract
        assert result.contract_name == "TEST_SCRIPT_CONTRACT"
        assert result.discovery_method == "direct_import"
        assert result.error_message is None

    @patch("importlib.import_module")
    def test_discover_by_direct_import_module_not_found(
        self, mock_import, discovery_manager
    ):
        """Test direct import discovery when module not found"""
        mock_import.side_effect = ImportError("No module named 'test_module'")

        result = discovery_manager._discover_by_direct_import("test_script")

        assert result.contract is None
        assert result.contract_name == "not_found"
        assert result.discovery_method == "direct_import"
        assert "No contract module found" in result.error_message

    @patch("importlib.import_module")
    def test_discover_by_direct_import_entry_point_mismatch(
        self, mock_import, discovery_manager
    ):
        """Test direct import discovery with entry point mismatch"""
        # Create mock contract with wrong entry point
        mock_contract = Mock(spec=ScriptContract)
        mock_contract.entry_point = "wrong_script.py"

        # Create mock module
        mock_module = Mock()
        mock_module.TEST_SCRIPT_CONTRACT = mock_contract
        mock_import.return_value = mock_module

        result = discovery_manager._discover_by_direct_import("test_script")

        # Should continue searching and eventually fail
        assert result.contract is None
        assert result.discovery_method == "direct_import"

    def test_discover_by_pattern_matching_no_modules(self, discovery_manager):
        """Test pattern matching discovery when no modules available"""
        result = discovery_manager._discover_by_pattern_matching("test_script")

        assert result.contract is None
        assert result.contract_name == "not_found"
        assert result.discovery_method == "pattern_matching"
        assert result.error_message == "Pattern matching failed"

    def test_discover_by_fuzzy_search_not_implemented(self, discovery_manager):
        """Test fuzzy search discovery (not implemented)"""
        result = discovery_manager._discover_by_fuzzy_search("test_script")

        assert result.contract is None
        assert result.contract_name == "not_found"
        assert result.discovery_method == "fuzzy_search"
        assert result.error_message == "Fuzzy search not implemented"

    def test_is_contract_match_with_entry_point(self, discovery_manager):
        """Test contract matching using entry_point field"""
        mock_contract = Mock(spec=ScriptContract)
        mock_contract.entry_point = "test_script.py"

        result = discovery_manager._is_contract_match(mock_contract, "test_script")

        assert result is True

    def test_is_contract_match_entry_point_mismatch(self, discovery_manager):
        """Test contract matching with entry_point mismatch"""
        mock_contract = Mock(spec=ScriptContract)
        mock_contract.entry_point = "other_script.py"

        result = discovery_manager._is_contract_match(mock_contract, "test_script")

        assert result is False

    def test_is_contract_match_no_entry_point_fallback(self, discovery_manager):
        """Test contract matching fallback when no entry_point"""
        mock_contract = Mock(spec=ScriptContract)
        mock_contract.__class__.__name__ = "TestScriptContract"
        # No entry_point attribute
        del mock_contract.entry_point

        with patch.object(
            discovery_manager, "_fallback_name_matching", return_value=True
        ):
            result = discovery_manager._is_contract_match(mock_contract, "test_script")

            assert result is True

    def test_fallback_name_matching_substring_match(self, discovery_manager):
        """Test fallback name matching with substring match"""
        # This should match because "testscript" is in "testscriptcontract" when lowercased
        result = discovery_manager._fallback_name_matching(
            "TestScriptContract", "testscript"
        )

        assert result is True

    def test_fallback_name_matching_no_match(self, discovery_manager):
        """Test fallback name matching with no match"""
        result = discovery_manager._fallback_name_matching(
            "CompletelyDifferentContract", "test_script"
        )

        assert result is False

    def test_fallback_name_matching_canonical_name(self, discovery_manager):
        """Test fallback name matching with canonical name"""
        result = discovery_manager._fallback_name_matching(
            "TabularPreprocessingContract", "tabular_preprocess", "TabularPreprocessing"
        )

        assert result is True

    def test_to_constant_case_basic(self, discovery_manager):
        """Test converting PascalCase to CONSTANT_CASE"""
        result = discovery_manager._to_constant_case("TabularPreprocessing")

        assert result == "TABULAR_PREPROCESSING"

    def test_to_constant_case_special_cases(self, discovery_manager):
        """Test converting special cases to CONSTANT_CASE"""
        test_cases = [
            ("XGBoostTraining", "XGBOOST_TRAINING"),
            ("PyTorchModel", "PYTORCH_MODEL"),
            ("MLFlowTracking", "MLFLOW_TRACKING"),
            ("TensorFlowServing", "TENSORFLOW_SERVING"),
            ("SageMakerEndpoint", "SAGEMAKER_ENDPOINT"),
            ("AutoMLPipeline", "AUTOML_PIPELINE"),
        ]

        for input_case, expected in test_cases:
            result = discovery_manager._to_constant_case(input_case)
            assert result == expected

    def test_get_contract_input_paths_empty(self, discovery_manager):
        """Test getting contract input paths when none exist"""
        mock_contract = Mock(spec=ScriptContract)
        mock_contract.expected_input_paths = None

        result = discovery_manager.get_contract_input_paths(
            mock_contract, "test_script"
        )

        assert result == {}

    def test_get_contract_input_paths_with_paths(self, discovery_manager, temp_dir):
        """Test getting contract input paths with adaptation"""
        mock_contract = Mock(spec=ScriptContract)
        mock_contract.expected_input_paths = {
            "training_data": "/opt/ml/input/data/training",
            "validation_data": "/opt/ml/input/data/validation",
        }

        result = discovery_manager.get_contract_input_paths(
            mock_contract, "test_script"
        )

        assert len(result) == 2
        assert "training_data" in result
        assert "validation_data" in result
        # Should adapt SageMaker paths to local paths
        assert temp_dir in result["training_data"]
        assert temp_dir in result["validation_data"]

    def test_get_contract_output_paths_with_paths(self, discovery_manager, temp_dir):
        """Test getting contract output paths with adaptation"""
        mock_contract = Mock(spec=ScriptContract)
        mock_contract.expected_output_paths = {
            "model_output": "/opt/ml/output/model",
            "metrics_output": "/opt/ml/output/metrics",
        }

        result = discovery_manager.get_contract_output_paths(
            mock_contract, "test_script"
        )

        assert len(result) == 2
        assert "model_output" in result
        assert "metrics_output" in result
        # Should adapt SageMaker paths to local paths
        assert temp_dir in result["model_output"]
        assert temp_dir in result["metrics_output"]

    def test_get_contract_environ_vars_required_vars(self, discovery_manager):
        """Test getting environment variables from contract"""
        mock_contract = Mock(spec=ScriptContract)
        mock_contract.required_env_vars = ["MODEL_TYPE", "BATCH_SIZE"]
        mock_contract.optional_env_vars = ["DEBUG_MODE"]

        result = discovery_manager.get_contract_environ_vars(mock_contract)

        assert "MODEL_TYPE" in result
        assert "BATCH_SIZE" in result
        assert "DEBUG_MODE" in result
        assert "PYTHONPATH" in result  # Standard testing vars
        assert "CURSUS_ENV" in result
        assert result["CURSUS_ENV"] == "testing"

    def test_get_contract_environ_vars_dict_format(self, discovery_manager):
        """Test getting environment variables in dict format"""
        mock_contract = Mock(spec=ScriptContract)
        mock_contract.required_env_vars = [
            {"MODEL_TYPE": "xgboost", "BATCH_SIZE": "1000"}
        ]
        mock_contract.optional_env_vars = [{"DEBUG_MODE": "false"}]

        result = discovery_manager.get_contract_environ_vars(mock_contract)

        assert result["MODEL_TYPE"] == "xgboost"
        assert result["BATCH_SIZE"] == "1000"
        assert result["DEBUG_MODE"] == "false"

    def test_get_contract_job_args_basic(self, discovery_manager):
        """Test getting job arguments from contract"""
        mock_contract = Mock(spec=ScriptContract)
        mock_contract.job_args = {"max_depth": 6, "learning_rate": 0.1}

        result = discovery_manager.get_contract_job_args(mock_contract, "test_script")

        assert result["script_name"] == "test_script"
        assert result["execution_mode"] == "testing"
        assert result["log_level"] == "INFO"
        assert result["max_depth"] == 6
        assert result["learning_rate"] == 0.1

    def test_get_contract_job_args_from_metadata(self, discovery_manager):
        """Test getting job arguments from contract metadata"""
        mock_contract = Mock(spec=ScriptContract)
        mock_contract.job_args = None
        mock_contract.metadata = {"job_args": {"custom_param": "custom_value"}}

        result = discovery_manager.get_contract_job_args(mock_contract, "test_script")

        assert result["custom_param"] == "custom_value"
        assert result["script_name"] == "test_script"

    def test_adapt_path_for_local_testing_sagemaker_patterns(
        self, discovery_manager, temp_dir
    ):
        """Test adapting SageMaker paths to local testing paths"""
        base_data_dir = Path(temp_dir) / "test_script"

        test_cases = [
            ("/opt/ml/input/data/training", "input/training"),
            ("/opt/ml/output/model", "output/model"),
            ("/opt/ml/processing/input/features", "input/features"),
            ("/opt/ml/processing/output/predictions", "output/predictions"),
        ]

        for original_path, expected_suffix in test_cases:
            result = discovery_manager._adapt_path_for_local_testing(
                original_path, base_data_dir, "input"
            )

            assert str(result).endswith(expected_suffix)
            assert str(base_data_dir) in str(result)

    def test_adapt_path_for_local_testing_non_sagemaker(
        self, discovery_manager, temp_dir
    ):
        """Test adapting non-SageMaker paths"""
        base_data_dir = Path(temp_dir) / "test_script"

        result = discovery_manager._adapt_path_for_local_testing(
            "/custom/path/data", base_data_dir, "input"
        )

        # Should create reasonable local path
        assert str(base_data_dir) in str(result)
        assert "input" in str(result)
        assert "data" in str(result)

    def test_discover_contract_full_workflow_success(self, discovery_manager):
        """Test complete contract discovery workflow with success"""
        with patch.object(
            discovery_manager, "_discover_by_direct_import"
        ) as mock_direct:
            mock_contract = Mock(spec=ScriptContract)
            mock_direct.return_value = ContractDiscoveryResult(
                contract=mock_contract,
                contract_name="TEST_SCRIPT_CONTRACT",
                discovery_method="direct_import",
                error_message=None,
            )

            result = discovery_manager.discover_contract("test_script")

            assert result.contract == mock_contract
            assert result.discovery_method == "direct_import"
            assert result.error_message is None

            # Should be cached
            cache_key = "test_script:None"
            assert cache_key in discovery_manager._contract_cache

    def test_discover_contract_full_workflow_failure(self, discovery_manager):
        """Test complete contract discovery workflow with failure"""
        with patch.object(
            discovery_manager, "_discover_by_direct_import"
        ) as mock_direct:
            with patch.object(
                discovery_manager, "_discover_by_pattern_matching"
            ) as mock_pattern:
                with patch.object(
                    discovery_manager, "_discover_by_fuzzy_search"
                ) as mock_fuzzy:
                    # All methods fail
                    failure_result = ContractDiscoveryResult(
                        contract=None,
                        contract_name="not_found",
                        discovery_method="none",
                        error_message="Failed",
                    )
                    mock_direct.return_value = failure_result
                    mock_pattern.return_value = failure_result
                    mock_fuzzy.return_value = failure_result

                    result = discovery_manager.discover_contract("nonexistent_script")

                    assert result.contract is None
                    assert "No contract found" in result.error_message

                    # Should be cached
                    cache_key = "nonexistent_script:None"
                    assert cache_key in discovery_manager._contract_cache


class TestContractDiscoveryIntegration:
    """Integration tests for contract discovery system"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def discovery_manager(self, temp_dir):
        """Create ContractDiscoveryManager instance"""
        return ContractDiscoveryManager(test_data_dir=temp_dir)

    def test_end_to_end_contract_discovery_and_adaptation(
        self, discovery_manager, temp_dir
    ):
        """Test complete end-to-end contract discovery and path adaptation"""
        # Create mock contract with realistic SageMaker paths
        mock_contract = Mock(spec=ScriptContract)
        mock_contract.entry_point = "tabular_preprocessing.py"
        mock_contract.expected_input_paths = {
            "raw_data": "/opt/ml/input/data/training",
            "config": "/opt/ml/input/config/preprocessing.json",
        }
        mock_contract.expected_output_paths = {
            "processed_data": "/opt/ml/output/data",
            "feature_metadata": "/opt/ml/output/metadata",
        }
        mock_contract.required_env_vars = ["PREPROCESSING_MODE", "FEATURE_COLUMNS"]
        mock_contract.job_args = {"batch_size": 1000, "normalize": True}

        with patch.object(
            discovery_manager, "_discover_by_direct_import"
        ) as mock_direct:
            mock_direct.return_value = ContractDiscoveryResult(
                contract=mock_contract,
                contract_name="TABULAR_PREPROCESSING_CONTRACT",
                discovery_method="direct_import",
                error_message=None,
            )

            # Discover contract
            result = discovery_manager.discover_contract("tabular_preprocessing")

            assert result.contract == mock_contract
            assert result.discovery_method == "direct_import"

            # Test path adaptation
            input_paths = discovery_manager.get_contract_input_paths(
                mock_contract, "tabular_preprocessing"
            )
            output_paths = discovery_manager.get_contract_output_paths(
                mock_contract, "tabular_preprocessing"
            )

            # Verify paths are adapted to local testing structure
            assert len(input_paths) == 2
            assert len(output_paths) == 2
            assert temp_dir in input_paths["raw_data"]
            assert temp_dir in output_paths["processed_data"]

            # Test environment variables
            environ_vars = discovery_manager.get_contract_environ_vars(mock_contract)
            assert "PREPROCESSING_MODE" in environ_vars
            assert "FEATURE_COLUMNS" in environ_vars
            assert environ_vars["CURSUS_ENV"] == "testing"

            # Test job arguments
            job_args = discovery_manager.get_contract_job_args(
                mock_contract, "tabular_preprocessing"
            )
            assert job_args["batch_size"] == 1000
            assert job_args["normalize"] is True
            assert job_args["script_name"] == "tabular_preprocessing"

    def test_contract_discovery_with_canonical_name(self, discovery_manager):
        """Test contract discovery using canonical name"""
        mock_contract = Mock(spec=ScriptContract)
        mock_contract.entry_point = "tabular_preprocessing.py"

        with patch.object(
            discovery_manager, "_discover_by_direct_import"
        ) as mock_direct:
            mock_direct.return_value = ContractDiscoveryResult(
                contract=mock_contract,
                contract_name="TABULAR_PREPROCESSING_CONTRACT",
                discovery_method="direct_import",
                error_message=None,
            )

            result = discovery_manager.discover_contract(
                "tabular_preprocessing", canonical_name="TabularPreprocessing"
            )

            assert result.contract == mock_contract

            # Verify canonical name was used in discovery
            mock_direct.assert_called_once_with(
                "tabular_preprocessing", "TabularPreprocessing"
            )

    def test_multiple_contract_discovery_caching(self, discovery_manager):
        """Test that multiple contract discoveries use caching effectively"""
        mock_contract = Mock(spec=ScriptContract)

        with patch.object(
            discovery_manager, "_discover_by_direct_import"
        ) as mock_direct:
            mock_direct.return_value = ContractDiscoveryResult(
                contract=mock_contract,
                contract_name="TEST_CONTRACT",
                discovery_method="direct_import",
                error_message=None,
            )

            # First discovery
            result1 = discovery_manager.discover_contract("test_script")

            # Second discovery (should use cache)
            result2 = discovery_manager.discover_contract("test_script")

            assert result1 == result2
            # Direct import should only be called once
            mock_direct.assert_called_once()
