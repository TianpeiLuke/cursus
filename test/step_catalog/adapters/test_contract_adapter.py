"""
Unit tests for step_catalog.adapters.contract_adapter module.

Tests the ContractDiscoveryEngineAdapter and ContractDiscoveryManagerAdapter classes
that provide backward compatibility with legacy contract discovery systems.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional, List

from cursus.step_catalog.adapters.contract_adapter import (
    ContractDiscoveryResult,
    ContractDiscoveryEngineAdapter,
    ContractDiscoveryManagerAdapter
)


class TestContractDiscoveryResult:
    """Test ContractDiscoveryResult functionality."""
    
    def test_init_success(self):
        """Test successful ContractDiscoveryResult initialization."""
        mock_contract = Mock()
        result = ContractDiscoveryResult(
            contract=mock_contract,
            contract_name="TEST_CONTRACT",
            discovery_method="step_catalog",
            error_message=None
        )
        
        assert result.contract == mock_contract
        assert result.contract_name == "TEST_CONTRACT"
        assert result.discovery_method == "step_catalog"
        assert result.error_message is None
        assert result.success is True
    
    def test_init_failure(self):
        """Test failed ContractDiscoveryResult initialization."""
        result = ContractDiscoveryResult(
            contract=None,
            contract_name=None,
            discovery_method="step_catalog",
            error_message="Contract not found"
        )
        
        assert result.contract is None
        assert result.contract_name is None
        assert result.discovery_method == "step_catalog"
        assert result.error_message == "Contract not found"
        assert result.success is False
    
    def test_init_defaults(self):
        """Test ContractDiscoveryResult with default values."""
        result = ContractDiscoveryResult()
        
        assert result.contract is None
        assert result.contract_name is None
        assert result.discovery_method == "step_catalog"
        assert result.error_message is None
        assert result.success is False
    
    def test_repr_success(self):
        """Test string representation for successful result."""
        mock_contract = Mock()
        result = ContractDiscoveryResult(
            contract=mock_contract,
            contract_name="TEST_CONTRACT",
            discovery_method="step_catalog"
        )
        
        repr_str = repr(result)
        assert "ContractDiscoveryResult(contract=TEST_CONTRACT" in repr_str
        assert "method=step_catalog)" in repr_str
    
    def test_repr_failure(self):
        """Test string representation for failed result."""
        result = ContractDiscoveryResult(
            error_message="Contract not found"
        )
        
        repr_str = repr(result)
        assert "ContractDiscoveryResult(error=Contract not found)" in repr_str


class TestContractDiscoveryEngineAdapter:
    """Test ContractDiscoveryEngineAdapter functionality."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_init(self, temp_workspace):
        """Test ContractDiscoveryEngineAdapter initialization."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog') as mock_catalog:
            adapter = ContractDiscoveryEngineAdapter(temp_workspace)
            
            # Should initialize StepCatalog with workspace_dirs=[workspace_root]
            mock_catalog.assert_called_once_with(workspace_dirs=[temp_workspace])
            assert adapter.logger is not None
    
    def test_discover_contracts_with_scripts_success(self, temp_workspace):
        """Test successful discovery of contracts with scripts."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.discover_contracts_with_scripts.return_value = ["step1", "step2", "step3"]
            mock_catalog_class.return_value = mock_catalog
            
            adapter = ContractDiscoveryEngineAdapter(temp_workspace)
            result = adapter.discover_contracts_with_scripts()
            
            assert result == ["step1", "step2", "step3"]
            mock_catalog.discover_contracts_with_scripts.assert_called_once()
    
    def test_discover_contracts_with_scripts_error(self, temp_workspace):
        """Test error handling in discover_contracts_with_scripts."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.discover_contracts_with_scripts.side_effect = Exception("Discovery failed")
            mock_catalog_class.return_value = mock_catalog
            
            adapter = ContractDiscoveryEngineAdapter(temp_workspace)
            result = adapter.discover_contracts_with_scripts()
            
            assert result == []
    
    def test_discover_all_contracts_success(self, temp_workspace):
        """Test successful discovery of all contracts."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.list_available_steps.return_value = ["step1", "step2", "step3"]
            
            # Mock step info with contract components
            mock_step_info1 = Mock()
            mock_step_info1.file_components = {"contract": Mock(), "script": Mock()}
            mock_step_info2 = Mock()
            mock_step_info2.file_components = {"script": Mock()}  # No contract
            mock_step_info3 = Mock()
            mock_step_info3.file_components = {"contract": Mock(), "spec": Mock()}
            
            mock_catalog.get_step_info.side_effect = [mock_step_info1, mock_step_info2, mock_step_info3]
            mock_catalog_class.return_value = mock_catalog
            
            adapter = ContractDiscoveryEngineAdapter(temp_workspace)
            result = adapter.discover_all_contracts()
            
            # Should return only steps that have contracts
            assert result == ["step1", "step3"]
    
    def test_discover_all_contracts_error(self, temp_workspace):
        """Test error handling in discover_all_contracts."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.list_available_steps.side_effect = Exception("List failed")
            mock_catalog_class.return_value = mock_catalog
            
            adapter = ContractDiscoveryEngineAdapter(temp_workspace)
            result = adapter.discover_all_contracts()
            
            assert result == []
    
    def test_extract_contract_reference_from_spec_success(self, temp_workspace):
        """Test successful extraction of contract reference from spec."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.find_step_by_component.return_value = "test_step"
            
            mock_step_info = Mock()
            mock_step_info.file_components = {"contract": Mock(), "spec": Mock()}
            mock_catalog.get_step_info.return_value = mock_step_info
            mock_catalog_class.return_value = mock_catalog
            
            adapter = ContractDiscoveryEngineAdapter(temp_workspace)
            result = adapter.extract_contract_reference_from_spec("test_step_spec.py")
            
            assert result == "test_step"
            mock_catalog.find_step_by_component.assert_called_once_with("test_step_spec.py")
    
    def test_extract_contract_reference_from_spec_no_step(self, temp_workspace):
        """Test extraction when no step is found."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.find_step_by_component.return_value = None
            mock_catalog_class.return_value = mock_catalog
            
            adapter = ContractDiscoveryEngineAdapter(temp_workspace)
            result = adapter.extract_contract_reference_from_spec("nonexistent_spec.py")
            
            assert result is None
    
    def test_extract_contract_reference_from_spec_no_contract(self, temp_workspace):
        """Test extraction when step has no contract."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.find_step_by_component.return_value = "test_step"
            
            mock_step_info = Mock()
            mock_step_info.file_components = {"spec": Mock()}  # No contract
            mock_catalog.get_step_info.return_value = mock_step_info
            mock_catalog_class.return_value = mock_catalog
            
            adapter = ContractDiscoveryEngineAdapter(temp_workspace)
            result = adapter.extract_contract_reference_from_spec("test_step_spec.py")
            
            assert result is None
    
    def test_extract_contract_reference_from_spec_error(self, temp_workspace):
        """Test error handling in extract_contract_reference_from_spec."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.find_step_by_component.side_effect = Exception("Find failed")
            mock_catalog_class.return_value = mock_catalog
            
            adapter = ContractDiscoveryEngineAdapter(temp_workspace)
            result = adapter.extract_contract_reference_from_spec("test_spec.py")
            
            assert result is None
    
    def test_build_entry_point_mapping_success(self, temp_workspace):
        """Test successful building of entry point mapping."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.list_available_steps.return_value = ["step1", "step2"]
            
            # Mock step info with script components
            mock_step_info1 = Mock()
            mock_script1 = Mock()
            mock_script1.path = Path("/path/to/step1.py")
            mock_step_info1.file_components = {"script": mock_script1}
            
            mock_step_info2 = Mock()
            mock_step_info2.file_components = {}  # No script
            
            mock_catalog.get_step_info.side_effect = [mock_step_info1, mock_step_info2]
            mock_catalog_class.return_value = mock_catalog
            
            adapter = ContractDiscoveryEngineAdapter(temp_workspace)
            result = adapter.build_entry_point_mapping()
            
            # Should only include steps with scripts
            assert result == {"step1": "/path/to/step1.py"}
    
    def test_build_entry_point_mapping_error(self, temp_workspace):
        """Test error handling in build_entry_point_mapping."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.list_available_steps.side_effect = Exception("List failed")
            mock_catalog_class.return_value = mock_catalog
            
            adapter = ContractDiscoveryEngineAdapter(temp_workspace)
            result = adapter.build_entry_point_mapping()
            
            assert result == {}


class TestContractDiscoveryManagerAdapter:
    """Test ContractDiscoveryManagerAdapter functionality."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_init_with_test_data_dir(self):
        """Test initialization with test_data_dir."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog') as mock_catalog:
            adapter = ContractDiscoveryManagerAdapter(test_data_dir="/test/data")
            
            assert adapter.test_data_dir == Path("/test/data")
            mock_catalog.assert_called_once_with(workspace_dirs=[Path("/test/data")])
            assert adapter._contract_cache == {}
    
    def test_init_with_workspace_root(self, temp_workspace):
        """Test initialization with workspace_root."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog') as mock_catalog:
            adapter = ContractDiscoveryManagerAdapter(workspace_root=temp_workspace)
            
            assert adapter.test_data_dir == temp_workspace
            mock_catalog.assert_called_once_with(workspace_dirs=[temp_workspace])
    
    def test_init_with_defaults(self):
        """Test initialization with default values."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog') as mock_catalog:
            adapter = ContractDiscoveryManagerAdapter()
            
            assert adapter.test_data_dir == Path('.')
            mock_catalog.assert_called_once_with(workspace_dirs=[Path('.')])
    
    def test_discover_contract_success_with_path(self, temp_workspace):
        """Test successful contract discovery returning file path."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_contract = Mock()
            mock_catalog.load_contract_class.return_value = mock_contract
            
            # Mock step info with contract file path
            mock_step_info = Mock()
            mock_contract_file = Mock()
            mock_contract_file.path = Path("/path/to/contract.py")
            mock_step_info.file_components = {"contract": mock_contract_file}
            mock_catalog.get_step_info.return_value = mock_step_info
            mock_catalog_class.return_value = mock_catalog
            
            adapter = ContractDiscoveryManagerAdapter(workspace_root=temp_workspace)
            result = adapter.discover_contract("test_step")
            
            # Should return file path for backward compatibility
            assert result == "/path/to/contract.py"
            mock_catalog.load_contract_class.assert_called_once_with("test_step")
    
    def test_discover_contract_success_without_path(self, temp_workspace):
        """Test successful contract discovery without file path."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_contract = Mock()
            mock_catalog.load_contract_class.return_value = mock_contract
            
            # Mock step info without contract file path
            mock_step_info = Mock()
            mock_step_info.file_components = {}
            mock_catalog.get_step_info.return_value = mock_step_info
            mock_catalog_class.return_value = mock_catalog
            
            adapter = ContractDiscoveryManagerAdapter(workspace_root=temp_workspace)
            result = adapter.discover_contract("test_step")
            
            # Should return ContractDiscoveryResult object
            assert isinstance(result, ContractDiscoveryResult)
            assert result.success is True
            assert result.contract == mock_contract
            assert result.contract_name == "TEST_STEP_CONTRACT"
    
    def test_discover_contract_not_found(self, temp_workspace):
        """Test contract discovery when contract is not found."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.load_contract_class.return_value = None
            mock_catalog_class.return_value = mock_catalog
            
            adapter = ContractDiscoveryManagerAdapter(workspace_root=temp_workspace)
            result = adapter.discover_contract("nonexistent_step")
            
            assert result is None
    
    def test_discover_contract_caching(self, temp_workspace):
        """Test that contract discovery results are cached."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_contract = Mock()
            mock_catalog.load_contract_class.return_value = mock_contract
            
            mock_step_info = Mock()
            mock_contract_file = Mock()
            mock_contract_file.path = Path("/path/to/contract.py")
            mock_step_info.file_components = {"contract": mock_contract_file}
            mock_catalog.get_step_info.return_value = mock_step_info
            mock_catalog_class.return_value = mock_catalog
            
            adapter = ContractDiscoveryManagerAdapter(workspace_root=temp_workspace)
            
            # First call
            result1 = adapter.discover_contract("test_step")
            
            # Second call should use cache
            result2 = adapter.discover_contract("test_step")
            
            assert result1 == result2
            # Should only call load_contract_class once
            mock_catalog.load_contract_class.assert_called_once_with("test_step")
    
    def test_discover_contract_with_canonical_name(self, temp_workspace):
        """Test contract discovery with canonical name."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_contract = Mock()
            mock_catalog.load_contract_class.return_value = mock_contract
            
            mock_step_info = Mock()
            mock_contract_file = Mock()
            mock_contract_file.path = Path("/path/to/contract.py")
            mock_step_info.file_components = {"contract": mock_contract_file}
            mock_catalog.get_step_info.return_value = mock_step_info
            mock_catalog_class.return_value = mock_catalog
            
            adapter = ContractDiscoveryManagerAdapter(workspace_root=temp_workspace)
            result = adapter.discover_contract("test_step", canonical_name="canonical_test")
            
            assert result == "/path/to/contract.py"
            # Should cache with both step_name and canonical_name
            assert "test_step:canonical_test" in adapter._contract_cache
    
    def test_discover_contract_error(self, temp_workspace):
        """Test error handling in discover_contract."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.load_contract_class.side_effect = Exception("Load failed")
            mock_catalog_class.return_value = mock_catalog
            
            adapter = ContractDiscoveryManagerAdapter(workspace_root=temp_workspace)
            result = adapter.discover_contract("test_step")
            
            assert result is None
    
    def test_get_contract_input_paths_success(self, temp_workspace):
        """Test successful extraction of contract input paths."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog'):
            adapter = ContractDiscoveryManagerAdapter(workspace_root=temp_workspace)
            
            mock_contract = Mock()
            mock_contract.expected_input_paths = {
                "input_data": "/opt/ml/input/data/train.csv",
                "model_config": "/opt/ml/input/config/model.json"
            }
            
            result = adapter.get_contract_input_paths(mock_contract, "test_step")
            
            assert len(result) == 2
            assert "input_data" in result
            assert "model_config" in result
            # Paths should be adapted for local testing
            assert "/input/" in result["input_data"]
            assert "/input/" in result["model_config"]
    
    def test_get_contract_input_paths_no_paths(self, temp_workspace):
        """Test get_contract_input_paths when contract has no input paths."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog'):
            adapter = ContractDiscoveryManagerAdapter(workspace_root=temp_workspace)
            
            mock_contract = Mock()
            mock_contract.expected_input_paths = None
            
            result = adapter.get_contract_input_paths(mock_contract, "test_step")
            
            assert result == {}
    
    def test_get_contract_input_paths_error(self, temp_workspace):
        """Test error handling in get_contract_input_paths."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog'):
            adapter = ContractDiscoveryManagerAdapter(workspace_root=temp_workspace)
            
            mock_contract = Mock()
            # Mock contract that raises exception when accessing expected_input_paths
            type(mock_contract).expected_input_paths = Mock(side_effect=Exception("Access failed"))
            
            result = adapter.get_contract_input_paths(mock_contract, "test_step")
            
            assert result == {}
    
    def test_get_contract_output_paths_success(self, temp_workspace):
        """Test successful extraction of contract output paths."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog'):
            adapter = ContractDiscoveryManagerAdapter(workspace_root=temp_workspace)
            
            mock_contract = Mock()
            mock_contract.expected_output_paths = {
                "model": "/opt/ml/output/model.tar.gz",
                "metrics": "/opt/ml/output/metrics.json"
            }
            
            result = adapter.get_contract_output_paths(mock_contract, "test_step")
            
            assert len(result) == 2
            assert "model" in result
            assert "metrics" in result
            # Paths should be adapted for local testing
            assert "/output/" in result["model"]
            assert "/output/" in result["metrics"]
    
    def test_get_contract_output_paths_no_paths(self, temp_workspace):
        """Test get_contract_output_paths when contract has no output paths."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog'):
            adapter = ContractDiscoveryManagerAdapter(workspace_root=temp_workspace)
            
            mock_contract = Mock()
            mock_contract.expected_output_paths = None
            
            result = adapter.get_contract_output_paths(mock_contract, "test_step")
            
            assert result == {}
    
    def test_get_contract_environ_vars_success(self, temp_workspace):
        """Test successful extraction of contract environment variables."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog'):
            adapter = ContractDiscoveryManagerAdapter(workspace_root=temp_workspace)
            
            mock_contract = Mock()
            mock_contract.required_env_vars = ["REQUIRED_VAR", {"CUSTOM_VAR": "custom_value"}]
            mock_contract.optional_env_vars = ["OPTIONAL_VAR", {"ANOTHER_VAR": "another_value"}]
            
            result = adapter.get_contract_environ_vars(mock_contract)
            
            # Should include default vars plus contract vars
            assert "PYTHONPATH" in result
            assert "CURSUS_ENV" in result
            assert result["CURSUS_ENV"] == "testing"
            assert "REQUIRED_VAR" in result
            assert "OPTIONAL_VAR" in result
            assert result["CUSTOM_VAR"] == "custom_value"
            assert result["ANOTHER_VAR"] == "another_value"
    
    def test_get_contract_environ_vars_no_vars(self, temp_workspace):
        """Test get_contract_environ_vars when contract has no environment variables."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog'):
            adapter = ContractDiscoveryManagerAdapter(workspace_root=temp_workspace)
            
            mock_contract = Mock()
            mock_contract.required_env_vars = None
            mock_contract.optional_env_vars = None
            
            result = adapter.get_contract_environ_vars(mock_contract)
            
            # Should include default vars
            assert "PYTHONPATH" in result
            assert "CURSUS_ENV" in result
            assert result["CURSUS_ENV"] == "testing"
    
    def test_get_contract_environ_vars_error(self, temp_workspace):
        """Test error handling in get_contract_environ_vars."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog'):
            adapter = ContractDiscoveryManagerAdapter(workspace_root=temp_workspace)
            
            mock_contract = Mock()
            # Mock contract that raises exception
            type(mock_contract).required_env_vars = Mock(side_effect=Exception("Access failed"))
            
            result = adapter.get_contract_environ_vars(mock_contract)
            
            # Should return default environment
            assert result == {"CURSUS_ENV": "testing"}
    
    def test_get_contract_job_args_success(self, temp_workspace):
        """Test successful extraction of contract job arguments."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog'):
            adapter = ContractDiscoveryManagerAdapter(workspace_root=temp_workspace)
            
            mock_contract = Mock()
            mock_contract.job_args = {
                "custom_arg": "custom_value",
                "batch_size": 32
            }
            
            result = adapter.get_contract_job_args(mock_contract, "test_step")
            
            # Should include default args plus contract args
            assert result["script_name"] == "test_step"
            assert result["execution_mode"] == "testing"
            assert result["log_level"] == "INFO"
            assert result["custom_arg"] == "custom_value"
            assert result["batch_size"] == 32
    
    def test_get_contract_job_args_from_metadata(self, temp_workspace):
        """Test extraction of job args from contract metadata."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog'):
            adapter = ContractDiscoveryManagerAdapter(workspace_root=temp_workspace)
            
            mock_contract = Mock()
            mock_contract.job_args = None
            mock_contract.metadata = {
                "job_args": {
                    "metadata_arg": "metadata_value"
                }
            }
            
            result = adapter.get_contract_job_args(mock_contract, "test_step")
            
            assert result["metadata_arg"] == "metadata_value"
    
    def test_get_contract_job_args_no_args(self, temp_workspace):
        """Test get_contract_job_args when contract has no job arguments."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog'):
            adapter = ContractDiscoveryManagerAdapter(workspace_root=temp_workspace)
            
            mock_contract = Mock()
            mock_contract.job_args = None
            mock_contract.metadata = None
            
            result = adapter.get_contract_job_args(mock_contract, "test_step")
            
            # Should include default args
            assert result["script_name"] == "test_step"
            assert result["execution_mode"] == "testing"
            assert result["log_level"] == "INFO"
    
    def test_get_contract_job_args_error(self, temp_workspace):
        """Test error handling in get_contract_job_args."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog'):
            adapter = ContractDiscoveryManagerAdapter(workspace_root=temp_workspace)
            
            mock_contract = Mock()
            # Mock contract that raises exception
            type(mock_contract).job_args = Mock(side_effect=Exception("Access failed"))
            
            result = adapter.get_contract_job_args(mock_contract, "test_step")
            
            # Should return default args
            assert result["script_name"] == "test_step"
            assert result["execution_mode"] == "testing"
    
    def test_adapt_path_for_local_testing_sagemaker_input(self, temp_workspace):
        """Test path adaptation for SageMaker input paths."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog'):
            adapter = ContractDiscoveryManagerAdapter(workspace_root=temp_workspace)
            
            base_data_dir = temp_workspace / "test_step"
            result = adapter._adapt_path_for_local_testing(
                "/opt/ml/input/data/train.csv", 
                base_data_dir, 
                "input"
            )
            
            assert result == base_data_dir / "input" / "data" / "train.csv"
    
    def test_adapt_path_for_local_testing_sagemaker_output(self, temp_workspace):
        """Test path adaptation for SageMaker output paths."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog'):
            adapter = ContractDiscoveryManagerAdapter(workspace_root=temp_workspace)
            
            base_data_dir = temp_workspace / "test_step"
            result = adapter._adapt_path_for_local_testing(
                "/opt/ml/output/model.tar.gz", 
                base_data_dir, 
                "output"
            )
            
            assert result == base_data_dir / "output" / "model.tar.gz"
    
    def test_adapt_path_for_local_testing_processing_input(self, temp_workspace):
        """Test path adaptation for SageMaker processing input paths."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog'):
            adapter = ContractDiscoveryManagerAdapter(workspace_root=temp_workspace)
            
            base_data_dir = temp_workspace / "test_step"
            result = adapter._adapt_path_for_local_testing(
                "/opt/ml/processing/input/data/features.csv", 
                base_data_dir, 
                "input"
            )
            
            assert result == base_data_dir / "input" / "data" / "features.csv"
    
    def test_adapt_path_for_local_testing_processing_output(self, temp_workspace):
        """Test path adaptation for SageMaker processing output paths."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog'):
            adapter = ContractDiscoveryManagerAdapter(workspace_root=temp_workspace)
            
            base_data_dir = temp_workspace / "test_step"
            result = adapter._adapt_path_for_local_testing(
                "/opt/ml/processing/output/results/metrics.json", 
                base_data_dir, 
                "output"
            )
            
            assert result == base_data_dir / "output" / "results" / "metrics.json"
    
    def test_adapt_path_for_local_testing_custom_path(self, temp_workspace):
        """Test path adaptation for custom paths."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog'):
            adapter = ContractDiscoveryManagerAdapter(workspace_root=temp_workspace)
            
            base_data_dir = temp_workspace / "test_step"
            result = adapter._adapt_path_for_local_testing(
                "/custom/path/to/data.csv", 
                base_data_dir, 
                "input"
            )
            
            assert result == base_data_dir / "input" / "data.csv"
    
    def test_adapt_path_for_local_testing_error(self, temp_workspace):
        """Test error handling in path adaptation."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog'):
            adapter = ContractDiscoveryManagerAdapter(workspace_root=temp_workspace)
            
            base_data_dir = temp_workspace / "test_step"
            
            # Test with invalid path that causes exception
            with patch('cursus.step_catalog.adapters.contract_adapter.Path', side_effect=Exception("Path error")):
                result = adapter._adapt_path_for_local_testing(
                    "invalid_path", 
                    base_data_dir, 
                    "input"
                )
                
                # Should return fallback path
                assert result == base_data_dir / "input" / "data"


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_complete_contract_discovery_workflow(self, temp_workspace):
        """Test complete contract discovery workflow."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            
            # Mock discovery methods
            mock_catalog.discover_contracts_with_scripts.return_value = ["step1", "step2"]
            mock_catalog.list_available_steps.return_value = ["step1", "step2", "step3"]
            
            # Mock step info
            mock_step_info1 = Mock()
            mock_step_info1.file_components = {"contract": Mock(), "script": Mock()}
            mock_step_info2 = Mock()
            mock_step_info2.file_components = {"contract": Mock()}
            mock_step_info3 = Mock()
            mock_step_info3.file_components = {"script": Mock()}  # No contract
            
            mock_catalog.get_step_info.side_effect = [mock_step_info1, mock_step_info2, mock_step_info3]
            mock_catalog_class.return_value = mock_catalog
            
            # Test engine adapter
            engine = ContractDiscoveryEngineAdapter(temp_workspace)
            contracts_with_scripts = engine.discover_contracts_with_scripts()
            all_contracts = engine.discover_all_contracts()
            
            assert contracts_with_scripts == ["step1", "step2"]
            assert all_contracts == ["step1", "step2"]  # step3 has no contract
    
    def test_manager_adapter_contract_analysis_workflow(self, temp_workspace):
        """Test manager adapter contract analysis workflow."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_contract = Mock()
            
            # Setup contract with all attributes
            mock_contract.expected_input_paths = {"data": "/opt/ml/input/data/train.csv"}
            mock_contract.expected_output_paths = {"model": "/opt/ml/output/model.tar.gz"}
            mock_contract.required_env_vars = ["MODEL_TYPE"]
            mock_contract.optional_env_vars = ["DEBUG_MODE"]
            mock_contract.job_args = {"epochs": 10}
            
            mock_catalog.load_contract_class.return_value = mock_contract
            
            mock_step_info = Mock()
            mock_contract_file = Mock()
            mock_contract_file.path = Path("/path/to/contract.py")
            mock_step_info.file_components = {"contract": mock_contract_file}
            mock_catalog.get_step_info.return_value = mock_step_info
            mock_catalog_class.return_value = mock_catalog
            
            manager = ContractDiscoveryManagerAdapter(workspace_root=temp_workspace)
            
            # Test complete workflow
            contract_path = manager.discover_contract("test_step")
            input_paths = manager.get_contract_input_paths(mock_contract, "test_step")
            output_paths = manager.get_contract_output_paths(mock_contract, "test_step")
            env_vars = manager.get_contract_environ_vars(mock_contract)
            job_args = manager.get_contract_job_args(mock_contract, "test_step")
            
            assert contract_path == "/path/to/contract.py"
            assert len(input_paths) == 1
            assert len(output_paths) == 1
            assert "MODEL_TYPE" in env_vars
            assert "DEBUG_MODE" in env_vars
            assert job_args["epochs"] == 10


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_engine_adapter_with_catalog_failure(self, temp_workspace):
        """Test engine adapter behavior when catalog fails."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog') as mock_catalog_class:
            mock_catalog_class.side_effect = Exception("Catalog initialization failed")
            
            # Should still initialize but methods will fail gracefully
            adapter = ContractDiscoveryEngineAdapter(temp_workspace)
            
            # All methods should return empty results on error
            assert adapter.discover_contracts_with_scripts() == []
            assert adapter.discover_all_contracts() == []
            assert adapter.extract_contract_reference_from_spec("test.py") is None
            assert adapter.build_entry_point_mapping() == {}
    
    def test_manager_adapter_with_invalid_contract(self, temp_workspace):
        """Test manager adapter with invalid contract object."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            
            # Return invalid contract object
            invalid_contract = "not_a_contract_object"
            mock_catalog.load_contract_class.return_value = invalid_contract
            mock_catalog_class.return_value = mock_catalog
            
            manager = ContractDiscoveryManagerAdapter(workspace_root=temp_workspace)
            
            # Should handle gracefully
            input_paths = manager.get_contract_input_paths(invalid_contract, "test_step")
            output_paths = manager.get_contract_output_paths(invalid_contract, "test_step")
            env_vars = manager.get_contract_environ_vars(invalid_contract)
            job_args = manager.get_contract_job_args(invalid_contract, "test_step")
            
            assert input_paths == {}
            assert output_paths == {}
            assert env_vars["CURSUS_ENV"] == "testing"  # Should have defaults
            assert job_args["script_name"] == "test_step"  # Should have defaults
    
    def test_path_adaptation_edge_cases(self, temp_workspace):
        """Test path adaptation with edge cases."""
        with patch('cursus.step_catalog.adapters.contract_adapter.StepCatalog'):
            adapter = ContractDiscoveryManagerAdapter(workspace_root=temp_workspace)
            
            base_data_dir = temp_workspace / "test_step"
            
            # Test with empty path
            result = adapter._adapt_path_for_local_testing("", base_data_dir, "input")
            assert result == base_data_dir / "input" / "data"
            
            # Test with single component path
            result = adapter._adapt_path_for_local_testing("data.csv", base_data_dir, "output")
            assert result == base_data_dir / "output" / "data"
            
            # Test with None path (should handle gracefully)
            try:
                result = adapter._adapt_path_for_local_testing(None, base_data_dir, "input")
                # If it doesn't raise an exception, it should return a reasonable default
                assert base_data_dir in result.parents or result == base_data_dir / "input" / "data"
            except (TypeError, AttributeError):
                # It's acceptable for this to raise an exception with None input
                pass
