"""
Unit tests for cursus.validation.alignment.loaders.contract_loader module.

Tests the ContractLoader class that handles loading and parsing of script
contracts from Python files with robust import handling and contract object
extraction.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import tempfile
from typing import Dict, Any, Optional

from cursus.validation.alignment.loaders.contract_loader import ContractLoader


class TestContractLoader:
    """Test cases for ContractLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.contracts_dir = Path(self.temp_dir) / "contracts"
        self.contracts_dir.mkdir(exist_ok=True)
        
        self.loader = ContractLoader(str(self.contracts_dir))
        
        # Sample contract object
        self.sample_contract = Mock()
        self.sample_contract.entry_point = "model_training.py"
        self.sample_contract.expected_input_paths = {
            'training_data': '/opt/ml/input/data/training',
            'validation_data': '/opt/ml/input/data/validation'
        }
        self.sample_contract.expected_output_paths = {
            'model': '/opt/ml/model',
            'metrics': '/opt/ml/output/metrics.json'
        }
        self.sample_contract.expected_arguments = {
            'learning_rate': 0.01,
            'max_depth': 6,
            'required_param': None
        }
        self.sample_contract.required_env_vars = ['AWS_REGION', 'SAGEMAKER_JOB_NAME']
        self.sample_contract.optional_env_vars = {'DEBUG': 'false', 'VERBOSE': 'true'}
        self.sample_contract.description = "Model training contract"
        self.sample_contract.framework_requirements = {'xgboost': '>=1.0.0'}
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test ContractLoader initialization."""
        loader = ContractLoader("/path/to/contracts")
        
        assert loader.contracts_dir == Path("/path/to/contracts")
    
    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_load_contract_success(self, mock_module_from_spec, mock_spec_from_file):
        """Test successful contract loading."""
        # Mock the module loading
        mock_spec = Mock()
        mock_loader = Mock()
        mock_spec.loader = mock_loader
        mock_spec_from_file.return_value = mock_spec
        
        mock_module = Mock()
        mock_module_from_spec.return_value = mock_module
        
        # Set up module with contract object
        mock_module.__dict__ = {'MODEL_TRAINING_CONTRACT': self.sample_contract}
        
        with patch.object(self.loader, '_find_contract_object', return_value=self.sample_contract):
            contract_path = Path("model_training_contract.py")
            result = self.loader.load_contract(contract_path, "model_training")
            
            # Verify the result structure
            assert result['entry_point'] == "model_training.py"
            assert 'inputs' in result
            assert 'outputs' in result
            assert 'arguments' in result
            assert 'environment_variables' in result
            assert result['description'] == "Model training contract"
    
    @patch('importlib.util.spec_from_file_location')
    def test_load_contract_no_spec(self, mock_spec_from_file):
        """Test contract loading when spec creation fails."""
        mock_spec_from_file.return_value = None
        
        contract_path = Path("test_contract.py")
        
        with pytest.raises(Exception) as exc_info:
            self.loader.load_contract(contract_path, "test")
        
        assert "Could not load contract module" in str(exc_info.value)
    
    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_load_contract_no_contract_object(self, mock_module_from_spec, mock_spec_from_file):
        """Test contract loading when no contract object is found."""
        # Mock the module loading
        mock_spec = Mock()
        mock_loader = Mock()
        mock_spec.loader = mock_loader
        mock_spec_from_file.return_value = mock_spec
        
        mock_module = Mock()
        mock_module_from_spec.return_value = mock_module
        
        with patch.object(self.loader, '_find_contract_object', return_value=None):
            contract_path = Path("test_contract.py")
            
            with pytest.raises(Exception) as exc_info:
                self.loader.load_contract(contract_path, "test")
            
            assert "No contract object found" in str(exc_info.value)
    
    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_load_contract_sys_path_management(self, mock_module_from_spec, mock_spec_from_file):
        """Test that sys.path is properly managed during contract loading."""
        original_path = sys.path.copy()
        
        # Mock the module loading
        mock_spec = Mock()
        mock_loader = Mock()
        mock_spec.loader = mock_loader
        mock_spec_from_file.return_value = mock_spec
        
        mock_module = Mock()
        mock_module_from_spec.return_value = mock_module
        
        with patch.object(self.loader, '_find_contract_object', return_value=self.sample_contract):
            contract_path = Path("test_contract.py")
            self.loader.load_contract(contract_path, "test")
            
            # Verify sys.path is restored
            assert sys.path == original_path
    
    def test_find_contract_object_standard_naming(self):
        """Test finding contract object with standard naming patterns."""
        mock_module = Mock()
        mock_module.__dict__ = {
            'MODEL_TRAINING_CONTRACT': self.sample_contract,
            'other_attr': 'value'
        }
        
        with patch('builtins.dir', return_value=['MODEL_TRAINING_CONTRACT', 'other_attr']):
            result = self.loader._find_contract_object(mock_module, "model_training")
            
            assert result == self.sample_contract
    
    def test_find_contract_object_generic_naming(self):
        """Test finding contract object with generic CONTRACT naming."""
        mock_module = Mock()
        mock_module.__dict__ = {
            'CONTRACT': self.sample_contract,
            'other_attr': 'value'
        }
        
        with patch('builtins.dir', return_value=['CONTRACT', 'other_attr']):
            result = self.loader._find_contract_object(mock_module, "model_training")
            
            assert result == self.sample_contract
    
    def test_find_contract_object_dynamic_discovery(self):
        """Test finding contract object through dynamic discovery."""
        mock_module = Mock()
        mock_module.__dict__ = {
            'CUSTOM_TRAINING_CONTRACT': self.sample_contract,
            'other_attr': 'value'
        }
        
        with patch('builtins.dir', return_value=['CUSTOM_TRAINING_CONTRACT', 'other_attr']):
            result = self.loader._find_contract_object(mock_module, "model_training")
            
            assert result == self.sample_contract
    
    def test_find_contract_object_no_entry_point(self):
        """Test finding contract object when object has no entry_point."""
        mock_invalid_contract = Mock()
        del mock_invalid_contract.entry_point  # Remove entry_point attribute
        
        mock_module = Mock()
        mock_module.__dict__ = {
            'INVALID_CONTRACT': mock_invalid_contract,
            'MODEL_TRAINING_CONTRACT': self.sample_contract
        }
        
        with patch('builtins.dir', return_value=['INVALID_CONTRACT', 'MODEL_TRAINING_CONTRACT']):
            result = self.loader._find_contract_object(mock_module, "model_training")
            
            # Should find the valid contract, not the invalid one
            assert result == self.sample_contract
    
    def test_find_contract_object_not_found(self):
        """Test finding contract object when none exists."""
        mock_module = Mock()
        mock_module.__dict__ = {'other_attr': 'value'}
        
        with patch('builtins.dir', return_value=['other_attr']):
            result = self.loader._find_contract_object(mock_module, "model_training")
            
            assert result is None
    
    def test_find_contract_object_specific_patterns(self):
        """Test finding contract object with specific naming patterns."""
        # Test XGBOOST_MODEL_EVAL_CONTRACT pattern
        mock_module = Mock()
        mock_module.__dict__ = {'XGBOOST_MODEL_EVAL_CONTRACT': self.sample_contract}
        
        with patch('builtins.dir', return_value=['XGBOOST_MODEL_EVAL_CONTRACT']):
            result = self.loader._find_contract_object(mock_module, "model_evaluation_xgb")
            
            assert result == self.sample_contract
    
    def test_contract_to_dict_complete(self):
        """Test converting complete contract object to dictionary."""
        result = self.loader._contract_to_dict(self.sample_contract, "model_training")
        
        # Verify structure
        assert result['entry_point'] == "model_training.py"
        assert result['description'] == "Model training contract"
        assert result['framework_requirements'] == {'xgboost': '>=1.0.0'}
        
        # Verify inputs conversion
        assert 'training_data' in result['inputs']
        assert result['inputs']['training_data']['path'] == '/opt/ml/input/data/training'
        
        # Verify outputs conversion
        assert 'model' in result['outputs']
        assert result['outputs']['model']['path'] == '/opt/ml/model'
        
        # Verify arguments conversion
        assert 'learning_rate' in result['arguments']
        assert result['arguments']['learning_rate']['default'] == 0.01
        assert result['arguments']['learning_rate']['required'] is False
        assert result['arguments']['required_param']['required'] is True
        
        # Verify environment variables
        assert result['environment_variables']['required'] == ['AWS_REGION', 'SAGEMAKER_JOB_NAME']
        assert result['environment_variables']['optional'] == {'DEBUG': 'false', 'VERBOSE': 'true'}
    
    def test_contract_to_dict_minimal(self):
        """Test converting minimal contract object to dictionary."""
        minimal_contract = Mock()
        minimal_contract.entry_point = "simple_script.py"
        
        # Remove optional attributes
        for attr in ['expected_input_paths', 'expected_output_paths', 'expected_arguments',
                    'required_env_vars', 'optional_env_vars', 'description', 'framework_requirements']:
            if hasattr(minimal_contract, attr):
                delattr(minimal_contract, attr)
        
        result = self.loader._contract_to_dict(minimal_contract, "simple_script")
        
        # Verify defaults
        assert result['entry_point'] == "simple_script.py"
        assert result['inputs'] == {}
        assert result['outputs'] == {}
        assert result['arguments'] == {}
        assert result['environment_variables']['required'] == []
        assert result['environment_variables']['optional'] == {}
        assert result['description'] == ''
        assert result['framework_requirements'] == {}
    
    def test_contract_to_dict_default_entry_point(self):
        """Test contract to dict with default entry point generation."""
        contract_no_entry = Mock()
        # Remove entry_point attribute
        if hasattr(contract_no_entry, 'entry_point'):
            delattr(contract_no_entry, 'entry_point')
        
        result = self.loader._contract_to_dict(contract_no_entry, "test_contract")
        
        assert result['entry_point'] == "test_contract.py"
    
    def test_contract_to_dict_empty_collections(self):
        """Test contract to dict with empty collections."""
        empty_contract = Mock()
        empty_contract.entry_point = "test.py"
        empty_contract.expected_input_paths = {}
        empty_contract.expected_output_paths = {}
        empty_contract.expected_arguments = {}
        empty_contract.required_env_vars = []
        empty_contract.optional_env_vars = {}
        
        result = self.loader._contract_to_dict(empty_contract, "test")
        
        assert result['inputs'] == {}
        assert result['outputs'] == {}
        assert result['arguments'] == {}
        assert result['environment_variables']['required'] == []
        assert result['environment_variables']['optional'] == {}


class TestContractLoaderIntegration:
    """Integration test cases for ContractLoader."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.contracts_dir = Path(self.temp_dir) / "contracts"
        self.contracts_dir.mkdir(exist_ok=True)
        
        self.loader = ContractLoader(str(self.contracts_dir))
    
    def teardown_method(self):
        """Clean up integration test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_contract_error_handling(self):
        """Test contract loading with various error conditions."""
        # Test with non-existent file
        nonexistent_path = Path("nonexistent_contract.py")
        
        with pytest.raises(Exception) as exc_info:
            self.loader.load_contract(nonexistent_path, "nonexistent")
        
        assert "Failed to load Python contract" in str(exc_info.value)
    
    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_load_contract_import_error(self, mock_module_from_spec, mock_spec_from_file):
        """Test contract loading when module import fails."""
        # Mock spec creation success but loader execution failure
        mock_spec = Mock()
        mock_loader = Mock()
        mock_loader.exec_module.side_effect = ImportError("Import failed")
        mock_spec.loader = mock_loader
        mock_spec_from_file.return_value = mock_spec
        
        mock_module = Mock()
        mock_module_from_spec.return_value = mock_module
        
        contract_path = Path("failing_contract.py")
        
        with pytest.raises(Exception) as exc_info:
            self.loader.load_contract(contract_path, "failing")
        
        assert "Failed to load Python contract" in str(exc_info.value)
    
    def test_find_contract_object_edge_cases(self):
        """Test finding contract object with edge cases."""
        # Test with module that has attributes but none are contracts
        mock_module = Mock()
        mock_non_contract = Mock()
        # Remove entry_point to make it invalid
        if hasattr(mock_non_contract, 'entry_point'):
            delattr(mock_non_contract, 'entry_point')
        
        mock_module.__dict__ = {
            'SOME_CONTRACT': mock_non_contract,
            'OTHER_ATTR': 'value'
        }
        
        with patch('builtins.dir', return_value=['SOME_CONTRACT', 'OTHER_ATTR']):
            result = self.loader._find_contract_object(mock_module, "test")
            
            assert result is None
    
    def test_contract_to_dict_with_none_values(self):
        """Test contract to dict conversion with None values."""
        contract_with_nones = Mock()
        contract_with_nones.entry_point = "test.py"
        contract_with_nones.expected_input_paths = None
        contract_with_nones.expected_output_paths = None
        contract_with_nones.expected_arguments = None
        contract_with_nones.required_env_vars = None
        contract_with_nones.optional_env_vars = None
        contract_with_nones.description = None
        contract_with_nones.framework_requirements = None
        
        # This should handle None values gracefully
        result = self.loader._contract_to_dict(contract_with_nones, "test")
        
        assert result['entry_point'] == "test.py"
        # Should handle None values by using defaults
        assert isinstance(result['inputs'], dict)
        assert isinstance(result['outputs'], dict)
        assert isinstance(result['arguments'], dict)


class TestContractLoaderErrorScenarios:
    """Test cases for error scenarios and edge cases."""
    
    def setup_method(self):
        """Set up error scenario test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.contracts_dir = Path(self.temp_dir) / "contracts"
        self.contracts_dir.mkdir(exist_ok=True)
        
        self.loader = ContractLoader(str(self.contracts_dir))
    
    def teardown_method(self):
        """Clean up error scenario test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_load_contract_module_execution_error(self, mock_module_from_spec, mock_spec_from_file):
        """Test contract loading when module execution fails."""
        mock_spec = Mock()
        mock_loader = Mock()
        mock_loader.exec_module.side_effect = RuntimeError("Module execution failed")
        mock_spec.loader = mock_loader
        mock_spec_from_file.return_value = mock_spec
        
        mock_module = Mock()
        mock_module_from_spec.return_value = mock_module
        
        contract_path = Path("error_contract.py")
        
        with pytest.raises(Exception) as exc_info:
            self.loader.load_contract(contract_path, "error")
        
        assert "Failed to load Python contract" in str(exc_info.value)
        assert "Module execution failed" in str(exc_info.value)
    
    def test_find_contract_object_multiple_valid_contracts(self):
        """Test finding contract object when multiple valid contracts exist."""
        contract1 = Mock()
        contract1.entry_point = "script1.py"
        
        contract2 = Mock()
        contract2.entry_point = "script2.py"
        
        mock_module = Mock()
        mock_module.__dict__ = {
            'FIRST_CONTRACT': contract1,
            'SECOND_CONTRACT': contract2
        }
        
        with patch('builtins.dir', return_value=['FIRST_CONTRACT', 'SECOND_CONTRACT']):
            # Should return the first valid contract found
            result = self.loader._find_contract_object(mock_module, "test")
            
            assert result in [contract1, contract2]  # Either is acceptable
            assert hasattr(result, 'entry_point')
    
    def test_contract_to_dict_complex_arguments(self):
        """Test contract to dict with complex argument types."""
        complex_contract = Mock()
        complex_contract.entry_point = "complex.py"
        complex_contract.expected_arguments = {
            'string_param': 'default_string',
            'int_param': 42,
            'float_param': 3.14,
            'bool_param': True,
            'list_param': [1, 2, 3],
            'dict_param': {'key': 'value'},
            'none_param': None
        }
        
        result = self.loader._contract_to_dict(complex_contract, "complex")
        
        # Verify all argument types are handled
        args = result['arguments']
        assert args['string_param']['default'] == 'default_string'
        assert args['string_param']['required'] is False
        assert args['int_param']['default'] == 42
        assert args['float_param']['default'] == 3.14
        assert args['bool_param']['default'] is True
        assert args['list_param']['default'] == [1, 2, 3]
        assert args['dict_param']['default'] == {'key': 'value'}
        assert args['none_param']['required'] is True  # None means required


if __name__ == '__main__':
    pytest.main([__file__])
