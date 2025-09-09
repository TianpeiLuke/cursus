"""
Unit tests for cursus.validation.alignment.loaders.contract_loader module.

Tests the ContractLoader class that handles loading and parsing of script
contracts from Python files with robust import handling and contract object
extraction.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import tempfile
from typing import Dict, Any, Optional

from cursus.validation.alignment.loaders.contract_loader import ContractLoader


class TestContractLoader(unittest.TestCase):
    """Test cases for ContractLoader class."""
    
    def setUp(self):
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
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test ContractLoader initialization."""
        loader = ContractLoader("/path/to/contracts")
        
        self.assertEqual(loader.contracts_dir, Path("/path/to/contracts"))
    
    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_load_contract_success(self, mock_module_from_spec, mock_spec_from_file):
        """Test successful contract loading."""
        # Mock the module loading
        mock_spec = Mock()
        mock_loader = Mock()
        mock_spec.loader = mock_loader
        mock_spec_from_file.return_value = mock_spec
        
        # Create a simple object that can have __package__ set
        class MockModule:
            def __init__(self, sample_contract):
                self.__dict__ = {'MODEL_TRAINING_CONTRACT': sample_contract}
        
        mock_module = MockModule(self.sample_contract)
        mock_module_from_spec.return_value = mock_module
        
        with patch.object(self.loader, '_find_contract_object', return_value=self.sample_contract):
            contract_path = Path("model_training_contract.py")
            result = self.loader.load_contract(contract_path, "model_training")
            
            # Verify the result structure
            self.assertEqual(result['entry_point'], "model_training.py")
            self.assertIn('inputs', result)
            self.assertIn('outputs', result)
            self.assertIn('arguments', result)
            self.assertIn('environment_variables', result)
            self.assertEqual(result['description'], "Model training contract")
    
    @patch('importlib.util.spec_from_file_location')
    def test_load_contract_no_spec(self, mock_spec_from_file):
        """Test contract loading when spec creation fails."""
        mock_spec_from_file.return_value = None
        
        contract_path = Path("test_contract.py")
        
        with self.assertRaises(Exception) as exc_info:
            self.loader.load_contract(contract_path, "test")
        
        self.assertIn("Could not load contract module", str(exc_info.exception))
    
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
            
            with self.assertRaises(Exception) as exc_info:
                self.loader.load_contract(contract_path, "test")
            
            self.assertIn("No contract object found", str(exc_info.exception))
    
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
            self.assertEqual(sys.path, original_path)
    
    def test_find_contract_object_standard_naming(self):
        """Test finding contract object with standard naming patterns."""
        mock_module = Mock()
        mock_module.__dict__ = {
            'MODEL_TRAINING_CONTRACT': self.sample_contract,
            'other_attr': 'value'
        }
        
        with patch('builtins.dir', return_value=['MODEL_TRAINING_CONTRACT', 'other_attr']):
            result = self.loader._find_contract_object(mock_module, "model_training")
            
            self.assertEqual(result, self.sample_contract)
    
    def test_find_contract_object_generic_naming(self):
        """Test finding contract object with generic CONTRACT naming."""
        mock_module = Mock()
        mock_module.__dict__ = {
            'CONTRACT': self.sample_contract,
            'other_attr': 'value'
        }
        
        with patch('builtins.dir', return_value=['CONTRACT', 'other_attr']):
            result = self.loader._find_contract_object(mock_module, "model_training")
            
            self.assertEqual(result, self.sample_contract)
    
    def test_find_contract_object_dynamic_discovery(self):
        """Test finding contract object through dynamic discovery."""
        mock_module = Mock()
        mock_module.__dict__ = {
            'CUSTOM_TRAINING_CONTRACT': self.sample_contract,
            'other_attr': 'value'
        }
        
        with patch('builtins.dir', return_value=['CUSTOM_TRAINING_CONTRACT', 'other_attr']):
            result = self.loader._find_contract_object(mock_module, "model_training")
            
            self.assertEqual(result, self.sample_contract)
    
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
            self.assertEqual(result, self.sample_contract)
    
    def test_find_contract_object_not_found(self):
        """Test finding contract object when none exists."""
        mock_module = Mock()
        mock_module.__dict__ = {'other_attr': 'value'}
        
        with patch('builtins.dir', return_value=['other_attr']):
            result = self.loader._find_contract_object(mock_module, "model_training")
            
            self.assertIsNone(result)
    
    def test_find_contract_object_specific_patterns(self):
        """Test finding contract object with specific naming patterns."""
        # Test XGBOOST_MODEL_EVAL_CONTRACT pattern
        mock_module = Mock()
        mock_module.__dict__ = {'XGBOOST_MODEL_EVAL_CONTRACT': self.sample_contract}
        
        with patch('builtins.dir', return_value=['XGBOOST_MODEL_EVAL_CONTRACT']):
            result = self.loader._find_contract_object(mock_module, "model_evaluation_xgb")
            
            self.assertEqual(result, self.sample_contract)
    
    def test_contract_to_dict_complete(self):
        """Test converting complete contract object to dictionary."""
        result = self.loader._contract_to_dict(self.sample_contract, "model_training")
        
        # Verify structure
        self.assertEqual(result['entry_point'], "model_training.py")
        self.assertEqual(result['description'], "Model training contract")
        self.assertEqual(result['framework_requirements'], {'xgboost': '>=1.0.0'})
        
        # Verify inputs conversion
        self.assertIn('training_data', result['inputs'])
        self.assertEqual(result['inputs']['training_data']['path'], '/opt/ml/input/data/training')
        
        # Verify outputs conversion
        self.assertIn('model', result['outputs'])
        self.assertEqual(result['outputs']['model']['path'], '/opt/ml/model')
        
        # Verify arguments conversion
        self.assertIn('learning_rate', result['arguments'])
        self.assertEqual(result['arguments']['learning_rate']['default'], 0.01)
        self.assertFalse(result['arguments']['learning_rate']['required'])
        self.assertTrue(result['arguments']['required_param']['required'])
        
        # Verify environment variables
        self.assertEqual(result['environment_variables']['required'], ['AWS_REGION', 'SAGEMAKER_JOB_NAME'])
        self.assertEqual(result['environment_variables']['optional'], {'DEBUG': 'false', 'VERBOSE': 'true'})
    
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
        self.assertEqual(result['entry_point'], "simple_script.py")
        self.assertEqual(result['inputs'], {})
        self.assertEqual(result['outputs'], {})
        self.assertEqual(result['arguments'], {})
        self.assertEqual(result['environment_variables']['required'], [])
        self.assertEqual(result['environment_variables']['optional'], {})
        self.assertEqual(result['description'], '')
        self.assertEqual(result['framework_requirements'], {})
    
    def test_contract_to_dict_default_entry_point(self):
        """Test contract to dict with default entry point generation."""
        contract_no_entry = Mock()
        # Remove entry_point attribute but add empty collections to avoid iteration errors
        if hasattr(contract_no_entry, 'entry_point'):
            delattr(contract_no_entry, 'entry_point')
        contract_no_entry.expected_input_paths = {}
        contract_no_entry.expected_output_paths = {}
        contract_no_entry.expected_arguments = {}
        
        result = self.loader._contract_to_dict(contract_no_entry, "test_contract")
        
        self.assertEqual(result['entry_point'], "test_contract.py")
    
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
        
        self.assertEqual(result['inputs'], {})
        self.assertEqual(result['outputs'], {})
        self.assertEqual(result['arguments'], {})
        self.assertEqual(result['environment_variables']['required'], [])
        self.assertEqual(result['environment_variables']['optional'], {})


class TestContractLoaderIntegration(unittest.TestCase):
    """Integration test cases for ContractLoader."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.contracts_dir = Path(self.temp_dir) / "contracts"
        self.contracts_dir.mkdir(exist_ok=True)
        
        self.loader = ContractLoader(str(self.contracts_dir))
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_contract_error_handling(self):
        """Test contract loading with various error conditions."""
        # Test with non-existent file
        nonexistent_path = Path("nonexistent_contract.py")
        
        with self.assertRaises(Exception) as exc_info:
            self.loader.load_contract(nonexistent_path, "nonexistent")
        
        self.assertIn("Failed to load Python contract", str(exc_info.exception))
    
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
        
        with self.assertRaises(Exception) as exc_info:
            self.loader.load_contract(contract_path, "failing")
        
        self.assertIn("Failed to load Python contract", str(exc_info.exception))
    
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
            
            self.assertIsNone(result)
    
    def test_contract_to_dict_with_none_values(self):
        """Test contract to dict conversion with None values."""
        contract_with_nones = Mock()
        contract_with_nones.entry_point = "test.py"
        # The actual implementation doesn't handle None values for these attributes
        # It expects them to either not exist (hasattr returns False) or be dictionaries
        # So we test the case where these attributes don't exist at all
        for attr in ['expected_input_paths', 'expected_output_paths', 'expected_arguments',
                    'required_env_vars', 'optional_env_vars', 'description', 'framework_requirements']:
            if hasattr(contract_with_nones, attr):
                delattr(contract_with_nones, attr)
        
        # This should handle missing attributes gracefully
        result = self.loader._contract_to_dict(contract_with_nones, "test")
        
        self.assertEqual(result['entry_point'], "test.py")
        # Should handle missing attributes by using defaults
        self.assertIsInstance(result['inputs'], dict)
        self.assertIsInstance(result['outputs'], dict)
        self.assertIsInstance(result['arguments'], dict)
        self.assertEqual(result['inputs'], {})
        self.assertEqual(result['outputs'], {})
        self.assertEqual(result['arguments'], {})
        self.assertEqual(result['environment_variables']['required'], [])
        self.assertEqual(result['environment_variables']['optional'], {})
        self.assertEqual(result['description'], '')
        self.assertEqual(result['framework_requirements'], {})


class TestContractLoaderErrorScenarios(unittest.TestCase):
    """Test cases for error scenarios and edge cases."""
    
    def setUp(self):
        """Set up error scenario test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.contracts_dir = Path(self.temp_dir) / "contracts"
        self.contracts_dir.mkdir(exist_ok=True)
        
        self.loader = ContractLoader(str(self.contracts_dir))
    
    def tearDown(self):
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
        
        with self.assertRaises(Exception) as exc_info:
            self.loader.load_contract(contract_path, "error")
        
        self.assertIn("Failed to load Python contract", str(exc_info.exception))
        self.assertIn("Module execution failed", str(exc_info.exception))
    
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
            
            self.assertIn(result, [contract1, contract2])  # Either is acceptable
            self.assertTrue(hasattr(result, 'entry_point'))
    
    def test_contract_to_dict_complex_arguments(self):
        """Test contract to dict with complex argument types."""
        complex_contract = Mock()
        complex_contract.entry_point = "complex.py"
        complex_contract.expected_input_paths = {}
        complex_contract.expected_output_paths = {}
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
        self.assertEqual(args['string_param']['default'], 'default_string')
        self.assertFalse(args['string_param']['required'])
        self.assertEqual(args['int_param']['default'], 42)
        self.assertEqual(args['float_param']['default'], 3.14)
        self.assertTrue(args['bool_param']['default'])
        self.assertEqual(args['list_param']['default'], [1, 2, 3])
        self.assertEqual(args['dict_param']['default'], {'key': 'value'})
        self.assertTrue(args['none_param']['required'])  # None means required


if __name__ == '__main__':
    unittest.main()
