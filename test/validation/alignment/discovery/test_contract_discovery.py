"""
Unit tests for cursus.validation.alignment.discovery.contract_discovery module.

Tests the ContractDiscoveryEngine class that handles discovery and mapping of
contract files, including finding contracts by script name, extracting contract
references from specification files, and building entry point mappings.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import sys
import tempfile
import os
from typing import Dict, List, Optional, Any

from cursus.validation.alignment.discovery.contract_discovery import ContractDiscoveryEngine


class TestContractDiscoveryEngine:
    """Test cases for ContractDiscoveryEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.contracts_dir = Path(self.temp_dir) / "contracts"
        self.contracts_dir.mkdir(exist_ok=True)
        
        self.engine = ContractDiscoveryEngine(str(self.contracts_dir))
        
        # Sample contract files
        self.sample_contracts = [
            "model_training_contract.py",
            "data_preprocessing_contract.py",
            "model_evaluation_contract.py",
            "__init__.py"  # Should be ignored
        ]
        
        # Create sample contract files
        for contract_file in self.sample_contracts:
            (self.contracts_dir / contract_file).touch()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test ContractDiscoveryEngine initialization."""
        engine = ContractDiscoveryEngine("/path/to/contracts")
        
        assert engine.contracts_dir == Path("/path/to/contracts")
        assert engine._entry_point_mapping is None
    
    def test_discover_all_contracts_with_files(self):
        """Test discovering all contracts when files exist."""
        contracts = self.engine.discover_all_contracts()
        
        expected = ["data_preprocessing", "model_evaluation", "model_training"]
        assert contracts == expected
    
    def test_discover_all_contracts_empty_directory(self):
        """Test discovering contracts in empty directory."""
        empty_dir = Path(self.temp_dir) / "empty_contracts"
        empty_dir.mkdir()
        
        engine = ContractDiscoveryEngine(str(empty_dir))
        contracts = engine.discover_all_contracts()
        
        assert contracts == []
    
    def test_discover_all_contracts_nonexistent_directory(self):
        """Test discovering contracts when directory doesn't exist."""
        nonexistent_dir = Path(self.temp_dir) / "nonexistent"
        
        engine = ContractDiscoveryEngine(str(nonexistent_dir))
        contracts = engine.discover_all_contracts()
        
        assert contracts == []
    
    def test_discover_all_contracts_ignores_init_files(self):
        """Test that __init__.py files are ignored."""
        contracts = self.engine.discover_all_contracts()
        
        # Should not include __init__ in the results
        assert "__init__" not in contracts
        assert "data_preprocessing" in contracts
    
    @patch('cursus.validation.alignment.discovery.contract_discovery.UnifiedAlignmentTester')
    def test_discover_contracts_with_scripts_success(self, mock_tester_class):
        """Test discovering contracts that have corresponding scripts."""
        # Mock the tester
        mock_tester = Mock()
        mock_tester.discover_scripts.return_value = ["model_training", "data_preprocessing"]
        mock_tester_class.return_value = mock_tester
        
        # Mock contract loading
        with patch.object(self.engine, '_load_contract_for_entry_point') as mock_load:
            mock_load.side_effect = [
                {'entry_point': 'model_training.py'},
                {'entry_point': 'data_preprocessing.py'},
                {'entry_point': 'nonexistent_script.py'}
            ]
            
            contracts = self.engine.discover_contracts_with_scripts()
            
            expected = ["data_preprocessing", "model_training"]
            assert contracts == expected
    
    @patch('cursus.validation.alignment.discovery.contract_discovery.UnifiedAlignmentTester')
    def test_discover_contracts_with_scripts_no_entry_point(self, mock_tester_class):
        """Test discovering contracts when some have no entry_point."""
        # Mock the tester
        mock_tester = Mock()
        mock_tester.discover_scripts.return_value = ["model_training"]
        mock_tester_class.return_value = mock_tester
        
        # Mock contract loading - some with no entry_point
        with patch.object(self.engine, '_load_contract_for_entry_point') as mock_load:
            mock_load.side_effect = [
                {'entry_point': 'model_training.py'},
                {'entry_point': ''},  # No entry point
                {'entry_point': 'model_evaluation.py'}
            ]
            
            contracts = self.engine.discover_contracts_with_scripts()
            
            # Only the one with valid entry_point and existing script
            assert contracts == ["model_training"]
    
    @patch('cursus.validation.alignment.discovery.contract_discovery.UnifiedAlignmentTester')
    def test_discover_contracts_with_scripts_load_error(self, mock_tester_class):
        """Test discovering contracts when some fail to load."""
        # Mock the tester
        mock_tester = Mock()
        mock_tester.discover_scripts.return_value = ["model_training"]
        mock_tester_class.return_value = mock_tester
        
        # Mock contract loading with errors
        with patch.object(self.engine, '_load_contract_for_entry_point') as mock_load:
            mock_load.side_effect = [
                {'entry_point': 'model_training.py'},
                Exception("Failed to load contract"),
                {'entry_point': 'nonexistent.py'}
            ]
            
            contracts = self.engine.discover_contracts_with_scripts()
            
            # Only the successfully loaded contract with existing script
            assert contracts == ["model_training"]
    
    @patch('cursus.validation.alignment.discovery.contract_discovery.UnifiedAlignmentTester')
    def test_discover_contracts_with_scripts_nonexistent_directory(self, mock_tester_class):
        """Test discovering contracts when contracts directory doesn't exist."""
        nonexistent_dir = Path(self.temp_dir) / "nonexistent"
        engine = ContractDiscoveryEngine(str(nonexistent_dir))
        
        contracts = engine.discover_contracts_with_scripts()
        
        assert contracts == []
    
    def test_extract_contract_reference_from_spec_import_patterns(self):
        """Test extracting contract reference from spec file using import patterns."""
        spec_content = """
from ..contracts.model_training import CONTRACT
from ..contracts.data_preprocessing_contract import PREPROCESSING_CONTRACT
"""
        
        with patch('builtins.open', mock_open(read_data=spec_content)):
            spec_file = Path("test_spec.py")
            
            # Test first pattern
            result = self.engine.extract_contract_reference_from_spec(spec_file)
            assert result == "model_training_contract"
    
    def test_extract_contract_reference_from_spec_no_match(self):
        """Test extracting contract reference when no patterns match."""
        spec_content = """
import some_other_module
from pathlib import Path
"""
        
        with patch('builtins.open', mock_open(read_data=spec_content)):
            spec_file = Path("test_spec.py")
            
            result = self.engine.extract_contract_reference_from_spec(spec_file)
            assert result is None
    
    def test_extract_contract_reference_from_spec_file_error(self):
        """Test extracting contract reference when file can't be read."""
        spec_file = Path("nonexistent_spec.py")
        
        with patch('builtins.open', side_effect=FileNotFoundError()):
            result = self.engine.extract_contract_reference_from_spec(spec_file)
            assert result is None
    
    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_extract_script_contract_from_spec_success(self, mock_module_from_spec, mock_spec_from_file):
        """Test extracting script_contract from specification file."""
        # Mock the module loading
        mock_spec = Mock()
        mock_loader = Mock()
        mock_spec.loader = mock_loader
        mock_spec_from_file.return_value = mock_spec
        
        mock_module = Mock()
        mock_module_from_spec.return_value = mock_module
        
        # Mock the specification object
        mock_contract = Mock()
        mock_contract.entry_point = "model_training.py"
        
        mock_spec_obj = Mock()
        mock_spec_obj.script_contract = mock_contract
        
        # Set up module attributes
        mock_module.__dict__ = {
            'MODEL_TRAINING_SPEC': mock_spec_obj,
            'other_attr': 'value'
        }
        
        with patch('builtins.dir', return_value=['MODEL_TRAINING_SPEC', 'other_attr']):
            spec_file = Path("model_training_spec.py")
            result = self.engine.extract_script_contract_from_spec(spec_file)
            
            assert result == "model_training"
    
    @patch('importlib.util.spec_from_file_location')
    def test_extract_script_contract_from_spec_no_spec(self, mock_spec_from_file):
        """Test extracting script_contract when spec loading fails."""
        mock_spec_from_file.return_value = None
        
        spec_file = Path("test_spec.py")
        result = self.engine.extract_script_contract_from_spec(spec_file)
        
        assert result is None
    
    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_extract_script_contract_from_spec_callable_contract(self, mock_module_from_spec, mock_spec_from_file):
        """Test extracting script_contract when contract is callable."""
        # Mock the module loading
        mock_spec = Mock()
        mock_loader = Mock()
        mock_spec.loader = mock_loader
        mock_spec_from_file.return_value = mock_spec
        
        mock_module = Mock()
        mock_module_from_spec.return_value = mock_module
        
        # Mock callable contract
        mock_contract_result = Mock()
        mock_contract_result.entry_point = "data_preprocessing.py"
        
        mock_contract_func = Mock(return_value=mock_contract_result)
        
        mock_spec_obj = Mock()
        mock_spec_obj.script_contract = mock_contract_func
        
        # Set up module attributes
        mock_module.__dict__ = {'DATA_PREPROCESSING_SPEC': mock_spec_obj}
        
        with patch('builtins.dir', return_value=['DATA_PREPROCESSING_SPEC']):
            spec_file = Path("data_preprocessing_spec.py")
            result = self.engine.extract_script_contract_from_spec(spec_file)
            
            assert result == "data_preprocessing"
            mock_contract_func.assert_called_once()
    
    def test_contracts_match_direct_match(self):
        """Test contract matching with direct match."""
        assert self.engine.contracts_match("model_training", "model_training") is True
    
    def test_contracts_match_with_py_extension(self):
        """Test contract matching when spec has .py extension."""
        assert self.engine.contracts_match("model_training.py", "model_training") is True
    
    def test_contracts_match_prefix_match(self):
        """Test contract matching with prefix matching."""
        assert self.engine.contracts_match("model_evaluation", "model_evaluation_xgb") is True
        assert self.engine.contracts_match("model_evaluation_xgb", "model_evaluation") is True
    
    def test_contracts_match_no_match(self):
        """Test contract matching when contracts don't match."""
        assert self.engine.contracts_match("model_training", "data_preprocessing") is False
        assert self.engine.contracts_match("completely_different", "model_training") is False
    
    def test_build_entry_point_mapping_success(self):
        """Test building entry point mapping successfully."""
        with patch.object(self.engine, '_extract_entry_point_from_contract') as mock_extract:
            mock_extract.side_effect = [
                "model_training.py",
                "data_preprocessing.py", 
                "model_evaluation.py",
                None  # __init__.py should be skipped anyway
            ]
            
            mapping = self.engine.build_entry_point_mapping()
            
            expected = {
                "model_training.py": "model_training_contract.py",
                "data_preprocessing.py": "data_preprocessing_contract.py",
                "model_evaluation.py": "model_evaluation_contract.py"
            }
            
            assert mapping == expected
    
    def test_build_entry_point_mapping_cached(self):
        """Test that entry point mapping is cached."""
        # Set up cached mapping
        cached_mapping = {"test.py": "test_contract.py"}
        self.engine._entry_point_mapping = cached_mapping
        
        result = self.engine.build_entry_point_mapping()
        
        assert result == cached_mapping
    
    def test_build_entry_point_mapping_nonexistent_directory(self):
        """Test building entry point mapping when directory doesn't exist."""
        nonexistent_dir = Path(self.temp_dir) / "nonexistent"
        engine = ContractDiscoveryEngine(str(nonexistent_dir))
        
        mapping = engine.build_entry_point_mapping()
        
        assert mapping == {}
        assert engine._entry_point_mapping == {}
    
    def test_build_entry_point_mapping_with_errors(self):
        """Test building entry point mapping when some contracts fail to load."""
        with patch.object(self.engine, '_extract_entry_point_from_contract') as mock_extract:
            mock_extract.side_effect = [
                "model_training.py",
                Exception("Failed to load"),
                "model_evaluation.py"
            ]
            
            mapping = self.engine.build_entry_point_mapping()
            
            # Should only include successfully loaded contracts
            expected = {
                "model_training.py": "model_training_contract.py",
                "model_evaluation.py": "model_evaluation_contract.py"
            }
            
            assert mapping == expected
    
    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_extract_entry_point_from_contract_success(self, mock_module_from_spec, mock_spec_from_file):
        """Test extracting entry point from contract file."""
        # Mock the module loading
        mock_spec = Mock()
        mock_loader = Mock()
        mock_spec.loader = mock_loader
        mock_spec_from_file.return_value = mock_spec
        
        mock_module = Mock()
        mock_module_from_spec.return_value = mock_module
        
        # Mock contract object
        mock_contract = Mock()
        mock_contract.entry_point = "model_training.py"
        
        # Set up module attributes
        mock_module.__dict__ = {
            'MODEL_TRAINING_CONTRACT': mock_contract,
            'other_attr': 'value'
        }
        
        with patch('builtins.dir', return_value=['MODEL_TRAINING_CONTRACT', 'other_attr']):
            contract_path = Path("model_training_contract.py")
            result = self.engine._extract_entry_point_from_contract(contract_path)
            
            assert result == "model_training.py"
    
    @patch('importlib.util.spec_from_file_location')
    def test_extract_entry_point_from_contract_no_spec(self, mock_spec_from_file):
        """Test extracting entry point when spec loading fails."""
        mock_spec_from_file.return_value = None
        
        contract_path = Path("test_contract.py")
        result = self.engine._extract_entry_point_from_contract(contract_path)
        
        assert result is None
    
    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_extract_entry_point_from_contract_no_contract_object(self, mock_module_from_spec, mock_spec_from_file):
        """Test extracting entry point when no contract object found."""
        # Mock the module loading
        mock_spec = Mock()
        mock_loader = Mock()
        mock_spec.loader = mock_loader
        mock_spec_from_file.return_value = mock_spec
        
        mock_module = Mock()
        mock_module_from_spec.return_value = mock_module
        
        # Set up module with no contract objects
        mock_module.__dict__ = {'other_attr': 'value'}
        
        with patch('builtins.dir', return_value=['other_attr']):
            contract_path = Path("test_contract.py")
            result = self.engine._extract_entry_point_from_contract(contract_path)
            
            assert result is None
    
    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_load_contract_for_entry_point_success(self, mock_module_from_spec, mock_spec_from_file):
        """Test loading contract for entry point extraction."""
        # Mock the module loading
        mock_spec = Mock()
        mock_loader = Mock()
        mock_spec.loader = mock_loader
        mock_spec_from_file.return_value = mock_spec
        
        mock_module = Mock()
        mock_module_from_spec.return_value = mock_module
        
        # Mock contract object
        mock_contract = Mock()
        mock_contract.entry_point = "model_training.py"
        
        # Set up module attributes
        mock_module.__dict__ = {'MODEL_TRAINING_CONTRACT': mock_contract}
        
        with patch('builtins.dir', return_value=['MODEL_TRAINING_CONTRACT']):
            contract_path = Path("model_training_contract.py")
            result = self.engine._load_contract_for_entry_point(contract_path, "model_training")
            
            expected = {'entry_point': 'model_training.py'}
            assert result == expected
    
    @patch('importlib.util.spec_from_file_location')
    def test_load_contract_for_entry_point_no_spec(self, mock_spec_from_file):
        """Test loading contract when spec creation fails."""
        mock_spec_from_file.return_value = None
        
        contract_path = Path("test_contract.py")
        
        with pytest.raises(Exception) as exc_info:
            self.engine._load_contract_for_entry_point(contract_path, "test")
        
        assert "Could not load contract module" in str(exc_info.value)
    
    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_load_contract_for_entry_point_no_contract_found(self, mock_module_from_spec, mock_spec_from_file):
        """Test loading contract when no contract object is found."""
        # Mock the module loading
        mock_spec = Mock()
        mock_loader = Mock()
        mock_spec.loader = mock_loader
        mock_spec_from_file.return_value = mock_spec
        
        mock_module = Mock()
        mock_module_from_spec.return_value = mock_module
        
        # Set up module with no valid contract objects
        mock_module.__dict__ = {'other_attr': 'value'}
        
        with patch('builtins.dir', return_value=['other_attr']):
            contract_path = Path("test_contract.py")
            
            with pytest.raises(Exception) as exc_info:
                self.engine._load_contract_for_entry_point(contract_path, "test")
            
            assert "No contract object found" in str(exc_info.value)
    
    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_load_contract_for_entry_point_multiple_naming_patterns(self, mock_module_from_spec, mock_spec_from_file):
        """Test loading contract with various naming patterns."""
        # Mock the module loading
        mock_spec = Mock()
        mock_loader = Mock()
        mock_spec.loader = mock_loader
        mock_spec_from_file.return_value = mock_spec
        
        mock_module = Mock()
        mock_module_from_spec.return_value = mock_module
        
        # Mock contract object with different naming
        mock_contract = Mock()
        mock_contract.entry_point = "model_evaluation.py"
        
        # Set up module with CONTRACT naming pattern
        mock_module.__dict__ = {'CONTRACT': mock_contract}
        
        with patch('builtins.dir', return_value=['CONTRACT']):
            contract_path = Path("model_evaluation_xgb_contract.py")
            result = self.engine._load_contract_for_entry_point(contract_path, "model_evaluation_xgb")
            
            expected = {'entry_point': 'model_evaluation.py'}
            assert result == expected


class TestContractDiscoveryEngineIntegration:
    """Integration test cases for ContractDiscoveryEngine."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.contracts_dir = Path(self.temp_dir) / "contracts"
        self.contracts_dir.mkdir(exist_ok=True)
        
        self.engine = ContractDiscoveryEngine(str(self.contracts_dir))
    
    def teardown_method(self):
        """Clean up integration test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_discovery_workflow(self):
        """Test complete discovery workflow with real file operations."""
        # Create contract files
        contract_files = [
            "model_training_contract.py",
            "data_preprocessing_contract.py",
            "invalid_contract.py"
        ]
        
        for contract_file in contract_files:
            (self.contracts_dir / contract_file).touch()
        
        # Test discovery
        all_contracts = self.engine.discover_all_contracts()
        expected_contracts = ["data_preprocessing", "invalid_contract", "model_training"]
        
        assert all_contracts == expected_contracts
    
    def test_sys_path_management(self):
        """Test that sys.path is properly managed during contract loading."""
        original_path = sys.path.copy()
        
        # Create a contract file
        contract_file = self.contracts_dir / "test_contract.py"
        contract_file.touch()
        
        # Mock the loading to simulate path manipulation
        with patch.object(self.engine, '_extract_entry_point_from_contract') as mock_extract:
            mock_extract.return_value = "test.py"
            
            # This should not permanently modify sys.path
            self.engine.build_entry_point_mapping()
            
            # Verify sys.path is restored
            assert sys.path == original_path
    
    def test_error_resilience_in_discovery(self):
        """Test that discovery continues even when some contracts fail to load."""
        # Create mix of valid and invalid contract files
        (self.contracts_dir / "valid_contract.py").touch()
        (self.contracts_dir / "another_valid_contract.py").touch()
        
        with patch.object(self.engine, '_extract_entry_point_from_contract') as mock_extract:
            # Simulate some contracts failing to load
            mock_extract.side_effect = [
                "valid.py",
                Exception("Load failed"),
            ]
            
            mapping = self.engine.build_entry_point_mapping()
            
            # Should still get the valid contract
            assert "valid.py" in mapping
            assert mapping["valid.py"] == "valid_contract.py"


if __name__ == '__main__':
    pytest.main([__file__])
