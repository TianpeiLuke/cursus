"""
Test suite for script contract argument validation.
"""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.cursus.validation.alignment.script_contract_alignment import (
    ScriptContractAlignmentTester
)


class TestArgumentValidation(unittest.TestCase):
    """Test argument validation in script contract alignment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.scripts_dir = Path(self.temp_dir) / "scripts"
        self.contracts_dir = Path(self.temp_dir) / "contracts"
        
        self.scripts_dir.mkdir()
        self.contracts_dir.mkdir()
        
        self.tester = ScriptContractAlignmentTester(
            str(self.scripts_dir),
            str(self.contracts_dir)
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_argument_validation_success(self):
        """Test successful argument validation."""
        # Create test script with arguments
        script_content = '''
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model-name', required=True, type=str)
parser.add_argument('--epochs', required=False, type=int, default=10)
args = parser.parse_args()
'''
        script_path = self.scripts_dir / "test_script.py"
        script_path.write_text(script_content)
        
        # Create matching contract
        contract = {
            "inputs": {},
            "outputs": {},
            "environment_variables": {"required": [], "optional": []},
            "arguments": {
                "model-name": {
                    "required": True,
                    "type": "str",
                    "description": "Name of the model"
                },
                "epochs": {
                    "required": False,
                    "type": "int",
                    "default": 10,
                    "description": "Number of training epochs"
                }
            }
        }
        
        contract_path = self.contracts_dir / "test_script_contract.json"
        contract_path.write_text(json.dumps(contract, indent=2))
        
        # Mock the script analyzer
        with patch('src.cursus.validation.alignment.script_contract_alignment.ScriptAnalyzer') as mock_analyzer:
            mock_instance = MagicMock()
            mock_analyzer.return_value = mock_instance
            
            # Mock argument definitions
            mock_arg1 = MagicMock()
            mock_arg1.argument_name = "model-name"
            mock_arg1.is_required = True
            mock_arg1.argument_type = "str"
            
            mock_arg2 = MagicMock()
            mock_arg2.argument_name = "epochs"
            mock_arg2.is_required = False
            mock_arg2.argument_type = "int"
            
            mock_instance.get_all_analysis_results.return_value = {
                'path_references': [],
                'env_var_accesses': [],
                'argument_definitions': [mock_arg1, mock_arg2],
                'file_operations': []
            }
            
            result = self.tester.validate_script("test_script")
            
            self.assertTrue(result['passed'])
            # Should have no argument-related issues
            arg_issues = [issue for issue in result['issues'] if issue['category'] == 'arguments']
            self.assertEqual(len(arg_issues), 0)
    
    def test_missing_required_argument(self):
        """Test validation when script is missing required argument."""
        # Create test script missing required argument
        script_content = '''
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', required=False, type=int, default=10)
args = parser.parse_args()
'''
        script_path = self.scripts_dir / "test_script.py"
        script_path.write_text(script_content)
        
        # Contract requires model-name argument
        contract = {
            "inputs": {},
            "outputs": {},
            "environment_variables": {"required": [], "optional": []},
            "arguments": {
                "model-name": {
                    "required": True,
                    "type": "str",
                    "description": "Name of the model"
                },
                "epochs": {
                    "required": False,
                    "type": "int",
                    "default": 10
                }
            }
        }
        
        contract_path = self.contracts_dir / "test_script_contract.json"
        contract_path.write_text(json.dumps(contract, indent=2))
        
        # Mock the script analyzer
        with patch('src.cursus.validation.alignment.script_contract_alignment.ScriptAnalyzer') as mock_analyzer:
            mock_instance = MagicMock()
            mock_analyzer.return_value = mock_instance
            
            # Only epochs argument defined in script
            mock_arg = MagicMock()
            mock_arg.argument_name = "epochs"
            mock_arg.is_required = False
            mock_arg.argument_type = "int"
            
            mock_instance.get_all_analysis_results.return_value = {
                'path_references': [],
                'env_var_accesses': [],
                'argument_definitions': [mock_arg],
                'file_operations': []
            }
            
            result = self.tester.validate_script("test_script")
            
            self.assertFalse(result['passed'])
            
            # Check for missing argument issue
            missing_arg_issues = [issue for issue in result['issues'] 
                                 if issue['category'] == 'arguments' and 'not defined in script' in issue['message']]
            self.assertGreater(len(missing_arg_issues), 0)
    
    def test_extra_argument_in_script(self):
        """Test validation when script has extra argument not in contract."""
        # Create test script with extra argument
        script_content = '''
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model-name', required=True, type=str)
parser.add_argument('--extra-arg', required=False, type=str)
args = parser.parse_args()
'''
        script_path = self.scripts_dir / "test_script.py"
        script_path.write_text(script_content)
        
        # Contract only has model-name
        contract = {
            "inputs": {},
            "outputs": {},
            "environment_variables": {"required": [], "optional": []},
            "arguments": {
                "model-name": {
                    "required": True,
                    "type": "str",
                    "description": "Name of the model"
                }
            }
        }
        
        contract_path = self.contracts_dir / "test_script_contract.json"
        contract_path.write_text(json.dumps(contract, indent=2))
        
        # Mock the script analyzer
        with patch('src.cursus.validation.alignment.script_contract_alignment.ScriptAnalyzer') as mock_analyzer:
            mock_instance = MagicMock()
            mock_analyzer.return_value = mock_instance
            
            # Both arguments defined in script
            mock_arg1 = MagicMock()
            mock_arg1.argument_name = "model-name"
            mock_arg1.is_required = True
            mock_arg1.argument_type = "str"
            
            mock_arg2 = MagicMock()
            mock_arg2.argument_name = "extra-arg"
            mock_arg2.is_required = False
            mock_arg2.argument_type = "str"
            
            mock_instance.get_all_analysis_results.return_value = {
                'path_references': [],
                'env_var_accesses': [],
                'argument_definitions': [mock_arg1, mock_arg2],
                'file_operations': []
            }
            
            result = self.tester.validate_script("test_script")
            
            # Should pass but have warnings
            self.assertTrue(result['passed'])
            
            # Check for extra argument warning
            extra_arg_issues = [issue for issue in result['issues'] 
                               if issue['category'] == 'arguments' and 'not in contract' in issue['message']]
            self.assertGreater(len(extra_arg_issues), 0)
    
    def test_argument_type_mismatch(self):
        """Test validation when argument types don't match."""
        # Create test script
        script_content = '''
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', required=True, type=str)  # Contract expects int
args = parser.parse_args()
'''
        script_path = self.scripts_dir / "test_script.py"
        script_path.write_text(script_content)
        
        # Contract expects int type
        contract = {
            "inputs": {},
            "outputs": {},
            "environment_variables": {"required": [], "optional": []},
            "arguments": {
                "epochs": {
                    "required": True,
                    "type": "int",
                    "description": "Number of epochs"
                }
            }
        }
        
        contract_path = self.contracts_dir / "test_script_contract.json"
        contract_path.write_text(json.dumps(contract, indent=2))
        
        # Mock the script analyzer
        with patch('src.cursus.validation.alignment.script_contract_alignment.ScriptAnalyzer') as mock_analyzer:
            mock_instance = MagicMock()
            mock_analyzer.return_value = mock_instance
            
            # Argument with wrong type
            mock_arg = MagicMock()
            mock_arg.argument_name = "epochs"
            mock_arg.is_required = True
            mock_arg.argument_type = "str"  # Script has str, contract expects int
            
            mock_instance.get_all_analysis_results.return_value = {
                'path_references': [],
                'env_var_accesses': [],
                'argument_definitions': [mock_arg],
                'file_operations': []
            }
            
            result = self.tester.validate_script("test_script")
            
            # Should pass but have warnings about type mismatch
            self.assertTrue(result['passed'])
            
            # Check for type mismatch warning
            type_issues = [issue for issue in result['issues'] 
                          if issue['category'] == 'arguments' and 'type mismatch' in issue['message']]
            self.assertGreater(len(type_issues), 0)


if __name__ == '__main__':
    unittest.main()
