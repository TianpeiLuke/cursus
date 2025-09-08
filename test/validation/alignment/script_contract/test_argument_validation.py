"""
Test suite for script contract argument validation.
"""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from cursus.validation.alignment.script_contract_alignment import (
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
        """Test successful argument validation with argparse hyphen-to-underscore conversion."""
        # Create test script with arguments using CLI convention (hyphens)
        script_content = '''
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model-name', required=True, type=str)  # CLI uses hyphens
parser.add_argument('--epochs', required=False, type=int, default=10)
args = parser.parse_args()

# Script accesses with underscores (argparse automatic conversion)
model = args.model_name  # model-name → model_name
num_epochs = args.epochs
'''
        script_path = self.scripts_dir / "test_script.py"
        script_path.write_text(script_content)
        
        # Create matching contract as Python module
        contract_content = '''
"""Contract for test_script.py"""

from cursus.core.base.contract_base import ScriptContract

class TestScriptContract(ScriptContract):
    def __init__(self):
        super().__init__(
            entry_point="test_script.py",
            expected_input_paths={},
            expected_output_paths={},
            expected_arguments={
                "model-name": "",  # Required argument (empty string indicates required)
                "epochs": "10"  # CLI argument with default
            },
            required_env_vars=[],
            optional_env_vars={}
        )

TEST_SCRIPT_CONTRACT = TestScriptContract()
'''
        contract_path = self.contracts_dir / "test_script_contract.py"
        contract_path.write_text(contract_content)
        
        # Mock the script analyzer
        with patch('cursus.validation.alignment.script_contract_alignment.ScriptAnalyzer') as mock_analyzer:
            mock_instance = MagicMock()
            mock_analyzer.return_value = mock_instance
            
            # Mock argument definitions - script analyzer detects Python attribute names (underscores)
            mock_arg1 = MagicMock()
            mock_arg1.argument_name = "model_name"  # Python attribute name (underscores)
            mock_arg1.is_required = True
            mock_arg1.argument_type = "str"
            
            mock_arg2 = MagicMock()
            mock_arg2.argument_name = "epochs"  # No conversion needed
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
            # Should have no argument-related issues due to proper argparse normalization
            arg_issues = [issue for issue in result['issues'] if issue['category'] == 'arguments']
            self.assertEqual(len(arg_issues), 0, f"Unexpected argument issues: {arg_issues}")
    
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
        
        # Contract requires model-name argument as Python module
        contract_content = '''
"""Contract for test_script.py"""

from cursus.core.base.contract_base import ScriptContract

class TestScriptContract(ScriptContract):
    def __init__(self):
        super().__init__(
            entry_point="test_script.py",
            expected_input_paths={},
            expected_output_paths={},
            expected_arguments={
                "model-name": "",  # Required (empty string indicates required)
                "epochs": "10"  # Optional with default
            },
            required_env_vars=[],
            optional_env_vars={}
        )

TEST_SCRIPT_CONTRACT = TestScriptContract()
'''
        contract_path = self.contracts_dir / "test_script_contract.py"
        contract_path.write_text(contract_content)
        
        # Mock the script analyzer
        with patch('cursus.validation.alignment.script_contract_alignment.ScriptAnalyzer') as mock_analyzer:
            mock_instance = MagicMock()
            mock_analyzer.return_value = mock_instance
            
            # Only epochs argument defined in script (Python attribute name)
            mock_arg = MagicMock()
            mock_arg.argument_name = "epochs"  # Python attribute name
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
        
        # Contract only has model-name as Python module
        contract_content = '''
"""Contract for test_script.py"""

from cursus.core.base.contract_base import ScriptContract

class TestScriptContract(ScriptContract):
    def __init__(self):
        super().__init__(
            entry_point="test_script.py",
            expected_input_paths={},
            expected_output_paths={},
            expected_arguments={
                "model-name": ""  # Required (empty string indicates required)
            },
            required_env_vars=[],
            optional_env_vars={}
        )

TEST_SCRIPT_CONTRACT = TestScriptContract()
'''
        contract_path = self.contracts_dir / "test_script_contract.py"
        contract_path.write_text(contract_content)
        
        # Mock the script analyzer
        with patch('cursus.validation.alignment.script_contract_alignment.ScriptAnalyzer') as mock_analyzer:
            mock_instance = MagicMock()
            mock_analyzer.return_value = mock_instance
            
            # Both arguments defined in script (Python attribute names)
            mock_arg1 = MagicMock()
            mock_arg1.argument_name = "model_name"  # Python attribute name (underscores)
            mock_arg1.is_required = True
            mock_arg1.argument_type = "str"
            
            mock_arg2 = MagicMock()
            mock_arg2.argument_name = "extra_arg"  # Python attribute name (underscores)
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
        
        # Contract expects int type as Python module
        contract_content = '''
"""Contract for test_script.py"""

from cursus.core.base.contract_base import ScriptContract

class TestScriptContract(ScriptContract):
    def __init__(self):
        super().__init__(
            entry_point="test_script.py",
            expected_input_paths={},
            expected_output_paths={},
            expected_arguments={
                "epochs": ""  # Required (empty string indicates required)
            },
            required_env_vars=[],
            optional_env_vars={}
        )

TEST_SCRIPT_CONTRACT = TestScriptContract()
'''
        # Create a mock contract that expects int type for direct testing
        mock_contract = {
            'arguments': {
                'epochs': {
                    'required': True,
                    'type': 'int'  # Contract expects int
                }
            }
        }
        
        # Mock analysis with script argument having string type
        mock_analysis = {
            'path_references': [],
            'env_var_accesses': [],
            'file_operations': [],
            'argument_definitions': [
                type('ArgDef', (), {
                    'argument_name': 'epochs',
                    'is_required': True,
                    'argument_type': 'str'  # Script has str, contract expects int
                })()
            ]
        }
        
        # Test the argument validation directly
        issues = self.tester.script_validator.validate_argument_usage(mock_analysis, mock_contract, "test_script", set())
        
        # Check for type mismatch warning
        type_issues = [issue for issue in issues 
                      if issue['category'] == 'arguments' and 'type mismatch' in issue['message']]
        self.assertGreater(len(type_issues), 0)
    
    def test_argparse_hyphen_to_underscore_normalization(self):
        """
        Test that the argparse hyphen-to-underscore fix works correctly.
        
        This is the specific test for the fix that eliminates false positives
        when contract uses CLI convention (hyphens) and script uses Python
        convention (underscores) - which is standard SageMaker + argparse behavior.
        """
        # Mock analysis with script arguments using Python attribute names (underscores)
        mock_analysis = {
            'path_references': [],
            'env_var_accesses': [],
            'file_operations': [],
            'argument_definitions': [
                type('ArgDef', (), {
                    'argument_name': 'job_type',  # Python attribute name (underscores)
                    'is_required': True,
                    'argument_type': 'str'
                })(),
                type('ArgDef', (), {
                    'argument_name': 'marketplace_id_col',  # Python attribute name (underscores)
                    'is_required': True,
                    'argument_type': 'str'
                })(),
                type('ArgDef', (), {
                    'argument_name': 'default_currency',  # Python attribute name (underscores)
                    'is_required': False,
                    'argument_type': 'str'
                })(),
                type('ArgDef', (), {
                    'argument_name': 'n_workers',  # Python attribute name (underscores)
                    'is_required': False,
                    'argument_type': 'int'
                })()
            ]
        }
        
        # Mock contract with CLI arguments using hyphens (CLI convention)
        mock_contract = {
            'arguments': {
                'job-type': {  # CLI argument name (hyphens)
                    'required': True,
                    'type': 'str'
                },
                'marketplace-id-col': {  # CLI argument name (hyphens)
                    'required': True,
                    'type': 'str'
                },
                'default-currency': {  # CLI argument name (hyphens)
                    'required': False,
                    'type': 'str'
                },
                'n-workers': {  # CLI argument name (hyphens)
                    'required': False,
                    'type': 'int'
                }
            }
        }
        
        # Test the argument validation with the fix
        issues = self.tester.script_validator.validate_argument_usage(mock_analysis, mock_contract, "currency_conversion", set())
        
        # Should have NO issues - the fix should handle argparse normalization correctly
        self.assertEqual(len(issues), 0, 
                        f"Expected no issues with argparse normalization, but found: {[issue['message'] for issue in issues]}")
        
        print("✅ Argparse hyphen-to-underscore normalization test passed!")
        print("   Contract 'job-type' correctly matches script 'args.job_type'")
        print("   Contract 'marketplace-id-col' correctly matches script 'args.marketplace_id_col'")
        print("   Contract 'default-currency' correctly matches script 'args.default_currency'")
        print("   Contract 'n-workers' correctly matches script 'args.n_workers'")
    
    def test_argparse_normalization_with_missing_argument(self):
        """Test that missing arguments are still detected correctly with argparse normalization."""
        # Script missing one argument
        mock_analysis = {
            'path_references': [],
            'env_var_accesses': [],
            'file_operations': [],
            'argument_definitions': [
                type('ArgDef', (), {
                    'argument_name': 'job_type',  # Only has job_type (Python attribute name)
                    'is_required': True,
                    'argument_type': 'str'
                })()
            ]
        }
        
        # Contract expects two arguments
        mock_contract = {
            'arguments': {
                'job-type': {  # CLI argument name (hyphens)
                    'required': True,
                    'type': 'str'
                },
                'marketplace-id-col': {  # This should be flagged as missing
                    'required': True,
                    'type': 'str'
                }
            }
        }
        
        issues = self.tester.script_validator.validate_argument_usage(mock_analysis, mock_contract, "test_script", set())
        
        # Should find exactly 1 missing argument issue
        missing_arg_issues = [issue for issue in issues if 'not defined in script' in issue['message']]
        self.assertEqual(len(missing_arg_issues), 1, 
                        f"Expected 1 missing argument issue, found {len(missing_arg_issues)}")
        
        # Check that the error message includes both CLI and Python names
        issue = missing_arg_issues[0]
        self.assertIn('marketplace-id-col', issue['message'])  # CLI name
        self.assertIn('marketplace_id_col', issue['message'])  # Python attribute name
        
        print("✅ Missing argument detection with argparse normalization works correctly!")
    
    def test_argparse_normalization_with_extra_argument(self):
        """Test that extra arguments are detected correctly with argparse normalization."""
        # Script has extra argument
        mock_analysis = {
            'path_references': [],
            'env_var_accesses': [],
            'file_operations': [],
            'argument_definitions': [
                type('ArgDef', (), {
                    'argument_name': 'job_type',  # Python attribute name
                    'is_required': True,
                    'argument_type': 'str'
                })(),
                type('ArgDef', (), {
                    'argument_name': 'extra_param',  # This should be flagged as extra
                    'is_required': False,
                    'argument_type': 'str'
                })()
            ]
        }
        
        # Contract only expects one argument
        mock_contract = {
            'arguments': {
                'job-type': {  # CLI argument name (hyphens)
                    'required': True,
                    'type': 'str'
                }
            }
        }
        
        issues = self.tester.script_validator.validate_argument_usage(mock_analysis, mock_contract, "test_script", set())
        
        # Should find exactly 1 extra argument issue
        extra_arg_issues = [issue for issue in issues if 'not in contract' in issue['message']]
        self.assertEqual(len(extra_arg_issues), 1, 
                        f"Expected 1 extra argument issue, found {len(extra_arg_issues)}")
        
        # Check that the error message includes both CLI and Python names
        issue = extra_arg_issues[0]
        self.assertIn('--extra-param', issue['message'])  # CLI name
        self.assertIn('extra_param', issue['message'])  # Python attribute name
        
        print("✅ Extra argument detection with argparse normalization works correctly!")

if __name__ == '__main__':
    unittest.main()
