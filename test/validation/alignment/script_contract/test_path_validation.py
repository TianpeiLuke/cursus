"""
Test suite for script contract path validation.
"""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.cursus.validation.alignment.script_contract_alignment import (
    ScriptContractAlignmentTester
)


class TestPathValidation(unittest.TestCase):
    """Test path validation in script contract alignment."""
    
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
    
    def test_path_validation_success(self):
        """Test successful path validation."""
        # Create test script
        script_content = '''
import os
import pandas as pd

# Read from contract-declared input
data = pd.read_csv("/opt/ml/processing/input/train/data.csv")

# Write to contract-declared output
data.to_csv("/opt/ml/processing/output/processed.csv", index=False)
'''
        script_path = self.scripts_dir / "test_script.py"
        script_path.write_text(script_content)
        
        # Create test contract
        contract = {
            "inputs": {
                "train": {
                    "path": "/opt/ml/processing/input/train/data.csv",
                    "type": "file"
                }
            },
            "outputs": {
                "processed": {
                    "path": "/opt/ml/processing/output/processed.csv",
                    "type": "file"
                }
            },
            "environment_variables": {
                "required": [],
                "optional": []
            },
            "arguments": {}
        }
        
        contract_path = self.contracts_dir / "test_script_contract.json"
        contract_path.write_text(json.dumps(contract, indent=2))
        
        # Mock the script analyzer to return expected results
        with patch('src.cursus.validation.alignment.script_contract_alignment.ScriptAnalyzer') as mock_analyzer:
            mock_instance = MagicMock()
            mock_analyzer.return_value = mock_instance
            
            # Mock path references that match the contract exactly
            mock_path_ref1 = MagicMock()
            mock_path_ref1.path = "/opt/ml/processing/input/train/data.csv"
            mock_path_ref2 = MagicMock()
            mock_path_ref2.path = "/opt/ml/processing/output/processed.csv"
            
            # Mock file operations that match the paths
            mock_file_op1 = MagicMock()
            mock_file_op1.file_path = "/opt/ml/processing/input/train/data.csv"
            mock_file_op1.operation_type = "read"
            
            mock_file_op2 = MagicMock()
            mock_file_op2.file_path = "/opt/ml/processing/output/processed.csv"
            mock_file_op2.operation_type = "write"
            
            mock_instance.get_all_analysis_results.return_value = {
                'path_references': [mock_path_ref1, mock_path_ref2],
                'env_var_accesses': [],
                'argument_definitions': [],
                'file_operations': [mock_file_op1, mock_file_op2]
            }
            
            result = self.tester.validate_script("test_script")
            
            self.assertTrue(result['passed'], f"Validation failed with issues: {result.get('issues', [])}")
            # Should have no critical or error issues
            critical_or_error_issues = [
                issue for issue in result.get('issues', [])
                if issue.get('severity') in ['CRITICAL', 'ERROR']
            ]
            self.assertEqual(len(critical_or_error_issues), 0)
    
    def test_path_validation_undeclared_path(self):
        """Test validation with undeclared path usage."""
        # Create test script with undeclared path
        script_content = '''
import pandas as pd

# This path is not in the contract
data = pd.read_csv("/opt/ml/processing/input/undeclared/data.csv")
'''
        script_path = self.scripts_dir / "test_script.py"
        script_path.write_text(script_content)
        
        # Create minimal contract
        contract = {
            "inputs": {},
            "outputs": {},
            "environment_variables": {"required": [], "optional": []},
            "arguments": {}
        }
        
        contract_path = self.contracts_dir / "test_script_contract.json"
        contract_path.write_text(json.dumps(contract, indent=2))
        
        # Mock the script analyzer
        with patch('src.cursus.validation.alignment.script_contract_alignment.ScriptAnalyzer') as mock_analyzer:
            mock_instance = MagicMock()
            mock_analyzer.return_value = mock_instance
            
            mock_path_ref = MagicMock()
            mock_path_ref.path = "/opt/ml/processing/input/undeclared/data.csv"
            
            mock_instance.get_all_analysis_results.return_value = {
                'path_references': [mock_path_ref],
                'env_var_accesses': [],
                'argument_definitions': [],
                'file_operations': []
            }
            
            result = self.tester.validate_script("test_script")
            
            self.assertFalse(result['passed'])
            self.assertGreater(len(result['issues']), 0)
            
            # Check for undeclared path issue
            path_issues = [issue for issue in result['issues'] 
                          if issue['category'] == 'path_usage' and 'undeclared' in issue['message']]
            self.assertGreater(len(path_issues), 0)
    
    def test_missing_script_file(self):
        """Test validation when script file is missing."""
        # Create contract but no script
        contract = {
            "inputs": {},
            "outputs": {},
            "environment_variables": {"required": [], "optional": []},
            "arguments": {}
        }
        
        contract_path = self.contracts_dir / "missing_script_contract.json"
        contract_path.write_text(json.dumps(contract, indent=2))
        
        result = self.tester.validate_script("missing_script")
        
        self.assertFalse(result['passed'])
        self.assertGreater(len(result['issues']), 0)
        
        # Check for missing file issue
        missing_file_issues = [issue for issue in result['issues'] 
                              if issue['category'] == 'missing_file']
        self.assertGreater(len(missing_file_issues), 0)
    
    def test_missing_contract_file(self):
        """Test validation when contract file is missing."""
        # Create script but no contract
        script_content = '''
print("Hello, world!")
'''
        script_path = self.scripts_dir / "no_contract.py"
        script_path.write_text(script_content)
        
        result = self.tester.validate_script("no_contract")
        
        self.assertFalse(result['passed'])
        self.assertGreater(len(result['issues']), 0)
        
        # Check for missing contract issue
        missing_contract_issues = [issue for issue in result['issues'] 
                                  if issue['category'] == 'missing_contract']
        self.assertGreater(len(missing_contract_issues), 0)


if __name__ == '__main__':
    unittest.main()
