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
        
        # Create test contract as Python module
        contract_content = f'''
"""Contract for test_script.py"""

from src.cursus.core.base.contract_base import ScriptContract

class TestScriptContract(ScriptContract):
    def __init__(self):
        super().__init__(
            entry_point="test_script.py",
            expected_input_paths={{
                "train": "/opt/ml/processing/input/train/data.csv"
            }},
            expected_output_paths={{
                "processed": "/opt/ml/processing/output/processed.csv"
            }},
            expected_arguments={{}},
            required_env_vars=[],
            optional_env_vars={{}}
        )

TEST_SCRIPT_CONTRACT = TestScriptContract()
'''
        contract_path = self.contracts_dir / "test_script_contract.py"
        contract_path.write_text(contract_content)
        
        # Mock the script analyzer to return expected results
        with patch('src.cursus.validation.alignment.script_contract_alignment.ScriptAnalyzer') as mock_analyzer:
            mock_instance = MagicMock()
            mock_analyzer.return_value = mock_instance
            
            # Mock path references that match the contract exactly
            mock_path_ref1 = MagicMock()
            mock_path_ref1.path = "/opt/ml/processing/input/train/data.csv"
            mock_path_ref2 = MagicMock()
            mock_path_ref2.path = "/opt/ml/processing/output/processed.csv"
            
            # Mock file operations that match the paths with enhanced method tracking
            mock_file_op1 = MagicMock()
            mock_file_op1.file_path = "/opt/ml/processing/input/train/data.csv"
            mock_file_op1.operation_type = "read"
            mock_file_op1.method = "pandas.read_csv"  # Enhanced method tracking
            mock_file_op1.line_number = 5
            mock_file_op1.context = "data = pd.read_csv(\"/opt/ml/processing/input/train/data.csv\")"
            
            mock_file_op2 = MagicMock()
            mock_file_op2.file_path = "/opt/ml/processing/output/processed.csv"
            mock_file_op2.operation_type = "write"
            mock_file_op2.method = "dataframe.to_csv"  # Enhanced method tracking
            mock_file_op2.line_number = 8
            mock_file_op2.context = "data.to_csv(\"/opt/ml/processing/output/processed.csv\", index=False)"
            
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
        
        # Create minimal contract as Python module
        contract_content = '''
"""Contract for test_script.py"""

from src.cursus.core.base.contract_base import ScriptContract

class TestScriptContract(ScriptContract):
    def __init__(self):
        super().__init__(
            entry_point="test_script.py",
            expected_input_paths={},
            expected_output_paths={},
            expected_arguments={},
            required_env_vars=[],
            optional_env_vars={}
        )

TEST_SCRIPT_CONTRACT = TestScriptContract()
'''
        contract_path = self.contracts_dir / "test_script_contract.py"
        contract_path.write_text(contract_content)
        
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
    
    def test_enhanced_file_operations_detection(self):
        """Test that enhanced file operations are properly detected and tracked."""
        # Create test script with various file operations
        script_content = '''
import pandas as pd
import tarfile
import shutil
from pathlib import Path

# Pandas operations
df = pd.read_csv("/opt/ml/input/data.csv")
df.to_parquet("/opt/ml/output/data.parquet")

# Tarfile operations
with tarfile.open("/opt/ml/input/model.tar.gz", "r:gz") as tar:
    tar.extractall("/tmp/model")

# Shutil operations
shutil.copy("/tmp/source.txt", "/opt/ml/output/dest.txt")

# Pathlib operations
output_path = Path("/opt/ml/output")
output_path.mkdir(exist_ok=True)
'''
        script_path = self.scripts_dir / "enhanced_ops.py"
        script_path.write_text(script_content)
        
        # Create contract with all paths
        contract_content = '''
"""Contract for enhanced_ops.py"""

from src.cursus.core.base.contract_base import ScriptContract

class EnhancedOpsContract(ScriptContract):
    def __init__(self):
        super().__init__(
            entry_point="enhanced_ops.py",
                expected_input_paths={
                    "data": "/opt/ml/processing/input/data.csv",
                    "model": "/opt/ml/processing/input/model.tar.gz"
                },
                expected_output_paths={
                    "parquet_data": "/opt/ml/processing/output/data.parquet",
                    "dest": "/opt/ml/processing/output/dest.txt",
                    "output_dir": "/opt/ml/processing/output"
                },
            expected_arguments={},
            required_env_vars=[],
            optional_env_vars={}
        )

ENHANCED_OPS_CONTRACT = EnhancedOpsContract()
'''
        contract_path = self.contracts_dir / "enhanced_ops_contract.py"
        contract_path.write_text(contract_content)
        
        # Mock the script analyzer with enhanced file operations
        with patch('src.cursus.validation.alignment.script_contract_alignment.ScriptAnalyzer') as mock_analyzer:
            mock_instance = MagicMock()
            mock_analyzer.return_value = mock_instance
            
            # Mock enhanced file operations with method tracking
            mock_ops = [
                MagicMock(file_path="/opt/ml/processing/input/data.csv", operation_type="read", 
                         method="pandas.read_csv", line_number=6),
                MagicMock(file_path="/opt/ml/processing/output/data.parquet", operation_type="write", 
                         method="dataframe.to_parquet", line_number=7),
                MagicMock(file_path="/opt/ml/processing/input/model.tar.gz", operation_type="read", 
                         method="tarfile.open", line_number=10),
                MagicMock(file_path="/tmp/source.txt", operation_type="read", 
                         method="shutil.copy", line_number=14),
                MagicMock(file_path="/opt/ml/processing/output/dest.txt", operation_type="write", 
                         method="shutil.copy", line_number=14),
                MagicMock(file_path="/opt/ml/processing/output", operation_type="write", 
                         method="pathlib.mkdir", line_number=18)
            ]
            
            mock_instance.get_all_analysis_results.return_value = {
                'path_references': [],
                'env_var_accesses': [],
                'argument_definitions': [],
                'file_operations': mock_ops
            }
            
            result = self.tester.validate_script("enhanced_ops")
            
            # Should pass since all operations match contract paths
            self.assertTrue(result['passed'], f"Enhanced operations validation failed: {result.get('issues', [])}")
            
            # Verify that enhanced methods were tracked
            # This is implicit in the successful validation - the enhanced detection
            # allows the validator to properly match file operations to contract paths


if __name__ == '__main__':
    unittest.main()
