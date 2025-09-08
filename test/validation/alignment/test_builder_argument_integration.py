"""
Integration test for builder argument detection in script contract alignment.

This test verifies that the builder argument detection is properly integrated
into the script contract alignment validation.
"""

import unittest
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from cursus.validation.alignment.script_contract_alignment import ScriptContractAlignmentTester

class TestBuilderArgumentIntegration(unittest.TestCase):
    """Test builder argument integration in script contract alignment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.scripts_dir = Path(self.temp_dir) / "scripts"
        self.contracts_dir = Path(self.temp_dir) / "contracts"
        self.builders_dir = Path(self.temp_dir) / "builders"
        
        self.scripts_dir.mkdir()
        self.contracts_dir.mkdir()
        self.builders_dir.mkdir()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_builder_argument_integration_tabular_preprocess(self):
        """Test that builder arguments are properly integrated for tabular_preprocess."""
        print("\n🔍 Testing builder argument integration for tabular_preprocess...")
        
        # Create mock script file
        script_content = '''
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--job_type', required=True, type=str)
args = parser.parse_args()

job_type = args.job_type
'''
        script_path = self.scripts_dir / "tabular_preprocess.py"
        script_path.write_text(script_content)
        
        # Create mock contract file (without job_type in expected_arguments)
        contract_content = '''
"""Contract for tabular_preprocess.py"""

from cursus.core.base.contract_base import ScriptContract

class TabularPreprocessContract(ScriptContract):
    def __init__(self):
        super().__init__(
            entry_point="tabular_preprocess.py",
            expected_input_paths={},
            expected_output_paths={},
            expected_arguments={
                # Note: job_type is NOT in contract - it's provided by builder
            },
            required_env_vars=[],
            optional_env_vars={}
        )

TABULAR_PREPROCESS_CONTRACT = TabularPreprocessContract()
'''
        contract_path = self.contracts_dir / "tabular_preprocess_contract.py"
        contract_path.write_text(contract_content)
        
        # Create mock builder file
        builder_content = '''
"""Tabular Preprocessing Step Builder"""

from typing import List

class TabularPreprocessingStepBuilder:
    def _get_job_arguments(self) -> List[str]:
        job_type = self.config.job_type
        return ["--job_type", job_type]
'''
        builder_path = self.builders_dir / "builder_tabular_preprocessing_step.py"
        builder_path.write_text(builder_content)
        
        # Create tester with builders directory
        tester = ScriptContractAlignmentTester(
            str(self.scripts_dir),
            str(self.contracts_dir),
            str(self.builders_dir)
        )
        
        # Mock the script analyzer
        with patch('src.cursus.validation.alignment.script_contract_alignment.ScriptAnalyzer') as mock_analyzer:
            mock_instance = MagicMock()
            mock_analyzer.return_value = mock_instance
            
            # Mock argument definition for job_type
            mock_arg = MagicMock()
            mock_arg.argument_name = "job_type"
            mock_arg.is_required = True
            mock_arg.argument_type = "str"
            
            mock_instance.get_all_analysis_results.return_value = {
                'path_references': [],
                'env_var_accesses': [],
                'argument_definitions': [mock_arg],
                'file_operations': []
            }
            
            # Run validation
            result = tester.validate_script("tabular_preprocess")
            
            print(f"✅ Validation result: {result['passed']}")
            print(f"📋 Issues found: {len(result['issues'])}")
            
            # Print all issues for debugging
            for i, issue in enumerate(result['issues']):
                print(f"  Issue {i+1}: [{issue['severity']}] {issue['category']}: {issue['message']}")
            
            # Check that job_type is handled correctly
            job_type_issues = [issue for issue in result['issues'] 
                             if 'job_type' in issue.get('message', '').lower()]
            
            print(f"🎯 job_type related issues: {len(job_type_issues)}")
            for issue in job_type_issues:
                print(f"  - [{issue['severity']}] {issue['message']}")
            
            # Assertions
            self.assertTrue(result['passed'], f"Validation should pass, but got issues: {result['issues']}")
            
            # Should have INFO message about builder-provided argument
            info_issues = [issue for issue in job_type_issues if issue['severity'] == 'INFO']
            self.assertGreater(len(info_issues), 0, 
                             "Expected INFO message about builder-provided job_type argument")
            
            # Should NOT have ERROR or WARNING about job_type
            error_warning_issues = [issue for issue in job_type_issues 
                                  if issue['severity'] in ['ERROR', 'WARNING']]
            self.assertEqual(len(error_warning_issues), 0,
                           f"Should not have ERROR/WARNING for builder-provided job_type, but got: {error_warning_issues}")
    
    def test_real_builder_argument_extraction(self):
        """Test builder argument extraction with real project files."""
        print("\n🔍 Testing real builder argument extraction...")
        
        # Use real project directories with absolute paths
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent.parent
        real_scripts_dir = str(project_root / "src/cursus/steps/scripts")
        real_contracts_dir = str(project_root / "src/cursus/steps/contracts")
        real_builders_dir = str(project_root / "src/cursus/steps/builders")
        
        tester = ScriptContractAlignmentTester(
            real_scripts_dir,
            real_contracts_dir,
            real_builders_dir
        )
        
        # Test builder argument extraction directly
        from cursus.validation.alignment.static_analysis.builder_analyzer import extract_builder_arguments
        
        builder_args = extract_builder_arguments("tabular_preprocess", real_builders_dir)
        print(f"🎯 Builder arguments for tabular_preprocess: {builder_args}")
        
        # Test the _validate_argument_usage method directly
        mock_analysis = {
            'path_references': [],
            'env_var_accesses': [],
            'file_operations': [],
            'argument_definitions': [
                type('ArgDef', (), {
                    'argument_name': 'job_type',
                    'is_required': True,
                    'argument_type': 'str'
                })()
            ]
        }
        
        mock_contract = {
            'arguments': {}  # Empty - job_type not in contract
        }
        
        # Test the argument validation through the validator
        issues = tester.script_validator.validate_argument_usage(mock_analysis, mock_contract, "tabular_preprocess", builder_args)
        
        print(f"📋 Argument validation issues: {len(issues)}")
        for issue in issues:
            print(f"  - [{issue['severity']}] {issue['category']}: {issue['message']}")
        
        # Check for job_type handling
        job_type_issues = [issue for issue in issues if 'job_type' in issue.get('message', '').lower()]
        print(f"🎯 job_type issues: {len(job_type_issues)}")
        
        if job_type_issues:
            for issue in job_type_issues:
                print(f"  - [{issue['severity']}] {issue['message']}")
                if 'details' in issue:
                    print(f"    Details: {issue['details']}")
        
        # Assertions
        self.assertIn("job_type", builder_args, "Builder should provide job_type argument")
        
        # Should have INFO message about builder-provided argument
        info_issues = [issue for issue in job_type_issues if issue['severity'] == 'INFO']
        self.assertGreater(len(info_issues), 0, 
                         "Expected INFO message about builder-provided job_type")

if __name__ == '__main__':
    unittest.main(verbosity=2)
