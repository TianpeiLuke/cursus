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
