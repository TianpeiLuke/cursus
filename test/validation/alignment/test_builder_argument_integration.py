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

# Import the necessary modules
from cursus.validation.alignment.script_contract_alignment import ScriptContractAlignmentTester


class TestBuilderArgumentIntegration(unittest.TestCase):
    """Test builder argument integration in script contract alignment."""
    
    def test_builder_argument_detection_integration(self):
        """Test that builder arguments are properly integrated into validation."""
        # Define workspace directory structure
        # workspace_dir points to src/cursus (the main workspace)
        current_file = Path(__file__).resolve()
        workspace_dir = current_file.parent.parent.parent.parent / "src" / "cursus"
        
        # Define component directories within the workspace
        real_scripts_dir = str(workspace_dir / "steps" / "scripts")
        real_contracts_dir = str(workspace_dir / "steps" / "contracts")
        real_builders_dir = str(workspace_dir / "steps" / "builders")
        
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
