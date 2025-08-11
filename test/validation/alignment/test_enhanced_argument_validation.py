"""
Test Enhanced Argument Validation with Builder Integration

Tests the enhanced Level 1 validation that checks builder arguments
before declaring script argument failures.
"""

import unittest
import tempfile
import os
import sys
from pathlib import Path

# Add src to path for imports
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from src.cursus.validation.alignment.script_contract_alignment import ScriptContractAlignmentTester
from src.cursus.validation.alignment.static_analysis.builder_analyzer import BuilderArgumentExtractor, extract_builder_arguments


class TestEnhancedArgumentValidation(unittest.TestCase):
    """Test enhanced argument validation with builder integration."""
    
    def setUp(self):
        """Set up test environment with temporary directories."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.scripts_dir = self.temp_dir / "scripts"
        self.contracts_dir = self.temp_dir / "contracts"
        self.builders_dir = self.temp_dir / "builders"
        
        for dir_path in [self.scripts_dir, self.contracts_dir, self.builders_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up temporary directories."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
    
    def test_builder_argument_extraction(self):
        """Test that builder argument extraction works correctly."""
        # Create a sample builder with _get_job_arguments method
        builder_content = '''"""
Sample builder for testing argument extraction.
"""

class SampleBuilder:
    def __init__(self, config):
        self.config = config
    
    def _get_job_arguments(self):
        """Get job arguments for the processing script."""
        args = [
            "--job-type", self.config.job_type,
            "--mode", self.config.mode,
            "--marketplace-id-col", self.config.marketplace_id_col
        ]
        
        # Add optional arguments
        if hasattr(self.config, "currency_col") and self.config.currency_col:
            args.extend(["--currency-col", self.config.currency_col])
        
        return args
'''
        
        builder_file = self.builders_dir / "builder_sample_step.py"
        builder_file.write_text(builder_content)
        
        # Test argument extraction
        extractor = BuilderArgumentExtractor(str(builder_file))
        arguments = extractor.extract_job_arguments()
        
        expected_args = {"job-type", "mode", "marketplace-id-col", "currency-col"}
        self.assertEqual(arguments, expected_args)
    
    def test_enhanced_validation_with_builder_args(self):
        """Test that validation correctly handles builder-provided arguments."""
        # Create a script that defines arguments provided by builder
        script_content = '''#!/usr/bin/env python3
"""
Sample script with config-driven arguments.
"""

import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Sample processing script")
    parser.add_argument("--job-type", required=True, help="Job type")
    parser.add_argument("--mode", required=True, help="Processing mode")
    parser.add_argument("--marketplace-id-col", required=True, help="Marketplace ID column")
    parser.add_argument("--currency-col", help="Currency column")
    
    args = parser.parse_args()
    
    # Use environment variables
    job_name = os.getenv("SM_TRAINING_JOB_NAME", "unknown")
    
    print(f"Processing with job type: {args.job_type}")

if __name__ == "__main__":
    main()
'''
        
        # Create a contract that doesn't include builder-provided arguments
        contract_content = '''"""
Contract for sample script.
"""

from cursus.steps.contracts.script_contract import ScriptContract

SAMPLE_CONTRACT = ScriptContract(
    entry_point="sample.py",
    description="Sample processing script contract",
    expected_input_paths={},
    expected_output_paths={},
    expected_arguments={},  # Empty - arguments provided by builder
    required_env_vars=["SM_TRAINING_JOB_NAME"],
    optional_env_vars={}
)
'''
        
        # Create a builder that provides the arguments
        builder_content = '''"""
Sample builder that provides config-driven arguments.
"""

class SampleBuilder:
    def __init__(self, config):
        self.config = config
    
    def _get_job_arguments(self):
        """Get job arguments for the processing script."""
        return [
            "--job-type", self.config.job_type,
            "--mode", self.config.mode,
            "--marketplace-id-col", self.config.marketplace_id_col,
            "--currency-col", self.config.currency_col
        ]
'''
        
        # Write test files
        (self.scripts_dir / "sample.py").write_text(script_content)
        (self.contracts_dir / "sample_contract.py").write_text(contract_content)
        (self.builders_dir / "builder_sample_step.py").write_text(builder_content)
        
        # Test validation with builder integration
        tester = ScriptContractAlignmentTester(
            str(self.scripts_dir),
            str(self.contracts_dir),
            str(self.builders_dir)
        )
        
        result = tester.validate_script("sample")
        
        # Check that validation passes or has only INFO issues for builder-provided args
        issues = result.get('issues', [])
        
        # Filter for argument-related issues
        arg_issues = [issue for issue in issues if issue.get('category') == 'arguments']
        
        # Should have INFO issues for builder-provided arguments, not WARNING/ERROR
        builder_arg_issues = [
            issue for issue in arg_issues 
            if issue.get('details', {}).get('source') == 'builder'
        ]
        
        # Verify that builder-provided arguments are marked as INFO, not errors
        for issue in builder_arg_issues:
            self.assertEqual(issue.get('severity'), 'INFO')
            self.assertIn('provided by builder', issue.get('message', ''))
    
    def test_orphaned_arguments_still_flagged(self):
        """Test that arguments not in contract or builder are still flagged as issues."""
        # Create a script with an argument not provided by builder or contract
        script_content = '''#!/usr/bin/env python3
"""
Script with orphaned argument.
"""

import argparse

def main():
    parser = argparse.ArgumentParser(description="Sample script")
    parser.add_argument("--job-type", required=True, help="Job type")
    parser.add_argument("--orphaned-arg", help="This argument is not in contract or builder")
    
    args = parser.parse_args()
    print(f"Job type: {args.job_type}")

if __name__ == "__main__":
    main()
'''
        
        # Create contract and builder that don't provide the orphaned argument
        contract_content = '''"""
Contract for orphaned test.
"""

class MockScriptContract:
    def __init__(self, entry_point, description, expected_input_paths, 
                 expected_output_paths, expected_arguments, required_env_vars, optional_env_vars):
        self.entry_point = entry_point
        self.description = description
        self.expected_input_paths = expected_input_paths
        self.expected_output_paths = expected_output_paths
        self.expected_arguments = expected_arguments
        self.required_env_vars = required_env_vars
        self.optional_env_vars = optional_env_vars

ORPHANED_CONTRACT = MockScriptContract(
    entry_point="orphaned.py",
    description="Test contract",
    expected_input_paths={},
    expected_output_paths={},
    expected_arguments={},
    required_env_vars=[],
    optional_env_vars={}
)
'''
        
        builder_content = '''"""
Builder that only provides job-type.
"""

class OrphanedBuilder:
    def __init__(self, config):
        self.config = config
    
    def _get_job_arguments(self):
        return ["--job-type", self.config.job_type]
'''
        
        # Write test files
        (self.scripts_dir / "orphaned.py").write_text(script_content)
        (self.contracts_dir / "orphaned_contract.py").write_text(contract_content)
        (self.builders_dir / "builder_orphaned_step.py").write_text(builder_content)
        
        # Test validation
        tester = ScriptContractAlignmentTester(
            str(self.scripts_dir),
            str(self.contracts_dir),
            str(self.builders_dir)
        )
        
        result = tester.validate_script("orphaned")
        
        # Check that orphaned argument is still flagged as WARNING
        issues = result.get('issues', [])
        arg_issues = [issue for issue in issues if issue.get('category') == 'arguments']
        
        orphaned_issues = [
            issue for issue in arg_issues 
            if 'orphaned-arg' in issue.get('message', '') and 
            issue.get('severity') == 'WARNING'
        ]
        
        self.assertTrue(len(orphaned_issues) > 0, "Orphaned argument should be flagged as WARNING")
    
    def test_builder_registry_mapping(self):
        """Test that builder registry correctly maps scripts to builders."""
        # Create multiple builders with different naming patterns
        builders = {
            "builder_currency_conversion_step.py": "currency_conversion",
            "builder_tabular_preprocessing_step.py": "tabular_preprocessing", 
            "builder_model_calibration_step.py": "model_calibration"
        }
        
        for builder_file, expected_script in builders.items():
            builder_content = f'''"""
Builder for {expected_script}.
"""

class {expected_script.title().replace('_', '')}Builder:
    def _get_job_arguments(self):
        return ["--job-type", "test"]
'''
            (self.builders_dir / builder_file).write_text(builder_content)
        
        # Test extraction for each script
        for expected_script in builders.values():
            args = extract_builder_arguments(expected_script, str(self.builders_dir))
            self.assertEqual(args, {"job-type"})


if __name__ == '__main__':
    unittest.main()
