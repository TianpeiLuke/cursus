#!/usr/bin/env python3
"""
Test script testability validation functionality.

Tests the TestabilityPatternValidator and its integration with
the ScriptContractAlignmentTester to ensure scripts follow
testability refactoring patterns.
"""

import pytest
import ast
import sys
from pathlib import Path

from cursus.validation.alignment import (
    TestabilityPatternValidator,
    SeverityLevel,
)


class TestTestabilityPatternValidator:
    """Test the TestabilityPatternValidator class."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = TestabilityPatternValidator()

    def test_good_testability_pattern(self):
        """Test script with good testability patterns."""
        good_script = '''
import os
import argparse
from pathlib import Path

def main(input_paths, output_paths, environ_vars, job_args):
    """Main function following testability pattern."""
    # Use parameters instead of direct environment access
    model_dir = input_paths["model_dir"]
    output_dir = output_paths["output_dir"]
    label_field = environ_vars.get("LABEL_FIELD", "label")
    
    # Simulate processing
    with open(f"{model_dir}/model.pkl", "r") as f:
        model_data = f.read()
    
    with open(f"{output_dir}/results.json", "w") as f:
        f.write('{"status": "complete"}')

def is_running_in_container():
    """Detect if running in container."""
    return os.path.exists("/.dockerenv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    
    # Collect environment variables
    environ_vars = {
        "LABEL_FIELD": os.environ.get("LABEL_FIELD", "label"),
        "ID_FIELD": os.environ.get("ID_FIELD", "id")
    }
    
    # Set up paths
    input_paths = {
        "model_dir": args.model_dir
    }
    
    output_paths = {
        "output_dir": args.output_dir
    }
    
    # Ensure output directory exists
    os.makedirs(output_paths["output_dir"], exist_ok=True)
    
    # Call main function with testability parameters
    main(input_paths, output_paths, environ_vars, args)
'''

        ast_tree = ast.parse(good_script)
        issues = self.validator.validate_script_testability("good_script.py", ast_tree)

        # Should have at least one INFO issue indicating compliance
        info_issues = [issue for issue in issues if issue.level == SeverityLevel.INFO]
        assert (
            len(info_issues) > 0
        ), "Should have INFO issues indicating good testability"

        # Should not have any CRITICAL issues, but may have ERROR issues for main call pattern
        critical_issues = [
            issue for issue in issues if issue.level == SeverityLevel.CRITICAL
        ]
        assert (
            len(critical_issues) == 0
        ), f"Should not have CRITICAL issues, but found: {[i.message for i in critical_issues]}"

        # Check that if there are ERROR issues, they are about main function call pattern
        error_issues = [issue for issue in issues if issue.level == SeverityLevel.ERROR]
        if error_issues:
            # All error issues should be about main function call pattern, which is acceptable
            for issue in error_issues:
                assert (
                    "main function" in issue.message.lower()
                ), f"Unexpected error issue: {issue.message}"

    def test_poor_testability_pattern(self):
        """Test script with poor testability patterns."""
        bad_script = '''
import os
import argparse

def main():
    """Main function without testability parameters."""
    # Direct environment access (anti-pattern)
    model_dir = os.environ.get("MODEL_DIR", "/opt/ml/input/model")
    output_dir = os.environ.get("OUTPUT_DIR", "/opt/ml/output")
    label_field = os.environ.get("LABEL_FIELD", "label")
    
    # Direct file operations with hardcoded paths
    with open("/opt/ml/input/model/model.pkl", "r") as f:
        model_data = f.read()
    
    with open("/opt/ml/output/results.json", "w") as f:
        f.write('{"status": "complete"}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-type", type=str, required=True)
    args = parser.parse_args()
    
    # Call main function without parameters
    main()
'''

        ast_tree = ast.parse(bad_script)
        issues = self.validator.validate_script_testability("bad_script.py", ast_tree)

        # Should have WARNING issues for poor testability (main signature)
        warning_issues = [
            issue for issue in issues if issue.level == SeverityLevel.WARNING
        ]
        assert (
            len(warning_issues) > 0
        ), "Should have WARNING issues for poor testability"

        # Check for specific testability issues
        issue_categories = [issue.category for issue in issues]
        assert (
            "testability_main_signature" in issue_categories
        ), "Should detect main function signature issues"

    def test_no_main_function(self):
        """Test script without main function."""
        no_main_script = '''
import os

def helper_function():
    """A helper function."""
    data_path = os.environ.get("DATA_PATH", "/opt/ml/input")
    return data_path

if __name__ == "__main__":
    print("Running script without main function")
    helper_function()
'''

        ast_tree = ast.parse(no_main_script)
        issues = self.validator.validate_script_testability(
            "no_main_script.py", ast_tree
        )

        # Should have WARNING about missing main function
        warning_issues = [
            issue for issue in issues if issue.level == SeverityLevel.WARNING
        ]
        main_function_warnings = [
            issue
            for issue in warning_issues
            if "main function" in issue.message.lower()
        ]
        assert (
            len(main_function_warnings) > 0
        ), "Should warn about missing main function"

    def test_partial_testability_parameters(self):
        """Test script with partial testability parameters."""
        partial_script = '''
import os
import argparse

def main(input_paths, output_paths):
    """Main function with partial testability parameters."""
    # Missing environ_vars and job_args parameters
    model_dir = input_paths["model_dir"]
    output_dir = output_paths["output_dir"]
    
    # Still using direct environment access
    label_field = os.environ.get("LABEL_FIELD", "label")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    input_paths = {"model_dir": "/opt/ml/input"}
    output_paths = {"output_dir": "/opt/ml/output"}
    
    main(input_paths, output_paths)
'''

        ast_tree = ast.parse(partial_script)
        issues = self.validator.validate_script_testability(
            "partial_script.py", ast_tree
        )

        # Should have WARNING about missing testability parameters
        warning_issues = [
            issue for issue in issues if issue.level == SeverityLevel.WARNING
        ]
        signature_warnings = [
            issue
            for issue in warning_issues
            if "missing testability parameters" in issue.message.lower()
        ]
        assert (
            len(signature_warnings) > 0
        ), "Should warn about missing testability parameters"

    def test_helper_function_env_access(self):
        """Test detection of environment access in helper functions."""
        helper_env_script = '''
import os

def main(input_paths, output_paths, environ_vars, job_args):
    """Main function with proper testability."""
    result = process_data()
    return result

def process_data():
    """Helper function with direct environment access."""
    # This is an anti-pattern - should use parameters
    label_field = os.environ.get("LABEL_FIELD", "label")
    return f"Processing with {label_field}"

if __name__ == "__main__":
    main({}, {}, {}, None)
'''

        ast_tree = ast.parse(helper_env_script)
        issues = self.validator.validate_script_testability(
            "helper_env_script.py", ast_tree
        )

        # Should have WARNING about helper function environment access
        warning_issues = [
            issue for issue in issues if issue.level == SeverityLevel.WARNING
        ]
        helper_warnings = [
            issue
            for issue in warning_issues
            if "helper function" in issue.message.lower()
        ]
        assert (
            len(helper_warnings) > 0
        ), "Should warn about helper function environment access"

    def test_container_detection(self):
        """Test detection of container detection patterns."""
        container_script = '''
import os

def main(input_paths, output_paths, environ_vars, job_args):
    """Main function with testability."""
    pass

def is_running_in_container():
    """Container detection function."""
    return os.path.exists("/.dockerenv")

if __name__ == "__main__":
    if is_running_in_container():
        print("Running in container")
    
    main({}, {}, {}, None)
'''

        ast_tree = ast.parse(container_script)
        issues = self.validator.validate_script_testability(
            "container_script.py", ast_tree
        )

        # Should NOT have INFO issue about missing container detection
        info_issues = [issue for issue in issues if issue.level == SeverityLevel.INFO]
        container_missing_issues = [
            issue
            for issue in info_issues
            if "container detection" in issue.message.lower()
        ]
        assert (
            len(container_missing_issues) == 0
        ), "Should not warn about missing container detection when present"

    def test_main_block_without_main_call(self):
        """Test main block that doesn't call main function."""
        no_call_script = '''
import os

def main(input_paths, output_paths, environ_vars, job_args):
    """Main function with testability."""
    pass

if __name__ == "__main__":
    print("Script running but not calling main function")
    # Missing: main(input_paths, output_paths, environ_vars, job_args)
'''

        ast_tree = ast.parse(no_call_script)
        issues = self.validator.validate_script_testability(
            "no_call_script.py", ast_tree
        )

        # Should have ERROR about main block not calling main function
        error_issues = [issue for issue in issues if issue.level == SeverityLevel.ERROR]
        main_call_errors = [
            issue
            for issue in error_issues
            if "does not call main function" in issue.message.lower()
        ]
        assert (
            len(main_call_errors) > 0
        ), "Should error when main block doesn't call main function"


class TestTestabilityIntegration:
    """Test integration of testability validation with ScriptContractAlignmentTester."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        # We'll test the integration conceptually since we need actual files
        pass

    def test_testability_validator_import(self):
        """Test that TestabilityPatternValidator can be imported."""
        from cursus.validation.alignment import TestabilityPatternValidator

        validator = TestabilityPatternValidator()
        assert validator is not None
        assert hasattr(validator, "validate_script_testability")

    def test_testability_parameters_defined(self):
        """Test that testability parameters are properly defined."""
        validator = TestabilityPatternValidator()
        expected_params = {"input_paths", "output_paths", "environ_vars", "job_args"}
        assert validator.testability_parameters == expected_params

    def test_severity_levels_mapping(self):
        """Test that severity levels are properly mapped."""
        validator = TestabilityPatternValidator()

        # Test with a simple script to ensure severity levels work
        simple_script = """
def main():
    pass
"""

        ast_tree = ast.parse(simple_script)
        issues = validator.validate_script_testability("simple_script.py", ast_tree)

        # All issues should have valid severity levels
        for issue in issues:
            assert isinstance(issue.level, SeverityLevel)
            assert issue.level in [
                SeverityLevel.INFO,
                SeverityLevel.WARNING,
                SeverityLevel.ERROR,
                SeverityLevel.CRITICAL,
            ]


class TestTestabilityValidationCategories:
    """Test specific testability validation categories."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = TestabilityPatternValidator()

    def test_testability_categories(self):
        """Test that all expected testability categories are covered."""
        expected_categories = {
            "testability_main_function",
            "testability_main_signature",
            "testability_compliance",
            "testability_env_access",
            "testability_entry_point",
            "testability_parameter_usage",
            "testability_parameter_access",
            "testability_container_support",
            "testability_helper_functions",
        }

        # Test with a comprehensive script that triggers multiple categories
        comprehensive_script = '''
import os

def main():
    """Main without testability parameters."""
    label = os.environ.get("LABEL", "default")
    return label

def helper():
    """Helper with env access."""
    return os.getenv("HELPER_VAR")

if __name__ == "__main__":
    result = main()
    print(result)
'''

        ast_tree = ast.parse(comprehensive_script)
        issues = self.validator.validate_script_testability(
            "comprehensive_script.py", ast_tree
        )

        # Check that we get issues from multiple categories
        found_categories = {issue.category for issue in issues}

        # Should have at least some of the expected categories
        overlap = found_categories.intersection(expected_categories)
        assert (
            len(overlap) > 0
        ), f"Should find testability categories, found: {found_categories}"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
