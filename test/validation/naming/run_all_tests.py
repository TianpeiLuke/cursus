"""
Test runner for all validation tests.

This script runs all the individual test files and provides a summary.
"""

# Import conftest to ensure path setup
import sys
import os
from pathlib import Path

import pytest


def run_all_validation_tests():
    """Run all validation test modules."""

    # List of test files to run
    current_dir = Path(__file__).parent
    test_files = [
        "test_naming_violation.py",
        "test_validator_basic.py",
        "test_canonical_step_name_validation.py",
        "test_config_class_name_validation.py",
        "test_builder_class_name_validation.py",
        "test_logical_name_validation.py",
        "test_file_naming_validation.py",
        "test_class_validation.py",
        "test_registry_validation.py",
    ]

    # Build list of test file paths
    test_paths = []
    for test_file in test_files:
        test_path = current_dir / test_file
        if test_path.exists():
            test_paths.append(str(test_path))
        else:
            print(f"  ⚠️  Test file not found: {test_file}")

    print(f"🔍 Running {len(test_paths)} validation test files...")

    # Run pytest with verbose output and summary
    args = [
        "-v",  # verbose
        "--tb=short",  # short traceback format
        "--durations=10",  # show 10 slowest tests
        "--color=yes",  # colored output
    ] + test_paths

    # Run the tests
    print("\n🚀 Running validation tests...")
    exit_code = pytest.main(args)

    # Print final status
    if exit_code == 0:
        print(f"\n✅ All tests passed!")
    else:
        print(f"\n❌ Some tests failed!")

    return exit_code == 0


if __name__ == "__main__":
    success = run_all_validation_tests()
    sys.exit(0 if success else 1)
