#!/usr/bin/env python3
"""
Test runner for CLI tests.

This script runs all CLI-related tests with enhanced output formatting,
including validation CLI, runtime CLI, alignment CLI, and builder test CLI.
"""

import sys
import os
import unittest
from io import StringIO

# Add the project root to the Python path
# Note: sys.path setup is handled by conftest.py
# No manual path manipulation needed


def run_specific_test_module(module_name, description):
    """Run tests for a specific module with detailed reporting."""
    print(f"🔍 Running {description}...")

    # Load specific test module
    loader = unittest.TestLoader()
    try:
        suite = loader.loadTestsFromName(module_name)
        test_count = suite.countTestCases()

        if test_count == 0:
            print(f"  ⚠️  No tests found in {module_name}")
            return True, 0, 0, 0, 0

        print(f"  📋 Found {test_count} tests")

        # Run tests with minimal verbosity for cleaner output
        runner = unittest.TextTestRunner(verbosity=1, stream=sys.stdout, buffer=True)

        result = runner.run(suite)

        # Report results
        success = result.wasSuccessful()
        status_icon = "✅" if success else "❌"
        print(
            f"  {status_icon} {description}: {result.testsRun} tests, "
            f"{len(result.failures)} failures, {len(result.errors)} errors"
        )

        return (
            success,
            result.testsRun,
            len(result.failures),
            len(result.errors),
            len(result.skipped) if hasattr(result, "skipped") else 0,
        )

    except Exception as e:
        print(f"  ❌ Error loading {module_name}: {str(e)}")
        return False, 0, 0, 1, 0


def main():
    """Run all CLI tests with enhanced output and detailed reporting."""
    print("🚀 CLI Test Suite Runner")
    print("=" * 50)

    # Define test modules to run
    test_modules = [
        ("test_runtime_testing_cli", "Runtime Testing CLI Tests"),
        ("test_alignment_cli", "Alignment CLI Tests"),
        ("test_builder_test_cli", "Builder Test CLI Tests"),
        ("test_workspace_cli", "Workspace CLI Tests"),
        ("test_registry_cli", "Registry CLI Tests"),
        ("test_catalog_cli", "Step Catalog CLI Tests"),
        ("test_pipeline_cli", "Pipeline Catalog CLI Tests"),
    ]

    total_success = True
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0

    # Run each test module
    for module_name, description in test_modules:
        success, tests, failures, errors, skipped = run_specific_test_module(
            module_name, description
        )
        total_success = total_success and success
        total_tests += tests
        total_failures += failures
        total_errors += errors
        total_skipped += skipped
        print()

    # Also run any additional tests using discovery
    print("🔍 Discovering additional CLI tests...")
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)

    # Discover all tests, but exclude the ones we already ran specifically
    suite = loader.discover(start_dir, pattern="test_*.py")
    discovered_tests = suite.countTestCases()

    # Only run discovered tests if they're different from what we already ran
    if discovered_tests > total_tests:
        additional_tests = discovered_tests - total_tests
        print(f"  📋 Found {additional_tests} additional tests")

        runner = unittest.TextTestRunner(verbosity=1, stream=sys.stdout, buffer=True)

        result = runner.run(suite)

        # Update totals with any additional tests
        if result.testsRun > total_tests:
            additional_run = result.testsRun - total_tests
            additional_failures = len(result.failures) - total_failures
            additional_errors = len(result.errors) - total_errors
            additional_skipped = (
                len(result.skipped) if hasattr(result, "skipped") else 0
            ) - total_skipped

            total_tests = result.testsRun
            total_failures = len(result.failures)
            total_errors = len(result.errors)
            total_skipped = len(result.skipped) if hasattr(result, "skipped") else 0
            total_success = total_success and result.wasSuccessful()

            status_icon = (
                "✅" if additional_failures == 0 and additional_errors == 0 else "❌"
            )
            print(
                f"  {status_icon} Additional tests: {additional_run} tests, "
                f"{additional_failures} failures, {additional_errors} errors"
            )
    else:
        print("  ✅ No additional tests found")

    print()
    print("📊 Final Test Summary:")
    print("=" * 50)
    print(f"  Total Tests Run: {total_tests}")
    print(f"  Failures: {total_failures}")
    print(f"  Errors: {total_errors}")
    print(f"  Skipped: {total_skipped}")
    print(
        f"  Success Rate: {((total_tests - total_failures - total_errors) / max(total_tests, 1) * 100):.1f}%"
    )
    print()

    if total_success and total_failures == 0 and total_errors == 0:
        print("🎉 All CLI tests passed successfully!")
        return 0
    else:
        print("❌ Some CLI tests failed!")
        if total_failures > 0:
            print(f"   - {total_failures} test(s) failed")
        if total_errors > 0:
            print(f"   - {total_errors} test(s) had errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())
