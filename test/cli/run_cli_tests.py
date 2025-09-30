#!/usr/bin/env python3
"""
Test runner for CLI tests using pytest.

This script runs all CLI-related tests with enhanced output formatting,
including validation CLI, runtime CLI, alignment CLI, and builder test CLI.
"""

import sys
import os
import subprocess
from pathlib import Path


def run_pytest_for_module(test_file, description):
    """Run pytest for a specific test module with detailed reporting."""
    print(f"🔍 Running {description}...")
    
    test_path = Path(__file__).parent / test_file
    if not test_path.exists():
        print(f"  ⚠️  Test file not found: {test_file}")
        return True, 0, 0, 0, 0
    
    try:
        # Run pytest with verbose output and capture results
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            str(test_path), 
            "-v", 
            "--tb=short",
            "--no-header",
            "--quiet"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
        
        # Parse pytest output to extract test counts
        output_lines = result.stdout.split('\n')
        
        # Look for the summary line
        passed = 0
        failed = 0
        errors = 0
        skipped = 0
        
        for line in output_lines:
            if "passed" in line or "failed" in line or "error" in line or "skipped" in line:
                # Parse pytest summary line
                if " passed" in line:
                    try:
                        passed = int(line.split()[0])
                    except (ValueError, IndexError):
                        pass
                if " failed" in line:
                    try:
                        failed = int(line.split()[0])
                    except (ValueError, IndexError):
                        pass
                if " error" in line:
                    try:
                        errors = int(line.split()[0])
                    except (ValueError, IndexError):
                        pass
                if " skipped" in line:
                    try:
                        skipped = int(line.split()[0])
                    except (ValueError, IndexError):
                        pass
        
        total_tests = passed + failed + errors + skipped
        
        if total_tests == 0:
            print(f"  ⚠️  No tests found in {test_file}")
            return True, 0, 0, 0, 0
        
        print(f"  📋 Found {total_tests} tests")
        
        success = result.returncode == 0 and failed == 0 and errors == 0
        status_icon = "✅" if success else "❌"
        
        print(f"  {status_icon} {description}: {total_tests} tests, {failed} failures, {errors} errors, {skipped} skipped")
        
        # Show failed test details if any
        if failed > 0 or errors > 0:
            print("  📝 Test output:")
            for line in output_lines[-10:]:  # Show last 10 lines for context
                if line.strip():
                    print(f"    {line}")
        
        return success, total_tests, failed, errors, skipped
        
    except Exception as e:
        print(f"  ❌ Error running pytest for {test_file}: {str(e)}")
        return False, 0, 0, 1, 0


def main():
    """Run all CLI tests using pytest with enhanced output and detailed reporting."""
    print("🚀 CLI Test Suite Runner (pytest)")
    print("=" * 50)

    # Define test files to run
    test_files = [
        ("test_runtime_testing_cli.py", "Runtime Testing CLI Tests"),
        ("test_alignment_cli.py", "Alignment CLI Tests"),
        ("test_builder_test_cli.py", "Builder Test CLI Tests"),
        ("test_workspace_cli.py", "Workspace CLI Tests"),
        ("test_registry_cli.py", "Registry CLI Tests"),
        ("test_catalog_cli.py", "Step Catalog CLI Tests"),
        ("test_pipeline_cli.py", "Pipeline Catalog CLI Tests"),
    ]

    total_success = True
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0

    # Run each test file
    for test_file, description in test_files:
        success, tests, failures, errors, skipped = run_pytest_for_module(
            test_file, description
        )
        total_success = total_success and success
        total_tests += tests
        total_failures += failures
        total_errors += errors
        total_skipped += skipped
        print()

    print("📊 Final Test Summary:")
    print("=" * 50)
    print(f"  Total Tests Run: {total_tests}")
    print(f"  Passed: {total_tests - total_failures - total_errors}")
    print(f"  Failures: {total_failures}")
    print(f"  Errors: {total_errors}")
    print(f"  Skipped: {total_skipped}")
    
    if total_tests > 0:
        success_rate = ((total_tests - total_failures - total_errors) / total_tests * 100)
        print(f"  Success Rate: {success_rate:.1f}%")
    else:
        print("  Success Rate: N/A")
    
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
