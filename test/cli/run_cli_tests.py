#!/usr/bin/env python3
"""
Test runner for CLI validation tests.

This script runs all CLI-related tests with enhanced output formatting.
"""

import sys
import os
import unittest
from io import StringIO

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def main():
    """Run all CLI tests with enhanced output."""
    print("🔍 Loading CLI test modules...")
    
    # Discover and load tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Count total tests
    total_tests = suite.countTestCases()
    print(f"  ✅ Found {total_tests} tests")
    print()
    
    print("🚀 Running CLI validation tests...")
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        buffer=True
    )
    
    result = runner.run(suite)
    
    print()
    print("📊 Test Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    print()
    
    if result.wasSuccessful():
        print("✅ All CLI tests passed!")
        return 0
    else:
        print("❌ Some CLI tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
