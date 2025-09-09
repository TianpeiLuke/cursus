"""
Test runner for all validation tests.

This script runs all the individual test files and provides a summary.
"""

# Import conftest to ensure path setup
import sys
import os
from pathlib import Path

# Import conftest to trigger path setup
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import conftest

import unittest
import importlib

def run_all_validation_tests():
    """Run all validation test modules."""
    
    # List of test modules to run (using relative imports from current directory)
    current_dir = Path(__file__).parent
    test_files = [
        'test_naming_violation',
        'test_validator_basic',
        'test_canonical_step_name_validation',
        'test_config_class_name_validation',
        'test_builder_class_name_validation',
        'test_logical_name_validation',
        'test_file_naming_validation',
        'test_class_validation',
        'test_registry_validation'
    ]
    
    # Add current directory to path for relative imports
    current_dir_str = str(current_dir.resolve())
    if current_dir_str not in sys.path:
        sys.path.insert(0, current_dir_str)
    
    test_modules = test_files
    
    # Create a test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    print("🔍 Loading validation test modules...")
    
    # Load tests from each module
    for module_name in test_modules:
        try:
            module = importlib.import_module(module_name)
            module_suite = loader.loadTestsFromModule(module)
            suite.addTest(module_suite)
            print(f"  ✅ Loaded {module_name}")
        except ImportError as e:
            print(f"  ❌ Failed to load {module_name}: {e}")
    
    # Run the tests
    print("\n🚀 Running validation tests...")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📊 Test Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print(f"\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    # Return success status
    success = len(result.failures) == 0 and len(result.errors) == 0
    if success:
        print(f"\n✅ All tests passed!")
    else:
        print(f"\n❌ Some tests failed!")
    
    return success

if __name__ == '__main__':
    success = run_all_validation_tests()
    sys.exit(0 if success else 1)
