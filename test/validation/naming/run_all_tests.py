"""
Test runner for all validation tests.

This script runs all the individual test files and provides a summary.
"""

# Add the project root to the Python path to allow for absolute imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import unittest
import importlib


def run_all_validation_tests():
    """Run all validation test modules."""
    
    # List of test modules to run
    test_modules = [
        'test.validation.naming.test_naming_violation',
        'test.validation.naming.test_validator_basic',
        'test.validation.naming.test_canonical_step_name_validation',
        'test.validation.naming.test_config_class_name_validation',
        'test.validation.naming.test_builder_class_name_validation',
        'test.validation.naming.test_logical_name_validation',
        'test.validation.naming.test_file_naming_validation',
        'test.validation.naming.test_class_validation',
        'test.validation.naming.test_registry_validation'
    ]
    
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
