"""
Comprehensive test runner for all validation tests.

This script discovers and runs all test files in the validation directory
and its subdirectories, providing a complete summary.
"""

# Import conftest to ensure path setup
import sys
import os
from pathlib import Path

# Import conftest to trigger path setup
sys.path.insert(0, str(Path(__file__).parent.parent))
import conftest

import unittest
import importlib

def discover_test_modules(base_path):
    """Discover all test modules in the validation directory tree."""
    test_modules = []
    base_path = Path(base_path)
    
    # Walk through all subdirectories
    for test_file in base_path.rglob("test_*.py"):
        # Convert file path to module path
        relative_path = test_file.relative_to(base_path.parent.parent)  # Relative to project root
        module_path = str(relative_path.with_suffix('')).replace(os.sep, '.')
        test_modules.append((module_path, str(test_file)))
    
    return test_modules

def run_all_validation_tests():
    """Run all validation test modules."""
    
    # Discover all test modules
    validation_path = Path(__file__).parent
    test_modules = discover_test_modules(validation_path)
    
    # Create a test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    print("🔍 Discovering and loading validation test modules...")
    print(f"Found {len(test_modules)} test files")
    
    loaded_count = 0
    failed_count = 0
    
    # Load tests from each module
    for module_name, file_path in sorted(test_modules):
        try:
            module = importlib.import_module(module_name)
            module_suite = loader.loadTestsFromModule(module)
            if module_suite.countTestCases() > 0:
                suite.addTest(module_suite)
                print(f"  ✅ Loaded {module_name} ({module_suite.countTestCases()} tests)")
                loaded_count += 1
            else:
                print(f"  ⚠️  {module_name} (no tests found)")
        except ImportError as e:
            print(f"  ❌ Failed to load {module_name}: {e}")
            failed_count += 1
        except Exception as e:
            print(f"  ❌ Error loading {module_name}: {e}")
            failed_count += 1
    
    print(f"\n📈 Discovery Summary:")
    print(f"  Total test files found: {len(test_modules)}")
    print(f"  Successfully loaded: {loaded_count}")
    print(f"  Failed to load: {failed_count}")
    print(f"  Total test cases: {suite.countTestCases()}")
    
    if suite.countTestCases() == 0:
        print("\n⚠️  No tests to run!")
        return True
    
    # Run the tests
    print(f"\n🚀 Running {suite.countTestCases()} validation tests...")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📊 Test Execution Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print(f"\n❌ Failures ({len(result.failures)}):")
        for i, (test, traceback) in enumerate(result.failures, 1):
            print(f"  {i}. {test}")
            # Print first few lines of traceback for brevity
            lines = traceback.strip().split('\n')
            for line in lines[-3:]:  # Show last 3 lines
                print(f"     {line}")
            print()
    
    if result.errors:
        print(f"\n💥 Errors ({len(result.errors)}):")
        for i, (test, traceback) in enumerate(result.errors, 1):
            print(f"  {i}. {test}")
            # Print first few lines of traceback for brevity
            lines = traceback.strip().split('\n')
            for line in lines[-3:]:  # Show last 3 lines
                print(f"     {line}")
            print()
    
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
