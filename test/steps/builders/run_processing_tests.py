#!/usr/bin/env python3
"""
Test runner for Processing step builders.

This script provides an easy way to run all Processing step builder tests
with proper output formatting and error handling.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

def run_processing_tests():
    """Run all Processing step builder tests."""
    print("=" * 80)
    print("PROCESSING STEP BUILDERS TEST SUITE")
    print("=" * 80)
    print()
    
    try:
        # Import and run the test suite
        from test_processing_step_builders import ProcessingStepBuilderTestSuite, TestProcessingStepBuilders
        import unittest
        
        # Create test suite
        test_suite = ProcessingStepBuilderTestSuite()
        
        # Show available builders
        available_builders = test_suite.get_available_processing_builders()
        print(f"Found {len(available_builders)} Processing step builders to test:")
        for step_name, builder_class in available_builders:
            print(f"  ✓ {step_name} ({builder_class.__name__})")
        print()
        
        if not available_builders:
            print("❌ No Processing step builders found. Check your imports and paths.")
            return False
        
        # Run the tests
        print("Running comprehensive tests...")
        print()
        
        # Create test loader and suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestProcessingStepBuilders)
        
        # Run tests with detailed output
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        result = runner.run(suite)
        
        # Summary
        print()
        print("=" * 80)
        print("TEST EXECUTION SUMMARY")
        print("=" * 80)
        
        if result.wasSuccessful():
            print("✅ All tests passed successfully!")
            return True
        else:
            print(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
            
            if result.failures:
                print("\nFailures:")
                for test, traceback in result.failures:
                    print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
            
            if result.errors:
                print("\nErrors:")
                for test, traceback in result.errors:
                    print(f"  - {test}: {traceback.split('\\n')[-2]}")
            
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the correct directory and all dependencies are installed.")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_individual_builder_test(step_name: str):
    """Run tests for a specific Processing step builder."""
    print(f"Testing {step_name} Processing Step Builder")
    print("=" * 60)
    
    try:
        from test_processing_step_builders import ProcessingStepBuilderTestSuite
        from cursus.validation.builders.universal_test import UniversalStepBuilderTest
        
        # Create test suite
        test_suite = ProcessingStepBuilderTestSuite()
        
        # Load the specific builder
        builder_class = test_suite.load_builder_class(step_name)
        if not builder_class:
            print(f"❌ Could not load builder for {step_name}")
            return False
        
        print(f"Testing {builder_class.__name__}...")
        print()
        
        # Run universal tests
        print("Running Universal Step Builder Tests...")
        tester = UniversalStepBuilderTest(
            builder_class, 
            verbose=True,
            enable_scoring=True,
            enable_structured_reporting=False  # Keep simple for individual testing
        )
        results = tester.run_all_tests()
        # Extract test results from enhanced format
        universal_results = results.get('test_results', results) if isinstance(results, dict) and 'test_results' in results else results
        
        # Run Processing-specific tests
        print("\nRunning Processing-Specific Tests...")
        processing_results = test_suite.run_processing_specific_tests(step_name, builder_class)
        
        # Combine and report results
        all_results = {**universal_results, **processing_results}
        
        passed_tests = sum(1 for result in all_results.values() 
                          if isinstance(result, dict) and result.get("passed", False))
        total_tests = len([r for r in all_results.values() if isinstance(r, dict)])
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n{step_name} Test Results:")
        print(f"  Tests passed: {passed_tests}/{total_tests} ({pass_rate:.1f}%)")
        
        # Show failed tests
        failed_tests = {k: v for k, v in all_results.items() 
                       if isinstance(v, dict) and not v.get("passed", True)}
        
        if failed_tests:
            print("  Failed tests:")
            for test_name, result in failed_tests.items():
                print(f"    ❌ {test_name}: {result.get('error', 'Unknown error')}")
            return False
        else:
            print("  ✅ All tests passed!")
            return True
            
    except Exception as e:
        print(f"❌ Error testing {step_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Test specific builder
        step_name = sys.argv[1]
        success = run_individual_builder_test(step_name)
    else:
        # Test all builders
        success = run_processing_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
