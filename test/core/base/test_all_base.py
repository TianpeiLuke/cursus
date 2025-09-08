"""
Comprehensive test runner for all base classes in cursus.core.base

This module runs all tests for the base classes and provides a summary.
Note: This file is renamed to avoid pytest auto-discovery to prevent duplicate test execution.
"""

import unittest
import sys
from io import StringIO

# Import test modules dynamically to avoid pytest auto-discovery conflicts
def import_test_modules():
    """Import test modules dynamically to avoid conflicts."""
    from test.core.base.test_config_base import TestBasePipelineConfig
    from test.core.base.test_builder_base import TestStepBuilderBase
    from test.core.base.test_specification_base import (
        TestOutputSpec, TestDependencySpec, TestValidationResult, 
        TestAlignmentResult, TestStepSpecification
    )
    from test.core.base.test_contract_base import (
        TestValidationResult as TestContractValidationResult,
        TestScriptContract, TestScriptAnalyzer
    )
    from test.core.base.test_hyperparameters_base import TestModelHyperparameters
    from test.core.base.test_enums import (
        TestDependencyType, TestNodeType, TestEnumInteraction, TestEnumEdgeCases
    )
    
    return {
        'TestBasePipelineConfig': TestBasePipelineConfig,
        'TestStepBuilderBase': TestStepBuilderBase,
        'TestOutputSpec': TestOutputSpec,
        'TestDependencySpec': TestDependencySpec,
        'TestValidationResult': TestValidationResult,
        'TestAlignmentResult': TestAlignmentResult,
        'TestStepSpecification': TestStepSpecification,
        'TestContractValidationResult': TestContractValidationResult,
        'TestScriptContract': TestScriptContract,
        'TestScriptAnalyzer': TestScriptAnalyzer,
        'TestModelHyperparameters': TestModelHyperparameters,
        'TestDependencyType': TestDependencyType,
        'TestNodeType': TestNodeType,
        'TestEnumInteraction': TestEnumInteraction,
        'TestEnumEdgeCases': TestEnumEdgeCases,
    }

def create_test_suite():
    """Create a comprehensive test suite for all base classes."""
    suite = unittest.TestSuite()
    
    # Import test classes
    test_classes = import_test_modules()
    
    # Add config_base tests
    suite.addTest(unittest.makeSuite(test_classes['TestBasePipelineConfig']))
    
    # Add builder_base tests
    suite.addTest(unittest.makeSuite(test_classes['TestStepBuilderBase']))
    
    # Add specification_base tests
    suite.addTest(unittest.makeSuite(test_classes['TestOutputSpec']))
    suite.addTest(unittest.makeSuite(test_classes['TestDependencySpec']))
    suite.addTest(unittest.makeSuite(test_classes['TestValidationResult']))
    suite.addTest(unittest.makeSuite(test_classes['TestAlignmentResult']))
    suite.addTest(unittest.makeSuite(test_classes['TestStepSpecification']))
    
    # Add contract_base tests
    suite.addTest(unittest.makeSuite(test_classes['TestContractValidationResult']))
    suite.addTest(unittest.makeSuite(test_classes['TestScriptContract']))
    suite.addTest(unittest.makeSuite(test_classes['TestScriptAnalyzer']))
    
    # Add hyperparameters_base tests
    suite.addTest(unittest.makeSuite(test_classes['TestModelHyperparameters']))
    
    # Add enums tests
    suite.addTest(unittest.makeSuite(test_classes['TestDependencyType']))
    suite.addTest(unittest.makeSuite(test_classes['TestNodeType']))
    suite.addTest(unittest.makeSuite(test_classes['TestEnumInteraction']))
    suite.addTest(unittest.makeSuite(test_classes['TestEnumEdgeCases']))
    
    return suite

def run_tests_with_summary():
    """Run all tests and provide a detailed summary."""
    print("=" * 80)
    print("RUNNING COMPREHENSIVE TESTS FOR CURSUS.CORE.BASE CLASSES")
    print("=" * 80)
    
    # Create test suite
    suite = create_test_suite()
    
    # Create test runner with detailed output
    stream = StringIO()
    runner = unittest.TextTestRunner(
        stream=stream,
        verbosity=2,
        descriptions=True,
        failfast=False
    )
    
    # Run tests
    result = runner.run(suite)
    
    # Print results to console
    output = stream.getvalue()
    print(output)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    passed = total_tests - failures - errors - skipped
    
    print(f"Total Tests Run: {total_tests}")
    print(f"Passed: {passed}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Skipped: {skipped}")
    
    if failures > 0:
        print(f"\nFAILURES ({failures}):")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if errors > 0:
        print(f"\nERRORS ({errors}):")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    # Overall result
    if failures == 0 and errors == 0:
        print(f"\n🎉 ALL TESTS PASSED! ({passed}/{total_tests})")
        success = True
    else:
        print(f"\n❌ SOME TESTS FAILED! ({passed}/{total_tests} passed)")
        success = False
    
    print("=" * 80)
    
    return success, result

def run_individual_test_modules():
    """Run each test module individually and report results."""
    print("=" * 80)
    print("RUNNING INDIVIDUAL TEST MODULES")
    print("=" * 80)
    
    test_modules = [
        ('config_base', 'test.core.base.test_config_base'),
        ('builder_base', 'test.core.base.test_builder_base'),
        ('specification_base', 'test.core.base.test_specification_base'),
        ('contract_base', 'test.core.base.test_contract_base'),
        ('hyperparameters_base', 'test.core.base.test_hyperparameters_base'),
        ('enums', 'test.core.base.test_enums'),
    ]
    
    results = {}
    
    for module_name, module_path in test_modules:
        print(f"\n--- Testing {module_name} ---")
        
        # Load and run the module
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName(module_path)
        
        runner = unittest.TextTestRunner(verbosity=1)
        result = runner.run(suite)
        
        # Store results
        results[module_name] = {
            'total': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'passed': result.testsRun - len(result.failures) - len(result.errors)
        }
    
    # Print summary table
    print("\n" + "=" * 80)
    print("INDIVIDUAL MODULE RESULTS")
    print("=" * 80)
    print(f"{'Module':<20} {'Total':<8} {'Passed':<8} {'Failed':<8} {'Errors':<8}")
    print("-" * 80)
    
    total_all = 0
    passed_all = 0
    failed_all = 0
    errors_all = 0
    
    for module_name, stats in results.items():
        total_all += stats['total']
        passed_all += stats['passed']
        failed_all += stats['failures']
        errors_all += stats['errors']
        
        print(f"{module_name:<20} {stats['total']:<8} {stats['passed']:<8} {stats['failures']:<8} {stats['errors']:<8}")
    
    print("-" * 80)
    print(f"{'TOTAL':<20} {total_all:<8} {passed_all:<8} {failed_all:<8} {errors_all:<8}")
    print("=" * 80)
    
    return results

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run tests for cursus.core.base classes')
    parser.add_argument('--individual', action='store_true', 
                       help='Run individual test modules separately')
    parser.add_argument('--summary', action='store_true', default=True,
                       help='Run comprehensive test suite with summary (default)')
    
    args = parser.parse_args()
    
    if args.individual:
        results = run_individual_test_modules()
    else:
        success, result = run_tests_with_summary()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
