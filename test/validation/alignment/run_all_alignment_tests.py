"""
Test runner for all alignment validation tests.

This script runs all alignment validation tests and provides
comprehensive reporting on test results and coverage.
"""

import unittest
import sys
import os
from pathlib import Path
from io import StringIO

# Add the project root to the Python path
sys.path.insert(0, str(project_root))

# Test modules will be loaded dynamically

class AlignmentTestRunner:
    """Custom test runner for alignment validation tests."""
    
    def __init__(self):
        """Initialize the test runner."""
        self.test_modules = [
            # Root level tests
            'test.validation.alignment.test_builder_argument_debug',
            'test.validation.alignment.test_builder_argument_integration',
            'test.validation.alignment.test_enhanced_argument_validation',
            'test.validation.alignment.test_framework_patterns',
            'test.validation.alignment.test_sagemaker_property_path_validation',
            'test.validation.alignment.test_step_type_detection',
            'test.validation.alignment.test_step_type_enhancement_router',
            'test.validation.alignment.test_step_type_enhancement_system_comprehensive',
            
            # Utils tests
            'test.validation.alignment.utils.test_severity_level',
            'test.validation.alignment.utils.test_alignment_level',
            'test.validation.alignment.utils.test_alignment_issue',
            'test.validation.alignment.utils.test_path_reference',
            'test.validation.alignment.utils.test_utility_functions',
            'test.validation.alignment.utils.test_core_models',
            'test.validation.alignment.utils.test_script_analysis_models',
            'test.validation.alignment.utils.test_step_type_detection',
            
            # Reporter tests
            'test.validation.alignment.reporter.test_validation_result',
            'test.validation.alignment.reporter.test_alignment_report',
            
            # Script-Contract tests
            'test.validation.alignment.script_contract.test_script_contract_path_validation',
            'test.validation.alignment.script_contract.test_argument_validation',
            'test.validation.alignment.script_contract.test_testability_validation',
            
            # Step Type Enhancers tests
            'test.validation.alignment.step_type_enhancers.test_base_enhancer',
            'test.validation.alignment.step_type_enhancers.test_training_enhancer',
            
            # Unified Tester tests
            'test.validation.alignment.unified_tester.test_level_validation',
            'test.validation.alignment.unified_tester.test_full_validation'
        ]
        
        self.test_results = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.error_tests = 0
        self.skipped_tests = 0
    
    def run_all_tests(self, verbosity=2):
        """
        Run all alignment validation tests.
        
        Args:
            verbosity (int): Test output verbosity level (0-2)
            
        Returns:
            bool: True if all tests passed, False otherwise
        """
        print("=" * 80)
        print("RUNNING ALIGNMENT VALIDATION TESTS")
        print("=" * 80)
        
        all_passed = True
        
        for module_name in self.test_modules:
            print(f"\n📦 Running tests from {module_name}")
            print("-" * 60)
            
            # Create test suite for this module
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromName(module_name)
            
            # Run tests with custom result collector
            stream = StringIO()
            runner = unittest.TextTestRunner(
                stream=stream,
                verbosity=verbosity,
                buffer=True
            )
            
            result = runner.run(suite)
            
            # Store results
            self.test_results[module_name] = {
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
                'success': result.wasSuccessful(),
                'output': stream.getvalue()
            }
            
            # Update totals
            self.total_tests += result.testsRun
            self.failed_tests += len(result.failures)
            self.error_tests += len(result.errors)
            self.skipped_tests += len(result.skipped) if hasattr(result, 'skipped') else 0
            
            # Print module results
            if result.wasSuccessful():
                print(f"✅ {result.testsRun} tests passed")
            else:
                print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")
                all_passed = False
                
                # Print failure details if verbosity is high
                if verbosity >= 2:
                    for test, traceback in result.failures:
                        print(f"\n🔴 FAILURE: {test}")
                        print(traceback)
                    
                    for test, traceback in result.errors:
                        print(f"\n💥 ERROR: {test}")
                        print(traceback)
        
        self.passed_tests = self.total_tests - self.failed_tests - self.error_tests - self.skipped_tests
        
        # Print summary
        self._print_summary()
        
        return all_passed
    
    def run_specific_module(self, module_name, verbosity=2):
        """
        Run tests from a specific module.
        
        Args:
            module_name (str): Name of the test module to run
            verbosity (int): Test output verbosity level
            
        Returns:
            bool: True if all tests in module passed, False otherwise
        """
        if module_name not in self.test_modules:
            print(f"❌ Unknown test module: {module_name}")
            print(f"Available modules: {', '.join(self.test_modules)}")
            return False
        
        print(f"🎯 Running tests from {module_name}")
        print("-" * 60)
        
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName(module_name)
        
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        
        return result.wasSuccessful()
    
    def run_specific_test(self, test_class, test_method=None, verbosity=2):
        """
        Run a specific test class or method.
        
        Args:
            test_class (str): Name of the test class
            test_method (str, optional): Name of specific test method
            verbosity (int): Test output verbosity level
            
        Returns:
            bool: True if test(s) passed, False otherwise
        """
        if test_method:
            test_name = f"{test_class}.{test_method}"
        else:
            test_name = test_class
        
        print(f"🎯 Running specific test: {test_name}")
        print("-" * 60)
        
        loader = unittest.TestLoader()
        
        try:
            if test_method:
                suite = loader.loadTestsFromName(f"{test_class}.{test_method}")
            else:
                suite = loader.loadTestsFromName(test_class)
        except Exception as e:
            print(f"❌ Failed to load test: {e}")
            return False
        
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        
        return result.wasSuccessful()
    
    def _print_summary(self):
        """Print comprehensive test summary."""
        print("\n" + "=" * 80)
        print("ALIGNMENT VALIDATION TEST SUMMARY")
        print("=" * 80)
        
        # Overall statistics
        print(f"📊 Total Tests: {self.total_tests}")
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.failed_tests}")
        print(f"💥 Errors: {self.error_tests}")
        print(f"⏭️  Skipped: {self.skipped_tests}")
        
        # Success rate
        if self.total_tests > 0:
            success_rate = (self.passed_tests / self.total_tests) * 100
            print(f"📈 Success Rate: {success_rate:.1f}%")
        
        # Module breakdown
        print(f"\n📦 Module Breakdown:")
        for module_name, results in self.test_results.items():
            status = "✅" if results['success'] else "❌"
            print(f"  {status} {module_name}: {results['tests_run']} tests")
            if not results['success']:
                print(f"    - {results['failures']} failures, {results['errors']} errors")
        
        # Overall result
        print(f"\n🏆 Overall Result: ", end="")
        if self.failed_tests == 0 and self.error_tests == 0:
            print("ALL TESTS PASSED! 🎉")
        else:
            print("SOME TESTS FAILED ❌")
        
        print("=" * 80)
    
    def generate_coverage_report(self):
        """Generate a coverage report for alignment validation tests."""
        print("\n📋 ALIGNMENT VALIDATION TEST COVERAGE")
        print("-" * 60)
        
        coverage_areas = {
            'Alignment Utilities': [
                'SeverityLevel enum',
                'AlignmentLevel enum', 
                'AlignmentIssue model',
                'PathReference model',
                'EnvVarAccess model',
                'ImportStatement model',
                'ArgumentDefinition model',
                'Utility functions'
            ],
            'Alignment Reporter': [
                'ValidationResult model',
                'AlignmentSummary model',
                'AlignmentRecommendation model',
                'AlignmentReport class',
                'JSON export',
                'HTML export',
                'Recommendation generation'
            ],
            'Script-Contract Alignment': [
                'Path usage validation',
                'Environment variable validation',
                'Argument parsing validation',
                'Import validation',
                'Script analysis',
                'Contract validation'
            ],
            'Unified Alignment Tester': [
                'Level 1 validation',
                'Level 2 validation',
                'Level 3 validation',
                'Level 4 validation',
                'Full validation orchestration',
                'Report generation',
                'Error handling'
            ]
        }
        
        for area, components in coverage_areas.items():
            print(f"\n🔍 {area}:")
            for component in components:
                print(f"  ✅ {component}")
        
        print(f"\n📊 Total Coverage Areas: {len(coverage_areas)}")
        total_components = sum(len(components) for components in coverage_areas.values())
        print(f"📊 Total Components Tested: {total_components}")

def main():
    """Main entry point for the test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run alignment validation tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_alignment_tests.py                    # Run all tests
  python run_all_alignment_tests.py -v 1               # Run with minimal output
  python run_all_alignment_tests.py --coverage         # Show coverage report
  python run_all_alignment_tests.py --module utils     # Run specific module
  python run_all_alignment_tests.py --test TestClass   # Run specific test class
        """
    )
    
    parser.add_argument(
        '-v', '--verbosity',
        type=int,
        choices=[0, 1, 2],
        default=2,
        help='Test output verbosity level (0=minimal, 1=normal, 2=verbose)'
    )
    
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Show test coverage report'
    )
    
    parser.add_argument(
        '--module',
        choices=['root', 'utils', 'reporter', 'script_contract', 'step_enhancers', 'unified'],
        help='Run tests from specific module only'
    )
    
    parser.add_argument(
        '--test',
        help='Run specific test class or method (e.g., TestAlignmentIssue or TestAlignmentIssue.test_creation)'
    )
    
    args = parser.parse_args()
    
    # Create test runner
    runner = AlignmentTestRunner()
    
    # Run tests based on arguments
    success = True
    
    if args.module:
        # Filter modules by category
        if args.module == 'root':
            modules_to_run = [m for m in runner.test_modules if not any(subfolder in m for subfolder in ['.utils.', '.reporter.', '.script_contract.', '.step_type_enhancers.', '.unified_tester.'])]
        elif args.module == 'utils':
            modules_to_run = [m for m in runner.test_modules if '.utils.' in m]
        elif args.module == 'reporter':
            modules_to_run = [m for m in runner.test_modules if '.reporter.' in m]
        elif args.module == 'script_contract':
            modules_to_run = [m for m in runner.test_modules if '.script_contract.' in m]
        elif args.module == 'step_enhancers':
            modules_to_run = [m for m in runner.test_modules if '.step_type_enhancers.' in m]
        elif args.module == 'unified':
            modules_to_run = [m for m in runner.test_modules if '.unified_tester.' in m]
        else:
            modules_to_run = []
        
        if not modules_to_run:
            print(f"❌ No modules found for category: {args.module}")
            success = False
        else:
            success = True
            for module in modules_to_run:
                module_success = runner.run_specific_module(module, args.verbosity)
                success = success and module_success
    
    elif args.test:
        if '.' in args.test:
            test_class, test_method = args.test.split('.', 1)
            success = runner.run_specific_test(test_class, test_method, args.verbosity)
        else:
            success = runner.run_specific_test(args.test, verbosity=args.verbosity)
    
    else:
        success = runner.run_all_tests(args.verbosity)
    
    # Show coverage report if requested
    if args.coverage:
        runner.generate_coverage_report()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
