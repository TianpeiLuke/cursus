"""
Test runner for Jupyter integration tests

This script runs all unit tests for the Jupyter integration modules
and provides comprehensive test coverage reporting.
"""

import unittest
import sys
import os
from pathlib import Path
import argparse
from io import StringIO
import time

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import test modules
from test_notebook_interface import *
from test_visualization import *
from test_debugger import *
from test_templates import *
from test_advanced import *


class TestResult:
    """Custom test result class to track test statistics"""
    
    def __init__(self):
        self.tests_run = 0
        self.failures = 0
        self.errors = 0
        self.skipped = 0
        self.success_rate = 0.0
        self.execution_time = 0.0
        self.failure_details = []
        self.error_details = []
        self.skipped_details = []


class JupyterTestRunner:
    """Custom test runner for Jupyter integration tests"""
    
    def __init__(self, verbosity=2):
        self.verbosity = verbosity
        self.results = {}
    
    def run_module_tests(self, module_name, test_classes):
        """Run tests for a specific module"""
        print(f"\n{'='*60}")
        print(f"Running tests for {module_name}")
        print(f"{'='*60}")
        
        suite = unittest.TestSuite()
        
        # Add all test classes to the suite
        for test_class in test_classes:
            tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
            suite.addTests(tests)
        
        # Run the tests
        stream = StringIO()
        runner = unittest.TextTestRunner(
            stream=stream, 
            verbosity=self.verbosity,
            buffer=True
        )
        
        start_time = time.time()
        result = runner.run(suite)
        end_time = time.time()
        
        # Process results
        test_result = TestResult()
        test_result.tests_run = result.testsRun
        test_result.failures = len(result.failures)
        test_result.errors = len(result.errors)
        test_result.skipped = len(result.skipped)
        test_result.execution_time = end_time - start_time
        
        if test_result.tests_run > 0:
            test_result.success_rate = (
                (test_result.tests_run - test_result.failures - test_result.errors) 
                / test_result.tests_run * 100
            )
        
        # Store failure and error details
        test_result.failure_details = [
            f"{test}: {traceback}" for test, traceback in result.failures
        ]
        test_result.error_details = [
            f"{test}: {traceback}" for test, traceback in result.errors
        ]
        test_result.skipped_details = [
            f"{test}: {reason}" for test, reason in result.skipped
        ]
        
        self.results[module_name] = test_result
        
        # Print results
        self._print_module_results(module_name, test_result)
        
        # Print detailed output if verbose
        if self.verbosity > 1:
            print("\nDetailed Output:")
            print(stream.getvalue())
        
        return test_result
    
    def _print_module_results(self, module_name, result):
        """Print results for a module"""
        print(f"\nResults for {module_name}:")
        print(f"  Tests run: {result.tests_run}")
        print(f"  Failures: {result.failures}")
        print(f"  Errors: {result.errors}")
        print(f"  Skipped: {result.skipped}")
        print(f"  Success rate: {result.success_rate:.1f}%")
        print(f"  Execution time: {result.execution_time:.2f}s")
        
        if result.failures > 0:
            print(f"  ❌ {result.failures} test(s) failed")
        if result.errors > 0:
            print(f"  💥 {result.errors} test(s) had errors")
        if result.skipped > 0:
            print(f"  ⏭️  {result.skipped} test(s) skipped")
        if result.failures == 0 and result.errors == 0:
            print(f"  ✅ All tests passed!")
    
    def run_all_tests(self):
        """Run all Jupyter integration tests"""
        print("🧪 Starting Jupyter Integration Test Suite")
        print(f"Python version: {sys.version}")
        print(f"Test directory: {Path(__file__).parent}")
        
        # Define test modules and their test classes
        test_modules = {
            "notebook_interface": [
                TestNotebookSession,
                TestNotebookInterface,
                TestNotebookInterfaceWithJupyter,
                TestNotebookInterfaceEdgeCases
            ],
            "visualization": [
                TestVisualizationConfig,
                TestTestResultMetrics,
                TestVisualizationReporter,
                TestVisualizationReporterWithJupyter,
                TestVisualizationReporterEdgeCases
            ],
            "debugger": [
                TestDebugSession,
                TestBreakpointManager,
                TestInteractiveDebugger,
                TestInteractiveDebuggerWithJupyter,
                TestInteractiveDebuggerEdgeCases
            ],
            "templates": [
                TestNotebookTemplate,
                TestNotebookTemplateManager,
                TestNotebookTemplateManagerWithJupyter,
                TestNotebookTemplateManagerEdgeCases
            ],
            "advanced": [
                TestNotebookSession,
                TestCollaborationManager,
                TestAutomatedReportGenerator,
                TestPerformanceMonitor,
                TestAdvancedWidgetFactory,
                TestAdvancedWidgetFactoryWithJupyter,
                TestAdvancedNotebookFeatures,
                TestAdvancedNotebookFeaturesWithJupyter,
                TestAdvancedNotebookFeaturesEdgeCases
            ]
        }
        
        # Run tests for each module
        total_start_time = time.time()
        
        for module_name, test_classes in test_modules.items():
            try:
                self.run_module_tests(module_name, test_classes)
            except Exception as e:
                print(f"❌ Error running tests for {module_name}: {e}")
                # Create a dummy result for failed module
                error_result = TestResult()
                error_result.errors = 1
                error_result.error_details = [f"Module loading error: {e}"]
                self.results[module_name] = error_result
        
        total_end_time = time.time()
        
        # Print overall summary
        self._print_overall_summary(total_end_time - total_start_time)
        
        return self.results
    
    def _print_overall_summary(self, total_time):
        """Print overall test summary"""
        print(f"\n{'='*60}")
        print("OVERALL TEST SUMMARY")
        print(f"{'='*60}")
        
        total_tests = sum(r.tests_run for r in self.results.values())
        total_failures = sum(r.failures for r in self.results.values())
        total_errors = sum(r.errors for r in self.results.values())
        total_skipped = sum(r.skipped for r in self.results.values())
        
        overall_success_rate = 0.0
        if total_tests > 0:
            overall_success_rate = (
                (total_tests - total_failures - total_errors) / total_tests * 100
            )
        
        print(f"Total modules tested: {len(self.results)}")
        print(f"Total tests run: {total_tests}")
        print(f"Total failures: {total_failures}")
        print(f"Total errors: {total_errors}")
        print(f"Total skipped: {total_skipped}")
        print(f"Overall success rate: {overall_success_rate:.1f}%")
        print(f"Total execution time: {total_time:.2f}s")
        
        # Module-by-module summary
        print(f"\nModule Summary:")
        for module_name, result in self.results.items():
            status = "✅ PASS" if result.failures == 0 and result.errors == 0 else "❌ FAIL"
            print(f"  {module_name:20} {status:8} ({result.tests_run} tests, {result.success_rate:.1f}%)")
        
        # Print detailed failures and errors if any
        if total_failures > 0 or total_errors > 0:
            print(f"\n{'='*60}")
            print("DETAILED FAILURE/ERROR REPORT")
            print(f"{'='*60}")
            
            for module_name, result in self.results.items():
                if result.failure_details or result.error_details:
                    print(f"\n{module_name}:")
                    
                    for failure in result.failure_details:
                        print(f"  FAILURE: {failure}")
                    
                    for error in result.error_details:
                        print(f"  ERROR: {error}")
        
        # Final status
        if total_failures == 0 and total_errors == 0:
            print(f"\n🎉 ALL TESTS PASSED! 🎉")
            return True
        else:
            print(f"\n💥 SOME TESTS FAILED 💥")
            return False


def check_dependencies():
    """Check if required dependencies are available"""
    print("Checking dependencies...")
    
    dependencies = {
        "pandas": "pandas",
        "pytest": "pytest", 
        "unittest.mock": "unittest.mock (built-in)",
        "pathlib": "pathlib (built-in)",
        "tempfile": "tempfile (built-in)",
        "datetime": "datetime (built-in)"
    }
    
    missing_deps = []
    
    for dep_name, import_name in dependencies.items():
        try:
            if dep_name == "pandas":
                import pandas
            elif dep_name == "pytest":
                import pytest
            elif dep_name == "unittest.mock":
                import unittest.mock
            elif dep_name == "pathlib":
                import pathlib
            elif dep_name == "tempfile":
                import tempfile
            elif dep_name == "datetime":
                import datetime
            print(f"  ✅ {import_name}")
        except ImportError:
            print(f"  ❌ {import_name}")
            missing_deps.append(dep_name)
    
    # Check optional Jupyter dependencies
    jupyter_deps = {
        "IPython": "IPython",
        "ipywidgets": "ipywidgets", 
        "plotly": "plotly",
        "nbformat": "nbformat"
    }
    
    print("\nOptional Jupyter dependencies:")
    for dep_name, import_name in jupyter_deps.items():
        try:
            if dep_name == "IPython":
                import IPython
            elif dep_name == "ipywidgets":
                import ipywidgets
            elif dep_name == "plotly":
                import plotly
            elif dep_name == "nbformat":
                import nbformat
            print(f"  ✅ {import_name}")
        except ImportError:
            print(f"  ⚠️  {import_name} (optional - some tests will be skipped)")
    
    if missing_deps:
        print(f"\n❌ Missing required dependencies: {', '.join(missing_deps)}")
        print("Please install missing dependencies before running tests.")
        return False
    
    print(f"\n✅ All required dependencies available!")
    return True


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="Run Jupyter integration tests")
    parser.add_argument(
        "-v", "--verbosity", 
        type=int, 
        choices=[0, 1, 2], 
        default=2,
        help="Test output verbosity (0=quiet, 1=normal, 2=verbose)"
    )
    parser.add_argument(
        "--check-deps", 
        action="store_true",
        help="Check dependencies and exit"
    )
    parser.add_argument(
        "--module",
        choices=["notebook_interface", "visualization", "debugger", "templates", "advanced"],
        help="Run tests for specific module only"
    )
    
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check_deps:
        check_dependencies()
        return
    
    # Check dependencies before running tests
    if not check_dependencies():
        sys.exit(1)
    
    # Create and run test runner
    runner = JupyterTestRunner(verbosity=args.verbosity)
    
    if args.module:
        # Run tests for specific module
        test_modules = {
            "notebook_interface": [
                TestNotebookSession,
                TestNotebookInterface,
                TestNotebookInterfaceWithJupyter,
                TestNotebookInterfaceEdgeCases
            ],
            "visualization": [
                TestVisualizationConfig,
                TestTestResultMetrics,
                TestVisualizationReporter,
                TestVisualizationReporterWithJupyter,
                TestVisualizationReporterEdgeCases
            ],
            "debugger": [
                TestDebugSession,
                TestBreakpointManager,
                TestInteractiveDebugger,
                TestInteractiveDebuggerWithJupyter,
                TestInteractiveDebuggerEdgeCases
            ],
            "templates": [
                TestNotebookTemplate,
                TestNotebookTemplateManager,
                TestNotebookTemplateManagerWithJupyter,
                TestNotebookTemplateManagerEdgeCases
            ],
            "advanced": [
                TestNotebookSession,
                TestCollaborationManager,
                TestAutomatedReportGenerator,
                TestPerformanceMonitor,
                TestAdvancedWidgetFactory,
                TestAdvancedWidgetFactoryWithJupyter,
                TestAdvancedNotebookFeatures,
                TestAdvancedNotebookFeaturesWithJupyter,
                TestAdvancedNotebookFeaturesEdgeCases
            ]
        }
        
        if args.module in test_modules:
            result = runner.run_module_tests(args.module, test_modules[args.module])
            success = result.failures == 0 and result.errors == 0
        else:
            print(f"Unknown module: {args.module}")
            success = False
    else:
        # Run all tests
        success = runner.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
