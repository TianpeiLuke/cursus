"""
Comprehensive test runner for production validation tests.

This script runs all unit tests for production validation components
with detailed reporting and dependency checking.
"""

import sys
import unittest
import time
import importlib
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class ProductionTestRunner:
    """Test runner for production validation components."""
    
    def __init__(self):
        self.test_modules = [
            'test_e2e_validator',
            'test_performance_optimizer', 
            'test_health_checker',
            'test_deployment_validator'
        ]
        self.results = {}
        self.total_tests = 0
        self.total_failures = 0
        self.total_errors = 0
        self.total_skipped = 0
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check if required dependencies are available."""
        dependencies = {
            'unittest': True,  # Built-in
            'unittest.mock': True,  # Built-in
            'tempfile': True,  # Built-in
            'pathlib': True,  # Built-in
            'datetime': True,  # Built-in
            'json': True,  # Built-in
            'yaml': False,
            'subprocess': True,  # Built-in
        }
        
        # Check optional dependencies
        try:
            import yaml
            dependencies['yaml'] = True
        except ImportError:
            pass
        
        try:
            import psutil
            dependencies['psutil'] = True
        except ImportError:
            dependencies['psutil'] = False
        
        try:
            import boto3
            dependencies['boto3'] = True
        except ImportError:
            dependencies['boto3'] = False
        
        try:
            import pydantic
            dependencies['pydantic'] = True
        except ImportError:
            dependencies['pydantic'] = False
        
        return dependencies
    
    def print_dependency_status(self, dependencies: Dict[str, bool]):
        """Print dependency status."""
        print("Dependency Status:")
        print("-" * 50)
        
        for dep, available in dependencies.items():
            status = "✓ Available" if available else "✗ Missing"
            print(f"  {dep:<20} {status}")
        
        missing_deps = [dep for dep, available in dependencies.items() if not available]
        if missing_deps:
            print(f"\nMissing dependencies: {', '.join(missing_deps)}")
            print("Note: Tests will use graceful fallbacks for missing dependencies")
        
        print()
    
    def discover_and_run_tests(self, module_name: str) -> unittest.TestResult:
        """Discover and run tests for a specific module."""
        try:
            # Import the test module
            test_module = importlib.import_module(f'test.validation.runtime.production.{module_name}')
            
            # Create test suite
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(test_module)
            
            # Run tests
            runner = unittest.TextTestRunner(
                verbosity=0,
                stream=open('/dev/null', 'w') if sys.platform != 'win32' else open('nul', 'w'),
                buffer=True
            )
            result = runner.run(suite)
            
            return result
            
        except ImportError as e:
            print(f"Warning: Could not import {module_name}: {e}")
            # Create a mock result for missing modules
            result = unittest.TestResult()
            result.testsRun = 0
            return result
        except Exception as e:
            print(f"Error running tests for {module_name}: {e}")
            result = unittest.TestResult()
            result.testsRun = 0
            result.errors = [(None, str(e))]
            return result
    
    def run_all_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run all production validation tests."""
        print("Production Validation Test Suite")
        print("=" * 50)
        
        # Check dependencies
        dependencies = self.check_dependencies()
        if verbose:
            self.print_dependency_status(dependencies)
        
        start_time = time.time()
        
        # Run tests for each module
        for module_name in self.test_modules:
            print(f"Running tests for {module_name}...")
            
            module_start_time = time.time()
            result = self.discover_and_run_tests(module_name)
            module_duration = time.time() - module_start_time
            
            # Store results
            self.results[module_name] = {
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
                'duration': module_duration,
                'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1) * 100
            }
            
            # Update totals
            self.total_tests += result.testsRun
            self.total_failures += len(result.failures)
            self.total_errors += len(result.errors)
            self.total_skipped += len(result.skipped) if hasattr(result, 'skipped') else 0
            
            # Print module results
            if result.testsRun > 0:
                success_rate = self.results[module_name]['success_rate']
                print(f"  ✓ {result.testsRun} tests, {success_rate:.1f}% success rate")
            else:
                print(f"  - No tests found or module unavailable")
            
            if verbose and (result.failures or result.errors):
                self._print_detailed_failures(module_name, result)
        
        total_duration = time.time() - start_time
        
        # Print summary
        self._print_summary(total_duration)
        
        return {
            'total_tests': self.total_tests,
            'total_failures': self.total_failures,
            'total_errors': self.total_errors,
            'total_skipped': self.total_skipped,
            'total_duration': total_duration,
            'module_results': self.results,
            'overall_success_rate': (self.total_tests - self.total_failures - self.total_errors) / max(self.total_tests, 1) * 100
        }
    
    def _print_detailed_failures(self, module_name: str, result: unittest.TestResult):
        """Print detailed failure information."""
        if result.failures:
            print(f"    Failures in {module_name}:")
            for test, traceback in result.failures:
                print(f"      - {test}: {traceback.split('AssertionError:')[-1].strip()}")
        
        if result.errors:
            print(f"    Errors in {module_name}:")
            for test, traceback in result.errors:
                print(f"      - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    def _print_summary(self, total_duration: float):
        """Print test execution summary."""
        print("\n" + "=" * 50)
        print("Test Execution Summary")
        print("=" * 50)
        
        print(f"Total Tests Run: {self.total_tests}")
        print(f"Failures: {self.total_failures}")
        print(f"Errors: {self.total_errors}")
        print(f"Skipped: {self.total_skipped}")
        print(f"Total Duration: {total_duration:.2f} seconds")
        
        if self.total_tests > 0:
            success_rate = (self.total_tests - self.total_failures - self.total_errors) / self.total_tests * 100
            print(f"Overall Success Rate: {success_rate:.1f}%")
            
            if success_rate == 100.0:
                print("🎉 All tests passed!")
            elif success_rate >= 90.0:
                print("✅ Most tests passed - excellent!")
            elif success_rate >= 75.0:
                print("⚠️  Good test coverage - some issues to address")
            else:
                print("❌ Multiple test failures - needs attention")
        else:
            print("⚠️  No tests were executed")
        
        # Print module breakdown
        print("\nModule Breakdown:")
        print("-" * 30)
        for module_name, results in self.results.items():
            if results['tests_run'] > 0:
                print(f"  {module_name:<25} {results['tests_run']:>3} tests ({results['success_rate']:>5.1f}%)")
            else:
                print(f"  {module_name:<25} {'N/A':>9}")
    
    def run_specific_module(self, module_name: str, verbose: bool = False) -> Dict[str, Any]:
        """Run tests for a specific module."""
        if module_name not in self.test_modules:
            print(f"Error: Unknown module '{module_name}'")
            print(f"Available modules: {', '.join(self.test_modules)}")
            return {}
        
        print(f"Running tests for {module_name}")
        print("=" * 50)
        
        # Check dependencies
        dependencies = self.check_dependencies()
        if verbose:
            self.print_dependency_status(dependencies)
        
        start_time = time.time()
        result = self.discover_and_run_tests(module_name)
        duration = time.time() - start_time
        
        # Print results
        if result.testsRun > 0:
            success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
            print(f"\nResults: {result.testsRun} tests, {success_rate:.1f}% success rate")
            print(f"Duration: {duration:.2f} seconds")
            
            if result.failures or result.errors:
                self._print_detailed_failures(module_name, result)
        else:
            print("No tests found or module unavailable")
        
        return {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
            'duration': duration,
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1) * 100
        }


def main():
    """Main entry point for the test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run production validation tests')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose output')
    parser.add_argument('--module', '-m', type=str,
                       help='Run tests for specific module only')
    parser.add_argument('--list-modules', '-l', action='store_true',
                       help='List available test modules')
    
    args = parser.parse_args()
    
    runner = ProductionTestRunner()
    
    if args.list_modules:
        print("Available test modules:")
        for module in runner.test_modules:
            print(f"  - {module}")
        return
    
    if args.module:
        results = runner.run_specific_module(args.module, args.verbose)
    else:
        results = runner.run_all_tests(args.verbose)
    
    # Exit with appropriate code
    if results and results.get('total_tests', 0) > 0:
        if results.get('total_failures', 0) > 0 or results.get('total_errors', 0) > 0:
            sys.exit(1)  # Tests failed
        else:
            sys.exit(0)  # All tests passed
    else:
        sys.exit(2)  # No tests run


if __name__ == '__main__':
    main()
