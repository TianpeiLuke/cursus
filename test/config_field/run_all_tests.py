"""
Comprehensive test runner for config_fields module.

This module runs all tests in the config_field directory and provides
detailed reporting on test results, coverage, and any issues found.
"""

import unittest
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import importlib.util

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class ConfigFieldTestRunner:
    """Comprehensive test runner for config_field tests."""
    
    def __init__(self):
        """Initialize the test runner."""
        self.test_dir = Path(__file__).parent
        self.results = {}
        self.total_tests = 0
        self.total_failures = 0
        self.total_errors = 0
        self.total_skipped = 0
        
    def discover_test_files(self) -> List[str]:
        """Discover all test files in the config_field directory."""
        test_files = []
        
        for file_path in self.test_dir.glob("test_*.py"):
            # Skip this runner file itself
            if file_path.name == "run_all_tests.py":
                continue
                
            test_files.append(file_path.stem)
        
        return sorted(test_files)
    
    def load_test_module(self, module_name: str) -> Optional[unittest.TestSuite]:
        """Load a test module and return its test suite."""
        try:
            # Import the module
            spec = importlib.util.spec_from_file_location(
                module_name, 
                self.test_dir / f"{module_name}.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Create test suite from the module
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(module)
            
            return suite
            
        except Exception as e:
            print(f"❌ Failed to load {module_name}: {e}")
            return None
    
    def run_test_suite(self, module_name: str, suite: unittest.TestSuite) -> Dict:
        """Run a test suite and return results."""
        print(f"\n🧪 Running tests from {module_name}...")
        
        # Run the tests
        start_time = time.time()
        result = unittest.TextTestRunner(
            stream=sys.stdout,
            verbosity=1,
            buffer=True
        ).run(suite)
        end_time = time.time()
        
        # Collect results
        test_result = {
            'module': module_name,
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped),
            'success': result.wasSuccessful(),
            'duration': end_time - start_time,
            'failure_details': result.failures,
            'error_details': result.errors,
            'skipped_details': result.skipped
        }
        
        # Update totals
        self.total_tests += result.testsRun
        self.total_failures += len(result.failures)
        self.total_errors += len(result.errors)
        self.total_skipped += len(result.skipped)
        
        return test_result
    
    def print_module_summary(self, result: Dict):
        """Print summary for a single test module."""
        module = result['module']
        tests_run = result['tests_run']
        failures = result['failures']
        errors = result['errors']
        skipped = result['skipped']
        duration = result['duration']
        
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        
        print(f"  {status} {module}: {tests_run} tests, "
              f"{failures} failures, {errors} errors, {skipped} skipped "
              f"({duration:.2f}s)")
        
        # Print failure details if any
        if failures > 0:
            print(f"    Failures:")
            for test, traceback in result['failure_details']:
                print(f"      - {test}: {traceback.split('AssertionError:')[-1].strip()}")
        
        # Print error details if any
        if errors > 0:
            print(f"    Errors:")
            for test, traceback in result['error_details']:
                error_msg = traceback.split('\n')[-2] if '\n' in traceback else traceback
                print(f"      - {test}: {error_msg}")
    
    def print_final_summary(self):
        """Print final summary of all test results."""
        print("\n" + "="*80)
        print("CONFIG FIELDS TEST SUMMARY")
        print("="*80)
        
        total_modules = len(self.results)
        successful_modules = sum(1 for r in self.results.values() if r['success'])
        failed_modules = total_modules - successful_modules
        
        print(f"\nModules: {total_modules} total, {successful_modules} passed, {failed_modules} failed")
        print(f"Tests: {self.total_tests} total, "
              f"{self.total_tests - self.total_failures - self.total_errors} passed, "
              f"{self.total_failures} failed, {self.total_errors} errors, "
              f"{self.total_skipped} skipped")
        
        # Calculate success rate
        if self.total_tests > 0:
            success_rate = ((self.total_tests - self.total_failures - self.total_errors) / self.total_tests) * 100
            print(f"Success Rate: {success_rate:.1f}%")
        
        # Print module-by-module results
        print(f"\nDetailed Results:")
        for module_name, result in self.results.items():
            status = "✅" if result['success'] else "❌"
            print(f"  {status} {module_name}: {result['tests_run']} tests "
                  f"({result['duration']:.2f}s)")
        
        # Print recommendations
        print(f"\nRecommendations:")
        if failed_modules == 0:
            print("  🎉 All tests passed! Great job!")
        else:
            print(f"  🔧 {failed_modules} module(s) have failing tests - review and fix")
        
        if self.total_skipped > 0:
            print(f"  ⚠️  {self.total_skipped} test(s) were skipped - investigate if needed")
        
        # Print coverage insights
        print(f"\nCoverage Insights:")
        print(f"  • Core functionality: {self._assess_core_coverage()}")
        print(f"  • Edge cases: {self._assess_edge_case_coverage()}")
        print(f"  • Integration: {self._assess_integration_coverage()}")
    
    def _assess_core_coverage(self) -> str:
        """Assess core functionality coverage."""
        core_modules = [
            'test_config_field_categorizer',
            'test_config_merger', 
            'test_type_aware_serialization',
            'test_circular_reference_tracker'
        ]
        
        covered = sum(1 for module in core_modules if module in self.results)
        total = len(core_modules)
        
        if covered == total:
            return f"Excellent ({covered}/{total} modules)"
        elif covered >= total * 0.75:
            return f"Good ({covered}/{total} modules)"
        else:
            return f"Needs improvement ({covered}/{total} modules)"
    
    def _assess_edge_case_coverage(self) -> str:
        """Assess edge case coverage."""
        edge_case_modules = [
            'test_circular_reference_consolidated',
            'test_bug_fixes_consolidated',
            'test_constants'
        ]
        
        covered = sum(1 for module in edge_case_modules if module in self.results)
        total = len(edge_case_modules)
        
        if covered == total:
            return f"Excellent ({covered}/{total} modules)"
        elif covered >= total * 0.67:
            return f"Good ({covered}/{total} modules)"
        else:
            return f"Needs improvement ({covered}/{total} modules)"
    
    def _assess_integration_coverage(self) -> str:
        """Assess integration test coverage."""
        integration_modules = [
            'test_integration',
            'test_config_class_store',
            'test_tier_registry'
        ]
        
        covered = sum(1 for module in integration_modules if module in self.results)
        total = len(integration_modules)
        
        if covered == total:
            return f"Excellent ({covered}/{total} modules)"
        elif covered >= total * 0.67:
            return f"Good ({covered}/{total} modules)"
        else:
            return f"Needs improvement ({covered}/{total} modules)"
    
    def run_all_tests(self):
        """Run all tests and provide comprehensive reporting."""
        print("🚀 Starting comprehensive config_fields test run...")
        print(f"📁 Test directory: {self.test_dir}")
        
        # Discover test files
        test_files = self.discover_test_files()
        print(f"📋 Found {len(test_files)} test modules:")
        for test_file in test_files:
            print(f"   • {test_file}")
        
        if not test_files:
            print("❌ No test files found!")
            return
        
        # Run each test module
        start_time = time.time()
        
        for test_file in test_files:
            suite = self.load_test_module(test_file)
            if suite:
                result = self.run_test_suite(test_file, suite)
                self.results[test_file] = result
                self.print_module_summary(result)
            else:
                print(f"⚠️  Skipped {test_file} due to loading errors")
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Print final summary
        self.print_final_summary()
        print(f"\n⏱️  Total execution time: {total_duration:.2f} seconds")
        
        # Return overall success
        return self.total_failures == 0 and self.total_errors == 0


def main():
    """Main entry point for the test runner."""
    print("Config Fields Comprehensive Test Runner")
    print("=" * 50)
    
    # Create and run the test runner
    runner = ConfigFieldTestRunner()
    success = runner.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
