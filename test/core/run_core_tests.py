#!/usr/bin/env python3
"""
Comprehensive Test Runner for Cursus Core Package

This program runs all tests that cover the core package components:
- assembler
- base  
- compiler
- config_fields (config_field in test directory)
- deps

It provides detailed reporting on test results, coverage analysis, and redundancy assessment.
"""

import sys
import os
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
import ast
import re

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
)

@dataclass
class TestResult:
    """Data class to store test results for a module."""
    module: str
    component: str
    tests_run: int
    failures: int
    errors: int
    skipped: int
    success: bool
    duration: float
    failure_details: List[Tuple[str, str]]
    error_details: List[Tuple[str, str]]
    skipped_details: List[Tuple[str, str]]

@dataclass
class CoverageAnalysis:
    """Data class to store coverage analysis results."""
    component: str
    test_files: List[str]
    source_files: List[str]
    tested_functions: Set[str]
    untested_functions: Set[str]
    coverage_percentage: float
    redundant_tests: List[str]
    missing_edge_cases: List[str]

class CoreTestRunner:
    """Comprehensive test runner for cursus core package."""
    
    CORE_COMPONENTS = {
        'assembler': 'assembler',
        'base': 'base', 
        'compiler': 'compiler',
        'config_fields': 'config_fields',  # Note: directory name is config_fields
        'deps': 'deps'
    }
    
    SOURCE_COMPONENTS = {
        'assembler': 'src/cursus/core/assembler',
        'base': 'src/cursus/core/base',
        'compiler': 'src/cursus/core/compiler', 
        'config_fields': 'src/cursus/core/config_fields',
        'deps': 'src/cursus/core/deps'
    }
    
    def __init__(self):
        """Initialize the test runner."""
        self.project_root = PROJECT_ROOT
        self.results: Dict[str, List[TestResult]] = defaultdict(list)
        self.coverage_analysis: Dict[str, CoverageAnalysis] = {}
        self.total_tests = 0
        self.total_failures = 0
        self.total_errors = 0
        self.total_skipped = 0
        self.start_time = None
        self.end_time = None
        
    def discover_test_files(self, component: str) -> List[str]:
        """Discover all test files for a given component."""
        test_dir = self.project_root / "test" / "core" / self.CORE_COMPONENTS[component]
        
        if not test_dir.exists():
            print(f"⚠️  Test directory {test_dir} does not exist for component {component}")
            return []
            
        test_files = []
        for file_path in test_dir.glob("test_*.py"):
            # Skip special files
            if file_path.name in ["run_all_tests.py", "__init__.py"]:
                continue
            test_files.append(file_path.stem)
            
        return sorted(test_files)
    
    def discover_source_files(self, component: str) -> List[str]:
        """Discover all source files for a given component."""
        source_dir = self.project_root / self.SOURCE_COMPONENTS[component]
        
        if not source_dir.exists():
            print(f"⚠️  Source directory {source_dir} does not exist for component {component}")
            return []
            
        source_files = []
        for file_path in source_dir.rglob("*.py"):
            if file_path.name != "__init__.py":
                source_files.append(str(file_path.relative_to(self.project_root)))
                
        return sorted(source_files)
    
    def run_pytest_module(self, component: str, module_name: str) -> Optional[Dict]:
        """Run a test module using pytest and return results."""
        try:
            # Use absolute paths from project root
            test_dir = self.project_root / "test" / "core" / self.CORE_COMPONENTS[component]
            module_path = test_dir / f"{module_name}.py"
            
            if not module_path.exists():
                print(f"❌ Test file {module_path} does not exist")
                return None
            
            # Run pytest with verbose output using absolute path
            cmd = [
                sys.executable, "-m", "pytest", 
                str(module_path),
                "-v", "--tb=short"
            ]
            
            # Change to the project root to ensure proper imports
            original_cwd = os.getcwd()
            os.chdir(self.project_root)
            
            try:
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=300  # 5 minute timeout
                )
                
                # Parse pytest output manually
                return self.parse_pytest_output(result.stdout, result.stderr, result.returncode)
                
            finally:
                os.chdir(original_cwd)
                
        except Exception as e:
            print(f"❌ Failed to run {component}/{module_name}: {e}")
            return None
    
    def parse_pytest_output(self, stdout: str, stderr: str, returncode: int) -> Dict:
        """Parse pytest output to extract test results."""
        lines = stdout.split('\n')
        
        # Extract test counts from summary line
        tests_run = 0
        failures = 0
        errors = 0
        skipped = 0
        passed = 0
        
        for line in lines:
            if '=====' in line and ('passed' in line or 'failed' in line or 'error' in line):
                # Parse summary line like "===== 26 passed, 17 warnings in 0.88s ====="
                if 'passed' in line:
                    passed_match = re.search(r'(\d+) passed', line)
                    if passed_match:
                        passed = int(passed_match.group(1))
                
                if 'failed' in line:
                    failed_match = re.search(r'(\d+) failed', line)
                    if failed_match:
                        failures = int(failed_match.group(1))
                
                if 'error' in line:
                    error_match = re.search(r'(\d+) error', line)
                    if error_match:
                        errors = int(error_match.group(1))
                
                if 'skipped' in line:
                    skipped_match = re.search(r'(\d+) skipped', line)
                    if skipped_match:
                        skipped = int(skipped_match.group(1))
        
        tests_run = passed + failures + errors + skipped
        
        # Extract failure details
        failure_details = []
        error_details = []
        
        # Look for FAILURES section
        in_failures = False
        current_test = None
        current_traceback = []
        
        for line in lines:
            if line.startswith('FAILURES'):
                in_failures = True
                continue
            elif line.startswith('ERRORS'):
                in_failures = False
                continue
            elif line.startswith('====='):
                in_failures = False
                continue
            
            if in_failures:
                if line.startswith('_____'):
                    # New test failure
                    if current_test and current_traceback:
                        failure_details.append((current_test, '\n'.join(current_traceback)))
                    
                    # Extract test name
                    test_match = re.search(r'_+ (.+) _+', line)
                    current_test = test_match.group(1) if test_match else line.strip('_').strip()
                    current_traceback = []
                elif current_test:
                    current_traceback.append(line)
        
        # Add the last failure if any
        if current_test and current_traceback:
            failure_details.append((current_test, '\n'.join(current_traceback)))
        
        return {
            'tests_run': tests_run,
            'passed': passed,
            'failures': failures,
            'errors': errors,
            'skipped': skipped,
            'success': failures == 0 and errors == 0,
            'failure_details': failure_details,
            'error_details': error_details,
            'stdout': stdout,
            'stderr': stderr,
            'returncode': returncode
        }
    
    def run_test_module(self, component: str, module_name: str) -> TestResult:
        """Run a test module using pytest and return results."""
        print(f"  🧪 Running {module_name}...")
        
        start_time = time.time()
        pytest_result = self.run_pytest_module(component, module_name)
        end_time = time.time()
        
        if not pytest_result:
            # Return empty result if pytest failed to run
            return TestResult(
                module=module_name,
                component=component,
                tests_run=0,
                failures=0,
                errors=1,
                skipped=0,
                success=False,
                duration=end_time - start_time,
                failure_details=[],
                error_details=[("Module execution", "Failed to run pytest")],
                skipped_details=[]
            )
        
        # Update totals
        self.total_tests += pytest_result['tests_run']
        self.total_failures += pytest_result['failures']
        self.total_errors += pytest_result['errors']
        self.total_skipped += pytest_result['skipped']
        
        # Create result object
        test_result = TestResult(
            module=module_name,
            component=component,
            tests_run=pytest_result['tests_run'],
            failures=pytest_result['failures'],
            errors=pytest_result['errors'],
            skipped=pytest_result['skipped'],
            success=pytest_result['success'],
            duration=end_time - start_time,
            failure_details=pytest_result['failure_details'],
            error_details=pytest_result['error_details'],
            skipped_details=[]
        )
        
        return test_result
    
    def analyze_function_coverage(self, component: str) -> CoverageAnalysis:
        """Analyze function coverage for a component."""
        test_files = self.discover_test_files(component)
        source_files = self.discover_source_files(component)
        
        # Extract functions from source files
        source_functions = set()
        for source_file in source_files:
            try:
                # Use absolute path for file reading
                absolute_path = self.project_root / source_file
                with open(absolute_path, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                    
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if not node.name.startswith('_'):  # Skip private functions
                            source_functions.add(f"{Path(source_file).stem}.{node.name}")
                    elif isinstance(node, ast.ClassDef):
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef) and not item.name.startswith('_'):
                                source_functions.add(f"{node.name}.{item.name}")
            except Exception as e:
                print(f"⚠️  Could not parse {source_file}: {e}")
        
        # Extract tested functions from test files
        tested_functions = set()
        redundant_tests = []
        test_function_counts = defaultdict(int)
        
        for test_file in test_files:
            try:
                test_path = self.project_root / "test" / "core" / self.CORE_COMPONENTS[component] / f"{test_file}.py"
                with open(test_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Look for function calls and imports that indicate testing
                for line in content.split('\n'):
                    # Find test method names that might indicate what's being tested
                    if 'def test_' in line:
                        test_name = re.search(r'def (test_\w+)', line)
                        if test_name:
                            test_function_counts[test_name.group(1)] += 1
                    
                    # Look for direct function calls or imports
                    for func in source_functions:
                        if func.split('.')[-1] in line and 'import' not in line:
                            tested_functions.add(func)
                            
            except Exception as e:
                print(f"⚠️  Could not analyze {test_file}: {e}")
        
        # Find redundant tests (same test name appearing multiple times)
        for test_name, count in test_function_counts.items():
            if count > 1:
                redundant_tests.append(f"{test_name} (appears {count} times)")
        
        # Calculate coverage
        untested_functions = source_functions - tested_functions
        coverage_percentage = (len(tested_functions) / len(source_functions) * 100) if source_functions else 0
        
        # Identify missing edge cases (heuristic based on common patterns)
        missing_edge_cases = []
        common_edge_cases = ['empty', 'null', 'invalid', 'error', 'exception', 'boundary', 'edge']
        
        for test_file in test_files:
            test_path = self.project_root / "test" / "core" / self.CORE_COMPONENTS[component] / f"{test_file}.py"
            try:
                with open(test_path, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                missing_cases = []
                for case in common_edge_cases:
                    if case not in content:
                        missing_cases.append(case)
                        
                if missing_cases:
                    missing_edge_cases.append(f"{test_file}: missing {', '.join(missing_cases)} tests")
            except Exception:
                pass
        
        return CoverageAnalysis(
            component=component,
            test_files=test_files,
            source_files=source_files,
            tested_functions=tested_functions,
            untested_functions=untested_functions,
            coverage_percentage=coverage_percentage,
            redundant_tests=redundant_tests,
            missing_edge_cases=missing_edge_cases
        )
    
    def run_component_tests(self, component: str) -> List[TestResult]:
        """Run all tests for a specific component."""
        print(f"\n📦 Testing {component.upper()} component...")
        
        test_files = self.discover_test_files(component)
        if not test_files:
            print(f"  ⚠️  No test files found for {component}")
            return []
        
        print(f"  📋 Found {len(test_files)} test modules")
        
        component_results = []
        for test_file in test_files:
            result = self.run_test_module(component, test_file)
            component_results.append(result)
            
            # Print immediate feedback
            status = "✅" if result.success else "❌"
            print(f"    {status} {test_file}: {result.tests_run} tests, "
                  f"{result.failures} failures, {result.errors} errors "
                  f"({result.duration:.2f}s)")
        
        return component_results
    
    def run_all_tests(self):
        """Run all tests for all core components."""
        print("🚀 Starting comprehensive core package test run...")
        print(f"📁 Project root: {self.project_root}")
        print(f"🎯 Testing components: {', '.join(self.CORE_COMPONENTS.keys())}")
        
        self.start_time = time.time()
        
        # Run tests for each component
        for component in self.CORE_COMPONENTS.keys():
            component_results = self.run_component_tests(component)
            self.results[component] = component_results
            
            # Analyze coverage for this component
            print(f"  📊 Analyzing coverage for {component}...")
            self.coverage_analysis[component] = self.analyze_function_coverage(component)
        
        self.end_time = time.time()
        
        # Generate comprehensive report
        self.generate_report()
    
    def print_component_summary(self, component: str, results: List[TestResult]):
        """Print summary for a component."""
        if not results:
            print(f"  ❌ {component.upper()}: No tests run")
            return
            
        total_tests = sum(r.tests_run for r in results)
        total_failures = sum(r.failures for r in results)
        total_errors = sum(r.errors for r in results)
        total_skipped = sum(r.skipped for r in results)
        successful_modules = sum(1 for r in results if r.success)
        total_duration = sum(r.duration for r in results)
        
        status = "✅" if total_failures == 0 and total_errors == 0 else "❌"
        print(f"  {status} {component.upper()}: {len(results)} modules, {total_tests} tests, "
              f"{successful_modules}/{len(results)} modules passed ({total_duration:.2f}s)")
        
        if total_failures > 0 or total_errors > 0:
            print(f"    Failures: {total_failures}, Errors: {total_errors}")
    
    def print_coverage_summary(self, component: str, analysis: CoverageAnalysis):
        """Print coverage summary for a component."""
        print(f"\n  📊 Coverage Analysis for {component.upper()}:")
        print(f"    Source files: {len(analysis.source_files)}")
        print(f"    Test files: {len(analysis.test_files)}")
        print(f"    Functions tested: {len(analysis.tested_functions)}")
        print(f"    Functions untested: {len(analysis.untested_functions)}")
        print(f"    Coverage: {analysis.coverage_percentage:.1f}%")
        
        if analysis.redundant_tests:
            print(f"    ⚠️  Redundant tests: {len(analysis.redundant_tests)}")
        
        if analysis.missing_edge_cases:
            print(f"    ⚠️  Missing edge cases: {len(analysis.missing_edge_cases)}")
    
    def generate_report(self):
        """Generate comprehensive test and coverage report."""
        print("\n" + "="*100)
        print("CURSUS CORE PACKAGE TEST REPORT")
        print("="*100)
        
        # Overall summary
        total_duration = self.end_time - self.start_time
        total_modules = sum(len(results) for results in self.results.values())
        successful_tests = self.total_tests - self.total_failures - self.total_errors
        
        print(f"\n📈 OVERALL SUMMARY")
        print(f"   Components tested: {len(self.CORE_COMPONENTS)}")
        print(f"   Test modules: {total_modules}")
        print(f"   Total tests: {self.total_tests}")
        print(f"   Passed: {successful_tests}")
        print(f"   Failed: {self.total_failures}")
        print(f"   Errors: {self.total_errors}")
        print(f"   Skipped: {self.total_skipped}")
        print(f"   Success rate: {(successful_tests/self.total_tests*100):.1f}%" if self.total_tests > 0 else "N/A")
        print(f"   Total duration: {total_duration:.2f} seconds")
        
        # Component-by-component results
        print(f"\n📦 COMPONENT RESULTS")
        for component, results in self.results.items():
            self.print_component_summary(component, results)
            if component in self.coverage_analysis:
                self.print_coverage_summary(component, self.coverage_analysis[component])
        
        # Detailed failure analysis
        self.print_failure_analysis()
        
        # Coverage and redundancy analysis
        self.print_coverage_analysis()
        
        # Recommendations
        self.print_recommendations()
        
        # Save detailed report to file
        self.save_detailed_report()
    
    def print_failure_analysis(self):
        """Print detailed failure analysis."""
        print(f"\n🔍 FAILURE ANALYSIS")
        
        has_failures = False
        for component, results in self.results.items():
            component_failures = [r for r in results if not r.success]
            if component_failures:
                has_failures = True
                print(f"\n  ❌ {component.upper()} failures:")
                
                for result in component_failures:
                    print(f"    📄 {result.module}:")
                    
                    for test, traceback in result.failure_details:
                        # Extract just the assertion error message
                        error_msg = traceback.split('AssertionError:')[-1].strip() if 'AssertionError:' in traceback else "Unknown failure"
                        print(f"      • {test}: {error_msg[:100]}...")
                    
                    for test, traceback in result.error_details:
                        # Extract the error type and message
                        lines = traceback.strip().split('\n')
                        error_line = lines[-1] if lines else "Unknown error"
                        print(f"      • {test}: {error_line[:100]}...")
        
        if not has_failures:
            print("  🎉 No failures detected!")
    
    def print_coverage_analysis(self):
        """Print comprehensive coverage and redundancy analysis."""
        print(f"\n📊 COVERAGE & REDUNDANCY ANALYSIS")
        
        # Overall coverage statistics
        total_source_files = sum(len(analysis.source_files) for analysis in self.coverage_analysis.values())
        total_test_files = sum(len(analysis.test_files) for analysis in self.coverage_analysis.values())
        total_tested_functions = sum(len(analysis.tested_functions) for analysis in self.coverage_analysis.values())
        total_untested_functions = sum(len(analysis.untested_functions) for analysis in self.coverage_analysis.values())
        
        overall_coverage = (total_tested_functions / (total_tested_functions + total_untested_functions) * 100) if (total_tested_functions + total_untested_functions) > 0 else 0
        
        print(f"\n  📈 Overall Coverage Statistics:")
        print(f"    Source files: {total_source_files}")
        print(f"    Test files: {total_test_files}")
        print(f"    Functions tested: {total_tested_functions}")
        print(f"    Functions untested: {total_untested_functions}")
        print(f"    Overall coverage: {overall_coverage:.1f}%")
        
        # Component-specific analysis
        print(f"\n  📦 Component Coverage Details:")
        for component, analysis in self.coverage_analysis.items():
            coverage_status = "🟢" if analysis.coverage_percentage >= 80 else "🟡" if analysis.coverage_percentage >= 60 else "🔴"
            print(f"    {coverage_status} {component}: {analysis.coverage_percentage:.1f}% coverage")
            
            if analysis.untested_functions:
                print(f"      Untested functions: {', '.join(list(analysis.untested_functions)[:5])}" + 
                      (f" (+{len(analysis.untested_functions)-5} more)" if len(analysis.untested_functions) > 5 else ""))
        
        # Redundancy analysis
        print(f"\n  🔄 Redundancy Analysis:")
        total_redundant = sum(len(analysis.redundant_tests) for analysis in self.coverage_analysis.values())
        
        if total_redundant > 0:
            print(f"    ⚠️  Found {total_redundant} potential redundant tests:")
            for component, analysis in self.coverage_analysis.items():
                if analysis.redundant_tests:
                    print(f"      {component}: {len(analysis.redundant_tests)} redundant tests")
                    for redundant in analysis.redundant_tests[:3]:  # Show first 3
                        print(f"        • {redundant}")
        else:
            print(f"    ✅ No obvious test redundancy detected")
        
        # Edge case analysis
        print(f"\n  🎯 Edge Case Analysis:")
        total_missing_edge_cases = sum(len(analysis.missing_edge_cases) for analysis in self.coverage_analysis.values())
        
        if total_missing_edge_cases > 0:
            print(f"    ⚠️  Potential missing edge cases in {total_missing_edge_cases} areas:")
            for component, analysis in self.coverage_analysis.items():
                if analysis.missing_edge_cases:
                    for missing in analysis.missing_edge_cases[:2]:  # Show first 2
                        print(f"      {component}: {missing}")
        else:
            print(f"    ✅ Good edge case coverage detected")
    
    def print_recommendations(self):
        """Print actionable recommendations."""
        print(f"\n💡 RECOMMENDATIONS")
        
        recommendations = []
        
        # Test failure recommendations
        if self.total_failures > 0 or self.total_errors > 0:
            recommendations.append(f"🔧 Fix {self.total_failures + self.total_errors} failing tests before deployment")
        
        # Coverage recommendations
        low_coverage_components = [
            component for component, analysis in self.coverage_analysis.items()
            if analysis.coverage_percentage < 70
        ]
        
        if low_coverage_components:
            recommendations.append(f"📈 Improve test coverage for: {', '.join(low_coverage_components)}")
        
        # Redundancy recommendations
        redundant_components = [
            component for component, analysis in self.coverage_analysis.items()
            if analysis.redundant_tests
        ]
        
        if redundant_components:
            recommendations.append(f"🔄 Review and consolidate redundant tests in: {', '.join(redundant_components)}")
        
        # Edge case recommendations
        edge_case_components = [
            component for component, analysis in self.coverage_analysis.items()
            if analysis.missing_edge_cases
        ]
        
        if edge_case_components:
            recommendations.append(f"🎯 Add edge case tests for: {', '.join(edge_case_components)}")
        
        # Print recommendations
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        else:
            print("  🎉 All tests are in excellent condition!")
        
        # Priority recommendations
        print(f"\n  🚨 Priority Actions:")
        if self.total_failures > 0:
            print(f"    1. Fix {self.total_failures} test failures immediately")
        if self.total_errors > 0:
            print(f"    2. Resolve {self.total_errors} test errors")
        
        overall_coverage = sum(analysis.coverage_percentage for analysis in self.coverage_analysis.values()) / len(self.coverage_analysis) if self.coverage_analysis else 0
        if overall_coverage < 80:
            print(f"    3. Increase overall test coverage from {overall_coverage:.1f}% to 80%+")
    
    def save_detailed_report(self):
        """Save detailed report to JSON file."""
        report_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total_tests': self.total_tests,
                'total_failures': self.total_failures,
                'total_errors': self.total_errors,
                'total_skipped': self.total_skipped,
                'success_rate': (self.total_tests - self.total_failures - self.total_errors) / self.total_tests * 100 if self.total_tests > 0 else 0,
                'duration': self.end_time - self.start_time if self.start_time and self.end_time else 0
            },
            'components': {},
            'coverage_analysis': {}
        }
        
        # Add component results
        for component, results in self.results.items():
            report_data['components'][component] = [asdict(result) for result in results]
        
        # Add coverage analysis (convert sets to lists for JSON serialization)
        for component, analysis in self.coverage_analysis.items():
            analysis_dict = asdict(analysis)
            analysis_dict['tested_functions'] = list(analysis_dict['tested_functions'])
            analysis_dict['untested_functions'] = list(analysis_dict['untested_functions'])
            report_data['coverage_analysis'][component] = analysis_dict
        
        # Save to file in test/core directory
        report_file = self.project_root / 'test' / 'core' / 'core_test_report.json'
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, default=str)
            print(f"\n💾 Detailed report saved to: {report_file}")
        except Exception as e:
            print(f"⚠️  Could not save detailed report: {e}")

def main():
    """Main entry point for the core test runner."""
    print("Cursus Core Package Comprehensive Test Runner")
    print("=" * 60)
    
    # Create and run the test runner
    runner = CoreTestRunner()
    runner.run_all_tests()
    
    # Exit with appropriate code
    success = runner.total_failures == 0 and runner.total_errors == 0
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
