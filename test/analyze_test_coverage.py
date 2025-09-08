#!/usr/bin/env python3
"""
Comprehensive Test Coverage Analysis for Cursus Core Package

This program analyzes test coverage and redundancy for the core package components:
- assembler
- base
- compiler
- config_fields (config_field in test directory)
- deps

It provides detailed reporting on test results, coverage analysis, and redundancy assessment.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter
import ast
import re

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
)

class TestCoverageAnalyzer:
    """Comprehensive test coverage analyzer for cursus core package."""
    
    CORE_COMPONENTS = {
        'assembler': ('core/assembler', '../src/cursus/core/assembler'),
        'base': ('core/base', '../src/cursus/core/base'),
        'compiler': ('core/compiler', '../src/cursus/core/compiler'),
        'config_fields': ('core/config_fields', '../src/cursus/core/config_fields'),
        'deps': ('core/deps', '../src/cursus/core/deps')
    }
    
    def __init__(self):
        """Initialize the analyzer."""
        self.project_root = PROJECT_ROOT
        self.test_report = None
        self.coverage_data = {}
        self.redundancy_data = {}
        self.function_coverage = {}
        
    def load_test_report(self, report_path: str = "test/core_test_report.json"):
        """Load the test report from JSON file."""
        report_file = self.project_root / report_path
        if not report_file.exists():
            print(f"❌ Test report not found: {report_file}")
            return False
            
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                self.test_report = json.load(f)
            print(f"✅ Loaded test report from: {report_file}")
            return True
        except Exception as e:
            print(f"❌ Failed to load test report: {e}")
            return False
    
    def analyze_source_functions(self, component: str) -> Dict[str, Set[str]]:
        """Analyze functions in source files for a component."""
        _, source_path = self.CORE_COMPONENTS[component]
        source_dir = self.project_root / source_path.replace('../', '')
        
        if not source_dir.exists():
            print(f"⚠️  Source directory not found: {source_dir}")
            return {}
        
        functions_by_file = {}
        
        for py_file in source_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                tree = ast.parse(content)
                functions = set()
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if not node.name.startswith('_'):  # Skip private functions
                            functions.add(node.name)
                    elif isinstance(node, ast.ClassDef):
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef) and not item.name.startswith('_'):
                                functions.add(f"{node.name}.{item.name}")
                
                if functions:
                    rel_path = str(py_file.relative_to(self.project_root))
                    functions_by_file[rel_path] = functions
                    
            except Exception as e:
                print(f"⚠️  Could not parse {py_file}: {e}")
        
        return functions_by_file
    
    def analyze_test_functions(self, component: str) -> Dict[str, Set[str]]:
        """Analyze test functions for a component."""
        test_path, _ = self.CORE_COMPONENTS[component]
        # Use absolute path from project root
        test_dir = self.project_root / 'test' / test_path
        
        if not test_dir.exists():
            print(f"⚠️  Test directory not found: {test_dir}")
            return {}
        
        test_functions_by_file = {}
        
        for py_file in test_dir.rglob("test_*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                tree = ast.parse(content)
                test_functions = set()
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                        test_functions.add(node.name)
                
                if test_functions:
                    rel_path = str(py_file.relative_to(self.project_root))
                    test_functions_by_file[rel_path] = test_functions
                    
            except Exception as e:
                print(f"⚠️  Could not parse {py_file}: {e}")
        
        return test_functions_by_file
    
    def analyze_function_coverage(self, component: str) -> Dict:
        """Analyze function coverage for a component."""
        source_functions = self.analyze_source_functions(component)
        test_functions = self.analyze_test_functions(component)
        
        # Flatten source functions
        all_source_functions = set()
        for functions in source_functions.values():
            all_source_functions.update(functions)
        
        # Flatten test functions
        all_test_functions = set()
        for functions in test_functions.values():
            all_test_functions.update(functions)
        
        # Analyze which source functions are likely tested
        tested_functions = set()
        for test_func in all_test_functions:
            # Extract potential function name from test name
            # e.g., test_create_pipeline -> create_pipeline
            if test_func.startswith('test_'):
                potential_func = test_func[5:]  # Remove 'test_' prefix
                
                # Look for matching source functions
                for source_func in all_source_functions:
                    if potential_func in source_func.lower() or source_func.lower() in potential_func:
                        tested_functions.add(source_func)
        
        # Calculate coverage
        total_functions = len(all_source_functions)
        tested_count = len(tested_functions)
        coverage_percentage = (tested_count / total_functions * 100) if total_functions > 0 else 0
        
        return {
            'component': component,
            'source_files': list(source_functions.keys()),
            'test_files': list(test_functions.keys()),
            'total_source_functions': total_functions,
            'tested_functions': tested_count,
            'untested_functions': total_functions - tested_count,
            'coverage_percentage': coverage_percentage,
            'source_functions_by_file': {k: list(v) for k, v in source_functions.items()},
            'test_functions_by_file': {k: list(v) for k, v in test_functions.items()},
            'likely_tested_functions': list(tested_functions),
            'likely_untested_functions': list(all_source_functions - tested_functions)
        }
    
    def analyze_test_redundancy(self, component: str) -> Dict:
        """Analyze test redundancy for a component."""
        test_functions = self.analyze_test_functions(component)
        
        # Count test function names across all files
        function_counts = Counter()
        function_locations = defaultdict(list)
        
        for file_path, functions in test_functions.items():
            for func in functions:
                function_counts[func] += 1
                function_locations[func].append(file_path)
        
        # Find redundant tests (same name in multiple files)
        redundant_tests = {}
        for func_name, count in function_counts.items():
            if count > 1:
                redundant_tests[func_name] = {
                    'count': count,
                    'locations': function_locations[func_name]
                }
        
        # Analyze test patterns for potential redundancy
        pattern_analysis = self._analyze_test_patterns(test_functions)
        
        return {
            'component': component,
            'total_test_functions': sum(len(funcs) for funcs in test_functions.values()),
            'unique_test_names': len(function_counts),
            'redundant_test_names': len(redundant_tests),
            'redundant_tests': redundant_tests,
            'pattern_analysis': pattern_analysis
        }
    
    def _analyze_test_patterns(self, test_functions: Dict[str, Set[str]]) -> Dict:
        """Analyze test patterns for potential redundancy."""
        patterns = defaultdict(list)
        
        for file_path, functions in test_functions.items():
            for func in functions:
                # Extract pattern from test name
                # e.g., test_init_with_required_fields -> init_with_required_fields
                if func.startswith('test_'):
                    pattern = func[5:]
                    patterns[pattern].append((file_path, func))
        
        # Find patterns that appear in multiple files
        redundant_patterns = {}
        for pattern, occurrences in patterns.items():
            if len(occurrences) > 1:
                redundant_patterns[pattern] = occurrences
        
        return redundant_patterns
    
    def analyze_test_quality(self, component: str) -> Dict:
        """Analyze test quality metrics for a component."""
        test_path, _ = self.CORE_COMPONENTS[component]
        test_dir = Path(test_path)
        
        if not test_dir.exists():
            return {}
        
        quality_metrics = {
            'component': component,
            'edge_case_coverage': self._analyze_edge_case_coverage(test_dir),
            'assertion_patterns': self._analyze_assertion_patterns(test_dir),
            'mock_usage': self._analyze_mock_usage(test_dir),
            'test_isolation': self._analyze_test_isolation(test_dir)
        }
        
        return quality_metrics
    
    def _analyze_edge_case_coverage(self, test_dir: Path) -> Dict:
        """Analyze edge case coverage in tests."""
        edge_case_keywords = [
            'empty', 'null', 'none', 'invalid', 'error', 'exception',
            'boundary', 'edge', 'limit', 'max', 'min', 'zero'
        ]
        
        edge_case_coverage = {}
        
        for py_file in test_dir.rglob("test_*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                
                found_keywords = []
                for keyword in edge_case_keywords:
                    if keyword in content:
                        found_keywords.append(keyword)
                
                if found_keywords:
                    rel_path = str(py_file.relative_to(test_dir))
                    edge_case_coverage[rel_path] = found_keywords
                    
            except Exception as e:
                print(f"⚠️  Could not analyze {py_file}: {e}")
        
        return edge_case_coverage
    
    def _analyze_assertion_patterns(self, test_dir: Path) -> Dict:
        """Analyze assertion patterns in tests."""
        assertion_patterns = defaultdict(int)
        
        for py_file in test_dir.rglob("test_*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Count different assertion types
                assertions = [
                    'assertEqual', 'assertNotEqual', 'assertTrue', 'assertFalse',
                    'assertIn', 'assertNotIn', 'assertIsNone', 'assertIsNotNone',
                    'assertRaises', 'assertGreater', 'assertLess', 'assertIsInstance'
                ]
                
                for assertion in assertions:
                    count = content.count(assertion)
                    if count > 0:
                        assertion_patterns[assertion] += count
                        
            except Exception as e:
                print(f"⚠️  Could not analyze {py_file}: {e}")
        
        return dict(assertion_patterns)
    
    def _analyze_mock_usage(self, test_dir: Path) -> Dict:
        """Analyze mock usage in tests."""
        mock_patterns = defaultdict(int)
        
        for py_file in test_dir.rglob("test_*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Count mock usage patterns
                mock_keywords = [
                    'Mock(', 'MagicMock(', '@patch', 'mock.patch',
                    'return_value', 'side_effect', 'assert_called'
                ]
                
                for keyword in mock_keywords:
                    count = content.count(keyword)
                    if count > 0:
                        mock_patterns[keyword] += count
                        
            except Exception as e:
                print(f"⚠️  Could not analyze {py_file}: {e}")
        
        return dict(mock_patterns)
    
    def _analyze_test_isolation(self, test_dir: Path) -> Dict:
        """Analyze test isolation patterns."""
        isolation_patterns = {
            'setUp_methods': 0,
            'tearDown_methods': 0,
            'isolated_test_case_usage': 0,
            'global_state_resets': 0
        }
        
        for py_file in test_dir.rglob("test_*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                isolation_patterns['setUp_methods'] += content.count('def setUp(')
                isolation_patterns['tearDown_methods'] += content.count('def tearDown(')
                isolation_patterns['isolated_test_case_usage'] += content.count('IsolatedTestCase')
                isolation_patterns['global_state_resets'] += content.count('reset_all_global_state')
                        
            except Exception as e:
                print(f"⚠️  Could not analyze {py_file}: {e}")
        
        return isolation_patterns
    
    def generate_comprehensive_report(self):
        """Generate comprehensive test coverage and redundancy report."""
        print("\n" + "="*100)
        print("CURSUS CORE PACKAGE - COMPREHENSIVE TEST COVERAGE ANALYSIS")
        print("="*100)
        
        if not self.test_report:
            print("❌ No test report loaded. Please run the core tests first.")
            return
        
        # Overall summary from test report
        summary = self.test_report.get('summary', {})
        print(f"\n📊 TEST EXECUTION SUMMARY")
        print(f"   Total tests: {summary.get('total_tests', 0)}")
        print(f"   Success rate: {summary.get('success_rate', 0):.1f}%")
        print(f"   Duration: {summary.get('duration', 0):.2f} seconds")
        
        # Analyze each component
        for component in self.CORE_COMPONENTS.keys():
            print(f"\n{'='*60}")
            print(f"COMPONENT: {component.upper()}")
            print(f"{'='*60}")
            
            # Function coverage analysis
            coverage_data = self.analyze_function_coverage(component)
            self.coverage_data[component] = coverage_data
            
            print(f"\n📈 FUNCTION COVERAGE ANALYSIS")
            print(f"   Source files: {len(coverage_data['source_files'])}")
            print(f"   Test files: {len(coverage_data['test_files'])}")
            print(f"   Total source functions: {coverage_data['total_source_functions']}")
            print(f"   Likely tested functions: {coverage_data['tested_functions']}")
            print(f"   Coverage estimate: {coverage_data['coverage_percentage']:.1f}%")
            
            if coverage_data['likely_untested_functions']:
                print(f"\n   🔍 Potentially untested functions:")
                for func in coverage_data['likely_untested_functions'][:10]:  # Show first 10
                    print(f"      • {func}")
                if len(coverage_data['likely_untested_functions']) > 10:
                    print(f"      ... and {len(coverage_data['likely_untested_functions']) - 10} more")
            
            # Redundancy analysis
            redundancy_data = self.analyze_test_redundancy(component)
            self.redundancy_data[component] = redundancy_data
            
            print(f"\n🔄 TEST REDUNDANCY ANALYSIS")
            print(f"   Total test functions: {redundancy_data['total_test_functions']}")
            print(f"   Unique test names: {redundancy_data['unique_test_names']}")
            print(f"   Redundant test names: {redundancy_data['redundant_test_names']}")
            
            if redundancy_data['redundant_tests']:
                print(f"\n   ⚠️  Redundant test functions:")
                for func_name, data in list(redundancy_data['redundant_tests'].items())[:5]:
                    print(f"      • {func_name} (appears {data['count']} times)")
                    for location in data['locations'][:3]:  # Show first 3 locations
                        print(f"        - {location}")
            
            # Quality analysis
            quality_data = self.analyze_test_quality(component)
            
            print(f"\n🎯 TEST QUALITY ANALYSIS")
            edge_cases = quality_data.get('edge_case_coverage', {})
            print(f"   Files with edge case tests: {len(edge_cases)}")
            
            assertions = quality_data.get('assertion_patterns', {})
            total_assertions = sum(assertions.values())
            print(f"   Total assertions: {total_assertions}")
            
            mocks = quality_data.get('mock_usage', {})
            total_mocks = sum(mocks.values())
            print(f"   Mock usage instances: {total_mocks}")
            
            isolation = quality_data.get('test_isolation', {})
            print(f"   setUp methods: {isolation.get('setUp_methods', 0)}")
            print(f"   IsolatedTestCase usage: {isolation.get('isolated_test_case_usage', 0)}")
        
        # Cross-component analysis
        self._generate_cross_component_analysis()
        
        # Recommendations
        self._generate_recommendations()
        
        # Save detailed analysis
        self._save_analysis_report()
    
    def _generate_cross_component_analysis(self):
        """Generate cross-component analysis."""
        print(f"\n{'='*60}")
        print("CROSS-COMPONENT ANALYSIS")
        print(f"{'='*60}")
        
        # Overall coverage statistics
        total_source_functions = sum(data['total_source_functions'] for data in self.coverage_data.values())
        total_tested_functions = sum(data['tested_functions'] for data in self.coverage_data.values())
        overall_coverage = (total_tested_functions / total_source_functions * 100) if total_source_functions > 0 else 0
        
        print(f"\n📊 OVERALL COVERAGE STATISTICS")
        print(f"   Total source functions: {total_source_functions}")
        print(f"   Total tested functions: {total_tested_functions}")
        print(f"   Overall coverage estimate: {overall_coverage:.1f}%")
        
        # Component coverage comparison
        print(f"\n📈 COMPONENT COVERAGE COMPARISON")
        coverage_by_component = []
        for component, data in self.coverage_data.items():
            coverage_by_component.append((component, data['coverage_percentage']))
        
        coverage_by_component.sort(key=lambda x: x[1], reverse=True)
        
        for component, coverage in coverage_by_component:
            status = "🟢" if coverage >= 80 else "🟡" if coverage >= 60 else "🔴"
            print(f"   {status} {component}: {coverage:.1f}%")
        
        # Redundancy comparison
        print(f"\n🔄 REDUNDANCY COMPARISON")
        for component, data in self.redundancy_data.items():
            redundancy_ratio = (data['redundant_test_names'] / data['unique_test_names'] * 100) if data['unique_test_names'] > 0 else 0
            status = "🔴" if redundancy_ratio > 20 else "🟡" if redundancy_ratio > 10 else "🟢"
            print(f"   {status} {component}: {redundancy_ratio:.1f}% redundancy ({data['redundant_test_names']}/{data['unique_test_names']})")
    
    def _generate_recommendations(self):
        """Generate actionable recommendations."""
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS")
        print(f"{'='*60}")
        
        recommendations = []
        
        # Coverage recommendations
        low_coverage_components = [
            component for component, data in self.coverage_data.items()
            if data['coverage_percentage'] < 70
        ]
        
        if low_coverage_components:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Coverage',
                'action': f"Improve test coverage for: {', '.join(low_coverage_components)}",
                'details': "Add tests for untested functions and edge cases"
            })
        
        # Redundancy recommendations
        high_redundancy_components = [
            component for component, data in self.redundancy_data.items()
            if data['redundant_test_names'] > 5
        ]
        
        if high_redundancy_components:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Redundancy',
                'action': f"Reduce test redundancy in: {', '.join(high_redundancy_components)}",
                'details': "Consolidate duplicate test functions and patterns"
            })
        
        # Test failures from report
        if self.test_report:
            total_failures = self.test_report['summary'].get('total_failures', 0)
            total_errors = self.test_report['summary'].get('total_errors', 0)
            
            if total_failures > 0 or total_errors > 0:
                recommendations.append({
                    'priority': 'CRITICAL',
                    'category': 'Failures',
                    'action': f"Fix {total_failures} failures and {total_errors} errors",
                    'details': "Address failing tests before deployment"
                })
        
        # Print recommendations
        if recommendations:
            recommendations.sort(key=lambda x: {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}[x['priority']])
            
            for i, rec in enumerate(recommendations, 1):
                priority_icon = {'CRITICAL': '🚨', 'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}[rec['priority']]
                print(f"\n{i}. {priority_icon} {rec['priority']} - {rec['category']}")
                print(f"   Action: {rec['action']}")
                print(f"   Details: {rec['details']}")
        else:
            print("\n🎉 All components are in excellent condition!")
    
    def _save_analysis_report(self):
        """Save detailed analysis report to file."""
        analysis_data = {
            'timestamp': self.test_report.get('timestamp') if self.test_report else None,
            'coverage_analysis': self.coverage_data,
            'redundancy_analysis': self.redundancy_data,
            'summary': {
                'total_components': len(self.CORE_COMPONENTS),
                'total_source_functions': sum(data['total_source_functions'] for data in self.coverage_data.values()),
                'total_tested_functions': sum(data['tested_functions'] for data in self.coverage_data.values()),
                'overall_coverage': sum(data['coverage_percentage'] for data in self.coverage_data.values()) / len(self.coverage_data) if self.coverage_data else 0
            }
        }
        
        report_file = self.project_root / 'test' / 'core_coverage_analysis.json'
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2, default=str)
            print(f"\n💾 Detailed analysis saved to: {report_file}")
        except Exception as e:
            print(f"⚠️  Could not save analysis report: {e}")

def main():
    """Main entry point for the coverage analyzer."""
    print("Cursus Core Package - Test Coverage Analysis")
    print("=" * 60)
    
    analyzer = TestCoverageAnalyzer()
    
    # Load test report
    if not analyzer.load_test_report():
        print("⚠️  Continuing without test report data...")
    
    # Generate comprehensive analysis
    analyzer.generate_comprehensive_report()

if __name__ == "__main__":
    main()
