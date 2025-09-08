#!/usr/bin/env python3
"""
Comprehensive Test Coverage Analysis for Cursus Core Package

This program analyzes test coverage for the core package components:
- assembler
- base
- compiler
- config_fields (config_field in test directory)
- deps

It provides detailed reporting on test results and coverage analysis.
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
# Note: project_root setup handled by conftest.py

class TestCoverageAnalyzer:
    """Comprehensive test coverage analyzer for cursus core package."""
    
    def __init__(self):
        """Initialize the test coverage analyzer."""
        self.root = Path(__file__).resolve().parent.parent
        self.src_dir = self.root / "src" / "cursus" / "core"
        self.test_dir = self.root / "test" / "core"
        
        # Component mapping
        self.components = {
            "assembler": {
                "source_dir": self.src_dir / "assembler",
                "test_dir": self.test_dir / "assembler"
            },
            "base": {
                "source_dir": self.src_dir / "base", 
                "test_dir": self.test_dir / "base"
            },
            "compiler": {
                "source_dir": self.src_dir / "compiler",
                "test_dir": self.test_dir / "compiler"
            },
            "config_fields": {
                "source_dir": self.src_dir / "config_fields",
                "test_dir": self.test_dir / "config_fields"
            },
            "deps": {
                "source_dir": self.src_dir / "deps",
                "test_dir": self.test_dir / "deps"
            }
        }
        
        self.coverage_data = {}
    
    def extract_functions_from_file(self, file_path: Path) -> List[str]:
        """Extract function and method names from a Python file."""
        functions = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    class_name = node.name
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            functions.append(f"{class_name}.{item.name}")
                            functions.append(item.name)  # Also add standalone method name
                            
        except Exception as e:
            print(f"Warning: Could not parse {file_path}: {e}")
            
        return functions
    
    def analyze_component_coverage(self, component_name: str) -> Dict:
        """Analyze test coverage for a specific component."""
        component_info = self.components[component_name]
        source_dir = component_info["source_dir"]
        test_dir = component_info["test_dir"]
        
        # Get source files and functions
        source_files = []
        source_functions_by_file = {}
        all_source_functions = []
        
        if source_dir.exists():
            for py_file in source_dir.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue
                    
                rel_path = f"src/cursus/core/{component_name}/{py_file.name}"
                source_files.append(rel_path)
                
                functions = self.extract_functions_from_file(py_file)
                source_functions_by_file[rel_path] = functions
                all_source_functions.extend(functions)
        
        # Get test files and functions
        test_files = []
        test_functions_by_file = {}
        all_test_functions = []
        
        if test_dir.exists():
            for py_file in test_dir.glob("test_*.py"):
                rel_path = f"test/core/{component_name}/{py_file.name}"
                test_files.append(rel_path)
                
                functions = self.extract_functions_from_file(py_file)
                test_functions_by_file[rel_path] = functions
                all_test_functions.extend(functions)
        
        # Determine likely tested functions based on naming patterns
        likely_tested = []
        likely_untested = []
        
        for func in all_source_functions:
            # Simple heuristic: if there's a test function that might test this function
            is_likely_tested = False
            
            # Check for direct test name matches
            func_base = func.split('.')[-1]  # Get method name without class
            for test_func in all_test_functions:
                if func_base.lower() in test_func.lower() or func.lower() in test_func.lower():
                    is_likely_tested = True
                    break
            
            # Check for class-based testing patterns
            if not is_likely_tested and '.' in func:
                class_name = func.split('.')[0]
                for test_func in all_test_functions:
                    if class_name.lower() in test_func.lower():
                        is_likely_tested = True
                        break
            
            if is_likely_tested:
                likely_tested.append(func)
            else:
                likely_untested.append(func)
        
        # Calculate coverage metrics
        total_functions = len(all_source_functions)
        tested_functions = len(likely_tested)
        untested_functions = len(likely_untested)
        coverage_percentage = (tested_functions / total_functions * 100) if total_functions > 0 else 0
        
        return {
            "component": component_name,
            "source_files": source_files,
            "test_files": test_files,
            "total_source_functions": total_functions,
            "tested_functions": tested_functions,
            "untested_functions": untested_functions,
            "coverage_percentage": coverage_percentage,
            "source_functions_by_file": source_functions_by_file,
            "test_functions_by_file": test_functions_by_file,
            "likely_tested_functions": likely_tested,
            "likely_untested_functions": likely_untested
        }
    
    def get_test_count(self, component_name: str) -> int:
        """Get total number of test functions for a component."""
        component_info = self.components[component_name]
        test_dir = component_info["test_dir"]
        
        total_test_functions = 0
        
        if test_dir.exists():
            for py_file in test_dir.glob("test_*.py"):
                functions = self.extract_functions_from_file(py_file)
                total_test_functions += len(functions)
        
        return total_test_functions
    
    def run_full_analysis(self) -> Dict:
        """Run complete coverage analysis for all components."""
        print("🚀 Starting Comprehensive Test Coverage Analysis")
        print(f"📁 Source Directory: {self.src_dir}")
        print(f"📁 Test Directory: {self.test_dir}")
        
        coverage_analysis = {}
        
        # Analyze each component
        for component_name in self.components.keys():
            print(f"\n🔍 Analyzing component: {component_name}")
            
            # Coverage analysis
            coverage_data = self.analyze_component_coverage(component_name)
            coverage_analysis[component_name] = coverage_data
            
            print(f"   📊 Coverage: {coverage_data['coverage_percentage']:.1f}% "
                  f"({coverage_data['tested_functions']}/{coverage_data['total_source_functions']} functions)")
            
            # Test count
            test_count = self.get_test_count(component_name)
            print(f"   🧪 Test Functions: {test_count}")
        
        # Calculate overall summary
        total_functions = sum(comp['total_source_functions'] for comp in coverage_analysis.values())
        total_tested = sum(comp['tested_functions'] for comp in coverage_analysis.values())
        overall_coverage = (total_tested / total_functions * 100) if total_functions > 0 else 0
        
        summary = {
            "total_components": len(self.components),
            "total_source_functions": total_functions,
            "total_tested_functions": total_tested,
            "overall_coverage": overall_coverage
        }
        
        return {
            "timestamp": self._get_timestamp(),
            "coverage_analysis": coverage_analysis,
            "summary": summary
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def save_analysis_results(self, results: Dict, output_file: str = "core_coverage_analysis.json"):
        """Save analysis results to JSON file."""
        output_path = self.root / "test" / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Analysis results saved to: {output_path}")
        return output_path
    
    def print_summary_report(self, results: Dict):
        """Print a summary report of the analysis results."""
        print(f"\n{'='*80}")
        print("📊 TEST COVERAGE ANALYSIS SUMMARY")
        print(f"{'='*80}")
        
        summary = results['summary']
        print(f"📈 Overall Coverage: {summary['overall_coverage']:.1f}% "
              f"({summary['total_tested_functions']}/{summary['total_source_functions']} functions)")
        print(f"🧩 Components Analyzed: {summary['total_components']}")
        
        print(f"\n{'Component':<15} {'Coverage':<12} {'Functions':<12} {'Test Count':<12}")
        print("-" * 60)
        
        coverage_data = results['coverage_analysis']
        
        for component in coverage_data.keys():
            cov = coverage_data[component]
            
            coverage_pct = f"{cov['coverage_percentage']:.1f}%"
            functions_str = f"{cov['tested_functions']}/{cov['total_source_functions']}"
            test_count = len(cov['test_functions_by_file'].get(list(cov['test_functions_by_file'].keys())[0], [])) if cov['test_functions_by_file'] else 0
            test_count_str = f"{test_count} tests"
            
            print(f"{component:<15} {coverage_pct:<12} {functions_str:<12} {test_count_str:<12}")
        
        print(f"\n{'='*80}")
        
        # Highlight critical issues
        print("🚨 CRITICAL ISSUES:")
        for component, data in coverage_data.items():
            if data['coverage_percentage'] < 50:
                print(f"   ❌ {component}: Low coverage ({data['coverage_percentage']:.1f}%)")
        
        print(f"\n📁 Detailed results saved in JSON format")
        print(f" Run with --component <name> for detailed component analysis")


def main():
    """Main entry point for the test coverage analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze test coverage for cursus core package')
    parser.add_argument('--component', help='Analyze specific component only')
    parser.add_argument('--output', default='core_coverage_analysis.json',
                       help='Output file name (default: core_coverage_analysis.json)')
    
    args = parser.parse_args()
    
    try:
        analyzer = TestCoverageAnalyzer()
        
        if args.component:
            # Analyze single component
            if args.component not in analyzer.components:
                print(f"❌ Unknown component: {args.component}")
                print(f"Available components: {', '.join(analyzer.components.keys())}")
                sys.exit(1)
            
            print(f"🔍 Analyzing component: {args.component}")
            coverage_data = analyzer.analyze_component_coverage(args.component)
            test_count = analyzer.get_test_count(args.component)
            
            # Print detailed results for single component
            print(f"\n📊 COVERAGE ANALYSIS - {args.component.upper()}")
            print(f"   Total Functions: {coverage_data['total_source_functions']}")
            print(f"   Tested Functions: {coverage_data['tested_functions']}")
            print(f"   Coverage: {coverage_data['coverage_percentage']:.1f}%")
            print(f"   Test Functions: {test_count}")
            
            if coverage_data['likely_untested_functions']:
                print(f"\n📋 LIKELY UNTESTED FUNCTIONS:")
                for func in coverage_data['likely_untested_functions']:
                    print(f"   • {func}")
        else:
            # Run full analysis
            results = analyzer.run_full_analysis()
            analyzer.print_summary_report(results)
            analyzer.save_analysis_results(results, args.output)
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Fatal error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
