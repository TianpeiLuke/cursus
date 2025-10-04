#!/usr/bin/env python3
"""
Comprehensive Test Coverage Analysis for Cursus Package

This program provides comprehensive test coverage analysis for the entire cursus package,
including all components: api, cli, core, mods, pipeline_catalog, processing, registry,
step_catalog, steps, validation, and workspace.

Features:
- Full package coverage analysis using pytest-cov
- Component-by-component breakdown
- Function-level coverage analysis
- HTML and JSON reporting
- Integration with existing test infrastructure
- Performance metrics and timing analysis
- Coverage trend tracking
"""

import json
import sys
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, Counter
import ast
import re
from datetime import datetime

# Add the project root to the Python path
# Note: project_root setup handled by conftest.py


class TestCoverageAnalyzer:
    """Comprehensive test coverage analyzer for the entire cursus package."""

    def __init__(self):
        """Initialize the test coverage analyzer."""
        self.root = Path(__file__).resolve().parent.parent
        self.src_dir = self.root / "src" / "cursus"
        self.test_dir = self.root / "test"

        # All cursus package components
        self.components = {
            "api": {
                "source_dir": self.src_dir / "api",
                "test_dir": self.test_dir / "api",
                "description": "DAG and pipeline API components"
            },
            "cli": {
                "source_dir": self.src_dir / "cli",
                "test_dir": self.test_dir / "cli",
                "description": "Command-line interface tools"
            },
            "core": {
                "source_dir": self.src_dir / "core",
                "test_dir": self.test_dir / "core",
                "description": "Core framework components (assembler, base, compiler, config_fields, deps)"
            },
            "mods": {
                "source_dir": self.src_dir / "mods",
                "test_dir": self.test_dir / "mods",
                "description": "MODS integration and execution documents"
            },
            "pipeline_catalog": {
                "source_dir": self.src_dir / "pipeline_catalog",
                "test_dir": self.test_dir / "pipeline_catalog",
                "description": "Pipeline catalog and registry management"
            },
            "processing": {
                "source_dir": self.src_dir / "processing",
                "test_dir": self.test_dir / "processing",
                "description": "Data processing and transformation components"
            },
            "registry": {
                "source_dir": self.src_dir / "registry",
                "test_dir": self.test_dir / "registry",
                "description": "Step and hyperparameter registry management"
            },
            "step_catalog": {
                "source_dir": self.src_dir / "step_catalog",
                "test_dir": self.test_dir / "step_catalog",
                "description": "Step catalog discovery and management"
            },
            "steps": {
                "source_dir": self.src_dir / "steps",
                "test_dir": self.test_dir / "steps",
                "description": "Pipeline step builders, configs, and specifications"
            },
            "validation": {
                "source_dir": self.src_dir / "validation",
                "test_dir": self.test_dir / "validation",
                "description": "Validation framework and alignment testing"
            },
            "workspace": {
                "source_dir": self.src_dir / "workspace",
                "test_dir": self.test_dir / "workspace",
                "description": "Workspace management and integration"
            }
        }

        self.coverage_data = {}
        self.start_time = None
        self.end_time = None

    def extract_functions_from_file(self, file_path: Path) -> List[str]:
        """Extract function and method names from a Python file."""
        functions = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
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
                            functions.append(
                                item.name
                            )  # Also add standalone method name

        except Exception as e:
            print(f"Warning: Could not parse {file_path}: {e}")

        return functions

    def run_pytest_coverage(self, component: Optional[str] = None, 
                           html_report: bool = True, 
                           fail_under: Optional[float] = None) -> Dict:
        """Run pytest with coverage analysis."""
        print("🚀 Running pytest with coverage analysis...")
        self.start_time = time.time()
        
        # Base pytest command with coverage
        cmd = [
            "python", "-m", "pytest",
            "--cov=src/cursus",
            "--cov-report=json:test/coverage.json",
            "--cov-report=term-missing",
            "-v"
        ]
        
        # Add HTML report if requested
        if html_report:
            cmd.extend(["--cov-report=html:htmlcov"])
            
        # Add fail-under threshold if specified
        if fail_under:
            cmd.extend([f"--cov-fail-under={fail_under}"])
            
        # Target specific component if specified
        if component and component in self.components:
            test_path = self.components[component]["test_dir"]
            if test_path.exists():
                cmd.append(str(test_path))
            else:
                print(f"⚠️  Test directory not found for {component}: {test_path}")
                return {}
        else:
            # Run all tests
            cmd.append("test/")
            
        # Change to project root directory
        original_cwd = os.getcwd()
        os.chdir(self.root)
        
        try:
            print(f"📋 Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            self.end_time = time.time()
            execution_time = self.end_time - self.start_time
            
            # Parse results
            coverage_data = {
                "command": " ".join(cmd),
                "execution_time": execution_time,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }
            
            # Load JSON coverage report if available
            coverage_json_path = self.root / "test" / "coverage.json"
            if coverage_json_path.exists():
                with open(coverage_json_path, 'r') as f:
                    coverage_data["detailed_coverage"] = json.load(f)
                    
            return coverage_data
            
        except subprocess.TimeoutExpired:
            print("❌ Test execution timed out after 10 minutes")
            return {"error": "timeout", "execution_time": 600}
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return {"error": str(e)}
        finally:
            os.chdir(original_cwd)

    def analyze_component_coverage(self, component_name: str) -> Dict:
        """Analyze test coverage for a specific component."""
        component_info = self.components[component_name]
        source_dir = component_info["source_dir"]
        test_dir = component_info["test_dir"]

        # Get source files and functions
        source_files = []
        source_functions_by_file = {}
        all_source_functions = []
        total_source_lines = 0

        if source_dir.exists():
            for py_file in source_dir.rglob("*.py"):
                if py_file.name.startswith("__"):
                    continue

                rel_path = py_file.relative_to(self.src_dir)
                source_files.append(str(rel_path))

                # Count lines
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        lines = len(f.readlines())
                        total_source_lines += lines
                except:
                    pass

                functions = self.extract_functions_from_file(py_file)
                source_functions_by_file[str(rel_path)] = functions
                all_source_functions.extend(functions)

        # Get test files and functions
        test_files = []
        test_functions_by_file = {}
        all_test_functions = []
        total_test_lines = 0

        if test_dir.exists():
            for py_file in test_dir.rglob("test_*.py"):
                rel_path = py_file.relative_to(self.test_dir)
                test_files.append(str(rel_path))

                # Count lines
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        lines = len(f.readlines())
                        total_test_lines += lines
                except:
                    pass

                functions = self.extract_functions_from_file(py_file)
                test_functions_by_file[str(rel_path)] = functions
                all_test_functions.extend(functions)

        # Determine likely tested functions based on naming patterns
        likely_tested = []
        likely_untested = []

        for func in all_source_functions:
            # Simple heuristic: if there's a test function that might test this function
            is_likely_tested = False

            # Check for direct test name matches
            func_base = func.split(".")[-1]  # Get method name without class
            for test_func in all_test_functions:
                if (
                    func_base.lower() in test_func.lower()
                    or func.lower() in test_func.lower()
                ):
                    is_likely_tested = True
                    break

            # Check for class-based testing patterns
            if not is_likely_tested and "." in func:
                class_name = func.split(".")[0]
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
        coverage_percentage = (
            (tested_functions / total_functions * 100) if total_functions > 0 else 0
        )

        return {
            "component": component_name,
            "description": component_info["description"],
            "source_exists": source_dir.exists(),
            "test_exists": test_dir.exists(),
            "source_files": source_files,
            "test_files": test_files,
            "total_source_functions": total_functions,
            "tested_functions": tested_functions,
            "untested_functions": untested_functions,
            "coverage_percentage": coverage_percentage,
            "total_source_lines": total_source_lines,
            "total_test_lines": total_test_lines,
            "test_to_source_ratio": len(all_test_functions) / max(total_functions, 1),
            "source_functions_by_file": source_functions_by_file,
            "test_functions_by_file": test_functions_by_file,
            "likely_tested_functions": likely_tested,
            "likely_untested_functions": likely_untested,
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

            print(
                f"   📊 Coverage: {coverage_data['coverage_percentage']:.1f}% "
                f"({coverage_data['tested_functions']}/{coverage_data['total_source_functions']} functions)"
            )

            # Test count
            test_count = self.get_test_count(component_name)
            print(f"   🧪 Test Functions: {test_count}")

        # Calculate overall summary
        total_functions = sum(
            comp["total_source_functions"] for comp in coverage_analysis.values()
        )
        total_tested = sum(
            comp["tested_functions"] for comp in coverage_analysis.values()
        )
        overall_coverage = (
            (total_tested / total_functions * 100) if total_functions > 0 else 0
        )

        summary = {
            "total_components": len(self.components),
            "total_source_functions": total_functions,
            "total_tested_functions": total_tested,
            "overall_coverage": overall_coverage,
        }

        return {
            "timestamp": self._get_timestamp(),
            "coverage_analysis": coverage_analysis,
            "summary": summary,
        }

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def save_analysis_results(
        self, results: Dict, output_file: str = "core_coverage_analysis.json"
    ):
        """Save analysis results to JSON file."""
        output_path = self.root / "test" / output_file

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n📄 Analysis results saved to: {output_path}")
        return output_path

    def print_summary_report(self, results: Dict):
        """Print a summary report of the analysis results."""
        print(f"\n{'='*80}")
        print("📊 TEST COVERAGE ANALYSIS SUMMARY")
        print(f"{'='*80}")

        summary = results["summary"]
        print(
            f"📈 Overall Coverage: {summary['overall_coverage']:.1f}% "
            f"({summary['total_tested_functions']}/{summary['total_source_functions']} functions)"
        )
        print(f"🧩 Components Analyzed: {summary['total_components']}")

        print(
            f"\n{'Component':<15} {'Coverage':<12} {'Functions':<12} {'Test Count':<12}"
        )
        print("-" * 60)

        coverage_data = results["coverage_analysis"]

        for component in coverage_data.keys():
            cov = coverage_data[component]

            coverage_pct = f"{cov['coverage_percentage']:.1f}%"
            functions_str = f"{cov['tested_functions']}/{cov['total_source_functions']}"
            test_count = (
                len(
                    cov["test_functions_by_file"].get(
                        list(cov["test_functions_by_file"].keys())[0], []
                    )
                )
                if cov["test_functions_by_file"]
                else 0
            )
            test_count_str = f"{test_count} tests"

            print(
                f"{component:<15} {coverage_pct:<12} {functions_str:<12} {test_count_str:<12}"
            )

        print(f"\n{'='*80}")

        # Highlight critical issues
        print("🚨 CRITICAL ISSUES:")
        for component, data in coverage_data.items():
            if data["coverage_percentage"] < 50:
                print(
                    f"   ❌ {component}: Low coverage ({data['coverage_percentage']:.1f}%)"
                )

        print(f"\n📁 Detailed results saved in JSON format")
        print(f" Run with --component <name> for detailed component analysis")


def main():
    """Main entry point for the test coverage analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive test coverage analysis for cursus package",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_test_coverage.py                    # Full analysis of all components
  python analyze_test_coverage.py --component core   # Analyze only core component
  python analyze_test_coverage.py --pytest          # Run pytest with coverage
  python analyze_test_coverage.py --pytest --html   # Run pytest with HTML report
  python analyze_test_coverage.py --list-components  # List all available components
        """
    )
    
    parser.add_argument("--component", help="Analyze specific component only")
    parser.add_argument(
        "--output",
        default="comprehensive_coverage_analysis.json",
        help="Output file name (default: comprehensive_coverage_analysis.json)",
    )
    parser.add_argument(
        "--pytest", 
        action="store_true",
        help="Run pytest with coverage analysis"
    )
    parser.add_argument(
        "--html", 
        action="store_true",
        help="Generate HTML coverage report (requires --pytest)"
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        help="Fail if coverage is under this percentage"
    )
    parser.add_argument(
        "--list-components",
        action="store_true",
        help="List all available components and exit"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    try:
        analyzer = TestCoverageAnalyzer()

        # List components and exit
        if args.list_components:
            print("📋 Available Components:")
            print("=" * 50)
            for name, info in analyzer.components.items():
                status = "✅" if info["source_dir"].exists() else "❌"
                test_status = "🧪" if info["test_dir"].exists() else "❌"
                print(f"{status} {name:<15} {test_status} {info['description']}")
            print("\nLegend: ✅ = Source exists, 🧪 = Tests exist, ❌ = Missing")
            return

        # Run pytest with coverage
        if args.pytest:
            print("🚀 Running pytest with coverage analysis...")
            pytest_results = analyzer.run_pytest_coverage(
                component=args.component,
                html_report=args.html,
                fail_under=args.fail_under
            )
            
            if pytest_results.get("success"):
                print("✅ Pytest execution completed successfully")
                if args.html:
                    print("📊 HTML coverage report generated in htmlcov/")
            else:
                print("❌ Pytest execution failed")
                if args.verbose:
                    print("STDOUT:", pytest_results.get("stdout", ""))
                    print("STDERR:", pytest_results.get("stderr", ""))
                    
            # Save pytest results
            pytest_output = args.output.replace(".json", "_pytest.json")
            analyzer.save_analysis_results(pytest_results, pytest_output)
            
            # Also run function-level analysis
            print("\n" + "="*60)
            print("Running function-level analysis...")

        if args.component:
            # Analyze single component
            if args.component not in analyzer.components:
                print(f"❌ Unknown component: {args.component}")
                print(f"Available components: {', '.join(analyzer.components.keys())}")
                sys.exit(1)

            print(f"🔍 Analyzing component: {args.component}")
            coverage_data = analyzer.analyze_component_coverage(args.component)

            # Print detailed results for single component
            print(f"\n📊 COVERAGE ANALYSIS - {args.component.upper()}")
            print(f"   Description: {coverage_data['description']}")
            print(f"   Source exists: {'✅' if coverage_data['source_exists'] else '❌'}")
            print(f"   Tests exist: {'✅' if coverage_data['test_exists'] else '❌'}")
            print(f"   Total Functions: {coverage_data['total_source_functions']}")
            print(f"   Tested Functions: {coverage_data['tested_functions']}")
            print(f"   Coverage: {coverage_data['coverage_percentage']:.1f}%")
            print(f"   Source Lines: {coverage_data['total_source_lines']}")
            print(f"   Test Lines: {coverage_data['total_test_lines']}")
            print(f"   Test/Source Ratio: {coverage_data['test_to_source_ratio']:.2f}")

            if coverage_data["likely_untested_functions"] and args.verbose:
                print(f"\n📋 LIKELY UNTESTED FUNCTIONS:")
                for func in coverage_data["likely_untested_functions"]:
                    print(f"   • {func}")
                    
            # Save single component results
            single_results = {
                "timestamp": analyzer._get_timestamp(),
                "component_analysis": coverage_data,
                "summary": {
                    "component": args.component,
                    "coverage_percentage": coverage_data['coverage_percentage'],
                    "total_functions": coverage_data['total_source_functions'],
                    "tested_functions": coverage_data['tested_functions']
                }
            }
            analyzer.save_analysis_results(single_results, args.output)
        else:
            # Run full analysis
            results = analyzer.run_full_analysis()
            analyzer.print_summary_report(results)
            analyzer.save_analysis_results(results, args.output)
            
            # Print recommendations
            print(f"\n💡 RECOMMENDATIONS:")
            coverage_data = results["coverage_analysis"]
            
            # Find components with no tests
            no_tests = [name for name, data in coverage_data.items() 
                       if not data["test_exists"]]
            if no_tests:
                print(f"   🚨 Components without tests: {', '.join(no_tests)}")
                
            # Find components with low coverage
            low_coverage = [name for name, data in coverage_data.items() 
                           if data["coverage_percentage"] < 30 and data["source_exists"]]
            if low_coverage:
                print(f"   ⚠️  Low coverage components: {', '.join(low_coverage)}")
                
            # Find components with good coverage
            good_coverage = [name for name, data in coverage_data.items() 
                            if data["coverage_percentage"] > 70]
            if good_coverage:
                print(f"   ✅ Well-tested components: {', '.join(good_coverage)}")

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
