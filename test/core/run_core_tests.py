#!/usr/bin/env python3
"""
Comprehensive Test Runner for Cursus Core Package

This program runs all tests that cover the core package components:
- assembler
- base  
- compiler
- config_fields
- deps

It provides detailed reporting on test results, coverage analysis, and redundancy assessment.
"""

import sys
import os
import time
import json
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
import argparse


@dataclass
class TestResult:
    """Data class to hold test execution results."""

    module: str
    passed: int
    failed: int
    skipped: int
    errors: int
    duration: float
    exit_code: int
    output: str
    error_output: str


@dataclass
class TestSummary:
    """Data class to hold overall test summary."""

    total_tests: int
    total_passed: int
    total_failed: int
    total_skipped: int
    total_errors: int
    total_duration: float
    modules_tested: List[str]
    failed_modules: List[str]
    results: List[TestResult]


class CoreTestRunner:
    """Test runner for all core package tests."""

    def __init__(
        self, verbose: bool = False, coverage: bool = False, parallel: bool = False
    ):
        self.verbose = verbose
        self.coverage = coverage
        self.parallel = parallel
        self.test_root = Path(__file__).parent
        self.project_root = self.test_root.parent.parent

        # Core modules to test
        self.core_modules = ["assembler", "base", "compiler", "config_fields", "deps"]

    def discover_test_files(self, module: str) -> List[Path]:
        """Discover all test files in a module directory."""
        module_path = self.test_root / module
        if not module_path.exists():
            print(f"Warning: Module directory {module_path} does not exist")
            return []

        test_files = []
        for test_file in module_path.rglob("test_*.py"):
            test_files.append(test_file)

        return sorted(test_files)

    def run_pytest_for_module(self, module: str) -> TestResult:
        """Run pytest for a specific module and return results."""
        module_path = self.test_root / module

        if not module_path.exists():
            return TestResult(
                module=module,
                passed=0,
                failed=0,
                skipped=0,
                errors=1,
                duration=0.0,
                exit_code=1,
                output="",
                error_output=f"Module directory {module_path} does not exist",
            )

        # Build pytest command
        cmd = ["python", "-m", "pytest"]

        # Add coverage if requested
        if self.coverage:
            cmd.extend(
                [
                    "--cov=cursus.core",
                    "--cov-report=term-missing",
                    "--cov-report=json:coverage.json",
                ]
            )

        # Add verbosity
        if self.verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")

        # Add parallel execution if requested
        if self.parallel:
            cmd.extend(["-n", "auto"])

        # Don't use JSON report as it requires additional plugin
        json_report = None

        # Add the module path
        cmd.append(str(module_path))

        print(f"\n{'='*60}")
        print(f"Running tests for module: {module}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}")

        start_time = time.time()

        try:
            # Run pytest
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout per module
            )

            duration = time.time() - start_time

            # Parse pytest output to extract test counts
            passed, failed, skipped, errors = self._parse_pytest_output(result.stdout)

            return TestResult(
                module=module,
                passed=passed,
                failed=failed,
                skipped=skipped,
                errors=errors,
                duration=duration,
                exit_code=result.returncode,
                output=result.stdout,
                error_output=result.stderr,
            )

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return TestResult(
                module=module,
                passed=0,
                failed=0,
                skipped=0,
                errors=1,
                duration=duration,
                exit_code=124,  # Timeout exit code
                output="",
                error_output=f"Test execution timed out after 5 minutes",
            )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                module=module,
                passed=0,
                failed=0,
                skipped=0,
                errors=1,
                duration=duration,
                exit_code=1,
                output="",
                error_output=f"Error running tests: {str(e)}",
            )

    def _parse_pytest_output(self, output: str) -> Tuple[int, int, int, int]:
        """Parse pytest output to extract test counts."""
        passed, failed, skipped, errors = 0, 0, 0, 0

        lines = output.split("\n")

        # First, look for the final summary line
        for line in reversed(lines):  # Start from the end to find the final summary
            line = line.strip()
            # Look for summary line with "in X.XXs" at the end
            if (
                (
                    "passed" in line
                    or "failed" in line
                    or "skipped" in line
                    or "error" in line
                )
                and "in " in line
                and "s" in line
            ):
                # Extract numbers followed by keywords
                passed_match = re.search(r"(\d+)\s+passed", line)
                failed_match = re.search(r"(\d+)\s+failed", line)
                skipped_match = re.search(r"(\d+)\s+skipped", line)
                error_match = re.search(r"(\d+)\s+error", line)

                if passed_match:
                    passed = int(passed_match.group(1))
                if failed_match:
                    failed = int(failed_match.group(1))
                if skipped_match:
                    skipped = int(skipped_match.group(1))
                if error_match:
                    errors = int(error_match.group(1))

                # If we found any counts, we're done
                if passed > 0 or failed > 0 or skipped > 0 or errors > 0:
                    break

        # If no summary found, count the test result indicators directly
        if passed == 0 and failed == 0 and skipped == 0 and errors == 0:
            for line in lines:
                # Look for lines with test result indicators and percentage
                # In quiet mode, pytest shows: ....F..s..E.. [ 58%]
                if "[" in line and "%]" in line:
                    # Extract the part before the percentage
                    test_part = line.split("[")[0].strip()
                    if re.match(r"^[\.FsE\s]+$", test_part):
                        passed += test_part.count(".")
                        failed += test_part.count("F")
                        skipped += test_part.count("s")
                        errors += test_part.count("E")

        return passed, failed, skipped, errors

    def run_all_tests(self) -> TestSummary:
        """Run tests for all core modules."""
        print(f"Starting comprehensive test run for cursus.core package")
        print(f"Test root: {self.test_root}")
        print(f"Project root: {self.project_root}")
        print(f"Coverage enabled: {self.coverage}")
        print(f"Parallel execution: {self.parallel}")
        print(f"Verbose output: {self.verbose}")

        results = []
        failed_modules = []

        overall_start = time.time()

        for module in self.core_modules:
            result = self.run_pytest_for_module(module)
            results.append(result)

            if result.exit_code != 0 or result.failed > 0 or result.errors > 0:
                failed_modules.append(module)

            # Print immediate results
            self._print_module_result(result)

        overall_duration = time.time() - overall_start

        # Calculate totals
        total_passed = sum(r.passed for r in results)
        total_failed = sum(r.failed for r in results)
        total_skipped = sum(r.skipped for r in results)
        total_errors = sum(r.errors for r in results)
        total_tests = total_passed + total_failed + total_skipped + total_errors

        summary = TestSummary(
            total_tests=total_tests,
            total_passed=total_passed,
            total_failed=total_failed,
            total_skipped=total_skipped,
            total_errors=total_errors,
            total_duration=overall_duration,
            modules_tested=self.core_modules,
            failed_modules=failed_modules,
            results=results,
        )

        return summary

    def _print_module_result(self, result: TestResult):
        """Print results for a single module."""
        status = (
            "✅ PASSED"
            if result.exit_code == 0 and result.failed == 0 and result.errors == 0
            else "❌ FAILED"
        )

        print(f"\n{result.module}: {status}")
        print(
            f"  Tests: {result.passed + result.failed + result.skipped + result.errors}"
        )
        print(f"  Passed: {result.passed}")
        if result.failed > 0:
            print(f"  Failed: {result.failed}")
        if result.skipped > 0:
            print(f"  Skipped: {result.skipped}")
        if result.errors > 0:
            print(f"  Errors: {result.errors}")
        print(f"  Duration: {result.duration:.2f}s")

        if result.exit_code != 0 and result.error_output:
            print(f"  Error: {result.error_output}")

    def print_summary(self, summary: TestSummary):
        """Print overall test summary."""
        print(f"\n{'='*80}")
        print(f"CORE PACKAGE TEST SUMMARY")
        print(f"{'='*80}")

        print(f"Total Tests: {summary.total_tests}")
        print(f"Passed: {summary.total_passed}")
        print(f"Failed: {summary.total_failed}")
        print(f"Skipped: {summary.total_skipped}")
        print(f"Errors: {summary.total_errors}")
        print(f"Total Duration: {summary.total_duration:.2f}s")

        print(f"\nModules Tested: {len(summary.modules_tested)}")
        for module in summary.modules_tested:
            status = "✅" if module not in summary.failed_modules else "❌"
            print(f"  {status} {module}")

        if summary.failed_modules:
            print(f"\nFailed Modules ({len(summary.failed_modules)}):")
            for module in summary.failed_modules:
                print(f"  ❌ {module}")

        # Overall status
        if summary.total_failed == 0 and summary.total_errors == 0:
            print(f"\n🎉 ALL TESTS PASSED!")
            return 0
        else:
            print(f"\n💥 SOME TESTS FAILED!")
            return 1

    def save_report(
        self, summary: TestSummary, filename: str = "core_test_report.json"
    ):
        """Save test results to JSON file."""
        report_path = self.test_root / filename

        # Convert to dict for JSON serialization
        report_data = asdict(summary)

        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)

        print(f"\nDetailed report saved to: {report_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run all tests for cursus.core package"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "-c", "--coverage", action="store_true", help="Enable coverage reporting"
    )
    parser.add_argument(
        "-p", "--parallel", action="store_true", help="Run tests in parallel"
    )
    parser.add_argument("-m", "--module", help="Run tests for specific module only")
    parser.add_argument(
        "--report", default="core_test_report.json", help="Report filename"
    )

    args = parser.parse_args()

    runner = CoreTestRunner(
        verbose=args.verbose, coverage=args.coverage, parallel=args.parallel
    )

    if args.module:
        # Run single module
        if args.module not in runner.core_modules:
            print(
                f"Error: Unknown module '{args.module}'. Available modules: {runner.core_modules}"
            )
            return 1

        print(f"Running tests for module: {args.module}")
        result = runner.run_pytest_for_module(args.module)
        runner._print_module_result(result)

        return (
            0
            if result.exit_code == 0 and result.failed == 0 and result.errors == 0
            else 1
        )
    else:
        # Run all modules
        summary = runner.run_all_tests()
        exit_code = runner.print_summary(summary)
        runner.save_report(summary, args.report)

        return exit_code


if __name__ == "__main__":
    sys.exit(main())
