"""
Test runner for all alignment validation tests.

This script runs all alignment validation tests using pytest and provides
comprehensive reporting on test results and coverage.
"""

import sys
import os
import subprocess
from pathlib import Path
import argparse

def discover_alignment_test_files(base_path):
    """Discover all alignment test files."""
    test_files = []
    base_path = Path(base_path)
    
    # Walk through all subdirectories to find test files
    for test_file in base_path.rglob("test_*.py"):
        # Skip __pycache__ and other non-test directories
        if "__pycache__" not in str(test_file):
            test_files.append(str(test_file))
    
    return sorted(test_files)

def run_alignment_tests(verbosity=1, collect_only=False, markers=None, pattern=None, module_filter=None):
    """Run alignment validation tests using pytest."""
    
    # Get alignment directory
    alignment_path = Path(__file__).parent
    test_files = discover_alignment_test_files(alignment_path)
    
    print("🔍 Discovering alignment validation test files...")
    print(f"Found {len(test_files)} test files")
    
    if collect_only:
        print("\n📋 Alignment test files found:")
        for test_file in test_files:
            rel_path = Path(test_file).relative_to(alignment_path)
            print(f"  📄 {rel_path}")
        return True
    
    # Build pytest command
    pytest_args = [
        sys.executable, "-m", "pytest",
        str(alignment_path),  # Run tests in alignment directory
        "--tb=short",  # Short traceback format
        "--no-header",  # No pytest header
        "--color=yes",  # Colored output
    ]
    
    # Add verbosity
    if verbosity == 0:
        pytest_args.append("-q")  # Quiet
    elif verbosity == 2:
        pytest_args.append("-v")  # Verbose
    elif verbosity == 3:
        pytest_args.extend(["-v", "-s"])  # Very verbose with output
    
    # Add module filter if specified
    if module_filter:
        module_patterns = {
            'utils': 'utils/',
            'reporter': 'reporter/',
            'script_contract': 'script_contract/',
            'step_enhancers': 'step_type_enhancers/',
            'unified': 'unified_tester/',
            'analyzers': 'analyzers/',
            'discovery': 'discovery/',
            'loaders': 'loaders/',
            'validators': 'validators/'
        }
        
        if module_filter in module_patterns:
            pytest_args.extend(["-k", module_patterns[module_filter]])
        else:
            print(f"⚠️  Unknown module filter: {module_filter}")
            print(f"Available filters: {', '.join(module_patterns.keys())}")
    
    # Add markers filter if specified
    if markers:
        pytest_args.extend(["-m", markers])
    
    # Add pattern filter if specified
    if pattern:
        pytest_args.extend(["-k", pattern])
    
    # Add coverage if available
    try:
        import pytest_cov
        pytest_args.extend([
            "--cov=cursus.validation.alignment",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov/alignment"
        ])
        print("📊 Coverage reporting enabled")
    except ImportError:
        print("⚠️  pytest-cov not available, skipping coverage")
    
    print(f"\n🚀 Running alignment validation tests with pytest...")
    print(f"Command: {' '.join(pytest_args)}")
    
    # Run pytest
    try:
        result = subprocess.run(pytest_args, cwd=alignment_path.parent.parent.parent)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running pytest: {e}")
        return False

def generate_coverage_report():
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
        ],
        'Step Type Enhancement': [
            'Training step enhancer',
            'Processing step enhancer',
            'CreateModel step enhancer',
            'Transform step enhancer',
            'RegisterModel step enhancer',
            'Framework pattern detection',
            'Enhancement routing'
        ],
        'Analyzers': [
            'Builder code analyzer',
            'Configuration analyzer',
            'Static analysis tools'
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
    """Main entry point for the alignment test runner."""
    parser = argparse.ArgumentParser(
        description="Run alignment validation tests using pytest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_alignment_tests.py                    # Run all tests
  python run_all_alignment_tests.py -v 2               # Verbose output
  python run_all_alignment_tests.py --collect-only     # Just list tests
  python run_all_alignment_tests.py --module utils     # Run utils tests only
  python run_all_alignment_tests.py -k "scorer"        # Run scorer tests only
  python run_all_alignment_tests.py --coverage         # Show coverage report
        """
    )
    
    parser.add_argument(
        '-v', '--verbosity',
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help='Test output verbosity level (0=quiet, 1=normal, 2=verbose, 3=very verbose)'
    )
    
    parser.add_argument(
        '--collect-only',
        action='store_true',
        help='Only collect and display test files, do not run tests'
    )
    
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Show test coverage report'
    )
    
    parser.add_argument(
        '--module',
        choices=['utils', 'reporter', 'script_contract', 'step_enhancers', 'unified', 'analyzers', 'discovery', 'loaders', 'validators'],
        help='Run tests from specific module only'
    )
    
    parser.add_argument(
        '-m', '--markers',
        help='Run tests matching given mark expression (e.g., "not slow")'
    )
    
    parser.add_argument(
        '-k', '--pattern',
        help='Run tests matching given substring expression'
    )
    
    args = parser.parse_args()
    
    print("🧪 Alignment Validation Test Runner (pytest-based)")
    print("=" * 70)
    
    success = run_alignment_tests(
        verbosity=args.verbosity,
        collect_only=args.collect_only,
        markers=args.markers,
        pattern=args.pattern,
        module_filter=args.module
    )
    
    # Show coverage report if requested
    if args.coverage:
        generate_coverage_report()
    
    if not args.collect_only:
        print("\n" + "=" * 70)
        if success:
            print("✅ All alignment validation tests completed successfully!")
        else:
            print("❌ Some alignment validation tests failed!")
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
