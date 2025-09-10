"""
Comprehensive test runner for all validation tests.

This script discovers and runs all test files in the validation directory
and its subdirectories using pytest, providing a complete summary.
"""

import sys
import os
import subprocess
from pathlib import Path
import argparse

def discover_test_files(base_path):
    """Discover all test files in the validation directory tree."""
    test_files = []
    base_path = Path(base_path)
    
    # Walk through all subdirectories to find test files
    for test_file in base_path.rglob("test_*.py"):
        # Skip __pycache__ and other non-test directories
        if "__pycache__" not in str(test_file):
            test_files.append(str(test_file))
    
    return sorted(test_files)

def run_all_validation_tests(verbosity=1, collect_only=False, markers=None, pattern=None):
    """Run all validation test files using pytest."""
    
    # Discover all test files
    validation_path = Path(__file__).parent
    test_files = discover_test_files(validation_path)
    
    print("🔍 Discovering validation test files...")
    print(f"Found {len(test_files)} test files")
    
    if collect_only:
        print("\n📋 Test files found:")
        for test_file in test_files:
            rel_path = Path(test_file).relative_to(validation_path)
            print(f"  📄 {rel_path}")
        return True
    
    # Build pytest command
    pytest_args = [
        sys.executable, "-m", "pytest",
        str(validation_path),  # Run tests in validation directory
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
            "--cov=cursus.validation",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov/validation"
        ])
        print("📊 Coverage reporting enabled")
    except ImportError:
        print("⚠️  pytest-cov not available, skipping coverage")
    
    print(f"\n🚀 Running validation tests with pytest...")
    print(f"Command: {' '.join(pytest_args)}")
    
    # Run pytest
    try:
        result = subprocess.run(pytest_args, cwd=validation_path.parent.parent)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running pytest: {e}")
        return False

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run all validation tests using pytest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_validation_tests.py                    # Run all tests
  python run_all_validation_tests.py -v 2               # Verbose output
  python run_all_validation_tests.py --collect-only     # Just list tests
  python run_all_validation_tests.py -m "not slow"      # Skip slow tests
  python run_all_validation_tests.py -k "alignment"     # Run alignment tests only
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
        '-m', '--markers',
        help='Run tests matching given mark expression (e.g., "not slow")'
    )
    
    parser.add_argument(
        '-k', '--pattern',
        help='Run tests matching given substring expression'
    )
    
    args = parser.parse_args()
    
    print("🧪 Validation Test Runner (pytest-based)")
    print("=" * 60)
    
    success = run_all_validation_tests(
        verbosity=args.verbosity,
        collect_only=args.collect_only,
        markers=args.markers,
        pattern=args.pattern
    )
    
    if not args.collect_only:
        print("\n" + "=" * 60)
        if success:
            print("✅ All validation tests completed successfully!")
        else:
            print("❌ Some validation tests failed!")
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
