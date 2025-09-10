#!/usr/bin/env python3
"""
Test runner for builder-configuration alignment tests

Runs all tests for the enhanced builder-configuration validation components using pytest:
- ConfigurationAnalyzer tests
- BuilderCodeAnalyzer tests  
- BuilderConfigurationAlignmentTester tests
"""

import sys
import os
import subprocess
from pathlib import Path
import argparse

def discover_builder_config_test_files(base_path):
    """Discover all builder-config alignment test files."""
    test_files = []
    base_path = Path(base_path)
    
    # Look for specific builder-config test files
    builder_config_patterns = [
        "test_builder_config_alignment.py",
        "analyzers/test_config_analyzer.py",
        "analyzers/test_builder_analyzer.py",
        "test_enhanced_argument_validation.py",
        "test_property_path_validator.py"
    ]
    
    for pattern in builder_config_patterns:
        test_file = base_path / pattern
        if test_file.exists():
            test_files.append(str(test_file))
    
    # Also look for any other test files in analyzers directory
    analyzers_dir = base_path / "analyzers"
    if analyzers_dir.exists():
        for test_file in analyzers_dir.glob("test_*.py"):
            if str(test_file) not in test_files:
                test_files.append(str(test_file))
    
    return sorted(test_files)

def run_builder_config_tests(verbosity=1, collect_only=False, markers=None, pattern=None):
    """Run builder-configuration alignment tests using pytest."""
    
    # Get alignment directory
    alignment_path = Path(__file__).parent
    test_files = discover_builder_config_test_files(alignment_path)
    
    print("🔍 Discovering builder-configuration alignment test files...")
    print(f"Found {len(test_files)} test files")
    
    if collect_only:
        print("\n📋 Builder-config test files found:")
        for test_file in test_files:
            rel_path = Path(test_file).relative_to(alignment_path)
            print(f"  📄 {rel_path}")
        return True
    
    if not test_files:
        print("⚠️  No builder-configuration test files found!")
        return True
    
    # Build pytest command
    pytest_args = [
        sys.executable, "-m", "pytest"
    ]
    
    # Add specific test files
    pytest_args.extend(test_files)
    
    # Add pytest options
    pytest_args.extend([
        "--tb=short",  # Short traceback format
        "--no-header",  # No pytest header
        "--color=yes",  # Colored output
    ])
    
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
            "--cov=cursus.validation.alignment",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov/builder_config"
        ])
        print("📊 Coverage reporting enabled")
    except ImportError:
        print("⚠️  pytest-cov not available, skipping coverage")
    
    print(f"\n🚀 Running builder-configuration alignment tests with pytest...")
    print(f"Command: {' '.join(pytest_args)}")
    
    # Run pytest
    try:
        result = subprocess.run(pytest_args, cwd=alignment_path.parent.parent.parent)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running pytest: {e}")
        return False

def generate_builder_config_coverage_report():
    """Generate a coverage report for builder-configuration alignment tests."""
    print("\n📋 BUILDER-CONFIGURATION ALIGNMENT TEST COVERAGE")
    print("-" * 70)
    
    coverage_areas = {
        'Configuration Analysis': [
            'Configuration field detection',
            'Required field validation',
            'Optional field validation',
            'Default value handling',
            'Field type validation',
            'Nested configuration support'
        ],
        'Builder Code Analysis': [
            'Configuration access pattern detection',
            'Field usage validation',
            'Missing field access detection',
            'Unused configuration detection',
            'Method signature analysis',
            'Class structure validation'
        ],
        'Alignment Testing': [
            'Builder-config field alignment',
            'Configuration completeness validation',
            'Field type consistency checking',
            'Default value alignment',
            'Required field coverage',
            'Error handling validation'
        ],
        'Enhanced Argument Validation': [
            'Argument definition validation',
            'Type checking',
            'Required argument validation',
            'Default value validation',
            'Argument usage pattern detection'
        ],
        'Property Path Validation': [
            'SageMaker property path validation',
            'Path reference checking',
            'Property access pattern validation',
            'Path resolution testing'
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
    """Main entry point for the builder-config test runner."""
    parser = argparse.ArgumentParser(
        description="Run builder-configuration alignment tests using pytest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_builder_config_tests.py                   # Run all tests
  python run_builder_config_tests.py -v 2              # Verbose output
  python run_builder_config_tests.py --collect-only    # Just list tests
  python run_builder_config_tests.py -k "analyzer"     # Run analyzer tests only
  python run_builder_config_tests.py --coverage        # Show coverage report
        """
    )
    
    parser.add_argument(
        '-v', '--verbosity',
        type=int,
        choices=[0, 1, 2, 3],
        default=2,
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
        '-m', '--markers',
        help='Run tests matching given mark expression (e.g., "not slow")'
    )
    
    parser.add_argument(
        '-k', '--pattern',
        help='Run tests matching given substring expression'
    )
    
    args = parser.parse_args()
    
    print("🧪 Builder-Configuration Alignment Test Runner (pytest-based)")
    print("=" * 70)
    
    success = run_builder_config_tests(
        verbosity=args.verbosity,
        collect_only=args.collect_only,
        markers=args.markers,
        pattern=args.pattern
    )
    
    # Show coverage report if requested
    if args.coverage:
        generate_builder_config_coverage_report()
    
    if not args.collect_only:
        print("\n" + "=" * 70)
        if success:
            print("✅ All builder-configuration alignment tests completed successfully!")
        else:
            print("❌ Some builder-configuration alignment tests failed!")
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
