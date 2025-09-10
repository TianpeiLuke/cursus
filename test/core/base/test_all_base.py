"""
Comprehensive test runner for all base classes in cursus.core.base

This module runs all tests for the base classes and provides a summary.
Note: This file can be used to run pytest on all base test modules.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_pytest_on_base_tests():
    """Run pytest on all base test modules and provide a summary."""
    print("=" * 80)
    print("RUNNING COMPREHENSIVE TESTS FOR CURSUS.CORE.BASE CLASSES WITH PYTEST")
    print("=" * 80)
    
    # Get the base test directory
    base_test_dir = Path(__file__).parent
    
    # List of test modules to run
    test_modules = [
        'test_config_base.py',
        'test_builder_base.py', 
        'test_specification_base.py',
        'test_contract_base.py',
        'test_hyperparameters_base.py',
        'test_enums.py',
    ]
    
    # Run pytest with verbose output
    cmd = [
        sys.executable, '-m', 'pytest',
        '-v',
        '--tb=short',
        '--no-header',
        *[str(base_test_dir / module) for module in test_modules]
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 80)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        # Print stdout
        if result.stdout:
            print(result.stdout)
        
        # Print stderr if there are errors
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print("\n" + "=" * 80)
        print("PYTEST EXECUTION SUMMARY")
        print("=" * 80)
        
        if result.returncode == 0:
            print("🎉 ALL TESTS PASSED!")
        else:
            print("❌ SOME TESTS FAILED!")
        
        print(f"Exit code: {result.returncode}")
        print("=" * 80)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running pytest: {e}")
        return False

def run_individual_test_modules():
    """Run each test module individually with pytest and report results."""
    print("=" * 80)
    print("RUNNING INDIVIDUAL TEST MODULES WITH PYTEST")
    print("=" * 80)
    
    base_test_dir = Path(__file__).parent
    
    test_modules = [
        ('config_base', 'test_config_base.py'),
        ('builder_base', 'test_builder_base.py'),
        ('specification_base', 'test_specification_base.py'),
        ('contract_base', 'test_contract_base.py'),
        ('hyperparameters_base', 'test_hyperparameters_base.py'),
        ('enums', 'test_enums.py'),
    ]
    
    results = {}
    
    for module_name, module_file in test_modules:
        print(f"\n--- Testing {module_name} ---")
        
        cmd = [
            sys.executable, '-m', 'pytest',
            '-v',
            '--tb=line',
            '--no-header',
            str(base_test_dir / module_file)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            # Parse pytest output to get test counts
            output = result.stdout
            print(output)
            
            # Simple parsing of pytest output
            passed = output.count(' PASSED')
            failed = output.count(' FAILED')
            errors = output.count(' ERROR')
            skipped = output.count(' SKIPPED')
            total = passed + failed + errors + skipped
            
            results[module_name] = {
                'total': total,
                'passed': passed,
                'failed': failed,
                'errors': errors,
                'skipped': skipped,
                'exit_code': result.returncode
            }
            
        except Exception as e:
            print(f"Error running {module_name}: {e}")
            results[module_name] = {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'errors': 1,
                'skipped': 0,
                'exit_code': 1
            }
    
    # Print summary table
    print("\n" + "=" * 80)
    print("INDIVIDUAL MODULE RESULTS")
    print("=" * 80)
    print(f"{'Module':<20} {'Total':<8} {'Passed':<8} {'Failed':<8} {'Errors':<8} {'Skipped':<8}")
    print("-" * 80)
    
    total_all = 0
    passed_all = 0
    failed_all = 0
    errors_all = 0
    skipped_all = 0
    
    for module_name, stats in results.items():
        total_all += stats['total']
        passed_all += stats['passed']
        failed_all += stats['failed']
        errors_all += stats['errors']
        skipped_all += stats['skipped']
        
        print(f"{module_name:<20} {stats['total']:<8} {stats['passed']:<8} {stats['failed']:<8} {stats['errors']:<8} {stats['skipped']:<8}")
    
    print("-" * 80)
    print(f"{'TOTAL':<20} {total_all:<8} {passed_all:<8} {failed_all:<8} {errors_all:<8} {skipped_all:<8}")
    print("=" * 80)
    
    return results

def run_with_coverage():
    """Run pytest with coverage reporting."""
    print("=" * 80)
    print("RUNNING TESTS WITH COVERAGE")
    print("=" * 80)
    
    base_test_dir = Path(__file__).parent
    
    cmd = [
        sys.executable, '-m', 'pytest',
        '--cov=cursus.core.base',
        '--cov-report=term-missing',
        '--cov-report=html:htmlcov',
        '-v',
        str(base_test_dir)
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 80)
    
    try:
        result = subprocess.run(cmd, cwd=os.getcwd())
        return result.returncode == 0
    except Exception as e:
        print(f"Error running pytest with coverage: {e}")
        return False

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run tests for cursus.core.base classes with pytest')
    parser.add_argument('--individual', action='store_true', 
                       help='Run individual test modules separately')
    parser.add_argument('--coverage', action='store_true',
                       help='Run tests with coverage reporting')
    parser.add_argument('--summary', action='store_true', default=True,
                       help='Run comprehensive test suite with summary (default)')
    
    args = parser.parse_args()
    
    if args.coverage:
        success = run_with_coverage()
    elif args.individual:
        results = run_individual_test_modules()
        success = all(r['exit_code'] == 0 for r in results.values())
    else:
        success = run_pytest_on_base_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
