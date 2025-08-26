"""Test runner for all runtime testing module tests."""

import unittest
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def discover_and_run_tests():
    """Discover and run all tests in the runtime testing module."""
    
    # Get the directory containing this script
    test_dir = Path(__file__).parent
    
    # Create test loader
    loader = unittest.TestLoader()
    
    # Discover tests in the runtime directory
    suite = loader.discover(
        start_dir=str(test_dir),
        pattern='test_*.py',
        top_level_dir=str(project_root)
    )
    
    # Create test runner with verbosity
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    # Run the tests
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("RUNTIME TESTING MODULE TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1

def run_specific_module_tests(module_name):
    """Run tests for a specific module."""
    
    test_dir = Path(__file__).parent
    loader = unittest.TestLoader()
    
    # Map module names to test file patterns
    module_patterns = {
        # Directory-based patterns
        'core': 'core/test_*.py',
        'utils': 'utils/test_*.py',
        'data': 'data/test_*.py',
        'config': 'config/test_*.py',
        'execution': 'execution/test_*.py',
        'integration': 'integration/test_*.py',
        'jupyter': 'jupyter/test_*.py',
        'pipeline_testing': 'pipeline_testing/test_*.py',
        'production': 'production/test_*.py',
        'testing': 'testing/test_*.py',
        
        # Specific core module tests
        'pipeline_script_executor': 'core/test_pipeline_script_executor.py',
        'script_import_manager': 'core/test_script_import_manager.py',
        'data_flow_manager': 'core/test_data_flow_manager.py',
        
        # Specific data module tests
        'enhanced_data_flow_manager': 'data/test_enhanced_data_flow_manager.py',
        'local_data_manager': 'data/test_local_data_manager.py',
        's3_output_registry': 'data/test_s3_output_registry.py',
        'base_synthetic_data_generator': 'data/test_base_synthetic_data_generator.py',
        'default_synthetic_data_generator': 'data/test_default_synthetic_data_generator.py',
        
        # Specific utils module tests
        'result_models': 'utils/test_result_models.py',
        'execution_context': 'utils/test_execution_context.py',
        'error_handling': 'utils/test_error_handling.py',
        
        # Specific jupyter module tests
        'notebook_interface': 'jupyter/test_notebook_interface.py',
        'visualization': 'jupyter/test_visualization.py',
        'debugger': 'jupyter/test_debugger.py',
        'templates': 'jupyter/test_templates.py',
        'advanced': 'jupyter/test_advanced.py'
    }
    
    if module_name not in module_patterns:
        print(f"Unknown module: {module_name}")
        print(f"Available modules: {', '.join(module_patterns.keys())}")
        return 1
    
    pattern = module_patterns[module_name]
    
    if '/' in pattern and not pattern.endswith('*.py'):
        # Specific file
        test_file = test_dir / pattern
        if test_file.exists():
            suite = loader.loadTestsFromName(f"test.validation.runtime.{pattern.replace('/', '.').replace('.py', '')}")
        else:
            print(f"Test file not found: {test_file}")
            return 1
    else:
        # Pattern-based discovery (includes patterns like 'data/test_*.py')
        if '/' in pattern:
            # Extract directory and pattern
            parts = pattern.split('/')
            subdir = parts[0]
            file_pattern = parts[1]
            search_dir = test_dir / subdir
        else:
            # Simple pattern like 'test_*.py'
            search_dir = test_dir
            file_pattern = pattern
        
        suite = loader.discover(
            start_dir=str(search_dir),
            pattern=file_pattern,
            top_level_dir=str(project_root)
        )
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1

def main():
    """Main entry point for test runner."""
    
    if len(sys.argv) > 1:
        module_name = sys.argv[1]
        exit_code = run_specific_module_tests(module_name)
    else:
        exit_code = discover_and_run_tests()
    
    sys.exit(exit_code)

if __name__ == '__main__':
    main()
