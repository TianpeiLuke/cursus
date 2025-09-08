"""
Test for circular imports in the cursus package.

This module systematically tests all modules in the cursus package to detect
circular import dependencies that could cause import failures or runtime issues.
"""

import unittest
import sys
import os
import importlib
import importlib.util
from typing import Set, List, Dict, Optional, Tuple
from pathlib import Path
import traceback
import warnings

class CircularImportDetector:
    """
    Utility class to detect circular imports in a package.
    
    This class systematically imports all modules in a package and tracks
    the import chain to detect circular dependencies.
    """
    
    def __init__(self, package_root: str):
        """
        Initialize the circular import detector.
        
        Args:
            package_root: Root directory of the package to test
        """
        self.package_root = Path(package_root)
        self.import_stack: List[str] = []
        self.imported_modules: Set[str] = set()
        self.failed_imports: Dict[str, str] = {}
        self.circular_imports: List[Tuple[str, List[str]]] = []
        
    def discover_modules(self) -> List[str]:
        """
        Discover all Python modules in the package.
        
        Returns:
            List of module names that can be imported
        """
        modules = []
        
        for py_file in self.package_root.rglob("*.py"):
            # Skip __pycache__ and other non-module files
            if "__pycache__" in str(py_file):
                continue
                
            # Convert file path to module name
            relative_path = py_file.relative_to(self.package_root.parent)
            module_parts = list(relative_path.parts)
            
            # Remove .py extension
            if module_parts[-1].endswith('.py'):
                module_parts[-1] = module_parts[-1][:-3]
            
            # Skip __init__ modules for now (we'll handle them separately)
            if module_parts[-1] == '__init__':
                module_parts = module_parts[:-1]
                if not module_parts:  # Skip root __init__.py
                    continue
            
            module_name = '.'.join(module_parts)
            modules.append(module_name)
        
        return sorted(modules)
    
    def test_import(self, module_name: str) -> Tuple[bool, Optional[str]]:
        """
        Test importing a single module and detect circular imports.
        
        Args:
            module_name: Name of the module to import
            
        Returns:
            Tuple of (success, error_message)
        """
        if module_name in self.imported_modules:
            return True, None
            
        if module_name in self.import_stack:
            # Circular import detected
            cycle_start = self.import_stack.index(module_name)
            cycle = self.import_stack[cycle_start:] + [module_name]
            self.circular_imports.append((module_name, cycle))
            return False, f"Circular import detected: {' -> '.join(cycle)}"
        
        self.import_stack.append(module_name)
        
        try:
            # Clear any existing module to force fresh import
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            # Attempt to import the module
            importlib.import_module(module_name)
            self.imported_modules.add(module_name)
            success = True
            error = None
            
        except ImportError as e:
            error_msg = str(e)
            # Check if this is a circular import error
            if "circular import" in error_msg.lower() or "cannot import name" in error_msg.lower():
                cycle_info = f"Circular import in {module_name}: {error_msg}"
                self.circular_imports.append((module_name, self.import_stack + [module_name]))
                error = cycle_info
            else:
                error = f"Import error in {module_name}: {error_msg}"
            
            self.failed_imports[module_name] = error
            success = False
            
        except Exception as e:
            error = f"Unexpected error importing {module_name}: {str(e)}"
            self.failed_imports[module_name] = error
            success = False
            
        finally:
            # Remove from import stack
            if module_name in self.import_stack:
                self.import_stack.remove(module_name)
        
        return success, error
    
    def test_all_imports(self) -> Dict[str, any]:
        """
        Test importing all discovered modules.
        
        Returns:
            Dictionary with test results
        """
        modules = self.discover_modules()
        results = {
            'total_modules': len(modules),
            'successful_imports': 0,
            'failed_imports': 0,
            'circular_imports': 0,
            'modules_tested': [],
            'import_failures': {},
            'circular_import_chains': []
        }
        
        for module_name in modules:
            success, error = self.test_import(module_name)
            results['modules_tested'].append({
                'module': module_name,
                'success': success,
                'error': error
            })
            
            if success:
                results['successful_imports'] += 1
            else:
                results['failed_imports'] += 1
                results['import_failures'][module_name] = error
                
                # Check if it's a circular import
                if any(module_name == circ[0] for circ in self.circular_imports):
                    results['circular_imports'] += 1
        
        # Add circular import chain details
        results['circular_import_chains'] = [
            {'module': module, 'chain': chain} 
            for module, chain in self.circular_imports
        ]
        
        return results

class TestCircularImports(unittest.TestCase):
    """Test suite for detecting circular imports in the cursus package."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""
        cls.cursus_root = os.path.join(project_root, 'src', 'cursus')
        cls.detector = CircularImportDetector(cls.cursus_root)
        
    def test_no_circular_imports_in_cursus_package(self):
        """Test that there are no circular imports in the cursus package."""
        print("\n" + "="*80)
        print("TESTING CURSUS PACKAGE FOR CIRCULAR IMPORTS")
        print("="*80)
        
        # Run the circular import detection
        results = self.detector.test_all_imports()
        
        # Print detailed results
        print(f"\nTotal modules discovered: {results['total_modules']}")
        print(f"Successful imports: {results['successful_imports']}")
        print(f"Failed imports: {results['failed_imports']}")
        print(f"Circular imports detected: {results['circular_imports']}")
        
        # Print failed imports
        if results['import_failures']:
            print(f"\nFAILED IMPORTS ({len(results['import_failures'])}):")
            for module, error in results['import_failures'].items():
                print(f"  - {module}: {error}")
        
        # Print circular import chains
        if results['circular_import_chains']:
            print(f"\nCIRCULAR IMPORT CHAINS ({len(results['circular_import_chains'])}):")
            for chain_info in results['circular_import_chains']:
                module = chain_info['module']
                chain = chain_info['chain']
                print(f"  - {module}: {' -> '.join(chain)}")
        
        # Print successful imports for reference
        successful_modules = [
            test['module'] for test in results['modules_tested'] 
            if test['success']
        ]
        if successful_modules:
            print(f"\nSUCCESSFUL IMPORTS ({len(successful_modules)}):")
            for module in successful_modules[:10]:  # Show first 10
                print(f"  - {module}")
            if len(successful_modules) > 10:
                print(f"  ... and {len(successful_modules) - 10} more")
        
        print("="*80)
        
        # Assert no circular imports
        self.assertEqual(
            results['circular_imports'], 0,
            f"Found {results['circular_imports']} circular imports in cursus package. "
            f"Chains: {results['circular_import_chains']}"
        )
        
        # We allow some import failures (e.g., missing dependencies) but not circular imports
        # However, if there are too many failures, it might indicate a problem
        failure_rate = results['failed_imports'] / results['total_modules'] if results['total_modules'] > 0 else 0
        if failure_rate > 0.5:  # More than 50% failures might indicate a systemic issue
            warnings.warn(
                f"High import failure rate: {failure_rate:.1%} "
                f"({results['failed_imports']}/{results['total_modules']})"
            )
    
    def test_core_modules_import_successfully(self):
        """Test that core modules can be imported without circular dependencies."""
        core_modules = [
            'src.cursus.core.base.config_base',
            'src.cursus.core.base.builder_base', 
            'src.cursus.core.base.specification_base',
            'src.cursus.core.base.contract_base',
            'src.cursus.core.base.hyperparameters_base',
            'src.cursus.core.base.enums',
        ]
        
        print(f"\nTesting core modules for circular imports...")
        
        failed_core_modules = []
        for module_name in core_modules:
            success, error = self.detector.test_import(module_name)
            if not success:
                failed_core_modules.append((module_name, error))
                print(f"  ❌ {module_name}: {error}")
            else:
                print(f"  ✅ {module_name}")
        
        self.assertEqual(
            len(failed_core_modules), 0,
            f"Core modules failed to import: {failed_core_modules}"
        )
    
    def test_api_modules_import_successfully(self):
        """Test that API modules can be imported without circular dependencies."""
        api_modules = [
            'src.cursus.api.dag.base_dag',
            'src.cursus.api.dag.edge_types',
            'src.cursus.api.dag.enhanced_dag',
        ]
        
        print(f"\nTesting API modules for circular imports...")
        
        failed_api_modules = []
        for module_name in api_modules:
            success, error = self.detector.test_import(module_name)
            if not success:
                failed_api_modules.append((module_name, error))
                print(f"  ❌ {module_name}: {error}")
            else:
                print(f"  ✅ {module_name}")
        
        self.assertEqual(
            len(failed_api_modules), 0,
            f"API modules failed to import: {failed_api_modules}"
        )
    
    def test_step_modules_import_successfully(self):
        """Test that step-related modules can be imported without circular dependencies."""
        step_modules = [
            'src.cursus.steps.registry.builder_registry',
            'src.cursus.steps.registry.hyperparameter_registry',
            'src.cursus.steps.registry.step_names',
        ]
        
        print(f"\nTesting step modules for circular imports...")
        
        failed_step_modules = []
        for module_name in step_modules:
            success, error = self.detector.test_import(module_name)
            if not success:
                failed_step_modules.append((module_name, error))
                print(f"  ❌ {module_name}: {error}")
            else:
                print(f"  ✅ {module_name}")
        
        # Allow some failures in step modules as they might have external dependencies
        # but ensure no circular imports
        circular_failures = [
            (module, error) for module, error in failed_step_modules
            if "circular import" in error.lower() or "cannot import name" in error.lower()
        ]
        
        self.assertEqual(
            len(circular_failures), 0,
            f"Step modules have circular imports: {circular_failures}"
        )
    
    def test_import_order_independence(self):
        """Test that modules can be imported in different orders without issues."""
        # Test a few key modules in different orders
        test_modules = [
            'src.cursus.core.base.config_base',
            'src.cursus.core.base.builder_base',
            'src.cursus.core.base.specification_base',
        ]
        
        print(f"\nTesting import order independence...")
        
        # Clear any existing imports
        for module in test_modules:
            if module in sys.modules:
                del sys.modules[module]
        
        # Test forward order
        forward_results = []
        for module in test_modules:
            success, error = self.detector.test_import(module)
            forward_results.append((module, success, error))
        
        # Clear imports again
        for module in test_modules:
            if module in sys.modules:
                del sys.modules[module]
        
        # Test reverse order
        reverse_results = []
        for module in reversed(test_modules):
            success, error = self.detector.test_import(module)
            reverse_results.append((module, success, error))
        
        # Compare results - they should be the same
        forward_success = [r[1] for r in forward_results]
        reverse_success = [r[1] for r in reversed(reverse_results)]
        
        self.assertEqual(
            forward_success, reverse_success,
            "Import success differs based on import order, indicating potential circular dependency"
        )
        
        print(f"  ✅ Import order independence verified")

def run_circular_import_tests():
    """Run the circular import tests with detailed output."""
    import io
    from datetime import datetime
    
    # Create a string buffer to capture all output
    output_buffer = io.StringIO()
    
    def tee_print(*args, **kwargs):
        """Print to both stdout and our buffer."""
        print(*args, **kwargs)
        print(*args, **kwargs, file=output_buffer)
    
    tee_print("="*80)
    tee_print("CURSUS PACKAGE CIRCULAR IMPORT TEST SUITE")
    tee_print("="*80)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCircularImports)
    
    # Create a custom stream that writes to both stdout and our buffer
    class TeeStream:
        def __init__(self, *streams):
            self.streams = streams
        
        def write(self, data):
            for stream in self.streams:
                stream.write(data)
        
        def flush(self):
            for stream in self.streams:
                stream.flush()
    
    tee_stream = TeeStream(sys.stdout, output_buffer)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=tee_stream)
    result = runner.run(suite)
    
    # Print summary
    tee_print("\n" + "="*80)
    tee_print("CIRCULAR IMPORT TEST SUMMARY")
    tee_print("="*80)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total_tests - failures - errors
    
    tee_print(f"Total Tests: {total_tests}")
    tee_print(f"Passed: {passed}")
    tee_print(f"Failed: {failures}")
    tee_print(f"Errors: {errors}")
    
    if failures > 0:
        tee_print(f"\nFAILURES:")
        for test, traceback in result.failures:
            tee_print(f"  - {test}")
            tee_print(f"    {traceback.split('AssertionError:')[-1].strip()}")
    
    if errors > 0:
        tee_print(f"\nERRORS:")
        for test, traceback in result.errors:
            tee_print(f"  - {test}")
    
    success = failures == 0 and errors == 0
    if success:
        tee_print(f"\n🎉 ALL CIRCULAR IMPORT TESTS PASSED!")
    else:
        tee_print(f"\n❌ SOME CIRCULAR IMPORT TESTS FAILED!")
    
    tee_print("="*80)
    
    # Save output to file in slipbox/test folder
    try:
        # Ensure slipbox/test directory exists
        slipbox_test_dir = os.path.join(project_root, 'slipbox', 'test')
        os.makedirs(slipbox_test_dir, exist_ok=True)
        
        # Generate output file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(slipbox_test_dir, f'circular_import_test_output_{timestamp}.txt')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Cursus Package Circular Import Test Output\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")
            f.write(output_buffer.getvalue())
        
        print(f"\n📄 Test output saved to: {output_file}")
        
    except Exception as e:
        print(f"\n⚠️  Warning: Could not save output file: {e}")
    
    return success

if __name__ == '__main__':
    # Run the tests
    success = run_circular_import_tests()
    sys.exit(0 if success else 1)
