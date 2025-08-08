"""
Runner script for Tabular Preprocessing Step Builder tests using existing validators.

This script runs comprehensive tests using the existing validation infrastructure
from src/cursus/validation/builders.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

try:
    from test_with_existing_validators import run_comprehensive_test
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Error importing test module: {e}")
    IMPORTS_AVAILABLE = False


def main():
    """Main runner function."""
    print("🚀" * 50)
    print("TABULAR PREPROCESSING STEP BUILDER VALIDATION")
    print("Using Existing Cursus Validation Infrastructure")
    print("🚀" * 50)
    
    if not IMPORTS_AVAILABLE:
        print("❌ Cannot run tests - required imports not available")
        print("\nPlease ensure the following components are implemented:")
        print("- TabularPreprocessingStepBuilder in src/cursus/steps/builders/")
        print("- TabularPreprocessingConfig in src/cursus/steps/configs/")
        print("- PREPROCESSING_TRAINING_SPEC in src/cursus/steps/specs/")
        print("- TABULAR_PREPROCESS_CONTRACT in src/cursus/steps/contracts/")
        print("- tabular_preprocess.py in src/cursus/steps/scripts/")
        return False
    
    try:
        # Run comprehensive test
        result = run_comprehensive_test()
        
        # Determine success
        success = (result.testsRun > 0 and 
                  len(result.failures) == 0 and 
                  len(result.errors) == 0)
        
        if success:
            print("\n✅ ALL TESTS PASSED!")
            print("The TabularPreprocessingStepBuilder is ready for use.")
        else:
            print("\n⚠️  SOME TESTS FAILED")
            print("Please review the test output above for details.")
            
            if result.failures:
                print(f"\nFailures ({len(result.failures)}):")
                for test, traceback in result.failures:
                    print(f"- {test}")
            
            if result.errors:
                print(f"\nErrors ({len(result.errors)}):")
                for test, traceback in result.errors:
                    print(f"- {test}")
        
        return success
        
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
