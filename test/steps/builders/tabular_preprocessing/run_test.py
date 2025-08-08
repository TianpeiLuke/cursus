"""
Runner script for Tabular Preprocessing Step Builder tests using Enhanced 4-Level Processing Tester.

This script runs comprehensive tests using the enhanced 4-level Processing tester
from src/cursus/validation/builders/variants/processing_test.py.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

try:
    from test_tabular_preprocessing import run_comprehensive_4_level_test
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Error importing test module: {e}")
    IMPORTS_AVAILABLE = False


def main():
    """Main runner function using enhanced 4-level Processing tester."""
    print("🚀" * 50)
    print("TABULAR PREPROCESSING STEP BUILDER VALIDATION")
    print("Using Enhanced 4-Level Processing Tester")
    print("🚀" * 50)
    
    if not IMPORTS_AVAILABLE:
        print("❌ Cannot run tests - required imports not available")
        print("\nPlease ensure the following components are implemented:")
        print("- TabularPreprocessingStepBuilder in src/cursus/steps/builders/")
        print("- TabularPreprocessingConfig in src/cursus/steps/configs/")
        print("- PREPROCESSING_TRAINING_SPEC in src/cursus/steps/specs/")
        print("- TABULAR_PREPROCESS_CONTRACT in src/cursus/steps/contracts/")
        print("- tabular_preprocess.py in src/cursus/steps/scripts/")
        print("- Enhanced 4-level Processing tester in src/cursus/validation/builders/variants/")
        return False
    
    try:
        # Run comprehensive 4-level test
        print("\n🧪 Running Enhanced 4-Level Processing Tests...")
        print("This will test all Processing step patterns identified in the analysis:")
        print("• Level 1: Interface Tests (processor creation, configuration attributes)")
        print("• Level 2: Specification Tests (job types, environment variables, arguments)")
        print("• Level 3: Path Mapping Tests (input/output creation, special patterns)")
        print("• Level 4: Integration Tests (end-to-end step creation, dependencies)")
        print()
        
        result = run_comprehensive_4_level_test()
        
        # Determine success
        success = (result.testsRun > 0 and 
                  len(result.failures) == 0 and 
                  len(result.errors) == 0)
        
        if success:
            print("\n✅ ALL 4-LEVEL PROCESSING TESTS PASSED!")
            print("The TabularPreprocessingStepBuilder fully complies with Processing step patterns.")
            print("\n🎯 Pattern Compliance Summary:")
            print("• ✅ Processor Creation Patterns (SKLearn/XGBoost)")
            print("• ✅ Step Creation Patterns (Pattern A/Pattern B)")
            print("• ✅ Environment Variable Patterns (Basic/JSON/Step-specific)")
            print("• ✅ Input/Output Handling Patterns")
            print("• ✅ Special Input Patterns (Local path override, File upload)")
            print("• ✅ Specification-Driven Architecture")
            print("• ✅ Contract-Based Path Mapping")
            print("• ✅ Multi-Job-Type Support")
        else:
            print("\n⚠️  SOME 4-LEVEL PROCESSING TESTS FAILED")
            print("Please review the test output above for details.")
            
            if result.failures:
                print(f"\nFailures ({len(result.failures)}):")
                for test, traceback in result.failures:
                    print(f"- {test}")
                    # Extract level information from test name
                    test_str = str(test)
                    if "level1_" in test_str:
                        print("  → Level 1 (Interface) issue detected")
                    elif "level2_" in test_str:
                        print("  → Level 2 (Specification) issue detected")
                    elif "level3_" in test_str:
                        print("  → Level 3 (Path Mapping) issue detected")
                    elif "level4_" in test_str:
                        print("  → Level 4 (Integration) issue detected")
            
            if result.errors:
                print(f"\nErrors ({len(result.errors)}):")
                for test, traceback in result.errors:
                    print(f"- {test}")
                    # Extract level information from test name
                    test_str = str(test)
                    if "level1_" in test_str:
                        print("  → Level 1 (Interface) error detected")
                    elif "level2_" in test_str:
                        print("  → Level 2 (Specification) error detected")
                    elif "level3_" in test_str:
                        print("  → Level 3 (Path Mapping) error detected")
                    elif "level4_" in test_str:
                        print("  → Level 4 (Integration) error detected")
            
            print("\n🔧 Troubleshooting Guide:")
            print("• Level 1 Issues: Check processor creation and configuration attributes")
            print("• Level 2 Issues: Review specification loading and environment variables")
            print("• Level 3 Issues: Verify input/output creation and path mapping")
            print("• Level 4 Issues: Debug step creation and dependency resolution")
        
        return success
        
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_quick_validation():
    """Run a quick validation to check if the 4-level tester is working."""
    print("\n🔍 Quick 4-Level Tester Validation...")
    
    try:
        from cursus.validation.builders.variants.processing_test import ProcessingStepBuilderTest
        from cursus.steps.builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
        from cursus.steps.configs.config_tabular_preprocessing_step import TabularPreprocessingConfig
        
        # Create minimal config
        config = TabularPreprocessingConfig(
            label_name="target",
            processing_entry_point="tabular_preprocess.py",
            job_type="training",
            region="NA",
            pipeline_name="test-pipeline",
            source_dir="src/cursus/steps/scripts",
            processing_instance_count=1,
            processing_volume_size=30,
            processing_instance_type_large="ml.m5.xlarge",
            processing_instance_type_small="ml.m5.large",
            processing_framework_version="0.23-1",
            use_large_processing_instance=False,
            py_version="py3"
        )
        
        # Create tester
        tester = ProcessingStepBuilderTest(
            builder_class=TabularPreprocessingStepBuilder,
            config=config,
            step_name="TabularPreprocessingStep",
            verbose=False
        )
        
        # Check if 4-level methods are available
        level_methods = {
            "Level 1": ["level1_test_processor_creation_method", "level1_test_processing_configuration_attributes"],
            "Level 2": ["level2_test_environment_variable_patterns", "level2_test_job_arguments_patterns"],
            "Level 3": ["level3_test_processing_input_creation", "level3_test_processing_output_creation"],
            "Level 4": ["level4_test_end_to_end_step_creation", "level4_test_processing_dependency_resolution"]
        }
        
        print("   Checking 4-level method availability:")
        all_available = True
        for level_name, methods in level_methods.items():
            available_methods = [method for method in methods if hasattr(tester, method)]
            print(f"   • {level_name}: {len(available_methods)}/{len(methods)} methods available")
            if len(available_methods) < len(methods):
                all_available = False
                missing = [method for method in methods if not hasattr(tester, method)]
                print(f"     Missing: {', '.join(missing)}")
        
        if all_available:
            print("   ✅ All 4-level methods are available!")
            return True
        else:
            print("   ❌ Some 4-level methods are missing!")
            return False
            
    except Exception as e:
        print(f"   ❌ Quick validation failed: {e}")
        return False


def show_usage_examples():
    """Show usage examples for the 4-level tester."""
    print("\n📚 4-Level Processing Tester Usage Examples:")
    print("=" * 50)
    
    print("\n1️⃣ Run all 4-level tests:")
    print("   python run_test.py")
    
    print("\n2️⃣ Run specific level tests:")
    print("   python test_tabular_preprocessing.py TestTabularPreprocessingWith4LevelTester.test_level1_interface_tests")
    print("   python test_tabular_preprocessing.py TestTabularPreprocessingWith4LevelTester.test_level2_specification_tests")
    print("   python test_tabular_preprocessing.py TestTabularPreprocessingWith4LevelTester.test_level3_path_mapping_tests")
    print("   python test_tabular_preprocessing.py TestTabularPreprocessingWith4LevelTester.test_level4_integration_tests")
    
    print("\n3️⃣ Run scoring tests:")
    print("   python test_scoring.py")
    
    print("\n4️⃣ Run legacy compatibility tests:")
    print("   python test_tabular_preprocessing.py TestTabularPreprocessingWith4LevelTester.test_legacy_compatibility")
    
    print("\n5️⃣ Run multiple job type tests:")
    print("   python test_tabular_preprocessing.py TestTabularPreprocessingMultipleJobTypes.test_all_job_types_with_4_level_tester")
    
    print("\n🎯 What Each Level Tests:")
    print("• Level 1 (Interface): Method signatures, processor creation, configuration attributes")
    print("• Level 2 (Specification): Job types, environment variables, arguments, contract mapping")
    print("• Level 3 (Path Mapping): Input/output creation, special patterns, S3 handling")
    print("• Level 4 (Integration): End-to-end step creation, dependencies, caching")


if __name__ == '__main__':
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            show_usage_examples()
            sys.exit(0)
        elif sys.argv[1] == "--quick" or sys.argv[1] == "-q":
            success = run_quick_validation()
            sys.exit(0 if success else 1)
        elif sys.argv[1] == "--examples":
            show_usage_examples()
            sys.exit(0)
    
    # Run main test
    success = main()
    sys.exit(0 if success else 1)
