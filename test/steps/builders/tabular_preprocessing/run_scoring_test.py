"""
Runner script for Tabular Preprocessing Step Builder scoring tests using Enhanced 4-Level Processing Tester.

This script runs comprehensive scoring tests using the enhanced 4-level Processing tester
and generates detailed reports with pattern-specific analysis.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

try:
    from test_scoring import main as run_scoring_main, run_comprehensive_scoring_test, run_4_level_scoring_comparison_test
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Error importing scoring test module: {e}")
    IMPORTS_AVAILABLE = False


def main():
    """Main runner function for 4-level scoring tests."""
    print("🎯" * 50)
    print("TABULAR PREPROCESSING STEP BUILDER SCORING")
    print("Using Enhanced 4-Level Processing Tester")
    print("🎯" * 50)
    
    if not IMPORTS_AVAILABLE:
        print("❌ Cannot run scoring tests - required imports not available")
        print("\nPlease ensure the following components are implemented:")
        print("- TabularPreprocessingStepBuilder in src/cursus/steps/builders/")
        print("- TabularPreprocessingConfig in src/cursus/steps/configs/")
        print("- PREPROCESSING_TRAINING_SPEC in src/cursus/steps/specs/")
        print("- TABULAR_PREPROCESS_CONTRACT in src/cursus/steps/contracts/")
        print("- Enhanced 4-level Processing tester in src/cursus/validation/builders/variants/")
        print("- Scoring system in src/cursus/validation/builders/scoring.py")
        return False
    
    try:
        print("\n📊 4-Level Processing Pattern Scoring Overview:")
        print("This scoring system evaluates Processing step builders against:")
        print("• Level 1: Interface compliance (processor creation, method signatures)")
        print("• Level 2: Specification compliance (job types, env vars, arguments)")
        print("• Level 3: Path mapping compliance (inputs/outputs, special patterns)")
        print("• Level 4: Integration compliance (end-to-end creation, dependencies)")
        print()
        
        # Run the main scoring function
        run_scoring_main()
        
        print("\n✅ 4-Level scoring tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Scoring test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_quick_scoring():
    """Run a quick scoring test to verify the 4-level tester integration."""
    print("\n🔍 Quick 4-Level Scoring Validation...")
    
    try:
        # Create output directory
        output_dir = Path(__file__).parent / "quick_scoring_reports"
        output_dir.mkdir(exist_ok=True)
        
        # Run comprehensive scoring test
        score_report = run_comprehensive_scoring_test(
            save_report=True,
            generate_chart=False,  # Skip chart for quick test
            output_dir=str(output_dir),
            verbose=False
        )
        
        if score_report:
            overall_score = score_report.get("overall", {}).get("score", 0)
            rating = score_report.get("overall", {}).get("rating", "Unknown")
            
            print(f"   ✅ Quick scoring completed!")
            print(f"   • Overall Score: {overall_score:.1f}% ({rating})")
            
            # Check level scores
            levels = score_report.get("levels", {})
            for level_name, level_data in levels.items():
                level_score = level_data.get("score", 0)
                display_name = level_name.replace("level", "Level ").replace("_", " ").title()
                print(f"   • {display_name}: {level_score:.1f}%")
            
            return True
        else:
            print("   ❌ Quick scoring failed - no report generated")
            return False
            
    except Exception as e:
        print(f"   ❌ Quick scoring validation failed: {e}")
        return False


def run_pattern_analysis():
    """Run pattern-specific analysis using the 4-level tester."""
    print("\n🔍 Processing Pattern Analysis...")
    
    try:
        from cursus.validation.builders.variants.processing_test import ProcessingStepBuilderTest
        from cursus.steps.builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
        from cursus.steps.configs.config_tabular_preprocessing_step import TabularPreprocessingConfig
        from cursus.steps.specs.preprocessing_training_spec import PREPROCESSING_TRAINING_SPEC
        from cursus.steps.contracts.tabular_preprocess_contract import TABULAR_PREPROCESS_CONTRACT
        
        # Create test configuration
        config = TabularPreprocessingConfig(
            label_name="target",
            processing_entry_point="tabular_preprocess.py",
            job_type="training",
            region="NA",
            pipeline_name="test-pattern-analysis",
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
            spec=PREPROCESSING_TRAINING_SPEC,
            contract=TABULAR_PREPROCESS_CONTRACT,
            step_name="TabularPreprocessingStep",
            verbose=False
        )
        
        # Analyze Processing patterns
        print("   Analyzing Processing step patterns...")
        
        # Pattern categories from processing_step_builder_patterns.md
        pattern_categories = {
            "Processor Creation": {
                "tests": ["level1_test_processor_creation_method", "level1_test_framework_specific_methods"],
                "description": "SKLearn vs XGBoost processor creation patterns"
            },
            "Step Creation Pattern": {
                "tests": ["level1_test_step_creation_pattern_compliance", "level4_test_step_creation_pattern_execution"],
                "description": "Pattern A (direct ProcessingStep) vs Pattern B (processor.run + step_args)"
            },
            "Environment Variables": {
                "tests": ["level1_test_environment_variables_method", "level2_test_environment_variable_patterns"],
                "description": "Basic, JSON-serialized, and step-specific environment variable patterns"
            },
            "Job Arguments": {
                "tests": ["level1_test_job_arguments_method", "level2_test_job_arguments_patterns"],
                "description": "Simple job type, complex arguments, and no arguments patterns"
            },
            "Input/Output Handling": {
                "tests": ["level1_test_processing_input_output_methods", "level3_test_processing_input_creation", "level3_test_processing_output_creation"],
                "description": "ProcessingInput/ProcessingOutput creation and specification-driven patterns"
            },
            "Special Input Patterns": {
                "tests": ["level3_test_special_input_handling", "level3_test_local_path_override_patterns", "level3_test_file_upload_patterns"],
                "description": "Local path override, file upload, and S3 handling patterns"
            },
            "Specification-Driven": {
                "tests": ["level2_test_specification_driven_inputs", "level2_test_specification_driven_outputs", "level2_test_contract_path_mapping"],
                "description": "Specification-driven architecture and contract-based path mapping"
            },
            "Multi-Job-Type Support": {
                "tests": ["level2_test_job_type_specification_loading", "level2_test_multi_job_type_support"],
                "description": "Job type-based specification loading patterns"
            }
        }
        
        # Run pattern analysis
        pattern_results = {}
        for pattern_name, pattern_info in pattern_categories.items():
            test_results = []
            for test_name in pattern_info["tests"]:
                if hasattr(tester, test_name):
                    try:
                        getattr(tester, test_name)()
                        test_results.append(True)
                    except Exception:
                        test_results.append(False)
                else:
                    test_results.append(False)
            
            if test_results:
                passed_count = sum(test_results)
                total_count = len(test_results)
                pass_rate = (passed_count / total_count) * 100
                
                # Determine compliance level
                if pass_rate >= 90:
                    compliance = "🟢 FULLY COMPLIANT"
                elif pass_rate >= 80:
                    compliance = "🟡 MOSTLY COMPLIANT"
                elif pass_rate >= 60:
                    compliance = "🟠 PARTIALLY COMPLIANT"
                else:
                    compliance = "🔴 NON-COMPLIANT"
                
                pattern_results[pattern_name] = {
                    "pass_rate": pass_rate,
                    "passed": passed_count,
                    "total": total_count,
                    "compliance": compliance,
                    "description": pattern_info["description"]
                }
        
        # Print pattern analysis results
        print("\n📊 Processing Pattern Compliance Analysis:")
        print("-" * 70)
        
        for pattern_name, result in pattern_results.items():
            print(f"\n🔍 {pattern_name}:")
            print(f"   {result['description']}")
            print(f"   Compliance: {result['passed']}/{result['total']} tests ({result['pass_rate']:.1f}%) {result['compliance']}")
        
        # Overall pattern compliance
        overall_pass_rate = sum(r['pass_rate'] for r in pattern_results.values()) / len(pattern_results)
        print(f"\n🎯 Overall Pattern Compliance: {overall_pass_rate:.1f}%")
        
        if overall_pass_rate >= 90:
            print("   🎉 Excellent! This Processing step fully complies with all identified patterns.")
        elif overall_pass_rate >= 80:
            print("   👍 Good compliance with minor pattern deviations.")
        elif overall_pass_rate >= 70:
            print("   ⚠️  Satisfactory but needs attention in some pattern areas.")
        else:
            print("   🚨 Significant pattern compliance issues detected.")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Pattern analysis failed: {e}")
        return False


def show_scoring_usage():
    """Show usage examples for the 4-level scoring system."""
    print("\n📚 4-Level Processing Scoring Usage Examples:")
    print("=" * 60)
    
    print("\n1️⃣ Run comprehensive scoring with reports:")
    print("   python run_scoring_test.py")
    
    print("\n2️⃣ Run quick scoring validation:")
    print("   python run_scoring_test.py --quick")
    
    print("\n3️⃣ Run pattern analysis:")
    print("   python run_scoring_test.py --patterns")
    
    print("\n4️⃣ Run configuration comparison:")
    print("   python test_scoring.py")
    
    print("\n5️⃣ Generate detailed reports:")
    print("   python test_scoring.py  # Creates reports in scoring_reports/")
    
    print("\n🎯 4-Level Scoring Breakdown:")
    print("• Level 1 Score: Interface compliance (method signatures, processor creation)")
    print("• Level 2 Score: Specification compliance (job types, env vars, arguments)")
    print("• Level 3 Score: Path mapping compliance (inputs/outputs, special patterns)")
    print("• Level 4 Score: Integration compliance (end-to-end creation, dependencies)")
    print("• Overall Score: Weighted average of all levels with pattern-specific analysis")
    
    print("\n📊 Score Interpretation:")
    print("• 90-100%: 🟢 Excellent - Fully compliant with Processing patterns")
    print("• 80-89%:  🟡 Good - Minor pattern deviations")
    print("• 70-79%:  🟠 Satisfactory - Some pattern issues")
    print("• 60-69%:  🔴 Needs Work - Significant pattern problems")
    print("• <60%:    ⚫ Poor - Major pattern violations")


if __name__ == '__main__':
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            show_scoring_usage()
            sys.exit(0)
        elif sys.argv[1] == "--quick" or sys.argv[1] == "-q":
            success = run_quick_scoring()
            sys.exit(0 if success else 1)
        elif sys.argv[1] == "--patterns" or sys.argv[1] == "-p":
            success = run_pattern_analysis()
            sys.exit(0 if success else 1)
        elif sys.argv[1] == "--usage":
            show_scoring_usage()
            sys.exit(0)
    
    # Run main scoring
    success = main()
    sys.exit(0 if success else 1)
