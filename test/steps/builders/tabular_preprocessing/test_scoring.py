"""
Scoring test program for TabularPreprocessingStepBuilder using Enhanced 4-Level Processing Tester.

This module provides comprehensive scoring and evaluation of the TabularPreprocessingStepBuilder
using the enhanced 4-level Processing tester and scoring system from 
src/cursus/validation/builders/scoring.py.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

try:
    from cursus.validation.builders.variants.processing_test import ProcessingStepBuilderTest
    from cursus.validation.builders.scoring import score_builder_results, StepBuilderScorer
    from cursus.steps.builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
    from cursus.steps.configs.config_tabular_preprocessing_step import TabularPreprocessingConfig
    from cursus.steps.specs.preprocessing_training_spec import PREPROCESSING_TRAINING_SPEC
    from cursus.steps.contracts.tabular_preprocess_contract import TABULAR_PREPROCESS_CONTRACT
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure the src directory is in your Python path and all dependencies are installed.")
    sys.exit(1)


def create_comprehensive_test_config() -> TabularPreprocessingConfig:
    """
    Create a comprehensive test configuration for scoring.
    
    Returns:
        TabularPreprocessingConfig instance
    """
    # Get the actual source directory using current working directory
    # This ensures we get the correct path regardless of how the script is called
    current_dir = Path.cwd()
    if current_dir.name == 'tabular_preprocessing':
        # We're in the test directory, go up to project root
        src_dir = current_dir.parent.parent.parent.parent / 'src' / 'cursus' / 'steps' / 'scripts'
    else:
        # We're in project root or elsewhere, calculate from __file__
        src_dir = Path(__file__).parent.parent.parent.parent / 'src' / 'cursus' / 'steps' / 'scripts'
    
    src_dir = src_dir.resolve()  # Convert to absolute path
    
    # Verify the path exists and contains the script
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src_dir}")
    
    script_path = src_dir / 'tabular_preprocess.py'
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    
    return TabularPreprocessingConfig(
        # Essential User Inputs (Tier 1)
        label_name="target",
        
        # System Fields with Defaults (Tier 2)
        processing_entry_point="tabular_preprocess.py",
        job_type="training",
        
        # Base configuration fields (required by parent classes)
        region="NA",
        pipeline_name="test-tabular-preprocessing-pipeline",
        pipeline_s3_loc="s3://test-bucket/test-pipeline",
        source_dir=str(src_dir),
        author="test-author",
        bucket="test-bucket",
        role="arn:aws:iam::123456789012:role/TestRole",
        service_name="test-service",
        pipeline_version="1.0.0",
        
        # Processing configuration
        processing_instance_count=1,
        processing_volume_size=30,
        processing_instance_type_large="ml.m5.xlarge",
        processing_instance_type_small="ml.m5.large",
        processing_framework_version="0.23-1",
        use_large_processing_instance=False,
        py_version="py3",
        
        # Optional configurations for comprehensive testing
        train_ratio=0.7,
        test_val_ratio=0.5,
        categorical_columns=["category_col1", "category_col2"],
        numerical_columns=["num_col1", "num_col2"],
        text_columns=["text_col1"],
        date_columns=["date_col1"]
    )


def run_comprehensive_scoring_test(
    save_report: bool = True,
    generate_chart: bool = True,
    output_dir: str = "test_reports",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run comprehensive scoring test for TabularPreprocessingStepBuilder using 4-level tester.
    
    Args:
        save_report: Whether to save the score report to a file
        generate_chart: Whether to generate a chart visualization
        output_dir: Directory to save reports and charts
        verbose: Whether to print verbose output
        
    Returns:
        Score report dictionary
    """
    print("🚀" * 50)
    print("TABULAR PREPROCESSING STEP BUILDER SCORING TEST")
    print("Using Enhanced 4-Level Processing Tester")
    print("🚀" * 50)
    
    try:
        # Create test configuration
        if verbose:
            print("\n📋 Creating comprehensive test configuration...")
        config = create_comprehensive_test_config()
        
        if verbose:
            print(f"   ✅ Configuration created successfully")
            print(f"   • source_dir: {config.source_dir}")
            print(f"   • processing_entry_point: {config.processing_entry_point}")
            print(f"   • job_type: {config.job_type}")
            print(f"   • label_name: {config.label_name}")
        
        # Run tests using enhanced 4-level Processing tester
        if verbose:
            print("\n🧪 Running enhanced 4-level Processing validation tests...")
        
        # Create enhanced 4-level processing tester
        tester = ProcessingStepBuilderTest(
            builder_class=TabularPreprocessingStepBuilder,
            config=config,
            spec=PREPROCESSING_TRAINING_SPEC,
            contract=TABULAR_PREPROCESS_CONTRACT,
            step_name="TabularPreprocessingStep",
            verbose=verbose
        )
        
        # Run all tests
        results = tester.run_all_tests()
        
        if verbose:
            print(f"\n✅ Test execution completed")
            print(f"   • Total tests run: {len(results)}")
            print(f"   • Tests passed: {sum(1 for r in results.values() if r.get('passed', False))}")
            print(f"   • Tests failed: {sum(1 for r in results.values() if not r.get('passed', False))}")
            
            # Print level-by-level summary
            _print_level_summary(results)
        
        # Generate score report
        if verbose:
            print("\n📊 Generating score report...")
        
        score_report = score_builder_results(
            results=results,
            builder_name="TabularPreprocessingStepBuilder",
            save_report=save_report,
            output_dir=output_dir,
            generate_chart=generate_chart
        )
        
        # Additional analysis
        if verbose:
            print("\n📈 Additional Analysis:")
            _print_detailed_analysis(score_report, results)
        
        return score_report
        
    except Exception as e:
        print(f"\n❌ Error during scoring test: {e}")
        import traceback
        traceback.print_exc()
        return {}


def _print_level_summary(results: Dict[str, Any]) -> None:
    """
    Print a summary of results organized by the 4 levels.
    
    Args:
        results: Raw test results
    """
    print("\n📊 4-LEVEL TEST SUMMARY:")
    print("-" * 50)
    
    levels = {
        "Level 1 (Interface)": [k for k in results.keys() if k.startswith("level1_")],
        "Level 2 (Specification)": [k for k in results.keys() if k.startswith("level2_")],
        "Level 3 (Path Mapping)": [k for k in results.keys() if k.startswith("level3_")],
        "Level 4 (Integration)": [k for k in results.keys() if k.startswith("level4_")],
        "Legacy/Other": [k for k in results.keys() if not any(k.startswith(f"level{i}_") for i in range(1, 5))]
    }
    
    for level_name, test_names in levels.items():
        if test_names:
            passed = sum(1 for name in test_names if results.get(name, {}).get("passed", False))
            total = len(test_names)
            pass_rate = (passed / total) * 100 if total > 0 else 0
            
            # Performance indicator
            if pass_rate >= 90:
                indicator = "🟢 Excellent"
            elif pass_rate >= 80:
                indicator = "🟡 Good"
            elif pass_rate >= 70:
                indicator = "🟠 Satisfactory"
            elif pass_rate >= 60:
                indicator = "🔴 Needs Work"
            else:
                indicator = "⚫ Poor"
            
            print(f"📁 {level_name}: {passed}/{total} passed ({pass_rate:.1f}%) {indicator}")
            
            # Show failed tests for this level
            failed_tests = [name for name in test_names if not results.get(name, {}).get("passed", False)]
            if failed_tests:
                print(f"   Failed: {', '.join([t.replace('level1_', '').replace('level2_', '').replace('level3_', '').replace('level4_', '') for t in failed_tests])}")


def _print_detailed_analysis(score_report: Dict[str, Any], raw_results: Dict[str, Any]) -> None:
    """
    Print detailed analysis of the scoring results with 4-level focus.
    
    Args:
        score_report: The generated score report
        raw_results: Raw test results
    """
    print("\n" + "=" * 60)
    print("DETAILED 4-LEVEL ANALYSIS")
    print("=" * 60)
    
    # Level-by-level analysis
    levels = score_report.get("levels", {})
    
    print("\n📊 Level Performance Analysis:")
    for level_name, level_data in levels.items():
        display_name = level_name.replace("level", "Level ").replace("_", " ").title()
        score = level_data.get("score", 0)
        passed = level_data.get("passed", 0)
        total = level_data.get("total", 0)
        
        # Performance indicator
        if score >= 90:
            indicator = "🟢 Excellent"
        elif score >= 80:
            indicator = "🟡 Good"
        elif score >= 70:
            indicator = "🟠 Satisfactory"
        elif score >= 60:
            indicator = "🔴 Needs Work"
        else:
            indicator = "⚫ Poor"
        
        print(f"   {display_name}: {score:.1f}% ({passed}/{total}) {indicator}")
        
        # Show failed tests for this level
        failed_tests = [
            test_name for test_name, result in level_data.get("tests", {}).items()
            if not result
        ]
        if failed_tests:
            print(f"      Failed: {', '.join(failed_tests)}")
    
    # Processing-specific pattern analysis
    print("\n🔍 Processing Pattern Analysis:")
    
    # Pattern A vs Pattern B compliance
    pattern_tests = {
        "Step Creation Pattern": ["level1_test_step_creation_pattern_compliance", "level4_test_step_creation_pattern_execution"],
        "Framework Detection": ["level1_test_framework_specific_methods", "level2_test_framework_specific_specifications"],
        "Environment Variables": ["level1_test_environment_variables_method", "level2_test_environment_variable_patterns"],
        "Input/Output Handling": ["level1_test_processing_input_output_methods", "level3_test_processing_input_creation"],
        "Special Patterns": ["level3_test_special_input_handling", "level3_test_file_upload_patterns", "level3_test_local_path_override_patterns"]
    }
    
    for pattern_name, test_names in pattern_tests.items():
        pattern_results = []
        for test_name in test_names:
            if test_name in raw_results:
                pattern_results.append(raw_results[test_name].get("passed", False))
        
        if pattern_results:
            passed_count = sum(pattern_results)
            total_count = len(pattern_results)
            pass_rate = (passed_count / total_count) * 100
            status = "✅ COMPLIANT" if pass_rate >= 80 else "⚠️ NEEDS ATTENTION" if pass_rate >= 60 else "❌ NON-COMPLIANT"
            print(f"   {pattern_name}: {passed_count}/{total_count} ({pass_rate:.1f}%) {status}")
    
    # Critical test analysis
    print("\n🔍 Critical Test Analysis:")
    critical_tests = [
        ("Processor Creation", "level1_test_processor_creation_method"),
        ("Environment Variables", "level2_test_environment_variable_patterns"),
        ("Input Creation", "level3_test_processing_input_creation"),
        ("End-to-End Creation", "level4_test_end_to_end_step_creation"),
        ("Specification Usage", "level2_test_specification_driven_inputs"),
        ("Contract Alignment", "level2_test_contract_path_mapping")
    ]
    
    for test_display_name, test_name in critical_tests:
        if test_name in raw_results:
            result = raw_results[test_name]
            status = "✅ PASSED" if result.get("passed", False) else "❌ FAILED"
            print(f"   {test_display_name}: {status}")
            if not result.get("passed", False) and "error" in result:
                print(f"      Error: {result['error']}")
    
    # Recommendations based on 4-level results
    overall_score = score_report.get("overall", {}).get("score", 0)
    print(f"\n💡 4-Level Processing Recommendations:")
    
    if overall_score >= 90:
        print("   🎉 Excellent work! The Processing step builder meets all pattern requirements.")
        print("   This implementation demonstrates best practices for Processing steps.")
    elif overall_score >= 80:
        print("   👍 Good Processing step implementation with minor areas for improvement.")
        print("   Focus on failed Level 3 and Level 4 tests for pattern compliance.")
    elif overall_score >= 70:
        print("   ⚠️  Satisfactory but needs attention in Processing-specific patterns.")
        print("   Prioritize Level 2 specification compliance and Level 3 path mapping.")
    elif overall_score >= 60:
        print("   🚨 Significant Processing pattern compliance issues detected.")
        print("   Focus on Level 1 interface requirements and processor creation patterns.")
    else:
        print("   🆘 Major Processing step pattern violations detected.")
        print("   Comprehensive refactoring needed - start with Level 1 interface compliance.")
    
    # Specific Processing pattern recommendations
    print(f"\n🔧 Processing Pattern Specific Recommendations:")
    
    # Check for common Processing issues
    if "level1_test_processor_creation_method" in raw_results and not raw_results["level1_test_processor_creation_method"].get("passed", False):
        print("   • Implement _create_processor() method correctly for SKLearn/XGBoost")
    
    if "level2_test_environment_variable_patterns" in raw_results and not raw_results["level2_test_environment_variable_patterns"].get("passed", False):
        print("   • Review environment variable patterns (basic, JSON-serialized, step-specific)")
    
    if "level3_test_special_input_handling" in raw_results and not raw_results["level3_test_special_input_handling"].get("passed", False):
        print("   • Implement special input handling patterns (local path override, file upload)")
    
    if "level4_test_step_creation_pattern_execution" in raw_results and not raw_results["level4_test_step_creation_pattern_execution"].get("passed", False):
        print("   • Ensure compliance with Pattern A (direct ProcessingStep) or Pattern B (processor.run)")


def run_4_level_scoring_comparison_test() -> None:
    """
    Run scoring tests with different configurations to compare 4-level results.
    """
    print("\n🔄" * 30)
    print("4-LEVEL SCORING COMPARISON TEST")
    print("🔄" * 30)
    
    configurations = {
        "minimal": {
            "label_name": "target",
            "processing_entry_point": "tabular_preprocess.py",
            "job_type": "training"
        },
        "standard": {
            "label_name": "target",
            "processing_entry_point": "tabular_preprocess.py", 
            "job_type": "training",
            "train_ratio": 0.7,
            "test_val_ratio": 0.5
        },
        "comprehensive": "full_config"  # Use the comprehensive config
    }
    
    results_comparison = {}
    
    for config_name, config_data in configurations.items():
        print(f"\n📊 Testing with {config_name} configuration...")
        
        try:
            if config_data == "full_config":
                config = create_comprehensive_test_config()
            else:
                # Create minimal config for comparison
                src_dir = Path(__file__).parent.parent.parent.parent / 'src' / 'cursus' / 'steps' / 'scripts'
                config_dict = {
                    **config_data,
                    "region": "NA",
                    "pipeline_name": f"test-{config_name}-pipeline",
                    "pipeline_s3_loc": "s3://test-bucket/test-pipeline",
                    "source_dir": str(src_dir),
                    "author": "test-author",
                    "bucket": "test-bucket",
                    "role": "arn:aws:iam::123456789012:role/TestRole",
                    "service_name": "test-service",
                    "pipeline_version": "1.0.0",
                    "processing_instance_count": 1,
                    "processing_volume_size": 30,
                    "processing_instance_type_large": "ml.m5.xlarge",
                    "processing_instance_type_small": "ml.m5.large",
                    "processing_framework_version": "0.23-1",
                    "use_large_processing_instance": False,
                    "py_version": "py3"
                }
                config = TabularPreprocessingConfig(**config_dict)
            
            # Run tests with 4-level tester
            tester = ProcessingStepBuilderTest(
                builder_class=TabularPreprocessingStepBuilder,
                config=config,
                spec=PREPROCESSING_TRAINING_SPEC,
                contract=TABULAR_PREPROCESS_CONTRACT,
                step_name=f"TabularPreprocessingStep_{config_name}",
                verbose=False
            )
            test_results = tester.run_all_tests()
            
            # Score results
            scorer = StepBuilderScorer(test_results)
            score_report = scorer.generate_report()
            
            # Calculate level-specific scores
            level_scores = {}
            for level_num in range(1, 5):
                level_tests = [k for k in test_results.keys() if k.startswith(f"level{level_num}_")]
                if level_tests:
                    level_passed = sum(1 for k in level_tests if test_results[k].get("passed", False))
                    level_total = len(level_tests)
                    level_scores[f"Level {level_num}"] = (level_passed / level_total) * 100 if level_total > 0 else 0
            
            results_comparison[config_name] = {
                "overall_score": score_report["overall"]["score"],
                "rating": score_report["overall"]["rating"],
                "pass_rate": score_report["overall"]["pass_rate"],
                "level_scores": level_scores
            }
            
            print(f"   Overall Score: {score_report['overall']['score']:.1f} ({score_report['overall']['rating']})")
            for level, score in level_scores.items():
                print(f"   {level}: {score:.1f}%")
            
        except Exception as e:
            print(f"   ❌ Error with {config_name} config: {e}")
            results_comparison[config_name] = {"error": str(e)}
    
    # Print comparison summary
    print(f"\n📈 4-Level Configuration Comparison Summary:")
    print("-" * 70)
    print(f"{'Config':<15} {'Overall':<8} {'Level 1':<8} {'Level 2':<8} {'Level 3':<8} {'Level 4':<8}")
    print("-" * 70)
    
    for config_name, result in results_comparison.items():
        if "error" in result:
            print(f"{config_name:<15}: ERROR - {result['error']}")
        else:
            overall = result['overall_score']
            level_scores = result['level_scores']
            l1 = level_scores.get('Level 1', 0)
            l2 = level_scores.get('Level 2', 0)
            l3 = level_scores.get('Level 3', 0)
            l4 = level_scores.get('Level 4', 0)
            print(f"{config_name:<15} {overall:>6.1f}%  {l1:>6.1f}%  {l2:>6.1f}%  {l3:>6.1f}%  {l4:>6.1f}%")


def main():
    """Main function to run the 4-level scoring tests."""
    print("🎯 TabularPreprocessingStepBuilder 4-Level Scoring Test Suite")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path(__file__).parent / "scoring_reports"
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Run comprehensive scoring test
        print("\n1️⃣ Running Comprehensive 4-Level Scoring Test...")
        score_report = run_comprehensive_scoring_test(
            save_report=True,
            generate_chart=True,
            output_dir=str(output_dir),
            verbose=True
        )
        
        if score_report:
            print(f"\n📁 Reports saved to: {output_dir}")
            
            # List generated files
            report_files = list(output_dir.glob("*"))
            if report_files:
                print("   Generated files:")
                for file_path in report_files:
                    print(f"   • {file_path.name}")
        
        # Run comparison test
        print("\n2️⃣ Running 4-Level Configuration Comparison Test...")
        run_4_level_scoring_comparison_test()
        
        print(f"\n🎉 4-Level scoring tests completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
