"""
Scoring test program for TabularPreprocessingStepBuilder.

This module provides comprehensive scoring and evaluation of the TabularPreprocessingStepBuilder
using the scoring system from src/cursus/validation/builders/scoring.py.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

try:
    from cursus.validation.builders.test_factory import UniversalStepBuilderTestFactory
    from cursus.validation.builders.scoring import score_builder_results, StepBuilderScorer
    from cursus.steps.builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
    from cursus.steps.configs.config_tabular_preprocessing_step import TabularPreprocessingConfig
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
    Run comprehensive scoring test for TabularPreprocessingStepBuilder.
    
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
        
        # Run tests using UniversalStepBuilderTestFactory
        if verbose:
            print("\n🧪 Running comprehensive validation tests...")
        
        # Create tester using factory
        tester = UniversalStepBuilderTestFactory.create_tester(
            TabularPreprocessingStepBuilder,
            verbose=verbose
        )
        
        # Run all tests
        results = tester.run_all_tests()
        
        if verbose:
            print(f"\n✅ Test execution completed")
            print(f"   • Total tests run: {len(results)}")
            print(f"   • Tests passed: {sum(1 for r in results.values() if r.get('passed', False))}")
            print(f"   • Tests failed: {sum(1 for r in results.values() if not r.get('passed', False))}")
        
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


def _print_detailed_analysis(score_report: Dict[str, Any], raw_results: Dict[str, Any]) -> None:
    """
    Print detailed analysis of the scoring results.
    
    Args:
        score_report: The generated score report
        raw_results: Raw test results
    """
    print("\n" + "=" * 60)
    print("DETAILED ANALYSIS")
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
    
    # Critical test analysis
    print("\n🔍 Critical Test Analysis:")
    critical_tests = [
        "test_inheritance",
        "test_required_methods", 
        "test_specification_usage",
        "test_contract_alignment",
        "test_step_creation"
    ]
    
    for test_name in critical_tests:
        if test_name in raw_results:
            result = raw_results[test_name]
            status = "✅ PASSED" if result.get("passed", False) else "❌ FAILED"
            print(f"   {test_name}: {status}")
            if not result.get("passed", False) and "error" in result:
                print(f"      Error: {result['error']}")
    
    # Recommendations
    overall_score = score_report.get("overall", {}).get("score", 0)
    print(f"\n💡 Recommendations:")
    
    if overall_score >= 90:
        print("   🎉 Excellent work! The step builder meets all quality standards.")
        print("   Consider this implementation as a reference for other step builders.")
    elif overall_score >= 80:
        print("   👍 Good implementation with minor areas for improvement.")
        print("   Focus on the failed tests to achieve excellence.")
    elif overall_score >= 70:
        print("   ⚠️  Satisfactory but needs attention in several areas.")
        print("   Prioritize fixing Level 3 and Level 4 issues.")
    elif overall_score >= 60:
        print("   🚨 Significant improvements needed.")
        print("   Focus on basic interface and specification compliance first.")
    else:
        print("   🆘 Major issues detected. Comprehensive refactoring recommended.")
        print("   Start with Level 1 interface tests and work upward.")


def run_scoring_comparison_test() -> None:
    """
    Run scoring tests with different configurations to compare results.
    """
    print("\n🔄" * 30)
    print("SCORING COMPARISON TEST")
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
            
            # Run tests
            tester = UniversalStepBuilderTestFactory.create_tester(
                TabularPreprocessingStepBuilder,
                verbose=False
            )
            test_results = tester.run_all_tests()
            
            # Score results
            scorer = StepBuilderScorer(test_results)
            score_report = scorer.generate_report()
            
            results_comparison[config_name] = {
                "score": score_report["overall"]["score"],
                "rating": score_report["overall"]["rating"],
                "pass_rate": score_report["overall"]["pass_rate"]
            }
            
            print(f"   Score: {score_report['overall']['score']:.1f} ({score_report['overall']['rating']})")
            
        except Exception as e:
            print(f"   ❌ Error with {config_name} config: {e}")
            results_comparison[config_name] = {"error": str(e)}
    
    # Print comparison summary
    print(f"\n📈 Configuration Comparison Summary:")
    print("-" * 50)
    for config_name, result in results_comparison.items():
        if "error" in result:
            print(f"{config_name:15}: ERROR - {result['error']}")
        else:
            print(f"{config_name:15}: {result['score']:5.1f} ({result['rating']}) - {result['pass_rate']:.1f}% pass rate")


def main():
    """Main function to run the scoring tests."""
    print("🎯 TabularPreprocessingStepBuilder Scoring Test Suite")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(__file__).parent / "scoring_reports"
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Run comprehensive scoring test
        print("\n1️⃣ Running Comprehensive Scoring Test...")
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
        print("\n2️⃣ Running Configuration Comparison Test...")
        run_scoring_comparison_test()
        
        print(f"\n🎉 Scoring tests completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
