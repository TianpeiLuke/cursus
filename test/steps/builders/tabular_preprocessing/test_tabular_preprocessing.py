"""
Tabular Preprocessing Step Builder Test using Existing Validation Infrastructure.

This test leverages the existing UniversalStepBuilderTest and ProcessingStepBuilderTest
classes from src/cursus/validation/builders to provide comprehensive validation.
"""

import sys
import os
import unittest
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

try:
    # Import the existing validation infrastructure
    from cursus.validation.builders.universal_test import UniversalStepBuilderTest
    from cursus.validation.builders.variants.processing_test import ProcessingStepBuilderTest
    
    # Import the components we need to test
    from cursus.steps.builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
    from cursus.steps.configs.config_tabular_preprocessing_step import TabularPreprocessingConfig
    from cursus.steps.specs.preprocessing_training_spec import PREPROCESSING_TRAINING_SPEC
    from cursus.steps.contracts.tabular_preprocess_contract import TABULAR_PREPROCESS_CONTRACT
    
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    IMPORTS_AVAILABLE = False


class TestTabularPreprocessingWithExistingValidators(unittest.TestCase):
    """
    Test TabularPreprocessingStepBuilder using the existing validation infrastructure.
    
    This test class demonstrates how to use the existing UniversalStepBuilderTest
    and ProcessingStepBuilderTest classes to validate the TabularPreprocessingStepBuilder.
    """
    
    def setUp(self):
        """Set up test configuration."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")
        
        # Create comprehensive test configuration
        self.config = TabularPreprocessingConfig(
            # Essential User Inputs (Tier 1)
            label_name="target",
            
            # System Fields with Defaults (Tier 2)
            processing_entry_point="tabular_preprocess.py",
            job_type="training",
            train_ratio=0.7,
            test_val_ratio=0.5,
            
            # Base configuration fields
            region="NA",
            pipeline_name="test-tabular-preprocessing-pipeline",
            source_dir="src/cursus/steps/scripts",
            
            # Processing configuration
            processing_instance_count=1,
            processing_volume_size=30,
            processing_instance_type_large="ml.m5.xlarge",
            processing_instance_type_small="ml.m5.large",
            processing_framework_version="0.23-1",
            use_large_processing_instance=False,
            py_version="py3",
            
            # Optional column configurations
            categorical_columns=["category_col1", "category_col2"],
            numerical_columns=["num_col1", "num_col2"],
            text_columns=["text_col1"],
            date_columns=["date_col1"],
            
            # Additional processing fields
            processing_instance_type="ml.m5.large",
            image_uri="683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            pipeline_s3_loc="s3://test-bucket/pipeline",
            
            # Step configuration
            step_name="TabularPreprocessingStep",
            depends_on=[]
        )
    
    def test_universal_step_builder_validation(self):
        """Test using UniversalStepBuilderTest."""
        print("\n" + "="*80)
        print("TESTING WITH UNIVERSAL STEP BUILDER TEST")
        print("="*80)
        
        # Create universal tester
        tester = UniversalStepBuilderTest(
            builder_class=TabularPreprocessingStepBuilder,
            config=self.config,
            spec=PREPROCESSING_TRAINING_SPEC,
            contract=TABULAR_PREPROCESS_CONTRACT,
            step_name="TabularPreprocessingStep",
            verbose=True
        )
        
        # Run all tests
        results = tester.run_all_tests()
        
        # Validate key results
        self.assertTrue(results.get("test_inheritance", {}).get("passed", False), 
                       "Inheritance test should pass")
        self.assertTrue(results.get("test_required_methods", {}).get("passed", False), 
                       "Required methods test should pass")
        
        # Print summary
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result.get("passed", False))
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\nUniversal Test Results: {passed_tests}/{total_tests} tests passed ({pass_rate:.1f}%)")
        
        # Report failed tests
        failed_tests = {k: v for k, v in results.items() if not v.get("passed", False)}
        if failed_tests:
            print("\nFailed Tests:")
            for test_name, result in failed_tests.items():
                print(f"❌ {test_name}: {result.get('error', 'Unknown error')}")
        
        return results
    
    def test_processing_step_builder_validation(self):
        """Test using ProcessingStepBuilderTest."""
        print("\n" + "="*80)
        print("TESTING WITH PROCESSING STEP BUILDER TEST")
        print("="*80)
        
        # Create processing-specific tester
        tester = ProcessingStepBuilderTest(
            builder_class=TabularPreprocessingStepBuilder,
            config=self.config,
            spec=PREPROCESSING_TRAINING_SPEC,
            contract=TABULAR_PREPROCESS_CONTRACT,
            step_name="TabularPreprocessingStep",
            verbose=True
        )
        
        # Run all tests
        results = tester.run_all_tests()
        
        # Validate processing-specific results
        processing_tests = [
            "test_processor_creation",
            "test_processing_inputs_outputs",
            "test_processing_job_arguments",
            "test_environment_variables_processing",
            "test_property_files_configuration",
            "test_processing_code_handling"
        ]
        
        for test_name in processing_tests:
            if test_name in results:
                test_result = results[test_name]
                print(f"Processing Test {test_name}: {'✅ PASSED' if test_result.get('passed', False) else '❌ FAILED'}")
                if not test_result.get('passed', False):
                    print(f"  Error: {test_result.get('error', 'Unknown error')}")
        
        # Print summary
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result.get("passed", False))
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\nProcessing Test Results: {passed_tests}/{total_tests} tests passed ({pass_rate:.1f}%)")
        
        return results
    
    def test_combined_validation(self):
        """Test combining both universal and processing-specific validation."""
        print("\n" + "="*80)
        print("COMBINED VALIDATION TEST")
        print("="*80)
        
        # Run universal tests
        universal_results = self.test_universal_step_builder_validation()
        
        # Run processing-specific tests
        processing_results = self.test_processing_step_builder_validation()
        
        # Combine results
        all_results = {}
        all_results.update(universal_results)
        all_results.update(processing_results)
        
        # Calculate overall statistics
        total_tests = len(all_results)
        passed_tests = sum(1 for result in all_results.values() if result.get("passed", False))
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n" + "="*80)
        print(f"OVERALL RESULTS: {passed_tests}/{total_tests} tests passed ({pass_rate:.1f}%)")
        print("="*80)
        
        # Ensure minimum pass rate
        self.assertGreaterEqual(pass_rate, 70.0, 
                               f"Pass rate {pass_rate:.1f}% is below minimum threshold of 70%")
        
        return all_results


class TestTabularPreprocessingMultipleJobTypes(unittest.TestCase):
    """Test TabularPreprocessingStepBuilder with multiple job types."""
    
    def setUp(self):
        """Set up test configurations for different job types."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")
        
        # Base configuration
        self.base_config = {
            "label_name": "target",
            "processing_entry_point": "tabular_preprocess.py",
            "train_ratio": 0.7,
            "test_val_ratio": 0.5,
            "region": "NA",
            "pipeline_name": "test-tabular-preprocessing-pipeline",
            "source_dir": "src/cursus/steps/scripts",
            "processing_instance_count": 1,
            "processing_volume_size": 30,
            "processing_instance_type_large": "ml.m5.xlarge",
            "processing_instance_type_small": "ml.m5.large",
            "processing_framework_version": "0.23-1",
            "use_large_processing_instance": False,
            "py_version": "py3",
            "categorical_columns": ["category_col1", "category_col2"],
            "numerical_columns": ["num_col1", "num_col2"],
            "text_columns": ["text_col1"],
            "date_columns": ["date_col1"],
            "processing_instance_type": "ml.m5.large",
            "image_uri": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
            "role_arn": "arn:aws:iam::123456789012:role/SageMakerRole",
            "pipeline_s3_loc": "s3://test-bucket/pipeline",
            "step_name": "TabularPreprocessingStep",
            "depends_on": []
        }
        
        # Job types to test
        self.job_types = ["training", "validation", "testing", "calibration"]
    
    def test_all_job_types(self):
        """Test TabularPreprocessingStepBuilder with all job types."""
        print("\n" + "="*80)
        print("TESTING MULTIPLE JOB TYPES")
        print("="*80)
        
        all_results = {}
        
        for job_type in self.job_types:
            print(f"\n🔍 Testing job_type: {job_type}")
            print("-" * 40)
            
            # Create config for this job type
            config_dict = self.base_config.copy()
            config_dict["job_type"] = job_type
            config = TabularPreprocessingConfig(**config_dict)
            
            # Create tester
            tester = UniversalStepBuilderTest(
                builder_class=TabularPreprocessingStepBuilder,
                config=config,
                spec=PREPROCESSING_TRAINING_SPEC,
                contract=TABULAR_PREPROCESS_CONTRACT,
                step_name=f"TabularPreprocessingStep_{job_type}",
                verbose=False  # Reduce verbosity for multiple tests
            )
            
            # Run tests
            results = tester.run_all_tests()
            
            # Store results
            all_results[job_type] = results
            
            # Calculate pass rate for this job type
            total_tests = len(results)
            passed_tests = sum(1 for result in results.values() if result.get("passed", False))
            pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            
            print(f"Job type {job_type}: {passed_tests}/{total_tests} tests passed ({pass_rate:.1f}%)")
            
            # Check critical tests
            critical_tests = ["test_inheritance", "test_required_methods"]
            for test_name in critical_tests:
                if test_name in results:
                    test_result = results[test_name]
                    status = "✅ PASSED" if test_result.get("passed", False) else "❌ FAILED"
                    print(f"  {test_name}: {status}")
        
        # Overall summary
        print(f"\n" + "="*40)
        print("JOB TYPE TESTING SUMMARY")
        print("="*40)
        
        for job_type, results in all_results.items():
            total_tests = len(results)
            passed_tests = sum(1 for result in results.values() if result.get("passed", False))
            pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            print(f"{job_type:12}: {passed_tests:2}/{total_tests:2} tests ({pass_rate:5.1f}%)")
        
        return all_results


def run_comprehensive_test():
    """Run comprehensive test suite using existing validators."""
    print("🧪" * 40)
    print("COMPREHENSIVE TABULAR PREPROCESSING TESTS")
    print("Using Existing Validation Infrastructure")
    print("🧪" * 40)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(TestTabularPreprocessingWithExistingValidators('test_universal_step_builder_validation'))
    suite.addTest(TestTabularPreprocessingWithExistingValidators('test_processing_step_builder_validation'))
    suite.addTest(TestTabularPreprocessingWithExistingValidators('test_combined_validation'))
    suite.addTest(TestTabularPreprocessingMultipleJobTypes('test_all_job_types'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print final summary
    print("\n" + "🎯" * 40)
    print("FINAL TEST SUMMARY")
    print("🎯" * 40)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result


if __name__ == '__main__':
    if IMPORTS_AVAILABLE:
        run_comprehensive_test()
    else:
        print("❌ Cannot run tests - required imports not available")
        print("Please ensure the following components are implemented:")
        print("- TabularPreprocessingStepBuilder")
        print("- TabularPreprocessingConfig")
        print("- PREPROCESSING_TRAINING_SPEC")
        print("- TABULAR_PREPROCESS_CONTRACT")
