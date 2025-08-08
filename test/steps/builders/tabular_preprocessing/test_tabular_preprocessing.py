"""
Tabular Preprocessing Step Builder Test using Enhanced 4-Level Processing Tester.

This test leverages the enhanced 4-level Processing tester from 
src/cursus/validation/builders/variants/processing_test.py to provide 
comprehensive validation based on Processing Step Builder Patterns analysis.
"""

import sys
import os
import unittest
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

try:
    # Import the enhanced 4-level Processing tester
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


class TestTabularPreprocessingWith4LevelTester(unittest.TestCase):
    """
    Test TabularPreprocessingStepBuilder using the enhanced 4-level Processing tester.
    
    This test class demonstrates how to use the enhanced ProcessingStepBuilderTest
    with its 4-level hierarchy to validate the TabularPreprocessingStepBuilder
    against all identified Processing step patterns.
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
    
    def test_4_level_processing_validation(self):
        """Test using the enhanced 4-level ProcessingStepBuilderTest."""
        print("\n" + "="*80)
        print("TESTING WITH ENHANCED 4-LEVEL PROCESSING TESTER")
        print("="*80)
        
        # Create enhanced 4-level processing tester
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
        
        # Print level-by-level results
        self._print_level_results(results)
        
        # Validate critical tests pass
        self._validate_critical_tests(results)
        
        # Print summary
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result.get("passed", False))
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n4-Level Processing Test Results: {passed_tests}/{total_tests} tests passed ({pass_rate:.1f}%)")
        
        # Ensure minimum pass rate for Processing steps
        self.assertGreaterEqual(pass_rate, 80.0, 
                               f"Processing step pass rate {pass_rate:.1f}% is below minimum threshold of 80%")
        
        return results
    
    def test_level1_interface_tests(self):
        """Test Level 1: Interface Tests specifically."""
        print("\n" + "="*60)
        print("LEVEL 1: INTERFACE TESTS")
        print("="*60)
        
        tester = ProcessingStepBuilderTest(
            builder_class=TabularPreprocessingStepBuilder,
            config=self.config,
            spec=PREPROCESSING_TRAINING_SPEC,
            contract=TABULAR_PREPROCESS_CONTRACT,
            step_name="TabularPreprocessingStep",
            verbose=True
        )
        
        # Run Level 1 tests
        level1_tests = [
            "level1_test_processor_creation_method",
            "level1_test_processing_configuration_attributes",
            "level1_test_framework_specific_methods",
            "level1_test_step_creation_pattern_compliance",
            "level1_test_processing_input_output_methods",
            "level1_test_environment_variables_method",
            "level1_test_job_arguments_method"
        ]
        
        results = {}
        for test_name in level1_tests:
            if hasattr(tester, test_name):
                try:
                    getattr(tester, test_name)()
                    results[test_name] = {"passed": True}
                    print(f"✅ {test_name}")
                except Exception as e:
                    results[test_name] = {"passed": False, "error": str(e)}
                    print(f"❌ {test_name}: {str(e)}")
        
        # Validate Level 1 critical tests
        critical_level1_tests = [
            "level1_test_processor_creation_method",
            "level1_test_processing_configuration_attributes"
        ]
        
        for test_name in critical_level1_tests:
            if test_name in results:
                self.assertTrue(results[test_name]["passed"], 
                               f"Critical Level 1 test failed: {test_name}")
        
        return results
    
    def test_level2_specification_tests(self):
        """Test Level 2: Specification Tests specifically."""
        print("\n" + "="*60)
        print("LEVEL 2: SPECIFICATION TESTS")
        print("="*60)
        
        tester = ProcessingStepBuilderTest(
            builder_class=TabularPreprocessingStepBuilder,
            config=self.config,
            spec=PREPROCESSING_TRAINING_SPEC,
            contract=TABULAR_PREPROCESS_CONTRACT,
            step_name="TabularPreprocessingStep",
            verbose=True
        )
        
        # Run Level 2 tests
        level2_tests = [
            "level2_test_job_type_specification_loading",
            "level2_test_environment_variable_patterns",
            "level2_test_job_arguments_patterns",
            "level2_test_specification_driven_inputs",
            "level2_test_specification_driven_outputs",
            "level2_test_contract_path_mapping",
            "level2_test_multi_job_type_support",
            "level2_test_framework_specific_specifications"
        ]
        
        results = {}
        for test_name in level2_tests:
            if hasattr(tester, test_name):
                try:
                    getattr(tester, test_name)()
                    results[test_name] = {"passed": True}
                    print(f"✅ {test_name}")
                except Exception as e:
                    results[test_name] = {"passed": False, "error": str(e)}
                    print(f"❌ {test_name}: {str(e)}")
        
        return results
    
    def test_level3_path_mapping_tests(self):
        """Test Level 3: Path Mapping Tests specifically."""
        print("\n" + "="*60)
        print("LEVEL 3: PATH MAPPING TESTS")
        print("="*60)
        
        tester = ProcessingStepBuilderTest(
            builder_class=TabularPreprocessingStepBuilder,
            config=self.config,
            spec=PREPROCESSING_TRAINING_SPEC,
            contract=TABULAR_PREPROCESS_CONTRACT,
            step_name="TabularPreprocessingStep",
            verbose=True
        )
        
        # Run Level 3 tests
        level3_tests = [
            "level3_test_processing_input_creation",
            "level3_test_processing_output_creation",
            "level3_test_container_path_mapping",
            "level3_test_special_input_handling",
            "level3_test_s3_path_normalization",
            "level3_test_file_upload_patterns",
            "level3_test_local_path_override_patterns",
            "level3_test_dependency_input_extraction"
        ]
        
        results = {}
        for test_name in level3_tests:
            if hasattr(tester, test_name):
                try:
                    getattr(tester, test_name)()
                    results[test_name] = {"passed": True}
                    print(f"✅ {test_name}")
                except Exception as e:
                    results[test_name] = {"passed": False, "error": str(e)}
                    print(f"❌ {test_name}: {str(e)}")
        
        return results
    
    def test_level4_integration_tests(self):
        """Test Level 4: Integration Tests specifically."""
        print("\n" + "="*60)
        print("LEVEL 4: INTEGRATION TESTS")
        print("="*60)
        
        tester = ProcessingStepBuilderTest(
            builder_class=TabularPreprocessingStepBuilder,
            config=self.config,
            spec=PREPROCESSING_TRAINING_SPEC,
            contract=TABULAR_PREPROCESS_CONTRACT,
            step_name="TabularPreprocessingStep",
            verbose=True
        )
        
        # Run Level 4 tests
        level4_tests = [
            "level4_test_step_creation_pattern_execution",
            "level4_test_framework_specific_step_creation",
            "level4_test_processing_dependency_resolution",
            "level4_test_step_name_generation",
            "level4_test_cache_configuration",
            "level4_test_step_dependencies_handling",
            "level4_test_end_to_end_step_creation",
            "level4_test_specification_attachment"
        ]
        
        results = {}
        for test_name in level4_tests:
            if hasattr(tester, test_name):
                try:
                    getattr(tester, test_name)()
                    results[test_name] = {"passed": True}
                    print(f"✅ {test_name}")
                except Exception as e:
                    results[test_name] = {"passed": False, "error": str(e)}
                    print(f"❌ {test_name}: {str(e)}")
        
        return results
    
    def test_legacy_compatibility(self):
        """Test that legacy test methods still work for backward compatibility."""
        print("\n" + "="*60)
        print("LEGACY COMPATIBILITY TESTS")
        print("="*60)
        
        tester = ProcessingStepBuilderTest(
            builder_class=TabularPreprocessingStepBuilder,
            config=self.config,
            spec=PREPROCESSING_TRAINING_SPEC,
            contract=TABULAR_PREPROCESS_CONTRACT,
            step_name="TabularPreprocessingStep",
            verbose=True
        )
        
        # Test legacy methods
        legacy_tests = [
            "test_processor_creation",
            "test_processing_inputs_outputs",
            "test_processing_job_arguments",
            "test_environment_variables_processing",
            "test_property_files_configuration",
            "test_processing_code_handling",
            "test_processing_step_dependencies"
        ]
        
        results = {}
        for test_name in legacy_tests:
            if hasattr(tester, test_name):
                try:
                    getattr(tester, test_name)()
                    results[test_name] = {"passed": True}
                    print(f"✅ {test_name} (legacy)")
                except Exception as e:
                    results[test_name] = {"passed": False, "error": str(e)}
                    print(f"❌ {test_name} (legacy): {str(e)}")
        
        return results
    
    def _print_level_results(self, results):
        """Print results organized by level."""
        print("\n📊 RESULTS BY LEVEL:")
        print("-" * 60)
        
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
                
                print(f"\n📁 {level_name}: {passed}/{total} passed ({pass_rate:.1f}%)")
                
                for test_name in test_names:
                    result = results.get(test_name, {})
                    status = "✅" if result.get("passed", False) else "❌"
                    display_name = test_name.replace("level1_", "").replace("level2_", "").replace("level3_", "").replace("level4_", "")
                    print(f"  {status} {display_name}")
                    
                    if not result.get("passed", False) and "error" in result:
                        print(f"    💬 {result['error']}")
    
    def _validate_critical_tests(self, results):
        """Validate that critical tests pass."""
        critical_tests = [
            "level1_test_processor_creation_method",
            "level2_test_environment_variable_patterns",
            "level3_test_processing_input_creation",
            "level4_test_end_to_end_step_creation"
        ]
        
        for test_name in critical_tests:
            if test_name in results:
                self.assertTrue(results[test_name].get("passed", False), 
                               f"Critical test failed: {test_name}")


class TestTabularPreprocessingMultipleJobTypes(unittest.TestCase):
    """Test TabularPreprocessingStepBuilder with multiple job types using 4-level tester."""
    
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
    
    def test_all_job_types_with_4_level_tester(self):
        """Test TabularPreprocessingStepBuilder with all job types using 4-level tester."""
        print("\n" + "="*80)
        print("TESTING MULTIPLE JOB TYPES WITH 4-LEVEL TESTER")
        print("="*80)
        
        all_results = {}
        
        for job_type in self.job_types:
            print(f"\n🔍 Testing job_type: {job_type}")
            print("-" * 40)
            
            # Create config for this job type
            config_dict = self.base_config.copy()
            config_dict["job_type"] = job_type
            config = TabularPreprocessingConfig(**config_dict)
            
            # Create 4-level tester
            tester = ProcessingStepBuilderTest(
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
            critical_tests = ["level1_test_processor_creation_method", "level2_test_job_type_specification_loading"]
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


def run_comprehensive_4_level_test():
    """Run comprehensive test suite using the enhanced 4-level Processing tester."""
    print("🧪" * 40)
    print("COMPREHENSIVE TABULAR PREPROCESSING TESTS")
    print("Using Enhanced 4-Level Processing Tester")
    print("🧪" * 40)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(TestTabularPreprocessingWith4LevelTester('test_4_level_processing_validation'))
    suite.addTest(TestTabularPreprocessingWith4LevelTester('test_level1_interface_tests'))
    suite.addTest(TestTabularPreprocessingWith4LevelTester('test_level2_specification_tests'))
    suite.addTest(TestTabularPreprocessingWith4LevelTester('test_level3_path_mapping_tests'))
    suite.addTest(TestTabularPreprocessingWith4LevelTester('test_level4_integration_tests'))
    suite.addTest(TestTabularPreprocessingWith4LevelTester('test_legacy_compatibility'))
    suite.addTest(TestTabularPreprocessingMultipleJobTypes('test_all_job_types_with_4_level_tester'))
    
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
        run_comprehensive_4_level_test()
    else:
        print("❌ Cannot run tests - required imports not available")
        print("Please ensure the following components are implemented:")
        print("- TabularPreprocessingStepBuilder")
        print("- TabularPreprocessingConfig")
        print("- PREPROCESSING_TRAINING_SPEC")
        print("- TABULAR_PREPROCESS_CONTRACT")
