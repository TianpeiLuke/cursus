"""
Test script to demonstrate the enhanced pattern-based scoring system.

This script shows how the scoring system automatically detects test levels
using different strategies without requiring manual TEST_LEVEL_MAP updates.
"""

import sys
import unittest
from pathlib import Path
from typing import Dict, Any

from cursus.validation.builders.scoring import StepBuilderScorer, score_builder_results

class TestPatternBasedScoring(unittest.TestCase):
    """Test cases for the enhanced pattern-based scoring system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_results = self._create_sample_test_results()
        self.processing_results = self._create_processing_variant_results()
        self.mixed_results = self._create_mixed_scenario_results()
    
    def _create_sample_test_results(self) -> Dict[str, Dict[str, Any]]:
        """Create sample test results to demonstrate pattern-based detection."""
        return {
            # Explicit level prefix tests (Processing variant style)
            "level1_test_processor_creation_method": {"passed": True},
            "level1_test_processing_configuration_attributes": {"passed": True},
            "level1_test_framework_specific_methods": {"passed": False, "error": "Framework not detected"},
            "level1_test_step_creation_pattern_compliance": {"passed": True},
            
            "level2_test_job_type_specification_loading": {"passed": True},
            "level2_test_environment_variable_patterns": {"passed": True},
            "level2_test_specification_driven_inputs": {"passed": False, "error": "Spec not found"},
            "level2_test_contract_path_mapping": {"passed": True},
            
            "level3_test_processing_input_creation": {"passed": True},
            "level3_test_processing_output_creation": {"passed": True},
            "level3_test_special_input_handling": {"passed": False, "error": "Special patterns not implemented"},
            "level3_test_s3_path_normalization": {"passed": True},
            
            "level4_test_step_creation_pattern_execution": {"passed": True},
            "level4_test_end_to_end_step_creation": {"passed": True},
            "level4_test_processing_dependency_resolution": {"passed": False, "error": "Dependency resolution failed"},
            "level4_test_specification_attachment": {"passed": True},
            
            # Legacy tests (keyword-based detection)
            "test_inheritance": {"passed": True},
            "test_processor_creation": {"passed": True},
            "test_specification_usage": {"passed": True},
            "test_environment_variable_handling": {"passed": False, "error": "Environment vars not set"},
            "test_input_path_mapping": {"passed": True},
            "test_processing_inputs_outputs": {"passed": True},
            "test_dependency_resolution": {"passed": True},
            "test_step_creation": {"passed": True},
            
            # Tests that should be detected by keywords
            "test_custom_processor_interface": {"passed": True},  # Should be Level 1
            "test_job_arguments_configuration": {"passed": True},  # Should be Level 2
            "test_output_path_validation": {"passed": False, "error": "Invalid output path"},  # Should be Level 3
            "test_integration_with_dependencies": {"passed": True},  # Should be Level 4
            
            # Undetected test (should not be assigned to any level)
            "test_random_functionality": {"passed": True},
        }
    
    def _create_processing_variant_results(self) -> Dict[str, Dict[str, Any]]:
        """Create Processing variant style test results."""
        return {
            # All Processing variant tests use explicit level prefixes
            "level1_test_processor_creation_method": {"passed": True},
            "level1_test_processing_configuration_attributes": {"passed": True},
            "level1_test_framework_specific_methods": {"passed": True},
            "level1_test_step_creation_pattern_compliance": {"passed": True},
            "level1_test_processing_input_output_methods": {"passed": True},
            "level1_test_environment_variables_method": {"passed": True},
            "level1_test_job_arguments_method": {"passed": True},
            
            "level2_test_job_type_specification_loading": {"passed": True},
            "level2_test_environment_variable_patterns": {"passed": True},
            "level2_test_job_arguments_patterns": {"passed": True},
            "level2_test_specification_driven_inputs": {"passed": True},
            "level2_test_specification_driven_outputs": {"passed": True},
            "level2_test_contract_path_mapping": {"passed": True},
            "level2_test_multi_job_type_support": {"passed": True},
            "level2_test_framework_specific_specifications": {"passed": True},
            
            "level3_test_processing_input_creation": {"passed": True},
            "level3_test_processing_output_creation": {"passed": True},
            "level3_test_container_path_mapping": {"passed": True},
            "level3_test_special_input_handling": {"passed": True},
            "level3_test_s3_path_normalization": {"passed": True},
            "level3_test_file_upload_patterns": {"passed": True},
            "level3_test_local_path_override_patterns": {"passed": True},
            "level3_test_dependency_input_extraction": {"passed": True},
            
            "level4_test_step_creation_pattern_execution": {"passed": True},
            "level4_test_framework_specific_step_creation": {"passed": True},
            "level4_test_processing_dependency_resolution": {"passed": True},
            "level4_test_step_name_generation": {"passed": True},
            "level4_test_cache_configuration": {"passed": True},
            "level4_test_step_dependencies_handling": {"passed": True},
            "level4_test_end_to_end_step_creation": {"passed": True},
            "level4_test_specification_attachment": {"passed": True},
        }
    
    def _create_mixed_scenario_results(self) -> Dict[str, Dict[str, Any]]:
        """Create mixed scenario with different naming conventions."""
        return {
            # New explicit prefix tests
            "level1_test_new_interface_method": {"passed": True},
            "level2_test_new_specification_feature": {"passed": True},
            "level3_test_new_path_mapping": {"passed": False, "error": "Path mapping failed"},
            "level4_test_new_integration_feature": {"passed": True},
            
            # Legacy tests with keyword detection
            "test_inheritance": {"passed": True},
            "test_processor_creation": {"passed": True},
            "test_specification_usage": {"passed": False, "error": "Spec not loaded"},
            "test_environment_variable_handling": {"passed": True},
            "test_input_path_mapping": {"passed": True},
            "test_output_path_mapping": {"passed": True},
            "test_dependency_resolution": {"passed": True},
            "test_step_creation": {"passed": True},
            
            # Descriptive tests that should be detected by keywords
            "test_custom_processor_interface_validation": {"passed": True},  # Level 1
            "test_job_arguments_specification_loading": {"passed": True},    # Level 2
            "test_processing_input_creation_validation": {"passed": True},   # Level 3
            "test_end_to_end_integration_workflow": {"passed": True},        # Level 4
            
            # Edge case: test that might not be detected
            "test_utility_helper_function": {"passed": True},
        }
    
    def test_explicit_prefix_detection(self):
        """Test that explicit level prefixes are correctly detected."""
        scorer = StepBuilderScorer(self.sample_results)
        
        # Test explicit prefix detection
        self.assertEqual(scorer._detect_level_from_test_name("level1_test_processor_creation_method"), "level1_interface")
        self.assertEqual(scorer._detect_level_from_test_name("level2_test_job_type_specification_loading"), "level2_specification")
        self.assertEqual(scorer._detect_level_from_test_name("level3_test_processing_input_creation"), "level3_path_mapping")
        self.assertEqual(scorer._detect_level_from_test_name("level4_test_step_creation_pattern_execution"), "level4_integration")
    
    def test_keyword_based_detection(self):
        """Test that keyword-based detection works correctly."""
        scorer = StepBuilderScorer(self.sample_results)
        
        # Test keyword-based detection
        self.assertEqual(scorer._detect_level_from_test_name("test_inheritance"), "level1_interface")
        self.assertEqual(scorer._detect_level_from_test_name("test_processor_creation"), "level1_interface")
        self.assertEqual(scorer._detect_level_from_test_name("test_specification_usage"), "level2_specification")
        self.assertEqual(scorer._detect_level_from_test_name("test_environment_variable_handling"), "level2_specification")
        self.assertEqual(scorer._detect_level_from_test_name("test_input_path_mapping"), "level3_path_mapping")
        self.assertEqual(scorer._detect_level_from_test_name("test_processing_inputs_outputs"), "level3_path_mapping")
        self.assertEqual(scorer._detect_level_from_test_name("test_dependency_resolution"), "level4_integration")
        self.assertEqual(scorer._detect_level_from_test_name("test_step_creation"), "level4_integration")
    
    def test_detection_method_identification(self):
        """Test that detection methods are correctly identified."""
        scorer = StepBuilderScorer(self.sample_results)
        
        # Test detection method identification
        self.assertEqual(scorer._get_detection_method("level1_test_processor_creation_method"), "explicit_prefix")
        self.assertEqual(scorer._get_detection_method("test_inheritance"), "keyword_based")  # Detected via keywords
        self.assertEqual(scorer._get_detection_method("test_specification_usage"), "keyword_based")
        self.assertEqual(scorer._get_detection_method("test_random_functionality"), "undetected")
    
    def test_detection_summary(self):
        """Test that detection summary provides accurate statistics."""
        scorer = StepBuilderScorer(self.sample_results)
        summary = scorer.get_detection_summary()
        
        # Verify summary structure
        self.assertIn("summary", summary)
        self.assertIn("details", summary)
        
        # Verify summary counts
        summary_data = summary["summary"]
        self.assertGreater(summary_data["explicit_prefix"], 0)
        self.assertGreater(summary_data["keyword_based"], 0)
        # Note: fallback_map might be 0 if keyword detection is comprehensive
        self.assertGreaterEqual(summary_data["fallback_map"], 0)
        self.assertGreater(summary_data["undetected"], 0)
        self.assertEqual(summary_data["total"], len(self.sample_results))
        
        # Verify details structure
        details = summary["details"]
        self.assertIn("explicit_prefix", details)
        self.assertIn("keyword_based", details)
        self.assertIn("fallback_map", details)
        self.assertIn("undetected", details)
    
    def test_processing_variant_compatibility(self):
        """Test that Processing variant tests are correctly handled."""
        scorer = StepBuilderScorer(self.processing_results)
        detection_summary = scorer.get_detection_summary()
        
        # All Processing variant tests should use explicit prefix
        self.assertEqual(detection_summary["summary"]["explicit_prefix"], len(self.processing_results))
        self.assertEqual(detection_summary["summary"]["keyword_based"], 0)
        self.assertEqual(detection_summary["summary"]["fallback_map"], 0)
        self.assertEqual(detection_summary["summary"]["undetected"], 0)
        
        # Verify level distribution
        report = scorer.generate_report()
        levels = report["levels"]
        
        # Should have tests in all 4 levels
        self.assertGreater(levels["level1_interface"]["total"], 0)
        self.assertGreater(levels["level2_specification"]["total"], 0)
        self.assertGreater(levels["level3_path_mapping"]["total"], 0)
        self.assertGreater(levels["level4_integration"]["total"], 0)
    
    def test_mixed_scenario_handling(self):
        """Test that mixed test scenarios are handled correctly."""
        scorer = StepBuilderScorer(self.mixed_results)
        detection_summary = scorer.get_detection_summary()
        
        # Should have multiple detection methods used
        summary = detection_summary["summary"]
        self.assertGreater(summary["explicit_prefix"], 0)
        self.assertGreater(summary["keyword_based"], 0)
        # Note: fallback_map might be 0 if keyword detection is comprehensive
        self.assertGreaterEqual(summary["fallback_map"], 0)
        
        # Total should match input
        self.assertEqual(summary["total"], len(self.mixed_results))
    
    def test_level_score_calculation(self):
        """Test that level scores are calculated correctly."""
        scorer = StepBuilderScorer(self.sample_results)
        
        # Test level score calculation
        for level in ["level1_interface", "level2_specification", "level3_path_mapping", "level4_integration"]:
            score, passed, total = scorer.calculate_level_score(level)
            
            # Score should be between 0 and 100
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 100.0)
            
            # Passed should not exceed total
            self.assertLessEqual(passed, total)
            self.assertGreaterEqual(passed, 0)
            self.assertGreaterEqual(total, 0)
    
    def test_overall_score_calculation(self):
        """Test that overall score is calculated correctly."""
        scorer = StepBuilderScorer(self.sample_results)
        overall_score = scorer.calculate_overall_score()
        
        # Overall score should be between 0 and 100
        self.assertGreaterEqual(overall_score, 0.0)
        self.assertLessEqual(overall_score, 100.0)
    
    def test_report_generation(self):
        """Test that reports are generated correctly."""
        scorer = StepBuilderScorer(self.sample_results)
        report = scorer.generate_report()
        
        # Verify report structure
        self.assertIn("overall", report)
        self.assertIn("levels", report)
        self.assertIn("failed_tests", report)
        
        # Verify overall section
        overall = report["overall"]
        self.assertIn("score", overall)
        self.assertIn("rating", overall)
        self.assertIn("passed", overall)
        self.assertIn("total", overall)
        self.assertIn("pass_rate", overall)
        
        # Verify levels section
        levels = report["levels"]
        for level in ["level1_interface", "level2_specification", "level3_path_mapping", "level4_integration"]:
            self.assertIn(level, levels)
            level_data = levels[level]
            self.assertIn("score", level_data)
            self.assertIn("passed", level_data)
            self.assertIn("total", level_data)
            self.assertIn("tests", level_data)
    
    def test_zero_maintenance_requirement(self):
        """Test that new test variants require zero maintenance."""
        # Create hypothetical Training variant tests
        training_results = {
            "level1_test_training_job_creation": {"passed": True},
            "level1_test_estimator_configuration": {"passed": True},
            "level2_test_hyperparameter_specification": {"passed": True},
            "level2_test_training_arguments": {"passed": True},
            "level3_test_training_input_creation": {"passed": True},
            "level3_test_model_output_creation": {"passed": True},
            "level4_test_training_step_execution": {"passed": True},
            "level4_test_model_registration": {"passed": True},
        }
        
        # Should work without any TEST_LEVEL_MAP updates
        scorer = StepBuilderScorer(training_results)
        detection_summary = scorer.get_detection_summary()
        
        # All should be detected via explicit prefix
        self.assertEqual(detection_summary["summary"]["explicit_prefix"], len(training_results))
        self.assertEqual(detection_summary["summary"]["undetected"], 0)
        
        # Should generate valid report
        report = scorer.generate_report()
        self.assertGreater(report["overall"]["score"], 0)
    
    def test_backward_compatibility(self):
        """Test that legacy tests still work correctly."""
        legacy_results = {
            "test_inheritance": {"passed": True},
            "test_required_methods": {"passed": True},
            "test_specification_usage": {"passed": True},
            "test_contract_alignment": {"passed": True},
            "test_input_path_mapping": {"passed": True},
            "test_output_path_mapping": {"passed": True},
            "test_dependency_resolution": {"passed": True},
            "test_step_creation": {"passed": True},
        }
        
        scorer = StepBuilderScorer(legacy_results)
        detection_summary = scorer.get_detection_summary()
        
        # All should be detected (either keyword-based or fallback map)
        detected = (detection_summary["summary"]["keyword_based"] + 
                   detection_summary["summary"]["fallback_map"])
        self.assertEqual(detected, len(legacy_results))
        self.assertEqual(detection_summary["summary"]["undetected"], 0)

class TestPatternScoringIntegration(unittest.TestCase):
    """Integration tests for the pattern-based scoring system."""
    
    def test_score_builder_results_function(self):
        """Test the main score_builder_results function."""
        results = {
            "level1_test_interface": {"passed": True},
            "level2_test_specification": {"passed": True},
            "level3_test_path_mapping": {"passed": False, "error": "Path error"},
            "level4_test_integration": {"passed": True},
        }
        
        # Test without saving files
        report = score_builder_results(
            results=results,
            builder_name="TestBuilder",
            save_report=False,
            generate_chart=False
        )
        
        # Verify report structure
        self.assertIn("overall", report)
        self.assertIn("levels", report)
        self.assertIn("failed_tests", report)
        
        # Verify failed tests are captured
        self.assertEqual(len(report["failed_tests"]), 1)
        self.assertEqual(report["failed_tests"][0]["name"], "level3_test_path_mapping")

def demonstrate_pattern_detection():
    """Demonstrate the pattern-based detection capabilities."""
    print("🔍" * 60)
    print("PATTERN-BASED SCORING SYSTEM DEMONSTRATION")
    print("🔍" * 60)
    
    # Create sample test results
    results = {
        # Explicit level prefix tests
        "level1_test_processor_creation_method": {"passed": True},
        "level2_test_environment_variable_patterns": {"passed": True},
        "level3_test_processing_input_creation": {"passed": False, "error": "Input creation failed"},
        "level4_test_end_to_end_step_creation": {"passed": True},
        
        # Legacy tests
        "test_inheritance": {"passed": True},
        "test_specification_usage": {"passed": True},
        "test_input_path_mapping": {"passed": True},
        "test_dependency_resolution": {"passed": True},
        
        # Keyword-based detection
        "test_custom_processor_interface": {"passed": True},
        "test_job_arguments_configuration": {"passed": True},
        "test_output_path_validation": {"passed": True},
        "test_integration_with_dependencies": {"passed": True},
        
        # Undetected test
        "test_random_functionality": {"passed": True},
    }
    
    print(f"\n📊 Sample Test Results ({len(results)} tests):")
    for test_name, result in results.items():
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    # Create scorer and analyze
    scorer = StepBuilderScorer(results)
    
    print("\n🎯 Pattern Detection Analysis:")
    detection_summary = scorer.get_detection_summary()
    
    print(f"\nDetection Method Summary:")
    summary = detection_summary["summary"]
    print(f"  📌 Explicit prefix (level1_, level2_, etc.): {summary['explicit_prefix']} tests")
    print(f"  🔍 Keyword-based detection: {summary['keyword_based']} tests")
    print(f"  📋 Fallback to TEST_LEVEL_MAP: {summary['fallback_map']} tests")
    print(f"  ❓ Undetected (no level assigned): {summary['undetected']} tests")
    print(f"  📈 Total tests processed: {summary['total']} tests")
    
    # Generate and display score report
    print("\n📊 SCORING RESULTS:")
    print("=" * 50)
    
    # Print report with detection details
    scorer.print_report(show_test_detection=True)

if __name__ == "__main__":
    # Run demonstration if called directly
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demonstrate_pattern_detection()
    else:
        # Run unit tests
        unittest.main()
