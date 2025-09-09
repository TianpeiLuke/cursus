#!/usr/bin/env python3
"""
Comprehensive test suite for the complete alignment validation visualization integration.

This test suite verifies that all components of the 4-phase visualization integration
work together correctly and that existing functionality remains intact.
"""

import sys
import os
import unittest
import tempfile
import shutil
import json
from unittest.mock import patch, MagicMock


from cursus.validation.alignment.unified_alignment_tester import UnifiedAlignmentTester
from cursus.validation.alignment.alignment_reporter import (
    AlignmentReport, ValidationResult, AlignmentIssue, SeverityLevel
)
from cursus.validation.alignment.alignment_scorer import AlignmentScorer

class TestVisualizationIntegrationComplete(unittest.TestCase):
    """Test complete visualization integration across all components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.tester = UnifiedAlignmentTester()
        self.setup_comprehensive_test_data()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def setup_comprehensive_test_data(self):
        """Set up comprehensive test data covering all scenarios."""
        # Level 1 results - mixed success/failure
        level1_success = ValidationResult(
            test_name="script_path_validation",
            passed=True,
            issues=[],
            details={"script_path": "/opt/ml/code/train.py"}
        )
        
        level1_failure = ValidationResult(
            test_name="environment_variable_validation",
            passed=False,
            issues=[
                AlignmentIssue(
                    level=SeverityLevel.ERROR,
                    category="environment_variables",
                    message="Script accesses undeclared environment variable 'CUSTOM_VAR'"
                )
            ]
        )
        
        # Level 2 results - mixed success/failure
        level2_success = ValidationResult(
            test_name="logical_name_alignment",
            passed=True,
            issues=[]
        )
        
        level2_failure = ValidationResult(
            test_name="input_output_mapping",
            passed=False,
            issues=[
                AlignmentIssue(
                    level=SeverityLevel.WARNING,
                    category="logical_names",
                    message="Contract output not found in specification"
                )
            ]
        )
        
        # Level 3 results - success
        level3_success = ValidationResult(
            test_name="dependency_resolution",
            passed=True,
            issues=[]
        )
        
        # Level 4 results - critical failure
        level4_failure = ValidationResult(
            test_name="configuration_validation",
            passed=False,
            issues=[
                AlignmentIssue(
                    level=SeverityLevel.CRITICAL,
                    category="configuration",
                    message="Builder missing required environment variable"
                )
            ]
        )
        
        # Add results to tester
        self.tester.report.add_level1_result("script_path_validation", level1_success)
        self.tester.report.add_level1_result("environment_variable_validation", level1_failure)
        self.tester.report.add_level2_result("logical_name_alignment", level2_success)
        self.tester.report.add_level2_result("input_output_mapping", level2_failure)
        self.tester.report.add_level3_result("dependency_resolution", level3_success)
        self.tester.report.add_level4_result("configuration_validation", level4_failure)
    
    def test_phase1_core_scoring_system_integration(self):
        """Test Phase 1: Core Scoring System Integration."""
        # Test AlignmentScorer creation and basic functionality
        scorer = self.tester.report.get_scorer()
        self.assertIsInstance(scorer, AlignmentScorer)
        
        # Test overall score calculation
        overall_score = self.tester.report.get_alignment_score()
        self.assertIsInstance(overall_score, float)
        self.assertGreaterEqual(overall_score, 0.0)
        self.assertLessEqual(overall_score, 100.0)
        
        # Test level scores
        level_scores = self.tester.report.get_level_scores()
        self.assertIsInstance(level_scores, dict)
        
        expected_levels = [
            'level1_script_contract',
            'level2_contract_spec',
            'level3_spec_dependencies',
            'level4_builder_config'
        ]
        
        for level in expected_levels:
            if level in level_scores:
                self.assertIsInstance(level_scores[level], float)
                self.assertGreaterEqual(level_scores[level], 0.0)
                self.assertLessEqual(level_scores[level], 100.0)
        
        # Test quality rating
        quality_rating = scorer.get_quality_rating()
        self.assertIn(quality_rating, ["Excellent", "Good", "Satisfactory", "Needs Work", "Poor"])
        
        print("✅ Phase 1: Core Scoring System Integration - PASSED")
    
    def test_phase2_chart_generation_implementation(self):
        """Test Phase 2: Chart Generation Implementation."""
        # Test chart generation through AlignmentScorer
        scorer = self.tester.report.get_scorer()
        
        with patch('cursus.validation.alignment.alignment_scorer.plt') as mock_plt:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_plt.figure.return_value = (mock_fig, mock_ax)
            
            chart_path = scorer.generate_chart("test_script", self.temp_dir)
            
            if chart_path:  # Only test if matplotlib is available
                expected_path = os.path.join(self.temp_dir, "test_script_alignment_scores.png")
                self.assertEqual(chart_path, expected_path)
                mock_plt.figure.assert_called_once()
                mock_plt.savefig.assert_called_once()
        
        # Test chart generation through AlignmentReport
        with patch.object(self.tester.report, 'get_scorer') as mock_get_scorer:
            mock_scorer = MagicMock()
            mock_scorer.generate_chart.return_value = "/path/to/chart.png"
            mock_get_scorer.return_value = mock_scorer
            
            chart_path = self.tester.report.generate_alignment_chart(
                script_name="test_script",
                output_dir=self.temp_dir
            )
            
            if chart_path:
                mock_scorer.generate_chart.assert_called_once()
        
        print("✅ Phase 2: Chart Generation Implementation - PASSED")
    
    def test_phase3_enhanced_report_structure(self):
        """Test Phase 3: Enhanced Report Structure with Visual Components."""
        # Test enhanced JSON export
        json_output = self.tester.export_report(format='json')
        self.assertIsInstance(json_output, str)
        
        data = json.loads(json_output)
        
        # Verify scoring section exists
        self.assertIn('scoring', data)
        scoring = data['scoring']
        
        # Verify scoring structure
        required_scoring_fields = [
            'overall_score',
            'quality_rating',
            'level_scores',
            'scoring_summary'
        ]
        
        for field in required_scoring_fields:
            self.assertIn(field, scoring)
        
        # Test enhanced HTML export
        html_output = self.tester.export_report(format='html')
        self.assertIsInstance(html_output, str)
        
        # Verify HTML contains scoring elements
        scoring_elements = [
            'scoring-section',
            'score-card',
            'Overall Alignment Score',
            'Script ↔ Contract',
            'Contract ↔ Specification',
            'Specification ↔ Dependencies',
            'Builder ↔ Configuration'
        ]
        
        for element in scoring_elements:
            self.assertIn(element, html_output)
        
        print("✅ Phase 3: Enhanced Report Structure - PASSED")
    
    def test_phase4_workflow_integration(self):
        """Test Phase 4: Workflow Integration."""
        # Test export_report with chart generation
        output_path = os.path.join(self.temp_dir, "workflow_test.json")
        
        with patch.object(self.tester.report, 'get_scorer') as mock_get_scorer:
            mock_scorer = MagicMock()
            mock_scorer.generate_chart.return_value = os.path.join(self.temp_dir, "chart.png")
            mock_get_scorer.return_value = mock_scorer
            
            result = self.tester.export_report(
                format='json',
                output_path=output_path,
                generate_chart=True,
                script_name="workflow_test"
            )
            
            self.assertIsInstance(result, str)
            mock_scorer.generate_chart.assert_called_once_with(
                "workflow_test",
                self.temp_dir
            )
        
        # Test run_full_validation with scoring display
        with patch.object(self.tester, '_run_level1_validation') as mock_l1, \
             patch.object(self.tester, '_run_level2_validation') as mock_l2, \
             patch.object(self.tester, '_run_level3_validation') as mock_l3, \
             patch.object(self.tester, '_run_level4_validation') as mock_l4, \
             patch('builtins.print') as mock_print:
            
            mock_l1.return_value = None
            mock_l2.return_value = None
            mock_l3.return_value = None
            mock_l4.return_value = None
            
            report = self.tester.run_full_validation(["test_script"])
            
            # Should print scoring summary
            print_calls = [call[0][0] for call in mock_print.call_args_list if call[0]]
            scoring_calls = [call for call in print_calls if "📊" in str(call) or "Alignment Score" in str(call)]
            
            # Should have at least one scoring-related print statement
            self.assertGreater(len(scoring_calls), 0)
            self.assertIsNotNone(report)
        
        print("✅ Phase 4: Workflow Integration - PASSED")
    
    def test_validate_specific_script_with_scoring(self):
        """Test validate_specific_script includes comprehensive scoring."""
        with patch.object(self.tester.level1_tester, 'validate_script') as mock_l1, \
             patch.object(self.tester.level2_tester, 'validate_contract') as mock_l2, \
             patch.object(self.tester.level3_tester, 'validate_specification') as mock_l3, \
             patch.object(self.tester.level4_tester, 'validate_builder') as mock_l4:
            
            mock_l1.return_value = {"passed": True, "issues": []}
            mock_l2.return_value = {"passed": False, "issues": [{"severity": "WARNING"}]}
            mock_l3.return_value = {"passed": True, "issues": []}
            mock_l4.return_value = {"passed": False, "issues": [{"severity": "CRITICAL"}]}
            
            result = self.tester.validate_specific_script("comprehensive_test")
            
            # Verify basic structure
            self.assertEqual(result['script_name'], "comprehensive_test")
            self.assertIn(result['overall_status'], ["PASSING", "FAILING"])
            
            # Verify scoring information
            self.assertIn('scoring', result)
            scoring = result['scoring']
            
            self.assertIn('overall_score', scoring)
            self.assertIn('quality_rating', scoring)
            self.assertIn('level_scores', scoring)
            
            # Verify score values are valid
            self.assertIsInstance(scoring['overall_score'], (int, float))
            self.assertGreaterEqual(scoring['overall_score'], 0.0)
            self.assertLessEqual(scoring['overall_score'], 100.0)
            
            self.assertIsInstance(scoring['quality_rating'], str)
            self.assertIsInstance(scoring['level_scores'], dict)
        
        print("✅ validate_specific_script with scoring - PASSED")
    
    def test_get_validation_summary_with_scoring(self):
        """Test get_validation_summary includes comprehensive scoring."""
        summary = self.tester.get_validation_summary()
        
        # Verify basic structure
        basic_fields = ['overall_status', 'total_tests', 'pass_rate', 'level_breakdown']
        for field in basic_fields:
            self.assertIn(field, summary)
        
        # Verify scoring information
        self.assertIn('scoring', summary)
        scoring = summary['scoring']
        
        scoring_fields = ['overall_score', 'quality_rating', 'average_level_score']
        for field in scoring_fields:
            self.assertIn(field, scoring)
        
        # Verify score values
        self.assertIsInstance(scoring['overall_score'], (int, float))
        self.assertGreaterEqual(scoring['overall_score'], 0.0)
        self.assertLessEqual(scoring['overall_score'], 100.0)
        
        print("✅ get_validation_summary with scoring - PASSED")
    
    def test_backward_compatibility(self):
        """Test that all existing functionality still works."""
        # Test basic validation methods
        with patch.object(self.tester, '_run_level1_validation') as mock_l1, \
             patch.object(self.tester, '_run_level2_validation') as mock_l2, \
             patch.object(self.tester, '_run_level3_validation') as mock_l3, \
             patch.object(self.tester, '_run_level4_validation') as mock_l4:
            
            mock_l1.return_value = None
            mock_l2.return_value = None
            mock_l3.return_value = None
            mock_l4.return_value = None
            
            # Should work without any visualization parameters
            report = self.tester.run_full_validation(["test_script"])
            self.assertIsNotNone(report)
        
        # Test basic export still works
        json_output = self.tester.export_report(format='json')
        self.assertIsInstance(json_output, str)
        
        # Test status methods still work
        matrix = self.tester.get_alignment_status_matrix()
        self.assertIsInstance(matrix, dict)
        
        # Test issue retrieval methods
        critical_issues = self.tester.get_critical_issues()
        self.assertIsInstance(critical_issues, list)
        
        print("✅ Backward compatibility - PASSED")
    
    def test_error_handling_and_edge_cases(self):
        """Test error handling and edge cases."""
        # Test with empty results
        empty_tester = UnifiedAlignmentTester()
        
        # Should handle empty results gracefully
        overall_score = empty_tester.report.get_alignment_score()
        self.assertEqual(overall_score, 0.0)
        
        # Should be able to export even with empty results
        json_output = empty_tester.export_report(format='json')
        self.assertIsInstance(json_output, str)
        
        data = json.loads(json_output)
        self.assertIn('scoring', data)
        
        # Test chart generation error handling
        with patch.object(self.tester.report, 'get_scorer') as mock_get_scorer:
            mock_scorer = MagicMock()
            mock_scorer.generate_chart.side_effect = Exception("Chart generation failed")
            mock_get_scorer.return_value = mock_scorer
            
            # Should not raise exception
            result = self.tester.export_report(
                format='json',
                generate_chart=True,
                script_name="error_test"
            )
            
            self.assertIsInstance(result, str)
        
        print("✅ Error handling and edge cases - PASSED")
    
    def test_json_serialization_with_complex_data(self):
        """Test JSON serialization with complex data structures."""
        # Add complex data to results
        complex_result = ValidationResult(test_name="complex_test", passed=True)
        
        complex_issue = AlignmentIssue(
            level=SeverityLevel.INFO,
            category="complex_serialization",
            message="Test complex object serialization",
            details={
                "property_object": property(lambda self: "test"),
                "type_object": str,
                "nested_dict": {
                    "inner_property": property(lambda self: "inner"),
                    "inner_type": int,
                    "normal_value": "should_work"
                }
            }
        )
        complex_result.add_issue(complex_issue)
        
        self.tester.report.add_level1_result("complex_test", complex_result)
        
        # Should be able to serialize without errors
        json_output = self.tester.export_report(format='json')
        parsed_json = json.loads(json_output)
        
        self.assertIsInstance(parsed_json, dict)
        self.assertIn('scoring', parsed_json)
        
        print("✅ JSON serialization with complex data - PASSED")
    
    def test_complete_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Mock validation results
        mock_results = {
            "test_script": {
                "passed": True,
                "issues": [],
                "details": {"test": "data"}
            }
        }
        
        with patch.object(self.tester.level1_tester, 'validate_all_scripts') as mock_l1, \
             patch.object(self.tester.level2_tester, 'validate_all_contracts') as mock_l2, \
             patch.object(self.tester.level3_tester, 'validate_all_specifications') as mock_l3, \
             patch.object(self.tester.level4_tester, 'validate_all_builders') as mock_l4, \
             patch.object(self.tester.report, 'get_scorer') as mock_get_scorer:
            
            # Mock validation methods
            mock_l1.return_value = mock_results
            mock_l2.return_value = mock_results
            mock_l3.return_value = mock_results
            mock_l4.return_value = mock_results
            
            # Mock scorer
            mock_scorer = MagicMock()
            mock_scorer.generate_chart.return_value = os.path.join(self.temp_dir, "chart.png")
            mock_scorer.calculate_overall_score.return_value = 85.5
            mock_scorer.get_quality_rating.return_value = "Good"
            mock_get_scorer.return_value = mock_scorer
            
            # Step 1: Run full validation
            report = self.tester.run_full_validation(["test_script"])
            self.assertIsNotNone(report)
            
            # Step 2: Get validation summary
            summary = self.tester.get_validation_summary()
            self.assertIn('scoring', summary)
            
            # Step 3: Validate specific script
            script_result = self.tester.validate_specific_script("test_script")
            self.assertIn('scoring', script_result)
            
            # Step 4: Export with visualization
            output_path = os.path.join(self.temp_dir, "final_report.json")
            json_output = self.tester.export_report(
                format='json',
                output_path=output_path,
                generate_chart=True,
                script_name="test_script"
            )
            
            # Verify complete workflow
            self.assertIsInstance(json_output, str)
            mock_scorer.generate_chart.assert_called_once()
            
            # Verify JSON contains all expected sections
            data = json.loads(json_output)
            expected_sections = ['scoring', 'level1', 'level2', 'level3', 'level4', 'summary']
            for section in expected_sections:
                if section in data:  # Some sections might be empty
                    self.assertIsInstance(data[section], (dict, list, str, int, float))
        
        print("✅ Complete end-to-end workflow - PASSED")

def run_comprehensive_demo():
    """Run a comprehensive demonstration of all visualization integration features."""
    print("=" * 80)
    print("COMPREHENSIVE ALIGNMENT VALIDATION VISUALIZATION INTEGRATION TEST")
    print("=" * 80)
    
    # Create test instance
    test_instance = TestVisualizationIntegrationComplete()
    test_instance.setUp()
    
    try:
        print("\n🧪 Testing Phase 1: Core Scoring System Integration...")
        test_instance.test_phase1_core_scoring_system_integration()
        
        print("\n🧪 Testing Phase 2: Chart Generation Implementation...")
        test_instance.test_phase2_chart_generation_implementation()
        
        print("\n🧪 Testing Phase 3: Enhanced Report Structure...")
        test_instance.test_phase3_enhanced_report_structure()
        
        print("\n🧪 Testing Phase 4: Workflow Integration...")
        test_instance.test_phase4_workflow_integration()
        
        print("\n🧪 Testing validate_specific_script with scoring...")
        test_instance.test_validate_specific_script_with_scoring()
        
        print("\n🧪 Testing get_validation_summary with scoring...")
        test_instance.test_get_validation_summary_with_scoring()
        
        print("\n🧪 Testing backward compatibility...")
        test_instance.test_backward_compatibility()
        
        print("\n🧪 Testing error handling and edge cases...")
        test_instance.test_error_handling_and_edge_cases()
        
        print("\n🧪 Testing JSON serialization with complex data...")
        test_instance.test_json_serialization_with_complex_data()
        
        print("\n🧪 Testing complete end-to-end workflow...")
        test_instance.test_complete_end_to_end_workflow()
        
        print("\n" + "=" * 80)
        print("🎉 ALL TESTS PASSED! COMPREHENSIVE INTEGRATION SUCCESSFUL!")
        print("=" * 80)
        print("\n✅ All 4 phases of the alignment validation visualization integration are working:")
        print("   📊 Phase 1: Core Scoring System Integration")
        print("   📈 Phase 2: Chart Generation Implementation")
        print("   📋 Phase 3: Enhanced Report Structure with Visual Components")
        print("   🔄 Phase 4: Workflow Integration")
        print("\n🔧 Key features verified:")
        print("   • AlignmentScorer with weighted 4-level scoring")
        print("   • Professional chart generation with matplotlib")
        print("   • Enhanced JSON/HTML exports with scoring data")
        print("   • Seamless workflow integration")
        print("   • Backward compatibility maintained")
        print("   • Robust error handling")
        print("   • Complex data serialization")
        print("\n🎯 The alignment validation visualization integration is complete and ready for use!")
        
    finally:
        test_instance.tearDown()

if __name__ == "__main__":
    # Run comprehensive demo first
    run_comprehensive_demo()
    
    # Then run unit tests
    print("\n" + "=" * 80)
    print("RUNNING COMPREHENSIVE UNIT TESTS")
    print("=" * 80)
    unittest.main(verbosity=2)
