#!/usr/bin/env python3
"""
Unit tests for UnifiedAlignmentTester visualization integration functionality.

This test suite focuses on testing the enhanced UnifiedAlignmentTester functionality
that includes scoring, chart generation, and visualization integration.
"""

import sys
import os
import unittest
import tempfile
import shutil
import json
from unittest.mock import patch, MagicMock, call
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from cursus.validation.alignment.unified_alignment_tester import UnifiedAlignmentTester
from cursus.validation.alignment.alignment_reporter import (
    AlignmentReport, ValidationResult, AlignmentIssue, SeverityLevel
)
from cursus.validation.alignment.alignment_scorer import AlignmentScorer


class TestUnifiedAlignmentTesterVisualization(unittest.TestCase):
    """Test UnifiedAlignmentTester visualization integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.tester = UnifiedAlignmentTester()
        self.setup_mock_validation_results()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def setup_mock_validation_results(self):
        """Set up mock validation results for testing."""
        # Add some sample results to the tester's report
        level1_result = ValidationResult(
            test_name="script_path_validation",
            passed=True,
            issues=[],
            details={"script_path": "/opt/ml/code/train.py"}
        )
        
        level2_result = ValidationResult(
            test_name="logical_name_alignment",
            passed=False,
            issues=[
                AlignmentIssue(
                    level=SeverityLevel.WARNING,
                    category="logical_names",
                    message="Contract output not found in specification"
                )
            ]
        )
        
        level3_result = ValidationResult(
            test_name="dependency_resolution",
            passed=True,
            issues=[]
        )
        
        level4_result = ValidationResult(
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
        
        self.tester.report.add_level1_result("script_path_validation", level1_result)
        self.tester.report.add_level2_result("logical_name_alignment", level2_result)
        self.tester.report.add_level3_result("dependency_resolution", level3_result)
        self.tester.report.add_level4_result("configuration_validation", level4_result)
    
    def test_export_report_with_chart_generation(self):
        """Test export_report method with chart generation enabled."""
        output_path = os.path.join(self.temp_dir, "test_report.json")
        
        # Mock the scorer and chart generation
        with patch.object(self.tester.report, 'get_scorer') as mock_get_scorer:
            mock_scorer = MagicMock()
            mock_scorer.generate_chart.return_value = os.path.join(self.temp_dir, "test_chart.png")
            mock_get_scorer.return_value = mock_scorer
            
            # Test with chart generation enabled
            result = self.tester.export_report(
                format='json',
                output_path=output_path,
                generate_chart=True,
                script_name="test_script"
            )
            
            # Should call generate_chart with correct parameters
            mock_scorer.generate_chart.assert_called_once_with(
                "test_script",
                str(Path(output_path).parent)
            )
            
            # Should return the JSON output
            self.assertIsInstance(result, str)
    
    def test_export_report_without_chart_generation(self):
        """Test export_report method with chart generation disabled."""
        output_path = os.path.join(self.temp_dir, "test_report.json")
        
        with patch.object(self.tester.report, 'get_scorer') as mock_get_scorer:
            mock_scorer = MagicMock()
            mock_get_scorer.return_value = mock_scorer
            
            # Test with chart generation disabled
            result = self.tester.export_report(
                format='json',
                output_path=output_path,
                generate_chart=False,
                script_name="test_script"
            )
            
            # Should not call generate_chart
            mock_scorer.generate_chart.assert_not_called()
            
            # Should still return the JSON output
            self.assertIsInstance(result, str)
    
    def test_export_report_chart_generation_error_handling(self):
        """Test export_report handles chart generation errors gracefully."""
        output_path = os.path.join(self.temp_dir, "test_report.json")
        
        with patch.object(self.tester.report, 'get_scorer') as mock_get_scorer:
            mock_scorer = MagicMock()
            mock_scorer.generate_chart.side_effect = Exception("Chart generation failed")
            mock_get_scorer.return_value = mock_scorer
            
            # Should not raise exception, but continue with report export
            result = self.tester.export_report(
                format='json',
                output_path=output_path,
                generate_chart=True,
                script_name="test_script"
            )
            
            # Should still return the JSON output
            self.assertIsInstance(result, str)
    
    def test_export_report_html_with_chart_generation(self):
        """Test HTML export with chart generation."""
        output_path = os.path.join(self.temp_dir, "test_report.html")
        
        with patch.object(self.tester.report, 'get_scorer') as mock_get_scorer, \
             patch.object(self.tester.report, 'export_to_html') as mock_export_html:
            
            mock_scorer = MagicMock()
            mock_scorer.generate_chart.return_value = os.path.join(self.temp_dir, "test_chart.png")
            mock_get_scorer.return_value = mock_scorer
            mock_export_html.return_value = "<html>Test Report</html>"
            
            result = self.tester.export_report(
                format='html',
                output_path=output_path,
                generate_chart=True,
                script_name="test_script"
            )
            
            # Should generate chart and export HTML
            mock_scorer.generate_chart.assert_called_once()
            mock_export_html.assert_called_once()
            self.assertIsInstance(result, str)
    
    def test_run_full_validation_with_scoring_summary(self):
        """Test run_full_validation displays scoring summary."""
        # Mock all level validation methods
        with patch.object(self.tester, '_run_level1_validation') as mock_l1, \
             patch.object(self.tester, '_run_level2_validation') as mock_l2, \
             patch.object(self.tester, '_run_level3_validation') as mock_l3, \
             patch.object(self.tester, '_run_level4_validation') as mock_l4, \
             patch('builtins.print') as mock_print:
            
            mock_l1.return_value = None
            mock_l2.return_value = None
            mock_l3.return_value = None
            mock_l4.return_value = None
            
            # Run full validation
            report = self.tester.run_full_validation(["test_script"])
            
            # Should print scoring summary
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            scoring_calls = [call for call in print_calls if "📊" in call or "Alignment Score" in call]
            
            # Should have at least one scoring-related print statement
            self.assertGreater(len(scoring_calls), 0)
            
            self.assertIsNotNone(report)
    
    def test_scoring_integration_with_report(self):
        """Test that scoring is properly integrated with the report."""
        # Test that the report has scoring capabilities
        self.assertTrue(hasattr(self.tester.report, 'get_scorer'))
        self.assertTrue(hasattr(self.tester.report, 'get_alignment_score'))
        self.assertTrue(hasattr(self.tester.report, 'get_level_scores'))
        
        # Test that scorer can be created
        scorer = self.tester.report.get_scorer()
        self.assertIsInstance(scorer, AlignmentScorer)
        
        # Test that scores can be calculated
        overall_score = self.tester.report.get_alignment_score()
        self.assertIsInstance(overall_score, float)
        self.assertGreaterEqual(overall_score, 0.0)
        self.assertLessEqual(overall_score, 100.0)
        
        level_scores = self.tester.report.get_level_scores()
        self.assertIsInstance(level_scores, dict)
    
    def test_chart_generation_integration(self):
        """Test chart generation integration."""
        with patch.object(self.tester.report, 'get_scorer') as mock_get_scorer:
            mock_scorer = MagicMock()
            mock_scorer.generate_chart.return_value = "/path/to/chart.png"
            mock_get_scorer.return_value = mock_scorer
            
            # Test chart generation through report
            chart_path = self.tester.report.generate_alignment_chart(
                script_name="test_script",
                output_dir=self.temp_dir
            )
            
            if chart_path:  # Only test if chart generation is available
                mock_scorer.generate_chart.assert_called_once()
    
    def test_enhanced_json_export_includes_scoring(self):
        """Test that JSON export includes scoring information."""
        json_output = self.tester.export_report(format='json')
        
        # Parse JSON to verify structure
        data = json.loads(json_output)
        
        # Should contain scoring section
        self.assertIn('scoring', data)
        scoring = data['scoring']
        
        # Verify scoring structure
        self.assertIn('overall_score', scoring)
        self.assertIn('quality_rating', scoring)
        self.assertIn('level_scores', scoring)
        self.assertIn('scoring_summary', scoring)
        
        # Verify score values are valid
        self.assertIsInstance(scoring['overall_score'], (int, float))
        self.assertIsInstance(scoring['quality_rating'], str)
        self.assertIsInstance(scoring['level_scores'], dict)
    
    def test_enhanced_html_export_includes_scoring(self):
        """Test that HTML export includes scoring visualizations."""
        html_output = self.tester.export_report(format='html')
        
        # Should contain scoring elements
        self.assertIn('scoring-section', html_output)
        self.assertIn('score-card', html_output)
        self.assertIn('Overall Alignment Score', html_output)
        
        # Should contain level-specific score cards
        self.assertIn('Script ↔ Contract', html_output)
        self.assertIn('Contract ↔ Specification', html_output)
        self.assertIn('Specification ↔ Dependencies', html_output)
        self.assertIn('Builder ↔ Configuration', html_output)
    
    def test_validate_specific_script_with_scoring(self):
        """Test validate_specific_script includes scoring information."""
        # Mock individual level validation methods
        with patch.object(self.tester.level1_tester, 'validate_script') as mock_l1, \
             patch.object(self.tester.level2_tester, 'validate_contract') as mock_l2, \
             patch.object(self.tester.level3_tester, 'validate_specification') as mock_l3, \
             patch.object(self.tester.level4_tester, 'validate_builder') as mock_l4:
            
            mock_l1.return_value = {"passed": True, "issues": []}
            mock_l2.return_value = {"passed": True, "issues": []}
            mock_l3.return_value = {"passed": True, "issues": []}
            mock_l4.return_value = {"passed": True, "issues": []}
            
            result = self.tester.validate_specific_script("test_script")
            
            # Should include scoring information
            self.assertIn('scoring', result)
            scoring = result['scoring']
            
            self.assertIn('overall_score', scoring)
            self.assertIn('quality_rating', scoring)
            self.assertIn('level_scores', scoring)
    
    def test_get_validation_summary_includes_scoring(self):
        """Test that validation summary includes scoring information."""
        summary = self.tester.get_validation_summary()
        
        # Should include scoring information
        self.assertIn('scoring', summary)
        scoring = summary['scoring']
        
        self.assertIn('overall_score', scoring)
        self.assertIn('quality_rating', scoring)
        self.assertIn('average_level_score', scoring)
    
    def test_chart_generation_with_different_output_formats(self):
        """Test chart generation works with different output formats."""
        # Test with JSON output
        json_path = os.path.join(self.temp_dir, "test.json")
        
        with patch.object(self.tester.report, 'get_scorer') as mock_get_scorer:
            mock_scorer = MagicMock()
            mock_scorer.generate_chart.return_value = os.path.join(self.temp_dir, "chart.png")
            mock_get_scorer.return_value = mock_scorer
            
            self.tester.export_report(
                format='json',
                output_path=json_path,
                generate_chart=True,
                script_name="test"
            )
            
            # Should generate chart in same directory as JSON
            mock_scorer.generate_chart.assert_called_with("test", self.temp_dir)
        
        # Test with HTML output
        html_path = os.path.join(self.temp_dir, "subdir", "test.html")
        os.makedirs(os.path.dirname(html_path), exist_ok=True)
        
        with patch.object(self.tester.report, 'get_scorer') as mock_get_scorer, \
             patch.object(self.tester.report, 'export_to_html') as mock_export_html:
            
            mock_scorer = MagicMock()
            mock_scorer.generate_chart.return_value = os.path.join(self.temp_dir, "subdir", "chart.png")
            mock_get_scorer.return_value = mock_scorer
            mock_export_html.return_value = "<html>Test</html>"
            
            self.tester.export_report(
                format='html',
                output_path=html_path,
                generate_chart=True,
                script_name="test"
            )
            
            # Should generate chart in same directory as HTML
            expected_dir = os.path.join(self.temp_dir, "subdir")
            mock_scorer.generate_chart.assert_called_with("test", expected_dir)
    
    def test_backward_compatibility_with_existing_functionality(self):
        """Test that existing functionality still works with visualization integration."""
        # Test basic validation methods still work
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
        
        summary = self.tester.get_validation_summary()
        self.assertIsInstance(summary, dict)
    
    def test_scoring_with_empty_results(self):
        """Test scoring functionality with empty validation results."""
        empty_tester = UnifiedAlignmentTester()
        
        # Should handle empty results gracefully
        overall_score = empty_tester.report.get_alignment_score()
        self.assertEqual(overall_score, 0.0)
        
        level_scores = empty_tester.report.get_level_scores()
        self.assertIsInstance(level_scores, dict)
        
        # Should be able to export even with empty results
        json_output = empty_tester.export_report(format='json')
        self.assertIsInstance(json_output, str)
        
        data = json.loads(json_output)
        self.assertIn('scoring', data)
    
    def test_chart_generation_file_path_handling(self):
        """Test chart generation handles file paths correctly."""
        # Test with nested directory structure
        nested_dir = os.path.join(self.temp_dir, "reports", "alignment")
        os.makedirs(nested_dir, exist_ok=True)
        
        output_path = os.path.join(nested_dir, "test_report.json")
        
        with patch.object(self.tester.report, 'get_scorer') as mock_get_scorer:
            mock_scorer = MagicMock()
            mock_scorer.generate_chart.return_value = os.path.join(nested_dir, "chart.png")
            mock_get_scorer.return_value = mock_scorer
            
            self.tester.export_report(
                format='json',
                output_path=output_path,
                generate_chart=True,
                script_name="test_script"
            )
            
            # Should use the correct directory
            mock_scorer.generate_chart.assert_called_with("test_script", nested_dir)
    
    def test_multiple_chart_generations_dont_conflict(self):
        """Test that multiple chart generations don't conflict."""
        with patch.object(self.tester.report, 'get_scorer') as mock_get_scorer:
            mock_scorer = MagicMock()
            mock_scorer.generate_chart.side_effect = [
                os.path.join(self.temp_dir, "chart1.png"),
                os.path.join(self.temp_dir, "chart2.png")
            ]
            mock_get_scorer.return_value = mock_scorer
            
            # Generate first chart
            self.tester.export_report(
                format='json',
                output_path=os.path.join(self.temp_dir, "report1.json"),
                generate_chart=True,
                script_name="script1"
            )
            
            # Generate second chart
            self.tester.export_report(
                format='json',
                output_path=os.path.join(self.temp_dir, "report2.json"),
                generate_chart=True,
                script_name="script2"
            )
            
            # Should have called generate_chart twice with different script names
            expected_calls = [
                call("script1", self.temp_dir),
                call("script2", self.temp_dir)
            ]
            mock_scorer.generate_chart.assert_has_calls(expected_calls)


class TestVisualizationWorkflowIntegration(unittest.TestCase):
    """Test end-to-end visualization workflow integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.tester = UnifiedAlignmentTester()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_visualization_workflow(self):
        """Test complete workflow from validation to visualization."""
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
            
            # Run full validation
            report = self.tester.run_full_validation(["test_script"])
            
            # Export with visualization
            output_path = os.path.join(self.temp_dir, "final_report.json")
            json_output = self.tester.export_report(
                format='json',
                output_path=output_path,
                generate_chart=True,
                script_name="test_script"
            )
            
            # Verify complete workflow
            self.assertIsNotNone(report)
            self.assertIsInstance(json_output, str)
            
            # Verify chart was generated
            mock_scorer.generate_chart.assert_called_once()
            
            # Verify JSON contains scoring
            data = json.loads(json_output)
            self.assertIn('scoring', data)
    
    def test_batch_processing_with_visualization(self):
        """Test batch processing with visualization generation."""
        scripts = ["script1", "script2", "script3"]
        
        with patch.object(self.tester.report, 'get_scorer') as mock_get_scorer:
            mock_scorer = MagicMock()
            mock_scorer.generate_chart.side_effect = [
                os.path.join(self.temp_dir, f"chart_{i}.png") for i in range(len(scripts))
            ]
            mock_get_scorer.return_value = mock_scorer
            
            # Process each script with visualization
            for i, script in enumerate(scripts):
                output_path = os.path.join(self.temp_dir, f"report_{script}.json")
                
                self.tester.export_report(
                    format='json',
                    output_path=output_path,
                    generate_chart=True,
                    script_name=script
                )
            
            # Should have generated charts for all scripts
            self.assertEqual(mock_scorer.generate_chart.call_count, len(scripts))
            
            # Verify each call had correct script name
            for i, script in enumerate(scripts):
                call_args = mock_scorer.generate_chart.call_args_list[i]
                self.assertEqual(call_args[0][0], script)  # First argument should be script name


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
