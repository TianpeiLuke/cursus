#!/usr/bin/env python3
"""
Unit tests for AlignmentScorer class and visualization integration functionality.

This test suite focuses on testing the core AlignmentScorer functionality,
chart generation capabilities, and integration with the alignment validation system.
"""

import sys
import os
import unittest
import tempfile
import shutil
import json
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from cursus.validation.alignment.alignment_scorer import AlignmentScorer
from cursus.validation.alignment.alignment_reporter import (
    AlignmentReport, ValidationResult, AlignmentIssue, SeverityLevel
)


class TestAlignmentScorer(unittest.TestCase):
    """Test the AlignmentScorer class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_results = self.create_sample_alignment_results()
        self.scorer = AlignmentScorer(self.sample_results)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_sample_alignment_results(self):
        """Create sample alignment results in the expected format."""
        return {
            'level1': {
                'script_path_validation': {
                    'passed': True,
                    'issues': [],
                    'details': {'script_path': '/opt/ml/code/train.py'}
                },
                'environment_variable_validation': {
                    'passed': False,
                    'issues': [
                        {
                            'level': 'error',
                            'category': 'environment_variables',
                            'message': 'Script accesses undeclared environment variable CUSTOM_VAR'
                        }
                    ]
                }
            },
            'level2': {
                'logical_name_alignment': {
                    'passed': True,
                    'issues': []
                },
                'input_output_mapping': {
                    'passed': False,
                    'issues': [
                        {
                            'level': 'warning',
                            'category': 'logical_names',
                            'message': 'Contract output model_artifacts not found in specification'
                        }
                    ]
                }
            },
            'level3': {
                'dependency_resolution': {
                    'passed': True,
                    'issues': []
                }
            },
            'level4': {
                'configuration_validation': {
                    'passed': False,
                    'issues': [
                        {
                            'level': 'critical',
                            'category': 'configuration',
                            'message': 'Builder does not set required environment variable MODEL_TYPE'
                        }
                    ]
                }
            }
        }
    
    def test_scorer_initialization(self):
        """Test AlignmentScorer initialization."""
        self.assertIsNotNone(self.scorer)
        self.assertEqual(self.scorer.results, self.sample_results)
        
        # Test with empty results
        empty_scorer = AlignmentScorer({})
        self.assertEqual(empty_scorer.results, {})
    
    def test_group_by_level(self):
        """Test the _group_by_level method."""
        grouped = self.scorer._group_by_level()
        
        # Should have all 4 levels
        expected_levels = [
            'level1_script_contract',
            'level2_contract_spec', 
            'level3_spec_dependencies',
            'level4_builder_config'
        ]
        
        for level in expected_levels:
            self.assertIn(level, grouped)
        
        # Check that level1 data is properly mapped
        self.assertIn('level1', grouped['level1_script_contract'])
        self.assertEqual(
            grouped['level1_script_contract']['level1'],
            self.sample_results['level1']
        )
        
        # Check that level2 data is properly mapped
        self.assertIn('level2', grouped['level2_contract_spec'])
        self.assertEqual(
            grouped['level2_contract_spec']['level2'],
            self.sample_results['level2']
        )
    
    def test_calculate_level_score(self):
        """Test level score calculation."""
        # Test level with mixed results
        level1_data = self.sample_results['level1']
        score = self.scorer._calculate_level_score(level1_data)
        
        # Should be 50% (1 passed, 1 failed)
        self.assertEqual(score, 50.0)
        
        # Test level with all passing
        level3_data = self.sample_results['level3']
        score = self.scorer._calculate_level_score(level3_data)
        
        # Should be 100% (1 passed, 0 failed)
        self.assertEqual(score, 100.0)
        
        # Test level with all failing
        failing_data = {
            'test1': {'passed': False},
            'test2': {'passed': False}
        }
        score = self.scorer._calculate_level_score(failing_data)
        
        # Should be 0% (0 passed, 2 failed)
        self.assertEqual(score, 0.0)
        
        # Test empty level
        empty_score = self.scorer._calculate_level_score({})
        self.assertEqual(empty_score, 0.0)
    
    def test_calculate_overall_score(self):
        """Test overall score calculation with weighted levels."""
        overall_score = self.scorer.calculate_overall_score()
        
        # Verify it's a valid percentage
        self.assertIsInstance(overall_score, float)
        self.assertGreaterEqual(overall_score, 0.0)
        self.assertLessEqual(overall_score, 100.0)
        
        # Calculate expected score manually
        # Level 1: 50% * 1.0 = 50.0
        # Level 2: 50% * 1.5 = 75.0  
        # Level 3: 100% * 2.0 = 200.0
        # Level 4: 0% * 2.5 = 0.0
        # Total weighted: 325.0, Total weights: 7.0
        # Expected: 325.0 / 7.0 = 46.43 (approximately)
        
        expected_score = 325.0 / 7.0
        self.assertAlmostEqual(overall_score, expected_score, places=1)
    
    def test_get_level_scores(self):
        """Test getting individual level scores."""
        level_scores = self.scorer.get_level_scores()
        
        # Should have all 4 levels
        expected_levels = [
            'level1_script_contract',
            'level2_contract_spec',
            'level3_spec_dependencies', 
            'level4_builder_config'
        ]
        
        for level in expected_levels:
            self.assertIn(level, level_scores)
            score = level_scores[level]
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 100.0)
        
        # Check specific expected scores
        self.assertEqual(level_scores['level1_script_contract'], 50.0)
        self.assertEqual(level_scores['level2_contract_spec'], 50.0)
        self.assertEqual(level_scores['level3_spec_dependencies'], 100.0)
        self.assertEqual(level_scores['level4_builder_config'], 0.0)
    
    def test_get_quality_rating(self):
        """Test quality rating calculation."""
        # Test with current sample data (should be around 46%)
        rating = self.scorer.get_quality_rating()
        self.assertEqual(rating, "Needs Work")  # 40-59% range
        
        # Test with high score
        high_score_results = {
            'level1': {'test1': {'passed': True}},
            'level2': {'test1': {'passed': True}},
            'level3': {'test1': {'passed': True}},
            'level4': {'test1': {'passed': True}}
        }
        high_scorer = AlignmentScorer(high_score_results)
        rating = high_scorer.get_quality_rating()
        self.assertEqual(rating, "Excellent")  # 90-100% range
        
        # Test with low score
        low_score_results = {
            'level1': {'test1': {'passed': False}},
            'level2': {'test1': {'passed': False}},
            'level3': {'test1': {'passed': False}},
            'level4': {'test1': {'passed': False}}
        }
        low_scorer = AlignmentScorer(low_score_results)
        rating = low_scorer.get_quality_rating()
        self.assertEqual(rating, "Poor")  # 0-39% range
    
    def test_get_scoring_summary(self):
        """Test comprehensive scoring summary."""
        summary = self.scorer.get_scoring_summary()
        
        # Check required fields
        required_fields = [
            'overall_score',
            'quality_rating',
            'level_scores',
            'level_weights',
            'total_tests',
            'passed_tests',
            'failed_tests',
            'pass_rate'
        ]
        
        for field in required_fields:
            self.assertIn(field, summary)
        
        # Verify data types and ranges
        self.assertIsInstance(summary['overall_score'], float)
        self.assertIsInstance(summary['quality_rating'], str)
        self.assertIsInstance(summary['level_scores'], dict)
        self.assertIsInstance(summary['level_weights'], dict)
        self.assertIsInstance(summary['total_tests'], int)
        self.assertIsInstance(summary['passed_tests'], int)
        self.assertIsInstance(summary['failed_tests'], int)
        self.assertIsInstance(summary['pass_rate'], float)
        
        # Verify calculations
        self.assertEqual(
            summary['total_tests'],
            summary['passed_tests'] + summary['failed_tests']
        )
        
        expected_pass_rate = (summary['passed_tests'] / summary['total_tests']) * 100
        self.assertAlmostEqual(summary['pass_rate'], expected_pass_rate, places=1)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.figure')
    def test_generate_chart_success(self, mock_figure, mock_savefig):
        """Test successful chart generation."""
        # Mock matplotlib components
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = (mock_fig, mock_ax)
        
        script_name = "test_script"
        chart_path = self.scorer.generate_chart(script_name, self.temp_dir)
        
        # Should return the expected path
        expected_path = os.path.join(self.temp_dir, f"{script_name}_alignment_scores.png")
        self.assertEqual(chart_path, expected_path)
        
        # Verify matplotlib was called
        mock_figure.assert_called_once()
        mock_savefig.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.figure')
    def test_generate_chart_with_custom_filename(self, mock_figure, mock_savefig):
        """Test chart generation with custom filename."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = (mock_fig, mock_ax)
        
        custom_filename = "custom_chart.png"
        chart_path = self.scorer.generate_chart("test", self.temp_dir, custom_filename)
        
        expected_path = os.path.join(self.temp_dir, custom_filename)
        self.assertEqual(chart_path, expected_path)
    
    def test_generate_chart_matplotlib_not_available(self):
        """Test chart generation when matplotlib is not available."""
        with patch('cursus.validation.alignment.alignment_scorer.plt', None):
            chart_path = self.scorer.generate_chart("test", self.temp_dir)
            self.assertIsNone(chart_path)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.figure')
    def test_generate_chart_error_handling(self, mock_figure, mock_savefig):
        """Test chart generation error handling."""
        # Mock an error during chart generation
        mock_figure.side_effect = Exception("Chart generation error")
        
        chart_path = self.scorer.generate_chart("test", self.temp_dir)
        self.assertIsNone(chart_path)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with no results
        empty_scorer = AlignmentScorer({})
        self.assertEqual(empty_scorer.calculate_overall_score(), 0.0)
        self.assertEqual(empty_scorer.get_quality_rating(), "Poor")
        
        # Test with only some levels
        partial_results = {
            'level1': {'test1': {'passed': True}},
            'level3': {'test1': {'passed': False}}
        }
        partial_scorer = AlignmentScorer(partial_results)
        score = partial_scorer.calculate_overall_score()
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)
        
        # Test with malformed data
        malformed_results = {
            'level1': {
                'test1': {'passed': True},
                'test2': {}  # Missing 'passed' field
            }
        }
        malformed_scorer = AlignmentScorer(malformed_results)
        # Should handle gracefully without crashing
        score = malformed_scorer.calculate_overall_score()
        self.assertIsInstance(score, float)
    
    def test_level_weights(self):
        """Test that level weights are applied correctly."""
        # Create results where all levels have the same pass rate
        uniform_results = {
            'level1': {
                'test1': {'passed': True},
                'test2': {'passed': False}
            },
            'level2': {
                'test1': {'passed': True},
                'test2': {'passed': False}
            },
            'level3': {
                'test1': {'passed': True},
                'test2': {'passed': False}
            },
            'level4': {
                'test1': {'passed': True},
                'test2': {'passed': False}
            }
        }
        
        uniform_scorer = AlignmentScorer(uniform_results)
        level_scores = uniform_scorer.get_level_scores()
        
        # All levels should have 50% pass rate
        for score in level_scores.values():
            self.assertEqual(score, 50.0)
        
        # But overall score should be exactly 50% since weights don't change
        # the result when all levels have the same score
        overall_score = uniform_scorer.calculate_overall_score()
        self.assertEqual(overall_score, 50.0)
    
    def test_scoring_with_issues_severity(self):
        """Test that scoring considers issue severity levels."""
        # Create results with different severity levels
        severity_results = {
            'level1': {
                'critical_test': {
                    'passed': False,
                    'issues': [{'level': 'critical', 'message': 'Critical issue'}]
                },
                'error_test': {
                    'passed': False,
                    'issues': [{'level': 'error', 'message': 'Error issue'}]
                },
                'warning_test': {
                    'passed': False,
                    'issues': [{'level': 'warning', 'message': 'Warning issue'}]
                },
                'passing_test': {
                    'passed': True,
                    'issues': []
                }
            }
        }
        
        severity_scorer = AlignmentScorer(severity_results)
        
        # Should still calculate based on pass/fail, not severity
        # (severity is used for reporting, not scoring)
        level_scores = severity_scorer.get_level_scores()
        self.assertEqual(level_scores['level1_script_contract'], 25.0)  # 1/4 passed
    
    def test_json_serialization_compatibility(self):
        """Test that scoring results can be JSON serialized."""
        summary = self.scorer.get_scoring_summary()
        
        # Should be able to serialize to JSON without errors
        json_str = json.dumps(summary)
        self.assertIsInstance(json_str, str)
        
        # Should be able to deserialize back
        deserialized = json.loads(json_str)
        self.assertEqual(deserialized['overall_score'], summary['overall_score'])
        self.assertEqual(deserialized['quality_rating'], summary['quality_rating'])


class TestAlignmentScorerIntegration(unittest.TestCase):
    """Test AlignmentScorer integration with other components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.report = self.create_sample_report()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_sample_report(self):
        """Create a sample AlignmentReport with test data."""
        report = AlignmentReport()
        
        # Add Level 1 results
        level1_result1 = ValidationResult(
            test_name="script_path_validation",
            passed=True,
            issues=[],
            details={"script_path": "/opt/ml/code/train.py"}
        )
        
        level1_result2 = ValidationResult(
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
        
        report.add_level1_result("script_path_validation", level1_result1)
        report.add_level1_result("environment_variable_validation", level1_result2)
        
        # Add Level 2 results
        level2_result = ValidationResult(
            test_name="logical_name_alignment",
            passed=True,
            issues=[]
        )
        
        report.add_level2_result("logical_name_alignment", level2_result)
        
        return report
    
    def test_scorer_creation_from_report(self):
        """Test creating scorer from AlignmentReport."""
        scorer = self.report.get_scorer()
        
        self.assertIsNotNone(scorer)
        self.assertIsInstance(scorer, AlignmentScorer)
        
        # Should have results from the report
        results = scorer.results
        self.assertIn('level1', results)
        self.assertIn('level2', results)
    
    def test_report_scoring_methods(self):
        """Test AlignmentReport scoring methods."""
        # Test overall score
        overall_score = self.report.get_alignment_score()
        self.assertIsInstance(overall_score, float)
        self.assertGreaterEqual(overall_score, 0.0)
        self.assertLessEqual(overall_score, 100.0)
        
        # Test level scores
        level_scores = self.report.get_level_scores()
        self.assertIsInstance(level_scores, dict)
        
        # Test scoring report
        scoring_report = self.report.get_scoring_report()
        self.assertIsInstance(scoring_report, dict)
        self.assertIn('overall_score', scoring_report)
        self.assertIn('level_scores', scoring_report)
    
    def test_chart_generation_integration(self):
        """Test chart generation through AlignmentReport."""
        try:
            chart_path = self.report.generate_alignment_chart(
                filename="integration_test_chart.png",
                output_dir=self.temp_dir
            )
            
            if chart_path:  # Only test if matplotlib is available
                self.assertIsInstance(chart_path, str)
                self.assertTrue(chart_path.endswith("integration_test_chart.png"))
        except ImportError:
            # matplotlib not available, skip test
            self.skipTest("matplotlib not available for chart generation")
    
    def test_enhanced_json_export_with_scoring(self):
        """Test that JSON export includes scoring information."""
        json_str = self.report.export_to_json()
        self.assertIsInstance(json_str, str)
        
        # Parse JSON to verify structure
        data = json.loads(json_str)
        
        # Should contain scoring section
        self.assertIn('scoring', data)
        scoring = data['scoring']
        
        self.assertIn('overall_score', scoring)
        self.assertIn('quality_rating', scoring)
        self.assertIn('level_scores', scoring)
        self.assertIn('scoring_summary', scoring)
    
    def test_enhanced_html_export_with_scoring(self):
        """Test that HTML export includes scoring visualizations."""
        html_str = self.report.export_to_html()
        self.assertIsInstance(html_str, str)
        
        # Should contain scoring elements
        self.assertIn('scoring-section', html_str)
        self.assertIn('score-card', html_str)
        self.assertIn('Overall Alignment Score', html_str)
        
        # Should contain level score cards
        self.assertIn('Script ↔ Contract', html_str)
        self.assertIn('Contract ↔ Specification', html_str)
    
    def test_backward_compatibility(self):
        """Test that existing functionality still works with scoring integration."""
        # Test basic report methods
        summary = self.report.generate_summary()
        self.assertIsNotNone(summary)
        
        # Test issue retrieval
        all_issues = self.report.get_all_issues()
        self.assertIsInstance(all_issues, list)
        
        # Test status methods
        self.assertIsInstance(self.report.is_passing(), bool)
        self.assertIsInstance(self.report.has_errors(), bool)
        
        # Test that these methods don't interfere with scoring
        overall_score = self.report.get_alignment_score()
        self.assertIsInstance(overall_score, float)


class TestVisualizationIntegration(unittest.TestCase):
    """Test visualization integration with UnifiedAlignmentTester."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('cursus.validation.alignment.unified_alignment_tester.UnifiedAlignmentTester')
    def test_unified_tester_chart_generation(self, mock_tester_class):
        """Test chart generation integration with UnifiedAlignmentTester."""
        # Mock the unified tester
        mock_tester = MagicMock()
        mock_tester_class.return_value = mock_tester
        
        # Mock the report with scoring capabilities
        mock_report = MagicMock()
        mock_scorer = MagicMock()
        mock_scorer.generate_chart.return_value = "/path/to/chart.png"
        mock_report.get_scorer.return_value = mock_scorer
        mock_tester.report = mock_report
        
        # Test export_report with chart generation
        from cursus.validation.alignment.unified_alignment_tester import UnifiedAlignmentTester
        tester = UnifiedAlignmentTester()
        
        # Mock the export_report method to test chart generation
        with patch.object(tester, 'export_report') as mock_export:
            mock_export.return_value = "report_output"
            
            result = tester.export_report(
                format='json',
                output_path=os.path.join(self.temp_dir, 'test_report.json'),
                generate_chart=True,
                script_name="test_script"
            )
            
            # Should call export_report with chart generation enabled
            mock_export.assert_called_once_with(
                format='json',
                output_path=os.path.join(self.temp_dir, 'test_report.json'),
                generate_chart=True,
                script_name="test_script"
            )
    
    def test_chart_file_naming_conventions(self):
        """Test chart file naming conventions."""
        scorer = AlignmentScorer({})
        
        # Test default naming
        with patch('cursus.validation.alignment.alignment_scorer.plt') as mock_plt:
            mock_plt.figure.return_value = (MagicMock(), MagicMock())
            
            chart_path = scorer.generate_chart("my_script", self.temp_dir)
            expected_path = os.path.join(self.temp_dir, "my_script_alignment_scores.png")
            self.assertEqual(chart_path, expected_path)
        
        # Test custom filename
        with patch('cursus.validation.alignment.alignment_scorer.plt') as mock_plt:
            mock_plt.figure.return_value = (MagicMock(), MagicMock())
            
            chart_path = scorer.generate_chart("my_script", self.temp_dir, "custom.png")
            expected_path = os.path.join(self.temp_dir, "custom.png")
            self.assertEqual(chart_path, expected_path)
    
    def test_chart_generation_error_recovery(self):
        """Test that chart generation errors don't break the workflow."""
        scorer = AlignmentScorer({})
        
        # Test with matplotlib import error
        with patch('cursus.validation.alignment.alignment_scorer.plt', None):
            chart_path = scorer.generate_chart("test", self.temp_dir)
            self.assertIsNone(chart_path)
        
        # Test with chart generation error
        with patch('cursus.validation.alignment.alignment_scorer.plt') as mock_plt:
            mock_plt.figure.side_effect = Exception("Chart error")
            
            chart_path = scorer.generate_chart("test", self.temp_dir)
            self.assertIsNone(chart_path)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
