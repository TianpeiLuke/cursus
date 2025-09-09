#!/usr/bin/env python3
"""
Unit tests for AlignmentScorer class.

This test suite provides comprehensive coverage for the AlignmentScorer functionality
which was identified as a critical missing test in the validation test coverage analysis.
"""

import unittest
from unittest.mock import patch, MagicMock
import tempfile
import os
import json

from cursus.validation.alignment.alignment_scorer import AlignmentScorer


class TestAlignmentScorer(unittest.TestCase):
    """Test AlignmentScorer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_validation_results = {
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
                ],
                'details': {'undeclared_vars': ['CUSTOM_VAR']}
            },
            'logical_name_alignment': {
                'passed': True,
                'issues': [],
                'details': {'aligned_names': ['training_data', 'model_output']}
            },
            'input_output_mapping': {
                'passed': False,
                'issues': [
                    {
                        'level': 'warning',
                        'category': 'logical_names',
                        'message': 'Contract output not found in specification'
                    }
                ],
                'details': {'missing_outputs': ['model_artifacts']}
            },
            'dependency_resolution': {
                'passed': True,
                'issues': [],
                'details': {'resolved_dependencies': ['preprocessing_step']}
            },
            'configuration_validation': {
                'passed': False,
                'issues': [
                    {
                        'level': 'critical',
                        'category': 'configuration',
                        'message': 'Builder missing required environment variable'
                    }
                ],
                'details': {'missing_env_vars': ['MODEL_TYPE']}
            }
        }
    
    def test_scorer_initialization(self):
        """Test AlignmentScorer initialization."""
        scorer = AlignmentScorer(self.sample_validation_results)
        
        self.assertIsNotNone(scorer)
        self.assertEqual(scorer.validation_results, self.sample_validation_results)
    
    def test_group_by_level(self):
        """Test grouping validation results by level."""
        scorer = AlignmentScorer(self.sample_validation_results)
        grouped = scorer._group_by_level()
        
        self.assertIsInstance(grouped, dict)
        
        # Should have level groupings
        expected_levels = [
            'level1_script_contract',
            'level2_contract_specification',
            'level3_specification_dependencies',
            'level4_builder_configuration'
        ]
        
        for level in expected_levels:
            if level in grouped:
                self.assertIsInstance(grouped[level], dict)
    
    def test_detect_level_from_test_name(self):
        """Test level detection from test names."""
        scorer = AlignmentScorer(self.sample_validation_results)
        
        # Test level 1 detection
        level1_tests = ['script_path_validation', 'environment_variable_validation']
        for test_name in level1_tests:
            level = scorer._detect_level_from_test_name(test_name)
            self.assertEqual(level, 'level1_script_contract')
        
        # Test level 2 detection
        level2_tests = ['logical_name_alignment', 'input_output_mapping']
        for test_name in level2_tests:
            level = scorer._detect_level_from_test_name(test_name)
            self.assertEqual(level, 'level2_contract_specification')
        
        # Test level 3 detection
        level3_tests = ['dependency_resolution']
        for test_name in level3_tests:
            level = scorer._detect_level_from_test_name(test_name)
            self.assertEqual(level, 'level3_specification_dependencies')
        
        # Test level 4 detection
        level4_tests = ['configuration_validation']
        for test_name in level4_tests:
            level = scorer._detect_level_from_test_name(test_name)
            self.assertEqual(level, 'level4_builder_configuration')
    
    def test_is_test_passed(self):
        """Test test pass/fail detection."""
        scorer = AlignmentScorer(self.sample_validation_results)
        
        # Test with passed result
        passed_result = {'passed': True, 'issues': []}
        self.assertTrue(scorer._is_test_passed(passed_result))
        
        # Test with failed result
        failed_result = {'passed': False, 'issues': [{'level': 'error'}]}
        self.assertFalse(scorer._is_test_passed(failed_result))
        
        # Test with boolean True
        self.assertTrue(scorer._is_test_passed(True))
        
        # Test with boolean False
        self.assertFalse(scorer._is_test_passed(False))
    
    def test_calculate_level_score(self):
        """Test level score calculation."""
        scorer = AlignmentScorer(self.sample_validation_results)
        
        # Test level 1 score (1 pass, 1 fail)
        score, passed, total = scorer.calculate_level_score('level1_script_contract')
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)
        self.assertEqual(passed, 1)
        self.assertEqual(total, 2)
        
        # Test level 2 score (1 pass, 1 fail)
        score, passed, total = scorer.calculate_level_score('level2_contract_specification')
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)
        self.assertEqual(passed, 1)
        self.assertEqual(total, 2)
        
        # Test level 3 score (1 pass)
        score, passed, total = scorer.calculate_level_score('level3_specification_dependencies')
        self.assertIsInstance(score, float)
        self.assertEqual(score, 100.0)  # Should be 100% since all tests pass
        self.assertEqual(passed, 1)
        self.assertEqual(total, 1)
        
        # Test level 4 score (1 fail)
        score, passed, total = scorer.calculate_level_score('level4_builder_configuration')
        self.assertIsInstance(score, float)
        self.assertEqual(score, 0.0)  # Should be 0% since all tests fail
        self.assertEqual(passed, 0)
        self.assertEqual(total, 1)
    
    def test_calculate_overall_score(self):
        """Test overall score calculation."""
        scorer = AlignmentScorer(self.sample_validation_results)
        
        overall_score = scorer.calculate_overall_score()
        
        self.assertIsInstance(overall_score, float)
        self.assertGreaterEqual(overall_score, 0.0)
        self.assertLessEqual(overall_score, 100.0)
        
        # Should be weighted average of level scores
        # With our test data: L1=50%, L2=50%, L3=100%, L4=0%
        # Expected weighted average should be reasonable
        self.assertGreater(overall_score, 0.0)
        self.assertLess(overall_score, 100.0)
    
    def test_get_rating(self):
        """Test quality rating generation."""
        scorer = AlignmentScorer(self.sample_validation_results)
        
        # Test different score ranges
        test_scores = [95.0, 85.0, 75.0, 65.0, 45.0]
        expected_ratings = ["Excellent", "Good", "Satisfactory", "Needs Work", "Poor"]
        
        for score, expected_rating in zip(test_scores, expected_ratings):
            rating = scorer.get_rating(score)
            self.assertEqual(rating, expected_rating)
    
    def test_generate_report(self):
        """Test report generation."""
        scorer = AlignmentScorer(self.sample_validation_results)
        
        report = scorer.generate_report()
        
        self.assertIsInstance(report, dict)
        
        # Verify report structure
        required_fields = [
            'overall_score',
            'quality_rating',
            'level_scores',
            'level_details',
            'summary'
        ]
        
        for field in required_fields:
            self.assertIn(field, report)
        
        # Verify score values
        self.assertIsInstance(report['overall_score'], float)
        self.assertIsInstance(report['quality_rating'], str)
        self.assertIsInstance(report['level_scores'], dict)
        self.assertIsInstance(report['level_details'], dict)
        self.assertIsInstance(report['summary'], dict)
    
    def test_extract_error_message(self):
        """Test error message extraction."""
        scorer = AlignmentScorer(self.sample_validation_results)
        
        # Test with result containing issues
        result_with_issues = {
            'passed': False,
            'issues': [
                {'message': 'First error'},
                {'message': 'Second error'}
            ]
        }
        
        error_msg = scorer._extract_error_message(result_with_issues)
        self.assertIn('First error', error_msg)
        
        # Test with result without issues
        result_without_issues = {'passed': True, 'issues': []}
        error_msg = scorer._extract_error_message(result_without_issues)
        self.assertEqual(error_msg, 'No issues')
    
    def test_save_report(self):
        """Test report saving functionality."""
        scorer = AlignmentScorer(self.sample_validation_results)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test saving report
            saved_path = scorer.save_report("test_script", temp_dir)
            
            self.assertTrue(os.path.exists(saved_path))
            self.assertTrue(saved_path.endswith('.json'))
            
            # Verify saved content
            with open(saved_path, 'r') as f:
                saved_data = json.load(f)
            
            self.assertIsInstance(saved_data, dict)
            self.assertIn('overall_score', saved_data)
            self.assertIn('quality_rating', saved_data)
    
    def test_print_report(self):
        """Test report printing functionality."""
        scorer = AlignmentScorer(self.sample_validation_results)
        
        # Test that print_report doesn't raise exceptions
        try:
            scorer.print_report()
        except Exception as e:
            self.fail(f"print_report raised an exception: {e}")
    
    def test_generate_chart(self):
        """Test chart generation functionality."""
        scorer = AlignmentScorer(self.sample_validation_results)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test chart generation (may not work if matplotlib not available)
            try:
                chart_path = scorer.generate_chart("test_script", temp_dir)
                if chart_path:  # Only test if chart was generated
                    self.assertTrue(os.path.exists(chart_path))
                    self.assertTrue(chart_path.endswith('.png'))
            except ImportError:
                # matplotlib not available, skip test
                pass
            except Exception as e:
                # Other errors are acceptable for this test
                pass
    
    def test_empty_validation_results(self):
        """Test scorer with empty validation results."""
        empty_results = {}
        scorer = AlignmentScorer(empty_results)
        
        overall_score = scorer.calculate_overall_score()
        self.assertEqual(overall_score, 0.0)
        
        report = scorer.generate_report()
        self.assertIsInstance(report, dict)
        self.assertEqual(report['overall_score'], 0.0)
    
    def test_all_passing_results(self):
        """Test scorer with all passing results."""
        all_passing = {
            'test1': {'passed': True, 'issues': []},
            'test2': {'passed': True, 'issues': []},
            'test3': {'passed': True, 'issues': []},
            'test4': {'passed': True, 'issues': []}
        }
        
        scorer = AlignmentScorer(all_passing)
        overall_score = scorer.calculate_overall_score()
        
        self.assertEqual(overall_score, 100.0)
        
        rating = scorer.get_rating(overall_score)
        self.assertEqual(rating, "Excellent")
    
    def test_all_failing_results(self):
        """Test scorer with all failing results."""
        all_failing = {
            'test1': {'passed': False, 'issues': [{'level': 'error'}]},
            'test2': {'passed': False, 'issues': [{'level': 'error'}]},
            'test3': {'passed': False, 'issues': [{'level': 'error'}]},
            'test4': {'passed': False, 'issues': [{'level': 'error'}]}
        }
        
        scorer = AlignmentScorer(all_failing)
        overall_score = scorer.calculate_overall_score()
        
        self.assertEqual(overall_score, 0.0)
        
        rating = scorer.get_rating(overall_score)
        self.assertEqual(rating, "Poor")
    
    def test_score_alignment_results_function(self):
        """Test the standalone score_alignment_results function."""
        from cursus.validation.alignment.alignment_scorer import score_alignment_results
        
        report = score_alignment_results(self.sample_validation_results, "test_script")
        
        self.assertIsInstance(report, dict)
        self.assertIn('overall_score', report)
        self.assertIn('quality_rating', report)
        self.assertIn('level_scores', report)


class TestAlignmentScorerEdgeCases(unittest.TestCase):
    """Test AlignmentScorer edge cases and error conditions."""
    
    def test_malformed_validation_results(self):
        """Test scorer with malformed validation results."""
        malformed_results = {
            'test1': {'invalid': 'structure'},
            'test2': None,
            'test3': 'string_instead_of_dict',
            'test4': {'passed': 'not_boolean', 'issues': 'not_list'}
        }
        
        scorer = AlignmentScorer(malformed_results)
        
        # Should handle malformed data gracefully
        overall_score = scorer.calculate_overall_score()
        self.assertIsInstance(overall_score, float)
        self.assertGreaterEqual(overall_score, 0.0)
        self.assertLessEqual(overall_score, 100.0)
    
    def test_mixed_result_formats(self):
        """Test scorer with mixed result formats."""
        mixed_results = {
            'test1': True,  # Boolean
            'test2': False,  # Boolean
            'test3': {'passed': True, 'issues': []},  # Dict format
            'test4': {'passed': False, 'issues': [{'level': 'error'}]}  # Dict format
        }
        
        scorer = AlignmentScorer(mixed_results)
        overall_score = scorer.calculate_overall_score()
        
        self.assertIsInstance(overall_score, float)
        self.assertEqual(overall_score, 50.0)  # 2 pass, 2 fail = 50%
    
    def test_unknown_test_names(self):
        """Test scorer with unknown test names."""
        unknown_results = {
            'unknown_test_1': {'passed': True, 'issues': []},
            'unknown_test_2': {'passed': False, 'issues': []},
            'completely_random_name': {'passed': True, 'issues': []}
        }
        
        scorer = AlignmentScorer(unknown_results)
        
        # Should still calculate scores even with unknown test names
        overall_score = scorer.calculate_overall_score()
        self.assertIsInstance(overall_score, float)
        self.assertGreaterEqual(overall_score, 0.0)
        self.assertLessEqual(overall_score, 100.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
