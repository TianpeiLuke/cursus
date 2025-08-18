#!/usr/bin/env python3
"""
Unit tests for AlignmentScorer to verify the fix works correctly.
"""

import json
import sys
import unittest
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

# Define weights for each alignment level (higher = more important)
ALIGNMENT_LEVEL_WEIGHTS = {
    "level1_script_contract": 1.0,      # Basic script-contract alignment
    "level2_contract_spec": 1.5,        # Contract-specification alignment
    "level3_spec_dependencies": 2.0,    # Specification-dependencies alignment
    "level4_builder_config": 2.5,       # Builder-configuration alignment
}

# Quality rating thresholds
ALIGNMENT_RATING_LEVELS = {
    90: "Excellent",     # 90-100: Excellent alignment
    80: "Good",          # 80-89: Good alignment
    70: "Satisfactory",  # 70-79: Satisfactory alignment
    60: "Needs Work",    # 60-69: Needs improvement
    0: "Poor"            # 0-59: Poor alignment
}

# Test importance weights for fine-tuning
ALIGNMENT_TEST_IMPORTANCE = {
    "script_contract_path_alignment": 1.5,
    "contract_spec_logical_names": 1.4,
    "spec_dependency_resolution": 1.3,
    "builder_config_environment_vars": 1.2,
    "script_contract_environment_vars": 1.2,
    "contract_spec_dependency_mapping": 1.3,
    "spec_dependency_property_paths": 1.4,
    "builder_config_specification_alignment": 1.5,
    # Default weight for other tests
}


class AlignmentScorer:
    """
    Scorer for evaluating alignment validation quality based on validation results.
    """
    
    def __init__(self, validation_results: Dict[str, Any]):
        """Initialize with alignment validation results."""
        self.results = validation_results
        self.level_results = self._group_by_level()
        
    def _group_by_level(self) -> Dict[str, Dict[str, Any]]:
        """Group validation results by alignment level using smart pattern detection."""
        grouped = {level: {} for level in ALIGNMENT_LEVEL_WEIGHTS.keys()}
        
        # Handle the actual alignment report format with level1_results, level2_results, etc.
        for key, value in self.results.items():
            if key.endswith('_results') and isinstance(value, dict):
                # Map level1_results -> level1_script_contract, etc.
                if key == 'level1_results':
                    grouped['level1_script_contract'] = value
                elif key == 'level2_results':
                    grouped['level2_contract_spec'] = value
                elif key == 'level3_results':
                    grouped['level3_spec_dependencies'] = value
                elif key == 'level4_results':
                    grouped['level4_builder_config'] = value
        
        return grouped
    
    def _is_test_passed(self, result: Any) -> bool:
        """Determine if a test passed based on its result structure."""
        if isinstance(result, dict):
            # Check for common pass/fail indicators
            if 'passed' in result:
                return bool(result['passed'])
            elif 'success' in result:
                return bool(result['success'])
            elif 'status' in result:
                return result['status'] in ['passed', 'success', 'ok']
            elif 'errors' in result:
                return len(result.get('errors', [])) == 0
            elif 'issues' in result:
                # Consider passed if no critical or error issues
                issues = result.get('issues', [])
                critical_errors = [i for i in issues if i.get('severity') in ['critical', 'error']]
                return len(critical_errors) == 0
        elif isinstance(result, bool):
            return result
        elif isinstance(result, str):
            return result.lower() in ['passed', 'success', 'ok', 'true']
        
        # Default to False if we can't determine
        return False
    
    def calculate_level_score(self, level: str) -> Tuple[float, int, int]:
        """Calculate score for a specific alignment level."""
        if level not in self.level_results:
            return 0.0, 0, 0
            
        level_tests = self.level_results[level]
        if not level_tests:
            return 0.0, 0, 0
            
        total_weight = 0.0
        weighted_score = 0.0
        
        for test_name, result in level_tests.items():
            # Get test importance weight (default to 1.0 if not specified)
            importance = ALIGNMENT_TEST_IMPORTANCE.get(test_name, 1.0)
            total_weight += importance
            
            # Determine if test passed based on result structure
            test_passed = self._is_test_passed(result)
            if test_passed:
                weighted_score += importance
        
        # Calculate percentage score
        score = (weighted_score / total_weight) * 100.0 if total_weight > 0 else 0.0
        
        # Count passed tests
        passed = sum(1 for result in level_tests.values() if self._is_test_passed(result))
        total = len(level_tests)
        
        return score, passed, total
    
    def calculate_overall_score(self) -> float:
        """Calculate overall alignment score across all levels."""
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for level, weight in ALIGNMENT_LEVEL_WEIGHTS.items():
            level_score, _, _ = self.calculate_level_score(level)
            total_weighted_score += level_score * weight
            total_weight += weight
        
        overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        return min(100.0, max(0.0, overall_score))
    
    def get_rating(self, score: float) -> str:
        """Get quality rating based on score."""
        for threshold, rating in sorted(ALIGNMENT_RATING_LEVELS.items(), reverse=True):
            if score >= threshold:
                return rating
        return "Invalid"
    
    def print_report(self) -> None:
        """Print a formatted alignment score report to the console."""
        print("\n" + "=" * 80)
        print(f"ALIGNMENT VALIDATION QUALITY SCORE REPORT")
        print("=" * 80)
        
        # Overall score and rating
        overall_score = self.calculate_overall_score()
        overall_rating = self.get_rating(overall_score)
        
        total_passed = 0
        total_tests = 0
        
        print(f"\nOverall Score: {overall_score:.1f}/100 - {overall_rating}")
        
        # Level scores
        print("\nScores by Alignment Level:")
        level_names = {
            "level1_script_contract": "Level 1 (Script ↔ Contract)",
            "level2_contract_spec": "Level 2 (Contract ↔ Specification)",
            "level3_spec_dependencies": "Level 3 (Specification ↔ Dependencies)",
            "level4_builder_config": "Level 4 (Builder ↔ Configuration)"
        }
        
        for level in ALIGNMENT_LEVEL_WEIGHTS.keys():
            score, passed, total = self.calculate_level_score(level)
            display_name = level_names.get(level, level)
            print(f"  {display_name}: {score:.1f}/100 ({passed}/{total} tests)")
            total_passed += passed
            total_tests += total
        
        pass_rate = (total_passed / total_tests) * 100.0 if total_tests > 0 else 0.0
        print(f"\nPass Rate: {pass_rate:.1f}% ({total_passed}/{total_tests} tests)")
        
        print("\n" + "=" * 80)


class TestAlignmentScorer(unittest.TestCase):
    """Test cases for AlignmentScorer."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Sample validation results in the expected format
        self.sample_results = {
            "level1_results": {
                "script_contract_path_alignment": {"passed": True, "message": "Path alignment verified"},
                "script_contract_environment_vars": {"passed": True, "message": "Environment variables aligned"}
            },
            "level2_results": {
                "contract_spec_logical_names": {"passed": True, "message": "Logical names aligned"},
                "contract_spec_dependency_mapping": {"passed": False, "message": "Dependency mapping failed"}
            },
            "level3_results": {
                "spec_dependency_resolution": {"passed": True, "message": "Dependencies resolved"},
                "spec_dependency_property_paths": {"passed": True, "message": "Property paths verified"}
            },
            "level4_results": {
                "builder_config_environment_vars": {"passed": True, "message": "Config environment vars aligned"},
                "builder_config_specification_alignment": {"passed": True, "message": "Specification alignment verified"}
            }
        }
    
    def test_group_by_level(self):
        """Test that results are correctly grouped by level."""
        scorer = AlignmentScorer(self.sample_results)
        
        # Check that all levels are present
        expected_levels = ["level1_script_contract", "level2_contract_spec", 
                          "level3_spec_dependencies", "level4_builder_config"]
        self.assertEqual(set(scorer.level_results.keys()), set(expected_levels))
        
        # Check that level1_results maps to level1_script_contract
        self.assertEqual(scorer.level_results["level1_script_contract"], 
                        self.sample_results["level1_results"])
        
        # Check that level2_results maps to level2_contract_spec
        self.assertEqual(scorer.level_results["level2_contract_spec"], 
                        self.sample_results["level2_results"])
    
    def test_is_test_passed(self):
        """Test the _is_test_passed method with various result formats."""
        scorer = AlignmentScorer(self.sample_results)
        
        # Test with passed=True
        self.assertTrue(scorer._is_test_passed({"passed": True}))
        self.assertFalse(scorer._is_test_passed({"passed": False}))
        
        # Test with success=True
        self.assertTrue(scorer._is_test_passed({"success": True}))
        self.assertFalse(scorer._is_test_passed({"success": False}))
        
        # Test with status
        self.assertTrue(scorer._is_test_passed({"status": "passed"}))
        self.assertTrue(scorer._is_test_passed({"status": "success"}))
        self.assertFalse(scorer._is_test_passed({"status": "failed"}))
        
        # Test with errors
        self.assertTrue(scorer._is_test_passed({"errors": []}))
        self.assertFalse(scorer._is_test_passed({"errors": ["some error"]}))
        
        # Test with boolean
        self.assertTrue(scorer._is_test_passed(True))
        self.assertFalse(scorer._is_test_passed(False))
        
        # Test with string
        self.assertTrue(scorer._is_test_passed("passed"))
        self.assertTrue(scorer._is_test_passed("success"))
        self.assertFalse(scorer._is_test_passed("failed"))
    
    def test_calculate_level_score(self):
        """Test level score calculation."""
        scorer = AlignmentScorer(self.sample_results)
        
        # Test level1 (all tests pass)
        score, passed, total = scorer.calculate_level_score("level1_script_contract")
        self.assertEqual(score, 100.0)
        self.assertEqual(passed, 2)
        self.assertEqual(total, 2)
        
        # Test level2 (1 pass, 1 fail)
        score, passed, total = scorer.calculate_level_score("level2_contract_spec")
        self.assertEqual(passed, 1)
        self.assertEqual(total, 2)
        # Score should be less than 100 but greater than 0
        self.assertGreater(score, 0)
        self.assertLess(score, 100)
    
    def test_calculate_overall_score(self):
        """Test overall score calculation."""
        scorer = AlignmentScorer(self.sample_results)
        overall_score = scorer.calculate_overall_score()
        
        # Should be between 0 and 100
        self.assertGreaterEqual(overall_score, 0)
        self.assertLessEqual(overall_score, 100)
        
        # Should be greater than 0 since most tests pass
        self.assertGreater(overall_score, 0)
    
    def test_get_rating(self):
        """Test rating assignment based on score."""
        scorer = AlignmentScorer(self.sample_results)
        
        self.assertEqual(scorer.get_rating(95), "Excellent")
        self.assertEqual(scorer.get_rating(85), "Good")
        self.assertEqual(scorer.get_rating(75), "Satisfactory")
        self.assertEqual(scorer.get_rating(65), "Needs Work")
        self.assertEqual(scorer.get_rating(50), "Poor")
    
    def test_with_real_report_data(self):
        """Test with actual report data if available."""
        report_path = Path("test/steps/scripts/alignment_validation/reports/json/xgboost_model_evaluation_alignment_report.json")
        
        if report_path.exists():
            with open(report_path, 'r') as f:
                report_data = json.load(f)
            
            scorer = AlignmentScorer(report_data)
            
            # Test that we can calculate scores without errors
            overall_score = scorer.calculate_overall_score()
            self.assertIsInstance(overall_score, float)
            self.assertGreaterEqual(overall_score, 0)
            self.assertLessEqual(overall_score, 100)
            
            # Test that we can get a rating
            rating = scorer.get_rating(overall_score)
            self.assertIn(rating, ["Excellent", "Good", "Satisfactory", "Needs Work", "Poor"])
            
            print(f"\nReal report test - Overall score: {overall_score:.1f}/100 - {rating}")
            
            # Print detailed results for debugging
            for level in ALIGNMENT_LEVEL_WEIGHTS.keys():
                score, passed, total = scorer.calculate_level_score(level)
                print(f"  {level}: {score:.1f}/100 ({passed}/{total} tests)")


def main():
    """Run the tests and demonstrate the scorer with real data."""
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Test with real report data if available
    report_path = Path("test/steps/scripts/alignment_validation/reports/json/xgboost_model_evaluation_alignment_report.json")
    
    if report_path.exists():
        print("\n" + "="*80)
        print("TESTING WITH REAL REPORT DATA")
        print("="*80)
        
        with open(report_path, 'r') as f:
            report_data = json.load(f)
        
        print(f'Report keys: {list(report_data.keys())}')
        
        scorer = AlignmentScorer(report_data)
        print(f'Level results keys: {list(scorer.level_results.keys())}')
        
        # Check what data we have for each level
        for level, data in scorer.level_results.items():
            if len(data) > 3:
                print(f'{level}: {len(data)} items - {list(data.keys())[:3]}...')
            else:
                print(f'{level}: {len(data)} items - {list(data.keys())}')
        
        # Calculate and display results
        overall_score = scorer.calculate_overall_score()
        print(f'\nOverall score: {overall_score:.1f}/100')
        
        scorer.print_report()
    else:
        print(f"\nReal report file not found: {report_path}")


if __name__ == "__main__":
    main()
