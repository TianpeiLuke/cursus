"""
Alignment Validation Scoring System.

This module provides scoring capabilities for alignment validation results,
similar to the step builder scoring system. It evaluates alignment quality
across multiple levels and generates comprehensive quality metrics.
"""

from typing import Dict, List, Any, Tuple, Optional
import json
from pathlib import Path
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

# Chart styling configuration
CHART_CONFIG = {
    "figure_size": (10, 6),
    "colors": {
        "excellent": "#28a745",    # Green
        "good": "#90ee90",         # Light green
        "satisfactory": "#ffa500", # Orange
        "needs_work": "#fa8072",   # Salmon
        "poor": "#dc3545"          # Red
    },
    "grid_style": {
        "axis": "y",
        "linestyle": "--",
        "alpha": 0.7
    }
}


class AlignmentScorer:
    """
    Scorer for evaluating alignment validation quality based on validation results.
    
    This class calculates scores for each alignment level and provides an overall
    score and rating for alignment validation quality.
    """
    
    def __init__(self, validation_results: Dict[str, Any]):
        """
        Initialize with alignment validation results.
        
        Args:
            validation_results: Dictionary containing validation results from alignment validation
        """
        self.results = validation_results
        self.level_results = self._group_by_level()
        
    def _group_by_level(self) -> Dict[str, Dict[str, Any]]:
        """
        Group validation results by alignment level using smart pattern detection.
        
        Returns:
            Dictionary mapping levels to their validation results
        """
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
            elif key.startswith('level') and isinstance(value, dict):
                # Handle direct level keys (level1, level2, etc.)
                if key == 'level1':
                    grouped['level1_script_contract'][key] = value
                elif key == 'level2':
                    grouped['level2_contract_spec'][key] = value
                elif key == 'level3':
                    grouped['level3_spec_dependencies'][key] = value
                elif key == 'level4':
                    grouped['level4_builder_config'][key] = value
        
        # Also handle individual test results if they exist
        test_results = self.results.get('tests', self.results.get('validations', {}))
        if isinstance(test_results, dict):
            for test_name, result in test_results.items():
                level = self._detect_level_from_test_name(test_name)
                if level:
                    grouped[level][test_name] = result
        
        return grouped
    
    def _detect_level_from_test_name(self, test_name: str) -> Optional[str]:
        """
        Detect alignment level from test name using pattern detection.
        
        Args:
            test_name: Name of the validation test
            
        Returns:
            Level name or None if level cannot be determined
        """
        test_lower = test_name.lower()
        
        # Level 1: Script ↔ Contract alignment
        level1_keywords = [
            "script_contract", "script", "contract", "path_alignment",
            "environment_vars", "entry_point", "script_path"
        ]
        if any(keyword in test_lower for keyword in level1_keywords):
            return "level1_script_contract"
        
        # Level 2: Contract ↔ Specification alignment
        level2_keywords = [
            "contract_spec", "logical_names", "spec_contract",
            "specification", "contract_alignment", "spec_alignment"
        ]
        if any(keyword in test_lower for keyword in level2_keywords):
            return "level2_contract_spec"
        
        # Level 3: Specification ↔ Dependencies alignment
        level3_keywords = [
            "spec_dependencies", "dependency", "dependencies",
            "property_paths", "dependency_resolution", "spec_dependency"
        ]
        if any(keyword in test_lower for keyword in level3_keywords):
            return "level3_spec_dependencies"
        
        # Level 4: Builder ↔ Configuration alignment
        level4_keywords = [
            "builder_config", "configuration", "builder", "config_alignment",
            "specification_alignment", "builder_specification"
        ]
        if any(keyword in test_lower for keyword in level4_keywords):
            return "level4_builder_config"
        
        return None
    
    def calculate_level_score(self, level: str) -> Tuple[float, int, int]:
        """
        Calculate score for a specific alignment level.
        
        Args:
            level: Name of the level to score
            
        Returns:
            Tuple containing (score, passed_tests, total_tests)
        """
        if level not in self.level_results:
            return 0.0, 0, 0
            
        level_tests = self.level_results[level]
        if not level_tests:
            return 0.0, 0, 0
            
        total_weight = 0.0
        weighted_score = 0.0
        passed_count = 0
        total_count = 0
        
        for script_name, script_result in level_tests.items():
            # Each script_result should have a 'passed' field
            total_count += 1
            
            # Get test importance weight (default to 1.0 if not specified)
            importance = ALIGNMENT_TEST_IMPORTANCE.get(script_name, 1.0)
            total_weight += importance
            
            # Determine if test passed based on result structure
            test_passed = self._is_test_passed(script_result)
            if test_passed:
                weighted_score += importance
                passed_count += 1
        
        # Calculate percentage score
        score = (weighted_score / total_weight) * 100.0 if total_weight > 0 else 0.0
        
        return score, passed_count, total_count
    
    def _is_test_passed(self, result: Any) -> bool:
        """
        Determine if a test passed based on its result structure.
        
        Args:
            result: Test result (can be various formats)
            
        Returns:
            True if test passed, False otherwise
        """
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
    
    def calculate_overall_score(self) -> float:
        """
        Calculate overall alignment score across all levels.
        
        Returns:
            Overall score (0-100)
        """
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for level, weight in ALIGNMENT_LEVEL_WEIGHTS.items():
            level_score, _, _ = self.calculate_level_score(level)
            total_weighted_score += level_score * weight
            total_weight += weight
        
        overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        return min(100.0, max(0.0, overall_score))
    
    def get_rating(self, score: float) -> str:
        """
        Get quality rating based on score.
        
        Args:
            score: Score to rate (0-100)
            
        Returns:
            Rating string
        """
        for threshold, rating in sorted(ALIGNMENT_RATING_LEVELS.items(), reverse=True):
            if score >= threshold:
                return rating
        return "Invalid"
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive alignment score report.
        
        Returns:
            Dictionary containing the full score report
        """
        level_scores = {}
        total_passed = 0
        total_tests = 0
        
        # Calculate scores for each level
        for level in ALIGNMENT_LEVEL_WEIGHTS.keys():
            score, passed, total = self.calculate_level_score(level)
            level_scores[level] = {
                "score": score,
                "passed": passed,
                "total": total,
                "tests": {
                    test_name: self._is_test_passed(result)
                    for test_name, result in self.level_results.get(level, {}).items()
                }
            }
            total_passed += passed
            total_tests += total
        
        # Calculate overall score
        overall_score = self.calculate_overall_score()
        overall_rating = self.get_rating(overall_score)
        
        # Create report
        report = {
            "overall": {
                "score": overall_score,
                "rating": overall_rating,
                "passed": total_passed,
                "total": total_tests,
                "pass_rate": (total_passed / total_tests) * 100.0 if total_tests > 0 else 0.0
            },
            "levels": level_scores,
            "failed_tests": [
                {
                    "name": test_name,
                    "level": level,
                    "error": self._extract_error_message(result)
                }
                for level, tests in self.level_results.items()
                for test_name, result in tests.items()
                if not self._is_test_passed(result)
            ],
            "metadata": {
                "scoring_system": "alignment_validation",
                "level_weights": ALIGNMENT_LEVEL_WEIGHTS,
                "test_importance": ALIGNMENT_TEST_IMPORTANCE
            }
        }
        
        return report
    
    def _extract_error_message(self, result: Any) -> str:
        """
        Extract error message from test result.
        
        Args:
            result: Test result
            
        Returns:
            Error message string
        """
        if isinstance(result, dict):
            if 'error' in result:
                return str(result['error'])
            elif 'message' in result:
                return str(result['message'])
            elif 'issues' in result:
                issues = result.get('issues', [])
                if issues:
                    return f"{len(issues)} alignment issues found"
            elif 'errors' in result:
                errors = result.get('errors', [])
                if errors:
                    return f"{len(errors)} errors found"
        
        return "Test failed"
    
    def save_report(self, script_name: str, output_dir: str = "alignment_reports") -> str:
        """
        Save the alignment score report to a JSON file.
        
        Args:
            script_name: Name of the script being validated
            output_dir: Directory to save the report in
            
        Returns:
            Path to the saved report
        """
        report = self.generate_report()
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create filename
        filename = f"{output_dir}/{script_name}_alignment_score_report.json"
        
        # Save report
        with open(filename, "w") as f:
            json.dump(report, f, indent=2)
            
        return filename
    
    def print_report(self) -> None:
        """Print a formatted alignment score report to the console."""
        report = self.generate_report()
        
        print("\n" + "=" * 80)
        print(f"ALIGNMENT VALIDATION QUALITY SCORE REPORT")
        print("=" * 80)
        
        # Overall score and rating
        overall = report["overall"]
        print(f"\nOverall Score: {overall['score']:.1f}/100 - {overall['rating']}")
        print(f"Pass Rate: {overall['pass_rate']:.1f}% ({overall['passed']}/{overall['total']} tests)")
        
        # Level scores
        print("\nScores by Alignment Level:")
        level_names = {
            "level1_script_contract": "Level 1 (Script ↔ Contract)",
            "level2_contract_spec": "Level 2 (Contract ↔ Specification)",
            "level3_spec_dependencies": "Level 3 (Specification ↔ Dependencies)",
            "level4_builder_config": "Level 4 (Builder ↔ Configuration)"
        }
        
        for level, data in report["levels"].items():
            display_name = level_names.get(level, level)
            print(f"  {display_name}: {data['score']:.1f}/100 ({data['passed']}/{data['total']} tests)")
        
        # Failed tests
        if report["failed_tests"]:
            print("\nFailed Tests:")
            for test in report["failed_tests"]:
                level_name = level_names.get(test['level'], test['level'])
                print(f"  ❌ {test['name']} ({level_name}): {test['error']}")
        
        print("\n" + "=" * 80)
    
    def generate_chart(self, script_name: str, output_dir: str = "alignment_reports") -> Optional[str]:
        """
        Generate a chart visualization of the alignment score report.
        
        Args:
            script_name: Name of the script being validated
            output_dir: Directory to save the chart in
            
        Returns:
            Path to the saved chart or None if matplotlib is not available
        """
        try:
            import matplotlib.pyplot as plt
            
            report = self.generate_report()
            
            # Create level names and scores
            level_names = [
                "L1 Script↔Contract",
                "L2 Contract↔Spec", 
                "L3 Spec↔Dependencies",
                "L4 Builder↔Config"
            ]
            
            levels = []
            scores = []
            colors = []
            
            level_mapping = {
                "level1_script_contract": 0,
                "level2_contract_spec": 1,
                "level3_spec_dependencies": 2,
                "level4_builder_config": 3
            }
            
            # Initialize with zeros
            level_scores_array = [0.0] * 4
            
            for level, data in report["levels"].items():
                if level in level_mapping:
                    index = level_mapping[level]
                    level_scores_array[index] = data["score"]
            
            # Create final arrays
            for i, (name, score) in enumerate(zip(level_names, level_scores_array)):
                levels.append(name)
                scores.append(score)
                
                # Choose color based on score
                if score >= 90:
                    colors.append(CHART_CONFIG["colors"]["excellent"])
                elif score >= 80:
                    colors.append(CHART_CONFIG["colors"]["good"])
                elif score >= 70:
                    colors.append(CHART_CONFIG["colors"]["satisfactory"])
                elif score >= 60:
                    colors.append(CHART_CONFIG["colors"]["needs_work"])
                else:
                    colors.append(CHART_CONFIG["colors"]["poor"])
            
            # Create the figure
            plt.figure(figsize=CHART_CONFIG["figure_size"])
            
            # Create bar chart
            bars = plt.bar(levels, scores, color=colors)
            
            # Add overall score line
            overall_score = report["overall"]["score"]
            plt.axhline(y=overall_score, color='blue', linestyle='-', alpha=0.7)
            plt.text(len(levels)-0.5, overall_score+2, 
                    f"Overall: {overall_score:.1f} ({report['overall']['rating']})", 
                    color='blue')
            
            # Add labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f"{height:.1f}%", ha='center', va='bottom')
            
            # Set chart properties
            plt.title(f"Alignment Validation Quality Score: {script_name}")
            plt.ylabel("Score (%)")
            plt.ylim(0, 105)
            plt.grid(**CHART_CONFIG["grid_style"])
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Save figure
            filename = f"{output_dir}/{script_name}_alignment_score_chart.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
        except ImportError:
            print("matplotlib not available, skipping chart generation")
            return None
        except Exception as e:
            print(f"Could not generate chart: {str(e)}")
            return None


def score_alignment_results(validation_results: Dict[str, Any], 
                          script_name: str = "Unknown",
                          save_report: bool = True,
                          output_dir: str = "alignment_reports",
                          generate_chart: bool = True) -> Dict[str, Any]:
    """
    Score alignment validation results for a script.
    
    Args:
        validation_results: Dictionary containing validation results
        script_name: Name of the script being validated
        save_report: Whether to save the report to a file
        output_dir: Directory to save the report in
        generate_chart: Whether to generate a chart visualization
        
    Returns:
        Alignment score report dictionary
    """
    scorer = AlignmentScorer(validation_results)
    report = scorer.generate_report()
    
    # Print report
    scorer.print_report()
    
    # Save report
    if save_report:
        scorer.save_report(script_name, output_dir)
    
    # Generate chart
    if generate_chart:
        scorer.generate_chart(script_name, output_dir)
    
    return report
