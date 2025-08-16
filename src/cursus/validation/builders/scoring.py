"""
Scoring system for Universal Step Builder Tests.

This module provides a scoring mechanism to evaluate step builders based on their 
performance across the four test levels. The scoring system assigns different weights 
to different test levels, reflecting their importance in the overall architecture.
"""

from typing import Dict, Any, List, Tuple, Optional
import json
from pathlib import Path
import numpy as np

# Define weights for each test level
LEVEL_WEIGHTS = {
    "level1_interface": 1.0,    # Basic interface compliance
    "level2_specification": 1.5, # Specification and contract compliance
    "level3_step_creation": 2.0,  # Step creation and configuration validation
    "level4_integration": 2.5,   # System integration
}

# Define test to level mapping
TEST_LEVEL_MAP = {
    # Level 1: Interface tests
    "test_inheritance": "level1_interface",
    "test_required_methods": "level1_interface",
    "test_error_handling": "level1_interface",
    "test_generic_step_creation": "level1_interface",
    "test_processor_creation": "level1_interface",
    "test_generic_configuration_validation": "level1_interface",
    
    # Level 2: Specification and contract tests
    "test_specification_usage": "level2_specification",
    "test_contract_alignment": "level2_specification",
    "test_environment_variable_handling": "level2_specification",
    "test_job_arguments": "level2_specification",
    "test_environment_variables_processing": "level2_specification",
    "test_processing_job_arguments": "level2_specification",
    "test_property_files_configuration": "level2_specification",
    
    # Level 3: Step creation tests (updated from path mapping)
    "test_step_instantiation": "level3_step_creation",
    "test_step_configuration_validity": "level3_step_creation",
    "test_step_dependencies_attachment": "level3_step_creation",
    "test_step_name_generation": "level3_step_creation",
    "test_processing_step_creation": "level3_step_creation",
    "test_training_step_creation": "level3_step_creation",
    "test_transform_step_creation": "level3_step_creation",
    "test_create_model_step_creation": "level3_step_creation",
    # Legacy path mapping tests (for backward compatibility)
    "test_input_path_mapping": "level3_step_creation",
    "test_output_path_mapping": "level3_step_creation",
    "test_property_path_validity": "level3_step_creation",
    "test_processing_inputs_outputs": "level3_step_creation",
    "test_processing_code_handling": "level3_step_creation",
    
    # Level 4: Integration tests
    "test_dependency_resolution": "level4_integration",
    "test_step_creation": "level4_integration",
    "test_step_name": "level4_integration",
    "test_generic_dependency_handling": "level4_integration",
    "test_processing_step_dependencies": "level4_integration",
}

# Define importance weights for specific tests
TEST_IMPORTANCE = {
    # All tests default to 1.0, override specific tests if needed
    # Level 1: Interface tests
    "test_inheritance": 1.0,
    "test_required_methods": 1.2,
    
    # Level 2: Specification and contract tests
    "test_specification_usage": 1.2,
    "test_contract_alignment": 1.3,
    
    # Level 3: Step creation tests (high importance)
    "test_step_instantiation": 1.4,
    "test_step_configuration_validity": 1.5,
    "test_step_dependencies_attachment": 1.3,
    "test_step_name_generation": 1.2,
    "test_processing_step_creation": 1.4,
    "test_training_step_creation": 1.4,
    "test_transform_step_creation": 1.4,
    "test_create_model_step_creation": 1.4,
    # Legacy path mapping tests (lower importance)
    "test_property_path_validity": 1.1,
    
    # Level 4: Integration tests
    "test_dependency_resolution": 1.4,
    "test_step_creation": 1.5,
}

# Rating levels
RATING_LEVELS = {
    90: "Excellent",   # 90-100: Excellent
    80: "Good",        # 80-89: Good
    70: "Satisfactory",# 70-79: Satisfactory
    60: "Needs Work",  # 60-69: Needs Work
    0: "Poor"          # 0-59: Poor
}


class StepBuilderScorer:
    """
    A scorer for evaluating step builder quality based on test results.
    
    This class calculates scores for each test level and provides an overall
    score and rating for a step builder.
    """
    
    def __init__(self, results: Dict[str, Dict[str, Any]]):
        """
        Initialize with test results.
        
        Args:
            results: Dictionary mapping test names to their results
        """
        self.results = results
        self.level_results = self._group_by_level()
        
    def _group_by_level(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Group test results by level using smart pattern detection.
        
        Returns:
            Dictionary mapping levels to their test results
        """
        grouped = {level: {} for level in LEVEL_WEIGHTS.keys()}
        
        for test_name, result in self.results.items():
            level = self._detect_level_from_test_name(test_name)
            if level:
                grouped[level][test_name] = result
        
        return grouped
    
    def _detect_level_from_test_name(self, test_name: str) -> Optional[str]:
        """
        Detect test level from method name using smart pattern detection.
        
        This method uses multiple strategies to determine the test level:
        1. Explicit level prefix (level1_, level2_, etc.) - preferred for new variants
        2. Keyword-based detection for legacy and descriptive test names
        3. Fallback to explicit TEST_LEVEL_MAP for edge cases
        
        Args:
            test_name: Name of the test method
            
        Returns:
            Level name or None if level cannot be determined
        """
        # Strategy 1: Explicit level prefix (preferred for new variants)
        if test_name.startswith("level1_"):
            return "level1_interface"
        elif test_name.startswith("level2_"):
            return "level2_specification"
        elif test_name.startswith("level3_"):
            return "level3_step_creation"
        elif test_name.startswith("level4_"):
            return "level4_integration"
        
        # Strategy 2: Keyword-based detection for legacy and descriptive tests
        test_lower = test_name.lower()
        
        # Level 1 keywords: interface, methods, creation, inheritance, basic functionality
        level1_keywords = [
            "inheritance", "required_methods", "processor_creation", "interface",
            "error_handling", "generic_step_creation", "generic_configuration",
            "framework_specific_methods", "step_creation_pattern_compliance",
            "processing_input_output_methods", "environment_variables_method",
            "job_arguments_method", "processing_configuration_attributes"
        ]
        if any(keyword in test_lower for keyword in level1_keywords):
            return "level1_interface"
        
        # Level 2 keywords: specification, contract, environment, arguments, job types
        level2_keywords = [
            "specification", "contract", "environment", "arguments", "job_type",
            "environment_variable_handling", "processing_job_arguments",
            "property_files_configuration", "job_type_specification_loading",
            "environment_variable_patterns", "job_arguments_patterns",
            "specification_driven", "contract_path_mapping", "multi_job_type",
            "framework_specific_specifications"
        ]
        if any(keyword in test_lower for keyword in level2_keywords):
            return "level2_specification"
        
        # Level 3 keywords: step creation, configuration validation, and legacy path mapping
        level3_keywords = [
            # New step creation tests
            "step_instantiation", "step_configuration_validity", "step_dependencies_attachment",
            "step_name_generation", "processing_step_creation", "training_step_creation",
            "transform_step_creation", "create_model_step_creation", "configuration_validity",
            # Legacy path mapping tests (for backward compatibility)
            "path_mapping", "input", "output", "property_path", "processing_inputs_outputs",
            "processing_code_handling", "processing_input_creation", "processing_output_creation",
            "container_path_mapping", "special_input_handling", "s3_path_normalization",
            "file_upload_patterns", "local_path_override_patterns", "dependency_input_extraction"
        ]
        if any(keyword in test_lower for keyword in level3_keywords):
            return "level3_step_creation"
        
        # Level 4 keywords: integration, dependency, step_creation, end-to-end
        level4_keywords = [
            "dependency", "step_creation", "integration", "end_to_end", "step_name",
            "generic_dependency_handling", "processing_step_dependencies",
            "step_creation_pattern_execution", "framework_specific_step_creation",
            "processing_dependency_resolution", "step_name_generation",
            "cache_configuration", "step_dependencies_handling", "specification_attachment"
        ]
        if any(keyword in test_lower for keyword in level4_keywords):
            return "level4_integration"
        
        # Strategy 3: Fallback to explicit TEST_LEVEL_MAP for edge cases
        return TEST_LEVEL_MAP.get(test_name)
    
    def calculate_level_score(self, level: str) -> Tuple[float, int, int]:
        """
        Calculate score for a specific level.
        
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
        
        for test_name, result in level_tests.items():
            # Get test importance weight (default to 1.0 if not specified)
            importance = TEST_IMPORTANCE.get(test_name, 1.0)
            total_weight += importance
            
            # Add to weighted score if test passed
            if result.get("passed", False):
                weighted_score += importance
        
        # Calculate percentage score
        score = (weighted_score / total_weight) * 100.0 if total_weight > 0 else 0.0
        
        # Count passed tests
        passed = sum(1 for result in level_tests.values() if result.get("passed", False))
        total = len(level_tests)
        
        return score, passed, total
    
    def calculate_overall_score(self) -> float:
        """
        Calculate overall score across all levels.
        
        Returns:
            Overall score (0-100)
        """
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for level, weight in LEVEL_WEIGHTS.items():
            level_score, _, _ = self.calculate_level_score(level)
            total_weighted_score += level_score * weight
            total_weight += weight
        
        overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        return min(100.0, max(0.0, overall_score))
    
    def get_rating(self, score: float) -> str:
        """
        Get rating based on score.
        
        Args:
            score: Score to rate (0-100)
            
        Returns:
            Rating string
        """
        for threshold, rating in sorted(RATING_LEVELS.items(), reverse=True):
            if score >= threshold:
                return rating
        return "Invalid"
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive score report.
        
        Returns:
            Dictionary containing the full score report
        """
        level_scores = {}
        total_passed = 0
        total_tests = 0
        
        # Calculate scores for each level
        for level in LEVEL_WEIGHTS.keys():
            score, passed, total = self.calculate_level_score(level)
            level_scores[level] = {
                "score": score,
                "passed": passed,
                "total": total,
                "tests": {
                    test_name: result.get("passed", False) 
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
                    "error": result.get("error", "Unknown error")
                }
                for test_name, result in self.results.items()
                if not result.get("passed", False)
            ]
        }
        
        return report
    
    def save_report(self, builder_name: str, output_dir: str = "test_reports") -> str:
        """
        Save the score report to a JSON file.
        
        Args:
            builder_name: Name of the builder
            output_dir: Directory to save the report in
            
        Returns:
            Path to the saved report
        """
        report = self.generate_report()
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create filename
        filename = f"{output_dir}/{builder_name}_score_report.json"
        
        # Save report
        with open(filename, "w") as f:
            json.dump(report, f, indent=2)
            
        return filename
    
    def print_report(self, show_test_detection: bool = False) -> None:
        """
        Print a formatted score report to the console.
        
        Args:
            show_test_detection: Whether to show test level detection details
        """
        report = self.generate_report()
        
        print("\n" + "=" * 80)
        print(f"STEP BUILDER QUALITY SCORE REPORT")
        print("=" * 80)
        
        # Overall score and rating
        overall = report["overall"]
        print(f"\nOverall Score: {overall['score']:.1f}/100 - {overall['rating']}")
        print(f"Pass Rate: {overall['pass_rate']:.1f}% ({overall['passed']}/{overall['total']} tests)")
        
        # Level scores
        print("\nScores by Level:")
        for level, data in report["levels"].items():
            # Get a nicer level name for display
            display_level = level.replace("level", "Level ").replace("_", " ").title()
            print(f"  {display_level}: {data['score']:.1f}/100 ({data['passed']}/{data['total']} tests)")
        
        # Show test detection details if requested
        if show_test_detection:
            self._print_test_detection_details()
        
        # Failed tests
        if report["failed_tests"]:
            print("\nFailed Tests:")
            for test in report["failed_tests"]:
                print(f"  ❌ {test['name']}: {test['error']}")
        
        print("\n" + "=" * 80)
    
    def _print_test_detection_details(self) -> None:
        """Print details about how tests were detected and categorized."""
        print("\nTest Level Detection Details:")
        print("-" * 50)
        
        detection_stats = {
            "explicit_prefix": 0,
            "keyword_based": 0,
            "fallback_map": 0,
            "undetected": 0
        }
        
        for test_name in self.results.keys():
            detection_method = self._get_detection_method(test_name)
            detection_stats[detection_method] += 1
            
        print(f"  Explicit prefix (level1_, level2_, etc.): {detection_stats['explicit_prefix']} tests")
        print(f"  Keyword-based detection: {detection_stats['keyword_based']} tests")
        print(f"  Fallback to TEST_LEVEL_MAP: {detection_stats['fallback_map']} tests")
        print(f"  Undetected (no level assigned): {detection_stats['undetected']} tests")
        
        # Show undetected tests if any
        if detection_stats['undetected'] > 0:
            undetected_tests = [
                test_name for test_name in self.results.keys()
                if self._detect_level_from_test_name(test_name) is None
            ]
            print(f"\n  Undetected tests: {', '.join(undetected_tests)}")
    
    def _get_detection_method(self, test_name: str) -> str:
        """
        Determine how a test name was detected for level assignment.
        
        Args:
            test_name: Name of the test method
            
        Returns:
            Detection method: 'explicit_prefix', 'keyword_based', 'fallback_map', or 'undetected'
        """
        # Check explicit prefix first
        if test_name.startswith(("level1_", "level2_", "level3_", "level4_")):
            return "explicit_prefix"
        
        # Check if it would be detected by keywords
        test_lower = test_name.lower()
        
        all_keywords = [
            # Level 1 keywords
            "inheritance", "required_methods", "processor_creation", "interface",
            "error_handling", "generic_step_creation", "generic_configuration",
            "framework_specific_methods", "step_creation_pattern_compliance",
            "processing_input_output_methods", "environment_variables_method",
            "job_arguments_method", "processing_configuration_attributes",
            
            # Level 2 keywords
            "specification", "contract", "environment", "arguments", "job_type",
            "environment_variable_handling", "processing_job_arguments",
            "property_files_configuration", "job_type_specification_loading",
            "environment_variable_patterns", "job_arguments_patterns",
            "specification_driven", "contract_path_mapping", "multi_job_type",
            "framework_specific_specifications",
            
            # Level 3 keywords
            "path_mapping", "input", "output", "property_path", "processing_inputs_outputs",
            "processing_code_handling", "processing_input_creation", "processing_output_creation",
            "container_path_mapping", "special_input_handling", "s3_path_normalization",
            "file_upload_patterns", "local_path_override_patterns", "dependency_input_extraction",
            
            # Level 4 keywords
            "dependency", "step_creation", "integration", "end_to_end", "step_name",
            "generic_dependency_handling", "processing_step_dependencies",
            "step_creation_pattern_execution", "framework_specific_step_creation",
            "processing_dependency_resolution", "step_name_generation",
            "cache_configuration", "step_dependencies_handling", "specification_attachment"
        ]
        
        if any(keyword in test_lower for keyword in all_keywords):
            return "keyword_based"
        
        # Check if it's in the fallback map
        if test_name in TEST_LEVEL_MAP:
            return "fallback_map"
        
        return "undetected"
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """
        Get a summary of test detection methods used.
        
        Returns:
            Dictionary with detection statistics and details
        """
        detection_stats = {
            "explicit_prefix": [],
            "keyword_based": [],
            "fallback_map": [],
            "undetected": []
        }
        
        for test_name in self.results.keys():
            detection_method = self._get_detection_method(test_name)
            detection_stats[detection_method].append(test_name)
        
        return {
            "summary": {
                "explicit_prefix": len(detection_stats["explicit_prefix"]),
                "keyword_based": len(detection_stats["keyword_based"]),
                "fallback_map": len(detection_stats["fallback_map"]),
                "undetected": len(detection_stats["undetected"]),
                "total": len(self.results)
            },
            "details": detection_stats
        }
    
    def generate_chart(self, builder_name: str, output_dir: str = "test_reports") -> Optional[str]:
        """
        Generate a chart visualization of the score report.
        
        Args:
            builder_name: Name of the builder
            output_dir: Directory to save the chart in
            
        Returns:
            Path to the saved chart or None if matplotlib is not available
        """
        try:
            import matplotlib.pyplot as plt
            
            report = self.generate_report()
            
            # Create level names and scores
            levels = []
            scores = []
            colors = []
            
            for level in ["level1_interface", "level2_specification", "level3_step_creation", "level4_integration"]:
                if level in report["levels"]:
                    # Get a nicer level name for display
                    display_level = level.replace("level", "L").replace("_", " ").title()
                    levels.append(display_level)
                    score = report["levels"][level]["score"]
                    scores.append(score)
                    
                    # Choose color based on score
                    if score >= 90:
                        colors.append("green")
                    elif score >= 80:
                        colors.append("lightgreen")
                    elif score >= 70:
                        colors.append("orange")
                    elif score >= 60:
                        colors.append("salmon")
                    else:
                        colors.append("red")
            
            # Create the figure
            plt.figure(figsize=(10, 6))
            
            # Create bar chart
            bars = plt.bar(levels, scores, color=colors)
            
            # Add overall score line
            plt.axhline(y=report["overall"]["score"], color='blue', linestyle='-', alpha=0.7)
            plt.text(len(levels)-0.5, report["overall"]["score"]+2, 
                    f"Overall: {report['overall']['score']:.1f} ({report['overall']['rating']})", 
                    color='blue')
            
            # Add labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f"{height:.1f}%", ha='center', va='bottom')
            
            # Set chart properties
            plt.title(f"Step Builder Quality Score: {builder_name}")
            plt.ylabel("Score (%)")
            plt.ylim(0, 105)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Save figure
            filename = f"{output_dir}/{builder_name}_score_chart.png"
            plt.savefig(filename)
            plt.close()
            
            return filename
        except ImportError:
            print("matplotlib not available, skipping chart generation")
            return None
        except Exception as e:
            print(f"Could not generate chart: {str(e)}")
            return None


def score_builder_results(results: Dict[str, Dict[str, Any]], 
                          builder_name: str = "Unknown",
                          save_report: bool = True,
                          output_dir: str = "test_reports",
                          generate_chart: bool = True) -> Dict[str, Any]:
    """
    Score test results for a step builder.
    
    Args:
        results: Dictionary mapping test names to their results
        builder_name: Name of the builder
        save_report: Whether to save the report to a file
        output_dir: Directory to save the report in
        generate_chart: Whether to generate a chart visualization
        
    Returns:
        Score report dictionary
    """
    scorer = StepBuilderScorer(results)
    report = scorer.generate_report()
    
    # Print report
    scorer.print_report()
    
    # Save report
    if save_report:
        scorer.save_report(builder_name, output_dir)
    
    # Generate chart
    if generate_chart:
        scorer.generate_chart(builder_name, output_dir)
    
    return report
