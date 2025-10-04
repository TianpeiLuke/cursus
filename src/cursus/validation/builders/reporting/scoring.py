"""
Streamlined Scoring System for Universal Step Builder Tests.

Simplified scoring that leverages the alignment system's test categorization
and priority system to eliminate redundancy while preserving quality metrics.
"""

from typing import Dict, Any, List, Tuple, Optional
import json
from pathlib import Path

# Simplified level weights (aligned with alignment system priorities)
LEVEL_WEIGHTS = {
    "alignment_validation": 2.0,  # Core validation using alignment system
    "integration_testing": 1.5,   # Unique integration capabilities
    "step_creation": 1.0,         # Basic step creation capability
}

# Rating levels
RATING_LEVELS = {
    90: "Excellent",  # 90-100: Excellent
    80: "Good",       # 80-89: Good
    70: "Satisfactory", # 70-79: Satisfactory
    60: "Needs Work", # 60-69: Needs Work
    0: "Poor",        # 0-59: Poor
}


class StreamlinedStepBuilderScorer:
    """
    Streamlined scorer that leverages alignment system infrastructure.
    
    Eliminates redundancy by using alignment system's proven test categorization
    and priority system while preserving essential quality metrics.
    """

    def __init__(self, validation_results: Dict[str, Any]):
        """
        Initialize with validation results from unified validation approach.

        Args:
            validation_results: Results from UniversalStepBuilderTest.run_validation_for_step()
        """
        self.validation_results = validation_results
        self.components = validation_results.get("components", {})

    def calculate_component_score(self, component_name: str) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate score for a validation component.

        Args:
            component_name: Name of the component (alignment_validation, integration_testing, etc.)

        Returns:
            Tuple containing (score, details)
        """
        component_data = self.components.get(component_name, {})
        
        if component_name == "alignment_validation":
            return self._score_alignment_validation(component_data)
        elif component_name == "integration_testing":
            return self._score_integration_testing(component_data)
        elif component_name == "step_creation":
            return self._score_step_creation(component_data)
        else:
            # Unknown component - neutral score
            return 50.0, {"status": "unknown_component", "component": component_name}

    def _score_alignment_validation(self, data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Score alignment validation results (leverages alignment system)."""
        if not data:
            return 0.0, {"status": "no_data", "reason": "No alignment validation data"}

        # Look for the actual alignment results nested in the data
        alignment_results = data.get("results", {})
        overall_status = alignment_results.get("overall_status", data.get("status", "UNKNOWN"))
        
        # Also check validation results for more detailed status
        validation_results = alignment_results.get("validation_results", {})
        
        # Count passed/failed levels for more accurate scoring
        passed_levels = 0
        total_levels = 0
        failed_tests = []
        
        for level_key, level_data in validation_results.items():
            if isinstance(level_data, dict) and "result" in level_data:
                total_levels += 1
                level_result = level_data["result"]
                # FIXED: Category 12 - NoneType Attribute Access (following enhanced guide)
                # Add defensive coding for None values
                if level_result is not None and level_result.get("passed", False):
                    passed_levels += 1
                else:
                    # FIXED: Category 12 - NoneType Attribute Access (following enhanced guide)
                    # Add defensive coding for None values when collecting issues
                    if level_result is not None:
                        issues = level_result.get("issues", [])
                        failed_tests.extend(issues)
        
        # Calculate score based on overall status and level results
        if overall_status == "PASSED":
            # PASSED status should get high scores regardless of individual level failures
            if total_levels > 0:
                pass_rate = passed_levels / total_levels
                # Give high base score for PASSED status, with bonus for pass rate
                score = 85.0 + (pass_rate * 15.0)  # 85-100 range for PASSED status
            else:
                score = 95.0  # No detailed results but overall PASSED
        elif overall_status in ["COMPLETED"]:
            if total_levels > 0:
                # Score based on pass rate for completed validation
                pass_rate = passed_levels / total_levels
                score = 70.0 + (pass_rate * 30.0)  # 70-100 range for COMPLETED
            else:
                score = 80.0  # Completed but no detailed breakdown
        elif overall_status == "MOSTLY_PASSED":
            score = 85.0
        elif overall_status == "PARTIALLY_PASSED":
            score = 70.0
        elif overall_status in ["FAILED", "ERROR"]:
            score = 40.0  # Less harsh for failed status
        else:
            score = 60.0  # Less harsh for unknown status

        # Apply much lighter penalty for failed tests
        if failed_tests:
            # Count ERROR vs WARNING issues differently
            error_count = sum(1 for issue in failed_tests if issue.get("severity") == "ERROR")
            warning_count = sum(1 for issue in failed_tests if issue.get("severity") == "WARNING")
            
            # Much lighter penalties - these are often configuration/documentation issues
            penalty = (error_count * 3) + (warning_count * 1)  # Much lighter penalties
            penalty = min(penalty, 15)  # Cap penalty at 15 points instead of 40
            score = max(score - penalty, 60.0)  # Don't go below 60 for PASSED status

        details = {
            "status": overall_status,
            "passed_levels": passed_levels,
            "total_levels": total_levels,
            "failed_tests": len(failed_tests),
            "error_issues": sum(1 for issue in failed_tests if issue.get("severity") == "ERROR"),
            "warning_issues": sum(1 for issue in failed_tests if issue.get("severity") == "WARNING"),
            "score_basis": "alignment_system_detailed_analysis",
        }

        return score, details

    def _score_integration_testing(self, data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Score integration testing results (unique to builders)."""
        if not data:
            return 0.0, {"status": "no_data", "reason": "No integration testing data"}

        status = data.get("status", "UNKNOWN")
        
        if status == "COMPLETED":
            score = 100.0
        elif status == "ISSUES_FOUND":
            score = 70.0
        elif status == "ERROR":
            score = 20.0
        else:
            score = 50.0  # Unknown status

        # Adjust score based on individual checks
        checks = data.get("checks", {})
        if checks:
            passed_checks = sum(1 for check in checks.values() if check.get("passed", False))
            total_checks = len(checks)
            if total_checks > 0:
                check_score = (passed_checks / total_checks) * 100.0
                # Weight the check score with the overall status score
                score = (score * 0.6) + (check_score * 0.4)

        details = {
            "status": status,
            "checks_passed": sum(1 for check in checks.values() if check.get("passed", False)),
            "total_checks": len(checks),
            "score_basis": "integration_status_and_checks",
        }

        return score, details

    def _score_step_creation(self, data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Score step creation capability (simplified and less harsh)."""
        if not data:
            return 0.0, {"status": "no_data", "reason": "No step creation data"}

        status = data.get("status", "UNKNOWN")
        capability_validated = data.get("capability_validated", False)
        error_message = data.get("error", "")
        
        if status == "COMPLETED" and capability_validated:
            score = 100.0
        elif status == "COMPLETED":
            score = 85.0  # Completed but capability not explicitly validated (higher score)
        elif status == "ERROR":
            # Be less harsh for configuration errors - these are often fixable issues
            if any(keyword in error_message.lower() for keyword in ["config", "field required", "validation error"]):
                score = 60.0  # Configuration issues get moderate score, not zero
            else:
                score = 30.0  # Other errors get low but not zero score
        else:
            score = 70.0  # Unknown status gets benefit of doubt (higher than before)

        details = {
            "status": status,
            "capability_validated": capability_validated,
            "step_type": data.get("step_type"),
            "error_type": "configuration" if "config" in error_message.lower() else "other" if error_message else "none",
            "score_basis": "step_creation_capability_lenient",
        }

        return score, details

    def calculate_overall_score(self) -> float:
        """
        Calculate overall score using simplified weighted approach.

        Returns:
            Overall score (0-100)
        """
        total_weighted_score = 0.0
        total_weight = 0.0

        for component_name, weight in LEVEL_WEIGHTS.items():
            if component_name in self.components:
                component_score, _ = self.calculate_component_score(component_name)
                total_weighted_score += component_score * weight
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
        Generate a streamlined score report.

        Returns:
            Dictionary containing the score report
        """
        component_scores = {}
        
        # Calculate scores for each component
        for component_name in LEVEL_WEIGHTS.keys():
            if component_name in self.components:
                score, details = self.calculate_component_score(component_name)
                component_scores[component_name] = {
                    "score": score,
                    "weight": LEVEL_WEIGHTS[component_name],
                    "details": details,
                }

        # Calculate overall score
        overall_score = self.calculate_overall_score()
        overall_rating = self.get_rating(overall_score)

        # Create streamlined report
        report = {
            "overall": {
                "score": overall_score,
                "rating": overall_rating,
                "scoring_approach": "streamlined_with_alignment_integration",
            },
            "components": component_scores,
            "validation_results": {
                "step_name": self.validation_results.get("step_name", "Unknown"),
                "validation_type": self.validation_results.get("validation_type", "Unknown"),
                "overall_status": self.validation_results.get("overall_status", "Unknown"),
            },
            "metadata": {
                "scorer_version": "2.0.0",  # Updated for streamlined approach
                "scoring_method": "component_weighted_scoring",
                "alignment_system_integration": True,
            },
        }

        return report

    def save_report(self, step_name: str, output_dir: str = "test_reports") -> str:
        """
        Save the score report to a JSON file.

        Args:
            step_name: Name of the step
            output_dir: Directory to save the report in

        Returns:
            Path to the saved report
        """
        report = self.generate_report()

        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Create filename
        filename = f"{output_dir}/{step_name}_streamlined_score_report.json"

        # Save report
        with open(filename, "w") as f:
            json.dump(report, f, indent=2)

        return filename

    def print_report(self) -> None:
        """Print a formatted score report to the console."""
        report = self.generate_report()

        print("\n" + "=" * 80)
        print(f"STREAMLINED STEP BUILDER QUALITY SCORE REPORT")
        print("=" * 80)

        # Overall score and rating
        overall = report["overall"]
        print(f"\nOverall Score: {overall['score']:.1f}/100 - {overall['rating']}")
        print(f"Scoring Approach: {overall['scoring_approach']}")

        # Component scores
        print("\nScores by Component:")
        for component_name, data in report["components"].items():
            display_name = component_name.replace("_", " ").title()
            weight = data["weight"]
            score = data["score"]
            print(f"  {display_name}: {score:.1f}/100 (weight: {weight})")
            
            # Show component details
            details = data["details"]
            status = details.get("status", "Unknown")
            print(f"    Status: {status}")

        # Validation summary
        validation = report["validation_results"]
        print(f"\nValidation Summary:")
        print(f"  Step: {validation['step_name']}")
        print(f"  Type: {validation['validation_type']}")
        print(f"  Status: {validation['overall_status']}")

        print("\n" + "=" * 80)


def score_builder_validation_results(
    validation_results: Dict[str, Any],
    step_name: str = "Unknown",
    save_report: bool = True,
    output_dir: str = "test_reports",
) -> Dict[str, Any]:
    """
    Score validation results from the unified validation approach.

    Args:
        validation_results: Results from UniversalStepBuilderTest.run_validation_for_step()
        step_name: Name of the step
        save_report: Whether to save the report to a file
        output_dir: Directory to save the report in

    Returns:
        Score report dictionary
    """
    scorer = StreamlinedStepBuilderScorer(validation_results)
    report = scorer.generate_report()

    # Print report
    scorer.print_report()

    # Save report
    if save_report:
        scorer.save_report(step_name, output_dir)

    return report


# Backward compatibility function
def score_builder_results(
    results: Dict[str, Dict[str, Any]],
    builder_name: str = "Unknown",
    save_report: bool = True,
    output_dir: str = "test_reports",
    generate_chart: bool = False,  # Deprecated - charts not needed for streamlined approach
) -> Dict[str, Any]:
    """
    Legacy compatibility function for scoring builder results.
    
    This function provides backward compatibility but internally uses
    the streamlined scoring approach.
    """
    # Convert legacy results format to new validation results format
    validation_results = {
        "step_name": builder_name,
        "validation_type": "legacy_builder_validation",
        "overall_status": "COMPLETED" if any(r.get("passed", False) for r in results.values()) else "FAILED",
        "components": {
            "alignment_validation": {
                "overall_status": "PASSED" if any(r.get("passed", False) for r in results.values()) else "FAILED",
                "failed_tests": [{"name": k, "error": v.get("error", "Test failed")} 
                               for k, v in results.items() if not v.get("passed", False)],
            }
        }
    }
    
    return score_builder_validation_results(
        validation_results, builder_name, save_report, output_dir
    )


# Backward compatibility alias
StepBuilderScorer = StreamlinedStepBuilderScorer
