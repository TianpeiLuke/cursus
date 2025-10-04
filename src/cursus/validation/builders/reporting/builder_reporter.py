"""
Streamlined Step Builder Test Reporting System.

Simplified reporting that leverages the alignment system infrastructure
to eliminate redundancy while preserving unique builder testing capabilities.
"""

import json
from typing import Dict, List, Any, Optional, Type
from datetime import datetime
from pathlib import Path

from ....core.base.builder_base import StepBuilderBase


class StreamlinedBuilderTestReport:
    """
    Simplified builder test report that leverages alignment system infrastructure.
    
    Eliminates redundancy by using alignment system's proven reporting patterns
    while preserving unique builder testing capabilities.
    """

    def __init__(self, builder_name: str, builder_class: str, sagemaker_step_type: str):
        self.builder_name = builder_name
        self.builder_class = builder_class
        self.sagemaker_step_type = sagemaker_step_type
        self.validation_timestamp = datetime.now()
        
        # Simplified structure - let alignment system handle complexity
        self.test_results: Dict[str, Any] = {}
        self.alignment_results: Optional[Dict[str, Any]] = None
        self.integration_results: Optional[Dict[str, Any]] = None
        self.scoring_data: Optional[Dict[str, Any]] = None
        self.metadata: Dict[str, Any] = {}

    def add_alignment_results(self, results: Dict[str, Any]):
        """Add results from alignment system validation."""
        self.alignment_results = results

    def add_integration_results(self, results: Dict[str, Any]):
        """Add results from integration testing (unique to builders)."""
        self.integration_results = results

    def add_scoring_data(self, scoring: Dict[str, Any]):
        """Add scoring data from scoring system."""
        self.scoring_data = scoring

    def get_overall_status(self) -> str:
        """Get overall validation status."""
        # Use alignment system's status if available
        if self.alignment_results:
            alignment_status = self.alignment_results.get("overall_status", "UNKNOWN")
            if alignment_status in ["PASSED", "COMPLETED"]:
                # Check integration results for final status
                if self.integration_results:
                    integration_status = self.integration_results.get("status", "UNKNOWN")
                    if integration_status == "COMPLETED":
                        return "PASSING"
                    elif integration_status == "ISSUES_FOUND":
                        return "MOSTLY_PASSING"
                    else:
                        return "PARTIALLY_PASSING"
                return "PASSING"
            else:
                return "FAILING"
        
        # Fallback to basic status determination
        if self.integration_results:
            integration_status = self.integration_results.get("status", "UNKNOWN")
            return "PASSING" if integration_status == "COMPLETED" else "FAILING"
        
        return "UNKNOWN"

    def get_quality_score(self) -> float:
        """Get quality score from scoring system."""
        if self.scoring_data:
            return self.scoring_data.get("overall", {}).get("score", 0.0)
        return 0.0

    def get_quality_rating(self) -> str:
        """Get quality rating from scoring system."""
        if self.scoring_data:
            return self.scoring_data.get("overall", {}).get("rating", "Unknown")
        return "Unknown"

    def is_passing(self) -> bool:
        """Check if the overall validation is passing."""
        status = self.get_overall_status()
        return status in ["PASSING", "MOSTLY_PASSING"]

    def get_critical_issues(self) -> List[str]:
        """Get critical issues from alignment system."""
        issues = []
        
        # Extract issues from alignment results
        if self.alignment_results and "failed_tests" in self.alignment_results:
            for test in self.alignment_results["failed_tests"]:
                if isinstance(test, dict) and "error" in test:
                    issues.append(test["error"])
        
        # Extract issues from integration results
        if self.integration_results and "error" in self.integration_results:
            issues.append(self.integration_results["error"])
        
        return issues

    def export_to_json(self) -> str:
        """Export report to JSON format compatible with alignment system reports."""
        report_data = {
            "builder_name": self.builder_name,
            "builder_class": self.builder_class,
            "sagemaker_step_type": self.sagemaker_step_type,
            "validation_timestamp": self.validation_timestamp.isoformat(),
            "overall_status": self.get_overall_status(),
            "quality_score": self.get_quality_score(),
            "quality_rating": self.get_quality_rating(),
            "is_passing": self.is_passing(),
            
            # Include alignment system results (leverages proven infrastructure)
            "alignment_validation": self.alignment_results or {},
            
            # Include unique integration testing results
            "integration_testing": self.integration_results or {},
            
            # Include scoring data
            "scoring": self.scoring_data or {},
            
            # Metadata
            "metadata": {
                "builder_name": self.builder_name,
                "builder_class": self.builder_class,
                "sagemaker_step_type": self.sagemaker_step_type,
                "validation_timestamp": self.validation_timestamp.isoformat(),
                "validator_version": "2.0.0",  # Updated version for streamlined approach
                "test_framework": "UniversalStepBuilderTest",
                "reporting_approach": "streamlined_with_alignment_integration",
                **self.metadata
            }
        }

        return json.dumps(report_data, indent=2, default=str)

    def save_to_file(self, output_path: Path):
        """Save report to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(self.export_to_json())

    def print_summary(self):
        """Print a formatted summary to console."""
        print("\n" + "=" * 80)
        print(f"STEP BUILDER TEST REPORT: {self.builder_name}")
        print("=" * 80)

        print(f"\nBuilder: {self.builder_class}")
        print(f"SageMaker Step Type: {self.sagemaker_step_type}")
        
        status = self.get_overall_status()
        status_icon = "✅" if self.is_passing() else "❌"
        print(f"Overall Status: {status_icon} {status}")
        
        # Print quality score
        quality_score = self.get_quality_score()
        quality_rating = self.get_quality_rating()
        if quality_score > 0:
            print(f"📊 Quality Score: {quality_score:.1f}/100 - {quality_rating}")

        # Print alignment validation summary
        if self.alignment_results:
            print(f"\n🔍 Alignment Validation:")
            alignment_status = self.alignment_results.get("overall_status", "UNKNOWN")
            print(f"  Status: {alignment_status}")
            
            # Show failed tests if any
            failed_tests = self.alignment_results.get("failed_tests", [])
            if failed_tests:
                print(f"  Failed Tests: {len(failed_tests)}")
                for test in failed_tests[:3]:  # Show first 3
                    if isinstance(test, dict):
                        test_name = test.get("name", "Unknown")
                        print(f"    • {test_name}")
                if len(failed_tests) > 3:
                    print(f"    ... and {len(failed_tests) - 3} more")

        # Print integration testing summary
        if self.integration_results:
            print(f"\n🔧 Integration Testing:")
            integration_status = self.integration_results.get("status", "UNKNOWN")
            print(f"  Status: {integration_status}")
            
            # Show integration checks if available
            checks = self.integration_results.get("checks", {})
            if checks:
                for check_name, check_result in checks.items():
                    check_passed = check_result.get("passed", False)
                    check_icon = "✅" if check_passed else "❌"
                    print(f"  {check_icon} {check_name.replace('_', ' ').title()}")

        # Print scoring breakdown if available
        if self.scoring_data and "levels" in self.scoring_data:
            print(f"\n📈 Quality Score Breakdown:")
            for level_key, level_data in self.scoring_data["levels"].items():
                display_name = level_key.replace("level", "L").replace("_", " ").title()
                score = level_data.get("score", 0.0)
                passed = level_data.get("passed", 0)
                total = level_data.get("total", 0)
                print(f"  {display_name}: {score:.1f}/100 ({passed}/{total} tests)")

        # Print critical issues
        critical_issues = self.get_critical_issues()
        if critical_issues:
            print(f"\n🚨 CRITICAL ISSUES ({len(critical_issues)}):")
            for issue in critical_issues[:5]:  # Show first 5
                print(f"  • {issue}")
            if len(critical_issues) > 5:
                print(f"  ... and {len(critical_issues) - 5} more")

        print("\n" + "=" * 80)


class StreamlinedBuilderTestReporter:
    """
    Streamlined reporter that leverages alignment system infrastructure.
    
    Eliminates redundancy by using proven alignment system patterns
    while preserving unique builder testing capabilities.
    """

    def __init__(self, output_dir: Path = None):
        """Initialize the streamlined reporter."""
        self.output_dir = (
            output_dir or Path.cwd() / "test" / "validation" / "builders" / "reports"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_and_report_builder(
        self, builder_class: Type[StepBuilderBase], step_name: str = None
    ) -> StreamlinedBuilderTestReport:
        """
        Test a step builder using the unified validation approach.
        
        Leverages the refactored UniversalStepBuilderTest that integrates
        with the alignment system to eliminate redundancy.
        """
        # Infer step name if not provided
        if not step_name:
            step_name = self._infer_step_name(builder_class)

        # Get step information
        from ....registry.step_names import STEP_NAMES
        step_info = STEP_NAMES.get(step_name, {})
        sagemaker_step_type = step_info.get("sagemaker_step_type", "Unknown")

        print(f"Testing {step_name} ({builder_class.__name__})...")

        # Create streamlined report
        report = StreamlinedBuilderTestReport(
            step_name, builder_class.__name__, sagemaker_step_type
        )

        try:
            # Use the refactored UniversalStepBuilderTest that leverages alignment system
            from ..universal_test import UniversalStepBuilderTest

            # Create tester using the new unified approach
            tester = UniversalStepBuilderTest(workspace_dirs=["."], verbose=False)
            
            # Run validation for this specific step (leverages alignment system)
            validation_results = tester.run_validation_for_step(step_name)
            
            # Extract components from unified validation results
            components = validation_results.get("components", {})
            
            # Add alignment validation results (eliminates redundancy)
            if "alignment_validation" in components:
                report.add_alignment_results(components["alignment_validation"])
            
            # Add integration testing results (unique value preserved)
            if "integration_testing" in components:
                report.add_integration_results(components["integration_testing"])
            
            # Add scoring data if available (leverages existing scoring system)
            if hasattr(tester, 'enable_scoring') and tester.enable_scoring:
                # Scoring data is now included in validation_results
                if "scoring" in validation_results:
                    report.add_scoring_data(validation_results["scoring"])

        except Exception as e:
            print(f"❌ Failed to test {step_name}: {e}")
            # Add error information to report
            report.metadata["error"] = str(e)
            report.metadata["error_type"] = type(e).__name__

        return report

    def test_and_save_builder_report(
        self, builder_class: Type[StepBuilderBase], step_name: str = None
    ) -> StreamlinedBuilderTestReport:
        """Test a builder and save the streamlined report to file."""
        report = self.test_and_report_builder(builder_class, step_name)

        # Save to report file
        filename = f"{report.builder_name.lower()}_builder_test_report.json"
        output_path = self.output_dir / filename
        report.save_to_file(output_path)

        print(f"✅ Streamlined report saved: {output_path}")
        return report

    def test_step_type_builders(
        self, sagemaker_step_type: str
    ) -> Dict[str, StreamlinedBuilderTestReport]:
        """Test all builders of a specific SageMaker step type using streamlined approach."""
        print(f"Testing all {sagemaker_step_type} step builders (streamlined)...")
        print("=" * 60)

        try:
            # Get all steps of this type
            from ....registry.step_names import get_steps_by_sagemaker_type
            step_names = get_steps_by_sagemaker_type(sagemaker_step_type)

            if not step_names:
                print(f"❌ No {sagemaker_step_type} step builders found")
                return {}

            reports = {}

            for step_name in step_names:
                try:
                    # Load builder class using step catalog
                    builder_class = self._load_builder_class(step_name)
                    if not builder_class:
                        print(f"  ❌ Could not load builder class for {step_name}")
                        continue

                    # Test and save report
                    report = self.test_and_save_builder_report(builder_class, step_name)
                    reports[step_name] = report

                except Exception as e:
                    print(f"  ❌ Failed to test {step_name}: {e}")

            # Generate streamlined step type summary
            self._generate_streamlined_step_type_summary(sagemaker_step_type, reports)

            return reports

        except Exception as e:
            print(f"❌ Failed to process {sagemaker_step_type} step builders: {e}")
            return {}  # Graceful fallback

    def _infer_step_name(self, builder_class: Type[StepBuilderBase]) -> str:
        """Infer step name from builder class name."""
        class_name = builder_class.__name__

        # Remove "StepBuilder" suffix
        if class_name.endswith("StepBuilder"):
            step_name = class_name[:-11]  # Remove "StepBuilder"
        else:
            step_name = class_name

        # Look for matching step name in registry
        from ....registry.step_names import STEP_NAMES
        for name, info in STEP_NAMES.items():
            if (
                info.get("builder_step_name", "").replace("StepBuilder", "")
                == step_name
            ):
                return name

        return step_name

    def _load_builder_class(self, step_name: str) -> Optional[Type[StepBuilderBase]]:
        """Load a builder class by step name using StepCatalog discovery."""
        try:
            # Use StepCatalog's built-in builder discovery mechanism
            if not hasattr(self, '_step_catalog'):
                from ....step_catalog.step_catalog import StepCatalog
                self._step_catalog = StepCatalog()
            
            builder_class = self._step_catalog.load_builder_class(step_name)
            if builder_class:
                return builder_class
            else:
                print(f"No builder class found for step: {step_name}")
                return None
                
        except Exception as e:
            print(f"Failed to load {step_name} builder using StepCatalog: {e}")
            return None

    def _generate_streamlined_step_type_summary(
        self, step_type: str, reports: Dict[str, StreamlinedBuilderTestReport]
    ):
        """Generate and save a streamlined summary report for a step type."""
        summary_data = {
            "step_type": step_type,
            "summary": {
                "total_builders": len(reports),
                "passing_builders": sum(1 for r in reports.values() if r.is_passing()),
                "failing_builders": sum(1 for r in reports.values() if not r.is_passing()),
                "average_quality_score": (
                    sum(r.get_quality_score() for r in reports.values()) / len(reports)
                    if reports else 0.0
                ),
            },
            "builder_reports": {
                name: {
                    "builder_class": report.builder_class,
                    "overall_status": report.get_overall_status(),
                    "quality_score": report.get_quality_score(),
                    "quality_rating": report.get_quality_rating(),
                    "is_passing": report.is_passing(),
                }
                for name, report in reports.items()
            },
            "metadata": {
                "step_type": step_type,
                "generation_timestamp": datetime.now().isoformat(),
                "validator_version": "2.0.0",  # Updated for streamlined approach
                "test_framework": "UniversalStepBuilderTest",
                "reporting_approach": "streamlined_with_alignment_integration",
            },
        }

        # Save streamlined step type summary
        summary_file = self.output_dir / f"{step_type.lower()}_builder_test_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary_data, f, indent=2, default=str)

        print(f"✅ {step_type} streamlined summary saved: {summary_file}")


# Backward compatibility aliases
BuilderTestReport = StreamlinedBuilderTestReport
BuilderTestReporter = StreamlinedBuilderTestReporter
