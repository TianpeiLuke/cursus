"""
Step Builder Test Reporting System.

Provides comprehensive reporting capabilities for step builder test results,
including summary generation, issue analysis, and export functionality.
"""

import json
from typing import Dict, List, Any, Optional, Type
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field

from ...core.base.builder_base import StepBuilderBase
from ...registry.step_names import get_steps_by_sagemaker_type, STEP_NAMES


class BuilderTestIssue(BaseModel):
    """
    Issue found during step builder testing.
    
    Similar to AlignmentIssue but specific to builder testing.
    """
    severity: str  # INFO, WARNING, ERROR, CRITICAL
    category: str  # interface, specification, path_mapping, integration, step_type_specific
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    recommendation: Optional[str] = None
    test_name: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()


class BuilderTestResult(BaseModel):
    """
    Result of a single builder test.
    
    Contains detailed information about what was tested,
    whether it passed, and specific issues found.
    """
    test_name: str
    passed: bool
    issues: List[BuilderTestIssue] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    test_level: str  # interface, specification, path_mapping, integration, step_type_specific
        
    def add_issue(self, issue: BuilderTestIssue):
        """Add a builder test issue to this result."""
        self.issues.append(issue)
        # Update passed status based on severity
        if issue.severity in ["ERROR", "CRITICAL"]:
            self.passed = False
            
    def get_highest_severity(self) -> Optional[str]:
        """Get the highest severity level among all issues."""
        if not self.issues:
            return None
        
        severity_order = {"INFO": 0, "WARNING": 1, "ERROR": 2, "CRITICAL": 3}
        highest = max(self.issues, key=lambda x: severity_order.get(x.severity, 0))
        return highest.severity
    
    def has_critical_issues(self) -> bool:
        """Check if this result has critical issues."""
        return any(issue.severity == "CRITICAL" for issue in self.issues)
    
    def has_errors(self) -> bool:
        """Check if this result has error-level issues."""
        return any(issue.severity == "ERROR" for issue in self.issues)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'test_name': self.test_name,
            'passed': self.passed,
            'timestamp': self.timestamp.isoformat(),
            'test_level': self.test_level,
            'issues': [issue.to_dict() for issue in self.issues],
            'details': self.details,
            'highest_severity': self.get_highest_severity()
        }


class BuilderTestSummary(BaseModel):
    """
    Executive summary of builder test results.
    
    Provides high-level statistics and key findings from the testing.
    """
    builder_name: str
    builder_class: str
    sagemaker_step_type: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    pass_rate: float
    total_issues: int
    critical_issues: int
    error_issues: int
    warning_issues: int
    info_issues: int
    highest_severity: Optional[str]
    overall_status: str  # PASSING, MOSTLY_PASSING, PARTIALLY_PASSING, FAILING
    validation_timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)  # For scoring and other metadata
    
    @classmethod
    def from_results(cls, builder_name: str, builder_class: str, sagemaker_step_type: str,
                    results: Dict[str, BuilderTestResult]) -> 'BuilderTestSummary':
        """Create BuilderTestSummary from test results."""
        validation_timestamp = datetime.now()
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r.passed)
        failed_tests = total_tests - passed_tests
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Collect all issues
        all_issues = []
        for result in results.values():
            all_issues.extend(result.issues)
        
        total_issues = len(all_issues)
        
        # Count issues by severity
        critical_issues = sum(1 for issue in all_issues if issue.severity == "CRITICAL")
        error_issues = sum(1 for issue in all_issues if issue.severity == "ERROR")
        warning_issues = sum(1 for issue in all_issues if issue.severity == "WARNING")
        info_issues = sum(1 for issue in all_issues if issue.severity == "INFO")
        
        # Determine highest severity
        highest_severity = None
        if critical_issues > 0:
            highest_severity = "CRITICAL"
        elif error_issues > 0:
            highest_severity = "ERROR"
        elif warning_issues > 0:
            highest_severity = "WARNING"
        elif info_issues > 0:
            highest_severity = "INFO"
        
        # Determine overall status
        if critical_issues > 0 or error_issues > 0:
            if pass_rate < 50:
                overall_status = "FAILING"
            else:
                overall_status = "PARTIALLY_PASSING"
        elif pass_rate == 100:
            overall_status = "PASSING"
        elif pass_rate >= 80:
            overall_status = "MOSTLY_PASSING"
        else:
            overall_status = "PARTIALLY_PASSING"
        
        return cls(
            builder_name=builder_name,
            builder_class=builder_class,
            sagemaker_step_type=sagemaker_step_type,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            pass_rate=pass_rate,
            total_issues=total_issues,
            critical_issues=critical_issues,
            error_issues=error_issues,
            warning_issues=warning_issues,
            info_issues=info_issues,
            highest_severity=highest_severity,
            overall_status=overall_status,
            validation_timestamp=validation_timestamp
        )
    
    def is_passing(self) -> bool:
        """Check if the overall testing is passing (no critical or error issues)."""
        return self.critical_issues == 0 and self.error_issues == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'builder_name': self.builder_name,
            'builder_class': self.builder_class,
            'sagemaker_step_type': self.sagemaker_step_type,
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'pass_rate': self.pass_rate,
            'total_issues': self.total_issues,
            'critical_issues': self.critical_issues,
            'error_issues': self.error_issues,
            'warning_issues': self.warning_issues,
            'info_issues': self.info_issues,
            'highest_severity': self.highest_severity,
            'overall_status': self.overall_status,
            'validation_timestamp': self.validation_timestamp.isoformat(),
            'is_passing': self.is_passing()
        }


class BuilderTestRecommendation(BaseModel):
    """
    Actionable recommendation for fixing builder test issues.
    
    Attributes:
        category: Category of the recommendation
        priority: Priority level (HIGH, MEDIUM, LOW)
        title: Short title of the recommendation
        description: Detailed description
        affected_components: List of components this affects
        steps: Step-by-step instructions for implementing the fix
    """
    category: str
    priority: str
    title: str
    description: str
    affected_components: List[str]
    steps: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()


class BuilderTestReport:
    """
    Comprehensive report of step builder test results.
    
    Contains results from all test levels with detailed
    analysis and actionable recommendations.
    """
    
    def __init__(self, builder_name: str, builder_class: str, sagemaker_step_type: str):
        self.builder_name = builder_name
        self.builder_class = builder_class
        self.sagemaker_step_type = sagemaker_step_type
        
        self.level1_interface: Dict[str, BuilderTestResult] = {}
        self.level2_specification: Dict[str, BuilderTestResult] = {}
        self.level3_step_creation: Dict[str, BuilderTestResult] = {}
        self.level4_integration: Dict[str, BuilderTestResult] = {}
        self.step_type_specific: Dict[str, BuilderTestResult] = {}
        
        self.summary: Optional[BuilderTestSummary] = None
        self.recommendations: List[BuilderTestRecommendation] = []
        self.metadata: Dict[str, Any] = {}
        
    def add_level1_result(self, test_name: str, result: BuilderTestResult):
        """Add a Level 1 (Interface) test result."""
        result.test_level = "interface"
        self.level1_interface[test_name] = result
        
    def add_level2_result(self, test_name: str, result: BuilderTestResult):
        """Add a Level 2 (Specification) test result."""
        result.test_level = "specification"
        self.level2_specification[test_name] = result
        
    def add_level3_result(self, test_name: str, result: BuilderTestResult):
        """Add a Level 3 (Step Creation) test result."""
        result.test_level = "step_creation"
        self.level3_step_creation[test_name] = result
        
    def add_level4_result(self, test_name: str, result: BuilderTestResult):
        """Add a Level 4 (Integration) test result."""
        result.test_level = "integration"
        self.level4_integration[test_name] = result
    
    def add_step_type_result(self, test_name: str, result: BuilderTestResult):
        """Add a step type-specific test result."""
        result.test_level = "step_type_specific"
        self.step_type_specific[test_name] = result
    
    def get_all_results(self) -> Dict[str, BuilderTestResult]:
        """Get all test results across all levels."""
        all_results = {}
        all_results.update(self.level1_interface)
        all_results.update(self.level2_specification)
        all_results.update(self.level3_step_creation)
        all_results.update(self.level4_integration)
        all_results.update(self.step_type_specific)
        return all_results
    
    def generate_summary(self) -> BuilderTestSummary:
        """Generate executive summary of test status."""
        all_results = self.get_all_results()
        self.summary = BuilderTestSummary.from_results(
            self.builder_name, self.builder_class, self.sagemaker_step_type, all_results
        )
        return self.summary
        
    def get_critical_issues(self) -> List[BuilderTestIssue]:
        """Get all critical test issues requiring immediate attention."""
        critical_issues = []
        for result in self.get_all_results().values():
            critical_issues.extend([
                issue for issue in result.issues 
                if issue.severity == "CRITICAL"
            ])
        return critical_issues
        
    def get_error_issues(self) -> List[BuilderTestIssue]:
        """Get all error-level test issues."""
        error_issues = []
        for result in self.get_all_results().values():
            error_issues.extend([
                issue for issue in result.issues 
                if issue.severity == "ERROR"
            ])
        return error_issues
    
    def has_critical_issues(self) -> bool:
        """Check if the report has any critical issues."""
        return len(self.get_critical_issues()) > 0
    
    def has_errors(self) -> bool:
        """Check if the report has any error-level issues."""
        return len(self.get_error_issues()) > 0
    
    def is_passing(self) -> bool:
        """Check if the overall test validation is passing."""
        return not self.has_critical_issues() and not self.has_errors()
        
    def get_recommendations(self) -> List[BuilderTestRecommendation]:
        """Get actionable recommendations for fixing test issues."""
        if not self.recommendations:
            self._generate_recommendations()
        return self.recommendations
    
    def _generate_recommendations(self):
        """Generate recommendations based on found issues."""
        all_issues = []
        for result in self.get_all_results().values():
            all_issues.extend(result.issues)
        
        # Group issues by category to generate targeted recommendations
        issue_categories = {}
        for issue in all_issues:
            if issue.category not in issue_categories:
                issue_categories[issue.category] = []
            issue_categories[issue.category].append(issue)
        
        # Generate recommendations for each category
        for category, issues in issue_categories.items():
            if category == "interface":
                self._add_interface_recommendation(issues)
            elif category == "specification":
                self._add_specification_recommendation(issues)
            elif category == "path_mapping":
                self._add_path_mapping_recommendation(issues)
            elif category == "integration":
                self._add_integration_recommendation(issues)
            elif category.endswith("_specific"):
                self._add_step_type_recommendation(issues, category)
    
    def _add_interface_recommendation(self, issues: List[BuilderTestIssue]):
        """Add recommendation for interface issues."""
        if not issues:
            return
            
        priority = "HIGH" if any(i.severity == "CRITICAL" for i in issues) else "MEDIUM"
        
        recommendation = BuilderTestRecommendation(
            category="interface",
            priority=priority,
            title="Fix Builder Interface Compliance",
            description="Builder does not properly implement required interface methods or inheritance",
            affected_components=["builder", "interface"],
            steps=[
                "Ensure builder inherits from StepBuilderBase",
                "Implement all required methods with correct signatures",
                "Add proper type hints and documentation",
                "Follow naming conventions for methods and classes",
                "Register builder with @register_builder decorator"
            ]
        )
        self.recommendations.append(recommendation)
    
    def _add_specification_recommendation(self, issues: List[BuilderTestIssue]):
        """Add recommendation for specification issues."""
        if not issues:
            return
            
        priority = "HIGH" if any(i.severity == "CRITICAL" for i in issues) else "MEDIUM"
        
        recommendation = BuilderTestRecommendation(
            category="specification",
            priority=priority,
            title="Fix Specification Alignment",
            description="Builder is not properly aligned with step specifications and contracts",
            affected_components=["builder", "specification", "contract"],
            steps=[
                "Review step specification for correct dependencies and outputs",
                "Ensure contract alignment with specification logical names",
                "Verify environment variable handling matches contract",
                "Update builder to use specification-driven approach"
            ]
        )
        self.recommendations.append(recommendation)
    
    def _add_path_mapping_recommendation(self, issues: List[BuilderTestIssue]):
        """Add recommendation for path mapping issues."""
        if not issues:
            return
            
        priority = "HIGH" if any(i.severity == "CRITICAL" for i in issues) else "MEDIUM"
        
        recommendation = BuilderTestRecommendation(
            category="path_mapping",
            priority=priority,
            title="Fix Input/Output Path Mapping",
            description="Builder is not correctly mapping inputs/outputs or property paths",
            affected_components=["builder", "specification", "contract"],
            steps=[
                "Review input/output mapping in _get_inputs() and _get_outputs()",
                "Ensure ProcessingInput/ProcessingOutput objects are created correctly",
                "Verify property paths are valid for the step type",
                "Check container path mapping from contract"
            ]
        )
        self.recommendations.append(recommendation)
    
    def _add_integration_recommendation(self, issues: List[BuilderTestIssue]):
        """Add recommendation for integration issues."""
        if not issues:
            return
            
        priority = "HIGH" if any(i.severity == "CRITICAL" for i in issues) else "MEDIUM"
        
        recommendation = BuilderTestRecommendation(
            category="integration",
            priority=priority,
            title="Fix Integration and Step Creation",
            description="Builder has issues with dependency resolution or step creation",
            affected_components=["builder", "dependencies", "step_creation"],
            steps=[
                "Fix dependency resolution in extract_inputs_from_dependencies()",
                "Ensure create_step() method works correctly",
                "Verify step type detection and classification",
                "Test step creation with mock dependencies"
            ]
        )
        self.recommendations.append(recommendation)
    
    def _add_step_type_recommendation(self, issues: List[BuilderTestIssue], category: str):
        """Add recommendation for step type-specific issues."""
        if not issues:
            return
            
        step_type = category.replace("_specific", "").title()
        priority = "HIGH" if any(i.severity == "CRITICAL" for i in issues) else "MEDIUM"
        
        recommendation = BuilderTestRecommendation(
            category=category,
            priority=priority,
            title=f"Fix {step_type}-Specific Requirements",
            description=f"Builder does not meet {step_type} step type-specific requirements",
            affected_components=["builder", f"{step_type.lower()}_processor", "step_creation"],
            steps=[
                f"Implement {step_type}-specific processor creation methods",
                f"Ensure correct {step_type} input/output handling",
                f"Verify {step_type} step creation and configuration",
                f"Test {step_type}-specific functionality"
            ]
        )
        self.recommendations.append(recommendation)
        
    def export_to_json(self) -> str:
        """Export report to JSON format (similar to alignment validation reports)."""
        if not self.summary:
            self.generate_summary()
            
        report_data = {
            'builder_name': self.builder_name,
            'builder_class': self.builder_class,
            'sagemaker_step_type': self.sagemaker_step_type,
            'level1_interface': {
                'passed': all(r.passed for r in self.level1_interface.values()),
                'issues': [issue.to_dict() for result in self.level1_interface.values() for issue in result.issues],
                'test_results': {k: v.to_dict() for k, v in self.level1_interface.items()}
            },
            'level2_specification': {
                'passed': all(r.passed for r in self.level2_specification.values()),
                'issues': [issue.to_dict() for result in self.level2_specification.values() for issue in result.issues],
                'test_results': {k: v.to_dict() for k, v in self.level2_specification.items()}
            },
            'level3_step_creation': {
                'passed': all(r.passed for r in self.level3_step_creation.values()),
                'issues': [issue.to_dict() for result in self.level3_step_creation.values() for issue in result.issues],
                'test_results': {k: v.to_dict() for k, v in self.level3_step_creation.items()}
            },
            'level4_integration': {
                'passed': all(r.passed for r in self.level4_integration.values()),
                'issues': [issue.to_dict() for result in self.level4_integration.values() for issue in result.issues],
                'test_results': {k: v.to_dict() for k, v in self.level4_integration.items()}
            },
            'step_type_specific': {
                'passed': all(r.passed for r in self.step_type_specific.values()),
                'issues': [issue.to_dict() for result in self.step_type_specific.values() for issue in result.issues],
                'test_results': {k: v.to_dict() for k, v in self.step_type_specific.items()}
            },
            'overall_status': self.summary.overall_status,
            'summary': self.summary.to_dict(),
            'recommendations': [r.to_dict() for r in self.get_recommendations()],
            'metadata': {
                'builder_name': self.builder_name,
                'builder_class': self.builder_class,
                'sagemaker_step_type': self.sagemaker_step_type,
                'validation_timestamp': self.summary.validation_timestamp.isoformat(),
                'validator_version': '1.0.0',
                'test_framework': 'UniversalStepBuilderTest'
            }
        }
        
        # Add scoring data if available
        if 'scoring' in self.metadata:
            report_data['scoring'] = self.metadata['scoring']
            # Also add scoring summary to top level for easy access
            scoring_overall = self.metadata['scoring'].get('overall', {})
            report_data['quality_score'] = scoring_overall.get('score', 0.0)
            report_data['quality_rating'] = scoring_overall.get('rating', 'Unknown')
        
        return json.dumps(report_data, indent=2, default=str)
        
    def save_to_file(self, output_path: Path):
        """Save report to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(self.export_to_json())
    
    def print_summary(self):
        """Print a formatted summary to console."""
        if not self.summary:
            self.generate_summary()
            
        print("\n" + "=" * 80)
        print(f"STEP BUILDER TEST REPORT: {self.builder_name}")
        print("=" * 80)
        
        print(f"\nBuilder: {self.builder_class}")
        print(f"SageMaker Step Type: {self.sagemaker_step_type}")
        print(f"Overall Status: {'✅ ' + self.summary.overall_status if self.is_passing() else '❌ ' + self.summary.overall_status}")
        print(f"Pass Rate: {self.summary.pass_rate:.1f}% ({self.summary.passed_tests}/{self.summary.total_tests})")
        
        # Print quality score if available
        if 'scoring' in self.metadata:
            scoring_overall = self.metadata['scoring'].get('overall', {})
            quality_score = scoring_overall.get('score', 0.0)
            quality_rating = scoring_overall.get('rating', 'Unknown')
            print(f"📊 Quality Score: {quality_score:.1f}/100 - {quality_rating}")
        
        print(f"Total Issues: {self.summary.total_issues}")
        
        if self.summary.total_issues > 0:
            print(f"  🚨 Critical: {self.summary.critical_issues}")
            print(f"  ❌ Error: {self.summary.error_issues}")
            print(f"  ⚠️  Warning: {self.summary.warning_issues}")
            print(f"  ℹ️  Info: {self.summary.info_issues}")
        
        # Print level summaries with scoring if available
        levels = [
            ("Level 1 (Interface)", self.level1_interface, "level1_interface"),
            ("Level 2 (Specification)", self.level2_specification, "level2_specification"),
            ("Level 3 (Step Creation)", self.level3_step_creation, "level3_step_creation"),
            ("Level 4 (Integration)", self.level4_integration, "level4_integration"),
            ("Step Type Specific", self.step_type_specific, "step_type_specific")
        ]
        
        for level_name, level_results, level_key in levels:
            if level_results:
                passed = sum(1 for r in level_results.values() if r.passed)
                total = len(level_results)
                rate = (passed / total * 100) if total > 0 else 0
                
                # Add scoring information if available
                score_info = ""
                if 'scoring' in self.metadata:
                    level_scores = self.metadata['scoring'].get('levels', {})
                    if level_key in level_scores:
                        level_score = level_scores[level_key].get('score', 0.0)
                        score_info = f" - Score: {level_score:.1f}/100"
                
                print(f"\n{level_name}: {passed}/{total} tests passed ({rate:.1f}%){score_info}")
        
        # Print scoring breakdown if available
        if 'scoring' in self.metadata:
            level_scores = self.metadata['scoring'].get('levels', {})
            if level_scores:
                print(f"\n📈 Quality Score Breakdown:")
                for level_key, level_data in level_scores.items():
                    display_name = level_key.replace("level", "L").replace("_", " ").title()
                    score = level_data.get('score', 0.0)
                    # Get weight information (would need to import LEVEL_WEIGHTS or pass it)
                    print(f"  {display_name}: {score:.1f}/100")
        
        # Print critical issues
        critical_issues = self.get_critical_issues()
        if critical_issues:
            print(f"\n🚨 CRITICAL ISSUES ({len(critical_issues)}):")
            for issue in critical_issues:
                print(f"  • {issue.message}")
                if issue.recommendation:
                    print(f"    💡 {issue.recommendation}")
        
        # Print error issues
        error_issues = self.get_error_issues()
        if error_issues:
            print(f"\n❌ ERROR ISSUES ({len(error_issues)}):")
            for issue in error_issues:
                print(f"  • {issue.message}")
                if issue.recommendation:
                    print(f"    💡 {issue.recommendation}")
        
        print("\n" + "=" * 80)


class BuilderTestReporter:
    """
    Main reporter class for generating step builder test reports.
    
    Provides methods to test builders and generate comprehensive reports
    in the same format as alignment validation reports.
    """
    
    def __init__(self, output_dir: Path = None):
        """Initialize the reporter."""
        self.output_dir = output_dir or Path.cwd() / "test" / "steps" / "builders" / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "individual").mkdir(exist_ok=True)
        (self.output_dir / "json").mkdir(exist_ok=True)
        (self.output_dir / "html").mkdir(exist_ok=True)
    
    def test_and_report_builder(self, builder_class: Type[StepBuilderBase], 
                               step_name: str = None) -> BuilderTestReport:
        """
        Test a step builder and generate a comprehensive report.
        
        Args:
            builder_class: The step builder class to test
            step_name: Optional step name (will be inferred if not provided)
            
        Returns:
            BuilderTestReport containing all test results and analysis
        """
        # Infer step name if not provided
        if not step_name:
            step_name = self._infer_step_name(builder_class)
        
        # Get step information
        step_info = STEP_NAMES.get(step_name, {})
        sagemaker_step_type = step_info.get("sagemaker_step_type", "Unknown")
        
        print(f"Testing {step_name} ({builder_class.__name__})...")
        
        # Create report
        report = BuilderTestReport(step_name, builder_class.__name__, sagemaker_step_type)
        
        # Import locally to avoid circular import
        from .universal_test import UniversalStepBuilderTest
        
        # Run universal tests with scoring enabled
        tester = UniversalStepBuilderTest(builder_class, verbose=False, enable_scoring=True)
        test_data = tester.run_all_tests()
        
        # Extract test results and scoring data
        if isinstance(test_data, dict) and 'test_results' in test_data:
            # New format with scoring
            universal_results = test_data['test_results']
            scoring_data = test_data.get('scoring')
        else:
            # Legacy format (raw results)
            universal_results = test_data
            scoring_data = None
        
        # Convert results to BuilderTestResult objects and organize by level
        self._organize_results_into_report(universal_results, report)
        
        # Add scoring data to report metadata if available
        if scoring_data:
            report.metadata['scoring'] = scoring_data
            # Add scoring summary to the report summary
            if report.summary:
                report.summary.metadata['quality_score'] = scoring_data.get('overall', {}).get('score', 0.0)
                report.summary.metadata['quality_rating'] = scoring_data.get('overall', {}).get('rating', 'Unknown')
        
        # Generate summary and recommendations
        report.generate_summary()
        
        return report
    
    def test_and_save_builder_report(self, builder_class: Type[StepBuilderBase], 
                                   step_name: str = None) -> BuilderTestReport:
        """Test a builder and save the report to file."""
        report = self.test_and_report_builder(builder_class, step_name)
        
        # Save to individual report file
        filename = f"{report.builder_name.lower()}_builder_test_report.json"
        output_path = self.output_dir / "individual" / filename
        report.save_to_file(output_path)
        
        print(f"✅ Report saved: {output_path}")
        return report
    
    def test_step_type_builders(self, sagemaker_step_type: str) -> Dict[str, BuilderTestReport]:
        """Test all builders of a specific SageMaker step type."""
        print(f"Testing all {sagemaker_step_type} step builders...")
        print("=" * 60)
        
        # Get all steps of this type
        step_names = get_steps_by_sagemaker_type(sagemaker_step_type)
        
        if not step_names:
            print(f"❌ No {sagemaker_step_type} step builders found")
            return {}
        
        reports = {}
        
        for step_name in step_names:
            try:
                # Load builder class
                builder_class = self._load_builder_class(step_name)
                if not builder_class:
                    print(f"  ❌ Could not load builder class for {step_name}")
                    continue
                
                # Test and save report
                report = self.test_and_save_builder_report(builder_class, step_name)
                reports[step_name] = report
                
            except Exception as e:
                print(f"  ❌ Failed to test {step_name}: {e}")
        
        # Generate step type summary
        self._generate_step_type_summary(sagemaker_step_type, reports)
        
        return reports
    
    def _organize_results_into_report(self, results: Dict[str, Any], report: BuilderTestReport):
        """Organize universal test results into the report structure."""
        
        # Define test level mappings
        level1_tests = [
            "test_inheritance", "test_required_methods", "test_configuration_validation",
            "test_documentation_standards", "test_naming_conventions", "test_registry_integration",
            "test_type_hints", "test_method_return_types", "test_error_handling"
        ]
        
        level2_tests = [
            "test_specification_usage", "test_contract_alignment", 
            "test_environment_variable_handling", "test_job_arguments"
        ]
        
        level3_tests = [
            "test_input_path_mapping", "test_output_path_mapping", "test_property_path_validity"
        ]
        
        level4_tests = [
            "test_dependency_resolution", "test_step_creation", "test_step_name"
        ]
        
        step_type_tests = [
            "test_step_type_detection", "test_step_type_classification", "test_step_type_compliance"
        ]
        
        # Convert and organize results
        for test_name, result in results.items():
            if not isinstance(result, dict):
                continue
                
            # Create BuilderTestResult
            test_result = self._convert_to_builder_test_result(test_name, result)
            
            # Add to appropriate level
            if test_name in level1_tests:
                report.add_level1_result(test_name, test_result)
            elif test_name in level2_tests:
                report.add_level2_result(test_name, test_result)
            elif test_name in level3_tests:
                report.add_level3_result(test_name, test_result)
            elif test_name in level4_tests:
                report.add_level4_result(test_name, test_result)
            elif test_name in step_type_tests or test_name.startswith("test_processing") or test_name.startswith("test_training"):
                report.add_step_type_result(test_name, test_result)
    
    def _convert_to_builder_test_result(self, test_name: str, result: Dict[str, Any]) -> BuilderTestResult:
        """Convert universal test result to BuilderTestResult."""
        test_result = BuilderTestResult(
            test_name=test_name,
            passed=result.get("passed", False),
            details=result.get("details", {}),
            test_level="unknown"  # Will be set when added to report
        )
        
        # Only create issues for failed tests - passed tests don't need issues
        if not result.get("passed", False):
            error_message = result.get("error", f"{test_name} failed")
            issue = BuilderTestIssue(
                severity="ERROR",
                category="test_failure",
                message=error_message,
                details=result.get("details", {}),
                recommendation=f"Fix {test_name} failure",
                test_name=test_name
            )
            test_result.add_issue(issue)
        # Don't add INFO issues for passed tests - this inflates the issue count
        # The test_result.passed=True is sufficient to indicate success
        
        return test_result
    
    def _infer_step_name(self, builder_class: Type[StepBuilderBase]) -> str:
        """Infer step name from builder class name."""
        class_name = builder_class.__name__
        
        # Remove "StepBuilder" suffix
        if class_name.endswith("StepBuilder"):
            step_name = class_name[:-11]  # Remove "StepBuilder"
        else:
            step_name = class_name
        
        # Look for matching step name in registry
        for name, info in STEP_NAMES.items():
            if info.get("builder_step_name", "").replace("StepBuilder", "") == step_name:
                return name
        
        return step_name
    
    def _load_builder_class(self, step_name: str) -> Optional[Type[StepBuilderBase]]:
        """Load a builder class by step name."""
        step_info = STEP_NAMES.get(step_name, {})
        builder_class_name = step_info.get("builder_step_name")
        
        if not builder_class_name:
            return None
        
        # Try to import the builder class
        try:
            import importlib
            
            # Map step names to actual module names based on existing files
            module_mapping = {
                "TabularPreprocessing": "builder_tabular_preprocessing_step",
                "RiskTableMapping": "builder_risk_table_mapping_step",
                "CurrencyConversion": "builder_currency_conversion_step",
                "DummyTraining": "builder_dummy_training_step",
                "XGBoostModelEval": "builder_xgboost_model_eval_step",
                "ModelCalibration": "builder_model_calibration_step",
                "Package": "builder_package_step",
                "Payload": "builder_payload_step",
                "PyTorchTraining": "builder_pytorch_training_step",
                "XGBoostTraining": "builder_xgboost_training_step",
                "PyTorchModel": "builder_pytorch_model_step",
                "XGBoostModel": "builder_xgboost_model_step",
                "BatchTransform": "builder_batch_transform_step",
                "CradleDataLoading": "builder_cradle_data_loading_step",
                "Registration": "builder_registration_step"
            }
            
            module_name = module_mapping.get(step_name)
            if not module_name:
                # Fallback to generic naming
                module_name = f"builder_{step_name.lower()}_step"
            
            module_path = f"src.cursus.steps.builders.{module_name}"
            module = importlib.import_module(module_path)
            return getattr(module, builder_class_name)
        except (ImportError, AttributeError) as e:
            print(f"Failed to load {step_name} builder from {module_path}: {e}")
            return None
    
    def _generate_step_type_summary(self, step_type: str, reports: Dict[str, BuilderTestReport]):
        """Generate and save a summary report for a step type."""
        summary_data = {
            "step_type": step_type,
            "summary": {
                "total_builders": len(reports),
                "passing_builders": sum(1 for r in reports.values() if r.is_passing()),
                "mostly_passing_builders": sum(1 for r in reports.values() if r.summary and r.summary.overall_status == "MOSTLY_PASSING"),
                "partially_passing_builders": sum(1 for r in reports.values() if r.summary and r.summary.overall_status == "PARTIALLY_PASSING"),
                "failing_builders": sum(1 for r in reports.values() if r.summary and r.summary.overall_status == "FAILING"),
            },
            "builder_reports": {name: report.summary.to_dict() if report.summary else {} for name, report in reports.items()},
            "metadata": {
                "step_type": step_type,
                "generation_timestamp": datetime.now().isoformat(),
                "validator_version": "1.0.0",
                "test_framework": "UniversalStepBuilderTest"
            }
        }
        
        # Save step type summary
        summary_file = self.output_dir / "json" / f"{step_type.lower()}_builder_test_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        print(f"✅ {step_type} summary saved: {summary_file}")
