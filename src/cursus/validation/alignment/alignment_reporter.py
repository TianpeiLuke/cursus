"""
Alignment validation reporting system.

Provides comprehensive reporting capabilities for alignment validation results,
including summary generation, issue analysis, scoring, and export functionality.
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from .alignment_utils import (
    AlignmentIssue, SeverityLevel, AlignmentLevel,
    group_issues_by_severity, get_highest_severity,
    format_alignment_issue
)
from .alignment_scorer import AlignmentScorer


class ValidationResult(BaseModel):
    """
    Result of a single validation check.
    
    Contains detailed information about what was tested,
    whether it passed, and specific issues found.
    """
    test_name: str
    passed: bool
    issues: List[AlignmentIssue] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
        
    def add_issue(self, issue: AlignmentIssue):
        """Add an alignment issue to this result."""
        self.issues.append(issue)
        # Update passed status based on severity
        if issue.level in [SeverityLevel.ERROR, SeverityLevel.CRITICAL]:
            self.passed = False
            
    def get_severity_level(self) -> Optional[SeverityLevel]:
        """Get the highest severity level among all issues."""
        return get_highest_severity(self.issues)
    
    def has_critical_issues(self) -> bool:
        """Check if this result has critical issues."""
        return any(issue.level == SeverityLevel.CRITICAL for issue in self.issues)
    
    def has_errors(self) -> bool:
        """Check if this result has error-level issues."""
        return any(issue.level == SeverityLevel.ERROR for issue in self.issues)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'test_name': self.test_name,
            'passed': self.passed,
            'timestamp': self.timestamp.isoformat(),
            'issues': [issue.model_dump() for issue in self.issues],
            'details': self.details,
            'severity_level': self.get_severity_level().value if self.get_severity_level() else None
        }


class AlignmentSummary(BaseModel):
    """
    Executive summary of alignment validation results.
    
    Provides high-level statistics and key findings from the validation.
    """
    total_tests: int
    passed_tests: int
    failed_tests: int
    pass_rate: float
    total_issues: int
    critical_issues: int
    error_issues: int
    warning_issues: int
    info_issues: int
    highest_severity: Optional[SeverityLevel]
    validation_timestamp: datetime = Field(default_factory=datetime.now)
    
    @classmethod
    def from_results(cls, results: Dict[str, ValidationResult]) -> 'AlignmentSummary':
        """Create AlignmentSummary from validation results."""
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
        grouped_issues = group_issues_by_severity(all_issues)
        critical_issues = len(grouped_issues[SeverityLevel.CRITICAL])
        error_issues = len(grouped_issues[SeverityLevel.ERROR])
        warning_issues = len(grouped_issues[SeverityLevel.WARNING])
        info_issues = len(grouped_issues[SeverityLevel.INFO])
        
        highest_severity = get_highest_severity(all_issues)
        
        return cls(
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
            validation_timestamp=validation_timestamp
        )
    
    def is_passing(self) -> bool:
        """Check if the overall validation is passing (no critical or error issues)."""
        return self.critical_issues == 0 and self.error_issues == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'pass_rate': self.pass_rate,
            'total_issues': self.total_issues,
            'critical_issues': self.critical_issues,
            'error_issues': self.error_issues,
            'warning_issues': self.warning_issues,
            'info_issues': self.info_issues,
            'highest_severity': self.highest_severity.value if self.highest_severity else None,
            'validation_timestamp': self.validation_timestamp.isoformat(),
            'is_passing': self.is_passing()
        }


class AlignmentRecommendation(BaseModel):
    """
    Actionable recommendation for fixing alignment issues.
    
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


class AlignmentReport:
    """
    Comprehensive report of alignment validation results.
    
    Contains results from all four alignment levels with detailed
    analysis, scoring, and actionable recommendations.
    """
    
    def __init__(self):
        self.level1_results: Dict[str, ValidationResult] = {}  # Script ↔ Contract
        self.level2_results: Dict[str, ValidationResult] = {}  # Contract ↔ Specification
        self.level3_results: Dict[str, ValidationResult] = {}  # Specification ↔ Dependencies
        self.level4_results: Dict[str, ValidationResult] = {}  # Builder ↔ Configuration
        self.summary: Optional[AlignmentSummary] = None
        self.recommendations: List[AlignmentRecommendation] = []
        self.metadata: Dict[str, Any] = {}
        self._scorer: Optional[AlignmentScorer] = None
        
    def add_level1_result(self, test_name: str, result: ValidationResult):
        """Add a Level 1 (Script ↔ Contract) validation result."""
        self.level1_results[test_name] = result
        
    def add_level2_result(self, test_name: str, result: ValidationResult):
        """Add a Level 2 (Contract ↔ Specification) validation result."""
        self.level2_results[test_name] = result
        
    def add_level3_result(self, test_name: str, result: ValidationResult):
        """Add a Level 3 (Specification ↔ Dependencies) validation result."""
        self.level3_results[test_name] = result
        
    def add_level4_result(self, test_name: str, result: ValidationResult):
        """Add a Level 4 (Builder ↔ Configuration) validation result."""
        self.level4_results[test_name] = result
    
    def get_all_results(self) -> Dict[str, ValidationResult]:
        """Get all validation results across all levels."""
        all_results = {}
        all_results.update(self.level1_results)
        all_results.update(self.level2_results)
        all_results.update(self.level3_results)
        all_results.update(self.level4_results)
        return all_results
    
    def generate_summary(self) -> AlignmentSummary:
        """Generate executive summary of alignment status."""
        all_results = self.get_all_results()
        self.summary = AlignmentSummary.from_results(all_results)
        return self.summary
        
    def get_critical_issues(self) -> List[AlignmentIssue]:
        """Get all critical alignment issues requiring immediate attention."""
        critical_issues = []
        for result in self.get_all_results().values():
            critical_issues.extend([
                issue for issue in result.issues 
                if issue.level == SeverityLevel.CRITICAL
            ])
        return critical_issues
        
    def get_error_issues(self) -> List[AlignmentIssue]:
        """Get all error-level alignment issues."""
        error_issues = []
        for result in self.get_all_results().values():
            error_issues.extend([
                issue for issue in result.issues 
                if issue.level == SeverityLevel.ERROR
            ])
        return error_issues
    
    def has_critical_issues(self) -> bool:
        """Check if the report has any critical issues."""
        return len(self.get_critical_issues()) > 0
    
    def has_errors(self) -> bool:
        """Check if the report has any error-level issues."""
        return len(self.get_error_issues()) > 0
    
    def is_passing(self) -> bool:
        """Check if the overall alignment validation is passing."""
        return not self.has_critical_issues() and not self.has_errors()
    
    def get_scorer(self) -> AlignmentScorer:
        """Get the alignment scorer for this report."""
        if self._scorer is None:
            # Convert validation results to scorer format
            scorer_results = self._convert_to_scorer_format()
            self._scorer = AlignmentScorer(scorer_results)
        return self._scorer
    
    def _convert_to_scorer_format(self) -> Dict[str, Any]:
        """Convert validation results to format expected by AlignmentScorer."""
        scorer_results = {}
        
        # Add level results with proper naming that matches AlignmentScorer expectations
        if self.level1_results:
            scorer_results['level1_results'] = {
                name: {
                    'passed': result.passed,
                    'issues': [issue.model_dump() for issue in result.issues],
                    'details': result.details
                }
                for name, result in self.level1_results.items()
            }
        
        if self.level2_results:
            scorer_results['level2_results'] = {
                name: {
                    'passed': result.passed,
                    'issues': [issue.model_dump() for issue in result.issues],
                    'details': result.details
                }
                for name, result in self.level2_results.items()
            }
        
        if self.level3_results:
            scorer_results['level3_results'] = {
                name: {
                    'passed': result.passed,
                    'issues': [issue.model_dump() for issue in result.issues],
                    'details': result.details
                }
                for name, result in self.level3_results.items()
            }
        
        if self.level4_results:
            scorer_results['level4_results'] = {
                name: {
                    'passed': result.passed,
                    'issues': [issue.model_dump() for issue in result.issues],
                    'details': result.details
                }
                for name, result in self.level4_results.items()
            }
        
        return scorer_results
    
    def get_alignment_score(self) -> float:
        """Get the overall alignment score (0.0 to 100.0)."""
        return self.get_scorer().calculate_overall_score()
    
    def get_level_scores(self) -> Dict[str, float]:
        """Get alignment scores for each level."""
        scorer = self.get_scorer()
        return {
            'level1_script_contract': scorer.calculate_level_score('level1_script_contract')[0],
            'level2_contract_spec': scorer.calculate_level_score('level2_contract_spec')[0],
            'level3_spec_dependencies': scorer.calculate_level_score('level3_spec_dependencies')[0],
            'level4_builder_config': scorer.calculate_level_score('level4_builder_config')[0]
        }
    
    def generate_alignment_chart(self, output_path: str = None) -> str:
        """Generate alignment score visualization chart."""
        return self.get_scorer().generate_chart(output_path)
    
    def get_scoring_report(self) -> Dict[str, Any]:
        """Get comprehensive scoring report."""
        return self.get_scorer().generate_report()
    
    def print_scoring_summary(self):
        """Print alignment scoring summary to console."""
        self.get_scorer().print_summary()
        
    def get_recommendations(self) -> List[AlignmentRecommendation]:
        """Get actionable recommendations for fixing alignment issues."""
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
            if category == "path_usage":
                self._add_path_usage_recommendation(issues)
            elif category == "environment_variables":
                self._add_env_var_recommendation(issues)
            elif category == "logical_names":
                self._add_logical_name_recommendation(issues)
            elif category == "dependency_resolution":
                self._add_dependency_recommendation(issues)
            elif category == "configuration":
                self._add_configuration_recommendation(issues)
    
    def _add_path_usage_recommendation(self, issues: List[AlignmentIssue]):
        """Add recommendation for path usage issues."""
        if not issues:
            return
            
        priority = "HIGH" if any(i.level == SeverityLevel.CRITICAL for i in issues) else "MEDIUM"
        
        recommendation = AlignmentRecommendation(
            category="path_usage",
            priority=priority,
            title="Fix Script Path Usage",
            description="Scripts are not using contract-defined paths correctly",
            affected_components=["script", "contract"],
            steps=[
                "Review script contract for expected input/output paths",
                "Update script to use contract paths exactly",
                "Remove any hardcoded paths not in contract",
                "Test script with contract validation"
            ]
        )
        self.recommendations.append(recommendation)
    
    def _add_env_var_recommendation(self, issues: List[AlignmentIssue]):
        """Add recommendation for environment variable issues."""
        if not issues:
            return
            
        priority = "HIGH" if any(i.level == SeverityLevel.CRITICAL for i in issues) else "MEDIUM"
        
        recommendation = AlignmentRecommendation(
            category="environment_variables",
            priority=priority,
            title="Fix Environment Variable Usage",
            description="Scripts are not accessing environment variables as declared in contract",
            affected_components=["script", "contract", "builder"],
            steps=[
                "Review contract for required and optional environment variables",
                "Update script to access only declared environment variables",
                "Ensure builder sets all required environment variables",
                "Add proper defaults for optional variables"
            ]
        )
        self.recommendations.append(recommendation)
    
    def _add_logical_name_recommendation(self, issues: List[AlignmentIssue]):
        """Add recommendation for logical name alignment issues."""
        if not issues:
            return
            
        priority = "HIGH" if any(i.level == SeverityLevel.CRITICAL for i in issues) else "MEDIUM"
        
        recommendation = AlignmentRecommendation(
            category="logical_names",
            priority=priority,
            title="Align Logical Names",
            description="Logical names between contract and specification do not match",
            affected_components=["contract", "specification"],
            steps=[
                "Review contract input/output logical names",
                "Review specification dependency and output names",
                "Ensure all logical names match exactly",
                "Update either contract or specification for consistency"
            ]
        )
        self.recommendations.append(recommendation)
    
    def _add_dependency_recommendation(self, issues: List[AlignmentIssue]):
        """Add recommendation for dependency resolution issues."""
        if not issues:
            return
            
        priority = "HIGH" if any(i.level == SeverityLevel.CRITICAL for i in issues) else "MEDIUM"
        
        recommendation = AlignmentRecommendation(
            category="dependency_resolution",
            priority=priority,
            title="Fix Dependency Resolution",
            description="Dependencies cannot be resolved or have incorrect compatible sources",
            affected_components=["specification", "dependencies"],
            steps=[
                "Review specification dependencies",
                "Check compatible_sources lists for accuracy",
                "Verify upstream steps produce expected outputs",
                "Update dependency specifications as needed"
            ]
        )
        self.recommendations.append(recommendation)
    
    def _add_configuration_recommendation(self, issues: List[AlignmentIssue]):
        """Add recommendation for configuration issues."""
        if not issues:
            return
            
        priority = "HIGH" if any(i.level == SeverityLevel.CRITICAL for i in issues) else "MEDIUM"
        
        recommendation = AlignmentRecommendation(
            category="configuration",
            priority=priority,
            title="Fix Builder Configuration",
            description="Builder is not handling configuration parameters correctly",
            affected_components=["builder", "configuration"],
            steps=[
                "Review builder configuration usage",
                "Ensure all config parameters are used appropriately",
                "Verify environment variables are set from config",
                "Test builder with different configuration values"
            ]
        )
        self.recommendations.append(recommendation)
        
    def export_to_json(self) -> str:
        """Export report to JSON format with scoring information."""
        if not self.summary:
            self.generate_summary()
            
        # Get scoring information
        overall_score = self.get_alignment_score()
        scorer = self.get_scorer()
        quality_rating = scorer.get_rating(overall_score)
        
        report_data = {
            'summary': self.summary.to_dict(),
            'scoring': {
                'overall_score': overall_score,
                'quality_rating': quality_rating,
                'level_scores': self.get_level_scores(),
                'scoring_report': self.get_scoring_report()
            },
            'level1_results': {k: v.to_dict() for k, v in self.level1_results.items()},
            'level2_results': {k: v.to_dict() for k, v in self.level2_results.items()},
            'level3_results': {k: v.to_dict() for k, v in self.level3_results.items()},
            'level4_results': {k: v.to_dict() for k, v in self.level4_results.items()},
            'recommendations': [r.to_dict() for r in self.get_recommendations()],
            'metadata': self.metadata
        }
        
        return json.dumps(report_data, indent=2, default=str)
        
    def export_to_html(self) -> str:
        """Export report to HTML format with scoring visualizations."""
        if not self.summary:
            self.generate_summary()
            
        # Get scoring information
        overall_score = self.get_alignment_score()
        level_scores = self.get_level_scores()
        
        html_template = """<!DOCTYPE html>
<html>
<head>
    <title>Alignment Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .metric {{ text-align: center; padding: 10px; }}
        .metric h3 {{ margin: 0; font-size: 2em; }}
        .metric p {{ margin: 5px 0; color: #666; }}
        .passing {{ color: #28a745; }}
        .failing {{ color: #dc3545; }}
        .warning {{ color: #ffc107; }}
        .scoring-section {{ margin: 20px 0; padding: 20px; background-color: #f8f9fa; border-radius: 5px; }}
        .score-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0; }}
        .score-card {{ background: white; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; text-align: center; }}
        .score-excellent {{ border-left-color: #28a745; }}
        .score-good {{ border-left-color: #17a2b8; }}
        .score-fair {{ border-left-color: #ffc107; }}
        .score-poor {{ border-left-color: #dc3545; }}
        .score-value {{ font-size: 2em; font-weight: bold; margin: 10px 0; }}
        .level-section {{ margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }}
        .level-header {{ background-color: #e9ecef; padding: 10px; font-weight: bold; }}
        .test-result {{ padding: 10px; border-bottom: 1px solid #eee; }}
        .test-passed {{ border-left: 4px solid #28a745; }}
        .test-failed {{ border-left: 4px solid #dc3545; }}
        .issue {{ margin: 5px 0; padding: 5px; background-color: #f8f9fa; border-radius: 3px; }}
        .critical {{ border-left: 4px solid #dc3545; }}
        .error {{ border-left: 4px solid #fd7e14; }}
        .warning {{ border-left: 4px solid #ffc107; }}
        .info {{ border-left: 4px solid #17a2b8; }}
        .recommendations {{ margin: 20px 0; }}
        .recommendation {{ margin: 10px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .high-priority {{ border-left: 4px solid #dc3545; }}
        .medium-priority {{ border-left: 4px solid #ffc107; }}
        .low-priority {{ border-left: 4px solid #28a745; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Alignment Validation Report</h1>
        <p>Generated: {timestamp}</p>
        <p>Overall Status: <span class="{status_class}">{status}</span></p>
        <p>Overall Alignment Score: <strong>{overall_score:.1f}/100</strong></p>
    </div>
    
    <div class="summary">
        <div class="metric">
            <h3>{pass_rate:.1f}%</h3>
            <p>Pass Rate</p>
        </div>
        <div class="metric">
            <h3>{total_tests}</h3>
            <p>Total Tests</p>
        </div>
        <div class="metric">
            <h3>{total_issues}</h3>
            <p>Total Issues</p>
        </div>
        <div class="metric">
            <h3>{critical_issues}</h3>
            <p>Critical Issues</p>
        </div>
    </div>
    
    <div class="scoring-section">
        <h2>Alignment Scoring</h2>
        <div class="score-grid">
            {score_cards}
        </div>
    </div>
    
    {level_sections}
    
    <div class="recommendations">
        <h2>Recommendations</h2>
        {recommendations_html}
    </div>
</body>
</html>"""
        
        # Generate level sections
        level_sections = ""
        for level_num, (level_name, results) in enumerate([
            ("Level 1: Script ↔ Contract", self.level1_results),
            ("Level 2: Contract ↔ Specification", self.level2_results),
            ("Level 3: Specification ↔ Dependencies", self.level3_results),
            ("Level 4: Builder ↔ Configuration", self.level4_results)
        ], 1):
            if results:
                level_sections += self._generate_level_html(level_name, results)
        
        # Generate recommendations HTML
        recommendations_html = ""
        for rec in self.get_recommendations():
            priority_class = f"{rec.priority.lower()}-priority"
            steps_html = "".join(f"<li>{step}</li>" for step in rec.steps)
            recommendations_html += f"""
            <div class="recommendation {priority_class}">
                <h3>{rec.title} ({rec.priority} Priority)</h3>
                <p>{rec.description}</p>
                <p><strong>Affected Components:</strong> {', '.join(rec.affected_components)}</p>
                <p><strong>Steps:</strong></p>
                <ol>{steps_html}</ol>
            </div>
            """
        
        # Generate score cards
        score_cards = self._generate_score_cards(overall_score, level_scores)
        
        return html_template.format(
            timestamp=self.summary.validation_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            status="PASSING" if self.is_passing() else "FAILING",
            status_class="passing" if self.is_passing() else "failing",
            overall_score=overall_score,
            pass_rate=self.summary.pass_rate,
            total_tests=self.summary.total_tests,
            total_issues=self.summary.total_issues,
            critical_issues=self.summary.critical_issues,
            score_cards=score_cards,
            level_sections=level_sections,
            recommendations_html=recommendations_html
        )
    
    def _generate_level_html(self, level_name: str, results: Dict[str, ValidationResult]) -> str:
        """Generate HTML for a specific alignment level."""
        level_html = f"""
        <div class="level-section">
            <div class="level-header">{level_name}</div>
        """
        
        for test_name, result in results.items():
            result_class = "test-passed" if result.passed else "test-failed"
            issues_html = ""
            
            for issue in result.issues:
                issue_class = issue.level.value.lower()
                issues_html += f"""
                <div class="issue {issue_class}">
                    <strong>{issue.level.value}:</strong> {issue.message}
                    {f'<br><em>Recommendation: {issue.recommendation}</em>' if issue.recommendation else ''}
                </div>
                """
            
            level_html += f"""
            <div class="test-result {result_class}">
                <h4>{test_name}</h4>
                <p>Status: {'PASSED' if result.passed else 'FAILED'}</p>
                {issues_html}
            </div>
            """
        
        level_html += "</div>"
        return level_html
    
    def _generate_score_cards(self, overall_score: float, level_scores: Dict[str, float]) -> str:
        """Generate HTML score cards for alignment levels."""
        def get_score_class(score: float) -> str:
            """Get CSS class based on score."""
            if score >= 90:
                return "score-excellent"
            elif score >= 75:
                return "score-good"
            elif score >= 60:
                return "score-fair"
            else:
                return "score-poor"
        
        def get_score_quality(score: float) -> str:
            """Get quality description based on score."""
            if score >= 90:
                return "Excellent"
            elif score >= 75:
                return "Good"
            elif score >= 60:
                return "Fair"
            else:
                return "Poor"
        
        # Overall score card
        overall_class = get_score_class(overall_score)
        overall_quality = get_score_quality(overall_score)
        
        score_cards = f"""
        <div class="score-card {overall_class}">
            <h4>Overall Alignment</h4>
            <div class="score-value">{overall_score:.1f}</div>
            <p>{overall_quality}</p>
        </div>
        """
        
        # Level score cards
        level_names = {
            'level1_script_contract': 'Script ↔ Contract',
            'level2_contract_spec': 'Contract ↔ Specification',
            'level3_spec_dependencies': 'Specification ↔ Dependencies',
            'level4_builder_config': 'Builder ↔ Configuration'
        }
        
        for level_key, level_name in level_names.items():
            if level_key in level_scores:
                score = level_scores[level_key]
                score_class = get_score_class(score)
                quality = get_score_quality(score)
                
                score_cards += f"""
        <div class="score-card {score_class}">
            <h4>{level_name}</h4>
            <div class="score-value">{score:.1f}</div>
            <p>{quality}</p>
        </div>
                """
        
        return score_cards
    
    def print_summary(self):
        """Print a formatted summary to console."""
        if not self.summary:
            self.generate_summary()
            
        print("\n" + "=" * 80)
        print("ALIGNMENT VALIDATION REPORT")
        print("=" * 80)
        
        print(f"\nOverall Status: {'✅ PASSING' if self.is_passing() else '❌ FAILING'}")
        print(f"Pass Rate: {self.summary.pass_rate:.1f}% ({self.summary.passed_tests}/{self.summary.total_tests})")
        print(f"Total Issues: {self.summary.total_issues}")
        
        if self.summary.total_issues > 0:
            print(f"  🚨 Critical: {self.summary.critical_issues}")
            print(f"  ❌ Error: {self.summary.error_issues}")
            print(f"  ⚠️  Warning: {self.summary.warning_issues}")
            print(f"  ℹ️  Info: {self.summary.info_issues}")
        
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
