"""
Consolidated Validation Reporter

This module provides comprehensive reporting capabilities for the validation alignment system.
Consolidates functionality from alignment_reporter.py, alignment_scorer.py, and enhanced_reporter.py.
"""

from typing import Dict, Any, List, Optional, Union, TextIO
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sys
from datetime import datetime

from ..utils.validation_models import (
    ValidationResult, ValidationSummary, ValidationIssue, ValidationStatus,
    IssueLevel, ValidationLevel, format_validation_summary
)

logger = logging.getLogger(__name__)


@dataclass
class ReportingConfig:
    """Configuration for validation reporting."""
    include_passed: bool = True
    include_excluded: bool = False
    include_metadata: bool = True
    include_issue_details: bool = True
    max_issues_per_step: Optional[int] = None
    output_format: str = "text"  # text, json, html
    color_output: bool = True
    verbose: bool = False


class ValidationReporter:
    """
    Comprehensive validation reporter that handles all reporting needs.
    
    This class consolidates the functionality of multiple reporter classes:
    - Basic alignment reporting
    - Scoring and metrics calculation
    - Enhanced formatting and output options
    """
    
    def __init__(self, config: Optional[ReportingConfig] = None):
        """Initialize the validation reporter."""
        self.config = config or ReportingConfig()
        self.summary = ValidationSummary()
        self._color_codes = {
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'magenta': '\033[95m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'bold': '\033[1m',
            'underline': '\033[4m',
            'reset': '\033[0m'
        }
    
    def add_result(self, result: ValidationResult):
        """Add a validation result to the reporter."""
        self.summary.add_result(result)
    
    def add_results(self, results: List[ValidationResult]):
        """Add multiple validation results to the reporter."""
        for result in results:
            self.add_result(result)
    
    def generate_report(self, output_file: Optional[Union[str, Path, TextIO]] = None) -> str:
        """
        Generate a comprehensive validation report.
        
        Args:
            output_file: Optional file path or file object to write the report to
            
        Returns:
            The generated report as a string
        """
        if self.config.output_format == "json":
            report = self._generate_json_report()
        elif self.config.output_format == "html":
            report = self._generate_html_report()
        else:
            report = self._generate_text_report()
        
        # Write to file if specified
        if output_file:
            if isinstance(output_file, (str, Path)):
                with open(output_file, 'w') as f:
                    f.write(report)
            else:
                output_file.write(report)
        
        return report
    
    def _generate_text_report(self) -> str:
        """Generate a text-based validation report."""
        lines = []
        
        # Header
        lines.append(self._colorize("=" * 80, 'bold'))
        lines.append(self._colorize("VALIDATION ALIGNMENT REPORT", 'bold'))
        lines.append(self._colorize("=" * 80, 'bold'))
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Summary
        lines.append(self._colorize("SUMMARY", 'bold', 'underline'))
        lines.append(format_validation_summary(self.summary))
        lines.append("")
        
        # Detailed results
        if self.config.verbose or self.summary.failed_steps > 0:
            lines.append(self._colorize("DETAILED RESULTS", 'bold', 'underline'))
            lines.append("")
            
            for result in self.summary.results:
                if not self._should_include_result(result):
                    continue
                
                lines.extend(self._format_result_text(result))
                lines.append("")
        
        # Issue summary by type (show in detailed mode or when there are issues)
        if self.summary.total_issues > 0 or self.config.verbose:
            lines.append(self._colorize("Issue Breakdown", 'bold', 'underline'))
            lines.extend(self._generate_issue_breakdown())
            lines.append("")
        
        # Recommendations
        if self.summary.failed_steps > 0:
            lines.append(self._colorize("RECOMMENDATIONS", 'bold', 'underline'))
            lines.extend(self._generate_recommendations())
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_json_report(self) -> str:
        """Generate a JSON-based validation report."""
        report_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_version": "1.0",
                "config": {
                    "include_passed": self.config.include_passed,
                    "include_excluded": self.config.include_excluded,
                    "include_metadata": self.config.include_metadata,
                    "include_issue_details": self.config.include_issue_details,
                    "max_issues_per_step": self.config.max_issues_per_step,
                    "verbose": self.config.verbose
                }
            },
            "summary": self.summary.to_dict(),
            "results": []
        }
        
        # Add detailed results
        for result in self.summary.results:
            if self._should_include_result(result):
                result_data = result.to_dict()
                
                # Limit issues if configured
                if (self.config.max_issues_per_step and 
                    len(result_data["issues"]) > self.config.max_issues_per_step):
                    result_data["issues"] = result_data["issues"][:self.config.max_issues_per_step]
                    result_data["truncated_issues"] = True
                
                report_data["results"].append(result_data)
        
        return json.dumps(report_data, indent=2)
    
    def _generate_html_report(self) -> str:
        """Generate an HTML-based validation report."""
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>Validation Alignment Report</title>",
            "<style>",
            self._get_html_styles(),
            "</style>",
            "</head>",
            "<body>",
            "<div class='container'>",
            "<h1>Validation Alignment Report</h1>",
            f"<p class='timestamp'>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
            
            # Summary section
            "<div class='summary'>",
            "<h2>Summary</h2>",
            self._generate_html_summary(),
            "</div>",
            
            # Results section
            "<div class='results'>",
            "<h2>Detailed Results</h2>",
            self._generate_html_results(),
            "</div>",
            
            "</div>",
            "</body>",
            "</html>"
        ]
        
        return "\n".join(html_parts)
    
    def _format_result_text(self, result: ValidationResult) -> List[str]:
        """Format a single validation result as text."""
        lines = []
        
        # Status indicator
        status_color = self._get_status_color(result.status)
        status_symbol = self._get_status_symbol(result.status)
        
        lines.append(self._colorize(
            f"{status_symbol} {result.step_name} [{result.status.value}]",
            status_color, 'bold'
        ))
        
        # Validation level
        if result.validation_level:
            lines.append(f"  Level: {result.validation_level.value}")
        
        # Issue counts
        if result.total_issues > 0:
            issue_parts = []
            if result.error_count > 0:
                issue_parts.append(self._colorize(f"{result.error_count} errors", 'red'))
            if result.warning_count > 0:
                issue_parts.append(self._colorize(f"{result.warning_count} warnings", 'yellow'))
            if result.info_count > 0:
                issue_parts.append(self._colorize(f"{result.info_count} info", 'blue'))
            
            lines.append(f"  Issues: {', '.join(issue_parts)}")
        
        # Error message
        if result.error_message:
            lines.append(f"  Error: {result.error_message}")
        
        # Issues details
        if self.config.include_issue_details and result.issues:
            lines.append("  Details:")
            issues_to_show = result.issues
            if self.config.max_issues_per_step:
                issues_to_show = issues_to_show[:self.config.max_issues_per_step]
            
            for issue in issues_to_show:
                issue_color = self._get_issue_color(issue.level)
                lines.append(f"    {self._colorize('•', issue_color)} {issue.message}")
                if issue.method_name:
                    lines.append(f"      Method: {issue.method_name}")
        
        # Metadata
        if self.config.include_metadata and result.metadata:
            lines.append("  Metadata:")
            for key, value in result.metadata.items():
                lines.append(f"    {key}: {value}")
        
        return lines
    
    def _generate_issue_breakdown(self) -> List[str]:
        """Generate a breakdown of issues by type and severity."""
        lines = []
        
        # Group issues by level
        issue_counts = {
            IssueLevel.ERROR: 0,
            IssueLevel.WARNING: 0,
            IssueLevel.INFO: 0
        }
        
        method_issues = {}
        rule_type_issues = {}
        
        for result in self.summary.results:
            for issue in result.issues:
                issue_counts[issue.level] += 1
                
                # Group by method
                method = issue.method_name or "unknown"
                if method not in method_issues:
                    method_issues[method] = 0
                method_issues[method] += 1
                
                # Group by rule type
                rule_type = issue.rule_type.value if issue.rule_type else "unknown"
                if rule_type not in rule_type_issues:
                    rule_type_issues[rule_type] = 0
                rule_type_issues[rule_type] += 1
        
        # Issue counts by severity
        lines.append("By Severity:")
        for level, count in issue_counts.items():
            if count > 0:
                color = self._get_issue_color(level)
                lines.append(f"  {self._colorize(level.value, color)}: {count}")
        
        # Top methods with issues
        if method_issues:
            lines.append("")
            lines.append("Top Methods with Issues:")
            sorted_methods = sorted(method_issues.items(), key=lambda x: x[1], reverse=True)
            for method, count in sorted_methods[:5]:
                lines.append(f"  {method}: {count}")
        
        # Issues by rule type
        if rule_type_issues:
            lines.append("")
            lines.append("By Rule Type:")
            sorted_rules = sorted(rule_type_issues.items(), key=lambda x: x[1], reverse=True)
            for rule_type, count in sorted_rules:
                lines.append(f"  {rule_type}: {count}")
        
        return lines
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        lines = []
        
        if self.summary.total_errors > 0:
            lines.append("• Address all ERROR-level issues first - these indicate critical problems")
        
        if self.summary.total_warnings > 0:
            lines.append("• Review WARNING-level issues - these may indicate potential problems")
        
        if self.summary.success_rate < 0.8:
            lines.append("• Success rate is below 80% - consider reviewing validation rules")
        
        # Method-specific recommendations
        method_error_counts = {}
        for result in self.summary.results:
            for issue in result.issues:
                if issue.level == IssueLevel.ERROR and issue.method_name:
                    method = issue.method_name
                    method_error_counts[method] = method_error_counts.get(method, 0) + 1
        
        if method_error_counts:
            top_method = max(method_error_counts.items(), key=lambda x: x[1])
            lines.append(f"• Focus on '{top_method[0]}' method - has {top_method[1]} errors")
        
        return lines
    
    def _generate_html_summary(self) -> str:
        """Generate HTML summary section."""
        return f"""
        <div class='summary-grid'>
            <div class='summary-item'>
                <div class='summary-number'>{self.summary.total_steps}</div>
                <div class='summary-label'>Total Steps</div>
            </div>
            <div class='summary-item success'>
                <div class='summary-number'>{self.summary.passed_steps}</div>
                <div class='summary-label'>Passed</div>
            </div>
            <div class='summary-item error'>
                <div class='summary-number'>{self.summary.failed_steps}</div>
                <div class='summary-label'>Failed</div>
            </div>
            <div class='summary-item'>
                <div class='summary-number'>{self.summary.success_rate:.1%}</div>
                <div class='summary-label'>Success Rate</div>
            </div>
        </div>
        """
    
    def _generate_html_results(self) -> str:
        """Generate HTML results section."""
        html_parts = []
        
        for result in self.summary.results:
            if not self._should_include_result(result):
                continue
            
            status_class = result.status.value.lower().replace('_', '-')
            html_parts.append(f"<div class='result-item {status_class}'>")
            html_parts.append(f"<h3>{result.step_name}</h3>")
            html_parts.append(f"<div class='status'>Status: {result.status.value}</div>")
            
            if result.issues:
                html_parts.append("<div class='issues'>")
                html_parts.append("<h4>Issues:</h4>")
                html_parts.append("<ul>")
                for issue in result.issues:
                    issue_class = issue.level.value.lower()
                    html_parts.append(f"<li class='{issue_class}'>{issue.message}</li>")
                html_parts.append("</ul>")
                html_parts.append("</div>")
            
            html_parts.append("</div>")
        
        return "\n".join(html_parts)
    
    def _get_html_styles(self) -> str:
        """Get CSS styles for HTML report."""
        return """
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        .timestamp { color: #666; font-style: italic; }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 20px; margin: 20px 0; }
        .summary-item { text-align: center; padding: 20px; border-radius: 8px; background: #f8f9fa; }
        .summary-item.success { background: #d4edda; }
        .summary-item.error { background: #f8d7da; }
        .summary-number { font-size: 2em; font-weight: bold; color: #333; }
        .summary-label { color: #666; margin-top: 5px; }
        .result-item { margin: 20px 0; padding: 15px; border-radius: 8px; border-left: 4px solid #ccc; }
        .result-item.passed { border-left-color: #28a745; background: #d4edda; }
        .result-item.failed { border-left-color: #dc3545; background: #f8d7da; }
        .result-item.excluded { border-left-color: #6c757d; background: #e2e3e5; }
        .status { font-weight: bold; margin-bottom: 10px; }
        .issues ul { margin: 10px 0; }
        .issues li.error { color: #dc3545; }
        .issues li.warning { color: #ffc107; }
        .issues li.info { color: #17a2b8; }
        """
    
    def _should_include_result(self, result: ValidationResult) -> bool:
        """Determine if a result should be included in the report."""
        if result.status == ValidationStatus.PASSED and not self.config.include_passed:
            return False
        if result.status == ValidationStatus.EXCLUDED and not self.config.include_excluded:
            return False
        return True
    
    def _colorize(self, text: str, *colors: str) -> str:
        """Apply color codes to text if color output is enabled."""
        if not self.config.color_output:
            return text
        
        color_codes = ''.join(self._color_codes.get(color, '') for color in colors)
        reset_code = self._color_codes['reset']
        return f"{color_codes}{text}{reset_code}"
    
    def _get_status_color(self, status: ValidationStatus) -> str:
        """Get color for validation status."""
        color_map = {
            ValidationStatus.PASSED: 'green',
            ValidationStatus.FAILED: 'red',
            ValidationStatus.EXCLUDED: 'yellow',
            ValidationStatus.ERROR: 'red',
            ValidationStatus.ISSUES_FOUND: 'red',
            ValidationStatus.PASSED_WITH_WARNINGS: 'yellow',
            ValidationStatus.NO_VALIDATOR: 'magenta'
        }
        return color_map.get(status, 'white')
    
    def _get_status_symbol(self, status: ValidationStatus) -> str:
        """Get symbol for validation status."""
        symbol_map = {
            ValidationStatus.PASSED: '✓',
            ValidationStatus.FAILED: '✗',
            ValidationStatus.EXCLUDED: '○',
            ValidationStatus.ERROR: '✗',
            ValidationStatus.ISSUES_FOUND: '✗',
            ValidationStatus.PASSED_WITH_WARNINGS: '⚠',
            ValidationStatus.NO_VALIDATOR: '?'
        }
        return symbol_map.get(status, '•')
    
    def _get_issue_color(self, level: IssueLevel) -> str:
        """Get color for issue level."""
        color_map = {
            IssueLevel.ERROR: 'red',
            IssueLevel.WARNING: 'yellow',
            IssueLevel.INFO: 'blue'
        }
        return color_map.get(level, 'white')
    
    def calculate_score(self) -> Dict[str, float]:
        """
        Calculate validation scores and metrics.
        
        Returns:
            Dictionary containing various scoring metrics
        """
        if self.summary.total_steps == 0:
            return {
                'overall_score': 0.0,
                'success_rate': 0.0,
                'error_rate': 0.0,
                'warning_rate': 0.0,
                'coverage_score': 0.0,
                'quality_score': 0.0
            }
        
        # Basic rates
        success_rate = self.summary.success_rate
        error_rate = self.summary.total_errors / max(self.summary.total_steps, 1)
        warning_rate = self.summary.total_warnings / max(self.summary.total_steps, 1)
        
        # Coverage score (non-excluded steps)
        non_excluded_steps = self.summary.total_steps - self.summary.excluded_steps
        coverage_score = non_excluded_steps / max(self.summary.total_steps, 1)
        
        # Quality score (weighted by issue severity)
        total_weighted_issues = (
            self.summary.total_errors * 3 +      # Errors weighted heavily
            self.summary.total_warnings * 1 +    # Warnings weighted normally
            self.summary.total_info * 0.1        # Info weighted lightly
        )
        max_possible_weighted_issues = self.summary.total_steps * 3  # All errors
        quality_score = max(0.0, 1.0 - (total_weighted_issues / max(max_possible_weighted_issues, 1)))
        
        # Overall score (combination of success rate and quality)
        overall_score = (success_rate * 0.6 + quality_score * 0.4) * coverage_score
        
        return {
            'overall_score': round(overall_score, 3),
            'success_rate': round(success_rate, 3),
            'error_rate': round(error_rate, 3),
            'warning_rate': round(warning_rate, 3),
            'coverage_score': round(coverage_score, 3),
            'quality_score': round(quality_score, 3)
        }
    
    def print_summary(self, file: TextIO = None):
        """Print a quick summary to stdout or specified file."""
        output = file or sys.stdout
        
        scores = self.calculate_score()
        
        print(f"Validation Summary:", file=output)
        print(f"  Steps: {self.summary.passed_steps}/{self.summary.total_steps} passed", file=output)
        print(f"  Success Rate: {scores['success_rate']:.1%}", file=output)
        print(f"  Overall Score: {scores['overall_score']:.1%}", file=output)
        print(f"  Issues: {self.summary.total_errors} errors, {self.summary.total_warnings} warnings", file=output)
    
    def export_to_json(self, output_file: Optional[Union[str, Path]] = None) -> str:
        """
        Export validation results to JSON format.
        
        Args:
            output_file: Optional file path to write the JSON to
            
        Returns:
            The JSON report as a string
        """
        # Temporarily set format to JSON
        original_format = self.config.output_format
        self.config.output_format = "json"
        
        try:
            json_report = self.generate_report(output_file)
            return json_report
        finally:
            # Restore original format
            self.config.output_format = original_format
    
    def export_to_html(self, output_file: Optional[Union[str, Path]] = None) -> str:
        """
        Export validation results to HTML format.
        
        Args:
            output_file: Optional file path to write the HTML to
            
        Returns:
            The HTML report as a string
        """
        # Temporarily set format to HTML
        original_format = self.config.output_format
        self.config.output_format = "html"
        
        try:
            html_report = self.generate_report(output_file)
            return html_report
        finally:
            # Restore original format
            self.config.output_format = original_format
    
    def generate_console_report(self, summary: ValidationSummary, detailed: bool = False) -> str:
        """
        Generate a console-friendly validation report.
        
        Args:
            summary: ValidationSummary to generate report from
            detailed: Whether to include detailed information
            
        Returns:
            The console report as a string
        """
        # Update our internal summary with the provided one
        self.summary = summary
        
        # Temporarily set format to text for console output and verbose mode
        original_format = self.config.output_format
        original_verbose = self.config.verbose
        self.config.output_format = "text"
        self.config.verbose = detailed
        
        try:
            console_report = self.generate_report()
            return console_report
        finally:
            # Restore original settings
            self.config.output_format = original_format
            self.config.verbose = original_verbose


# Convenience functions for quick reporting

def generate_quick_report(results: List[ValidationResult], 
                         output_file: Optional[str] = None,
                         format: str = "text") -> str:
    """Generate a quick validation report from results."""
    config = ReportingConfig(output_format=format)
    reporter = ValidationReporter(config)
    reporter.add_results(results)
    return reporter.generate_report(output_file)


def print_validation_summary(results: List[ValidationResult]):
    """Print a quick validation summary to stdout."""
    reporter = ValidationReporter()
    reporter.add_results(results)
    reporter.print_summary()


def calculate_validation_scores(results: List[ValidationResult]) -> Dict[str, float]:
    """Calculate validation scores from results."""
    reporter = ValidationReporter()
    reporter.add_results(results)
    return reporter.calculate_score()
