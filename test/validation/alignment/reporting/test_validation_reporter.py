"""
Test Validation Reporter Module

Tests for consolidated reporting functionality.
Tests multiple output formats, scoring, and metrics.
"""

import pytest
import tempfile
import os
import json
from unittest.mock import patch, MagicMock, Mock
from datetime import datetime

from cursus.validation.alignment.reporting.validation_reporter import ValidationReporter, ReportingConfig
from cursus.validation.alignment.utils.validation_models import (
    ValidationResult,
    ValidationIssue,
    ValidationSummary,
    ValidationLevel,
    ValidationStatus,
    IssueLevel
)


class TestValidationReporter:
    """Test ValidationReporter functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.reporter = ValidationReporter()
        
        # Create sample validation results
        self.sample_issues = [
            ValidationIssue(
                level=IssueLevel.ERROR,
                message="Test error message",
                details={"step": "test_step"}
            ),
            ValidationIssue(
                level=IssueLevel.WARNING,
                message="Test warning message",
                details={"step": "test_step"}
            )
        ]
        
        self.sample_results = [
            ValidationResult(
                step_name="step1",
                validation_level=ValidationLevel.SCRIPT_CONTRACT,
                status=ValidationStatus.PASSED,
                issues=[]
            ),
            ValidationResult(
                step_name="step2",
                validation_level=ValidationLevel.CONTRACT_SPEC,
                status=ValidationStatus.FAILED,
                issues=self.sample_issues
            )
        ]
        
        self.sample_summary = ValidationSummary()
        for result in self.sample_results:
            self.sample_summary.add_result(result)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_reporter_initialization(self):
        """Test ValidationReporter initialization."""
        reporter = ValidationReporter()
        assert hasattr(reporter, 'generate_report')
        assert hasattr(reporter, 'export_to_json')
        assert hasattr(reporter, 'export_to_html')
        assert hasattr(reporter, 'generate_console_report')

    def test_generate_console_report(self):
        """Test console report generation."""
        report = self.reporter.generate_console_report(self.sample_summary)
        
        assert isinstance(report, str)
        assert "VALIDATION ALIGNMENT REPORT" in report
        assert "Total Steps: 2" in report
        assert "Passed: 1" in report
        assert "Failed: 1" in report

    def test_generate_detailed_console_report(self):
        """Test detailed console report with issue breakdown."""
        report = self.reporter.generate_console_report(self.sample_summary, detailed=True)
        
        assert isinstance(report, str)
        assert "Issue Breakdown" in report
        assert "ERROR" in report
        assert "WARNING" in report
        assert "step2" in report  # Failed step should be mentioned
        assert "Test error message" in report

    def test_export_to_json(self):
        """Test JSON export functionality."""
        output_path = os.path.join(self.temp_dir, "validation_report.json")
        
        # Add sample results to the reporter
        self.reporter.add_results(self.sample_results)
        
        json_report = self.reporter.export_to_json(output_path)
        
        # Verify file was created
        assert os.path.exists(output_path)
        
        # Verify JSON content
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert "summary" in data
        assert "results" in data
        assert data["summary"]["total_steps"] == 2
        assert data["summary"]["passed_steps"] == 1
        assert data["summary"]["failed_steps"] == 1

    def test_export_to_html(self):
        """Test HTML export functionality."""
        output_path = os.path.join(self.temp_dir, "validation_report.html")
        
        # Add sample results to the reporter
        self.reporter.add_results(self.sample_results)
        
        html_report = self.reporter.export_to_html(output_path)
        
        # Verify file was created
        assert os.path.exists(output_path)
        
        # Verify HTML content
        with open(output_path, 'r') as f:
            content = f.read()
        
        assert "<html>" in content
        assert "<title>" in content
        assert "Validation Alignment Report" in content
        assert "step1" in content
        assert "step2" in content

    def test_calculate_score(self):
        """Test validation scoring system."""
        score = self.reporter.calculate_score()
        
        assert "overall_score" in score
        assert "success_rate" in score
        assert "error_rate" in score
        assert "warning_rate" in score
        assert "coverage_score" in score
        assert "quality_score" in score
        
        assert 0 <= score["overall_score"] <= 1
        assert 0 <= score["success_rate"] <= 1

    def test_print_summary(self):
        """Test summary printing functionality."""
        # Test that print_summary doesn't raise an exception
        try:
            self.reporter.print_summary()
        except Exception as e:
            pytest.fail(f"print_summary raised an exception: {e}")

    def test_add_result(self):
        """Test adding individual results."""
        reporter = ValidationReporter()
        result = ValidationResult(
            step_name="test_step",
            status=ValidationStatus.PASSED,
            issues=[]
        )
        
        reporter.add_result(result)
        assert len(reporter.summary.results) == 1
        assert reporter.summary.total_steps == 1

    def test_add_results(self):
        """Test adding multiple results."""
        reporter = ValidationReporter()
        reporter.add_results(self.sample_results)
        
        assert len(reporter.summary.results) == 2
        assert reporter.summary.total_steps == 2
        assert reporter.summary.passed_steps == 1
        assert reporter.summary.failed_steps == 1

    def test_generate_text_report(self):
        """Test text report generation."""
        self.reporter.add_results(self.sample_results)
        report = self.reporter.generate_report()
        
        assert isinstance(report, str)
        assert "VALIDATION ALIGNMENT REPORT" in report
        assert "SUMMARY" in report

    def test_generate_json_report(self):
        """Test JSON report generation."""
        config = ReportingConfig(output_format="json")
        reporter = ValidationReporter(config)
        reporter.add_results(self.sample_results)
        
        report = reporter.generate_report()
        
        # Should be valid JSON
        data = json.loads(report)
        assert "metadata" in data
        assert "summary" in data
        assert "results" in data

    def test_generate_html_report(self):
        """Test HTML report generation."""
        config = ReportingConfig(output_format="html")
        reporter = ValidationReporter(config)
        reporter.add_results(self.sample_results)
        
        report = reporter.generate_report()
        
        assert "<html>" in report
        assert "<title>" in report
        assert "Validation Alignment Report" in report

    def test_reporting_config(self):
        """Test reporting configuration."""
        config = ReportingConfig(
            include_passed=False,
            include_excluded=True,
            color_output=False,
            verbose=True
        )
        
        reporter = ValidationReporter(config)
        assert reporter.config.include_passed == False
        assert reporter.config.include_excluded == True
        assert reporter.config.color_output == False
        assert reporter.config.verbose == True

    def test_colorize_functionality(self):
        """Test color output functionality."""
        # Test with color enabled
        config = ReportingConfig(color_output=True)
        reporter = ValidationReporter(config)
        
        colored_text = reporter._colorize("test", "red", "bold")
        assert "\033[" in colored_text  # Should contain ANSI codes
        
        # Test with color disabled
        config = ReportingConfig(color_output=False)
        reporter = ValidationReporter(config)
        
        plain_text = reporter._colorize("test", "red", "bold")
        assert plain_text == "test"  # Should not contain ANSI codes

    def test_status_symbols_and_colors(self):
        """Test status symbol and color mapping."""
        symbols = {
            ValidationStatus.PASSED: '✓',
            ValidationStatus.FAILED: '✗',
            ValidationStatus.EXCLUDED: '○',
            ValidationStatus.ERROR: '✗'
        }
        
        for status, expected_symbol in symbols.items():
            symbol = self.reporter._get_status_symbol(status)
            assert symbol == expected_symbol

    def test_issue_colors(self):
        """Test issue level color mapping."""
        colors = {
            IssueLevel.ERROR: 'red',
            IssueLevel.WARNING: 'yellow',
            IssueLevel.INFO: 'blue'
        }
        
        for level, expected_color in colors.items():
            color = self.reporter._get_issue_color(level)
            assert color == expected_color

    def test_empty_summary_handling(self):
        """Test handling of empty validation summary."""
        empty_summary = ValidationSummary()
        
        report = self.reporter.generate_console_report(empty_summary)
        
        assert "Total Steps: 0" in report

    def test_report_filtering(self):
        """Test report filtering functionality."""
        config = ReportingConfig(include_passed=False)
        reporter = ValidationReporter(config)
        
        # Should filter out passed results
        passed_result = ValidationResult(
            step_name="passed_step",
            status=ValidationStatus.PASSED,
            issues=[]
        )
        
        should_include = reporter._should_include_result(passed_result)
        assert should_include == False

    def test_large_dataset_handling(self):
        """Test handling of large validation datasets."""
        # Create a large number of results
        large_results = []
        for i in range(100):  # Reduced from 1000 for faster testing
            result = ValidationResult(
                step_name=f"step_{i}",
                validation_level=ValidationLevel.SCRIPT_CONTRACT,
                status=ValidationStatus.PASSED if i % 2 == 0 else ValidationStatus.FAILED,
                issues=[]
            )
            large_results.append(result)
        
        large_summary = ValidationSummary()
        for result in large_results:
            large_summary.add_result(result)
        
        # Should handle large dataset without issues
        report = self.reporter.generate_console_report(large_summary)
        assert "Total Steps: 100" in report

    def test_issue_breakdown_generation(self):
        """Test issue breakdown generation."""
        self.reporter.add_results(self.sample_results)
        
        # Generate breakdown
        breakdown = self.reporter._generate_issue_breakdown()
        
        assert isinstance(breakdown, list)
        assert len(breakdown) > 0
        
        # Should contain severity information
        breakdown_text = "\n".join(breakdown)
        assert "By Severity:" in breakdown_text

    def test_recommendations_generation(self):
        """Test recommendations generation."""
        self.reporter.add_results(self.sample_results)
        
        # Generate recommendations
        recommendations = self.reporter._generate_recommendations()
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should contain actionable recommendations
        recommendations_text = "\n".join(recommendations)
        assert "ERROR-level issues" in recommendations_text or "WARNING-level issues" in recommendations_text

    def test_report_error_handling(self):
        """Test error handling in report generation."""
        # Test with invalid output path
        invalid_path = "/invalid/path/report.json"
        
        with pytest.raises(Exception):
            self.reporter.export_to_json(invalid_path)

    def test_html_styles_generation(self):
        """Test HTML styles generation."""
        styles = self.reporter._get_html_styles()
        
        assert isinstance(styles, str)
        assert "body" in styles
        assert "font-family" in styles
        assert ".container" in styles

    def test_html_summary_generation(self):
        """Test HTML summary section generation."""
        self.reporter.add_results(self.sample_results)
        
        html_summary = self.reporter._generate_html_summary()
        
        assert isinstance(html_summary, str)
        assert "summary-grid" in html_summary
        assert "Total Steps" in html_summary

    def test_html_results_generation(self):
        """Test HTML results section generation."""
        self.reporter.add_results(self.sample_results)
        
        html_results = self.reporter._generate_html_results()
        
        assert isinstance(html_results, str)
        assert "result-item" in html_results
        assert "step1" in html_results
        assert "step2" in html_results
