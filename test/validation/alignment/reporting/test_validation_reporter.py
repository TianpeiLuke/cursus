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

from cursus.validation.alignment.reporting.validation_reporter import ValidationReporter
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

    def test_generate_console_report(self):
        """Test console report generation."""
        report = self.reporter.generate_console_report(self.sample_summary)
        
        assert isinstance(report, str)
        assert "Validation Summary" in report
        assert "Total Steps: 2" in report
        assert "Passed: 1" in report
        assert "Failed: 1" in report
        assert "Total Issues: 2" in report

    def test_generate_detailed_console_report(self):
        """Test detailed console report with issue breakdown."""
        report = self.reporter.generate_console_report(self.sample_summary, detailed=True)
        
        assert isinstance(report, str)
        assert "Issue Breakdown" in report
        assert "ERROR: 1" in report
        assert "WARNING: 1" in report
        assert "step2" in report  # Failed step should be mentioned
        assert "Test error message" in report

    def test_export_to_json(self):
        """Test JSON export functionality."""
        output_path = os.path.join(self.temp_dir, "validation_report.json")
        
        self.reporter.export_to_json(self.sample_summary, output_path)
        
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
        assert len(data["results"]) == 2

    def test_export_to_html(self):
        """Test HTML export functionality."""
        output_path = os.path.join(self.temp_dir, "validation_report.html")
        
        self.reporter.export_to_html(self.sample_summary, output_path)
        
        # Verify file was created
        assert os.path.exists(output_path)
        
        # Verify HTML content
        with open(output_path, 'r') as f:
            content = f.read()
        
        assert "<html>" in content
        assert "<title>" in content
        assert "Validation Report" in content
        assert "Total Steps: 2" in content
        assert "step1" in content
        assert "step2" in content

    def test_export_to_csv(self):
        """Test CSV export functionality."""
        output_path = os.path.join(self.temp_dir, "validation_report.csv")
        
        self.reporter.export_to_csv(self.sample_summary, output_path)
        
        # Verify file was created
        assert os.path.exists(output_path)
        
        # Verify CSV content
        with open(output_path, 'r') as f:
            content = f.read()
        
        assert "step_name,validation_level,status,issue_count" in content
        assert "step1,SCRIPT_CONTRACT,PASSED,0" in content
        assert "step2,CONTRACT_SPEC,FAILED,2" in content

    def test_generate_summary_statistics(self):
        """Test summary statistics generation."""
        stats = self.reporter.generate_summary_statistics(self.sample_summary)
        
        assert "total_steps" in stats
        assert "success_rate" in stats
        assert "issue_breakdown" in stats
        assert "execution_time" in stats
        
        assert stats["total_steps"] == 2
        assert stats["success_rate"] == 0.5  # 1/2
        assert stats["issue_breakdown"]["ERROR"] == 1
        assert stats["issue_breakdown"]["WARNING"] == 1
        assert stats["execution_time"]["total"] == 3.7  # 1.2 + 2.5

    def test_generate_issue_report(self):
        """Test issue-focused report generation."""
        issue_report = self.reporter.generate_issue_report(self.sample_summary)
        
        assert "critical_issues" in issue_report
        assert "error_issues" in issue_report
        assert "warning_issues" in issue_report
        assert "info_issues" in issue_report
        
        assert len(issue_report["error_issues"]) == 1
        assert len(issue_report["warning_issues"]) == 1
        assert issue_report["error_issues"][0]["message"] == "Test error message"

    def test_generate_step_breakdown_report(self):
        """Test step-by-step breakdown report."""
        breakdown = self.reporter.generate_step_breakdown_report(self.sample_summary)
        
        assert "steps" in breakdown
        assert "level_statistics" in breakdown
        
        assert len(breakdown["steps"]) == 2
        assert breakdown["steps"][0]["step_name"] == "step1"
        assert breakdown["steps"][1]["step_name"] == "step2"
        
        # Check level statistics
        level_stats = breakdown["level_statistics"]
        assert ValidationLevel.SCRIPT_CONTRACT.name in level_stats
        assert ValidationLevel.CONTRACT_SPEC.name in level_stats

    def test_generate_performance_report(self):
        """Test performance-focused report generation."""
        perf_report = self.reporter.generate_performance_report(self.sample_summary)
        
        assert "total_execution_time" in perf_report
        assert "average_execution_time" in perf_report
        assert "slowest_steps" in perf_report
        assert "fastest_steps" in perf_report
        
        assert perf_report["total_execution_time"] == 3.7
        assert perf_report["average_execution_time"] == 1.85
        assert len(perf_report["slowest_steps"]) <= 5  # Top 5 slowest
        assert len(perf_report["fastest_steps"]) <= 5  # Top 5 fastest

    def test_generate_validation_score(self):
        """Test validation scoring system."""
        score = self.reporter.generate_validation_score(self.sample_summary)
        
        assert "overall_score" in score
        assert "component_scores" in score
        assert "score_breakdown" in score
        
        assert 0 <= score["overall_score"] <= 100
        assert "success_rate" in score["component_scores"]
        assert "issue_severity" in score["component_scores"]

    def test_export_multiple_formats(self):
        """Test exporting to multiple formats simultaneously."""
        base_path = os.path.join(self.temp_dir, "multi_format_report")
        
        formats = ["json", "html", "csv"]
        self.reporter.export_multiple_formats(self.sample_summary, base_path, formats)
        
        # Verify all format files were created
        for fmt in formats:
            file_path = f"{base_path}.{fmt}"
            assert os.path.exists(file_path)

    def test_generate_comparison_report(self):
        """Test comparison report between multiple validation runs."""
        # Create a second summary for comparison
        comparison_results = [
            ValidationResult(
                step_name="step1",
                validation_level=ValidationLevel.SCRIPT_CONTRACT,
                status=ValidationStatus.PASSED,
                issues=[]
            ),
            ValidationResult(
                step_name="step2",
                validation_level=ValidationLevel.CONTRACT_SPEC,
                status=ValidationStatus.PASSED,  # Improved from FAILED
                issues=[]  # Fixed issues
            )
        ]
        
        comparison_summary = ValidationSummary()
        for result in comparison_results:
            comparison_summary.add_result(result)
        
        comparison = self.reporter.generate_comparison_report(
            baseline=self.sample_summary,
            current=comparison_summary
        )
        
        assert "baseline" in comparison
        assert "current" in comparison
        assert "improvements" in comparison
        assert "regressions" in comparison
        
        # Should show improvements
        assert comparison["improvements"]["failed_steps"] == 1  # 1 -> 0
        assert comparison["improvements"]["total_issues"] == 2  # 2 -> 0

    def test_generate_trend_report(self):
        """Test trend analysis across multiple validation runs."""
        # Create historical data
        # Create second summary for historical data
        second_summary = ValidationSummary()
        for i in range(2):
            result = ValidationResult(
                step_name=f"step{i+1}",
                status=ValidationStatus.PASSED,
                issues=[]
            )
            second_summary.add_result(result)
        
        historical_summaries = [
            self.sample_summary,
            second_summary
        ]
        
        trend_report = self.reporter.generate_trend_report(historical_summaries)
        
        assert "trend_data" in trend_report
        assert "trend_analysis" in trend_report
        
        trend_data = trend_report["trend_data"]
        assert len(trend_data) == 2
        assert trend_data[0]["passed_steps"] == 1
        assert trend_data[1]["passed_steps"] == 2

    def test_custom_report_template(self):
        """Test custom report template functionality."""
        template = {
            "title": "Custom Validation Report",
            "sections": ["summary", "issues", "performance"],
            "format": "detailed"
        }
        
        custom_report = self.reporter.generate_custom_report(
            self.sample_summary,
            template
        )
        
        assert "title" in custom_report
        assert custom_report["title"] == "Custom Validation Report"
        assert "sections" in custom_report
        assert len(custom_report["sections"]) == 3

    def test_report_filtering(self):
        """Test report filtering by various criteria."""
        # Filter by issue level
        error_only_report = self.reporter.generate_filtered_report(
            self.sample_summary,
            issue_level_filter=[IssueLevel.ERROR]
        )
        
        assert "filtered_results" in error_only_report
        # Should only include results with errors
        filtered_results = error_only_report["filtered_results"]
        assert len(filtered_results) == 1
        assert filtered_results[0]["step_name"] == "step2"

    def test_report_aggregation(self):
        """Test report aggregation across multiple workspaces."""
        # Create multiple summaries representing different workspaces
        # Create second workspace summary
        workspace2_summary = ValidationSummary()
        for i in range(3):
            status = ValidationStatus.PASSED if i < 2 else ValidationStatus.FAILED
            issues = [ValidationIssue(level=IssueLevel.ERROR, message="Error")] if i >= 2 else []
            result = ValidationResult(
                step_name=f"ws2_step{i+1}",
                status=status,
                issues=issues
            )
            workspace2_summary.add_result(result)
        
        workspace_summaries = {
            "workspace1": self.sample_summary,
            "workspace2": workspace2_summary
        }
        
        aggregated_report = self.reporter.generate_aggregated_report(workspace_summaries)
        
        assert "total_workspaces" in aggregated_report
        assert "combined_summary" in aggregated_report
        assert "workspace_breakdown" in aggregated_report
        
        combined = aggregated_report["combined_summary"]
        assert combined["total_steps"] == 5  # 2 + 3
        assert combined["passed_steps"] == 3  # 1 + 2
        assert combined["failed_steps"] == 2  # 1 + 1

    def test_report_configuration(self):
        """Test report configuration and customization."""
        config = {
            "include_execution_times": True,
            "include_recommendations": True,
            "max_issues_per_step": 10,
            "sort_by": "severity",
            "theme": "dark"
        }
        
        configured_reporter = ValidationReporter(config)
        report = configured_reporter.generate_console_report(self.sample_summary)
        
        # Should include execution times based on config
        assert "Execution Time:" in report

    def test_report_error_handling(self):
        """Test error handling in report generation."""
        # Test with invalid output path
        invalid_path = "/invalid/path/report.json"
        
        with pytest.raises(Exception):
            self.reporter.export_to_json(self.sample_summary, invalid_path)

    def test_report_caching(self):
        """Test report caching functionality."""
        # Generate report twice - second should be faster due to caching
        start_time = datetime.now()
        report1 = self.reporter.generate_console_report(self.sample_summary)
        first_duration = (datetime.now() - start_time).total_seconds()
        
        start_time = datetime.now()
        report2 = self.reporter.generate_console_report(self.sample_summary)
        second_duration = (datetime.now() - start_time).total_seconds()
        
        # Reports should be identical
        assert report1 == report2
        
        # Second generation might be faster due to caching (if implemented)
        # This is optional and depends on implementation

    def test_empty_summary_handling(self):
        """Test handling of empty validation summary."""
        empty_summary = ValidationSummary()
        
        report = self.reporter.generate_console_report(empty_summary)
        
        assert "No validation results" in report or "Total Steps: 0" in report

    def test_large_dataset_handling(self):
        """Test handling of large validation datasets."""
        # Create a large number of results
        large_results = []
        for i in range(1000):
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
        assert "Total Steps: 1000" in report
        
        # Performance report should handle large dataset
        perf_report = self.reporter.generate_performance_report(large_summary)
        assert perf_report["total_execution_time"] == 100.0  # 1000 * 0.1
