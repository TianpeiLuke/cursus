"""
Test suite for AlignmentReport class.
"""

import pytest
import json

from cursus.validation.alignment.alignment_reporter import (
    AlignmentReport, ValidationResult, AlignmentSummary
)
from cursus.validation.alignment.alignment_utils import (
    AlignmentIssue, SeverityLevel, AlignmentLevel
)


@pytest.fixture
def sample_issue():
    """Set up sample issue fixture."""
    return AlignmentIssue(
        level=SeverityLevel.ERROR,
        category="test_category",
        message="Test issue",
        details={"key": "value"}
    )


@pytest.fixture
def sample_result(sample_issue):
    """Set up sample validation result fixture."""
    result = ValidationResult(
        test_name="test_validation",
        passed=False
    )
    result.add_issue(sample_issue)
    return result


class TestAlignmentReport:
    """Test AlignmentReport class."""
    
    def test_alignment_report_creation(self):
        """Test basic AlignmentReport creation."""
        report = AlignmentReport()
        
        assert len(report.level1_results) == 0
        assert len(report.level2_results) == 0
        assert len(report.level3_results) == 0
        assert len(report.level4_results) == 0
        assert report.summary is None
        assert len(report.recommendations) == 0
        assert report.metadata == {}
    
    def test_add_result(self, sample_result):
        """Test adding results to different levels."""
        report = AlignmentReport()
        
        # Add results to each level
        report.add_level1_result("script_test", sample_result)
        report.add_level2_result("contract_test", sample_result)
        report.add_level3_result("spec_test", sample_result)
        report.add_level4_result("builder_test", sample_result)
        
        assert len(report.level1_results) == 1
        assert len(report.level2_results) == 1
        assert len(report.level3_results) == 1
        assert len(report.level4_results) == 1
        
        assert "script_test" in report.level1_results
        assert "contract_test" in report.level2_results
        assert "spec_test" in report.level3_results
        assert "builder_test" in report.level4_results
    
    def test_get_all_results(self, sample_result):
        """Test getting all results across levels."""
        report = AlignmentReport()
        
        report.add_level1_result("test1", sample_result)
        report.add_level2_result("test2", sample_result)
        
        all_results = report.get_all_results()
        
        assert len(all_results) == 2
        assert "test1" in all_results
        assert "test2" in all_results
    
    def test_get_summary(self, sample_result):
        """Test generating summary."""
        report = AlignmentReport()
        report.add_level1_result("test1", sample_result)
        
        summary = report.generate_summary()
        
        assert summary is not None
        assert summary.total_tests == 1
        assert summary.failed_tests == 1
        assert summary.passed_tests == 0
        assert summary.total_issues == 1
        assert summary.error_issues == 1
    
    def test_get_issues_by_level(self, sample_result):
        """Test getting issues by severity level."""
        report = AlignmentReport()
        
        # Add critical issue
        critical_issue = AlignmentIssue(
            level=SeverityLevel.CRITICAL,
            category="critical_test",
            message="Critical issue"
        )
        critical_result = ValidationResult(test_name="critical_test", passed=False)
        critical_result.add_issue(critical_issue)
        
        report.add_level1_result("critical_test", critical_result)
        report.add_level2_result("error_test", sample_result)
        
        critical_issues = report.get_critical_issues()
        error_issues = report.get_error_issues()
        
        assert len(critical_issues) == 1
        assert len(error_issues) == 1
        assert critical_issues[0].level == SeverityLevel.CRITICAL
        assert error_issues[0].level == SeverityLevel.ERROR
    
    def test_has_critical_issues(self):
        """Test checking for critical issues."""
        report = AlignmentReport()
        
        # No critical issues initially
        assert report.has_critical_issues() is False
        
        # Add critical issue
        critical_issue = AlignmentIssue(
            level=SeverityLevel.CRITICAL,
            category="critical_test",
            message="Critical issue"
        )
        critical_result = ValidationResult(test_name="critical_test", passed=False)
        critical_result.add_issue(critical_issue)
        report.add_level1_result("critical_test", critical_result)
        
        assert report.has_critical_issues() is True
    
    def test_export_to_json(self, sample_result):
        """Test exporting report to JSON."""
        report = AlignmentReport()
        report.add_level1_result("test1", sample_result)
        
        json_output = report.export_to_json()
        
        assert isinstance(json_output, str)
        
        # Parse JSON to verify structure
        data = json.loads(json_output)
        assert 'summary' in data
        assert 'level1_results' in data
        assert 'recommendations' in data
        assert 'metadata' in data
    
    def test_export_to_html(self, sample_result):
        """Test exporting report to HTML."""
        report = AlignmentReport()
        report.add_level1_result("test1", sample_result)
        
        html_output = report.export_to_html()
        
        assert isinstance(html_output, str)
        assert '<html>' in html_output
        assert 'Alignment Validation Report' in html_output
        assert 'test1' in html_output


if __name__ == '__main__':
    pytest.main([__file__])
