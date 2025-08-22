"""
Unit tests for RegistryValidator model classes.

Tests the validation issue models and enums used by the registry validator.
"""

import pytest
from src.cursus.pipeline_catalog.utils.registry_validator import (
    ValidationSeverity, ValidationIssue, AtomicityViolation,
    ConnectionError, MetadataError, TagConsistencyError, IndependenceError,
    ValidationReport
)


class TestValidationSeverity:
    """Test suite for ValidationSeverity enum."""
    
    def test_validation_severity_values(self):
        """Test ValidationSeverity enum values."""
        assert ValidationSeverity.ERROR.value == "error"
        assert ValidationSeverity.WARNING.value == "warning"
        assert ValidationSeverity.INFO.value == "info"


class TestValidationIssue:
    """Test suite for ValidationIssue model."""
    
    def test_validation_issue_creation(self):
        """Test creating ValidationIssue instance."""
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            category="test_category",
            pipeline_id="test_pipeline",
            message="Test validation issue",
            suggested_fix="Fix the issue"
        )
        
        assert issue.severity == ValidationSeverity.ERROR
        assert issue.category == "test_category"
        assert issue.pipeline_id == "test_pipeline"
        assert issue.message == "Test validation issue"
        assert issue.suggested_fix == "Fix the issue"
    
    def test_validation_issue_minimal(self):
        """Test creating ValidationIssue with minimal fields."""
        issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            category="minimal",
            pipeline_id=None,
            message="Minimal issue"
        )
        
        assert issue.severity == ValidationSeverity.WARNING
        assert issue.category == "minimal"
        assert issue.pipeline_id is None
        assert issue.message == "Minimal issue"
        assert issue.suggested_fix is None


class TestSpecializedValidationIssues:
    """Test suite for specialized validation issue classes."""
    
    def test_atomicity_violation(self):
        """Test AtomicityViolation creation."""
        violation = AtomicityViolation(
            pipeline_id="test_pipeline",
            violation_type="missing_responsibility",
            description="No single responsibility defined",
            suggested_fix="Define clear responsibility"
        )
        
        assert violation.severity == ValidationSeverity.ERROR
        assert violation.category == "atomicity"
        assert violation.pipeline_id == "test_pipeline"
        assert "missing_responsibility" in violation.message
        assert "No single responsibility defined" in violation.message
        assert violation.suggested_fix == "Define clear responsibility"
    
    def test_connection_error(self):
        """Test ConnectionError creation."""
        error = ConnectionError(
            source_id="source_pipeline",
            target_id="target_pipeline",
            error_type="missing_target",
            description="Target pipeline does not exist"
        )
        
        assert error.severity == ValidationSeverity.ERROR
        assert error.category == "connectivity"
        assert error.pipeline_id == "source_pipeline"
        assert "missing_target" in error.message
        assert "source_pipeline -> target_pipeline" in error.message
        assert "Target pipeline does not exist" in error.message
    
    def test_metadata_error(self):
        """Test MetadataError creation."""
        error = MetadataError(
            pipeline_id="test_pipeline",
            field="description",
            description="Missing description field",
            suggested_fix="Add description"
        )
        
        assert error.severity == ValidationSeverity.WARNING
        assert error.category == "metadata"
        assert error.pipeline_id == "test_pipeline"
        assert "description" in error.message
        assert "Missing description field" in error.message
        assert error.suggested_fix == "Add description"
    
    def test_tag_consistency_error(self):
        """Test TagConsistencyError creation."""
        error = TagConsistencyError(
            pipeline_id="test_pipeline",
            tag_category="framework_tags",
            description="Framework not in framework_tags",
            suggested_fix="Add framework to tags"
        )
        
        assert error.severity == ValidationSeverity.WARNING
        assert error.category == "tags"
        assert error.pipeline_id == "test_pipeline"
        assert "framework_tags" in error.message
        assert "Framework not in framework_tags" in error.message
        assert error.suggested_fix == "Add framework to tags"
    
    def test_independence_error(self):
        """Test IndependenceError creation."""
        error = IndependenceError(
            pipeline_id="test_pipeline",
            claim="fully_self_contained",
            evidence="has many dependencies",
            suggested_fix="Reduce dependencies"
        )
        
        assert error.severity == ValidationSeverity.WARNING
        assert error.category == "independence"
        assert error.pipeline_id == "test_pipeline"
        assert "fully_self_contained" in error.message
        assert "has many dependencies" in error.message
        assert error.suggested_fix == "Reduce dependencies"


class TestValidationReport:
    """Test suite for ValidationReport model."""
    
    def test_validation_report_creation(self):
        """Test creating ValidationReport instance."""
        issues = [
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="test",
                pipeline_id="test_pipeline",
                message="Test error"
            ),
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="test",
                pipeline_id="test_pipeline",
                message="Test warning"
            )
        ]
        
        report = ValidationReport(
            is_valid=False,
            total_issues=2,
            issues_by_severity={ValidationSeverity.ERROR: 1, ValidationSeverity.WARNING: 1},
            issues_by_category={"test": 2},
            all_issues=issues
        )
        
        assert report.is_valid is False
        assert report.total_issues == 2
        assert report.issues_by_severity[ValidationSeverity.ERROR] == 1
        assert report.issues_by_severity[ValidationSeverity.WARNING] == 1
        assert report.issues_by_category["test"] == 2
        assert len(report.all_issues) == 2
    
    def test_validation_report_summary_valid(self):
        """Test validation report summary for valid registry."""
        report = ValidationReport(
            is_valid=True,
            total_issues=0,
            issues_by_severity={},
            issues_by_category={},
            all_issues=[]
        )
        
        summary = report.summary()
        assert "validation passed" in summary.lower()
        assert "no critical issues" in summary.lower()
    
    def test_validation_report_summary_invalid(self):
        """Test validation report summary for invalid registry."""
        report = ValidationReport(
            is_valid=False,
            total_issues=3,
            issues_by_severity={
                ValidationSeverity.ERROR: 2,
                ValidationSeverity.WARNING: 1
            },
            issues_by_category={"test": 3},
            all_issues=[]
        )
        
        summary = report.summary()
        assert "2 errors" in summary
        assert "1 warnings" in summary
