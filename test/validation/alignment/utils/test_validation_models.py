"""
Test Validation Models Module

Tests for consolidated data models, enums, and utility functions.
Tests ValidationResult, ValidationStatus, IssueLevel, ValidationSummary, etc.
"""

import pytest
from datetime import datetime
from enum import Enum

from cursus.validation.alignment.utils.validation_models import (
    ValidationLevel,
    ValidationStatus,
    IssueLevel,
    ValidationResult,
    ValidationIssue,
    ValidationSummary
)


class TestValidationEnums:
    """Test validation enum classes."""

    def test_validation_level_enum(self):
        """Test ValidationLevel enum values."""
        assert ValidationLevel.SCRIPT_CONTRACT.value == 1
        assert ValidationLevel.CONTRACT_SPEC.value == 2
        assert ValidationLevel.SPEC_DEPENDENCY.value == 3
        assert ValidationLevel.BUILDER_CONFIG.value == 4
        
        # Test enum ordering
        levels = list(ValidationLevel)
        assert len(levels) == 4
        assert levels[0] == ValidationLevel.SCRIPT_CONTRACT
        assert levels[3] == ValidationLevel.BUILDER_CONFIG

    def test_validation_status_enum(self):
        """Test ValidationStatus enum values."""
        assert ValidationStatus.PASSED in ValidationStatus
        assert ValidationStatus.FAILED in ValidationStatus
        assert ValidationStatus.EXCLUDED in ValidationStatus
        assert ValidationStatus.SKIPPED in ValidationStatus
        
        # Test string representation
        assert str(ValidationStatus.PASSED) == "ValidationStatus.PASSED"

    def test_issue_level_enum(self):
        """Test IssueLevel enum values and ordering."""
        assert IssueLevel.INFO in IssueLevel
        assert IssueLevel.WARNING in IssueLevel
        assert IssueLevel.ERROR in IssueLevel
        assert IssueLevel.CRITICAL in IssueLevel
        
        # Test severity ordering (if implemented)
        levels = list(IssueLevel)
        assert len(levels) == 4

    # StepTypeCategory is defined in config module, not validation_models
    # This test is moved to config tests


class TestValidationIssue:
    """Test ValidationIssue data model."""

    def test_validation_issue_creation(self):
        """Test ValidationIssue creation with required fields."""
        issue = ValidationIssue(
            level=IssueLevel.ERROR,
            category="test_category",
            message="Test error message",
            details={"key": "value"},
            recommendation="Fix the issue"
        )
        
        assert issue.level == IssueLevel.ERROR
        assert issue.category == "test_category"
        assert issue.message == "Test error message"
        assert issue.details == {"key": "value"}
        assert issue.recommendation == "Fix the issue"

    def test_validation_issue_optional_fields(self):
        """Test ValidationIssue with optional fields."""
        issue = ValidationIssue(
            level=IssueLevel.WARNING,
            category="test_category",
            message="Test warning",
            file_path="/path/to/file.py",
            line_number=42
        )
        
        assert issue.file_path == "/path/to/file.py"
        assert issue.line_number == 42
        assert issue.details is None  # Optional field
        assert issue.recommendation is None  # Optional field

    def test_validation_issue_to_dict(self):
        """Test ValidationIssue conversion to dictionary."""
        issue = ValidationIssue(
            level=IssueLevel.ERROR,
            category="test_category",
            message="Test message",
            details={"test": "data"},
            recommendation="Test recommendation"
        )
        
        issue_dict = issue.to_dict()
        
        assert issue_dict["level"] == "ERROR"
        assert issue_dict["category"] == "test_category"
        assert issue_dict["message"] == "Test message"
        assert issue_dict["details"] == {"test": "data"}
        assert issue_dict["recommendation"] == "Test recommendation"

    def test_validation_issue_from_dict(self):
        """Test ValidationIssue creation from dictionary."""
        issue_data = {
            "level": "WARNING",
            "category": "test_category",
            "message": "Test message",
            "details": {"key": "value"},
            "recommendation": "Test recommendation"
        }
        
        issue = ValidationIssue.from_dict(issue_data)
        
        assert issue.level == IssueLevel.WARNING
        assert issue.category == "test_category"
        assert issue.message == "Test message"
        assert issue.details == {"key": "value"}
        assert issue.recommendation == "Test recommendation"


class TestValidationResult:
    """Test ValidationResult data model."""

    def test_validation_result_creation(self):
        """Test ValidationResult creation."""
        issues = [
            ValidationIssue(
                level=IssueLevel.ERROR,
                category="test",
                message="Test error"
            )
        ]
        
        result = ValidationResult(
            step_name="test_step",
            validation_level=ValidationLevel.SCRIPT_CONTRACT,
            status=ValidationStatus.FAILED,
            issues=issues,
            execution_time=1.5
        )
        
        assert result.step_name == "test_step"
        assert result.validation_level == ValidationLevel.SCRIPT_CONTRACT
        assert result.status == ValidationStatus.FAILED
        assert len(result.issues) == 1
        assert result.execution_time == 1.5

    def test_validation_result_passed_status(self):
        """Test ValidationResult with passed status."""
        result = ValidationResult(
            step_name="test_step",
            validation_level=ValidationLevel.BUILDER_CONFIG,
            status=ValidationStatus.PASSED,
            issues=[],
            metadata={"validator": "ProcessingStepBuilderValidator"}
        )
        
        assert result.status == ValidationStatus.PASSED
        assert len(result.issues) == 0
        assert result.metadata["validator"] == "ProcessingStepBuilderValidator"

    def test_validation_result_has_errors(self):
        """Test ValidationResult error detection."""
        error_issue = ValidationIssue(
            level=IssueLevel.ERROR,
            category="test",
            message="Error"
        )
        warning_issue = ValidationIssue(
            level=IssueLevel.WARNING,
            category="test",
            message="Warning"
        )
        
        # Result with errors
        result_with_errors = ValidationResult(
            step_name="test_step",
            validation_level=ValidationLevel.SCRIPT_CONTRACT,
            status=ValidationStatus.FAILED,
            issues=[error_issue, warning_issue]
        )
        
        assert result_with_errors.has_errors() is True
        assert result_with_errors.has_warnings() is True
        
        # Result with only warnings
        result_with_warnings = ValidationResult(
            step_name="test_step",
            validation_level=ValidationLevel.SCRIPT_CONTRACT,
            status=ValidationStatus.PASSED,
            issues=[warning_issue]
        )
        
        assert result_with_warnings.has_errors() is False
        assert result_with_warnings.has_warnings() is True

    def test_validation_result_get_issues_by_level(self):
        """Test filtering issues by level."""
        error_issue = ValidationIssue(level=IssueLevel.ERROR, category="test", message="Error")
        warning_issue = ValidationIssue(level=IssueLevel.WARNING, category="test", message="Warning")
        info_issue = ValidationIssue(level=IssueLevel.INFO, category="test", message="Info")
        
        result = ValidationResult(
            step_name="test_step",
            validation_level=ValidationLevel.SCRIPT_CONTRACT,
            status=ValidationStatus.FAILED,
            issues=[error_issue, warning_issue, info_issue]
        )
        
        errors = result.get_issues_by_level(IssueLevel.ERROR)
        warnings = result.get_issues_by_level(IssueLevel.WARNING)
        infos = result.get_issues_by_level(IssueLevel.INFO)
        
        assert len(errors) == 1
        assert len(warnings) == 1
        assert len(infos) == 1
        assert errors[0].message == "Error"
        assert warnings[0].message == "Warning"
        assert infos[0].message == "Info"


class TestValidationSummary:
    """Test ValidationSummary data model."""

    def test_validation_summary_creation(self):
        """Test ValidationSummary creation."""
        results = [
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
                issues=[ValidationIssue(level=IssueLevel.ERROR, category="test", message="Error")]
            )
        ]
        
        summary = ValidationSummary(
            total_steps=2,
            passed_steps=1,
            failed_steps=1,
            excluded_steps=0,
            total_issues=1,
            results=results
        )
        
        assert summary.total_steps == 2
        assert summary.passed_steps == 1
        assert summary.failed_steps == 1
        assert summary.excluded_steps == 0
        assert summary.total_issues == 1
        assert len(summary.results) == 2

    def test_validation_summary_success_rate(self):
        """Test ValidationSummary success rate calculation."""
        summary = ValidationSummary(
            total_steps=10,
            passed_steps=8,
            failed_steps=2,
            excluded_steps=0,
            total_issues=5,
            results=[]
        )
        
        success_rate = summary.get_success_rate()
        assert success_rate == 0.8  # 8/10

    def test_validation_summary_issue_breakdown(self):
        """Test ValidationSummary issue breakdown."""
        error_issue = ValidationIssue(level=IssueLevel.ERROR, category="test", message="Error")
        warning_issue = ValidationIssue(level=IssueLevel.WARNING, category="test", message="Warning")
        
        results = [
            ValidationResult(
                step_name="step1",
                validation_level=ValidationLevel.SCRIPT_CONTRACT,
                status=ValidationStatus.FAILED,
                issues=[error_issue, warning_issue]
            )
        ]
        
        summary = ValidationSummary(
            total_steps=1,
            passed_steps=0,
            failed_steps=1,
            excluded_steps=0,
            total_issues=2,
            results=results
        )
        
        breakdown = summary.get_issue_breakdown()
        
        assert breakdown[IssueLevel.ERROR] == 1
        assert breakdown[IssueLevel.WARNING] == 1
        assert breakdown.get(IssueLevel.INFO, 0) == 0
        assert breakdown.get(IssueLevel.CRITICAL, 0) == 0


# ValidationRuleset is defined in config module, not validation_models
# These tests are moved to config tests


class TestUtilityFunctions:
    """Test utility functions in validation models."""

    def test_create_validation_issue_helper(self):
        """Test helper function for creating validation issues."""
        # This would test any utility functions for creating issues
        issue = ValidationIssue(
            level=IssueLevel.ERROR,
            category="helper_test",
            message="Helper created issue"
        )
        
        assert issue.level == IssueLevel.ERROR
        assert issue.category == "helper_test"

    def test_validation_result_factory(self):
        """Test factory methods for creating validation results."""
        # Test creating a passed result
        passed_result = ValidationResult.create_passed_result(
            step_name="test_step",
            validation_level=ValidationLevel.SCRIPT_CONTRACT
        )
        
        assert passed_result.status == ValidationStatus.PASSED
        assert len(passed_result.issues) == 0
        
        # Test creating a failed result
        failed_result = ValidationResult.create_failed_result(
            step_name="test_step",
            validation_level=ValidationLevel.SCRIPT_CONTRACT,
            issues=[ValidationIssue(level=IssueLevel.ERROR, category="test", message="Error")]
        )
        
        assert failed_result.status == ValidationStatus.FAILED
        assert len(failed_result.issues) == 1

    def test_validation_models_serialization(self):
        """Test serialization of validation models."""
        issue = ValidationIssue(
            level=IssueLevel.WARNING,
            category="serialization_test",
            message="Test serialization"
        )
        
        result = ValidationResult(
            step_name="test_step",
            validation_level=ValidationLevel.BUILDER_CONFIG,
            status=ValidationStatus.PASSED,
            issues=[issue]
        )
        
        # Test to_dict methods exist and work
        issue_dict = issue.to_dict()
        result_dict = result.to_dict()
        
        assert isinstance(issue_dict, dict)
        assert isinstance(result_dict, dict)
        assert issue_dict["level"] == "WARNING"
        assert result_dict["status"] == "PASSED"
