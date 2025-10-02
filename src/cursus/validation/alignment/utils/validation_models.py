"""
Validation Models

This module provides consolidated data models and enums for the validation alignment system.
Replaces the previous script_analysis_models.py and core_models.py with a unified approach.
"""

from typing import Dict, Any, List, Optional, Union
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation levels in the alignment system."""
    SCRIPT_CONTRACT = 1      # Level 1: Script ↔ Contract
    CONTRACT_SPEC = 2        # Level 2: Contract ↔ Specification  
    SPEC_DEPENDENCY = 3      # Level 3: Specification ↔ Dependencies (Universal)
    BUILDER_CONFIG = 4       # Level 4: Builder ↔ Configuration


class ValidationStatus(Enum):
    """Status of validation operations."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    EXCLUDED = "EXCLUDED"
    ERROR = "ERROR"
    ISSUES_FOUND = "ISSUES_FOUND"
    PASSED_WITH_WARNINGS = "PASSED_WITH_WARNINGS"
    NO_VALIDATOR = "NO_VALIDATOR"
    COMPLETED = "COMPLETED"


class IssueLevel(Enum):
    """Severity levels for validation issues."""
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class RuleType(Enum):
    """Types of validation rules."""
    UNIVERSAL = "universal"
    STEP_SPECIFIC = "step_specific"
    METHOD_INTERFACE = "method_interface"
    CONFIGURATION = "configuration"


@dataclass
class ValidationIssue:
    """Represents a validation issue found during alignment checking."""
    level: IssueLevel
    message: str
    method_name: Optional[str] = None
    rule_type: Optional[RuleType] = None
    details: Optional[Dict[str, Any]] = None
    step_name: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert issue to dictionary format."""
        return {
            "level": self.level.value,
            "message": self.message,
            "method_name": self.method_name,
            "rule_type": self.rule_type.value if self.rule_type else None,
            "details": self.details or {},
            "step_name": self.step_name,
            "file_path": self.file_path,
            "line_number": self.line_number
        }


@dataclass
class ValidationResult:
    """Represents the result of a validation operation."""
    status: ValidationStatus
    step_name: str
    validation_level: Optional[ValidationLevel] = None
    issues: List[ValidationIssue] = None
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def error_count(self) -> int:
        """Count of error-level issues."""
        return sum(1 for issue in self.issues if issue.level == IssueLevel.ERROR)
    
    @property
    def warning_count(self) -> int:
        """Count of warning-level issues."""
        return sum(1 for issue in self.issues if issue.level == IssueLevel.WARNING)
    
    @property
    def info_count(self) -> int:
        """Count of info-level issues."""
        return sum(1 for issue in self.issues if issue.level == IssueLevel.INFO)
    
    @property
    def total_issues(self) -> int:
        """Total count of all issues."""
        return len(self.issues)
    
    def add_issue(self, issue: ValidationIssue):
        """Add a validation issue to the result."""
        self.issues.append(issue)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "status": self.status.value,
            "step_name": self.step_name,
            "validation_level": self.validation_level.value if self.validation_level else None,
            "issues": [issue.to_dict() for issue in self.issues],
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "total_issues": self.total_issues,
            "metadata": self.metadata,
            "error_message": self.error_message
        }


@dataclass
class MethodValidationInfo:
    """Information about method validation requirements."""
    method_name: str
    is_required: bool
    expected_signature: Optional[str] = None
    return_type: Optional[str] = None
    purpose: Optional[str] = None
    validation_rules: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.validation_rules is None:
            self.validation_rules = []


@dataclass
class StepValidationContext:
    """Context information for step validation."""
    step_name: str
    step_type: str
    builder_class_name: Optional[str] = None
    workspace_dirs: Optional[List[str]] = None
    validation_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.workspace_dirs is None:
            self.workspace_dirs = []
        if self.validation_config is None:
            self.validation_config = {}


class ValidationSummary:
    """Summary of validation results across multiple steps."""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.total_steps = 0
        self.passed_steps = 0
        self.failed_steps = 0
        self.excluded_steps = 0
        self.error_steps = 0
        self.total_issues = 0
        self.total_errors = 0
        self.total_warnings = 0
        self.total_info = 0
    
    def add_result(self, result: ValidationResult):
        """Add a validation result to the summary."""
        self.results.append(result)
        self.total_steps += 1
        
        # Update status counts
        if result.status == ValidationStatus.PASSED:
            self.passed_steps += 1
        elif result.status in [ValidationStatus.FAILED, ValidationStatus.ISSUES_FOUND]:
            self.failed_steps += 1
        elif result.status == ValidationStatus.EXCLUDED:
            self.excluded_steps += 1
        elif result.status == ValidationStatus.ERROR:
            self.error_steps += 1
        
        # Update issue counts
        self.total_issues += result.total_issues
        self.total_errors += result.error_count
        self.total_warnings += result.warning_count
        self.total_info += result.info_count
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate (passed / total non-excluded)."""
        non_excluded = self.total_steps - self.excluded_steps
        if non_excluded == 0:
            return 1.0
        return self.passed_steps / non_excluded
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary format."""
        return {
            "total_steps": self.total_steps,
            "passed_steps": self.passed_steps,
            "failed_steps": self.failed_steps,
            "excluded_steps": self.excluded_steps,
            "error_steps": self.error_steps,
            "success_rate": self.success_rate,
            "total_issues": self.total_issues,
            "total_errors": self.total_errors,
            "total_warnings": self.total_warnings,
            "total_info": self.total_info,
            "results": [result.to_dict() for result in self.results]
        }


# Utility functions for working with validation models

def create_validation_issue(
    level: Union[IssueLevel, str],
    message: str,
    method_name: Optional[str] = None,
    rule_type: Optional[Union[RuleType, str]] = None,
    details: Optional[Dict[str, Any]] = None,
    **kwargs
) -> ValidationIssue:
    """Create a validation issue with proper type conversion."""
    if isinstance(level, str):
        level = IssueLevel(level)
    
    if isinstance(rule_type, str):
        rule_type = RuleType(rule_type)
    
    return ValidationIssue(
        level=level,
        message=message,
        method_name=method_name,
        rule_type=rule_type,
        details=details,
        **kwargs
    )


def create_validation_result(
    status: Union[ValidationStatus, str],
    step_name: str,
    validation_level: Optional[Union[ValidationLevel, int]] = None,
    issues: Optional[List[ValidationIssue]] = None,
    **kwargs
) -> ValidationResult:
    """Create a validation result with proper type conversion."""
    if isinstance(status, str):
        status = ValidationStatus(status)
    
    if isinstance(validation_level, int):
        validation_level = ValidationLevel(validation_level)
    
    return ValidationResult(
        status=status,
        step_name=step_name,
        validation_level=validation_level,
        issues=issues or [],
        **kwargs
    )


def merge_validation_results(results: List[ValidationResult]) -> ValidationResult:
    """Merge multiple validation results into a single result."""
    if not results:
        raise ValueError("Cannot merge empty list of results")
    
    if len(results) == 1:
        return results[0]
    
    # Use the first result as base
    merged = ValidationResult(
        status=ValidationStatus.PASSED,
        step_name=results[0].step_name,
        validation_level=results[0].validation_level,
        issues=[],
        metadata={}
    )
    
    # Merge all issues
    all_issues = []
    has_errors = False
    has_warnings = False
    
    for result in results:
        all_issues.extend(result.issues)
        if result.error_count > 0:
            has_errors = True
        if result.warning_count > 0:
            has_warnings = True
        
        # Merge metadata
        if result.metadata:
            merged.metadata.update(result.metadata)
    
    merged.issues = all_issues
    
    # Determine final status
    if has_errors:
        merged.status = ValidationStatus.FAILED
    elif has_warnings:
        merged.status = ValidationStatus.PASSED_WITH_WARNINGS
    else:
        merged.status = ValidationStatus.PASSED
    
    return merged


def filter_issues_by_level(issues: List[ValidationIssue], level: IssueLevel) -> List[ValidationIssue]:
    """Filter validation issues by severity level."""
    return [issue for issue in issues if issue.level == level]


def group_issues_by_method(issues: List[ValidationIssue]) -> Dict[str, List[ValidationIssue]]:
    """Group validation issues by method name."""
    grouped = {}
    for issue in issues:
        method_name = issue.method_name or "unknown"
        if method_name not in grouped:
            grouped[method_name] = []
        grouped[method_name].append(issue)
    return grouped


def format_validation_summary(summary: ValidationSummary) -> str:
    """Format validation summary as a readable string."""
    lines = [
        f"Validation Summary:",
        f"  Total Steps: {summary.total_steps}",
        f"  Passed: {summary.passed_steps}",
        f"  Failed: {summary.failed_steps}",
        f"  Excluded: {summary.excluded_steps}",
        f"  Errors: {summary.error_steps}",
        f"  Success Rate: {summary.success_rate:.1%}",
        f"",
        f"Issues:",
        f"  Total: {summary.total_issues}",
        f"  Errors: {summary.total_errors}",
        f"  Warnings: {summary.total_warnings}",
        f"  Info: {summary.total_info}"
    ]
    return "\n".join(lines)
