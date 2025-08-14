"""
Core data models for alignment validation.

Contains the fundamental enums and base classes used across
all alignment validation components.
"""

from enum import Enum
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class SeverityLevel(Enum):
    """Severity levels for alignment issues."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AlignmentLevel(Enum):
    """Alignment validation levels."""
    SCRIPT_CONTRACT = 1
    CONTRACT_SPECIFICATION = 2
    SPECIFICATION_DEPENDENCY = 3
    BUILDER_CONFIGURATION = 4


class AlignmentIssue(BaseModel):
    """
    Represents an alignment issue found during validation.
    
    Attributes:
        level: Severity level of the issue
        category: Category of the alignment issue
        message: Human-readable description of the issue
        details: Additional details about the issue
        recommendation: Suggested fix for the issue
        alignment_level: Which alignment level this issue affects
    """
    level: SeverityLevel
    category: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    recommendation: Optional[str] = None
    alignment_level: Optional[AlignmentLevel] = None


class StepTypeAwareAlignmentIssue(AlignmentIssue):
    """
    Extends existing AlignmentIssue with step type context.
    
    Additional Attributes:
        step_type: SageMaker step type context (Processing, Training, etc.)
        framework_context: Framework-specific context (XGBoost, PyTorch, etc.)
        reference_examples: List of reference implementation examples
    """
    step_type: Optional[str] = None
    framework_context: Optional[str] = None
    reference_examples: List[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    """
    Represents the result of a validation operation.
    
    Attributes:
        is_valid: Whether the validation passed
        issues: List of issues found during validation
        summary: Summary of validation results
        metadata: Additional metadata about the validation
    """
    is_valid: bool
    issues: List[AlignmentIssue] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def has_errors(self) -> bool:
        """Check if validation result has any errors."""
        return any(issue.level in [SeverityLevel.ERROR, SeverityLevel.CRITICAL] for issue in self.issues)
    
    @property
    def has_warnings(self) -> bool:
        """Check if validation result has any warnings."""
        return any(issue.level == SeverityLevel.WARNING for issue in self.issues)
    
    def add_issue(self, issue: AlignmentIssue) -> None:
        """Add an issue to the validation result."""
        self.issues.append(issue)
        if issue.level in [SeverityLevel.ERROR, SeverityLevel.CRITICAL]:
            self.is_valid = False


def create_alignment_issue(
    level: SeverityLevel,
    category: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    recommendation: Optional[str] = None,
    alignment_level: Optional[AlignmentLevel] = None
) -> AlignmentIssue:
    """
    Create an alignment issue with proper defaults.
    
    Args:
        level: Severity level
        category: Issue category
        message: Issue message
        details: Additional details
        recommendation: Suggested fix
        alignment_level: Which alignment level this affects
        
    Returns:
        AlignmentIssue instance
    """
    return AlignmentIssue(
        level=level,
        category=category,
        message=message,
        details=details or {},
        recommendation=recommendation,
        alignment_level=alignment_level
    )


def create_step_type_aware_alignment_issue(
    level: SeverityLevel,
    category: str,
    message: str,
    step_type: Optional[str] = None,
    framework_context: Optional[str] = None,
    reference_examples: Optional[List[str]] = None,
    details: Optional[Dict[str, Any]] = None,
    recommendation: Optional[str] = None,
    alignment_level: Optional[AlignmentLevel] = None
) -> StepTypeAwareAlignmentIssue:
    """
    Create a step type-aware alignment issue with proper defaults.
    
    Args:
        level: Severity level
        category: Issue category
        message: Issue message
        step_type: SageMaker step type context
        framework_context: Framework-specific context
        reference_examples: List of reference implementation examples
        details: Additional details
        recommendation: Suggested fix
        alignment_level: Which alignment level this affects
        
    Returns:
        StepTypeAwareAlignmentIssue instance
    """
    return StepTypeAwareAlignmentIssue(
        level=level,
        category=category,
        message=message,
        step_type=step_type,
        framework_context=framework_context,
        reference_examples=reference_examples or [],
        details=details or {},
        recommendation=recommendation,
        alignment_level=alignment_level
    )
