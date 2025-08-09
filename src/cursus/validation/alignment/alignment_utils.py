"""
Common utilities for alignment validation.

Provides shared data structures, enums, and helper functions used across
all alignment validation components.
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
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


class PathReference(BaseModel):
    """
    Represents a path reference found in script analysis.
    
    Attributes:
        path: The path string found
        line_number: Line number where the path was found
        context: Surrounding code context
        is_hardcoded: Whether this is a hardcoded path
        construction_method: How the path is constructed (e.g., 'os.path.join')
    """
    path: str
    line_number: int
    context: str
    is_hardcoded: bool = True
    construction_method: Optional[str] = None


class EnvVarAccess(BaseModel):
    """
    Represents environment variable access found in script analysis.
    
    Attributes:
        variable_name: Name of the environment variable
        line_number: Line number where the access was found
        context: Surrounding code context
        access_method: How the variable is accessed (e.g., 'os.environ', 'os.getenv')
        has_default: Whether a default value is provided
        default_value: The default value if provided
    """
    variable_name: str
    line_number: int
    context: str
    access_method: str
    has_default: bool = False
    default_value: Optional[str] = None


class ImportStatement(BaseModel):
    """
    Represents an import statement found in script analysis.
    
    Attributes:
        module_name: Name of the imported module
        import_alias: Alias used for the import (if any)
        line_number: Line number where the import was found
        is_from_import: Whether this is a 'from X import Y' statement
        imported_items: List of specific items imported (for from imports)
    """
    module_name: str
    import_alias: Optional[str]
    line_number: int
    is_from_import: bool = False
    imported_items: List[str] = Field(default_factory=list)


class ArgumentDefinition(BaseModel):
    """
    Represents a command-line argument definition found in script analysis.
    
    Attributes:
        argument_name: Name of the argument (without dashes)
        line_number: Line number where the argument was defined
        is_required: Whether the argument is required
        has_default: Whether the argument has a default value
        default_value: The default value if provided
        argument_type: Type of the argument (str, int, etc.)
        choices: Valid choices for the argument (if any)
    """
    argument_name: str
    line_number: int
    is_required: bool = False
    has_default: bool = False
    default_value: Optional[Any] = None
    argument_type: Optional[str] = None
    choices: Optional[List[str]] = None


class PathConstruction(BaseModel):
    """
    Represents a dynamic path construction found in script analysis.
    
    Attributes:
        base_path: The base path being constructed from
        construction_parts: Parts used in the construction
        line_number: Line number where the construction was found
        context: Surrounding code context
        method: Method used for construction (e.g., 'os.path.join', 'pathlib')
    """
    base_path: str
    construction_parts: List[str]
    line_number: int
    context: str
    method: str


class FileOperation(BaseModel):
    """
    Represents a file operation found in script analysis.
    
    Attributes:
        file_path: Path to the file being operated on
        operation_type: Type of operation (read, write, append, etc.)
        line_number: Line number where the operation was found
        context: Surrounding code context
        mode: File mode used (if specified)
    """
    file_path: str
    operation_type: str
    line_number: int
    context: str
    mode: Optional[str] = None


def normalize_path(path: str) -> str:
    """
    Normalize a path for comparison purposes.
    
    Args:
        path: Path to normalize
        
    Returns:
        Normalized path string
    """
    import os
    return os.path.normpath(path).replace('\\', '/')


def extract_logical_name_from_path(path: str) -> Optional[str]:
    """
    Extract logical name from a SageMaker path.
    
    For paths like '/opt/ml/processing/input/data', extracts 'data'.
    
    Args:
        path: SageMaker path
        
    Returns:
        Logical name or None if not extractable
    """
    # Common SageMaker path patterns
    patterns = [
        '/opt/ml/processing/input/',
        '/opt/ml/processing/output/',
        '/opt/ml/input/data/',
        '/opt/ml/model/',
        '/opt/ml/output/'
    ]
    
    normalized_path = normalize_path(path)
    
    for pattern in patterns:
        if normalized_path.startswith(pattern):
            remainder = normalized_path[len(pattern):].strip('/')
            if remainder:
                # Return the first path component as logical name
                return remainder.split('/')[0]
    
    return None


def is_sagemaker_path(path: str) -> bool:
    """
    Check if a path is a SageMaker container path.
    
    Args:
        path: Path to check
        
    Returns:
        True if this is a SageMaker path
    """
    sagemaker_prefixes = [
        '/opt/ml/processing/',
        '/opt/ml/input/',
        '/opt/ml/model',
        '/opt/ml/output'
    ]
    
    normalized_path = normalize_path(path)
    return any(normalized_path.startswith(prefix) for prefix in sagemaker_prefixes)


def format_alignment_issue(issue: AlignmentIssue) -> str:
    """
    Format an alignment issue for display.
    
    Args:
        issue: The alignment issue to format
        
    Returns:
        Formatted string representation
    """
    level_emoji = {
        SeverityLevel.INFO: "ℹ️",
        SeverityLevel.WARNING: "⚠️", 
        SeverityLevel.ERROR: "❌",
        SeverityLevel.CRITICAL: "🚨"
    }
    
    emoji = level_emoji.get(issue.level, "")
    level_name = issue.level.value
    
    result = f"{emoji} {level_name}: {issue.message}"
    
    if issue.recommendation:
        result += f"\n  💡 Recommendation: {issue.recommendation}"
    
    if issue.details:
        result += f"\n  📋 Details: {issue.details}"
    
    return result


def group_issues_by_severity(issues: List[AlignmentIssue]) -> Dict[SeverityLevel, List[AlignmentIssue]]:
    """
    Group alignment issues by severity level.
    
    Args:
        issues: List of alignment issues
        
    Returns:
        Dictionary mapping severity levels to lists of issues
    """
    grouped = {level: [] for level in SeverityLevel}
    
    for issue in issues:
        grouped[issue.level].append(issue)
    
    return grouped


def get_highest_severity(issues: List[AlignmentIssue]) -> Optional[SeverityLevel]:
    """
    Get the highest severity level among a list of issues.
    
    Args:
        issues: List of alignment issues
        
    Returns:
        Highest severity level or None if no issues
    """
    if not issues:
        return None
    
    severity_order = [SeverityLevel.CRITICAL, SeverityLevel.ERROR, 
                     SeverityLevel.WARNING, SeverityLevel.INFO]
    
    for severity in severity_order:
        if any(issue.level == severity for issue in issues):
            return severity
    
    return None


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
