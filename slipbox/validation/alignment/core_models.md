---
tags:
  - code
  - validation
  - alignment
  - models
  - data_structures
keywords:
  - core models
  - alignment issues
  - severity levels
  - validation results
  - data models
  - alignment levels
  - step type awareness
  - issue classification
topics:
  - validation framework
  - data models
  - alignment validation
  - issue tracking
language: python
date of note: 2025-08-18
---

# Core Models

## Overview

The Core Models module contains the fundamental data structures and enums used across all alignment validation components. It provides standardized representations for alignment issues, validation results, and severity classifications.

## Core Enumerations

### SeverityLevel Enum

Defines the severity levels for alignment issues:

```python
class SeverityLevel(Enum):
    """Severity levels for alignment issues."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
```

**Severity Hierarchy:**
- **INFO**: Informational findings that don't affect validation
- **WARNING**: Potential issues that may indicate problems
- **ERROR**: Alignment violations that should fail validation
- **CRITICAL**: Severe issues that prevent validation from completing

### AlignmentLevel Enum

Defines the four levels of alignment validation:

```python
class AlignmentLevel(Enum):
    """Alignment validation levels."""
    SCRIPT_CONTRACT = 1
    CONTRACT_SPECIFICATION = 2
    SPECIFICATION_DEPENDENCY = 3
    BUILDER_CONFIGURATION = 4
```

**Alignment Levels:**
- **Level 1**: Script ↔ Contract alignment
- **Level 2**: Contract ↔ Specification alignment
- **Level 3**: Specification ↔ Dependencies alignment
- **Level 4**: Builder ↔ Configuration alignment

## Core Data Models

### AlignmentIssue Model

The fundamental model for representing alignment validation issues:

```python
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
```

**Field Descriptions:**

#### Core Fields
- **level**: Severity classification using SeverityLevel enum
- **category**: Issue category for grouping and filtering
- **message**: Human-readable description of the problem

#### Optional Fields
- **details**: Dictionary containing additional context and information
- **recommendation**: Suggested action to resolve the issue
- **alignment_level**: Which of the four alignment levels this issue affects

**Usage Example:**
```python
issue = AlignmentIssue(
    level=SeverityLevel.ERROR,
    category="configuration_fields",
    message="Builder accesses undeclared configuration field: batch_size",
    details={
        'field_name': 'batch_size',
        'builder': 'processing_builder',
        'accessed_line': 42
    },
    recommendation="Add batch_size to configuration class or remove from builder",
    alignment_level=AlignmentLevel.BUILDER_CONFIGURATION
)
```

### StepTypeAwareAlignmentIssue Model

Extended alignment issue model with SageMaker step type context:

```python
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
```

**Additional Context Fields:**
- **step_type**: SageMaker step type (Processing, Training, Transform, etc.)
- **framework_context**: ML framework context (XGBoost, PyTorch, TensorFlow, etc.)
- **reference_examples**: List of reference implementations or documentation links

**Usage Example:**
```python
step_aware_issue = StepTypeAwareAlignmentIssue(
    level=SeverityLevel.WARNING,
    category="framework_patterns",
    message="XGBoost-specific configuration pattern not followed",
    step_type="Processing",
    framework_context="XGBoost",
    reference_examples=[
        "examples/xgboost_processing_pattern.py",
        "docs/xgboost_best_practices.md"
    ],
    details={
        'expected_pattern': 'XGBoostProcessor',
        'found_pattern': 'SKLearnProcessor'
    },
    recommendation="Use XGBoostProcessor for XGBoost-based processing steps"
)
```

### ValidationResult Model

Represents the complete result of a validation operation:

```python
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
```

**Field Descriptions:**
- **is_valid**: Boolean indicating overall validation success
- **issues**: List of all alignment issues found
- **summary**: Summary statistics and key findings
- **metadata**: Additional context about the validation run

#### Computed Properties

```python
@property
def has_errors(self) -> bool:
    """Check if validation result has any errors."""
    return any(issue.level in [SeverityLevel.ERROR, SeverityLevel.CRITICAL] for issue in self.issues)

@property
def has_warnings(self) -> bool:
    """Check if validation result has any warnings."""
    return any(issue.level == SeverityLevel.WARNING for issue in self.issues)
```

#### Issue Management

```python
def add_issue(self, issue: AlignmentIssue) -> None:
    """Add an issue to the validation result."""
    self.issues.append(issue)
    if issue.level in [SeverityLevel.ERROR, SeverityLevel.CRITICAL]:
        self.is_valid = False
```

**Automatic Validation Status Update:**
- Adding ERROR or CRITICAL issues automatically sets `is_valid = False`
- WARNING and INFO issues don't affect validation status

**Usage Example:**
```python
result = ValidationResult(is_valid=True)

# Add a warning - doesn't affect validity
warning_issue = AlignmentIssue(
    level=SeverityLevel.WARNING,
    category="best_practices",
    message="Consider adding validation logic"
)
result.add_issue(warning_issue)
print(result.is_valid)  # Still True

# Add an error - automatically sets is_valid to False
error_issue = AlignmentIssue(
    level=SeverityLevel.ERROR,
    category="configuration_fields",
    message="Required field not found"
)
result.add_issue(error_issue)
print(result.is_valid)  # Now False
```

## Factory Functions

### Standard AlignmentIssue Creation

```python
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
    """
```

**Factory Benefits:**
- **Consistent Creation**: Ensures proper field initialization
- **Default Handling**: Provides sensible defaults for optional fields
- **Type Safety**: Enforces correct parameter types

**Usage Example:**
```python
issue = create_alignment_issue(
    level=SeverityLevel.ERROR,
    category="missing_file",
    message="Contract file not found",
    details={'searched_path': '/path/to/contract.py'},
    recommendation="Create the missing contract file",
    alignment_level=AlignmentLevel.SCRIPT_CONTRACT
)
```

### Step Type-Aware Issue Creation

```python
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
    """
```

**Enhanced Factory Features:**
- **Step Type Context**: Include SageMaker step type information
- **Framework Context**: Add ML framework-specific context
- **Reference Examples**: Provide helpful implementation examples

**Usage Example:**
```python
step_issue = create_step_type_aware_alignment_issue(
    level=SeverityLevel.WARNING,
    category="framework_optimization",
    message="Consider using framework-specific optimizations",
    step_type="Training",
    framework_context="PyTorch",
    reference_examples=[
        "examples/pytorch_distributed_training.py",
        "docs/pytorch_optimization_guide.md"
    ],
    details={
        'current_implementation': 'basic_training',
        'suggested_optimization': 'distributed_training'
    },
    recommendation="Implement distributed training for better performance"
)
```

## Integration Patterns

### Issue Collection

```python
# Collect issues from multiple validation steps
issues = []

# Level 1 validation
script_issues = validate_script_contract_alignment(script, contract)
issues.extend(script_issues)

# Level 2 validation  
contract_issues = validate_contract_spec_alignment(contract, spec)
issues.extend(contract_issues)

# Create comprehensive result
result = ValidationResult(
    is_valid=not any(issue.has_errors for issue in issues),
    issues=issues,
    summary={
        'total_issues': len(issues),
        'error_count': sum(1 for issue in issues if issue.level == SeverityLevel.ERROR),
        'warning_count': sum(1 for issue in issues if issue.level == SeverityLevel.WARNING)
    }
)
```

### Issue Filtering and Analysis

```python
# Filter issues by severity
critical_issues = [issue for issue in result.issues if issue.level == SeverityLevel.CRITICAL]
error_issues = [issue for issue in result.issues if issue.level == SeverityLevel.ERROR]

# Filter by alignment level
level1_issues = [issue for issue in result.issues if issue.alignment_level == AlignmentLevel.SCRIPT_CONTRACT]

# Filter by category
config_issues = [issue for issue in result.issues if issue.category == "configuration_fields"]

# Group by step type (for step-aware issues)
step_type_groups = {}
for issue in result.issues:
    if isinstance(issue, StepTypeAwareAlignmentIssue) and issue.step_type:
        if issue.step_type not in step_type_groups:
            step_type_groups[issue.step_type] = []
        step_type_groups[issue.step_type].append(issue)
```

### Reporting Integration

```python
# Convert to reporter format
from .alignment_reporter import AlignmentReport, ValidationResult as ReporterResult

reporter_result = ReporterResult(
    test_name="comprehensive_alignment_validation",
    passed=result.is_valid,
    issues=result.issues,
    details=result.metadata
)

report = AlignmentReport()
report.add_level1_result("script_validation", reporter_result)
```

## Best Practices

### Issue Creation

**Descriptive Messages:**
```python
# Good: Specific and actionable
create_alignment_issue(
    level=SeverityLevel.ERROR,
    category="configuration_fields",
    message="Builder accesses undeclared configuration field 'batch_size' at line 42",
    recommendation="Add 'batch_size' field to ProcessingStepConfig class"
)

# Avoid: Vague and unhelpful
create_alignment_issue(
    level=SeverityLevel.ERROR,
    category="error",
    message="Something is wrong"
)
```

**Comprehensive Details:**
```python
# Good: Rich context information
create_alignment_issue(
    level=SeverityLevel.WARNING,
    category="validation_patterns",
    message="No validation logic detected for required fields",
    details={
        'required_fields': ['input_path', 'output_path'],
        'builder_file': 'builder_processing_step.py',
        'config_class': 'ProcessingStepConfig',
        'validation_methods_found': []
    },
    recommendation="Add validation calls for required configuration fields"
)
```

### Severity Assignment

**Severity Guidelines:**
- **CRITICAL**: Issues that prevent validation from completing
- **ERROR**: Alignment violations that should fail validation
- **WARNING**: Potential problems that may cause issues
- **INFO**: Informational findings for awareness

### Category Naming

**Consistent Categories:**
- `missing_file`: Missing required files
- `configuration_fields`: Configuration field alignment issues
- `logical_names`: Logical name consistency issues
- `data_types`: Data type compatibility issues
- `dependency_resolution`: Dependency resolution problems
- `validation_patterns`: Validation logic issues

The Core Models provide the foundational data structures for the entire alignment validation framework, ensuring consistent issue representation and comprehensive validation result tracking across all alignment levels.
