---
tags:
  - test
  - validation
  - alignment
  - utilities
  - framework
keywords:
  - alignment utilities
  - validation utilities
  - step type detection
  - framework detection
  - file resolution
  - severity levels
topics:
  - validation utilities
  - alignment framework
  - utility functions
  - framework detection
language: python
date of note: 2025-08-18
---

# Alignment Utilities

The Alignment Utilities module provides essential utility functions and classes that support the alignment validation framework. It includes step type detection, framework detection, file resolution, severity level management, and other core utilities used throughout the validation system.

## Overview

The alignment utilities serve as the foundation for the alignment validation system, providing common functionality that is shared across all validation levels and components. These utilities ensure consistent behavior and enable advanced features like step type awareness and framework detection.

## Core Utilities

### Severity Level Management

The `SeverityLevel` enum defines the severity levels for validation issues:

```python
class SeverityLevel(Enum):
    CRITICAL = "CRITICAL"    # Issues that prevent system operation
    ERROR = "ERROR"          # Issues that cause validation failures
    WARNING = "WARNING"      # Issues that should be addressed
    INFO = "INFO"           # Informational messages
```

**Usage Example**:
```python
from cursus.validation.alignment.alignment_utils import SeverityLevel

# Create issues with appropriate severity
critical_issue = create_alignment_issue(
    level=SeverityLevel.CRITICAL,
    category='missing_file',
    message='Required file not found'
)
```

### Alignment Level Classification

The `AlignmentLevel` enum defines the four alignment levels:

```python
class AlignmentLevel(Enum):
    SCRIPT_CONTRACT = "script_contract"              # Level 1
    CONTRACT_SPECIFICATION = "contract_specification" # Level 2
    SPECIFICATION_DEPENDENCY = "specification_dependency" # Level 3
    BUILDER_CONFIGURATION = "builder_configuration"   # Level 4
```

### Issue Creation Functions

#### create_alignment_issue()

Creates standard alignment issues:

```python
def create_alignment_issue(
    level: SeverityLevel,
    category: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    recommendation: Optional[str] = None,
    alignment_level: Optional[AlignmentLevel] = None
) -> AlignmentIssue:
```

**Usage Example**:
```python
issue = create_alignment_issue(
    level=SeverityLevel.ERROR,
    category='path_mismatch',
    message='Script uses undeclared path',
    details={'script': 'tabular_preprocessing', 'path': '/tmp/data'},
    recommendation='Add path declaration to contract',
    alignment_level=AlignmentLevel.SCRIPT_CONTRACT
)
```

#### create_step_type_aware_alignment_issue()

Creates step type-aware alignment issues with enhanced context:

```python
def create_step_type_aware_alignment_issue(
    level: SeverityLevel,
    category: str,
    message: str,
    step_type: Optional[str] = None,
    framework_context: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    recommendation: Optional[str] = None,
    alignment_level: Optional[AlignmentLevel] = None
) -> StepTypeAwareAlignmentIssue:
```

**Usage Example**:
```python
issue = create_step_type_aware_alignment_issue(
    level=SeverityLevel.WARNING,
    category='training_pattern_missing',
    message='Training script missing model saving pattern',
    step_type='Training',
    framework_context='xgboost',
    recommendation='Add model.save() or xgb.save_model() call'
)
```

## Step Type Detection

### detect_step_type_from_registry()

Detects step type from the step registry:

```python
def detect_step_type_from_registry(script_name: str) -> Optional[str]:
    """
    Detect step type from step registry.
    
    Args:
        script_name: Name of the script
        
    Returns:
        Step type string or None if not found
    """
```

**Usage Example**:
```python
step_type = detect_step_type_from_registry('tabular_preprocessing')
print(f"Step type: {step_type}")  # Output: "Processing"

step_type = detect_step_type_from_registry('xgboost_training')
print(f"Step type: {step_type}")  # Output: "Training"
```

**Supported Step Types**:
- `Training`: Model training steps
- `Processing`: Data processing steps
- `Transform`: Data transformation steps
- `CreateModel`: Model creation steps
- `RegisterModel`: Model registration steps
- `Utility`: Utility and helper steps

## Framework Detection

### detect_framework_from_imports()

Detects ML framework from import statements:

```python
def detect_framework_from_imports(imports: List[str]) -> Optional[str]:
    """
    Detect ML framework from import statements.
    
    Args:
        imports: List of import statements
        
    Returns:
        Framework name or None if not detected
    """
```

**Usage Example**:
```python
imports = ['import xgboost as xgb', 'import pandas as pd']
framework = detect_framework_from_imports(imports)
print(f"Framework: {framework}")  # Output: "xgboost"

imports = ['import torch', 'import torch.nn as nn']
framework = detect_framework_from_imports(imports)
print(f"Framework: {framework}")  # Output: "pytorch"
```

**Supported Frameworks**:
- `xgboost`: XGBoost machine learning framework
- `pytorch`: PyTorch deep learning framework
- `sklearn`: Scikit-learn machine learning library
- `tensorflow`: TensorFlow deep learning framework
- `lightgbm`: LightGBM gradient boosting framework

## File Resolution

### FlexibleFileResolver

The `FlexibleFileResolver` class provides robust file discovery capabilities:

```python
class FlexibleFileResolver:
    """
    Flexible file resolver for finding related files across directories.
    
    Supports multiple discovery strategies:
    - Exact name matching
    - Pattern-based matching
    - Fuzzy name matching
    - Directory traversal
    """
    
    def __init__(self, base_directories: Dict[str, str]):
        """
        Initialize with base directories for different file types.
        
        Args:
            base_directories: Dictionary mapping file types to directories
        """
```

**Usage Example**:
```python
base_directories = {
    'contracts': 'src/cursus/steps/contracts',
    'builders': 'src/cursus/steps/builders',
    'scripts': 'src/cursus/steps/scripts'
}

resolver = FlexibleFileResolver(base_directories)

# Find contract file for script
contract_path = resolver.find_contract_file('tabular_preprocessing')

# Find builder file for script
builder_path = resolver.find_builder_file('tabular_preprocessing')
```

#### Key Methods

**find_contract_file()**
```python
def find_contract_file(self, script_name: str) -> Optional[str]:
    """Find contract file for given script name."""
```

**find_builder_file()**
```python
def find_builder_file(self, script_name: str) -> Optional[str]:
    """Find builder file for given script name."""
```

**find_specification_file()**
```python
def find_specification_file(self, script_name: str) -> Optional[str]:
    """Find specification file for given script name."""
```

## Path Utilities

### normalize_path()

Normalizes file paths for consistent comparison:

```python
def normalize_path(path: str) -> str:
    """
    Normalize file path for consistent comparison.
    
    Args:
        path: File path to normalize
        
    Returns:
        Normalized path string
    """
```

**Usage Example**:
```python
# Normalize different path formats
path1 = normalize_path('/opt/ml/input/data/training/')
path2 = normalize_path('/opt/ml/input/data/training')
path3 = normalize_path('\\opt\\ml\\input\\data\\training\\')

# All return the same normalized path
assert path1 == path2 == path3
```

**Normalization Rules**:
- Convert backslashes to forward slashes
- Remove trailing slashes
- Resolve relative path components (., ..)
- Convert to lowercase for case-insensitive comparison

## Issue Management

### AlignmentIssue Class

Base class for alignment validation issues:

```python
class AlignmentIssue:
    """Base class for alignment validation issues."""
    
    def __init__(self,
                 level: SeverityLevel,
                 category: str,
                 message: str,
                 details: Optional[Dict[str, Any]] = None,
                 recommendation: Optional[str] = None,
                 alignment_level: Optional[AlignmentLevel] = None):
```

**Attributes**:
- `level`: Severity level of the issue
- `category`: Category classification
- `message`: Human-readable description
- `details`: Additional context and information
- `recommendation`: Suggested fix or improvement
- `alignment_level`: Which alignment level detected the issue

### StepTypeAwareAlignmentIssue Class

Enhanced issue class with step type context:

```python
class StepTypeAwareAlignmentIssue(AlignmentIssue):
    """Enhanced alignment issue with step type awareness."""
    
    def __init__(self,
                 level: SeverityLevel,
                 category: str,
                 message: str,
                 step_type: Optional[str] = None,
                 framework_context: Optional[str] = None,
                 **kwargs):
```

**Additional Attributes**:
- `step_type`: Detected step type (Training, Processing, etc.)
- `framework_context`: Detected framework context (xgboost, pytorch, etc.)

## Framework Pattern Detection

### Training Pattern Detection

```python
def detect_training_patterns(script_content: str) -> Dict[str, bool]:
    """
    Detect training-specific patterns in script content.
    
    Returns:
        Dictionary of detected patterns
    """
```

**Detected Patterns**:
- `training_loop_patterns`: Model training loops
- `model_saving_patterns`: Model artifact saving
- `hyperparameter_loading_patterns`: Hyperparameter loading
- `evaluation_patterns`: Model evaluation logic

### XGBoost Pattern Detection

```python
def detect_xgboost_patterns(script_content: str) -> Dict[str, bool]:
    """
    Detect XGBoost-specific patterns in script content.
    
    Returns:
        Dictionary of detected XGBoost patterns
    """
```

**Detected Patterns**:
- `xgboost_imports`: XGBoost import statements
- `dmatrix_patterns`: DMatrix usage for data handling
- `xgboost_training`: XGBoost training calls
- `model_saving`: XGBoost model saving patterns

## Validation Context

### ValidationContext Class

Provides context information for validation operations:

```python
class ValidationContext:
    """Context information for validation operations."""
    
    def __init__(self,
                 script_name: str,
                 step_type: Optional[str] = None,
                 framework: Optional[str] = None,
                 validation_level: Optional[AlignmentLevel] = None):
```

**Usage Example**:
```python
context = ValidationContext(
    script_name='tabular_preprocessing',
    step_type='Processing',
    framework='pandas',
    validation_level=AlignmentLevel.SCRIPT_CONTRACT
)

# Use context in validation
issue = create_alignment_issue_with_context(
    context=context,
    level=SeverityLevel.WARNING,
    category='pattern_missing',
    message='Missing recommended pattern'
)
```

## Performance Utilities

### Caching Utilities

```python
def create_validation_cache_key(script_name: str, validation_type: str) -> str:
    """Create cache key for validation results."""
    return f"{validation_type}_{script_name}"

def is_cache_valid(cache_key: str, max_age_seconds: int = 3600) -> bool:
    """Check if cached result is still valid."""
    # Implementation...
```

### Batch Processing Utilities

```python
def batch_validate_scripts(
    scripts: List[str],
    validator_func: Callable,
    max_workers: int = 4
) -> Dict[str, Any]:
    """
    Validate multiple scripts in parallel.
    
    Args:
        scripts: List of script names to validate
        validator_func: Validation function to apply
        max_workers: Maximum number of worker threads
        
    Returns:
        Dictionary of validation results
    """
```

## Error Handling Utilities

### Exception Classes

```python
class AlignmentValidationError(Exception):
    """Base exception for alignment validation errors."""
    pass

class ScriptAnalysisError(AlignmentValidationError):
    """Error during script analysis."""
    pass

class ContractLoadError(AlignmentValidationError):
    """Error loading contract file."""
    pass

class FileResolutionError(AlignmentValidationError):
    """Error resolving file paths."""
    pass
```

### Error Recovery

```python
def safe_validate_with_fallback(
    primary_validator: Callable,
    fallback_validator: Callable,
    *args, **kwargs
) -> Dict[str, Any]:
    """
    Safely run validation with fallback on error.
    
    Args:
        primary_validator: Primary validation function
        fallback_validator: Fallback validation function
        
    Returns:
        Validation results from primary or fallback validator
    """
```

## Integration Points

### With Validation Framework

The alignment utilities are used throughout the validation framework:

```python
# In UnifiedAlignmentTester
from .alignment_utils import (
    SeverityLevel, AlignmentLevel, create_alignment_issue,
    detect_step_type_from_registry, detect_framework_from_imports
)

# Create issues with proper severity
issue = create_alignment_issue(
    level=SeverityLevel.ERROR,
    category='alignment_violation',
    message='Component alignment failed',
    alignment_level=AlignmentLevel.SCRIPT_CONTRACT
)
```

### With Step Type Enhancement

```python
# In step type enhancement system
step_type = detect_step_type_from_registry(script_name)
framework = detect_framework_from_imports(imports)

# Create enhanced issues
enhanced_issue = create_step_type_aware_alignment_issue(
    level=SeverityLevel.WARNING,
    category='framework_pattern',
    message='Missing framework-specific pattern',
    step_type=step_type,
    framework_context=framework
)
```

## Configuration and Constants

### Default Configurations

```python
# Default validation configuration
DEFAULT_VALIDATION_CONFIG = {
    'enable_step_type_awareness': True,
    'enable_framework_detection': True,
    'cache_validation_results': True,
    'max_cache_age_seconds': 3600,
    'parallel_validation': True,
    'max_workers': 4
}
```

### File Pattern Constants

```python
# File naming patterns
FILE_PATTERNS = {
    'script': '{script_name}.py',
    'contract': '{script_name}_contract.py',
    'specification': '{script_name}_spec.py',
    'builder': 'builder_{script_name}_step.py',
    'config': '{script_name}_config.py'
}
```

### Framework Detection Patterns

```python
# Framework import patterns
FRAMEWORK_PATTERNS = {
    'xgboost': ['xgboost', 'xgb'],
    'pytorch': ['torch', 'pytorch'],
    'sklearn': ['sklearn', 'scikit-learn'],
    'tensorflow': ['tensorflow', 'tf'],
    'lightgbm': ['lightgbm', 'lgb'],
    'pandas': ['pandas', 'pd'],
    'numpy': ['numpy', 'np']
}
```

## Utility Functions

### String Utilities

```python
def camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case."""
    
def snake_to_camel(name: str) -> str:
    """Convert snake_case to CamelCase."""
    
def normalize_identifier(name: str) -> str:
    """Normalize identifier for comparison."""
```

### Path Utilities

```python
def resolve_relative_path(base_path: str, relative_path: str) -> str:
    """Resolve relative path against base path."""
    
def get_common_path_prefix(paths: List[str]) -> str:
    """Get common prefix of multiple paths."""
    
def is_subpath(parent: str, child: str) -> bool:
    """Check if child path is under parent path."""
```

### Validation Utilities

```python
def merge_validation_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple validation results into single result."""
    
def filter_issues_by_severity(issues: List[Dict], min_severity: SeverityLevel) -> List[Dict]:
    """Filter issues by minimum severity level."""
    
def group_issues_by_category(issues: List[Dict]) -> Dict[str, List[Dict]]:
    """Group issues by category for reporting."""
```

## Performance Optimization

### Caching Mechanisms

The utilities include sophisticated caching mechanisms:

```python
# Result caching
validation_cache = {}

def cache_validation_result(cache_key: str, result: Dict[str, Any]) -> None:
    """Cache validation result with timestamp."""
    validation_cache[cache_key] = {
        'result': result,
        'timestamp': time.time()
    }

def get_cached_validation_result(cache_key: str, max_age: int = 3600) -> Optional[Dict[str, Any]]:
    """Get cached validation result if still valid."""
    if cache_key in validation_cache:
        cached = validation_cache[cache_key]
        if time.time() - cached['timestamp'] < max_age:
            return cached['result']
    return None
```

### Parallel Processing

```python
def parallel_validate(
    items: List[str],
    validator_func: Callable,
    max_workers: int = 4
) -> Dict[str, Any]:
    """
    Run validation in parallel for multiple items.
    
    Args:
        items: List of items to validate
        validator_func: Validation function
        max_workers: Maximum parallel workers
        
    Returns:
        Dictionary of validation results
    """
```

## Best Practices

### For Utility Usage

1. **Use Appropriate Severity**: Choose correct severity levels for issues
2. **Provide Context**: Include relevant details and recommendations
3. **Cache Results**: Leverage caching for performance
4. **Handle Errors**: Use error recovery utilities for robustness
5. **Normalize Data**: Use normalization utilities for consistency

### For Utility Development

1. **Keep Functions Pure**: Avoid side effects in utility functions
2. **Document Thoroughly**: Provide clear documentation and examples
3. **Handle Edge Cases**: Consider edge cases and error conditions
4. **Optimize Performance**: Use caching and efficient algorithms
5. **Maintain Compatibility**: Ensure backward compatibility

## Common Usage Patterns

### Issue Creation Pattern

```python
# Standard pattern for creating validation issues
def validate_component(component_name: str, component_data: Dict) -> List[Dict]:
    issues = []
    
    # Detect context
    step_type = detect_step_type_from_registry(component_name)
    
    # Validate and create issues
    if validation_fails:
        issue = create_alignment_issue(
            level=SeverityLevel.ERROR,
            category='validation_failure',
            message=f'Validation failed for {component_name}',
            details={'component': component_name, 'data': component_data},
            recommendation='Fix the validation issue',
            alignment_level=AlignmentLevel.SCRIPT_CONTRACT
        )
        issues.append(issue)
    
    return issues
```

### File Resolution Pattern

```python
# Standard pattern for file resolution
def find_related_files(script_name: str) -> Dict[str, Optional[str]]:
    resolver = FlexibleFileResolver(base_directories)
    
    return {
        'contract': resolver.find_contract_file(script_name),
        'builder': resolver.find_builder_file(script_name),
        'specification': resolver.find_specification_file(script_name)
    }
```

### Framework Detection Pattern

```python
# Standard pattern for framework detection
def analyze_script_with_framework(script_path: str) -> Dict[str, Any]:
    # Analyze script
    analyzer = ScriptAnalyzer(script_path)
    analysis = analyzer.get_all_analysis_results()
    
    # Detect framework
    framework = detect_framework_from_imports(analysis.get('imports', []))
    
    # Add framework context
    analysis['framework'] = framework
    
    return analysis
```

## Future Enhancements

Planned improvements to the alignment utilities:

1. **Enhanced Framework Detection**: Support for more ML frameworks and libraries
2. **Advanced Caching**: More sophisticated caching strategies
3. **Performance Metrics**: Detailed performance monitoring utilities
4. **Plugin System**: Extensible utility system through plugins
5. **AI-Powered Detection**: Machine learning-based pattern detection

The Alignment Utilities module provides the essential foundation for the entire alignment validation system, enabling sophisticated validation capabilities while maintaining performance and reliability.
