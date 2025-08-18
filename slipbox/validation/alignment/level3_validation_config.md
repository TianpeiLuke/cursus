---
tags:
  - code
  - validation
  - alignment
  - level3_validation
  - configuration
keywords:
  - level 3 validation
  - validation configuration
  - compatibility thresholds
  - validation modes
  - severity determination
  - dependency resolution
topics:
  - alignment validation
  - configuration management
  - validation thresholds
language: python
date of note: 2025-08-18
---

# Level 3 Validation Configuration

## Overview

The `Level3ValidationConfig` class provides configurable validation behavior for Level 3 (Specification ↔ Dependencies) alignment validation. This component enables flexible threshold management for compatibility score evaluation, allowing different validation strictness levels based on project requirements and development phases.

## Core Functionality

### Validation Mode System

The configuration system supports three distinct validation modes, each with different threshold requirements:

#### ValidationMode Enum

```python
class ValidationMode(Enum):
    """Validation modes with different threshold requirements."""
    STRICT = "strict"           # Current behavior (exact resolution required)
    RELAXED = "relaxed"         # Allow dependencies with reasonable compatibility
    PERMISSIVE = "permissive"   # Allow dependencies with minimal compatibility
```

### Threshold Configuration

Each validation mode defines specific thresholds for different severity levels:

#### STRICT Mode
- **PASS_THRESHOLD**: 0.8 (≥ 0.8: PASS)
- **WARNING_THRESHOLD**: 0.7 (0.7-0.79: WARNING)
- **ERROR_THRESHOLD**: 0.5 (0.5-0.69: ERROR)
- **CRITICAL**: < 0.5

#### RELAXED Mode (Default)
- **PASS_THRESHOLD**: 0.6 (≥ 0.6: PASS)
- **WARNING_THRESHOLD**: 0.4 (0.4-0.59: WARNING)
- **ERROR_THRESHOLD**: 0.2 (0.2-0.39: ERROR)
- **CRITICAL**: < 0.2

#### PERMISSIVE Mode
- **PASS_THRESHOLD**: 0.3 (≥ 0.3: PASS)
- **WARNING_THRESHOLD**: 0.2 (0.2-0.29: WARNING)
- **ERROR_THRESHOLD**: 0.1 (0.1-0.19: ERROR)
- **CRITICAL**: < 0.1

## Core Components

### Level3ValidationConfig Class

The main configuration class that manages validation thresholds and behavior:

```python
class Level3ValidationConfig:
    """Configuration for Level 3 validation thresholds and behavior."""
    
    def __init__(self, mode: ValidationMode = ValidationMode.RELAXED):
        """
        Initialize validation configuration.
        
        Args:
            mode: Validation mode determining threshold strictness
        """
```

### Key Configuration Parameters

#### Threshold Settings
- **PASS_THRESHOLD**: Minimum score for passing validation
- **WARNING_THRESHOLD**: Minimum score for warning level issues
- **ERROR_THRESHOLD**: Minimum score for error level issues
- **RESOLUTION_THRESHOLD**: Threshold for dependency resolver (≤ PASS_THRESHOLD)

#### Reporting Configuration
- **INCLUDE_SCORE_BREAKDOWN**: Enable detailed scoring in reports
- **INCLUDE_ALTERNATIVE_CANDIDATES**: Show alternative dependency candidates
- **MAX_ALTERNATIVE_CANDIDATES**: Maximum number of alternatives to show (default: 3)

#### Logging Configuration
- **LOG_SUCCESSFUL_RESOLUTIONS**: Log successful dependency resolutions
- **LOG_FAILED_RESOLUTIONS**: Log failed dependency resolutions
- **LOG_SCORE_DETAILS**: Enable detailed score logging for debugging

## Core Methods

### determine_severity_from_score()

Determines issue severity based on compatibility score and dependency requirement status:

**Parameters**:
- `score`: Compatibility score (0.0 to 1.0)
- `is_required`: Whether the dependency is required

**Logic**:
- Score ≥ PASS_THRESHOLD: INFO (should not occur in failed dependencies)
- Score ≥ WARNING_THRESHOLD: WARNING (or ERROR if required)
- Score ≥ ERROR_THRESHOLD: ERROR
- Score < ERROR_THRESHOLD: CRITICAL

**Special Handling**: Required dependencies receive higher severity levels than optional ones.

### should_pass_validation()

Determines if a compatibility score meets the pass threshold:

```python
def should_pass_validation(self, score: float) -> bool:
    """
    Determine if a compatibility score should pass validation.
    
    Args:
        score: Compatibility score (0.0 to 1.0)
        
    Returns:
        True if score meets pass threshold
    """
    return score >= self.PASS_THRESHOLD
```

### get_threshold_description()

Provides human-readable description of current threshold configuration:

**Returns**: Dictionary containing:
- `mode`: Current validation mode
- `thresholds`: Formatted threshold ranges
- `resolution_threshold`: Dependency resolution threshold
- `description`: Mode description text

## Factory Methods

### Predefined Configuration Creators

#### create_strict_config()
Creates a strict validation configuration for production environments requiring high compatibility.

#### create_relaxed_config()
Creates a relaxed validation configuration suitable for most development scenarios.

#### create_permissive_config()
Creates a permissive validation configuration for exploration and early development phases.

### Custom Configuration Creator

#### create_custom_config()
Creates a custom validation configuration with user-specified thresholds:

```python
@classmethod
def create_custom_config(cls, pass_threshold: float, warning_threshold: float, 
                       error_threshold: float) -> 'Level3ValidationConfig':
    """
    Create a custom validation configuration with specific thresholds.
    
    Args:
        pass_threshold: Minimum score for passing validation
        warning_threshold: Minimum score for warning level
        error_threshold: Minimum score for error level
        
    Returns:
        Custom Level3ValidationConfig instance
    """
```

## Integration Points

### Dependency Resolution System
- **RESOLUTION_THRESHOLD**: Controls which dependencies are considered resolvable
- **Compatibility Scoring**: Integrates with dependency compatibility scoring algorithms
- **Alternative Candidates**: Supports showing alternative dependency options

### Validation Reporting
- **Severity Mapping**: Maps compatibility scores to appropriate severity levels
- **Score Breakdown**: Provides detailed scoring information in validation reports
- **Threshold Context**: Includes threshold information in validation summaries

### Alignment Validation Framework
- **Level 3 Integration**: Specifically designed for Specification ↔ Dependencies validation
- **Multi-Level Support**: Compatible with the four-level alignment validation architecture
- **Configurable Behavior**: Allows different validation strictness based on project phase

## Usage Patterns

### Basic Configuration

```python
# Use default relaxed mode
config = Level3ValidationConfig()

# Use specific mode
strict_config = Level3ValidationConfig(ValidationMode.STRICT)
permissive_config = Level3ValidationConfig(ValidationMode.PERMISSIVE)
```

### Factory Method Usage

```python
# Predefined configurations
strict_config = Level3ValidationConfig.create_strict_config()
relaxed_config = Level3ValidationConfig.create_relaxed_config()
permissive_config = Level3ValidationConfig.create_permissive_config()

# Custom configuration
custom_config = Level3ValidationConfig.create_custom_config(
    pass_threshold=0.7,
    warning_threshold=0.5,
    error_threshold=0.3
)
```

### Validation Usage

```python
config = Level3ValidationConfig.create_relaxed_config()

# Determine if score passes validation
if config.should_pass_validation(compatibility_score):
    print("Dependency validation passed")

# Determine severity for reporting
severity = config.determine_severity_from_score(
    score=compatibility_score,
    is_required=True
)

# Get threshold information for reporting
threshold_info = config.get_threshold_description()
print(f"Using {threshold_info['mode']} validation mode")
```

## Benefits

### Flexible Validation Behavior
- **Mode-Based Configuration**: Different strictness levels for different project phases
- **Configurable Thresholds**: Customizable compatibility score requirements
- **Adaptive Severity**: Context-aware severity determination

### Development Workflow Support
- **Strict Mode**: Production-ready validation with high standards
- **Relaxed Mode**: Balanced validation for active development
- **Permissive Mode**: Exploratory validation for early development

### Enhanced Reporting
- **Score Context**: Provides threshold context in validation reports
- **Alternative Suggestions**: Shows alternative dependency candidates
- **Detailed Breakdown**: Optional detailed scoring information

## Design Considerations

### Threshold Relationships
- **Hierarchical Thresholds**: ERROR_THRESHOLD < WARNING_THRESHOLD < PASS_THRESHOLD
- **Resolution Alignment**: RESOLUTION_THRESHOLD ≤ PASS_THRESHOLD for consistency
- **Severity Logic**: Required dependencies receive stricter treatment

### Performance Considerations
- **Lightweight Configuration**: Minimal overhead for threshold checking
- **Efficient Severity Mapping**: Fast score-to-severity conversion
- **Optional Detailed Logging**: Performance-conscious logging controls

### Extensibility
- **Custom Threshold Support**: Easy creation of custom configurations
- **Mode Extension**: Simple addition of new validation modes
- **Configuration Evolution**: Backward-compatible configuration updates

## Future Enhancements

### Advanced Configuration
- **Context-Aware Thresholds**: Different thresholds based on dependency type
- **Dynamic Threshold Adjustment**: Runtime threshold modification based on validation results
- **Profile-Based Configuration**: Predefined configuration profiles for different use cases

### Enhanced Reporting
- **Threshold Visualization**: Graphical representation of threshold ranges
- **Historical Threshold Analysis**: Tracking of threshold effectiveness over time
- **Automated Threshold Recommendations**: Suggestions for optimal threshold values

### Integration Improvements
- **Configuration Persistence**: Save and load configuration settings
- **Environment-Based Configuration**: Different configurations for different environments
- **Team Configuration Sharing**: Standardized configuration across development teams

## Conclusion

The Level 3 Validation Configuration system provides essential flexibility for managing validation behavior in the alignment validation framework. By supporting multiple validation modes with configurable thresholds, it enables teams to adapt validation strictness to their development phase and quality requirements. This component is crucial for maintaining appropriate validation standards while supporting productive development workflows.
