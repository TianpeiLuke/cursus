---
tags:
  - code
  - validation
  - alignment
  - step_type_detection
  - sagemaker_steps
keywords:
  - step type detection
  - SageMaker step types
  - framework detection
  - script analysis
  - pattern matching
  - step registry integration
topics:
  - alignment validation
  - step type classification
  - script analysis
language: python
date of note: 2025-08-18
---

# Step Type Detection

## Overview

The Step Type Detection module provides comprehensive functionality for detecting SageMaker step types and ML frameworks from scripts and their analysis results. This component is essential for the alignment validation framework as it enables accurate classification of scripts to ensure proper validation against the correct step type specifications.

## Core Functionality

### Step Type Detection Methods

The module provides multiple approaches to step type detection:

1. **Registry-Based Detection**: Uses existing step registry for canonical step type determination
2. **Pattern-Based Detection**: Analyzes script content patterns to infer step types
3. **Framework Detection**: Identifies ML frameworks from import statements
4. **Comprehensive Context**: Combines multiple detection methods for robust classification

### Key Components

#### Registry Integration

The module integrates with the existing step registry system to provide authoritative step type information based on canonical naming conventions.

#### Pattern Recognition

Implements sophisticated pattern matching to identify step types based on script content analysis, including file paths, function calls, and ML-specific patterns.

#### Framework Analysis

Analyzes import statements to detect ML frameworks and libraries, providing additional context for step type classification.

## Core Functions

### detect_step_type_from_registry()

Uses the existing step registry to determine SageMaker step type:

**Purpose**: Provides authoritative step type detection based on canonical naming

**Process**:
1. Converts script name to canonical name format
2. Queries step registry for SageMaker step type
3. Returns detected step type or defaults to "Processing"

**Integration**: Leverages `cursus.steps.registry.step_names` for consistent step type resolution

**Error Handling**: Gracefully handles import errors and missing registry entries with fallback to "Processing"

### detect_framework_from_imports()

Detects ML framework from import analysis:

**Supported Frameworks**:
- **XGBoost**: `xgboost`, `xgb`
- **PyTorch**: `torch`, `pytorch`
- **Scikit-learn**: `sklearn`, `scikit-learn`, `scikit_learn`
- **SageMaker**: `sagemaker`
- **Pandas**: `pandas`, `pd`
- **NumPy**: `numpy`, `np`

**Priority Logic**:
1. ML frameworks take precedence: `xgboost`, `pytorch`, `sklearn`, `sagemaker`
2. Falls back to first detected framework if no priority match
3. Returns `None` if no frameworks detected

**Input Flexibility**: Handles both `ImportStatement` objects and string representations

### detect_step_type_from_script_patterns()

Detects step type from script content patterns:

#### Training Step Patterns
- `xgb.train(` - XGBoost training calls
- `model.fit(` - General model training
- `torch.save(` - PyTorch model saving
- `/opt/ml/model` - SageMaker model output path
- `hyperparameters.json` - Training hyperparameter file
- `model.save_model(` - Model persistence calls

#### Processing Step Patterns
- `/opt/ml/processing/input` - SageMaker processing input path
- `/opt/ml/processing/output` - SageMaker processing output path
- `pd.read_csv(` - Data loading operations
- `.transform(` - Data transformation calls
- `.fit_transform(` - Combined fit and transform operations

#### CreateModel Step Patterns
- `def model_fn(` - SageMaker inference function
- `def input_fn(` - Input preprocessing function
- `def predict_fn(` - Prediction function
- `def output_fn(` - Output postprocessing function
- `pickle.load(` - Model loading with pickle
- `joblib.load(` - Model loading with joblib

**Scoring System**: Counts pattern matches and returns step type with highest score

### get_step_type_context()

Provides comprehensive step type context for a script:

**Context Information**:
- `script_name`: Original script name
- `registry_step_type`: Step type from registry detection
- `pattern_step_type`: Step type from pattern analysis
- `final_step_type`: Resolved final step type
- `confidence`: Confidence level of detection

**Confidence Levels**:
- **High**: Registry and pattern detection agree
- **Medium**: Registry detection available (with or without pattern disagreement)
- **Low**: Only pattern detection or default fallback

**Resolution Logic**:
1. If both registry and pattern agree: High confidence, use agreed type
2. If both available but disagree: Medium confidence, prefer registry
3. If only registry available: Medium confidence, use registry
4. If only pattern available: Low confidence, use pattern
5. If neither available: Low confidence, default to "Processing"

## Integration Points

### Step Registry System
- **Canonical Name Resolution**: Converts file names to canonical step names
- **Step Type Mapping**: Maps canonical names to SageMaker step types
- **Consistency**: Ensures alignment with existing step classification system

### Script Analysis Framework
- **Import Analysis**: Integrates with import statement analysis
- **Content Analysis**: Works with script content parsing
- **Pattern Recognition**: Supports existing script analysis workflows

### Alignment Validation
- **Step Type Context**: Provides step type information for validation
- **Confidence Indicators**: Helps validation systems assess reliability
- **Fallback Mechanisms**: Ensures validation can proceed even with uncertain detection

## Usage Patterns

### Basic Step Type Detection

```python
# Registry-based detection
step_type = detect_step_type_from_registry("xgboost_training_script")
print(f"Detected step type: {step_type}")

# Pattern-based detection
with open("script.py", "r") as f:
    content = f.read()
pattern_type = detect_step_type_from_script_patterns(content)
print(f"Pattern-detected type: {pattern_type}")
```

### Framework Detection

```python
# From import analysis results
imports = [
    ImportStatement(module_name="xgboost", alias="xgb"),
    ImportStatement(module_name="pandas", alias="pd")
]
framework = detect_framework_from_imports(imports)
print(f"Detected framework: {framework}")
```

### Comprehensive Context Analysis

```python
# Get complete step type context
context = get_step_type_context(
    script_name="xgboost_training",
    script_content=script_content
)

print(f"Final step type: {context['final_step_type']}")
print(f"Confidence: {context['confidence']}")
print(f"Registry type: {context['registry_step_type']}")
print(f"Pattern type: {context['pattern_step_type']}")
```

### Validation Integration

```python
# Use in alignment validation
def validate_script_alignment(script_name, script_content):
    context = get_step_type_context(script_name, script_content)
    
    if context['confidence'] == 'low':
        print("Warning: Low confidence in step type detection")
    
    step_type = context['final_step_type']
    # Proceed with step-type-specific validation...
```

## Benefits

### Robust Detection
- **Multiple Methods**: Combines registry and pattern-based approaches
- **Fallback Mechanisms**: Ensures detection always produces a result
- **Confidence Assessment**: Provides reliability indicators for downstream use

### Framework Awareness
- **ML Framework Detection**: Identifies specific ML libraries and frameworks
- **Priority-Based Selection**: Intelligently selects primary framework
- **Extensible Patterns**: Easy addition of new framework detection patterns

### Integration Friendly
- **Registry Compatibility**: Works seamlessly with existing step registry
- **Flexible Input**: Handles various input formats and sources
- **Error Resilience**: Graceful handling of missing or malformed data

## Design Considerations

### Performance Optimization
- **Lazy Evaluation**: Registry queries only when needed
- **Efficient Pattern Matching**: Optimized string searching algorithms
- **Minimal Dependencies**: Lightweight imports and error handling

### Extensibility
- **Pattern Addition**: Easy addition of new step type patterns
- **Framework Extension**: Simple framework detection expansion
- **Detection Method Plugins**: Support for additional detection approaches

### Reliability
- **Error Handling**: Comprehensive exception handling
- **Default Fallbacks**: Sensible defaults when detection fails
- **Confidence Tracking**: Clear indication of detection reliability

## Future Enhancements

### Advanced Pattern Recognition
- **Machine Learning Classification**: ML-based step type classification
- **AST Analysis**: Abstract syntax tree analysis for deeper pattern recognition
- **Semantic Analysis**: Understanding of script semantics beyond pattern matching

### Enhanced Framework Detection
- **Version-Specific Detection**: Framework version identification
- **Dependency Analysis**: Transitive dependency framework detection
- **Configuration-Based Detection**: Framework detection from configuration files

### Integration Improvements
- **Real-Time Detection**: Live detection during script development
- **IDE Integration**: Integration with development environments
- **Validation Feedback**: Improved feedback for validation systems

## Conclusion

The Step Type Detection module provides essential functionality for accurate classification of scripts within the alignment validation framework. By combining registry-based authority with pattern-based inference and framework detection, it ensures robust and reliable step type identification. This component is crucial for enabling step-type-specific validation and maintaining alignment accuracy across different SageMaker step types and ML frameworks.
