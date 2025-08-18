---
tags:
  - test
  - builders
  - detection
  - validation
  - step_info
keywords:
  - step info detector
  - builder analysis
  - step type detection
  - framework detection
  - test pattern detection
  - step name resolution
  - registry integration
topics:
  - step builder analysis
  - automatic detection
  - validation infrastructure
  - step classification
language: python
date of note: 2025-08-18
---

# Step Information Detector

## Overview

The Step Information Detector provides automatic detection and analysis of step builder characteristics. The `StepInfoDetector` class analyzes builder classes to extract comprehensive step information including step names, SageMaker step types, frameworks, and test patterns, enabling intelligent test configuration and validation.

## Architecture

### Core Detector Class: StepInfoDetector

The `StepInfoDetector` class serves as the central analyzer for step builder classes:

```python
class StepInfoDetector:
    """Detects step information from builder classes."""
    
    def __init__(self, builder_class: Type[StepBuilderBase]):
        """
        Initialize detector with builder class.
        
        Args:
            builder_class: The step builder class to analyze
        """
        self.builder_class = builder_class
        self._step_info = None
```

### Key Components

#### Builder Class Analysis
- Analyzes builder class names and methods
- Extracts step names and framework information
- Determines test patterns and custom step indicators

#### Registry Integration
- Integrates with step name registry for validation
- Maps builder classes to registered step names
- Provides comprehensive step metadata

## Key Features

### Comprehensive Step Information Detection

The detector provides complete step analysis in a single method:

```python
def detect_step_info(self) -> Dict[str, Any]:
    """
    Detect comprehensive step information from builder class.
    
    Returns:
        Dictionary containing step information:
        - builder_class_name: Name of the builder class
        - step_name: Detected step name from registry
        - sagemaker_step_type: SageMaker step type (Processing, Training, etc.)
        - framework: Detected ML framework (xgboost, pytorch, etc.)
        - test_pattern: Test pattern classification
        - is_custom_step: Whether this is a custom step implementation
        - registry_info: Complete registry information for the step
    """
    if self._step_info is None:
        self._step_info = self._analyze_builder_class()
    return self._step_info
```

### Step Name Detection

Intelligent step name detection from builder class names:

```python
def _detect_step_name_from_class(self, class_name: str) -> Optional[str]:
    """Detect step name from builder class name."""
    # Remove common suffixes
    suffixes = ["StepBuilder", "Builder", "Step"]
    base_name = class_name
    for suffix in suffixes:
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]
            break
    
    # Try to find matching step name in registry
    for step_name, info in STEP_NAMES.items():
        builder_step_name = info.get("builder_step_name", "")
        if builder_step_name:
            # Extract base name from builder step name
            builder_base = builder_step_name.replace("StepBuilder", "").replace("Builder", "")
            if builder_base == base_name:
                return step_name
    
    return None
```

**Step Name Resolution Process**:
1. **Suffix Removal**: Removes common builder suffixes (StepBuilder, Builder, Step)
2. **Registry Matching**: Compares base name with registered step names
3. **Exact Matching**: Finds exact matches in the step registry
4. **Fallback Handling**: Returns None if no match is found

### Framework Detection

Multi-level framework detection from class names and methods:

```python
def _detect_framework(self) -> Optional[str]:
    """Detect framework used by the builder."""
    class_name = self.builder_class.__name__.lower()
    
    # Check for framework indicators in class name
    if "xgboost" in class_name:
        return "xgboost"
    elif "pytorch" in class_name:
        return "pytorch"
    elif "tensorflow" in class_name:
        return "tensorflow"
    elif "sklearn" in class_name:
        return "sklearn"
    
    # Check for framework indicators in methods
    method_names = [method.lower() for method in dir(self.builder_class)]
    method_string = " ".join(method_names)
    
    if "xgboost" in method_string:
        return "xgboost"
    elif "pytorch" in method_string:
        return "pytorch"
    elif "tensorflow" in method_string:
        return "tensorflow"
    elif "sklearn" in method_string:
        return "sklearn"
    
    return None
```

**Framework Detection Levels**:
1. **Class Name Analysis**: Searches for framework keywords in class names
2. **Method Name Analysis**: Examines method names for framework indicators
3. **Priority Order**: XGBoost → PyTorch → TensorFlow → Scikit-learn
4. **Fallback**: Returns None if no framework is detected

### Test Pattern Detection

Intelligent test pattern classification:

```python
def _detect_test_pattern(self, class_name: str, sagemaker_step_type: Optional[str]) -> str:
    """Detect test pattern for the builder."""
    # Check for custom step patterns
    if self._is_custom_step(class_name):
        return "custom_step"
    
    # Check for custom package patterns
    framework = self._detect_framework()
    if framework and framework != "sklearn":
        return "custom_package"
    
    # Default to standard pattern
    return "standard"
```

**Test Pattern Types**:
- **custom_step**: Custom step implementations (CradleDataLoading, MimsModelRegistration)
- **custom_package**: Non-sklearn framework implementations
- **standard**: Standard SageMaker step implementations

### Custom Step Detection

Identification of custom step implementations:

```python
def _is_custom_step(self, class_name: str) -> bool:
    """Check if this is a custom step implementation."""
    custom_step_indicators = [
        "CradleDataLoading",
        "MimsModelRegistration",
        "Custom"
    ]
    
    return any(indicator in class_name for indicator in custom_step_indicators)
```

**Custom Step Indicators**:
- **CradleDataLoading**: Cradle data loading steps
- **MimsModelRegistration**: MIMS model registration steps
- **Custom**: Explicitly custom implementations

## Detection Results Structure

### Complete Step Information

The detector returns comprehensive step information:

```python
{
    "builder_class_name": "XGBoostTrainingStepBuilder",
    "step_name": "xgboost_training",
    "sagemaker_step_type": "Training",
    "framework": "xgboost",
    "test_pattern": "custom_package",
    "is_custom_step": False,
    "registry_info": {
        "builder_step_name": "XGBoostTrainingStepBuilder",
        "sagemaker_step_type": "Training",
        "framework": "xgboost",
        # ... additional registry metadata
    }
}
```

### Registry Integration

Integration with the step name registry:

```python
# Get SageMaker step type from registry
sagemaker_step_type = get_sagemaker_step_type(step_name) if step_name else None

# Include complete registry information
"registry_info": STEP_NAMES.get(step_name, {}) if step_name else {}
```

## Detection Examples

### Processing Step Detection

```python
# TabularPreprocessingStepBuilder analysis
detector = StepInfoDetector(TabularPreprocessingStepBuilder)
info = detector.detect_step_info()

# Results:
{
    "builder_class_name": "TabularPreprocessingStepBuilder",
    "step_name": "tabular_preprocessing",
    "sagemaker_step_type": "Processing",
    "framework": "sklearn",
    "test_pattern": "standard",
    "is_custom_step": False,
    "registry_info": {...}
}
```

### Training Step Detection

```python
# XGBoostTrainingStepBuilder analysis
detector = StepInfoDetector(XGBoostTrainingStepBuilder)
info = detector.detect_step_info()

# Results:
{
    "builder_class_name": "XGBoostTrainingStepBuilder",
    "step_name": "xgboost_training",
    "sagemaker_step_type": "Training",
    "framework": "xgboost",
    "test_pattern": "custom_package",
    "is_custom_step": False,
    "registry_info": {...}
}
```

### Custom Step Detection

```python
# CradleDataLoadingStepBuilder analysis
detector = StepInfoDetector(CradleDataLoadingStepBuilder)
info = detector.detect_step_info()

# Results:
{
    "builder_class_name": "CradleDataLoadingStepBuilder",
    "step_name": "cradle_data_loading",
    "sagemaker_step_type": "Processing",
    "framework": None,
    "test_pattern": "custom_step",
    "is_custom_step": True,
    "registry_info": {...}
}
```

## Integration Points

### With Mock Factory
- Provides step information for mock configuration creation
- Enables framework-specific mock generation
- Supports test pattern-based mock selection

### With Base Test Framework
- Supplies step information for test environment setup
- Enables step type-specific test configuration
- Supports intelligent test execution

### With Registry System
- Integrates with step name registry for validation
- Provides complete step metadata
- Ensures consistency with registered steps

## Usage Examples

### Basic Step Information Detection

```python
from cursus.validation.builders.step_info_detector import StepInfoDetector

# Create detector for a builder class
detector = StepInfoDetector(XGBoostTrainingStepBuilder)

# Detect comprehensive step information
step_info = detector.detect_step_info()

# Access specific information
print(f"Step Name: {step_info['step_name']}")
print(f"SageMaker Step Type: {step_info['sagemaker_step_type']}")
print(f"Framework: {step_info['framework']}")
print(f"Test Pattern: {step_info['test_pattern']}")
print(f"Is Custom Step: {step_info['is_custom_step']}")
```

### Framework-Specific Detection

```python
# Detect framework for different builders
builders = [
    XGBoostTrainingStepBuilder,
    PyTorchTrainingStepBuilder,
    TabularPreprocessingStepBuilder,
    CradleDataLoadingStepBuilder
]

for builder_class in builders:
    detector = StepInfoDetector(builder_class)
    info = detector.detect_step_info()
    
    print(f"{info['builder_class_name']}:")
    print(f"  Framework: {info['framework']}")
    print(f"  Step Type: {info['sagemaker_step_type']}")
    print(f"  Pattern: {info['test_pattern']}")
```

### Registry Information Access

```python
# Access complete registry information
detector = StepInfoDetector(XGBoostTrainingStepBuilder)
step_info = detector.detect_step_info()

registry_info = step_info['registry_info']
print(f"Builder Step Name: {registry_info.get('builder_step_name')}")
print(f"Registry Step Type: {registry_info.get('sagemaker_step_type')}")
print(f"Registry Framework: {registry_info.get('framework')}")
```

### Custom Step Identification

```python
# Identify custom steps
def is_custom_implementation(builder_class):
    detector = StepInfoDetector(builder_class)
    info = detector.detect_step_info()
    return info['is_custom_step']

# Check multiple builders
custom_builders = []
standard_builders = []

for builder_class in all_builders:
    if is_custom_implementation(builder_class):
        custom_builders.append(builder_class)
    else:
        standard_builders.append(builder_class)

print(f"Custom Builders: {len(custom_builders)}")
print(f"Standard Builders: {len(standard_builders)}")
```

## Detection Accuracy

### Framework Detection Accuracy

The detector uses multiple detection methods for high accuracy:

1. **Class Name Matching**: Direct framework keywords in class names
2. **Method Name Analysis**: Framework indicators in method names
3. **Priority Ordering**: Consistent framework precedence
4. **Fallback Handling**: Graceful handling of undetected frameworks

### Step Name Resolution Accuracy

Step name detection leverages the registry system:

1. **Registry Matching**: Exact matching with registered step names
2. **Suffix Normalization**: Consistent handling of builder suffixes
3. **Base Name Extraction**: Accurate extraction of core step names
4. **Fallback Handling**: Graceful handling of unregistered steps

### Test Pattern Classification Accuracy

Test pattern detection provides reliable classification:

1. **Custom Step Detection**: Accurate identification of custom implementations
2. **Framework-based Classification**: Framework-aware pattern detection
3. **Default Fallback**: Standard pattern for unclassified builders

## Performance Considerations

### Caching

The detector implements caching for performance:

```python
def detect_step_info(self) -> Dict[str, Any]:
    """Detect comprehensive step information from builder class."""
    if self._step_info is None:
        self._step_info = self._analyze_builder_class()
    return self._step_info
```

**Caching Benefits**:
- **Single Analysis**: Step information is analyzed only once
- **Repeated Access**: Subsequent calls return cached results
- **Performance Optimization**: Reduces analysis overhead

### Analysis Efficiency

The detection process is optimized for efficiency:

1. **Early Termination**: Framework detection stops at first match
2. **Minimal Reflection**: Limited use of reflection for method analysis
3. **Registry Lookup**: Fast dictionary-based registry lookups

## Error Handling

### Graceful Degradation

The detector handles errors gracefully:

```python
# Framework detection with fallback
framework = self._detect_framework()
if framework is None:
    # Graceful handling of undetected frameworks
    pass

# Step name detection with fallback
step_name = self._detect_step_name_from_class(class_name)
if step_name is None:
    # Graceful handling of unregistered steps
    pass
```

### Robust Detection

Detection methods are designed for robustness:

1. **Multiple Detection Methods**: Framework detection uses multiple approaches
2. **Fallback Values**: Default values for undetected characteristics
3. **Exception Handling**: Graceful handling of analysis errors

## Best Practices

### Usage Patterns

Recommended usage patterns for the detector:

```python
# Single-use detection
detector = StepInfoDetector(builder_class)
step_info = detector.detect_step_info()

# Reuse detector instance for multiple accesses
detector = StepInfoDetector(builder_class)
step_info = detector.detect_step_info()
framework = step_info['framework']
step_type = step_info['sagemaker_step_type']
```

### Integration Guidelines

Guidelines for integrating with other components:

1. **Early Detection**: Detect step information early in test setup
2. **Information Sharing**: Share detected information across components
3. **Validation**: Validate detected information against expectations

### Extension Points

The detector can be extended for additional detection:

```python
class ExtendedStepInfoDetector(StepInfoDetector):
    """Extended detector with additional detection capabilities."""
    
    def _detect_additional_info(self) -> Dict[str, Any]:
        """Detect additional step information."""
        # Custom detection logic
        pass
    
    def _analyze_builder_class(self) -> Dict[str, Any]:
        """Extended analysis with additional information."""
        base_info = super()._analyze_builder_class()
        additional_info = self._detect_additional_info()
        return {**base_info, **additional_info}
```

## Conclusion

The Step Information Detector provides comprehensive and accurate detection of step builder characteristics. Through intelligent analysis of class names, methods, and registry integration, it enables automatic configuration and validation of step builders across the entire validation framework.

The detector's caching mechanism, robust error handling, and extensible design make it a reliable foundation for step builder analysis and test configuration automation.
