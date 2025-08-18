---
tags:
  - code
  - validation
  - alignment
  - property_paths
  - sagemaker
keywords:
  - property path validation
  - SageMaker property paths
  - step property validation
  - SageMaker documentation
  - property path patterns
  - step type validation
  - property path reference
  - validation patterns
topics:
  - validation framework
  - SageMaker integration
  - property path validation
  - step validation
language: python
date of note: 2025-08-18
---

# SageMaker Property Path Validator

## Overview

The SageMaker Property Path Validator implements Level 2 Property Path Validation for the unified alignment tester. It validates SageMaker Step Property Path References based on official SageMaker documentation, ensuring that property paths used in step specifications are valid for the specific SageMaker step type.

## Core Functionality

### SageMakerPropertyPathValidator Class

The main validator class that validates property paths against SageMaker documentation:

```python
class SageMakerPropertyPathValidator:
    """
    Validates SageMaker step property paths against official documentation.
    
    This validator ensures that property paths used in step specifications
    are valid for the specific SageMaker step type, preventing runtime errors
    in pipeline execution.
    """
```

### Documentation Reference

**Official Documentation:**
- **Version**: v2.92.2
- **URL**: https://sagemaker.readthedocs.io/en/v2.92.2/amazon_sagemaker_model_building_pipeline.html#data-dependency-property-reference

### Initialization

```python
def __init__(self):
    """Initialize the property path validator."""
    self.documentation_version = "v2.92.2"
    self.documentation_url = "https://sagemaker.readthedocs.io/en/v2.92.2/amazon_sagemaker_model_building_pipeline.html#data-dependency-property-reference"
    
    # Cache for property path definitions
    self._property_path_cache = {}
```

## Step Registry Integration

### Production Registry Resolution

```python
# STEP REGISTRY INTEGRATION: Resolve actual SageMaker step type
try:
    if STEP_REGISTRY_AVAILABLE and spec_step_type:
        # Get canonical step name from spec type (e.g., "CurrencyConversion_Training" -> "CurrencyConversion")
        canonical_name = get_step_name_from_spec_type(spec_step_type)
        
        # Get actual SageMaker step type from registry (e.g., "CurrencyConversion" -> "Processing")
        sagemaker_step_type = get_sagemaker_step_type(canonical_name)
        
        # Use the resolved SageMaker step type for validation
        resolved_step_type = sagemaker_step_type.lower()
```

**Registry Integration Benefits:**
- **Accurate Step Type Resolution**: Maps custom step names to actual SageMaker step types
- **Production Alignment**: Uses the same registry as production systems
- **Consistent Validation**: Ensures validation matches runtime behavior

### Fallback Mechanisms

```python
except Exception as e:
    # Fallback if registry resolution fails
    resolved_step_type = spec_step_type.lower()
    
    issues.append({
        'severity': 'WARNING',
        'category': 'step_type_resolution',
        'message': f'Step type resolution failed, using fallback: {str(e)}',
        'details': {
            'contract': contract_name,
            'original_spec_type': spec_step_type,
            'resolved_step_type': resolved_step_type,
            'error': str(e)
        },
        'recommendation': 'Check step registry configuration and imports'
    })
```

## Property Path Database

### Comprehensive Step Type Support

The validator includes comprehensive property path definitions for all major SageMaker step types:

#### TrainingStep Properties
```python
# TrainingStep - Properties from DescribeTrainingJob API
if 'training' in step_type_lower or node_type_lower == 'training':
    property_paths = {
        'model_artifacts': [
            'properties.ModelArtifacts.S3ModelArtifacts'
        ],
        'output_config': [
            'properties.OutputDataConfig.S3OutputPath',
            'properties.OutputDataConfig.KmsKeyId'
        ],
        'metrics': [
            'properties.FinalMetricDataList[*].Value',
            'properties.FinalMetricDataList[*].MetricName',
            'properties.FinalMetricDataList[*].Timestamp'
        ],
        'job_info': [
            'properties.TrainingJobName',
            'properties.TrainingJobArn',
            'properties.TrainingJobStatus'
        ]
    }
```

#### ProcessingStep Properties
```python
# ProcessingStep - Properties from DescribeProcessingJob API
elif 'processing' in step_type_lower or node_type_lower == 'processing':
    property_paths = {
        'outputs': [
            'properties.ProcessingOutputConfig.Outputs[*].S3Output.S3Uri',
            'properties.ProcessingOutputConfig.Outputs[*].S3Output.LocalPath',
            'properties.ProcessingOutputConfig.Outputs[*].OutputName'
        ],
        'inputs': [
            'properties.ProcessingInputs[*].S3Input.S3Uri',
            'properties.ProcessingInputs[*].S3Input.LocalPath',
            'properties.ProcessingInputs[*].InputName'
        ]
    }
```

#### TransformStep Properties
```python
# TransformStep - Properties from DescribeTransformJob API
elif 'transform' in step_type_lower or node_type_lower == 'transform':
    property_paths = {
        'outputs': [
            'properties.TransformOutput.S3OutputPath',
            'properties.TransformOutput.Accept',
            'properties.TransformOutput.AssembleWith'
        ],
        'job_info': [
            'properties.TransformJobName',
            'properties.TransformJobStatus'
        ]
    }
```

#### Additional Step Types
- **TuningStep**: Hyperparameter tuning job properties
- **CreateModelStep**: Model creation properties
- **LambdaStep**: Lambda function output parameters
- **CallbackStep**: Callback step output parameters
- **QualityCheckStep**: Model quality check properties
- **ClarifyCheckStep**: Clarify bias detection properties
- **EMRStep**: EMR cluster properties

## Validation Process

### Specification Property Path Validation

```python
def validate_specification_property_paths(self, specification: Dict[str, Any], 
                                        contract_name: str) -> List[Dict[str, Any]]:
    """
    Validate all property paths in a specification.
    
    Args:
        specification: Specification dictionary
        contract_name: Name of the contract being validated
        
    Returns:
        List of validation issues
    """
```

**Validation Steps:**

#### 1. Step Type Resolution
- Extract step_type and node_type from specification
- Use step registry to resolve actual SageMaker step type
- Handle fallback scenarios gracefully

#### 2. Property Path Database Lookup
```python
# Get valid property paths for the resolved step type
valid_property_paths = self._get_valid_property_paths_for_step_type(resolved_step_type, node_type)
```

#### 3. Output Property Path Validation
```python
# Validate property paths in outputs
for output in specification.get('outputs', []):
    property_path = output.get('property_path', '')
    logical_name = output.get('logical_name', '')
    
    if property_path:
        validation_result = self._validate_single_property_path(
            property_path, resolved_step_type, node_type, valid_property_paths
        )
```

#### 4. Issue Generation
```python
if not validation_result['valid']:
    issues.append({
        'severity': 'ERROR',
        'category': 'property_path_validation',
        'message': f'Invalid property path in output {logical_name}: {property_path}',
        'details': {
            'contract': contract_name,
            'logical_name': logical_name,
            'property_path': property_path,
            'step_type': resolved_step_type,
            'error': validation_result['error'],
            'valid_paths': validation_result['suggestions'],
            'documentation_reference': self.documentation_url
        },
        'recommendation': f'Use a valid property path for {resolved_step_type}. Valid paths include: {", ".join(validation_result["suggestions"][:5])}'
    })
```

## Pattern Matching System

### Advanced Pattern Recognition

```python
def _matches_property_path_pattern(self, property_path: str, pattern: str) -> bool:
    """
    Check if a property path matches a pattern with wildcards.
    
    Supports multiple pattern types from the reference database:
    - Exact matches: properties.ModelArtifacts.S3ModelArtifacts
    - Wildcard array access: properties.FinalMetricDataList[*].Value
    - Named array access: properties.FinalMetricDataList['accuracy'].Value
    - Indexed array access: properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri
    """
```

**Pattern Types Supported:**

#### 1. Exact Matches
```python
# Direct exact match
properties.ModelArtifacts.S3ModelArtifacts
```

#### 2. Wildcard Array Access
```python
# Wildcard pattern
properties.FinalMetricDataList[*].Value
```

#### 3. Named Array Access
```python
# Named key access
properties.FinalMetricDataList['accuracy'].Value
properties.FinalMetricDataList["loss"].Value
```

#### 4. Indexed Array Access
```python
# Numeric index access
properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri
properties.ProcessingOutputConfig.Outputs[1].S3Output.S3Uri
```

### Regex Pattern Conversion

```python
# Convert pattern to regex for advanced matching
escaped_pattern = re.escape(pattern)

# Replace escaped [*] with regex patterns for different array access types
escaped_pattern = escaped_pattern.replace(
    r'\[\*\]', 
    r'\[(?:[\'"][^\'\"]*[\'"]|\d+|\*)\]'  # Match ['key'], ["key"], [0], or [*]
)

# Create full regex pattern
regex_pattern = f'^{escaped_pattern}$'
```

## Suggestion System

### Intelligent Path Suggestions

```python
def _get_property_path_suggestions(self, property_path: str, all_valid_paths: List[str]) -> List[str]:
    """
    Get suggestions for a property path based on similarity to valid paths.
    """
```

**Suggestion Algorithm:**

#### 1. Similarity Scoring
```python
def _calculate_path_similarity(self, path1: str, path2: str) -> float:
    """Calculate similarity between two property paths."""
    
    # Split paths into components
    components1 = path1.replace('[', '.').replace(']', '.').split('.')
    components2 = path2.replace('[', '.').replace(']', '.').split('.')
    
    # Calculate component overlap
    common_components = set(components1) & set(components2)
    total_components = set(components1) | set(components2)
    
    component_score = len(common_components) / len(total_components)
    
    # Combine scores
    return (component_score * 0.7) + (substring_score * 0.3)
```

#### 2. Top Suggestions
```python
# Sort by score (descending) and take top suggestions
scored_paths.sort(key=lambda x: x[0], reverse=True)

# Take top 10 suggestions with score > 0
for score, path in scored_paths[:10]:
    if score > 0:
        suggestions.append(path)
```

## Documentation and Introspection

### Step Type Documentation

```python
def get_step_type_documentation(self, step_type: str, node_type: str = '') -> Dict[str, Any]:
    """Get documentation information for a specific step type."""
    
    return {
        'step_type': step_type,
        'node_type': node_type,
        'documentation_url': self.documentation_url,
        'documentation_version': self.documentation_version,
        'valid_property_paths': valid_paths,
        'total_valid_paths': sum(len(paths) for paths in valid_paths.values()),
        'categories': list(valid_paths.keys())
    }
```

### Supported Step Types Listing

```python
def list_supported_step_types(self) -> List[Dict[str, Any]]:
    """List all supported step types and their documentation."""
    
    supported_types = [
        {'step_type': 'training', 'node_type': 'training', 'description': 'TrainingStep - Properties from DescribeTrainingJob API'},
        {'step_type': 'processing', 'node_type': 'processing', 'description': 'ProcessingStep - Properties from DescribeProcessingJob API'},
        {'step_type': 'transform', 'node_type': 'transform', 'description': 'TransformStep - Properties from DescribeTransformJob API'},
        # ... more step types
    ]
```

## Integration Patterns

### Usage with Contract Specification Alignment

```python
# Integration in ContractSpecificationAlignmentTester
def _validate_property_paths(self, specification: Dict[str, Any], contract_name: str) -> List[Dict[str, Any]]:
    """Validate SageMaker Step Property Path References (Level 2 Enhancement)."""
    return self.property_path_validator.validate_specification_property_paths(
        specification, contract_name
    )
```

### Convenience Function

```python
def validate_property_paths(specification: Dict[str, Any], contract_name: str) -> List[Dict[str, Any]]:
    """
    Convenience function to validate property paths in a specification.
    """
    validator = SageMakerPropertyPathValidator()
    return validator.validate_specification_property_paths(specification, contract_name)
```

## Validation Results

### Issue Categories

**property_path_validation:**
- Invalid property paths for specific step types
- Detailed error messages with suggestions
- Documentation references

**step_type_resolution:**
- Step type resolution status and fallbacks
- Registry integration results
- Resolution warnings and errors

**property_path_validation_summary:**
- Overall validation statistics
- Coverage information
- Documentation version references

### Example Validation Results

```python
# Valid property path
{
    'severity': 'INFO',
    'category': 'property_path_validation',
    'message': 'Valid property path in output model_artifacts: properties.ModelArtifacts.S3ModelArtifacts',
    'details': {
        'contract': 'xgboost_training',
        'logical_name': 'model_artifacts',
        'property_path': 'properties.ModelArtifacts.S3ModelArtifacts',
        'step_type': 'training',
        'validation_source': 'SageMaker Documentation v2.92.2'
    }
}

# Invalid property path with suggestions
{
    'severity': 'ERROR',
    'category': 'property_path_validation',
    'message': 'Invalid property path in output metrics: properties.Metrics.Accuracy',
    'details': {
        'contract': 'xgboost_training',
        'logical_name': 'metrics',
        'property_path': 'properties.Metrics.Accuracy',
        'step_type': 'training',
        'error': 'Property path "properties.Metrics.Accuracy" is not valid for step type "training"',
        'valid_paths': [
            'properties.FinalMetricDataList[*].Value',
            'properties.FinalMetricDataList[*].MetricName'
        ]
    },
    'recommendation': 'Use a valid property path for training. Valid paths include: properties.FinalMetricDataList[*].Value, properties.FinalMetricDataList[*].MetricName'
}
```

## Best Practices

### Property Path Design

**Use Official Patterns:**
```python
# Good: Official SageMaker property paths
outputs = [
    {
        'logical_name': 'model_artifacts',
        'property_path': 'properties.ModelArtifacts.S3ModelArtifacts'
    },
    {
        'logical_name': 'training_metrics',
        'property_path': 'properties.FinalMetricDataList[*].Value'
    }
]
```

**Avoid Custom Patterns:**
```python
# Avoid: Custom or non-standard property paths
outputs = [
    {
        'logical_name': 'model_output',
        'property_path': 'custom.model.path'  # Not valid
    }
]
```

### Step Type Consistency

**Registry Integration:**
```python
# Good: Use step registry for consistent step type resolution
canonical_name = get_step_name_from_spec_type(spec_type)
sagemaker_step_type = get_sagemaker_step_type(canonical_name)
```

**Documentation Reference:**
```python
# Good: Include documentation references in specifications
specification = {
    'step_type': 'Training',
    'outputs': [
        {
            'logical_name': 'model_artifacts',
            'property_path': 'properties.ModelArtifacts.S3ModelArtifacts',
            'documentation_ref': 'https://sagemaker.readthedocs.io/en/v2.92.2/amazon_sagemaker_model_building_pipeline.html#data-dependency-property-reference'
        }
    ]
}
```

## Performance and Caching

### Property Path Caching

```python
# Cache for property path definitions
self._property_path_cache = {}

# Create cache key
cache_key = f"{step_type}_{node_type}"

if cache_key in self._property_path_cache:
    return self._property_path_cache[cache_key]

# Cache the result
self._property_path_cache[cache_key] = property_paths
```

### Efficient Pattern Matching

- **Regex Compilation**: Compile patterns once and reuse
- **Early Termination**: Stop on first exact match
- **Similarity Caching**: Cache similarity calculations for repeated comparisons

The SageMaker Property Path Validator provides essential validation capabilities for ensuring that property paths used in step specifications are valid according to official SageMaker documentation, preventing runtime errors and improving pipeline reliability.
