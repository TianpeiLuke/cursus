---
tags:
  - code
  - validation
  - alignment
  - specification
  - dependencies
keywords:
  - specification dependency alignment
  - dependency resolution
  - circular dependency validation
  - data type consistency
  - dependency chains
  - specification validation
  - dependency compatibility
  - canonical step names
topics:
  - validation framework
  - alignment validation
  - dependency resolution
  - specification analysis
language: python
date of note: 2025-08-18
---

# Specification Dependency Alignment

## Overview

The Specification Dependency Alignment Tester validates Level 3 alignment between step specifications and their dependency declarations. It ensures dependency chains are consistent, resolvable, and maintain data type compatibility across the entire pipeline.

## Core Functionality

### SpecificationDependencyAlignmentTester Class

The main class orchestrates comprehensive validation of specification-dependency alignment:

```python
class SpecificationDependencyAlignmentTester:
    """
    Tests alignment between step specifications and their dependencies.
    
    Validates:
    - Dependency chains are consistent
    - All dependencies can be resolved
    - No circular dependencies exist
    - Data types match across dependency chains
    """
```

### Component Architecture

The tester integrates multiple specialized components for robust dependency validation:

**Core Components:**
- **SpecificationLoader**: Loads and processes specification files
- **DependencyValidator**: Performs dependency validation checks
- **DependencyPatternClassifier**: Classifies dependency patterns
- **Level3ValidationConfig**: Configures validation thresholds and behavior
- **Pipeline Components**: Production dependency resolver and registry
- **Step Registry**: Canonical step name mapping and resolution

### Initialization and Configuration

```python
def __init__(self, specs_dir: str, validation_config: Level3ValidationConfig = None):
    """Initialize the specification-dependency alignment tester."""
```

**Configuration Management:**
```python
self.config = validation_config or Level3ValidationConfig.create_relaxed_config()
```

**Component Integration:**
- **SpecificationLoader**: Handles specification file discovery and loading
- **DependencyValidator**: Performs validation with configurable thresholds
- **DependencyPatternClassifier**: Identifies acceptable dependency patterns
- **Pipeline Components**: Production-grade dependency resolution
- **Registry Integration**: Uses production step name registry

**Logging Configuration:**
```python
threshold_desc = self.config.get_threshold_description()
logger.info(f"Level 3 validation initialized with {threshold_desc['mode']} mode")
logger.debug(f"Thresholds: {threshold_desc['thresholds']}")
```

## Validation Process

### Comprehensive Specification Validation

```python
def validate_all_specifications(self, target_scripts: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """Validate alignment for all specifications or specified target scripts."""
```

**Process Flow:**
1. **Discovery Phase**: Identify specifications to validate
2. **Individual Validation**: Process each specification separately
3. **Error Handling**: Capture and report validation failures
4. **Result Aggregation**: Compile comprehensive results

### Individual Specification Validation

```python
def validate_specification(self, spec_name: str) -> Dict[str, Any]:
    """Validate alignment for a specific specification."""
```

**Validation Steps:**

#### 1. Specification File Discovery
```python
spec_files = self._find_specification_files(spec_name)
```

**Discovery Features:**
- Multiple file support for job type variants
- Standard naming pattern recognition
- Comprehensive error reporting for missing files

#### 2. Specification Loading
```python
specifications = {}
for spec_file in spec_files:
    job_type = self._extract_job_type_from_spec_file(spec_file)
    spec = self._load_specification_from_python(spec_file, spec_name, job_type)
    specifications[job_type] = spec
```

**Multi-Specification Handling:**
- Load all job type variants
- Extract job type from file names
- Handle specification parsing errors
- Maintain job type context

#### 3. Dependency Context Loading
```python
all_specs = self._load_all_specifications()
```

**Context Features:**
- Load all available specifications for dependency resolution
- Create comprehensive dependency resolution context
- Support cross-specification dependency validation

#### 4. Alignment Validation
Performs multiple validation checks:
- Dependency resolution validation
- Circular dependency detection
- Data type consistency validation

## Validation Checks

### Dependency Resolution Validation

```python
def _validate_dependency_resolution(self, specification: Dict[str, Any], 
                                  all_specs: Dict[str, Dict[str, Any]], 
                                  spec_name: str) -> List[Dict[str, Any]]:
    """Enhanced dependency validation with compatibility scoring using extracted component."""
```

**Validation Features:**
- **Compatibility Scoring**: Quantitative assessment of dependency matches
- **Semantic Matching**: Intelligent name matching across specifications
- **Source Compatibility**: Validates compatible_sources declarations
- **Data Type Compatibility**: Ensures type consistency across dependencies

**Validation Process:**
1. Extract dependencies from specification
2. Find potential providers for each dependency
3. Calculate compatibility scores
4. Generate specific recommendations for improvements

### Circular Dependency Detection

```python
def _validate_circular_dependencies(self, specification: Dict[str, Any], 
                                  all_specs: Dict[str, Dict[str, Any]], 
                                  spec_name: str) -> List[Dict[str, Any]]:
    """Validate that no circular dependencies exist using extracted component."""
```

**Detection Features:**
- **Graph Analysis**: Builds dependency graph for cycle detection
- **Path Tracking**: Identifies specific circular dependency paths
- **Multi-Level Detection**: Finds both direct and indirect cycles
- **Comprehensive Reporting**: Reports all circular dependencies found

### Data Type Consistency Validation

```python
def _validate_dependency_data_types(self, specification: Dict[str, Any], 
                                  all_specs: Dict[str, Dict[str, Any]], 
                                  spec_name: str) -> List[Dict[str, Any]]:
    """Validate data type consistency across dependency chains using extracted component."""
```

**Type Validation Features:**
- **Chain Analysis**: Validates types across entire dependency chains
- **Type Compatibility**: Checks for compatible type conversions
- **Format Consistency**: Ensures data format compatibility
- **Schema Validation**: Validates data schema consistency

## Production Registry Integration

### Canonical Step Name Resolution

```python
def _get_canonical_step_name(self, spec_file_name: str) -> str:
    """Convert specification file name to canonical step name using the registry."""
```

**Resolution Strategy:**
1. **Registry Mapping**: Use centralized FILE_NAME_TO_CANONICAL mapping
2. **Specification Loading**: Extract step_type from specification file
3. **Step Type Conversion**: Convert step_type to canonical name
4. **Fallback Conversion**: Parse file name to spec_type format

**Registry Integration:**
```python
# Use the centralized registry mapping (single source of truth)
canonical_name = get_canonical_name_from_file_name(spec_file_name)
```

### Available Step Names Discovery

```python
def _get_available_canonical_step_names(self, all_specs: Dict[str, Dict[str, Any]]) -> List[str]:
    """Get available canonical step names using the registry as single source of truth."""
```

**Registry Query:**
```python
from ...steps.registry.step_names import get_all_step_names
canonical_names = get_all_step_names()
```

### Dependency Resolver Integration

```python
def _populate_resolver_registry(self, all_specs: Dict[str, Dict[str, Any]]):
    """Populate the dependency resolver registry with all specifications using canonical names."""
```

**Integration Process:**
1. Convert file-based spec names to canonical names
2. Convert specification dictionaries to StepSpecification objects
3. Register specifications with production dependency resolver
4. Enable production-grade dependency resolution

## Compatibility Analysis

### Flexible Output Matching

```python
def _is_compatible_output(self, required_logical_name: str, output_logical_name: str) -> bool:
    """Check if an output logical name is compatible with a required logical name using flexible matching."""
```

**Matching Strategies:**
- **Exact Match**: Direct name matching
- **Pattern Recognition**: Common data input/output patterns
- **Semantic Equivalence**: Logically equivalent names
- **Bidirectional Matching**: Reverse pattern matching

**Data Pattern Examples:**
```python
data_patterns = {
    'data_input': ['processed_data', 'training_data', 'input_data', 'data', 'model_input_data'],
    'input_data': ['processed_data', 'training_data', 'data_input', 'data', 'model_input_data'],
    'training_data': ['processed_data', 'data_input', 'input_data', 'data', 'model_input_data'],
    # ... more patterns
}
```

### Compatibility Recommendation Generation

```python
def _generate_compatibility_recommendation(self, dep_name: str, best_candidate: Dict) -> str:
    """Generate specific recommendations based on compatibility analysis."""
```

**Recommendation Categories:**
- **Type Compatibility**: Suggests type alignment improvements
- **Semantic Similarity**: Recommends naming improvements
- **Source Compatibility**: Suggests compatible_sources updates
- **Data Type Compatibility**: Guides data type alignment

**Example Recommendations:**
```python
if score_breakdown.get('type_compatibility', 0) < 0.2:
    recommendations.append(f"Consider changing dependency type or output type for better compatibility")

if score_breakdown.get('semantic_similarity', 0) < 0.15:
    recommendations.append(f"Consider renaming '{dep_name}' or adding aliases to improve semantic matching")
```

## Specification Object Conversion

### Dictionary to StepSpecification Conversion

```python
def _dict_to_step_specification(self, spec_dict: Dict[str, Any]) -> StepSpecification:
    """Convert specification dictionary back to StepSpecification object."""
```

**Conversion Process:**
1. **Dependency Conversion**: Create DependencySpec objects
2. **Output Conversion**: Create OutputSpec objects
3. **Specification Assembly**: Create complete StepSpecification object
4. **Type Preservation**: Maintain type information during conversion

**Dependency Conversion:**
```python
dep_data = {
    'logical_name': dep['logical_name'],
    'dependency_type': dep['dependency_type'],
    'required': dep['required'],
    'compatible_sources': dep.get('compatible_sources', []),
    'data_type': dep['data_type'],
    'description': dep.get('description', ''),
    'semantic_keywords': dep.get('semantic_keywords', [])
}
dep_spec = DependencySpec(**dep_data)
```

## Reporting and Analysis

### Dependency Resolution Report

```python
def get_dependency_resolution_report(self, all_specs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Generate detailed dependency resolution report using production resolver."""
```

**Report Features:**
- **Resolution Status**: Overall dependency resolution status
- **Unresolved Dependencies**: List of dependencies that cannot be resolved
- **Resolution Paths**: Successful dependency resolution paths
- **Compatibility Scores**: Quantitative compatibility assessment
- **Recommendations**: Specific improvement suggestions

**Report Generation Process:**
1. Populate resolver registry with all specifications
2. Convert file names to canonical names
3. Generate comprehensive resolution report
4. Include compatibility analysis and recommendations

## Error Handling and Diagnostics

### Missing File Diagnostics

**Specification File Not Found:**
```python
{
    'passed': False,
    'issues': [{
        'severity': 'CRITICAL',
        'category': 'missing_file',
        'message': f'Specification file not found: {self.specs_dir / f"{spec_name}_spec.py"}',
        'recommendation': f'Create the specification file {spec_name}_spec.py'
    }]
}
```

### Parsing Error Diagnostics

**Specification Parse Errors:**
```python
{
    'passed': False,
    'issues': [{
        'severity': 'CRITICAL',
        'category': 'spec_parse_error',
        'message': f'Failed to parse specification from {spec_file}: {str(e)}',
        'recommendation': 'Fix Python syntax or specification structure'
    }]
}
```

### Validation Error Handling

**General Validation Errors:**
```python
{
    'passed': False,
    'error': str(e),
    'issues': [{
        'severity': 'CRITICAL',
        'category': 'validation_error',
        'message': f'Failed to validate specification {spec_name}: {str(e)}'
    }]
}
```

## Integration with Validation Framework

### Result Format

The tester returns comprehensive validation results:

```python
{
    'passed': bool,                    # Overall pass/fail status
    'issues': List[Dict[str, Any]],    # List of alignment issues
    'specification': Dict[str, Any]    # Loaded specification data
}
```

### Issue Categories

- **missing_file**: Specification files not found
- **spec_parse_error**: Specification file parsing failures
- **validation_error**: General validation failures
- **dependency_resolution**: Dependency resolution issues
- **circular_dependencies**: Circular dependency detection
- **data_type_consistency**: Data type consistency issues

### Severity Levels

- **CRITICAL**: Prevents validation from completing
- **ERROR**: Alignment violations that should fail validation
- **WARNING**: Potential issues that may indicate problems
- **INFO**: Informational findings

## Best Practices

### Specification Design

**Clear Dependency Declarations:**
```python
# Good: Explicit dependency specification
dependencies = [
    {
        'logical_name': 'training_data',
        'dependency_type': 'data',
        'required': True,
        'compatible_sources': ['DataPreprocessing', 'FeatureEngineering'],
        'data_type': 'tabular'
    }
]
```

**Consistent Naming:**
```python
# Good: Consistent logical names across specifications
# Provider: output 'processed_data'
# Consumer: dependency 'processed_data' (exact match)
```

### Dependency Management

**Compatible Sources Declaration:**
```python
# Good: Explicit compatible sources
'compatible_sources': ['DataPreprocessing', 'FeatureEngineering', 'DataValidation']
```

**Data Type Consistency:**
```python
# Good: Consistent data types across dependency chain
# Provider: output data_type: 'tabular'
# Consumer: dependency data_type: 'tabular'
```

### File Organization

**Standard Naming Patterns:**
- Specification files: `{spec_name}_spec.py`
- Job type variants: `{spec_name}_{job_type}_spec.py`
- Consistent canonical name mapping

**Registry Integration:**
- Use production STEP_NAMES registry for canonical mapping
- Ensure specification step_type matches registry expectations
- Handle job type variants consistently

The Specification Dependency Alignment Tester provides essential Level 3 validation capabilities, ensuring that dependency chains are consistent, resolvable, and maintain data type compatibility across the entire pipeline through production-grade dependency resolution and comprehensive compatibility analysis.
