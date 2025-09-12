---
tags:
  - entry_point
  - validation
  - runtime
  - documentation
  - overview
keywords:
  - runtime validation
  - pipeline testing
  - script validation
  - logical name matching
  - data compatibility
  - module overview
topics:
  - validation framework
  - runtime testing
  - module documentation
  - API overview
language: python
date of note: 2025-09-12
---

# Runtime Validation Module

Comprehensive runtime validation system for pipeline scripts and data compatibility testing. Provides sophisticated testing capabilities with intelligent path matching, semantic analysis, and topological execution ordering.

## Overview

The runtime validation module is a complete testing framework designed to validate pipeline scripts and ensure data compatibility between pipeline components. It combines traditional script testing with advanced logical name matching and semantic analysis to provide robust validation capabilities for complex ML pipelines.

The module supports both basic semantic matching for backward compatibility and advanced logical name matching with alias systems for sophisticated pipeline validation. It integrates seamlessly with the broader cursus ecosystem while maintaining clean separation of concerns and extensible architecture.

## Key Features

### 🔧 Core Testing Capabilities
- **Script Execution Validation** - Test individual scripts with proper parameter handling
- **Data Compatibility Testing** - Validate data flow between pipeline components
- **Pipeline Flow Validation** - End-to-end pipeline testing with dependency ordering
- **Error Handling & Reporting** - Clear, actionable error messages and detailed reports

### 🧠 Intelligent Matching
- **Logical Name Matching** - Exact logical name matching with confidence scoring
- **Alias System Support** - Flexible alias-based matching for different naming conventions
- **Semantic Similarity** - AI-powered semantic matching using SemanticMatcher infrastructure
- **Multi-tier Matching** - Hierarchical matching with fallback strategies

### 🏗️ Advanced Architecture
- **Topological Execution** - Proper dependency ordering for pipeline validation
- **Workspace Integration** - Seamless integration with workspace-aware development
- **Configuration Management** - Centralized configuration with automatic feature detection
- **Extensible Design** - Clean interfaces for custom extensions and integrations

## Module Structure

```
src/cursus/validation/runtime/
├── __init__.py                          # Module initialization
├── runtime_testing.py                   # Core testing engine with logical matching integration
├── logical_name_matching.py             # Sophisticated path matching system
├── runtime_models.py                    # Pydantic data models for validation
├── runtime_spec_builder.py              # Builder pattern for creating specifications
├── contract_discovery.py                # Script contract discovery system
└── workspace_aware_spec_builder.py      # Workspace-aware specification building
```

## Quick Start

### Basic Usage

```python
from cursus.validation.runtime.runtime_testing import RuntimeTester
from cursus.validation.runtime.runtime_models import (
    RuntimeTestingConfiguration, 
    PipelineTestingSpec, 
    ScriptExecutionSpec
)
from cursus.api.dag.base_dag import PipelineDAG

# Create script specifications
preprocessing_spec = ScriptExecutionSpec(
    script_name='tabular_preprocessing',
    step_name='TabularPreprocessing_training',
    input_paths={'input_path': '/data/input'},
    output_paths={'processed_data': '/data/output'},
    environ_vars={'PREPROCESSING_MODE': 'standard'},
    job_args={'preprocessing_mode': 'standard'}
)

training_spec = ScriptExecutionSpec(
    script_name='xgboost_training',
    step_name='XGBoostTraining_training',
    input_paths={'training_data': '/data/training'},
    output_paths={'model_output': '/data/model'},
    environ_vars={'MODEL_TYPE': 'xgboost'},
    job_args={'max_depth': '6'}
)

# Create pipeline specification
dag = PipelineDAG(
    nodes=['preprocessing', 'training'],
    edges=[('preprocessing', 'training')]
)

pipeline_spec = PipelineTestingSpec(
    dag=dag,
    script_specs={'preprocessing': preprocessing_spec, 'training': training_spec},
    test_workspace_root='/workspace'
)

# Initialize runtime tester
config = RuntimeTestingConfiguration(pipeline_spec=pipeline_spec)
tester = RuntimeTester(config)

# Test individual script
main_params = tester.builder.get_script_main_params(preprocessing_spec)
script_result = tester.test_script_with_spec(preprocessing_spec, main_params)

if script_result.success:
    print(f"✅ Script {script_result.script_name} executed successfully")
else:
    print(f"❌ Script failed: {script_result.error_message}")

# Test data compatibility
compatibility_result = tester.test_data_compatibility_with_specs(
    preprocessing_spec, training_spec
)

if compatibility_result.compatible:
    print("✅ Scripts are compatible for data flow")
else:
    print(f"❌ Compatibility issues: {compatibility_result.compatibility_issues}")

# Test complete pipeline
pipeline_result = tester.test_pipeline_flow_with_spec(pipeline_spec)

if pipeline_result['pipeline_success']:
    print(f"✅ Pipeline executed successfully")
    print(f"Execution order: {pipeline_result.get('execution_order', [])}")
else:
    print(f"❌ Pipeline errors: {pipeline_result['errors']}")
```

### Advanced Features

```python
# Use logical name matching capabilities
if tester.enable_logical_matching:
    # Get detailed path matches
    path_matches = tester.get_path_matches(preprocessing_spec, training_spec)
    
    for match in path_matches:
        print(f"Match: {match.matched_source_name} -> {match.matched_dest_name}")
        print(f"Type: {match.match_type.value}, Confidence: {match.confidence:.3f}")
    
    # Generate matching report
    matching_report = tester.generate_matching_report(preprocessing_spec, training_spec)
    print(f"Total matches: {matching_report['total_matches']}")
    print(f"High confidence: {matching_report['high_confidence_matches']}")
    
    # Validate entire pipeline
    validation_results = tester.validate_pipeline_logical_names(pipeline_spec)
    print(f"Pipeline valid: {validation_results['overall_valid']}")
    print(f"Validation rate: {validation_results['summary']['validation_rate']:.1%}")
```

## Core Components

### RuntimeTester
The main testing engine that orchestrates all validation activities. Integrates logical name matching when available and falls back to semantic matching for backward compatibility.

**Key Methods:**
- `test_script_with_spec()` - Test individual script functionality
- `test_data_compatibility_with_specs()` - Test data compatibility between scripts
- `test_pipeline_flow_with_spec()` - Test complete pipeline flow
- `get_path_matches()` - Get logical name matches (when available)
- `validate_pipeline_logical_names()` - Validate pipeline-wide compatibility

### Logical Name Matching System
Sophisticated path matching system that uses multiple algorithms to connect script outputs with inputs.

**Matching Hierarchy:**
1. **Exact Logical Match** (confidence: 1.0) - Perfect logical name match
2. **Logical-to-Alias Match** (confidence: 0.95) - Logical name matches alias
3. **Alias-to-Logical Match** (confidence: 0.95) - Alias matches logical name
4. **Alias-to-Alias Match** (confidence: 0.9) - Alias matches alias
5. **Semantic Match** (confidence: variable) - AI-powered semantic similarity

### Data Models
Pydantic-based models providing validation, serialization, and documentation:

- **ScriptExecutionSpec** - Individual script specifications
- **PipelineTestingSpec** - Complete pipeline specifications
- **RuntimeTestingConfiguration** - System configuration
- **ScriptTestResult** - Individual script test results
- **DataCompatibilityResult** - Data compatibility test results

## Integration Points

### Workspace Integration
```python
# Automatic workspace-aware path discovery
tester = RuntimeTester("/workspace/directory")  # Backward compatibility

# Or use configuration object (recommended)
config = RuntimeTestingConfiguration(pipeline_spec=pipeline_spec)
tester = RuntimeTester(config)
```

### Builder Pattern Integration
```python
from cursus.validation.runtime.runtime_spec_builder import PipelineTestingSpecBuilder

# Use builder for parameter extraction
builder = PipelineTestingSpecBuilder(test_data_dir='/workspace')
main_params = builder.get_script_main_params(script_spec)
```

### Semantic Matcher Integration
```python
# Automatic integration with existing SemanticMatcher
from cursus.core.deps.semantic_matcher import SemanticMatcher

# RuntimeTester automatically uses SemanticMatcher for similarity scoring
tester = RuntimeTester(config, semantic_threshold=0.8)

# Semantic matching provides detailed explanations
matches = tester.get_path_matches(spec_a, spec_b)
for match in matches:
    if match.semantic_details:
        print(f"Semantic explanation: {match.semantic_details}")
```

## Configuration Options

### RuntimeTestingConfiguration
```python
config = RuntimeTestingConfiguration(
    pipeline_spec=pipeline_spec,
    enable_enhanced_features=True,      # Auto-detected if None
    enable_logical_matching=True,       # Auto-detected if None
    semantic_threshold=0.7,             # Minimum similarity for semantic matches
    max_execution_time=300,             # Maximum script execution time (seconds)
    output_file_timeout=60,             # Timeout for output file detection
    debug_mode=False                    # Enable debug logging
)
```

### Feature Auto-Detection
The system automatically detects available features based on your pipeline specification:

```python
# Enhanced features auto-enabled if pipeline contains EnhancedScriptExecutionSpec
if pipeline_spec.has_enhanced_specs():
    config.enable_enhanced_features = True
    config.enable_logical_matching = True

# Logical matching auto-enabled if logical_name_matching module available
try:
    from cursus.validation.runtime.logical_name_matching import PathMatcher
    config.enable_logical_matching = True
except ImportError:
    config.enable_logical_matching = False
```

## Error Handling and Debugging

### Clear Error Messages
```python
result = tester.test_script_with_spec(script_spec, main_params)

if not result.success:
    if "requires the following input data" in result.error_message:
        print("Missing input data - check ScriptExecutionSpec paths")
        # Error message includes specific missing files
    elif "missing main() function" in result.error_message:
        print("Script structure issue - ensure main() function exists")
    elif result.is_timeout():
        print(f"Script timed out after {result.execution_time:.2f}s")
```

### Debug Mode
```python
# Enable debug mode for detailed logging
config = RuntimeTestingConfiguration(
    pipeline_spec=pipeline_spec,
    debug_mode=True
)

tester = RuntimeTester(config)
# Detailed logging will show:
# - Script discovery paths
# - Parameter extraction details
# - Matching algorithm steps
# - File system operations
```

## Performance Considerations

### Efficient Testing
```python
# Test scripts in topological order for efficiency
if tester.enable_logical_matching:
    result = tester.test_pipeline_flow_with_topological_execution(pipeline_spec)
    print(f"Execution order: {result['execution_order']}")
else:
    result = tester.test_pipeline_flow_with_spec(pipeline_spec)
```

### Caching and Reuse
```python
# Reuse tester instance for multiple tests
tester = RuntimeTester(config)

# Test multiple script pairs efficiently
for spec_a, spec_b in script_pairs:
    result = tester.test_data_compatibility_with_specs(spec_a, spec_b)
    print(f"{spec_a.script_name} -> {spec_b.script_name}: {result.compatible}")
```

## Testing and Validation

### Unit Testing Support
```python
import pytest
from cursus.validation.runtime.runtime_testing import RuntimeTester

def test_script_validation():
    """Test script validation functionality"""
    tester = RuntimeTester(config)
    
    # Mock script execution for testing
    with patch.object(tester, '_find_script_path', return_value='test_script.py'):
        result = tester.test_script_with_spec(script_spec, main_params)
        assert result.success
```

### Integration Testing
```python
def test_pipeline_integration():
    """Test complete pipeline integration"""
    tester = RuntimeTester(config)
    
    result = tester.test_pipeline_flow_with_spec(pipeline_spec)
    
    assert result['pipeline_success']
    assert len(result['script_results']) == len(pipeline_spec.dag.nodes)
    assert len(result['data_flow_results']) == len(pipeline_spec.dag.edges)
```

## Migration Guide

### From Basic to Enhanced Features
```python
# Old approach (still supported)
tester = RuntimeTester("/workspace/directory")
result = tester.test_data_compatibility_with_specs(spec_a, spec_b)

# New approach (recommended)
config = RuntimeTestingConfiguration(pipeline_spec=pipeline_spec)
tester = RuntimeTester(config)

# Enhanced capabilities automatically available
if tester.enable_logical_matching:
    path_matches = tester.get_path_matches(spec_a, spec_b)
    matching_report = tester.generate_matching_report(spec_a, spec_b)
```

### Adding Logical Name Matching
```python
# Convert basic specs to enhanced specs
from cursus.validation.runtime.logical_name_matching import EnhancedScriptExecutionSpec

enhanced_spec = EnhancedScriptExecutionSpec.from_script_execution_spec(
    basic_spec,
    input_aliases={'raw_data': ['source_data', 'input_dataset']},
    output_aliases={'processed_data': ['clean_data', 'training_ready_data']}
)
```

## API Reference

### Complete Module Documentation
- **[Runtime Testing](runtime_testing.md)** - Core testing engine with logical matching integration
- **[Logical Name Matching](logical_name_matching.md)** - Sophisticated path matching system
- **[Runtime Models](runtime_models.md)** - Pydantic data models for validation
- **[Integration Demo](logical_name_matching_integration_demo.md)** - Complete integration example

### Key Classes
- `RuntimeTester` - Main testing engine
- `PathMatcher` - Logical name matching with confidence scoring
- `TopologicalExecutor` - Pipeline execution ordering
- `ScriptExecutionSpec` - Individual script specifications
- `PipelineTestingSpec` - Complete pipeline specifications

## Examples and Tutorials

### Real-World Pipeline Testing
```python
# Use a real shared DAG from the pipeline catalog
from cursus.pipeline_catalog.shared_dags.xgboost.complete_e2e_dag import create_xgboost_complete_e2e_dag

# Create the complete XGBoost end-to-end DAG
dag = create_xgboost_complete_e2e_dag()

# Create specifications for each step using actual script names
script_specs = {
    'CradleDataLoading_training': ScriptExecutionSpec.create_default(
        'cradle_data_loading', 'CradleDataLoading_training', workspace
    ),
    'TabularPreprocessing_training': ScriptExecutionSpec.create_default(
        'tabular_preprocessing', 'TabularPreprocessing_training', workspace
    ),
    'XGBoostTraining': ScriptExecutionSpec.create_default(
        'xgboost_training', 'XGBoostTraining', workspace
    ),
    'ModelCalibration_calibration': ScriptExecutionSpec.create_default(
        'model_calibration', 'ModelCalibration_calibration', workspace
    ),
    'Package': ScriptExecutionSpec.create_default(
        'package_model', 'Package', workspace
    ),
    'Registration': ScriptExecutionSpec.create_default(
        'model_registration', 'Registration', workspace
    ),
    'Payload': ScriptExecutionSpec.create_default(
        'payload_generation', 'Payload', workspace
    ),
    'CradleDataLoading_calibration': ScriptExecutionSpec.create_default(
        'cradle_data_loading', 'CradleDataLoading_calibration', workspace
    ),
    'TabularPreprocessing_calibration': ScriptExecutionSpec.create_default(
        'tabular_preprocessing', 'TabularPreprocessing_calibration', workspace
    ),
    'XGBoostModelEval_calibration': ScriptExecutionSpec.create_default(
        'xgboost_model_evaluation', 'XGBoostModelEval_calibration', workspace
    )
}

pipeline_spec = PipelineTestingSpec(
    dag=dag,
    script_specs=script_specs,
    test_workspace_root=workspace,
    pipeline_name='XGBoost Complete E2E Pipeline',
    description='Complete XGBoost end-to-end pipeline with training, calibration, packaging, registration, and evaluation'
)

# Test the complete pipeline
config = RuntimeTestingConfiguration(pipeline_spec=pipeline_spec)
tester = RuntimeTester(config)

result = tester.test_pipeline_flow_with_spec(pipeline_spec)

if result['pipeline_success']:
    print("🎉 Complete XGBoost E2E pipeline validated successfully!")
    print(f"Pipeline has {len(dag.nodes)} nodes and {len(dag.edges)} edges")
    print(f"Execution order: {' → '.join(result.get('execution_order', []))}")
    
    # Show data flow validation results
    for edge_key, edge_result in result.get('data_flow_results', {}).items():
        status = '✅' if edge_result.compatible else '❌'
        print(f"  {edge_key}: {status}")
else:
    print("❌ Pipeline validation failed:")
    for error in result['errors']:
        print(f"  - {error}")

# Validate pipeline structure using shared DAG validation
from cursus.pipeline_catalog.shared_dags.xgboost.complete_e2e_dag import validate_dag_structure

dag_validation = validate_dag_structure(dag)
if dag_validation['is_valid']:
    print("✅ DAG structure validation passed")
else:
    print("❌ DAG structure validation failed:")
    for error in dag_validation['errors']:
        print(f"  - {error}")
```

## Contributing

### Adding New Features
1. **Extend Models** - Add new fields to existing Pydantic models
2. **Enhance Matching** - Implement new matching algorithms in PathMatcher
3. **Improve Testing** - Add new validation capabilities to RuntimeTester
4. **Update Documentation** - Keep API documentation current

### Testing Guidelines
- Write comprehensive unit tests for new features
- Include integration tests for end-to-end workflows
- Test both basic and enhanced feature modes
- Validate backward compatibility

## Related Documentation

- **[Validation Framework Overview](../README.md)** - Complete validation system overview
- **[Core Dependencies](../../core/deps/README.md)** - Semantic matching infrastructure
- **[API DAG System](../../api/dag/README.md)** - Pipeline DAG definitions
- **[Workspace Management](../../workspace/README.md)** - Workspace-aware development

---

*This module represents the culmination of sophisticated runtime validation capabilities, combining traditional testing with AI-powered semantic analysis and logical name matching for comprehensive pipeline validation.*
