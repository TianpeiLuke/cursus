---
tags:
  - code
  - validation
  - logical_name_matching
  - path_matching
  - semantic_matching
keywords:
  - PathMatcher
  - TopologicalExecutor
  - LogicalNameMatchingTester
  - EnhancedScriptExecutionSpec
  - PathSpec
  - semantic similarity
  - alias matching
topics:
  - logical name matching
  - path matching algorithms
  - topological execution
  - enhanced validation
language: python
date of note: 2025-09-12
---

# Logical Name Matching System

Intelligent path matching system for pipeline runtime testing using semantic similarity, alias systems, and topological execution ordering. Provides sophisticated matching capabilities between script outputs and inputs.

## Overview

The logical name matching system implements advanced path matching algorithms that go beyond simple string matching to provide intelligent connections between script outputs and inputs. The system uses a multi-tiered approach: exact logical name matching, alias-based matching (logical-to-alias, alias-to-logical, alias-to-alias), and semantic similarity matching using the SemanticMatcher infrastructure.

The system is designed to handle complex pipeline scenarios where scripts may use different naming conventions but refer to the same logical data entities. It provides confidence scoring, detailed matching reports, and supports topological execution ordering for proper pipeline validation.

Key components include PathMatcher for intelligent path matching with confidence scoring, TopologicalExecutor for proper pipeline execution ordering, LogicalNameMatchingTester for enhanced compatibility testing, and EnhancedScriptExecutionSpec with alias support following OutputSpec patterns.

## Classes and Methods

### Core Classes
- [`PathMatcher`](#pathmatcher) - Handles logical name matching with semantic capabilities
- [`TopologicalExecutor`](#topologicalexecutor) - Manages topological execution ordering
- [`LogicalNameMatchingTester`](#logicalnamematchingtester) - Enhanced runtime tester with logical matching

### Data Models
- [`PathSpec`](#pathspec) - Enhanced path specification with alias support
- [`PathMatch`](#pathmatch) - Represents successful matches with confidence scoring
- [`EnhancedScriptExecutionSpec`](#enhancedscriptexecutionspec) - Enhanced script spec with alias system
- [`EnhancedDataCompatibilityResult`](#enhanceddatacompatibilityresult) - Enhanced results with matching details

### Enums
- [`MatchType`](#matchtype) - Types of path matches between outputs and inputs

## API Reference

### PathMatcher

_class_ cursus.validation.runtime.logical_name_matching.PathMatcher(_semantic_threshold=0.7_)

Handles logical name matching between ScriptExecutionSpecs using semantic matching capabilities. Implements a multi-tiered matching approach with confidence scoring.

**Parameters:**
- **semantic_threshold** (_float_) – Minimum similarity score for semantic matches. Defaults to 0.7.

```python
from cursus.validation.runtime.logical_name_matching import PathMatcher

matcher = PathMatcher(semantic_threshold=0.8)
```

#### find_path_matches

find_path_matches(_source_spec_, _dest_spec_)

Find matches between source outputs and destination inputs using prioritized matching algorithms.

**Matching Priority:**
1. Exact logical name match (confidence: 1.0)
2. Logical name to alias match (confidence: 0.95)
3. Alias to logical name match (confidence: 0.95)
4. Alias to alias match (confidence: 0.9)
5. Semantic similarity match (confidence: variable based on similarity)

**Parameters:**
- **source_spec** (_EnhancedScriptExecutionSpec_) – Source script specification with outputs.
- **dest_spec** (_EnhancedScriptExecutionSpec_) – Destination script specification with inputs.

**Returns:**
- **List[PathMatch]** – List of PathMatch objects sorted by confidence (highest first).

```python
from cursus.validation.runtime.logical_name_matching import EnhancedScriptExecutionSpec, PathSpec

# Create enhanced specs with aliases
source_spec = EnhancedScriptExecutionSpec(
    script_name='preprocessing',
    output_path_specs={
        'processed_data': PathSpec(
            logical_name='processed_data',
            path='/output/processed.csv',
            aliases=['clean_data', 'training_ready_data']
        )
    }
)

dest_spec = EnhancedScriptExecutionSpec(
    script_name='training',
    input_path_specs={
        'training_data': PathSpec(
            logical_name='training_data',
            path='/input/training.csv',
            aliases=['processed_data', 'clean_data']
        )
    }
)

matches = matcher.find_path_matches(source_spec, dest_spec)

for match in matches:
    print(f"Match: {match.matched_source_name} -> {match.matched_dest_name}")
    print(f"Type: {match.match_type.value}, Confidence: {match.confidence:.3f}")
```

#### generate_matching_report

generate_matching_report(_matches_)

Generate a detailed report of path matching results with statistics and recommendations.

**Parameters:**
- **matches** (_List[PathMatch]_) – List of PathMatch objects to analyze.

**Returns:**
- **Dict[str, Any]** – Dictionary containing detailed matching information and recommendations.

```python
matches = matcher.find_path_matches(source_spec, dest_spec)
report = matcher.generate_matching_report(matches)

print(f"Total matches: {report['total_matches']}")
print(f"High confidence matches: {report['high_confidence_matches']}")
print(f"Average confidence: {report['average_confidence']}")

for recommendation in report['recommendations']:
    print(f"Recommendation: {recommendation}")
```

### TopologicalExecutor

_class_ cursus.validation.runtime.logical_name_matching.TopologicalExecutor()

Handles topological execution ordering for pipeline testing. Ensures proper dependency ordering and validates DAG structure.

```python
from cursus.validation.runtime.logical_name_matching import TopologicalExecutor

executor = TopologicalExecutor()
```

#### get_execution_order

get_execution_order(_dag_)

Get topological execution order from DAG. Uses the DAG's built-in topological sorting capabilities.

**Parameters:**
- **dag** (_PipelineDAG_) – PipelineDAG object to analyze.

**Returns:**
- **List[str]** – List of node names in topological order.

**Raises:**
- **ValueError** – If DAG contains cycles or has topology errors.

```python
from cursus.api.dag.base_dag import PipelineDAG

dag = PipelineDAG(
    nodes=['data_prep', 'feature_eng', 'model_train', 'model_eval'],
    edges=[
        ('data_prep', 'feature_eng'),
        ('feature_eng', 'model_train'),
        ('model_train', 'model_eval')
    ]
)

execution_order = executor.get_execution_order(dag)
print(f"Execution order: {execution_order}")
```

#### validate_dag_structure

validate_dag_structure(_dag_, _script_specs_)

Validate DAG structure and script spec alignment. Ensures all nodes have corresponding specs and identifies structural issues.

**Parameters:**
- **dag** (_PipelineDAG_) – PipelineDAG object to validate.
- **script_specs** (_Dict[str, Any]_) – Dictionary of script specifications.

**Returns:**
- **List[str]** – List of validation errors (empty if valid).

```python
script_specs = {
    'data_prep': preprocessing_spec,
    'feature_eng': feature_spec,
    'model_train': training_spec,
    'model_eval': evaluation_spec
}

errors = executor.validate_dag_structure(dag, script_specs)

if errors:
    print("DAG validation errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("DAG structure is valid")
```

### LogicalNameMatchingTester

_class_ cursus.validation.runtime.logical_name_matching.LogicalNameMatchingTester(_semantic_threshold=0.7_)

Enhanced runtime tester with logical name matching capabilities. Provides sophisticated compatibility testing and pipeline validation.

**Parameters:**
- **semantic_threshold** (_float_) – Minimum similarity score for semantic matches. Defaults to 0.7.

```python
from cursus.validation.runtime.logical_name_matching import LogicalNameMatchingTester

tester = LogicalNameMatchingTester(semantic_threshold=0.8)
```

#### test_data_compatibility_with_logical_matching

test_data_compatibility_with_logical_matching(_spec_a_, _spec_b_, _output_files_)

Test data compatibility between scripts using intelligent path matching. Provides enhanced results with detailed matching information.

**Parameters:**
- **spec_a** (_EnhancedScriptExecutionSpec_) – Source script specification.
- **spec_b** (_EnhancedScriptExecutionSpec_) – Destination script specification.
- **output_files** (_List[Path]_) – List of actual output files from script A.

**Returns:**
- **EnhancedDataCompatibilityResult** – Enhanced results with detailed matching information.

```python
from pathlib import Path

output_files = [Path('/output/processed_data.csv'), Path('/output/metadata.json')]

result = tester.test_data_compatibility_with_logical_matching(
    source_spec, dest_spec, output_files
)

print(f"Compatible: {result.compatible}")
print(f"Path matches found: {len(result.path_matches)}")
print(f"Files tested: {result.files_tested}")

if result.matching_details:
    print(f"Matching report: {result.matching_details}")
```

#### test_pipeline_with_topological_execution

test_pipeline_with_topological_execution(_dag_, _script_specs_, _script_tester_func_)

Test pipeline with proper topological execution order. Ensures dependencies are respected and provides comprehensive validation.

**Parameters:**
- **dag** (_PipelineDAG_) – PipelineDAG object defining pipeline structure.
- **script_specs** (_Dict[str, EnhancedScriptExecutionSpec]_) – Dictionary of enhanced script specifications.
- **script_tester_func** (_Callable_) – Function to test individual scripts.

**Returns:**
- **Dict[str, Any]** – Dictionary with comprehensive pipeline test results.

```python
def script_tester_func(enhanced_spec):
    # Convert enhanced spec back to basic spec for testing
    basic_spec = ScriptExecutionSpec(
        script_name=enhanced_spec.script_name,
        step_name=enhanced_spec.step_name,
        input_paths=enhanced_spec.input_paths,
        output_paths=enhanced_spec.output_paths,
        environ_vars=enhanced_spec.environ_vars,
        job_args=enhanced_spec.job_args
    )
    
    # Test the script (implementation depends on your testing framework)
    return test_script_implementation(basic_spec)

result = tester.test_pipeline_with_topological_execution(
    dag, enhanced_script_specs, script_tester_func
)

print(f"Pipeline success: {result['pipeline_success']}")
print(f"Execution order: {result['execution_order']}")
print(f"Logical matching results: {result['logical_matching_results']}")
```

## Data Models

### PathSpec

_class_ cursus.validation.runtime.logical_name_matching.PathSpec(_logical_name_, _path_, _aliases=[]_)

Enhanced path specification with alias support following OutputSpec pattern. Provides flexible naming for logical data entities.

**Parameters:**
- **logical_name** (_str_) – Primary logical name for the path.
- **path** (_str_) – File system path.
- **aliases** (_List[str]_) – Alternative names for the same logical entity. Defaults to empty list.

```python
from cursus.validation.runtime.logical_name_matching import PathSpec

path_spec = PathSpec(
    logical_name='processed_data',
    path='/output/processed.csv',
    aliases=['clean_data', 'training_ready_data', 'preprocessed_dataset']
)

# Check if a name matches this spec
if path_spec.matches_name_or_alias('clean_data'):
    print("Name matches this path spec")
```

#### matches_name_or_alias

matches_name_or_alias(_name_)

Check if name matches logical_name or any alias.

**Parameters:**
- **name** (_str_) – Name to check against logical name and aliases.

**Returns:**
- **bool** – True if name matches logical_name or any alias.

### PathMatch

_class_ cursus.validation.runtime.logical_name_matching.PathMatch(_source_logical_name_, _dest_logical_name_, _match_type_, _confidence_, _matched_source_name_, _matched_dest_name_, _semantic_details=None_)

Represents a successful match between source output and destination input with detailed matching information.

**Parameters:**
- **source_logical_name** (_str_) – Source output logical name.
- **dest_logical_name** (_str_) – Destination input logical name.
- **match_type** (_MatchType_) – Type of match found.
- **confidence** (_float_) – Confidence score (0.0 to 1.0).
- **matched_source_name** (_str_) – Actual source name that matched.
- **matched_dest_name** (_str_) – Actual destination name that matched.
- **semantic_details** (_Optional[Dict[str, Any]]_) – Detailed semantic matching information.

```python
# PathMatch objects are typically created by PathMatcher.find_path_matches()
matches = matcher.find_path_matches(source_spec, dest_spec)

for match in matches:
    print(f"Match: {match.source_logical_name} -> {match.dest_logical_name}")
    print(f"Actual match: {match.matched_source_name} -> {match.matched_dest_name}")
    print(f"Type: {match.match_type.value}")
    print(f"Confidence: {match.confidence:.3f}")
    
    if match.semantic_details:
        print(f"Semantic details: {match.semantic_details}")
```

### EnhancedScriptExecutionSpec

_class_ cursus.validation.runtime.logical_name_matching.EnhancedScriptExecutionSpec(_script_name_, _step_name_, _input_path_specs={}_, _output_path_specs={}_, _**kwargs_)

Enhanced ScriptExecutionSpec with alias system support. Extends the basic ScriptExecutionSpec with sophisticated path specifications and alias matching capabilities.

**Parameters:**
- **script_name** (_str_) – Name of the script.
- **step_name** (_str_) – Name of the pipeline step.
- **input_path_specs** (_Dict[str, PathSpec]_) – Input path specifications with aliases.
- **output_path_specs** (_Dict[str, PathSpec]_) – Output path specifications with aliases.
- ****kwargs** – Additional arguments passed to base ScriptExecutionSpec.

```python
from cursus.validation.runtime.logical_name_matching import EnhancedScriptExecutionSpec, PathSpec

enhanced_spec = EnhancedScriptExecutionSpec(
    script_name='xgboost_training',
    step_name='XGBoostTraining_training',
    input_path_specs={
        'training_data': PathSpec(
            logical_name='training_data',
            path='/input/training.csv',
            aliases=['processed_data', 'clean_data', 'model_input']
        )
    },
    output_path_specs={
        'model_output': PathSpec(
            logical_name='model_output',
            path='/output/model.pkl',
            aliases=['trained_model', 'model_artifact']
        )
    },
    environ_vars={'MODEL_TYPE': 'xgboost'},
    job_args={'max_depth': '6'}
)
```

#### from_script_execution_spec

_classmethod_ from_script_execution_spec(_original_spec_, _input_aliases=None_, _output_aliases=None_)

Create enhanced spec from original ScriptExecutionSpec. Converts basic specs to enhanced specs with optional alias mappings.

**Parameters:**
- **original_spec** (_ScriptExecutionSpec_) – Original script specification to enhance.
- **input_aliases** (_Optional[Dict[str, List[str]]]_) – Optional input aliases mapping.
- **output_aliases** (_Optional[Dict[str, List[str]]]_) – Optional output aliases mapping.

**Returns:**
- **EnhancedScriptExecutionSpec** – Enhanced specification with alias support.

```python
from cursus.validation.runtime.runtime_models import ScriptExecutionSpec

# Create basic spec
basic_spec = ScriptExecutionSpec(
    script_name='preprocessing',
    input_paths={'raw_data': '/input/raw.csv'},
    output_paths={'processed_data': '/output/processed.csv'}
)

# Convert to enhanced spec with aliases
input_aliases = {
    'raw_data': ['source_data', 'input_dataset']
}
output_aliases = {
    'processed_data': ['clean_data', 'training_ready_data']
}

enhanced_spec = EnhancedScriptExecutionSpec.from_script_execution_spec(
    basic_spec, input_aliases, output_aliases
)
```

### EnhancedDataCompatibilityResult

_class_ cursus.validation.runtime.logical_name_matching.EnhancedDataCompatibilityResult(_script_a_, _script_b_, _compatible_, _path_matches=[]_, _matching_details=None_, _**kwargs_)

Enhanced result model with path matching information. Extends basic DataCompatibilityResult with detailed matching analysis.

**Parameters:**
- **script_a** (_str_) – Source script name.
- **script_b** (_str_) – Destination script name.
- **compatible** (_bool_) – Whether scripts are compatible.
- **path_matches** (_List[PathMatch]_) – List of successful path matches.
- **matching_details** (_Optional[Dict[str, Any]]_) – Detailed matching report.
- ****kwargs** – Additional arguments from base DataCompatibilityResult.

```python
# Enhanced results are typically returned by LogicalNameMatchingTester
result = tester.test_data_compatibility_with_logical_matching(
    source_spec, dest_spec, output_files
)

print(f"Compatible: {result.compatible}")
print(f"Path matches: {len(result.path_matches)}")

if result.matching_details:
    details = result.matching_details
    print(f"Total matches: {details['total_matches']}")
    print(f"High confidence: {details['high_confidence_matches']}")
    
    for rec in details['recommendations']:
        print(f"Recommendation: {rec}")
```

### MatchType

_enum_ cursus.validation.runtime.logical_name_matching.MatchType

Types of path matches between source outputs and destination inputs. Defines the hierarchy of matching algorithms.

**Values:**
- **EXACT_LOGICAL** – Exact logical name match (highest confidence)
- **LOGICAL_TO_ALIAS** – Logical name matches destination alias
- **ALIAS_TO_LOGICAL** – Source alias matches logical name
- **ALIAS_TO_ALIAS** – Alias matches alias
- **SEMANTIC** – Semantic similarity match (variable confidence)

```python
from cursus.validation.runtime.logical_name_matching import MatchType

# Match types are used in PathMatch objects
for match in path_matches:
    if match.match_type == MatchType.EXACT_LOGICAL:
        print("Perfect logical name match!")
    elif match.match_type == MatchType.SEMANTIC:
        print(f"Semantic match with confidence {match.confidence:.3f}")
```

## Advanced Features

### Semantic Integration

The system integrates with the existing SemanticMatcher infrastructure:

```python
# PathMatcher automatically uses SemanticMatcher for similarity scoring
matcher = PathMatcher(semantic_threshold=0.8)

# Semantic matching provides detailed explanations
matches = matcher.find_path_matches(source_spec, dest_spec)

for match in matches:
    if match.match_type == MatchType.SEMANTIC and match.semantic_details:
        print(f"Semantic explanation: {match.semantic_details}")
```

### File Mapping and Detection

The system provides intelligent file mapping based on logical names:

```python
# LogicalNameMatchingTester maps logical names to actual files
output_files = [
    Path('/output/processed_data.csv'),
    Path('/output/feature_metadata.json'),
    Path('/output/preprocessing_log.txt')
]

result = tester.test_data_compatibility_with_logical_matching(
    source_spec, dest_spec, output_files
)

# System automatically maps files to logical names based on naming patterns
print(f"Files tested: {result.files_tested}")
print(f"Primary format detected: {result.data_format_a}")
```

### Confidence Scoring and Recommendations

The system provides detailed confidence analysis:

```python
matches = matcher.find_path_matches(source_spec, dest_spec)
report = matcher.generate_matching_report(matches)

# Analyze confidence distribution
confidence_dist = report['confidence_distribution']
print(f"Average confidence: {confidence_dist['average']:.3f}")
print(f"High confidence matches: {confidence_dist['high_confidence']}")
print(f"Low confidence matches: {confidence_dist['low_confidence']}")

# Get actionable recommendations
for recommendation in report['recommendations']:
    print(f"💡 {recommendation}")
```

## Integration with Runtime Testing

The logical name matching system is designed to integrate seamlessly with the RuntimeTester:

```python
from cursus.validation.runtime.runtime_testing import RuntimeTester

# RuntimeTester automatically uses logical matching when available
tester = RuntimeTester(config, enable_logical_matching=True)

# All enhanced capabilities are available through the main interface
path_matches = tester.get_path_matches(spec_a, spec_b)
matching_report = tester.generate_matching_report(spec_a, spec_b)
validation_results = tester.validate_pipeline_logical_names(pipeline_spec)
```

## Related Documentation

- [Runtime Testing](runtime_testing.md) - Main runtime testing interface with logical matching integration
- [Runtime Models](runtime_models.md) - Base data models extended by logical matching
- [Integration Demo](logical_name_matching_integration_demo.md) - Complete integration example
- [Semantic Matcher](../../core/deps/semantic_matcher.md) - Underlying semantic matching infrastructure
