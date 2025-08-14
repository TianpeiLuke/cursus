---
tags:
  - design
  - dependency
  - resolution
  - system
  - semantic_matching
keywords:
  - dependency resolution
  - semantic matching
  - property references
  - specification registry
  - unified dependency resolver
  - alias support
  - compatibility scoring
  - intelligent matching
topics:
  - dependency management
  - semantic similarity
  - pipeline orchestration
  - specification-driven design
language: python
date of note: 2025-08-12
---

# Dependency Resolution System

## Overview

The Dependency Resolution System is a comprehensive solution for automatically connecting pipeline step dependencies with outputs from previous steps using intelligent matching algorithms. Based on the implementation in `src/cursus/core/deps`, this system enables declarative pipeline construction by automatically resolving step dependencies without requiring explicit wiring.

## Core Architecture

The system consists of several key components working together:

```
┌──────────────────┐    ┌─────────────────────┐    ┌──────────────────┐
│ Specification    │    │ Unified Dependency  │    │ Semantic         │
│ Registry         │◄───┤ Resolver            ├───►│ Matcher          │
└──────────────────┘    └─────────────────────┘    └──────────────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │ Property            │
                        │ References          │
                        └─────────────────────┘
```

### Key Components

1. **UnifiedDependencyResolver** (`dependency_resolver.py`) - Core orchestrator for dependency resolution
2. **SemanticMatcher** (`semantic_matcher.py`) - Calculates semantic similarity between names
3. **SpecificationRegistry** (`specification_registry.py`) - Manages step specifications
4. **PropertyReference** (`property_reference.py`) - Represents resolved dependencies
5. **Factory** (`factory.py`) - Creates configured resolver instances

## Resolution Process

The dependency resolution follows this process:

1. **Registration**: Steps register their specifications with the registry
2. **Candidate Search**: For each dependency, find all potential provider outputs
3. **Compatibility Scoring**: Calculate compatibility scores using multiple criteria
4. **Selection**: Choose the highest-scoring match above threshold (0.5)
5. **Property Reference Creation**: Create property references for resolved dependencies

### Detailed Resolution Algorithm

```python
def resolve_step_dependencies(self, consumer_step: str, available_steps: List[str]) -> Dict[str, PropertyReference]:
    """Resolve dependencies for a single step."""
    consumer_spec = self.registry.get_specification(consumer_step)
    resolved = {}
    
    for dep_name, dep_spec in consumer_spec.dependencies.items():
        # Find best matching output from available steps
        best_match = self._resolve_single_dependency(dep_spec, consumer_step, available_steps)
        if best_match:
            resolved[dep_name] = best_match
        elif dep_spec.required:
            raise DependencyResolutionError(f"Required dependency '{dep_name}' not resolved")
    
    return resolved
```

## Compatibility Scoring System

The system uses a weighted scoring algorithm to determine compatibility between dependencies and outputs:

### Scoring Components (Total: 100%)

1. **Dependency Type Compatibility (40%)**
   - Exact match: 0.4 points
   - Compatible types: 0.2 points
   - Incompatible: 0.0 points (eliminates candidate)

2. **Data Type Compatibility (20%)**
   - Exact match: 0.2 points
   - Compatible types: 0.1 points

3. **Semantic Name Matching with Alias Support (25%)**
   - Uses `SemanticMatcher.calculate_similarity_with_aliases()`
   - Considers both logical_name and all aliases
   - Returns highest similarity score

4. **Exact Match Bonus (5%)**
   - Exact logical_name match: 0.05 points
   - Exact alias match: 0.05 points

5. **Compatible Source Check (10%)**
   - Provider in compatible_sources: 0.1 points
   - No sources specified: 0.05 points

6. **Keyword Matching (5%)**
   - Based on semantic_keywords overlap

### Type Compatibility Matrix

```python
compatibility_matrix = {
    DependencyType.MODEL_ARTIFACTS: [DependencyType.MODEL_ARTIFACTS],
    DependencyType.TRAINING_DATA: [DependencyType.PROCESSING_OUTPUT, DependencyType.TRAINING_DATA],
    DependencyType.PROCESSING_OUTPUT: [DependencyType.PROCESSING_OUTPUT, DependencyType.TRAINING_DATA],
    DependencyType.HYPERPARAMETERS: [DependencyType.HYPERPARAMETERS, DependencyType.CUSTOM_PROPERTY],
    DependencyType.PAYLOAD_SAMPLES: [DependencyType.PAYLOAD_SAMPLES, DependencyType.PROCESSING_OUTPUT],
    DependencyType.CUSTOM_PROPERTY: [DependencyType.CUSTOM_PROPERTY]
}
```

## Semantic Matching with Alias Support

The `SemanticMatcher` provides sophisticated name matching capabilities:

### Enhanced Alias Support

```python
def calculate_similarity_with_aliases(self, name: str, output_spec: OutputSpec) -> float:
    """Calculate similarity considering both logical_name and aliases."""
    # Start with logical_name similarity
    best_score = self.calculate_similarity(name, output_spec.logical_name)
    best_match = output_spec.logical_name
    
    # Check each alias for better matches
    for alias in output_spec.aliases:
        alias_score = self.calculate_similarity(name, alias)
        if alias_score > best_score:
            best_score = alias_score
            best_match = alias
    
    return best_score
```

### Multi-Algorithm Similarity Calculation

The base similarity calculation uses four algorithms:

1. **String Similarity (30%)** - Character-by-character comparison using SequenceMatcher
2. **Token Overlap (25%)** - Jaccard similarity between word sets
3. **Semantic Similarity (25%)** - Synonym and concept matching
4. **Substring Matching (20%)** - Substring containment analysis

### Name Normalization

Names are normalized before comparison:
- Convert to lowercase
- Replace separators (`_`, `-`, `.`) with spaces
- Remove special characters
- Expand abbreviations (`config` → `configuration`)
- Remove stop words

## Alias Support Implementation

The system fully utilizes the `aliases` field in `OutputSpec`:

```python
class OutputSpec(BaseModel):
    logical_name: str = Field(...)
    aliases: List[str] = Field(default_factory=list)
    output_type: DependencyType = Field(...)
    # ...other fields
```

### Use Cases for Aliases

1. **Evolving Naming Standards**:
   ```python
   OutputSpec(
       logical_name="processed_features",  # New standard
       aliases=["processed_data"],         # Legacy name
       # ...
   )
   ```

2. **Cross-Team Compatibility**:
   ```python
   OutputSpec(
       logical_name="model_artifacts",
       aliases=["model_data", "trained_model", "model_output"],
       # ...
   )
   ```

3. **Domain-Specific Terminology**:
   ```python
   OutputSpec(
       logical_name="feature_vectors",    # Technical term
       aliases=["customer_profiles"],     # Business term
       # ...
   )
   ```

## Advanced Features

### Job Type Normalization

The system includes intelligent job type normalization to handle step type variants with job suffixes:

```python
def _normalize_step_type_for_compatibility(self, step_type: str) -> str:
    """
    Normalize step type by removing job type suffixes for compatibility checking.
    
    This handles the classical job type variants issue where step types like
    "TabularPreprocessing_Training" need to be normalized to "TabularPreprocessing"
    for compatibility checking against compatible_sources.
    """
    try:
        # Import here to avoid circular imports
        from src.cursus.steps.registry.step_names import get_step_name_from_spec_type, get_spec_step_type
        
        # Use the registry function to get canonical name, then get the base spec type
        canonical_name = get_step_name_from_spec_type(step_type)
        normalized = get_spec_step_type(canonical_name)
        
        if normalized != step_type:
            logger.debug(f"Normalized step type '{step_type}' -> '{normalized}' for compatibility checking")
        
        return normalized
        
    except Exception as e:
        # Fallback to manual normalization if registry lookup fails
        logger.debug(f"Registry normalization failed for '{step_type}': {e}, using fallback")
        
        job_type_suffixes = ['_Training', '_Testing', '_Validation', '_Calibration']
        for suffix in job_type_suffixes:
            if step_type.endswith(suffix):
                normalized = step_type[:-len(suffix)]
                logger.debug(f"Fallback normalized step type '{step_type}' -> '{normalized}'")
                return normalized
        
        return step_type
```

**Benefits:**
- **Handles Job Type Variants**: Automatically normalizes `TabularPreprocessing_Training` → `TabularPreprocessing`
- **Registry Integration**: Uses centralized step name registry for consistent normalization
- **Fallback Support**: Manual normalization if registry lookup fails
- **Improved Resolution**: Enables proper source compatibility matching for job type variants

### Resolution with Detailed Scoring

```python
def resolve_with_scoring(self, consumer_step: str, available_steps: List[str]) -> Dict[str, any]:
    """Resolve dependencies with detailed compatibility scoring."""
    # Returns detailed information including:
    # - resolved: Successfully resolved dependencies
    # - failed_with_scores: Failed resolutions with candidate details
    # - resolution_details: Context information
```

### Resolution Reporting

```python
def get_resolution_report(self, available_steps: List[str]) -> Dict[str, any]:
    """Generate detailed resolution report for debugging."""
    # Provides comprehensive analysis of resolution process
```

### Caching

The resolver includes intelligent caching:
- Cache key based on consumer step and available steps
- Automatic cache invalidation when specifications change
- Significant performance improvement for repeated resolutions

## Error Handling and Diagnostics

### Exception Types

```python
class DependencyResolutionError(Exception):
    """Raised when dependencies cannot be resolved."""
    pass
```

### Logging and Debugging

The system provides extensive logging:
- Resolution attempts and scores
- Best matches and alternatives
- Detailed scoring breakdowns
- Cache hits and misses

### Resolution Diagnostics

```python
# Example diagnostic output
logger.info(f"Best match for training_data: preprocessing.processed_data (confidence: 0.920)")
logger.debug(f"Alternative matches: [('training.evaluation_output', 0.740)]")
```

## Implementation Improvements

### Historical Issues and Solutions

1. **Conflicting Matches**: Resolved by removing unused outputs from training specifications
2. **Property Reference Handling**: Improved direct logical name matching
3. **Ambiguous Dependencies**: Enhanced with alias support and better scoring

### Key Improvements Made

1. **Removed Unused Outputs**: Cleaned up training specifications to remove confusing outputs
2. **Direct Logical Name Matching**: Changed dependency names to match output names exactly
3. **Enhanced Alias Support**: Full utilization of existing alias fields
4. **Improved Error Messages**: Better diagnostic information for failed resolutions

## Usage Examples

### Basic Resolution

```python
from src.cursus.core.deps.factory import create_dependency_resolver

# Create resolver
resolver = create_dependency_resolver()

# Register specifications
resolver.register_specification("preprocessing", preprocess_spec)
resolver.register_specification("training", training_spec)

# Resolve dependencies
available_steps = ["preprocessing", "training"]
resolved = resolver.resolve_all_dependencies(available_steps)

# Result: {'training': {'training_data': PropertyReference(...)}}
```

### Advanced Resolution with Scoring

```python
# Get detailed resolution information
result = resolver.resolve_with_scoring("training", available_steps)

print(f"Resolved: {result['resolved']}")
print(f"Failed: {result['failed_with_scores']}")
print(f"Details: {result['resolution_details']}")
```

### Resolution Reporting

```python
# Generate comprehensive report
report = resolver.get_resolution_report(available_steps)

print(f"Resolution rate: {report['resolution_summary']['resolution_rate']:.2%}")
print(f"Steps with errors: {report['resolution_summary']['steps_with_errors']}")
```

## Performance Considerations

1. **Caching**: Resolution results are cached to avoid repeated calculations
2. **Early Termination**: Type incompatibility immediately eliminates candidates
3. **Threshold Filtering**: Only candidates above 0.5 threshold are considered
4. **Efficient Algorithms**: Optimized string matching and similarity calculations

## Testing Strategy

The system includes comprehensive testing:

1. **Unit Tests**: Individual component testing (SemanticMatcher, compatibility scoring)
2. **Integration Tests**: Full resolution process testing
3. **Edge Case Testing**: Empty aliases, multiple matches, no matches
4. **Performance Testing**: Large-scale resolution scenarios

## Future Enhancements

1. **Machine Learning Integration**: Learn from successful resolutions to improve scoring
2. **Dynamic Threshold Adjustment**: Adapt thresholds based on pipeline complexity
3. **Conflict Resolution**: Better handling of multiple high-scoring matches
4. **Visualization Tools**: Graphical representation of resolution process

## References

- Implementation: `src/cursus/core/deps/`
- Base Classes: `src/cursus/core/base.py`
- Step Specifications: Pipeline step specification files
- Property References: SageMaker property handling system

## Conclusion

The Dependency Resolution System provides a robust, intelligent solution for automatically connecting pipeline dependencies. Through sophisticated semantic matching, alias support, and comprehensive scoring algorithms, it enables declarative pipeline construction while maintaining flexibility and reliability. The system's implementation in `cursus/core/deps` demonstrates a mature, production-ready approach to dependency management in complex ML pipelines.
