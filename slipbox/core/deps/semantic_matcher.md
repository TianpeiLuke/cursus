---
tags:
  - code
  - deps
  - semantic_matcher
  - similarity_calculation
  - intelligent_matching
keywords:
  - SemanticMatcher
  - semantic similarity
  - dependency matching
  - string similarity
  - token overlap
  - synonym matching
  - intelligent resolution
topics:
  - semantic matching
  - similarity algorithms
  - intelligent dependency resolution
language: python
date of note: 2024-12-07
---

# Semantic Matcher

Semantic matching utilities for intelligent dependency resolution that calculate semantic similarity between dependency names and output names to enable intelligent auto-resolution.

## Overview

The `SemanticMatcher` class provides sophisticated algorithms for calculating semantic similarity between names in the dependency resolution system. This class enables intelligent matching by analyzing multiple similarity dimensions including string similarity, token overlap, semantic relationships through synonyms, and substring matching patterns.

The matcher uses a multi-faceted scoring approach that combines string similarity using sequence matching (30% weight), token overlap with Jaccard similarity (25% weight), semantic similarity through synonym dictionaries (25% weight), and substring matching for partial name recognition (20% weight). This comprehensive approach ensures accurate matching across various naming conventions and patterns.

The system supports advanced features including alias-aware similarity calculation for output specifications, synonym dictionaries for domain-specific concepts, abbreviation expansion for common terms, stop word filtering for noise reduction, and detailed similarity explanations for debugging and optimization.

## Classes and Methods

### Classes
- [`SemanticMatcher`](#semanticmatcher) - Semantic similarity matching engine for dependency resolution

## API Reference

### SemanticMatcher

_class_ cursus.core.deps.semantic_matcher.SemanticMatcher()

Semantic similarity matching for dependency resolution. This class implements sophisticated algorithms for calculating semantic similarity between names, enabling intelligent automatic dependency resolution based on name similarity and semantic relationships.

```python
from cursus.core.deps.semantic_matcher import SemanticMatcher

# Create semantic matcher
matcher = SemanticMatcher()

# Calculate similarity between names
similarity = matcher.calculate_similarity("model_artifacts", "trained_model")
print(f"Similarity: {similarity:.3f}")
```

#### calculate_similarity

calculate_similarity(_name1_, _name2_)

Calculate semantic similarity between two names. This method uses a multi-dimensional approach combining string similarity, token overlap, semantic relationships, and substring matching to produce a comprehensive similarity score.

**Parameters:**
- **name1** (_str_) – First name to compare.
- **name2** (_str_) – Second name to compare.

**Returns:**
- **float** – Similarity score between 0.0 and 1.0, where 1.0 indicates identical semantic meaning.

```python
# Compare different naming patterns
similarity1 = matcher.calculate_similarity("training_data", "processed_dataset")
similarity2 = matcher.calculate_similarity("model_config", "hyperparameters")
similarity3 = matcher.calculate_similarity("output_artifacts", "generated_results")

print(f"Training data similarity: {similarity1:.3f}")
print(f"Config similarity: {similarity2:.3f}")
print(f"Output similarity: {similarity3:.3f}")
```

#### calculate_similarity_with_aliases

calculate_similarity_with_aliases(_name_, _output_spec_)

Calculate semantic similarity between a name and an output specification, considering both logical_name and all aliases. This method finds the best match among all possible names in the output specification.

**Parameters:**
- **name** (_str_) – The name to compare (typically the dependency's logical_name).
- **output_spec** (_OutputSpec_) – OutputSpec with logical_name and potential aliases to match against.

**Returns:**
- **float** – The highest similarity score (0.0 to 1.0) between name and any name in output_spec.

```python
from cursus.core.base import OutputSpec, DependencyType

# Create output spec with aliases
output_spec = OutputSpec(
    logical_name="model_artifacts",
    property_path="properties.ModelArtifacts.S3ModelArtifacts",
    output_type=DependencyType.MODEL_ARTIFACTS,
    data_type="S3Uri",
    aliases=["trained_model", "model", "artifacts"]
)

# Calculate similarity with alias support
similarity = matcher.calculate_similarity_with_aliases("trained_model", output_spec)
print(f"Best match similarity: {similarity:.3f}")
# Will match against "trained_model" alias with high score
```

#### find_best_matches

find_best_matches(_target_name_, _candidate_names_, _threshold=0.5_)

Find the best matching names from a list of candidates. This method evaluates all candidates and returns those above the similarity threshold, sorted by score.

**Parameters:**
- **target_name** (_str_) – Name to match against.
- **candidate_names** (_List[str]_) – List of candidate names to evaluate.
- **threshold** (_float_) – Minimum similarity threshold for inclusion. Defaults to 0.5.

**Returns:**
- **List[Tuple[str, float]]** – List of (name, score) tuples sorted by score (highest first).

```python
# Find best matches from candidates
target = "model_artifacts"
candidates = [
    "trained_model",
    "preprocessing_output", 
    "model_config",
    "training_artifacts",
    "evaluation_results"
]

matches = matcher.find_best_matches(target, candidates, threshold=0.3)
for name, score in matches:
    print(f"{name}: {score:.3f}")

# Output:
# trained_model: 0.75
# training_artifacts: 0.68
# model_config: 0.45
```

#### explain_similarity

explain_similarity(_name1_, _name2_)

Provide detailed explanation of similarity calculation. This method breaks down the similarity calculation into its component parts for debugging and optimization purposes.

**Parameters:**
- **name1** (_str_) – First name to compare.
- **name2** (_str_) – Second name to compare.

**Returns:**
- **Dict[str, float]** – Dictionary with detailed similarity breakdown including overall score and component scores.

```python
# Get detailed similarity explanation
explanation = matcher.explain_similarity("model_artifacts", "trained_model")

print(f"Overall score: {explanation['overall_score']:.3f}")
print(f"String similarity: {explanation['string_similarity']:.3f}")
print(f"Token similarity: {explanation['token_similarity']:.3f}")
print(f"Semantic similarity: {explanation['semantic_similarity']:.3f}")
print(f"Substring similarity: {explanation['substring_similarity']:.3f}")
print(f"Normalized names: {explanation['normalized_names']}")
```

## Similarity Components

The SemanticMatcher uses multiple similarity components to calculate comprehensive scores:

### String Similarity (30% weight)
Uses sequence matching to compare character-level similarity between normalized names.

```python
# High string similarity for similar character patterns
matcher.calculate_similarity("preprocessing", "preprocess")  # High score
matcher.calculate_similarity("model_artifacts", "model_artifact")  # High score
```

### Token Overlap (25% weight)
Calculates Jaccard similarity based on word token overlap after normalization.

```python
# High token overlap for shared words
matcher.calculate_similarity("training_data_output", "processed_training_data")  # High score
matcher.calculate_similarity("model_config_params", "hyperparameter_config")  # Medium score
```

### Semantic Similarity (25% weight)
Uses synonym dictionaries to identify semantically related terms.

```python
# Semantic relationships through synonyms
matcher.calculate_similarity("model", "artifact")  # Medium score (synonyms)
matcher.calculate_similarity("config", "parameters")  # Medium score (synonyms)
matcher.calculate_similarity("data", "dataset")  # Medium score (synonyms)
```

### Substring Matching (20% weight)
Identifies partial matches and common substrings between names.

```python
# Substring matching for partial names
matcher.calculate_similarity("model", "trained_model")  # Medium score (substring)
matcher.calculate_similarity("config", "model_config")  # Medium score (substring)
```

## Usage Examples

### Basic Similarity Calculation
```python
from cursus.core.deps.semantic_matcher import SemanticMatcher

# Create matcher
matcher = SemanticMatcher()

# Test various similarity patterns
test_pairs = [
    ("model_artifacts", "trained_model"),
    ("preprocessing_output", "processed_data"),
    ("hyperparameters", "model_config"),
    ("training_data", "dataset"),
    ("evaluation_results", "test_output")
]

for name1, name2 in test_pairs:
    similarity = matcher.calculate_similarity(name1, name2)
    print(f"'{name1}' vs '{name2}': {similarity:.3f}")
```

### Dependency Resolution Integration
```python
# Use in dependency resolution context
def find_compatible_outputs(dependency_name, available_outputs):
    matcher = SemanticMatcher()
    
    compatible = []
    for output_name, output_spec in available_outputs.items():
        similarity = matcher.calculate_similarity_with_aliases(
            dependency_name, output_spec
        )
        
        if similarity > 0.5:  # Threshold for compatibility
            compatible.append((output_name, similarity))
    
    # Sort by similarity score
    compatible.sort(key=lambda x: x[1], reverse=True)
    return compatible

# Example usage
dependency_name = "model_artifacts"
available_outputs = {
    "preprocessing_output": preprocessing_spec,
    "training_model": training_spec,
    "evaluation_results": evaluation_spec
}

matches = find_compatible_outputs(dependency_name, available_outputs)
```

### Debugging Similarity Calculations
```python
# Debug similarity calculation
def debug_similarity(name1, name2):
    matcher = SemanticMatcher()
    explanation = matcher.explain_similarity(name1, name2)
    
    print(f"Comparing '{name1}' vs '{name2}':")
    print(f"  Normalized: {explanation['normalized_names']}")
    print(f"  Overall: {explanation['overall_score']:.3f}")
    print(f"  String: {explanation['string_similarity']:.3f}")
    print(f"  Token: {explanation['token_similarity']:.3f}")
    print(f"  Semantic: {explanation['semantic_similarity']:.3f}")
    print(f"  Substring: {explanation['substring_similarity']:.3f}")

# Debug specific comparisons
debug_similarity("model_artifacts", "trained_model")
debug_similarity("preprocessing_config", "hyperparameters")
```

## Related Documentation

- [Dependency Resolver](dependency_resolver.md) - Uses semantic matcher for intelligent dependency resolution
- [Output Specification](../base/output_spec.md) - Output specifications used in alias-aware matching
- [Factory](factory.md) - Factory functions that create semantic matcher instances
- [Specification Registry](specification_registry.md) - Registry system that benefits from semantic matching
- [Pipeline Assembler](../assembler/pipeline_assembler.md) - Uses semantic matching for step connection
