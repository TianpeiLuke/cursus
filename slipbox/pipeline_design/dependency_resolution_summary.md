# Dependency Resolution: Connecting Step Dependencies to Outputs

**Date:** July 8, 2025  
**Topic:** How DependencySpec connects to OutputSpec  
**Status:** ✅ IMPLEMENTED (Alias Support Added)

## Core Mechanism Overview

The dependency resolution system automatically connects a step's dependencies (what it needs) to outputs from previous steps (what they provide) using intelligent matching rather than requiring exact name matches.

### Key Components

1. **DependencySpec** - Defines input requirements for a step
2. **OutputSpec** - Defines what outputs a step produces
3. **UnifiedDependencyResolver** - Orchestrates the matching process
4. **SemanticMatcher** - Calculates name similarities and matches

## Resolution Process

```
┌──────────────┐     ┌────────────────┐      ┌──────────────┐
│ Previous Step│     │Dependency      │      │ Consumer Step│
│ (OutputSpec) │────▶│Resolver+Matcher│─────▶│(DependencySpec)
└──────────────┘     └────────────────┘      └──────────────┘
```

1. **Identification**: Consumer step needs input "training_data"
2. **Candidate Search**: Resolver finds all available outputs from previous steps
3. **Scoring**: Each output gets a compatibility score (0.0-1.0) based on:
   - Type compatibility (40%)
   - Data type compatibility (20%)
   - Name similarity with alias support (25%)
   - Compatible source check (10%)
   - Keyword matching (5%)
4. **Selection**: Highest scoring output above threshold (0.5) is chosen
5. **Connection**: PropertyReference created linking consumer to provider

## Semantic Name Matching with Alias Support (25% of total score)

The SemanticMatcher now uses a new method `calculate_similarity_with_aliases()` that checks both the logical name and all aliases, returning the highest similarity score:

```python
def calculate_similarity_with_aliases(self, name: str, output_spec: OutputSpec) -> float:
    """Find highest similarity score among logical_name and all aliases."""
    # Start with logical_name similarity
    best_score = self.calculate_similarity(name, output_spec.logical_name)
    best_match = output_spec.logical_name
    
    # Check each alias and take the best score
    for alias in output_spec.aliases:
        alias_score = self.calculate_similarity(name, alias)
        if alias_score > best_score:
            best_score = alias_score
            best_match = alias
            
    # Log which name gave the best match
    if best_score > 0.5:
        logger.debug(f"Best match for '{name}': '{best_match}' (score: {best_score:.3f})")
        
    return best_score
```

The base similarity calculation uses:

1. **String Similarity (30%)**: How similar the strings are character by character
2. **Token Overlap (25%)**: How many words they share
3. **Semantic Similarity (25%)**: If words are synonyms or related
4. **Substring Matching (20%)**: If one is contained in the other

## Alias Support Implementation

The OutputSpec class already includes an `aliases` field that is now fully utilized:

```python
class OutputSpec(BaseModel):
    logical_name: str = Field(...)
    aliases: List[str] = Field(default_factory=list)
    output_type: DependencyType = Field(...)
    # ...other fields
```

### Key Benefits

1. **Better Matching**: Finds connections that would otherwise be missed
2. **Backward Compatibility**: Supports evolving naming conventions while maintaining older ones
3. **Developer Flexibility**: Can define multiple names for outputs without breaking connections
4. **Perfect Matches**: Exact alias matches receive the same score as logical name matches

## Example: How Alias Support Improves Matching

### Before Enhancement

A training step with dependency `training_data` would get a moderate match (~0.61) with a preprocessing step's output `processed_data`:

```
Similarity 'training_data' vs 'processed_data': 0.612
```

### After Enhancement

If preprocessing output includes `"training_data"` in its aliases:

```python
OutputSpec(
    logical_name="processed_data",
    aliases=["training_data", "model_input"],
    # ...
)
```

Then we get a perfect match (1.0) plus an exact match bonus, resulting in a significantly higher score:

```
Similarity 'training_data' vs 'training_data' (alias): 1.0
Total score with exact match bonus: 1.05 (capped at 1.0)
```

## Use Cases for Aliases

1. **Evolution of Naming Standards**:
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
       aliases=["model_data", "trained_model"],
       # ...
   )
   ```

3. **Technical/Business Terminology**:
   ```python
   OutputSpec(
       logical_name="feature_vectors",
       aliases=["customer_profiles"], 
       # ...
   )
   ```

## Testing Strategy

The implementation has been thoroughly tested:

1. **Unit Tests**: Verifying `calculate_similarity_with_aliases()` works correctly
2. **Integration Tests**: Testing the full dependency resolution process with aliases
3. **Handling Edge Cases**: Empty alias lists, multiple similar matches, etc.

This enhancement makes full use of the existing aliases field in OutputSpec that was previously underutilized, significantly improving the robustness of the pipeline dependency resolution system.
