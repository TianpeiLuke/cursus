---
tags:
  - code
  - deps
  - dependency_resolver
  - specification_matching
  - intelligent_resolution
keywords:
  - UnifiedDependencyResolver
  - dependency resolution
  - specification matching
  - compatibility scoring
  - semantic matching
  - property references
  - pipeline dependencies
topics:
  - dependency resolution
  - specification matching
  - intelligent pipeline assembly
language: python
date of note: 2024-12-07
---

# Dependency Resolver

Unified dependency resolver for intelligent pipeline dependency management that automatically matches step dependencies with compatible outputs from other steps using specification-based compatibility scoring.

## Overview

The `UnifiedDependencyResolver` class provides the core dependency resolution logic that intelligently connects pipeline steps by analyzing their specifications and calculating compatibility scores. This system eliminates the need for manual dependency wiring by automatically matching step inputs to compatible outputs based on multiple criteria including type compatibility, semantic similarity, and source compatibility.

The resolver uses a sophisticated scoring algorithm that evaluates dependency type compatibility (40% weight), data type compatibility (20% weight), semantic name matching with alias support (25% weight), exact name match bonuses (5% weight), compatible source checking with job type normalization (10% weight), and keyword matching (5% weight). This multi-faceted approach ensures accurate and reliable dependency resolution across complex pipeline structures.

The system supports advanced features including resolution caching for performance optimization, detailed compatibility scoring with breakdown analysis, comprehensive resolution reporting for debugging, and integration with semantic matchers for intelligent name similarity calculations.

## Classes and Methods

### Classes
- [`UnifiedDependencyResolver`](#unifieddependencyresolver) - Main dependency resolution engine with intelligent matching
- [`DependencyResolutionError`](#dependencyresolutionerror) - Exception for unresolvable dependencies

### Functions
- [`create_dependency_resolver`](#create_dependency_resolver) - Factory function for creating configured resolver instances

## API Reference

### UnifiedDependencyResolver

_class_ cursus.core.deps.dependency_resolver.UnifiedDependencyResolver(_registry_, _semantic_matcher_)

Intelligent dependency resolver using declarative specifications. This class implements sophisticated dependency resolution logic that automatically matches step dependencies with compatible outputs from other steps using multi-criteria compatibility scoring.

**Parameters:**
- **registry** (_SpecificationRegistry_) – Specification registry containing step specifications for dependency resolution.
- **semantic_matcher** (_SemanticMatcher_) – Semantic matcher for calculating name similarity and alias matching.

```python
from cursus.core.deps.dependency_resolver import UnifiedDependencyResolver
from cursus.core.deps.specification_registry import SpecificationRegistry
from cursus.core.deps.semantic_matcher import SemanticMatcher

# Create resolver components
registry = SpecificationRegistry()
semantic_matcher = SemanticMatcher()

# Create resolver
resolver = UnifiedDependencyResolver(registry, semantic_matcher)

# Register step specifications
resolver.register_specification("preprocessing", preprocessing_spec)
resolver.register_specification("training", training_spec)
```

#### resolve_all_dependencies

resolve_all_dependencies(_available_steps_)

Resolve dependencies for all registered steps. This method processes all available steps and attempts to resolve their dependencies using the intelligent matching algorithm.

**Parameters:**
- **available_steps** (_List[str]_) – List of step names that are available in the pipeline for dependency resolution.

**Returns:**
- **Dict[str, Dict[str, PropertyReference]]** – Dictionary mapping step names to their resolved dependencies as property references.

```python
available_steps = ["preprocessing", "training", "evaluation"]
resolved = resolver.resolve_all_dependencies(available_steps)

for step_name, dependencies in resolved.items():
    print(f"Step {step_name} has {len(dependencies)} resolved dependencies")
```

#### resolve_step_dependencies

resolve_step_dependencies(_consumer_step_, _available_steps_)

Resolve dependencies for a single step. This method analyzes the consumer step's dependency specifications and matches them with compatible outputs from available provider steps.

**Parameters:**
- **consumer_step** (_str_) – Name of the step whose dependencies need to be resolved.
- **available_steps** (_List[str]_) – List of available step names that can provide dependencies.

**Returns:**
- **Dict[str, PropertyReference]** – Dictionary mapping dependency names to property references for resolved dependencies.

```python
dependencies = resolver.resolve_step_dependencies("training", available_steps)
for dep_name, prop_ref in dependencies.items():
    print(f"Dependency {dep_name} resolved to {prop_ref}")
```

#### resolve_with_scoring

resolve_with_scoring(_consumer_step_, _available_steps_)

Resolve dependencies with detailed compatibility scoring. This method provides comprehensive scoring information for both successful and failed dependency resolutions, useful for debugging and optimization.

**Parameters:**
- **consumer_step** (_str_) – Name of the step whose dependencies need to be resolved.
- **available_steps** (_List[str]_) – List of available step names for dependency matching.

**Returns:**
- **Dict[str, Any]** – Dictionary containing resolved dependencies, failed resolutions with scores, and detailed resolution context.

```python
result = resolver.resolve_with_scoring("training", available_steps)
print(f"Resolved: {len(result['resolved'])} dependencies")
print(f"Failed: {len(result['failed_with_scores'])} dependencies")

# Examine failed resolutions
for dep_name, failure_info in result['failed_with_scores'].items():
    best_candidate = failure_info['best_candidate']
    if best_candidate:
        print(f"Best match for {dep_name}: score {best_candidate['score']:.3f}")
```

#### register_specification

register_specification(_step_name_, _spec_)

Register a step specification with the resolver. This method adds a step specification to the registry and clears the resolution cache to ensure fresh calculations.

**Parameters:**
- **step_name** (_str_) – Name of the step to register.
- **spec** (_StepSpecification_) – Step specification containing dependencies and outputs.

```python
resolver.register_specification("new_step", step_specification)
```

#### get_resolution_report

get_resolution_report(_available_steps_)

Generate a detailed resolution report for debugging. This method provides comprehensive information about the resolution process including success rates, unresolved dependencies, and step-by-step analysis.

**Parameters:**
- **available_steps** (_List[str]_) – List of available step names for resolution analysis.

**Returns:**
- **Dict[str, Any]** – Detailed report containing resolution statistics, step details, and summary information.

```python
report = resolver.get_resolution_report(available_steps)
print(f"Resolution rate: {report['resolution_summary']['resolution_rate']:.2%}")
print(f"Steps with errors: {report['resolution_summary']['steps_with_errors']}")

# Examine step details
for step_name, details in report['step_details'].items():
    print(f"{step_name}: {details['resolved_dependencies']} resolved")
```

#### clear_cache

clear_cache()

Clear the resolution cache. This method removes all cached resolution results, forcing fresh calculations on the next resolution request.

```python
resolver.clear_cache()
```

### DependencyResolutionError

_class_ cursus.core.deps.dependency_resolver.DependencyResolutionError

Exception raised when dependencies cannot be resolved. This exception is thrown when required dependencies cannot be matched with compatible outputs from available steps.

```python
try:
    dependencies = resolver.resolve_step_dependencies("training", available_steps)
except DependencyResolutionError as e:
    print(f"Resolution failed: {e}")
```

### create_dependency_resolver

create_dependency_resolver(_registry=None_, _semantic_matcher=None_)

Create a properly configured dependency resolver. This factory function creates a UnifiedDependencyResolver instance with optional registry and semantic matcher components.

**Parameters:**
- **registry** (_Optional[SpecificationRegistry]_) – Optional specification registry. If None, creates a new instance.
- **semantic_matcher** (_Optional[SemanticMatcher]_) – Optional semantic matcher. If None, creates a new instance.

**Returns:**
- **UnifiedDependencyResolver** – Configured dependency resolver instance with all required components.

```python
from cursus.core.deps.dependency_resolver import create_dependency_resolver

# Create with default components
resolver = create_dependency_resolver()

# Create with custom registry
custom_registry = SpecificationRegistry()
resolver = create_dependency_resolver(registry=custom_registry)
```

## Related Documentation

- [Specification Registry](specification_registry.md) - Registry for managing step specifications
- [Semantic Matcher](semantic_matcher.md) - Semantic similarity calculation for dependency matching
- [Property Reference](property_reference.md) - Property reference system for pipeline dependencies
- [Factory](factory.md) - Factory functions for creating dependency resolution components
- [Step Specification](../base/step_specification.md) - Step specification system for dependency definitions
