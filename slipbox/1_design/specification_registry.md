---
tags:
  - design
  - implementation
  - pipeline_deps
  - specification_registry
keywords:
  - specification registry
  - context isolation
  - compatibility analysis
  - step specifications
  - dependency resolution
  - registry pattern
  - specification storage
topics:
  - specification management
  - context-aware registries
  - dependency analysis
language: python
date of note: 2025-07-31
---

# Specification Registry

## What is the Purpose of Specification Registry?

Specification Registries serve as **context-aware specification storage containers** that enable isolated, testable, and maintainable dependency management. They represent the evolution from monolithic, global registries to atomized, context-specific registries that respect pipeline and component boundaries.

## Core Purpose

Specification Registries provide a **context-aware specification management system** that enables:

1. **Context Isolation** - Prevent cross-context interference in multi-pipeline and testing environments
2. **Specification Storage** - Centralized storage and retrieval of step specifications
3. **Compatibility Analysis** - Built-in compatibility checking between dependencies and outputs
4. **Testable Architecture** - Enable isolated unit testing of registry functionality
5. **Maintainable Design** - Clear separation of concerns from higher-level orchestration

## Separation of Concerns

The design maintains a clear separation of concerns:

- **SpecificationRegistry** focuses solely on managing specifications within a single context
- **RegistryManager** (separate component) handles the orchestration of multiple registry instances

This separation follows the core design principle: "Each component in the architecture has a specific, well-defined responsibility"

## Key Features

### 1. Context-Aware Registry Creation

Each registry is bound to a specific context for complete isolation:

```python
from src.pipeline_deps import SpecificationRegistry

# Create context-specific registries
training_registry = SpecificationRegistry("training_pipeline")
inference_registry = SpecificationRegistry("inference_pipeline")
test_registry = SpecificationRegistry("unit_test_context")

# Each registry maintains its own isolated state
training_registry.register("data_loading", TRAINING_DATA_LOADING_SPEC)
inference_registry.register("data_loading", INFERENCE_DATA_LOADING_SPEC)

# No cross-contamination between contexts
assert training_registry.get_specification("data_loading") != inference_registry.get_specification("data_loading")
```

### 2. Enhanced Specification Management

Comprehensive specification storage and retrieval with improved error handling:

```python
# Register step specifications
registry = SpecificationRegistry("my_pipeline")

# Register with validation
registry.register("preprocessing", PREPROCESSING_SPEC)
registry.register("training", TRAINING_SPEC)
registry.register("evaluation", EVALUATION_SPEC)

# Unregister when no longer needed
registry.unregister("preprocessing")  # Returns True if successful

# Retrieve specifications
training_spec = registry.get_specification("training")
training_specs = registry.get_specifications_by_type("XGBoostTraining")

# List available specifications
step_names = registry.list_step_names()
step_types = registry.list_step_types()

print(f"Registry contains {len(registry)} steps")  # Pythonic length access
print(f"Training step registered: {'training' in registry}")  # Containment check
```

### 3. Built-in Compatibility Analysis

Intelligent compatibility checking between dependencies and outputs:

```python
# Define a dependency that needs to be satisfied
dependency_spec = DependencySpec(
    logical_name="training_data",
    dependency_type=DependencyType.PROCESSING_OUTPUT,
    required=True,
    compatible_sources=["PreprocessingStep", "DataLoadingStep"],
    semantic_keywords=["processed", "data", "training"],
    data_type="S3Uri"
)

# Find compatible outputs across all registered specifications
compatible_outputs = registry.find_compatible_outputs(dependency_spec)

# Results are scored and sorted by compatibility
for step_name, output_name, output_spec, score in compatible_outputs:
    print(f"Compatible: {step_name}.{output_name} (score: {score:.3f})")
    
# Example output:
# Compatible: preprocessing.processed_data (score: 0.850)
# Compatible: data_loading.raw_data (score: 0.650)
```

### 4. Validation and Error Prevention

Enhanced validation during registration with better error reporting:

```python
# Invalid specification will raise ValueError with detailed message
try:
    registry.register("invalid_step", "not_a_specification")
except ValueError as e:
    print(f"Registration failed: {e}")

# Duplicate step name detection
try:
    registry.register("training", TRAINING_SPEC)
    registry.register("training", OTHER_TRAINING_SPEC)  # Same name
except ValueError as e:
    print(f"Registration failed: {e}")  # "Step name 'training' already registered"

# Specification validation
spec = StepSpecification(
    step_type="TestStep",
    node_type=NodeType.SOURCE,
    dependencies=[],  # SOURCE nodes cannot have dependencies
    outputs=[...]
)

# Validation happens automatically during registration
registry.register("test_step", spec)  # Success
```

### 5. State Management and Persistence

State management capabilities for registry control:

```python
# Clear all specifications from registry
registry.clear()
print(f"Registry now contains {len(registry)} steps")

# Export registry state to dictionary or JSON
state_dict = registry.export_to_dict()
json_state = registry.export_to_json(indent=2)

# Print JSON state for debugging or logging
print(json_state)

# Import state (within same registry)
registry.import_from_dict(state_dict)
```

### 6. Pythonic Interface

More natural Python interface with special methods:

```python
# Iterate through all specifications
for step_name, specification in registry:
    print(f"Step: {step_name}, Type: {specification.step_type}")

# Length checking
if len(registry) > 0:
    print("Registry has specifications")

# Containment checking
if "preprocessing" in registry:
    print("Preprocessing step is registered")
```

## Integration with Other Components

### With Registry Manager

SpecificationRegistry instances are managed by the RegistryManager:

```python
from src.pipeline_deps import get_registry, registry_manager

# Get registry through manager (creates if needed)
pipeline_registry = get_registry("my_pipeline")

# Registry manager coordinates multiple registries
all_contexts = registry_manager.list_contexts()
context_stats = registry_manager.get_context_stats()

# Cleanup when needed
registry_manager.clear_context("old_pipeline")
```

### With Dependency Resolver

SpecificationRegistry provides the foundation for dependency resolution:

```python
from src.pipeline_deps import UnifiedDependencyResolver

# Create resolver with specific registry
registry = SpecificationRegistry("training_pipeline")
resolver = UnifiedDependencyResolver(registry)

# Register specifications
registry.register("data_loading", DATA_LOADING_SPEC)
registry.register("preprocessing", PREPROCESSING_SPEC)
registry.register("training", TRAINING_SPEC)

# Resolve dependencies using registry data
resolved_dependencies = resolver.resolve_all_dependencies([
    "data_loading", "preprocessing", "training"
])
```

### With Step Builders

Step builders can query registries for specifications:

```python
class XGBoostTrainingStepBuilder(BuilderStepBase):
    def __init__(self, config, registry=None):
        super().__init__(config)
        self.registry = registry or get_registry("default")
    
    @classmethod
    def get_specification(cls) -> StepSpecification:
        return XGBOOST_TRAINING_SPEC
    
    def build_step(self):
        # Query registry for compatible inputs
        spec = self.get_specification()
        for dep_name, dep_spec in spec.dependencies.items():
            compatible = self.registry.find_compatible_outputs(dep_spec)
            if compatible:
                best_match = compatible[0]  # Highest scored match
                # Use best match for step construction
```

## Strategic Value

Specification Registries enable:

1. **Testability**: Complete isolation enables comprehensive unit testing
2. **Maintainability**: Clear separation of concerns from orchestration logic
3. **Scalability**: Context-specific registries scale independently
4. **Flexibility**: Multiple registries can coexist without interference
5. **Reliability**: Built-in validation prevents invalid configurations
6. **Debuggability**: Context-aware logging and error reporting
7. **Composability**: Registries can be combined and coordinated as needed

## Architecture Benefits

### Focused Design Pattern

The focused registry pattern provides several architectural advantages:

```python
class SpecificationRegistry:
    """Context-aware specification storage with built-in compatibility analysis."""
    
    def __init__(self, context_name: str):
        self.context_name = context_name
        self._specifications: Dict[str, StepSpecification] = {}
        self._step_type_to_names: Dict[str, List[str]] = {}
        self._logger = logging.getLogger(f"{__name__}.{context_name}")
    
    def register(self, step_name: str, specification: StepSpecification):
        """Register specification with validation."""
        if not isinstance(specification, StepSpecification):
            raise ValueError(f"Expected StepSpecification, got {type(specification)}")
        
        if step_name in self._specifications:
            raise ValueError(f"Step name '{step_name}' already registered")
            
        self._specifications[step_name] = specification
        
        # Track step type
        step_type = specification.step_type
        if step_type not in self._step_type_to_names:
            self._step_type_to_names[step_type] = []
        self._step_type_to_names[step_type].append(step_name)
        
        self._logger.info(f"Registered specification for step '{step_name}'")
    
    def unregister(self, step_name: str) -> bool:
        """Unregister a specification."""
        if step_name not in self._specifications:
            return False
            
        spec = self._specifications[step_name]
        step_type = spec.step_type
        
        # Remove from type mapping
        if step_type in self._step_type_to_names:
            self._step_type_to_names[step_type].remove(step_name)
            if not self._step_type_to_names[step_type]:
                del self._step_type_to_names[step_type]
        
        # Remove from specifications
        del self._specifications[step_name]
        self._logger.info(f"Unregistered specification for step '{step_name}'")
        return True
    
    def clear(self):
        """Clear all specifications."""
        self._specifications.clear()
        self._step_type_to_names.clear()
        self._logger.info("Cleared all specifications")
    
    def find_compatible_outputs(self, dependency_spec: DependencySpec) -> List[Tuple[str, str, OutputSpec, float]]:
        """Find compatible outputs with scoring."""
        # Implementation provides intelligent compatibility analysis
        # Returns scored and sorted results
        
    def __len__(self) -> int:
        """Get count of registered specifications."""
        return len(self._specifications)
        
    def __contains__(self, step_name: str) -> bool:
        """Check if step name exists in registry."""
        return step_name in self._specifications
        
    def __iter__(self):
        """Iterate through step names and specifications."""
        return iter(self._specifications.items())
```

### Benefits:

1. **Single Responsibility**: Registry focuses solely on specification management
2. **Context Awareness**: Each registry knows its context and scope
3. **Built-in Intelligence**: Compatibility analysis is embedded in the registry
4. **Testable Design**: Easy to create isolated test registries
5. **Clear Interfaces**: Well-defined API for specification management
6. **Pythonic Usage**: Natural Python interface with special methods
7. **Extensible**: Easy to add new compatibility algorithms

## Example Usage

### Basic Registry Operations

```python
from src.pipeline_deps import SpecificationRegistry, StepSpecification, DependencySpec, OutputSpec

# Create registry for specific context
registry = SpecificationRegistry("fraud_detection_pipeline")

# Define step specifications
data_loading_spec = StepSpecification(
    step_type="DataLoadingStep",
    node_type=NodeType.SOURCE,
    dependencies=[],
    outputs=[
        OutputSpec(
            logical_name="raw_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['RawData'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Raw transaction data"
        )
    ]
)

preprocessing_spec = StepSpecification(
    step_type="PreprocessingStep",
    node_type=NodeType.INTERNAL,
    dependencies=[
        DependencySpec(
            logical_name="input_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["DataLoadingStep"],
            semantic_keywords=["data", "input", "raw"],
            data_type="S3Uri"
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="processed_features",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['ProcessedFeatures'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Processed feature data"
        )
    ]
)

# Register specifications
registry.register("data_loading", data_loading_spec)
registry.register("preprocessing", preprocessing_spec)

# Query registry
print(f"Registered steps: {registry.list_step_names()}")
print(f"Step types: {registry.list_step_types()}")

# Find compatible outputs
input_dependency = DependencySpec(
    logical_name="input_data",
    dependency_type=DependencyType.PROCESSING_OUTPUT,
    data_type="S3Uri"
)

compatible = registry.find_compatible_outputs(input_dependency)
for step_name, output_name, output_spec, score in compatible:
    print(f"Compatible: {step_name}.{output_name} (score: {score:.3f})")
```

### Advanced Compatibility Analysis

```python
# Create registry with multiple step types
registry = SpecificationRegistry("ml_pipeline")

# Register various step specifications
registry.register("data_loading", DATA_LOADING_SPEC)
registry.register("feature_engineering", FEATURE_ENGINEERING_SPEC)
registry.register("xgboost_training", XGBOOST_TRAINING_SPEC)
registry.register("pytorch_training", PYTORCH_TRAINING_SPEC)
registry.register("model_evaluation", MODEL_EVALUATION_SPEC)

# Define complex dependency with multiple criteria
complex_dependency = DependencySpec(
    logical_name="model_artifacts",
    dependency_type=DependencyType.MODEL_ARTIFACTS,
    required=True,
    compatible_sources=["XGBoostTraining", "PyTorchTraining"],
    semantic_keywords=["model", "trained", "artifacts"],
    data_type="S3Uri",
    description="Trained model artifacts for evaluation"
)

# Find all compatible outputs
compatible_outputs = registry.find_compatible_outputs(complex_dependency)

print(f"Found {len(compatible_outputs)} compatible outputs:")
for step_name, output_name, output_spec, score in compatible_outputs:
    step_spec = registry.get_specification(step_name)
    print(f"  {step_name} ({step_spec.step_type})")
    print(f"    Output: {output_name}")
    print(f"    Score: {score:.3f}")
    print(f"    Type: {output_spec.output_type.value}")
    print()

# Get specifications by type
training_steps = registry.get_specifications_by_type("XGBoostTraining")
print(f"XGBoost training steps: {len(training_steps)}")
```

### Testing with Isolated Registries

```python
import unittest
from src.pipeline_deps import SpecificationRegistry

class TestSpecificationRegistry(unittest.TestCase):
    def setUp(self):
        """Create isolated test registry."""
        self.registry = SpecificationRegistry("test_context")
        
    def test_specification_registration(self):
        """Test specification registration and retrieval."""
        # Register test specification
        self.registry.register("test_step", TEST_STEP_SPEC)
        
        # Verify registration
        self.assertIn("test_step", self.registry)
        self.assertEqual(len(self.registry), 1)
        
        # Verify retrieval
        retrieved_spec = self.registry.get_specification("test_step")
        self.assertEqual(retrieved_spec.step_type, "TestStep")
    
    def test_compatibility_analysis(self):
        """Test compatibility checking."""
        # Register compatible specifications
        self.registry.register("source", SOURCE_SPEC)
        self.registry.register("consumer", CONSUMER_SPEC)
        
        # Test compatibility
        dependency = DependencySpec(
            logical_name="test_input",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            data_type="S3Uri"
        )
        
        compatible = self.registry.find_compatible_outputs(dependency)
        self.assertGreater(len(compatible), 0)
        
        # Verify scoring
        step_name, output_name, output_spec, score = compatible[0]
        self.assertGreater(score, 0.5)
        
    def test_state_management(self):
        """Test registry state management."""
        # Register specifications
        self.registry.register("step1", TEST_STEP_SPEC)
        self.registry.register("step2", OTHER_TEST_SPEC)
        
        # Export state
        state = self.registry.export_to_dict()
        
        # Clear and verify
        self.registry.clear()
        self.assertEqual(len(self.registry), 0)
        
        # Import and verify restoration
        self.registry.import_from_dict(state)
        self.assertEqual(len(self.registry), 2)
        self.assertIn("step1", self.registry)
        self.assertIn("step2", self.registry)
```

## Migration from Monolithic Registry

### Old Pattern (Monolithic)
```python
# Old monolithic approach
from src.pipeline_deps import PipelineRegistry

# Single global registry
global_registry = PipelineRegistry()
global_registry.register("step1", SPEC1)
global_registry.register("step2", SPEC2)

# All pipelines share the same registry
# Risk of cross-pipeline interference
```

### New Pattern (Atomized)
```python
# New atomized approach
from src.pipeline_deps import get_registry

# Context-specific registries
training_registry = get_registry("training_pipeline")
inference_registry = get_registry("inference_pipeline")

# Each pipeline has isolated registry
training_registry.register("training_step", TRAINING_SPEC)
inference_registry.register("inference_step", INFERENCE_SPEC)

# Complete isolation between contexts
# No risk of cross-pipeline interference
```

### Enhanced Features

```python
# New capabilities
registry.unregister("step1")  # Remove specific step
registry.clear()  # Remove all steps

# Export/import
state = registry.export_to_dict()
registry.import_from_dict(state)

# Pythonic interface
for step_name, spec in registry:
    print(f"{step_name}: {spec.step_type}")
    
if "step1" in registry:
    print("Step1 is registered")
```

---

Specification Registries represent the **atomized foundation** for context-aware specification management, with a clear separation of concerns from the Registry Manager that orchestrates multiple registry instances. This design aligns with the core design principles of separation of concerns, specification-driven design, and maintainable architecture.
