# Pipeline Registry

## What is the Purpose of Pipeline Registry?

Pipeline Registries serve as **isolated dependency management containers** that enable multiple pipelines to coexist without interference. They represent the evolution from global, singleton registries to pipeline-scoped registries that respect pipeline boundaries.

## Core Purpose

Pipeline Registries provide a **unified pipeline-scoped dependency management system** that combines specification storage with intelligent dependency resolution, enabling:

1. **Pipeline Isolation** - Prevent cross-pipeline interference in multi-pipeline environments
2. **Embedded Dependency Resolution** - Built-in intelligent dependency matching within pipeline scope
3. **Unified API** - Single object provides both registry and resolution capabilities
4. **Centralized Management** - Manage multiple registries through a unified interface
5. **Simplified Architecture** - Eliminate manual coordination between registry and resolver components

## Key Features

### 1. Pipeline-Scoped Registries

Each pipeline gets its own isolated registry instance:

```python
# Create pipeline-specific registries
training_registry = PipelineRegistry("training_pipeline")
inference_registry = PipelineRegistry("inference_pipeline")

# Register specifications in their respective registries
training_registry.register("training_step", TRAINING_SPEC)
inference_registry.register("inference_step", INFERENCE_SPEC)

# No cross-contamination between registries
assert inference_registry.get_specification("training_step") is None
```

### 2. Registry Manager

Centralized management of multiple pipeline registries:

```python
# Get registry for a specific pipeline
pipeline_registry = registry_manager.get_pipeline_registry("my_pipeline")

# List all registered pipelines
pipeline_names = registry_manager.list_pipeline_registries()

# Clear a specific pipeline registry
registry_manager.clear_pipeline_registry("old_pipeline")

# Access the default registry for shared specifications
default_registry = registry_manager.get_default_registry()
```

### 3. Automatic Pipeline Builder Integration

Decorator-based integration with pipeline builders:

```python
@integrate_with_pipeline_builder
class MyPipelineBuilder:
    def __init__(self, base_config):
        self.base_config = base_config
        # self.registry is automatically set to the pipeline-specific registry
        # based on base_config.pipeline_name
    
    def build(self):
        # Use self.registry for dependency resolution
        step = self.create_step("my_step")
        spec = self.registry.get_specification("my_step")
```

### 4. Embedded Dependency Resolution

Each pipeline registry includes built-in dependency resolution capabilities:

```python
# Get pipeline registry with embedded resolver
pipeline_registry = get_pipeline_registry("my_pipeline")

# Register step specifications
pipeline_registry.register("preprocessing", PREPROCESSING_SPEC)
pipeline_registry.register("training", TRAINING_SPEC)
pipeline_registry.register("evaluation", EVALUATION_SPEC)

# Direct dependency resolution through registry
resolved_dependencies = pipeline_registry.resolve_pipeline_dependencies([
    "preprocessing", "training", "evaluation"
])

# Access the embedded resolver if needed for advanced operations
resolver = pipeline_registry.dependency_resolver
resolution_report = resolver.get_resolution_report(["preprocessing", "training"])
```

### 5. Convenience Functions

Simplified access to registries:

```python
# Get registry for a specific pipeline
pipeline_registry = get_pipeline_registry("my_pipeline")

# Get the default registry
default_registry = get_default_registry()
```

## Integration with Other Components

### With Embedded Dependency Resolver

The pipeline registry includes an embedded dependency resolver that provides intelligent dependency matching:

```python
# Get pipeline registry (resolver is embedded)
pipeline_registry = get_pipeline_registry("my_pipeline")

# Register step specifications
pipeline_registry.register("data_loading", DATA_LOADING_SPEC)
pipeline_registry.register("preprocessing", PREPROCESSING_SPEC)
pipeline_registry.register("training", TRAINING_SPEC)

# Direct dependency resolution through registry
resolved = pipeline_registry.resolve_pipeline_dependencies([
    "data_loading", "preprocessing", "training"
])

# Result: 
# {
#   "preprocessing": {
#     "input_data": PropertyReference(step="data_loading", output="processed_data")
#   },
#   "training": {
#     "training_data": PropertyReference(step="preprocessing", output="processed_features")
#   }
# }

# Access embedded resolver for advanced operations
resolver = pipeline_registry.dependency_resolver
compatibility_report = resolver.get_resolution_report(["data_loading", "preprocessing", "training"])
```

### With Step Builders
```python
@integrate_with_pipeline_builder
class XGBoostTrainingStepBuilder:
    def build(self):
        # Access pipeline-specific registry
        spec = self.registry.get_specification("xgboost_training")
        # Build step using specification
```

### With Pipeline Builder
```python
@integrate_with_pipeline_builder
class PipelineBuilder:
    def __init__(self, base_config):
        self.base_config = base_config
        # self.registry is automatically set with embedded resolver
    
    def build_pipeline(self):
        # Register all step specifications
        for step_name, spec in self.step_specifications.items():
            self.registry.register(step_name, spec)
        
        # Resolve dependencies automatically
        resolved_dependencies = self.registry.resolve_pipeline_dependencies(self.steps)
        
        # Build steps with resolved dependencies
        for step_name in self.steps:
            spec = self.registry.get_specification(step_name)
            dependencies = resolved_dependencies.get(step_name, {})
            step = self.build_step_with_dependencies(step_name, spec, dependencies)
```

## Strategic Value

Pipeline Registries enable:

1. **Multi-Pipeline Support**: Run multiple pipelines in the same environment without interference
2. **Unified Architecture**: Single object provides both specification storage and dependency resolution
3. **Simplified API**: Eliminate manual coordination between registry and resolver components
4. **Intelligent Automation**: Built-in dependency resolution with semantic compatibility matching
5. **Reduced Complexity**: Fewer objects to manage and coordinate
6. **Pipeline Isolation**: Changes to one pipeline don't affect others
7. **Contextual Awareness**: Steps and dependencies are resolved within pipeline scope
8. **Backward Compatibility**: Legacy code continues to work with minimal changes

## Example Usage

### Basic Pipeline Registry with Embedded Resolution

```python
# Create and configure pipeline registries
training_registry = get_pipeline_registry("training_pipeline")
inference_registry = get_pipeline_registry("inference_pipeline")

# Register step specifications in their respective registries
training_registry.register("data_loading", DATA_LOADING_SPEC)
training_registry.register("preprocessing", PREPROCESSING_SPEC)
training_registry.register("training", TRAINING_SPEC)

inference_registry.register("model_loading", MODEL_LOADING_SPEC)
inference_registry.register("inference", INFERENCE_SPEC)

# Automatic dependency resolution within each pipeline
training_dependencies = training_registry.resolve_pipeline_dependencies([
    "data_loading", "preprocessing", "training"
])

inference_dependencies = inference_registry.resolve_pipeline_dependencies([
    "model_loading", "inference"
])

print("Training Pipeline Dependencies:")
for step, deps in training_dependencies.items():
    print(f"  {step}: {deps}")

print("Inference Pipeline Dependencies:")
for step, deps in inference_dependencies.items():
    print(f"  {step}: {deps}")
```

### Advanced Pipeline Builder Integration

```python
@integrate_with_pipeline_builder
class TrainingPipelineBuilder:
    def __init__(self, config):
        self.base_config = config
        self.base_config.pipeline_name = "training_pipeline"
        # self.registry is automatically set with embedded resolver
        
        # Define step specifications
        self.step_specifications = {
            "data_loading": DATA_LOADING_SPEC,
            "preprocessing": PREPROCESSING_SPEC,
            "training": TRAINING_SPEC,
            "evaluation": EVALUATION_SPEC
        }
    
    def build(self):
        # Register all specifications
        for step_name, spec in self.step_specifications.items():
            self.registry.register(step_name, spec)
        
        # Automatic dependency resolution
        step_names = list(self.step_specifications.keys())
        resolved_dependencies = self.registry.resolve_pipeline_dependencies(step_names)
        
        # Build steps with resolved dependencies
        steps = []
        for step_name in step_names:
            spec = self.registry.get_specification(step_name)
            dependencies = resolved_dependencies.get(step_name, {})
            
            step = self.create_step_with_dependencies(step_name, spec, dependencies)
            steps.append(step)
        
        return steps
    
    def create_step_with_dependencies(self, step_name, spec, dependencies):
        """Create step with automatically resolved dependencies."""
        # Build step using specification and resolved dependencies
        step_config = self.get_step_config(step_name)
        step_builder = self.get_step_builder(spec.step_type)(step_config)
        
        # Apply resolved dependencies
        inputs = {}
        for dep_name, prop_ref in dependencies.items():
            inputs[dep_name] = prop_ref.get_property_reference()
        
        return step_builder.build_step(inputs)

# Create and run the pipeline
builder = TrainingPipelineBuilder(config)
pipeline_steps = builder.build()

# Get resolution report for debugging
resolution_report = builder.registry.dependency_resolver.get_resolution_report([
    "data_loading", "preprocessing", "training", "evaluation"
])
print(f"Resolution rate: {resolution_report['resolution_summary']['resolution_rate']:.1%}")
```

### Migration from Manual Resolver Pattern

```python
# OLD PATTERN (Manual Coordination)
registry = get_pipeline_registry("my_pipeline")
resolver = UnifiedDependencyResolver(registry)  # Manual resolver creation
resolved = resolver.resolve_all_dependencies(available_steps)

# NEW PATTERN (Embedded Resolver)
registry = get_pipeline_registry("my_pipeline")
resolved = registry.resolve_pipeline_dependencies(available_steps)  # Direct resolution

# Access embedded resolver for advanced operations if needed
resolver = registry.dependency_resolver
detailed_report = resolver.get_resolution_report(available_steps)
```

## Architecture Benefits

### Unified Design Pattern

The embedded resolver pattern provides several architectural advantages:

```python
class PipelineRegistry(SpecificationRegistry):
    """Pipeline registry with embedded dependency resolution."""
    
    def __init__(self, pipeline_name: str):
        super().__init__()
        self.pipeline_name = pipeline_name
        self._dependency_resolver = None  # Lazy-loaded
    
    @property
    def dependency_resolver(self) -> 'UnifiedDependencyResolver':
        """Lazy-loaded dependency resolver for this pipeline."""
        if self._dependency_resolver is None:
            from .dependency_resolver import UnifiedDependencyResolver
            self._dependency_resolver = UnifiedDependencyResolver(self)
        return self._dependency_resolver
    
    def resolve_pipeline_dependencies(self, step_names: List[str]) -> Dict[str, Dict[str, PropertyReference]]:
        """Direct resolution through embedded resolver."""
        return self.dependency_resolver.resolve_all_dependencies(step_names)
```

### Benefits:

1. **Single Responsibility**: Registry manages both specifications and their resolution
2. **Lazy Loading**: Resolver is created only when needed
3. **Automatic Lifecycle**: Resolver lifecycle is managed by the registry
4. **Simplified API**: One object provides complete functionality
5. **Better Encapsulation**: Implementation details are hidden
6. **Reduced Coupling**: No manual coordination required

---

Pipeline Registries represent the **unified architectural foundation** for multi-pipeline environments, combining specification management with intelligent dependency resolution in a single, cohesive component. They work seamlessly with other components like [Step Specifications](step_specification.md), [Smart Proxies](smart_proxy.md), and [Pipeline Builders](pipeline_template_builder_v2.md) to create a robust, scalable, and simplified pipeline architecture.
