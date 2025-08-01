# Design Principles

## What is the Purpose of Design Principles?

Design Principles serve as the **architectural philosophy** that guides the development and evolution of the ML pipeline system. They represent the core beliefs and strategic decisions that shape how components interact, how complexity is managed, and how the system evolves over time.

## Core Purpose

Design Principles provide the **architectural philosophy layer** that:

1. **Architectural Philosophy and Guidelines** - Establish fundamental beliefs about system design
2. **Consistency Across Components** - Ensure coherent design decisions throughout the system
3. **Decision-Making Framework** - Guide trade-offs and architectural choices
4. **Evolution Strategy** - Provide direction for system growth and adaptation
5. **Quality Assurance Foundation** - Establish standards for code quality and maintainability

## Key Design Principles

### 1. Declarative Over Imperative

**Principle**: Favor declarative specifications over imperative implementations.

**Rationale**: Declarative approaches are more maintainable, testable, and enable intelligent automation.

```python
# Imperative Approach (Discouraged)
def build_training_pipeline():
    data_step = DataLoadingStep()
    data_step.set_input_path("s3://bucket/data")
    data_step.set_output_path("s3://bucket/processed")
    
    preprocess_step = PreprocessingStep()
    preprocess_step.set_input(data_step.get_output())
    preprocess_step.configure_transformations(["normalize", "encode"])
    
    training_step = TrainingStep()
    training_step.set_input(preprocess_step.get_output())
    training_step.set_hyperparameters({"max_depth": 6})
    
    return Pipeline([data_step, preprocess_step, training_step])

# Declarative Approach (Preferred)
TRAINING_PIPELINE_SPEC = PipelineSpecification(
    name="training_pipeline",
    steps=[
        StepSpecification(
            step_type="DataLoading",
            node_type=NodeType.SOURCE,
            outputs=["processed_data"]
        ),
        StepSpecification(
            step_type="Preprocessing",
            node_type=NodeType.INTERNAL,
            dependencies=["processed_data"],
            outputs=["features"]
        ),
        StepSpecification(
            step_type="Training",
            node_type=NodeType.SINK,
            dependencies=["features"]
        )
    ]
)
```

**Benefits**:
- Specifications can be analyzed, validated, and optimized
- Enable automatic code generation and intelligent tooling
- Separate "what" from "how" for better maintainability
- Support multiple implementation strategies

### 2. Composition Over Inheritance

**Principle**: Favor composition and dependency injection over deep inheritance hierarchies.

**Rationale**: Composition provides better flexibility, testability, and reduces coupling.

```python
# Inheritance Approach (Discouraged)
class BaseStep:
    def execute(self): pass
    def validate(self): pass

class ProcessingStep(BaseStep):
    def execute(self): 
        # Processing logic mixed with base functionality
        pass

class XGBoostTrainingStep(ProcessingStep):
    def execute(self):
        # Training logic mixed with processing and base functionality
        pass

# Composition Approach (Preferred)
class StepBuilder:
    def __init__(self, config: ConfigBase, validator: Validator, executor: Executor):
        self.config = config          # Injected configuration
        self.validator = validator    # Injected validation logic
        self.executor = executor      # Injected execution logic
    
    def build_step(self, inputs):
        self.validator.validate(inputs)
        return self.executor.execute(self.config, inputs)

class XGBoostTrainingStepBuilder(StepBuilder):
    def __init__(self, config: XGBoostConfig):
        super().__init__(
            config=config,
            validator=XGBoostValidator(),
            executor=XGBoostExecutor()
        )
```

**Benefits**:
- Components can be tested in isolation
- Easy to swap implementations (e.g., different validators)
- Reduces coupling between components
- Supports dependency injection patterns

### 3. Fail Fast and Explicit

**Principle**: Detect and report errors as early as possible with explicit, actionable messages.

**Rationale**: Early error detection reduces debugging time and prevents cascading failures.

```python
# Implicit Failure (Discouraged)
def connect_steps(source_step, target_step):
    # Silently fails or produces cryptic errors
    target_step.input = source_step.output
    return target_step

# Explicit Failure (Preferred)
def connect_steps(source_step: StepProxy, target_step: StepProxy, output_name: str = None):
    """Connect steps with explicit validation and clear error messages"""
    
    # Validate step compatibility
    source_spec = source_step.get_specification()
    target_spec = target_step.get_specification()
    
    if not source_spec.outputs:
        raise ConnectionError(
            f"Source step '{source_step.step_name}' has no outputs. "
            f"Available step types with outputs: {get_steps_with_outputs()}"
        )
    
    compatible_outputs = find_compatible_outputs(source_spec, target_spec)
    if not compatible_outputs:
        raise ConnectionError(
            f"No compatible outputs found between '{source_step.step_name}' and '{target_step.step_name}'. "
            f"Source outputs: {list(source_spec.outputs.keys())} "
            f"Target dependencies: {list(target_spec.dependencies.keys())} "
            f"Suggestion: Check output types and dependency requirements."
        )
    
    # Proceed with connection
    return create_connection(source_step, target_step, compatible_outputs)
```

**Benefits**:
- Reduces debugging time with clear error messages
- Prevents silent failures that are hard to diagnose
- Provides actionable suggestions for fixing issues
- Improves developer experience

### 4. Single Responsibility Principle

**Principle**: Each component should have a single, well-defined responsibility.

**Rationale**: Single responsibility improves maintainability, testability, and reduces coupling.

```python
# Multiple Responsibilities (Discouraged)
class XGBoostStep:
    def __init__(self):
        pass
    
    def validate_inputs(self, inputs): pass      # Validation responsibility
    def load_data(self, path): pass             # Data loading responsibility
    def preprocess_data(self, data): pass       # Preprocessing responsibility
    def train_model(self, data): pass           # Training responsibility
    def save_model(self, model): pass           # Persistence responsibility
    def create_sagemaker_step(self): pass       # SageMaker integration responsibility

# Single Responsibility (Preferred)
class InputValidator:
    """Responsible only for input validation"""
    def validate(self, inputs, specification): pass

class XGBoostEstimatorFactory:
    """Responsible only for creating XGBoost estimators"""
    def create_estimator(self, config): pass

class SageMakerStepFactory:
    """Responsible only for creating SageMaker steps"""
    def create_training_step(self, estimator, inputs): pass

class XGBoostTrainingStepBuilder:
    """Responsible only for orchestrating XGBoost training step creation"""
    def __init__(self, validator: InputValidator, 
                 estimator_factory: XGBoostEstimatorFactory,
                 step_factory: SageMakerStepFactory):
        self.validator = validator
        self.estimator_factory = estimator_factory
        self.step_factory = step_factory
    
    def build_step(self, config, inputs):
        self.validator.validate(inputs, self.get_specification())
        estimator = self.estimator_factory.create_estimator(config)
        return self.step_factory.create_training_step(estimator, inputs)
```

**Benefits**:
- Each component is easier to understand and test
- Changes to one responsibility don't affect others
- Components can be reused in different contexts
- Reduces complexity and coupling

### 5. Open/Closed Principle

**Principle**: Components should be open for extension but closed for modification.

**Rationale**: Enables system evolution without breaking existing functionality.

```python
# Modification Required (Discouraged)
class StepBuilder:
    def build_step(self, step_type, config):
        if step_type == "xgboost":
            return self._build_xgboost_step(config)
        elif step_type == "pytorch":
            return self._build_pytorch_step(config)
        elif step_type == "sklearn":  # New type requires modification
            return self._build_sklearn_step(config)
        else:
            raise ValueError(f"Unknown step type: {step_type}")

# Extension Without Modification (Preferred)
class StepBuilderRegistry:
    def __init__(self):
        self._builders = {}
    
    def register_builder(self, step_type: str, builder_class: Type[BuilderStepBase]):
        """Register new builder without modifying existing code"""
        self._builders[step_type] = builder_class
    
    def create_builder(self, step_type: str, config) -> BuilderStepBase:
        if step_type not in self._builders:
            raise ValueError(f"Unknown step type: {step_type}")
        
        builder_class = self._builders[step_type]
        return builder_class(config)

# Usage - Adding new step types without modification
registry = StepBuilderRegistry()
registry.register_builder("xgboost", XGBoostTrainingStepBuilder)
registry.register_builder("pytorch", PyTorchTrainingStepBuilder)
registry.register_builder("sklearn", SklearnTrainingStepBuilder)  # Extension
```

**Benefits**:
- New functionality can be added without changing existing code
- Reduces risk of introducing bugs in stable components
- Supports plugin architectures and extensibility
- Enables third-party extensions

### 6. Dependency Inversion Principle

**Principle**: Depend on abstractions, not concretions.

**Rationale**: Reduces coupling and improves testability and flexibility.

```python
# Concrete Dependencies (Discouraged)
class XGBoostTrainingStepBuilder:
    def __init__(self, config):
        self.config = config
        self.s3_client = boto3.client('s3')           # Concrete dependency
        self.sagemaker_client = boto3.client('sagemaker')  # Concrete dependency
    
    def build_step(self, inputs):
        # Tightly coupled to AWS services
        data = self.s3_client.get_object(...)
        return self.sagemaker_client.create_training_job(...)

# Abstract Dependencies (Preferred)
class XGBoostTrainingStepBuilder:
    def __init__(self, config: XGBoostConfig, 
                 storage_service: StorageService,      # Abstract dependency
                 ml_service: MLService):               # Abstract dependency
        self.config = config
        self.storage_service = storage_service
        self.ml_service = ml_service
    
    def build_step(self, inputs):
        # Loosely coupled to abstractions
        data = self.storage_service.get_data(inputs['data_path'])
        return self.ml_service.create_training_job(self.config, data)

# Concrete implementations injected at runtime
class S3StorageService(StorageService):
    def get_data(self, path): pass

class SageMakerMLService(MLService):
    def create_training_job(self, config, data): pass
```

**Benefits**:
- Components can be tested with mock implementations
- Easy to swap implementations (e.g., different cloud providers)
- Reduces coupling between layers
- Supports dependency injection frameworks

### 7. Convention Over Configuration

**Principle**: Provide sensible defaults and conventions to reduce configuration burden.

**Rationale**: Reduces cognitive load and setup time while maintaining flexibility.

```python
# Excessive Configuration (Discouraged)
config = XGBoostTrainingStepConfig(
    entry_point="train.py",
    source_dir="./code",
    framework_version="1.5-1",
    py_version="py38",
    instance_type="ml.m5.xlarge",
    instance_count=1,
    volume_size_in_gb=30,
    max_runtime_in_seconds=7200,
    output_path="s3://bucket/output",
    checkpoint_s3_uri="s3://bucket/checkpoints",
    enable_sagemaker_metrics=True,
    metric_definitions=[
        {"Name": "train:auc", "Regex": "train-auc:(\\S+)"},
        {"Name": "validation:auc", "Regex": "validation-auc:(\\S+)"}
    ]
)

# Convention-Based Configuration (Preferred)
config = XGBoostTrainingStepConfig(
    # Only specify what differs from conventions
    hyperparameters={"max_depth": 6, "eta": 0.3}
    # Conventions automatically applied:
    # - entry_point="train.py" (standard name)
    # - framework_version=latest_stable
    # - instance_type based on data size estimation
    # - standard metric definitions for XGBoost
    # - output paths follow naming conventions
)

class XGBoostTrainingStepConfig(ConfigBase):
    def __init__(self, **kwargs):
        # Apply conventions first
        self.apply_xgboost_conventions()
        
        # Override with user-specified values
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def apply_xgboost_conventions(self):
        """Apply standard conventions for XGBoost training"""
        self.entry_point = "train.py"
        self.framework_version = self.get_latest_stable_version()
        self.metric_definitions = self.get_standard_xgboost_metrics()
        self.instance_type = self.estimate_instance_type()
```

**Benefits**:
- Reduces setup time and configuration complexity
- Provides good defaults for common use cases
- Still allows customization when needed
- Improves consistency across projects

### 8. Explicit Dependencies

**Principle**: Make dependencies explicit and visible in interfaces.

**Rationale**: Improves understanding, testability, and prevents hidden coupling.

```python
# Hidden Dependencies (Discouraged)
class TrainingStepBuilder:
    def build_step(self, inputs):
        # Hidden dependency on global configuration
        role = os.environ['SAGEMAKER_ROLE']
        
        # Hidden dependency on global registry
        estimator_class = GLOBAL_ESTIMATOR_REGISTRY.get('xgboost')
        
        # Hidden dependency on global S3 client
        s3_client = get_global_s3_client()
        
        return TrainingStep(...)

# Explicit Dependencies (Preferred)
class TrainingStepBuilder:
    def __init__(self, 
                 config: TrainingStepConfig,           # Explicit config dependency
                 estimator_factory: EstimatorFactory,  # Explicit factory dependency
                 storage_service: StorageService):     # Explicit storage dependency
        self.config = config
        self.estimator_factory = estimator_factory
        self.storage_service = storage_service
    
    def build_step(self, inputs: Dict[str, Any]) -> TrainingStep:
        """Build training step with explicit dependencies"""
        estimator = self.estimator_factory.create_estimator(self.config)
        return TrainingStep(
            name=self.config.step_name,
            estimator=estimator,
            inputs=inputs
        )
```

**Benefits**:
- Dependencies are clear from the interface
- Easier to test with mock dependencies
- Prevents hidden global state issues
- Improves code understanding and maintenance

## Integration Principles

### 1. Layered Architecture

**Principle**: Organize components into clear layers with defined responsibilities.

```python
# Layer Structure
"""
┌─────────────────────────────────────┐
│           Fluent API Layer          │  # User-facing natural language interface
├─────────────────────────────────────┤
│          Smart Proxy Layer          │  # Intelligent abstraction and automation
├─────────────────────────────────────┤
│         Step Builder Layer          │  # Implementation bridge to SageMaker
├─────────────────────────────────────┤
│       Configuration Layer           │  # Centralized configuration management
├─────────────────────────────────────┤
│       Specification Layer           │  # Declarative metadata and contracts
├─────────────────────────────────────┤
│         Foundation Layer            │  # DAG, registry, core utilities
└─────────────────────────────────────┘
"""
```

**Benefits**:
- Clear separation of concerns
- Dependencies flow in one direction (downward)
- Each layer can be tested independently
- Enables incremental development and refactoring

### 2. Registry Pattern

**Principle**: Use registries for component discovery and loose coupling.

```python
class ComponentRegistry:
    def __init__(self):
        self._specifications = {}
        self._builders = {}
        self._configs = {}
    
    def register_specification(self, step_type: str, spec: StepSpecification):
        self._specifications[step_type] = spec
    
    def register_builder(self, step_type: str, builder_class: Type[BuilderStepBase]):
        self._builders[step_type] = builder_class
    
    def get_compatible_outputs(self, dependency_spec: DependencySpec) -> List[OutputSpec]:
        """Find compatible outputs across all registered specifications"""
        compatible = []
        for spec in self._specifications.values():
            for output_spec in spec.outputs.values():
                if self._is_compatible(output_spec, dependency_spec):
                    compatible.append(output_spec)
        return compatible
```

**Benefits**:
- Enables component discovery and introspection
- Supports plugin architectures
- Reduces coupling between components
- Enables intelligent automation and tooling

## Strategic Value

Design Principles provide:

1. **Architectural Consistency**: Ensure coherent design decisions across the system
2. **Maintainability**: Guide decisions that improve long-term maintainability
3. **Extensibility**: Enable system growth without architectural debt
4. **Quality Assurance**: Establish standards for code quality and design
5. **Team Alignment**: Provide shared understanding of design philosophy
6. **Decision Framework**: Guide trade-offs and architectural choices

## Example Application

```python
# Applying design principles in practice
class PipelineBuilder:
    """Example of design principles in action"""
    
    def __init__(self, 
                 registry: ComponentRegistry,        # Explicit dependency
                 config_factory: ConfigFactory,     # Explicit dependency
                 validator: PipelineValidator):      # Explicit dependency
        self.registry = registry                    # Dependency injection
        self.config_factory = config_factory
        self.validator = validator
    
    def build_pipeline(self, pipeline_spec: PipelineSpecification) -> Pipeline:
        """Build pipeline following design principles"""
        
        # Fail fast - validate specification early
        validation_errors = self.validator.validate_specification(pipeline_spec)
        if validation_errors:
            raise ValidationError(f"Invalid pipeline specification: {validation_errors}")
        
        # Use registry for component discovery (Open/Closed principle)
        steps = []
        for step_spec in pipeline_spec.steps:
            builder_class = self.registry.get_builder(step_spec.step_type)
            if not builder_class:
                raise ValueError(
                    f"No builder registered for step type '{step_spec.step_type}'. "
                    f"Available types: {list(self.registry.get_available_step_types())}"
                )
            
            # Apply conventions for configuration (Convention over Configuration)
            config = self.config_factory.create_config(step_spec)
            
            # Use composition and dependency injection
            builder = builder_class(config)
            steps.append(builder)
        
        return Pipeline(steps)
```

Design Principles serve as the **architectural DNA** of the system, encoding the fundamental beliefs and strategic decisions that guide how components are designed, how they interact, and how the system evolves over time to maintain quality and extensibility.
