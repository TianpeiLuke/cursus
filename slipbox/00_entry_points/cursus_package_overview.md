---
tags:
  - entry_point
  - overview
  - architecture
  - package
  - analysis
keywords:
  - cursus package
  - ML pipeline system
  - specification-driven design
  - automatic pipeline generation
  - system architecture
topics:
  - package overview
  - system architecture
  - design analysis
language: python
date of note: 2025-10-23
---

# Cursus Package Overview and Analysis

## Executive Summary

Cursus is a sophisticated ML pipeline orchestration system that implements a **specification-driven design** philosophy to enable automatic pipeline generation, standardized step sharing, and intelligent dependency resolution. The system transforms the traditional imperative approach to pipeline construction into a declarative, maintainable, and highly automated framework.

## What is Cursus?

Cursus is a comprehensive ML pipeline framework that addresses the core challenges of modern machine learning workflows:

- **Automatic Pipeline Generation**: Declarative specifications automatically resolve dependencies and assemble pipelines
- **Standardized Step Library**: Reusable, well-tested components that promote knowledge sharing and accelerate experimentation
- **Intelligent Dependency Resolution**: Semantic matching automatically connects compatible steps
- **Specification-Driven Architecture**: Separates "what" (specifications) from "how" (implementations) for better maintainability

### Core Value Proposition

1. **Efficiency**: Reduce pipeline development time from days to hours through automatic generation
2. **Standardization**: Promote consistent, reusable components across teams and projects
3. **Knowledge Sharing**: Enable rapid experimentation through shared, validated step implementations
4. **Maintainability**: Declarative specifications are easier to understand, modify, and extend

## System Architecture

Cursus implements a layered architecture with five core subsystems:

```
┌─────────────────────────────────────────────────────────────┐
│                    I/O System                               │
│  PipelineDAG • DAG Config Factory • Config Field Manager   │
├─────────────────────────────────────────────────────────────┤
│                 Orchestrator System                         │
│  Pipeline Assembler • DAG Compiler • Pipeline Templates    │
├─────────────────────────────────────────────────────────────┤
│                Step Library System                          │
│  Step Implementations • Registry • Step Catalog            │
├─────────────────────────────────────────────────────────────┤
│                   Core System                               │
│  Dependency Resolver • Semantic Matcher • Specifications   │
├─────────────────────────────────────────────────────────────┤
│              Validation & Workspace System                  │
│  Alignment Validation • Builder Validation • Workspace     │
└─────────────────────────────────────────────────────────────┘
```

## Core System Components

### 1. Orchestrator System (`cursus/core/compiler` & `cursus/core/assembler`)

The orchestrator system is the central coordination layer that transforms declarative specifications into executable pipelines.

**Key Components:**
- **PipelineDAGCompiler**: Compiles pipeline DAGs into executable SageMaker pipelines
- **Dynamic Pipeline Template**: Enables runtime pipeline generation based on specifications
- **Pipeline Template Base**: Provides the foundation for template-based pipeline construction
- **Pipeline Assembler**: The core orchestrator that coordinates all pipeline assembly operations

**Functionality:**
```python
# Example: Automatic pipeline assembly from specifications
template = XGBoostPipelineTemplate(
    config_path="configs/pipeline_config.json",
    sagemaker_session=sagemaker_session,
    role=role
)
pipeline = template.generate_pipeline()  # Automatic assembly
```

**Benefits:**
- Transforms complex pipeline construction into simple template instantiation
- Enables automatic optimization and validation during compilation
- Supports both static and dynamic pipeline generation patterns

### 2. Step Library System (`cursus/steps`, `cursus/registry`, `cursus/step_catalog`)

The step library system provides a comprehensive catalog of reusable, standardized ML pipeline components.

**Key Components:**
- **Step Implementations**: Complete implementations for common ML operations (training, preprocessing, evaluation)
- **Step Registry**: Centralized registry for step types, hyperparameters, and metadata
- **Step Catalog**: Auto-discovery system that automatically finds and registers step components
- **Pipeline Catalog**: Extended catalog system for sharing complete pipeline structures

**Step Structure:**
```
cursus/steps/
├── specs/          # Declarative step specifications
├── builders/       # Step builders that create SageMaker steps
├── configs/        # Configuration classes for each step type
├── contracts/      # Script contracts defining input/output expectations
├── scripts/        # Actual implementation scripts
└── hyperparams/    # Hyperparameter definitions and defaults
```

**Example Step Specification:**
```python
XGBOOST_TRAINING_SPEC = StepSpecification(
    step_type="XGBoostTraining",
    node_type=NodeType.INTERNAL,
    dependencies=[
        DependencySpec(
            logical_name="input_path",
            dependency_type=DependencyType.TRAINING_DATA,
            required=True,
            compatible_sources=["TabularPreprocessing", "DataLoad"],
            semantic_keywords=["data", "training", "processed"]
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="model_output",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts"
        )
    ]
)
```

**Benefits:**
- Standardized, tested implementations reduce development time
- Auto-discovery eliminates manual registration overhead
- Shared specifications promote consistency across teams
- Extensible architecture supports custom step types

### 3. Core System (`cursus/core/deps`)

The core system implements the intelligent dependency resolution that enables automatic pipeline assembly.

**Key Components:**
- **Dependency Resolver**: Automatically matches step outputs to compatible inputs
- **Semantic Matcher**: Uses keyword-based matching to find compatible connections
- **Specification Registry**: Maintains metadata about all available steps and their capabilities
- **Property Reference System**: Handles SageMaker property path resolution

**Automatic Dependency Resolution:**
```python
# The resolver automatically connects these steps:
data_loading_step = DataLoadingStepBuilder(config)
preprocessing_step = PreprocessingStepBuilder(config)
training_step = XGBoostTrainingStepBuilder(config)

# Automatic connection based on semantic matching:
# data_loading.output["processed_data"] -> preprocessing.input["input_data"]
# preprocessing.output["features"] -> training.input["training_data"]
```

**Semantic Matching Example:**
```python
# Output specification
OutputSpec(
    logical_name="processed_data",
    semantic_keywords=["data", "processed", "tabular", "features"]
)

# Dependency specification  
DependencySpec(
    logical_name="training_input",
    semantic_keywords=["data", "input", "training", "features"]
)

# Automatic match: "data" + "features" keywords overlap
```

**Benefits:**
- Eliminates manual input/output wiring
- Reduces pipeline construction errors
- Enables intelligent pipeline optimization
- Supports complex multi-step dependency chains

### 4. Validation & Workspace System (`cursus/validation`, `cursus/workspace`)

The validation system ensures pipeline correctness and consistency across different environments.

**Key Components:**
- **Alignment Validation**: Ensures specifications, contracts, builders, and configs are properly aligned
- **Builder Validation**: Validates step builder implementations against specifications
- **Script Testing**: Tests script implementations for correctness and compatibility
- **Workspace System**: Manages multi-developer environments and configuration isolation

**Validation Layers:**
```python
# Four-level validation hierarchy
Level1: Script Contract Alignment    # Scripts match their contracts
Level2: Contract Specification Alignment  # Contracts match specifications  
Level3: Specification Dependency Alignment  # Dependencies are satisfiable
Level4: Builder Configuration Alignment  # Builders implement specifications correctly
```

**Benefits:**
- Prevents runtime errors through comprehensive validation
- Ensures consistency between specifications and implementations
- Supports multi-developer workflows with workspace isolation
- Provides clear error messages and debugging guidance

### 5. I/O System (`cursus/api/dag`, `cursus/api/factory`, `cursus/core/config_fields`)

The I/O system manages pipeline definitions, user interactions, and configuration management.

**Key Components:**
- **PipelineDAG**: Core data structure representing pipeline topology as a directed acyclic graph
- **DAG Config Factory**: Provides interactive widgets for collecting user input and building configurations
- **Config Field Manager**: Manages loading, saving, and validation of configuration files

**PipelineDAG Structure:**
```python
class PipelineDAG:
    def __init__(self, nodes: List[str] = None, edges: List[tuple] = None):
        self.nodes = nodes or []           # Step names
        self.edges = edges or []           # (from_step, to_step) tuples
        self.adj_list = {}                 # Adjacency list representation
        self.reverse_adj = {}              # Reverse adjacency for dependency lookup
    
    def topological_sort(self) -> List[str]:
        """Return nodes in execution order"""
        # Kahn's algorithm implementation
```

**Interactive Configuration:**
```python
# DAG Config Factory provides Jupyter widgets for pipeline configuration
factory = DAGConfigFactory()
config_widget = factory.create_config_widget(pipeline_template="xgboost_training")
# Users interact with widgets to specify hyperparameters, data paths, etc.
final_config = factory.build_config_from_widget(config_widget)
```

**Benefits:**
- Clean separation between pipeline logic and data representation
- Interactive configuration reduces setup complexity
- Centralized config management ensures consistency
- DAG representation enables analysis and optimization

## Design Principles

Cursus is built on a foundation of proven software engineering principles:

### 1. Declarative Over Imperative
- **Principle**: Favor declarative specifications over imperative implementations
- **Benefit**: Specifications can be analyzed, validated, and optimized automatically
- **Example**: Define what steps are needed and their dependencies; the system figures out how to connect them

### 2. Composition Over Inheritance
- **Principle**: Use composition and dependency injection instead of deep inheritance hierarchies
- **Benefit**: Better testability, flexibility, and reduced coupling
- **Example**: Step builders compose validators, executors, and config managers rather than inheriting complex behavior

### 3. Fail Fast and Explicit
- **Principle**: Detect errors early with clear, actionable messages
- **Benefit**: Reduces debugging time and prevents cascading failures
- **Example**: Specification validation catches incompatible dependencies before pipeline execution

### 4. Single Responsibility Principle
- **Principle**: Each component has one well-defined responsibility
- **Benefit**: Easier testing, maintenance, and understanding
- **Example**: Separate classes for validation, execution, and configuration management

### 5. Open/Closed Principle
- **Principle**: Open for extension, closed for modification
- **Benefit**: New functionality without breaking existing code
- **Example**: Registry pattern allows new step types without modifying core system

### 6. Convention Over Configuration
- **Principle**: Provide sensible defaults and conventions
- **Benefit**: Reduces setup complexity while maintaining flexibility
- **Example**: Standard naming conventions for scripts, configs, and outputs

### Anti-Over-Engineering Principles

### 7. Demand Validation Principle
- **Principle**: Validate actual user demand before implementing complex features
- **Benefit**: Prevents building sophisticated solutions for theoretical problems
- **Framework**: Require evidence of user requests and problem reports before feature development

### 8. Simplicity First Principle
- **Principle**: Start with the simplest solution, add complexity only when validated
- **Benefit**: Faster development, easier maintenance, better performance
- **Guideline**: Prefer 50-line simple solutions over 500-line complex ones unless complexity is proven necessary

### 9. Evidence-Based Architecture
- **Principle**: Base architectural decisions on evidence of actual usage patterns
- **Benefit**: Avoids solving non-existent problems
- **Practice**: Collect usage analytics and user feedback before architectural changes

## The Message Passing Algorithm and Automatic Pipeline Generation

The heart of Cursus's automatic pipeline generation capability lies in its sophisticated message passing algorithm implemented in the PipelineAssembler. This system demonstrates how specification-driven design, step builder patterns, and intelligent dependency resolution work together to create a fully automated pipeline construction system.

### Message Passing Architecture

The PipelineAssembler implements a **message-driven communication system** between step builders that enables automatic dependency matching without manual intervention. This architecture follows these key principles:

#### 1. Specification-Based Message Propagation

```python
def _propagate_messages(self) -> None:
    """
    Initialize step connections using the dependency resolver.
    
    This method analyzes the DAG structure and uses the dependency resolver
    to intelligently match inputs to outputs based on specifications.
    """
    # Process each edge in the DAG
    for src_step, dst_step in self.dag.edges:
        src_builder = self.step_builders[src_step]
        dst_builder = self.step_builders[dst_step]
        
        # Let resolver match outputs to inputs
        for dep_name, dep_spec in dst_builder.spec.dependencies.items():
            matches = []
            
            # Check if source step can provide this dependency
            for out_name, out_spec in src_builder.spec.outputs.items():
                compatibility = resolver._calculate_compatibility(
                    dep_spec, out_spec, src_builder.spec
                )
                if compatibility > 0.5:  # Compatibility threshold
                    matches.append((out_name, out_spec, compatibility))
            
            # Use best match if found
            if matches:
                matches.sort(key=lambda x: x[2], reverse=True)
                best_match = matches[0]
                
                # Store connection message
                self.step_messages[dst_step][dep_name] = {
                    "source_step": src_step,
                    "source_output": best_match[0],
                    "match_type": "specification_match",
                    "compatibility": best_match[2],
                }
```

#### 2. Intelligent Compatibility Scoring

The system uses semantic matching to calculate compatibility scores between step outputs and inputs:

```python
# Example compatibility calculation
compatibility = resolver._calculate_compatibility(dep_spec, out_spec, src_builder.spec)

# Factors considered:
# - Semantic keyword overlap (e.g., "data", "training", "processed")
# - Data type compatibility (e.g., S3Uri, JsonPath)
# - Dependency type matching (e.g., TRAINING_DATA, MODEL_ARTIFACTS)
# - Compatible source validation
```

#### 3. Message-Based Step Instantiation

When instantiating steps, the assembler uses stored messages to automatically wire inputs:

```python
def _instantiate_step(self, step_name: str) -> Step:
    """Create step with automatic input wiring from messages"""
    
    # Extract inputs from stored messages
    inputs = {}
    if step_name in self.step_messages:
        for input_name, message in self.step_messages[step_name].items():
            src_step = message["source_step"]
            src_output = message["source_output"]
            
            # Create PropertyReference for SageMaker property paths
            prop_ref = PropertyReference(
                step_name=src_step, 
                output_spec=output_spec
            )
            runtime_prop = prop_ref.to_runtime_property(self.step_instances)
            inputs[input_name] = runtime_prop
    
    # Delegate to step builder with resolved inputs
    return builder.create_step(inputs=inputs, outputs=outputs, dependencies=dependencies)
```

### Step Builder Design Patterns

The message passing system leverages several key design patterns in step builders:

#### 1. Specification-Driven Interface

Each step builder implements a standardized specification that declares its dependencies and outputs:

```python
XGBOOST_TRAINING_SPEC = StepSpecification(
    step_type="XGBoostTraining",
    dependencies=[
        DependencySpec(
            logical_name="input_path",
            dependency_type=DependencyType.TRAINING_DATA,
            semantic_keywords=["data", "training", "processed"],
            compatible_sources=["TabularPreprocessing", "DataLoad"]
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="model_output",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts"
        )
    ]
)
```

#### 2. Dependency Injection Pattern

Step builders receive dependency resolution components through constructor injection:

```python
class XGBoostTrainingStepBuilder(StepBuilderBase):
    def __init__(self, config, sagemaker_session, role, 
                 registry_manager, dependency_resolver):
        self.config = config
        self.registry_manager = registry_manager
        self.dependency_resolver = dependency_resolver
        self.spec = XGBOOST_TRAINING_SPEC  # Specification attachment
```

#### 3. Message-Aware Step Creation

Step builders implement a standardized `create_step` method that accepts resolved inputs from the message passing system:

```python
def create_step(self, inputs=None, outputs=None, dependencies=None, **kwargs):
    """Create step with message-resolved inputs"""
    
    # Use specification to validate inputs
    if self.spec:
        validated_inputs = self.extract_inputs_from_dependencies(dependencies)
        inputs.update(validated_inputs)
    
    # Create SageMaker step with resolved connections
    return TrainingStep(
        name=self._get_step_name(),
        estimator=self._create_estimator(),
        inputs=self._process_training_inputs(inputs),
        depends_on=dependencies
    )
```

### Data Structures Enabling Automation

The automatic pipeline generation relies on several key data structures:

#### 1. PipelineDAG - Topology Representation

```python
class PipelineDAG:
    def __init__(self, nodes=None, edges=None):
        self.nodes = nodes or []           # Step names
        self.edges = edges or []           # (from_step, to_step) tuples
        self.adj_list = {}                 # Adjacency list for traversal
        self.reverse_adj = {}              # Reverse adjacency for dependencies
    
    def topological_sort(self) -> List[str]:
        """Return execution order using Kahn's algorithm"""
```

#### 2. Step Messages - Connection Storage

```python
# Message structure storing step connections
self.step_messages: DefaultDict[str, Dict[str, Any]] = defaultdict(dict)

# Example message:
{
    "training_step": {
        "input_path": {
            "source_step": "preprocessing_step",
            "source_output": "processed_data",
            "match_type": "specification_match",
            "compatibility": 0.85
        }
    }
}
```

#### 3. Specification Registry - Metadata Repository

```python
# Registry maintains step specifications for dependency resolution
registry.register_specification("XGBoostTraining", XGBOOST_TRAINING_SPEC)
registry.register_specification("TabularPreprocessing", PREPROCESSING_SPEC)

# Enables semantic matching across all registered steps
```

### The Complete Automation Flow

The integration of these components creates a fully automated pipeline generation process:

#### Phase 1: Initialization
1. **DAG Analysis**: Parse pipeline topology from PipelineDAG
2. **Builder Creation**: Instantiate step builders with dependency injection
3. **Specification Registration**: Register all step specifications with resolver

#### Phase 2: Message Propagation
1. **Edge Processing**: Analyze each DAG edge for potential connections
2. **Compatibility Calculation**: Score semantic compatibility between outputs and inputs
3. **Message Storage**: Store best matches in step_messages structure
4. **Conflict Resolution**: Handle multiple potential matches by selecting highest compatibility

#### Phase 3: Step Instantiation
1. **Topological Ordering**: Determine execution order using DAG topological sort
2. **Message Resolution**: Convert stored messages to SageMaker PropertyReferences
3. **Step Creation**: Delegate to step builders with resolved inputs and outputs
4. **Pipeline Assembly**: Combine instantiated steps into SageMaker Pipeline

### Key Advantages of This Architecture

1. **Zero Manual Wiring**: Developers never specify input/output connections manually
2. **Semantic Intelligence**: System understands step compatibility through keyword matching
3. **Extensible Design**: New step types automatically integrate through specifications
4. **Error Prevention**: Incompatible connections are detected before pipeline execution
5. **Distributed Construction**: Each step builder operates independently while coordinating through messages

This sophisticated message passing architecture demonstrates how specification-driven design enables truly automatic pipeline generation, transforming complex manual pipeline construction into a declarative, maintainable, and highly automated process.

## Benefits of Using Cursus

### 1. Efficient Automatic Pipeline Generation

**Traditional Approach:**
```python
# Manual pipeline construction (100+ lines)
data_step = ProcessingStep(
    name="data-loading",
    processor=SKLearnProcessor(...),
    inputs=[ProcessingInput(...)],
    outputs=[ProcessingOutput(...)],
    code="data_loading.py"
)

preprocess_step = ProcessingStep(
    name="preprocessing", 
    processor=SKLearnProcessor(...),
    inputs=[ProcessingInput(source=data_step.properties.ProcessingOutputConfig.Outputs["output"].S3Output.S3Uri)],
    outputs=[ProcessingOutput(...)],
    code="preprocessing.py"
)

training_step = TrainingStep(
    name="training",
    estimator=XGBoost(...),
    inputs={"training": TrainingInput(s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs["output"].S3Output.S3Uri)}
)

pipeline = Pipeline(
    name="ml-pipeline",
    steps=[data_step, preprocess_step, training_step]
)
```

**Cursus Approach:**
```python
# Automatic pipeline generation (10 lines)
template = XGBoostPipelineTemplate(
    config_path="configs/pipeline_config.json"
)
pipeline = template.generate_pipeline()  # Automatic dependency resolution and assembly
```

**Benefits:**
- **90% reduction in code**: From 100+ lines to 10 lines
- **Zero manual wiring**: Dependencies resolved automatically
- **Built-in validation**: Specifications validated before execution
- **Optimization opportunities**: System can optimize pipeline structure

### 2. Shareable and Standardized Steps

**Knowledge Sharing:**
```python
# Teams can share validated step implementations
from cursus.steps.specs import XGBOOST_TRAINING_SPEC
from cursus.steps.builders import XGBoostTrainingStepBuilder

# Instant access to battle-tested implementations
# No need to reimplement common ML operations
# Consistent interfaces across all step types
```

**Standardization Benefits:**
- **Consistent interfaces**: All steps follow the same specification pattern
- **Validated implementations**: Steps are tested and proven in production
- **Documentation included**: Specifications serve as living documentation
- **Version control**: Step evolution is tracked and managed

### 3. Fast Experimentation

**Rapid Prototyping:**
```python
# Experiment with different algorithms by changing one line
configs = {
    'Training': XGBoostTrainingConfig(max_depth=6),  # Try XGBoost
    # 'Training': PyTorchTrainingConfig(epochs=10),  # Or PyTorch
    # 'Training': LightGBMTrainingConfig(num_leaves=31)  # Or LightGBM
}

pipeline = template.generate_pipeline(configs)
```

**A/B Testing Support:**
```python
# Easy to create pipeline variants for comparison
baseline_pipeline = create_pipeline(algorithm="xgboost", features="basic")
experiment_pipeline = create_pipeline(algorithm="pytorch", features="advanced")
```

**Benefits:**
- **Algorithm swapping**: Change algorithms without rewriting pipelines
- **Feature experimentation**: Easy to test different preprocessing approaches
- **Hyperparameter optimization**: Built-in support for parameter sweeps
- **Reproducible experiments**: Configurations are versioned and shareable

### 4. Reduced Maintenance Burden

**Centralized Updates:**
```python
# Update step implementation once, benefits all pipelines using it
# No need to update dozens of individual pipeline definitions
# Backward compatibility maintained through specification versioning
```

**Error Prevention:**
```python
# Automatic validation prevents common errors:
# - Missing dependencies
# - Type mismatches  
# - Invalid configurations
# - Broken property references
```

**Benefits:**
- **Single source of truth**: Step implementations maintained centrally
- **Automatic error detection**: Validation catches issues before execution
- **Consistent updates**: All users benefit from improvements automatically
- **Reduced debugging**: Clear error messages and validation feedback

## Package Structure

```
src/cursus/
├── api/                    # I/O System
│   ├── dag/               # Pipeline DAG definitions
│   │   ├── base_dag.py           # Core DAG data structure
│   │   └── pipeline_dag_resolver.py  # DAG analysis and resolution
│   └── factory/           # Configuration factories
│       └── dag_config_factory.py    # Interactive config widgets
│
├── core/                  # Core Systems
│   ├── assembler/         # Orchestrator System - Pipeline Assembly
│   │   ├── pipeline_assembler.py    # Main pipeline orchestrator
│   │   └── pipeline_template_base.py # Base template class
│   ├── compiler/          # Orchestrator System - Pipeline Compilation
│   │   ├── dag_compiler.py          # DAG to pipeline compiler
│   │   ├── dynamic_template.py      # Runtime template generation
│   │   └── validation.py            # Compilation validation
│   ├── deps/              # Core System - Dependency Resolution
│   │   ├── dependency_resolver.py   # Main dependency resolver
│   │   ├── semantic_matcher.py      # Semantic keyword matching
│   │   ├── specification_registry.py # Step specification registry
│   │   └── property_reference.py    # SageMaker property handling
│   ├── config_fields/     # I/O System - Configuration Management
│   │   └── [config field managers]  # Config loading/saving/validation
│   └── base/              # Foundation classes
│       └── [base classes]           # Common base implementations
│
├── steps/                 # Step Library System
│   ├── specs/             # Step specifications (declarative metadata)
│   ├── builders/          # Step builders (SageMaker step creation)
│   ├── configs/           # Configuration classes
│   ├── contracts/         # Script contracts (input/output definitions)
│   ├── scripts/           # Implementation scripts
│   └── hyperparams/       # Hyperparameter definitions
│
├── step_catalog/          # Step Library System - Auto-discovery
│   ├── step_catalog.py           # Main catalog system
│   ├── spec_discovery.py         # Specification discovery
│   ├── builder_discovery.py      # Builder discovery
│   ├── config_discovery.py       # Configuration discovery
│   └── contract_discovery.py     # Contract discovery
│
├── registry/              # Step Library System - Registration
│   ├── step_names.py             # Step type registry
│   ├── hyperparameter_registry.py # Hyperparameter registry
│   └── hybrid/                   # Hybrid registry components
│
├── pipeline_catalog/      # Extended catalog for pipeline sharing
│   └── [pipeline catalog components]
│
├── validation/            # Validation & Workspace System
│   └── [validation components]    # Multi-level validation framework
│
├── workspace/             # Validation & Workspace System  
│   └── [workspace components]     # Multi-developer support
│
├── cli/                   # Command-line interface
├── mods/                  # MODS-specific extensions
└── processing/            # Processing utilities
```

## Component Functionality Details

### Orchestrator System Components

**PipelineAssembler** (`cursus/core/assembler/pipeline_assembler.py`):
- Coordinates the entire pipeline assembly process using a sophisticated message passing algorithm
- Implements message-driven communication between step builders to enable automatic dependency matching
- Orchestrates step creation by passing dependency information between builders through structured messages
- Manages automatic dependency resolution by facilitating message exchanges that allow builders to discover and connect to compatible outputs
- Handles error recovery and validation during assembly through message-based feedback loops
- Enables distributed pipeline construction where each step builder operates independently while coordinating through the message passing system

**PipelineTemplateBase** (`cursus/core/assembler/pipeline_template_base.py`):
- Provides the foundational base class for all pipeline templates
- Defines the standard interface for template-based pipeline construction
- Implements common template functionality including DAG creation, config mapping, and step builder registration
- Enables consistent template patterns across different pipeline types

**DAGCompiler** (`cursus/core/compiler/dag_compiler.py`):
- Compiles PipelineDAG structures into executable SageMaker pipelines
- Optimizes pipeline structure for performance and cost
- Validates DAG correctness and dependency satisfaction

**Dynamic Template** (`cursus/core/compiler/dynamic_template.py`):
- Enables runtime pipeline generation based on user specifications
- Supports conditional pipeline construction based on data characteristics
- Allows for adaptive pipeline behavior

### Step Library System Components

**Step Catalog** (`cursus/step_catalog/step_catalog.py`):
- Automatically discovers and registers step components
- Provides search and filtering capabilities for step discovery
- Maintains metadata about step capabilities and requirements

**Auto-Discovery Components**:
The step catalog system implements five specialized discovery components that automatically find and register step components:

- **Spec Discovery** (`cursus/step_catalog/spec_discovery.py`):
  - Automatically discovers step specifications across the codebase
  - Scans for StepSpecification objects and registers them with metadata
  - Handles specification validation and dependency analysis
  - Enables dynamic specification loading without manual registration

- **Builder Discovery** (`cursus/step_catalog/builder_discovery.py`):
  - Automatically finds and registers step builder classes
  - Identifies classes inheriting from StepBuilderBase
  - Maps builder classes to their corresponding step types
  - Supports plugin-style architecture for extensible step implementations

- **Config Discovery** (`cursus/step_catalog/config_discovery.py`):
  - Discovers configuration classes for each step type
  - Automatically maps config classes to their corresponding steps
  - Validates configuration schemas and default values
  - Enables automatic config generation and validation

- **Contract Discovery** (`cursus/step_catalog/contract_discovery.py`):
  - Finds and registers script contracts defining input/output expectations
  - Maps contracts to their corresponding step implementations
  - Validates contract alignment with specifications
  - Ensures consistency between script expectations and step definitions

- **Script Discovery** (`cursus/step_catalog/script_discovery.py`):
  - Automatically discovers and registers step implementation scripts
  - Scans for Python scripts that implement step functionality
  - Maps scripts to their corresponding step types and contracts
  - Validates script compatibility with specifications and contracts
  - Enables dynamic script loading and execution

**Adapters** (`cursus/step_catalog/adapters/`):
- Provide integration layers between different component types
- Handle format conversions and compatibility mappings
- Enable seamless integration between specifications, builders, configs, and contracts
- Support legacy component integration and migration paths

**Registry System** (`cursus/registry/`):
- Centralized registration of step types and hyperparameters
- Provides validation and consistency checking for registered components
- Supports versioning and evolution of step definitions

### Core System Components

**Dependency Resolver** (`cursus/core/deps/dependency_resolver.py`):
- Implements the core automatic dependency resolution algorithm
- Handles complex multi-step dependency chains
- Provides intelligent error messages for unsatisfiable dependencies

**Semantic Matcher** (`cursus/core/deps/semantic_matcher.py`):
- Performs keyword-based matching between step outputs and inputs
- Supports fuzzy matching and similarity scoring
- Enables intelligent connection suggestions

### I/O System Components

**PipelineDAG** (`cursus/api/dag/base_dag.py`):
- Core data structure representing pipeline topology as a directed acyclic graph
- Implements graph operations including topological sorting for execution order
- Provides dependency tracking and adjacency list management
- Enables pipeline analysis and optimization through graph algorithms

**Pipeline DAG Resolver** (`cursus/api/dag/pipeline_dag_resolver.py`):
- Analyzes pipeline DAG structures for optimization opportunities
- Resolves complex dependency relationships and execution patterns
- Identifies parallel execution opportunities and bottlenecks
- Provides DAG validation and consistency checking

**DAG Config Factory** (`cursus/api/factory/dag_config_factory.py`):
- Provides interactive Jupyter widgets for pipeline configuration
- Enables user-friendly configuration collection and validation
- Supports dynamic config generation based on pipeline templates
- Handles config serialization and deserialization for persistence

**Config Field Manager** (`cursus/core/config_fields/`):
- Manages loading, saving, and validation of configuration files
- Provides centralized config field management and type validation
- Supports hierarchical configuration structures and inheritance
- Enables config portability across different environments and deployments

## Usage Patterns

### 1. Template-Based Pipeline Creation
```python
# Define pipeline structure declaratively
template = XGBoostPipelineTemplate(
    config_path="configs/xgboost_config.json"
)

# Generate pipeline automatically
pipeline = template.generate_pipeline()

# Execute pipeline
execution = pipeline.start()
```

### 2. Custom Step Development
```python
# 1. Define specification
CUSTOM_STEP_SPEC = StepSpecification(
    step_type="CustomProcessing",
    dependencies=[...],
    outputs=[...]
)

# 2. Implement builder
class CustomStepBuilder(StepBuilderBase):
    def create_step(self, **kwargs):
        # Implementation
        pass

# 3. Register with system
registry.register_builder("CustomProcessing", CustomStepBuilder)
```

### 3. Interactive Configuration
```python
# Create interactive configuration widget
factory = DAGConfigFactory()
widget = factory.create_config_widget("xgboost_training")

# User interacts with widget in Jupyter notebook
# Generate final configuration
config = factory.build_config_from_widget(widget)
```

## References

### Design Documentation
- [Specification-Driven Design](../1_design/specification_driven_design.md) - Core architectural approach
- [Design Principles](../1_design/design_principles.md) - Fundamental design philosophy
- [Hybrid Design](../1_design/hybrid_design.md) - Implementation strategy balancing specifications and configurations
- [Config-Driven Design](../1_design/config_driven_design.md) - Previous implementation approach
- [Pipeline Assembler](../1_design/pipeline_assembler.md) - Pipeline assembly orchestration
- [Pipeline Compiler](../1_design/pipeline_compiler.md) - DAG compilation system
- [Step Specification](../1_design/step_specification.md) - Step specification format and structure
- [Step Builder](../1_design/step_builder.md) - Step builder implementation patterns
- [Script Contract](../1_design/script_contract.md) - Script contract system
- [Dependency Resolver](../1_design/dependency_resolver.md) - Automatic dependency resolution
- [Pipeline DAG](../1_design/pipeline_dag.md) - DAG data structure and operations

### Architecture Components
- [Dynamic Template System](../1_design/dynamic_template_system.md) - Runtime pipeline generation
- [Step Catalog Design](../1_design/unified_step_catalog_system_design.md) - Auto-discovery system
- [Registry Design](../1_design/registry_single_source_of_truth.md) - Component registration
- [Validation Engine](../1_design/validation_engine.md) - Multi-level validation framework
- [Config Field Management](../1_design/config_field_manager_refactoring.md) - Configuration system

### Developer Guides
- [Developer Guide](../0_developer_guide/README.md) - Complete development guide
- [Adding New Pipeline Step](../0_developer_guide/adding_new_pipeline_step.md) - Step development process
- [Step Builder Guide](../0_developer_guide/step_builder.md) - Builder implementation
- [Script Development Guide](../0_developer_guide/script_development_guide.md) - Script implementation
- [Validation Framework Guide](../0_developer_guide/validation_framework_guide.md) - Validation system usage

### Rules and Standard Operating Procedures
- [Alignment Rules](../0_developer_guide/alignment_rules.md) - Component alignment requirements and validation rules
- [Best Practices](../0_developer_guide/best_practices.md) - Development best practices and coding standards
- [Common Pitfalls](../0_developer_guide/common_pitfalls.md) - Common mistakes and how to avoid them
- [Standardization Rules](../0_developer_guide/standardization_rules.md) - Naming conventions and code organization standards
- [Validation Checklist](../0_developer_guide/validation_checklist.md) - Comprehensive validation checklist for development
- [Prerequisites](../0_developer_guide/prerequisites.md) - Development environment setup and requirements
- [Creation Process](../0_developer_guide/creation_process.md) - Step-by-step component creation process
- [Component Guide](../0_developer_guide/component_guide.md) - Comprehensive guide to system components
- [Design Principles](../0_developer_guide/design_principles.md) - Core design principles and architectural guidelines

### Workspace-Aware System
- [Workspace-Aware System Master Design](../1_design/workspace_aware_system_master_design.md) - Multi-developer support
- [Workspace Setup Guide](../01_developer_guide_workspace_aware/ws_workspace_setup_guide.md) - Workspace configuration
- [Workspace CLI Reference](../01_developer_guide_workspace_aware/ws_workspace_cli_reference.md) - Command-line tools

### Model-Specific Design and Optimization
- [MTGBM Loss Functions Refactoring Design](../1_design/mtgbm_models_refactoring_design.md) - Comprehensive refactoring design for MTGBM loss function implementations with abstract base classes, strategy pattern, and factory pattern
- [MTGBM Model Classes Refactoring Design](../1_design/mtgbm_model_classes_refactoring_design.md) - Architectural design for refactoring MTGBM model class implementations with template method pattern, base model abstractions, and state management
- [MTGBM Models Optimization Analysis](../4_analysis/2025-11-11_mtgbm_models_optimization_analysis.md) - Detailed analysis identifying optimization opportunities for MTGBM implementations

### Advanced Topics
- [Agentic Workflow Design](../1_design/agentic_workflow_design.md) - AI-powered workflow automation
- [MCP Integration](../1_design/mcp_agentic_workflow_master_design.md) - Model Context Protocol integration
- [Pipeline Catalog Design](../1_design/pipeline_catalog_design.md) - Pipeline sharing system
- [Performance Optimization](../1_design/cursus_framework_output_management.md) - Output management and optimization

### Testing and Validation
- [Pipeline Runtime Testing](../1_design/pipeline_runtime_testing_simplified_design.md) - Runtime testing framework
- [Alignment Validation](../1_design/alignment_validation_data_structures.md) - Component alignment validation
- [Universal Step Builder Test](../1_design/universal_step_builder_test.md) - Comprehensive testing system

### Configuration and UI
- [DAG Config Factory Design](../1_design/dag_config_factory_design.md) - Interactive configuration widgets
- [Config UI Design](../1_design/generalized_config_ui_design.md) - User interface design
- [Three-Tier Config Design](../1_design/three_tier_config_design.md) - Configuration architecture

### API and CLI
- [API Reference Documentation](../api/) - Complete API documentation
- [CLI Reference](../cli/) - Command-line interface documentation
- [Workspace CLI](../01_developer_guide_workspace_aware/ws_workspace_cli_reference.md) - Workspace-specific commands

### Examples and Tutorials
- [Examples](../examples/) - Usage examples and tutorials
- [Tutorials](../5_tutorials/) - Step-by-step tutorials
- [ML Examples](../ml/) - Machine learning specific examples

### Analysis and Planning
- [Project Planning](../2_project_planning/) - Development roadmap and planning
- [LLM Developer Resources](../3_llm_developer/) - AI-assisted development
- [Analysis Reports](../4_analysis/) - System analysis and metrics

This comprehensive reference collection provides complete documentation for understanding, using, and extending the Cursus ML pipeline system.
