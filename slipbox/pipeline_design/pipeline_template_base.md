---
tags:
  - design
  - foundation
  - pipeline_design
  - template_architecture
keywords:
  - pipeline template
  - abstract base class
  - component management
  - thread safety
  - execution document
  - factory methods
topics:
  - pipeline architecture
  - template patterns
  - component lifecycle
language: python
date of note: 2025-07-31
---

# Pipeline Template Base

The `PipelineTemplateBase` provides a standardized foundation for all pipeline templates in the system. It defines a consistent structure, manages dependency components properly, and enforces best practices across different pipeline implementations.

## Purpose

The pipeline template base solves several challenges:

1. **Code Duplication**: Eliminates redundant boilerplate across pipeline templates
2. **Component Lifecycle**: Provides consistent component creation and management
3. **Best Practices**: Enforces standardized approaches to pipeline construction
4. **Thread Safety**: Enables concurrent pipeline execution in multi-threaded environments

## Class Structure

```python
class PipelineTemplateBase(ABC):
    def __init__(self,
                 config_path: str,
                 sagemaker_session: Optional[PipelineSession] = None,
                 role: Optional[str] = None,
                 notebook_root: Optional[Path] = None,
                 registry_manager: Optional[RegistryManager] = None,
                 dependency_resolver: Optional[UnifiedDependencyResolver] = None):
        """Initialize base template."""
        pass
        
    # Abstract methods that must be implemented by subclasses
    @abstractmethod
    def _validate_configuration(self) -> None: pass
    
    @abstractmethod
    def _create_pipeline_dag(self) -> PipelineDAG: pass
    
    @abstractmethod
    def _create_config_map(self) -> Dict[str, BasePipelineConfig]: pass
    
    @abstractmethod
    def _create_step_builder_map(self) -> Dict[str, Type[StepBuilderBase]]: pass
    
    # Core method to generate pipeline
    def generate_pipeline(self) -> Pipeline:
        """Generate the SageMaker Pipeline."""
        pass
        
    # Important method for execution document filling
    def fill_execution_document(self, execution_document: Dict[str, Any]) -> Dict[str, Any]:
        """Fill execution document with pipeline metadata."""
        pass
        
    # Factory methods
    @classmethod
    def create_with_components(cls, config_path: str, context_name: Optional[str] = None, **kwargs): pass
    
    @classmethod
    def build_with_context(cls, config_path: str, **kwargs) -> Pipeline: pass
    
    @classmethod
    def build_in_thread(cls, config_path: str, **kwargs) -> Pipeline: pass
```

## Key Components

### 1. Configuration Loading

The template handles loading configurations from a file path:

```python
self.configs = self._load_configs(config_path)
self.base_config = self._get_base_config()
```

This eliminates redundant configuration loading code across templates. The template also provides a standardized approach for validating configurations against step specifications, avoiding the need to create temporary builders.

### 2. Component Management

The template manages dependency components (registry_manager and dependency_resolver):

```python
# Store dependency components
self._registry_manager = registry_manager
self._dependency_resolver = dependency_resolver

# Initialize components if not provided
if not self._registry_manager or not self._dependency_resolver:
    self._initialize_components()
```

This ensures proper component creation and management across all templates.

### 3. Pipeline Generation

The template coordinates the pipeline generation process:

```python
def generate_pipeline(self) -> Pipeline:
    # Create the DAG, config_map, and step builder map
    dag = self._create_pipeline_dag()
    config_map = self._create_config_map()
    step_builder_map = self._create_step_builder_map()
    
    # Create the assembler
    assembler = PipelineAssembler(
        dag=dag,
        config_map=config_map,
        step_builder_map=step_builder_map,
        # ...other parameters...
        registry_manager=self._registry_manager,
        dependency_resolver=self._dependency_resolver
    )
    
    # Generate the pipeline
    pipeline = assembler.generate_pipeline(pipeline_name)
    
    return pipeline
```

This provides a consistent approach to pipeline generation across all templates.

### 4. Factory Methods

The template includes factory methods for component creation:

```python
@classmethod
def create_with_components(cls, config_path: str, context_name: Optional[str] = None, **kwargs):
    components = create_pipeline_components(context_name)
    return cls(
        config_path=config_path,
        registry_manager=components["registry_manager"],
        dependency_resolver=components["resolver"],
        **kwargs
    )
```

These factory methods make it easier to create templates with properly configured components.

## Relationship to PipelineAssembler

The `PipelineTemplateBase` operates at a higher level than `PipelineAssembler`:

1. `PipelineTemplateBase` is responsible for:
   - Loading configurations from file
   - Creating the DAG, config_map, and step_builder_map
   - Managing dependency components
   - Providing a standard structure for templates

2. `PipelineAssembler` is responsible for:
   - Assembling the pipeline from the DAG, config_map, and step_builder_map
   - Handling the low-level details of step instantiation
   - Connecting steps according to the DAG

The relationship can be visualized as:

```
PipelineTemplateBase (High-level template)
├── Loads configurations from file
├── Creates DAG, config_map, step_builder_map
├── Manages dependency components
└── Uses PipelineAssembler for pipeline assembly
    ├── PipelineAssembler (Low-level assembler)
    ├── Instantiates steps based on the DAG
    ├── Connects steps according to the DAG
    └── Generates the final pipeline
```

## Implementation Pattern

To implement a pipeline template using `PipelineTemplateBase`:

1. Inherit from `PipelineTemplateBase`
2. Define the `CONFIG_CLASSES` class variable
3. Implement the required abstract methods
4. Override any other methods as needed

Example:

```python
class ExamplePipelineTemplate(PipelineTemplateBase):
    CONFIG_CLASSES = {
        'BasePipelineConfig': BasePipelineConfig,
        'ProcessingConfig': ProcessingConfig,
        'TrainingConfig': TrainingConfig,
        # ... other config classes ...
    }
    
    def _validate_configuration(self) -> None:
        # Perform lightweight validation of config structure and presence
        tp_configs = [cfg for name, cfg in self.configs.items() 
                     if isinstance(cfg, PreprocessingConfig)]
        
        if len(tp_configs) < 2:
            raise ValueError("Expected at least two PreprocessingConfig instances")
            
        # Check for presence of training and calibration configs
        training_config = next((cfg for cfg in tp_configs 
                              if getattr(cfg, 'job_type', None) == 'training'), None)
        if not training_config:
            raise ValueError("No PreprocessingConfig found with job_type='training'")
        
    def _create_pipeline_dag(self) -> PipelineDAG:
        # Create the DAG
        dag = PipelineDAG()
        dag.add_node("data_load")
        dag.add_node("preprocessing")
        dag.add_node("training")
        dag.add_edge("data_load", "preprocessing")
        dag.add_edge("preprocessing", "training")
        return dag
        
    def _create_config_map(self) -> Dict[str, BasePipelineConfig]:
        # Create the config map
        return {
            "data_load": self.configs.get('DataLoad'),
            "preprocessing": self.configs.get('Preprocessing'),
            "training": self.configs.get('Training'),
        }
        
    def _create_step_builder_map(self) -> Dict[str, Type[StepBuilderBase]]:
        # Create the step builder map
        return {
            "DataLoad": DataLoadingStepBuilder,
            "Preprocessing": PreprocessingStepBuilder,
            "Training": TrainingStepBuilder,
        }
```

## Thread Safety

The template provides several ways to ensure thread safety:

1. **Context Manager**: Use `build_with_context` to create a scoped context
   ```python
   pipeline = ExamplePipelineTemplate.build_with_context(config_path)
   ```

2. **Thread-Local Storage**: Use `build_in_thread` for thread-local components
   ```python
   pipeline = ExamplePipelineTemplate.build_in_thread(config_path)
   ```

3. **Factory Method**: Use `create_with_components` for isolated components
   ```python
   template = ExamplePipelineTemplate.create_with_components(config_path)
   pipeline = template.generate_pipeline()
   ```

## Execution Document Management

The template includes a crucial method for filling execution documents:

```python
def fill_execution_document(self, execution_document: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fill in the execution document with pipeline metadata.
    
    This method is designed to be overridden by subclasses to fill in
    execution documents with step-specific metadata from the pipeline.
    
    Args:
        execution_document: Execution document to fill
        
    Returns:
        Updated execution document
    """
    # Default implementation does nothing
    return execution_document
```

This method is particularly important for:

1. **Metadata Integration**: Adding pipeline-specific metadata to execution documents
2. **Step Configuration**: Including step-specific configurations like Cradle data loading requests
3. **Model Registration**: Adding model registration information for downstream systems
4. **Execution Tracking**: Adding pipeline execution details for monitoring and tracking

Example implementation:

```python
def fill_execution_document(self, execution_document: Dict[str, Any]) -> Dict[str, Any]:
    """Fill execution document with pipeline metadata."""
    if "PIPELINE_STEP_CONFIGS" not in execution_document:
        raise KeyError("Execution document missing 'PIPELINE_STEP_CONFIGS' key")
    
    pipeline_configs = execution_document["PIPELINE_STEP_CONFIGS"]

    # Fill Cradle configurations
    for step_name, request_dict in self.pipeline_metadata.get('cradle_loading_requests', {}).items():
        if step_name not in pipeline_configs:
            continue
        pipeline_configs[step_name]["STEP_CONFIG"] = request_dict
        
    # Fill Registration configurations
    for step_name, config in self.pipeline_metadata.get('registration_configs', {}).items():
        registration_step_name = f"Registration_{self.base_config.region}"
        if registration_step_name not in pipeline_configs:
            continue
        pipeline_configs[registration_step_name]["STEP_CONFIG"] = config

    return execution_document
```

Each template implementation should override this method to provide its specific execution document integration.

## Best Practices

1. **Configuration Loading and Lightweight Validation**:
   - Use the `CONFIG_CLASSES` class variable to specify expected configs
   - Override `_validate_configuration` to perform lightweight validation:
     - Validate presence/absence of required configuration objects
     - Check basic parameter types, ranges, and validity
     - Focus on non-dependency validation concerns
   - Leave dependency resolution to the `UnifiedDependencyResolver` during pipeline building

2. **DAG Construction**:
   - Keep the DAG simple and focused on the pipeline's purpose
   - Use descriptive node names that match config keys

3. **Config Mapping**:
   - Map step names to configuration instances
   - Validate that all required configs are present

4. **Step Builder Mapping**:
   - Map step types to appropriate builder classes
   - Ensure all step types are covered

5. **Component Management**:
   - Use the provided factory methods for component creation
   - Avoid creating components directly

6. **Thread Safety**:
   - Use thread-local storage for multi-threaded environments
   - Use context managers for proper cleanup
   
7. **Pipeline Parameters**:
   - Override `_get_pipeline_parameters` to add pipeline parameters
   - Include parameters for things like encryption, security, etc.

By following these best practices, you can create pipeline templates that are consistent, maintainable, and thread-safe.

## References

- [Dynamic Pipeline Template](dynamic_template.md) - Advanced template implementation built on this base class
- [Step Config Resolver](step_config_resolver.md) - Resolver for mapping DAG nodes to configurations
- [Pipeline DAG](pipeline_dag.md) - DAG structure used by pipeline templates
- [Step Builder](step_builder.md) - Builder pattern used for step instantiation
- [Registry Manager](registry_manager.md) - Registry manager for component registration
