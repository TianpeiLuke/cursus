# Pipeline Template Base

## Overview

The `PipelineTemplateBase` is an abstract base class that provides a consistent structure and common functionality for all pipeline templates. It handles configuration loading, component lifecycle management, and pipeline generation, enforcing best practices across different pipeline implementations.

## Class Definition

```python
class PipelineTemplateBase(ABC):
    """Base class for all pipeline templates."""
    
    # This should be overridden by subclasses to specify the config classes
    # that are expected in the configuration file
    CONFIG_CLASSES: Dict[str, Type[BasePipelineConfig]] = {}
    
    def __init__(
        self,
        config_path: str,
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None,
        registry_manager: Optional[RegistryManager] = None,
        dependency_resolver: Optional[UnifiedDependencyResolver] = None
    ):
        """Initialize base template."""
```

## Pipeline Template Workflow

The template follows these steps to build a pipeline:

1. Load configurations from file
2. Initialize component dependencies (registry_manager, dependency_resolver)
3. Create the DAG, config_map, and step_builder_map
4. Use PipelineAssembler to assemble the pipeline

## Key Methods

### Configuration Loading

```python
def _load_configs(self, config_path: str) -> Dict[str, BasePipelineConfig]:
    """
    Load configurations from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary of configurations
    """
    
def _get_base_config(self) -> BasePipelineConfig:
    """
    Get base configuration.
    
    Returns:
        Base configuration
        
    Raises:
        ValueError: If base configuration not found
    """
```

The template uses a configuration-driven approach where pipeline parameters are loaded from JSON files. The `CONFIG_CLASSES` class attribute defines which configuration classes are expected in the config file, and the `_load_configs` method loads these configurations accordingly.

### Component Management

```python
def _initialize_components(self) -> None:
    """
    Initialize dependency resolution components.
    
    This method creates registry manager and dependency resolver if they
    were not provided during initialization.
    """
```

This method handles the initialization of dependency resolution components, using the factory module to create properly configured instances if they weren't provided during initialization.

### Abstract Methods

These methods must be implemented by subclasses:

```python
@abstractmethod
def _validate_configuration(self) -> None:
    """Perform lightweight validation of configuration structure and essential parameters."""
    
@abstractmethod
def _create_pipeline_dag(self) -> PipelineDAG:
    """Create the DAG structure for the pipeline."""
    
@abstractmethod
def _create_config_map(self) -> Dict[str, BasePipelineConfig]:
    """Create a mapping from step names to config instances."""
    
@abstractmethod
def _create_step_builder_map(self) -> Dict[str, Type[StepBuilderBase]]:
    """Create a mapping from step types to builder classes."""
```

These abstract methods define the contract that subclasses must fulfill to create a valid pipeline template:

- **_validate_configuration**: Validates the structure and essential parameters of loaded configurations
- **_create_pipeline_dag**: Defines the pipeline's DAG structure
- **_create_config_map**: Maps step names to their respective configurations
- **_create_step_builder_map**: Maps step types to builder classes

### Pipeline Generation

```python
def generate_pipeline(self) -> Pipeline:
    """
    Generate the SageMaker Pipeline.
    
    This method coordinates the pipeline generation process:
    1. Create the DAG, config_map, and step_builder_map
    2. Create the PipelineAssembler
    3. Generate the pipeline
    4. Store pipeline metadata
    
    Returns:
        SageMaker Pipeline
    """
```

This method orchestrates the pipeline generation process by:
1. Creating the DAG, config_map, and step_builder_map using the abstract methods
2. Creating a PipelineAssembler with these components
3. Generating the pipeline using the assembler
4. Storing pipeline metadata for later use

### Factory Methods

```python
@classmethod
def create_with_components(cls, config_path: str, context_name: Optional[str] = None, **kwargs):
    """Create template with managed dependency components."""
    
@classmethod
def build_with_context(cls, config_path: str, **kwargs) -> Pipeline:
    """Build pipeline with scoped dependency resolution context."""
    
@classmethod
def build_in_thread(cls, config_path: str, **kwargs) -> Pipeline:
    """Build pipeline using thread-local component instances."""
```

These factory methods provide different ways to create pipeline templates:

- **create_with_components**: Creates a template with managed dependency components
- **build_with_context**: Builds a pipeline with a scoped dependency resolution context
- **build_in_thread**: Builds a pipeline using thread-local component instances

### Execution Document Support

```python
def fill_execution_document(self, execution_document: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fill in the execution document with pipeline metadata.
    
    This method is a placeholder that subclasses can override to fill in
    execution documents with step-specific metadata from the pipeline.
    
    Args:
        execution_document: Execution document to fill
        
    Returns:
        Updated execution document
    """
```

This method allows subclasses to fill execution documents with step-specific metadata from the pipeline, such as Cradle requests or execution configurations.

## Creating Custom Pipeline Templates

To create a custom pipeline template, extend the `PipelineTemplateBase` class:

```python
from src.pipeline_builder.pipeline_template_base import PipelineTemplateBase
from src.pipeline_dag.base_dag import PipelineDAG
from src.pipeline_steps.config_base import BasePipelineConfig

class MyCustomTemplate(PipelineTemplateBase):
    # Define the configuration classes expected in the config file
    CONFIG_CLASSES = {
        'Base': BasePipelineConfig,
        'DataLoading': CradleDataLoadingConfig,
        'Preprocessing': TabularPreprocessingConfig,
        'Training': XGBoostTrainingConfig,
    }
    
    def _validate_configuration(self) -> None:
        # Validate that required configs are present
        if 'DataLoading' not in self.configs:
            raise ValueError("DataLoading configuration is required")
        
        if 'Training' not in self.configs:
            raise ValueError("Training configuration is required")
        
        # Validate specific parameters
        data_config = self.configs['DataLoading']
        if not hasattr(data_config, 'source_type') or not data_config.source_type:
            raise ValueError("DataLoading configuration must specify source_type")
    
    def _create_pipeline_dag(self) -> PipelineDAG:
        # Create the DAG structure
        dag = PipelineDAG()
        
        # Define nodes and edges
        dag.add_node("data_loading")
        dag.add_node("preprocessing")
        dag.add_node("training")
        
        dag.add_edge("data_loading", "preprocessing")
        dag.add_edge("preprocessing", "training")
        
        return dag
    
    def _create_config_map(self) -> Dict[str, BasePipelineConfig]:
        # Map step names to configurations
        return {
            "data_loading": self.configs['DataLoading'],
            "preprocessing": self.configs['Preprocessing'],
            "training": self.configs['Training'],
        }
    
    def _create_step_builder_map(self) -> Dict[str, Type[StepBuilderBase]]:
        # Map step types to builder classes
        return {
            "CradleDataLoading": CradleDataLoadingStepBuilder,
            "TabularPreprocessingStep": TabularPreprocessingStepBuilder,
            "XGBoostTrainingStep": XGBoostTrainingStepBuilder,
        }
        
    def fill_execution_document(self, execution_document: Dict[str, Any]) -> Dict[str, Any]:
        # Fill execution document with Cradle requests
        cradle_requests = self.pipeline_metadata.get('cradle_loading_requests', {})
        
        if 'execution' not in execution_document:
            execution_document['execution'] = {}
            
        execution_document['execution']['cradle_requests'] = cradle_requests
        
        return execution_document
```

## Usage Examples

### Using a Custom Template

```python
# Create the template
template = MyCustomTemplate(
    config_path="configs/xgboost_config.json",
    sagemaker_session=sagemaker_session,
    role="arn:aws:iam::123456789012:role/SageMakerRole",
)

# Generate the pipeline
pipeline = template.generate_pipeline()

# Execute the pipeline
pipeline.upsert()
execution = pipeline.start()
```

### Using Factory Methods

```python
# Create with managed components
template = MyCustomTemplate.create_with_components(
    config_path="configs/xgboost_config.json",
    context_name="my_pipeline",
    sagemaker_session=sagemaker_session,
    role="arn:aws:iam::123456789012:role/SageMakerRole",
)

# Build with context
pipeline = MyCustomTemplate.build_with_context(
    config_path="configs/xgboost_config.json",
    sagemaker_session=sagemaker_session,
    role="arn:aws:iam::123456789012:role/SageMakerRole",
)

# Build in thread
pipeline = MyCustomTemplate.build_in_thread(
    config_path="configs/xgboost_config.json",
    sagemaker_session=sagemaker_session,
    role="arn:aws:iam::123456789012:role/SageMakerRole",
)
```

### Working with Execution Documents

```python
# Create execution document template
execution_doc = {
    "execution": {
        "name": "XGBoost Training Pipeline",
        "steps": []
    }
}

# Fill execution document with pipeline metadata
filled_doc = template.fill_execution_document(execution_doc)

print(f"Execution document with Cradle requests: {filled_doc}")
```

## Related Documentation

- [Pipeline Assembler](pipeline_assembler.md)
- [Pipeline DAG](../pipeline_dag/pipeline_dag.md)
- [Template Implementation](template_implementation.md)
- [Pipeline Examples](pipeline_examples.md)
