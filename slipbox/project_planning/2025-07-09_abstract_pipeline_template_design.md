# Abstract Pipeline Template Design

**Date:** July 11, 2025  
**Status:** ✅ COMPLETED  
**Priority:** 🔥 HIGH - Foundation for Pipeline Template Modernization  
**Related Documents:**
- [2025-07-09_pipeline_template_modernization_plan.md](./2025-07-09_pipeline_template_modernization_plan.md)
- [2025-07-08_remove_global_singletons.md](./2025-07-08_remove_global_singletons.md)
- [2025-07-07_specification_driven_step_builder_plan.md](./2025-07-07_specification_driven_step_builder_plan.md)
- [2025-07-04_job_type_variant_solution.md](./2025-07-04_job_type_variant_solution.md)
- [2025-07-05_corrected_alignment_architecture_plan.md](./2025-07-05_corrected_alignment_architecture_plan.md)
- [specification_driven_xgboost_pipeline_plan.md](./specification_driven_xgboost_pipeline_plan.md)

## Executive Summary

This document outlines the design for a new abstract base template class that serves as the foundation for all pipeline templates. This class provides a consistent structure, enforces best practices, and implements common functionality, making it easier to create and maintain pipeline templates. The abstract base class is the cornerstone of the Pipeline Template Modernization Plan, providing a solid foundation for all pipeline templates to build upon. As of July 11, 2025, the implementation has been completed and successfully integrated with all pipeline templates.

## Design Goals

1. **Consistent Structure**: Ensure all pipeline templates follow the same basic structure
2. **Component Lifecycle Management**: Properly handle creation and disposal of dependency components
3. **Dependency Injection**: Support passing of registry_manager and dependency_resolver
4. **Thread Safety**: Provide mechanisms for thread-safe pipeline creation
5. **Best Practices Enforcement**: Guide developers toward recommended patterns
6. **Extensibility**: Allow for template-specific customization
7. **Backward Compatibility**: Support existing pipeline templates with minimal changes

## Implemented Class Structure

### PipelineTemplateBase

```python
class PipelineTemplateBase(ABC):
    """
    Abstract base class for all pipeline templates.
    
    This class provides a consistent structure and common functionality for
    all pipeline templates, enforcing best practices and ensuring proper
    component lifecycle management.
    """
    
    def __init__(
        self,
        config_path: str,
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None,
        registry_manager: Optional[RegistryManager] = None,
        dependency_resolver: Optional[UnifiedDependencyResolver] = None
    ):
        """
        Initialize base template.
        
        Args:
            config_path: Path to configuration file
            sagemaker_session: SageMaker session
            role: IAM role
            notebook_root: Root directory of notebook
            registry_manager: Optional registry manager for dependency injection
            dependency_resolver: Optional dependency resolver for dependency injection
        """
        self.config_path = config_path
        self.session = sagemaker_session
        self.role = role
        self.notebook_root = notebook_root or Path.cwd()
        
        # Load configurations
        self.configs = self._load_configs(config_path)
        self.base_config = self._get_base_config()
        
        # Store dependency components
        self._registry_manager = registry_manager
        self._dependency_resolver = dependency_resolver
        
        # Initialize components if not provided
        if not self._registry_manager or not self._dependency_resolver:
            self._initialize_components()
            
        # Validate configuration
        self._validate_configuration()
        
        # Initialize pipeline metadata storage
        self.pipeline_metadata = {}
        
    def _load_configs(self, config_path: str) -> Dict[str, BasePipelineConfig]:
        """
        Load configurations from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary of configurations
        """
        return load_configs(config_path, self.CONFIG_CLASSES)
        
    def _get_base_config(self) -> BasePipelineConfig:
        """
        Get base configuration.
        
        Returns:
            Base configuration
            
        Raises:
            ValueError: If base configuration not found
        """
        base_config = self.configs.get('Base')
        if not base_config:
            raise ValueError("Base configuration not found in config file")
        return base_config
        
    def _initialize_components(self) -> None:
        """
        Initialize dependency resolution components.
        
        This method creates registry manager and dependency resolver if they
        were not provided during initialization.
        """
        context_name = getattr(self.base_config, 'pipeline_name', None)
        components = create_pipeline_components(context_name)
        
        if not self._registry_manager:
            self._registry_manager = components["registry_manager"]
            
        if not self._dependency_resolver:
            self._dependency_resolver = components["resolver"]
            
    @abstractmethod
    def _validate_configuration(self) -> None:
        """
        Validate configuration.
        
        This method should be implemented by subclasses to validate their
        specific configuration requirements. Instead of creating temporary
        builders, implementations should directly use step specifications
        to validate inputs and outputs.
        
        Typical implementation pattern:
        ```python
        def _validate_configuration(self) -> None:
            # Import specifications directly
            from ..pipeline_step_specs.my_step_spec import MY_STEP_SPEC
            
            # Check for required dependencies in the specification
            for dependency in MY_STEP_SPEC.dependencies.values():
                if dependency.required:
                    # Validation logic here
                    ...
        ```
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
        
    @abstractmethod
    def _create_pipeline_dag(self) -> PipelineDAG:
        """
        Create the DAG structure for the pipeline.
        
        This method should be implemented by subclasses to define the
        pipeline's DAG structure.
        
        Returns:
            PipelineDAG instance
        """
        pass
        
    @abstractmethod
    def _create_config_map(self) -> Dict[str, BasePipelineConfig]:
        """
        Create a mapping from step names to config instances.
        
        This method should be implemented by subclasses to map step names
        to their respective configurations.
        
        Returns:
            Dictionary mapping step names to configurations
        """
        pass
        
    @abstractmethod
    def _create_step_builder_map(self) -> Dict[str, Type[StepBuilderBase]]:
        """
        Create a mapping from step types to builder classes.
        
        This method should be implemented by subclasses to map step types
        to their builder classes.
        
        Returns:
            Dictionary mapping step types to builder classes
        """
        pass
        
    def _get_pipeline_parameters(self) -> List[ParameterString]:
        """
        Get pipeline parameters.
        
        Returns:
            List of pipeline parameters
        """
        # Default implementation with common parameters
        return []
        
    def _get_pipeline_name(self) -> str:
        """
        Get pipeline name.
        
        Returns:
            Pipeline name
        """
        return getattr(self.base_config, 'pipeline_name', 'default-pipeline')
        
    def generate_pipeline(self) -> Pipeline:
        """
        Generate the SageMaker Pipeline.
        
        Returns:
            SageMaker Pipeline
        """
        pipeline_name = self._get_pipeline_name()
        
        # Create the DAG, config map, and step builder map
        dag = self._create_pipeline_dag()
        config_map = self._create_config_map()
        step_builder_map = self._create_step_builder_map()
        
        # Create the assembler
        assembler = PipelineAssembler(
            dag=dag,
            config_map=config_map,
            step_builder_map=step_builder_map,
            sagemaker_session=self.session,
            role=self.role,
            pipeline_parameters=self._get_pipeline_parameters(),
            notebook_root=self.notebook_root,
            registry_manager=self._registry_manager,
            dependency_resolver=self._dependency_resolver
        )
        
        # Generate the pipeline
        pipeline = assembler.generate_pipeline(pipeline_name)
        
        # Store pipeline metadata
        self._store_pipeline_metadata(assembler)
        
        return pipeline
        
    def _store_pipeline_metadata(self, assembler: PipelineAssembler) -> None:
        """
        Store pipeline metadata from assembler.
        
        This method can be overridden by subclasses to store step-specific
        metadata like Cradle requests or execution document configurations.
        
        Args:
            assembler: PipelineAssembler instance
        """
        # Default implementation stores basic metadata
        self.pipeline_metadata = {
            "step_names": list(assembler.step_instances.keys()),
            "step_types": {name: type(step).__name__ for name, step in assembler.step_instances.items()},
            "step_dependencies": {name: [dep.name for dep in step.step.depends_on] for name, step in assembler.step_instances.items() if hasattr(step, 'step') and hasattr(step.step, 'depends_on')}
        }
        
    def fill_execution_document(self, execution_document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fill execution document with pipeline metadata.
        
        This method is designed to be overridden by subclasses to fill in
        execution documents with step-specific metadata from the pipeline.
        
        Args:
            execution_document: Execution document to fill
            
        Returns:
            Updated execution document
        """
        # Default implementation does nothing
        return execution_document
        
    @classmethod
    def create_with_components(cls, config_path: str, context_name: Optional[str] = None, **kwargs):
        """
        Create template with managed dependency components.
        
        This factory method creates a template with properly configured
        dependency resolution components from the factory module.
        
        Args:
            config_path: Path to configuration file
            context_name: Optional context name for registry isolation
            **kwargs: Additional arguments to pass to constructor
            
        Returns:
            Template instance with managed components
        """
        components = create_pipeline_components(context_name)
        return cls(
            config_path=config_path,
            registry_manager=components["registry_manager"],
            dependency_resolver=components["resolver"],
            **kwargs
        )
        
    @classmethod
    def build_with_context(cls, config_path: str, **kwargs) -> Pipeline:
        """
        Build pipeline with scoped dependency resolution context.
        
        This method creates a template with a dependency resolution context
        that ensures proper cleanup of resources after pipeline generation.
        
        Args:
            config_path: Path to configuration file
            **kwargs: Additional arguments to pass to constructor
            
        Returns:
            Generated pipeline
        """
        with dependency_resolution_context(clear_on_exit=True) as components:
            template = cls(
                config_path=config_path,
                registry_manager=components["registry_manager"],
                dependency_resolver=components["resolver"],
                **kwargs
            )
            return template.generate_pipeline()
            
    @classmethod
    def build_in_thread(cls, config_path: str, **kwargs) -> Pipeline:
        """
        Build pipeline using thread-local component instances.
        
        This method creates a template with thread-local component instances,
        ensuring thread safety in multi-threaded environments.
        
        Args:
            config_path: Path to configuration file
            **kwargs: Additional arguments to pass to constructor
            
        Returns:
            Generated pipeline
        """
        components = get_thread_components()
        template = cls(
            config_path=config_path,
            registry_manager=components["registry_manager"],
            dependency_resolver=components["resolver"],
            **kwargs
        )
        return template.generate_pipeline()
```

## Integration with Other Components

### Integration with PipelineAssembler

The PipelineTemplateBase works in conjunction with the PipelineAssembler, which is responsible for the low-level assembly of the pipeline:

```python
# In PipelineTemplateBase.generate_pipeline()
assembler = PipelineAssembler(
    dag=dag,
    config_map=config_map,
    step_builder_map=step_builder_map,
    sagemaker_session=self.session,
    role=self.role,
    pipeline_parameters=self._get_pipeline_parameters(),
    notebook_root=self.notebook_root,
    registry_manager=self._registry_manager,
    dependency_resolver=self._dependency_resolver
)

pipeline = assembler.generate_pipeline(pipeline_name)
```

The assembler handles:
1. Instantiating steps based on the DAG
2. Connecting steps according to the DAG
3. Resolving property references between steps
4. Building the final pipeline

### Integration with PropertyReference System

The template system works seamlessly with the enhanced property reference system through the PipelineAssembler:

```python
# In PipelineAssembler._instantiate_step()
prop_ref = PropertyReference(
    step_name=src_step,
    property_path=output_spec.property_path,
    output_spec=output_spec
)

runtime_prop = prop_ref.to_runtime_property(self.step_instances)
inputs[input_name] = runtime_prop
```

The template provides all the necessary context for property references to be resolved correctly:
- Step instances through the assembler
- Output specifications through step builders
- Property paths through specifications

### Integration with Job Type Variants

The template system now integrates with job type variants through specification selection:

```python
# In XGBoostEndToEndTemplate._create_config_map()
config_map = {}

# Add training configs with job_type='training'
train_dl_config = self._get_config_by_type(CradleDataLoadConfig, "training")
if train_dl_config:
    config_map["train_data_load"] = train_dl_config
    
# Add calibration configs with job_type='calibration'
calib_dl_config = self._get_config_by_type(CradleDataLoadConfig, "calibration")
if calib_dl_config:
    config_map["calib_data_load"] = calib_dl_config
```

This ensures that each step gets the right configuration with the correct job type, which then leads to the correct job type variant specification being selected.

## Template Implementation Examples

### XGBoostEndToEndTemplate

```python
class XGBoostEndToEndTemplate(PipelineTemplateBase):
    """Template-based builder for XGBoost end-to-end pipeline."""
    
    CONFIG_CLASSES = {
        'Base': BasePipelineConfig,
        'CradleDataLoad': CradleDataLoadConfig,
        'TabularPreprocessing': TabularPreprocessingConfig,
        'XGBoostTraining': XGBoostTrainingConfig,
        'PackageStep': PackageStepConfig,
        'PayloadTest': PayloadConfig,
        'ModelRegistration': ModelRegistrationConfig,
    }
    
    def _validate_configuration(self) -> None:
        """Validate the configuration structure."""
        # Check for preprocessing configs
        tp_configs = [cfg for name, cfg in self.configs.items() 
                     if isinstance(cfg, TabularPreprocessingConfig)]
        
        if len(tp_configs) < 2:
            raise ValueError("Expected at least two TabularPreprocessingConfig instances")
        
        # Check for training/calibration configs
        training_config = next((cfg for cfg in tp_configs 
                              if getattr(cfg, 'job_type', None) == 'training'), None)
        if not training_config:
            raise ValueError("No TabularPreprocessingConfig found with job_type='training'")
            
        calibration_config = next((cfg for cfg in tp_configs 
                                 if getattr(cfg, 'job_type', None) == 'calibration'), None)
        if not calibration_config:
            raise ValueError("No TabularPreprocessingConfig found with job_type='calibration'")
        
        # Check for single-instance configs
        for config_type, name in [
            (XGBoostTrainingConfig, "XGBoost training"),
            (PackageStepConfig, "model packaging"),
            (PayloadConfig, "payload testing"),
            (ModelRegistrationConfig, "model registration")
        ]:
            instances = [cfg for _, cfg in self.configs.items() if type(cfg) is config_type]
            if not instances:
                raise ValueError(f"No {name} configuration found")
            if len(instances) > 1:
                raise ValueError(f"Multiple {name} configurations found")
    
    def _create_pipeline_dag(self) -> PipelineDAG:
        """Create the pipeline DAG structure."""
        dag = PipelineDAG()
        
        # Add nodes
        dag.add_node("train_data_load")
        dag.add_node("train_preprocess")
        dag.add_node("xgboost_train")
        dag.add_node("model_packaging")
        dag.add_node("payload_test")
        dag.add_node("model_registration")
        dag.add_node("calib_data_load")
        dag.add_node("calib_preprocess")
        
        # Add edges
        dag.add_edge("train_data_load", "train_preprocess")
        dag.add_edge("train_preprocess", "xgboost_train")
        dag.add_edge("xgboost_train", "model_packaging")
        dag.add_edge("xgboost_train", "payload_test")
        dag.add_edge("model_packaging", "model_registration")
        dag.add_edge("payload_test", "model_registration")
        dag.add_edge("calib_data_load", "calib_preprocess")
        
        return dag
```

### PytorchEndToEndTemplate

```python
class PytorchEndToEndTemplate(PipelineTemplateBase):
    """Template-based builder for PyTorch end-to-end pipeline."""
    
    CONFIG_CLASSES = {
        'Base': BasePipelineConfig,
        'CradleDataLoad': CradleDataLoadConfig,
        'TabularPreprocessing': TabularPreprocessingConfig,
        'PytorchTraining': PytorchTrainingConfig,
        'PackageStep': PackageStepConfig,
        'PayloadTest': PayloadConfig,
        'ModelRegistration': ModelRegistrationConfig,
    }
    
    # Implementation of abstract methods...
```

## Execution Document Support

A key feature of the PipelineTemplateBase is support for execution document filling:

```python
def fill_execution_document(self, execution_document: Dict[str, Any]) -> Dict[str, Any]:
    """Fill in the execution document with pipeline metadata."""
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

This allows templates to fill in execution documents with step-specific metadata, which is crucial for:
1. Cradle data loading requests
2. Model registration information
3. Step configuration details
4. Pipeline execution details

## Usage Examples

### Basic Usage

```python
# Create a template
template = XGBoostEndToEndTemplate(
    config_path="path/to/config.json",
    sagemaker_session=session,
    role=role
)

# Generate a pipeline
pipeline = template.generate_pipeline()
```

### Using Factory Method

```python
# Create a template with managed components
template = XGBoostEndToEndTemplate.create_with_components(
    config_path="path/to/config.json",
    context_name="my-pipeline",
    sagemaker_session=session,
    role=role
)

# Generate a pipeline
pipeline = template.generate_pipeline()
```

### Using Context Manager

```python
# Generate a pipeline with a scoped context
pipeline = XGBoostEndToEndTemplate.build_with_context(
    config_path="path/to/config.json",
    sagemaker_session=session,
    role=role
)
```

### Using Thread-Local Storage

```python
# Generate a pipeline with thread-local components
pipeline = XGBoostEndToEndTemplate.build_in_thread(
    config_path="path/to/config.json",
    sagemaker_session=session,
    role=role
)
```

## Implementation Results

The PipelineTemplateBase implementation has delivered significant benefits:

1. **Code Reduction**: ~250 lines of code removed from templates (~40% reduction)
2. **Consistent Structure**: All templates now follow the same basic structure
3. **Improved Error Handling**: Consistent validation and error messages
4. **Better Thread Safety**: Support for thread-local components
5. **Reduced Boilerplate**: Common functionality implemented in the base class
6. **Enhanced Extensibility**: Clear extension points for subclasses
7. **Improved Documentation**: Clear guidance for template implementation
8. **Better Testing**: Isolation of components for testing

## Implementation Plan

### Phase 1: Create Abstract Base Class ✅ COMPLETED

1. ✅ Create `pipeline_template_base.py` module in `pipeline_builder` package
2. ✅ Implement `PipelineTemplateBase` class
3. ✅ Add unit tests for abstract base class

### Phase 2: Create Reference Implementation ✅ COMPLETED

1. ✅ Create `xgboost_end_to_end_template.py` module
2. ✅ Implement `XGBoostEndToEndTemplate` class extending `PipelineTemplateBase`
3. ✅ Add comprehensive documentation
4. ✅ Add unit tests for reference implementation

### Phase 3: Update Existing Templates ✅ COMPLETED

1. ✅ Update each existing template to inherit from `PipelineTemplateBase`
   - ✅ XGBoost end-to-end template
   - ✅ PyTorch end-to-end template
   - ✅ XGBoost training-only template
   - ✅ XGBoost data loading and preprocessing template
2. ✅ Remove redundant code that is now handled by the base class
3. ✅ Ensure backward compatibility

### Phase 4: Add Tutorial and Documentation ✅ COMPLETED

1. ✅ Create tutorial notebook for creating templates with the abstract base class
2. ✅ Document best practices for template implementation
3. ✅ Add examples of common patterns and extension points

## Key Design Decisions

### Why an Abstract Base Class?

An abstract base class provides several benefits for our pipeline templates:

1. **Consistency**: All templates now follow the same basic structure
2. **Code Reuse**: Common functionality is implemented once in the base class
3. **Enforcement**: Required methods must be implemented by subclasses
4. **Guidance**: Default implementations guide developers toward best practices
5. **Evolution**: Base class can evolve independently of templates

### Why Dependency Injection?

Dependency injection improves our pipeline templates in several ways:

1. **Testability**: Components can be mocked for testing
2. **Flexibility**: Different implementations can be provided
3. **Lifecycle Management**: Components can be created and disposed together
4. **Thread Safety**: Components can be isolated between threads

### Why Factory Methods?

Factory methods simplify the creation of templates:

1. **Convenience**: Encapsulate component creation
2. **Best Practices**: Guide developers toward recommended patterns
3. **Flexibility**: Allow different creation strategies
4. **Isolation**: Support isolated components per template

## Integration with Specification-Driven Architecture

The PipelineTemplateBase integrates seamlessly with the specification-driven architecture:

1. **Configuration Validation**: Templates validate against specifications rather than creating temporary builders
2. **Dependency Resolution**: Templates use the UnifiedDependencyResolver for dependency resolution
3. **Job Type Variants**: Templates support job type variants through specification selection
4. **Property References**: Templates work with the enhanced property reference system

## Conclusion

The PipelineTemplateBase class provides a solid foundation for all pipeline templates, ensuring consistency, enforcing best practices, and implementing common functionality. This makes it easier to create and maintain pipeline templates, and supports the goals of the Pipeline Template Modernization Plan.

The implementation uses lightweight configuration validation instead of comprehensive dependency validation, providing:

1. **Proper Separation of Concerns**: Configuration structure validation is separate from dependency resolution
2. **Leveraging Existing Components**: The UnifiedDependencyResolver handles dependency validation during pipeline building
3. **Improved Performance**: No temporary builders created during initialization means faster template startup
4. **Cleaner Code**: Template implementations focus on validating their own configuration structure
5. **Better Error Messages**: Configuration errors are reported early, while dependency errors include detailed diagnostics from the resolver

The implementation has been successfully completed and integrated with all components of the pipeline system. It provides a robust foundation for future pipeline development and maintenance.
