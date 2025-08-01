# Pipeline Template Base Design

**Date:** July 9, 2025  
**Status:** ✅ COMPLETED  
**Priority:** 🔥 HIGH - Foundation for Pipeline Template Modernization  
**Related Documents:**
- [2025-07-09_pipeline_template_modernization_plan.md](./2025-07-09_pipeline_template_modernization_plan.md)
- [2025-07-08_remove_global_singletons.md](./2025-07-08_remove_global_singletons.md)

## Executive Summary

This document outlines the design for a new template base class that will serve as the foundation for all pipeline templates. This class will provide a consistent structure, enforce best practices, and implement common functionality, making it easier to create and maintain pipeline templates. The pipeline template base class will be the cornerstone of the Pipeline Template Modernization Plan, providing a solid foundation for all pipeline templates to build upon.

## Design Goals

1. **Consistent Structure**: Ensure all pipeline templates follow the same basic structure
2. **Component Lifecycle Management**: Properly handle creation and disposal of dependency components
3. **Dependency Injection**: Support passing of registry_manager and dependency_resolver
4. **Thread Safety**: Provide mechanisms for thread-safe pipeline creation
5. **Best Practices Enforcement**: Guide developers toward recommended patterns
6. **Extensibility**: Allow for template-specific customization
7. **Backward Compatibility**: Support existing pipeline templates with minimal changes

## Class Structure

### PipelineTemplateBase

```python
class PipelineTemplateBase(ABC):
    """
    Base class for all pipeline templates.
    
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
        
    def _get_pipeline_name(self) -> str:
        """
        Get pipeline name.
        
        Returns:
            Pipeline name
        """
        return getattr(self.base_config, 'pipeline_name', 'default-pipeline')
        
    def _store_pipeline_metadata(self, assembler: PipelineAssembler) -> None:
        """
        Store pipeline metadata from assembler.
        
        This method can be overridden by subclasses to store step-specific
        metadata like Cradle requests or execution document configurations.
        
        Args:
            assembler: PipelineAssembler instance
        """
        pass
        
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

## Usage Examples

### Basic Usage

```python
# Create a template
template = MyPipelineTemplate(
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
template = MyPipelineTemplate.create_with_components(
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
pipeline = MyPipelineTemplate.build_with_context(
    config_path="path/to/config.json",
    sagemaker_session=session,
    role=role
)
```

### Using Thread-Local Storage

```python
# Generate a pipeline with thread-local components
pipeline = MyPipelineTemplate.build_in_thread(
    config_path="path/to/config.json",
    sagemaker_session=session,
    role=role
)
```

## Recent Updates

The PipelineTemplateBase has been updated with the following improvements:

1. **Renaming and Refactoring**:
   - Renamed from `AbstractPipelineTemplate` to `PipelineTemplateBase` for clearer naming
   - Updated all references from `PipelineBuilderTemplate` to `PipelineAssembler`
   - Updated parameter documentation from `template` to `assembler` for consistency

2. **Lightweight Configuration Validation**:
   - The `_validate_configuration()` method now focuses on basic configuration structure validation
   - Instead of performing dependency validation, it checks for presence/absence of required configurations
   - This approach leverages the dependency resolver for proper dependency validation during pipeline building
   - No temporary builders are created, resulting in cleaner and more efficient code

3. **Separation of Concerns**:
   - Configuration structure validation happens during initialization via `_validate_configuration()`
   - Dependency resolution happens during pipeline building via the `UnifiedDependencyResolver`
   - This clear separation improves maintainability and reduces redundant validation

4. **Reference Implementation**: 
   - The XGBoostTrainEvaluateE2ETemplate has been updated with a simplified validation approach
   - It performs checks like validating the presence of required configurations and checking job types
   - This makes the code easier to understand and reduces initialization overhead

5. **Enhanced Property Reference Handling**:
   - Removed reliance on the outdated `property_reference_wrapper.py` module
   - Integrated with the enhanced `PropertyReference` class in `base_specifications.py`
   - Simplified step builder implementations by removing redundant `handle_property_reference` calls
   - Fixed errors like `'dict' object has no attribute 'decode'` that occurred during pipeline execution
   - PipelineAssembler now correctly handles property references during pipeline assembly
   - All property references are properly passed to SageMaker's native property reference system
   - This enables cleaner, more maintainable templates without special case handling for property references

These changes are part of the broader effort to embrace the specification-driven approach across the entire pipeline system.

## Implementation Plan

### Phase 1: Create Template Base Class (Week 1) - ✅ COMPLETED

1. Create `pipeline_template_base.py` module in `pipeline_builder` package
2. Implement `PipelineTemplateBase` class
3. Add unit tests for template base class

### Phase 2: Create Reference Implementation (Week 1) - ✅ COMPLETED

1. Create `reference_pipeline_template.py` module
2. Implement `ReferencePipelineTemplate` class extending `PipelineTemplateBase`
3. Add comprehensive documentation
4. Add unit tests for reference implementation

### Phase 3: Update Existing Templates (Week 2) - ✅ COMPLETED

1. Update XGBoost Train-Evaluate E2E Template to inherit from `PipelineTemplateBase` - ✅ COMPLETED
2. Update remaining templates - 🔄 IN PROGRESS
3. Ensure backward compatibility - ✅ COMPLETED

### Phase 4: Add Tutorial and Documentation (Week 3) - ✅ COMPLETED

1. Create documentation for `PipelineTemplateBase` - ✅ COMPLETED
2. Create documentation for `PipelineAssembler` - ✅ COMPLETED
3. Document best practices for template implementation - ✅ COMPLETED
4. Add examples of common patterns and extension points - ✅ COMPLETED

## Key Design Decisions

### Why a Base Class?

A base class provides several benefits for our pipeline templates:

1. **Consistency**: All templates will follow the same basic structure
2. **Code Reuse**: Common functionality can be implemented once in the base class
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

## Conclusion

The PipelineTemplateBase class provides a solid foundation for all pipeline templates, ensuring consistency, enforcing best practices, and implementing common functionality. This makes it easier to create and maintain pipeline templates, and supports the goals of the Pipeline Template Modernization Plan.

The renaming from AbstractPipelineTemplate to PipelineTemplateBase and from PipelineBuilderTemplate to PipelineAssembler further improves clarity by:

1. **Better Semantics**: Names more clearly reflect component roles
2. **Consistent Naming**: Follows standard naming patterns in the codebase
3. **Clearer Distinction**: Makes the relationship between components more obvious
4. **Improved Readability**: Code is easier to understand at a glance

The recent update to use lightweight configuration validation instead of comprehensive dependency validation further strengthens this foundation by:

1. **Proper Separation of Concerns**: Configuration structure validation is separate from dependency resolution
2. **Leveraging Existing Components**: The UnifiedDependencyResolver handles dependency validation during pipeline building
3. **Improved Performance**: No temporary builders created during initialization means faster template startup
4. **Cleaner Code**: Template implementations focus on validating their own configuration structure
5. **Better Error Messages**: Configuration errors are reported early, while dependency errors include detailed diagnostics from the resolver

By implementing this template base class with lightweight validation and leveraging the existing dependency resolver, we have taken a significant step toward a more modern, maintainable pipeline architecture that follows best practices like separation of concerns and reuse of existing components.
