---
tags:
  - code
  - core
  - compiler
  - template
  - dynamic_generation
keywords:
  - dynamic template
  - pipeline template
  - DAG adaptation
  - template generation
  - pipeline assembly
topics:
  - dynamic templates
  - pipeline generation
  - template adaptation
language: python
date of note: 2025-09-07
---

# Dynamic Pipeline Template

## Overview

The `DynamicPipelineTemplate` is a flexible implementation of `PipelineTemplateBase` that can adapt to any PipelineDAG structure without requiring custom template classes. It implements the abstract methods of `PipelineTemplateBase` using intelligent resolution mechanisms to map DAG nodes to configurations and step builders.

## Class Definition

```python
class DynamicPipelineTemplate(PipelineTemplateBase):
    """
    Dynamic pipeline template that works with any PipelineDAG.
    
    This template automatically implements the abstract methods of
    PipelineTemplateBase by using intelligent resolution mechanisms
    to map DAG nodes to configurations and step builders.
    """
    
    # Initialize CONFIG_CLASSES as empty - will be populated dynamically
    CONFIG_CLASSES: Dict[str, Type[BasePipelineConfig]] = {}
    
    def __init__(
        self,
        dag: PipelineDAG,
        config_path: str,
        config_resolver: Optional[StepConfigResolver] = None,
        builder_registry: Optional[StepBuilderRegistry] = None,
        skip_validation: bool = False,
        **kwargs
    ):
        """Initialize dynamic template with intelligent resolution components."""
```

## Key Design Choices

### 1. Dynamic Configuration Class Detection

The template automatically detects required configuration classes from the configuration file, eliminating the need for manual class specification:

```python
def _detect_config_classes(self) -> Dict[str, Type[BasePipelineConfig]]:
    """
    Automatically detect required config classes from configuration file.
    
    This method analyzes the configuration file to determine which
    configuration classes are needed based on:
    1. Config type metadata in the configuration file
    2. Model type information in configuration entries
    3. Essential base classes needed for all pipelines
    
    Returns:
        Dictionary mapping config class names to config classes
    """
    # Import here to avoid circular imports
    from ...steps.configs.utils import detect_config_classes_from_json
    
    # Use the helper function to detect classes from the JSON file
    detected_classes = detect_config_classes_from_json(self.config_path)
    self.logger.debug(f"Detected {len(detected_classes)} required config classes from configuration file")
    
    return detected_classes
```

This approach provides several benefits:
- **Performance Optimization**: Only loads classes that are actually used
- **Reduced Memory Usage**: Avoids loading unnecessary configuration classes
- **Automatic Discovery**: No manual maintenance of class lists required
- **Error Prevention**: Reduces chances of missing required classes

### 2. Intelligent DAG-to-Configuration Mapping

The template uses `StepConfigResolver` to automatically map DAG nodes to configuration instances:

```python
def _create_config_map(self) -> Dict[str, BasePipelineConfig]:
    """
    Auto-map DAG nodes to configurations.
    
    Uses StepConfigResolver to intelligently match DAG node names
    to configuration instances from the loaded config file.
    
    Returns:
        Dictionary mapping DAG node names to configuration instances
        
    Raises:
        ConfigurationError: If nodes cannot be resolved to configurations
    """
    if self._resolved_config_map is not None:
        return self._resolved_config_map
    
    try:
        dag_nodes = list(self._dag.nodes)
        self.logger.info(f"Resolving {len(dag_nodes)} DAG nodes to configurations")
        
        # Extract metadata from loaded configurations if available
        if self._loaded_metadata is None and hasattr(self, 'loaded_config_data'):
            if isinstance(self.loaded_config_data, dict) and 'metadata' in self.loaded_config_data:
                self._loaded_metadata = self.loaded_config_data['metadata']
                self.logger.info(f"Using metadata from loaded configuration")
        
        # Use the config resolver to map nodes to configs
        self._resolved_config_map = self._config_resolver.resolve_config_map(
            dag_nodes=dag_nodes,
            available_configs=self.configs,
            metadata=self._loaded_metadata
        )
        
        self.logger.info(f"Successfully resolved all {len(self._resolved_config_map)} nodes")
        
        # Log resolution details
        for node, config in self._resolved_config_map.items():
            config_type = type(config).__name__
            job_type = getattr(config, 'job_type', 'N/A')
            self.logger.debug(f"  {node} → {config_type} (job_type: {job_type})")
        
        return self._resolved_config_map
        
    except Exception as e:
        self.logger.error(f"Failed to resolve DAG nodes to configurations: {e}")
        raise ConfigurationError(f"Configuration resolution failed: {e}")
```

The resolution process includes:
- **Metadata Integration**: Uses configuration metadata for improved matching
- **Detailed Logging**: Provides comprehensive feedback on resolution results
- **Caching**: Stores resolved mappings to avoid redundant computation
- **Error Handling**: Graceful failure with detailed error messages

### 3. Automatic Step Builder Registry Integration

The template integrates with `StepBuilderRegistry` to map configuration types to step builders:

```python
def _create_step_builder_map(self) -> Dict[str, Type[StepBuilderBase]]:
    """
    Auto-map step types to builders using registry.
    
    Uses StepBuilderRegistry to map configuration types to their
    corresponding step builder classes.
    
    Returns:
        Dictionary mapping step types to step builder classes
        
    Raises:
        RegistryError: If step builders cannot be found for config types
    """
    if self._resolved_builder_map is not None:
        return self._resolved_builder_map
    
    try:
        # Get the complete builder registry
        self._resolved_builder_map = self._builder_registry.get_builder_map()
        
        self.logger.info(f"Using {len(self._resolved_builder_map)} registered step builders")
        
        # Validate that all required builders are available
        config_map = self._create_config_map()
        missing_builders = []
        
        for node, config in config_map.items():
            try:
                # Pass the node name to the registry for better resolution
                builder_class = self._builder_registry.get_builder_for_config(config, node_name=node)
                step_type = self._builder_registry._config_class_to_step_type(
                    type(config).__name__, node_name=node, job_type=getattr(config, 'job_type', None))
                self.logger.debug(f"  {step_type} → {builder_class.__name__}")
            except RegistryError as e:
                missing_builders.append(f"{node} ({type(config).__name__})")
        
        if missing_builders:
            available_builders = list(self._resolved_builder_map.keys())
            raise RegistryError(
                f"Missing step builders for {len(missing_builders)} configurations",
                unresolvable_types=missing_builders,
                available_builders=available_builders
            )
        
        return self._resolved_builder_map
        
    except Exception as e:
        self.logger.error(f"Failed to create step builder map: {e}")
        raise RegistryError(f"Step builder mapping failed: {e}")
```

### 4. Comprehensive Validation Framework

The template includes a comprehensive validation system that checks multiple aspects of pipeline compatibility:

```python
def _validate_configuration(self) -> None:
    """
    Validate that all DAG nodes have corresponding configs.
    
    Performs comprehensive validation including:
    1. All DAG nodes have matching configurations
    2. All configurations have corresponding step builders
    3. Configuration-specific validation passes
    4. Dependency resolution is possible
    
    Raises:
        ValidationError: If validation fails
    """
    # Skip validation if requested (for testing purposes)
    if self._skip_validation:
        self.logger.info("Skipping configuration validation (requested)")
        return
    try:
        self.logger.info("Validating dynamic pipeline configuration")
        
        # Get resolved mappings
        dag_nodes = list(self._dag.nodes)
        config_map = self._create_config_map()
        builder_map = self._create_step_builder_map()
        
        # Run comprehensive validation
        validation_result = self._validation_engine.validate_dag_compatibility(
            dag_nodes=dag_nodes,
            available_configs=self.configs,
            config_map=config_map,
            builder_registry=builder_map
        )
        
        if not validation_result.is_valid:
            self.logger.error("Configuration validation failed")
            self.logger.error(validation_result.detailed_report())
            raise ValidationError(
                "Dynamic pipeline configuration validation failed",
                validation_errors={
                    'missing_configs': validation_result.missing_configs,
                    'unresolvable_builders': validation_result.unresolvable_builders,
                    'config_errors': validation_result.config_errors,
                    'dependency_issues': validation_result.dependency_issues
                }
            )
        
        # Log warnings if any
        if validation_result.warnings:
            for warning in validation_result.warnings:
                self.logger.warning(warning)
        
        self.logger.info("Configuration validation passed successfully")
        
    except Exception as e:
        self.logger.error(f"Configuration validation failed: {e}")
        raise ValidationError(f"Validation failed: {e}")
```

The validation framework provides:
- **Multi-Level Validation**: Checks DAG, configuration, and builder compatibility
- **Detailed Error Reporting**: Provides specific information about validation failures
- **Warning System**: Reports non-critical issues that may affect performance
- **Skip Option**: Allows validation to be bypassed for testing scenarios

## Advanced Features

### 1. Resolution Preview and Debugging

The template provides detailed preview capabilities for understanding how DAG nodes will be resolved:

```python
def get_resolution_preview(self) -> Dict[str, Any]:
    """
    Get a preview of how DAG nodes will be resolved.
    
    Returns:
        Dictionary with resolution preview information
    """
    try:
        dag_nodes = list(self._dag.nodes)
        preview_data = self._config_resolver.preview_resolution(
            dag_nodes=dag_nodes,
            available_configs=self.configs,
            metadata=self._loaded_metadata
        )
        
        # Convert to display format
        preview = {
            'nodes': len(dag_nodes),
            'resolutions': {}
        }
        
        for node, candidates in preview_data.items():
            if candidates:
                best_candidate = candidates[0]
                preview['resolutions'][node] = {
                    'config_type': best_candidate['config_type'],
                    'confidence': best_candidate['confidence'],
                    'method': best_candidate['method'],
                    'job_type': best_candidate['job_type'],
                    'alternatives': len(candidates) - 1
                }
            else:
                preview['resolutions'][node] = {
                    'config_type': 'UNRESOLVED',
                    'confidence': 0.0,
                    'method': 'none',
                    'job_type': 'N/A',
                    'alternatives': 0
                }
        
        return preview
        
    except Exception as e:
        self.logger.error(f"Failed to generate resolution preview: {e}")
        return {'error': str(e)}
```

### 2. Pipeline Metadata Management

The template includes sophisticated metadata management for integration with execution systems:

```python
def _store_pipeline_metadata(self, assembler: "PipelineAssembler") -> None:
    """
    Store pipeline metadata from template.
    
    This method dynamically discovers and stores metadata from various step types,
    particularly focused on Cradle data loading requests and registration step configurations
    for use in filling execution documents.
    
    Args:
        assembler: PipelineAssembler instance
    """
    # Store Cradle data loading requests if available
    if hasattr(assembler, 'cradle_loading_requests'):
        self.pipeline_metadata['cradle_loading_requests'] = assembler.cradle_loading_requests
        self.logger.info(f"Stored {len(assembler.cradle_loading_requests)} Cradle loading requests")
        
    # Find and store registration steps and configurations
    try:
        # Find all registration steps in the pipeline
        registration_steps = []
        
        # Approach 1: Check step instances dictionary if available
        if hasattr(assembler, 'step_instances'):
            for step_name, step_instance in assembler.step_instances.items():
                # Check for registration step using name pattern or type
                if ("registration" in step_name.lower() or 
                    "registration" in str(type(step_instance)).lower()):
                    registration_steps.append(step_instance)
                    self.logger.info(f"Found registration step: {step_name}")
```

### 3. Execution Document Integration

The template provides comprehensive support for execution document filling, particularly for MODS integration:

```python
def fill_execution_document(self, execution_document: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fill in the execution document with pipeline metadata.
    
    This method populates the execution document with:
    1. Cradle data loading requests (if present in the pipeline)
    2. Registration configurations (if present in the pipeline)
    
    Args:
        execution_document: Execution document to fill
        
    Returns:
        Updated execution document
    """
    if "PIPELINE_STEP_CONFIGS" not in execution_document:
        self.logger.warning("Execution document missing 'PIPELINE_STEP_CONFIGS' key")
        return execution_document
    
    pipeline_configs = execution_document["PIPELINE_STEP_CONFIGS"]

    # 1. Handle Cradle data loading requests
    self._fill_cradle_configurations(pipeline_configs)
    
    # 2. Handle Registration configurations
    self._fill_registration_configurations(pipeline_configs)
    
    return execution_document
```

### 4. Registration Configuration Management

The template includes sophisticated logic for handling model registration configurations:

```python
def _create_execution_doc_config(self, image_uri: str) -> Dict[str, Any]:
    """
    Create the execution document configuration dictionary.
    
    This method dynamically creates an execution document configuration
    by extracting information from registration, payload, and package configurations.
    
    Args:
        image_uri: The URI of the inference image to use
        
    Returns:
        Dictionary with execution document configuration
    """
    # Find needed configs using type name pattern matching
    registration_cfg = None
    payload_cfg = None
    package_cfg = None
    
    for _, cfg in self.configs.items():
        cfg_type_name = type(cfg).__name__.lower()
        if "registration" in cfg_type_name and not "payload" in cfg_type_name:
            registration_cfg = cfg
        elif "payload" in cfg_type_name:
            payload_cfg = cfg
        elif "package" in cfg_type_name:
            package_cfg = cfg
            
    if not registration_cfg:
        self.logger.warning("No registration configuration found for execution document")
        return {}
        
    # Create a basic configuration with required fields
    exec_config = {
        "source_model_inference_image_arn": image_uri,
    }
    
    # Add registration configuration fields
    for field in [
        "model_domain", "model_objective", "source_model_inference_content_types",
        "source_model_inference_response_types", "source_model_inference_input_variable_list",
        "source_model_inference_output_variable_list", "model_registration_region",
        "source_model_region", "aws_region", "model_owner", "region"
    ]:
        if hasattr(registration_cfg, field):
            # Map certain fields to their execution doc names
            if field == "aws_region":
                exec_config["source_model_region"] = getattr(registration_cfg, field)
            elif field == "region":
                exec_config["model_registration_region"] = getattr(registration_cfg, field)
            else:
                exec_config[field] = getattr(registration_cfg, field)
    
    # Add environment variables if entry point is available
    if hasattr(registration_cfg, "inference_entry_point"):
        exec_config["source_model_environment_variable_map"] = {
            "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
            "SAGEMAKER_PROGRAM": registration_cfg.inference_entry_point,
            "SAGEMAKER_REGION": getattr(registration_cfg, "aws_region", "us-east-1"),
            "SAGEMAKER_SUBMIT_DIRECTORY": '/opt/ml/model/code',
        }
        
    return exec_config
```

## Pipeline Parameters and Network Configuration

The template provides standard pipeline parameters and network configuration:

```python
def _get_pipeline_parameters(self) -> List[ParameterString]:
    """
    Get pipeline parameters.
    
    Returns standard parameters used by most pipelines:
    - PIPELINE_EXECUTION_TEMP_DIR: S3 prefix for execution data
    - KMS_ENCRYPTION_KEY_PARAM: KMS key for encryption
    - SECURITY_GROUP_ID: Security group for network isolation
    - VPC_SUBNET: VPC subnet for network isolation
    
    Returns:
        List of pipeline parameters
    """
    return [
        PIPELINE_EXECUTION_TEMP_DIR, KMS_ENCRYPTION_KEY_PARAM,
        SECURITY_GROUP_ID, VPC_SUBNET,
    ]
```

The template imports these constants from the MODS workflow core library with fallback definitions:

```python
# Import constants from core library (with fallback)
try:
    from mods_workflow_core.utils.constants import (
        PIPELINE_EXECUTION_TEMP_DIR,
        KMS_ENCRYPTION_KEY_PARAM,
        PROCESSING_JOB_SHARED_NETWORK_CONFIG,
        SECURITY_GROUP_ID,
        VPC_SUBNET,
    )
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import constants from mods_workflow_core, using local definitions")
    # Define pipeline parameters locally if import fails
    PIPELINE_EXECUTION_TEMP_DIR = ParameterString(name="EXECUTION_S3_PREFIX")
    KMS_ENCRYPTION_KEY_PARAM = ParameterString(name="KMS_ENCRYPTION_KEY_PARAM")
    SECURITY_GROUP_ID = ParameterString(name="SECURITY_GROUP_ID")
    VPC_SUBNET = ParameterString(name="VPC_SUBNET")
    # Also create the network config
    PROCESSING_JOB_SHARED_NETWORK_CONFIG = NetworkConfig(
        enable_network_isolation=False,
        security_group_ids=[SECURITY_GROUP_ID],
        subnets=[VPC_SUBNET],
        encrypt_inter_container_traffic=True,
    )
```

## Utility Methods

### DAG Analysis and Execution Order

```python
def get_step_dependencies(self) -> Dict[str, list]:
    """
    Get the dependencies for each step based on the DAG.
    
    Returns:
        Dictionary mapping step names to their dependencies
    """
    dependencies = {}
    for node in self._dag.nodes:
        dependencies[node] = list(self._dag.get_dependencies(node))
    return dependencies

def get_execution_order(self) -> list:
    """
    Get the topological execution order of steps.
    
    Returns:
        List of step names in execution order
    """
    try:
        return self._dag.topological_sort()
    except Exception as e:
        self.logger.error(f"Failed to get execution order: {e}")
        return list(self._dag.nodes)
```

### Registry Statistics and Validation

```python
def get_builder_registry_stats(self) -> Dict[str, Any]:
    """
    Get statistics about the builder registry.
    
    Returns:
        Dictionary with registry statistics
    """
    return self._builder_registry.get_registry_stats()

def validate_before_build(self) -> bool:
    """
    Validate the configuration before building the pipeline.
    
    Returns:
        True if validation passes, False otherwise
    """
    try:
        self._validate_configuration()
        return True
    except ValidationError:
        return False
```

## Usage Examples

### Basic Usage

```python
from src.cursus.core.compiler.dynamic_template import DynamicPipelineTemplate
from src.cursus.api.dag.base_dag import PipelineDAG

# Create a DAG
dag = PipelineDAG()
dag.add_node("data_load")
dag.add_node("preprocess")
dag.add_node("train")
dag.add_edge("data_load", "preprocess")
dag.add_edge("preprocess", "train")

# Create dynamic template
template = DynamicPipelineTemplate(
    dag=dag,
    config_path="configs/my_pipeline.json",
    sagemaker_session=session,
    role=role
)

# Generate pipeline
pipeline = template.generate_pipeline()
```

### Advanced Usage with Custom Components

```python
from src.cursus.core.compiler.config_resolver import StepConfigResolver
from src.cursus.registry.builder_registry import StepBuilderRegistry

# Create custom resolver and registry
custom_resolver = StepConfigResolver()
custom_registry = StepBuilderRegistry()

# Create template with custom components
template = DynamicPipelineTemplate(
    dag=dag,
    config_path="configs/my_pipeline.json",
    config_resolver=custom_resolver,
    builder_registry=custom_registry,
    sagemaker_session=session,
    role=role
)

# Preview resolution before building
preview = template.get_resolution_preview()
print(f"Resolution preview: {preview}")

# Validate configuration
if template.validate_before_build():
    pipeline = template.generate_pipeline()
else:
    print("Validation failed")
```

### Integration with MODS

The `DynamicPipelineTemplate` serves as the base for MODS integration through the `MODSPipelineDAGCompiler`, which decorates this class with the `MODSTemplate` decorator before instantiation:

```python
# In MODSPipelineDAGCompiler
@MODSTemplate
class MODSDynamicPipelineTemplate(DynamicPipelineTemplate):
    pass

# Usage
template = MODSDynamicPipelineTemplate(
    dag=dag,
    config_path=config_path,
    **kwargs
)
```

This allows dynamic pipeline generation while maintaining compatibility with the MODS system for execution document filling and other MODS-specific features.

## Benefits of the Design

The `DynamicPipelineTemplate` design provides several key benefits:

1. **Universal Compatibility**: Works with any PipelineDAG structure without custom template classes
2. **Intelligent Resolution**: Automatically maps DAG nodes to configurations and builders
3. **Comprehensive Validation**: Ensures compatibility before pipeline generation
4. **Detailed Debugging**: Provides extensive preview and debugging capabilities
5. **MODS Integration**: Seamlessly integrates with MODS execution document system
6. **Performance Optimization**: Only loads required configuration classes
7. **Error Resilience**: Graceful error handling with detailed feedback
8. **Extensibility**: Supports custom resolvers and registries

## Related Components

- [Config Resolver](config_resolver.md): Intelligent DAG node-to-configuration mapping
- [Step Builder Registry](../registry/builder_registry.md): Configuration-to-builder mapping
- [Validation Engine](validation.md): Comprehensive validation framework
- [Pipeline Template Base](../assembler/pipeline_template_base.md): Base template class
- [Pipeline DAG](../../api/dag/base_dag.md): DAG structure definition
