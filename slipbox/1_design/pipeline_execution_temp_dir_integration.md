---
tags:
  - design
  - pipeline_api
  - cursus
  - configuration
  - output_destinations
keywords:
  - PIPELINE_EXECUTION_TEMP_DIR
  - pipeline parameters
  - output destinations
  - step builders
  - portability
  - dynamic configuration
  - s3 locations
  - backward compatibility
  - parameter substitution
  - runtime configuration
topics:
  - pipeline configuration management
  - output destination handling
  - parameter integration
  - step builder architecture
language: python
date of note: 2025-09-17
---

# Integration of PIPELINE_EXECUTION_TEMP_DIR in Cursus Framework

## Overview

This design document outlines a proposal to enhance the Cursus framework to support using `PIPELINE_EXECUTION_TEMP_DIR` as an alternative base directory for output destinations. This modification enables better portability and runtime configuration of output locations without significantly changing the current system architecture.

## Background

### Current Implementation in Cursus Framework

Currently, the Cursus framework relies on `pipeline_s3_loc` from configuration files to determine output destinations in step builders. This value is constructed based on:

```python
@property
def pipeline_s3_loc(self) -> str:
    """Get S3 location for pipeline artifacts."""
    if self._pipeline_s3_loc is None:
        pipeline_subdirectory = "MODS"
        pipeline_subsubdirectory = f"{self.pipeline_name}_{self.pipeline_version}"
        self._pipeline_s3_loc = (
            f"s3://{self.bucket}/{pipeline_subdirectory}/{pipeline_subsubdirectory}"
        )
    return self._pipeline_s3_loc
```

Examining the actual implementation in concrete step builders like `PackageStepBuilder`, we see that destinations are hard-coded using string interpolation:

```python
# From PackageStepBuilder._get_outputs:
destination = f"{self.config.pipeline_s3_loc}/packaging/{logical_name}"
```

This approach doesn't support runtime configuration via pipeline parameters.

### Alternative Approach in regional_xgboost.py

The `regional_xgboost.py` implementation demonstrates a more flexible approach: all destinations are constructed by joining `PIPELINE_EXECUTION_TEMP_DIR` with appropriate subdirectories. This allows the base output location to be specified at runtime as a parameter:

```python
# In regional_xgboost.py
class AtoZRegionalXGBoostModel:
    ARTIFACT_LOCATION = Join(on="/", values=[PIPELINE_EXECUTION_TEMP_DIR, "Artifacts"])
    TRAINING_DATA_LOCATION = Join(on="/", values=[PIPELINE_EXECUTION_TEMP_DIR, "data/training"])
    VALIDATION_DATA_LOCATION = Join(on="/", values=[PIPELINE_EXECUTION_TEMP_DIR, "data/validation"])
    # ... other locations defined similarly
```

The `PIPELINE_EXECUTION_TEMP_DIR` parameter is already defined in the Cursus core (`dynamic_template.py`):

```python
# In dynamic_template.py
# Import constants from core library (with fallback)
try:
    from mods_workflow_core.utils.constants import (
        PIPELINE_EXECUTION_TEMP_DIR,
        # ... other constants
    )
except ImportError:
    # Define pipeline parameters locally if import fails
    PIPELINE_EXECUTION_TEMP_DIR = ParameterString(name="EXECUTION_S3_PREFIX")
    # ... other fallback definitions
```

It's included in pipeline parameters via the `_get_pipeline_parameters` method:

```python
def _get_pipeline_parameters(self) -> List[ParameterString]:
    """
    Get pipeline parameters.
    """
    return [
        PIPELINE_EXECUTION_TEMP_DIR,
        KMS_ENCRYPTION_KEY_PARAM,
        SECURITY_GROUP_ID,
        VPC_SUBNET,
    ]
```

However, this parameter is not currently being utilized for output destination generation in step builders. After analyzing the actual code:

1. `StepBuilderBase` has no mechanism to access pipeline parameters
2. `PipelineAssembler` stores `pipeline_parameters` but doesn't pass them to step builders
3. Step builders directly use `config.pipeline_s3_loc` with string interpolation
4. `regional_xgboost.py` successfully uses `Join` with `PIPELINE_EXECUTION_TEMP_DIR`

## Goals

- Enable the use of `PIPELINE_EXECUTION_TEMP_DIR` as the base directory for output destinations when specified
- Provide a fallback to the current `pipeline_s3_loc` approach when `PIPELINE_EXECUTION_TEMP_DIR` is not specified
- Implement this change with minimal disruption to existing code and functionality
- Maintain backward compatibility with existing configurations

## Design

### Complete Data Flow Analysis

After examining the entire chain from `mods_pipeline_adapter.py` → `dag_compiler.py` → `dynamic_template.py` → `pipeline_template_base.py` → `pipeline_assembler.py`, the actual data flow reveals **multiple critical gaps**:

#### **Current Broken Flow:**
1. **XGBoostCursusPipelineAdapter** → Creates `PipelineDAGCompiler` but **DOESN'T pass PIPELINE_EXECUTION_TEMP_DIR**
2. **PipelineDAGCompiler.create_template()** → Creates `DynamicPipelineTemplate` but **DOESN'T pass pipeline parameters**
3. **DynamicPipelineTemplate._get_pipeline_parameters()** → Returns `[PIPELINE_EXECUTION_TEMP_DIR, ...]` but **METHOD NEVER CALLED**
4. **PipelineTemplateBase.generate_pipeline()** → Calls assembler with `pipeline_parameters=self._get_pipeline_parameters()` 
5. **PipelineAssembler.__init__()** → Stores `self.pipeline_parameters = pipeline_parameters or []` 
6. **PipelineAssembler._initialize_step_builders()** → Creates step builders but **DOESN'T pass parameters to builders**
7. **PipelineAssembler._generate_outputs()** → Uses `config.pipeline_s3_loc` directly, **ignores pipeline_parameters**

#### **Critical Gaps Identified:**

**Gap 1: Top-Level Parameter Source Missing**
```python
# In mods_pipeline_adapter.py - NO PIPELINE_EXECUTION_TEMP_DIR usage
self.dag_compiler = PipelineDAGCompiler(
    config_path=self.config_path,
    sagemaker_session=self.sagemaker_session,
    role=self.execution_role,
    # MISSING: pipeline_parameters or execution_s3_prefix
)
```

**Gap 2: DAG Compiler Parameter Passing Missing**
```python
# In dag_compiler.py create_template() - NO pipeline parameters passed
template = DynamicPipelineTemplate(
    dag=dag,
    config_path=self.config_path,
    config_resolver=self.config_resolver,
    builder_registry=self.builder_registry,
    sagemaker_session=self.sagemaker_session,
    role=self.role,
    # MISSING: pipeline_parameters
)
```

**Gap 3: Template to Assembler Connection Works**
```python
# In pipeline_template_base.py generate_pipeline() - THIS WORKS
template = PipelineAssembler(
    # ...
    pipeline_parameters=self._get_pipeline_parameters(),  # ✓ This works
    # ...
)
```

**Gap 4: Assembler Parameter Usage Missing** 
```python
# In PipelineAssembler._initialize_step_builders() - DOESN'T use parameters
builder = builder_cls(config=config, ...)
# MISSING: Pass execution prefix to builder here
```

#### **End-to-End Flow Must Be:**
```
mods_pipeline_adapter → dag_compiler → dynamic_template → pipeline_template_base → pipeline_assembler → step_builders
        ↓                    ↓              ↓                        ↓                      ↓                ↓
   Provide param    →   Pass param   →   Store param    →    Call _get_params()  →    Use param     →  Apply param
```

**Critical Missing Layer: PipelineTemplateBase**

After examining `pipeline_template_base.py`, there's a **critical intermediate layer** that bridges the template and assembler:

```python
# In PipelineTemplateBase.generate_pipeline()
template = PipelineAssembler(
    dag=dag,
    config_map=config_map,
    step_builder_map=step_builder_map,
    sagemaker_session=self.session,
    role=self.role,
    pipeline_parameters=self._get_pipeline_parameters(),  # KEY INTEGRATION POINT!
    notebook_root=self.notebook_root,
    registry_manager=self._registry_manager,
    dependency_resolver=self._dependency_resolver,
)
```

**Gap 3.5: PipelineTemplateBase Integration Missing**
- `PipelineTemplateBase.generate_pipeline()` calls `self._get_pipeline_parameters()` 
- `DynamicPipelineTemplate` overrides `_get_pipeline_parameters()` to return custom parameters
- `PipelineTemplateBase` passes these parameters to `PipelineAssembler` constructor
- This is the **actual bridge** between template parameter storage and assembler parameter usage

In contrast, `regional_xgboost.py` bypasses the entire Cursus framework and uses `Join` operations with `PIPELINE_EXECUTION_TEMP_DIR` directly.

### Proposed Solution

Based on the detailed code analysis, we propose a comprehensive solution with three main components:

1. **Parameter Access**: Give step builders access to pipeline parameters by modifying `StepBuilderBase` and `PipelineAssembler`
2. **Path Resolution**: Add a new method in `StepBuilderBase` to intelligently select between runtime parameter and static configuration
3. **Path Construction**: Replace string interpolation with `Join` operations to ensure proper parameter substitution at runtime

This approach enables the use of `PIPELINE_EXECUTION_TEMP_DIR` while maintaining backward compatibility with the current approach.

#### 1. Enhance `StepBuilderBase` with Output Path Management (Dependency Direction: PipelineAssembler → StepBuilderBase)

**Note**: Step builders do not depend on or import PipelineAssembler. The dependency direction is important: PipelineAssembler calls and initializes step builders, setting the pipeline_parameters attribute that step builders can use.

First, add a `pipeline_parameters` attribute to the `StepBuilderBase` class:

```python
class StepBuilderBase(ABC):
    # ...existing code...
    
    def __init__(
        self,
        config: BasePipelineConfig,
        spec: Optional[StepSpecification] = None,
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None,
        registry_manager: Optional[RegistryManager] = None,
        dependency_resolver: Optional[UnifiedDependencyResolver] = None,
    ):
        """
        Initialize base step builder.
        """
        self.config = config
        self.spec = spec
        self.session = sagemaker_session
        self.role = role
        self.notebook_root = notebook_root or Path.cwd()
        self._registry_manager = registry_manager
        self._dependency_resolver = dependency_resolver
        self.execution_prefix: Optional[Union[ParameterString, str]] = None  # Initialize execution prefix
        
        # ...existing code...

    def set_execution_prefix(self, execution_prefix: Optional[Union[ParameterString, str]] = None) -> None:
        """
        Set the execution prefix for dynamic output path resolution.
        
        This method is called by PipelineAssembler to provide the execution prefix
        that step builders use for dynamic output path generation.
        
        Based on analysis of regional_xgboost.py, only PIPELINE_EXECUTION_TEMP_DIR
        is used by step builders for output paths. Other pipeline parameters 
        (KMS_ENCRYPTION_KEY_PARAM, VPC_SUBNET, SECURITY_GROUP_ID) are used at
        the pipeline level, not in step builders.
        
        Args:
            execution_prefix: The execution prefix that can be either:
                           - ParameterString: PIPELINE_EXECUTION_TEMP_DIR from pipeline parameters
                           - str: config.pipeline_s3_loc as fallback
                           - None: No parameter found, will fall back to config.pipeline_s3_loc
        """
        self.execution_prefix = execution_prefix
        self.log_debug("Set execution prefix: %s", execution_prefix)
```

Then, add a new method to `StepBuilderBase` that will resolve the base path for output destinations:

```python
def _get_base_output_path(self):
    """
    Get base path for output destinations with PIPELINE_EXECUTION_TEMP_DIR support.
    
    This method checks for the execution_prefix (set by PipelineAssembler) and falls
    back to the traditional pipeline_s3_loc from config.
    
    Returns:
        The base path for output destinations. Returns a ParameterString if
        execution_prefix was set from PIPELINE_EXECUTION_TEMP_DIR, otherwise 
        returns the string value from config.pipeline_s3_loc.
    """
    # Check if execution_prefix has been set by PipelineAssembler
    if hasattr(self, "execution_prefix") and self.execution_prefix is not None:
        self.log_info("Using execution_prefix for base output path")
        return self.execution_prefix
    
    # Fall back to pipeline_s3_loc from config (current behavior)
    base_path = self.config.pipeline_s3_loc
    self.log_debug("No execution_prefix set, using config.pipeline_s3_loc for base output path")
    return base_path
```

**Critical Discovery: Unified Path Construction with Join()**

After analyzing existing Cursus step builders and `regional_xgboost.py`, we've confirmed that `sagemaker.workflow.functions.Join()` is the unified approach that works correctly with both `str` and `ParameterString` objects.

**Evidence from Current Code:**
```python
# From builder_xgboost_training_step.py - ALREADY USING Join() correctly:
from sagemaker.workflow.functions import Join
channels = {
    "train": TrainingInput(s3_data=Join(on="/", values=[base_path, "train/"])),
    "val": TrainingInput(s3_data=Join(on="/", values=[base_path, "val/"])),  
    "test": TrainingInput(s3_data=Join(on="/", values=[base_path, "test/"])),
}
# Here base_path can be either str or ParameterString - Join() handles both!

# From builder_package_step.py - INCONSISTENT approach:
destination = f"{self.config.pipeline_s3_loc}/packaging/{logical_name}"  # BREAKS with ParameterString
```

**The Solution: Always Use Join()**
```python
# ALWAYS use this unified approach:
from sagemaker.workflow.functions import Join
destination = Join(on="/", values=[base_output_path, "packaging", logical_name])

# NEVER use f-strings for paths that might contain parameters:
destination = f"{base_output_path}/packaging/{logical_name}"  # BREAKS with ParameterString
```

This unified approach ensures that SageMaker correctly handles parameter substitution at runtime for both string values (from `pipeline_s3_loc`) and `ParameterString` objects (from `PIPELINE_EXECUTION_TEMP_DIR`).

The solution maintains consistency with patterns already established in `mods_workflow_core.utils.constants`, such as the utility function for input configuration:

```python
# Existing utility function in mods_workflow_core.utils.constants
def get_input_config_processing_input_for_pipeline_execution(step_name):
    return ProcessingInput(
        source=Join(on="/", values=[PIPELINE_EXECUTION_TEMP_DIR, step_name, "input", "config"]),
        destination="/opt/ml/processing/config/",
        input_name="config",
    )
```

We could potentially extend our solution with similar utility methods for common output path patterns.

#### 2. Modify `PipelineAssembler` to Pass Pipeline Parameters to Builders

Update `PipelineAssembler._initialize_step_builders()` to pass pipeline parameters to builders:

```python
def _initialize_step_builders(self) -> None:
    """Initialize step builders for all steps in the DAG."""
    logger.info("Initializing step builders")
    start_time = time.time()

    for step_name in self.dag.nodes:
        try:
            config = self.config_map[step_name]
            config_class_name = type(config).__name__
            step_type = CONFIG_STEP_REGISTRY.get(config_class_name)
            if not step_type:
                step_type = BasePipelineConfig.get_step_name(config_class_name)
                logger.warning(
                    f"Config class {config_class_name} not found in registry, using derived name: {step_type}"
                )

            builder_cls = self.step_builder_map[step_type]

            # Initialize the builder with dependency components
            builder = builder_cls(
                config=config,
                sagemaker_session=self.sagemaker_session,
                role=self.role,
                notebook_root=self.notebook_root,
                registry_manager=self._registry_manager,  
                dependency_resolver=self._dependency_resolver,
            )
            
            # Pass execution prefix to the builder using the public method
            # Find PIPELINE_EXECUTION_TEMP_DIR in pipeline_parameters and pass it to the builder
            execution_prefix = None
            for param in self.pipeline_parameters:
                if hasattr(param, "name") and param.name == "EXECUTION_S3_PREFIX":
                    execution_prefix = param
                    break
            
            if execution_prefix:
                builder.set_execution_prefix(execution_prefix)
            # If no PIPELINE_EXECUTION_TEMP_DIR found, builder will fall back to config.pipeline_s3_loc
            
            self.step_builders[step_name] = builder
            logger.info(f"Initialized builder for step {step_name} of type {step_type}")
        except Exception as e:
            logger.error(f"Error initializing builder for step {step_name}: {e}")
            raise ValueError(f"Failed to initialize step builder for {step_name}: {e}") from e

    elapsed_time = time.time() - start_time
    logger.info(
        f"Initialized {len(self.step_builders)} step builders in {elapsed_time:.2f} seconds"
    )
```

#### 3. Update `_generate_outputs` in `PipelineAssembler`

Modify the `_generate_outputs` method to use the new `_get_base_output_path` method:

```python
def _generate_outputs(self, step_name: str) -> Dict[str, Any]:
    """Generate outputs dictionary using step builder's specification."""
    builder = self.step_builders[step_name]
    
    # If builder has no specification, return empty dict
    if not hasattr(builder, "spec") or not builder.spec:
        logger.warning(f"Step {step_name} has no specification, returning empty outputs")
        return {}

    # Get base S3 location using the new method
    base_s3_loc = builder._get_base_output_path()

    # Generate outputs dictionary based on specification
    outputs = {}
    step_type = builder.spec.step_type.lower()

    # Use each output specification to generate standard output path
    for logical_name, output_spec in builder.spec.outputs.items():
        # Standard path pattern using Join instead of f-string to ensure proper parameter substitution
        from sagemaker.workflow.functions import Join
        outputs[logical_name] = Join(on="/", values=[base_s3_loc, step_type, logical_name])

        # Add debug log
        logger.debug(
            f"Generated output for {step_name}.{logical_name}: {outputs[logical_name]}"
        )

    return outputs
```

#### 4. Update Individual Step Builders

Update the `_get_outputs` method in step builders to use the new base path resolution:

```python
def _get_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
    """Get outputs for the step using specification and contract."""
    if not self.spec:
        raise ValueError("Step specification is required")

    if not self.contract:
        raise ValueError("Script contract is required for output mapping")

    processing_outputs = []

    # Get the base output path (using PIPELINE_EXECUTION_TEMP_DIR if available)
    base_output_path = self._get_base_output_path()

    # Process each output in the specification
    for _, output_spec in self.spec.outputs.items():
        logical_name = output_spec.logical_name

        # Get container path from contract
        container_path = None
        if logical_name in self.contract.expected_output_paths:
            container_path = self.contract.expected_output_paths[logical_name]
        else:
            raise ValueError(f"No container path found for output: {logical_name}")

        # Try to find destination in outputs
        destination = None

        # Look in outputs by logical name
        if logical_name in outputs:
            destination = outputs[logical_name]
        else:
            # Generate destination from base path using Join instead of f-string
            from sagemaker.workflow.functions import Join
            destination = Join(on="/", values=[base_output_path, self.spec.step_type.lower(), logical_name])
            self.log_info(
                "Using generated destination for '%s': %s",
                logical_name,
                destination,
            )

        processing_outputs.append(
            ProcessingOutput(
                output_name=logical_name,
                source=container_path,
                destination=destination,
            )
        )

    return processing_outputs
```

## Extended Implementation Plan

Based on the complete end-to-end analysis, we need to implement changes across **five critical layers** to establish the complete parameter flow:

### Phase 0: Top-Level Parameter Integration (Days 1-2)

**0a. Update XGBoostCursusPipelineAdapter to Support PIPELINE_EXECUTION_TEMP_DIR**
```python
# File: mods_pipeline_adapter/mods_pipeline_adapter.py
# Location: XGBoostCursusPipelineAdapter.__init__ method

def __init__(
    self,
    sagemaker_session=None,
    execution_role=None,
    regional_alias=DEFAULT_REGION,
    execution_s3_prefix=None,  # NEW: Add execution prefix parameter
):
    """
    Initialize the adapter with configuration and session details.

    Args:
        sagemaker_session: SageMaker pipeline session
        execution_role: IAM role for pipeline execution
        regional_alias: Region code (NA, EU, FE, etc.) for configuration path
        execution_s3_prefix: Optional S3 prefix for pipeline execution artifacts
    """
    # Store execution prefix for pipeline parameters
    self.execution_s3_prefix = execution_s3_prefix
    
    # ...existing initialization code...
    
    # Initialize compiler with pipeline parameters
    pipeline_parameters = []
    if self.execution_s3_prefix:
        # Create PIPELINE_EXECUTION_TEMP_DIR parameter with user-provided value
        from sagemaker.workflow.parameters import ParameterString
        pipeline_execution_param = ParameterString(
            name="EXECUTION_S3_PREFIX", 
            default_value=self.execution_s3_prefix
        )
        pipeline_parameters.append(pipeline_execution_param)
    
    self.dag_compiler = PipelineDAGCompiler(
        config_path=self.config_path,
        sagemaker_session=self.sagemaker_session,
        role=self.execution_role,
        pipeline_parameters=pipeline_parameters,  # NEW: Pass parameters to compiler
    )
```

**0b. Update PipelineDAGCompiler to Accept and Forward Pipeline Parameters**
```python
# File: cursus/core/compiler/dag_compiler.py
# Location: PipelineDAGCompiler.__init__ method

def __init__(
    self,
    config_path: str,
    sagemaker_session: Optional[PipelineSession] = None,
    role: Optional[str] = None,
    config_resolver: Optional[StepConfigResolver] = None,
    builder_registry: Optional[StepBuilderRegistry] = None,
    pipeline_parameters: Optional[List[ParameterString]] = None,  # NEW parameter
    **kwargs,
):
    """
    Initialize compiler with configuration and session.

    Args:
        config_path: Path to configuration file
        sagemaker_session: SageMaker session for pipeline execution
        role: IAM role for pipeline execution
        config_resolver: Custom config resolver (optional)
        builder_registry: Custom builder registry (optional)
        pipeline_parameters: Pipeline parameters to pass to template (optional)
        **kwargs: Additional arguments for template constructor
    """
    # ...existing initialization code...
    
    # Store pipeline parameters for template creation
    self.pipeline_parameters = pipeline_parameters or []
    
    # ...rest of existing code...
```

**0c. Update PipelineDAGCompiler.create_template() to Pass Parameters**
```python
# File: cursus/core/compiler/dag_compiler.py  
# Location: create_template method

def create_template(self, dag: PipelineDAG, **kwargs) -> "DynamicPipelineTemplate":
    """
    Create a pipeline template from the DAG without generating the pipeline.
    """
    try:
        from .dynamic_template import DynamicPipelineTemplate

        # Merge kwargs with default values
        template_kwargs = {**self.template_kwargs}
        template_kwargs.update(kwargs)

        # Pass pipeline parameters to template
        template = DynamicPipelineTemplate(
            dag=dag,
            config_path=self.config_path,
            config_resolver=self.config_resolver,
            builder_registry=self.builder_registry,
            sagemaker_session=self.sagemaker_session,
            role=self.role,
            pipeline_parameters=self.pipeline_parameters,  # NEW: Pass parameters
            **template_kwargs,
        )

        return template
        
    except Exception as e:
        raise PipelineAPIError(f"Template creation failed: {e}") from e
```

**0d. Update DynamicPipelineTemplate to Store and Use Pipeline Parameters**
```python
# File: cursus/core/compiler/dynamic_template.py
# Location: DynamicPipelineTemplate.__init__ method

def __init__(
    self,
    dag: PipelineDAG,
    config_path: str,
    config_resolver: Optional[StepConfigResolver] = None,
    builder_registry: Optional[StepBuilderRegistry] = None,
    skip_validation: bool = False,
    pipeline_parameters: Optional[List[ParameterString]] = None,  # NEW parameter
    **kwargs,
):
    """
    Initialize dynamic template.

    Args:
        dag: PipelineDAG instance defining pipeline structure
        config_path: Path to configuration file
        config_resolver: Custom config resolver (optional)
        builder_registry: Custom builder registry (optional)
        pipeline_parameters: Custom pipeline parameters (optional)
        **kwargs: Additional arguments for base template
    """
    # Store custom pipeline parameters
    self._custom_pipeline_parameters = pipeline_parameters or []
    
    # ...existing initialization code...
```

**0e. Update DynamicPipelineTemplate._get_pipeline_parameters() to Use Custom Parameters**
```python
# File: cursus/core/compiler/dynamic_template.py
# Location: _get_pipeline_parameters method

def _get_pipeline_parameters(self) -> List[ParameterString]:
    """
    Get pipeline parameters.

    Returns custom parameters if provided, otherwise returns standard parameters:
    - PIPELINE_EXECUTION_TEMP_DIR: S3 prefix for execution data
    - KMS_ENCRYPTION_KEY_PARAM: KMS key for encryption
    - SECURITY_GROUP_ID: Security group for network isolation
    - VPC_SUBNET: VPC subnet for network isolation

    Returns:
        List of pipeline parameters
    """
    # If custom parameters provided, use them (with standard parameters as fallback)
    if self._custom_pipeline_parameters:
        # Merge custom and standard parameters, avoiding duplicates
        all_parameters = list(self._custom_pipeline_parameters)
        standard_params = [
            KMS_ENCRYPTION_KEY_PARAM,
            SECURITY_GROUP_ID,
            VPC_SUBNET,
        ]
        
        # Add standard parameters that aren't already in custom parameters
        custom_param_names = {p.name for p in self._custom_pipeline_parameters if hasattr(p, 'name')}
        for param in standard_params:
            if hasattr(param, 'name') and param.name not in custom_param_names:
                all_parameters.append(param)
                
        return all_parameters
    
    # Default behavior: return standard parameters
    return [
        PIPELINE_EXECUTION_TEMP_DIR,
        KMS_ENCRYPTION_KEY_PARAM,
        SECURITY_GROUP_ID,
        VPC_SUBNET,
    ]
```

## Implementation Plan

Based on the actual code analysis, we'll implement targeted changes at the precise integration points identified:

### Phase 1: StepBuilderBase Enhancement (Days 1-2)

1. **Update StepBuilderBase constructor and add execution prefix method**
   ```python
   # File: cursus/core/base/builder_base.py
   # Location: StepBuilderBase.__init__ method
   
   def __init__(self, ...):
       # ...existing code...
       self.execution_prefix: Optional[Union[ParameterString, str]] = None  # Initialize execution prefix
   
   def set_execution_prefix(self, execution_prefix: Optional[Union[ParameterString, str]] = None) -> None:
       """Set execution prefix for output path resolution."""
       self.execution_prefix = execution_prefix
       self.log_debug("Set execution prefix: %s", execution_prefix)
   
   def _get_base_output_path(self):
       """Get base output path with PIPELINE_EXECUTION_TEMP_DIR support."""
       if hasattr(self, "execution_prefix") and self.execution_prefix is not None:
           self.log_info("Using execution_prefix for base output path")
           return self.execution_prefix
       
       base_path = self.config.pipeline_s3_loc
       self.log_debug("No execution_prefix set, using config.pipeline_s3_loc")
       return base_path
   ```

### Phase 2: PipelineAssembler Modifications (Days 3-4)

2. **Update PipelineAssembler._initialize_step_builders (~line 107)**
   ```python
   # File: cursus/core/assembler/pipeline_assembler.py
   # Location: _initialize_step_builders method, after builder instantiation
   
   # CRITICAL FIX: Add parameter passing after builder creation
   builder = builder_cls(config=config, ...)  # existing code
   
   # NEW CODE: Extract and pass PIPELINE_EXECUTION_TEMP_DIR
   execution_prefix = None
   for param in self.pipeline_parameters:
       if hasattr(param, "name") and param.name == "EXECUTION_S3_PREFIX":
           execution_prefix = param
           break
   
   if execution_prefix:
       builder.set_execution_prefix(execution_prefix)
       logger.info(f"Set execution prefix for {step_name}")
   
   self.step_builders[step_name] = builder
   ```

3. **Update PipelineAssembler._generate_outputs (~line 200)**
   ```python
   # File: cursus/core/assembler/pipeline_assembler.py
   # Location: _generate_outputs method
   
   def _generate_outputs(self, step_name: str) -> Dict[str, Any]:
       builder = self.step_builders[step_name]
       
       # REPLACE existing config-based approach with builder method
       # OLD: base_s3_loc = getattr(config, "pipeline_s3_loc", "s3://default-bucket/pipeline")
       base_s3_loc = builder._get_base_output_path()  # NEW
       
       outputs = {}
       step_type = builder.spec.step_type.lower()
       
       for logical_name, output_spec in builder.spec.outputs.items():
           # REPLACE f-string with Join for ParameterString compatibility
           # OLD: outputs[logical_name] = f"{base_s3_loc}/{step_type}/{logical_name}"
           from sagemaker.workflow.functions import Join  # NEW
           outputs[logical_name] = Join(on="/", values=[base_s3_loc, step_type, logical_name])  # NEW
           
       return outputs
   ```

### Phase 3: Step Builder Updates (Days 5-6)

4. **Update PackageStepBuilder._get_outputs**
   ```python
   # File: cursus/steps/builders/builder_package_step.py
   # Location: _get_outputs method, destination generation
   
   # REPLACE f-string approach with Join + base path method
   # OLD: destination = f"{self.config.pipeline_s3_loc}/packaging/{logical_name}"
   base_output_path = self._get_base_output_path()  # NEW
   from sagemaker.workflow.functions import Join  # NEW
   destination = Join(on="/", values=[base_output_path, "packaging", logical_name])  # NEW
   ```

5. **Update other step builders** (using same pattern as PackageStepBuilder)
   - Apply same Join() + _get_base_output_path() pattern to all step builders in `cursus/steps/builders/`
   - Priority: ProcessingStep builders first, then TrainingStep, ModelStep builders

### Phase 4: Testing and Validation (Days 7-8)

6. **Unit Tests**
   - Test `_get_base_output_path()` with and without execution_prefix
   - Test parameter extraction in `_initialize_step_builders()`
   - Test Join-based path construction in `_generate_outputs()`
   - Test backward compatibility with existing config-based approach

7. **Integration Tests**
   - Test complete pipeline with PIPELINE_EXECUTION_TEMP_DIR parameter
   - Verify f-string to Join() migration doesn't break existing pipelines
   - Test parameter substitution at SageMaker runtime

### Phase 5: Documentation (Day 9-10)

8. **Update Documentation**
   - Document the new _get_base_output_path() method
   - Add examples of using PIPELINE_EXECUTION_TEMP_DIR
   - Create migration guide for existing step builders

### Rollout Strategy

1. **Backward Compatibility**: The solution preserves existing behavior for all current code
2. **Gradual Adoption**: Teams can opt-in to using PIPELINE_EXECUTION_TEMP_DIR at their own pace
3. **Feature Flag**: We'll use runtime detection of parameters rather than explicit feature flags

## Benefits

This design provides several key benefits:

### Technical Benefits
- **Runtime Configurability**: Enables dynamic output location configuration without code changes
- **Consistent Path Construction**: Unified approach for all step builders using `Join`
- **Proper Parameter Handling**: Ensures reliable parameter substitution at runtime
- **Backward Compatibility**: Maintains support for existing `pipeline_s3_loc`-based code

### Operational Benefits
- **Environment Portability**: Easier to move pipelines between development, testing, and production environments
- **Simplifies CI/CD**: Allows automated systems to inject different output locations per environment
- **Reduces Configuration Management**: No need to maintain different configs per environment
- **Cleaner Execution**: Allows isolation of artifacts between execution runs

### Developer Benefits
- **Aligned with Existing Patterns**: Matches the successful approach in `regional_xgboost.py`
- **Intuitive Implementation**: Follows natural path construction patterns
- **Reduced Boilerplate**: Less custom code needed in step builders
- **Explicit Path Construction**: `Join` makes the construction pattern more obvious

## Considerations

- The use of `PIPELINE_EXECUTION_TEMP_DIR` in property references needs to be handled carefully to ensure compatibility with SageMaker Pipeline execution
- Parameter handling in SageMaker requires consideration of when string interpolation happens vs. when parameters are resolved
- When `PIPELINE_EXECUTION_TEMP_DIR` is used, we need to ensure that all steps consistently use it for output destinations
- The SageMaker Python SDK handles the parameter substitution when using both string values (`pipeline_s3_loc`) and `ParameterString` objects (`EXECUTION_S3_PREFIX`) in paths
- The actual parameter name expected at runtime is "EXECUTION_S3_PREFIX", not "PIPELINE_EXECUTION_TEMP_DIR", so our code must check for this parameter name
- **Critical**: String interpolation with f-strings (`f"{base_s3_loc}/{step_type}/{logical_name}"`) does NOT work reliably with `ParameterString` objects. Always use `Join(on="/", values=[base_s3_loc, step_type, logical_name])` for combining paths with parameters
- In `regional_xgboost.py`, the `sagemaker.workflow.functions.Join` utility is used for this reason: `Join(on="/", values=[PIPELINE_EXECUTION_TEMP_DIR, "Artifacts"])`
- Using `Join` ensures proper parameter substitution at runtime, which is essential for SageMaker's execution model

## References

- PIPELINE_EXECUTION_TEMP_DIR is imported from `mods_workflow_core.utils.constants` in `dynamic_template.py`, defined as:
  ```python
  # Predefined Pipeline Parameters
  PIPELINE_EXECUTION_TEMP_DIR = ParameterString(name="EXECUTION_S3_PREFIX")
  KMS_ENCRYPTION_KEY_PARAM = ParameterString(name="KMS_ENCRYPTION_KEY_PARAM")
  VPC_SUBNET = ParameterString(name="VPC_SUBNET")
  SECURITY_GROUP_ID = ParameterString(name="SECURITY_GROUP_ID")
  ```

- There's also a convenience function in `mods_workflow_core.utils.constants` for common input config patterns:
  ```python
  # Predefined Pipeline input location for all input config
  def get_input_config_processing_input_for_pipeline_execution(step_name):
      return ProcessingInput(
          source=Join(on="/", values=[PIPELINE_EXECUTION_TEMP_DIR, step_name, "input", "config"]),
          destination="/opt/ml/processing/config/",
          input_name="config",
      )
  ```

- The approach in `regional_xgboost.py` uses `Join(on="/", values=[PIPELINE_EXECUTION_TEMP_DIR, "subdir"])` to create destinations
- Step builders currently use `self.config.pipeline_s3_loc` to generate output destinations
- `pipeline_s3_loc` is constructed as `s3://{bucket}/MODS/{pipeline_name}_{pipeline_version}`

## Related Documents

- [Cursus Framework Architecture](./cursus_framework_architecture.md) - Core framework architecture document
- [Cursus Framework Output Management](./cursus_framework_output_management.md) - Output destination management strategies
