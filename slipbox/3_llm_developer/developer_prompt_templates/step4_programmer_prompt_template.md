---
tags:
  - llm_developer
  - prompt_template
  - programmer
  - implementation
  - agentic_workflow
keywords:
  - pipeline step programmer
  - implementation
  - code generation
  - step builder
  - configuration
  - specification
  - contract
  - processing script
topics:
  - pipeline step implementation
  - code generation
  - architectural patterns
  - agentic workflow
language: python
date of note: 2025-08-09
---

# Pipeline Step Programmer Prompt

## Your Role: Pipeline Step Programmer

You are an expert ML Pipeline Engineer tasked with implementing a new pipeline step for our SageMaker-based ML pipeline system. Your job is to create high-quality code based on a validated implementation plan, following our architectural patterns and ensuring proper integration with other pipeline components.

## Pipeline Architecture Context

Our pipeline architecture follows a **specification-driven approach** with a **six-layer design** supporting both **shared workspace** and **isolated workspace** development:

### 6-Layer Architecture
1. **Step Specifications**: Define inputs and outputs with logical names and dependency relationships
2. **Script Contracts**: Define container paths and environment variables for script execution
3. **Processing Scripts**: Implement business logic using unified main function interface for testability
4. **Step Builders**: Connect specifications and contracts via SageMaker with UnifiedRegistryManager integration
5. **Configuration Classes**: Manage step parameters using three-tier field classification (Essential/System/Derived)
6. **Hyperparameters**: Handle ML-specific parameter tuning and optimization

### Key Modern Features
- **UnifiedRegistryManager System**: Single consolidated registry replacing legacy patterns
- **Workspace-Aware Development**: Support for both shared and isolated development approaches
- **Pipeline Catalog Integration**: Zettelkasten-inspired pipeline catalog with connection-based discovery
- **Enhanced Validation Framework**: Workspace-aware validation with isolation capabilities
- **Three-Tier Configuration Design**: Essential/System/Derived field categorization for better maintainability

## User Input Requirements

Please provide the following information:

1. **SageMaker Step Type**: What type of SageMaker step is this?
   - Options: `Processing`, `Training`, `Transform`, `CreateModel`, `RegisterModel`, `Condition`, `Lambda`, etc.

2. **Design Pattern References**: Which design patterns should be followed?
   - Example: `slipbox/1_design/processing_step_builder_patterns.md`
   - Example: `slipbox/1_design/training_step_builder_patterns.md`

3. **Special Implementation Requirements**: Any special patterns needed?
   - Example: Local hyperparameter to S3 saving patterns
   - Example: Model artifact handling patterns
   - Example: Custom resource configuration patterns

## Your Task

Based on the provided implementation plan, create all necessary code files for the new pipeline step. Your implementation should:

1. Follow the validated implementation plan exactly
2. Adhere to our architectural principles and standardization rules
3. Ensure proper alignment between layers (contract, specification, builder, script)
4. Implement robust error handling and validation
5. Create comprehensive unit tests
6. Place all files in their correct locations within the project structure

## Implementation Plan

[INJECT VALIDATED IMPLEMENTATION PLAN HERE]

## Knowledge Base - Implementation References

### Core Implementation Guides
- [Script Development Guide](../../0_developer_guide/script_development_guide.md) - Unified main function interface
- [Script Contract Development](../../0_developer_guide/script_contract.md) - Contract implementation patterns
- [Step Specification Development](../../0_developer_guide/step_specification.md) - Specification patterns
- [Step Builder Implementation](../../0_developer_guide/step_builder.md) - Builder implementation guide
- [Three-Tier Config Design](../../0_developer_guide/three_tier_config_design.md) - Configuration patterns
- [Hyperparameter Class](../../0_developer_guide/hyperparameter_class.md) - Hyperparameter implementation

### Registry and Validation
- [Step Builder Registry Guide](../../0_developer_guide/step_builder_registry_guide.md) - UnifiedRegistryManager system
- [Step Builder Registry Usage](../../0_developer_guide/step_builder_registry_usage.md) - Practical registry examples
- [Validation Framework Guide](../../0_developer_guide/validation_framework_guide.md) - Workspace-aware validation

### Design Pattern References
- [Processing Step Builder Patterns](../../1_design/processing_step_builder_patterns.md) - Processing step patterns
- [Training Step Builder Patterns](../../1_design/training_step_builder_patterns.md) - Training step patterns
- [CreateModel Step Builder Patterns](../../1_design/createmodel_step_builder_patterns.md) - Model creation patterns
- [Transform Step Builder Patterns](../../1_design/transform_step_builder_patterns.md) - Transform step patterns

### Code Implementation Examples
- Builder Implementations: `src/cursus/steps/builders/` - Complete builder examples by step type
- Configuration Classes: `src/cursus/steps/configs/` - Three-tier configuration examples
- Step Specifications: `src/cursus/steps/specs/` - Specification implementation examples
- Script Contracts: `src/cursus/steps/contracts/` - Contract implementation examples
- Processing Scripts: `src/cursus/steps/scripts/` - Script implementation examples with unified main function
- Registry Integration: `src/cursus/registry/step_names_original.py` - Step registration examples

## IMPORTANT: Domain Knowledge Acquisition

**Before beginning any implementation work, you MUST:**

1. **Read Relevant Documentation**: Thoroughly review the documentation references provided above to understand:
   - Current architectural patterns and design principles
   - Workspace-aware development approaches
   - Modern registry and validation systems
   - Component alignment requirements and best practices

2. **Study Code Examples**: Examine the referenced code examples to understand:
   - Implementation patterns and coding standards
   - Registry integration approaches
   - Configuration and validation patterns
   - Script development and testing approaches

3. **Understand Design Context**: Gain deep understanding of:
   - Why the 6-layer architecture was adopted
   - How workspace-aware development supports multi-developer collaboration
   - The relationship between specifications, contracts, builders, and scripts
   - Modern validation and testing approaches

**Your implementation quality depends on your understanding of these domain-specific patterns and principles. Take time to read and understand the referenced documentation and code examples before generating any implementation.**

## Workspace-Aware Implementation Patterns

### Script Development Integration
**Source**: `slipbox/0_developer_guide/script_development_guide.md`
- Unified main function interface for enhanced testability
- SageMaker compatibility patterns and requirements
- Contract-based path access patterns and implementations
- Error handling and validation approaches for scripts
- Integration with testing frameworks and validation tools

### Enhanced Script Pattern

**Enhanced Script Pattern**:
```python
#!/usr/bin/env python
"""
[StepName] processing script with unified main function interface.
"""

import logging
from pathlib import Path
from typing import Optional

# Workspace-aware imports
if __name__ == "__main__":
    # Runtime imports for script execution
    from cursus.core.contract_enforcer import ContractEnforcer
    from cursus.steps.contracts.[name]_contract import [NAME]_CONTRACT

logger = logging.getLogger(__name__)

def main() -> int:
    """
    Main entry point with unified interface and workspace awareness.
    
    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    try:
        # Contract validation and enforcement
        contract = [NAME]_CONTRACT
        with ContractEnforcer(contract) as enforcer:
            # Get paths from contract (workspace-aware)
            input_path = enforcer.get_input_path("data")
            output_path = enforcer.get_output_path("output")
            
            # Execute main processing logic
            result = process_data(input_path, output_path)
            logger.info(f"Processing completed successfully: {result}")
            return 0
            
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return 2
    except Exception as e:
        logger.error(f"Error in processing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 3

def process_data(input_path: Path, output_path: Path) -> dict:
    """
    Process data with workspace-aware path handling.
    
    Args:
        input_path: Input data path from contract
        output_path: Output data path from contract
        
    Returns:
        dict: Processing results and metadata
    """
    # Implementation logic here
    pass

if __name__ == "__main__":
    exit(main())
```

### Configuration Pattern Examples

**Enhanced Configuration Template with Portable Path Support**:
```python
"""
[StepName] Configuration with Three-Tier Field Categorization and Portable Path Support

This module implements the configuration class for [StepName] steps using the refactored 
design with automatic configuration discovery and portable path support:
1. Essential User Inputs (Tier 1) - Required fields that must be provided by users
2. System Fields (Tier 2) - Fields with reasonable defaults that can be overridden
3. Derived Fields (Tier 3) - Fields calculated from other fields, private with read-only properties
"""

from pydantic import BaseModel, Field, field_validator, model_validator, PrivateAttr
from typing import Dict, Optional, Any, TYPE_CHECKING
from pathlib import Path
import logging

# Current implementation imports (refactored system)
if TYPE_CHECKING:
    from cursus.core.base.contract_base import ScriptContract

# Import appropriate base config class (current implementation)
from cursus.steps.configs.config_processing_step_base import ProcessingStepConfigBase

# Import contract (current implementation)
from cursus.steps.contracts.[name]_contract import [NAME]_CONTRACT

logger = logging.getLogger(__name__)

class [StepName]Config(ProcessingStepConfigBase):
    """
    Configuration for the [StepName] step with three-tier field categorization and portable path support.
    
    Features:
    - Automatic discovery through step catalog integration (no manual registration required)
    - Portable path support for universal deployment compatibility
    - Three-tier field categorization for better maintainability
    
    Fields are categorized into:
    - Tier 1: Essential User Inputs - Required from users
    - Tier 2: System Fields - Default values that can be overridden
    - Tier 3: Derived Fields - Private with read-only property access
    """

    # ===== Essential User Inputs (Tier 1) =====
    step_specific_param: str = Field(
        description="Step-specific required parameter that users must provide."
    )
    
    # ===== System Fields with Defaults (Tier 2) =====
    processing_entry_point: str = Field(
        default="[name].py",
        description="Relative path (within processing_source_dir) to the [name] script."
    )
    
    # ===== Derived Fields (Tier 3) =====
    _derived_value: Optional[str] = PrivateAttr(default=None)
    
    @property
    def derived_value(self) -> str:
        """Get derived value calculated from step-specific parameters."""
        if self._derived_value is None:
            self._derived_value = f"{self.step_specific_param}_processed"
        return self._derived_value
    
    # ===== Portable Path Support (Inherited from ProcessingStepConfigBase) =====
    # The following properties are automatically available:
    # - portable_source_dir: Portable version of source_dir
    # - portable_processing_source_dir: Portable version of processing_source_dir
    # - portable_effective_source_dir: Portable version of effective_source_dir
    # - get_portable_script_path(): Get portable script path
    
    def get_script_contract(self) -> 'ScriptContract':
        """Get script contract for this configuration."""
        return [NAME]_CONTRACT
    
    # ===== Enhanced model_dump with Portable Paths =====
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include portable paths for universal deployment."""
        data = super().model_dump(**kwargs)
        
        # Add derived properties
        data["derived_value"] = self.derived_value
        
        # Portable paths are automatically included by ProcessingStepConfigBase
        # This ensures configuration files work across all deployment environments
        
        return data
```

## Knowledge Base - Design Pattern References

### Processing Step Implementation Patterns
**Source**: `slipbox/1_design/processing_step_builder_patterns.md`
- Processing step implementation patterns and requirements
- Input/output handling for processing steps
- Resource configuration patterns for processing workloads
- Error handling specific to processing operations
- Integration patterns with upstream and downstream components

### Training Step Implementation Patterns
**Source**: `slipbox/1_design/training_step_builder_patterns.md`
- Training step implementation patterns and requirements
- Model training specific input/output handling
- Resource configuration for training workloads
- Hyperparameter management patterns
- Model artifact handling and validation

### Model Creation Implementation Patterns
**Source**: `slipbox/1_design/createmodel_step_builder_patterns.md`
- Model creation implementation patterns and requirements
- Model packaging and deployment preparation
- Model metadata and versioning patterns
- Integration with model registry systems
- Model validation and testing patterns

### Transform Step Implementation Patterns
**Source**: `slipbox/1_design/transform_step_builder_patterns.md`
- Transform step implementation patterns and requirements
- Data transformation input/output handling
- Batch processing and streaming patterns
- Data quality validation requirements
- Performance optimization for transform operations

### Specification-Driven Implementation
**Source**: `slipbox/1_design/specification_driven_design.md`
- Specification-driven implementation architecture
- Component integration through specifications
- Dependency resolution integration patterns
- Cross-component consistency requirements

### Step Builder Registry Integration
**Source**: `slipbox/1_design/step_builder_registry_design.md`
- Registry integration patterns and requirements
- Step registration and discovery patterns
- Naming consistency across registry components
- Registry-based validation approaches

## Knowledge Base - Implementation Reference Documents

### Configuration Field Patterns
**Source**: `slipbox/1_design/config_field_categorization_three_tier.md`
- Configuration field implementation patterns
- Three-tier configuration field categorization
- Field validation and type checking patterns
- Configuration inheritance and composition

### Environment Variable Patterns
**Source**: `slipbox/1_design/environment_variable_contract_enforcement.md`
- Environment variable implementation patterns
- Contract-based environment variable enforcement
- Variable validation and error handling
- Integration with container environments

### Step Naming Patterns
**Source**: `slipbox/1_design/registry_based_step_name_generation.md`
- Step naming implementation patterns
- Registry-based name generation approaches
- Naming consistency across components
- Name validation and conflict resolution

## Knowledge Base - Code Implementation Examples

### Builder Implementation Examples
**Source**: `src/cursus/steps/builders/`
- Complete builder implementations by step type
- Proven implementation patterns and approaches
- Integration examples with SageMaker components
- Error handling and validation implementations
- Input/output handling patterns
- Resource configuration examples

### Configuration Class Examples
**Source**: `src/cursus/steps/configs/`
- Configuration class implementations
- Three-tier configuration pattern implementations
- Parameter validation and type checking examples
- Configuration inheritance and composition patterns
- Integration with builder classes

### Step Specification Examples
**Source**: `src/cursus/steps/specs/`
- Step specification implementations
- Input/output specification patterns
- Dependency specification implementations
- Compatible sources specification examples
- Integration with dependency resolution

### Script Contract Examples
**Source**: `src/cursus/steps/contracts/`
- Script contract implementations
- Path specification and environment variable patterns
- Container integration patterns
- Contract-specification alignment examples
- Environment variable declaration patterns

### Processing Script Examples
**Source**: `src/cursus/steps/scripts/`
- Processing script implementations
- Contract-based path access patterns
- Error handling and validation approaches
- Logging and monitoring integration
- Business logic implementation patterns

### Hyperparameter Class Examples
**Source**: `src/cursus/steps/hyperparams/`
- Hyperparameter class implementations
- Parameter validation and serialization
- Integration with configuration classes
- Type checking and validation patterns

### Registry Integration Examples
**Source**: `src/cursus/steps/registry/`
- Registry integration examples
- Step registration patterns and requirements
- Naming consistency implementation approaches
- Registry-based validation implementations
- Step discovery and instantiation patterns

## Critical Implementation Patterns

### Builder Implementation Patterns

#### 1. **Class Naming and Registration Patterns**
```python
# Step builder class names follow the pattern: [StepName]StepBuilder
@register_builder()
class TabularPreprocessingStepBuilder(StepBuilderBase):
    """Builder for a Tabular Preprocessing ProcessingStep."""

# For training steps:
@register_builder()
class XGBoostTrainingStepBuilder(StepBuilderBase):
    """Builder for an XGBoost Training Step."""

# For model steps:
@register_builder()
class XGBoostModelStepBuilder(StepBuilderBase):
    """Builder for an XGBoost Model Step."""
```

#### 2. **Job Type Handling Patterns**
```python
def __init__(self, config, sagemaker_session=None, role=None, 
             registry_manager=None, dependency_resolver=None):
    """Initialize with specification based on job type."""
    
    # For steps with job type variants (e.g., preprocessing)
    if not hasattr(config, 'job_type'):
        raise ValueError("config.job_type must be specified")
        
    job_type = config.job_type.lower()
    
    # Get specification based on job type
    spec = None
    if job_type == "training" and PREPROCESSING_TRAINING_SPEC is not None:
        spec = PREPROCESSING_TRAINING_SPEC
    elif job_type == "calibration" and PREPROCESSING_CALIBRATION_SPEC is not None:
        spec = PREPROCESSING_CALIBRATION_SPEC
    # ... other job types
    
    if not spec:
        raise ValueError(f"No specification found for job type: {job_type}")
        
    super().__init__(config=config, spec=spec, sagemaker_session=sagemaker_session,
                     role=role,
                     registry_manager=registry_manager,
                     dependency_resolver=dependency_resolver)
```

#### 3. **Specification and Contract Validation**
```python
def __init__(self, config, ...):
    # For single specification steps (e.g., training, model)
    if not SPEC_AVAILABLE or XGBOOST_TRAINING_SPEC is None:
        raise ValueError("XGBoost training specification not available")
        
    super().__init__(config=config, spec=XGBOOST_TRAINING_SPEC, ...)

def validate_configuration(self) -> None:
    """Validate required configuration."""
    self.log_info("Validating [StepName]Config...")
    
    # Validate required attributes specific to step type
    required_attrs = [
        'processing_instance_count',  # For processing steps
        'training_instance_type',     # For training steps
        'instance_type',              # For model steps
        # ... other step-specific attributes
    ]
    
    for attr in required_attrs:
        if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
            raise ValueError(f"Config missing required attribute: {attr}")
```

#### 4. **SageMaker Step Type Patterns**

**Processing Steps:**
```python
def _create_processor(self) -> SKLearnProcessor:
    """Create the SKLearn processor for the processing job."""
    instance_type = (self.config.processing_instance_type_large 
                    if self.config.use_large_processing_instance 
                    else self.config.processing_instance_type_small)
    
    return SKLearnProcessor(
        framework_version=self.config.processing_framework_version,
        role=self.role,
        instance_type=instance_type,
        instance_count=self.config.processing_instance_count,
        volume_size_in_gb=self.config.processing_volume_size,
        base_job_name=self._generate_job_name(),
        sagemaker_session=self.session,
        env=self._get_environment_variables(),
    )

def create_step(self, **kwargs) -> ProcessingStep:
    """Create the ProcessingStep."""
    processor = self._create_processor()
    proc_inputs = self._get_inputs(inputs)
    proc_outputs = self._get_outputs(outputs)
    job_args = self._get_job_arguments()
    
    return ProcessingStep(
        name=self._get_step_name(),
        processor=processor,
        inputs=proc_inputs,
        outputs=proc_outputs,
        code=script_path,
        job_arguments=job_args,
        depends_on=dependencies,
        cache_config=self._get_cache_config(enable_caching)
    )
```

**Training Steps:**
```python
def _create_estimator(self, output_path=None) -> XGBoost:
    """Create and configure the XGBoost estimator."""
    return XGBoost(
        entry_point=self.config.training_entry_point,
        source_dir=self.config.source_dir,
        framework_version=self.config.framework_version,
        py_version=self.config.py_version,
        role=self.role,
        instance_type=self.config.training_instance_type,
        instance_count=self.config.training_instance_count,
        volume_size=self.config.training_volume_size,
        base_job_name=self._generate_job_name(),
        sagemaker_session=self.session,
        output_path=output_path,
        environment=self._get_environment_variables(),
    )

def create_step(self, **kwargs) -> TrainingStep:
    """Create the TrainingStep."""
    training_inputs = self._get_inputs(inputs)
    output_path = self._get_outputs({})
    estimator = self._create_estimator(output_path)
    
    return TrainingStep(
        name=self._get_step_name(),
        estimator=estimator,
        inputs=training_inputs,
        depends_on=dependencies,
        cache_config=self._get_cache_config(enable_caching)
    )
```

**CreateModel Steps:**
```python
def _create_model(self, model_data: str) -> XGBoostModel:
    """Create and configure the XGBoostModel."""
    image_uri = self._get_image_uri()
        
    return XGBoostModel(
        model_data=model_data,
        role=self.role,
        entry_point=self.config.entry_point,
        source_dir=self.config.source_dir,
        framework_version=self.config.framework_version,
        py_version=self.config.py_version,
        image_uri=image_uri,
        sagemaker_session=self.session,
        env=self._get_environment_variables(),
    )

def create_step(self, **kwargs) -> CreateModelStep:
    """Create the CreateModelStep."""
    model_inputs = self._get_inputs(extracted_inputs)
    model = self._create_model(model_inputs['model_data'])
    
    return CreateModelStep(
        name=self._get_step_name(),
        step_args=model.create(
            instance_type=self.config.instance_type,
            accelerator_type=getattr(self.config, 'accelerator_type', None),
            tags=getattr(self.config, 'tags', None),
            model_name=self.config.get_model_name() if hasattr(self.config, 'get_model_name') else None
        ),
        depends_on=dependencies or []
    )
```

#### 5. **Input/Output Handling with Dependency Resolver**

**Processing Steps - _get_inputs:**
```python
def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
    """Get inputs using specification and contract."""
    if not self.spec or not self.contract:
        raise ValueError("Step specification and contract are required")
        
    processing_inputs = []
    
    # Process each dependency in the specification
    for _, dependency_spec in self.spec.dependencies.items():
        logical_name = dependency_spec.logical_name
        
        # Skip if optional and not provided
        if not dependency_spec.required and logical_name not in inputs:
            continue
            
        # Make sure required inputs are present
        if dependency_spec.required and logical_name not in inputs:
            raise ValueError(f"Required input '{logical_name}' not provided")
        
        # Get container path from contract
        if logical_name in self.contract.expected_input_paths:
            container_path = self.contract.expected_input_paths[logical_name]
            
            # Use the input value directly - property references are handled by PipelineAssembler
            processing_inputs.append(
                ProcessingInput(
                    input_name=logical_name,
                    source=inputs[logical_name],
                    destination=container_path
                )
            )
        else:
            raise ValueError(f"No container path found for input: {logical_name}")
            
    return processing_inputs
```

**Training Steps - _get_inputs:**
```python
def _get_inputs(self, inputs: Dict[str, Any]) -> Dict[str, TrainingInput]:
    """Get inputs using specification and contract."""
    if not self.spec or not self.contract:
        raise ValueError("Step specification and contract are required")
        
    training_inputs = {}
    
    # SPECIAL CASE: Always generate hyperparameters internally first
    hyperparameters_key = "hyperparameters_s3_uri"
    internal_hyperparameters_s3_uri = self._prepare_hyperparameters_file()
    
    # Get container path and create channel
    if hyperparameters_key in self.contract.expected_input_paths:
        container_path = self.contract.expected_input_paths[hyperparameters_key]
        # Extract channel name from container path
        parts = container_path.split('/')
        if len(parts) > 5 and parts[5]:
            channel_name = parts[5]  # e.g., 'config' from '/opt/ml/input/data/config/hyperparameters.json'
            training_inputs[channel_name] = TrainingInput(s3_data=internal_hyperparameters_s3_uri)
    
    # Process other dependencies
    for _, dependency_spec in self.spec.dependencies.items():
        logical_name = dependency_spec.logical_name
        
        if logical_name == hyperparameters_key:
            continue  # Already handled
            
        if logical_name in inputs:
            container_path = self.contract.expected_input_paths[logical_name]
            
            # SPECIAL HANDLING FOR input_path - create train/val/test channels
            if logical_name == "input_path":
                base_path = inputs[logical_name]
                data_channels = self._create_data_channels_from_source(base_path)
                training_inputs.update(data_channels)
            else:
                # Extract channel name from container path
                parts = container_path.split('/')
                if len(parts) > 5:
                    channel_name = parts[5]
                    training_inputs[channel_name] = TrainingInput(s3_data=inputs[logical_name])
                    
    return training_inputs
```

**Model Steps - _get_inputs:**
```python
def _get_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Use specification dependencies to get model_data."""
    model_data_key = "model_data"  # From spec.dependencies
    
    if model_data_key not in inputs:
        raise ValueError(f"Required input '{model_data_key}' not found")
        
    return {model_data_key: inputs[model_data_key]}
```

#### 6. **Output Handling Patterns**

**Processing Steps - _get_outputs:**
```python
def _get_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
    """Get outputs using specification and contract."""
    if not self.spec or not self.contract:
        raise ValueError("Step specification and contract are required")
        
    processing_outputs = []
    
    # Process each output in the specification
    for _, output_spec in self.spec.outputs.items():
        logical_name = output_spec.logical_name
        
        # Get container path from contract
        if logical_name in self.contract.expected_output_paths:
            container_path = self.contract.expected_output_paths[logical_name]
            
            # Try to find destination in outputs
            if logical_name in outputs:
                destination = outputs[logical_name]
            else:
                # Generate destination from config
                destination = f"{self.config.pipeline_s3_loc}/{step_name}/{self.config.job_type}/{logical_name}"
                self.log_info("Using generated destination for '%s': %s", logical_name, destination)
            
            processing_outputs.append(
                ProcessingOutput(
                    output_name=logical_name,
                    source=container_path,
                    destination=destination
                )
            )
        else:
            raise ValueError(f"No container path found for output: {logical_name}")
            
    return processing_outputs
```

**Training Steps - _get_outputs:**
```python
def _get_outputs(self, outputs: Dict[str, Any]) -> str:
    """Get output path for model artifacts and evaluation results."""
    if not self.spec or not self.contract:
        raise ValueError("Step specification and contract are required")
        
    # Check if any output path is explicitly provided
    primary_output_path = None
    output_logical_names = [spec.logical_name for _, spec in self.spec.outputs.items()]
    
    for logical_name in output_logical_names:
        if logical_name in outputs:
            primary_output_path = outputs[logical_name]
            break
            
    # If no output path was provided, generate a default one
    if primary_output_path is None:
        primary_output_path = f"{self.config.pipeline_s3_loc}/xgboost_training/"
        
    return primary_output_path.rstrip('/')
```

#### 7. **Dependency Extraction Patterns**
```python
def create_step(self, **kwargs) -> Step:
    """Create the step with dependency extraction."""
    inputs_raw = kwargs.get('inputs', {})
    dependencies = kwargs.get('dependencies', [])
    
    # Handle inputs
    inputs = {}
    
    # If dependencies are provided, extract inputs from them
    if dependencies:
        try:
            extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
            inputs.update(extracted_inputs)
        except Exception as e:
            self.log_warning("Failed to extract inputs from dependencies: %s", e)
            
    # Add explicitly provided inputs (overriding any extracted ones)
    inputs.update(inputs_raw)
    
    # Add direct keyword arguments (e.g., DATA, METADATA from template)
    for key in ["DATA", "METADATA", "SIGNATURE"]:
        if key in kwargs and key not in inputs:
            inputs[key] = kwargs[key]
    
    # Continue with step creation...
```

#### 8. **S3 Path Handling and Property References**
```python
def _normalize_s3_uri(self, uri: str, description: str = "S3 URI") -> str:
    """Normalize S3 URI handling PipelineVariable objects."""
    # Handle PipelineVariable objects
    if hasattr(uri, 'expr'):
        uri = str(uri.expr)
    
    # Handle Pipeline step references with Get key - return as is
    if isinstance(uri, dict) and 'Get' in uri:
        self.log_info("Found Pipeline step reference: %s", uri)
        return uri
    
    return S3PathHandler.normalize(uri, description)

def _create_data_channels_from_source(self, base_path):
    """Create train, validation, and test channel inputs from a base path."""
    from sagemaker.workflow.functions import Join
    
    # Base path is used directly - property references are handled by PipelineAssembler
    channels = {
        "train": TrainingInput(s3_data=Join(on='/', values=[base_path, "train/"])),
        "val": TrainingInput(s3_data=Join(on='/', values=[base_path, "val/"])),
        "test": TrainingInput(s3_data=Join(on='/', values=[base_path, "test/"]))
    }
    
    return channels
```

#### 9. **Environment Variables and Job Arguments**
```python
def _get_environment_variables(self) -> Dict[str, str]:
    """Create environment variables from contract and config."""
    # Get base environment variables from contract
    env_vars = super()._get_environment_variables()
    
    # Add step-specific environment variables
    if hasattr(self.config, 'label_name'):
        env_vars["LABEL_FIELD"] = self.config.label_name
    
    if hasattr(self.config, 'train_ratio'):
        env_vars["TRAIN_RATIO"] = str(self.config.train_ratio)
        
    return env_vars

def _get_job_arguments(self) -> List[str]:
    """Construct command-line arguments from config."""
    # Get job_type from configuration (takes precedence over contract)
    job_type = self.config.job_type
    self.log_info("Setting job_type argument to: %s", job_type)
    
    return ["--job_type", job_type]
```

#### 10. **Error Handling and Logging Patterns**
```python
def create_step(self, **kwargs) -> Step:
    """Create step with comprehensive error handling."""
    try:
        # Step creation logic
        step = self._create_step_implementation(**kwargs)
        
        # Attach specification to the step for future reference
        setattr(step, '_spec', self.spec)
        
        self.log_info("Created %s with name: %s", step.__class__.__name__, step.name)
        return step
        
    except Exception as e:
        self.log_error("Error creating %s: %s", self.__class__.__name__, str(e))
        import traceback
        self.log_error(traceback.format_exc())
        raise ValueError(f"Failed to create {self.__class__.__name__}: {str(e)}") from e

# Safe logging for Pipeline variables
def log_info(self, message, *args, **kwargs):
    """Safely log info messages, handling Pipeline variables."""
    try:
        # Use safe_value_for_logging from base class
        safe_args = [safe_value_for_logging(arg) for arg in args]
        logger.info(message, *safe_args, **kwargs)
    except Exception as e:
        logger.info(f"Original logging failed ({e}), logging raw message: {message}")
```

## Implementation Details

### 1. Step Registry Addition

The registry system uses a centralized approach with `src/cursus/steps/registry/step_names.py` as the single source of truth. Here's how to add a new step:

#### A. Add to Central Step Registry

```python
# In src/cursus/steps/registry/step_names.py
STEP_NAMES = {
    # ... existing steps ...
    
    "[StepName]": {
        "config_class": "[StepName]Config",
        "builder_step_name": "[StepName]StepBuilder", 
        "spec_type": "[StepName]",
        "sagemaker_step_type": "Processing",  # or "Training", "CreateModel", "Transform"
        "description": "[Brief description of what this step does]"
    },
}
```

**SageMaker Step Type Options:**
- `"Processing"` - For data processing, preprocessing, evaluation steps
- `"Training"` - For model training steps
- `"CreateModel"` - For model creation/registration steps
- `"Transform"` - For batch transform steps
- `"Lambda"` - For utility steps (like hyperparameter preparation)
- Custom types for special cases

#### B. Builder Registration with Decorator (Current Implementation)

The step builder will be automatically registered using the `@register_builder()` decorator:

```python
# In src/cursus/steps/builders/builder_[name]_step.py
from cursus.steps.builders.base import StepBuilderBase
from cursus.registry.builder_registry import register_builder

@register_builder()  # Auto-detects step type from STEP_NAMES registry
class [StepName]StepBuilder(StepBuilderBase):
    """Builder for [StepName] step with portable path support."""
    
    def __init__(self, config, sagemaker_session=None, role=None, 
                 registry_manager=None, dependency_resolver=None):
        """Initialize with portable path support."""
        super().__init__(
            config=config,
            spec=STEP_SPEC,  # Always pass the specification
            sagemaker_session=sagemaker_session,
            role=role,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver
        )
    
    def create_step(self, **kwargs):
        """Create step with portable path support."""
        # Use portable paths from config
        script_path = self.config.get_portable_script_path() or self.config.get_script_path()
        source_dir = self.config.portable_source_dir or self.config.source_dir
        
        # Continue with step creation using portable paths...
        pass
```

### 2. Script Contract Implementation

```python
# src/cursus/steps/contracts/[name]_contract.py
from ...core.base.contract_base import ScriptContract

[NAME]_CONTRACT = ScriptContract(
    entry_point="[name].py",
    expected_input_paths={
        # Input paths with SageMaker container locations
        "logical_input_name": "/opt/ml/processing/input/data",
    },
    expected_output_paths={
        # Output paths with SageMaker container locations
        "logical_output_name": "/opt/ml/processing/output",
    },
    expected_arguments={
        # Expected command-line arguments (if any)
    },
    required_env_vars=[
        "REQUIRED_ENV_VAR_1",
        "REQUIRED_ENV_VAR_2"
    ],
    optional_env_vars={
        "OPTIONAL_ENV_VAR_1": "default_value",
        "OPTIONAL_ENV_VAR_2": ""
    },
    framework_requirements={
        "pandas": ">=1.3.0",
        "numpy": ">=1.21.0",
        "scikit-learn": ">=1.0.0"
    },
    description="""
    Contract for [step description].
    
    Detailed description of what the script does:
    1. Input processing details
    2. Output generation details
    3. Environment variable usage
    """
)
```

### 3. Step Specification Implementation

```python
# src/cursus/steps/specs/[name]_spec.py
from ...core.base.specification_base import StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType
from ..registry.step_names import get_spec_step_type_with_job_type

def _get_[name]_contract():
    from ..contracts.[name]_contract import [NAME]_CONTRACT
    return [NAME]_CONTRACT

[NAME]_SPEC = StepSpecification(
    step_type=get_spec_step_type_with_job_type("[StepName]", "Training"),  # Use job type function
    node_type=NodeType.INTERNAL,
    script_contract=_get_[name]_contract(),
    dependencies=[
        DependencySpec(
            logical_name="logical_input_name",
            dependency_type=DependencyType.PROCESSING_OUTPUT,  # Use appropriate dependency type
            required=True,
            compatible_sources=["UpstreamStepName", "AnotherCompatibleStep"],
            semantic_keywords=["data", "input", "raw", "dataset"],
            data_type="S3Uri",
            description="Description of input dependency"
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="logical_output_name",
            output_type=DependencyType.PROCESSING_OUTPUT,  # Use appropriate output type
            property_path="properties.ProcessingOutputConfig.Outputs['logical_output_name'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Description of output"
        )
    ]
)
```

### 4. Configuration Class Implementation

```python
# src/cursus/steps/configs/config_[name]_step.py
"""
[StepName] Configuration with Self-Contained Derivation Logic

This module implements the configuration class for [StepName] steps
using a self-contained design where each field is properly categorized 
according to the three-tier design:
1. Essential User Inputs (Tier 1) - Required fields that must be provided by users
2. System Fields (Tier 2) - Fields with reasonable defaults that can be overridden
3. Derived Fields (Tier 3) - Fields calculated from other fields, private with read-only properties
"""

from pydantic import BaseModel, Field, field_validator, model_validator, PrivateAttr
from typing import Dict, Optional, Any, TYPE_CHECKING
from pathlib import Path
import logging

# Import appropriate base config class based on step type
from .config_processing_step_base import ProcessingStepConfigBase  # For Processing steps
# OR from ...core.base.config_base import BasePipelineConfig  # For other step types

# Import contract
from ..contracts.[name]_contract import [NAME]_CONTRACT

# Import for type hints only
if TYPE_CHECKING:
    from ...core.base.contract_base import ScriptContract

logger = logging.getLogger(__name__)


class [StepName]Config(ProcessingStepConfigBase):  # Or BasePipelineConfig for non-processing steps
    """
    Configuration for the [StepName] step with three-tier field categorization.
    Inherits from ProcessingStepConfigBase (or BasePipelineConfig for non-processing steps).
    
    Fields are categorized into:
    - Tier 1: Essential User Inputs - Required from users
    - Tier 2: System Fields - Default values that can be overridden
    - Tier 3: Derived Fields - Private with read-only property access
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide
    
    step_specific_param: str = Field(
        description="Step-specific required parameter that users must provide."
    )
    
    another_required_param: int = Field(
        description="Another required parameter for step configuration."
    )
    
    # ===== System Fields with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override
    
    processing_entry_point: str = Field(
        default="[name].py",
        description="Relative path (within processing_source_dir) to the [name] script."
    )
    
    optional_param: str = Field(
        default="default_value",
        description="Optional parameter with a sensible default value."
    )
    
    custom_instance_count: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Custom instance count for this specific step (overrides base processing_instance_count if needed)."
    )
    
    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields
    # They are private with public read-only property access
    
    _derived_value: Optional[str] = PrivateAttr(default=None)
    _computed_s3_path: Optional[str] = PrivateAttr(default=None)
    _full_script_path: Optional[str] = PrivateAttr(default=None)

    # Update to Pydantic V2 style model_config (based on real patterns from codebase)
    model_config = ProcessingStepConfigBase.model_config.copy()
    model_config.update({
        'arbitrary_types_allowed': True,
        'validate_assignment': True
    })
    
    # ===== Properties for Derived Fields =====
    
    @property
    def derived_value(self) -> str:
        """
        Get derived value calculated from step-specific parameters.
        
        Returns:
            Derived value combining step parameters
        """
        if self._derived_value is None:
            self._derived_value = f"{self.step_specific_param}_{self.another_required_param}"
        return self._derived_value
    
    @property
    def computed_s3_path(self) -> str:
        """
        Get computed S3 path for step outputs.
        
        Returns:
            S3 path for step outputs
        """
        if self._computed_s3_path is None:
            self._computed_s3_path = f"{self.pipeline_s3_loc}/[step_name]/{self.derived_value}"
        return self._computed_s3_path
    
    @property
    def full_script_path(self) -> Optional[str]:
        """
        Get full path to the [name] script.
        
        Returns:
            Full path to the script
        """
        if self._full_script_path is None:
            # Get effective source directory
            source_dir = self.effective_source_dir
            if source_dir is None:
                return None
                
            # Combine with entry point
            if source_dir.startswith('s3://'):
                self._full_script_path = f"{source_dir.rstrip('/')}/{self.processing_entry_point}"
            else:
                self._full_script_path = str(Path(source_dir) / self.processing_entry_point)
                
        return self._full_script_path

    # ===== Validators =====
    
    @field_validator("step_specific_param")
    @classmethod
    def validate_step_specific_param(cls, v: str) -> str:
        """
        Ensure step_specific_param meets requirements.
        """
        if not v or not v.strip():
            raise ValueError("step_specific_param must be a non-empty string")
        return v
    
    @field_validator("processing_entry_point")
    @classmethod
    def validate_entry_point_relative(cls, v: Optional[str]) -> Optional[str]:
        """
        Ensure processing_entry_point is a non‐empty relative path.
        """
        if v is None or not v.strip():
            raise ValueError("processing_entry_point must be a non‐empty relative path")
        if Path(v).is_absolute() or v.startswith("/") or v.startswith("s3://"):
            raise ValueError("processing_entry_point must be a relative path within source directory")
        return v

    @field_validator("another_required_param")
    @classmethod
    def validate_another_required_param(cls, v: int) -> int:
        """
        Ensure another_required_param is within valid range.
        """
        if v <= 0:
            raise ValueError("another_required_param must be positive")
        return v
        
    # Initialize derived fields at creation time
    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "[StepName]Config":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()
        
        # Initialize step-specific derived fields
        self._derived_value = f"{self.step_specific_param}_{self.another_required_param}"
        self._computed_s3_path = f"{self.pipeline_s3_loc}/[step_name]/{self._derived_value}"
        
        # Initialize full script path if possible
        source_dir = self.effective_source_dir
        if source_dir is not None:
            if source_dir.startswith('s3://'):
                self._full_script_path = f"{source_dir.rstrip('/')}/{self.processing_entry_point}"
            else:
                self._full_script_path = str(Path(source_dir) / self.processing_entry_point)
            
        return self

    # ===== Script Contract =====
        
    def get_script_contract(self) -> 'ScriptContract':
        """
        Get script contract for this configuration.
        
        Returns:
            The [name] script contract
        """
        return [NAME]_CONTRACT
        
    def get_script_path(self, default_path: str = None) -> str:
        """
        Get script path with priority order:
        1. Use full_script_path property if available
        2. Use default_path if provided
        
        Returns:
            Script path or default_path if no entry point can be determined
        """
        if self.full_script_path:
            return self.full_script_path
        return default_path
    
    # ===== Overrides for Inheritance =====
    
    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include [name] specific fields.
        
        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class
        base_fields = super().get_public_init_fields()
        
        # Add [name] specific fields
        step_fields = {
            'step_specific_param': self.step_specific_param,
            'another_required_param': self.another_required_param,
            'processing_entry_point': self.processing_entry_point,
            'optional_param': self.optional_param,
            'custom_instance_count': self.custom_instance_count,
        }
        
        # Combine fields (step fields take precedence if overlap)
        init_fields = {**base_fields, **step_fields}
        
        return init_fields
        
    # ===== Serialization =====
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include derived properties."""
        # Get base fields first
        data = super().model_dump(**kwargs)
        
        # Add derived properties
        data["derived_value"] = self.derived_value
        data["computed_s3_path"] = self.computed_s3_path
        if self.full_script_path:
            data["full_script_path"] = self.full_script_path
            
        return data
    
    # ===== Training-Specific Methods (if applicable) =====
    
    def to_hyperparameter_dict(self) -> Dict[str, Any]:
        """Convert configuration to hyperparameter dictionary for training steps."""
        return {
            "step_specific_param": self.step_specific_param,
            "another_required_param": self.another_required_param,
            "derived_value": self.derived_value,
            "optional_param": self.optional_param
        }
```

### 5. Processing Script Implementation (Error Handling)

```python
# src/cursus/steps/scripts/[name].py
#!/usr/bin/env python

def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        # Execute main processing
        result = process_data(args.input_path, args.output_path)
        logger.info(f"Processing completed successfully: {result}")
        return 0
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return 2
    except Exception as e:
        logger.error(f"Error in processing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 3
```

## Instructions

Create implementation files in the following locations, ensuring complete adherence to the implementation plan:

1. **Script Contract**
   - Location: `src/cursus/steps/contracts/[name]_contract.py`
   - Follow the ScriptContract schema with proper input/output paths and environment variables
   - Ensure logical names match what's specified in the implementation plan

2. **Step Specification**
   - Location: `src/cursus/steps/specs/[name]_spec.py`
   - Define dependencies and outputs as specified in the plan
   - Ensure correct dependency types, compatible sources, and semantic keywords
   - Make property paths consistent with SageMaker standards

3. **Configuration**
   - Location: `src/cursus/steps/configs/config_[name]_step.py`
   - Implement the config class with all parameters specified in the plan
   - Inherit from the appropriate base config class
   - Implement required methods like get_script_contract()

4. **Step Builder**
   - Location: `src/cursus/steps/builders/builder_[name]_step.py`
   - Implement the builder class following the StepBuilderBase pattern
   - Create methods for handling inputs, outputs, processor creation, and step creation
   - Ensure proper error handling and validation

5. **Processing Script**
   - Location: `src/cursus/steps/scripts/[name].py`
   - Implement the script following the contract's input/output paths
   - Include robust error handling and validation
   - Add comprehensive logging at appropriate levels

6. **Update Registry Files**
   - Update `src/cursus/steps/registry/step_names.py` to include the new step
   - Update appropriate `__init__.py` files to expose the new components

7. **Unit Tests**
   - Create appropriate test files in `test/` directory following the project's test structure

## Key Implementation Requirements

1. **Strict Path Adherence**: Always use paths from contracts, never hardcode paths
2. **Alignment Consistency**: Ensure logical names are consistent across all components
3. **Dependency Types**: Use correct dependency types to ensure compatibility with upstream/downstream steps
4. **Error Handling**: Implement comprehensive error handling with meaningful messages
5. **Documentation**: Add thorough docstrings to all classes and methods
6. **Type Hints**: Use proper Python type hints for all parameters and return values
7. **Standardization**: Follow naming conventions and interface standards precisely
8. **Validation**: Add validation for all inputs, configuration, and runtime conditions
9. **Spec/Contract Validation**: Always verify spec and contract availability in builder methods:
   ```python
   if not self.spec:
       raise ValueError("Step specification is required")
           
   if not self.contract:
       raise ValueError("Script contract is required for input mapping")
   ```
10. **S3 Path Handling**: Implement helper methods for S3 path handling:
    ```python
    def _normalize_s3_uri(self, uri: str, description: str = "S3 URI") -> str:
        # Handle PipelineVariable objects
        if hasattr(uri, 'expr'):
            uri = str(uri.expr)
        
        # Handle Pipeline step references
        if isinstance(uri, dict) and 'Get' in uri:
            self.log_info("Found Pipeline step reference: %s", uri)
            return uri
        
        return S3PathHandler.normalize(uri, description)
    ```
11. **PipelineVariable Handling**: Always handle PipelineVariable objects in inputs/outputs
12. **Configuration Validation**: Add comprehensive validation in validate_configuration:
    ```python
    required_attrs = ['attribute1', 'attribute2', ...]
    for attr in required_attrs:
        if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
            raise ValueError(f"Config missing required attribute: {attr}")
    ```

## Required Builder Methods

Ensure your step builder implementation includes these essential methods:

### 1. Base Methods

```python
def __init__(self, config: CustomConfig, sagemaker_session=None, role=None, registry_manager=None, dependency_resolver=None):
    """Initialize the step builder with configuration and dependencies."""
    if not isinstance(config, CustomConfig):
        raise ValueError("Builder requires a CustomConfig instance.")
    
    super().__init__(
        config=config,
        spec=CUSTOM_SPEC,  # Always pass the specification
        sagemaker_session=sagemaker_session,
        role=role,
        registry_manager=registry_manager,
        dependency_resolver=dependency_resolver
    )
    self.config: CustomConfig = config  # Type hint for IDE support

def validate_configuration(self) -> None:
    """
    Validate the configuration thoroughly before any step creation.
    This should check all required attributes and validate file paths.
    """
    self.log_info("Validating CustomConfig...")
    
    # Check required attributes
    required_attrs = ['attribute1', 'attribute2', 'attribute3']
    for attr in required_attrs:
        if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
            raise ValueError(f"CustomConfig missing required attribute: {attr}")
    
    # Validate paths if needed
    if not hasattr(self.config.some_path, 'expr'):  # Skip validation for PipelineVariables
        path = Path(self.config.some_path)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
    
    self.log_info("CustomConfig validation succeeded.")

def create_step(self, **kwargs) -> ProcessingStep:
    """
    Create the SageMaker step with full error handling and dependency extraction.
    
    Args:
        **kwargs: Keyword args including dependencies and optional params
    
    Returns:
        ProcessingStep: The configured processing step
        
    Raises:
        ValueError: If inputs cannot be extracted or config is invalid
    """
    try:
        # Extract inputs from dependencies
        dependencies = kwargs.get('dependencies', [])
        inputs = {}
        if dependencies:
            inputs = self.extract_inputs_from_dependencies(dependencies)
        
        # Get processor inputs and outputs
        processing_inputs = self._get_inputs(inputs)
        processing_outputs = self._get_outputs({})
        
        # Create processor
        processor = self._get_processor()
        
        # Get cache configuration
        cache_config = self._get_cache_config(kwargs.get('enable_caching', True))
        
        # Create the step
        step = processor.run(
            code=self.config.get_script_path(),
            inputs=processing_inputs,
            outputs=processing_outputs,
            arguments=self._get_script_arguments(),
            job_name=self._generate_job_name('CustomStep'),
            wait=False,
            cache_config=cache_config
        )
        
        # Store specification in step for future reference
        setattr(step, '_spec', self.spec)
        
        return step
    
    except Exception as e:
        self.log_error(f"Error creating CustomStep: {e}")
        import traceback
        self.log_error(traceback.format_exc())
        raise ValueError(f"Failed to create CustomStep: {str(e)}") from e
```

### 2. Input/Output Methods

```python
def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
    """
    Get inputs for the processor using spec and contract for mapping.
    
    Args:
        inputs: Dictionary of input sources keyed by logical name
        
    Returns:
        List of ProcessingInput objects for the processor
    """
    if not self.spec:
        raise ValueError("Step specification is required")
        
    if not self.contract:
        raise ValueError("Script contract is required for input mapping")
        
    processing_inputs = []
    
    # Process each dependency in the specification
    for logical_name, dependency_spec in self.spec.dependencies.items():
        # Skip if optional and not provided
        if not dependency_spec.required and logical_name not in inputs:
            continue
            
        # Check required inputs
        if dependency_spec.required and logical_name not in inputs:
            raise ValueError(f"Required input '{logical_name}' not provided")
        
        # Get container path from contract
        if logical_name in self.contract.expected_input_paths:
            container_path = self.contract.expected_input_paths[logical_name]
            
            # Add input to processing inputs
            processing_inputs.append(
                ProcessingInput(
                    source=inputs[logical_name],
                    destination=container_path,
                    input_name=logical_name
                )
            )
        else:
            raise ValueError(f"No container path found for input: {logical_name}")
            
    return processing_inputs

def _get_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
    """
    Get outputs for the processor using spec and contract for mapping.
    
    Args:
        outputs: Dictionary of output destinations keyed by logical name
        
    Returns:
        List of ProcessingOutput objects for the processor
    """
    if not self.spec:
        raise ValueError("Step specification is required")
        
    if not self.contract:
        raise ValueError("Script contract is required for output mapping")
        
    processing_outputs = []
    
    # Process each output in the specification
    for logical_name, output_spec in self.spec.outputs.items():
        # Get container path from contract
        if logical_name in self.contract.expected_output_paths:
            container_path = self.contract.expected_output_paths[logical_name]
            
            # Try to find destination in outputs
            destination = None
            
            # Look in outputs by logical name
            if logical_name in outputs:
                destination = outputs[logical_name]
            else:
                # Generate destination from base path using Join instead of f-string
                from sagemaker.workflow.functions import Join
                base_output_path = self._get_base_output_path()
                step_type = self.spec.step_type.lower() if hasattr(self.spec, 'step_type') else 'processing'
                destination = Join(on="/", values=[base_output_path, step_type, logical_name])
                self.log_info(
                    "Using generated destination for '%s': %s",
                    logical_name,
                    destination,
                )
            
            # Add output to processing outputs
            processing_outputs.append(
                ProcessingOutput(
                    source=container_path,
                    destination=destination,
                    output_name=logical_name
                )
            )
        else:
            raise ValueError(f"No container path found for output: {logical_name}")
            
    return processing_outputs
```

### 3. Helper Methods

```python
def _get_processor(self):
    """
    Create and configure the processor for the step.
    
    Returns:
        The configured processor for running the step
    """
    return ScriptProcessor(
        image_uri="137112412989.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
        command=["python3"],
        instance_type=self.config.instance_type,
        instance_count=self.config.instance_count,
        volume_size_in_gb=self.config.volume_size_gb,
        max_runtime_in_seconds=self.config.max_runtime_seconds,
        role=self.role,
        sagemaker_session=self.session,
        base_job_name=self._sanitize_name_for_sagemaker(
            f"{self._get_step_name('CustomStep')}"
        )
    )

def _get_script_arguments(self) -> List[str]:
    """
    Generate script arguments from configuration parameters.
    
    Returns:
        List of arguments to pass to the script
    """
    args = []
    
    # Add arguments from config
    args.extend(["--param1", str(self.config.param1)])
    args.extend(["--param2", str(self.config.param2)])
    
    return args

def _validate_s3_uri(self, uri: str, description: str = "S3 URI") -> bool:
    """
    Validate that a string is a properly formatted S3 URI.
    
    Args:
        uri: The URI to validate
        description: Description for error messages
        
    Returns:
        True if valid, False otherwise
    """
    # Handle PipelineVariable objects
    if hasattr(uri, 'expr'):
        return True
        
    # Handle Pipeline step references with Get key
    if isinstance(uri, dict) and 'Get' in uri:
        return True
    
    if not isinstance(uri, str):
        self.log_warning("Invalid %s URI: type %s", description, type(uri).__name__)
        return False
    
    return S3PathHandler.is_valid(uri)
```

## Expected Output Format

For each file you create, follow this format:

```
# File: [path/to/file.py]
```python
# Full content of the file here, including imports, docstrings, and implementation
```

Ensure each file is complete, properly formatted, and ready to be saved directly to the specified location. Include all necessary imports, docstrings, and implementation details.

Remember that your implementation will be validated against our architectural standards, with special focus on alignment rules adherence and cross-component compatibility.
