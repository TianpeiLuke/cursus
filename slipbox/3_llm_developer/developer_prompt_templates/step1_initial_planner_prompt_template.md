# Initial Pipeline Step Planner Prompt (Enhanced)

## Your Role: Pipeline Step Planner

You are an expert ML Pipeline Architect tasked with planning a new pipeline step for our SageMaker-based ML pipeline system. Your job is to analyze requirements, determine what components need to be created or modified, and create a comprehensive plan for implementing the new step.

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

## Workspace-Aware Development Support

### Developer Workflow Type Detection

Please identify the developer workflow type for this implementation:

1. **Shared Workspace Developer**:
   - Direct access to `src/cursus/steps/` for modification
   - Core maintainer or senior developer role
   - Working on shared/production components
   - Full system access and permissions

2. **Isolated Workspace Developer**:
   - Working in `development/projects/project_xxx/` environment
   - Project team or external contributor
   - Read-only access to shared `src/cursus/steps/` code
   - Uses hybrid registry with project-specific components

**USER INPUT REQUIRED**: Please specify:
- **Developer Type**: Shared Workspace or Isolated Workspace
- **Project ID** (if Isolated Workspace): `project_xxx`
- **Workspace Path** (if Isolated Workspace): `development/projects/project_xxx/`

## Your Task

Based on the provided requirements, create a detailed plan for implementing a new pipeline step. Your plan should include:

1. Analysis of the requirements and their architectural implications
2. List of components to create (script contract, step specification, configuration, step builder, processing script)
3. List of existing files to update (registries, imports, etc.)
4. Dependency analysis (upstream and downstream steps)
5. Job type variants to consider (if any)
6. Edge cases and error handling considerations
7. Alignment strategy between script contract, specification, and builder

## Requirements for the New Step

**USER INPUT REQUIRED**: Please provide the following information:

1. **Step Requirements and Description**: 
   - What does this step need to accomplish?
   - What business logic should it implement?
   - What are the specific functional requirements?

2. **SageMaker Step Type Categorization**: 
   - Processing
   - Training  
   - Transform
   - CreateModel
   - Other (specify)

3. **Plan Documentation Location**: 
   - Where should the implementation plan be documented?
   - Example: `slipbox/2_project_planning/2025-MM-DD_[step_name]_implementation_plan.md`

4. **Relevant Design Patterns**: 
   - Which design patterns should be referenced?
   - Example: `processing_step_builder_patterns.md`, `training_step_builder_patterns.md`

## Knowledge Base - Developer Guide References

### Core Developer Guide
- [Developer Guide README](../../0_developer_guide/README.md) - Updated September 2025 with 6-layer architecture
- [Design Principles](../../0_developer_guide/design_principles.md) - Core architectural principles and patterns
- [Creation Process](../../0_developer_guide/creation_process.md) - Complete 10-step process with consistent numbering
- [Prerequisites](../../0_developer_guide/prerequisites.md) - Updated for modern system requirements
- [Component Guide](../../0_developer_guide/component_guide.md) - 6-layer architecture overview
- [Best Practices](../../0_developer_guide/best_practices.md) - Development best practices
- [Common Pitfalls](../../0_developer_guide/common_pitfalls.md) - Common mistakes to avoid
- [Alignment Rules](../../0_developer_guide/alignment_rules.md) - Component alignment requirements

### Workspace-Aware Developer Guide
- [Workspace-Aware Developer Guide README](../../01_developer_guide_workspace_aware/README.md) - Isolated project development
- [Workspace Setup Guide](../../01_developer_guide_workspace_aware/ws_workspace_setup_guide.md) - Project initialization
- [Adding New Pipeline Step (Workspace-Aware)](../../01_developer_guide_workspace_aware/ws_adding_new_pipeline_step.md) - Workspace-specific development
- [Hybrid Registry Integration](../../01_developer_guide_workspace_aware/ws_hybrid_registry_integration.md) - Registry patterns

### Design Document References
- [Workspace-Aware System Master Design](../../1_design/workspace_aware_system_master_design.md) - Complete system architecture
- [Workspace-Aware Multi-Developer Management Design](../../1_design/workspace_aware_multi_developer_management_design.md) - Multi-developer framework
- [Agentic Workflow Design](../../1_design/agentic_workflow_design.md) - Complete system architecture for agentic workflows

### Code Examples References (Current Implementation)
- Registry Examples: `src/cursus/registry/step_names_original.py` - Actual STEP_NAMES dictionary structure
- Builder Examples: `src/cursus/steps/builders/` - Complete builder implementations by step type with portable path support
- Configuration Examples: `src/cursus/steps/configs/` - Configuration class implementations with three-tier design and portable paths
- Contract Examples: `src/cursus/steps/contracts/` - Script contract implementations
- Specification Examples: `src/cursus/steps/specs/` - Step specification implementations
- Current Implementation: All examples reflect the refactored system with automatic discovery and portable path support

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

## Relevant Documentation

### Creation Process Overview

**Step Creation Workflow and Process** (from `slipbox/0_developer_guide/creation_process.md`):

The step creation process follows a systematic approach:

1. **Requirements Analysis**: Understand the business requirements and technical constraints
2. **Architectural Design**: Design the step components following our 6-layer architecture
3. **Component Implementation**: Implement script contract, step specification, configuration, and step builder
4. **Integration**: Integrate with existing pipeline components and registries
5. **Validation**: Ensure alignment between all components and compliance with standards
6. **Testing**: Comprehensive testing at unit and integration levels

**Key Principles**:
- Specification-driven design: Start with clear input/output specifications
- Contract-first approach: Define script contracts before implementation
- Alignment validation: Ensure consistency between all layers
- Registry integration: Proper registration for step discovery

### Prerequisites

**Prerequisites for Step Development** (from `slipbox/0_developer_guide/prerequisites.md`):

**Technical Prerequisites**:
- Understanding of SageMaker Processing, Training, and Transform steps
- Knowledge of our four-layer architecture (specifications, contracts, builders, scripts)
- Familiarity with dependency resolution patterns
- Understanding of S3 path handling and PipelineVariable usage

**Architectural Prerequisites**:
- Clear understanding of upstream and downstream step dependencies
- Knowledge of existing step patterns and design principles
- Understanding of registry-based step discovery
- Familiarity with configuration management patterns

**Development Prerequisites**:
- Access to existing step implementations for pattern reference
- Understanding of validation and testing requirements
- Knowledge of error handling and logging patterns
- Familiarity with deployment and integration processes

### Alignment Rules

**Critical Alignment Requirements** (from `slipbox/0_developer_guide/alignment_rules.md`):

**Script-Contract Alignment**:
- All paths defined in script contract must be used in processing script
- Environment variables declared in contract must be accessed in script
- Script arguments must match contract specifications exactly

**Contract-Specification Alignment**:
- Logical names in contract input paths must match specification dependency names
- Logical names in contract output paths must match specification output names
- Contract and specification must have consistent data flow definitions

**Specification-Builder Alignment**:
- Builder must use specification dependencies for input mapping
- Builder must use specification outputs for output mapping
- Builder configuration must align with specification requirements

**Cross-Component Consistency**:
- Naming conventions must be consistent across all components
- Data types and formats must be consistent throughout the pipeline
- Error handling patterns must be uniform across components

### Standardization Rules

**Naming Conventions and Interface Standards** (from `slipbox/0_developer_guide/standardization_rules.md`):

**Naming Conventions**:
- Step builders must end with "StepBuilder" (e.g., `TabularPreprocessingStepBuilder`)
- Configuration classes must end with "Config" (e.g., `TabularPreprocessingConfig`)
- Step specifications must end with "Spec" (e.g., `TabularPreprocessingSpec`)
- Script contracts must end with "Contract" (e.g., `TabularPreprocessingContract`)
- File naming follows patterns: `builder_*_step.py`, `config_*_step.py`, `spec_*_step.py`, `contract_*_step.py`

**Interface Standards**:
- Step builders must inherit from `StepBuilderBase`
- Required methods: `validate_configuration`, `_get_inputs`, `_get_outputs`, `create_step`
- Configuration classes must inherit from appropriate base classes
- Method signatures must follow established patterns

**Registry Integration**:
- All steps must be registered in `src/cursus/steps/registry/step_names.py`
- Registry entries must match builder class names
- Auto-discovery patterns must be followed for undecorated builders

**Documentation Standards**:
- All classes must have comprehensive docstrings
- Method documentation must include parameter and return type information
- Examples must be provided for complex functionality

## Design Pattern References

### Processing Step Builder Patterns (from `slipbox/1_design/processing_step_builder_patterns.md`)

**Key Patterns for Processing Steps**:
- Input validation and preprocessing
- S3 path handling with PipelineVariable support
- Container path mapping from contracts
- Error handling and logging patterns
- Output validation and postprocessing

### Training Step Builder Patterns (from `slipbox/1_design/training_step_builder_patterns.md`)

**Key Patterns for Training Steps**:
- Model artifact handling
- Hyperparameter configuration management
- Training job configuration patterns
- Model validation and metrics handling
- Checkpoint and model saving patterns

### CreateModel Step Builder Patterns (from `slipbox/1_design/createmodel_step_builder_patterns.md`)

**Key Patterns for Model Creation Steps**:
- Model artifact registration
- Inference configuration setup
- Model packaging and deployment preparation
- Model metadata management
- Version control and model registry integration

### Transform Step Builder Patterns (from `slipbox/1_design/transform_step_builder_patterns.md`)

**Key Patterns for Transform Steps**:
- Batch transform configuration
- Input/output data format handling
- Transform job parameter management
- Result aggregation and validation
- Performance optimization patterns

### Step Builder Patterns Summary (from `slipbox/1_design/step_builder_patterns_summary.md`)

**Universal Patterns Across All Step Types**:
- Specification and contract validation
- Configuration validation patterns
- Input/output mapping strategies
- Error handling and recovery
- Logging and monitoring integration
- Registry integration patterns

## Implementation Examples

### Existing Step Builder Implementations (from `src/cursus/steps/builders/`)

**Reference Implementations by Step Type**:
- Processing steps: Tabular preprocessing, data validation, feature engineering
- Training steps: XGBoost training, PyTorch training, model evaluation
- Transform steps: Batch inference, data transformation, model scoring
- CreateModel steps: Model registration, endpoint creation, model packaging

### Configuration Class Examples (from `src/cursus/steps/configs/`)

**Configuration Patterns**:
- Three-tier configuration design
- Parameter validation and defaults
- Environment variable integration
- SageMaker-specific parameter handling

### Step Specification Examples (from `src/cursus/steps/specs/`)

**Specification Patterns**:
- Dependency specification with compatible sources
- Output specification with property paths
- Job type variant handling
- External dependency integration

### Script Contract Examples (from `src/cursus/steps/contracts/`)

**Contract Patterns**:
- Input/output path definitions
- Environment variable specifications
- Framework requirement declarations
- Container configuration patterns

### Processing Script Examples (from `src/cursus/steps/scripts/`)

**Script Implementation Patterns**:
- Argument parsing and validation
- File I/O and S3 integration
- Error handling and logging
- Business logic implementation

### Registry Integration Examples (from `src/cursus/steps/registry/step_names_original.py`)

**Step Registration Patterns**:
- Step name definitions
- Builder class associations
- Step type classifications
- Description and metadata

## Workspace-Specific Implementation Patterns

### For Shared Workspace Developers

**File Locations**:
- Script Contract: `src/cursus/steps/contracts/[name]_contract.py`
- Step Specification: `src/cursus/steps/specs/[name]_spec.py`
- Configuration: `src/cursus/steps/configs/config_[name]_step.py`
- Step Builder: `src/cursus/steps/builders/builder_[name]_step.py`
- Processing Script: `src/cursus/steps/scripts/[name].py`

**Registry Integration**:
```python
# Update src/cursus/registry/step_names_original.py
STEP_NAMES = {
    "[StepName]": {
        "config_class": "[StepName]Config",
        "builder_step_name": "[StepName]StepBuilder",
        "spec_type": "[StepName]",
        "sagemaker_step_type": "Processing",
        "description": "[Brief description]"
    },
}
```

### For Isolated Workspace Developers

**File Locations**:
- Script Contract: `development/projects/[project_id]/src/cursus_dev/steps/contracts/[name]_contract.py`
- Step Specification: `development/projects/[project_id]/src/cursus_dev/steps/specs/[name]_spec.py`
- Configuration: `development/projects/[project_id]/src/cursus_dev/steps/configs/config_[name]_step.py`
- Step Builder: `development/projects/[project_id]/src/cursus_dev/steps/builders/builder_[name]_step.py`
- Processing Script: `development/projects/[project_id]/src/cursus_dev/steps/scripts/[name].py`

**Workspace Registry Integration**:
```python
# Create development/projects/[project_id]/src/cursus_dev/registry/project_step_names.py
PROJECT_STEP_NAMES = {
    "[StepName]": {
        "config_class": "[StepName]Config",
        "builder_step_name": "[StepName]StepBuilder",
        "spec_type": "[StepName]",
        "sagemaker_step_type": "Processing",
        "description": "[Brief description]",
        "workspace_id": "[project_id]"
    },
}

# Register with workspace-aware registry system
from src.cursus.registry.step_names import set_workspace_context
with set_workspace_context("[project_id]"):
    # Project-specific steps are accessible through hybrid registry
    pass
```

**Hybrid Registry Access Pattern**:
```python
# Isolated workspace builders can access shared components
from src.cursus.steps.builders.builder_shared_step import SharedStepBuilder  # Read-only access
from ..builders.builder_project_step import ProjectStepBuilder  # Project-specific
```

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
def __init__(self, config, sagemaker_session=None, role=None, notebook_root=None, 
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
                     role=role, notebook_root=notebook_root,
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

#### 11. **Hyperparameter Handling Patterns (Training Steps)**

**Reference Implementation**: For complete hyperparameter handling patterns, see `src/cursus/steps/builders/builder_xgboost_training_step.py` which provides the definitive implementation for:

- **Lambda-Optimized File Operations**: Robust S3 upload with retry logic and resource management
- **Serverless Environment Compatibility**: Proper error handling for AWS Lambda execution
- **Configuration Serialization**: Converting config objects to hyperparameter dictionaries
- **S3 Path Generation**: Using Join() pattern for consistent path construction

**Key Hyperparameter Method Pattern**:
```python
def _prepare_hyperparameters_file(self) -> str:
    """
    Prepare hyperparameters file and upload to S3.
    
    This method follows the Lambda-optimized pattern from XGBoostTrainingStepBuilder
    with comprehensive error handling and resource management.
    
    Returns:
        S3 URI of the uploaded hyperparameters file
    """
    from sagemaker.workflow.functions import Join
    import json
    import tempfile
    import boto3
    from botocore.exceptions import ClientError
    import time
    import os
    
    # Generate hyperparameters dictionary from config
    hyperparams = self.config.to_hyperparameter_dict()
    
    # Create S3 path using Join pattern for consistency
    base_output_path = self._get_base_output_path()
    hyperparams_s3_uri = Join(
        on="/", 
        values=[base_output_path, "xgboost_training", "hyperparameters", "hyperparameters.json"]
    )
    
    # Lambda-optimized file upload with retry logic
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        temp_file_path = None
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                json.dump(hyperparams, temp_file, indent=2)
                temp_file_path = temp_file.name
            
            # Upload to S3 with proper error handling
            s3_client = boto3.client('s3')
            bucket, key = self._parse_s3_uri(hyperparams_s3_uri)
            
            with open(temp_file_path, 'rb') as file_obj:
                s3_client.upload_fileobj(file_obj, bucket, key)
            
            self.log_info(f"Successfully uploaded hyperparameters to: {hyperparams_s3_uri}")
            return hyperparams_s3_uri
            
        except ClientError as e:
            self.log_warning(f"S3 upload attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
            else:
                raise ValueError(f"Failed to upload hyperparameters after {max_retries} attempts: {e}")
                
        except Exception as e:
            self.log_error(f"Unexpected error during hyperparameters upload: {e}")
            raise
            
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except OSError as e:
                    self.log_warning(f"Failed to clean up temporary file {temp_file_path}: {e}")

def _parse_s3_uri(self, s3_uri: str) -> tuple:
    """Parse S3 URI into bucket and key components."""
    if not s3_uri.startswith('s3://'):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    
    parts = s3_uri[5:].split('/', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid S3 URI format: {s3_uri}")
    
    return parts[0], parts[1]
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
    """
    Get outputs for the step using specification and contract.

    For training steps, this returns the output path where model artifacts and evaluation results will be stored.
    SageMaker uses this single output_path parameter for both:
    - model.tar.gz (from /opt/ml/model/)
    - output.tar.gz (from /opt/ml/output/data/)

    Args:
        outputs: Output destinations keyed by logical name

    Returns:
        Output path for model artifacts and evaluation results

    Raises:
        ValueError: If no specification or contract is available
    """
    if not self.spec:
        raise ValueError("Step specification is required")

    if not self.contract:
        raise ValueError("Script contract is required for output mapping")

    # First, check if any output path is explicitly provided in the outputs dictionary
    primary_output_path = None

    # Check if model_output or evaluation_output are in the outputs dictionary
    output_logical_names = [
        spec.logical_name for _, spec in self.spec.outputs.items()
    ]

    for logical_name in output_logical_names:
        if logical_name in outputs:
            primary_output_path = outputs[logical_name]
            self.log_info(
                f"Using provided output path from '{logical_name}': {primary_output_path}"
            )
            break

    # If no output path was provided, generate a default one
    if primary_output_path is None:
        # Generate a clean path using base output path and Join for parameter compatibility
        from sagemaker.workflow.functions import Join
        base_output_path = self._get_base_output_path()
        primary_output_path = Join(on="/", values=[base_output_path, "xgboost_training"])
        self.log_info(f"Using generated base output path: {primary_output_path}")

    # Remove trailing slash if present for consistency with S3 path handling
    if primary_output_path.endswith("/"):
        primary_output_path = primary_output_path[:-1]

    # Get base job name for logging purposes
    base_job_name = self._generate_job_name()

    # Log how SageMaker will structure outputs under this path
    self.log_info(
        f"SageMaker will organize outputs using base job name: {base_job_name}"
    )
    self.log_info(f"Full job name will be: {base_job_name}-[timestamp]")
    self.log_info(
        f"Output path structure will be: {primary_output_path}/{base_job_name}-[timestamp]/"
    )
    self.log_info(
        f"  - Model artifacts will be in: {primary_output_path}/{base_job_name}-[timestamp]/output/model.tar.gz"
    )
    self.log_info(
        f"  - Evaluation results will be in: {primary_output_path}/{base_job_name}-[timestamp]/output/output.tar.gz"
    )

    return primary_output_path
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

## Expected Output Format

Present your plan in the following format:


# Implementation Plan for [Step Name]

## 1. Step Overview
- Purpose: [Brief description of the step's purpose]
- Inputs: [List of required inputs]
- Outputs: [List of produced outputs]
- Position in pipeline: [Where this step fits in the pipeline]
- Architectural considerations: [Key design decisions and their rationale]
- Alignment with design principles: [How this step follows our architectural patterns]

## 2. Components to Create
- Script Contract: src/pipeline_script_contracts/[name]_contract.py
  - Input paths: [List logical names and container paths]
  - Output paths: [List logical names and container paths]
  - Environment variables: [List required and optional env vars]
  
- Step Specification: src/pipeline_step_specs/[name]_spec.py
  - Dependencies: [List dependency specs with compatible sources]
  - Outputs: [List output specs with property paths]
  - Job type variants: [List any variants needed]
  
- Configuration: src/pipeline_steps/config_[name].py
  - Step-specific parameters: [List parameters with defaults]
  - SageMaker parameters: [List instance type, count, etc.]
  - Required validation checks: [List of validation checks to implement]
  
- Step Builder: src/pipeline_steps/builder_[name].py
  - Special handling: [Any special logic needed]
  - Required helper methods: [List of helper methods to implement]
  - Input/output handling: [How _get_inputs and _get_outputs should be implemented]
  
- Processing Script: src/pipeline_scripts/[name].py
  - Algorithm: [Brief description of algorithm]
  - Main functions: [List of main functions]
  - Error handling strategy: [How to handle errors in the script]

## 3. Files to Update
- src/cursus/steps/registry/step_names.py
- src/cursus/steps/builders/__init__.py
- src/cursus/steps/configs/__init__.py
- src/cursus/steps/specs/__init__.py
- src/cursus/steps/contracts/__init__.py
- src/cursus/steps/scripts/__init__.py
- [Any template files that need updating]

## 4. Integration Strategy
- Upstream steps: [List steps that can provide inputs]
- Downstream steps: [List steps that can consume outputs]
- DAG updates: [How to update the pipeline DAG]

## 5. Contract-Specification Alignment
- Input alignment: [How contract input paths map to specification dependency names]
- Output alignment: [How contract output paths map to specification output names]
- Validation strategy: [How to ensure alignment during development]

## 6. Error Handling Strategy
- Input validation: [How to validate inputs]
- Script robustness: [How to handle common failure modes]
- Logging strategy: [What to log and at what levels]
- Error reporting: [How errors are communicated to the pipeline]

## 7. Testing and Validation Plan
- Unit tests: [Tests for individual components]
- Integration tests: [Tests for step in pipeline context]
- Validation criteria: [How to verify step is working correctly]
- S3 path handling tests: [How to test S3 path handling]
- PipelineVariable handling tests: [How to test PipelineVariable handling]

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

#### B. Builder Registration with Decorator

The step builder will be automatically registered using the `@register_builder()` decorator:

```python
# In src/cursus/steps/builders/builder_[name]_step.py
from ..registry.builder_registry import register_builder

@register_builder()  # Auto-detects step type from STEP_NAMES registry
class [StepName]StepBuilder(StepBuilderBase):
    """Builder for [StepName] step."""
    pass
```

**Alternative Manual Registration:**
```python
@register_builder(step_type="[StepName]")  # Explicit step type
class [StepName]StepBuilder(StepBuilderBase):
    pass
```

#### C. Registry Validation Functions

The registry provides validation functions to ensure consistency:

```python
# Validate step name exists
from src.cursus.steps.registry.step_names import validate_step_name
assert validate_step_name("[StepName]") == True

# Get step information
from src.cursus.steps.registry.step_names import get_step_description
description = get_step_description("[StepName]")

# Get SageMaker step type
from src.cursus.steps.registry.step_names import get_sagemaker_step_type
sagemaker_type = get_sagemaker_step_type("[StepName]")
```

#### D. Job Type Variants (if applicable)

For steps that have job type variants (like preprocessing with training/validation/testing):

```python
# In step_names.py - base entry
"TabularPreprocessing": {
    "config_class": "TabularPreprocessingConfig",
    "builder_step_name": "TabularPreprocessingStepBuilder",
    "spec_type": "TabularPreprocessing", 
    "sagemaker_step_type": "Processing",
    "description": "Tabular data preprocessing with job type variants"
},

# Job type handling in specifications
from src.cursus.steps.registry.step_names import get_spec_step_type_with_job_type

# Creates "TabularPreprocessing_Training", "TabularPreprocessing_Validation", etc.
training_spec_type = get_spec_step_type_with_job_type("TabularPreprocessing", "Training")
```

#### E. Legacy Alias Support (if needed)

If the new step replaces an old step or needs backward compatibility:

```python
# In src/cursus/steps/registry/builder_registry.py
class StepBuilderRegistry:
    LEGACY_ALIASES = {
        # ... existing aliases ...
        "OldStepName": "[StepName]",  # Maps old name to new canonical name
    }
```

#### F. Registry Discovery and Validation

The registry automatically discovers builders, but you can validate the setup:

```python
# Check registry consistency
from src.cursus.steps.registry.builder_registry import get_global_registry

registry = get_global_registry()

# Validate all mappings
validation_results = registry.validate_registry()
print("Valid mappings:", validation_results['valid'])
print("Invalid mappings:", validation_results['invalid']) 
print("Missing builders:", validation_results['missing'])

# Get registry statistics
stats = registry.get_registry_stats()
print(f"Total builders: {stats['total_builders']}")

# List all supported step types
supported_types = registry.list_supported_step_types()
print("Supported step types:", supported_types)
```

#### G. Complete Registry Integration Example

Here's a complete example for adding a new "DataValidation" step:

```python
# 1. Add to step_names.py
STEP_NAMES = {
    # ... existing entries ...
    "DataValidation": {
        "config_class": "DataValidationConfig",
        "builder_step_name": "DataValidationStepBuilder",
        "spec_type": "DataValidation",
        "sagemaker_step_type": "Processing",
        "description": "Validates input data quality and schema compliance"
    },
}

# 2. Create builder with auto-registration
# src/cursus/steps/builders/builder_data_validation_step.py
from ..registry.builder_registry import register_builder
from ...core.base.builder_base import StepBuilderBase

@register_builder()  # Automatically maps to "DataValidation" from STEP_NAMES
class DataValidationStepBuilder(StepBuilderBase):
    """Builder for Data Validation ProcessingStep."""
    
    def __init__(self, config, sagemaker_session=None, role=None, 
                 notebook_root=None, registry_manager=None, dependency_resolver=None):
        # Specification loading
        if not DATA_VALIDATION_SPEC:
            raise ValueError("Data validation specification not available")
            
        super().__init__(config=config, spec=DATA_VALIDATION_SPEC, 
                         sagemaker_session=sagemaker_session, role=role,
                         notebook_root=notebook_root, registry_manager=registry_manager,
                         dependency_resolver=dependency_resolver)

# 3. Verify registration
from src.cursus.steps.registry.builder_registry import get_global_registry

registry = get_global_registry()
builder_class = registry.get_builder_for_step_type("DataValidation")
print(f"Registered builder: {builder_class.__name__}")

# 4. Test with configuration
from src.cursus.steps.configs.config_data_validation_step import DataValidationConfig

config = DataValidationConfig(
    author="test",
    bucket="test-bucket", 
    role="test-role",
    region="NA",
    service_name="test-service",
    pipeline_version="1.0"
)

builder_class = registry.get_builder_for_config(config)
print(f"Builder for config: {builder_class.__name__}")
```

#### H. Registry Helper Functions Usage

The registry provides many helper functions for step management:

```python
from src.cursus.steps.registry.step_names import (
    get_config_class_name,
    get_builder_step_name, 
    get_spec_step_type,
    get_all_step_names,
    get_steps_by_sagemaker_type,
    get_sagemaker_step_type_mapping
)

# Get configuration class name for a step
config_class = get_config_class_name("[StepName]")  # Returns "[StepName]Config"

# Get builder class name for a step  
builder_name = get_builder_step_name("[StepName]")  # Returns "[StepName]StepBuilder"

# Get specification type for a step
spec_type = get_spec_step_type("[StepName]")  # Returns "[StepName]"

# List all registered step names
all_steps = get_all_step_names()

# Get all processing steps
processing_steps = get_steps_by_sagemaker_type("Processing")

# Get complete mapping of SageMaker types to steps
type_mapping = get_sagemaker_step_type_mapping()
```

#### I. Import Updates Required

After adding the new step to the registry, update the following import files:

```python
# src/cursus/steps/builders/__init__.py
from .builder_[name]_step import [StepName]StepBuilder

# src/cursus/steps/configs/__init__.py  
from .config_[name]_step import [StepName]Config

# src/cursus/steps/specs/__init__.py
from .[name]_spec import [NAME]_SPEC

# src/cursus/steps/contracts/__init__.py
from .[name]_contract import [NAME]_CONTRACT

# src/cursus/steps/scripts/__init__.py (if applicable)
# No import needed - scripts are executed directly
```

#### J. Registry Testing and Validation

Create tests to ensure registry integration works correctly:

```python
# test/steps/registry/test_[name]_registry.py
import pytest
from src.cursus.steps.registry.builder_registry import get_global_registry
from src.cursus.steps.registry.step_names import validate_step_name, get_sagemaker_step_type
from src.cursus.steps.configs.config_[name]_step import [StepName]Config

def test_step_registry_integration():
    """Test that [StepName] is properly registered."""
    # Test step name validation
    assert validate_step_name("[StepName]") == True
    
    # Test SageMaker step type
    sagemaker_type = get_sagemaker_step_type("[StepName]")
    assert sagemaker_type in ["Processing", "Training", "CreateModel", "Transform"]
    
    # Test builder registry
    registry = get_global_registry()
    assert registry.is_step_type_supported("[StepName]")
    
    # Test builder retrieval
    builder_class = registry.get_builder_for_step_type("[StepName]")
    assert builder_class.__name__ == "[StepName]StepBuilder"

def test_config_to_builder_mapping():
    """Test that configuration maps to correct builder."""
    config = [StepName]Config(
        author="test", bucket="test-bucket", role="test-role",
        region="NA", service_name="test", pipeline_version="1.0",
        # ... step-specific required parameters
    )
    
    registry = get_global_registry()
    builder_class = registry.get_builder_for_config(config)
    assert builder_class.__name__ == "[StepName]StepBuilder"
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

### 5. Step Builder Implementation (Critical Methods)

```python
# src/cursus/steps/builders/builder_[name]_step.py

def validate_configuration(self) -> None:
    """
    Validates the provided configuration to ensure all required fields for this
    specific step are present and valid before attempting to build the step.

    Raises:
        ValueError: If any required configuration is missing or invalid.
    """
    self.log_info("Validating [StepName]Config...")
    
    # Validate required attributes
    required_attrs = [
        'attribute1',
        'attribute2',
        # ...
    ]
    
    for attr in required_attrs:
        if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
            raise ValueError(f"[StepName]Config missing required attribute: {attr}")
            
    self.log_info("[StepName]Config validation succeeded.")

def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
    """
    Get inputs for the processor using the specification and contract.
    
    Args:
        inputs: Dictionary of input sources keyed by logical name
        
    Returns:
        List of ProcessingInput objects for the processor
        
    Raises:
        ValueError: If no specification or contract is available
    """
    if not self.spec:
        raise ValueError("Step specification is required")
        
    if not self.contract:
        raise ValueError("Script contract is required for input mapping")
        
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
        container_path = None
        if logical_name in self.contract.expected_input_paths:
            container_path = self.contract.expected_input_paths[logical_name]
            # Map input to container path
            # ...
        else:
            raise ValueError(f"No container path found for input: {logical_name}")
            
    return processing_inputs
```

### 6. Processing Script Implementation (Error Handling)

```python
# src/pipeline_scripts/[name].py
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
```

Remember to follow the Step Creation Process outlined in the documentation, carefully considering alignment rules between layers and ensuring your plan adheres to our design principles and standardization rules. Pay special attention to downstream component compatibility, especially with dependency resolver requirements.
