# Initial Pipeline Step Planner Prompt (Enhanced)

## Your Role: Pipeline Step Planner

You are an expert ML Pipeline Architect tasked with planning a new pipeline step for our SageMaker-based ML pipeline system. Your job is to analyze requirements, determine what components need to be created or modified, and create a comprehensive plan for implementing the new step.

## Pipeline Architecture Context

Our pipeline architecture follows a specification-driven approach with a four-layer design:

1. **Step Specifications**: Define inputs and outputs with logical names
2. **Script Contracts**: Define container paths for script inputs/outputs
3. **Step Builders**: Connect specifications and contracts via SageMaker
4. **Processing Scripts**: Implement the actual business logic

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

The Model Calibration step is responsible for taking a trained model's raw prediction scores and calibrating them to better reflect true probabilities. This is essential for:

1. Ensuring model scores accurately represent true probabilities
2. Making correct risk-based decisions
3. Properly setting classification thresholds
4. Interpreting model outputs with confidence

Key requirements for the step:

1. Accept a dataset with features, ground truth labels, and precomputed model prediction scores (likely a byproduct of training)
2. Apply one of several calibration methods:
   - GAM (Generalized Additive Model) with monotonic constraint
   - Isotonic Regression
   - Platt Scaling
3. Create a mapping between original scores and calibrated probabilities
4. Output this calibration mapping so it can be applied to future predictions
5. Generate calibration metrics showing before/after reliability diagrams and calibration error
6. Support configuration parameters including:
   - Calibration method selection
   - Method-specific parameters (e.g., number of splines for GAM)
   - Thresholds for acceptable calibration error
   - Output formats and visualization options
7. Handle error cases including:
   - Calibration failing to converge
   - Insufficient data for reliable calibration
   - Problematic model score distributions

The calibration step should fit after XGBoost model training and before the MIMS packaging step. You can find the XGBoost training script in dockers/xgboost_atoz/train_xgb.py. In particular, the input of model calibration is the "evaluation_output" of XGBoost model training step (see src/pipeline_step_specs) which is the evaluted dataset with ground truth label and prediction score. (see dockers/xgboost_atoz/train_xgb.py and verify the content saved in /opt/ml/output/data within the container) 

## Relevant Documentation

### Creation Process Overview

Refer to: `slipbox/developer_guide/creation_process.md`

Creating a new pipeline step involves creating several components that work together:

1. **Register the new step name in the step registry**:
   - Add your step to `src/pipeline_registry/step_names.py`
   - Define config class, builder name, and step type

2. **Create the Step Configuration**:
   - Create `src/pipeline_steps/config_your_new_step_step.py`
   - Include SageMaker parameters and step-specific parameters
   - Add method to retrieve script contract

3. **Develop the Script Contract**:
   - Define input/output paths, environment variables, and dependencies
   - Create `src/pipeline_script_contracts/your_new_step_contract.py`

4. **Create the Step Specification**:
   - Define dependencies and outputs with logical names
   - Create `src/pipeline_step_specs/your_new_step_spec.py`

5. **Implement the Step Builder**:
   - Create `src/pipeline_steps/builder_your_new_step_step.py`
   - Implement validation, input/output mapping, and step creation

6. **Update Required Registry Files**:
   - Update import statements in `__init__.py` files

7. **Create Unit Tests**:
   - Test builder, specification, and alignment

8. **Integrate with Pipeline Templates**:
   - Add your step to pipeline DAG creation

### Prerequisites

Refer to: `slipbox/developer_guide/prerequisites.md`

Before you begin developing a new pipeline step, ensure you have:

1. **Task Description**:
   - Clear purpose, input requirements, expected outputs, pipeline position
   - Success criteria for verifying the step works correctly

2. **Processing Script Requirements**:
   - Understanding of calibration algorithms (GAM, isotonic regression, Platt scaling)
   - Error handling and logging needs
   - Path requirements and environment variables

3. **Step Identification**:
   - Canonical name following conventions (e.g., `ModelCalibration`)
   - Step type and job type variants
   - Logical names for semantic matching

4. **SageMaker Component Understanding**:
   - ProcessingStep requirements for calibration operations
   - Understanding of container paths and environment configuration

### Alignment Rules

Refer to: `slipbox/developer_guide/alignment_rules.md`

Follow these alignment principles when developing the new step:

1. **Script ↔ Contract**:
   - Scripts must use exactly the paths defined in their Script Contract
   - Environment variables, input/output structures, and file patterns must match

2. **Contract ↔ Specification**:
   - Logical names in the Script Contract must match dependency names in the Step Specification
   - Property paths in OutputSpec must correspond to the contract's output paths

3. **Specification ↔ Dependencies**:
   - Dependencies declared must match upstream step outputs by logical name or alias
   - Compatible sources must list all steps that produce the required output

4. **Builder ↔ Configuration**:
   - Step Builders must pass configuration parameters according to the config class
   - Environment variables must cover all required_env_vars from the contract

### Standardization Rules

Refer to: `slipbox/developer_guide/standardization_rules.md`

Adhere to these standardization rules for consistency:

1. **Naming Conventions**:
   - Step Types: Use PascalCase (e.g., `ModelCalibration`)
   - Logical Names: Use snake_case (e.g., `calibration_data_input`)
   - Config Classes: Use PascalCase + Config suffix (e.g., `ModelCalibrationConfig`)
   - Builder Classes: Use PascalCase + StepBuilder suffix (e.g., `ModelCalibrationStepBuilder`)

2. **Interface Standardization**:
   - Step Builders: Inherit from `StepBuilderBase` and implement required methods
   - Config Classes: Inherit from base config class and implement required methods
   - Standardized documentation for all components

3. **Error Handling Standards**:
   - Use standard exception hierarchy
   - Provide meaningful error messages with codes
   - Include suggestions for resolution
   - Log errors appropriately

## Example of Similar Steps

The model calibration step will have similarities to the existing model evaluation step. Here are some key aspects of the model evaluation step for reference:

### Model Evaluation Step Specification
```python
# Key parts of model_eval_spec.py
MODEL_EVAL_SPEC = StepSpecification(
    step_type=get_spec_step_type("XGBoostModelEval"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_model_evaluation_contract(),
    dependencies=[
        DependencySpec(
            logical_name="model_input",
            dependency_type=DependencyType.MODEL_ARTIFACTS,
            required=True,
            compatible_sources=["XGBoostTraining", "TrainingStep", "ModelStep"],
            semantic_keywords=["model", "artifacts", "trained", "output", "ModelArtifacts"],
            data_type="S3Uri",
            description="Trained model artifacts to be evaluated (includes hyperparameters.json)"
        ),
        DependencySpec(
            logical_name="processed_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["TabularPreprocessing", "ProcessingStep", "DataLoad"],
            semantic_keywords=["data", "evaluation", "calibration", "validation", "test", "processed"],
            data_type="S3Uri",
            description="Evaluation dataset for model assessment"
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="eval_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['eval_output'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Model evaluation results including predictions"
        ),
        OutputSpec(
            logical_name="metrics_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['metrics_output'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Model evaluation metrics (AUC, precision, recall, etc.)"
        )
    ]
)
```

### Model Evaluation Builder
```python
# Key methods from builder_model_eval_step_xgboost.py
class XGBoostModelEvalStepBuilder(StepBuilderBase):
    """Builder for an XGBoost Model Evaluation ProcessingStep."""
    
    def validate_configuration(self) -> None:
        """Validates the provided configuration."""
        self.log_info("Validating XGBoostModelEvalConfig...")
        
        # Validate required attributes
        required_attrs = [
            'processing_entry_point',
            'processing_source_dir',
            'processing_instance_count', 
            'processing_volume_size',
            'pipeline_name',
            'job_type',
            'hyperparameters',
            'xgboost_framework_version'
        ]
        
        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
                raise ValueError(f"XGBoostModelEvalConfig missing required attribute: {attr}")

    def _get_environment_variables(self) -> Dict[str, str]:
        """Constructs environment variables for the processing job."""
        # Get base environment variables from contract
        env_vars = super()._get_environment_variables()
        
        # Add evaluation-specific environment variables
        if hasattr(self.config, 'hyperparameters'):
            if hasattr(self.config.hyperparameters, 'id_name'):
                env_vars["ID_FIELD"] = str(self.config.hyperparameters.id_name)
            if hasattr(self.config.hyperparameters, 'label_name'):
                env_vars["LABEL_FIELD"] = str(self.config.hyperparameters.label_name)
        
        return env_vars
```

## Critical Implementation Patterns

### Builder Implementation Patterns
- **Specification and Contract Validation**: Always verify that specification and contract are available in the builder and raise appropriate errors if not:
```python
if not SPEC_AVAILABLE or STEP_SPEC is None:
    raise ValueError("Step specification not available")

if not self.spec:
    raise ValueError("Step specification is required")
            
if not self.contract:
    raise ValueError("Script contract is required for input mapping")
```

- **S3 Path Handling**: Include helper methods for consistent S3 path handling:
```python
def _normalize_s3_uri(self, uri: str, description: str = "S3 URI") -> str:
    # Handle PipelineVariable objects
    if hasattr(uri, 'expr'):
        uri = str(uri.expr)
    
    # Handle Pipeline step references with Get key - return as is
    if isinstance(uri, dict) and 'Get' in uri:
        self.log_info("Found Pipeline step reference: %s", uri)
        return uri
    
    return S3PathHandler.normalize(uri, description)
```

- **Input/Output Methods**: Always use specification and contract for mapping inputs and outputs:
```python
def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
    """
    Get inputs for the processor using the specification and contract.
    Must check for specification and contract availability first.
    """
    if not self.spec:
        raise ValueError("Step specification is required")
        
    if not self.contract:
        raise ValueError("Script contract is required for input mapping")
        
    # Process each dependency in the specification
    processing_inputs = []
    for _, dependency_spec in self.spec.dependencies.items():
        logical_name = dependency_spec.logical_name
        
        # Get container path from contract
        if logical_name in self.contract.expected_input_paths:
            container_path = self.contract.expected_input_paths[logical_name]
            # Map input to container path
            # ...
    return processing_inputs
```

### Configuration Validation Patterns
- **Required Attribute Validation**:
```python
def validate_configuration(self) -> None:
    self.log_info("Validating StepConfig...")
    
    # Validate required attributes
    required_attrs = [
        'attribute1',
        'attribute2',
        # ...
    ]
    
    for attr in required_attrs:
        if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
            raise ValueError(f"Config missing required attribute: {attr}")
```

### Error Handling Patterns
- **Comprehensive Try/Except Blocks**:
```python
try:
    # Critical operation
    result = self._perform_operation()
    return result
except Exception as e:
    self.log_error(f"Failed to perform operation: {e}")
    import traceback
    self.log_error(traceback.format_exc())
    raise ValueError(f"Operation failed: {str(e)}") from e
```

## Expected Output Format

Present your plan in the following format:

```
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
- src/pipeline_registry/step_names.py
- src/pipeline_steps/__init__.py
- src/pipeline_step_specs/__init__.py
- src/pipeline_script_contracts/__init__.py
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

```python
# In src/pipeline_registry/step_names.py
STEP_NAMES = {
    # ... existing steps ...
    
    "[Step Name]": {
        "config_class": "[StepName]Config",
        "builder_step_name": "[StepName]StepBuilder",
        "spec_type": "[StepName]",
        "description": "[Brief description]"
    },
}
```

### 2. Script Contract Implementation

```python
# src/pipeline_script_contracts/[name]_contract.py
from .base_script_contract import ScriptContract

[NAME]_CONTRACT = ScriptContract(
    entry_point="[name].py",
    expected_input_paths={
        # Input paths with SageMaker container locations
    },
    expected_output_paths={
        # Output paths with SageMaker container locations
    },
    required_env_vars=[],
    optional_env_vars={},
    framework_requirements={},
    description="Contract for [step description]"
)
```

### 3. Step Specification Implementation

```python
# src/pipeline_step_specs/[name]_spec.py
from ..pipeline_deps.base_specifications import StepSpecification, NodeType, DependencySpec, OutputSpec, DependencyType
from ..pipeline_registry.step_names import get_spec_step_type

def _get_[name]_contract():
    from ..pipeline_script_contracts.[name]_contract import [NAME]_CONTRACT
    return [NAME]_CONTRACT

[NAME]_SPEC = StepSpecification(
    step_type=get_spec_step_type("[StepName]"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_[name]_contract(),
    dependencies=[
        # List of dependency specifications
    ],
    outputs=[
        # List of output specifications
    ]
)
```

### 4. Configuration Class Implementation

```python
# src/pipeline_steps/config_[name].py
from .[appropriate_base_config] import [BaseConfig]

class [StepName]Config([BaseConfig]):
    """Configuration for [StepName]."""
    
    def __init__(
        self,
        region: str,
        pipeline_s3_loc: str,
        # Additional parameters
    ):
        """Initialize [StepName] configuration."""
        super().__init__(region, pipeline_s3_loc)
        # Set additional parameters
    
    def get_script_contract(self):
        """Return the script contract for this step."""
        from ..pipeline_script_contracts.[name]_contract import [NAME]_CONTRACT
        return [NAME]_CONTRACT
```

### 5. Step Builder Implementation (Critical Methods)

```python
# src/pipeline_steps/builder_[name].py

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
