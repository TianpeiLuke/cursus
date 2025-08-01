# Dummy Training Step Enhancement Implementation Plan

## Overview

The Dummy Training step currently copies a pretrained model.tar.gz file from the input location to the output location. To integrate with MIMS packaging and payload steps, we need to enhance the step to unpack the model.tar.gz file, add a hyperparameters.json file inside, and repack it. This ensures that downstream steps have access to the hyperparameter information they require.

## Required Changes

### 1. Modify Dummy Training Script

**File:** `src/pipeline_scripts/dummy_training.py`

**Current functionality:**
- Validates model.tar.gz file format
- Copies the file from input to output location

**Changes required:**
- Add a new parameter for the hyperparameter.json file path
- Implement functionality to:
  1. Extract the model.tar.gz to a working directory
  2. Add the hyperparameter.json file to the extracted contents
  3. Repack everything into a new model.tar.gz
  4. Output the new model.tar.gz to the specified location

**Implementation approach:**
- Use the existing mims_package.py implementation as a reference
- Adapt the code to work with:
  - `/opt/ml/processing/input/model/model.tar.gz` as the input model
  - `/opt/ml/processing/input/config/hyperparameters.json` as the hyperparameters input
  - `/opt/ml/processing/output/model` as the output directory

### 2. Update Script Contract

**File:** `src/pipeline_script_contracts/dummy_training_contract.py`

**Current contract:**
```python
DUMMY_TRAINING_CONTRACT = ScriptContract(
    entry_point="dummy_training.py",
    expected_input_paths={
        "pretrained_model_path": "/opt/ml/processing/input/model/model.tar.gz"
    },
    expected_output_paths={
        "model_input": "/opt/ml/processing/output/model"
    },
    required_env_vars=[],
    optional_env_vars={},
    framework_requirements={
        "boto3": ">=1.26.0",
        "pathlib": ">=1.0.0"
    },
    description="Contract for dummy training step that copies a pretrained model.tar.gz to output location"
)
```

**Changes required:**
- Add a new input path for hyperparameters.json
- Update the description to reflect the new functionality
- Add any new framework requirements if needed

**Implementation approach:**
- Add `"hyperparameters_path": "/opt/ml/processing/input/config/hyperparameters.json"` to `expected_input_paths`
- Update the description to mention the hyperparameter inclusion process
- Ensure backward compatibility by maintaining the existing input and output paths

### 3. Update Step Specification

**File:** `src/pipeline_step_specs/dummy_training_spec.py`

**Current specification:**
```python
DUMMY_TRAINING_SPEC = StepSpecification(
    step_type=get_spec_step_type("DummyTraining"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_dummy_training_contract(),
    dependencies=[
        DependencySpec(
            logical_name="pretrained_model_path",
            dependency_type=DependencyType.PROCESSING_INPUT,
            required=True,
            compatible_sources=["ProcessingStep", "XGBoostTraining", "PytorchTraining", "TabularPreprocessing"],
            semantic_keywords=["model", "pretrained", "artifact", "weights", "training_output", "model_data"],
            data_type="S3Uri",
            description="Path to pretrained model.tar.gz file"
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="model_input",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ProcessingOutputConfig.Outputs['model_input'].S3Output.S3Uri",
            data_type="S3Uri",
            description="S3 path to model artifacts",
            aliases=["ModelOutputPath", "ModelArtifacts", "model_data", "output_path"]
        )
    ]
)
```

**Changes required:**
- Add a new dependency for hyperparameters
- Ensure the dependency is properly configured with appropriate:
  - logical_name
  - dependency_type
  - required flag
  - compatible sources
  - semantic keywords
  - data type
  - description

**Implementation approach:**
- Add a new `DependencySpec` with `logical_name="hyperparameters_path"` similar to the XGBoost training spec
- Set `dependency_type=DependencyType.HYPERPARAMETERS`
- Set `required=True` as we now need the hyperparameters
- Configure compatible sources and semantic keywords to match the expected inputs

### 4. Update Step Builder and Config

**File:** `src/pipeline_steps/builder_dummy_training_step.py` (might need to confirm existence or create)

**Changes required:**
- Implement functionality to handle hyperparameters as input
- Add mechanisms to:
  - Save hyperparameters from config to temp S3
  - Use the saved hyperparameters as input to the step
  - Similar to XGBoost training step builder's methods:
    - `_get_inputs`
    - `_prepare_hyperparameters_file`

**Implementation approach:**
- If a builder for dummy training exists, update it with the required functionality
- If not, create a new builder based on other training step builders
- Implement methods to prepare and provide hyperparameters to the step

## Testing and Validation

1. **Unit Testing:**
   - Test the modified dummy_training.py script with various inputs
   - Verify that the hyperparameters.json file is correctly included in the output model.tar.gz

2. **Integration Testing:**
   - Test the entire pipeline flow from dummy training to packaging and payload steps
   - Verify that downstream steps can access and use the hyperparameters.json

3. **Validation Checks:**
   - Ensure backward compatibility is maintained
   - Verify that the implementation properly handles edge cases like:
     - Missing hyperparameters
     - Invalid model.tar.gz files
     - File permission issues

## Expected Benefits

1. Seamless integration between dummy training step and downstream MIMS packaging and payload steps
2. Consistent model artifact structure across different training approaches
3. Ability to pass hyperparameter information through the pipeline even when using pretrained models

## Implementation Timeline

1. Script modification: 2-3 hours
2. Contract updates: 30 minutes
3. Step specification updates: 1 hour
4. Step builder modifications: 2-3 hours
5. Testing and validation: 2-3 hours
6. Documentation and code review: 1-2 hours

Total estimated time: 8-12 hours

## Dependencies and Prerequisites

- Understanding of the current dummy training step implementation
- Familiarity with the MIMS packaging and payload step requirements
- Access to the codebase and ability to test pipeline execution

## Risks and Mitigations

1. **Risk:** Breaking existing pipelines that use dummy training step
   **Mitigation:** Ensure backward compatibility by keeping existing functionality intact

2. **Risk:** Performance impact when unpacking/repacking large model files
   **Mitigation:** Optimize file operations and consider streaming approaches for large files

3. **Risk:** File permission issues in container environment
   **Mitigation:** Add robust error handling and clear error messages
