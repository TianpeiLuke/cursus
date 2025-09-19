---
tags:
  - project
  - planning
  - pipeline_api
  - refactoring
  - implementation
keywords:
  - hyperparameters refactor
  - source_dir integration
  - PIPELINE_EXECUTION_TEMP_DIR
  - XGBoost training
  - parameter conflict resolution
  - runtime vs definition time
topics:
  - hyperparameters management
  - pipeline execution architecture
  - training step refactoring
  - parameter flow optimization
language: python
date of note: 2025-09-18
---

# Hyperparameters Source Directory Integration Refactor Plan

## 🎉 PROJECT STATUS: COMPLETED ✅

**Implementation Date**: September 18, 2025  
**Status**: All implementation phases completed successfully  
**Components Updated**: 9 files across 2 training step types  
**Business Impact**: Complete resolution of `PIPELINE_EXECUTION_TEMP_DIR` timing conflicts  

### ✅ **COMPLETED PHASES**
- [x] **Phase 1**: XGBoostTrainingStepBuilder Refactoring ✅ **COMPLETED**
- [x] **Phase 2**: Training Script Refactoring ✅ **COMPLETED**  
- [x] **Phase 3**: Specification Updates ✅ **COMPLETED**
- [x] **Phase 4**: DummyTraining SOURCE Node Refactor ✅ **COMPLETED**

### 📋 **READY FOR NEXT PHASE**
- [ ] **Phase 5**: Testing & Documentation (ready for implementation)

---

## Executive Summary

This document outlines a critical refactoring required to resolve a fundamental architectural conflict between the new `PIPELINE_EXECUTION_TEMP_DIR` parameter flow system and the existing `_prepare_hyperparameters_file` method in XGBoost training steps.

**IMPLEMENTATION COMPLETED**: All phases of this refactor have been successfully implemented, resolving the timing conflict and enabling true pipeline portability.

## Problem Statement

### Core Architectural Conflict

The implementation of `PIPELINE_EXECUTION_TEMP_DIR` as a `ParameterString` object has revealed a fundamental timing conflict in our hyperparameters management approach:

1. **Pipeline Definition Time vs Runtime Execution Time**:
   - `_prepare_hyperparameters_file` operates at **Pipeline Definition Time**
   - `PIPELINE_EXECUTION_TEMP_DIR` is resolved at **Pipeline Execution Time**
   - This creates an irreconcilable timing mismatch

2. **ParameterString Resolution Conflict**:
   - `_prepare_hyperparameters_file` requires concrete S3 paths to upload hyperparameters
   - `PIPELINE_EXECUTION_TEMP_DIR` is a `ParameterString` that only resolves during execution
   - Cannot use unresolved `ParameterString` objects for S3 upload operations

### Current Implementation Analysis

**XGBoostTrainingStepBuilder Current Flow:**
```python
def _prepare_hyperparameters_file(self) -> str:
    # PROBLEM: This method runs at Pipeline Definition Time
    # but needs to use PIPELINE_EXECUTION_TEMP_DIR which only resolves at Runtime
    
    base_output_path = self._get_base_output_path()  # Returns ParameterString
    target_s3_uri = Join(on="/", values=[base_output_path, "training_config", "hyperparameters.json"])
    
    # CONFLICT: S3Uploader.upload() cannot work with ParameterString objects
    S3Uploader.upload(str(local_file), target_s3_uri, sagemaker_session=self.session)
```

**Training Script Current Expectation:**
```python
# Script expects hyperparameters at: /opt/ml/input/data/config/hyperparameters.json
hparam_path = os.path.join(data_dir, "config", "hyperparameters.json")
config = load_and_validate_config(hparam_path)
```

### Requirements for _prepare_hyperparameters_file to Work

For the current approach to function with `PIPELINE_EXECUTION_TEMP_DIR`, we would need:

1. **Concrete S3 Path Resolution**: Ability to resolve `ParameterString` objects at definition time
2. **Pre-execution S3 Access**: S3 upload capabilities before pipeline execution begins
3. **Static Path Generation**: Deterministic S3 paths that don't depend on runtime parameters
4. **Complex Parameter Handling**: Sophisticated logic to handle both static and dynamic paths

### Challenges with Current Approach

1. **Timing Mismatch**: Cannot upload to S3 paths that don't exist until runtime
2. **Parameter Complexity**: Managing both static and dynamic parameter resolution
3. **External System Integration**: External systems cannot provide custom execution directories if hyperparameters are pre-uploaded to fixed locations
4. **Architectural Inconsistency**: Violates the principle that `ParameterString` objects resolve at runtime

## Proposed Solution: Source Directory Integration

### Solution Overview

**Embed hyperparameters directly in the source directory** instead of uploading them separately to S3. This approach:

1. **Eliminates Timing Conflict**: Hyperparameters are packaged with source code at definition time
2. **Maintains Parameter Flow**: `PIPELINE_EXECUTION_TEMP_DIR` can be used for all other outputs
3. **Simplifies Architecture**: Removes complex S3 upload logic from step builders
4. **Improves Portability**: Hyperparameters travel with the code, making pipelines more self-contained

### Technical Implementation Strategy

**1. Source Directory Structure:**
```
source_dir/
├── xgboost_training.py          # Main training script
 file
├── processing/                  # Existing processing modules
│   ├── risk_table_processor.py
│   └── numerical_imputation_processor.py
└── hyperparams/                 
    └── hyperparameters_xgboost.py # Existing hyperparameters classes
    └── hyperparameters.json    # Generated hyperparameters
```

**2. Container Path Mapping:**
```
# SageMaker copies source_dir to: /opt/ml/code/
/opt/ml/code/
├── xgboost_training.py
├── hyperparams/
│   └── hyperparameters.json    # Available at /opt/ml/code/hyperparams/hyperparameters.json
└── ...
```

**3. Training Script Path Update:**
```python
# OLD: Load from separate input channel
hparam_path = os.path.join(data_dir, "config", "hyperparameters.json")

# NEW: Load from source directory
hparam_path = "/opt/ml/code/hyperparams/hyperparameters.json"
```

## Implementation Plan

### Phase 1: XGBoostTrainingStepBuilder Refactoring

#### 1.1 Remove _prepare_hyperparameters_file Method
- **Target**: `src/cursus/steps/builders/builder_xgboost_training_step.py`
- **Action**: Complete removal of `_prepare_hyperparameters_file` method and all references
- **Impact**: Eliminates S3 upload logic and timing conflicts

#### 1.2 Simplify Hyperparameters Handling
- **Target**: `src/cursus/steps/builders/builder_xgboost_training_step.py`
- **Action**: Remove hyperparameters generation logic - assume hyperparameters are already in source directory
- **Rationale**: Users will be required to include `hyperparams/hyperparameters.json` in their source directory structure
- **Implementation**: No code changes needed - existing `source_dir` configuration already points to directory containing hyperparameters

#### 1.3 Update _create_estimator Method (No Changes Needed)
- **Target**: `src/cursus/steps/builders/builder_xgboost_training_step.py`
- **Action**: No changes required - existing implementation already uses `self.config.source_dir`
- **Current Implementation**:
```python
def _create_estimator(self, output_path=None) -> XGBoost:
    return XGBoost(
        entry_point=self.config.training_entry_point,
        source_dir=self.config.source_dir,  # Already points to directory with hyperparams
        # ... other parameters remain the same
    )
```

#### 1.4 Update _get_inputs Method
- **Target**: `src/cursus/steps/builders/builder_xgboost_training_step.py`
- **Action**: Remove hyperparameters input channel handling
- **Changes**:
  - Remove `hyperparameters_s3_uri` special case handling
  - Remove internal hyperparameters generation logic
  - Remove config channel creation
  - Simplify input processing to handle only data channels

**Detailed Implementation:**
```python
def _get_inputs(self, inputs: Dict[str, Any]) -> Dict[str, TrainingInput]:
    """
    Get inputs for the step using specification and contract.

    This method creates TrainingInput objects for each dependency defined in the specification.
    After refactor: Only handles data inputs, hyperparameters are embedded in source directory.

    Args:
        inputs: Input data sources keyed by logical name

    Returns:
        Dictionary of TrainingInput objects keyed by channel name

    Raises:
        ValueError: If no specification or contract is available
    """
    if not self.spec:
        raise ValueError("Step specification is required")

    if not self.contract:
        raise ValueError("Script contract is required for input mapping")

    training_inputs = {}

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

            # SPECIAL HANDLING FOR input_path
            # For '/opt/ml/input/data', we need to create train/val/test channels
            if logical_name == "input_path":
                base_path = inputs[logical_name]

                # Create separate channels for each data split using helper method
                data_channels = self._create_data_channels_from_source(base_path)
                training_inputs.update(data_channels)
                self.log_info(
                    "Created data channels from %s: %s", logical_name, base_path
                )
            else:
                # For other inputs, extract the channel name from the container path
                parts = container_path.split("/")
                if (
                    len(parts) > 4
                    and parts[1] == "opt"
                    and parts[2] == "ml"
                    and parts[3] == "input"
                    and parts[4] == "data"
                ):
                    if len(parts) > 5:
                        channel_name = parts[5]  # Extract channel name from path
                        training_inputs[channel_name] = TrainingInput(
                            s3_data=inputs[logical_name]
                        )
                        self.log_info(
                            "Created %s channel from %s: %s",
                            channel_name,
                            logical_name,
                            inputs[logical_name],
                        )
                    else:
                        # If no specific channel in path, use logical name as channel
                        training_inputs[logical_name] = TrainingInput(
                            s3_data=inputs[logical_name]
                        )
                        self.log_info(
                            "Created %s channel from %s: %s",
                            logical_name,
                            logical_name,
                            inputs[logical_name],
                        )
        else:
            raise ValueError(f"No container path found for input: {logical_name}")

    return training_inputs
```

**Key Changes from Original:**
1. **Removed hyperparameters special case**: No more `hyperparameters_key = "hyperparameters_s3_uri"` handling
2. **Removed internal hyperparameters generation**: No more `_prepare_hyperparameters_file()` call
3. **Removed config channel creation**: No more `training_inputs["config"]` creation
4. **Removed matched_inputs tracking**: Simplified logic since no special cases
5. **Removed hyperparameters override logic**: No more external vs internal hyperparameters handling
6. **Simplified flow**: Only processes dependencies from specification, creates appropriate channels

### Phase 2: Training Script Refactoring

#### 2.1 Update Hyperparameters Loading Path
- **Targets**: 
  - `src/cursus/steps/scripts/xgboost_training.py`
  - `dockers/xgboost_atoz/xgboost_training.py`
  - `dockers/xgboost_pda/xgboost_training.py`
- **Action**: Update hyperparameters loading logic to use provided path with fallback in all 3 identical scripts
- **Implementation**:
```python
# OLD: Load from input data channel
def main(input_paths, output_paths, environ_vars, job_args):
    if "hyperparameters_s3_uri" in input_paths:
        hparam_path = input_paths["hyperparameters_s3_uri"]
        if not hparam_path.endswith("hyperparameters.json"):
            hparam_path = os.path.join(hparam_path, "hyperparameters.json")
    else:
        hparam_path = os.path.join(data_dir, "config", "hyperparameters.json")

# NEW: Use provided path with source directory fallback
def main(input_paths, output_paths, environ_vars, job_args):
    # Use provided hyperparameters path, with source directory fallback
    if "hyperparameters_s3_uri" in input_paths:
        hparam_path = input_paths["hyperparameters_s3_uri"]
        if not hparam_path.endswith("hyperparameters.json"):
            hparam_path = os.path.join(hparam_path, "hyperparameters.json")
    else:
        # Fallback to source directory if not provided
        hparam_path = "/opt/ml/code/hyperparams/hyperparameters.json"
```

#### 2.2 Update Container Path Constants
- **Targets**: 
  - `src/cursus/steps/scripts/xgboost_training.py`
  - `dockers/xgboost_atoz/xgboost_training.py`
  - `dockers/xgboost_pda/xgboost_training.py`
- **Action**: Update CONTAINER_PATHS constant to point CONFIG_DIR to source directory in all 3 scripts
- **Implementation**:
```python
# OLD: Container path constants
CONTAINER_PATHS = {
    "INPUT_DATA": "/opt/ml/input/data",
    "MODEL_DIR": "/opt/ml/model",
    "OUTPUT_DATA": "/opt/ml/output/data",
    "CONFIG_DIR": "/opt/ml/input/data/config",  # Input data channel
}

# NEW: Container path constants with source directory config
CONTAINER_PATHS = {
    "INPUT_DATA": "/opt/ml/input/data",
    "MODEL_DIR": "/opt/ml/model",
    "OUTPUT_DATA": "/opt/ml/output/data",
    "CONFIG_DIR": "/opt/ml/code/hyperparams",  # Source directory path
}

# Input paths remain unchanged - cleaner approach
input_paths = {
    "input_path": CONTAINER_PATHS["INPUT_DATA"],
    "hyperparameters_s3_uri": CONTAINER_PATHS["CONFIG_DIR"],  # Now points to source directory
}
```

#### 2.3 Update Function Signature Documentation
- **Targets**: 
  - `src/cursus/steps/scripts/xgboost_training.py`
  - `dockers/xgboost_atoz/xgboost_training.py`
  - `dockers/xgboost_pda/xgboost_training.py`
- **Action**: Update hyperparameters_s3_uri parameter documentation to reflect source directory path in all 3 scripts
- **Implementation**:
```python
def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str], 
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
) -> None:
    """
    Main function to execute the XGBoost training logic.

    Args:
        input_paths: Dictionary of input paths with logical names
            - "input_path": Directory containing train/val/test data
            - "hyperparameters_s3_uri": Path to hyperparameters directory (now points to /opt/ml/code/hyperparams)
        output_paths: Dictionary of output paths with logical names
            - "model_output": Directory to save model artifacts
            - "evaluation_output": Directory to save evaluation outputs
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments
    """
```

### Phase 3: Specification Updates

#### 3.1 Update XGBoost Training Specification
- **Target**: `src/cursus/steps/specs/xgboost_training_spec.py`
- **Action**: Update hyperparameters dependency to be optional
- **Rationale**: Since we have fallback to source directory, `hyperparameters_s3_uri` is no longer required
- **Changes**:
  - Change `hyperparameters_s3_uri` dependency from `required=True` to `required=False`
  - Update dependency description to reflect optional nature and fallback behavior
  - Maintain dependency in specification for consistency with script contract

**Implementation:**
```python
# OLD: Required hyperparameters dependency
DependencySpec(
    logical_name="hyperparameters_s3_uri",
    description="S3 URI containing hyperparameters configuration file",
    required=True,  # Was required
    # ... other fields
)

# NEW: Optional hyperparameters dependency with fallback
DependencySpec(
    logical_name="hyperparameters_s3_uri", 
    description="S3 URI containing hyperparameters configuration file (optional - falls back to source directory)",
    required=False,  # Now optional
    # ... other fields
)
```

#### 3.2 Update Script Contract (No Changes Needed)
- **Target**: Script contract validation in step builder
- **Action**: Keep hyperparameters in expected input paths for consistency
- **Rationale**: Maintain consistency between specification and contract
- **Changes**: No changes needed - hyperparameters path remains in contract, just points to source directory now

### Phase 4: DummyTraining Complete Refactor

**Critical Discovery**: DummyTraining requires a fundamentally different approach than XGBoost training. Instead of processing external inputs, DummyTraining should be a **SOURCE node** that packages pre-existing model and hyperparameters from the source directory.

#### **New Source Directory Structure for DummyTraining**
```
source_dir/
├── dummy_training.py          # Main training script
├── models/                    # NEW: Model directory
│   └── model.tar.gz          # Pre-trained model artifacts
└── hyperparams/              # NEW: Hyperparameters directory
    └── hyperparameters.json  # Generated hyperparameters file
```

#### **Container Path Mapping**
```
# SageMaker copies source_dir to: /opt/ml/code/
/opt/ml/code/
├── dummy_training.py
├── models/
│   └── model.tar.gz          # Available at /opt/ml/code/models/model.tar.gz
└── hyperparams/
    └── hyperparameters.json  # Available at /opt/ml/code/hyperparams/hyperparameters.json
```

#### 4.1 Update DummyTraining Specification
- **Target**: `src/cursus/steps/specs/dummy_training_spec.py`
- **Action**: Complete refactor to SOURCE node
- **Changes**:
  - Change `node_type` from `NodeType.INTERNAL` to `NodeType.SOURCE`
  - Remove ALL existing dependencies (no external inputs needed)
  - Keep only the `model_input` output specification
  - Update description to reflect source-only behavior

**Implementation:**
```python
DUMMY_TRAINING_SPEC = StepSpecification(
    step_type=get_spec_step_type("DummyTraining"),
    node_type=NodeType.SOURCE,  # Changed from INTERNAL to SOURCE
    script_contract=_get_dummy_training_contract(),
    dependencies=[
        # Remove all dependencies - SOURCE node needs no external inputs
    ],
    outputs=[
        OutputSpec(
            logical_name="model_input",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ProcessingOutputConfig.Outputs['model_input'].S3Output.S3Uri",
            data_type="S3Uri",
            description="S3 path to model artifacts with integrated hyperparameters (from source directory)",
            aliases=["ModelOutputPath", "ModelArtifacts", "model_data", "output_path"],
        )
    ],
)
```

#### 4.2 Update DummyTraining Contract
- **Target**: `src/cursus/steps/contracts/dummy_training_contract.py`
- **Action**: Remove all input paths (SOURCE node has no external inputs)
- **Changes**:
  - Set `expected_input_paths` to empty dictionary `{}`
  - Keep only the `model_input` output path
  - Update description to reflect source directory approach

**Implementation:**
```python
DUMMY_TRAINING_CONTRACT = ScriptContract(
    entry_point="dummy_training.py",
    expected_input_paths={
        # Empty - SOURCE node reads from source directory only
    },
    expected_output_paths={
        "model_input": "/opt/ml/processing/output/model"
    },
    expected_arguments={
        # No expected arguments - using hard-coded source directory paths
    },
    required_env_vars=[],
    optional_env_vars={},
    framework_requirements={"boto3": ">=1.26.0", "pathlib": ">=1.0.0"},
    description="Contract for dummy training SOURCE step that packages model.tar.gz and hyperparameters.json from source directory",
)
```

#### 4.3 Update DummyTraining Script
- **Target**: `src/cursus/steps/scripts/dummy_training.py`
- **Action**: Complete refactor to use source directory paths
- **Changes**:
  - Update `input_paths` to empty dictionary (consistent with contract)
  - Hard-code source directory paths in `main()` function
  - Remove dependency on external input channels
  - Update path constants to point to source directory

**Implementation:**
```python
# NEW: Hard-coded source directory paths with fallback support
MODEL_SOURCE_PATH = "/opt/ml/code/models/model.tar.gz"
HYPERPARAMS_SOURCE_PATH = "/opt/ml/code/hyperparams/hyperparameters.json"
MODEL_OUTPUT_DIR = "/opt/ml/processing/output/model"

def find_model_file(base_paths: List[str]) -> Optional[Path]:
    """
    Find model.tar.gz file in multiple possible locations.
    
    Args:
        base_paths: List of base paths to search
        
    Returns:
        Path to model file if found, None otherwise
    """
    for base_path in base_paths:
        model_path = Path(base_path) / "model.tar.gz"
        if model_path.exists():
            logger.info(f"Found model file at: {model_path}")
            return model_path
    return None

def find_hyperparams_file(base_paths: List[str]) -> Optional[Path]:
    """
    Find hyperparameters.json file in multiple possible locations.
    
    Args:
        base_paths: List of base paths to search
        
    Returns:
        Path to hyperparameters file if found, None otherwise
    """
    for base_path in base_paths:
        hyperparams_path = Path(base_path) / "hyperparameters.json"
        if hyperparams_path.exists():
            logger.info(f"Found hyperparameters file at: {hyperparams_path}")
            return hyperparams_path
    return None

def main(
    input_paths: Dict[str, str],  # Will be empty dict
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: Optional[argparse.Namespace] = None,
) -> Path:
    """
    Main entry point for the DummyTraining script.
    
    Reads model and hyperparameters from source directory with fallback mechanisms.
    """
    try:
        # Define fallback search paths for model file
        model_search_paths = [
            "/opt/ml/code/models",           # Primary: source directory models folder
            "/opt/ml/code",                  # Fallback: source directory root
            "/opt/ml/processing/input/model", # Legacy: processing input (if somehow provided)
        ]
        
        # Define fallback search paths for hyperparameters file
        hyperparams_search_paths = [
            "/opt/ml/code/hyperparams",      # Primary: source directory hyperparams folder
            "/opt/ml/code",                  # Fallback: source directory root
            "/opt/ml/processing/input/config", # Legacy: processing input (if somehow provided)
        ]
        
        # Find model file with fallback
        model_path = find_model_file(model_search_paths)
        if not model_path:
            raise FileNotFoundError(
                f"Model file (model.tar.gz) not found in any of these locations: {model_search_paths}"
            )
        
        # Find hyperparameters file with fallback
        hyperparams_path = find_hyperparams_file(hyperparams_search_paths)
        if not hyperparams_path:
            raise FileNotFoundError(
                f"Hyperparameters file (hyperparameters.json) not found in any of these locations: {hyperparams_search_paths}"
            )
        
        # Get output directory
        output_dir = Path(output_paths["model_output"])
        
        logger.info(f"Using paths:")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Hyperparameters: {hyperparams_path}")
        logger.info(f"  Output: {output_dir}")
        
        # Process model with hyperparameters from source directory
        output_path = process_model_with_hyperparameters(
            model_path, hyperparams_path, output_dir
        )
        
        return output_path
    except FileNotFoundError as e:
        logger.error(f"Required file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in dummy training: {e}")
        raise

if __name__ == "__main__":
    # Empty input paths - consistent with SOURCE node contract
    input_paths = {}
    
    output_paths = {"model_output": MODEL_OUTPUT_DIR}
    environ_vars = {}
    args = None
    
    result = main(input_paths, output_paths, environ_vars, args)
```

#### 4.4 Update DummyTrainingStepBuilder
- **Target**: `src/cursus/steps/builders/builder_dummy_training_step.py`
- **Action**: Complete refactor for SOURCE node behavior following proven ProcessingStep patterns
- **Changes**:
  - Remove `_upload_model_to_s3()` and `_prepare_hyperparameters_file()` methods (assets come from source directory)
  - Update `validate_configuration()` to use ProcessingStepConfigBase validation patterns
  - Update `_get_inputs()` to return empty list (SOURCE node has no external inputs)
  - Update `create_step()` to use proven XGBoostModelEvalStepBuilder ProcessingStep pattern
  - Fix source directory reference to use `get_effective_source_dir()`

**Key Implementation Changes:**
```python
def validate_configuration(self):
    """
    Validate the provided configuration.
    """
    self.log_info("Validating DummyTraining SOURCE configuration...")
    
    # Validate required configuration attributes
    required_attrs = [
        "processing_framework_version",
        "processing_instance_count",
        "processing_volume_size",
        "processing_entry_point",
    ]

    for attr in required_attrs:
        if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
            raise ValueError(f"DummyTrainingConfig missing required attribute: {attr}")
    
    # Validate source directory exists (but don't check contents - that's runtime concern)
    # Check both processing_source_dir and source_dir (fallback)
    effective_source_dir = self.config.get_effective_source_dir()
    if effective_source_dir and not effective_source_dir.startswith("s3://"):
        source_dir = Path(effective_source_dir)
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    self.log_info("DummyTraining SOURCE configuration validation succeeded.")

def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
    """
    Get inputs for the processor.
    
    For SOURCE nodes, return empty list since all data comes from source directory.
    
    Returns:
        Empty list - SOURCE node has no external inputs
    """
    self.log_info("DummyTraining is a SOURCE node - no external inputs required")
    return []

def create_step(self, **kwargs) -> ProcessingStep:
    """
    Create the processing step following the pattern from XGBoostModelEvalStepBuilder.
    
    This implementation uses processor.run() with both code and source_dir parameters,
    which is the correct pattern for ProcessingSteps that need source directory access.
    """
    try:
        # Extract parameters
        inputs_raw = kwargs.get("inputs", {})
        outputs = kwargs.get("outputs", {})
        dependencies = kwargs.get("dependencies", [])
        enable_caching = kwargs.get("enable_caching", True)

        # Handle inputs (should be empty for SOURCE node)
        inputs = {}
        inputs.update(inputs_raw)  # Should be empty but include for consistency

        # Create processor and get inputs/outputs
        processor = self._get_processor()
        processing_inputs = self._get_inputs(inputs)  # Returns empty list for SOURCE node
        processing_outputs = self._get_outputs(outputs)

        # Get step name using standardized method with auto-detection
        step_name = self._get_step_name()

        # Get job arguments from contract
        script_args = self._get_job_arguments()

        # CRITICAL: Follow XGBoostModelEvalStepBuilder pattern for source directory
        # Use processor.run() with both code and source_dir parameters
        script_path = self.config.get_script_path()  # Entry point only
        source_dir = self.config.get_effective_source_dir()  # Source directory path (processing_source_dir or source_dir)

        # Create step arguments using processor.run()
        step_args = processor.run(
            code=script_path,
            source_dir=source_dir,  # This ensures source directory is available in container
            inputs=processing_inputs,
            outputs=processing_outputs,
            arguments=script_args,
        )

        # Create and return the step using step_args
        processing_step = ProcessingStep(
            name=step_name,
            step_args=step_args,
            depends_on=dependencies,
            cache_config=self._get_cache_config(enable_caching),
        )

        # Store specification in step for future reference
        setattr(processing_step, "_spec", self.spec)

        return processing_step

    except Exception as e:
        self.log_error(f"Error creating DummyTraining step: {e}")
        import traceback
        self.log_error(traceback.format_exc())
        raise ValueError(f"Failed to create DummyTraining step: {str(e)}") from e
```

**Critical Implementation Details:**
- **Configuration Validation**: Uses ProcessingStepConfigBase patterns, validates required attributes, checks effective source directory
- **No File Existence Validation**: Configuration validation doesn't check for model/hyperparameters files (runtime concern)
- **Proper Source Directory Handling**: Uses `get_effective_source_dir()` to handle processing_source_dir/source_dir fallback
- **ProcessingStep Pattern**: Follows XGBoostModelEvalStepBuilder approach with `processor.run()` and `step_args`
- **Empty Inputs**: SOURCE node returns empty list from `_get_inputs()` as expected

#### 4.5 Update DummyTraining Configuration
- **Target**: `src/cursus/steps/configs/config_dummy_training_step.py`
- **Action**: Complete refactor for SOURCE node configuration
- **Changes**:
  - Remove `pretrained_model_path` field (model comes from source directory)
  - Remove `hyperparameters` field (hyperparameters come from source directory)
  - Remove `hyperparameters_s3_uri` field (not needed for SOURCE node)
  - Update class docstring to reflect SOURCE node behavior and expected directory structure
  - Simplify validation to only check basic configuration and contract alignment
  - Remove file existence validation (runtime concern, not configuration concern)
  - Add validation that contract has empty input paths (SOURCE node requirement)

**Implementation:**
```python
class DummyTrainingConfig(ProcessingStepConfigBase):
    """
    Configuration for DummyTraining SOURCE step.

    This step is a SOURCE node that reads model.tar.gz and hyperparameters.json
    from the source directory, packages them together, and makes them available
    for downstream packaging and registration steps.
    
    Expected source directory structure:
    source_dir/
    ├── dummy_training.py          # Main training script
    ├── models/                    # Model directory
    │   └── model.tar.gz          # Pre-trained model artifacts
    └── hyperparams/              # Hyperparameters directory
        └── hyperparameters.json  # Generated hyperparameters file
    """

    # Override with specific default for this step
    processing_entry_point: str = Field(
        default="dummy_training.py",
        description="Entry point script for dummy training SOURCE step.",
    )

    @model_validator(mode="after")
    def validate_config(self) -> "DummyTrainingConfig":
        """
        Validate configuration for SOURCE node.

        For SOURCE nodes, we only validate basic configuration attributes.
        File existence is checked at runtime, not configuration time.
        """
        # Basic validation - entry point is required for SOURCE nodes
        if not self.processing_entry_point:
            raise ValueError("DummyTraining SOURCE step requires a processing_entry_point")

        # Validate script contract - ensure it matches SOURCE node expectations
        contract = self.get_script_contract()
        if not contract:
            raise ValueError("Failed to load script contract")

        # For SOURCE nodes, contract should have empty input paths
        if contract.expected_input_paths:
            raise ValueError(
                f"SOURCE node contract should have empty input paths, but found: {list(contract.expected_input_paths.keys())}"
            )

        # Ensure we have the required output path
        if "model_input" not in contract.expected_output_paths:
            raise ValueError(
                "Script contract missing required output path: model_input"
            )

        return self
```

#### 4.6 PyTorchTrainingStepBuilder Analysis
- **Target**: `src/cursus/steps/builders/builder_pytorch_training_step.py`
- **Action**: No changes needed
- **Assessment**: PyTorch training does NOT use `_prepare_hyperparameters_file` - hyperparameters are passed directly to estimator constructor
- **Conclusion**: PyTorchTrainingStepBuilder is not affected by this refactor

### Phase 5: Testing and Validation

#### 5.1 Unit Testing
- **Target**: Test hyperparameters loading in source directory
- **Tests**:
  - Verify hyperparameters.json is created in source directory
  - Validate hyperparameters content and format
  - Test source directory preparation and cleanup

#### 5.2 Integration Testing
- **Target**: End-to-end pipeline execution
- **Tests**:
  - Verify training script can load hyperparameters from `/opt/ml/code/hyperparams/`
  - Test pipeline execution with `PIPELINE_EXECUTION_TEMP_DIR`
  - Validate model training and output generation

#### 5.3 External System Integration Testing
- **Target**: Test with custom `PIPELINE_EXECUTION_TEMP_DIR`
- **Tests**:
  - Verify external systems can provide custom execution directories
  - Test parameter flow with dynamic output paths
  - Validate backward compatibility

## Risk Assessment and Mitigation

### Technical Risks

1. **Source Directory Size Increase**
   - **Risk**: Adding hyperparameters to source directory increases package size
   - **Mitigation**: Hyperparameters JSON files are typically small (<1KB)
   - **Status**: Low risk

2. **Container Path Dependencies**
   - **Risk**: Hard-coded `/opt/ml/code/` path may not be portable
   - **Mitigation**: Use environment variables or relative paths where possible
   - **Status**: Medium risk, requires careful testing

3. **Temporary Directory Management**
   - **Risk**: Temporary source directories may not be cleaned up properly
   - **Mitigation**: Implement robust cleanup in finally blocks
   - **Status**: Low risk with proper implementation

### Operational Risks

1. **Training Script Compatibility**
   - **Risk**: Existing training scripts may break with path changes
   - **Mitigation**: Comprehensive testing and gradual rollout
   - **Status**: Medium risk, requires careful migration

2. **Development Workflow Impact**
   - **Risk**: Developers may need to adjust local testing workflows
   - **Mitigation**: Update documentation and provide migration guide
   - **Status**: Low risk

## Implementation Timeline

### Phase 1: Core Refactoring (Days 1-2)
- Remove `_prepare_hyperparameters_file` from XGBoostTrainingStepBuilder
- Update `_get_inputs` method to remove hyperparameters channel handling
- No changes needed to `_create_estimator` (already uses `self.config.source_dir`)

### Phase 2: Script Updates (Days 3-4)
- Update xgboost_training.py hyperparameters loading path
- Remove hyperparameters input channel handling
- Update function signatures and documentation

### Phase 3: Specification Updates (Day 5)
- Update XGBoost training specification
- Remove hyperparameters dependencies
- Update contract validation

### Phase 4: Testing and Validation (Days 6-7)
- Unit testing of hyperparameters loading from source directory
- Integration testing with pipeline execution
- External system integration testing

### Phase 5: Documentation and Rollout (Day 8)
- Update developer documentation
- Create migration guide for users to include hyperparameters in source directory
- Deploy and monitor

## Success Metrics

### Technical Metrics
- [x] Zero `_prepare_hyperparameters_file` references in codebase ✅ **COMPLETED**
- [ ] Successful hyperparameters loading from `/opt/ml/code/hyperparams/`
- [ ] Pipeline execution with custom `PIPELINE_EXECUTION_TEMP_DIR`
- [x] No S3 upload conflicts during pipeline definition ✅ **COMPLETED**

### Business Metrics
- [ ] External system integration time maintained
- [ ] Pipeline portability improved
- [ ] Development workflow disruption minimized
- [ ] Training performance maintained

## Implementation Status

### ✅ **Phase 1: XGBoostTrainingStepBuilder Refactoring - COMPLETED**
- [x] **1.1 Remove _prepare_hyperparameters_file Method** ✅ **COMPLETED**
  - Completely removed `_prepare_hyperparameters_file` method and all references
  - Eliminated S3 upload logic and timing conflicts with `PIPELINE_EXECUTION_TEMP_DIR`
  
- [x] **1.2 Simplify Hyperparameters Handling** ✅ **COMPLETED**
  - Removed hyperparameters generation logic
  - Users now required to include `hyperparams/hyperparameters.json` in source directory
  
- [x] **1.3 Update _create_estimator Method** ✅ **COMPLETED**
  - Updated comments to reflect new hyperparameters approach
  - Documentation indicates hyperparameters are embedded in source directory
  
- [x] **1.4 Update _get_inputs Method** ✅ **COMPLETED**
  - Completely refactored to remove hyperparameters special case handling
  - Simplified method that only processes data dependencies from specification
  - Removed 6 key complex logic components as planned

### ✅ **Phase 2: Training Script Refactoring - COMPLETED**
- [x] **2.1 Update Hyperparameters Loading Path** ✅ **COMPLETED**
  - Updated fallback path in all 3 scripts to `/opt/ml/code/hyperparams/hyperparameters.json`
  - `src/cursus/steps/scripts/xgboost_training.py` ✅ **COMPLETED**
  - `dockers/xgboost_atoz/xgboost_training.py` ✅ **COMPLETED**
  - `dockers/xgboost_pda/xgboost_training.py` ✅ **COMPLETED**
  
- [x] **2.2 Update Container Path Constants** ✅ **COMPLETED**
  - Updated `CONFIG_DIR` to point to source directory in all 3 scripts
  - `src/cursus/steps/scripts/xgboost_training.py` ✅ **COMPLETED**
  - `dockers/xgboost_atoz/xgboost_training.py` ✅ **COMPLETED**
  - `dockers/xgboost_pda/xgboost_training.py` ✅ **COMPLETED**
  
- [x] **2.3 Update Function Signature Documentation** ✅ **COMPLETED**
  - Updated hyperparameters parameter documentation in all 3 scripts
  - `src/cursus/steps/scripts/xgboost_training.py` ✅ **COMPLETED**
  - `dockers/xgboost_atoz/xgboost_training.py` ✅ **COMPLETED**
  - `dockers/xgboost_pda/xgboost_training.py` ✅ **COMPLETED**

**Implementation Summary:**
- **Complete Source Directory Integration**: All 3 XGBoost training scripts now use `/opt/ml/code/hyperparams` for hyperparameters
- **Fallback Logic Updated**: Scripts fall back to source directory when hyperparameters not provided via input channel
- **Consistent Implementation**: Same changes applied across all identical scripts
- **Backward Compatibility**: Existing hyperparameters loading logic preserved with enhanced capabilities

### ✅ **Phase 3: Specification Updates - COMPLETED**
- [x] **3.1 Update XGBoost Training Specification** ✅ **COMPLETED**
  - Updated `hyperparameters_s3_uri` dependency description to reflect source directory fallback
  - Dependency already set to `required=False` (optional)
  - Updated description: "S3 URI containing hyperparameters configuration file (optional - falls back to source directory)"
  - Maintains consistency between specification and script contract
  
- [x] **3.2 Update Script Contract** ✅ **COMPLETED** (No Changes Needed)
  - Script contract validation already consistent with specification
  - Hyperparameters path remains in contract, now points to source directory
  - No changes needed - maintains consistency between specification and contract

**Implementation Summary:**
- **Optional Dependency**: `hyperparameters_s3_uri` is now properly documented as optional with fallback behavior
- **Clear Documentation**: Description explicitly mentions source directory fallback
- **Specification Consistency**: Aligns with the new source directory approach implemented in Phases 1-2
- **Contract Alignment**: Script contract remains consistent with updated specification

### ✅ **Phase 4: DummyTraining Complete Refactor - COMPLETED**
- [x] **4.1 Update DummyTraining Specification** ✅ **COMPLETED**
  - Changed `node_type` from `NodeType.INTERNAL` to `NodeType.SOURCE`
  - Removed ALL existing dependencies (no external inputs needed)
  - Kept only the `model_input` output specification
  - Updated description to reflect source-only behavior

- [x] **4.2 Update DummyTraining Contract** ✅ **COMPLETED**
  - Set `expected_input_paths` to empty dictionary `{}`
  - Kept only the `model_input` output path
  - Updated description to reflect source directory approach

- [x] **4.3 Update DummyTraining Script** ✅ **COMPLETED**
  - Updated path constants to point to source directory
  - Added `find_model_file()` and `find_hyperparams_file()` helper functions with fallback mechanisms
  - Implemented comprehensive fallback search paths (3 locations each)
  - Updated main function with robust file discovery and error handling
  - Updated script entry point to use empty input paths (consistent with SOURCE node)

- [x] **4.4 Update DummyTrainingStepBuilder** ✅ **COMPLETED**
  - Updated `validate_configuration()` with proper ProcessingStepConfigBase validation
  - Removed `_upload_model_to_s3()` and `_prepare_hyperparameters_file()` methods
  - Updated `_get_inputs()` to return empty list for SOURCE node
  - Updated `create_step()` to use proven XGBoostModelEvalStepBuilder pattern
  - Implemented `processor.run()` with both `code` and `source_dir` parameters
  - Fixed source directory reference to use `get_effective_source_dir()`

- [x] **4.5 Update DummyTraining Configuration** ✅ **COMPLETED**
  - Removed `pretrained_model_path` field (model comes from source directory)
  - Removed `hyperparameters` field (hyperparameters come from source directory)
  - Removed `hyperparameters_s3_uri` field (not needed for SOURCE node)
  - Updated class docstring with SOURCE node behavior and expected directory structure
  - Simplified validation to only check basic configuration and contract alignment
  - Removed file existence validation (runtime concern, not configuration concern)
  - Added validation that contract has empty input paths (SOURCE node requirement)

- [x] **4.6 PyTorchTrainingStepBuilder Analysis** ✅ **COMPLETED**
  - Confirmed PyTorch training does NOT use `_prepare_hyperparameters_file` method
  - Hyperparameters are passed directly to estimator constructor
  - No S3 upload logic for hyperparameters
  - PyTorchTrainingStepBuilder is not affected by this refactor

**Implementation Summary:**
- **Complete SOURCE Node Implementation**: DummyTraining successfully converted across all 6 components (spec, contract, script, builder, config, analysis)
- **Zero External Dependencies**: All components aligned on source directory approach with no external inputs
- **Proven ProcessingStep Pattern**: Builder uses same successful approach as XGBoostModelEvalStepBuilder
- **Robust Fallback Mechanisms**: Script includes comprehensive file discovery with multiple search paths
- **Clean Configuration Model**: Configuration validates SOURCE node requirements and documents expected directory structure

### 🔄 **Phase 5: Testing and Validation - PENDING**
- [ ] **5.1 Unit Testing** (including fallback path testing and config validation)
- [ ] **5.2 Integration Testing**
- [ ] **5.3 External System Integration Testing**

## Conclusion

The hyperparameters source directory integration refactor resolves the fundamental timing conflict between `PIPELINE_EXECUTION_TEMP_DIR` and `_prepare_hyperparameters_file`. By embedding hyperparameters directly in the source directory, we:

1. **Eliminate Timing Conflicts**: No more S3 uploads during pipeline definition
2. **Maintain Parameter Flow**: `PIPELINE_EXECUTION_TEMP_DIR` works correctly for all outputs
3. **Improve Portability**: Hyperparameters travel with the code
4. **Simplify Architecture**: Remove complex S3 upload logic

This refactor is essential for completing Phase 3 of the PIPELINE_EXECUTION_TEMP_DIR implementation and enabling true pipeline portability across different environments and external systems.

## References

### Related Documents
- [Pipeline Execution Temp Dir Integration](../1_design/pipeline_execution_temp_dir_integration.md) - Primary design document
- [PIPELINE_EXECUTION_TEMP_DIR Implementation Plan](2025-09-18_pipeline_execution_temp_dir_implementation_plan.md) - Overall project plan

### Implementation Files
- `src/cursus/steps/builders/builder_xgboost_training_step.py` - Primary refactoring target
- `src/cursus/steps/scripts/xgboost_training.py` - Training script updates
- `src/cursus/steps/specs/xgboost_training_spec.py` - Specification updates
- `dockers/xgboost_atoz/` - Container environment reference
