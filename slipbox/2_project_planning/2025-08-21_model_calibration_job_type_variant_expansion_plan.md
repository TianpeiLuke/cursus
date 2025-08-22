---
tags:
  - project
  - planning
  - job_type_variants
  - model_calibration
keywords:
  - job type
  - variant handling
  - model calibration
  - specification
  - pipeline
  - builder pattern
topics:
  - variant specifications
  - dependency resolution
  - model calibration enhancement
language: python
date of note: 2025-08-21
---

# Model Calibration Job Type Variant Expansion Plan

**Created**: August 21, 2025  
**Status**: 🚧 IN PROGRESS - Phase 5 Complete  
**Priority**: Medium  
**Timeline**: Estimated 3-5 days  
**Related**: [Job Type Variant Solution](./2025-07-04_job_type_variant_solution.md)

## Context

The `ModelCalibration` step currently does not support job type variants, while other processing steps like `TabularPreprocessing`, `RiskTableMapping`, `CurrencyConversion`, and most recently `XGBoostModelEval` do. This inconsistency creates challenges when constructing pipelines that require calibrating models with different data splits.

Since XGBoostModelEval already supports job_type variants (training, calibration, validation, testing), adding this capability to ModelCalibration would allow for more consistent pipeline construction and improved dependency resolution.

## Problem

Current limitations with the ModelCalibration step:

1. No job_type differentiation unlike other processing steps
2. Cannot clearly specify which data split the calibration is operating on
3. Pipeline templates must use hardcoded connections rather than leveraging semantic matching
4. Dependency resolver cannot match between job type variants of `XGBoostModelEval` and `ModelCalibration`

## Solution: Expand ModelCalibration with Job Type Variants

Following the established pattern in TabularPreprocessing, RiskTableMapping, and CurrencyConversion, we will expand the ModelCalibration step to support job_type variants.

### Implementation Plan

#### Phase 1: Update ModelCalibrationConfig (1 day)

1. Add `job_type` field to `ModelCalibrationConfig` class:

```python
job_type: str = Field(
    default="calibration",
    description="Which data split to use for calibration (e.g., 'training', 'calibration', 'validation', 'test')."
)
```

2. Add validation for job_type values:

```python
@model_validator(mode='after')
def validate_config(self) -> 'ModelCalibrationConfig':
    # Existing validation code...
    
    # Validate job_type
    valid_job_types = {"training", "calibration", "validation", "testing"}
    if self.job_type not in valid_job_types:
        raise ValueError(f"job_type must be one of {valid_job_types}, got '{self.job_type}'")
    
    # Rest of validation...
    
    return self
```

3. No need to update environment variables for JOB_TYPE, as it's passed through command line arguments instead (following the established pattern in other steps like TabularPreprocessing and RiskTableMapping).

4. Add job_type to `get_public_init_fields` (no need to add it to from_hyperparameters):

```python
def get_public_init_fields(self) -> Dict[str, Any]:
    # Get fields from parent class
    base_fields = super().get_public_init_fields()
    
    # Add calibration-specific fields
    calibration_fields = {
        # Existing fields...
        'job_type': self.job_type,
        # Rest of fields...
    }
    
    # Combine and return
    return {**base_fields, **calibration_fields}
```

#### Phase 2: Create Job Type-Specific Spec Files (1-2 days)

Create four new specification files following the pattern used in TabularPreprocessing:

1. `src/cursus/steps/specs/model_calibration_training_spec.py`:

```python
"""
Model Calibration Training Step Specification.

This module defines the declarative specification for model calibration steps
specifically for training data, including dependencies and outputs.
"""

from ...core.base.specification_base import StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType
from ..registry.step_names import get_spec_step_type_with_job_type

# Import the contract at runtime to avoid circular imports
def _get_model_calibration_contract():
    from ..contracts.model_calibration_contract import MODEL_CALIBRATION_CONTRACT
    return MODEL_CALIBRATION_CONTRACT

# Model Calibration Training Step Specification
MODEL_CALIBRATION_TRAINING_SPEC = StepSpecification(
    step_type=get_spec_step_type_with_job_type("ModelCalibration", "training"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_model_calibration_contract(),
    dependencies={
        "evaluation_data": DependencySpec(
            logical_name="evaluation_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["XGBoostTraining", "XGBoostModelEval", "ModelEvaluation", "TrainingEvaluation", "CrossValidation"],
            semantic_keywords=["training", "train", "evaluation", "predictions", "scores", "results", "model_output", "performance"],
            data_type="S3Uri",
            description="Training evaluation dataset with ground truth labels and model predictions"
        )
    },
    outputs={
        # Same outputs with training-specific keywords
        "calibration_output": OutputSpec(
            logical_name="calibration_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['calibration_output'].S3Output.S3Uri",
            aliases=["calibration_model", "training_calibration", "train_calibrator", "training_probability_calibration"],
            data_type="S3Uri",
            description="Training calibration mapping and artifacts"
        ),
        # Other outputs similar to original but with training-specific keywords...
    }
)
```

2. Similar files for calibration, validation, and testing variants:
   - `model_calibration_calibration_spec.py`
   - `model_calibration_validation_spec.py`
   - `model_calibration_testing_spec.py`

Each with appropriate semantic keywords for their respective job types.

#### Phase 3: Update ModelCalibrationStepBuilder (1 day)

Update the builder to dynamically select the appropriate specification based on job_type:

```python
# Import specifications based on job type
try:
    from ..specs.model_calibration_training_spec import MODEL_CALIBRATION_TRAINING_SPEC
    from ..specs.model_calibration_calibration_spec import MODEL_CALIBRATION_CALIBRATION_SPEC
    from ..specs.model_calibration_validation_spec import MODEL_CALIBRATION_VALIDATION_SPEC
    from ..specs.model_calibration_testing_spec import MODEL_CALIBRATION_TESTING_SPEC
    SPECS_AVAILABLE = True
except ImportError:
    MODEL_CALIBRATION_TRAINING_SPEC = MODEL_CALIBRATION_CALIBRATION_SPEC = MODEL_CALIBRATION_VALIDATION_SPEC = MODEL_CALIBRATION_TESTING_SPEC = None
    SPECS_AVAILABLE = False
```

Update the `__init__` method:

```python
def __init__(
    self, 
    config, 
    sagemaker_session=None, 
    role=None, 
    notebook_root=None,
    registry_manager=None,
    dependency_resolver=None
):
    """Initialize with specification based on job type."""
    if not isinstance(config, ModelCalibrationConfig):
        raise ValueError("ModelCalibrationStepBuilder requires a ModelCalibrationConfig instance.")
        
    # Get the appropriate spec based on job type
    spec = None
    if not hasattr(config, 'job_type'):
        raise ValueError("config.job_type must be specified")
        
    job_type = config.job_type.lower()
    
    # Get specification based on job type
    if job_type == "training" and MODEL_CALIBRATION_TRAINING_SPEC is not None:
        spec = MODEL_CALIBRATION_TRAINING_SPEC
    elif job_type == "calibration" and MODEL_CALIBRATION_CALIBRATION_SPEC is not None:
        spec = MODEL_CALIBRATION_CALIBRATION_SPEC
    elif job_type == "validation" and MODEL_CALIBRATION_VALIDATION_SPEC is not None:
        spec = MODEL_CALIBRATION_VALIDATION_SPEC
    elif job_type == "testing" and MODEL_CALIBRATION_TESTING_SPEC is not None:
        spec = MODEL_CALIBRATION_TESTING_SPEC
    else:
        # Try dynamic import
        try:
            module_path = f"..specs.model_calibration_{job_type}_spec"
            module = importlib.import_module(module_path, package=__package__)
            spec_var_name = f"MODEL_CALIBRATION_{job_type.upper()}_SPEC"
            if hasattr(module, spec_var_name):
                spec = getattr(module, spec_var_name)
        except (ImportError, AttributeError):
            self.log_warning("Could not import specification for job type: %s", job_type)
            
    if not spec:
        raise ValueError(f"No specification found for job type: {job_type}")
            
    self.log_info("Using specification for %s", job_type)
    
    super().__init__(
        config=config,
        spec=spec,
        sagemaker_session=sagemaker_session,
        role=role,
        notebook_root=notebook_root,
        registry_manager=registry_manager,
        dependency_resolver=dependency_resolver
    )
    self.config: ModelCalibrationConfig = config
```

Update validation method:

```python
def validate_configuration(self) -> None:
    """Validate the provided configuration."""
    self.log_info("Validating ModelCalibrationConfig...")
    
    # Validate required attributes
    required_attrs = [
        'processing_entry_point',
        'processing_source_dir',
        'processing_instance_count',
        'processing_volume_size',
        'calibration_method',
        'label_field',
        'score_field',
        'is_binary',
        'job_type'  # Add job_type to required attributes
    ]
    
    for attr in required_attrs:
        if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
            raise ValueError(f"ModelCalibrationConfig missing required attribute: {attr}")
    
    # Validate job_type
    valid_job_types = {"training", "calibration", "validation", "testing"}
    if self.config.job_type not in valid_job_types:
        raise ValueError(f"Invalid job_type: {self.config.job_type}")
    
    # Rest of validation...
```

Update destination path in `_get_outputs`:

```python
# Generate destination from config
destination = f"{self.config.pipeline_s3_loc}/model_calibration/{self.config.job_type}/{logical_name}"
```

#### Phase 4: Update _get_job_arguments Method (0.5 day)

Ensure the job_type is passed as an argument to the script, following the pattern used in TabularPreprocessing and RiskTableMapping steps:

```python
def _get_job_arguments(self) -> List[str]:
    """
    Constructs the list of command-line arguments to be passed to the processing script.
    
    This implementation uses job_type from the configuration.
    
    Returns:
        A list of strings representing the command-line arguments.
    """
    # Get job_type from configuration
    job_type = self.config.job_type
    self.log_info("Setting job_type argument to: %s", job_type)
    
    # Return job_type argument
    return ["--job_type", job_type]
```

#### Phase 5: Update Model Calibration Script (1 day)

The model_calibration.py script needs to be updated to handle different data formats based on job_type:

1. Add command-line argument parsing for job_type:

```python
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_type", type=str, required=True,
                        help="Job type - one of: training, calibration, validation, testing")
    args = parser.parse_args()
    
    # Create config with environment variables
    config = CalibrationConfig.from_env()
    
    # Call the main function
    try:
        main(config)
        logger.info("Calibration completed successfully")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Calibration failed: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
```

2. Add extract_and_load_nested_tarball_data function from PDA implementation:

```python
def extract_and_load_nested_tarball_data(config=None):
    """
    Extract and load data from nested tar.gz files in SageMaker output structure.
    
    Handles SageMaker's specific output structure:
    - output.tar.gz (outer archive)
      - val.tar.gz (inner archive)
        - val/predictions.csv (actual data)
        - val_metrics/... (metrics and plots)
      - test.tar.gz (inner archive)
        - test/predictions.csv (actual data)
        - test_metrics/... (metrics and plots)
    
    Also handles cases where the input path contains:
    - Direct output.tar.gz file
    - Path to a job directory that contains output/output.tar.gz
    - Path to a parent directory with job subdirectories
    
    Args:
        config: Configuration object (optional, created from environment if not provided)
        
    Returns:
        pd.DataFrame: Combined dataset with predictions from extracted tar.gz files
        
    Raises:
        FileNotFoundError: If necessary tar.gz files or prediction data not found
    """
    config = config or CalibrationConfig.from_env()
    input_dir = config.input_data_path
    log_section("NESTED TARBALL EXTRACTION")
    logger.info(f"Looking for SageMaker output archive in {input_dir}")
    
    # Check if we have a direct data file first (non-tarball case)
    try:
        direct_file = find_first_data_file(input_dir)
        if direct_file:
            logger.info(f"Found direct data file: {direct_file}, using standard loading")
            return load_data(config)
    except FileNotFoundError:
        # No direct data file, continue with tarball extraction
        pass
    
    # First check: Direct tarball in the input directory
    output_archive = None
    for fname in os.listdir(input_dir):
        if fname.lower() == "output.tar.gz":
            output_archive = os.path.join(input_dir, fname)
            logger.info(f"Found output.tar.gz directly in input directory")
            break
    
    # Second check: Look for job-specific directories containing output/output.tar.gz
    if not output_archive:
        logger.info("No output.tar.gz found directly in input directory, checking for job directories")
        for item in os.listdir(input_dir):
            item_path = os.path.join(input_dir, item)
            if os.path.isdir(item_path):
                # Check if this directory has an output/output.tar.gz file
                output_dir = os.path.join(item_path, "output")
                if os.path.isdir(output_dir):
                    nested_archive = os.path.join(output_dir, "output.tar.gz")
                    if os.path.isfile(nested_archive):
                        output_archive = nested_archive
                        logger.info(f"Found nested output.tar.gz at {output_archive}")
                        break
    
    # Third check: Recursive search for output.tar.gz (most robust but potentially slower)
    if not output_archive:
        logger.info("No output.tar.gz found in expected locations, performing recursive search")
        for root, _, files in os.walk(input_dir):
            for fname in files:
                if fname.lower() == "output.tar.gz":
                    output_archive = os.path.join(root, fname)
                    logger.info(f"Found output.tar.gz from recursive search at {output_archive}")
                    break
            if output_archive:
                break
    
    # If we still don't have it, fall back to standard data loading
    if not output_archive:
        logger.warning("No output.tar.gz found anywhere, falling back to standard data loading")
        return load_data(config)
    
    # (Rest of the extraction function as in the dockers/xgboost_pda/scripts/model_calibration.py implementation)
```

3. Update load_and_prepare_data function to use job_type from command line:

```python
def load_and_prepare_data(config=None, job_type="calibration"):
    """Load evaluation data and prepare it for calibration based on classification type.
    
    Args:
        config: Configuration object (optional, created from environment if not provided)
        job_type: The job type to determine how to load data
        
    Returns:
        tuple: Different return values based on classification type:
            - Binary: (df, y_true, y_prob, None)
            - Multi-class: (df, y_true, None, y_prob_matrix)
        
    Raises:
        FileNotFoundError: If no data file is found
        ValueError: If required columns are missing
    """
    config = config or CalibrationConfig.from_env()
    
    log_section("DATA PREPARATION")
    
    # Load data differently based on job_type
    if job_type == "training":
        # Training job outputs are nested tarballs from XGBoostTraining output
        logger.info(f"Loading data for job_type=training using nested tarball extraction")
        try:
            df = extract_and_load_nested_tarball_data(config)
        except Exception as e:
            logger.warning(f"Failed to extract data from nested tarballs: {e}")
            logger.warning(f"Exception details: {traceback.format_exc()}")
            logger.info("Falling back to standard data loading")
            df = load_data(config)
    else:
        # Calibration, validation, and testing job outputs are direct files from XGBoostModelEval
        logger.info(f"Loading data for job_type={job_type} using standard loading")
        df = load_data(config)
    
    if config.is_binary:
        # Binary case - single score field
        y_true = df[config.label_field].values
        y_prob = df[config.score_field].values
        return df, y_true, y_prob, None
    else:
        # Multi-class case - multiple probability columns
        y_true = df[config.label_field].values
        
        # Get all probability columns
        prob_columns = []
        for i in range(config.num_classes):
            class_name = config.multiclass_categories[i]
            col_name = f"{config.score_field_prefix}{class_name}"
            if col_name not in df.columns:
                # Try numeric index as fallback
                col_name = f"{config.score_field_prefix}{i}"
                if col_name not in df.columns:
                    raise ValueError(f"Could not find probability column for class {class_name}")
            prob_columns.append(col_name)
        
        logger.info(f"Found probability columns for multi-class: {prob_columns}")
        
        # Extract probability matrix (samples × classes)
        y_prob_matrix = df[prob_columns].values
        
        return df, y_true, None, y_prob_matrix
```

4. Update main() function to accept and use job_type:

```python
def main(config=None):
    """Main entry point for the calibration script."""
    try:
        # Use provided config or create from environment
        config = config or CalibrationConfig.from_env()
        logger.info("Starting model calibration")
        logger.info(f"Running in {'binary' if config.is_binary else 'multi-class'} mode with job_type={args.job_type}")
        
        # Create output directories
        create_directories(config)
        
        if config.is_binary:
            # Binary classification workflow
            # Load data and extract features and target based on job_type
            df, y_true, y_prob_uncalibrated, _ = load_and_prepare_data(config, args.job_type)
            
            # Rest of the code remains the same...
```

#### Phase 6: Testing and Integration (1-2 days)

1. Create unit tests for `ModelCalibrationConfig` with different job_type values
2. Test the `ModelCalibrationStepBuilder` with various job_type configurations
3. Create integration tests connecting `XGBoostModelEval` with job_type to corresponding `ModelCalibration`
4. Update example pipeline templates to leverage job_type variants
5. Test the model_calibration.py script with different job_type values and input formats

## Comparison with Existing Job Type Variant Steps

### 1. TabularPreprocessing

TabularPreprocessing already implements job_type variants with:

1. Separate specification files for each job type:
   - `tabular_preprocessing_training_spec.py`
   - `tabular_preprocessing_calibration_spec.py` 
   - `tabular_preprocessing_validation_spec.py`
   - `tabular_preprocessing_testing_spec.py`

2. Dynamic specification selection in builder:
   - Imports all variant specs
   - Selects appropriate spec based on config.job_type
   - Falls back to dynamic import for custom job types

3. Job_type passed to script via command-line arguments

### 2. CurrencyConversion and RiskTableMapping

Similar to TabularPreprocessing, these steps have:

1. Job_type field in config with validation
2. Separate specification files for each job type
3. Dynamic selection in builder
4. Output paths include job_type for separation

## Benefits

1. **Consistent Architecture**: Align ModelCalibration with other processing steps
2. **Improved Dependency Resolution**: Enable proper matching between XGBoostModelEval and ModelCalibration variants
3. **Clear Pipeline Structure**: Each data split's flow is properly separated
4. **Semantic Matching**: Leverage automatic dependency resolution with job type awareness
5. **Enhanced Reusability**: Make ModelCalibration more flexible for various pipeline configurations

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing pipelines | Maintain backward compatibility by defaulting to "calibration" job_type |
| Script implementation may not handle job_type | Update script implementation or handle job_type in step builder |
| Inconsistent semantic matching | Use consistent semantic keywords across all job type variants |
| Over-engineering for simple step | Follow established patterns to maintain consistency |

## Success Criteria

- [x] All job type variant specifications created (training, calibration, validation, testing)
- [x] ModelCalibrationConfig properly validates job_type
- [x] ModelCalibrationStepBuilder selects correct specification based on job_type
- [x] Job_type propagated to script via command-line arguments (following the pattern used in other steps)
- [x] Model calibration script properly handles different input data formats based on job_type
- [ ] XGBoostModelEval with specific job_type can properly connect to corresponding ModelCalibration
- [ ] >95% test coverage for new code

## Phase 1 Completion Status (✅ COMPLETED)

**Completed Items:**
- [x] Added `job_type` field to `ModelCalibrationConfig` with default "calibration"
- [x] Added validation for job_type values: {"training", "calibration", "validation", "testing"}
- [x] Updated `get_public_init_fields()` to include job_type
- [x] Fixed processing_source_dir path to align with actual script location
- [x] Standardized job_type values across all step configs (updated XGBoostModelEval)
- [x] Maintained backward compatibility with default job_type="calibration"

**Files Modified:**
- `src/cursus/steps/configs/config_model_calibration_step.py` - Added job_type support
- `src/cursus/steps/configs/config_xgboost_model_eval_step.py` - Standardized to use "testing"

## Phase 2 Completion Status (✅ COMPLETED)

**Completed Items:**
- [x] Created `model_calibration_training_spec.py` with training-specific semantic keywords
- [x] Created `model_calibration_calibration_spec.py` with calibration-specific semantic keywords
- [x] Created `model_calibration_validation_spec.py` with validation-specific semantic keywords
- [x] Created `model_calibration_testing_spec.py` with testing-specific semantic keywords
- [x] All specifications follow the established pattern with proper imports and structure
- [x] Each specification includes job_type-specific aliases and semantic keywords for dependency resolution

**Files Created:**
- `src/cursus/steps/specs/model_calibration_training_spec.py` - Training job type specification
- `src/cursus/steps/specs/model_calibration_calibration_spec.py` - Calibration job type specification
- `src/cursus/steps/specs/model_calibration_validation_spec.py` - Validation job type specification
- `src/cursus/steps/specs/model_calibration_testing_spec.py` - Testing job type specification

## Phase 3 Completion Status (✅ COMPLETED)

**Completed Items:**
- [x] Updated ModelCalibrationStepBuilder to import all job type-specific specifications
- [x] Added dynamic specification selection based on config.job_type in __init__ method
- [x] Added job_type validation in validate_configuration method
- [x] Updated output path generation to include job_type for proper separation
- [x] Updated _get_job_arguments to pass job_type as command-line argument to script
- [x] Added fallback dynamic import mechanism for custom job types
- [x] Maintained backward compatibility and error handling

**Files Modified:**
- `src/cursus/steps/builders/builder_model_calibration_step.py` - Updated for dynamic spec selection

**Key Changes:**
- Import all four job type specifications with graceful fallback
- Dynamic specification selection in constructor based on config.job_type
- Job_type validation in configuration validation
- Output paths now include job_type: `{pipeline_s3_loc}/model_calibration/{job_type}/{logical_name}`
- Command-line argument passing: `["--job_type", job_type]`
- Follows the exact same pattern as TabularPreprocessingStepBuilder

## Implementation Status Summary

**✅ COMPLETED PHASES:**
- **Phase 1**: ModelCalibrationConfig updated with job_type support
- **Phase 2**: All four job type-specific specification files created
- **Phase 3**: ModelCalibrationStepBuilder updated for dynamic spec selection
- **Phase 4**: Model calibration script updated for job_type command-line argument handling
- **Phase 5**: Nested tarball extraction functionality added for training job outputs

**🚧 REMAINING PHASES:**
- **Phase 6**: Testing and integration (NOT IMPLEMENTED)

## Phase 4 Completion Status (✅ COMPLETED)

**Completed Items:**
- [x] Added command-line argument parsing for `--job_type` parameter
- [x] Updated main function to extract job_type from command-line arguments
- [x] Modified load_and_prepare_data calls to pass job_type parameter
- [x] Added proper logging for job_type usage
- [x] Maintained backward compatibility with default job_type="calibration"

**Files Modified:**
- `dockers/xgboost_atoz/scripts/model_calibration.py` - Added job_type command-line argument support

**Key Changes:**
- Command-line argument parser: `parser.add_argument("--job_type", type=str, default="calibration")`
- Job_type extraction in main function: `job_type = job_args.job_type if job_args and hasattr(job_args, 'job_type') else "calibration"`
- Updated function calls: `load_and_prepare_data(config, job_type)`

## Phase 5 Completion Status (✅ COMPLETED)

**Completed Items:**
- [x] Added `extract_and_load_nested_tarball_data()` function from PDA implementation
- [x] Updated `load_and_prepare_data()` to use job_type for different loading strategies
- [x] Added comprehensive tarball extraction with multiple fallback strategies
- [x] Added proper logging and error handling for nested tarball extraction
- [x] Implemented graceful fallback to standard data loading when tarball extraction fails

**Files Modified:**
- `dockers/xgboost_atoz/scripts/model_calibration.py` - Added nested tarball extraction functionality

**Key Features:**
- **Training job_type**: Uses nested tarball extraction for XGBoostTraining outputs
- **Other job_types**: Uses standard data loading for XGBoostModelEval outputs
- **Robust extraction**: Handles multiple tarball nesting scenarios with fallbacks
- **Error resilience**: Falls back to standard loading if tarball extraction fails
- **Comprehensive logging**: Detailed logging for debugging extraction process

## Future Considerations

1. Standardize job_type handling across all processing steps
2. Consider automatic job_type propagation from connected upstream steps
3. Enhance semantic matcher to better handle job type variants
4. Create reusable job_type variant helpers to reduce code duplication

## Timeline

**Week 1 (August 21-27, 2025):**
- Day 1-2: Update ModelCalibrationConfig and create job type specifications
- Day 3: Update ModelCalibrationStepBuilder to handle job_type
- Day 4: Update model_calibration script to use job_type for different data loading logic
- Day 5: Testing, documentation and review

## Conclusion

Adding job_type variant support to the ModelCalibration step will align it with other processing steps, enable better pipeline construction, and improve dependency resolution. By following the established patterns in TabularPreprocessing, CurrencyConversion, and RiskTableMapping, we can implement this enhancement with minimal risk and maintain consistency across the codebase. The updated model_calibration script will properly handle different input data formats from both training and model evaluation steps based on the job_type parameter.
