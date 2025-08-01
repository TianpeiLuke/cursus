# Training Step Input Handling Improvements

## Background

The XGBoost Training step had two issues with its input handling:

1. It used an ad-hoc approach for creating inputs instead of following the specification-driven pattern used in other steps
2. It did not prioritize internally generated hyperparameters over dependency-provided ones

```python
# Original code in _get_inputs
# Handle different input types based on logical name
if logical_name == "input_path":
    # Training data - create train/val/test channels
    base_path = inputs[logical_name]
    
    # Create separate channels for each data split
    training_inputs["train"] = TrainingInput(s3_data=Join(on='/', values=[base_path, "train/"]))
    training_inputs["val"] = TrainingInput(s3_data=Join(on='/', values=[base_path, "val/"]))
    training_inputs["test"] = TrainingInput(s3_data=Join(on='/', values=[base_path, "test/"]))
    
    logger.info(f"Created train/val/test channels from input_path: {base_path}")
    
elif logical_name == "hyperparameters_s3_uri":
    # Hyperparameters config - single file
    config_uri = inputs[logical_name]
    
    # Ensure we're using the full file path
    if not S3PathHandler.get_name(config_uri) == "hyperparameters.json":
        config_uri = S3PathHandler.join(config_uri, "hyperparameters.json")
        
    training_inputs["config"] = TrainingInput(s3_data=config_uri)
    logger.info(f"Created config channel from hyperparameters_s3_uri: {config_uri}")
```

## The Problem

The issues with this approach were:

1. **Inconsistent Design**: It did not follow the specification-driven approach like other steps
2. **Hardcoded Names**: Channel names like "train", "val", "test", and "config" were hardcoded
3. **Dependency Order**: It would use dependency-provided hyperparameters instead of internally generated ones

This made the code less maintainable and caused inconsistent behavior with other steps.

## The Solution

We refactored the input handling to:

1. **Follow Specification-Driven Design**: Iterate through dependencies defined in the spec
2. **Prioritize Internal Hyperparameters**: Generate hyperparameters internally and ignore any dependency-provided ones
3. **Extract Channel Names from Container Paths**: Derive channel names from container paths in the contract
4. **Add Helper Methods**: Created a `_create_data_channels_from_source` helper for creating data channels

### 1. Prioritizing Internal Hyperparameters

```python
# SPECIAL CASE: Always generate hyperparameters internally first
hyperparameters_key = "hyperparameters_s3_uri"

# Generate hyperparameters file regardless of whether inputs contains it
internal_hyperparameters_s3_uri = self._prepare_hyperparameters_file()
logger.info(f"[TRAINING INPUT OVERRIDE] Generated hyperparameters internally at: {internal_hyperparameters_s3_uri}")
logger.info(f"[TRAINING INPUT OVERRIDE] This will be used regardless of any dependency-provided values")

# Create config channel with the internal hyperparameters
training_inputs["config"] = TrainingInput(s3_data=internal_hyperparameters_s3_uri)
matched_inputs.add(hyperparameters_key)  # Mark as handled

# Remove our special case from the inputs dictionary
if hyperparameters_key in working_inputs:
    external_path = working_inputs[hyperparameters_key]
    logger.info(f"[TRAINING INPUT OVERRIDE] Ignoring dependency-provided hyperparameters: {external_path}")
    logger.info(f"[TRAINING INPUT OVERRIDE] Using internal hyperparameters instead: {internal_hyperparameters_s3_uri}")
    del working_inputs[hyperparameters_key]
```

### 2. Creating Data Channel Helper

```python
def _create_data_channels_from_source(self, base_path):
    """
    Create train, validation, and test channel inputs from a base path.
    """
    channels = {
        "train": TrainingInput(s3_data=Join(on='/', values=[base_path, "train/"])),
        "val": TrainingInput(s3_data=Join(on='/', values=[base_path, "val/"])),
        "test": TrainingInput(s3_data=Join(on='/', values=[base_path, "test/"]))
    }
    
    return channels
```

### 3. Specification-Driven Processing

```python
# Process each dependency in the specification
for _, dependency_spec in self.spec.dependencies.items():
    logical_name = dependency_spec.logical_name
    
    # Skip inputs we've already handled
    if logical_name in matched_inputs:
        continue
        
    # Skip if optional and not provided
    if not dependency_spec.required and logical_name not in working_inputs:
        continue
        
    # Make sure required inputs are present
    if dependency_spec.required and logical_name not in working_inputs:
        raise ValueError(f"Required input '{logical_name}' not provided")
    
    # Get container path from contract
    container_path = None
    if logical_name in self.contract.expected_input_paths:
        container_path = self.contract.expected_input_paths[logical_name]
        
        # SPECIAL HANDLING FOR input_path
        if logical_name == "input_path":
            base_path = working_inputs[logical_name]
            
            # Create separate channels for each data split using helper method
            data_channels = self._create_data_channels_from_source(base_path)
            training_inputs.update(data_channels)
            logger.info(f"Created data channels from {logical_name}: {base_path}")
        else:
            # For other inputs, extract the channel name from the container path
            parts = container_path.split('/')
            if len(parts) > 4 and parts[1] == "opt" and parts[2] == "ml" and parts[3] == "input" and parts[4] == "data":
                if len(parts) > 5:
                    channel_name = parts[5]  # Extract channel name from path
                    training_inputs[channel_name] = TrainingInput(s3_data=working_inputs[logical_name])
                    logger.info(f"Created {channel_name} channel from {logical_name}: {working_inputs[logical_name]}")
                else:
                    # If no specific channel in path, use logical name as channel
                    training_inputs[logical_name] = TrainingInput(s3_data=working_inputs[logical_name])
                    logger.info(f"Created {logical_name} channel from {logical_name}: {working_inputs[logical_name]}")
    else:
        raise ValueError(f"No container path found for input: {logical_name}")
```

## Key Improvements

1. **Consistent Approach**: The training step now follows the same specification-driven approach as other steps
2. **Explicit Prioritization**: Internal hyperparameters are explicitly prioritized over dependency-provided ones
3. **Clear Logging**: Added clear logging with a `[TRAINING INPUT OVERRIDE]` prefix when special handling occurs
4. **Derived Channel Names**: Channel names are derived from container paths in the contract where possible
5. **Improved Maintainability**: Extracted reusable logic into helper methods

## Relationship to Container Paths

SageMaker Training Jobs use a channel-based input system where:

1. **Channel Names**: Inputs are organized by channel names (e.g., "train", "val", "test", "config")
2. **Container Paths**: Each channel is mounted at `/opt/ml/input/data/{channel_name}` in the container

The improved implementation extracts channel names from container paths where possible, making the connection between logical names in the spec and channel names in the container more explicit.

## Benefits

This change provides several benefits:

1. **Better Alignment**: The code now better aligns with the specification-driven approach
2. **More Consistent**: It behaves similarly to other steps like the packaging step
3. **Clearer Intent**: The prioritization of internal hyperparameters is explicit
4. **Better Logging**: The logging is more informative and follows a consistent pattern
5. **More Maintainable**: The code is more maintainable with extracted helper methods

## Simplified create_step Method

With the new `_get_inputs` method handling hyperparameters internally, we could also simplify the `create_step` method by removing redundant code:

```python
# Before: Explicitly handling hyperparameters in create_step
# Ensure we have hyperparameters - either generate them or use provided ones
if "hyperparameters_s3_uri" not in inputs:
    # Generate hyperparameters file
    hyperparameters_s3_uri = self._prepare_hyperparameters_file()
    inputs["hyperparameters_s3_uri"] = hyperparameters_s3_uri
    logger.info(f"Generated hyperparameters at: {hyperparameters_s3_uri}")
```

```python
# After: Let _get_inputs handle hyperparameters internally
# Get training inputs using specification-driven method
# Note: _get_inputs now handles generating hyperparameters internally
training_inputs = self._get_inputs(inputs)
```

This further improves code organization by:

1. **Separation of Concerns**: Each method has a single, well-defined responsibility
2. **Reduced Duplication**: Hyperparameter generation logic is only in one place
3. **Consistent Behavior**: Hyperparameters are always generated internally regardless of how the step is called
4. **Clearer Dependencies**: `_get_inputs` is now self-contained and doesn't rely on `create_step` to prepare hyperparameters

# Training Step Output Path Improvements

## Background

In addition to the input handling improvements, we also enhanced the output path handling in both the PyTorch and XGBoost training steps to be more consistent with other steps like TabularPreprocessingStep.

## Problem Statement

The training steps had inconsistent output path handling compared to other steps:

1. They used hardcoded fallback values:
   ```python
   bucket = getattr(self.config, 'bucket', 'default-bucket')
   pipeline_name = getattr(self.config, 'pipeline_name', 'pytorch-model')
   current_date = getattr(self.config, 'current_date', '2025-07-07')
   primary_output_path = f"s3://{bucket}/{pipeline_name}/training_output/{current_date}/model"
   ```

2. The path generation logic wasn't aligned with the specification-driven approach.

3. There was minimal logging about path generation.

## Solution

We updated both training step builders to use the `pipeline_s3_loc` property from the base config class, similar to how TabularPreprocessingStep works:

### Before (PyTorch & XGBoost):
```python
# Try to find destination in outputs
if logical_name in outputs:
    primary_output_path = outputs[logical_name]
else:
    # Generate default output path using base config
    bucket = getattr(self.config, 'bucket', 'default-bucket')
    pipeline_name = getattr(self.config, 'pipeline_name', 'pytorch-model')
    current_date = getattr(self.config, 'current_date', '2025-07-07')
    primary_output_path = f"s3://{bucket}/{pipeline_name}/training_output/{current_date}/model"
```

### After (PyTorch):
```python
# Try to find destination in outputs
if logical_name in outputs:
    primary_output_path = outputs[logical_name]
else:
    # Generate destination using pipeline_s3_loc like tabular preprocessing
    primary_output_path = f"{self.config.pipeline_s3_loc}/pytorch_training/{logical_name}"
    logger.info(f"Using generated destination for '{logical_name}': {primary_output_path}")
```

### After (XGBoost):
```python
# Try to find destination in outputs
if logical_name in outputs:
    primary_output_path = outputs[logical_name]
else:
    # Generate destination using pipeline_s3_loc like tabular preprocessing
    primary_output_path = f"{self.config.pipeline_s3_loc}/xgboost_training/{logical_name}"
    logger.info(f"Using generated destination for '{logical_name}': {primary_output_path}")
```

The fallback case was also improved:

```python
# If no model output found in spec, generate default output path
if primary_output_path is None:
    # Generate default path using pipeline_s3_loc
    primary_output_path = f"{self.config.pipeline_s3_loc}/xgboost_training/model"
    logger.warning(f"No model output found in specification. Using default path: {primary_output_path}")
```

## Path Format Comparison

### TabularPreprocessingStep:
```
{pipeline_s3_loc}/tabular_preprocessing/{job_type}/{logical_name}
```

### PyTorchTrainingStep:
```
{pipeline_s3_loc}/pytorch_training/{logical_name}
```

### XGBoostTrainingStep:
```
{pipeline_s3_loc}/xgboost_training/{logical_name}
```

## Benefits

1. **Consistency**: All step types now use the same pattern for output path generation.

2. **Configuration-Based**: Path generation relies on the `pipeline_s3_loc` configuration rather than hardcoded defaults.

3. **Better Logging**: Added logging statements make it clear which paths are being generated and why.

4. **Specification-Driven**: The output paths are derived from the step specification's output logical names.

## Special Considerations for Training Steps

Unlike processing steps which can have multiple outputs, training steps only accept a single output path. This requires identifying the primary model output from the specification's outputs.

The logic to identify the primary model output looks for:
1. An output with logical name exactly matching "model_output"
2. Or an output with "model" in its logical name

We also maintain a proper fallback mechanism if no suitable output is found in the specification.
