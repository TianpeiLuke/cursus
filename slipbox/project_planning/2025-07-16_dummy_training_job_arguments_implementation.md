# DummyTraining Job Arguments Implementation Example

This document provides a concrete implementation example of the job arguments enhancement for the DummyTraining step. This serves as a reference for how the job arguments field should be implemented in a script contract and utilized by step builders.

## 1. Updated Script Contract

```python
"""
Contract for dummy training step that processes a pretrained model.tar.gz with hyperparameters.

This script contract defines the expected input and output paths, environment variables,
and framework requirements for the DummyTraining step, which processes a pretrained model
by adding hyperparameters.json to it for downstream packaging and payload steps.
"""

from .base_script_contract import ScriptContract

DUMMY_TRAINING_CONTRACT = ScriptContract(
    entry_point="dummy_training.py",
    expected_input_paths={
        "pretrained_model_path": "/opt/ml/processing/input/model/model.tar.gz",
        "hyperparameters_s3_uri": "/opt/ml/processing/input/config/hyperparameters.json"
    },
    expected_output_paths={
        "model_input": "/opt/ml/processing/output/model"  # Matches specification logical name
    },
    expected_arguments={
        "pretrained-model-path": "/opt/ml/processing/input/model/model.tar.gz",
        "hyperparameters-s3-uri": "/opt/ml/processing/input/config/hyperparameters.json",
        "output-dir": "/opt/ml/processing/output/model"
    },
    required_env_vars=[],
    optional_env_vars={},
    framework_requirements={
        "boto3": ">=1.26.0",
        "pathlib": ">=1.0.0"
    },
    description="Contract for dummy training step that processes a pretrained model.tar.gz by unpacking it, "
                "adding a hyperparameters.json file inside, and repacking it for downstream steps"
)
```

## 2. Updated StepBuilderBase Method

In the `StepBuilderBase` class, we add a generic method for generating job arguments:

```python
def _get_job_arguments(self) -> Optional[List[str]]:
    """
    Constructs command-line arguments for the script based on script contract.
    If no arguments are defined in the contract, returns None (not an empty list).
    
    Returns:
        List of string arguments to pass to the script, or None if no arguments
    """
    if not hasattr(self, 'contract') or not self.contract:
        self.log_warning("No contract available for argument generation")
        return None
        
    # If contract has no expected arguments, return None
    if not hasattr(self.contract, 'expected_arguments') or not self.contract.expected_arguments:
        return None
        
    args = []
    
    # Add each expected argument with its value
    for arg_name, arg_value in self.contract.expected_arguments.items():
        args.extend([f"--{arg_name}", arg_value])
    
    # If we have arguments to return
    if args:
        self.log_info("Generated job arguments from contract: %s", args)
        return args
    
    # If we end up with an empty list, return None instead
    return None
```

## 3. DummyTrainingStepBuilder Implementation

The `DummyTrainingStepBuilder` class is updated to use the contract-based arguments:

```python
def _get_job_arguments(self) -> Optional[List[str]]:
    """
    Constructs job arguments for dummy training script based on contract.
    
    Returns:
        List of command-line arguments or None
    """
    # Use base implementation that reads from contract
    return super()._get_job_arguments()
```

And the `create_step` method is updated to use this method:

```python
def create_step(self, **kwargs) -> ProcessingStep:
    """
    Create the processing step.
    
    Args:
        **kwargs: Additional keyword arguments for step creation.
                 Should include 'dependencies' list if step has dependencies.
                 
    Returns:
        ProcessingStep: The configured processing step
        
    Raises:
        ValueError: If inputs cannot be extracted
        Exception: If step creation fails
    """
    try:
        # Extract inputs from dependencies using the resolver
        dependencies = kwargs.get('dependencies', [])
        inputs = {}
        if dependencies:
            try:
                extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
                inputs.update(extracted_inputs)
            except Exception as e:
                self.log_warning("Failed to extract inputs from dependencies: %s", e)
        
        # Add any explicitly provided inputs (overriding extracted ones)
        inputs_raw = kwargs.get('inputs', {})
        inputs.update(inputs_raw)
        
        # Create processor
        processor = self._get_processor()
        
        # Get processor inputs and outputs
        processing_inputs = self._get_inputs(inputs)
        processing_outputs = self._get_outputs(kwargs.get('outputs', {}))
        
        # Create the step
        step_name = kwargs.get('step_name', 'DummyTraining')
        
        # Get job arguments from contract
        script_args = self._get_job_arguments()
        
        # Get cache configuration
        cache_config = self._get_cache_config(kwargs.get('enable_caching', True))
        
        # Create the step
        step = processor.run(
            code=self.config.get_script_path(),
            inputs=processing_inputs,
            outputs=processing_outputs,
            arguments=script_args,
            job_name=self._generate_job_name(step_name),
            wait=False,
            cache_config=cache_config
        )
        
        # Attach specification to the step for future reference
        setattr(step, '_spec', self.spec)
        
        return step
        
    except Exception as e:
        self.log_error(f"Error creating DummyTraining step: {e}")
        import traceback
        self.log_error(traceback.format_exc())
        raise ValueError(f"Failed to create DummyTraining step: {str(e)}") from e
```

## 4. Benefits of This Approach

1. **Explicit Contract**: The script contract now explicitly documents what command-line arguments the script expects, making the requirements clearer.

2. **Single Source of Truth**: The arguments are defined in the contract alongside paths, creating a single source of truth.

3. **Consistency**: Arguments are consistently generated from the contract, reducing the risk of misalignment.

4. **Validation**: The system can now validate that the script actually uses the arguments declared in the contract.

5. **Easier Maintenance**: When script argument requirements change, only the contract needs to be updated, and the builder will automatically use the new arguments.

## 5. Backward Compatibility

This implementation maintains backward compatibility in two ways:

1. The `expected_arguments` field is optional in the contract, so existing contracts without this field will still work.

2. The `_get_job_arguments` method in StepBuilderBase returns None when no arguments are defined, which is the correct default for the ProcessingStep's job_arguments parameter.

## 6. Testing Considerations

When testing this implementation, verify:

1. Arguments are correctly generated from the contract
2. The script can parse and use the generated arguments
3. The correct paths from the contract are passed as argument values
4. Empty argument lists are properly handled as None

By following this implementation pattern, we ensure script arguments are explicitly defined and consistently handled throughout the pipeline system.
