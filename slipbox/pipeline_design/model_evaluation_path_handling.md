# Model Evaluation Step Path Handling

## Background

The XGBoost Model Evaluation step requires special path handling compared to other steps in the pipeline. This document explains the design decisions and implementation details.

## The Issue

When loading configurations and executing the pipeline, we discovered an inconsistency between:

1. How the `XGBoostModelEvalStepBuilder` uses script paths
2. How other step builders use script paths
3. How the `get_script_path()` method in config classes was implemented

The inconsistency caused errors during pipeline execution because:

- The notebook calls `model_eval_calibration_config.get_script_path()` which was returning a combined path
- The `XGBoostModelEvalStepBuilder.create_step()` method expects separate `processing_entry_point` and `processing_source_dir`

## Solution

We implemented a special version of `get_script_path()` in `XGBoostModelEvalConfig` that:

1. Returns only the entry point name without combining it with the source directory
2. Maintains the expected behavior for the notebook and pipeline execution
3. Includes clear documentation explaining this special case

This solution preserves the behavior of the `XGBoostModelEvalStepBuilder` while making the `get_script_path()` method behave correctly when called externally.

## Code Implementation

### XGBoostModelEvalConfig

```python
def get_script_path(self) -> str:
    """
    Get script path for XGBoost model evaluation.
    
    SPECIAL CASE: Unlike other step configs, XGBoostModelEvalStepBuilder provides 
    processing_source_dir and processing_entry_point directly to the processor.run() 
    method separately. Therefore, this method should return only the entry point name 
    without combining with source_dir.
    
    Returns:
        Script entry point name (without source_dir)
    """
    # Determine which entry point to use
    entry_point = None
    
    # First priority: Use processing_entry_point if provided
    if self.processing_entry_point:
        entry_point = self.processing_entry_point
    # Second priority: Use contract entry point
    elif hasattr(self, 'script_contract') and self.script_contract and hasattr(self.script_contract, 'entry_point'):
        entry_point = self.script_contract.entry_point
    
    # Return just the entry point name without combining with source directory
    return entry_point
```

### XGBoostModelEvalStepBuilder

```python
# Get script paths from config
# IMPORTANT: Using processing_entry_point directly rather than get_script_path()
# This is intentional - XGBoostModelEvalConfig.get_script_path() is designed to 
# return only the entry point without combining it with source directory
script_path = self.config.processing_entry_point
source_dir = self.config.processing_source_dir
```

## Alternative Approaches Considered

1. **Modify the builder**: We considered changing the builder to use `get_script_path()` and extract the filename, but this would make the builder more complex.

2. **Standard path resolution for all**: We considered making all step builders use the same approach, but this would require changes across multiple components.

3. **Always separate paths**: We could make all step configs return separate parts, but this would break compatibility with existing code.

The chosen solution is the most targeted fix with the least impact on other components.

## Future Considerations

As we evolve the pipeline system:

1. Consider standardizing path handling across all step types
2. Evaluate whether combined or separate path handling is better for future steps
3. Ensure all builder implementations are consistent in how they use config methods

## Related Components

- `XGBoostModelEvalConfig` - Special path handling implementation
- `XGBoostModelEvalStepBuilder` - Uses separate paths directly from config
- Pipeline template notebooks - Call `get_script_path()` for validation/display
