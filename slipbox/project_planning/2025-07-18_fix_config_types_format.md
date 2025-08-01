# Config Types Format Fix Plan

## Overview

This document outlines a plan to fix an issue discovered with the `config_types` format in the configuration serialization system. The current implementation in `ConfigMerger.save()` uses class names as both keys and values, breaking backward compatibility with the legacy format and causing validation failures during config loading.

## Issue Description

### Current Implementation (Problematic)

```python
# Current problematic code:
'config_types': {
    # This creates class name -> class name mapping
    getattr(cfg, "step_name_override", cfg.__class__.__name__): cfg.__class__.__name__
    for cfg in self.config_list
}
```

This generates a format like:

```json
"config_types": {
  "BasePipelineConfig": "BasePipelineConfig",
  "CradleDataLoadConfig": "CradleDataLoadConfig",
  "ModelRegistrationConfig": "ModelRegistrationConfig",
  ...
}
```

### Expected Legacy Format

The legacy format used step names (with job type variants) as keys:

```json
"config_types": {
  "Base": "BasePipelineConfig",
  "CradleDataLoading_training": "CradleDataLoadConfig",
  "CradleDataLoading_calibration": "CradleDataLoadConfig",
  "XGBoostTraining": "XGBoostTrainingConfig",
  ...
}
```

### Impact

This format mismatch causes the `load_configs` function to fail validation when creating configs from class names, as it can't find the corresponding entries in the `specific` section of the configuration.

The issue occurs because:
1. `load_configs` processes each entry in `config_types`
2. For each entry, it tries to find data in the `specific` section using the key from `config_types`
3. Since the keys are now class names (e.g., "XGBoostTrainingConfig") instead of step names (e.g., "XGBoostTraining"), no data is found
4. The function falls back to using only `shared` data, which often lacks required fields

## Solution Plan

### 1. Update ConfigMerger.save() Method

Modify the `save()` method in `src/config_field_manager/config_merger.py` to use step names as keys:

```python
def save(self, output_file: str) -> Dict[str, Any]:
    """
    Merge configurations and save to a file using the simplified structure.
    
    Args:
        output_file: Path to output file
        
    Returns:
        dict: Merged configuration
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Merge configurations
    merged = self.merge()
    
    # Create config_types with proper step_name -> class_name mapping
    config_types = {}
    
    for cfg in self.config_list:
        # Get class name
        class_name = cfg.__class__.__name__
        
        # Generate proper step name
        step_name = self._generate_step_name(cfg)
        
        # Add to config_types mapping
        config_types[step_name] = class_name
    
    # Create metadata with the corrected format
    metadata = {
        'created_at': datetime.now().isoformat(),
        'config_types': config_types
    }
    
    # Create the output structure with the simplified format
    output = {
        'metadata': metadata,
        'configuration': merged
    }
    
    # Serialize and save to file
    self.logger.info(f"Saving merged configuration to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, sort_keys=True)
    
    self.logger.info(f"Successfully saved merged configuration to {output_file}")
    return merged

def _generate_step_name(self, config: Any) -> str:
    """
    Generate a consistent step name for a config object.
    
    Args:
        config: Config object
        
    Returns:
        str: Step name
    """
    # Get base step name from class name
    class_name = config.__class__.__name__
    
    # Remove "Config" suffix if present
    base_step = class_name
    if base_step.endswith("Config"):
        base_step = base_step[:-6]
    
    step_name = base_step
    
    # Append distinguishing attributes (job_type, data_type, mode)
    for attr in ("job_type", "data_type", "mode"):
        if hasattr(config, attr):
            val = getattr(config, attr)
            if val is not None:
                step_name = f"{step_name}_{val}"
    
    # Use step name override if available
    return getattr(config, "step_name_override", step_name)
```

### 2. Ensure Consistent Step Name Generation

To maintain consistency throughout the system, we need to ensure step name generation follows the same logic everywhere:

1. Extract the common step name generation logic to a shared utility function
2. Use this utility in both `TypeAwareConfigSerializer` and `ConfigMerger`
3. Document the step name generation rules clearly

This will prevent divergent implementations that might cause format mismatches.

### 3. Add Unit Tests to Verify Format

Create tests that specifically verify the `config_types` format:

```python
def test_config_types_format(self):
    """Test that config_types uses step names as keys."""
    # Create test configs including ones with job_type
    config1 = XGBoostTrainingConfig(field1="value1")
    config2 = CradleDataLoadConfig(field2="value2", job_type="training")
    
    # Merge and save
    merger = ConfigMerger([config1, config2])
    with tempfile.NamedTemporaryFile() as tmp:
        merged = merger.save(tmp.name)
        
        # Load and check format
        with open(tmp.name, 'r') as f:
            saved_data = json.load(f)
        
        # Verify config_types format
        self.assertIn("metadata", saved_data)
        self.assertIn("config_types", saved_data["metadata"])
        
        config_types = saved_data["metadata"]["config_types"]
        
        # Keys should be step names
        self.assertIn("XGBoostTraining", config_types)
        self.assertIn("CradleDataLoading_training", config_types)
        
        # Values should be class names
        self.assertEqual("XGBoostTrainingConfig", config_types["XGBoostTraining"])
        self.assertEqual("CradleDataLoadConfig", config_types["CradleDataLoading_training"])
```

### 4. Create Backward Compatibility Utility

Create a utility function to fix existing configuration files with the wrong format:

```python
def fix_config_types_format(input_file: str, output_file: str = None) -> str:
    """
    Fix config_types format in an existing config file.
    
    Args:
        input_file: Path to input config file
        output_file: Path for output (defaults to input file)
        
    Returns:
        str: Message indicating result
    """
    if not output_file:
        output_file = input_file
        
    with open(input_file, 'r') as f:
        data = json.load(f)
        
    # Check if fix is needed
    needs_fix = False
    if "metadata" in data and "config_types" in data["metadata"]:
        config_types = data["metadata"]["config_types"]
        
        # Check if keys are class names instead of step names
        if any(key.endswith("Config") for key in config_types.keys()):
            needs_fix = True
            
    if needs_fix:
        # Create fixed config_types
        fixed_config_types = {}
        
        for class_name, _ in data["metadata"]["config_types"].items():
            # Convert class name to step name
            base_step = class_name
            if base_step.endswith("Config"):
                base_step = base_step[:-6]  # Remove "Config" suffix
                
            # Check for job_type variants in specific section
            specific = data.get("configuration", {}).get("specific", {})
            
            # Find matching step name in specific section
            matching_step = None
            for step_name in specific.keys():
                if step_name.startswith(base_step):
                    matching_step = step_name
                    break
            
            # Use matching step name or base step name
            key = matching_step or base_step
            fixed_config_types[key] = class_name
        
        # Update config_types
        data["metadata"]["config_types"] = fixed_config_types
        
        # Write fixed data
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return f"Fixed config_types format in {output_file}"
    else:
        return f"No fix needed for {input_file}"
```

### 5. Update Documentation

Update relevant documentation to clearly describe the expected format for `config_types`:

1. Add docstrings to the `ConfigMerger.save()` method
2. Update the design document to specify the format requirements
3. Include examples of correct and incorrect formats
4. Explain the relationship between step names and class names

## Implementation Steps

1. **Modify ConfigMerger.save() Method**
   - Update the method to use the step name generation function
   - Ensure proper format for the config_types mapping

2. **Add _generate_step_name Helper Method**
   - Implement consistent step name generation logic
   - Consider moving to a common utility module for reuse

3. **Add Tests**
   - Create specific tests for the config_types format
   - Test with various job type variants
   - Verify backward compatibility

4. **Create Fix Utility**
   - Implement the fix_config_types_format utility
   - Add command-line interface for batch fixing
   - Document usage instructions

5. **Update Documentation**
   - Add explicit format requirements to design documents
   - Update docstrings with format explanations
   - Create examples showing proper usage

## Verification

To verify the fix works correctly:

1. Run the updated implementation with test configs
2. Compare the output format with the legacy format
3. Verify that loading the saved configs succeeds
4. Run the backward compatibility test with existing configs

## References

- [Config Field Categorization Refactoring Plan](./2025-07-17_config_field_categorization_refactoring_plan.md)
- [Config Field Categorization Refactored Design](../pipeline_design/config_field_categorization_refactored.md)
- [Job Type Variant Solution](./2025-07-04_job_type_variant_solution.md)
