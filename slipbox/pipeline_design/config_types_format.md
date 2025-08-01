# Config Types Format

## Overview

This design document describes the format requirements for the `config_types` metadata in configuration files, focusing on how step names are mapped to class names and ensuring backward compatibility.

## Purpose

The purpose of the config types format is to:

1. **Map step names to class names** for proper config instantiation
2. **Support job type variants** with distinct step names
3. **Enable validation** during configuration loading
4. **Maintain backward compatibility** with existing systems

## Format Requirements

### 1. Step Name to Class Name Mapping

The `config_types` metadata must follow this format:

```json
"config_types": {
  "StepName": "ConfigClassName",
  "StepName_jobtype": "ConfigClassName"
}
```

Where:
- **Keys are step names** (not class names)
- **Values are class names** (the full class name including "Config" suffix)
- **Job type variants** have the job type appended to the step name

### 2. Step Name Format

Step names must follow this pattern:

1. **Base step name**: Derived from the config class name or registry
   - From registry: `CONFIG_STEP_REGISTRY[class_name]` (e.g., "CradleDataLoading")
   - Fallback: Remove "Config" suffix from class name (e.g., "CradleDataLoad")
   
2. **Variant suffixes**: Append distinguishing attributes
   - `job_type`: Appended directly (e.g., "CradleDataLoading_training")
   - `data_type`: Appended if present (e.g., "XGBoostTraining_tabular")
   - `mode`: Appended if present (e.g., "ModelRegistration_prod")

3. **Override**: Honor `step_name_override` attribute if present

### 3. Class Name Format

Class names must be the complete class name, including the "Config" suffix:

```json
"config_types": {
  "CradleDataLoading": "CradleDataLoadConfig",
  "XGBoostTraining": "XGBoostTrainingConfig"
}
```

## Example Formats

### Correct Format

```json
"config_types": {
  "Base": "BasePipelineConfig",
  "CradleDataLoading_training": "CradleDataLoadConfig",
  "CradleDataLoading_calibration": "CradleDataLoadConfig",
  "XGBoostTraining": "XGBoostTrainingConfig",
  "XGBoostModelEval_calibration": "XGBoostModelEvalConfig"
}
```

### Incorrect Format (Using Class Names as Keys)

```json
"config_types": {
  "BasePipelineConfig": "BasePipelineConfig",
  "CradleDataLoadConfig": "CradleDataLoadConfig",
  "XGBoostTrainingConfig": "XGBoostTrainingConfig"
}
```

This incorrect format causes validation failures because:
1. The `load_configs` function looks for entries in the `specific` section using keys from `config_types`
2. The `specific` section uses step names as keys, not class names
3. No data is found, causing validation failures

## Implementation Details

### 1. Generating Config Types During Save

```python
def save(self, output_file: str) -> Dict[str, Any]:
    # ... merge configurations ...
    
    # Create metadata with proper step name -> class name mapping for config_types
    # Using TypeAwareConfigSerializer for consistent step name generation
    serializer = TypeAwareConfigSerializer()
    config_types = {}
    
    for cfg in self.config_list:
        step_name = serializer.generate_step_name(cfg)
        class_name = cfg.__class__.__name__
        config_types[step_name] = class_name
    
    metadata = {
        'created_at': datetime.now().isoformat(),
        'config_types': config_types
    }
    
    # ... create output structure and save ...
```

The use of `TypeAwareConfigSerializer.generate_step_name()` ensures consistent step name generation across the system, following the Single Source of Truth principle.

### 2. Using Config Types During Load

```python
def load_configs(input_file: str, config_classes: Dict[str, Type[BaseModel]]) -> Dict[str, BaseModel]:
    # ... load file data ...
    
    meta = data['metadata']
    cfgs = data['configuration']
    types = meta['config_types']  # step_name -> class_name
    rebuilt = {}
    
    for step_name, class_name in types.items():
        if class_name not in config_classes:
            raise ValueError(f"Unknown config class: {class_name}")
        cls = config_classes[class_name]
        
        # Build field dictionary from shared and specific sections
        fields = {}
        
        # Add shared values (lowest priority)
        for k, v in cfgs['shared'].items():
            if k in cls.model_fields.keys():
                fields[k] = v
        
        # Add specific values (highest priority)
        for k, v in cfgs['specific'].get(step_name, {}).items():
            if k in cls.model_fields.keys():
                fields[k] = v
        
        # Create the instance
        try:
            instance = cls(**fields)
            rebuilt[step_name] = instance
        except Exception as e:
            logger.error(f"Failed to create instance for {step_name}: {str(e)}")
            raise
    
    return rebuilt
```

## Single Source of Truth Implementation

The refactored system ensures consistent step name generation by using a single implementation:

1. `TypeAwareConfigSerializer.generate_step_name()` is the single authoritative method for generating step names
2. All components that need to generate step names use this method:
   - `ConfigMerger`: For generating step name keys in config_types metadata
   - `utils.serialize_config`: For backward compatibility with legacy code
   - All serialization components: For consistent step name generation

This avoids duplicating the step name generation logic, ensuring consistency and making maintenance easier.

## Backward Compatibility

To maintain backward compatibility, the system:

1. **Accepts both formats**: Can load configs with either format
2. **Always saves in correct format**: Ensures saved configs use the correct format
3. **Provides migration utilities**: Offers tools to fix incorrect formats
4. **Falls back to legacy behaviors**: Uses string manipulation when registry is unavailable
5. **Maintains API stability**: Preserves the same function signatures and behaviors

### Migration Tool

```python
def fix_config_types_format(input_file: str, output_file: str = None) -> str:
    """Fix config_types format in an existing config file."""
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

## Best Practices

1. **Use Registry**: Always use the pipeline registry for step names
2. **Include Job Types**: Always include job_type in relevant steps
3. **Test Both Directions**: Test both saving and loading configurations
4. **Validate Output**: Validate `config_types` format after saving
5. **Use Fix Utility**: Use the fix utility on legacy configs

## Future Improvements

1. **Format Validation**: Add validation to ensure correct format
2. **Registry Completeness**: Ensure all classes are in the registry
3. **Automated Migration**: Automatically migrate during load if needed
4. **Documentation**: Expand documentation with examples

## References

- [Registry-Based Step Name Generation](./registry_based_step_name_generation.md)
- [Job Type Variant Handling](./job_type_variant_handling.md)
- [Config Types Format Fix Plan](../slipbox/project_planning/2025-07-18_fix_config_types_format.md)
