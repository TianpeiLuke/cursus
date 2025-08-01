# Simplified Config Field Categorization

## Overview

This design document describes the simplified config field categorization system, which determines how fields are organized into shared and specific sections in a configuration file.

## Purpose

The purpose of the simplified config field categorization is to:

1. **Reduce complexity** by flattening the structure
2. **Provide clear rules** for categorizing fields
3. **Support special fields** that need specific treatment
4. **Enable efficient serialization** and deserialization

## Simplified Structure

The simplified structure has two main sections:

```json
{
  "shared": {
    "field1": "value1",
    "field2": "value2"
  },
  "specific": {
    "StepName1": {
      "field3": "value3"
    },
    "StepName2": {
      "field4": "value4"
    }
  }
}
```

This is a flattened version of the legacy structure that had additional nested sections.

## Categorization Rules

Fields are categorized using these explicit rules with clear precedence:

### 1. Special Fields → `specific`

Special fields are always kept in the specific section for each step:
- Fields in the `SPECIAL_FIELDS_TO_KEEP_SPECIFIC` list
- Fields that are Pydantic models
- Complex nested data structures

```python
SPECIAL_FIELDS_TO_KEEP_SPECIFIC = {
    "hyperparameters", 
    "hyperparameters_s3_uri",
    "job_name_prefix",
    "job_type"
}
```

### 2. Fields in One Config → `specific`

Fields that only appear in a single configuration go to the specific section for that step.

### 3. Fields with Different Values → `specific`

Fields with the same name but different values across configurations go to the specific section for each step.

### 4. Non-Static Fields → `specific`

Fields identified as non-static (runtime values, input/output paths) go to the specific section.

Non-static fields are identified by:
- Names containing patterns like `_names`, `input_`, `output_`, `_specific`, `_count`
- Complex values (large dictionaries or lists)

### 5. Fields with Identical Values → `shared`

Fields with identical values across all configurations that are not caught by the above rules go to the shared section.

### 6. Default Case → `specific`

When in doubt, fields are placed in the specific section to ensure proper functioning.

## Implementation Details

### 1. Field Information Collection

```python
def _collect_field_info(self, config_list):
    """Collect comprehensive information about all fields."""
    field_info = {
        'values': defaultdict(set),            # field_name -> set of values (as JSON strings)
        'sources': defaultdict(list),          # field_name -> list of step names
        'is_static': defaultdict(bool),        # field_name -> is this field likely static
        'is_special': defaultdict(bool),       # field_name -> is this a special field
        'raw_values': defaultdict(dict)        # field_name -> {step_name: actual value}
    }
    
    # Collect information from all configs
    for config in config_list:
        serialized = serialize_config(config)
        step_name = serialized["_metadata"]["step_name"]
        
        for field_name, value in serialized.items():
            if field_name == "_metadata":
                continue
                
            # Track raw value
            field_info['raw_values'][field_name][step_name] = value
            
            # Track serialized value for comparison
            try:
                value_str = json.dumps(value, sort_keys=True)
                field_info['values'][field_name].add(value_str)
            except (TypeError, ValueError):
                # If not JSON serializable, use object ID as placeholder
                field_info['values'][field_name].add(f"__non_serializable_{id(value)}__")
            
            # Track sources
            field_info['sources'][field_name].append(step_name)
            
            # Check if special
            field_info['is_special'][field_name] = self._is_special_field(field_name, value, config)
            
            # Check if static
            field_info['is_static'][field_name] = self._is_likely_static(field_name, value)
    
    return field_info
```

### 2. Field Categorization

```python
def _categorize_fields(self, field_info):
    """Apply categorization rules to all fields."""
    categorization = {
        'shared': {},
        'specific': defaultdict(dict)
    }
    
    for field_name in field_info['sources']:
        # Apply rules in order of precedence
        if field_info['is_special'][field_name]:
            category = 'specific'
        elif len(field_info['sources'][field_name]) <= 1:
            category = 'specific'
        elif len(field_info['values'][field_name]) > 1:
            category = 'specific'
        elif not field_info['is_static'][field_name]:
            category = 'specific'
        else:
            category = 'shared'
        
        # Place field in appropriate category
        self._place_field(field_name, category, categorization, field_info)
    
    return categorization
```

### 3. Field Placement

```python
def _place_field(self, field_name, category, categorization, field_info):
    """Place field in the appropriate category."""
    if category == 'shared':
        # Use common value for all configs
        value_str = next(iter(field_info['values'][field_name]))
        categorization['shared'][field_name] = json.loads(value_str)
    else:
        # Add to specific section for each config
        for step_name in field_info['sources'][field_name]:
            value = field_info['raw_values'][field_name][step_name]
            categorization['specific'][step_name][field_name] = value
```

## Benefits

### 1. Simpler Mental Model

- **Flattened Structure**: Just two primary sections (shared and specific)
- **Clear Rules**: Explicit precedence-based rules for categorization
- **Consistent Behavior**: Fields are always categorized the same way

### 2. Improved Maintainability

- **Separation of Concerns**: Field categorization is isolated from serialization
- **Independent Components**: ConfigFieldCategorizer works independently
- **Explicit Logging**: Clear logging of categorization decisions

### 3. Better Performance

- **Single Pass Analysis**: All field information collected in one pass
- **Efficient Categorization**: Rules applied once per field
- **Reduced Redundancy**: No duplicate processing of fields

### 4. Enhanced Robustness

- **Special Field Handling**: Consistent treatment of special fields
- **Type-Safe Operations**: Strong typing prevents category errors
- **Default Safety**: When in doubt, fields go to specific sections

## Field Handling Examples

### 1. Special Fields

```python
# Special field 'hyperparameters'
config1.hyperparameters = {"max_depth": 6}
config2.hyperparameters = {"max_depth": 8}

# Result in configuration
{
  "specific": {
    "Step1": {"hyperparameters": {"max_depth": 6}},
    "Step2": {"hyperparameters": {"max_depth": 8}}
  }
}
```

### 2. Fields with Different Values

```python
# Different values for 'learning_rate'
config1.learning_rate = 0.1
config2.learning_rate = 0.01

# Result in configuration
{
  "specific": {
    "Step1": {"learning_rate": 0.1},
    "Step2": {"learning_rate": 0.01}
  }
}
```

### 3. Fields with Same Values

```python
# Same values for 'pipeline_name'
config1.pipeline_name = "my_pipeline"
config2.pipeline_name = "my_pipeline"

# Result in configuration
{
  "shared": {
    "pipeline_name": "my_pipeline"
  }
}
```

## Loading Priority

When loading configurations, fields are prioritized:

1. **Specific Values**: Fields from the specific section for this step
2. **Shared Values**: Fields from the shared section

```python
# Build field dictionary with priority
fields = {}

# Add shared values (lowest priority)
for k, v in shared.items():
    if k in valid_fields:
        fields[k] = v
        
# Add specific values (highest priority)
for k, v in specific.get(step_name, {}).items():
    if k in valid_fields:
        fields[k] = v
```

## Future Improvements

1. **Enhanced Field Analysis**: Better heuristics for static/dynamic field detection
2. **Categorization Hints**: Allow configs to specify preferred categorization
3. **Configuration Review**: Add tools to analyze and validate categorization
4. **Visualization Tools**: Create visualizations of field categorization

## References

- [Config Field Categorization](./config_field_categorization.md)
- [Registry-Based Step Name Generation](./registry_based_step_name_generation.md)
- [Config Types Format](./config_types_format.md)
