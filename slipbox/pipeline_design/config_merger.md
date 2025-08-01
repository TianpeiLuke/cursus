# Config Merger

## Overview

The ConfigMerger is responsible for merging multiple configuration objects into a unified structure with shared and specific sections, then saving the result to a file. It orchestrates the field categorization and serialization processes to produce a consistent, well-organized configuration file.

## Purpose

The purpose of ConfigMerger is to:

1. **Consolidate multiple configurations** into a single structure
2. **Organize fields** into shared and specific sections
3. **Handle special fields** appropriately
4. **Generate metadata** for the configuration
5. **Save merged configuration** to a file

## Key Components

### 1. Merger Constructor

The merger is initialized with the configurations to merge and optional components:

```python
class ConfigMerger:
    """
    Handles the merging of multiple configs based on field categorization.
    
    Coordinates the categorization, serialization, and validation processes
    to produce a properly merged configuration.
    """
    
    def __init__(self, config_list, categorizer=None, serializer=None):
        """
        Initialize with configs and optional components.
        
        Args:
            config_list: List of config objects to merge
            categorizer: Optional ConfigFieldCategorizer instance
            serializer: Optional TypeAwareSerializer instance
        """
        self.config_list = config_list
        # Dependency injection following Separation of Concerns
        self.categorizer = categorizer or ConfigFieldCategorizer(config_list)
        self.serializer = serializer or TypeAwareSerializer()
        self.logger = logging.getLogger(__name__)
```

### 2. Merge Method

The core method that performs the merging operation:

```python
def merge(self):
    """
    Merge configs based on field categorization.
    
    Returns:
        dict: Merged configuration
    """
    merged = self.categorizer.categorization
    
    # Serialize all values
    self._serialize_all_values(merged)
    
    # Handle special fields to ensure they're in the right place
    self._handle_special_fields(merged)
    
    # Ensure mutual exclusivity
    self._ensure_mutual_exclusivity(merged)
    
    return merged
```

### 3. Value Serialization

Serializes all values in the merged structure:

```python
def _serialize_all_values(self, merged):
    """
    Serialize all values in the merged config.
    
    Args:
        merged: The merged configuration to update
    """
    # Serialize shared fields
    for k, v in list(merged['shared'].items()):
        merged['shared'][k] = self.serializer.serialize(v)
            
    # Serialize specific fields
    for step, fields in merged['specific'].items():
        for k, v in list(fields.items()):
            merged['specific'][step][k] = self.serializer.serialize(v)
```

### 4. Special Field Handling

Ensures special fields are correctly placed in specific sections:

```python
def _handle_special_fields(self, merged):
    """
    Ensure special fields are in their appropriate sections.
    
    Args:
        merged: The merged configuration to update
    """
    # Handle special fields in shared
    for field_name in list(merged['shared'].keys()):
        if field_name in SPECIAL_FIELDS_TO_KEEP_SPECIFIC:
            self.logger.info(f"Moving special field '{field_name}' from shared")
            shared_value = merged['shared'].pop(field_name)
            
            # Add to specific configs that have this field
            for config in self.config_list:
                if hasattr(config, field_name):
                    step = self._generate_step_name(config)
                    value = getattr(config, field_name)
                    serialized_value = self.serializer.serialize(value)
                    
                    if step not in merged['specific']:
                        merged['specific'][step] = {}
                    merged['specific'][step][field_name] = serialized_value
```

### 5. Mutual Exclusivity Enforcement

Ensures fields don't appear in both shared and specific sections:

```python
def _ensure_mutual_exclusivity(self, merged):
    """
    Ensure mutual exclusivity between shared/specific sections.
    
    Args:
        merged: The merged configuration to update
    """
    # Check shared vs specific
    shared_fields = set(merged['shared'].keys())
    for step, fields in merged['specific'].items():
        overlap = shared_fields.intersection(set(fields.keys()))
        if overlap:
            self.logger.warning(f"Found fields {overlap} in both 'shared' and 'specific' for step {step}")
            for field in overlap:
                merged['specific'][step].pop(field)
```

### 6. Common Field Validation

Validates that common required fields are present across all configurations:

```python
def _check_required_fields(self, merged: Dict[str, Any]) -> None:
    """
    Check that all common required fields are present in the merged output.
    
    This verifies that mandatory fields shared across configs are included,
    without making assumptions about step-specific required fields.
    
    Args:
        merged: Merged configuration structure
    """
    # Get common required fields defined in base config classes
    shared_fields = set(merged["shared"].keys())
    
    # For fields that should be in shared but might be in specific sections,
    # check if they appear in any specific section instead
    for step_name, fields in merged["specific"].items():
        step_fields = set(fields.keys())
        for field in step_fields:
            # If this is a field that should typically be shared but
            # appears in a specific section, log it
            if field in common_fields_that_should_be_shared:
                self.logger.info(f"Field '{field}' found in specific section '{step_name}' but should be shared")
```

This validation focuses only on common fields required across all configurations rather than making assumptions about step-specific requirements based on naming patterns. The implementation avoids hardcoding field names by examining config classes to determine which fields are common requirements. This approach prevents false positive warnings that previously occurred when step names contained words like "training" or "processing" but the underlying classes had different field requirements.

### 6. Step Name Generation

Generates consistent step names using the registry:

```python
def _generate_step_name(self, config: Any) -> str:
    """
    Generate a consistent step name for a config object using the pipeline registry.
    
    Args:
        config: Config object
        
    Returns:
        str: Step name
    """
    # First check for step_name_override - highest priority
    if hasattr(config, "step_name_override") and config.step_name_override != config.__class__.__name__:
        return config.step_name_override
        
    # Get class name
    class_name = config.__class__.__name__
    
    # Look up the step name from the registry (primary source of truth)
    try:
        from src.pipeline_registry.step_names import CONFIG_STEP_REGISTRY
        if class_name in CONFIG_STEP_REGISTRY:
            base_step = CONFIG_STEP_REGISTRY[class_name]
        else:
            # Fall back to the old behavior if not in registry
            base_step = class_name
            if base_step.endswith("Config"):
                base_step = base_step[:-6]  # Remove "Config" suffix
    except (ImportError, AttributeError):
        # If registry not available, fall back to the old behavior
        base_step = class_name
        if base_step.endswith("Config"):
            base_step = base_step[:-6]  # Remove "Config" suffix
    
    step_name = base_step
    
    # Append distinguishing attributes (job_type, data_type, mode)
    for attr in ("job_type", "data_type", "mode"):
        if hasattr(config, attr):
            val = getattr(config, attr)
            if val is not None:
                step_name = f"{step_name}_{val}"
    
    return step_name
```

### 7. Save Method

Saves the merged configuration to a file:

```python
def save(self, output_file):
    """
    Save merged config to a file.
    
    Args:
        output_file: Path to output file
        
    Returns:
        dict: The merged configuration
    """
    merged = self.merge()
    
    # Create metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'config_types': {
            self._generate_step_name(c): c.__class__.__name__
            for c in self.config_list
        }
    }
    
    output = {'metadata': metadata, 'configuration': merged}
    
    try:
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, sort_keys=True)
        self.logger.info(f"Successfully wrote config to {output_file}")
    except Exception as e:
        self.logger.error(f"Error writing JSON: {str(e)}")
        
    return merged
```

## Output Format

The merged configuration follows this structure:

```json
{
  "metadata": {
    "created_at": "2025-07-18T09:15:30.123456",
    "config_types": {
      "CradleDataLoading_training": "CradleDataLoadConfig",
      "TabularPreprocessing_training": "TabularPreprocessingConfig",
      "XGBoostTraining": "XGBoostTrainingConfig"
    }
  },
  "configuration": {
    "shared": {
      "pipeline_name": "xgboost_training",
      "pipeline_version": "1.0.0",
      "region": "us-west-2",
      "bucket": "my-bucket"
    },
    "specific": {
      "CradleDataLoading_training": {
        "job_type": "training",
        "data_source_type": "s3"
      },
      "TabularPreprocessing_training": {
        "job_type": "training",
        "columns_to_drop": ["col1", "col2"]
      },
      "XGBoostTraining": {
        "hyperparameters": {
          "max_depth": 6,
          "eta": 0.3
        }
      }
    }
  }
}
```

## Usage Examples

### 1. Basic Usage

```python
# Create configs
cradle_config = CradleDataLoadConfig(job_type="training", region="us-west-2")
preprocess_config = TabularPreprocessingConfig(job_type="training", columns_to_drop=["col1"])
training_config = XGBoostTrainingConfig(hyperparameters={"max_depth": 6})

# Create merger
merger = ConfigMerger([cradle_config, preprocess_config, training_config])

# Save merged config
merged = merger.save("pipeline_config.json")

# Check structure
assert "shared" in merged
assert "specific" in merged
assert "CradleDataLoading_training" in merged["specific"]
```

### 2. With Custom Categorizer and Serializer

```python
# Create custom categorizer with specific rules
categorizer = ConfigFieldCategorizer(configs)
serializer = TypeAwareSerializer()

# Create merger with custom components
merger = ConfigMerger(configs, categorizer=categorizer, serializer=serializer)

# Merge and analyze
merged = merger.merge()

# Check categorization
for field, sources in categorizer.get_field_sources().items():
    print(f"Field '{field}' from sources: {sources}")
```

### 3. Checking Special Fields Handling

```python
# Config with special fields
training_config = XGBoostTrainingConfig(
    hyperparameters={"max_depth": 6},
    pipeline_name="xgboost_pipeline"
)

merger = ConfigMerger([training_config])
merged = merger.merge()

# Verify hyperparameters are in specific section
assert "hyperparameters" not in merged["shared"]
assert "hyperparameters" in merged["specific"]["XGBoostTraining"]

# Verify pipeline_name is in shared section
assert "pipeline_name" in merged["shared"]
```

## Benefits

1. **Consistent Field Organization**: Fields are consistently categorized using explicit rules
2. **Type Safety**: All values are properly serialized with type information
3. **Metadata Generation**: Creates helpful metadata for the configuration
4. **Special Field Handling**: Ensures special fields are always in the correct section
5. **Step Name Generation**: Uses registry-based step name generation for consistency
6. **Mutual Exclusivity**: Enforces mutual exclusivity between shared and specific sections

## Implementation Details

### 1. Dependency Injection

The ConfigMerger uses dependency injection to accept custom categorizers and serializers:

```python
def __init__(self, config_list, categorizer=None, serializer=None):
    self.config_list = config_list
    self.categorizer = categorizer or ConfigFieldCategorizer(config_list)
    self.serializer = serializer or TypeAwareSerializer()
```

This allows for flexible customization and easier testing.

### 2. Special Field Detection

The merger relies on a predefined list of special fields:

```python
SPECIAL_FIELDS_TO_KEEP_SPECIFIC = {
    "hyperparameters", 
    "hyperparameters_s3_uri",
    "job_name_prefix",
    "job_type"
}
```

These fields are always kept in the specific section for each step.

### 3. Step Name Generation

Step names are generated using a registry-based approach:

1. Check for step_name_override
2. Look up class name in CONFIG_STEP_REGISTRY
3. Fall back to removing "Config" suffix if not in registry
4. Append job_type, data_type, and mode attributes if present

### 4. Error Handling

The ConfigMerger includes robust error handling:

- Logs warnings when fields are moved from shared to specific
- Logs warnings when fields appear in both shared and specific
- Catches and logs exceptions during file writing

## Future Improvements

1. **Schema Validation**: Add schema validation before saving
2. **Automatic Field Detection**: Improve automatic detection of special fields
3. **Field Transformation**: Support field transformations during merging
4. **Versioning Support**: Add configuration versioning support
5. **Field Sources**: Add field source tracking to metadata

## References

- [Registry-Based Step Name Generation](./registry_based_step_name_generation.md)
- [Config Field Categorization](./simplified_config_field_categorization.md)
- [Type-Aware Serializer](./type_aware_serializer.md)
- [Job Type Variant Handling](./job_type_variant_handling.md)
