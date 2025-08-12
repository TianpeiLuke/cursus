---
tags:
  - code
  - core
  - config_merger
  - configuration_merging
  - conflict_resolution
keywords:
  - configuration merger
  - config merging
  - conflict resolution
  - merge directions
  - type-aware serialization
  - configuration validation
  - file operations
  - single source of truth
topics:
  - configuration merging
  - conflict resolution strategies
  - configuration validation
  - file I/O operations
language: python
date of note: 2025-08-12
---

# Configuration Merger

## Overview

The `ConfigMerger` is responsible for combining multiple configuration objects into a unified output structure. It uses categorization results from the `ConfigFieldCategorizer` to produce properly structured output files, implementing the **Single Source of Truth** principle by leveraging the categorizer's analysis.

## Class Definition

```python
class ConfigMerger:
    """
    Merger for combining multiple configuration objects into a unified output.
    
    Uses categorization results to produce properly structured output files.
    Implements the Explicit Over Implicit principle by clearly defining merge behavior.
    """
    
    def __init__(self, config_list: List[Any], processing_step_config_base_class: Optional[type] = None):
        """
        Initialize with list of config objects to merge.
        
        Args:
            config_list: List of configuration objects to merge
            processing_step_config_base_class: Optional base class for processing steps
        """
```

## Key Design Principles

### 1. Single Source of Truth

The merger delegates field categorization to the `ConfigFieldCategorizer`, ensuring consistent analysis:

```python
def __init__(self, config_list: List[Any], processing_step_config_base_class: Optional[type] = None):
    self.config_list = config_list
    self.logger = logging.getLogger(__name__)
    
    # Use ConfigFieldCategorizer to categorize fields - implementing Single Source of Truth
    self.logger.info(f"Categorizing fields for {len(config_list)} configs")
    self.categorizer = ConfigFieldCategorizer(config_list, processing_step_config_base_class)
    
    # Create serializer for saving output
    self.serializer = TypeAwareConfigSerializer()
```

### 2. Explicit Over Implicit

The merger clearly defines merge behavior and provides explicit conflict resolution strategies:

```python
@classmethod
def merge_with_direction(cls, source: Dict[str, Any], target: Dict[str, Any], 
                      direction: MergeDirection = MergeDirection.PREFER_SOURCE) -> Dict[str, Any]:
    """
    Merge two dictionaries with a specified merge direction.
    """
    if source_value != target_value:
        # Handle conflict based on direction
        if direction == MergeDirection.PREFER_SOURCE:
            result[key] = source_value
        elif direction == MergeDirection.PREFER_TARGET:
            pass  # Keep target value
        elif direction == MergeDirection.ERROR_ON_CONFLICT:
            raise ValueError(f"Conflict on key {key}: source={source_value}, target={target_value}")
```

### 3. Type-Safe Operations

The merger uses enum-based merge directions and integrates with type-aware serialization:

```python
from .constants import MergeDirection
from .type_aware_config_serializer import TypeAwareConfigSerializer
```

## Core Functionality

### Configuration Merging

The primary merge operation produces a simplified structure:

```python
def merge(self) -> Dict[str, Any]:
    """
    Merge configurations according to simplified categorization rules.
    
    Returns:
        dict: Merged configuration structure with just 'shared' and 'specific' sections
    """
    # Get categorized fields from categorizer - implementing Single Source of Truth
    categorized = self.categorizer.get_categorized_fields()
    
    # Create the merged output following the simplified structure
    merged = {
        "shared": categorized["shared"],
        "specific": categorized["specific"]
    }
    
    # Log statistics about the merged result
    shared_count = len(merged["shared"])
    specific_steps = len(merged["specific"])
    specific_fields = sum(len(fields) for step, fields in merged["specific"].items())
    
    self.logger.info(f"Merged result contains:")
    self.logger.info(f"  - {shared_count} shared fields")
    self.logger.info(f"  - {specific_steps} specific steps with {specific_fields} total fields")
    
    # Verify the merged result
    self._verify_merged_output(merged)
    
    return merged
```

### File Operations

The merger provides comprehensive file I/O operations:

```python
def save(self, output_file: str) -> Dict[str, Any]:
    """
    Save merged configuration to a file.
    
    Args:
        output_file: Path to output file
        
    Returns:
        dict: Merged configuration
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Merge configurations
    merged = self.merge()
    
    # Create metadata with proper step name -> class name mapping for config_types
    config_types = {}
    for cfg in self.config_list:
        step_name = self._generate_step_name(cfg)
        class_name = cfg.__class__.__name__
        config_types[step_name] = class_name
    
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
```

### Configuration Loading

The merger supports loading configurations from files with backward compatibility:

```python
@classmethod
def load(cls, input_file: str, config_classes: Optional[Dict[str, type]] = None) -> Dict[str, Any]:
    """
    Load a merged configuration from a file.
    
    Supports the simplified structure with just shared and specific sections.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading configuration from {input_file}")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Configuration file not found: {input_file}")
    
    # Load the JSON file
    with open(input_file, 'r') as f:
        file_data = json.load(f)
    
    # Check if we're dealing with the old format (with metadata and configuration keys)
    # or the new format (direct structure)
    if "configuration" in file_data and isinstance(file_data["configuration"], dict):
        # Old format - extract the actual configuration data
        logger.debug("Detected old configuration format with metadata wrapper")
        data = file_data["configuration"]
    else:
        # New format - direct structure
        logger.debug("Detected new configuration format (direct structure)")
        data = file_data
        
    # Create serializer
    serializer = TypeAwareConfigSerializer(config_classes=config_classes)
    
    # Process each section into the simplified structure
    result = {
        "shared": {},
        "specific": {}
    }
    
    # Deserialize shared fields
    if "shared" in data:
        for field, value in data["shared"].items():
            result["shared"][field] = serializer.deserialize(value)
    
    # Deserialize specific fields
    if "specific" in data:
        for step, fields in data["specific"].items():
            if step not in result["specific"]:
                result["specific"][step] = {}
            for field, value in fields.items():
                result["specific"][step][field] = serializer.deserialize(value)
                
    logger.info(f"Successfully loaded configuration from {input_file}")
    return result
```

## Verification and Validation

The merger includes comprehensive verification of merged results:

### Structure Verification

```python
def _verify_merged_output(self, merged: Dict[str, Any]) -> None:
    """
    Verify the merged output meets expectations for the simplified structure.
    """
    # Check structure has only shared and specific sections
    if set(merged.keys()) != {"shared", "specific"}:
        self.logger.warning(f"Merged structure has unexpected keys: {set(merged.keys())}. Expected 'shared' and 'specific' only.")
    
    # Check for mutual exclusivity violations
    self._check_mutual_exclusivity(merged)
    
    # Check for special fields in wrong sections
    self._check_special_fields_placement(merged)
    
    # Check for missing required fields
    self._check_required_fields(merged)
```

### Mutual Exclusivity Checking

```python
def _check_mutual_exclusivity(self, merged: Dict[str, Any]) -> None:
    """
    Check for field name collisions across categories in the simplified structure.
    """
    # Collect all field names by section
    shared_fields = set(merged["shared"].keys())
    
    specific_fields: Dict[str, Set[str]] = {}
    for step, fields in merged["specific"].items():
        specific_fields[step] = set(fields.keys())
        
    # Check for collisions between shared and specific sections
    for step, fields in specific_fields.items():
        collisions = shared_fields.intersection(fields)
        if collisions:
            self.logger.warning(f"Field name collision between shared and specific.{step}: {collisions}")
```

### Special Fields Validation

```python
def _check_special_fields_placement(self, merged: Dict[str, Any]) -> None:
    """
    Check that special fields are placed in specific sections in the simplified structure.
    """
    # Check shared section for special fields
    for field in merged["shared"]:
        if field in SPECIAL_FIELDS_TO_KEEP_SPECIFIC:
            self.logger.warning(f"Special field '{field}' found in shared section")
```

### Required Fields Analysis

```python
def _check_required_fields(self, merged: Dict[str, Any]) -> None:
    """
    Check that common fields are consistently placed across configurations.
    
    This method dynamically determines which fields appear in multiple config
    classes and checks their placement in the merged configuration.
    """
    if not self.config_list:
        return  # Nothing to check if no configs
        
    # Get all fields from each config
    config_fields = {}
    for config in self.config_list:
        step_name = self._generate_step_name(config)
        config_fields[step_name] = set(
            field for field in dir(config) 
            if not field.startswith('_') and not callable(getattr(config, field))
        )
    
    # Find fields that appear in all configs
    if config_fields:
        common_fields = set.intersection(*config_fields.values())
    else:
        common_fields = set()
        
    # Remove special fields that should stay in specific sections
    from .constants import SPECIAL_FIELDS_TO_KEEP_SPECIFIC        
    common_fields -= set(SPECIAL_FIELDS_TO_KEEP_SPECIFIC)
    
    # Fields that should be common (appear in multiple configs)
    potential_shared_fields = set()
    for field in set().union(*config_fields.values()):
        if field in SPECIAL_FIELDS_TO_KEEP_SPECIFIC:
            continue  # Skip special fields
            
        # Count configs that have this field
        count = sum(1 for fields in config_fields.values() if field in fields)
        if count > 1:
            # Field appears in multiple configs, should be shared
            potential_shared_fields.add(field)
    
    # Get fields from shared section
    shared_fields = set(merged["shared"].keys())
    
    # Find potentially shared fields that aren't in the shared section
    missing_from_shared = potential_shared_fields - shared_fields
    
    if not missing_from_shared:
        # All potential shared fields are already in shared section - perfect!
        return
        
    # Check if any of these fields appear in specific sections instead
    misplaced_fields = {}
    for step_name, fields in merged["specific"].items():
        step_fields = set(fields.keys())
        for field in missing_from_shared:
            if field in step_fields:
                if field not in misplaced_fields:
                    misplaced_fields[field] = []
                misplaced_fields[field].append(step_name)
    
    # Log fields that should be moved from specific to shared
    if misplaced_fields:
        for field, steps in misplaced_fields.items():
            self.logger.info(f"Field '{field}' appears in multiple configs but is in specific section(s): {steps}")
```

## Conflict Resolution

The merger provides flexible conflict resolution strategies:

### Merge Directions

```python
class MergeDirection(Enum):
    """
    Enumeration of merge directions.
    
    Specifies the direction to resolve conflicts when merging fields.
    """
    PREFER_SOURCE = auto()     # Use source value in case of conflict
    PREFER_TARGET = auto()     # Use target value in case of conflict
    ERROR_ON_CONFLICT = auto() # Raise an error on conflict
```

### Directional Merging

```python
@classmethod
def merge_with_direction(cls, source: Dict[str, Any], target: Dict[str, Any], 
                      direction: MergeDirection = MergeDirection.PREFER_SOURCE) -> Dict[str, Any]:
    """
    Merge two dictionaries with a specified merge direction.
    """
    result = target.copy()
    
    for key, source_value in source.items():
        if key not in result:
            # Key only in source, add it
            result[key] = source_value
        else:
            target_value = result[key]
            
            if isinstance(source_value, dict) and isinstance(target_value, dict):
                # Recursive merge for nested dictionaries
                result[key] = cls.merge_with_direction(source_value, target_value, direction)
            elif source_value != target_value:
                # Handle conflict based on direction
                if direction == MergeDirection.PREFER_SOURCE:
                    result[key] = source_value
                elif direction == MergeDirection.PREFER_TARGET:
                    pass  # Keep target value
                elif direction == MergeDirection.ERROR_ON_CONFLICT:
                    raise ValueError(f"Conflict on key {key}: source={source_value}, target={target_value}")
    
    return result
```

## Step Name Generation

The merger ensures consistent step naming across operations:

```python
def _generate_step_name(self, config: Any) -> str:
    """
    Generate a consistent step name for a config object using the pipeline registry.
    
    Args:
        config: Config object
        
    Returns:
        str: Step name
    """
    # Use the serializer's method to ensure consistency
    serializer = TypeAwareConfigSerializer()
    return serializer.generate_step_name(config)
```

## Usage Examples

### Basic Configuration Merging

```python
from src.cursus.core.config_fields.config_merger import ConfigMerger

# Create list of configuration objects
config_list = [data_config, preprocessing_config, training_config]

# Create merger
merger = ConfigMerger(config_list)

# Merge configurations
merged_config = merger.merge()

# Access merged structure
shared_fields = merged_config['shared']
specific_fields = merged_config['specific']

print(f"Shared fields: {list(shared_fields.keys())}")
for step_name, fields in specific_fields.items():
    print(f"Step {step_name}: {list(fields.keys())}")
```

### Save and Load Operations

```python
# Save merged configuration to file
merger = ConfigMerger(config_list)
merged_config = merger.save("output/merged_config.json")

# Load configuration from file
loaded_config = ConfigMerger.load("output/merged_config.json")

# Load with specific config classes for type restoration
config_classes = {
    'DataLoadingConfig': DataLoadingConfig,
    'PreprocessingConfig': PreprocessingConfig,
    'TrainingConfig': TrainingConfig
}
loaded_config = ConfigMerger.load("output/merged_config.json", config_classes)
```

### Directional Merging

```python
# Merge two configuration dictionaries with conflict resolution
source_config = {"field1": "value1", "field2": "value2"}
target_config = {"field1": "different_value", "field3": "value3"}

# Prefer source values in conflicts
merged = ConfigMerger.merge_with_direction(
    source_config, 
    target_config, 
    MergeDirection.PREFER_SOURCE
)
# Result: {"field1": "value1", "field2": "value2", "field3": "value3"}

# Prefer target values in conflicts
merged = ConfigMerger.merge_with_direction(
    source_config, 
    target_config, 
    MergeDirection.PREFER_TARGET
)
# Result: {"field1": "different_value", "field2": "value2", "field3": "value3"}

# Error on conflicts
try:
    merged = ConfigMerger.merge_with_direction(
        source_config, 
        target_config, 
        MergeDirection.ERROR_ON_CONFLICT
    )
except ValueError as e:
    print(f"Merge conflict: {e}")
```

### Convenience Functions

```python
from src.cursus.core.config_fields.config_merger import merge_and_save_configs, load_configs

# Convenience function for merge and save
merged_config = merge_and_save_configs(config_list, "output/merged_config.json")

# Convenience function for loading
loaded_config = load_configs("output/merged_config.json", config_classes)
```

## Output Structure

The merger produces a standardized output structure:

```json
{
  "metadata": {
    "created_at": "2025-08-12T09:54:00.000000",
    "config_types": {
      "data_loading": "DataLoadingConfig",
      "preprocessing": "PreprocessingConfig", 
      "training": "TrainingConfig"
    }
  },
  "configuration": {
    "shared": {
      "processing_instance_count": 1,
      "processing_volume_size": 30,
      "role": "arn:aws:iam::123456789012:role/SageMakerRole"
    },
    "specific": {
      "data_loading": {
        "cradle_endpoint": "https://cradle.example.com",
        "source_type": "EDX"
      },
      "preprocessing": {
        "image_uri": "123456789012.dkr.ecr.us-west-2.amazonaws.com/preprocessing:latest",
        "script_name": "preprocess.py"
      },
      "training": {
        "image_uri": "123456789012.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest",
        "hyperparameters": {
          "max_depth": 6,
          "eta": 0.3,
          "objective": "binary:logistic"
        }
      }
    }
  }
}
```

## Error Handling

The merger provides comprehensive error handling:

### File Operations

```python
def save(self, output_file: str) -> Dict[str, Any]:
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    try:
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, sort_keys=True)
    except IOError as e:
        self.logger.error(f"Failed to save configuration to {output_file}: {e}")
        raise

@classmethod
def load(cls, input_file: str, config_classes: Optional[Dict[str, type]] = None) -> Dict[str, Any]:
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Configuration file not found: {input_file}")
    
    try:
        with open(input_file, 'r') as f:
            file_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file {input_file}: {e}")
```

### Merge Conflicts

```python
elif direction == MergeDirection.ERROR_ON_CONFLICT:
    raise ValueError(f"Conflict on key {key}: source={source_value}, target={target_value}")
```

### Validation Errors

```python
def _verify_merged_output(self, merged: Dict[str, Any]) -> None:
    if set(merged.keys()) != {"shared", "specific"}:
        self.logger.warning(f"Merged structure has unexpected keys: {set(merged.keys())}")
```

## Integration with Other Components

### Configuration Field Categorizer

The merger relies on the categorizer for field analysis:

```python
self.categorizer = ConfigFieldCategorizer(config_list, processing_step_config_base_class)
categorized = self.categorizer.get_categorized_fields()
```

### Type-Aware Serializer

The merger uses the serializer for consistent data handling:

```python
self.serializer = TypeAwareConfigSerializer()
serializer = TypeAwareConfigSerializer(config_classes=config_classes)
result["shared"][field] = serializer.deserialize(value)
```

## Related Documentation

- [Configuration Field Categorizer](config_field_categorizer.md): Provides field categorization
- [Type-Aware Configuration Serializer](type_aware_config_serializer.md): Handles serialization
- [Configuration Constants](config_constants.md): Defines merge directions and patterns
- [Configuration Fields Overview](README.md): System overview and integration
