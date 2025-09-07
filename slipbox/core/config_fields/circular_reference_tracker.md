---
tags:
  - code
  - core
  - config_fields
  - circular_reference
  - object_tracking
keywords:
  - CircularReferenceTracker
  - circular reference detection
  - object graph traversal
  - deserialization tracking
  - reference tracking
topics:
  - configuration management
  - circular reference handling
  - object graph analysis
language: python
date of note: 2025-09-07
---

# Circular Reference Tracker

Dedicated data structure for tracking object references during deserialization, detecting circular references, and generating detailed diagnostic information about the path through the object graph.

## Overview

The `circular_reference_tracker` module provides a dedicated data structure for tracking object references during deserialization, detecting circular references, and generating detailed diagnostic information about the path through the object graph that led to the circular reference.

The tracker maintains a complete path through the object graph during traversal, enabling detailed diagnostic information when circular references are detected. It provides significantly enhanced error messages compared to simple set-based tracking, including the full path that led to the circular reference and context about the objects involved.

## Classes and Methods

### Classes
- [`CircularReferenceTracker`](#circularreferencetracker) - Main tracker for detecting and handling circular references

## API Reference

### CircularReferenceTracker

_class_ cursus.core.config_fields.circular_reference_tracker.CircularReferenceTracker(_max_depth=100_)

Tracks object references during deserialization to detect and handle circular references. This class maintains a complete path through the object graph during traversal, enabling detailed diagnostic information when circular references are detected.

**Parameters:**
- **max_depth** (_int_) – Maximum allowed depth in the object graph before considering it a potential infinite recursion

```python
from cursus.core.config_fields.circular_reference_tracker import CircularReferenceTracker

# Create tracker with default max depth
tracker = CircularReferenceTracker()

# Create tracker with custom max depth
tracker = CircularReferenceTracker(max_depth=50)
```

#### enter_object

enter_object(_obj_data_, _field_name=None_, _context=None_)

Start tracking a new object in the deserialization process.

**Parameters:**
- **obj_data** (_Any_) – The object being deserialized
- **field_name** (_Optional[str]_) – Name of the field containing this object
- **context** (_Optional[Dict[str, Any]]_) – Optional context information (e.g., parent object type)

**Returns:**
- **Tuple[bool, Optional[str]]** – (is_circular, error_message if any)

```python
# Track object during deserialization
obj_data = {"__model_type__": "DataSourceConfig", "name": "my_source"}
is_circular, error_msg = tracker.enter_object(obj_data, "data_source")

if is_circular:
    print(f"Circular reference detected: {error_msg}")
else:
    # Continue processing object
    # ... process object fields ...
    
    # Always exit when done
    tracker.exit_object()
```

#### exit_object

exit_object()

Mark that we've finished processing the current object. This must be called when the object is completely processed to maintain the correct state of the processing stack and current path.

```python
# Proper usage pattern
try:
    is_circular, error = tracker.enter_object(obj_data, field_name)
    if not is_circular:
        # Process object
        process_object_fields(obj_data)
finally:
    # Always exit, even if processing failed
    tracker.exit_object()
```

#### get_current_path_str

get_current_path_str()

Get string representation of the current object path.

**Returns:**
- **str** – A human-readable representation of the current path

```python
# Get current path for debugging
current_path = tracker.get_current_path_str()
print(f"Current object path: {current_path}")
# Output: "PipelineConfig() -> DataSourceConfig(name=my_source) -> MdsDataSourceConfig()"
```

## Related Documentation

- [Type-Aware Configuration Serializer](type_aware_config_serializer.md) - Primary consumer of the circular reference tracker
- [Configuration Fields Overview](README.md) - System overview and integration
