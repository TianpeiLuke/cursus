---
tags:
  - code
  - core
  - circular_reference_tracker
  - reference_detection
  - serialization_safety
keywords:
  - circular reference detection
  - reference tracking
  - serialization safety
  - object tracking
  - depth limiting
  - reference cycles
  - deserialization protection
  - stack overflow prevention
topics:
  - circular reference detection
  - serialization safety
  - object reference tracking
  - deserialization protection
language: python
date of note: 2025-08-12
---

# Circular Reference Tracker

## Overview

The `CircularReferenceTracker` provides advanced circular reference detection and handling during configuration serialization and deserialization. It prevents infinite loops and stack overflow errors by tracking object references and maintaining context information about the reference chain.

## Class Definition

```python
class CircularReferenceTracker:
    """
    Advanced circular reference detection and handling for configuration serialization.
    
    Tracks object references during serialization/deserialization to prevent infinite loops
    and provides detailed context information for debugging circular reference issues.
    """
    
    def __init__(self, max_depth: int = 100):
        """
        Initialize the tracker with a maximum depth limit.
        
        Args:
            max_depth: Maximum depth to allow before considering it a circular reference
        """
```

## Key Features

### 1. Depth-Limited Tracking

The tracker prevents infinite recursion by limiting the maximum depth of object traversal:

```python
def __init__(self, max_depth: int = 100):
    self.max_depth = max_depth
    self.reference_stack = []
    self.visited_objects = set()
    self.depth_counter = 0
```

### 2. Context-Aware Detection

The tracker maintains context information about each object in the reference chain:

```python
def enter_object(self, obj: Any, field_name: Optional[str] = None, 
                context: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    """
    Enter an object for tracking, checking for circular references.
    
    Args:
        obj: Object being processed
        field_name: Optional field name for context
        context: Optional additional context information
        
    Returns:
        Tuple of (is_circular, error_message)
    """
```

### 3. Detailed Error Reporting

When circular references are detected, the tracker provides detailed error messages:

```python
def _generate_error_message(self, obj: Any, field_name: Optional[str], 
                          context: Optional[Dict[str, Any]]) -> str:
    """
    Generate a detailed error message for circular reference detection.
    """
    error_parts = ["Circular reference detected"]
    
    if field_name:
        error_parts.append(f"at field '{field_name}'")
    
    if context and 'expected_type' in context:
        error_parts.append(f"(expected type: {context['expected_type']})")
    
    error_parts.append(f"at depth {self.depth_counter}")
    
    if self.reference_stack:
        stack_info = " -> ".join([
            f"{item.get('field_name', 'unknown')}" 
            for item in self.reference_stack[-3:]  # Show last 3 items
        ])
        error_parts.append(f"Reference chain: ...{stack_info}")
    
    return ". ".join(error_parts)
```

## Core Functionality

### Object Entry Tracking

```python
def enter_object(self, obj: Any, field_name: Optional[str] = None, 
                context: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    """
    Enter an object for tracking, checking for circular references.
    """
    # Increment depth counter
    self.depth_counter += 1
    
    # Check depth limit
    if self.depth_counter > self.max_depth:
        error_msg = self._generate_error_message(obj, field_name, context)
        return True, f"Maximum depth exceeded ({self.max_depth}). {error_msg}"
    
    # Generate object identifier
    obj_id = self._get_object_identifier(obj)
    
    # Check if we've seen this object before
    if obj_id in self.visited_objects:
        error_msg = self._generate_error_message(obj, field_name, context)
        return True, error_msg
    
    # Add to tracking structures
    self.visited_objects.add(obj_id)
    
    stack_entry = {
        'obj_id': obj_id,
        'field_name': field_name,
        'context': context or {},
        'depth': self.depth_counter
    }
    self.reference_stack.append(stack_entry)
    
    return False, ""
```

### Object Exit Tracking

```python
def exit_object(self) -> None:
    """
    Exit the current object, removing it from tracking.
    
    This should be called when finished processing an object to clean up
    the tracking state and allow the object to be processed again in
    different contexts.
    """
    if self.reference_stack:
        exited_entry = self.reference_stack.pop()
        obj_id = exited_entry['obj_id']
        
        # Remove from visited objects to allow reprocessing in different contexts
        self.visited_objects.discard(obj_id)
    
    # Decrement depth counter
    if self.depth_counter > 0:
        self.depth_counter -= 1
```

### Object Identification

```python
def _get_object_identifier(self, obj: Any) -> str:
    """
    Generate a unique identifier for an object.
    
    Args:
        obj: Object to identify
        
    Returns:
        str: Unique identifier for the object
    """
    if obj is None:
        return "None"
    
    # For dictionaries, use a combination of id and structure
    if isinstance(obj, dict):
        # Include type information if available
        if '__model_type__' in obj:
            model_type = obj['__model_type__']
            return f"dict_{id(obj)}_{model_type}"
        return f"dict_{id(obj)}"
    
    # For other objects, use id and type
    return f"{type(obj).__name__}_{id(obj)}"
```

## Usage Examples

### Basic Usage with Type-Aware Serializer

```python
from src.cursus.core.config_fields.circular_reference_tracker import CircularReferenceTracker

class TypeAwareConfigSerializer:
    def __init__(self):
        self.ref_tracker = CircularReferenceTracker(max_depth=100)
    
    def deserialize(self, field_data: Any, field_name: Optional[str] = None, 
                    expected_type: Optional[Type] = None) -> Any:
        # Check for circular references
        context = {}
        if expected_type:
            context['expected_type'] = expected_type.__name__
        
        is_circular, error = self.ref_tracker.enter_object(field_data, field_name, context)
        
        try:
            if is_circular:
                self.logger.warning(error)
                return self._create_circular_reference_placeholder(field_data, error)
            
            # Process the object normally
            return self._deserialize_object(field_data, expected_type)
        
        finally:
            # Always exit the object when done
            self.ref_tracker.exit_object()
```

### Standalone Usage

```python
# Create tracker
tracker = CircularReferenceTracker(max_depth=50)

# Process objects with circular reference detection
def process_object_tree(obj, field_name=None):
    is_circular, error = tracker.enter_object(obj, field_name)
    
    try:
        if is_circular:
            print(f"Circular reference detected: {error}")
            return None
        
        # Process object normally
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                result[key] = process_object_tree(value, key)
            return result
        elif isinstance(obj, list):
            return [process_object_tree(item, f"{field_name}[{i}]") 
                   for i, item in enumerate(obj)]
        else:
            return obj
    
    finally:
        tracker.exit_object()

# Example usage
data = {
    'name': 'test',
    'nested': {
        'value': 42
    }
}

# Create circular reference
data['nested']['parent'] = data

# Process with detection
result = process_object_tree(data, 'root')
```

### Context-Rich Tracking

```python
def deserialize_with_context(self, field_data: Any, field_name: str, 
                           expected_type: Type) -> Any:
    # Provide rich context for better error messages
    context = {
        'expected_type': expected_type.__name__,
        'field_path': self._get_current_field_path(),
        'operation': 'deserialization',
        'timestamp': datetime.now().isoformat()
    }
    
    is_circular, error = self.ref_tracker.enter_object(field_data, field_name, context)
    
    try:
        if is_circular:
            # Log detailed error with context
            self.logger.error(f"Circular reference in deserialization: {error}")
            self.logger.debug(f"Context: {context}")
            
            # Create appropriate placeholder
            return self._create_context_aware_placeholder(field_data, context, error)
        
        # Normal processing
        return self._deserialize_normal(field_data, expected_type)
    
    finally:
        self.ref_tracker.exit_object()
```

## Error Handling and Recovery

### Circular Reference Placeholders

When circular references are detected, the tracker helps create appropriate placeholders:

```python
def _create_circular_reference_placeholder(self, field_data: Any, error: str) -> Dict[str, Any]:
    """
    Create a placeholder object for circular references.
    """
    placeholder = {
        "__circular_ref__": True,
        "error": error,
        "detection_time": datetime.now().isoformat()
    }
    
    # Include type information if available
    if isinstance(field_data, dict):
        if '__model_type__' in field_data:
            placeholder['original_type'] = field_data['__model_type__']
        if '__model_module__' in field_data:
            placeholder['original_module'] = field_data['__model_module__']
    
    return placeholder
```

### Graceful Degradation

```python
def _handle_circular_reference_gracefully(self, obj: Any, error: str, 
                                        expected_type: Optional[Type] = None) -> Any:
    """
    Handle circular references gracefully with appropriate fallbacks.
    """
    self.logger.warning(f"Circular reference detected: {error}")
    
    # Try to create a minimal valid object if expected_type is provided
    if expected_type and hasattr(expected_type, 'model_construct'):
        try:
            # Create minimal object with required fields
            minimal_data = self._get_minimal_required_fields(expected_type)
            minimal_data['__circular_ref__'] = True
            return expected_type.model_construct(**minimal_data)
        except Exception as e:
            self.logger.warning(f"Failed to create minimal object: {e}")
    
    # Fall back to placeholder dictionary
    return {
        "__circular_ref__": True,
        "error": error,
        "original_type": type(obj).__name__ if obj is not None else "None"
    }
```

## Performance Considerations

### Memory Efficiency

The tracker is designed to minimize memory usage:

```python
def _cleanup_old_entries(self) -> None:
    """
    Clean up old entries to prevent memory leaks in long-running processes.
    """
    # Keep only recent entries in the stack
    if len(self.reference_stack) > self.max_depth * 2:
        # Remove oldest entries but keep recent context
        self.reference_stack = self.reference_stack[-self.max_depth:]
        
        # Rebuild visited_objects set from remaining stack
        self.visited_objects = {
            entry['obj_id'] for entry in self.reference_stack
        }
```

### Processing Efficiency

The tracker uses efficient data structures and algorithms:

```python
def _get_object_identifier(self, obj: Any) -> str:
    """
    Efficient object identification using built-in id() function.
    """
    # Use Python's built-in id() for O(1) object identification
    if isinstance(obj, dict) and '__model_type__' in obj:
        # Include type for better debugging, but still O(1)
        return f"dict_{id(obj)}_{obj['__model_type__']}"
    
    return f"{type(obj).__name__}_{id(obj)}"
```

## Integration with Configuration System

### Serializer Integration

```python
class TypeAwareConfigSerializer:
    def __init__(self):
        # Initialize with reasonable depth limit
        self.ref_tracker = CircularReferenceTracker(max_depth=100)
    
    def deserialize(self, field_data: Any, field_name: Optional[str] = None, 
                    expected_type: Optional[Type] = None) -> Any:
        # Use tracker for all deserialization operations
        context = {'expected_type': expected_type.__name__} if expected_type else {}
        is_circular, error = self.ref_tracker.enter_object(field_data, field_name, context)
        
        try:
            if is_circular:
                return self._handle_circular_reference(field_data, error, expected_type)
            
            # Normal deserialization logic
            return self._deserialize_normal(field_data, expected_type)
        finally:
            self.ref_tracker.exit_object()
```

### Configuration Merger Integration

```python
class ConfigMerger:
    def __init__(self):
        self.ref_tracker = CircularReferenceTracker(max_depth=50)
    
    def _merge_nested_objects(self, source: Any, target: Any, path: str = "") -> Any:
        """
        Merge nested objects with circular reference protection.
        """
        is_circular, error = self.ref_tracker.enter_object(source, path)
        
        try:
            if is_circular:
                self.logger.warning(f"Circular reference in merge at {path}: {error}")
                return target  # Keep target value when circular reference detected
            
            # Normal merge logic
            return self._perform_merge(source, target)
        finally:
            self.ref_tracker.exit_object()
```

## Configuration and Tuning

### Depth Limit Configuration

```python
# For simple configurations
tracker = CircularReferenceTracker(max_depth=20)

# For complex nested configurations
tracker = CircularReferenceTracker(max_depth=100)

# For very deep object hierarchies
tracker = CircularReferenceTracker(max_depth=500)
```

### Context Configuration

```python
def create_context_for_field(field_name: str, expected_type: Type, 
                           operation: str) -> Dict[str, Any]:
    """
    Create rich context information for tracking.
    """
    return {
        'field_name': field_name,
        'expected_type': expected_type.__name__ if expected_type else 'unknown',
        'operation': operation,
        'timestamp': datetime.now().isoformat(),
        'thread_id': threading.current_thread().ident
    }
```

## Debugging and Monitoring

### Debug Information

```python
def get_tracking_state(self) -> Dict[str, Any]:
    """
    Get current tracking state for debugging.
    """
    return {
        'depth': self.depth_counter,
        'max_depth': self.max_depth,
        'stack_size': len(self.reference_stack),
        'visited_count': len(self.visited_objects),
        'current_stack': [
            {
                'field_name': entry.get('field_name', 'unknown'),
                'depth': entry.get('depth', 0),
                'obj_type': entry.get('context', {}).get('expected_type', 'unknown')
            }
            for entry in self.reference_stack[-5:]  # Last 5 entries
        ]
    }
```

### Monitoring Integration

```python
def track_with_monitoring(self, obj: Any, field_name: str, 
                         monitor_callback: Optional[Callable] = None) -> Tuple[bool, str]:
    """
    Track object with optional monitoring callback.
    """
    is_circular, error = self.enter_object(obj, field_name)
    
    if monitor_callback:
        monitor_callback({
            'event': 'circular_reference_check',
            'field_name': field_name,
            'is_circular': is_circular,
            'depth': self.depth_counter,
            'error': error if is_circular else None
        })
    
    return is_circular, error
```

## Related Documentation

- [Type-Aware Configuration Serializer](type_aware_config_serializer.md): Primary user of circular reference tracking
- [Configuration Merger](config_merger.md): Uses tracking for safe nested object merging
- [Configuration Field Categorizer](config_field_categorizer.md): May use tracking for complex field analysis
- [Configuration Fields Overview](README.md): System overview and integration
