"""
Circular Reference Tracker for detecting and handling circular references in object graphs.

This module provides a dedicated data structure for tracking object references during
deserialization, detecting circular references, and generating detailed diagnostic information
about the path through the object graph that led to the circular reference.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class CircularReferenceTracker:
    """
    Tracks object references during deserialization to detect and handle circular references.
    
    This class maintains a complete path through the object graph during traversal, enabling
    detailed diagnostic information when circular references are detected. It provides
    significantly enhanced error messages compared to simple set-based tracking.
    """
    
    def __init__(self, max_depth: int = 100):
        """
        Initialize the tracker with a maximum recursion depth.
        
        Args:
            max_depth: Maximum allowed depth in the object graph before
                       considering it a potential infinite recursion
        """
        self.processing_stack: List[Dict[str, Any]] = []  # Stack of currently processing objects
        self.object_id_to_path: Dict[Any, List[Dict[str, Any]]] = {}  # Maps object IDs to their path
        self.current_path: List[Dict[str, Any]] = []  # Current path in the object graph
        self.max_depth = max_depth
        self.logger = logging.getLogger(__name__)
    
    def enter_object(self, obj_data: Any, field_name: Optional[str] = None,
                    context: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[str]]:
        """
        Start tracking a new object in the deserialization process.
        
        Args:
            obj_data: The object being deserialized
            field_name: Name of the field containing this object
            context: Optional context information (e.g., parent object type)
            
        Returns:
            (bool, str): (is_circular, error_message if any)
        """
        # Check depth limit first
        if len(self.current_path) >= self.max_depth:
            error_msg = self._format_depth_error(field_name)
            self.logger.error(error_msg)
            return True, error_msg
        
        # Generate object ID, passing along field name for better context
        obj_id = self._generate_object_id(obj_data, field_name)
        
        # Check for circular reference
        if obj_id in self.object_id_to_path:
            error_msg = self._format_cycle_error(obj_data, field_name, obj_id)
            self.logger.warning(error_msg)
            return True, error_msg
            
        # Update tracking
        node_info = {
            'id': obj_id,
            'type': obj_data.get('__model_type__', 'unknown') if isinstance(obj_data, dict) else str(type(obj_data).__name__),
            'module': obj_data.get('__model_module__', 'unknown') if isinstance(obj_data, dict) else '',
            'field_name': field_name,
            'context': context or {}
        }
        
        # Add identifying information if available
        if isinstance(obj_data, dict):
            for id_field in ['name', 'pipeline_name', 'id', 'step_name']:
                if id_field in obj_data and isinstance(obj_data[id_field], (str, int, float, bool)):
                    node_info['identifier'] = f"{id_field}={obj_data[id_field]}"
                    break
                
        self.processing_stack.append(node_info)
        self.current_path.append(node_info)
        self.object_id_to_path[obj_id] = list(self.current_path)  # Copy current path
        
        return False, None
        
    def exit_object(self) -> None:
        """
        Mark that we've finished processing the current object.
        
        This must be called when the object is completely processed to maintain
        the correct state of the processing stack and current path.
        """
        if self.processing_stack:
            node = self.processing_stack.pop()
            if self.current_path:
                self.current_path.pop()
            
    def get_current_path_str(self) -> str:
        """
        Get string representation of the current object path.
        
        Returns:
            str: A human-readable representation of the current path
        """
        return ' -> '.join(
            f"{node['type']}({node.get('identifier', '')})"
            for node in self.current_path
        )
            
    def _format_cycle_error(self, obj_data: Any, field_name: Optional[str], 
                          obj_id: Any) -> str:
        """
        Format a detailed error message for circular reference.
        
        Args:
            obj_data: The object data that caused the circular reference
            field_name: The field name containing the object
            obj_id: The object identifier
            
        Returns:
            str: A detailed error message
        """
        # Get the original path where this object was first seen
        original_path = self.object_id_to_path.get(obj_id, [])
        original_path_str = ' -> '.join(
            f"{node['type']}({node.get('identifier', '')})"
            for node in original_path
        )
        
        # Current path to this reference
        current_path_str = self.get_current_path_str()
        
        # Object details
        if isinstance(obj_data, dict):
            type_name = obj_data.get('__model_type__', 'unknown_type')
            module_name = obj_data.get('__model_module__', 'unknown_module')
        else:
            type_name = type(obj_data).__name__
            module_name = type(obj_data).__module__
        
        # Format the error
        return (f"Circular reference detected during model deserialization.\n"
                f"Object: {type_name} in {module_name}\n"
                f"Field: {field_name or 'unknown'}\n"
                f"Original definition path: {original_path_str}\n"
                f"Reference path: {current_path_str}\n"
                f"This creates a cycle in the object graph.")
                
    def _format_depth_error(self, field_name: Optional[str]) -> str:
        """
        Format a detailed error message for maximum recursion depth.
        
        Args:
            field_name: The field name being deserialized
            
        Returns:
            str: A detailed error message
        """
        path_str = self.get_current_path_str()
        
        return (f"Maximum recursion depth ({self.max_depth}) exceeded while deserializing {field_name or 'unknown'}\n"
                f"Current path: {path_str}\n"
                f"This suggests a potential circular reference or extremely nested structure.")
                
    def _generate_object_id(self, obj_data: Any, field_name: Optional[str] = None) -> Any:
        """
        Generate a reliable ID for an object to detect circular refs.
        Enhanced to avoid false positives for list items and common types.
        
        Args:
            obj_data: The object to identify
            field_name: The field name containing this object (may include array indices)
            
        Returns:
            Any: An identifier for the object
        """
        # For non-dict objects, use memory address
        if not isinstance(obj_data, dict):
            return id(obj_data)  # Fallback for non-dict objects
            
        # For dictionaries with model type info, create a more precise composite ID
        type_name = obj_data.get('__model_type__')
        if not type_name:
            return id(obj_data)  # No type info, use object ID
            
        # Build context-aware ID parts
        id_parts = [type_name]
        
        # Add field name context (including array indices) to distinguish list items
        if field_name:
            # If this is a list item, explicitly include the index in the ID to avoid 
            # false positives between different items in the same list
            if '[' in str(field_name):
                id_parts.append(f"list_item:{field_name}")
        
        # Include more discriminating fields for certain object types
        # For DataSourceConfig, include data_source_name as a primary identifier
        if type_name == "DataSourceConfig" or type_name.endswith(".DataSourceConfig"):
            for key in ['data_source_name', 'data_source_type']:
                if key in obj_data and isinstance(obj_data[key], (str, int, float, bool)):
                    id_parts.append(f"{key}:{obj_data[key]}")
        
        # Add other key identifiers if available
        for key in ['name', 'pipeline_name', 'id', 'step_name']:
            if key in obj_data and isinstance(obj_data[key], (str, int, float, bool)):
                id_parts.append(f"{key}:{obj_data[key]}")
        
        # Include current path depth to help distinguish objects at different nesting levels
        if self.current_path:
            id_parts.append(f"depth:{len(self.current_path)}")
                
        return hash(tuple(id_parts))
