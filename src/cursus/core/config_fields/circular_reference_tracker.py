"""
CircularReferenceTracker Adapter - Simplified replacement for over-engineered tracker.

This adapter provides a simplified circular reference tracking system that replaces
the complex 600+ line CircularReferenceTracker with minimal tier-aware tracking.

MIGRATION: CircularReferenceTracker simplified (600 lines → ~70 lines)
- Eliminated complex graph analysis and sophisticated detection algorithms
- Replaced with simple tier-aware tracking based on three-tier architecture
- Maintained essential circular reference protection
- Removed unnecessary edge case handling and theoretical problem solving
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class SimplifiedCircularReferenceTracker:
    """
    Simplified circular reference tracker with tier-aware detection.
    
    Replaces the complex CircularReferenceTracker (600+ lines) with minimal
    tracking that leverages the three-tier architecture to prevent most
    circular references naturally.
    """
    
    def __init__(self, max_depth: int = 50):
        """
        Initialize simplified tracker.
        
        Args:
            max_depth: Maximum allowed depth (reduced from 100 to 50)
        """
        self.visited: Set[int] = set()
        self.processing_stack: List[str] = []
        self.max_depth = max_depth
        self.logger = logging.getLogger(__name__)
    
    def enter_object(
        self,
        obj_data: Any,
        field_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Start tracking a new object with simplified detection.
        
        Args:
            obj_data: The object being deserialized
            field_name: Name of the field containing this object
            context: Optional context information (ignored in simplified version)
            
        Returns:
            (bool, str): (is_circular, error_message if any)
        """
        # Check depth limit first
        if len(self.processing_stack) >= self.max_depth:
            error_msg = f"Maximum recursion depth ({self.max_depth}) exceeded at field {field_name}"
            self.logger.warning(error_msg)
            return True, error_msg
        
        # Simple circular reference detection for dictionaries with type info
        if isinstance(obj_data, dict) and "__model_type__" in obj_data:
            obj_id = id(obj_data)
            if obj_id in self.visited:
                error_msg = self._format_simple_cycle_error(obj_data, field_name)
                self.logger.warning(error_msg)
                return True, error_msg
            self.visited.add(obj_id)
        
        # Track processing stack for depth management
        self.processing_stack.append(field_name or "unknown")
        return False, None
    
    def exit_object(self) -> None:
        """Mark that we've finished processing the current object."""
        if self.processing_stack:
            self.processing_stack.pop()
    
    def get_current_path_str(self) -> str:
        """Get simple string representation of current path."""
        return " -> ".join(self.processing_stack)
    
    def reset(self) -> None:
        """Reset tracker state."""
        self.visited.clear()
        self.processing_stack.clear()
    
    def _get_module_from_step_catalog(self, type_name: str) -> str:
        """
        Get module name from step catalog system.
        
        Args:
            type_name: The model type name (e.g., "XGBoostTrainingConfig")
            
        Returns:
            str: Module name or "unknown" if not found
        """
        try:
            # Try to import the step catalog registry
            from ...registry.step_names import CONFIG_STEP_REGISTRY, get_step_names
            
            # First try to find the config class in the registry
            if type_name in CONFIG_STEP_REGISTRY:
                step_name = CONFIG_STEP_REGISTRY[type_name]
                step_info = get_step_names().get(step_name, {})
                
                # If we have step info, we can infer the module structure
                if step_info:
                    # Most config classes follow the pattern: cursus.steps.configs.config_*
                    return f"cursus.steps.configs.config_{step_name.lower()}"
            
            # Fallback: try to import the class directly to get its module
            try:
                from ...steps.configs.utils import build_complete_config_classes
                config_classes = build_complete_config_classes()
                
                if type_name in config_classes:
                    config_class = config_classes[type_name]
                    return getattr(config_class, '__module__', 'unknown')
            except ImportError:
                pass
                
        except ImportError:
            # Step catalog not available, fall back to unknown
            pass
        
        return "unknown"

    def _format_simple_cycle_error(self, obj_data: Any, field_name: Optional[str]) -> str:
        """
        Format a simple error message for circular reference.
        
        Args:
            obj_data: The object data that caused the circular reference
            field_name: The field name containing the object
            
        Returns:
            str: A simple error message
        """
        # Get object details
        if isinstance(obj_data, dict):
            type_name = obj_data.get("__model_type__", "unknown")
            module_name = self._get_module_from_step_catalog(type_name)
        else:
            type_name = type(obj_data).__name__
            module_name = getattr(type(obj_data), '__module__', 'unknown')
        
        # Current path
        current_path_str = self.get_current_path_str()
        
        return (
            f"Circular reference detected during model deserialization.\n"
            f"Object: {type_name} in {module_name}\n"
            f"Field: {field_name or 'unknown'}\n"
            f"Original definition path: {type_name}()\n"
            f"Reference path: {current_path_str}\n"
            f"This creates a cycle in the object graph."
        )


# Backward compatibility alias
CircularReferenceTracker = SimplifiedCircularReferenceTracker
