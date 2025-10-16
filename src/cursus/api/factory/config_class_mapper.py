"""
Configuration Class Mapper

This module provides mapping functionality between DAG nodes and their corresponding
configuration classes using the existing registry system. It handles the resolution
of DAG node names to Pydantic configuration classes.

Key Components:
- ConfigClassMapper: Maps DAG nodes to configuration classes using registry system
- Registry integration for automatic DAG node-to-config mapping
- Fallback mechanisms for cases where registry is unavailable
"""

from typing import Dict, Type, Optional, Any
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class ConfigClassMapper:
    """
    Maps DAG nodes to configuration classes using existing registry system.
    
    This class provides the bridge between DAG structure and configuration classes,
    enabling automatic discovery of which configuration class should be used for
    each step in a pipeline DAG.
    """
    
    def __init__(self):
        """Initialize mapper with registry system integration."""
        self.resolver_adapter = None
        self.config_classes = {}
        self._initialize_registry()
    
    def _initialize_registry(self):
        """Initialize registry system components with graceful fallback."""
        try:
            # Try to use UnifiedConfigManager for config class discovery
            from ...core.config_fields.unified_config_manager import get_unified_config_manager
            
            self.unified_manager = get_unified_config_manager()
            self.config_classes = self.unified_manager.get_config_classes()
            logger.info(f"Registry system initialized successfully with {len(self.config_classes)} config classes")
            
        except ImportError as e:
            logger.warning(f"UnifiedConfigManager not available: {e}")
            logger.info("Operating in fallback mode without registry")
        except Exception as e:
            logger.warning(f"Failed to initialize registry: {e}")
            logger.info("Operating in fallback mode without registry")
    
    def map_dag_to_config_classes(self, dag) -> Dict[str, Type[BaseModel]]:
        """
        Map DAG node names to configuration classes (not instances).
        
        Args:
            dag: Pipeline DAG object with nodes to map
            
        Returns:
            Dictionary mapping node names to configuration classes
        """
        config_class_map = {}
        
        # Get DAG nodes - handle different DAG implementations
        nodes = self._get_dag_nodes(dag)
        
        for node_name in nodes:
            config_class = self.resolve_node_to_config_class(node_name)
            if config_class:
                config_class_map[node_name] = config_class
            else:
                logger.warning(f"No configuration class found for DAG node: {node_name}")
        
        return config_class_map
    
    def resolve_node_to_config_class(self, node_name: str) -> Optional[Type[BaseModel]]:
        """
        Resolve a single DAG node to its configuration class.
        
        Args:
            node_name: Name of the DAG node to resolve
            
        Returns:
            Configuration class for the node, or None if not found
        """
        if self.resolver_adapter:
            try:
                # Use registry system to resolve config class
                config_class = self.resolver_adapter.get_config_class(node_name)
                if config_class:
                    return config_class
            except Exception as e:
                logger.warning(f"Registry resolution failed for {node_name}: {e}")
        
        # Fallback: try direct lookup in config classes
        if self.config_classes:
            # Try exact match first
            if node_name in self.config_classes:
                return self.config_classes[node_name]
            
            # Try pattern matching for common naming conventions
            for class_name, config_class in self.config_classes.items():
                if self._matches_node_pattern(node_name, class_name):
                    return config_class
        
        # Final fallback: try to infer from common patterns
        return self._infer_config_class_from_node_name(node_name)
    
    def _get_dag_nodes(self, dag) -> list:
        """
        Extract node names from DAG object, handling different DAG implementations.
        
        Args:
            dag: Pipeline DAG object
            
        Returns:
            List of node names in the DAG
        """
        # Handle different DAG implementations
        if hasattr(dag, 'nodes'):
            if callable(dag.nodes):
                return list(dag.nodes())
            else:
                return list(dag.nodes)
        elif hasattr(dag, 'get_nodes'):
            return dag.get_nodes()
        elif hasattr(dag, 'steps'):
            return list(dag.steps.keys()) if isinstance(dag.steps, dict) else list(dag.steps)
        else:
            logger.warning(f"Unknown DAG structure: {type(dag)}")
            return []
    
    def _matches_node_pattern(self, node_name: str, class_name: str) -> bool:
        """
        Check if a configuration class name matches a DAG node name pattern.
        
        Args:
            node_name: DAG node name
            class_name: Configuration class name
            
        Returns:
            True if the class name matches the node pattern
        """
        # Convert to lowercase for comparison
        node_lower = node_name.lower()
        class_lower = class_name.lower()
        
        # Direct substring match
        if node_lower in class_lower or class_lower in node_lower:
            return True
        
        # Remove common suffixes/prefixes for matching
        node_clean = node_lower.replace('_step', '').replace('step_', '').replace('_config', '')
        class_clean = class_lower.replace('config', '').replace('step', '')
        
        return node_clean in class_clean or class_clean in node_clean
    
    def _infer_config_class_from_node_name(self, node_name: str) -> Optional[Type[BaseModel]]:
        """
        Attempt to infer configuration class from node name using common patterns.
        
        Args:
            node_name: DAG node name to infer config class for
            
        Returns:
            Inferred configuration class, or None if cannot infer
        """
        # This is a fallback method for when registry is not available
        # In a real implementation, this would contain logic to map common
        # node name patterns to known configuration classes
        
        # For now, return None to indicate no inference possible
        logger.debug(f"Cannot infer configuration class for node: {node_name}")
        return None
    
    def get_available_config_classes(self) -> Dict[str, Type[BaseModel]]:
        """
        Get all available configuration classes from the registry.
        
        Returns:
            Dictionary of available configuration classes
        """
        return self.config_classes.copy()
    
    def register_manual_mapping(self, node_name: str, config_class: Type[BaseModel]) -> None:
        """
        Manually register a mapping between node name and configuration class.
        
        This is useful for cases where automatic resolution fails or for
        custom configuration classes not in the registry.
        
        Args:
            node_name: DAG node name
            config_class: Configuration class to map to the node
        """
        if not self.config_classes:
            self.config_classes = {}
        
        self.config_classes[node_name] = config_class
        logger.info(f"Manually registered mapping: {node_name} -> {config_class.__name__}")
    
    def validate_mapping(self, config_class_map: Dict[str, Type[BaseModel]]) -> Dict[str, str]:
        """
        Validate that all mapped configuration classes are valid Pydantic models.
        
        Args:
            config_class_map: Dictionary of node name to config class mappings
            
        Returns:
            Dictionary of validation errors (empty if all valid)
        """
        validation_errors = {}
        
        for node_name, config_class in config_class_map.items():
            try:
                # Check if it's a Pydantic model
                if not issubclass(config_class, BaseModel):
                    validation_errors[node_name] = f"Class {config_class.__name__} is not a Pydantic model"
                
                # Try to instantiate with empty args to check basic structure
                try:
                    config_class()
                except Exception as e:
                    # This is expected for classes with required fields
                    pass
                    
            except Exception as e:
                validation_errors[node_name] = f"Invalid configuration class: {e}"
        
        return validation_errors


class ManualConfigClassMapper(ConfigClassMapper):
    """
    Manual configuration class mapper for cases where registry is not available.
    
    This mapper allows explicit registration of node-to-config-class mappings
    without relying on the registry system.
    """
    
    def __init__(self, manual_mappings: Optional[Dict[str, Type[BaseModel]]] = None):
        """
        Initialize with manual mappings.
        
        Args:
            manual_mappings: Dictionary of node name to config class mappings
        """
        # Skip registry initialization
        self.resolver_adapter = None
        self.config_classes = manual_mappings or {}
    
    def add_mapping(self, node_name: str, config_class: Type[BaseModel]) -> None:
        """
        Add a manual mapping between node name and configuration class.
        
        Args:
            node_name: DAG node name
            config_class: Configuration class for the node
        """
        self.register_manual_mapping(node_name, config_class)
    
    def add_mappings(self, mappings: Dict[str, Type[BaseModel]]) -> None:
        """
        Add multiple manual mappings at once.
        
        Args:
            mappings: Dictionary of node name to config class mappings
        """
        for node_name, config_class in mappings.items():
            self.add_mapping(node_name, config_class)
