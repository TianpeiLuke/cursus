"""
DAG Configuration Factory API

This module provides interactive pipeline configuration generation through a step-by-step workflow.
The system leverages existing Pydantic field definitions and the three-tier configuration system
to create user-friendly configuration interfaces for pipeline development.

Key Components:
- DAGConfigFactory: Main interactive factory for step-by-step configuration generation
- ConfigClassMapper: Maps DAG nodes to configuration classes using registry system
- ConfigurationGenerator: Generates final configuration instances with base config inheritance
- Field requirement extraction utilities for Pydantic classes

Usage:
    from cursus.api.factory import DAGConfigFactory
    
    # Create factory from DAG
    factory = DAGConfigFactory(dag)
    
    # Get base configuration requirements
    base_requirements = factory.get_base_config_requirements()
    
    # Set base configuration
    factory.set_base_config(**base_values)
    
    # Configure steps interactively
    for step_name in factory.get_pending_steps():
        step_requirements = factory.get_step_requirements(step_name)
        factory.set_step_config(step_name, **step_values)
    
    # Generate final configurations
    configs = factory.generate_all_configs()
"""

from .dag_config_factory import DAGConfigFactory, ConfigurationIncompleteError
from .config_class_mapper import ConfigClassMapper
from .configuration_generator import ConfigurationGenerator
from .field_extractor import extract_field_requirements, print_field_requirements, categorize_field_requirements

__all__ = [
    'DAGConfigFactory',
    'ConfigClassMapper', 
    'ConfigurationGenerator',
    'ConfigurationIncompleteError',
    'extract_field_requirements',
    'print_field_requirements',
    'categorize_field_requirements'
]
