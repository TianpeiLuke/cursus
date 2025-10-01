"""
Factory Classes Module

This module contains factory classes for component selection and instantiation
in the alignment validation system. Factories provide intelligent selection
of appropriate components based on step types, frameworks, and other criteria.

Components:
- smart_spec_selector.py: Intelligent specification selection based on context
- step_type_detection.py: Step type detection and classification utilities
- step_type_enhancement_router.py: Router for step-type specific enhancements

Factory Pattern Benefits:
- Encapsulates component creation logic
- Enables dynamic selection based on runtime criteria
- Supports extensibility for new step types and frameworks
- Centralizes component instantiation decisions
"""

# Specification selection
from .smart_spec_selector import SmartSpecificationSelector

# Step type detection and routing
from .step_type_detection import (
    detect_step_type_from_registry,
    detect_framework_from_imports,
    detect_step_type_from_script_patterns,
    get_step_type_context,
)
from .step_type_enhancement_router import StepTypeEnhancementRouter

__all__ = [
    # Specification selection
    "SmartSpecificationSelector",
    
    # Step type detection and routing
    "detect_step_type_from_registry",
    "detect_framework_from_imports",
    "detect_step_type_from_script_patterns",
    "get_step_type_context",
    "StepTypeEnhancementRouter",
]
