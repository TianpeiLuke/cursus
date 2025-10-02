"""
Factory Classes Module

This module contains factory classes for component selection and instantiation
in the alignment validation system. Factories provide intelligent selection
of appropriate components based on step types, frameworks, and other criteria.

Components:
- smart_spec_selector.py: Intelligent specification selection based on context
- step_type_enhancement_router.py: Router for step-type specific enhancements

Note: Step type detection functionality has been consolidated into the registry
and step_catalog modules for better architecture and reduced redundancy.

Factory Pattern Benefits:
- Encapsulates component creation logic
- Enables dynamic selection based on runtime criteria
- Supports extensibility for new step types and frameworks
- Centralizes component instantiation decisions
"""

# Specification selection
from .smart_spec_selector import SmartSpecificationSelector

# Step type enhancement routing
from .step_type_enhancement_router import StepTypeEnhancementRouter

__all__ = [
    # Specification selection
    "SmartSpecificationSelector",
    
    # Step type enhancement routing
    "StepTypeEnhancementRouter",
]
