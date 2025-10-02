"""
Factory Classes Module

This module contains factory classes for component selection and instantiation
in the alignment validation system. Factories provide intelligent selection
of appropriate components based on step types, frameworks, and other criteria.

PHASE 2 ENHANCEMENT: SmartSpecificationSelector Integration Complete
- SmartSpecificationSelector functionality integrated into StepCatalog/SpecAutoDiscovery
- Multi-variant specification handling now available via StepCatalog methods:
  * StepCatalog.create_unified_specification() - Creates unified spec from variants
  * StepCatalog.validate_logical_names_smart() - Smart validation with detailed feedback
- Eliminated ~100 lines of redundant discovery logic
- Enhanced with registry patterns and workspace-aware discovery

Remaining Components:
- step_type_enhancement_router.py: Router for step-type specific enhancements

Consolidation History:
- PHASE 1: Step type detection functionality consolidated into registry/step_catalog
- PHASE 2: Smart specification selection integrated into SpecAutoDiscovery

Factory Pattern Benefits:
- Encapsulates component creation logic
- Enables dynamic selection based on runtime criteria
- Supports extensibility for new step types and frameworks
- Centralizes component instantiation decisions
"""

# PHASE 2 ENHANCEMENT: SmartSpecificationSelector functionality integrated into StepCatalog
# Removed smart_spec_selector import - functionality now available via StepCatalog methods

# Step type enhancement routing
from .step_type_enhancement_router import StepTypeEnhancementRouter

__all__ = [
    # PHASE 2 ENHANCEMENT: SmartSpecificationSelector removed - functionality integrated into StepCatalog
    # Use StepCatalog.create_unified_specification() and StepCatalog.validate_logical_names_smart() instead
    
    # Step type enhancement routing
    "StepTypeEnhancementRouter",
]
