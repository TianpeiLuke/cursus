"""
PHASE 5 MIGRATION: Complete replacement with unified step catalog adapter.

This file has been replaced with a simple adapter import as part of the
unified step catalog system migration (Phase 5, Week 1).

All config class store functionality is preserved through the 
ConfigClassStoreAdapter which uses the unified StepCatalog for 
discovery operations while maintaining backward compatibility with existing APIs.

Migration Benefits:
- 95% code reduction (100+ lines → 1 import line)
- Unified discovery through StepCatalog with config class auto-discovery
- Eliminated TODO for auto-discovery (replaced with catalog's build_complete_config_classes)
- Maintained backward compatibility for registry functionality
- Eliminated code redundancy

Note: The TODO in build_complete_config_classes() function has been resolved by using
the unified StepCatalog's build_complete_config_classes() method, which provides
comprehensive config class discovery across all available sources.

The ConfigClassStore registry functionality is preserved through the adapter while
the build_complete_config_classes() function now uses the unified StepCatalog's
automated discovery capabilities.
"""

from ...step_catalog.adapters.config_class_detector import ConfigClassStoreAdapter as ConfigClassStore
from ...step_catalog.adapters.legacy_wrappers import build_complete_config_classes
