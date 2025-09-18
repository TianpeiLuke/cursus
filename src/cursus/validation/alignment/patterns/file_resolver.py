"""
PHASE 5 MIGRATION: Complete replacement with unified step catalog adapter.

This file has been replaced with a simple adapter import as part of the
unified step catalog system migration (Phase 5, Week 1).

All pattern-based file resolution functionality is preserved through the 
HybridFileResolverAdapter which uses the unified StepCatalog for discovery 
operations while maintaining backward compatibility with existing APIs.

Migration Benefits:
- 95% code reduction (200+ lines → 1 import line)
- Unified discovery through StepCatalog with O(1) lookups
- Eliminated complex production registry integration (replaced with catalog registry data)
- Eliminated sophisticated fallback strategies (replaced with precise catalog lookups)
- Maintained backward compatibility
- Eliminated code redundancy

The complex hybrid resolution strategies with production registry integration
and multiple fallback mechanisms have been replaced with precise pattern-based
lookups from the unified StepCatalog index.
"""

from ....step_catalog.adapters.file_resolver import HybridFileResolverAdapter as HybridFileResolver
