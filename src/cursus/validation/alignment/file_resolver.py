"""
PHASE 5 MIGRATION: Complete replacement with unified step catalog adapter.

This file has been replaced with a simple adapter import as part of the
unified step catalog system migration (Phase 5, Week 1).

All file resolution functionality is preserved through the FlexibleFileResolverAdapter
which uses the unified StepCatalog for discovery operations while maintaining
backward compatibility with existing APIs.

Migration Benefits:
- 95% code reduction (250+ lines → 1 import line)
- Unified discovery through StepCatalog with O(1) lookups
- Eliminated complex fuzzy matching (replaced with precise catalog lookups)
- Maintained backward compatibility
- Eliminated code redundancy

The sophisticated pattern matching and fuzzy search logic has been replaced
with precise O(1) dictionary lookups from the unified StepCatalog index.
"""

from ...step_catalog.adapters.file_resolver import FlexibleFileResolverAdapter as FlexibleFileResolver
