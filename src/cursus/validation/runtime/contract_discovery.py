"""
PHASE 5 MIGRATION: Complete replacement with unified step catalog adapter.

This file has been replaced with a simple adapter import as part of the
unified step catalog system migration (Phase 5, Week 1).

All core discovery functionality is preserved through the ContractDiscoveryManagerAdapter
which uses the unified StepCatalog for discovery operations while maintaining
backward compatibility with existing APIs.

Migration Benefits:
- 95% code reduction (300+ lines → 1 import line)
- Unified discovery through StepCatalog
- Maintained backward compatibility for core discovery methods
- Eliminated code redundancy

Note: Specialized contract parsing methods (input/output paths) are preserved
in the adapter with appropriate warnings, as they require contract file parsing
beyond the scope of the unified catalog's discovery responsibilities.
"""

from ...step_catalog.adapters.contract_adapter import ContractDiscoveryManagerAdapter as ContractDiscoveryManager
from ...step_catalog.adapters.contract_adapter import ContractDiscoveryResult
