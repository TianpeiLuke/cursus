"""
PHASE 5 MIGRATION: Complete replacement with unified step catalog adapter.

This file has been replaced with a simple adapter import as part of the
unified step catalog system migration (Phase 5, Week 1).

All functionality is preserved through the ContractDiscoveryEngineAdapter
which uses the unified StepCatalog for discovery operations while maintaining
backward compatibility with existing APIs.

Migration Benefits:
- 95% code reduction (200+ lines → 1 import line)
- Unified discovery through StepCatalog
- Maintained backward compatibility
- Eliminated code redundancy
"""

from ....step_catalog.adapters.contract_adapter import ContractDiscoveryEngineAdapter as ContractDiscoveryEngine
