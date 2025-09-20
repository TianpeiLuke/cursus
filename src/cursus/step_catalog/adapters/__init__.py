"""
Step catalog adapters package.

This package provides backward compatibility adapters that maintain existing APIs
during the migration from legacy discovery systems to the unified StepCatalog system.

The adapters are organized by functionality:
- contract_discovery: Contract discovery and management adapters
- file_resolver: File resolution and discovery adapters  
- workspace_discovery: Workspace discovery and management adapters
- config_resolver: Configuration resolution adapters
- config_class_detector: Configuration class detection adapters
"""

# Contract discovery adapters
from .contract_adapter import (
    ContractDiscoveryResult,
    ContractDiscoveryEngineAdapter,
    ContractDiscoveryManagerAdapter,
)

# File resolver adapters
from .file_resolver import (
    FlexibleFileResolverAdapter,
    DeveloperWorkspaceFileResolverAdapter,
    HybridFileResolverAdapter,
)

# Workspace discovery adapters
from .workspace_discovery import (
    WorkspaceDiscoveryManagerAdapter,
)

# Config resolver adapters
from .config_resolver import (
    StepConfigResolverAdapter,
)

# Config class detector adapters
from .config_class_detector import (
    ConfigClassDetectorAdapter,
    ConfigClassStoreAdapter,
)

# Legacy wrapper adapters
from .legacy_wrappers import (
    LegacyDiscoveryWrapper,
    build_complete_config_classes,
    detect_config_classes_from_json,
)

__all__ = [
    # Contract discovery
    "ContractDiscoveryResult",
    "ContractDiscoveryEngineAdapter", 
    "ContractDiscoveryManagerAdapter",
    
    # File resolvers
    "FlexibleFileResolverAdapter",
    "DeveloperWorkspaceFileResolverAdapter", 
    "HybridFileResolverAdapter",
    
    # Workspace discovery
    "WorkspaceDiscoveryManagerAdapter",
    
    # Config resolution
    "StepConfigResolverAdapter",
    
    # Config class detection
    "ConfigClassDetectorAdapter",
    "ConfigClassStoreAdapter",
    
    # Legacy wrappers
    "LegacyDiscoveryWrapper",
    "build_complete_config_classes",
    "detect_config_classes_from_json",
]
