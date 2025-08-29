---
tags:
  - design
  - configuration
  - workspace
  - multi-developer
  - architecture
  - three-tier
keywords:
  - workspace-aware config management
  - multi-developer configuration
  - workspace-scoped field categorization
  - workspace config merging
  - distributed config registry
  - workspace tier registry
topics:
  - workspace management
  - configuration management
  - multi-developer architecture
  - field categorization
  - config serialization
language: python
date of note: 2025-08-29
---

# Workspace-Aware Config Manager Design

## Overview

The Workspace-Aware Config Manager extends the existing configuration management system in `src/cursus/core/config_fields/` to support multi-developer workspace environments. This design addresses a critical gap in the current workspace implementation by enabling workspace-scoped configuration management, field categorization, and merging capabilities.

## Problem Statement

The current config management system operates globally without workspace awareness:

1. **ConfigFieldCategorizer** categorizes fields into "shared" and "specific" sections using global analysis
2. **ConfigMerger** merges configurations without workspace context
3. **ConfigFieldTierRegistry** uses static tier classifications without workspace-specific overrides
4. **TypeAwareConfigSerializer** generates step names without workspace context

This creates issues for multi-developer environments where:
- Different workspaces may have different shared/specific field patterns
- Workspace-specific configurations need isolated merging
- Cross-workspace dependencies require proper resolution
- Workspace-specific tier overrides are needed

## Design Principles

### Core Architectural Principles
- **Workspace Isolation**: Configuration management respects workspace boundaries
- **Backward Compatibility**: Existing non-workspace code continues to work unchanged
- **Single Source of Truth**: Workspace-aware components extend, don't replace, existing registries
- **Explicit Over Implicit**: Clear workspace context detection and propagation

### Extension Strategy
- **Composition Over Inheritance**: Workspace components wrap existing components
- **Context Propagation**: Workspace context flows through the entire config management pipeline
- **Graceful Degradation**: System works without workspace context (backward compatibility)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Workspace-Aware Config Manager               │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ WorkspaceConfig │  │ WorkspaceConfig │  │ WorkspaceConfig │  │
│  │ FieldCategorizer│  │     Merger      │  │ TierRegistry    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│           │                     │                     │          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ WorkspaceConfig │  │ WorkspaceType   │  │ WorkspaceConfig │  │
│  │   Serializer    │  │   Aware         │  │   Context       │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                     Existing Config System                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ ConfigField     │  │ ConfigMerger    │  │ ConfigFieldTier │  │
│  │ Categorizer     │  │                 │  │   Registry      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│  ┌─────────────────┐  ┌─────────────────┐                      │
│  │ TypeAwareConfig │  │ CircularRef     │                      │
│  │   Serializer    │  │   Tracker       │                      │
│  └─────────────────┘  └─────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. WorkspaceConfigContext

Central context manager for workspace-aware configuration operations.

```python
# File: src/cursus/core/config_fields/workspace_config_context.py
import contextvars
from typing import Optional, Dict, Any, ContextManager
from contextlib import contextmanager
from dataclasses import dataclass

@dataclass
class WorkspaceConfigInfo:
    """Information about a workspace configuration context."""
    workspace_id: str
    workspace_root: str
    developer_id: Optional[str] = None
    workspace_type: str = "single"  # "single" or "multi"
    metadata: Dict[str, Any] = None

# Thread-local workspace context
_workspace_config_context: contextvars.ContextVar[Optional[WorkspaceConfigInfo]] = \
    contextvars.ContextVar('workspace_config_context', default=None)

class WorkspaceConfigContext:
    """Context manager for workspace-aware configuration operations."""
    
    @classmethod
    def get_current(cls) -> Optional[WorkspaceConfigInfo]:
        """Get the current workspace configuration context."""
        return _workspace_config_context.get()
    
    @classmethod
    def set_current(cls, workspace_info: WorkspaceConfigInfo) -> None:
        """Set the current workspace configuration context."""
        _workspace_config_context.set(workspace_info)
    
    @classmethod
    def clear_current(cls) -> None:
        """Clear the current workspace configuration context."""
        _workspace_config_context.set(None)
    
    @classmethod
    @contextmanager
    def workspace_context(cls, workspace_info: WorkspaceConfigInfo) -> ContextManager[None]:
        """Context manager for temporary workspace context."""
        old_context = cls.get_current()
        try:
            cls.set_current(workspace_info)
            yield
        finally:
            if old_context:
                cls.set_current(old_context)
            else:
                cls.clear_current()
    
    @classmethod
    def from_workspace_root(cls, workspace_root: str, developer_id: str = None) -> WorkspaceConfigInfo:
        """Create workspace info from workspace root path."""
        workspace_id = f"{developer_id}@{workspace_root}" if developer_id else workspace_root
        return WorkspaceConfigInfo(
            workspace_id=workspace_id,
            workspace_root=workspace_root,
            developer_id=developer_id,
            workspace_type="multi" if developer_id else "single"
        )
```

### 2. WorkspaceConfigFieldCategorizer

Workspace-aware extension of ConfigFieldCategorizer.

```python
# File: src/cursus/core/config_fields/workspace_config_field_categorizer.py
from typing import Dict, List, Any, Optional, Set
import logging
from collections import defaultdict

from .config_field_categorizer import ConfigFieldCategorizer, CategoryType
from .workspace_config_context import WorkspaceConfigContext, WorkspaceConfigInfo
from .constants import SPECIAL_FIELDS_TO_KEEP_SPECIFIC

class WorkspaceConfigFieldCategorizer:
    """
    Workspace-aware field categorizer that handles workspace-scoped categorization.
    
    Extends the base ConfigFieldCategorizer to support:
    1. Workspace-scoped field analysis
    2. Cross-workspace field sharing detection
    3. Workspace-specific categorization rules
    4. Developer-specific field patterns
    """
    
    def __init__(self, 
                 workspace_configs: Dict[str, List[Any]], 
                 processing_step_config_base_class: Optional[type] = None,
                 workspace_context: Optional[WorkspaceConfigInfo] = None):
        """
        Initialize with workspace-organized configurations.
        
        Args:
            workspace_configs: Dict mapping workspace_id to list of configs
            processing_step_config_base_class: Base class for processing steps
            workspace_context: Optional workspace context
        """
        self.workspace_configs = workspace_configs
        self.workspace_context = workspace_context or WorkspaceConfigContext.get_current()
        self.logger = logging.getLogger(__name__)
        
        # Create individual categorizers for each workspace
        self.workspace_categorizers: Dict[str, ConfigFieldCategorizer] = {}
        for workspace_id, configs in workspace_configs.items():
            if configs:  # Only create categorizer if there are configs
                self.workspace_categorizers[workspace_id] = ConfigFieldCategorizer(
                    configs, processing_step_config_base_class
                )
        
        # Analyze cross-workspace patterns
        self.cross_workspace_analysis = self._analyze_cross_workspace_patterns()
        
        # Generate workspace-aware categorization
        self.workspace_categorization = self._categorize_workspace_fields()
    
    def _analyze_cross_workspace_patterns(self) -> Dict[str, Any]:
        """
        Analyze field patterns across workspaces.
        
        Returns:
            Dict containing cross-workspace analysis results
        """
        analysis = {
            'field_workspace_map': defaultdict(set),  # field_name -> set of workspace_ids
            'shared_across_workspaces': set(),        # fields shared across all workspaces
            'workspace_specific_fields': defaultdict(set),  # workspace_id -> set of unique fields
            'field_value_consistency': defaultdict(dict),   # field_name -> workspace_id -> values
            'workspace_field_stats': {}                     # workspace_id -> field statistics
        }
        
        # Collect field information from all workspaces
        for workspace_id, categorizer in self.workspace_categorizers.items():
            field_info = categorizer.field_info
            workspace_fields = set(field_info['sources'].keys())
            
            # Track which workspaces have which fields
            for field_name in workspace_fields:
                analysis['field_workspace_map'][field_name].add(workspace_id)
                
                # Track field values for consistency analysis
                if field_name not in analysis['field_value_consistency']:
                    analysis['field_value_consistency'][field_name] = {}
                analysis['field_value_consistency'][field_name][workspace_id] = \
                    field_info['values'][field_name]
            
            # Store workspace field statistics
            analysis['workspace_field_stats'][workspace_id] = {
                'total_fields': len(workspace_fields),
                'shared_fields': len([f for f in workspace_fields 
                                    if len(categorizer.field_info['sources'][f]) > 1]),
                'specific_fields': len([f for f in workspace_fields 
                                      if len(categorizer.field_info['sources'][f]) == 1])
            }
        
        # Identify fields shared across all workspaces
        all_workspace_ids = set(self.workspace_configs.keys())
        for field_name, workspace_set in analysis['field_workspace_map'].items():
            if workspace_set == all_workspace_ids:
                # Check if values are consistent across workspaces
                field_values = analysis['field_value_consistency'][field_name]
                all_values = set()
                for workspace_values in field_values.values():
                    all_values.update(workspace_values)
                
                if len(all_values) == 1:  # Same value across all workspaces
                    analysis['shared_across_workspaces'].add(field_name)
        
        # Identify workspace-specific fields
        for workspace_id in all_workspace_ids:
            for field_name, workspace_set in analysis['field_workspace_map'].items():
                if workspace_set == {workspace_id}:
                    analysis['workspace_specific_fields'][workspace_id].add(field_name)
        
        return analysis
    
    def _categorize_workspace_fields(self) -> Dict[str, Any]:
        """
        Generate workspace-aware field categorization.
        
        Returns:
            Dict containing workspace categorization results
        """
        categorization = {
            'global_shared': {},           # Fields shared across ALL workspaces
            'workspace_shared': {},        # workspace_id -> shared fields within workspace
            'workspace_specific': {},      # workspace_id -> step_name -> specific fields
            'cross_workspace_dependencies': []  # List of cross-workspace field dependencies
        }
        
        # Process global shared fields (shared across all workspaces with same values)
        for field_name in self.cross_workspace_analysis['shared_across_workspaces']:
            # Get the common value from any workspace
            for workspace_id, categorizer in self.workspace_categorizers.items():
                if field_name in categorizer.field_info['raw_values']:
                    # Get first available value (they should all be the same)
                    step_name = next(iter(categorizer.field_info['raw_values'][field_name].keys()))
                    categorization['global_shared'][field_name] = \
                        categorizer.field_info['raw_values'][field_name][step_name]
                    break
        
        # Process workspace-level categorization
        for workspace_id, categorizer in self.workspace_categorizers.items():
            workspace_categorization = categorizer.get_categorized_fields()
            
            # Store workspace shared fields (excluding global shared)
            workspace_shared = {}
            for field_name, value in workspace_categorization['shared'].items():
                if field_name not in categorization['global_shared']:
                    workspace_shared[field_name] = value
            categorization['workspace_shared'][workspace_id] = workspace_shared
            
            # Store workspace specific fields
            categorization['workspace_specific'][workspace_id] = \
                workspace_categorization['specific']
        
        # Detect cross-workspace dependencies
        categorization['cross_workspace_dependencies'] = \
            self._detect_cross_workspace_dependencies()
        
        return categorization
    
    def _detect_cross_workspace_dependencies(self) -> List[Dict[str, Any]]:
        """
        Detect dependencies between workspaces based on field references.
        
        Returns:
            List of cross-workspace dependency information
        """
        dependencies = []
        
        # This is a placeholder for more sophisticated dependency detection
        # In practice, this would analyze field values for references to other workspaces
        # For example, input_path fields that reference outputs from other workspaces
        
        for workspace_id, categorizer in self.workspace_categorizers.items():
            field_info = categorizer.field_info
            
            for field_name, raw_values in field_info['raw_values'].items():
                for step_name, value in raw_values.items():
                    # Check if value references other workspaces
                    if isinstance(value, str) and any(
                        other_workspace in value 
                        for other_workspace in self.workspace_configs.keys() 
                        if other_workspace != workspace_id
                    ):
                        dependencies.append({
                            'source_workspace': workspace_id,
                            'source_step': step_name,
                            'field_name': field_name,
                            'dependency_type': 'field_reference',
                            'referenced_value': value
                        })
        
        return dependencies
    
    def get_workspace_categorization(self, workspace_id: str = None) -> Dict[str, Any]:
        """
        Get categorization for a specific workspace or all workspaces.
        
        Args:
            workspace_id: Optional workspace ID, if None returns all
            
        Returns:
            Dict containing categorization results
        """
        if workspace_id:
            if workspace_id not in self.workspace_categorization['workspace_specific']:
                return {}
            
            return {
                'global_shared': self.workspace_categorization['global_shared'],
                'workspace_shared': self.workspace_categorization['workspace_shared'].get(workspace_id, {}),
                'workspace_specific': self.workspace_categorization['workspace_specific'].get(workspace_id, {}),
                'workspace_id': workspace_id
            }
        
        return self.workspace_categorization
    
    def get_field_category_for_workspace(self, field_name: str, workspace_id: str) -> Optional[CategoryType]:
        """
        Get the category of a field within a specific workspace context.
        
        Args:
            field_name: Name of the field
            workspace_id: Workspace identifier
            
        Returns:
            CategoryType or None if field not found
        """
        # Check global shared first
        if field_name in self.workspace_categorization['global_shared']:
            return CategoryType.SHARED
        
        # Check workspace shared
        workspace_shared = self.workspace_categorization['workspace_shared'].get(workspace_id, {})
        if field_name in workspace_shared:
            return CategoryType.SHARED
        
        # Check workspace specific
        workspace_specific = self.workspace_categorization['workspace_specific'].get(workspace_id, {})
        for step_fields in workspace_specific.values():
            if field_name in step_fields:
                return CategoryType.SPECIFIC
        
        return None
    
    def print_workspace_categorization_stats(self) -> None:
        """Print statistics about workspace field categorization."""
        print("Workspace Configuration Field Categorization Statistics:")
        print(f"  Global shared fields: {len(self.workspace_categorization['global_shared'])}")
        
        for workspace_id in self.workspace_configs.keys():
            workspace_shared = len(self.workspace_categorization['workspace_shared'].get(workspace_id, {}))
            workspace_specific_count = sum(
                len(step_fields) 
                for step_fields in self.workspace_categorization['workspace_specific'].get(workspace_id, {}).values()
            )
            
            print(f"  Workspace '{workspace_id}':")
            print(f"    Workspace shared: {workspace_shared}")
            print(f"    Workspace specific: {workspace_specific_count}")
        
        print(f"  Cross-workspace dependencies: {len(self.workspace_categorization['cross_workspace_dependencies'])}")
```

### 3. WorkspaceConfigMerger

Workspace-aware configuration merger.

```python
# File: src/cursus/core/config_fields/workspace_config_merger.py
from typing import Dict, List, Any, Optional
import logging
import os
import json
from datetime import datetime

from .config_merger import ConfigMerger, MergeDirection
from .workspace_config_field_categorizer import WorkspaceConfigFieldCategorizer
from .workspace_config_context import WorkspaceConfigContext, WorkspaceConfigInfo
from .type_aware_config_serializer import TypeAwareConfigSerializer

class WorkspaceConfigMerger:
    """
    Workspace-aware configuration merger that handles multi-workspace scenarios.
    
    Extends ConfigMerger to support:
    1. Workspace-scoped configuration merging
    2. Cross-workspace dependency resolution
    3. Workspace-specific output formats
    4. Global vs workspace-level shared fields
    """
    
    def __init__(self, 
                 workspace_configs: Dict[str, List[Any]], 
                 processing_step_config_base_class: Optional[type] = None,
                 workspace_context: Optional[WorkspaceConfigInfo] = None):
        """
        Initialize with workspace-organized configurations.
        
        Args:
            workspace_configs: Dict mapping workspace_id to list of configs
            processing_step_config_base_class: Base class for processing steps
            workspace_context: Optional workspace context
        """
        self.workspace_configs = workspace_configs
        self.workspace_context = workspace_context or WorkspaceConfigContext.get_current()
        self.logger = logging.getLogger(__name__)
        
        # Create workspace-aware categorizer
        self.categorizer = WorkspaceConfigFieldCategorizer(
            workspace_configs, processing_step_config_base_class, workspace_context
        )
        
        # Create serializer with workspace context
        self.serializer = WorkspaceTypeAwareConfigSerializer(workspace_context)
        
        # Create individual mergers for backward compatibility
        self.workspace_mergers: Dict[str, ConfigMerger] = {}
        for workspace_id, configs in workspace_configs.items():
            if configs:
                self.workspace_mergers[workspace_id] = ConfigMerger(
                    configs, processing_step_config_base_class
                )
    
    def merge(self) -> Dict[str, Any]:
        """
        Merge configurations with workspace awareness.
        
        Returns:
            Dict containing workspace-aware merged configuration
        """
        # Get workspace categorization
        workspace_categorization = self.categorizer.get_workspace_categorization()
        
        # Create workspace-aware merged structure
        merged = {
            "global_shared": workspace_categorization["global_shared"],
            "workspace_configurations": {}
        }
        
        # Process each workspace
        for workspace_id, configs in self.workspace_configs.items():
            if not configs:
                continue
            
            workspace_config = {
                "workspace_shared": workspace_categorization["workspace_shared"].get(workspace_id, {}),
                "workspace_specific": workspace_categorization["workspace_specific"].get(workspace_id, {})
            }
            
            merged["workspace_configurations"][workspace_id] = workspace_config
        
        # Add cross-workspace metadata
        merged["cross_workspace_dependencies"] = workspace_categorization["cross_workspace_dependencies"]
        
        # Log statistics
        self._log_merge_statistics(merged)
        
        return merged
    
    def merge_single_workspace(self, workspace_id: str) -> Dict[str, Any]:
        """
        Merge configuration for a single workspace (backward compatibility).
        
        Args:
            workspace_id: Workspace identifier
            
        Returns:
            Dict containing single workspace merged configuration
        """
        if workspace_id not in self.workspace_mergers:
            raise ValueError(f"Workspace '{workspace_id}' not found")
        
        # Use the individual workspace merger for backward compatibility
        workspace_merger = self.workspace_mergers[workspace_id]
        workspace_result = workspace_merger.merge()
        
        # Add global shared fields
        workspace_categorization = self.categorizer.get_workspace_categorization()
        global_shared = workspace_categorization["global_shared"]
        
        # Merge global shared into workspace shared
        if global_shared:
            if "shared" not in workspace_result:
                workspace_result["shared"] = {}
            workspace_result["shared"].update(global_shared)
        
        return workspace_result
    
    def save(self, output_file: str, workspace_id: str = None) -> Dict[str, Any]:
        """
        Save merged configuration to file.
        
        Args:
            output_file: Path to output file
            workspace_id: Optional workspace ID for single workspace save
            
        Returns:
            Dict containing merged configuration
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Determine what to merge and save
        if workspace_id:
            # Single workspace save (backward compatibility)
            merged = self.merge_single_workspace(workspace_id)
            save_format = "single_workspace"
        else:
            # Multi-workspace save
            merged = self.merge()
            save_format = "multi_workspace"
        
        # Create metadata
        metadata = self._create_save_metadata(save_format)
        
        # Create output structure
        output = {
            'metadata': metadata,
            'configuration': merged
        }
        
        # Save to file
        self.logger.info(f"Saving {save_format} configuration to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, sort_keys=True)
        
        self.logger.info(f"Successfully saved configuration to {output_file}")
        return merged
    
    def _create_save_metadata(self, save_format: str) -> Dict[str, Any]:
        """Create metadata for saved configuration."""
        metadata = {
            'created_at': datetime.now().isoformat(),
            'save_format': save_format,
            'workspace_context': None
        }
        
        # Add workspace context information
        if self.workspace_context:
            metadata['workspace_context'] = {
                'workspace_id': self.workspace_context.workspace_id,
                'workspace_root': self.workspace_context.workspace_root,
                'developer_id': self.workspace_context.developer_id,
                'workspace_type': self.workspace_context.workspace_type
            }
        
        # Add config type mappings for each workspace
        config_types = {}
        for workspace_id, configs in self.workspace_configs.items():
            workspace_config_types = {}
            for config in configs:
                step_name = self.serializer.generate_step_name(config)
                workspace_config_types[step_name] = config.__class__.__name__
            config_types[workspace_id] = workspace_config_types
        
        metadata['config_types'] = config_types
        
        return metadata
    
    def _log_merge_statistics(self, merged: Dict[str, Any]) -> None:
        """Log statistics about the merged configuration."""
        global_shared_count = len(merged.get("global_shared", {}))
        workspace_count = len(merged.get("workspace_configurations", {}))
        cross_deps_count = len(merged.get("cross_workspace_dependencies", []))
        
        self.logger.info(f"Workspace merge completed:")
        self.logger.info(f"  Global shared fields: {global_shared_count}")
        self.logger.info(f"  Workspaces: {workspace_count}")
        self.logger.info(f"  Cross-workspace dependencies: {cross_deps_count}")
        
        for workspace_id, workspace_config in merged.get("workspace_configurations", {}).items():
            workspace_shared = len(workspace_config.get("workspace_shared", {}))
            workspace_specific = sum(
                len(step_fields) 
                for step_fields in workspace_config.get("workspace_specific", {}).values()
            )
            self.logger.info(f"  Workspace '{workspace_id}': {workspace_shared} shared, {workspace_specific} specific")
    
    @classmethod
    def load(cls, input_file: str, config_classes: Optional[Dict[str, type]] = None) -> Dict[str, Any]:
        """
        Load workspace-aware configuration from file.
        
        Args:
            input_file: Path to input file
            config_classes: Optional mapping of class names to class objects
            
        Returns:
            Dict containing loaded configuration
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Loading workspace configuration from {input_file}")
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Configuration file not found: {input_file}")
        
        # Load the JSON file
        with open(input_file, 'r') as f:
            file_data = json.load(f)
        
        # Check format
        metadata = file_data.get("metadata", {})
        save_format = metadata.get("save_format", "unknown")
        
        if save_format == "single_workspace":
            # Use ConfigMerger.load for backward compatibility
            return ConfigMerger.load(input_file, config_classes)
        elif save_format == "multi_workspace":
            # Handle multi-workspace format
            return file_data.get("configuration", {})
        else:
            # Try to auto-detect format
            config_data = file_data.get("configuration", file_data)
            if "workspace_configurations" in config_data:
                return config_data
            else:
                # Fall back to single workspace format
                return ConfigMerger.load(input_file, config_classes)
```

### 4. WorkspaceConfigFieldTierRegistry

Workspace-aware tier registry with workspace-specific overrides.

```python
# File: src/cursus/core/config_fields/workspace_config_field_tier_registry.py
from typing import Dict, Set, Optional, Any
import logging
from collections import defaultdict

from .tier_registry import ConfigFieldTierRegistry
from .workspace_config_context import WorkspaceConfigContext, WorkspaceConfigInfo

class WorkspaceConfigFieldTierRegistry:
    """
    Workspace-aware tier registry that supports workspace-specific tier overrides.
    
    Extends ConfigFieldTierRegistry to support:
    1. Workspace-specific tier classifications
    2. Developer-specific tier overrides
    3. Inheritance from global tier registry
    4. Workspace context-aware tier resolution
    """
    
    def __init__(self, workspace_context: Optional[WorkspaceConfigInfo] = None):
        """
        Initialize workspace tier registry.
        
        Args:
            workspace_context: Optional workspace context
        """
        self.workspace_context = workspace_context or WorkspaceConfigContext.get_current()
        self.logger = logging.getLogger(__name__)
        
        # Workspace-specific tier overrides
        # Structure: workspace_id -> field_name -> tier
        self.workspace_tier_overrides: Dict[str, Dict[str, int]] = defaultdict(dict)
        
        # Developer-specific tier overrides
        # Structure: developer_id -> field_name -> tier
        self.developer_tier_overrides: Dict[str, Dict[str, int]] = defaultdict(dict)
    
    def get_tier(self, field_name: str, workspace_id: str = None, developer_id: str = None) -> int:
        """
        Get tier classification for a field with workspace context.
        
        Args:
            field_name: The name of the field
            workspace_id: Optional workspace ID
            developer_id: Optional developer ID
            
        Returns:
            int: Tier classification (1, 2, or 3)
        """
        # Use context if not provided
        if not workspace_id and not developer_id and self.workspace_context:
            workspace_id = self.workspace_context.workspace_id
            developer_id = self.workspace_context.developer_id
        
        # Check developer-specific overrides first (highest priority)
        if developer_id and developer_id in self.developer_tier_overrides:
            if field_name in self.developer_tier_overrides[developer_id]:
                return self.developer_tier_overrides[developer_id][field_name]
        
        # Check workspace-specific overrides
        if workspace_id and workspace_id in self.workspace_tier_overrides:
            if field_name in self.workspace_tier_overrides[workspace_id]:
                return self.workspace_tier_overrides[workspace_id][field_name]
        
        # Fall back to global tier registry
        return ConfigFieldTierRegistry.get_tier(field_name)
    
    def register_workspace_field(self, workspace_id: str, field_name: str, tier: int) -> None:
        """
        Register a workspace-specific field tier override.
        
        Args:
            workspace_id: Workspace identifier
            field_name: Field name
            tier: Tier classification (1, 2, or 3)
            
        Raises:
            ValueError: If tier is not 1, 2, or 3
        """
        if tier not in [1, 2, 3]:
            raise ValueError(f"Tier must be 1, 2, or 3, got {tier}")
        
        self.workspace_tier_overrides[workspace_id][field_name] = tier
        self.logger.info(f"Registered workspace tier override: {workspace_id}.{field_name} -> Tier {tier}")
    
    def register_developer_field(self, developer_id: str, field_name: str, tier: int) -> None:
        """
        Register a developer-specific field tier override.
        
        Args:
            developer_id: Developer identifier
            field_name: Field name
            tier: Tier classification (1, 2, or 3)
            
        Raises:
            ValueError: If tier is not 1, 2, or 3
        """
        if tier not in [1, 2, 3]:
            raise ValueError(f"Tier must be 1, 2, or 3, got {tier}")
        
        self.developer_tier_overrides[developer_id][field_name] = tier
        self.logger.info(f"Registered developer tier override: {developer_id}.{field_name} -> Tier {tier}")
    
    def unregister_workspace_field(self, workspace_id: str, field_name: str) -> bool:
        """
        Unregister a workspace-specific field tier override.
        
        Args:
            workspace_id: Workspace identifier
            field_name: Field name
            
        Returns:
            bool: True if field was unregistered, False if not found
        """
        if workspace_id in self.workspace_tier_overrides:
            if field_name in self.workspace_tier_overrides[workspace_id]:
                del self.workspace_tier_overrides[workspace_id][field_name]
                self.logger.info(f"Unregistered workspace tier override: {workspace_id}.{field_name}")
                return True
        return False
    
    def unregister_developer_field(self, developer_id: str, field_name: str) -> bool:
        """
        Unregister a developer-specific field tier override.
        
        Args:
            developer_id: Developer identifier
            field_name: Field name
            
        Returns:
            bool: True if field was unregistered, False if not found
        """
        if developer_id in self.developer_tier_overrides:
            if field_name in self.developer_tier_overrides[developer_id]:
                del self.developer_tier_overrides[developer_id][field_name]
                self.logger.info(f"Unregistered developer tier override: {developer_id}.{field_name}")
                return True
        return False
    
    def get_workspace_fields(self, workspace_id: str) -> Dict[str, int]:
        """
        Get all workspace-specific field tier overrides.
        
        Args:
            workspace_id: Workspace identifier
            
        Returns:
            Dict mapping field names to tier classifications
        """
        return dict(self.workspace_tier_overrides.get(workspace_id, {}))
    
    def get_developer_fields(self, developer_id: str) -> Dict[str, int]:
        """
        Get all developer-specific field tier overrides.
        
        Args:
            developer_id: Developer identifier
            
        Returns:
            Dict mapping field names to tier classifications
        """
        return dict(self.developer_tier_overrides.get(developer_id, {}))
    
    def get_all_workspace_overrides(self) -> Dict[str, Dict[str, int]]:
        """
        Get all workspace-specific tier overrides.
        
        Returns:
            Dict mapping workspace IDs to field tier mappings
        """
        return dict(self.workspace_tier_overrides)
    
    def get_all_developer_overrides(self) -> Dict[str, Dict[str, int]]:
        """
        Get all developer-specific tier overrides.
        
        Returns:
            Dict mapping developer IDs to field tier mappings
        """
        return dict(self.developer_tier_overrides)
    
    def get_effective_tier_mapping(self, workspace_id: str = None, developer_id: str = None) -> Dict[str, int]:
        """
        Get effective tier mapping considering all override levels.
        
        Args:
            workspace_id: Optional workspace ID
            developer_id: Optional developer ID
            
        Returns:
            Dict mapping field names to effective tier classifications
        """
        # Use context if not provided
        if not workspace_id and not developer_id and self.workspace_context:
            workspace_id = self.workspace_context.workspace_id
            developer_id = self.workspace_context.developer_id
        
        # Start with global tier registry
        effective_mapping = {}
        
        # Get all known fields from global registry
        global_fields = ConfigFieldTierRegistry.get_all_registered_fields()
        for field_name in global_fields:
            effective_mapping[field_name] = ConfigFieldTierRegistry.get_tier(field_name)
        
        # Apply workspace overrides
        if workspace_id and workspace_id in self.workspace_tier_overrides:
            effective_mapping.update(self.workspace_tier_overrides[workspace_id])
        
        # Apply developer overrides (highest priority)
        if developer_id and developer_id in self.developer_tier_overrides:
            effective_mapping.update(self.developer_tier_overrides[developer_id])
        
        return effective_mapping
    
    def print_tier_override_stats(self) -> None:
        """Print statistics about tier overrides."""
        workspace_count = len(self.workspace_tier_overrides)
        developer_count = len(self.developer_tier_overrides)
        
        total_workspace_overrides = sum(
            len(overrides) for overrides in self.workspace_tier_overrides.values()
        )
        total_developer_overrides = sum(
            len(overrides) for overrides in self.developer_tier_overrides.values()
        )
        
        print("Workspace Tier Registry Statistics:")
        print(f"  Workspaces with overrides: {workspace_count}")
        print(f"  Total workspace field overrides: {total_workspace_overrides}")
        print(f"  Developers with overrides: {developer_count}")
        print(f"  Total developer field overrides: {total_developer_overrides}")
        
        if self.workspace_context:
            print(f"  Current workspace context: {self.workspace_context.workspace_id}")
            if self.workspace_context.developer_id:
                print(f"  Current developer context: {self.workspace_context.developer_id}")
```

### 5. WorkspaceTypeAwareConfigSerializer

Workspace-aware configuration serializer.

```python
# File: src/cursus/core/config_fields/workspace_type_aware_config_serializer.py
from typing import Any, Optional, Dict
import logging

from .type_aware_config_serializer import TypeAwareConfigSerializer
from .workspace_config_context import WorkspaceConfigContext, WorkspaceConfigInfo

class WorkspaceTypeAwareConfigSerializer:
    """
    Workspace-aware configuration serializer that generates workspace-scoped step names.
    
    Extends TypeAwareConfigSerializer to support:
    1. Workspace-prefixed step name generation
    2. Developer-specific step naming
    3. Workspace context-aware serialization
    4. Cross-workspace step name collision avoidance
    """
    
    def __init__(self, workspace_context: Optional[WorkspaceConfigInfo] = None):
        """
        Initialize workspace-aware serializer.
        
        Args:
            workspace_context: Optional workspace context
        """
        self.workspace_context = workspace_context or WorkspaceConfigContext.get_current()
        self.logger = logging.getLogger(__name__)
        
        # Base serializer for fallback
        self.base_serializer = TypeAwareConfigSerializer()
    
    def generate_step_name(self, config: Any, workspace_id: str = None) -> str:
        """
        Generate workspace-aware step name.
        
        Args:
            config: Configuration object
            workspace_id: Optional workspace ID override
            
        Returns:
            str: Workspace-aware step name
        """
        # Get base step name
        base_step_name = self.base_serializer.generate_step_name(config)
        
        # Determine workspace context
        effective_workspace_id = workspace_id
        if not effective_workspace_id and self.workspace_context:
            effective_workspace_id = self.workspace_context.workspace_id
        
        # If no workspace context, return base name (backward compatibility)
        if not effective_workspace_id:
            return base_step_name
        
        # Generate workspace-prefixed step name
        workspace_prefix = self._generate_workspace_prefix(effective_workspace_id)
        workspace_step_name = f"{workspace_prefix}_{base_step_name}"
        
        self.logger.debug(f"Generated workspace step name: {base_step_name} -> {workspace_step_name}")
        return workspace_step_name
    
    def _generate_workspace_prefix(self, workspace_id: str) -> str:
        """
        Generate workspace prefix for step names.
        
        Args:
            workspace_id: Workspace identifier
            
        Returns:
            str: Workspace prefix
        """
        # Handle different workspace ID formats
        if "@" in workspace_id:
            # Format: developer@workspace_root
            developer_id, workspace_root = workspace_id.split("@", 1)
            # Use developer ID as prefix for multi-developer workspaces
            return developer_id.replace("-", "_").replace(".", "_")
        else:
            # Single workspace format
            # Use last part of workspace path as prefix
            workspace_name = workspace_id.split("/")[-1] if "/" in workspace_id else workspace_id
            return workspace_name.replace("-", "_").replace(".", "_")
    
    def generate_workspace_scoped_config_dict(self, config: Any, workspace_id: str = None) -> Dict[str, Any]:
        """
        Generate workspace-scoped configuration dictionary.
        
        Args:
            config: Configuration object
            workspace_id: Optional workspace ID override
            
        Returns:
            Dict containing workspace-scoped configuration
        """
        # Generate workspace-aware step name
        step_name = self.generate_step_name(config, workspace_id)
        
        # Get base config dict
        base_config_dict = self.base_serializer.generate_config_dict(config)
        
        # Add workspace metadata
        workspace_config_dict = {
            "step_name": step_name,
            "workspace_metadata": self._generate_workspace_metadata(workspace_id),
            "config": base_config_dict
        }
        
        return workspace_config_dict
    
    def _generate_workspace_metadata(self, workspace_id: str = None) -> Dict[str, Any]:
        """
        Generate workspace metadata for configuration.
        
        Args:
            workspace_id: Optional workspace ID override
            
        Returns:
            Dict containing workspace metadata
        """
        metadata = {}
        
        # Use provided workspace_id or context
        effective_workspace_id = workspace_id
        if not effective_workspace_id and self.workspace_context:
            effective_workspace_id = self.workspace_context.workspace_id
        
        if effective_workspace_id:
            metadata["workspace_id"] = effective_workspace_id
        
        if self.workspace_context:
            metadata.update({
                "workspace_root": self.workspace_context.workspace_root,
                "workspace_type": self.workspace_context.workspace_type
            })
            
            if self.workspace_context.developer_id:
                metadata["developer_id"] = self.workspace_context.developer_id
            
            if self.workspace_context.metadata:
                metadata["custom_metadata"] = self.workspace_context.metadata
        
        return metadata
    
    def is_workspace_aware(self) -> bool:
        """
        Check if serializer is operating in workspace-aware mode.
        
        Returns:
            bool: True if workspace context is available
        """
        return self.workspace_context is not None
    
    def get_workspace_context(self) -> Optional[WorkspaceConfigInfo]:
        """
        Get current workspace context.
        
        Returns:
            WorkspaceConfigInfo or None
        """
        return self.workspace_context
    
    def set_workspace_context(self, workspace_context: WorkspaceConfigInfo) -> None:
        """
        Set workspace context.
        
        Args:
            workspace_context: New workspace context
        """
        self.workspace_context = workspace_context
        self.logger.info(f"Updated workspace context: {workspace_context.workspace_id}")
```

## Integration with Existing System

### 1. Backward Compatibility Strategy

The workspace-aware config manager maintains full backward compatibility:

```python
# File: src/cursus/core/config_fields/__init__.py
# Add workspace-aware imports while maintaining existing imports

from .config_field_categorizer import ConfigFieldCategorizer
from .config_merger import ConfigMerger
from .tier_registry import ConfigFieldTierRegistry
from .type_aware_config_serializer import TypeAwareConfigSerializer

# New workspace-aware components
from .workspace_config_context import WorkspaceConfigContext, WorkspaceConfigInfo
from .workspace_config_field_categorizer import WorkspaceConfigFieldCategorizer
from .workspace_config_merger import WorkspaceConfigMerger
from .workspace_config_field_tier_registry import WorkspaceConfigFieldTierRegistry
from .workspace_type_aware_config_serializer import WorkspaceTypeAwareConfigSerializer

# Convenience factory function for workspace-aware operations
def create_workspace_config_manager(workspace_configs: Dict[str, List[Any]], 
                                   workspace_context: Optional[WorkspaceConfigInfo] = None):
    """
    Create a complete workspace-aware configuration manager.
    
    Args:
        workspace_configs: Dict mapping workspace_id to list of configs
        workspace_context: Optional workspace context
        
    Returns:
        Dict containing all workspace-aware components
    """
    return {
        'categorizer': WorkspaceConfigFieldCategorizer(workspace_configs, workspace_context=workspace_context),
        'merger': WorkspaceConfigMerger(workspace_configs, workspace_context=workspace_context),
        'tier_registry': WorkspaceConfigFieldTierRegistry(workspace_context),
        'serializer': WorkspaceTypeAwareConfigSerializer(workspace_context)
    }
```

### 2. Integration Points

#### A. Workspace DAG Compiler Integration

```python
# File: src/cursus/core/workspace/compiler.py
# Add workspace config manager integration

from ..config_fields import (
    WorkspaceConfigContext, 
    WorkspaceConfigMerger,
    create_workspace_config_manager
)

class WorkspaceDAGCompiler:
    def __init__(self, workspace_dag: WorkspaceAwareDAG):
        self.workspace_dag = workspace_dag
        # Initialize workspace config manager
        self._init_workspace_config_manager()
    
    def _init_workspace_config_manager(self):
        """Initialize workspace-aware configuration management."""
        # Collect configurations from all workspace steps
        workspace_configs = {}
        for workspace_id, steps in self.workspace_dag.workspace_steps.items():
            workspace_configs[workspace_id] = [step.config for step in steps if hasattr(step, 'config')]
        
        # Create workspace context
        workspace_context = WorkspaceConfigContext.from_workspace_root(
            self.workspace_dag.workspace_root,
            self.workspace_dag.developer_id
        )
        
        # Create workspace config manager
        self.config_manager = create_workspace_config_manager(
            workspace_configs, workspace_context
        )
    
    def compile_with_config_awareness(self) -> Dict[str, Any]:
        """Compile DAG with workspace-aware configuration management."""
        with WorkspaceConfigContext.workspace_context(self.config_manager['serializer'].workspace_context):
            # Merge configurations with workspace awareness
            merged_config = self.config_manager['merger'].merge()
            
            # Compile DAG with merged configuration
            compiled_dag = self.compile()
            
            # Add configuration metadata
            compiled_dag['workspace_configuration'] = merged_config
            
            return compiled_dag
```

#### B. Pipeline Runtime Integration

```python
# File: src/cursus/pipeline_runtime/core_engine.py
# Add workspace config context support

from ..core.config_fields import WorkspaceConfigContext

class PipelineRuntimeEngine:
    def execute_with_workspace_context(self, workspace_context: WorkspaceConfigInfo):
        """Execute pipeline with workspace configuration context."""
        with WorkspaceConfigContext.workspace_context(workspace_context):
            return self.execute()
```

### 3. Migration Path

#### Phase 1: Core Infrastructure (Current)
- Implement workspace config context management
- Create workspace-aware categorizer, merger, tier registry, and serializer
- Ensure backward compatibility with existing code

#### Phase 2: Integration
- Integrate with WorkspaceDAGCompiler
- Add workspace context to pipeline runtime
- Update CLI tools to support workspace-aware configuration

#### Phase 3: Advanced Features
- Cross-workspace dependency resolution
- Workspace-specific configuration validation
- Performance optimizations for large multi-workspace environments

## Testing Strategy

### 1. Unit Tests

```python
# File: test/core/config_fields/test_workspace_config_manager.py
import pytest
from src.cursus.core.config_fields import (
    WorkspaceConfigContext,
    WorkspaceConfigFieldCategorizer,
    WorkspaceConfigMerger,
    WorkspaceConfigFieldTierRegistry,
    WorkspaceTypeAwareConfigSerializer
)

class TestWorkspaceConfigContext:
    def test_context_management(self):
        """Test workspace context management."""
        workspace_info = WorkspaceConfigContext.from_workspace_root(
            "/path/to/workspace", "developer1"
        )
        
        with WorkspaceConfigContext.workspace_context(workspace_info):
            current = WorkspaceConfigContext.get_current()
            assert current.workspace_id == "developer1@/path/to/workspace"
            assert current.developer_id == "developer1"
        
        # Context should be cleared after exiting
        assert WorkspaceConfigContext.get_current() is None

class TestWorkspaceConfigFieldCategorizer:
    def test_cross_workspace_analysis(self):
        """Test cross-workspace field analysis."""
        # Create test configurations for multiple workspaces
        workspace_configs = {
            "workspace1": [MockConfig1(), MockConfig2()],
            "workspace2": [MockConfig1(), MockConfig3()]
        }
        
        categorizer = WorkspaceConfigFieldCategorizer(workspace_configs)
        analysis = categorizer.cross_workspace_analysis
        
        # Verify cross-workspace field detection
        assert "shared_field" in analysis['shared_across_workspaces']
        assert "workspace1_specific" in analysis['workspace_specific_fields']['workspace1']

class TestWorkspaceConfigMerger:
    def test_workspace_aware_merge(self):
        """Test workspace-aware configuration merging."""
        workspace_configs = {
            "workspace1": [MockConfig1()],
            "workspace2": [MockConfig2()]
        }
        
        merger = WorkspaceConfigMerger(workspace_configs)
        merged = merger.merge()
        
        # Verify workspace structure
        assert "global_shared" in merged
        assert "workspace_configurations" in merged
        assert "workspace1" in merged["workspace_configurations"]
        assert "workspace2" in merged["workspace_configurations"]

class TestWorkspaceConfigFieldTierRegistry:
    def test_workspace_tier_overrides(self):
        """Test workspace-specific tier overrides."""
        registry = WorkspaceConfigFieldTierRegistry()
        
        # Register workspace override
        registry.register_workspace_field("workspace1", "custom_field", 2)
        
        # Test tier resolution
        assert registry.get_tier("custom_field", "workspace1") == 2
        assert registry.get_tier("custom_field", "workspace2") == 3  # Default

class TestWorkspaceTypeAwareConfigSerializer:
    def test_workspace_step_name_generation(self):
        """Test workspace-aware step name generation."""
        workspace_context = WorkspaceConfigContext.from_workspace_root(
            "/path/to/workspace", "developer1"
        )
        
        serializer = WorkspaceTypeAwareConfigSerializer(workspace_context)
        step_name = serializer.generate_step_name(MockConfig1())
        
        # Should include workspace prefix
        assert step_name.startswith("developer1_")
```

### 2. Integration Tests

```python
# File: test/integration/test_workspace_config_integration.py
import pytest
from src.cursus.core.workspace import WorkspaceDAGCompiler
from src.cursus.api.dag import WorkspaceAwareDAG

class TestWorkspaceConfigIntegration:
    def test_end_to_end_workspace_config_flow(self):
        """Test complete workspace configuration flow."""
        # Create workspace DAG with multiple workspaces
        workspace_dag = WorkspaceAwareDAG("test_workspace")
        
        # Add steps from different workspaces
        workspace_dag.add_workspace_step("workspace1", MockStep1())
        workspace_dag.add_workspace_step("workspace2", MockStep2())
        
        # Compile with workspace config awareness
        compiler = WorkspaceDAGCompiler(workspace_dag)
        compiled = compiler.compile_with_config_awareness()
        
        # Verify workspace configuration structure
        assert "workspace_configuration" in compiled
        workspace_config = compiled["workspace_configuration"]
        assert "global_shared" in workspace_config
        assert "workspace_configurations" in workspace_config
```

## Performance Considerations

### 1. Memory Optimization
- Lazy loading of workspace configurations
- Shared field deduplication across workspaces
- Efficient cross-workspace dependency tracking

### 2. Computation Optimization
- Cached field categorization results
- Incremental workspace analysis updates
- Parallel processing for large workspace sets

### 3. Scalability
- Support for 100+ concurrent workspaces
- Efficient workspace context switching
- Minimal overhead for single-workspace scenarios

## Security Considerations

### 1. Workspace Isolation
- Prevent cross-workspace configuration leakage
- Secure workspace context propagation
- Access control for workspace-specific configurations

### 2. Developer Privacy
- Optional developer ID anonymization
- Secure storage of workspace metadata
- Audit logging for configuration access

## Documentation and Examples

### 1. Usage Examples

```python
# Example 1: Basic workspace-aware configuration
from src.cursus.core.config_fields import (
    WorkspaceConfigContext,
    WorkspaceConfigMerger
)

# Set up workspace context
workspace_context = WorkspaceConfigContext.from_workspace_root(
    "/path/to/workspace", "developer1"
)

# Organize configurations by workspace
workspace_configs = {
    "workspace1": [config1, config2],
    "workspace2": [config3, config4]
}

# Create workspace-aware merger
with WorkspaceConfigContext.workspace_context(workspace_context):
    merger = WorkspaceConfigMerger(workspace_configs)
    merged_config = merger.merge()
    
    # Save workspace-aware configuration
    merger.save("output/workspace_config.json")

# Example 2: Single workspace backward compatibility
single_workspace_config = merger.merge_single_workspace("workspace1")
merger.save("output/single_workspace_config.json", "workspace1")
```

### 2. CLI Integration

```bash
# Workspace-aware configuration merging
cursus config merge --workspace-aware --workspace-root /path/to/workspace --developer-id developer1

# Single workspace mode (backward compatible)
cursus config merge --workspace workspace1 --output single_config.json
```

## Future Enhancements

### 1. Advanced Dependency Resolution
- Automatic cross-workspace dependency detection
- Dependency graph visualization
- Circular dependency prevention

### 2. Configuration Validation
- Workspace-specific validation rules
- Cross-workspace consistency checks
- Configuration drift detection

### 3. Performance Monitoring
- Workspace configuration performance metrics
- Memory usage optimization
- Configuration access patterns analysis

### 4. Integration Extensions
- IDE plugin support for workspace configuration
- CI/CD pipeline integration
- Configuration versioning and rollback

## Conclusion

The Workspace-Aware Config Manager design provides a comprehensive solution for multi-developer workspace configuration management while maintaining full backward compatibility with the existing system. The design follows established patterns from the existing codebase and provides clear extension points for future enhancements.

Key benefits:
- **Workspace Isolation**: Configurations are properly scoped to workspaces
- **Developer Productivity**: Reduced configuration conflicts in multi-developer environments
- **Backward Compatibility**: Existing code continues to work unchanged
- **Extensibility**: Clear architecture for future workspace-aware features
- **Performance**: Efficient handling of large multi-workspace environments

This design addresses the critical gap identified in Phase 3 of the workspace implementation and provides the foundation for robust multi-developer workspace support in the cursus system.
