"""
Unified Registry Manager Implementation

This module provides a single, consolidated registry manager for the hybrid registry system.
Eliminates redundancy by consolidating CoreStepRegistry, LocalStepRegistry, and HybridRegistryManager
into one unified manager that handles all registry operations efficiently.
"""

import os
import logging
import sys
from typing import Dict, List, Optional, Union, Any, Type
from pathlib import Path
import threading
from contextlib import contextmanager
from functools import lru_cache


from .models import (
    StepDefinition,
    ResolutionContext,
    StepResolutionResult,
    RegistryValidationResult,
    ConflictAnalysis,
)
from .utils import (
    load_registry_module,
    from_legacy_format,
    to_legacy_format,
    convert_registry_dict,
    format_step_not_found_error,
    format_registry_load_error,
)

logger = logging.getLogger(__name__)


class UnifiedRegistryManager:
    """
    Unified registry manager that consolidates all registry operations.

    Replaces CoreStepRegistry, LocalStepRegistry, and HybridRegistryManager with a single,
    efficient manager that eliminates redundancy while maintaining all functionality.
    """

    def __init__(self, core_registry_path: str = None, workspaces_root: str = None):
        # PORTABLE: Use package-relative paths instead of hardcoded assumptions
        if core_registry_path:
            self.core_registry_path = core_registry_path
        else:
            # Find package root and use relative path
            package_root = self._find_package_root()
            self.core_registry_path = str(package_root / "registry" / "step_names.py")
        
        # PORTABLE: Only set workspaces_root if explicitly provided by user
        if workspaces_root:
            self.workspaces_root = Path(workspaces_root)
        else:
            # No default workspace assumption - user must provide if needed
            self.workspaces_root = None

        # Core registry data
        self._core_steps: Dict[str, StepDefinition] = {}
        self._core_loaded = False

        # Workspace registry data
        self._workspace_steps: Dict[str, Dict[str, StepDefinition]] = (
            {}
        )  # workspace_id -> steps
        self._workspace_overrides: Dict[str, Dict[str, StepDefinition]] = (
            {}
        )  # workspace_id -> overrides
        self._workspace_metadata: Dict[str, Dict[str, Any]] = (
            {}
        )  # workspace_id -> metadata

        # Performance optimization: Caching infrastructure
        self._legacy_cache: Dict[str, Dict[str, Dict[str, Any]]] = (
            {}
        )  # workspace_id -> legacy_dict
        self._definition_cache: Dict[str, Dict[str, StepDefinition]] = (
            {}
        )  # workspace_id -> definitions
        self._step_list_cache: Dict[str, List[str]] = {}  # workspace_id -> step_names

        # Thread safety
        self._lock = threading.RLock()

        # Initialize registries
        self._load_core_registry()
        self._discover_and_load_workspaces()

    def _find_package_root(self) -> Path:
        """
        Find cursus package root using relative path navigation.
        
        Works in all deployment scenarios:
        - PyPI: site-packages/cursus/
        - Source: src/cursus/
        - Submodule: parent_package/cursus/
        """
        # From cursus/registry/hybrid/manager.py, navigate to cursus package root
        current_file = Path(__file__)
        
        # Navigate up to find cursus package root
        current_dir = current_file.parent
        while current_dir.name != 'cursus' and current_dir.parent != current_dir:
            current_dir = current_dir.parent
        
        if current_dir.name == 'cursus':
            return current_dir
        else:
            # Fallback: assume we're in cursus package structure
            return current_file.parent.parent.parent  # hybrid -> registry -> cursus

    def _load_core_registry(self):
        """Load core registry from original step names to avoid circular imports."""
        from ..exceptions import RegistryLoadError

        try:
            # Import directly from step_names_original to avoid circular imports
            from ..step_names_original import STEP_NAMES as ORIGINAL_STEP_NAMES

            # Convert to StepDefinition objects
            for step_name, step_info in ORIGINAL_STEP_NAMES.items():
                step_def = from_legacy_format(
                    step_name, step_info, registry_type="core", workspace_id=None
                )
                self._core_steps[step_name] = step_def

            self._core_loaded = True
            logger.debug(
                f"Loaded {len(self._core_steps)} core steps from original registry"
            )

        except Exception as e:
            if isinstance(e, RegistryLoadError):
                raise
            raise RegistryLoadError(f"Failed to load core registry: {str(e)}")

    def _discover_and_load_workspaces(self):
        """Auto-discover and load workspace registries using step catalog."""
        # Avoid circular imports by checking if we're already in a step catalog context
        import sys
        
        # Check if step catalog is already being imported to avoid recursion
        if 'cursus.step_catalog' in sys.modules or 'src.cursus.step_catalog' in sys.modules:
            logger.debug("Step catalog already in import chain, skipping workspace discovery")
            return
            
        # Try using step catalog for workspace discovery (only if workspaces_root is provided)
        if self.workspaces_root:
            try:
                # Use lazy import to avoid circular dependency with relative import
                import importlib
                step_catalog_module = importlib.import_module('...step_catalog.step_catalog', package=__package__)
                StepCatalog = step_catalog_module.StepCatalog
                
                # PORTABLE: Use workspace-aware discovery with user-provided workspace root
                catalog = StepCatalog(workspace_dirs=[self.workspaces_root])
                
                # Use catalog's cross-workspace discovery
                cross_workspace_components = catalog.discover_cross_workspace_components()
                
                # Load workspaces discovered by catalog
                for workspace_id, components in cross_workspace_components.items():
                    if workspace_id != "core" and components:  # Skip core, focus on workspaces
                        workspace_path = self.workspaces_root / workspace_id
                        if workspace_path.exists():
                            self._load_workspace_registry(workspace_id, workspace_path)
                            logger.debug(f"Loaded workspace '{workspace_id}' via step catalog with {len(components)} components")
                            
            except ImportError:
                logger.debug("Step catalog not available - workspace discovery disabled")
            except Exception as e:
                logger.debug(f"Step catalog workspace discovery failed: {e}")
        else:
            logger.debug("No workspaces_root provided - skipping workspace discovery (package-only mode)")

    def _load_workspace_registry(self, workspace_id: str, workspace_path: Path):
        """Load workspace registry from the workspace path."""
        try:
            # Look for workspace registry file
            registry_file = (
                workspace_path
                / "src"
                / "cursus_dev"
                / "registry"
                / "workspace_registry.py"
            )

            if not registry_file.exists():
                return

            module = load_registry_module(str(registry_file))

            # Initialize workspace data
            self._workspace_steps[workspace_id] = {}
            self._workspace_overrides[workspace_id] = {}
            self._workspace_metadata[workspace_id] = {}

            # Load LOCAL_STEPS
            local_steps = getattr(module, "LOCAL_STEPS", {})
            for step_name, step_info in local_steps.items():
                step_def = from_legacy_format(
                    step_name,
                    step_info,
                    registry_type="workspace",
                    workspace_id=workspace_id,
                )
                self._workspace_steps[workspace_id][step_name] = step_def

            # Load STEP_OVERRIDES
            step_overrides = getattr(module, "STEP_OVERRIDES", {})
            for step_name, step_info in step_overrides.items():
                step_def = from_legacy_format(
                    step_name,
                    step_info,
                    registry_type="override",
                    workspace_id=workspace_id,
                )
                self._workspace_overrides[workspace_id][step_name] = step_def

            # Load WORKSPACE_METADATA
            workspace_metadata = getattr(module, "WORKSPACE_METADATA", {})
            self._workspace_metadata[workspace_id] = workspace_metadata

            # Update workspace_id from metadata if provided
            if "developer_id" in workspace_metadata:
                actual_workspace_id = workspace_metadata["developer_id"]
                if actual_workspace_id != workspace_id:
                    # Move data to correct workspace_id
                    self._workspace_steps[actual_workspace_id] = (
                        self._workspace_steps.pop(workspace_id)
                    )
                    self._workspace_overrides[actual_workspace_id] = (
                        self._workspace_overrides.pop(workspace_id)
                    )
                    self._workspace_metadata[actual_workspace_id] = (
                        self._workspace_metadata.pop(workspace_id)
                    )
                    workspace_id = actual_workspace_id

            logger.debug(
                f"Loaded workspace '{workspace_id}': {len(self._workspace_steps[workspace_id])} local steps, {len(self._workspace_overrides[workspace_id])} overrides"
            )

        except Exception as e:
            logger.debug(f"Failed to load workspace registry for '{workspace_id}': {e}")

    def get_step_definition(
        self, step_name: str, workspace_id: str = None
    ) -> Optional[StepDefinition]:
        """
        Get a step definition by name, with optional workspace context.

        Args:
            step_name: Name of the step to retrieve
            workspace_id: Optional workspace context for resolution

        Returns:
            StepDefinition if found, None otherwise
        """
        with self._lock:
            # Check workspace-specific registry first if workspace_id provided
            if workspace_id and workspace_id in self._workspace_steps:
                # Check local steps first
                if step_name in self._workspace_steps[workspace_id]:
                    return self._workspace_steps[workspace_id][step_name]

                # Check overrides
                if step_name in self._workspace_overrides[workspace_id]:
                    return self._workspace_overrides[workspace_id][step_name]

            # Check all workspace registries for the step
            for ws_id in self._workspace_steps:
                if step_name in self._workspace_steps[ws_id]:
                    return self._workspace_steps[ws_id][step_name]
                if step_name in self._workspace_overrides[ws_id]:
                    return self._workspace_overrides[ws_id][step_name]

            # Fallback to core registry
            return self._core_steps.get(step_name)

    @lru_cache(maxsize=32)
    def _get_cached_definitions(
        self, workspace_id: Optional[str]
    ) -> Dict[str, StepDefinition]:
        """Cached version of get_all_step_definitions for performance optimization."""
        cache_key = workspace_id or "core"

        if cache_key not in self._definition_cache:
            if workspace_id and workspace_id in self._workspace_steps:
                # Start with core definitions
                all_definitions = self._core_steps.copy()

                # Add workspace local steps
                all_definitions.update(self._workspace_steps[workspace_id])

                # Apply workspace overrides
                all_definitions.update(self._workspace_overrides[workspace_id])

                self._definition_cache[cache_key] = all_definitions
            else:
                # Return core definitions only
                self._definition_cache[cache_key] = self._core_steps.copy()

        return self._definition_cache[cache_key]

    def get_all_step_definitions(
        self, workspace_id: str = None
    ) -> Dict[str, StepDefinition]:
        """Get all step definitions with caching for performance optimization."""
        with self._lock:
            return self._get_cached_definitions(workspace_id)

    def get_local_only_definitions(
        self, workspace_id: str
    ) -> Dict[str, StepDefinition]:
        """Get only local and override definitions for a workspace (not core)."""
        with self._lock:
            if workspace_id not in self._workspace_steps:
                return {}

            local_only = {}
            local_only.update(self._workspace_steps[workspace_id])
            local_only.update(self._workspace_overrides[workspace_id])
            return local_only

    def get_step(
        self, step_name: str, context: Optional[ResolutionContext] = None
    ) -> StepResolutionResult:
        """
        Get a step definition with simple workspace priority resolution.

        Args:
            step_name: Name of the step to retrieve
            context: Resolution context for workspace handling

        Returns:
            StepResolutionResult containing the resolved step and metadata
        """
        if context is None:
            context = ResolutionContext(workspace_id="default")

        with self._lock:
            # Simple workspace priority resolution
            if context.workspace_id and context.workspace_id in self._workspace_steps:
                # Check workspace-specific registry first
                step_def = self.get_step_definition(step_name, context.workspace_id)
                if step_def:
                    source = (
                        context.workspace_id
                        if step_def.workspace_id == context.workspace_id
                        else "core"
                    )
                    return StepResolutionResult(
                        step_name=step_name,
                        resolved=True,
                        selected_definition=step_def,
                        source_registry=source,
                        workspace_id=context.workspace_id,
                        resolution_strategy="workspace_priority",
                        conflict_detected=False,
                        conflict_analysis=None,
                        errors=[],
                        warnings=[],
                    )

            # Check core registry
            core_step = self._core_steps.get(step_name)
            if core_step:
                return StepResolutionResult(
                    step_name=step_name,
                    resolved=True,
                    selected_definition=core_step,
                    source_registry="core",
                    workspace_id=context.workspace_id,
                    resolution_strategy="workspace_priority",
                    conflict_detected=False,
                    conflict_analysis=None,
                    errors=[],
                    warnings=[],
                )

            # Step not found
            error_msg = format_step_not_found_error(
                step_name, context.workspace_id, self.list_all_steps()
            )
            return StepResolutionResult(
                step_name=step_name,
                resolved=False,
                selected_definition=None,
                source_registry="none",
                workspace_id=context.workspace_id,
                resolution_strategy="workspace_priority",
                conflict_detected=False,
                conflict_analysis=None,
                errors=[error_msg],
                warnings=[],
            )

    def list_steps(self, workspace_id: str = None) -> List[str]:
        """List all available step names for a workspace or core."""
        with self._lock:
            if workspace_id and workspace_id in self._workspace_steps:
                all_steps = set(self._core_steps.keys())
                all_steps.update(self._workspace_steps[workspace_id].keys())
                all_steps.update(self._workspace_overrides[workspace_id].keys())
                return sorted(list(all_steps))
            else:
                return sorted(list(self._core_steps.keys()))

    def list_all_steps(
        self, include_source: bool = False
    ) -> Union[List[str], Dict[str, List[str]]]:
        """List all available steps across all registries."""
        with self._lock:
            if include_source:
                result = {}
                result["core"] = sorted(list(self._core_steps.keys()))
                for workspace_id in self._workspace_steps:
                    workspace_steps = set(self._workspace_steps[workspace_id].keys())
                    workspace_steps.update(
                        self._workspace_overrides[workspace_id].keys()
                    )
                    result[workspace_id] = sorted(list(workspace_steps))
                return result
            else:
                all_steps = set(self._core_steps.keys())
                for workspace_id in self._workspace_steps:
                    all_steps.update(self._workspace_steps[workspace_id].keys())
                    all_steps.update(self._workspace_overrides[workspace_id].keys())
                return sorted(list(all_steps))

    def has_step(self, step_name: str, workspace_id: str = None) -> bool:
        """Check if a step exists in the registry."""
        with self._lock:
            if workspace_id and workspace_id in self._workspace_steps:
                return (
                    step_name in self._workspace_steps[workspace_id]
                    or step_name in self._workspace_overrides[workspace_id]
                    or step_name in self._core_steps
                )
            else:
                # Check all workspaces and core
                if step_name in self._core_steps:
                    return True
                for ws_id in self._workspace_steps:
                    if (
                        step_name in self._workspace_steps[ws_id]
                        or step_name in self._workspace_overrides[ws_id]
                    ):
                        return True
                return False

    def get_step_count(self, workspace_id: str = None) -> int:
        """Get the total number of steps in the registry."""
        with self._lock:
            if workspace_id and workspace_id in self._workspace_steps:
                all_steps = set(self._core_steps.keys())
                all_steps.update(self._workspace_steps[workspace_id].keys())
                all_steps.update(self._workspace_overrides[workspace_id].keys())
                return len(all_steps)
            else:
                return len(self._core_steps)

    def add_workspace_registry(self, workspace_id: str, workspace_path: str) -> None:
        """Add a new workspace registry."""
        with self._lock:
            self._load_workspace_registry(workspace_id, Path(workspace_path))
            # Invalidate caches after adding workspace
            self._invalidate_cache(workspace_id)
            self._invalidate_all_caches()  # Also invalidate global caches
            logger.info(f"Added workspace registry: {workspace_id}")

    def remove_workspace_registry(self, workspace_id: str) -> bool:
        """Remove a workspace registry."""
        with self._lock:
            if workspace_id in self._workspace_steps:
                del self._workspace_steps[workspace_id]
                del self._workspace_overrides[workspace_id]
                del self._workspace_metadata[workspace_id]
                # Invalidate caches after removing workspace
                self._invalidate_cache(workspace_id)
                self._invalidate_all_caches()  # Also invalidate global caches
                logger.info(f"Removed workspace registry: {workspace_id}")
                return True
            return False

    def get_step_conflicts(self) -> Dict[str, List[StepDefinition]]:
        """Identify steps defined in multiple registries."""
        with self._lock:
            conflicts = {}
            all_step_names = set()

            # Collect all step names from all workspaces
            for workspace_id in self._workspace_steps:
                local_steps = self.get_local_only_definitions(workspace_id)
                for step_name, step_def in local_steps.items():
                    if step_name in all_step_names:
                        if step_name not in conflicts:
                            conflicts[step_name] = []
                        conflicts[step_name].append(step_def)
                    else:
                        all_step_names.add(step_name)

            return conflicts

    def get_registry_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status information for all registries."""
        with self._lock:
            status = {}

            # Core registry status
            status["core"] = {
                "loaded": self._core_loaded,
                "step_count": len(self._core_steps),
                "registry_path": str(self.core_registry_path),
            }

            # Workspace registry status
            for workspace_id in self._workspace_steps:
                local_count = len(self._workspace_steps[workspace_id])
                override_count = len(self._workspace_overrides[workspace_id])
                status[workspace_id] = {
                    "loaded": True,
                    "local_step_count": local_count,
                    "override_count": override_count,
                    "total_step_count": self.get_step_count(workspace_id),
                    "workspace_id": workspace_id,
                    "metadata": self._workspace_metadata.get(workspace_id, {}),
                }

            return status

    @lru_cache(maxsize=16)
    def _get_cached_legacy_dict(
        self, workspace_id: Optional[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Cached version of create_legacy_step_names_dict for performance optimization."""
        cache_key = workspace_id or "core"

        if cache_key not in self._legacy_cache:
            all_definitions = self._get_cached_definitions(workspace_id)
            legacy_dict = {}

            for step_name, definition in all_definitions.items():
                legacy_dict[step_name] = to_legacy_format(definition)

            self._legacy_cache[cache_key] = legacy_dict

        return self._legacy_cache[cache_key]

    def create_legacy_step_names_dict(
        self, workspace_id: str = None
    ) -> Dict[str, Dict[str, Any]]:
        """Create legacy STEP_NAMES dictionary for backward compatibility with caching."""
        with self._lock:
            return self._get_cached_legacy_dict(workspace_id)

    def _invalidate_cache(self, workspace_id: Optional[str] = None):
        """Invalidate cached data when registry changes occur."""
        if workspace_id:
            # Invalidate specific workspace cache
            cache_key = workspace_id
            self._legacy_cache.pop(cache_key, None)
            self._definition_cache.pop(cache_key, None)
            self._step_list_cache.pop(cache_key, None)
        else:
            # Invalidate all caches
            self._legacy_cache.clear()
            self._definition_cache.clear()
            self._step_list_cache.clear()

        # Clear LRU caches
        self._get_cached_definitions.cache_clear()
        self._get_cached_legacy_dict.cache_clear()

    def _invalidate_all_caches(self):
        """Invalidate all cached data across all workspaces."""
        self._invalidate_cache(None)

    @contextmanager
    def resolution_context(self, workspace_id: str):
        """Context manager for step resolution."""
        context = ResolutionContext(workspace_id=workspace_id)
        yield context

    # Component discovery caching methods for WorkspaceComponentRegistry integration
    def get_component_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached component discovery results."""
        return (
            self._component_cache.get(cache_key)
            if hasattr(self, "_component_cache")
            else None
        )

    def set_component_cache(self, cache_key: str, components: Dict[str, Any]) -> None:
        """Cache component discovery results."""
        if not hasattr(self, "_component_cache"):
            self._component_cache = {}
        self._component_cache[cache_key] = components

    def clear_component_cache(self) -> None:
        """Clear component discovery cache."""
        if hasattr(self, "_component_cache"):
            self._component_cache.clear()

    def get_builder_class_cache(self, cache_key: str) -> Optional[Type]:
        """Get cached builder class."""
        return (
            self._builder_class_cache.get(cache_key)
            if hasattr(self, "_builder_class_cache")
            else None
        )

    def set_builder_class_cache(self, cache_key: str, builder_class: Type) -> None:
        """Cache builder class."""
        if not hasattr(self, "_builder_class_cache"):
            self._builder_class_cache = {}
        self._builder_class_cache[cache_key] = builder_class

    def clear_builder_class_cache(self) -> None:
        """Clear builder class cache."""
        if hasattr(self, "_builder_class_cache"):
            self._builder_class_cache.clear()

    # Workspace context management methods
    def set_workspace_context(self, workspace_id: str) -> None:
        """Set current workspace context."""
        self._current_workspace_context = workspace_id
        self._invalidate_all_caches()  # Invalidate caches when context changes

    def get_workspace_context(self) -> Optional[str]:
        """Get current workspace context."""
        return getattr(self, "_current_workspace_context", None)

    def clear_workspace_context(self) -> None:
        """Clear current workspace context."""
        self._current_workspace_context = None
        self._invalidate_all_caches()  # Invalidate caches when context changes


# Backward compatibility aliases
CoreStepRegistry = UnifiedRegistryManager
LocalStepRegistry = UnifiedRegistryManager
HybridRegistryManager = UnifiedRegistryManager
