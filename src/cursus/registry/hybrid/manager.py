"""
Hybrid Registry Manager Implementation

This module provides the core registry management classes for the hybrid registry system,
including CoreStepRegistry, LocalStepRegistry, and HybridRegistryManager.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Set, Union, Any, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from contextlib import contextmanager
from pydantic import BaseModel, Field

from .models import (
    StepDefinition,
    NamespacedStepDefinition,
    ResolutionContext,
    StepResolutionResult,
    RegistryValidationResult,
    ConflictAnalysis,
    StepComponentResolution
)
from .utils import (
    RegistryLoader,
    StepDefinitionConverter,
    RegistryValidationUtils,
    RegistryErrorFormatter
)

logger = logging.getLogger(__name__)


class RegistryConfig(BaseModel):
    """Configuration for registry management."""
    core_registry_path: str
    local_registry_paths: List[str] = Field(default_factory=list)
    workspace_registry_paths: List[str] = Field(default_factory=list)
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    enable_validation: bool = True
    conflict_resolution_strategy: str = "workspace_priority"
    max_concurrent_loads: int = 4


class CoreStepRegistry:
    """
    Core step registry managing centralized step definitions.
    
    This registry handles the core/shared step definitions that are available
    to all workspaces and provides the foundation for the hybrid system.
    """
    
    def __init__(self, registry_path: str, config: Optional[RegistryConfig] = None):
        self.registry_path = Path(registry_path)
        self.config = config or RegistryConfig(core_registry_path=registry_path)
        self._steps: Dict[str, StepDefinition] = {}
        self._loader = RegistryLoader()
        self._converter = StepDefinitionConverter()
        self._validator = RegistryValidationUtils()
        self._error_formatter = RegistryErrorFormatter()
        self._lock = threading.RLock()
        self._loaded = False
        
    def load_registry(self) -> RegistryValidationResult:
        """Load the core registry from the specified path."""
        with self._lock:
            try:
                if not self.registry_path.exists():
                    logger.warning(f"Core registry path does not exist: {self.registry_path}")
                    return RegistryValidationResult(
                        is_valid=False,
                        errors=[f"Registry path not found: {self.registry_path}"],
                        warnings=[],
                        step_count=0
                    )
                
                # Load registry files
                registry_files = list(self.registry_path.glob("*.py"))
                if not registry_files:
                    logger.warning(f"No registry files found in: {self.registry_path}")
                    return RegistryValidationResult(
                        is_valid=True,
                        errors=[],
                        warnings=["No registry files found"],
                        step_count=0
                    )
                
                errors = []
                warnings = []
                loaded_steps = {}
                
                for registry_file in registry_files:
                    try:
                        module = self._loader.load_registry_module(str(registry_file))
                        registry_dict = self._loader.get_registry_attributes(module)
                        
                        # Convert and validate steps
                        for step_name, step_data in registry_dict.items():
                            try:
                                step_def = self._converter.from_legacy_format(step_name, step_data)
                                validation_result = self._validator.validate_step_definition_fields(step_def)
                                
                                if validation_result.is_valid:
                                    loaded_steps[step_name] = step_def
                                else:
                                    errors.extend(validation_result.errors)
                                    warnings.extend(validation_result.warnings)
                                    
                            except Exception as e:
                                error_msg = self._error_formatter.format_registry_load_error(
                                    step_name, str(registry_file), str(e)
                                )
                                errors.append(error_msg)
                                logger.error(error_msg)
                                
                    except Exception as e:
                        error_msg = self._error_formatter.format_registry_load_error(
                            "registry_file", str(registry_file), str(e)
                        )
                        errors.append(error_msg)
                        logger.error(error_msg)
                
                # Update internal state
                self._steps = loaded_steps
                self._loaded = True
                
                logger.info(f"Loaded {len(loaded_steps)} steps from core registry")
                
                return RegistryValidationResult(
                    is_valid=len(errors) == 0,
                    errors=errors,
                    warnings=warnings,
                    step_count=len(loaded_steps)
                )
                
            except Exception as e:
                error_msg = f"Failed to load core registry: {str(e)}"
                logger.error(error_msg)
                return RegistryValidationResult(
                    is_valid=False,
                    errors=[error_msg],
                    warnings=[],
                    step_count=0
                )
    
    def get_step(self, step_name: str) -> Optional[StepDefinition]:
        """Get a step definition by name."""
        with self._lock:
            if not self._loaded:
                self.load_registry()
            return self._steps.get(step_name)
    
    def list_steps(self) -> List[str]:
        """List all available step names."""
        with self._lock:
            if not self._loaded:
                self.load_registry()
            return list(self._steps.keys())
    
    def has_step(self, step_name: str) -> bool:
        """Check if a step exists in the registry."""
        with self._lock:
            if not self._loaded:
                self.load_registry()
            return step_name in self._steps
    
    def get_step_count(self) -> int:
        """Get the total number of steps in the registry."""
        with self._lock:
            if not self._loaded:
                self.load_registry()
            return len(self._steps)
    
    def reload_registry(self) -> RegistryValidationResult:
        """Reload the registry from disk."""
        with self._lock:
            self._loaded = False
            self._steps.clear()
            return self.load_registry()


class LocalStepRegistry:
    """
    Local step registry managing workspace-specific step definitions.
    
    This registry handles step definitions that are specific to a particular
    workspace or developer environment.
    """
    
    def __init__(self, workspace_id: str, registry_paths: List[str], 
                 config: Optional[RegistryConfig] = None):
        self.workspace_id = workspace_id
        self.registry_paths = [Path(p) for p in registry_paths]
        self.config = config or RegistryConfig(core_registry_path="")
        self._steps: Dict[str, NamespacedStepDefinition] = {}
        self._loader = RegistryLoader()
        self._converter = StepDefinitionConverter()
        self._validator = RegistryValidationUtils()
        self._error_formatter = RegistryErrorFormatter()
        self._lock = threading.RLock()
        self._loaded = False
        
    def load_registry(self) -> RegistryValidationResult:
        """Load the local registry from all specified paths."""
        with self._lock:
            try:
                errors = []
                warnings = []
                loaded_steps = {}
                
                for registry_path in self.registry_paths:
                    if not registry_path.exists():
                        warnings.append(f"Local registry path does not exist: {registry_path}")
                        continue
                    
                    # Load registry files from this path
                    registry_files = list(registry_path.glob("*.py"))
                    if not registry_files:
                        warnings.append(f"No registry files found in: {registry_path}")
                        continue
                    
                    for registry_file in registry_files:
                        try:
                            module = self._loader.load_registry_module(str(registry_file))
                            registry_dict = self._loader.get_registry_attributes(module)
                            
                            # Convert and validate steps
                            for step_name, step_data in registry_dict.items():
                                try:
                                    base_step = self._converter.from_legacy_format(step_name, step_data)
                                    
                                    # Create namespaced step definition
                                    namespaced_step = NamespacedStepDefinition(
                                        name=base_step.name,
                                        step_type=base_step.step_type,
                                        script_path=base_step.script_path,
                                        hyperparameters=base_step.hyperparameters,
                                        dependencies=base_step.dependencies,
                                        metadata=base_step.metadata,
                                        namespace=self.workspace_id,
                                        workspace_id=self.workspace_id,
                                        priority=1,  # Local steps have higher priority
                                        source_path=str(registry_file)
                                    )
                                    
                                    validation_result = self._validator.validate_step_definition_fields(namespaced_step)
                                    
                                    if validation_result.is_valid:
                                        loaded_steps[step_name] = namespaced_step
                                    else:
                                        errors.extend(validation_result.errors)
                                        warnings.extend(validation_result.warnings)
                                        
                                except Exception as e:
                                    error_msg = self._error_formatter.format_registry_load_error(
                                        step_name, str(registry_file), str(e)
                                    )
                                    errors.append(error_msg)
                                    logger.error(error_msg)
                                    
                        except Exception as e:
                            error_msg = self._error_formatter.format_registry_load_error(
                                "registry_file", str(registry_file), str(e)
                            )
                            errors.append(error_msg)
                            logger.error(error_msg)
                
                # Update internal state
                self._steps = loaded_steps
                self._loaded = True
                
                logger.info(f"Loaded {len(loaded_steps)} steps from local registry for workspace {self.workspace_id}")
                
                return RegistryValidationResult(
                    is_valid=len(errors) == 0,
                    errors=errors,
                    warnings=warnings,
                    step_count=len(loaded_steps)
                )
                
            except Exception as e:
                error_msg = f"Failed to load local registry for workspace {self.workspace_id}: {str(e)}"
                logger.error(error_msg)
                return RegistryValidationResult(
                    is_valid=False,
                    errors=[error_msg],
                    warnings=[],
                    step_count=0
                )
    
    def get_step(self, step_name: str) -> Optional[NamespacedStepDefinition]:
        """Get a step definition by name."""
        with self._lock:
            if not self._loaded:
                self.load_registry()
            return self._steps.get(step_name)
    
    def list_steps(self) -> List[str]:
        """List all available step names."""
        with self._lock:
            if not self._loaded:
                self.load_registry()
            return list(self._steps.keys())
    
    def has_step(self, step_name: str) -> bool:
        """Check if a step exists in the registry."""
        with self._lock:
            if not self._loaded:
                self.load_registry()
            return step_name in self._steps
    
    def get_step_count(self) -> int:
        """Get the total number of steps in the registry."""
        with self._lock:
            if not self._loaded:
                self.load_registry()
            return len(self._steps)
    
    def reload_registry(self) -> RegistryValidationResult:
        """Reload the registry from disk."""
        with self._lock:
            self._loaded = False
            self._steps.clear()
            return self.load_registry()


class HybridRegistryManager:
    """
    Hybrid registry manager coordinating core and local registries.
    
    This manager provides a unified interface for accessing step definitions
    from both core and local registries, handling conflicts and providing
    intelligent resolution strategies.
    """
    
    def __init__(self, config: RegistryConfig):
        self.config = config
        self.core_registry = CoreStepRegistry(config.core_registry_path, config)
        self.local_registries: Dict[str, LocalStepRegistry] = {}
        self._validator = RegistryValidationUtils()
        self._error_formatter = RegistryErrorFormatter()
        self._lock = threading.RLock()
        
        # Initialize local registries
        for i, path in enumerate(config.local_registry_paths):
            workspace_id = f"local_{i}"
            self.local_registries[workspace_id] = LocalStepRegistry(
                workspace_id, [path], config
            )
        
        # Initialize workspace registries
        for i, path in enumerate(config.workspace_registry_paths):
            workspace_id = f"workspace_{i}"
            self.local_registries[workspace_id] = LocalStepRegistry(
                workspace_id, [path], config
            )
    
    def add_workspace_registry(self, workspace_id: str, registry_paths: List[str]) -> None:
        """Add a new workspace registry."""
        with self._lock:
            if workspace_id in self.local_registries:
                logger.warning(f"Workspace registry {workspace_id} already exists, replacing")
            
            self.local_registries[workspace_id] = LocalStepRegistry(
                workspace_id, registry_paths, self.config
            )
            logger.info(f"Added workspace registry: {workspace_id}")
    
    def remove_workspace_registry(self, workspace_id: str) -> bool:
        """Remove a workspace registry."""
        with self._lock:
            if workspace_id in self.local_registries:
                del self.local_registries[workspace_id]
                logger.info(f"Removed workspace registry: {workspace_id}")
                return True
            return False
    
    def load_all_registries(self) -> Dict[str, RegistryValidationResult]:
        """Load all registries (core and local)."""
        results = {}
        
        # Load core registry
        logger.info("Loading core registry...")
        results["core"] = self.core_registry.load_registry()
        
        # Load local registries
        if self.config.max_concurrent_loads > 1:
            # Concurrent loading
            with ThreadPoolExecutor(max_workers=self.config.max_concurrent_loads) as executor:
                future_to_workspace = {
                    executor.submit(registry.load_registry): workspace_id
                    for workspace_id, registry in self.local_registries.items()
                }
                
                for future in as_completed(future_to_workspace):
                    workspace_id = future_to_workspace[future]
                    try:
                        results[workspace_id] = future.result()
                        logger.info(f"Loaded registry for workspace: {workspace_id}")
                    except Exception as e:
                        error_msg = f"Failed to load registry for workspace {workspace_id}: {str(e)}"
                        logger.error(error_msg)
                        results[workspace_id] = RegistryValidationResult(
                            is_valid=False,
                            errors=[error_msg],
                            warnings=[],
                            step_count=0
                        )
        else:
            # Sequential loading
            for workspace_id, registry in self.local_registries.items():
                logger.info(f"Loading registry for workspace: {workspace_id}")
                results[workspace_id] = registry.load_registry()
        
        return results
    
    def get_step(self, step_name: str, context: Optional[ResolutionContext] = None) -> StepResolutionResult:
        """
        Get a step definition with conflict resolution.
        
        Args:
            step_name: Name of the step to retrieve
            context: Resolution context for conflict handling
            
        Returns:
            StepResolutionResult containing the resolved step and metadata
        """
        if context is None:
            context = ResolutionContext(
                workspace_id="default",
                resolution_strategy=self.config.conflict_resolution_strategy
            )
        
        # Collect all matching steps
        candidates = []
        
        # Check core registry
        core_step = self.core_registry.get_step(step_name)
        if core_step:
            candidates.append((core_step, "core", 0))  # Core has lowest priority
        
        # Check local registries
        for workspace_id, registry in self.local_registries.items():
            local_step = registry.get_step(step_name)
            if local_step:
                priority = local_step.priority if hasattr(local_step, 'priority') else 1
                candidates.append((local_step, workspace_id, priority))
        
        if not candidates:
            error_msg = self._error_formatter.format_step_not_found_error(step_name, list(self.local_registries.keys()))
            return StepResolutionResult(
                step_definition=None,
                source_registry="none",
                workspace_id=context.workspace_id,
                resolution_strategy=context.resolution_strategy,
                conflict_detected=False,
                conflict_analysis=None,
                errors=[error_msg],
                warnings=[]
            )
        
        # Handle single candidate (no conflict)
        if len(candidates) == 1:
            step_def, source, priority = candidates[0]
            return StepResolutionResult(
                step_definition=step_def,
                source_registry=source,
                workspace_id=context.workspace_id,
                resolution_strategy=context.resolution_strategy,
                conflict_detected=False,
                conflict_analysis=None,
                errors=[],
                warnings=[]
            )
        
        # Handle multiple candidates (conflict resolution)
        return self._resolve_step_conflict(step_name, candidates, context)
    
    def _resolve_step_conflict(self, step_name: str, candidates: List[Tuple], 
                             context: ResolutionContext) -> StepResolutionResult:
        """Resolve conflicts between multiple step definitions."""
        
        # Create conflict analysis
        conflict_analysis = ConflictAnalysis(
            step_name=step_name,
            conflicting_sources=[source for _, source, _ in candidates],
            resolution_strategy=context.resolution_strategy,
            workspace_context=context.workspace_id
        )
        
        # Apply resolution strategy
        if context.resolution_strategy == "workspace_priority":
            # Prefer workspace-specific steps, then by priority
            workspace_candidates = [c for c in candidates if c[1] == context.workspace_id]
            if workspace_candidates:
                selected = max(workspace_candidates, key=lambda x: x[2])
            else:
                selected = max(candidates, key=lambda x: x[2])
                
        elif context.resolution_strategy == "highest_priority":
            # Select by highest priority
            selected = max(candidates, key=lambda x: x[2])
            
        elif context.resolution_strategy == "core_fallback":
            # Prefer core registry
            core_candidates = [c for c in candidates if c[1] == "core"]
            if core_candidates:
                selected = core_candidates[0]
            else:
                selected = max(candidates, key=lambda x: x[2])
                
        else:
            # Default to highest priority
            selected = max(candidates, key=lambda x: x[2])
        
        step_def, source, priority = selected
        
        # Generate warnings about conflicts
        warnings = []
        other_sources = [c[1] for c in candidates if c[1] != source]
        if other_sources:
            warnings.append(f"Step '{step_name}' found in multiple registries: {other_sources}. Using {source}.")
        
        return StepResolutionResult(
            step_definition=step_def,
            source_registry=source,
            workspace_id=context.workspace_id,
            resolution_strategy=context.resolution_strategy,
            conflict_detected=True,
            conflict_analysis=conflict_analysis,
            errors=[],
            warnings=warnings
        )
    
    def list_all_steps(self, include_source: bool = False) -> Union[List[str], Dict[str, List[str]]]:
        """List all available steps across all registries."""
        if include_source:
            result = {}
            result["core"] = self.core_registry.list_steps()
            for workspace_id, registry in self.local_registries.items():
                result[workspace_id] = registry.list_steps()
            return result
        else:
            all_steps = set(self.core_registry.list_steps())
            for registry in self.local_registries.values():
                all_steps.update(registry.list_steps())
            return sorted(list(all_steps))
    
    def get_registry_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status information for all registries."""
        status = {}
        
        # Core registry status
        status["core"] = {
            "loaded": self.core_registry._loaded,
            "step_count": self.core_registry.get_step_count(),
            "registry_path": str(self.core_registry.registry_path)
        }
        
        # Local registry status
        for workspace_id, registry in self.local_registries.items():
            status[workspace_id] = {
                "loaded": registry._loaded,
                "step_count": registry.get_step_count(),
                "registry_paths": [str(p) for p in registry.registry_paths],
                "workspace_id": registry.workspace_id
            }
        
        return status
    
    @contextmanager
    def resolution_context(self, workspace_id: str, strategy: str = None):
        """Context manager for step resolution."""
        context = ResolutionContext(
            workspace_id=workspace_id,
            resolution_strategy=strategy or self.config.conflict_resolution_strategy
        )
        yield context
