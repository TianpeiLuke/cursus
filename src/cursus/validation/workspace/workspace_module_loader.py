"""
Workspace Module Loader

Provides workspace-aware dynamic module loading with Python path management.
Handles module discovery, loading, and isolation for multi-developer workspaces.

Architecture:
- Extends existing dynamic module loading capabilities
- Manages Python sys.path for workspace isolation
- Supports developer workspace module discovery
- Maintains backward compatibility with single workspace mode
- Provides context managers for safe path manipulation

Features:
- Workspace-specific module path management
- Dynamic module loading with fallback to shared workspace
- Python path isolation and cleanup
- Module caching and invalidation
- Developer workspace discovery
"""

import os
import sys
import importlib
import importlib.util
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Type, ContextManager
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class WorkspaceModuleLoader:
    """
    Workspace-aware module loader that provides dynamic module loading
    with Python path management for multi-developer workspaces.
    
    Features:
    - Workspace-specific Python path management
    - Dynamic module loading with workspace isolation
    - Fallback to shared workspace modules
    - Module caching and invalidation
    - Context managers for safe path manipulation
    """
    
    def __init__(
        self,
        workspace_root: Optional[Union[str, Path]] = None,
        developer_id: Optional[str] = None,
        enable_shared_fallback: bool = True,
        cache_modules: bool = True
    ):
        """
        Initialize workspace module loader.
        
        Args:
            workspace_root: Root directory containing developer workspaces
            developer_id: Specific developer workspace to target
            enable_shared_fallback: Whether to fallback to shared workspace
            cache_modules: Whether to cache loaded modules
        """
        self.workspace_root = Path(workspace_root) if workspace_root else None
        self.developer_id = developer_id
        self.enable_shared_fallback = enable_shared_fallback
        self.cache_modules = cache_modules
        
        # Module cache
        self._module_cache: Dict[str, Any] = {}
        self._path_cache: Dict[str, List[str]] = {}
        
        # Workspace mode detection
        self.workspace_mode = workspace_root is not None
        
        if self.workspace_mode:
            self._validate_workspace_structure()
            self._build_workspace_paths()
        
        logger.info(f"Initialized workspace module loader for developer '{developer_id}' "
                   f"at '{workspace_root}'")
    
    def _validate_workspace_structure(self) -> None:
        """Validate that workspace root has expected structure."""
        if not self.workspace_root.exists():
            raise ValueError(f"Workspace root does not exist: {self.workspace_root}")
        
        developers_dir = self.workspace_root / "developers"
        shared_dir = self.workspace_root / "shared"
        
        if not developers_dir.exists() and not shared_dir.exists():
            raise ValueError(
                f"Workspace root must contain 'developers' or 'shared' directory: "
                f"{self.workspace_root}"
            )
        
        if self.developer_id:
            dev_workspace = developers_dir / self.developer_id
            if not dev_workspace.exists():
                raise ValueError(
                    f"Developer workspace does not exist: {dev_workspace}"
                )
    
    def _build_workspace_paths(self) -> None:
        """Build workspace-specific Python paths."""
        self.developer_paths = []
        self.shared_paths = []
        
        if self.developer_id:
            # Developer workspace paths
            dev_base = (self.workspace_root / "developers" / self.developer_id / 
                       "src")
            
            if dev_base.exists():
                self.developer_paths = [
                    str(dev_base),
                    str(dev_base / "cursus_dev"),
                    str(dev_base / "cursus_dev" / "steps")
                ]
        
        # Shared workspace paths
        if self.enable_shared_fallback:
            shared_base = self.workspace_root / "shared" / "src"
            
            if shared_base.exists():
                self.shared_paths = [
                    str(shared_base),
                    str(shared_base / "cursus_dev"),
                    str(shared_base / "cursus_dev" / "steps")
                ]
    
    @contextmanager
    def workspace_path_context(
        self,
        include_developer: bool = True,
        include_shared: bool = None
    ) -> ContextManager[List[str]]:
        """
        Context manager for workspace-aware Python path manipulation.
        
        Args:
            include_developer: Whether to include developer workspace paths
            include_shared: Whether to include shared workspace paths
                          (defaults to self.enable_shared_fallback)
        
        Yields:
            List of paths added to sys.path
        """
        if include_shared is None:
            include_shared = self.enable_shared_fallback
        
        if not self.workspace_mode:
            yield []
            return
        
        # Collect paths to add
        paths_to_add = []
        
        if include_developer and hasattr(self, 'developer_paths'):
            paths_to_add.extend(self.developer_paths)
        
        if include_shared and hasattr(self, 'shared_paths'):
            paths_to_add.extend(self.shared_paths)
        
        # Filter existing paths and add to sys.path
        original_path = sys.path.copy()
        added_paths = []
        
        for path in paths_to_add:
            if path and os.path.exists(path) and path not in sys.path:
                sys.path.insert(0, path)
                added_paths.append(path)
        
        try:
            yield added_paths
        finally:
            # Restore original sys.path
            sys.path[:] = original_path
    
    def load_builder_class(
        self,
        step_name: str,
        builder_module_name: Optional[str] = None,
        builder_class_name: Optional[str] = None
    ) -> Optional[Type]:
        """
        Load step builder class with workspace-aware module loading.
        
        Args:
            step_name: Name of the step
            builder_module_name: Specific module name to load
            builder_class_name: Specific class name to load
        
        Returns:
            Builder class if found, None otherwise
        """
        cache_key = f"builder:{step_name}:{builder_module_name}:{builder_class_name}"
        
        # Check cache first
        if self.cache_modules and cache_key in self._module_cache:
            return self._module_cache[cache_key]
        
        builder_class = None
        
        # Try workspace-aware loading
        if self.workspace_mode:
            with self.workspace_path_context():
                builder_class = self._load_builder_class_impl(
                    step_name, builder_module_name, builder_class_name
                )
        else:
            # Fallback to standard loading
            builder_class = self._load_builder_class_impl(
                step_name, builder_module_name, builder_class_name
            )
        
        # Cache result
        if self.cache_modules:
            self._module_cache[cache_key] = builder_class
        
        return builder_class
    
    def _load_builder_class_impl(
        self,
        step_name: str,
        builder_module_name: Optional[str],
        builder_class_name: Optional[str]
    ) -> Optional[Type]:
        """Implementation of builder class loading."""
        # Determine module and class names
        if not builder_module_name:
            builder_module_name = f"{step_name}_builder"
        
        if not builder_class_name:
            # Convert step_name to class name (e.g., "create_model" -> "CreateModelBuilder")
            class_name_parts = step_name.split('_')
            builder_class_name = ''.join(word.capitalize() for word in class_name_parts) + 'Builder'
        
        # Try different module path patterns
        module_patterns = [
            f"cursus_dev.steps.builders.{builder_module_name}",
            f"steps.builders.{builder_module_name}",
            f"builders.{builder_module_name}",
            builder_module_name
        ]
        
        for module_pattern in module_patterns:
            try:
                module = importlib.import_module(module_pattern)
                if hasattr(module, builder_class_name):
                    builder_class = getattr(module, builder_class_name)
                    logger.debug(f"Loaded builder class {builder_class_name} "
                               f"from {module_pattern}")
                    return builder_class
            except ImportError as e:
                logger.debug(f"Failed to import {module_pattern}: {e}")
                continue
        
        return None
    
    def load_contract_class(
        self,
        step_name: str,
        contract_module_name: Optional[str] = None,
        contract_class_name: Optional[str] = None
    ) -> Optional[Type]:
        """
        Load script contract class with workspace-aware module loading.
        
        Args:
            step_name: Name of the step
            contract_module_name: Specific module name to load
            contract_class_name: Specific class name to load
        
        Returns:
            Contract class if found, None otherwise
        """
        cache_key = f"contract:{step_name}:{contract_module_name}:{contract_class_name}"
        
        # Check cache first
        if self.cache_modules and cache_key in self._module_cache:
            return self._module_cache[cache_key]
        
        contract_class = None
        
        # Try workspace-aware loading
        if self.workspace_mode:
            with self.workspace_path_context():
                contract_class = self._load_contract_class_impl(
                    step_name, contract_module_name, contract_class_name
                )
        else:
            # Fallback to standard loading
            contract_class = self._load_contract_class_impl(
                step_name, contract_module_name, contract_class_name
            )
        
        # Cache result
        if self.cache_modules:
            self._module_cache[cache_key] = contract_class
        
        return contract_class
    
    def _load_contract_class_impl(
        self,
        step_name: str,
        contract_module_name: Optional[str],
        contract_class_name: Optional[str]
    ) -> Optional[Type]:
        """Implementation of contract class loading."""
        # Determine module and class names
        if not contract_module_name:
            contract_module_name = f"{step_name}_contract"
        
        if not contract_class_name:
            # Convert step_name to class name (e.g., "create_model" -> "CreateModelContract")
            class_name_parts = step_name.split('_')
            contract_class_name = ''.join(word.capitalize() for word in class_name_parts) + 'Contract'
        
        # Try different module path patterns
        module_patterns = [
            f"cursus_dev.steps.contracts.{contract_module_name}",
            f"steps.contracts.{contract_module_name}",
            f"contracts.{contract_module_name}",
            contract_module_name
        ]
        
        for module_pattern in module_patterns:
            try:
                module = importlib.import_module(module_pattern)
                if hasattr(module, contract_class_name):
                    contract_class = getattr(module, contract_class_name)
                    logger.debug(f"Loaded contract class {contract_class_name} "
                               f"from {module_pattern}")
                    return contract_class
            except ImportError as e:
                logger.debug(f"Failed to import {module_pattern}: {e}")
                continue
        
        return None
    
    def load_module_from_file(
        self,
        file_path: Union[str, Path],
        module_name: Optional[str] = None
    ) -> Optional[Any]:
        """
        Load module from specific file path.
        
        Args:
            file_path: Path to the Python file
            module_name: Name to assign to the module
        
        Returns:
            Loaded module if successful, None otherwise
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return None
        
        if not module_name:
            module_name = file_path.stem
        
        cache_key = f"file:{file_path}:{module_name}"
        
        # Check cache first
        if self.cache_modules and cache_key in self._module_cache:
            return self._module_cache[cache_key]
        
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Cache result
            if self.cache_modules:
                self._module_cache[cache_key] = module
            
            logger.debug(f"Loaded module {module_name} from {file_path}")
            return module
            
        except Exception as e:
            logger.error(f"Failed to load module from {file_path}: {e}")
            return None
    
    def discover_workspace_modules(
        self,
        module_type: str = "builders"
    ) -> Dict[str, List[str]]:
        """
        Discover available modules in workspace.
        
        Args:
            module_type: Type of modules to discover (builders, contracts, etc.)
        
        Returns:
            Dictionary mapping workspace to list of module names
        """
        if not self.workspace_mode:
            return {}
        
        discovered = {}
        
        # Discover in developer workspace
        if self.developer_id:
            dev_modules_dir = (self.workspace_root / "developers" / self.developer_id / 
                             "src" / "cursus_dev" / "steps" / module_type)
            
            if dev_modules_dir.exists():
                modules = self._discover_modules_in_directory(dev_modules_dir)
                if modules:
                    discovered[f"developer:{self.developer_id}"] = modules
        
        # Discover in shared workspace
        if self.enable_shared_fallback:
            shared_modules_dir = (self.workspace_root / "shared" / "src" / 
                                "cursus_dev" / "steps" / module_type)
            
            if shared_modules_dir.exists():
                modules = self._discover_modules_in_directory(shared_modules_dir)
                if modules:
                    discovered["shared"] = modules
        
        return discovered
    
    def _discover_modules_in_directory(self, directory: Path) -> List[str]:
        """Discover Python modules in a directory."""
        modules = []
        
        if not directory.exists():
            return modules
        
        for item in directory.iterdir():
            if item.is_file() and item.suffix == '.py' and item.name != '__init__.py':
                modules.append(item.stem)
            elif item.is_dir() and (item / '__init__.py').exists():
                modules.append(item.name)
        
        return sorted(modules)
    
    def clear_cache(self) -> None:
        """Clear module cache."""
        self._module_cache.clear()
        self._path_cache.clear()
        logger.debug("Cleared module cache")
    
    def invalidate_cache_for_step(self, step_name: str) -> None:
        """Invalidate cache entries for a specific step."""
        keys_to_remove = [
            key for key in self._module_cache.keys()
            if key.startswith(f"builder:{step_name}:") or 
               key.startswith(f"contract:{step_name}:")
        ]
        
        for key in keys_to_remove:
            del self._module_cache[key]
        
        logger.debug(f"Invalidated cache for step: {step_name}")
    
    def get_workspace_info(self) -> Dict[str, Any]:
        """Get information about current workspace configuration."""
        return {
            'workspace_mode': self.workspace_mode,
            'workspace_root': str(self.workspace_root) if self.workspace_root else None,
            'developer_id': self.developer_id,
            'enable_shared_fallback': self.enable_shared_fallback,
            'cache_modules': self.cache_modules,
            'developer_paths': getattr(self, 'developer_paths', []),
            'shared_paths': getattr(self, 'shared_paths', []),
            'cached_modules': len(self._module_cache),
        }
    
    def switch_developer(self, developer_id: str) -> None:
        """Switch to a different developer workspace."""
        if not self.workspace_mode:
            raise ValueError("Not in workspace mode")
        
        # Validate new developer workspace
        dev_workspace = self.workspace_root / "developers" / developer_id
        if not dev_workspace.exists():
            raise ValueError(f"Developer workspace not found: {developer_id}")
        
        self.developer_id = developer_id
        
        # Rebuild workspace paths
        self._build_workspace_paths()
        
        # Clear cache to avoid cross-contamination
        self.clear_cache()
        
        logger.info(f"Switched to developer workspace: {developer_id}")
