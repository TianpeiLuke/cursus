"""
Builder class auto-discovery with workspace support.

This module provides AST-based builder class discovery that mirrors the ConfigAutoDiscovery
architecture for consistency. It handles deployment portability internally and supports
both package and workspace builder discovery.
"""

import ast
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Type, Any
import importlib.util

logger = logging.getLogger(__name__)


class BuilderAutoDiscovery:
    """
    AST-based builder class discovery with workspace support.
    
    Mirrors ConfigAutoDiscovery architecture for consistency and handles
    deployment portability internally.
    """
    
    def __init__(self, package_root: Path, workspace_dirs: Optional[List[Path]] = None):
        """
        Initialize builder discovery with package and workspace search spaces.
        
        Args:
            package_root: Root directory of the cursus package
            workspace_dirs: Optional list of workspace directories to search
        """
        # Initialize logger FIRST before any other operations
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🔧 BuilderAutoDiscovery.__init__ starting - package_root: {package_root}")
        self.logger.info(f"🔧 BuilderAutoDiscovery.__init__ - workspace_dirs: {workspace_dirs}")
        
        try:
            # Handle sys.path setup internally for deployment portability
            self.logger.debug("🔧 Calling _ensure_cursus_importable()")
            self._ensure_cursus_importable()
            self.logger.debug("✅ _ensure_cursus_importable() completed successfully")
        except Exception as e:
            self.logger.error(f"❌ _ensure_cursus_importable() failed: {e}")
            raise
        
        self.package_root = package_root
        self.workspace_dirs = workspace_dirs or []
        self.logger.info(f"✅ BuilderAutoDiscovery basic initialization complete")
        
        # Caches for performance
        self._builder_cache: Dict[str, Type] = {}
        self._builder_paths: Dict[str, Path] = {}
        self._discovery_complete = False
        
        # Discovery results
        self._package_builders: Dict[str, Type] = {}
        self._workspace_builders: Dict[str, Dict[str, Type]] = {}  # workspace_id -> builders
        
        # Registry integration
        self._registry_info: Dict[str, Dict[str, Any]] = {}
        
        try:
            self.logger.debug("🔧 Loading registry info...")
            self._load_registry_info()
            self.logger.info(f"✅ Registry info loaded: {len(self._registry_info)} steps")
        except Exception as e:
            self.logger.error(f"❌ Registry info loading failed: {e}")
            self._registry_info = {}
        
        self.logger.info("🎉 BuilderAutoDiscovery initialization completed successfully")
    
    def _ensure_cursus_importable(self):
        """
        Internal sys.path setup for deployment portability.
        
        This handles the importlib deployment issue internally so consumers
        don't need to worry about it.
        """
        current_file = Path(__file__).resolve()
        current_path = current_file
        while current_path.parent != current_path:
            if current_path.name == 'cursus':
                cursus_parent = str(current_path.parent)
                if cursus_parent not in sys.path:
                    sys.path.insert(0, cursus_parent)
                    self.logger.debug(f"Added cursus parent to sys.path: {cursus_parent}")
                break
            current_path = current_path.parent
    
    def _load_registry_info(self):
        """
        Load registry information from cursus/registry/step_names.py.
        
        This provides authoritative information about step names, builder classes,
        and other metadata that can guide the discovery process.
        """
        try:
            from ..registry.step_names import get_step_names
            
            step_names_dict = get_step_names()
            for step_name, step_info in step_names_dict.items():
                self._registry_info[step_name] = step_info
            
            self.logger.debug(f"Loaded registry info for {len(self._registry_info)} steps")
            
        except ImportError as e:
            self.logger.warning(f"Could not import registry step_names: {e}")
            self._registry_info = {}
        except Exception as e:
            self.logger.error(f"Error loading registry info: {e}")
            self._registry_info = {}
    
    def _get_registry_builder_info(self, step_name: str) -> Optional[Dict[str, Any]]:
        """
        Get builder information from registry for a step.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Dictionary with builder information from registry or None
        """
        if step_name in self._registry_info:
            step_info = self._registry_info[step_name]
            return {
                "builder_step_name": step_info.get("builder_step_name"),
                "sagemaker_step_type": step_info.get("sagemaker_step_type"),
                "step_type": step_info.get("step_type"),
                "module_path": step_info.get("module_path"),
                "class_name": step_info.get("class_name")
            }
        return None
    
    def discover_builder_classes(self, project_id: Optional[str] = None) -> Dict[str, Type]:
        """
        Discover all builder classes from package and workspaces.
        
        Args:
            project_id: Optional project ID for workspace-specific discovery
            
        Returns:
            Dictionary mapping step names to builder class types
        """
        if not self._discovery_complete:
            self._run_discovery()
        
        # Combine package and workspace builders
        all_builders = {}
        
        # Add package builders
        all_builders.update(self._package_builders)
        
        # Add workspace builders (workspace overrides package)
        if project_id and project_id in self._workspace_builders:
            all_builders.update(self._workspace_builders[project_id])
        else:
            # Add all workspace builders if no specific project
            for workspace_builders in self._workspace_builders.values():
                all_builders.update(workspace_builders)
        
        self.logger.info(f"Discovered {len(all_builders)} builder classes")
        return all_builders
    
    def load_builder_class(self, step_name: str) -> Optional[Type]:
        """
        Load builder class for a specific step with workspace-aware discovery.
        
        Args:
            step_name: Name of the step to load builder for
            
        Returns:
            Builder class type or None if not found
        """
        # Check cache first
        if step_name in self._builder_cache:
            return self._builder_cache[step_name]
        
        # Ensure discovery is complete
        if not self._discovery_complete:
            self._run_discovery()
        
        # Try workspace builders first (higher priority)
        for workspace_id, workspace_builders in self._workspace_builders.items():
            if step_name in workspace_builders:
                builder_class = workspace_builders[step_name]
                self._builder_cache[step_name] = builder_class
                self.logger.debug(f"Loaded builder for {step_name} from workspace {workspace_id}")
                return builder_class
        
        # Try package builders
        if step_name in self._package_builders:
            builder_class = self._package_builders[step_name]
            self._builder_cache[step_name] = builder_class
            self.logger.debug(f"Loaded builder for {step_name} from package")
            return builder_class
        
        # Not found
        self.logger.warning(f"No builder class found for step: {step_name}")
        return None
    
    def _run_discovery(self):
        """Run the complete discovery process."""
        try:
            # Discover package builders
            self._discover_package_builders()
            
            # Discover workspace builders
            self._discover_workspace_builders()
            
            self._discovery_complete = True
            
            total_builders = len(self._package_builders) + sum(
                len(builders) for builders in self._workspace_builders.values()
            )
            self.logger.info(f"Builder discovery complete: {total_builders} builders found")
            
        except Exception as e:
            self.logger.error(f"Error during builder discovery: {e}")
            # Graceful degradation
            self._package_builders = {}
            self._workspace_builders = {}
    
    def _discover_package_builders(self):
        """Discover builders in the cursus package."""
        package_builders_dir = self.package_root / "steps" / "builders"
        if package_builders_dir.exists():
            self._package_builders = self._scan_builder_directory(
                package_builders_dir, "package"
            )
            self.logger.debug(f"Found {len(self._package_builders)} package builders")
    
    def _discover_workspace_builders(self):
        """Discover builders in workspace directories with simplified structure."""
        for workspace_dir in self.workspace_dirs:
            try:
                workspace_path = Path(workspace_dir)
                if not workspace_path.exists():
                    self.logger.warning(f"Workspace directory does not exist: {workspace_path}")
                    continue
                
                # Simplified structure: workspace_dir directly contains builders/
                workspace_builders_dir = workspace_path / "builders"
                if workspace_builders_dir.exists():
                    workspace_builders = self._scan_builder_directory(
                        workspace_builders_dir, workspace_path.name
                    )
                    if workspace_builders:
                        self._workspace_builders[workspace_path.name] = workspace_builders
                        self.logger.debug(f"Found {len(workspace_builders)} builders in workspace {workspace_path.name}")
                else:
                    self.logger.debug(f"No builders directory found in workspace: {workspace_path}")
                    
            except Exception as e:
                self.logger.error(f"Error discovering workspace builders in {workspace_dir}: {e}")
    
    def _scan_builder_directory(self, builders_dir: Path, workspace_id: str) -> Dict[str, Type]:
        """
        Scan directory for builder files using AST analysis.
        
        Args:
            builders_dir: Directory containing builder files
            workspace_id: ID of the workspace (for logging)
            
        Returns:
            Dictionary mapping step names to builder classes
        """
        builders = {}
        
        if not builders_dir.exists():
            return builders
        
        try:
            for py_file in builders_dir.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue
                
                try:
                    # Extract builder info using AST
                    builder_info = self._extract_builder_from_ast(py_file)
                    if builder_info:
                        step_name, builder_class = builder_info
                        builders[step_name] = builder_class
                        self._builder_paths[step_name] = py_file
                        
                except Exception as e:
                    self.logger.warning(f"Error processing builder file {py_file}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error scanning builder directory {builders_dir}: {e}")
        
        return builders
    
    def _extract_builder_from_ast(self, file_path: Path) -> Optional[tuple]:
        """
        Extract builder class from Python file using AST analysis.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Tuple of (step_name, builder_class) or None if not found
        """
        try:
            # Read and parse the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Find classes that inherit from StepBuilderBase
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if class inherits from StepBuilderBase
                    if self._inherits_from_step_builder_base(node):
                        # Extract step name from file name or class name
                        step_name = self._extract_step_name_from_builder_file(file_path, node.name)
                        if step_name:
                            # Load the actual class
                            builder_class = self._load_class_from_file(file_path, node.name)
                            if builder_class:
                                return (step_name, builder_class)
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error extracting builder from {file_path}: {e}")
            return None
    
    def _inherits_from_step_builder_base(self, class_node: ast.ClassDef) -> bool:
        """
        Check if a class inherits from StepBuilderBase.
        
        Args:
            class_node: AST class definition node
            
        Returns:
            True if class inherits from StepBuilderBase
        """
        for base in class_node.bases:
            if isinstance(base, ast.Name) and base.id == 'StepBuilderBase':
                return True
            elif isinstance(base, ast.Attribute):
                # Handle qualified names like module.StepBuilderBase
                if base.attr == 'StepBuilderBase':
                    return True
        return False
    
    def _extract_step_name_from_builder_file(self, file_path: Path, class_name: str) -> Optional[str]:
        """
        Extract step name from builder file name and class name using registry information.
        
        Args:
            file_path: Path to the builder file
            class_name: Name of the builder class
            
        Returns:
            Step name or None if not extractable
        """
        # First, try to find step name from registry by matching builder class name
        for step_name, step_info in self._registry_info.items():
            builder_step_name = step_info.get("builder_step_name")
            if builder_step_name and builder_step_name == class_name:
                self.logger.debug(f"Found step name {step_name} for builder {class_name} via registry")
                return step_name
        
        # Try to extract from file name (e.g., builder_xgboost_training_step.py)
        file_name = file_path.stem
        if file_name.startswith("builder_") and file_name.endswith("_step"):
            # Remove builder_ prefix and _step suffix
            step_name_parts = file_name[8:-5].split("_")  # Remove "builder_" and "_step"
            
            # Apply special case handling for known patterns
            step_name = self._convert_parts_to_pascal_case_with_special_cases(step_name_parts)
            
            # Validate against registry if possible
            if step_name in self._registry_info:
                return step_name
            
            # Try variations if exact match not found
            for registered_step in self._registry_info.keys():
                if registered_step.lower() == step_name.lower():
                    return registered_step
            
            return step_name
        
        # Try to extract from class name (e.g., XGBoostTrainingStepBuilder)
        if class_name.endswith("StepBuilder"):
            step_name = class_name[:-11]  # Remove "StepBuilder"
        elif class_name.endswith("Builder"):
            step_name = class_name[:-7]  # Remove "Builder"
        else:
            step_name = class_name
        
        # Validate against registry
        if step_name in self._registry_info:
            return step_name
        
        # Try case-insensitive match
        for registered_step in self._registry_info.keys():
            if registered_step.lower() == step_name.lower():
                return registered_step
        
        # Fallback: use extracted name as-is
        return step_name
    
    def _convert_parts_to_pascal_case_with_special_cases(self, parts: List[str]) -> str:
        """
        Convert file name parts to PascalCase with special case handling.
        
        Args:
            parts: List of file name parts (e.g., ['xgboost', 'training'])
            
        Returns:
            PascalCase step name with proper special case handling
        """
        result_parts = []
        
        for part in parts:
            # Handle special cases
            if part.lower() == 'xgboost':
                result_parts.append('XGBoost')
            elif part.lower() == 'pytorch':
                result_parts.append('PyTorch')
            else:
                result_parts.append(part.capitalize())
        
        return "".join(result_parts)
    
    def _load_class_from_file(self, file_path: Path, class_name: str) -> Optional[Type]:
        """
        Load class using relative imports with package parameter (deployment-agnostic).
        
        This approach uses importlib.import_module with relative paths and package parameter,
        which is cleaner than sys.path manipulation and works across all deployment scenarios.
        
        Args:
            file_path: Path to the Python file
            class_name: Name of the class to load
            
        Returns:
            Class type or None if loading fails
        """
        try:
            # Convert file path to relative module path
            relative_module_path = self._file_to_relative_module_path(file_path)
            if not relative_module_path:
                self.logger.warning(f"Could not determine relative module path for {file_path}")
                return None
            
            # Import the module using relative import with package parameter
            import importlib
            module = importlib.import_module(relative_module_path, package=__package__)
            
            # Get the class from the module
            if hasattr(module, class_name):
                return getattr(module, class_name)
            else:
                self.logger.warning(f"Class {class_name} not found in module {relative_module_path}")
                return None
                
        except Exception as e:
            self.logger.warning(f"Error loading class {class_name} from {file_path} (relative module: {relative_module_path}): {e}")
            return None
    
    def _file_to_relative_module_path(self, file_path: Path) -> Optional[str]:
        """
        Convert file path to relative module path for use with importlib.import_module.
        
        This creates relative import paths like "..steps.builders.builder_xgboost_training_step"
        that work with the package parameter in importlib.import_module.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Relative module path string (e.g., '..steps.builders.builder_xgboost_training_step')
        """
        try:
            # Get the path relative to the package root
            try:
                relative_path = file_path.relative_to(self.package_root)
            except ValueError:
                # File is not under package root, might be in workspace
                self.logger.debug(f"File {file_path} not under package root {self.package_root}")
                return None
            
            # Convert path to module format
            parts = list(relative_path.parts)
            
            # Remove .py extension from the last part
            if parts[-1].endswith('.py'):
                parts[-1] = parts[-1][:-3]
            
            # Create relative module path with .. prefix for relative import
            # This works with importlib.import_module(relative_path, package=__package__)
            relative_module_path = '..' + '.'.join(parts)
            
            self.logger.debug(f"Converted {file_path} to relative module path: {relative_module_path}")
            return relative_module_path
            
        except Exception as e:
            self.logger.warning(f"Error converting file path {file_path} to relative module path: {e}")
            return None
    
    def _file_to_module_path(self, file_path: Path) -> Optional[str]:
        """
        Convert file path to Python module path (legacy method for compatibility).
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Module path string (e.g., 'cursus.steps.builders.builder_xgboost_training_step')
        """
        try:
            # Get the path relative to the package root
            try:
                relative_path = file_path.relative_to(self.package_root)
            except ValueError:
                # File is not under package root, might be in workspace
                self.logger.debug(f"File {file_path} not under package root {self.package_root}")
                return None
            
            # Convert path to module format
            parts = list(relative_path.parts)
            
            # Remove .py extension from the last part
            if parts[-1].endswith('.py'):
                parts[-1] = parts[-1][:-3]
            
            # Create module path with cursus prefix
            module_path = 'cursus.' + '.'.join(parts)
            
            self.logger.debug(f"Converted {file_path} to module path: {module_path}")
            return module_path
            
        except Exception as e:
            self.logger.warning(f"Error converting file path {file_path} to module path: {e}")
            return None
    
    def get_builder_info(self, step_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a builder.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Dictionary with builder information or None if not found
        """
        builder_class = self.load_builder_class(step_name)
        if not builder_class:
            return None
        
        return {
            "step_name": step_name,
            "builder_class": builder_class.__name__,
            "module": builder_class.__module__,
            "file_path": str(self._builder_paths.get(step_name, "Unknown")),
            "workspace_id": self._get_workspace_for_step(step_name)
        }
    
    def _get_workspace_for_step(self, step_name: str) -> str:
        """Get workspace ID for a step."""
        for workspace_id, workspace_builders in self._workspace_builders.items():
            if step_name in workspace_builders:
                return workspace_id
        return "package"
    
    def list_available_builders(self) -> List[str]:
        """
        List all available builder step names.
        
        Returns:
            List of step names that have builders
        """
        if not self._discovery_complete:
            self._run_discovery()
        
        all_steps = set(self._package_builders.keys())
        for workspace_builders in self._workspace_builders.values():
            all_steps.update(workspace_builders.keys())
        
        return sorted(list(all_steps))
    
    def get_discovery_stats(self) -> Dict[str, Any]:
        """
        Get discovery statistics.
        
        Returns:
            Dictionary with discovery statistics
        """
        if not self._discovery_complete:
            self._run_discovery()
        
        return {
            "package_builders": len(self._package_builders),
            "workspace_builders": {
                workspace_id: len(builders) 
                for workspace_id, builders in self._workspace_builders.items()
            },
            "total_builders": len(self._package_builders) + sum(
                len(builders) for builders in self._workspace_builders.values()
            ),
            "cached_builders": len(self._builder_cache),
            "discovery_complete": self._discovery_complete
        }
