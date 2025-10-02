"""
Configuration class auto-discovery for the unified step catalog system.

This module implements AST-based configuration class discovery from both core
and workspace directories, integrating with the existing ConfigClassStore.
Extended to include hyperparameter class discovery.
"""

import ast
import importlib
import logging
from pathlib import Path
from typing import Dict, Type, Optional, Any, List, Union

logger = logging.getLogger(__name__)


class ConfigAutoDiscovery:
    """Simple configuration class auto-discovery."""
    
    def __init__(self, package_root: Path, workspace_dirs: List[Path]):
        """
        Initialize config auto-discovery with dual search space support.
        
        Args:
            package_root: Root of the cursus package
            workspace_dirs: List of workspace directories to search
        """
        self.package_root = package_root
        self.workspace_dirs = workspace_dirs
        self.logger = logging.getLogger(__name__)
    
    def discover_config_classes(self, project_id: Optional[str] = None) -> Dict[str, Type]:
        """
        Auto-discover configuration classes from package and workspace directories.
        
        Args:
            project_id: Optional project ID for workspace-specific discovery
            
        Returns:
            Dictionary mapping class names to class types
        """
        discovered_classes = {}
        
        # Always scan package core configs
        core_config_dir = self.package_root / "steps" / "configs"
        if core_config_dir.exists():
            try:
                core_classes = self._scan_config_directory(core_config_dir)
                discovered_classes.update(core_classes)
                self.logger.info(f"Discovered {len(core_classes)} core config classes")
            except Exception as e:
                self.logger.error(f"Error scanning core config directory: {e}")
        
        # Scan workspace configs if workspace directories provided
        if self.workspace_dirs:
            for workspace_dir in self.workspace_dirs:
                try:
                    workspace_classes = self._discover_workspace_configs(workspace_dir, project_id)
                    # Workspace configs override core configs with same names
                    discovered_classes.update(workspace_classes)
                except Exception as e:
                    self.logger.error(f"Error scanning workspace config directory {workspace_dir}: {e}")
        
        return discovered_classes
    
    def discover_hyperparameter_classes(self, project_id: Optional[str] = None) -> Dict[str, Type]:
        """
        Auto-discover hyperparameter classes from package and workspace directories.
        
        Args:
            project_id: Optional project ID for workspace-specific discovery
            
        Returns:
            Dictionary mapping class names to class types
        """
        discovered_classes = {}
        
        # Always scan package core hyperparams
        core_hyperparams_dir = self.package_root / "steps" / "hyperparams"
        if core_hyperparams_dir.exists():
            try:
                core_classes = self._scan_hyperparams_directory(core_hyperparams_dir)
                discovered_classes.update(core_classes)
                self.logger.info(f"Discovered {len(core_classes)} core hyperparameter classes")
            except Exception as e:
                self.logger.error(f"Error scanning core hyperparams directory: {e}")
        
        # Also scan core/base directory for base hyperparameter classes (deployment-agnostic)
        core_base_dir = self.package_root / "core" / "base"
        if core_base_dir.exists():
            try:
                base_classes = self._scan_hyperparams_directory(core_base_dir)
                discovered_classes.update(base_classes)
                self.logger.info(f"Discovered {len(base_classes)} base hyperparameter classes from core/base")
            except Exception as e:
                self.logger.error(f"Error scanning core/base directory: {e}")
        
        # Scan workspace hyperparams if workspace directories provided
        if self.workspace_dirs:
            for workspace_dir in self.workspace_dirs:
                try:
                    workspace_classes = self._discover_workspace_hyperparams(workspace_dir, project_id)
                    # Workspace hyperparams override core hyperparams with same names
                    discovered_classes.update(workspace_classes)
                except Exception as e:
                    self.logger.error(f"Error scanning workspace hyperparams directory {workspace_dir}: {e}")
        
        return discovered_classes

    def build_complete_config_classes(self, project_id: Optional[str] = None) -> Dict[str, Type]:
        """
        Build complete mapping using pure auto-discovery.
        Includes both config and hyperparameter classes for comprehensive discovery.
        
        ConfigClassStore was removed as part of the unified step catalog refactoring.
        This method now uses pure AST-based auto-discovery which is deployment-agnostic
        and works in all environments (installed, submodule, Lambda, Docker, etc.).
        
        Args:
            project_id: Optional project ID for workspace-specific discovery
            
        Returns:
            Complete dictionary of config and hyperparameter classes (auto-discovered)
        """
        config_classes = {}
        
        # Auto-discovered config classes
        discovered_config_classes = self.discover_config_classes(project_id)
        config_classes.update(discovered_config_classes)
        config_added_count = len(discovered_config_classes)
        
        # Auto-discovered hyperparameter classes
        discovered_hyperparam_classes = self.discover_hyperparameter_classes(project_id)
        config_classes.update(discovered_hyperparam_classes)
        hyperparam_added_count = len(discovered_hyperparam_classes)
        
        self.logger.info(f"Built complete config classes: {len(config_classes)} total "
                       f"({config_added_count} config + {hyperparam_added_count} hyperparameter auto-discovered)")
        return config_classes
    
    def _scan_config_directory(self, config_dir: Path) -> Dict[str, Type]:
        """
        Scan directory for configuration classes using AST parsing.
        
        Args:
            config_dir: Directory to scan for config files
            
        Returns:
            Dictionary mapping class names to class types
        """
        config_classes = {}
        
        try:
            for py_file in config_dir.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue
                
                try:
                    # Parse file with AST to find config classes
                    with open(py_file, 'r', encoding='utf-8') as f:
                        source = f.read()
                    
                    tree = ast.parse(source, filename=str(py_file))
                    
                    # Find config classes in the AST
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef) and self._is_config_class(node):
                            try:
                                # Import the class using relative import pattern
                                relative_module_path = self._file_to_relative_module_path(py_file)
                                if relative_module_path:
                                    module = importlib.import_module(relative_module_path, package=__package__)
                                    class_type = getattr(module, node.name)
                                    config_classes[node.name] = class_type
                                    self.logger.debug(f"Found config class: {node.name} in {py_file}")
                                else:
                                    self.logger.warning(f"Could not determine relative module path for {py_file}")
                            except Exception as e:
                                self.logger.warning(f"Error importing config class {node.name} from {py_file}: {e}")
                                continue
                
                except Exception as e:
                    self.logger.warning(f"Error processing config file {py_file}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error scanning config directory {config_dir}: {e}")
        
        return config_classes
    
    def _is_config_class(self, class_node: ast.ClassDef) -> bool:
        """
        Check if a class is a config class based on inheritance and naming.
        
        Args:
            class_node: AST class definition node
            
        Returns:
            True if the class appears to be a configuration class
        """
        # Check base classes for known config base classes
        for base in class_node.bases:
            if isinstance(base, ast.Name):
                if base.id in {'BasePipelineConfig', 'ProcessingStepConfigBase', 'BaseModel'}:
                    return True
            elif isinstance(base, ast.Attribute):
                if base.attr in {'BasePipelineConfig', 'ProcessingStepConfigBase', 'BaseModel'}:
                    return True
        
        # Check naming pattern (classes ending with Config or Configuration)
        if class_node.name.endswith('Config') or class_node.name.endswith('Configuration'):
            return True
        
        return False

    def _discover_workspace_configs(self, workspace_dir: Path, project_id: Optional[str] = None) -> Dict[str, Type]:
        """Discover config classes in a workspace directory with simplified structure."""
        discovered = {}
        
        # Simplified structure: workspace_dir directly contains configs/
        config_dir = workspace_dir / "configs"
        if config_dir.exists():
            discovered.update(self._scan_config_directory(config_dir))
        
        return discovered

    def _discover_workspace_hyperparams(self, workspace_dir: Path, project_id: Optional[str] = None) -> Dict[str, Type]:
        """Discover hyperparameter classes in a workspace directory with simplified structure."""
        discovered = {}
        
        # Simplified structure: workspace_dir directly contains hyperparams/
        hyperparams_dir = workspace_dir / "hyperparams"
        if hyperparams_dir.exists():
            discovered.update(self._scan_hyperparams_directory(hyperparams_dir))
        
        return discovered

    def _file_to_relative_module_path(self, file_path: Path) -> Optional[str]:
        """
        Convert file path to relative module path for use with importlib.import_module.
        
        This creates relative import paths like "..steps.configs.config_name"
        that work with the package parameter in importlib.import_module.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Relative module path string or None if conversion fails
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
    
    def _file_to_module_path(self, file_path: Path) -> str:
        """
        Convert file path to Python module path (legacy method for compatibility).
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Module path string (e.g., 'cursus.steps.configs.config_name')
        """
        parts = file_path.parts
        
        # Find src directory to determine module root
        if 'src' in parts:
            src_idx = parts.index('src')
            module_parts = parts[src_idx + 1:]
        else:
            # Fallback: use last few parts
            module_parts = parts[-3:] if len(parts) >= 3 else parts
        
        # Remove .py extension from the last part
        if module_parts[-1].endswith('.py'):
            module_parts = module_parts[:-1] + (module_parts[-1][:-3],)
        
        return '.'.join(module_parts)
    
    def _scan_hyperparams_directory(self, hyperparams_dir: Path) -> Dict[str, Type]:
        """
        Scan directory for hyperparameter classes using AST parsing.
        
        Args:
            hyperparams_dir: Directory to scan for hyperparameter files
            
        Returns:
            Dictionary mapping class names to class types
        """
        hyperparam_classes = {}
        
        try:
            for py_file in hyperparams_dir.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue
                
                try:
                    # Parse file with AST to find hyperparameter classes
                    with open(py_file, 'r', encoding='utf-8') as f:
                        source = f.read()
                    
                    tree = ast.parse(source, filename=str(py_file))
                    
                    # Find hyperparameter classes in the AST
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef) and self._is_hyperparameter_class(node):
                            try:
                                # Import the class using relative import pattern
                                relative_module_path = self._file_to_relative_module_path(py_file)
                                if relative_module_path:
                                    module = importlib.import_module(relative_module_path, package=__package__)
                                    class_type = getattr(module, node.name)
                                    hyperparam_classes[node.name] = class_type
                                    self.logger.debug(f"Found hyperparameter class: {node.name} in {py_file}")
                                else:
                                    self.logger.warning(f"Could not determine relative module path for {py_file}")
                            except Exception as e:
                                self.logger.warning(f"Error importing hyperparameter class {node.name} from {py_file}: {e}")
                                continue
                
                except Exception as e:
                    self.logger.warning(f"Error processing hyperparameter file {py_file}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error scanning hyperparameter directory {hyperparams_dir}: {e}")
        
        return hyperparam_classes
    
    def _is_hyperparameter_class(self, class_node: ast.ClassDef) -> bool:
        """
        Check if a class is a hyperparameter class based on inheritance and naming.
        
        Args:
            class_node: AST class definition node
            
        Returns:
            True if the class appears to be a hyperparameter class
        """
        # Check base classes for known hyperparameter base classes
        for base in class_node.bases:
            if isinstance(base, ast.Name):
                if base.id in {'ModelHyperparameters', 'BaseModel'}:
                    return True
            elif isinstance(base, ast.Attribute):
                if base.attr in {'ModelHyperparameters', 'BaseModel'}:
                    return True
        
        # Check naming pattern (classes ending with Hyperparameters or containing Hyperparam)
        if (class_node.name.endswith('Hyperparameters') or 
            'Hyperparam' in class_node.name or
            class_node.name.endswith('Hyperparams')):
            return True
        
        return False
