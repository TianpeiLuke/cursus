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
from typing import Dict, Type, Optional, Any

logger = logging.getLogger(__name__)


class ConfigAutoDiscovery:
    """Simple configuration class auto-discovery."""
    
    def __init__(self, workspace_root: Path):
        """
        Initialize config auto-discovery.
        
        Args:
            workspace_root: Root directory of the workspace
        """
        self.workspace_root = workspace_root
        self.logger = logging.getLogger(__name__)
    
    def discover_config_classes(self, project_id: Optional[str] = None) -> Dict[str, Type]:
        """
        Auto-discover configuration classes from core and workspace directories.
        
        Args:
            project_id: Optional project ID for workspace-specific discovery
            
        Returns:
            Dictionary mapping class names to class types
        """
        discovered_classes = {}
        
        # Always scan core configs
        core_config_dir = self.workspace_root / "src" / "cursus" / "steps" / "configs"
        if core_config_dir.exists():
            try:
                core_classes = self._scan_config_directory(core_config_dir)
                discovered_classes.update(core_classes)
                self.logger.info(f"Discovered {len(core_classes)} core config classes")
            except Exception as e:
                self.logger.error(f"Error scanning core config directory: {e}")
        
        # Scan workspace configs if project_id provided
        if project_id:
            workspace_config_dir = (
                self.workspace_root / "development" / "projects" / project_id / 
                "src" / "cursus_dev" / "steps" / "configs"
            )
            if workspace_config_dir.exists():
                try:
                    workspace_classes = self._scan_config_directory(workspace_config_dir)
                    # Workspace configs override core configs with same names
                    discovered_classes.update(workspace_classes)
                    self.logger.info(f"Discovered {len(workspace_classes)} workspace config classes for project {project_id}")
                except Exception as e:
                    self.logger.error(f"Error scanning workspace config directory: {e}")
        
        return discovered_classes
    
    def discover_hyperparameter_classes(self, project_id: Optional[str] = None) -> Dict[str, Type]:
        """
        Auto-discover hyperparameter classes from core and workspace directories.
        Workspace-aware design supports hyperparams in multiple locations.
        
        Args:
            project_id: Optional project ID for workspace-specific discovery
            
        Returns:
            Dictionary mapping class names to class types
        """
        discovered_classes = {}
        
        # Always scan core hyperparams
        core_hyperparams_dir = self.workspace_root / "src" / "cursus" / "steps" / "hyperparams"
        if core_hyperparams_dir.exists():
            try:
                core_classes = self._scan_hyperparams_directory(core_hyperparams_dir)
                discovered_classes.update(core_classes)
                self.logger.info(f"Discovered {len(core_classes)} core hyperparameter classes")
            except Exception as e:
                self.logger.error(f"Error scanning core hyperparams directory: {e}")
        
        # Also include the base ModelHyperparameters class from core/base
        try:
            from ..core.base.hyperparameters_base import ModelHyperparameters
            discovered_classes["ModelHyperparameters"] = ModelHyperparameters
            self.logger.debug("Added ModelHyperparameters base class")
        except ImportError as e:
            self.logger.warning(f"Could not import ModelHyperparameters base class: {e}")
        
        # Workspace-aware discovery: scan multiple potential locations
        if project_id:
            # Standard workspace location
            workspace_hyperparams_dir = (
                self.workspace_root / "development" / "projects" / project_id / 
                "src" / "cursus_dev" / "steps" / "hyperparams"
            )
            if workspace_hyperparams_dir.exists():
                try:
                    workspace_classes = self._scan_hyperparams_directory(workspace_hyperparams_dir)
                    discovered_classes.update(workspace_classes)
                    self.logger.info(f"Discovered {len(workspace_classes)} workspace hyperparameter classes for project {project_id}")
                except Exception as e:
                    self.logger.error(f"Error scanning workspace hyperparams directory: {e}")
            
            # Alternative workspace locations (workspace-aware design)
            alternative_locations = [
                self.workspace_root / "development" / "projects" / project_id / "hyperparams",
                self.workspace_root / "development" / project_id / "hyperparams",
                self.workspace_root / "workspaces" / project_id / "hyperparams",
                self.workspace_root / "workspaces" / project_id / "src" / "hyperparams",
            ]
            
            for alt_dir in alternative_locations:
                if alt_dir.exists():
                    try:
                        alt_classes = self._scan_hyperparams_directory(alt_dir)
                        if alt_classes:
                            discovered_classes.update(alt_classes)
                            self.logger.info(f"Discovered {len(alt_classes)} hyperparameter classes in {alt_dir}")
                    except Exception as e:
                        self.logger.warning(f"Error scanning alternative hyperparams directory {alt_dir}: {e}")
        
        # Also scan for hyperparams in development folder root (workspace-aware)
        dev_hyperparams_dir = self.workspace_root / "development" / "hyperparams"
        if dev_hyperparams_dir.exists():
            try:
                dev_classes = self._scan_hyperparams_directory(dev_hyperparams_dir)
                if dev_classes:
                    discovered_classes.update(dev_classes)
                    self.logger.info(f"Discovered {len(dev_classes)} development hyperparameter classes")
            except Exception as e:
                self.logger.warning(f"Error scanning development hyperparams directory: {e}")
        
        return discovered_classes

    def build_complete_config_classes(self, project_id: Optional[str] = None) -> Dict[str, Type]:
        """
        Build complete mapping integrating manual registration with auto-discovery.
        Now includes both config and hyperparameter classes for comprehensive discovery.
        
        This addresses the TODO in the existing build_complete_config_classes() function
        by providing auto-discovery capability while maintaining backward compatibility.
        
        Args:
            project_id: Optional project ID for workspace-specific discovery
            
        Returns:
            Complete dictionary of config and hyperparameter classes (manual + auto-discovered)
        """
        try:
            from ..core.config_fields.config_class_store import ConfigClassStore
            
            # Start with manually registered classes (highest priority)
            config_classes = ConfigClassStore.get_all_classes()
            self.logger.debug(f"Found {len(config_classes)} manually registered config classes")
            
            # Add auto-discovered config classes (manual registration takes precedence)
            discovered_config_classes = self.discover_config_classes(project_id)
            config_added_count = 0
            
            for class_name, class_type in discovered_config_classes.items():
                if class_name not in config_classes:
                    config_classes[class_name] = class_type
                    # Also register in store for consistency
                    try:
                        ConfigClassStore.register(class_type)
                        config_added_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to register auto-discovered config class {class_name}: {e}")
            
            # Add auto-discovered hyperparameter classes
            discovered_hyperparam_classes = self.discover_hyperparameter_classes(project_id)
            hyperparam_added_count = 0
            
            for class_name, class_type in discovered_hyperparam_classes.items():
                if class_name not in config_classes:
                    config_classes[class_name] = class_type
                    hyperparam_added_count += 1
                    self.logger.debug(f"Added hyperparameter class: {class_name}")
            
            total_added = config_added_count + hyperparam_added_count
            self.logger.info(f"Built complete config classes: {len(config_classes)} total "
                           f"({config_added_count} config + {hyperparam_added_count} hyperparameter auto-discovered)")
            return config_classes
            
        except ImportError as e:
            self.logger.error(f"Failed to import ConfigClassStore: {e}")
            # Fallback to just auto-discovery (both config and hyperparameter)
            config_classes = self.discover_config_classes(project_id)
            hyperparam_classes = self.discover_hyperparameter_classes(project_id)
            config_classes.update(hyperparam_classes)
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
                                # Import the class
                                module_path = self._file_to_module_path(py_file)
                                module = importlib.import_module(module_path)
                                class_type = getattr(module, node.name)
                                config_classes[node.name] = class_type
                                self.logger.debug(f"Found config class: {node.name} in {py_file}")
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
    
    def _file_to_module_path(self, file_path: Path) -> str:
        """
        Convert file path to Python module path.
        
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
                                # Import the class
                                module_path = self._file_to_module_path(py_file)
                                module = importlib.import_module(module_path)
                                class_type = getattr(module, node.name)
                                hyperparam_classes[node.name] = class_type
                                self.logger.debug(f"Found hyperparameter class: {node.name} in {py_file}")
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
