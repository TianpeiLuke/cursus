"""
Configuration class auto-discovery for the unified step catalog system.

This module implements AST-based configuration class discovery from both core
and workspace directories, integrating with the existing ConfigClassStore.
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
    
    def build_complete_config_classes(self, project_id: Optional[str] = None) -> Dict[str, Type]:
        """
        Build complete mapping integrating manual registration with auto-discovery.
        
        This addresses the TODO in the existing build_complete_config_classes() function
        by providing auto-discovery capability while maintaining backward compatibility.
        
        Args:
            project_id: Optional project ID for workspace-specific discovery
            
        Returns:
            Complete dictionary of config classes (manual + auto-discovered)
        """
        try:
            from ..core.config_fields.config_class_store import ConfigClassStore
            
            # Start with manually registered classes (highest priority)
            config_classes = ConfigClassStore.get_all_classes()
            self.logger.debug(f"Found {len(config_classes)} manually registered config classes")
            
            # Add auto-discovered classes (manual registration takes precedence)
            discovered_classes = self.discover_config_classes(project_id)
            added_count = 0
            
            for class_name, class_type in discovered_classes.items():
                if class_name not in config_classes:
                    config_classes[class_name] = class_type
                    # Also register in store for consistency
                    try:
                        ConfigClassStore.register(class_type)
                        added_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to register auto-discovered class {class_name}: {e}")
            
            self.logger.info(f"Built complete config classes: {len(config_classes)} total ({added_count} auto-discovered)")
            return config_classes
            
        except ImportError as e:
            self.logger.error(f"Failed to import ConfigClassStore: {e}")
            # Fallback to just auto-discovery
            return self.discover_config_classes(project_id)
    
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
