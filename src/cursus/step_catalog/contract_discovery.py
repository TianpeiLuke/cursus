"""
Contract class auto-discovery for the unified step catalog system.

This module implements AST-based contract class discovery from both core
and workspace directories, following the same pattern as ConfigAutoDiscovery.
"""

import ast
import importlib
import logging
from pathlib import Path
from typing import Dict, Type, Optional, Any, List, Union

logger = logging.getLogger(__name__)


class ContractAutoDiscovery:
    """Contract class auto-discovery following ConfigAutoDiscovery pattern."""
    
    def __init__(self, package_root: Path, workspace_dirs: List[Path]):
        """
        Initialize contract auto-discovery with dual search space support.
        
        Args:
            package_root: Root of the cursus package
            workspace_dirs: List of workspace directories to search
        """
        self.package_root = package_root
        self.workspace_dirs = workspace_dirs
        self.logger = logging.getLogger(__name__)
    
    def discover_contract_classes(self, project_id: Optional[str] = None) -> Dict[str, Type]:
        """
        Auto-discover contract classes from package and workspace directories.
        
        Args:
            project_id: Optional project ID for workspace-specific discovery
            
        Returns:
            Dictionary mapping class names to class types
        """
        discovered_classes = {}
        
        # Always scan package core contracts
        core_contract_dir = self.package_root / "steps" / "contracts"
        if core_contract_dir.exists():
            try:
                core_classes = self._scan_contract_directory(core_contract_dir)
                discovered_classes.update(core_classes)
                self.logger.info(f"Discovered {len(core_classes)} core contract classes")
            except Exception as e:
                self.logger.error(f"Error scanning core contract directory: {e}")
        
        # Scan workspace contracts if workspace directories provided
        if self.workspace_dirs:
            for workspace_dir in self.workspace_dirs:
                try:
                    workspace_classes = self._discover_workspace_contracts(workspace_dir, project_id)
                    # Workspace contracts override core contracts with same names
                    discovered_classes.update(workspace_classes)
                except Exception as e:
                    self.logger.error(f"Error scanning workspace contract directory {workspace_dir}: {e}")
        
        return discovered_classes
    
    def load_contract_class(self, step_name: str) -> Optional[Any]:
        """
        Load contract class for a specific step.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Contract object or None if not found/loadable
        """
        try:
            # First try direct import using relative imports
            contract = self._try_direct_import(step_name)
            if contract:
                self.logger.debug(f"Successfully loaded contract for {step_name} via direct import")
                return contract
            
            # Then try workspace-based discovery if workspace directories provided
            if self.workspace_dirs:
                for workspace_dir in self.workspace_dirs:
                    try:
                        contract = self._try_workspace_contract_import(step_name, workspace_dir)
                        if contract:
                            self.logger.debug(f"Successfully loaded contract for {step_name} from workspace {workspace_dir}")
                            return contract
                    except Exception as e:
                        self.logger.debug(f"Workspace contract import failed for {step_name} in {workspace_dir}: {e}")
                        continue
            
            self.logger.warning(f"No contract found for step: {step_name}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading contract for {step_name}: {e}")
            return None
    
    def _scan_contract_directory(self, contract_dir: Path) -> Dict[str, Type]:
        """
        Scan directory for contract classes using AST parsing.
        
        Args:
            contract_dir: Directory to scan for contract files
            
        Returns:
            Dictionary mapping class names to class types
        """
        contract_classes = {}
        
        try:
            for py_file in contract_dir.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue
                
                try:
                    # Parse file with AST to find contract classes
                    with open(py_file, 'r', encoding='utf-8') as f:
                        source = f.read()
                    
                    tree = ast.parse(source, filename=str(py_file))
                    
                    # Find contract classes in the AST
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef) and self._is_contract_class(node):
                            try:
                                # Import the class using relative import pattern
                                relative_module_path = self._file_to_relative_module_path(py_file)
                                if relative_module_path:
                                    module = importlib.import_module(relative_module_path, package=__package__)
                                    class_type = getattr(module, node.name)
                                    contract_classes[node.name] = class_type
                                    self.logger.debug(f"Found contract class: {node.name} in {py_file}")
                                else:
                                    self.logger.warning(f"Could not determine relative module path for {py_file}")
                            except Exception as e:
                                self.logger.warning(f"Error importing contract class {node.name} from {py_file}: {e}")
                                continue
                
                except Exception as e:
                    self.logger.warning(f"Error processing contract file {py_file}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error scanning contract directory {contract_dir}: {e}")
        
        return contract_classes
    
    def _is_contract_class(self, class_node: ast.ClassDef) -> bool:
        """
        Check if a class is a contract class based on inheritance and naming.
        
        Args:
            class_node: AST class definition node
            
        Returns:
            True if the class appears to be a contract class
        """
        # Check base classes for known contract base classes
        for base in class_node.bases:
            if isinstance(base, ast.Name):
                if base.id in {'BaseContract', 'StepContract', 'ProcessingContract'}:
                    return True
            elif isinstance(base, ast.Attribute):
                if base.attr in {'BaseContract', 'StepContract', 'ProcessingContract'}:
                    return True
        
        # Check naming pattern (classes ending with Contract)
        if class_node.name.endswith('Contract'):
            return True
        
        # Check for contract-like variable names (e.g., STEP_NAME_CONTRACT)
        for node in ast.walk(class_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.endswith('_CONTRACT'):
                        return True
        
        return False
    
    def _discover_workspace_contracts(self, workspace_dir: Path, project_id: Optional[str] = None) -> Dict[str, Type]:
        """Discover contract classes in a workspace directory."""
        discovered = {}
        projects_dir = workspace_dir / "development" / "projects"
        
        if not projects_dir.exists():
            return discovered
        
        if project_id:
            # Search specific project
            project_dir = projects_dir / project_id
            if project_dir.exists():
                contract_dir = project_dir / "src" / "cursus_dev" / "steps" / "contracts"
                if contract_dir.exists():
                    discovered.update(self._scan_contract_directory(contract_dir))
        else:
            # Search all projects
            for project_dir in projects_dir.iterdir():
                if project_dir.is_dir():
                    contract_dir = project_dir / "src" / "cursus_dev" / "steps" / "contracts"
                    if contract_dir.exists():
                        discovered.update(self._scan_contract_directory(contract_dir))
        
        return discovered
    
    def _file_to_relative_module_path(self, file_path: Path) -> Optional[str]:
        """
        Convert file path to relative module path for use with importlib.import_module.
        
        This creates relative import paths like "..steps.contracts.contract_name"
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
    
    def _try_direct_import(self, step_name: str) -> Optional[Any]:
        """
        Try direct import of contract using relative imports.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Contract object or None if import fails
        """
        try:
            # Try package contracts using relative imports
            relative_module_path = f"...steps.contracts.{step_name}_contract"
            module = importlib.import_module(relative_module_path, package=__package__)
            
            # Look for contract object
            contract_name = f"{step_name.upper()}_CONTRACT"
            if hasattr(module, contract_name):
                return getattr(module, contract_name)
            
            # Fallback: look for any contract-like object
            for attr_name in dir(module):
                if attr_name.endswith('_CONTRACT') or attr_name.endswith('Contract'):
                    return getattr(module, attr_name)
            
            return None
            
        except ImportError as e:
            self.logger.debug(f"Package contract import failed for {step_name}: {e}")
            return None
        except Exception as e:
            self.logger.warning(f"Error in direct contract import for {step_name}: {e}")
            return None
    
    def _try_workspace_contract_import(self, step_name: str, workspace_dir: Path) -> Optional[Any]:
        """
        Try to import contract from workspace using file-based loading.
        
        Args:
            step_name: Name of the step
            workspace_dir: Workspace directory to search in
            
        Returns:
            Contract object or None if not found
        """
        try:
            # Look for contract files in workspace projects
            projects_dir = workspace_dir / "development" / "projects"
            if not projects_dir.exists():
                return None
            
            # Search all projects for the contract
            for project_dir in projects_dir.iterdir():
                if not project_dir.is_dir():
                    continue
                
                contract_file = project_dir / "src" / "cursus_dev" / "steps" / "contracts" / f"{step_name}_contract.py"
                if contract_file.exists():
                    try:
                        # Load contract using file-based import
                        contract = self._load_contract_from_file(contract_file, step_name)
                        if contract:
                            return contract
                    except Exception as e:
                        self.logger.warning(f"Failed to load workspace contract from {contract_file}: {e}")
                        continue
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error in workspace contract import for {step_name}: {e}")
            return None
    
    def _load_contract_from_file(self, contract_path: Path, step_name: str) -> Optional[Any]:
        """
        Load contract object from file path.
        
        Args:
            contract_path: Path to the contract file
            step_name: Name of the step
            
        Returns:
            Contract object or None if loading fails
        """
        try:
            import importlib.util
            
            # Load module from file
            spec = importlib.util.spec_from_file_location("contract_module", contract_path)
            if spec is None or spec.loader is None:
                return None
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for contract object
            contract_name = f"{step_name.upper()}_CONTRACT"
            if hasattr(module, contract_name):
                return getattr(module, contract_name)
            
            # Fallback: look for any contract-like object
            for attr_name in dir(module):
                if attr_name.endswith('_CONTRACT') or attr_name.endswith('Contract'):
                    return getattr(module, attr_name)
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to load contract from {contract_path}: {e}")
            return None
