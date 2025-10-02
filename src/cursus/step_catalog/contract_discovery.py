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
            # Strategy 1: Try direct import using the step name as-is
            contract = self._try_direct_import(step_name)
            if contract:
                self.logger.debug(f"Successfully loaded contract for {step_name} via direct import")
                return contract
            
            # Strategy 2: Try with PascalCase to snake_case conversion
            snake_case_name = self._pascal_to_snake_case(step_name)
            if snake_case_name != step_name:
                contract = self._try_direct_import(snake_case_name)
                if contract:
                    self.logger.debug(f"Successfully loaded contract for {step_name} via snake_case conversion ({snake_case_name})")
                    return contract
            
            # Strategy 3: Try workspace-based discovery if workspace directories provided
            if self.workspace_dirs:
                for workspace_dir in self.workspace_dirs:
                    try:
                        # Try original name
                        contract = self._try_workspace_contract_import(step_name, workspace_dir)
                        if contract:
                            self.logger.debug(f"Successfully loaded contract for {step_name} from workspace {workspace_dir}")
                            return contract
                        
                        # Try snake_case name
                        if snake_case_name != step_name:
                            contract = self._try_workspace_contract_import(snake_case_name, workspace_dir)
                            if contract:
                                self.logger.debug(f"Successfully loaded contract for {step_name} from workspace {workspace_dir} via snake_case")
                                return contract
                    except Exception as e:
                        self.logger.debug(f"Workspace contract import failed for {step_name} in {workspace_dir}: {e}")
                        continue
            
            self.logger.warning(f"No contract found for step: {step_name}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading contract for {step_name}: {e}")
            return None
    
    def _pascal_to_snake_case(self, name: str) -> str:
        """
        Convert PascalCase to snake_case.
        
        Args:
            name: PascalCase string
            
        Returns:
            snake_case string
        """
        import re
        # Insert underscore before uppercase letters that follow lowercase letters or digits
        snake_case = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
        return snake_case.lower()
    
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
        
        # Assume workspace_dir already points to the equivalent of cursus/steps structure
        # e.g., workspace_dir = /path/to/workspace/development/projects/src/cursus_dev/steps
        contract_dir = workspace_dir / "contracts"
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
        Try direct import of contract using relative imports with automatic discovery.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Contract object or None if import fails
        """
        try:
            # Try package contracts using relative imports
            # Use ..steps.contracts instead of ...steps.contracts to avoid "beyond top-level package" error
            relative_module_path = f"..steps.contracts.{step_name}_contract"
            module = importlib.import_module(relative_module_path, package=__package__)
            
            # Strategy 1: Automatically discover all contract objects in the module
            contract_objects = self._discover_contract_objects_in_module(module)
            
            if contract_objects:
                # Return the first contract object found
                contract_name, contract_obj = contract_objects[0]
                self.logger.debug(f"Auto-discovered contract: {contract_name}")
                return contract_obj
            
            return None
            
        except ImportError as e:
            self.logger.debug(f"Package contract import failed for {step_name}: {e}")
            return None
        except Exception as e:
            self.logger.warning(f"Error in direct contract import for {step_name}: {e}")
            return None
    
    def _discover_contract_objects_in_module(self, module) -> List[tuple]:
        """
        Automatically discover all contract objects in a module.
        
        Args:
            module: Imported module to scan
            
        Returns:
            List of (name, object) tuples for contract objects found
        """
        contract_objects = []
        
        try:
            # Get all attributes in the module
            for attr_name in dir(module):
                # Skip private attributes and imports
                if attr_name.startswith('_'):
                    continue
                
                attr_value = getattr(module, attr_name)
                
                # Skip imported classes and functions
                if isinstance(attr_value, (type, type(lambda: None))):
                    continue
                
                # Check if it's a contract object
                if self._is_contract_object(attr_name, attr_value):
                    contract_objects.append((attr_name, attr_value))
                    self.logger.debug(f"Found contract object: {attr_name} of type {type(attr_value)}")
        
        except Exception as e:
            self.logger.warning(f"Error discovering contract objects in module: {e}")
        
        return contract_objects
    
    def _is_contract_object(self, name: str, obj: Any) -> bool:
        """
        Check if an object is a contract object.
        
        Args:
            name: Name of the object
            obj: Object to check
            
        Returns:
            True if the object appears to be a contract
        """
        # Check naming patterns
        if name.endswith('_CONTRACT') or name.endswith('Contract'):
            return True
        
        # Check if object has contract-like attributes
        if hasattr(obj, 'expected_input_paths') and hasattr(obj, 'expected_output_paths'):
            return True
        
        # Check if object has entry_point attribute (common in contracts)
        if hasattr(obj, 'entry_point'):
            return True
        
        # Check object type name
        obj_type_name = type(obj).__name__
        if 'Contract' in obj_type_name:
            return True
        
        return False
    
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
            # Assume workspace_dir already points to the equivalent of cursus/steps structure
            # e.g., workspace_dir = /path/to/workspace/development/projects/src/cursus_dev/steps
            contract_file = workspace_dir / "contracts" / f"{step_name}_contract.py"
            if contract_file.exists():
                try:
                    # Load contract using file-based import
                    contract = self._load_contract_from_file(contract_file, step_name)
                    if contract:
                        return contract
                except Exception as e:
                    self.logger.warning(f"Failed to load workspace contract from {contract_file}: {e}")
            
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
    
    def serialize_contract(self, contract_instance: Any) -> Dict[str, Any]:
        """
        Convert contract instance to dictionary format.
        
        This method provides standardized serialization of ScriptContract objects
        for use in script-contract alignment validation, following the same pattern
        as SpecAutoDiscovery.serialize_spec().
        
        Args:
            contract_instance: Contract instance to serialize
            
        Returns:
            Dictionary representation of the contract
        """
        try:
            if not self._is_contract_instance(contract_instance):
                raise ValueError("Object is not a valid contract instance")
            
            # Serialize contract fields using helper methods
            return {
                "entry_point": getattr(contract_instance, "entry_point", ""),
                "inputs": self._serialize_contract_inputs(contract_instance),
                "outputs": self._serialize_contract_outputs(contract_instance),
                "arguments": self._serialize_contract_arguments(contract_instance),
                "environment_variables": {
                    "required": getattr(contract_instance, "required_env_vars", []),
                    "optional": getattr(contract_instance, "optional_env_vars", {}),
                },
                "description": getattr(contract_instance, "description", ""),
                "framework_requirements": getattr(contract_instance, "framework_requirements", {}),
            }
            
        except Exception as e:
            self.logger.error(f"Error serializing contract: {e}")
            return {}
    
    def find_contracts_by_entry_point(self, entry_point: str) -> Dict[str, Any]:
        """
        Find contracts that reference a specific script entry point.
        
        This method enables script-contract alignment validation by finding
        contracts that are associated with a given script entry point.
        
        Args:
            entry_point: Script entry point (e.g., "model_evaluation_xgb.py")
            
        Returns:
            Dictionary mapping contract names to contract instances
        """
        try:
            matching_contracts = {}
            
            # Search core package contracts
            core_contract_dir = self.package_root / "steps" / "contracts"
            if core_contract_dir.exists():
                core_matches = self._find_contracts_by_entry_point_in_dir(core_contract_dir, entry_point)
                matching_contracts.update(core_matches)
            
            # Search workspace contracts
            if self.workspace_dirs:
                for workspace_dir in self.workspace_dirs:
                    workspace_matches = self._find_contracts_by_entry_point_in_workspace(workspace_dir, entry_point)
                    matching_contracts.update(workspace_matches)
            
            self.logger.debug(f"Found {len(matching_contracts)} contracts for entry point '{entry_point}'")
            return matching_contracts
            
        except Exception as e:
            self.logger.error(f"Error finding contracts for entry point {entry_point}: {e}")
            return {}
    
    def get_contract_entry_points(self) -> Dict[str, str]:
        """
        Get all contract entry points for validation.
        
        Returns:
            Dictionary mapping contract names to their entry points
        """
        try:
            entry_points = {}
            
            # Scan core contracts
            core_contract_dir = self.package_root / "steps" / "contracts"
            if core_contract_dir.exists():
                core_entry_points = self._extract_entry_points_from_dir(core_contract_dir)
                entry_points.update(core_entry_points)
            
            # Scan workspace contracts
            if self.workspace_dirs:
                for workspace_dir in self.workspace_dirs:
                    workspace_entry_points = self._extract_entry_points_from_workspace(workspace_dir)
                    entry_points.update(workspace_entry_points)
            
            self.logger.debug(f"Found {len(entry_points)} contract entry points")
            return entry_points
            
        except Exception as e:
            self.logger.error(f"Error getting contract entry points: {e}")
            return {}
    
    def _is_contract_instance(self, obj: Any) -> bool:
        """Check if an object is a valid contract instance."""
        try:
            # Check if it has the expected attributes of a contract
            return (hasattr(obj, 'entry_point') or 
                   hasattr(obj, 'expected_input_paths') or 
                   hasattr(obj, 'expected_output_paths'))
        except Exception:
            return False
    
    def _serialize_contract_inputs(self, contract_instance: Any) -> Dict[str, Any]:
        """Serialize contract input specifications."""
        inputs = {}
        try:
            if hasattr(contract_instance, 'expected_input_paths'):
                for logical_name, path in contract_instance.expected_input_paths.items():
                    inputs[logical_name] = {"path": path}
        except Exception as e:
            self.logger.warning(f"Error serializing contract inputs: {e}")
        return inputs
    
    def _serialize_contract_outputs(self, contract_instance: Any) -> Dict[str, Any]:
        """Serialize contract output specifications."""
        outputs = {}
        try:
            if hasattr(contract_instance, 'expected_output_paths'):
                for logical_name, path in contract_instance.expected_output_paths.items():
                    outputs[logical_name] = {"path": path}
        except Exception as e:
            self.logger.warning(f"Error serializing contract outputs: {e}")
        return outputs
    
    def _serialize_contract_arguments(self, contract_instance: Any) -> Dict[str, Any]:
        """Serialize contract argument specifications."""
        arguments = {}
        try:
            if hasattr(contract_instance, 'expected_arguments'):
                for arg_name, default_value in contract_instance.expected_arguments.items():
                    arguments[arg_name] = {
                        "default": default_value,
                        "required": default_value is None,
                    }
        except Exception as e:
            self.logger.warning(f"Error serializing contract arguments: {e}")
        return arguments
    
    def _find_contracts_by_entry_point_in_dir(self, contract_dir: Path, entry_point: str) -> Dict[str, Any]:
        """Find contracts that reference an entry point in a specific directory."""
        matching_contracts = {}
        
        try:
            for py_file in contract_dir.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue
                
                try:
                    if self._contract_file_references_entry_point(py_file, entry_point):
                        # Load the contract from this file
                        step_name = py_file.stem.replace("_contract", "")
                        contract_instance = self._load_contract_from_file(py_file, step_name)
                        if contract_instance:
                            contract_key = py_file.stem
                            matching_contracts[contract_key] = contract_instance
                            self.logger.debug(f"Found matching contract: {contract_key} for entry point {entry_point}")
                
                except Exception as e:
                    self.logger.warning(f"Error checking contract file {py_file} for entry point {entry_point}: {e}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Error scanning directory {contract_dir} for entry point {entry_point}: {e}")
        
        return matching_contracts
    
    def _find_contracts_by_entry_point_in_workspace(self, workspace_dir: Path, entry_point: str) -> Dict[str, Any]:
        """Find contracts that reference an entry point in workspace directories."""
        matching_contracts = {}
        
        try:
            # Assume workspace_dir already points to the equivalent of cursus/steps structure
            # e.g., workspace_dir = /path/to/workspace/development/projects/src/cursus_dev/steps
            contract_dir = workspace_dir / "contracts"
            if contract_dir.exists():
                workspace_matches = self._find_contracts_by_entry_point_in_dir(contract_dir, entry_point)
                matching_contracts.update(workspace_matches)
        
        except Exception as e:
            self.logger.error(f"Error scanning workspace {workspace_dir} for entry point {entry_point}: {e}")
        
        return matching_contracts
    
    def _contract_file_references_entry_point(self, contract_file: Path, entry_point: str) -> bool:
        """Check if a contract file references a specific entry point."""
        try:
            with open(contract_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Simple string search for entry point references
                if entry_point in content:
                    return True
                
                # Also check for entry point without extension
                entry_point_base = entry_point.replace('.py', '')
                if entry_point_base in content:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Error checking if {contract_file} references entry point {entry_point}: {e}")
            return False
    
    def _extract_entry_points_from_dir(self, contract_dir: Path) -> Dict[str, str]:
        """Extract entry points from contracts in a specific directory."""
        entry_points = {}
        
        try:
            for py_file in contract_dir.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue
                
                try:
                    # Load the contract from this file
                    step_name = py_file.stem.replace("_contract", "")
                    contract_instance = self._load_contract_from_file(py_file, step_name)
                    if contract_instance and hasattr(contract_instance, 'entry_point'):
                        contract_key = py_file.stem
                        entry_points[contract_key] = contract_instance.entry_point
                        self.logger.debug(f"Found entry point: {contract_instance.entry_point} for contract {contract_key}")
                
                except Exception as e:
                    self.logger.warning(f"Error extracting entry point from contract file {py_file}: {e}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Error extracting entry points from directory {contract_dir}: {e}")
        
        return entry_points
    
    def _extract_entry_points_from_workspace(self, workspace_dir: Path) -> Dict[str, str]:
        """Extract entry points from contracts in workspace directories."""
        entry_points = {}
        
        try:
            # Assume workspace_dir already points to the equivalent of cursus/steps structure
            # e.g., workspace_dir = /path/to/workspace/development/projects/src/cursus_dev/steps
            contract_dir = workspace_dir / "contracts"
            if contract_dir.exists():
                workspace_entry_points = self._extract_entry_points_from_dir(contract_dir)
                entry_points.update(workspace_entry_points)
        
        except Exception as e:
            self.logger.error(f"Error extracting entry points from workspace {workspace_dir}: {e}")
        
        return entry_points
