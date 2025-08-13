"""
Contract Discovery Engine

Handles discovery and mapping of contract files, including:
- Finding contract files by script name
- Extracting contract references from specification files
- Building entry point mappings
- Matching contracts with their corresponding scripts
"""

import sys
import importlib.util
from typing import Dict, List, Optional, Any
from pathlib import Path


class ContractDiscoveryEngine:
    """
    Engine for discovering and mapping contract files.
    
    Provides robust contract file discovery using multiple strategies:
    - Entry point mapping from contract files
    - Specification file contract references
    - Naming convention patterns
    """
    
    def __init__(self, contracts_dir: str):
        """
        Initialize the contract discovery engine.
        
        Args:
            contracts_dir: Directory containing script contracts
        """
        self.contracts_dir = Path(contracts_dir)
        self._entry_point_mapping = None
    
    def discover_all_contracts(self) -> List[str]:
        """Discover all contract files in the contracts directory."""
        contracts = []
        
        if self.contracts_dir.exists():
            for contract_file in self.contracts_dir.glob("*_contract.py"):
                if not contract_file.name.startswith('__'):
                    contract_name = contract_file.stem.replace('_contract', '')
                    contracts.append(contract_name)
        
        return sorted(contracts)
    
    def discover_contracts_with_scripts(self) -> List[str]:
        """
        Discover contracts that have corresponding scripts by checking their entry_point field.
        
        This method loads each contract and checks if the script file referenced in the
        entry_point field actually exists, preventing validation errors for contracts
        without corresponding scripts.
        
        Returns:
            List of contract names that have corresponding scripts
        """
        from ..unified_alignment_tester import UnifiedAlignmentTester
        
        # Get the list of actual scripts for verification
        tester = UnifiedAlignmentTester()
        actual_scripts = set(tester.discover_scripts())
        
        contracts_with_scripts = []
        
        if not self.contracts_dir.exists():
            return contracts_with_scripts
        
        for contract_file in self.contracts_dir.glob("*_contract.py"):
            if contract_file.name.startswith('__'):
                continue
                
            contract_name = contract_file.stem.replace('_contract', '')
            
            try:
                # Load the contract to get its entry_point
                contract = self._load_contract_for_entry_point(contract_file, contract_name)
                entry_point = contract.get('entry_point', '')
                
                if entry_point:
                    # Extract script name from entry_point (remove .py extension)
                    script_name = entry_point.replace('.py', '')
                    
                    # Check if this script exists in the discovered scripts
                    if script_name in actual_scripts:
                        contracts_with_scripts.append(contract_name)
                    else:
                        # Log that we're skipping this contract
                        print(f"ℹ️  Skipping contract '{contract_name}' - script '{script_name}' not found in discovered scripts")
                else:
                    # Contract has no entry_point, skip it
                    print(f"ℹ️  Skipping contract '{contract_name}' - no entry_point defined")
                    
            except Exception as e:
                # If we can't load the contract, skip it
                print(f"⚠️  Skipping contract '{contract_name}' - failed to load: {str(e)}")
                continue
        
        return sorted(contracts_with_scripts)
    
    def extract_contract_reference_from_spec(self, spec_file: Path) -> Optional[str]:
        """Extract the contract reference from a specification file."""
        try:
            with open(spec_file, 'r') as f:
                content = f.read()
            
            # Look for import patterns that reference contracts
            import_patterns = [
                r'from \.\.contracts\.(\w+) import',
                r'from \.\.contracts\.(\w+)_contract import',
                r'import \.\.contracts\.(\w+)_contract',
            ]
            
            import re
            for pattern in import_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    contract_name = matches[0]
                    # Handle the case where pattern captures just the base name
                    if not contract_name.endswith('_contract'):
                        contract_name += '_contract'
                    return contract_name
            
            return None
            
        except Exception:
            return None
    
    def extract_script_contract_from_spec(self, spec_file: Path) -> Optional[str]:
        """Extract the script_contract field from a specification file (primary method)."""
        try:
            # Add the project root to sys.path temporarily to handle relative imports
            project_root = str(spec_file.parent.parent.parent.parent)  # Go up to project root
            src_root = str(spec_file.parent.parent.parent)  # Go up to src/ level
            specs_dir = str(spec_file.parent)
            
            paths_to_add = [project_root, src_root, specs_dir]
            added_paths = []
            
            for path in paths_to_add:
                if path not in sys.path:
                    sys.path.insert(0, path)
                    added_paths.append(path)
            
            try:
                # Load the module
                spec = importlib.util.spec_from_file_location(f"{spec_file.stem}", spec_file)
                if spec is None or spec.loader is None:
                    return None
                
                module = importlib.util.module_from_spec(spec)
                module.__package__ = 'cursus.steps.specs'
                spec.loader.exec_module(module)
                
                # Look for specification objects and extract their script_contract
                for attr_name in dir(module):
                    if attr_name.endswith('_SPEC') and not attr_name.startswith('_'):
                        spec_obj = getattr(module, attr_name)
                        if hasattr(spec_obj, 'script_contract'):
                            contract_obj = spec_obj.script_contract
                            if callable(contract_obj):
                                # It's a function that returns the contract
                                contract_obj = contract_obj()
                            if hasattr(contract_obj, 'entry_point'):
                                return contract_obj.entry_point.replace('.py', '')
                
                return None
                
            finally:
                # Remove added paths from sys.path
                for path in added_paths:
                    if path in sys.path:
                        sys.path.remove(path)
                        
        except Exception:
            return None
    
    def contracts_match(self, contract_from_spec: str, target_contract_name: str) -> bool:
        """Check if the contract from spec matches the target contract name."""
        # Direct match
        if contract_from_spec == target_contract_name:
            return True
        
        # Handle cases where spec has entry_point like "model_evaluation_xgb.py" 
        # but we're looking for "model_evaluation_xgb"
        if contract_from_spec.replace('.py', '') == target_contract_name:
            return True
        
        # Handle cases where contract name is different from script name
        # e.g., model_evaluation_xgb -> model_evaluation
        if target_contract_name.startswith(contract_from_spec):
            return True
        if contract_from_spec.startswith(target_contract_name):
            return True
        
        return False
    
    def build_entry_point_mapping(self) -> Dict[str, str]:
        """
        Build a mapping from entry_point values to contract file names.
        
        Returns:
            Dictionary mapping entry_point (script filename) to contract filename
        """
        if self._entry_point_mapping is not None:
            return self._entry_point_mapping
        
        mapping = {}
        
        if not self.contracts_dir.exists():
            self._entry_point_mapping = mapping
            return mapping
        
        # Scan all contract files
        for contract_file in self.contracts_dir.glob("*_contract.py"):
            if contract_file.name.startswith('__'):
                continue
                
            try:
                # Extract entry_point from contract
                entry_point = self._extract_entry_point_from_contract(contract_file)
                if entry_point:
                    mapping[entry_point] = contract_file.name
            except Exception:
                # Skip contracts that can't be loaded
                continue
        
        self._entry_point_mapping = mapping
        return mapping
    
    def _extract_entry_point_from_contract(self, contract_path: Path) -> Optional[str]:
        """
        Extract the entry_point value from a contract file.
        
        Args:
            contract_path: Path to the contract file
            
        Returns:
            Entry point value or None if not found
        """
        try:
            # Add the project root to sys.path temporarily
            project_root = str(contract_path.parent.parent.parent.parent)
            src_root = str(contract_path.parent.parent.parent)
            contract_dir = str(contract_path.parent)
            
            paths_to_add = [project_root, src_root, contract_dir]
            added_paths = []
            
            for path in paths_to_add:
                if path not in sys.path:
                    sys.path.insert(0, path)
                    added_paths.append(path)
            
            try:
                # Load the module
                spec = importlib.util.spec_from_file_location(
                    f"contract_{contract_path.stem}", contract_path
                )
                if spec is None or spec.loader is None:
                    return None
                
                module = importlib.util.module_from_spec(spec)
                module.__package__ = 'cursus.steps.contracts'
                spec.loader.exec_module(module)
                
                # Look for contract objects and extract entry_point
                for attr_name in dir(module):
                    if attr_name.endswith('_CONTRACT') or attr_name == 'CONTRACT':
                        contract_obj = getattr(module, attr_name)
                        if hasattr(contract_obj, 'entry_point'):
                            return contract_obj.entry_point
                
                return None
                
            finally:
                # Clean up sys.path
                for path in added_paths:
                    if path in sys.path:
                        sys.path.remove(path)
                        
        except Exception:
            return None
    
    def _load_contract_for_entry_point(self, contract_path: Path, contract_name: str) -> Dict[str, Any]:
        """Load contract from Python module to extract entry_point (lightweight version)."""
        try:
            # Add the project root to sys.path temporarily to handle relative imports
            project_root = str(contract_path.parent.parent.parent.parent)  # Go up to project root
            src_root = str(contract_path.parent.parent.parent)  # Go up to src/ level
            contract_dir = str(contract_path.parent)
            
            paths_to_add = [project_root, src_root, contract_dir]
            added_paths = []
            
            for path in paths_to_add:
                if path not in sys.path:
                    sys.path.insert(0, path)
                    added_paths.append(path)
            
            try:
                # Load the module
                spec = importlib.util.spec_from_file_location(f"{contract_name}_contract", contract_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not load contract module from {contract_path}")
                
                module = importlib.util.module_from_spec(spec)
                module.__package__ = 'cursus.steps.contracts'
                spec.loader.exec_module(module)
            finally:
                # Remove added paths from sys.path
                for path in added_paths:
                    if path in sys.path:
                        sys.path.remove(path)
            
            # Look for the contract object - try multiple naming patterns
            contract_obj = None
            
            # Try various naming patterns
            possible_names = [
                f"{contract_name.upper()}_CONTRACT",
                f"{contract_name}_CONTRACT", 
                f"{contract_name}_contract",
                "MODEL_EVALUATION_CONTRACT",  # Specific for model_evaluation_xgb
                "CONTRACT",
                "contract"
            ]
            
            # Also try to find any variable ending with _CONTRACT
            for attr_name in dir(module):
                if attr_name.endswith('_CONTRACT') and not attr_name.startswith('_'):
                    possible_names.append(attr_name)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_names = []
            for name in possible_names:
                if name not in seen:
                    seen.add(name)
                    unique_names.append(name)
            
            for name in unique_names:
                if hasattr(module, name):
                    contract_obj = getattr(module, name)
                    # Verify it's actually a contract object
                    if hasattr(contract_obj, 'entry_point'):
                        break
                    else:
                        contract_obj = None
            
            if contract_obj is None:
                raise AttributeError(f"No contract object found in {contract_path}. Tried: {unique_names}")
            
            # Convert ScriptContract object to dictionary format (lightweight)
            return {
                'entry_point': getattr(contract_obj, 'entry_point', f"{contract_name}.py")
            }
            
        except Exception as e:
            raise Exception(f"Failed to load Python contract from {contract_path}: {str(e)}")
