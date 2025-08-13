"""
Specification Loader Module

Handles loading and parsing of step specifications from Python files.
Provides robust import handling and specification discovery.
"""

import os
import sys
import re
import importlib.util
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..alignment_utils import FlexibleFileResolver


class SpecificationLoader:
    """
    Loads step specifications from Python files with robust import handling.
    
    Handles:
    - Dynamic module loading with proper sys.path management
    - Multiple specification discovery methods
    - Specification object to dictionary conversion
    - Job type extraction and categorization
    """
    
    def __init__(self, specs_dir: str, contracts_dir: str):
        """
        Initialize the specification loader.
        
        Args:
            specs_dir: Directory containing specification files
            contracts_dir: Directory containing contract files (for FlexibleFileResolver)
        """
        self.specs_dir = Path(specs_dir)
        self.contracts_dir = Path(contracts_dir)
        
        # Initialize FlexibleFileResolver for robust file discovery
        base_directories = {
            'contracts': str(self.contracts_dir),
            'specs': str(self.specs_dir)
        }
        self.file_resolver = FlexibleFileResolver(base_directories)
    
    def find_specifications_by_contract(self, contract_name: str) -> Dict[Path, Dict[str, Any]]:
        """
        Find all specification files that reference the given contract.
        Uses hybrid approach: script_contract field (primary) + FlexibleFileResolver (fallback).
        
        Args:
            contract_name: Name of the contract to find specifications for
            
        Returns:
            Dictionary mapping spec file paths to spec info (job_type, spec_name)
        """
        matching_specs = {}
        
        if not self.specs_dir.exists():
            return matching_specs
        
        # PRIMARY METHOD: Look for specifications that have script_contract field pointing to our contract
        for spec_file in self.specs_dir.glob("*_spec.py"):
            if spec_file.name.startswith('__'):
                continue
                
            try:
                # Load the specification and check its script_contract field
                contract_from_spec = self._extract_script_contract_from_spec(spec_file)
                if contract_from_spec and self._contracts_match(contract_from_spec, contract_name):
                    # Extract job type and spec name
                    job_type = self._extract_job_type_from_spec_file(spec_file)
                    spec_name = self._extract_spec_name_from_file(spec_file)
                    contract_ref = self._extract_contract_reference(spec_file)
                    
                    matching_specs[spec_file] = {
                        'job_type': job_type,
                        'spec_name': spec_name,
                        'contract_ref': contract_ref,
                        'discovery_method': 'script_contract_field'
                    }
            except Exception as e:
                # Skip files that can't be processed
                continue
        
        # FALLBACK METHOD: Use FlexibleFileResolver for fuzzy name matching
        if not matching_specs:
            spec_file_path = self.file_resolver.find_specification_file(contract_name)
            
            if spec_file_path:
                spec_file = Path(spec_file_path)
                if spec_file.exists():
                    try:
                        # Extract job type and spec name
                        job_type = self._extract_job_type_from_spec_file(spec_file)
                        spec_name = self._extract_spec_name_from_file(spec_file)
                        contract_ref = self._extract_contract_reference(spec_file)
                        
                        matching_specs[spec_file] = {
                            'job_type': job_type,
                            'spec_name': spec_name,
                            'contract_ref': contract_ref,
                            'discovery_method': 'flexible_file_resolver'
                        }
                    except Exception as e:
                        # Skip files that can't be processed
                        pass
        
        # FINAL FALLBACK: Traditional import-based matching
        if not matching_specs:
            for spec_file in self.specs_dir.glob("*_spec.py"):
                if spec_file.name.startswith('__'):
                    continue
                    
                try:
                    # Check if this spec file references our contract
                    contract_ref = self._extract_contract_reference(spec_file)
                    if contract_ref and contract_ref == f"{contract_name}_contract":
                        # Extract job type and spec name
                        job_type = self._extract_job_type_from_spec_file(spec_file)
                        spec_name = self._extract_spec_name_from_file(spec_file)
                        
                        matching_specs[spec_file] = {
                            'job_type': job_type,
                            'spec_name': spec_name,
                            'contract_ref': contract_ref,
                            'discovery_method': 'import_pattern_matching'
                        }
                except Exception as e:
                    # Skip files that can't be processed
                    continue
        
        return matching_specs
    
    def load_specification(self, spec_path: Path, spec_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load specification from file using robust sys.path management.
        
        Args:
            spec_path: Path to the specification file
            spec_info: Specification info containing spec_name
            
        Returns:
            Specification dictionary
            
        Raises:
            ValueError: If specification loading fails
        """
        try:
            # Add the project root to sys.path temporarily to handle relative imports
            # Go up to the project root (where src/ is located)
            project_root = str(spec_path.parent.parent.parent.parent)  # Go up to project root
            src_root = str(spec_path.parent.parent.parent)  # Go up to src/ level
            specs_dir = str(spec_path.parent)
            
            paths_to_add = [project_root, src_root, specs_dir]
            added_paths = []
            
            for path in paths_to_add:
                if path not in sys.path:
                    sys.path.insert(0, path)
                    added_paths.append(path)
            
            try:
                # Load the module
                spec = importlib.util.spec_from_file_location(f"{spec_path.stem}", spec_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not load specification module from {spec_path}")
                
                module = importlib.util.module_from_spec(spec)
                
                # Set the module's package to handle relative imports
                module.__package__ = 'cursus.steps.specs'
                
                spec.loader.exec_module(module)
            finally:
                # Remove added paths from sys.path
                for path in added_paths:
                    if path in sys.path:
                        sys.path.remove(path)
            
            # Look for the specification object using the extracted name
            spec_var_name = spec_info['spec_name']
            
            if hasattr(module, spec_var_name):
                spec_obj = getattr(module, spec_var_name)
                # Convert StepSpecification object to dictionary
                return self._spec_to_dict(spec_obj)
            else:
                raise ValueError(f"Specification constant {spec_var_name} not found in {spec_path}")
                    
        except Exception as e:
            # If we still can't load it, provide a more detailed error
            raise ValueError(f"Failed to load specification from {spec_path}: {str(e)}")
    
    def _spec_to_dict(self, spec_obj) -> Dict[str, Any]:
        """
        Convert StepSpecification object to dictionary format.
        
        Args:
            spec_obj: Specification object
            
        Returns:
            Specification dictionary
        """
        dependencies = []
        for dep_name, dep_spec in spec_obj.dependencies.items():
            dependencies.append({
                'logical_name': dep_spec.logical_name,
                'dependency_type': dep_spec.dependency_type.value if hasattr(dep_spec.dependency_type, 'value') else str(dep_spec.dependency_type),
                'required': dep_spec.required,
                'compatible_sources': dep_spec.compatible_sources,
                'data_type': dep_spec.data_type,
                'description': dep_spec.description
            })
        
        outputs = []
        for out_name, out_spec in spec_obj.outputs.items():
            outputs.append({
                'logical_name': out_spec.logical_name,
                'output_type': out_spec.output_type.value if hasattr(out_spec.output_type, 'value') else str(out_spec.output_type),
                'property_path': out_spec.property_path,
                'data_type': out_spec.data_type,
                'description': out_spec.description
            })
        
        return {
            'step_type': spec_obj.step_type,
            'node_type': spec_obj.node_type.value if hasattr(spec_obj.node_type, 'value') else str(spec_obj.node_type),
            'dependencies': dependencies,
            'outputs': outputs
        }
    
    def _extract_script_contract_from_spec(self, spec_file: Path) -> Optional[str]:
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
    
    def _extract_contract_reference(self, spec_file: Path) -> Optional[str]:
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
    
    def _extract_spec_name_from_file(self, spec_file: Path) -> str:
        """Extract the specification constant name from a file."""
        try:
            with open(spec_file, 'r') as f:
                content = f.read()
            
            # Look for specification constant definitions
            spec_patterns = [
                r'(\w+_SPEC)\s*=\s*StepSpecification',
                r'(\w+)\s*=\s*StepSpecification'
            ]
            
            for pattern in spec_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    return matches[0]
            
            # Fallback: derive from filename
            stem = spec_file.stem
            return stem.upper().replace('_SPEC', '') + '_SPEC'
            
        except Exception:
            # Fallback: derive from filename
            stem = spec_file.stem
            return stem.upper().replace('_SPEC', '') + '_SPEC'
    
    def _extract_job_type_from_spec_file(self, spec_file: Path) -> str:
        """Extract job type from specification file name."""
        stem = spec_file.stem
        parts = stem.split('_')
        
        # Known job types to look for
        job_types = {'training', 'validation', 'testing', 'calibration'}
        
        # Pattern 1: {contract_name}_{job_type}_spec.py (job-specific)
        if len(parts) >= 3 and parts[-1] == 'spec':
            potential_job_type = parts[-2]
            if potential_job_type in job_types:
                return potential_job_type  # This is a job-specific spec
        
        # Pattern 2: {contract_name}_spec.py (generic, job-agnostic)
        # This includes cases like dummy_training_spec.py where "training" is part of the script name
        if len(parts) >= 2 and parts[-1] == 'spec':
            return 'generic'  # Generic spec that applies to all job types
        
        return 'unknown'
    
    def _contracts_match(self, contract_from_spec: str, target_contract_name: str) -> bool:
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
