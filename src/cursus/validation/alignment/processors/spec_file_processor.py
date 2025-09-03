"""
Specification File Processor

Handles loading and processing of specification files, including:
- Extracting specification names from files
- Determining job types from file names
- Loading specifications from Python modules
- Converting specification objects to dictionaries
"""

import sys
import importlib.util
from typing import Dict, List, Optional, Any
from pathlib import Path


class SpecificationFileProcessor:
    """
    Processor for loading and handling specification files.
    
    Provides robust specification file processing including:
    - Job type extraction from file names
    - Specification name extraction
    - Python module loading with proper import handling
    - Object to dictionary conversion
    """
    
    def __init__(self, specs_dir: str, contracts_dir: str):
        """
        Initialize the specification file processor.
        
        Args:
            specs_dir: Directory containing step specifications
            contracts_dir: Directory containing script contracts (for relative imports)
        """
        self.specs_dir = Path(specs_dir)
        self.contracts_dir = Path(contracts_dir)
    
    def extract_spec_name_from_file(self, spec_file: Path) -> str:
        """Extract the specification constant name from a file."""
        try:
            with open(spec_file, 'r') as f:
                content = f.read()
            
            # Look for specification constant definitions
            import re
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
    
    def extract_job_type_from_spec_file(self, spec_file: Path) -> str:
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
    
    def extract_job_type_from_spec_name(self, spec_name: str) -> str:
        """Extract job type from specification name."""
        spec_name_lower = spec_name.lower()
        
        if 'training' in spec_name_lower:
            return 'training'
        elif 'testing' in spec_name_lower:
            return 'testing'
        elif 'validation' in spec_name_lower:
            return 'validation'
        elif 'calibration' in spec_name_lower:
            return 'calibration'
        else:
            return 'generic'
    
    def load_specification_from_file(self, spec_path: Path, spec_info: Dict[str, Any]) -> Dict[str, Any]:
        """Load specification from file using robust sys.path management."""
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
                return self._convert_spec_object_to_dict(spec_obj)
            else:
                raise ValueError(f"Specification constant {spec_var_name} not found in {spec_path}")
                    
        except Exception as e:
            # If we still can't load it, provide a more detailed error
            raise ValueError(f"Failed to load specification from {spec_path}: {str(e)}")
    
    def load_specification_from_python(self, spec_path: Path, contract_name: str, job_type: str) -> Dict[str, Any]:
        """Load specification from Python file with content modification for imports."""
        try:
            # Read the file content and modify imports to be absolute
            with open(spec_path, 'r') as f:
                content = f.read()
            
            # Replace common relative imports with absolute imports
            modified_content = self._modify_imports_for_loading(content)
            
            # Add the project root to sys.path
            project_root = self.specs_dir.parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            try:
                # Create a temporary module from the modified content
                module_name = f"{contract_name}_{job_type}_spec_temp"
                spec = importlib.util.spec_from_loader(module_name, loader=None)
                module = importlib.util.module_from_spec(spec)
                
                # Execute the modified content in the module's namespace
                exec(modified_content, module.__dict__)
                
                # Look for the specification constant
                spec_var_name = self._determine_spec_var_name(contract_name, job_type)
                
                if hasattr(module, spec_var_name):
                    spec_obj = getattr(module, spec_var_name)
                    return self._convert_spec_object_to_dict(spec_obj)
                else:
                    raise ValueError(f"Specification constant {spec_var_name} not found in {spec_path}")
                    
            finally:
                # Clean up sys.path
                if str(project_root) in sys.path:
                    sys.path.remove(str(project_root))
                    
        except Exception as e:
            # If we still can't load it, provide a more detailed error
            raise ValueError(f"Failed to load specification from {spec_path}: {str(e)}")
    
    def _modify_imports_for_loading(self, content: str) -> str:
        """Modify relative imports to absolute imports for loading."""
        return content.replace(
            'from ...core.base.step_specification import StepSpecification',
            'from src.cursus.core.base.step_specification import StepSpecification'
        ).replace(
            'from ...core.base.dependency_specification import DependencySpecification',
            'from src.cursus.core.base.dependency_specification import DependencySpecification'
        ).replace(
            'from ...core.base.output_specification import OutputSpecification',
            'from src.cursus.core.base.output_specification import OutputSpecification'
        ).replace(
            'from ...core.base.enums import',
            'from src.cursus.core.base.enums import'
        ).replace(
            'from ...core.base.specification_base import',
            'from src.cursus.core.base.specification_base import'
        ).replace(
            'from ..registry.step_names import',
            'from src.cursus.registry.step_names import'
        ).replace(
            'from ..contracts.',
            'from src.cursus.steps.contracts.'
        )
    
    def _determine_spec_var_name(self, contract_name: str, job_type: str) -> str:
        """Determine the specification variable name based on contract and job type."""
        if job_type == 'generic':
            # For generic specs, try without job type first
            return f"{contract_name.upper()}_SPEC"
        else:
            # For job-specific specs, include job type
            return f"{contract_name.upper()}_{job_type.upper()}_SPEC"
    
    def _convert_spec_object_to_dict(self, spec_obj) -> Dict[str, Any]:
        """Convert StepSpecification object to dictionary format."""
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
