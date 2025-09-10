"""
Specification Loader

Handles loading and parsing of step specification files from Python modules.
Provides robust loading with proper sys.path management and job type awareness.
"""

import sys
import importlib.util
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from ..alignment_utils import FlexibleFileResolver
from ....core.base.specification_base import StepSpecification, DependencySpec, OutputSpec

logger = logging.getLogger(__name__)


class SpecificationLoader:
    """
    Loads and parses step specification files from Python modules.
    
    Features:
    - Robust sys.path management for imports
    - Job type awareness (training, validation, testing, calibration)
    - Multiple fallback strategies for finding specification constants
    - Conversion between StepSpecification objects and dictionaries
    """
    
    def __init__(self, specs_dir: str):
        """
        Initialize the specification loader.
        
        Args:
            specs_dir: Directory containing specification files
        """
        self.specs_dir = Path(specs_dir)
        
        # Initialize file resolver for finding specification files
        base_directories = {'specs': str(self.specs_dir)}
        self.file_resolver = FlexibleFileResolver(base_directories)
    
    def find_specification_files(self, spec_name: str) -> List[Path]:
        """
        Find all specification files for a specification using hybrid approach.
        PRIMARY: Direct file matching, FALLBACK: FlexibleFileResolver for fuzzy name matching.
        
        Args:
            spec_name: Name of the specification to find files for
            
        Returns:
            List of specification file paths
        """
        spec_files = []
        
        # PRIMARY METHOD: Direct file matching
        direct_spec_file = self.specs_dir / f"{spec_name}_spec.py"
        if direct_spec_file.exists():
            spec_files.append(direct_spec_file)
            
            # Look for job type variants in the same directory
            for job_type in ['training', 'validation', 'testing', 'calibration']:
                variant_file = self.specs_dir / f"{spec_name}_{job_type}_spec.py"
                if variant_file.exists() and variant_file not in spec_files:
                    spec_files.append(variant_file)
        
        # FALLBACK METHOD: Use FlexibleFileResolver for fuzzy name matching
        if not spec_files:
            primary_spec = self.file_resolver.find_spec_file(spec_name)
            if primary_spec:
                spec_files.append(Path(primary_spec))
                
                # Look for job type variants in the same directory
                spec_dir = Path(primary_spec).parent
                base_name = Path(primary_spec).stem.replace('_spec', '')
                
                # Find all variants: {base_name}_{job_type}_spec.py
                for job_type in ['training', 'validation', 'testing', 'calibration']:
                    variant_file = spec_dir / f"{base_name}_{job_type}_spec.py"
                    if variant_file.exists() and variant_file not in spec_files:
                        spec_files.append(variant_file)
        
        return spec_files
    
    def extract_job_type_from_spec_file(self, spec_file: Path) -> str:
        """
        Extract job type from specification file name.
        
        Args:
            spec_file: Path to the specification file
            
        Returns:
            Job type string ('training', 'validation', 'testing', 'calibration', or 'default')
        """
        # Pattern: {spec_name}_{job_type}_spec.py or {spec_name}_spec.py
        stem = spec_file.stem
        parts = stem.split('_')
        if len(parts) >= 3 and parts[-1] == 'spec':
            return parts[-2]  # job_type is second to last part
        return 'default'
    
    def load_specification_from_python(self, spec_path: Path, spec_name: str, job_type: str) -> Dict[str, Any]:
        """
        Load specification from Python file using robust sys.path management.
        
        Args:
            spec_path: Path to the specification file
            spec_name: Name of the specification
            job_type: Job type for the specification
            
        Returns:
            Dictionary representation of the specification
            
        Raises:
            ValueError: If specification cannot be loaded or parsed
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
            
            # Use job type-aware constant name resolution
            expected_constant = self.file_resolver.find_spec_constant_name(spec_name, job_type)
            
            # Try the expected constant name first
            possible_names = []
            if expected_constant:
                possible_names.append(expected_constant)
            
            # Add fallback patterns
            possible_names.extend([
                f"{spec_name.upper()}_{job_type.upper()}_SPEC",
                f"{spec_name.upper()}_SPEC",
                f"{job_type.upper()}_SPEC"
            ])
            
            # Add dynamic discovery - scan for any constants ending with _SPEC
            spec_constants = [name for name in dir(module) 
                            if name.endswith('_SPEC') and not name.startswith('_')]
            possible_names.extend(spec_constants)
            
            spec_obj = None
            for spec_var_name in possible_names:
                if hasattr(module, spec_var_name):
                    spec_obj = getattr(module, spec_var_name)
                    break
            
            if spec_obj is None:
                raise ValueError(f"No specification constant found in {spec_path}. Tried: {possible_names}")
            
            # Convert StepSpecification object to dictionary
            return self.step_specification_to_dict(spec_obj)
                
        except Exception as e:
            # If we still can't load it, provide a more detailed error
            raise ValueError(f"Failed to load specification from {spec_path}: {str(e)}")
    
    def step_specification_to_dict(self, spec_obj: StepSpecification) -> Dict[str, Any]:
        """
        Convert StepSpecification object to dictionary representation.
        
        Args:
            spec_obj: StepSpecification object to convert
            
        Returns:
            Dictionary representation of the specification
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
    
    def dict_to_step_specification(self, spec_dict: Dict[str, Any]) -> StepSpecification:
        """
        Convert specification dictionary back to StepSpecification object.
        
        Args:
            spec_dict: Dictionary representation of the specification
            
        Returns:
            StepSpecification object
        """
        # Convert dependencies
        dependencies = {}
        for dep in spec_dict.get('dependencies', []):
            # Create DependencySpec using keyword arguments
            dep_data = {
                'logical_name': dep['logical_name'],
                'dependency_type': dep['dependency_type'],  # Keep as string, validator will convert
                'required': dep['required'],
                'compatible_sources': dep.get('compatible_sources', []),
                'data_type': dep['data_type'],
                'description': dep.get('description', ''),
                'semantic_keywords': dep.get('semantic_keywords', [])
            }
            dep_spec = DependencySpec(**dep_data)
            dependencies[dep['logical_name']] = dep_spec
        
        # Convert outputs
        outputs = {}
        for out in spec_dict.get('outputs', []):
            # Create OutputSpec using keyword arguments
            out_data = {
                'logical_name': out['logical_name'],
                'output_type': out['output_type'],  # Keep as string, validator will convert
                'property_path': out['property_path'],
                'data_type': out['data_type'],
                'description': out.get('description', ''),
                'aliases': out.get('aliases', [])
            }
            out_spec = OutputSpec(**out_data)
            outputs[out['logical_name']] = out_spec
        
        # Create StepSpecification using keyword arguments
        spec_data = {
            'step_type': spec_dict['step_type'],
            'node_type': spec_dict['node_type'],  # Keep as string, validator will convert
            'dependencies': dependencies,
            'outputs': outputs
        }
        return StepSpecification(**spec_data)
    
    def load_all_specifications(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all specification files from the specifications directory.
        
        Returns:
            Dictionary mapping specification names to their dictionary representations
        """
        all_specs = {}
        
        if self.specs_dir.exists():
            for spec_file in self.specs_dir.glob("*_spec.py"):
                spec_name = spec_file.stem.replace('_spec', '')
                # Remove job type suffix if present
                parts = spec_name.split('_')
                if len(parts) > 1:
                    # Try to identify if last part is a job type
                    potential_job_types = ['training', 'validation', 'testing', 'calibration']
                    if parts[-1] in potential_job_types:
                        spec_name = '_'.join(parts[:-1])
                
                if spec_name not in all_specs:
                    try:
                        job_type = self.extract_job_type_from_spec_file(spec_file)
                        spec = self.load_specification_from_python(spec_file, spec_name, job_type)
                        all_specs[spec_name] = spec
                    except Exception as e:
                        logger.warning(f"Failed to load specification {spec_name} from {spec_file}: {e}")
                        # Skip files that can't be parsed
                        continue
        
        return all_specs
    
    def discover_specifications(self) -> List[str]:
        """
        Discover all specification files in the specifications directory.
        
        Only includes specifications that have actual files (not derived base names).
        This prevents validation errors for non-existent specification files.
        
        Returns:
            List of specification names that have actual files
        """
        specifications = set()
        
        if self.specs_dir.exists():
            for spec_file in self.specs_dir.glob("*_spec.py"):
                if spec_file.name.startswith('__'):
                    continue
                    
                # Use the actual file name (without .py extension) as the spec name
                # This ensures we only validate specifications that actually exist
                spec_name = spec_file.stem.replace('_spec', '')
                specifications.add(spec_name)
        
        return sorted(list(specifications))
    
    def find_specifications_by_contract(self, contract_name: str) -> Dict[Path, Dict[str, Any]]:
        """
        Find specification files that reference a specific contract.
        
        Args:
            contract_name: Name of the contract to find specifications for
            
        Returns:
            Dictionary mapping specification file paths to their info
        """
        matching_specs = {}
        
        if not self.specs_dir.exists():
            return matching_specs
        
        # Search through all specification files
        for spec_file in self.specs_dir.glob("*_spec.py"):
            if spec_file.name.startswith('__'):
                continue
                
            try:
                # Load the specification file to check for contract reference
                spec_name = spec_file.stem.replace('_spec', '')
                job_type = self.extract_job_type_from_spec_file(spec_file)
                
                # Try to load and check if it references the contract
                spec_dict = self.load_specification_from_python(spec_file, spec_name, job_type)
                
                # Check if this specification references the contract
                # This is a simplified check - in practice, you might need to check
                # the actual specification content for contract references
                if self._specification_references_contract(spec_dict, contract_name):
                    matching_specs[spec_file] = {
                        'spec_name': spec_name,
                        'job_type': job_type,
                        'spec_dict': spec_dict
                    }
                    
            except Exception as e:
                logger.debug(f"Could not load specification {spec_file} to check contract reference: {e}")
                continue
        
        return matching_specs
    
    def _specification_references_contract(self, spec_dict: Dict[str, Any], contract_name: str) -> bool:
        """
        Check if a specification references a specific contract.
        
        Args:
            spec_dict: Specification dictionary
            contract_name: Name of the contract to check for
            
        Returns:
            True if the specification references the contract
        """
        # Check if the specification has a script_contract field that matches
        # Note: The spec_dict comes from converting a StepSpecification object,
        # but the script_contract field might not be included in the conversion.
        # We need to check the original specification file for this.
        
        # For now, use a naming convention approach as the primary method
        spec_step_type = spec_dict.get('step_type', '')
        
        # Return False for empty, None, or missing step_type
        if not spec_step_type:
            return False
            
        spec_step_type = spec_step_type.lower()
            
        contract_base = contract_name.lower().replace('_contract', '')
        
        # Check if the step type matches the contract name
        if contract_base in spec_step_type or spec_step_type in contract_base:
            return True
        
        # Additional check: if the step_type contains the contract base name
        # This handles cases like "CurrencyConversion_Training" matching "currency_conversion"
        step_type_words = spec_step_type.replace('_', ' ').split()
        contract_words = contract_base.replace('_', ' ').split()
        
        # Check if all contract words are present in step type words
        if all(any(contract_word in step_word for step_word in step_type_words) 
               for contract_word in contract_words):
            return True
        
        # Handle specific naming variations
        # e.g., "xgboost_model_evaluation" -> "xgboost_model_eval"
        # or "xgboost_model_eval" -> "xgboost_model_evaluation"
        contract_normalized = contract_base.replace('evaluation', 'eval').replace('eval', 'evaluation')
        if contract_normalized != contract_base:
            if contract_normalized in spec_step_type or spec_step_type in contract_normalized:
                return True
        
        # Try with "eval" <-> "evaluation" substitution
        if 'evaluation' in contract_base:
            contract_eval = contract_base.replace('evaluation', 'eval')
            if contract_eval in spec_step_type or spec_step_type in contract_eval:
                return True
        elif 'eval' in contract_base:
            contract_evaluation = contract_base.replace('eval', 'evaluation')
            if contract_evaluation in spec_step_type or spec_step_type in contract_evaluation:
                return True
        
        return False
    
    def load_specification(self, spec_file: Path, spec_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load a specification from file with provided info.
        
        Args:
            spec_file: Path to the specification file
            spec_info: Information about the specification
            
        Returns:
            Dictionary representation of the specification
        """
        # If spec_dict is already in spec_info, return it
        if 'spec_dict' in spec_info:
            return spec_info['spec_dict']
        
        # Otherwise, load it
        spec_name = spec_info.get('spec_name', spec_file.stem.replace('_spec', ''))
        job_type = spec_info.get('job_type', self.extract_job_type_from_spec_file(spec_file))
        
        return self.load_specification_from_python(spec_file, spec_name, job_type)
