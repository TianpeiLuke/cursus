"""
Contract ↔ Specification Alignment Tester

Validates alignment between script contracts and step specifications.
Ensures logical names, data types, and dependencies are consistent.
"""

import os
import sys
import importlib.util
from typing import Dict, List, Any, Optional, Set
from pathlib import Path

from .alignment_utils import FlexibleFileResolver


class ContractSpecificationAlignmentTester:
    """
    Tests alignment between script contracts and step specifications.
    
    Validates:
    - Logical names match between contract and specification
    - Data types are consistent
    - Input/output specifications align
    - Dependencies are properly declared
    """
    
    def __init__(self, contracts_dir: str, specs_dir: str):
        """
        Initialize the contract-specification alignment tester.
        
        Args:
            contracts_dir: Directory containing script contracts
            specs_dir: Directory containing step specifications
        """
        self.contracts_dir = Path(contracts_dir)
        self.specs_dir = Path(specs_dir)
        
        # Initialize FlexibleFileResolver for robust file discovery
        base_directories = {
            'contracts': str(self.contracts_dir),
            'specs': str(self.specs_dir)
        }
        self.file_resolver = FlexibleFileResolver(base_directories)
        
        # Add the project root to Python path for imports
        project_root = self.contracts_dir.parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
    
    def validate_all_contracts(self, target_scripts: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Validate alignment for all contracts or specified target scripts.
        
        Args:
            target_scripts: Specific scripts to validate (None for all)
            
        Returns:
            Dictionary mapping contract names to validation results
        """
        results = {}
        
        # Discover contracts to validate
        if target_scripts:
            contracts_to_validate = target_scripts
        else:
            # Only validate contracts that have corresponding scripts
            contracts_to_validate = self._discover_contracts_with_scripts()
        
        for contract_name in contracts_to_validate:
            try:
                result = self.validate_contract(contract_name)
                results[contract_name] = result
            except Exception as e:
                results[contract_name] = {
                    'passed': False,
                    'error': str(e),
                    'issues': [{
                        'severity': 'CRITICAL',
                        'category': 'validation_error',
                        'message': f'Failed to validate contract {contract_name}: {str(e)}'
                    }]
                }
        
        return results
    
    def validate_contract(self, contract_name: str) -> Dict[str, Any]:
        """
        Validate alignment for a specific contract using Smart Specification Selection.
        
        Args:
            contract_name: Name of the contract to validate
            
        Returns:
            Validation result dictionary
        """
        # Use FlexibleFileResolver to find the correct contract file
        contract_file_path = self.file_resolver.find_contract_file(contract_name)
        
        # Check if contract file exists
        if not contract_file_path:
            return {
                'passed': False,
                'issues': [{
                    'severity': 'CRITICAL',
                    'category': 'missing_file',
                    'message': f'Contract file not found for script: {contract_name}',
                    'details': {
                        'script': contract_name,
                        'searched_patterns': [
                            f'{contract_name}_contract.py',
                            'Known naming patterns from FlexibleFileResolver'
                        ]
                    },
                    'recommendation': f'Create contract file for {contract_name} or check naming patterns'
                }]
            }
        
        contract_path = Path(contract_file_path)
        if not contract_path.exists():
            return {
                'passed': False,
                'issues': [{
                    'severity': 'CRITICAL',
                    'category': 'missing_file',
                    'message': f'Contract file not found: {contract_path}',
                    'recommendation': f'Create the contract file {contract_path.name}'
                }]
            }
        
        # Load contract from Python file
        try:
            contract = self._load_contract_from_python(contract_path, contract_name)
        except Exception as e:
            return {
                'passed': False,
                'issues': [{
                    'severity': 'CRITICAL',
                    'category': 'contract_load_error',
                    'message': f'Failed to load contract: {str(e)}',
                    'recommendation': 'Fix Python syntax or contract structure in contract file'
                }]
            }
        
        # Find specification files using script_contract field
        spec_files = self._find_specifications_by_contract(contract_name)
        
        if not spec_files:
            return {
                'passed': False,
                'issues': [{
                    'severity': 'ERROR',
                    'category': 'missing_specification',
                    'message': f'No specification files found for {contract_name}',
                    'recommendation': f'Create specification files that reference {contract_name}_contract'
                }]
            }
        
        # Load specifications from Python files
        specifications = {}
        for spec_file, spec_info in spec_files.items():
            try:
                spec = self._load_specification_from_file(spec_file, spec_info)
                # Use the spec file name as the key since job type comes from config, not spec
                spec_key = spec_file.stem
                specifications[spec_key] = spec
                    
            except Exception as e:
                return {
                    'passed': False,
                    'issues': [{
                        'severity': 'CRITICAL',
                        'category': 'spec_load_error',
                        'message': f'Failed to load specification from {spec_file}: {str(e)}',
                        'recommendation': 'Fix Python syntax or specification structure'
                    }]
                }
        
        # SMART SPECIFICATION SELECTION: Create unified specification model
        unified_spec = self._create_unified_specification(specifications, contract_name)
        
        # Perform alignment validation against unified specification
        all_issues = []
        
        # Validate logical name alignment using smart multi-variant logic
        logical_issues = self._validate_logical_names_smart(contract, unified_spec, contract_name)
        all_issues.extend(logical_issues)
        
        # Validate data type consistency
        type_issues = self._validate_data_types(contract, unified_spec['primary_spec'], contract_name)
        all_issues.extend(type_issues)
        
        # Validate input/output alignment
        io_issues = self._validate_input_output_alignment(contract, unified_spec['primary_spec'], contract_name)
        all_issues.extend(io_issues)
        
        # Determine overall pass/fail status
        has_critical_or_error = any(
            issue['severity'] in ['CRITICAL', 'ERROR'] for issue in all_issues
        )
        
        return {
            'passed': not has_critical_or_error,
            'issues': all_issues,
            'contract': contract,
            'specifications': specifications,
            'unified_specification': unified_spec
        }
    
    def _load_contract_from_python(self, contract_path: Path, contract_name: str) -> Dict[str, Any]:
        """Load contract from Python file using robust sys.path management (same approach as Level 1)."""
        try:
            # Add the project root to sys.path temporarily to handle relative imports
            # Go up to the project root (where src/ is located)
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
                
                # Set the module's package to handle relative imports
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
                "XGBOOST_MODEL_EVAL_CONTRACT",  # Specific for model_evaluation_xgb
                "MODEL_EVALUATION_CONTRACT",  # Legacy fallback
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
            
            # Convert ScriptContract object to dictionary format
            contract_dict = {
                'entry_point': getattr(contract_obj, 'entry_point', f"{contract_name}.py"),
                'inputs': {},
                'outputs': {},
                'arguments': {},
                'environment_variables': {
                    'required': getattr(contract_obj, 'required_env_vars', []),
                    'optional': getattr(contract_obj, 'optional_env_vars', {})
                },
                'description': getattr(contract_obj, 'description', ''),
                'framework_requirements': getattr(contract_obj, 'framework_requirements', {})
            }
            
            # Convert expected_input_paths to inputs format
            if hasattr(contract_obj, 'expected_input_paths'):
                for logical_name, path in contract_obj.expected_input_paths.items():
                    contract_dict['inputs'][logical_name] = {'path': path}
            
            # Convert expected_output_paths to outputs format
            if hasattr(contract_obj, 'expected_output_paths'):
                for logical_name, path in contract_obj.expected_output_paths.items():
                    contract_dict['outputs'][logical_name] = {'path': path}
            
            # Convert expected_arguments to arguments format
            if hasattr(contract_obj, 'expected_arguments'):
                for arg_name, default_value in contract_obj.expected_arguments.items():
                    contract_dict['arguments'][arg_name] = {
                        'default': default_value,
                        'required': default_value is None
                    }
            
            return contract_dict
            
        except Exception as e:
            raise Exception(f"Failed to load Python contract from {contract_path}: {str(e)}")
    
    def _find_specifications_by_contract(self, contract_name: str) -> Dict[Path, Dict[str, Any]]:
        """
        Find all specification files that reference the given contract.
        Uses hybrid approach: script_contract field (primary) + FlexibleFileResolver (fallback).
        
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
    
    def _extract_spec_name_from_file(self, spec_file: Path) -> str:
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
    
    def _load_specification_from_file(self, spec_path: Path, spec_info: Dict[str, Any]) -> Dict[str, Any]:
        """Load specification from file using robust sys.path management (same approach as Level 1)."""
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
            else:
                raise ValueError(f"Specification constant {spec_var_name} not found in {spec_path}")
                    
        except Exception as e:
            # If we still can't load it, provide a more detailed error
            raise ValueError(f"Failed to load specification from {spec_path}: {str(e)}")

    def _load_specification_from_python(self, spec_path: Path, contract_name: str, job_type: str) -> Dict[str, Any]:
        """Load specification from Python file."""
        try:
            # Read the file content and modify imports to be absolute
            with open(spec_path, 'r') as f:
                content = f.read()
            
            # Replace common relative imports with absolute imports
            modified_content = content.replace(
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
                'from src.cursus.steps.registry.step_names import'
            ).replace(
                'from ..contracts.',
                'from src.cursus.steps.contracts.'
            )
            
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
                if job_type == 'generic':
                    # For generic specs, try without job type first
                    spec_var_name = f"{contract_name.upper()}_SPEC"
                else:
                    # For job-specific specs, include job type
                    spec_var_name = f"{contract_name.upper()}_{job_type.upper()}_SPEC"
                
                if hasattr(module, spec_var_name):
                    spec_obj = getattr(module, spec_var_name)
                    # Convert StepSpecification object to dictionary
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
                else:
                    raise ValueError(f"Specification constant {spec_var_name} not found in {spec_path}")
                    
            finally:
                # Clean up sys.path
                if str(project_root) in sys.path:
                    sys.path.remove(str(project_root))
                    
        except Exception as e:
            # If we still can't load it, provide a more detailed error
            raise ValueError(f"Failed to load specification from {spec_path}: {str(e)}")

    def _validate_logical_names(self, contract: Dict[str, Any], specification: Dict[str, Any], contract_name: str, job_type: str = None) -> List[Dict[str, Any]]:
        """Validate that logical names match between contract and specification."""
        issues = []
        
        # Get logical names from contract
        contract_inputs = set(contract.get('inputs', {}).keys())
        contract_outputs = set(contract.get('outputs', {}).keys())
        
        # Get logical names from specification
        spec_dependencies = set()
        for dep in specification.get('dependencies', []):
            if 'logical_name' in dep:
                spec_dependencies.add(dep['logical_name'])
        
        spec_outputs = set()
        for output in specification.get('outputs', []):
            if 'logical_name' in output:
                spec_outputs.add(output['logical_name'])
        
        # Check for contract inputs not in spec dependencies
        missing_deps = contract_inputs - spec_dependencies
        for logical_name in missing_deps:
            issues.append({
                'severity': 'ERROR',
                'category': 'logical_names',
                'message': f'Contract input {logical_name} not declared as specification dependency',
                'details': {'logical_name': logical_name, 'contract': contract_name},
                'recommendation': f'Add {logical_name} to specification dependencies'
            })
        
        # Check for contract outputs not in spec outputs
        missing_outputs = contract_outputs - spec_outputs
        for logical_name in missing_outputs:
            issues.append({
                'severity': 'ERROR',
                'category': 'logical_names',
                'message': f'Contract output {logical_name} not declared as specification output',
                'details': {'logical_name': logical_name, 'contract': contract_name},
                'recommendation': f'Add {logical_name} to specification outputs'
            })
        
        return issues
    
    def _validate_data_types(self, contract: Dict[str, Any], specification: Dict[str, Any], contract_name: str) -> List[Dict[str, Any]]:
        """Validate data type consistency between contract and specification."""
        issues = []
        
        # Note: Contract inputs/outputs are typically stored as simple path strings,
        # while specifications have rich data type information.
        # For now, we'll skip detailed data type validation since the contract
        # format doesn't include explicit data type declarations.
        
        # This could be enhanced in the future if contracts are extended
        # to include data type information.
        
        return issues
    
    def _validate_input_output_alignment(self, contract: Dict[str, Any], specification: Dict[str, Any], contract_name: str) -> List[Dict[str, Any]]:
        """Validate input/output alignment between contract and specification."""
        issues = []
        
        # Check for specification dependencies without corresponding contract inputs
        spec_deps = {dep.get('logical_name') for dep in specification.get('dependencies', [])}
        contract_inputs = set(contract.get('inputs', {}).keys())
        
        unmatched_deps = spec_deps - contract_inputs
        for logical_name in unmatched_deps:
            if logical_name:  # Skip None values
                issues.append({
                    'severity': 'WARNING',
                    'category': 'input_output_alignment',
                    'message': f'Specification dependency {logical_name} has no corresponding contract input',
                    'details': {'logical_name': logical_name, 'contract': contract_name},
                    'recommendation': f'Add {logical_name} to contract inputs or remove from specification dependencies'
                })
        
        # Check for specification outputs without corresponding contract outputs
        spec_outputs = {out.get('logical_name') for out in specification.get('outputs', [])}
        contract_outputs = set(contract.get('outputs', {}).keys())
        
        unmatched_outputs = spec_outputs - contract_outputs
        for logical_name in unmatched_outputs:
            if logical_name:  # Skip None values
                issues.append({
                    'severity': 'WARNING',
                    'category': 'input_output_alignment',
                    'message': f'Specification output {logical_name} has no corresponding contract output',
                    'details': {'logical_name': logical_name, 'contract': contract_name},
                    'recommendation': f'Add {logical_name} to contract outputs or remove from specification outputs'
                })
        
        return issues
    
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

    def _create_unified_specification(self, specifications: Dict[str, Dict[str, Any]], contract_name: str) -> Dict[str, Any]:
        """
        Create a unified specification model from multiple specification variants.
        
        This implements Smart Specification Selection by:
        1. Detecting specification variants (training, testing, validation, calibration)
        2. Creating a union of all dependencies and outputs
        3. Providing metadata about which variants contribute what
        
        Args:
            specifications: Dictionary of loaded specifications
            contract_name: Name of the contract being validated
            
        Returns:
            Unified specification model with metadata
        """
        if not specifications:
            return {'primary_spec': {}, 'variants': {}, 'unified_dependencies': set(), 'unified_outputs': set()}
        
        # Group specifications by job type
        variants = {
            'training': None,
            'testing': None,
            'validation': None,
            'calibration': None,
            'generic': None
        }
        
        # Categorize specifications by job type
        for spec_key, spec_data in specifications.items():
            job_type = self._extract_job_type_from_spec_name(spec_key)
            if job_type in variants:
                variants[job_type] = spec_data
            else:
                # If we can't categorize it, treat it as generic
                variants['generic'] = spec_data
        
        # Remove None entries
        variants = {k: v for k, v in variants.items() if v is not None}
        
        # Create unified dependency and output sets
        unified_dependencies = {}
        unified_outputs = {}
        dependency_sources = {}  # Track which variants contribute each dependency
        output_sources = {}      # Track which variants contribute each output
        
        # Union all dependencies from all variants
        for variant_name, spec_data in variants.items():
            for dep in spec_data.get('dependencies', []):
                logical_name = dep.get('logical_name')
                if logical_name:
                    unified_dependencies[logical_name] = dep
                    if logical_name not in dependency_sources:
                        dependency_sources[logical_name] = []
                    dependency_sources[logical_name].append(variant_name)
            
            for output in spec_data.get('outputs', []):
                logical_name = output.get('logical_name')
                if logical_name:
                    unified_outputs[logical_name] = output
                    if logical_name not in output_sources:
                        output_sources[logical_name] = []
                    output_sources[logical_name].append(variant_name)
        
        # Select primary specification (prefer training, then generic, then first available)
        primary_spec = None
        if 'training' in variants:
            primary_spec = variants['training']
        elif 'generic' in variants:
            primary_spec = variants['generic']
        else:
            primary_spec = next(iter(variants.values()))
        
        return {
            'primary_spec': primary_spec,
            'variants': variants,
            'unified_dependencies': unified_dependencies,
            'unified_outputs': unified_outputs,
            'dependency_sources': dependency_sources,
            'output_sources': output_sources,
            'variant_count': len(variants)
        }
    
    def _extract_job_type_from_spec_name(self, spec_name: str) -> str:
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
    
    def _validate_logical_names_smart(self, contract: Dict[str, Any], unified_spec: Dict[str, Any], contract_name: str) -> List[Dict[str, Any]]:
        """
        Smart validation of logical names using multi-variant specification logic.
        
        This implements the core Smart Specification Selection validation:
        - Contract input is valid if it exists in ANY variant
        - Contract must cover intersection of REQUIRED dependencies
        - Provides detailed feedback about which variants need what
        
        Args:
            contract: Contract dictionary
            unified_spec: Unified specification model
            contract_name: Name of the contract
            
        Returns:
            List of validation issues
        """
        issues = []
        
        # Get logical names from contract
        contract_inputs = set(contract.get('inputs', {}).keys())
        contract_outputs = set(contract.get('outputs', {}).keys())
        
        # Get unified logical names from all specification variants
        unified_dependencies = unified_spec.get('unified_dependencies', {})
        unified_outputs = unified_spec.get('unified_outputs', {})
        dependency_sources = unified_spec.get('dependency_sources', {})
        output_sources = unified_spec.get('output_sources', {})
        variants = unified_spec.get('variants', {})
        
        # SMART VALIDATION LOGIC
        
        # 1. Check contract inputs against unified dependencies
        unified_dep_names = set(unified_dependencies.keys())
        
        # Contract inputs that are not in ANY variant are errors
        invalid_inputs = contract_inputs - unified_dep_names
        for logical_name in invalid_inputs:
            issues.append({
                'severity': 'ERROR',
                'category': 'logical_names',
                'message': f'Contract input {logical_name} not declared in any specification variant',
                'details': {
                    'logical_name': logical_name, 
                    'contract': contract_name,
                    'available_variants': list(variants.keys()),
                    'available_dependencies': list(unified_dep_names)
                },
                'recommendation': f'Add {logical_name} to specification dependencies or remove from contract'
            })
        
        # 2. Check for required dependencies that contract doesn't provide
        required_deps = set()
        optional_deps = set()
        
        for dep_name, dep_spec in unified_dependencies.items():
            if dep_spec.get('required', False):
                required_deps.add(dep_name)
            else:
                optional_deps.add(dep_name)
        
        missing_required = required_deps - contract_inputs
        for logical_name in missing_required:
            # Find which variants require this dependency
            requiring_variants = dependency_sources.get(logical_name, [])
            issues.append({
                'severity': 'ERROR',
                'category': 'logical_names',
                'message': f'Contract missing required dependency {logical_name}',
                'details': {
                    'logical_name': logical_name,
                    'contract': contract_name,
                    'requiring_variants': requiring_variants
                },
                'recommendation': f'Add {logical_name} to contract inputs (required by variants: {", ".join(requiring_variants)})'
            })
        
        # 3. Provide informational feedback for valid optional inputs
        valid_optional_inputs = contract_inputs & optional_deps
        for logical_name in valid_optional_inputs:
            supporting_variants = dependency_sources.get(logical_name, [])
            if len(supporting_variants) < len(variants):
                # This input is only used by some variants - provide info
                issues.append({
                    'severity': 'INFO',
                    'category': 'logical_names',
                    'message': f'Contract input {logical_name} used by variants: {", ".join(supporting_variants)}',
                    'details': {
                        'logical_name': logical_name,
                        'contract': contract_name,
                        'supporting_variants': supporting_variants,
                        'total_variants': len(variants)
                    },
                    'recommendation': f'Input {logical_name} is correctly declared for multi-variant support'
                })
        
        # 4. Check contract outputs against unified outputs
        unified_output_names = set(unified_outputs.keys())
        
        # Contract outputs that are not in ANY variant are errors
        invalid_outputs = contract_outputs - unified_output_names
        for logical_name in invalid_outputs:
            issues.append({
                'severity': 'ERROR',
                'category': 'logical_names',
                'message': f'Contract output {logical_name} not declared in any specification variant',
                'details': {
                    'logical_name': logical_name,
                    'contract': contract_name,
                    'available_variants': list(variants.keys()),
                    'available_outputs': list(unified_output_names)
                },
                'recommendation': f'Add {logical_name} to specification outputs or remove from contract'
            })
        
        # 5. Check for missing outputs (less critical since outputs are usually consistent)
        missing_outputs = unified_output_names - contract_outputs
        for logical_name in missing_outputs:
            producing_variants = output_sources.get(logical_name, [])
            issues.append({
                'severity': 'WARNING',
                'category': 'logical_names',
                'message': f'Contract missing output {logical_name}',
                'details': {
                    'logical_name': logical_name,
                    'contract': contract_name,
                    'producing_variants': producing_variants
                },
                'recommendation': f'Add {logical_name} to contract outputs (produced by variants: {", ".join(producing_variants)})'
            })
        
        # 6. Add summary information about multi-variant validation
        if len(variants) > 1:
            issues.append({
                'severity': 'INFO',
                'category': 'multi_variant_validation',
                'message': f'Smart Specification Selection: validated against {len(variants)} variants',
                'details': {
                    'contract': contract_name,
                    'variants': list(variants.keys()),
                    'total_dependencies': len(unified_dependencies),
                    'total_outputs': len(unified_outputs),
                    'contract_inputs': len(contract_inputs),
                    'contract_outputs': len(contract_outputs)
                },
                'recommendation': 'Multi-variant validation completed successfully'
            })
        
        return issues

    def _discover_contracts(self) -> List[str]:
        """Discover all contract files in the contracts directory."""
        contracts = []
        
        if self.contracts_dir.exists():
            for contract_file in self.contracts_dir.glob("*_contract.py"):
                if not contract_file.name.startswith('__'):
                    contract_name = contract_file.stem.replace('_contract', '')
                    contracts.append(contract_name)
        
        return sorted(contracts)
    
    def _discover_contracts_with_scripts(self) -> List[str]:
        """
        Discover contracts that have corresponding scripts by checking their entry_point field.
        
        This method loads each contract and checks if the script file referenced in the
        entry_point field actually exists, preventing validation errors for contracts
        without corresponding scripts.
        
        Returns:
            List of contract names that have corresponding scripts
        """
        from src.cursus.validation.alignment.unified_alignment_tester import UnifiedAlignmentTester
        
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
                contract = self._load_contract_from_python(contract_file, contract_name)
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
