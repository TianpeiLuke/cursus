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
            contracts_to_validate = self._discover_contracts()
        
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
        Validate alignment for a specific contract.
        
        Args:
            contract_name: Name of the contract to validate
            
        Returns:
            Validation result dictionary
        """
        # Look for Python contract file
        contract_path = self.contracts_dir / f"{contract_name}_contract.py"
        
        # Check if contract file exists
        if not contract_path.exists():
            return {
                'passed': False,
                'issues': [{
                    'severity': 'CRITICAL',
                    'category': 'missing_file',
                    'message': f'Contract file not found: {contract_path}',
                    'recommendation': f'Create the contract file {contract_name}_contract.py'
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
        
        # Find specification files (multiple files for different job types)
        spec_files = self._find_specification_files(contract_name)
        
        if not spec_files:
            return {
                'passed': False,
                'issues': [{
                    'severity': 'ERROR',
                    'category': 'missing_specification',
                    'message': f'No specification files found for {contract_name}',
                    'recommendation': f'Create specification files like {contract_name}_training_spec.py'
                }]
            }
        
        # Load specifications from Python files
        specifications = {}
        for spec_file in spec_files:
            try:
                job_type = self._extract_job_type_from_spec_file(spec_file)
                spec = self._load_specification_from_python(spec_file, contract_name, job_type)
                specifications[job_type] = spec
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
        
        # Perform alignment validation for each specification
        all_issues = []
        
        for job_type, specification in specifications.items():
            # Validate logical name alignment
            logical_issues = self._validate_logical_names(contract, specification, contract_name, job_type)
            all_issues.extend(logical_issues)
            
            # Validate data type consistency
            type_issues = self._validate_data_types(contract, specification, contract_name)
            all_issues.extend(type_issues)
            
            # Validate input/output alignment
            io_issues = self._validate_input_output_alignment(contract, specification, contract_name)
            all_issues.extend(io_issues)
        
        # Determine overall pass/fail status
        has_critical_or_error = any(
            issue['severity'] in ['CRITICAL', 'ERROR'] for issue in all_issues
        )
        
        return {
            'passed': not has_critical_or_error,
            'issues': all_issues,
            'contract': contract,
            'specifications': specifications
        }
    
    def _load_contract_from_python(self, contract_path: Path, contract_name: str) -> Dict[str, Any]:
        """Load contract from Python file."""
        try:
            # Read the file content and modify the import to be absolute
            with open(contract_path, 'r') as f:
                content = f.read()
            
            # Replace relative import with absolute import
            modified_content = content.replace(
                'from ...core.base.contract_base import ScriptContract',
                'from src.cursus.core.base.contract_base import ScriptContract'
            )
            
            # Add the project root to sys.path
            project_root = self.contracts_dir.parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            try:
                # Create a temporary module from the modified content
                module_name = f"{contract_name}_contract_temp"
                spec = importlib.util.spec_from_loader(module_name, loader=None)
                module = importlib.util.module_from_spec(spec)
                
                # Execute the modified content in the module's namespace
                exec(modified_content, module.__dict__)
                
                # Look for the contract constant (uppercase name)
                contract_var_name = f"{contract_name.upper()}_CONTRACT"
                if hasattr(module, contract_var_name):
                    contract_obj = getattr(module, contract_var_name)
                    # Convert ScriptContract object to dictionary
                    return {
                        'entry_point': contract_obj.entry_point,
                        'inputs': contract_obj.expected_input_paths,
                        'outputs': contract_obj.expected_output_paths,
                        'arguments': contract_obj.expected_arguments,
                        'environment_variables': {
                            'required': contract_obj.required_env_vars,
                            'optional': contract_obj.optional_env_vars
                        },
                        'framework_requirements': contract_obj.framework_requirements,
                        'description': contract_obj.description
                    }
                else:
                    raise ValueError(f"Contract constant {contract_var_name} not found in {contract_path}")
                    
            finally:
                # Clean up sys.path
                if str(project_root) in sys.path:
                    sys.path.remove(str(project_root))
                    
        except Exception as e:
            # If we still can't load it, provide a more detailed error
            raise ValueError(f"Failed to load contract from {contract_path}: {str(e)}")
    
    def _find_specification_files(self, contract_name: str) -> List[Path]:
        """Find all specification files for a contract."""
        spec_files = []
        if self.specs_dir.exists():
            # Look for files matching pattern: {contract_name}_{job_type}_spec.py
            for spec_file in self.specs_dir.glob(f"{contract_name}_*_spec.py"):
                spec_files.append(spec_file)
        return spec_files
    
    def _extract_job_type_from_spec_file(self, spec_file: Path) -> str:
        """Extract job type from specification file name."""
        # Pattern: {contract_name}_{job_type}_spec.py
        stem = spec_file.stem
        parts = stem.split('_')
        if len(parts) >= 3 and parts[-1] == 'spec':
            return parts[-2]  # job_type is second to last part
        return 'unknown'
    
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
    
    def _discover_contracts(self) -> List[str]:
        """Discover all contract files in the contracts directory."""
        contracts = []
        
        if self.contracts_dir.exists():
            for contract_file in self.contracts_dir.glob("*_contract.py"):
                if not contract_file.name.startswith('__'):
                    contract_name = contract_file.stem.replace('_contract', '')
                    contracts.append(contract_name)
        
        return sorted(contracts)
