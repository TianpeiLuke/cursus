"""
Contract-Specification Validator Module

Contains the core validation logic for contract-specification alignment.
Handles data type validation, input/output alignment, and basic logical name validation.
"""

from typing import Dict, Any, List


class ContractSpecValidator:
    """
    Handles core validation logic for contract-specification alignment.
    
    Provides methods for:
    - Data type consistency validation
    - Input/output alignment validation
    - Basic logical name validation (non-smart)
    """
    
    def validate_logical_names(self, contract: Dict[str, Any], specification: Dict[str, Any], contract_name: str, job_type: str = None) -> List[Dict[str, Any]]:
        """
        Validate that logical names match between contract and specification.
        
        This is the basic (non-smart) validation for single specifications.
        
        Args:
            contract: Contract dictionary
            specification: Specification dictionary
            contract_name: Name of the contract
            job_type: Job type (optional)
            
        Returns:
            List of validation issues
        """
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
    
    def validate_data_types(self, contract: Dict[str, Any], specification: Dict[str, Any], contract_name: str) -> List[Dict[str, Any]]:
        """
        Validate data type consistency between contract and specification.
        
        Args:
            contract: Contract dictionary
            specification: Specification dictionary
            contract_name: Name of the contract
            
        Returns:
            List of validation issues
        """
        issues = []
        
        # Note: Contract inputs/outputs are typically stored as simple path strings,
        # while specifications have rich data type information.
        # For now, we'll skip detailed data type validation since the contract
        # format doesn't include explicit data type declarations.
        
        # This could be enhanced in the future if contracts are extended
        # to include data type information.
        
        return issues
    
    def validate_input_output_alignment(self, contract: Dict[str, Any], specification: Dict[str, Any], contract_name: str) -> List[Dict[str, Any]]:
        """
        Validate input/output alignment between contract and specification.
        
        Args:
            contract: Contract dictionary
            specification: Specification dictionary
            contract_name: Name of the contract
            
        Returns:
            List of validation issues
        """
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
