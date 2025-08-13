"""
Legacy Validation Methods

Contains legacy validation methods for backward compatibility.
These methods provide single-variant validation logic that was used
before the Smart Specification Selection system was implemented.
"""

from typing import Dict, List, Any, Optional


class LegacyValidators:
    """
    Legacy validation methods for backward compatibility.
    
    These methods provide the original single-variant validation logic
    that was used before Smart Specification Selection was implemented.
    They are kept for compatibility and comparison purposes.
    """
    
    def __init__(self):
        """Initialize the legacy validators."""
        pass
    
    def validate_logical_names(self, contract: Dict[str, Any], specification: Dict[str, Any], 
                             contract_name: str, job_type: str = None) -> List[Dict[str, Any]]:
        """
        Validate that logical names match between contract and specification (legacy single-variant).
        
        This is the original validation logic that works with a single specification variant.
        It has been superseded by the Smart Specification Selection logic but is kept
        for backward compatibility.
        
        Args:
            contract: Contract dictionary
            specification: Single specification dictionary
            contract_name: Name of the contract
            job_type: Job type (optional, for context)
            
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
    
    def validate_data_types(self, contract: Dict[str, Any], specification: Dict[str, Any], 
                          contract_name: str) -> List[Dict[str, Any]]:
        """
        Validate data type consistency between contract and specification (legacy).
        
        Note: Contract inputs/outputs are typically stored as simple path strings,
        while specifications have rich data type information.
        For now, we'll skip detailed data type validation since the contract
        format doesn't include explicit data type declarations.
        
        This could be enhanced in the future if contracts are extended
        to include data type information.
        
        Args:
            contract: Contract dictionary
            specification: Specification dictionary
            contract_name: Name of the contract
            
        Returns:
            List of validation issues (currently empty)
        """
        issues = []
        
        # Note: Contract inputs/outputs are typically stored as simple path strings,
        # while specifications have rich data type information.
        # For now, we'll skip detailed data type validation since the contract
        # format doesn't include explicit data type declarations.
        
        # This could be enhanced in the future if contracts are extended
        # to include data type information.
        
        return issues
    
    def validate_input_output_alignment(self, contract: Dict[str, Any], specification: Dict[str, Any], 
                                      contract_name: str) -> List[Dict[str, Any]]:
        """
        Validate input/output alignment between contract and specification (legacy).
        
        This method checks for specification dependencies/outputs that don't have
        corresponding contract inputs/outputs, which can indicate missing or
        mismatched declarations.
        
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
    
    def create_unified_specification_legacy(self, specifications: Dict[str, Dict[str, Any]], 
                                          contract_name: str) -> Dict[str, Any]:
        """
        Create a unified specification model from multiple specification variants (legacy version).
        
        This is the original implementation before Smart Specification Selection.
        It provides a simpler union-based approach without the sophisticated
        multi-variant logic of the new system.
        
        Args:
            specifications: Dictionary of loaded specifications
            contract_name: Name of the contract being validated
            
        Returns:
            Unified specification model (legacy format)
        """
        if not specifications:
            return {'primary_spec': {}, 'variants': {}, 'unified_dependencies': set(), 'unified_outputs': set()}
        
        # Group specifications by job type (simplified)
        variants = {}
        for spec_key, spec_data in specifications.items():
            job_type = self._extract_job_type_from_spec_name_legacy(spec_key)
            variants[job_type] = spec_data
        
        # Create unified dependency and output sets (simple union)
        unified_dependencies = {}
        unified_outputs = {}
        
        # Union all dependencies from all variants
        for variant_name, spec_data in variants.items():
            for dep in spec_data.get('dependencies', []):
                logical_name = dep.get('logical_name')
                if logical_name:
                    unified_dependencies[logical_name] = dep
            
            for output in spec_data.get('outputs', []):
                logical_name = output.get('logical_name')
                if logical_name:
                    unified_outputs[logical_name] = output
        
        # Select primary specification (prefer training, then first available)
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
            'variant_count': len(variants)
        }
    
    def _extract_job_type_from_spec_name_legacy(self, spec_name: str) -> str:
        """Extract job type from specification name (legacy version)."""
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
