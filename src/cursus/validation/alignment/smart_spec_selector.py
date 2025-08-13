"""
Smart Specification Selection Module

Implements the Smart Specification Selection logic for multi-variant specification handling.
Creates unified specification models from multiple specification variants.
"""

from typing import Dict, Any, Set, List
from pathlib import Path


class SmartSpecificationSelector:
    """
    Handles Smart Specification Selection logic for multi-variant specifications.
    
    This implements the core logic for:
    1. Detecting specification variants (training, testing, validation, calibration)
    2. Creating a union of all dependencies and outputs
    3. Providing metadata about which variants contribute what
    4. Selecting primary specifications for validation
    """
    
    def create_unified_specification(self, specifications: Dict[str, Dict[str, Any]], contract_name: str) -> Dict[str, Any]:
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
            return {'primary_spec': {}, 'variants': {}, 'unified_dependencies': {}, 'unified_outputs': {}}
        
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
        primary_spec = self._select_primary_specification(variants)
        
        return {
            'primary_spec': primary_spec,
            'variants': variants,
            'unified_dependencies': unified_dependencies,
            'unified_outputs': unified_outputs,
            'dependency_sources': dependency_sources,
            'output_sources': output_sources,
            'variant_count': len(variants)
        }
    
    def _select_primary_specification(self, variants: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select the primary specification from available variants.
        
        Priority order:
        1. training (most common and comprehensive)
        2. generic (applies to all job types)
        3. first available variant
        
        Args:
            variants: Dictionary of specification variants
            
        Returns:
            Primary specification dictionary
        """
        if 'training' in variants:
            return variants['training']
        elif 'generic' in variants:
            return variants['generic']
        else:
            return next(iter(variants.values())) if variants else {}
    
    def _extract_job_type_from_spec_name(self, spec_name: str) -> str:
        """
        Extract job type from specification name.
        
        Args:
            spec_name: Name of the specification
            
        Returns:
            Job type string
        """
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
    
    def validate_logical_names_smart(self, contract: Dict[str, Any], unified_spec: Dict[str, Any], contract_name: str) -> List[Dict[str, Any]]:
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
