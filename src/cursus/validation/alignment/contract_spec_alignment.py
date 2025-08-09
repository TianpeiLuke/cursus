"""
Contract ↔ Specification Alignment Tester

Validates alignment between script contracts and step specifications.
Ensures logical names, data types, and dependencies are consistent.
"""

import os
import json
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
        contract_path = self.contracts_dir / f"{contract_name}_contract.json"
        spec_path = self.specs_dir / f"{contract_name}_spec.json"
        
        # Check if files exist
        if not contract_path.exists():
            return {
                'passed': False,
                'issues': [{
                    'severity': 'CRITICAL',
                    'category': 'missing_file',
                    'message': f'Contract file not found: {contract_path}',
                    'recommendation': f'Create the contract file {contract_name}_contract.json'
                }]
            }
        
        if not spec_path.exists():
            return {
                'passed': False,
                'issues': [{
                    'severity': 'ERROR',
                    'category': 'missing_specification',
                    'message': f'Specification file not found: {spec_path}',
                    'recommendation': f'Create specification file {contract_name}_spec.json'
                }]
            }
        
        # Load contract and specification
        try:
            with open(contract_path, 'r') as f:
                contract = json.load(f)
        except Exception as e:
            return {
                'passed': False,
                'issues': [{
                    'severity': 'CRITICAL',
                    'category': 'contract_parse_error',
                    'message': f'Failed to parse contract: {str(e)}',
                    'recommendation': 'Fix JSON syntax in contract file'
                }]
            }
        
        try:
            with open(spec_path, 'r') as f:
                specification = json.load(f)
        except Exception as e:
            return {
                'passed': False,
                'issues': [{
                    'severity': 'CRITICAL',
                    'category': 'spec_parse_error',
                    'message': f'Failed to parse specification: {str(e)}',
                    'recommendation': 'Fix JSON syntax in specification file'
                }]
            }
        
        # Perform alignment validation
        issues = []
        
        # Validate logical name alignment
        logical_issues = self._validate_logical_names(contract, specification, contract_name)
        issues.extend(logical_issues)
        
        # Validate data type consistency
        type_issues = self._validate_data_types(contract, specification, contract_name)
        issues.extend(type_issues)
        
        # Validate input/output alignment
        io_issues = self._validate_input_output_alignment(contract, specification, contract_name)
        issues.extend(io_issues)
        
        # Determine overall pass/fail status
        has_critical_or_error = any(
            issue['severity'] in ['CRITICAL', 'ERROR'] for issue in issues
        )
        
        return {
            'passed': not has_critical_or_error,
            'issues': issues,
            'contract': contract,
            'specification': specification
        }
    
    def _validate_logical_names(self, contract: Dict[str, Any], specification: Dict[str, Any], contract_name: str) -> List[Dict[str, Any]]:
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
        
        # Compare input data types
        contract_inputs = contract.get('inputs', {})
        spec_deps = {dep.get('logical_name'): dep for dep in specification.get('dependencies', [])}
        
        for logical_name, contract_input in contract_inputs.items():
            if logical_name in spec_deps:
                contract_type = contract_input.get('data_type')
                spec_type = spec_deps[logical_name].get('data_type')
                
                if contract_type and spec_type and contract_type != spec_type:
                    issues.append({
                        'severity': 'WARNING',
                        'category': 'data_types',
                        'message': f'Data type mismatch for {logical_name}: contract={contract_type}, spec={spec_type}',
                        'details': {
                            'logical_name': logical_name,
                            'contract_type': contract_type,
                            'spec_type': spec_type,
                            'contract': contract_name
                        },
                        'recommendation': f'Align data types for {logical_name} between contract and specification'
                    })
        
        # Compare output data types
        contract_outputs = contract.get('outputs', {})
        spec_outputs = {out.get('logical_name'): out for out in specification.get('outputs', [])}
        
        for logical_name, contract_output in contract_outputs.items():
            if logical_name in spec_outputs:
                contract_type = contract_output.get('data_type')
                spec_type = spec_outputs[logical_name].get('data_type')
                
                if contract_type and spec_type and contract_type != spec_type:
                    issues.append({
                        'severity': 'WARNING',
                        'category': 'data_types',
                        'message': f'Output data type mismatch for {logical_name}: contract={contract_type}, spec={spec_type}',
                        'details': {
                            'logical_name': logical_name,
                            'contract_type': contract_type,
                            'spec_type': spec_type,
                            'contract': contract_name
                        },
                        'recommendation': f'Align output data types for {logical_name} between contract and specification'
                    })
        
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
            for contract_file in self.contracts_dir.glob("*_contract.json"):
                contract_name = contract_file.stem.replace('_contract', '')
                contracts.append(contract_name)
        
        return sorted(contracts)
