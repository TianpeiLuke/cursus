"""
Contract ↔ Specification Alignment Tester

Validates alignment between script contracts and step specifications.
Ensures logical names, data types, and dependencies are consistent.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path

from .alignment_utils import FlexibleFileResolver
from .property_path_validator import SageMakerPropertyPathValidator
from .loaders import ContractLoader, SpecificationLoader
from .smart_spec_selector import SmartSpecificationSelector
from .validators import ContractSpecValidator
from .discovery import ContractDiscoveryEngine
from .processors import SpecificationFileProcessor
from .orchestration import ValidationOrchestrator


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
        
        # Initialize property path validator
        self.property_path_validator = SageMakerPropertyPathValidator()
        
        # Initialize loaders
        self.contract_loader = ContractLoader(str(self.contracts_dir))
        self.spec_loader = SpecificationLoader(str(self.specs_dir))
        
        # Initialize smart specification selector
        self.smart_spec_selector = SmartSpecificationSelector()
        
        # Initialize validator
        self.validator = ContractSpecValidator()
    
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
    
    def validate_contract(self, script_or_contract_name: str) -> Dict[str, Any]:
        """
        Validate alignment for a specific contract using Smart Specification Selection.
        
        Args:
            script_or_contract_name: Name of the script or contract to validate
            
        Returns:
            Validation result dictionary
        """
        # Use FlexibleFileResolver to find the correct contract file
        contract_file_path = self.file_resolver.find_contract_file(script_or_contract_name)
        
        # Check if contract file exists
        if not contract_file_path:
            return {
                'passed': False,
                'issues': [{
                    'severity': 'CRITICAL',
                    'category': 'missing_file',
                    'message': f'Contract file not found for script: {script_or_contract_name}',
                    'details': {
                        'script': script_or_contract_name,
                        'searched_patterns': [
                            f'{script_or_contract_name}_contract.py',
                            'Known naming patterns from FlexibleFileResolver'
                        ]
                    },
                    'recommendation': f'Create contract file for {script_or_contract_name} or check naming patterns'
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
        
        # Extract the actual contract name from the file path
        # e.g., "xgboost_model_eval_contract.py" -> "xgboost_model_eval_contract"
        actual_contract_name = contract_path.stem
        
        # Load contract from Python file
        try:
            contract = self.contract_loader.load_contract(contract_path, actual_contract_name)
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
        
        # Find specification files using the actual contract name
        spec_files = self.spec_loader.find_specifications_by_contract(actual_contract_name)
        
        if not spec_files:
            return {
                'passed': False,
                'issues': [{
                    'severity': 'ERROR',
                    'category': 'missing_specification',
                    'message': f'No specification files found for {actual_contract_name}',
                    'recommendation': f'Create specification files that reference {actual_contract_name}'
                }]
            }
        
        # Load specifications from Python files
        specifications = {}
        for spec_file, spec_info in spec_files.items():
            try:
                spec = self.spec_loader.load_specification(spec_file, spec_info)
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
        unified_spec = self.smart_spec_selector.create_unified_specification(specifications, actual_contract_name)
        
        # Perform alignment validation against unified specification
        all_issues = []
        
        # Validate logical name alignment using smart multi-variant logic
        logical_issues = self.smart_spec_selector.validate_logical_names_smart(contract, unified_spec, actual_contract_name)
        all_issues.extend(logical_issues)
        
        # Validate data type consistency
        type_issues = self.validator.validate_data_types(contract, unified_spec['primary_spec'], actual_contract_name)
        all_issues.extend(type_issues)
        
        # Validate input/output alignment
        io_issues = self.validator.validate_input_output_alignment(contract, unified_spec['primary_spec'], actual_contract_name)
        all_issues.extend(io_issues)
        
        # NEW: Validate property path references (Level 2 enhancement)
        property_path_issues = self._validate_property_paths(unified_spec['primary_spec'], actual_contract_name)
        all_issues.extend(property_path_issues)
        
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
    
    
    
    # Methods moved to extracted components:
    # - _extract_contract_reference -> ContractDiscoveryEngine
    # - _extract_spec_name_from_file -> SpecificationFileProcessor  
    # - _extract_job_type_from_spec_file -> SpecificationFileProcessor
    
    # Methods moved to extracted components:
    # - _load_specification_from_file -> SpecificationFileProcessor
    # - _load_specification_from_python -> SpecificationFileProcessor
    # - _validate_logical_names -> ContractSpecValidator (legacy version)
    # - _validate_data_types -> ContractSpecValidator
    # - _validate_input_output_alignment -> ContractSpecValidator
    
    # Methods moved to SmartSpecificationSelector:
    # - _extract_script_contract_from_spec -> ContractDiscoveryEngine
    # - _contracts_match -> ContractDiscoveryEngine  
    # - _create_unified_specification -> SmartSpecificationSelector
    # - _extract_job_type_from_spec_name -> SpecificationFileProcessor
    # - _validate_logical_names_smart -> SmartSpecificationSelector

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
        
        This method uses the ContractDiscoveryEngine to find contracts that have
        corresponding scripts, preventing validation errors for contracts without scripts.
        
        Returns:
            List of contract names that have corresponding scripts
        """
        # Use ContractDiscoveryEngine for robust contract discovery
        discovery_engine = ContractDiscoveryEngine(str(self.contracts_dir))
        return discovery_engine.discover_contracts_with_scripts()
    
    def _validate_property_paths(self, specification: Dict[str, Any], contract_name: str) -> List[Dict[str, Any]]:
        """
        Validate SageMaker Step Property Path References (Level 2 Enhancement).
        
        Uses the dedicated SageMakerPropertyPathValidator to validate that property paths
        used in specification outputs are valid for the specified SageMaker step type.
        
        Args:
            specification: Specification dictionary
            contract_name: Name of the contract being validated
            
        Returns:
            List of validation issues
        """
        return self.property_path_validator.validate_specification_property_paths(
            specification, contract_name
        )
