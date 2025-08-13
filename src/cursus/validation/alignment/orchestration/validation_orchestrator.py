"""
Validation Orchestrator

Coordinates the overall validation process by orchestrating different
validation components and managing the validation workflow.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path


class ValidationOrchestrator:
    """
    Orchestrates the validation process across multiple components.
    
    Coordinates:
    - Contract discovery and loading
    - Specification discovery and loading
    - Smart specification selection
    - Multi-level validation execution
    - Result aggregation and processing
    - Error handling and recovery
    """
    
    def __init__(self, contracts_dir: str, specs_dir: str):
        """
        Initialize the validation orchestrator.
        
        Args:
            contracts_dir: Directory containing script contracts
            specs_dir: Directory containing step specifications
        """
        self.contracts_dir = Path(contracts_dir)
        self.specs_dir = Path(specs_dir)
        
        # Initialize components (will be injected by main class)
        self.contract_discovery = None
        self.spec_processor = None
        self.contract_loader = None
        self.spec_loader = None
        self.smart_spec_selector = None
        self.validator = None
        self.property_path_validator = None
    
    def set_components(self, **components):
        """
        Set the validation components.
        
        Args:
            **components: Dictionary of component instances
        """
        for name, component in components.items():
            setattr(self, name, component)
    
    def orchestrate_contract_validation(self, contract_name: str) -> Dict[str, Any]:
        """
        Orchestrate the complete validation process for a single contract.
        
        This is the main orchestration method that coordinates all validation steps:
        1. Contract discovery and loading
        2. Specification discovery and loading
        3. Smart specification selection
        4. Multi-level validation
        5. Result aggregation
        
        Args:
            contract_name: Name of the contract to validate
            
        Returns:
            Complete validation result dictionary
        """
        try:
            # Step 1: Discover and validate contract file existence
            contract_file_path = self._discover_contract_file(contract_name)
            if not contract_file_path:
                return self._create_missing_contract_result(contract_name)
            
            # Step 2: Load contract
            contract = self._load_contract_safely(contract_file_path, contract_name)
            if 'error' in contract:
                return self._create_contract_load_error_result(contract_name, contract['error'])
            
            # Step 3: Discover and load specifications
            specifications = self._discover_and_load_specifications(contract_name)
            if not specifications:
                return self._create_missing_specifications_result(contract_name)
            
            # Step 4: Apply Smart Specification Selection
            unified_spec = self._create_unified_specification(specifications, contract_name)
            
            # Step 5: Execute validation pipeline
            validation_results = self._execute_validation_pipeline(
                contract, unified_spec, specifications, contract_name
            )
            
            # Step 6: Aggregate and finalize results
            return self._finalize_validation_results(
                validation_results, contract, specifications, unified_spec, contract_name
            )
            
        except Exception as e:
            return self._create_orchestration_error_result(contract_name, str(e))
    
    def orchestrate_batch_validation(self, target_scripts: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Orchestrate validation for multiple contracts.
        
        Args:
            target_scripts: Specific scripts to validate (None for all)
            
        Returns:
            Dictionary mapping contract names to validation results
        """
        results = {}
        
        # Determine contracts to validate
        if target_scripts:
            contracts_to_validate = target_scripts
        else:
            contracts_to_validate = self._discover_contracts_with_scripts()
        
        # Validate each contract
        for contract_name in contracts_to_validate:
            try:
                result = self.orchestrate_contract_validation(contract_name)
                results[contract_name] = result
            except Exception as e:
                results[contract_name] = self._create_orchestration_error_result(contract_name, str(e))
        
        return results
    
    def _discover_contract_file(self, contract_name: str) -> Optional[str]:
        """Discover contract file using the contract discovery engine."""
        if self.contract_discovery:
            # Use FlexibleFileResolver through contract discovery
            from ..alignment_utils import FlexibleFileResolver
            base_directories = {
                'contracts': str(self.contracts_dir),
                'specs': str(self.specs_dir)
            }
            file_resolver = FlexibleFileResolver(base_directories)
            return file_resolver.find_contract_file(contract_name)
        return None
    
    def _load_contract_safely(self, contract_file_path: str, contract_name: str) -> Dict[str, Any]:
        """Load contract with error handling."""
        try:
            if self.contract_loader:
                return self.contract_loader.load_contract(Path(contract_file_path), contract_name)
            else:
                return {'error': 'Contract loader not initialized'}
        except Exception as e:
            return {'error': str(e)}
    
    def _discover_and_load_specifications(self, contract_name: str) -> Dict[str, Dict[str, Any]]:
        """Discover and load all specifications for a contract."""
        specifications = {}
        
        if not self.spec_loader:
            return specifications
        
        try:
            # Find specification files using script_contract field
            spec_files = self.spec_loader.find_specifications_by_contract(contract_name)
            
            # Load specifications from Python files
            for spec_file, spec_info in spec_files.items():
                try:
                    spec = self.spec_loader.load_specification(spec_file, spec_info)
                    # Use the spec file name as the key since job type comes from config, not spec
                    spec_key = spec_file.stem
                    specifications[spec_key] = spec
                except Exception as e:
                    # Log specification loading error but continue with others
                    print(f"⚠️  Failed to load specification from {spec_file}: {str(e)}")
                    continue
            
            return specifications
            
        except Exception as e:
            print(f"⚠️  Error discovering specifications for {contract_name}: {str(e)}")
            return specifications
    
    def _create_unified_specification(self, specifications: Dict[str, Dict[str, Any]], 
                                    contract_name: str) -> Dict[str, Any]:
        """Create unified specification using Smart Specification Selection."""
        if self.smart_spec_selector:
            return self.smart_spec_selector.create_unified_specification(specifications, contract_name)
        else:
            # Fallback: use first specification as primary
            if specifications:
                primary_spec = next(iter(specifications.values()))
                return {
                    'primary_spec': primary_spec,
                    'variants': specifications,
                    'unified_dependencies': {},
                    'unified_outputs': {},
                    'variant_count': len(specifications)
                }
            return {'primary_spec': {}, 'variants': {}, 'unified_dependencies': {}, 'unified_outputs': {}, 'variant_count': 0}
    
    def _execute_validation_pipeline(self, contract: Dict[str, Any], unified_spec: Dict[str, Any], 
                                   specifications: Dict[str, Dict[str, Any]], contract_name: str) -> List[Dict[str, Any]]:
        """Execute the complete validation pipeline."""
        all_issues = []
        
        try:
            # Validate logical name alignment using smart multi-variant logic
            if self.smart_spec_selector:
                logical_issues = self.smart_spec_selector.validate_logical_names_smart(
                    contract, unified_spec, contract_name
                )
                all_issues.extend(logical_issues)
            
            # Validate data type consistency
            if self.validator:
                type_issues = self.validator.validate_data_types(
                    contract, unified_spec['primary_spec'], contract_name
                )
                all_issues.extend(type_issues)
                
                # Validate input/output alignment
                io_issues = self.validator.validate_input_output_alignment(
                    contract, unified_spec['primary_spec'], contract_name
                )
                all_issues.extend(io_issues)
            
            # Validate property path references (Level 2 enhancement)
            if self.property_path_validator:
                property_path_issues = self.property_path_validator.validate_specification_property_paths(
                    unified_spec['primary_spec'], contract_name
                )
                all_issues.extend(property_path_issues)
            
        except Exception as e:
            all_issues.append({
                'severity': 'CRITICAL',
                'category': 'validation_pipeline_error',
                'message': f'Validation pipeline error: {str(e)}',
                'details': {'contract': contract_name, 'error': str(e)},
                'recommendation': 'Check validation pipeline configuration and component initialization'
            })
        
        return all_issues
    
    def _finalize_validation_results(self, validation_issues: List[Dict[str, Any]], 
                                   contract: Dict[str, Any], specifications: Dict[str, Dict[str, Any]], 
                                   unified_spec: Dict[str, Any], contract_name: str) -> Dict[str, Any]:
        """Finalize and format validation results."""
        # Determine overall pass/fail status
        has_critical_or_error = any(
            issue['severity'] in ['CRITICAL', 'ERROR'] for issue in validation_issues
        )
        
        return {
            'passed': not has_critical_or_error,
            'issues': validation_issues,
            'contract': contract,
            'specifications': specifications,
            'unified_specification': unified_spec,
            'validation_metadata': {
                'contract_name': contract_name,
                'specification_count': len(specifications),
                'unified_dependencies_count': len(unified_spec.get('unified_dependencies', {})),
                'unified_outputs_count': len(unified_spec.get('unified_outputs', {})),
                'total_issues': len(validation_issues),
                'critical_issues': len([i for i in validation_issues if i['severity'] == 'CRITICAL']),
                'error_issues': len([i for i in validation_issues if i['severity'] == 'ERROR']),
                'warning_issues': len([i for i in validation_issues if i['severity'] == 'WARNING']),
                'info_issues': len([i for i in validation_issues if i['severity'] == 'INFO'])
            }
        }
    
    def _discover_contracts_with_scripts(self) -> List[str]:
        """Discover contracts that have corresponding scripts."""
        if self.contract_discovery:
            return self.contract_discovery.discover_contracts_with_scripts()
        else:
            # Fallback: discover all contracts
            contracts = []
            if self.contracts_dir.exists():
                for contract_file in self.contracts_dir.glob("*_contract.py"):
                    if not contract_file.name.startswith('__'):
                        contract_name = contract_file.stem.replace('_contract', '')
                        contracts.append(contract_name)
            return sorted(contracts)
    
    def _create_missing_contract_result(self, contract_name: str) -> Dict[str, Any]:
        """Create result for missing contract file."""
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
            }],
            'validation_metadata': {
                'contract_name': contract_name,
                'validation_stage': 'contract_discovery'
            }
        }
    
    def _create_contract_load_error_result(self, contract_name: str, error: str) -> Dict[str, Any]:
        """Create result for contract loading error."""
        return {
            'passed': False,
            'issues': [{
                'severity': 'CRITICAL',
                'category': 'contract_load_error',
                'message': f'Failed to load contract: {error}',
                'recommendation': 'Fix Python syntax or contract structure in contract file'
            }],
            'validation_metadata': {
                'contract_name': contract_name,
                'validation_stage': 'contract_loading'
            }
        }
    
    def _create_missing_specifications_result(self, contract_name: str) -> Dict[str, Any]:
        """Create result for missing specifications."""
        return {
            'passed': False,
            'issues': [{
                'severity': 'ERROR',
                'category': 'missing_specification',
                'message': f'No specification files found for {contract_name}',
                'recommendation': f'Create specification files that reference {contract_name}_contract'
            }],
            'validation_metadata': {
                'contract_name': contract_name,
                'validation_stage': 'specification_discovery'
            }
        }
    
    def _create_orchestration_error_result(self, contract_name: str, error: str) -> Dict[str, Any]:
        """Create result for orchestration error."""
        return {
            'passed': False,
            'error': error,
            'issues': [{
                'severity': 'CRITICAL',
                'category': 'orchestration_error',
                'message': f'Orchestration failed for contract {contract_name}: {error}',
                'recommendation': 'Check orchestration configuration and component initialization'
            }],
            'validation_metadata': {
                'contract_name': contract_name,
                'validation_stage': 'orchestration'
            }
        }
