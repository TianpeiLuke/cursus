"""
Script ↔ Contract Alignment Tester

Validates alignment between processing scripts and their contracts.
Ensures scripts use paths, environment variables, and arguments as declared in contracts.
"""

import os
import json
import sys
import importlib.util
from typing import Dict, List, Any, Optional, Set
from pathlib import Path

from .static_analysis.script_analyzer import ScriptAnalyzer
from .alignment_utils import (
    SeverityLevel, create_alignment_issue, normalize_path,
    extract_logical_name_from_path, is_sagemaker_path
)


class ScriptContractAlignmentTester:
    """
    Tests alignment between processing scripts and their contracts.
    
    Validates:
    - Path usage matches contract declarations
    - Environment variable access matches contract
    - Script arguments align with contract expectations
    - File operations match declared inputs/outputs
    """
    
    def __init__(self, scripts_dir: str, contracts_dir: str):
        """
        Initialize the script-contract alignment tester.
        
        Args:
            scripts_dir: Directory containing processing scripts
            contracts_dir: Directory containing script contracts
        """
        self.scripts_dir = Path(scripts_dir)
        self.contracts_dir = Path(contracts_dir)
        
        # Build entry_point to contract file mapping
        self._entry_point_to_contract = self._build_entry_point_mapping()
    
    def validate_all_scripts(self, target_scripts: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Validate alignment for all scripts or specified target scripts.
        
        Args:
            target_scripts: Specific scripts to validate (None for all)
            
        Returns:
            Dictionary mapping script names to validation results
        """
        results = {}
        
        # Discover scripts to validate
        if target_scripts:
            scripts_to_validate = target_scripts
        else:
            scripts_to_validate = self._discover_scripts()
        
        for script_name in scripts_to_validate:
            try:
                result = self.validate_script(script_name)
                results[script_name] = result
            except Exception as e:
                results[script_name] = {
                    'passed': False,
                    'error': str(e),
                    'issues': [{
                        'severity': 'CRITICAL',
                        'category': 'validation_error',
                        'message': f'Failed to validate script {script_name}: {str(e)}'
                    }]
                }
        
        return results
    
    def validate_script(self, script_name: str) -> Dict[str, Any]:
        """
        Validate alignment for a specific script.
        
        Args:
            script_name: Name of the script to validate
            
        Returns:
            Validation result dictionary
        """
        script_path = self.scripts_dir / f"{script_name}.py"
        
        # Use entry_point mapping to find the correct contract file
        script_filename = f"{script_name}.py"
        contract_file = self._entry_point_to_contract.get(script_filename)
        
        if contract_file:
            contract_path = self.contracts_dir / contract_file
        else:
            # Fallback to naming convention
            contract_path = self.contracts_dir / f"{script_name}_contract.py"
        
        # Check if files exist
        if not script_path.exists():
            return {
                'passed': False,
                'issues': [{
                    'severity': 'CRITICAL',
                    'category': 'missing_file',
                    'message': f'Script file not found: {script_path}',
                    'recommendation': f'Create the script file {script_name}.py'
                }]
            }
        
        if not contract_path.exists():
            return {
                'passed': False,
                'issues': [{
                    'severity': 'ERROR',
                    'category': 'missing_contract',
                    'message': f'Contract file not found: {contract_path}',
                    'recommendation': f'Create contract file {contract_file or f"{script_name}_contract.py"}'
                }]
            }
        
        # Load contract from Python module
        try:
            contract = self._load_python_contract(contract_path, script_name)
        except Exception as e:
            return {
                'passed': False,
                'issues': [{
                    'severity': 'CRITICAL',
                    'category': 'contract_parse_error',
                    'message': f'Failed to load contract: {str(e)}',
                    'recommendation': 'Fix Python syntax in contract file'
                }]
            }
        
        # Analyze script
        try:
            analyzer = ScriptAnalyzer(str(script_path))
            analysis = analyzer.get_all_analysis_results()
        except Exception as e:
            return {
                'passed': False,
                'issues': [{
                    'severity': 'CRITICAL',
                    'category': 'script_analysis_error',
                    'message': f'Failed to analyze script: {str(e)}',
                    'recommendation': 'Fix syntax errors in script'
                }]
            }
        
        # Perform alignment validation
        issues = []
        
        # Validate path usage
        path_issues = self._validate_path_usage(analysis, contract, script_name)
        issues.extend(path_issues)
        
        # Validate environment variable usage
        env_issues = self._validate_env_var_usage(analysis, contract, script_name)
        issues.extend(env_issues)
        
        # Validate argument usage
        arg_issues = self._validate_argument_usage(analysis, contract, script_name)
        issues.extend(arg_issues)
        
        # Validate file operations
        file_issues = self._validate_file_operations(analysis, contract, script_name)
        issues.extend(file_issues)
        
        # Determine overall pass/fail status
        has_critical_or_error = any(
            issue['severity'] in ['CRITICAL', 'ERROR'] for issue in issues
        )
        
        return {
            'passed': not has_critical_or_error,
            'issues': issues,
            'script_analysis': analysis,
            'contract': contract
        }
    
    def _load_python_contract(self, contract_path: Path, script_name: str) -> Dict[str, Any]:
        """Load contract from Python module and convert to dictionary format."""
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
                spec = importlib.util.spec_from_file_location(f"{script_name}_contract", contract_path)
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
                f"{script_name.upper()}_CONTRACT",
                f"{script_name}_CONTRACT", 
                f"{script_name}_contract",
                "MODEL_EVALUATION_CONTRACT",  # Specific for model_evaluation_xgb
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
                'entry_point': getattr(contract_obj, 'entry_point', f"{script_name}.py"),
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
                        'required': default_value is None or default_value == ""
                    }
            
            return contract_dict
            
        except Exception as e:
            raise Exception(f"Failed to load Python contract from {contract_path}: {str(e)}")
    
    def _validate_path_usage(self, analysis: Dict[str, Any], contract: Dict[str, Any], script_name: str) -> List[Dict[str, Any]]:
        """Validate that script path usage matches contract declarations."""
        issues = []
        
        # Get contract paths
        contract_inputs = contract.get('inputs', {})
        contract_outputs = contract.get('outputs', {})
        
        # Extract expected paths from contract
        expected_paths = set()
        for input_spec in contract_inputs.values():
            if 'path' in input_spec:
                expected_paths.add(normalize_path(input_spec['path']))
        
        for output_spec in contract_outputs.values():
            if 'path' in output_spec:
                expected_paths.add(normalize_path(output_spec['path']))
        
        # Get script paths
        script_paths = set()
        for path_ref in analysis.get('path_references', []):
            script_paths.add(normalize_path(path_ref.path))
        
        # Check for hardcoded paths not in contract
        undeclared_paths = script_paths - expected_paths
        for path in undeclared_paths:
            if is_sagemaker_path(path):
                issues.append({
                    'severity': 'ERROR',
                    'category': 'path_usage',
                    'message': f'Script uses undeclared SageMaker path: {path}',
                    'details': {'path': path, 'script': script_name},
                    'recommendation': f'Add path {path} to contract inputs or outputs'
                })
        
        # Check for contract paths not used in script
        unused_paths = expected_paths - script_paths
        for path in unused_paths:
            issues.append({
                'severity': 'WARNING',
                'category': 'path_usage',
                'message': f'Contract declares path not used in script: {path}',
                'details': {'path': path, 'script': script_name},
                'recommendation': f'Either use path {path} in script or remove from contract'
            })
        
        # Check for logical name consistency
        script_logical_names = set()
        for path in script_paths:
            logical_name = extract_logical_name_from_path(path)
            if logical_name:
                script_logical_names.add(logical_name)
        
        contract_logical_names = set()
        for input_name in contract_inputs.keys():
            contract_logical_names.add(input_name)
        for output_name in contract_outputs.keys():
            contract_logical_names.add(output_name)
        
        # Check for logical name mismatches
        script_only_names = script_logical_names - contract_logical_names
        for name in script_only_names:
            issues.append({
                'severity': 'WARNING',
                'category': 'logical_names',
                'message': f'Script uses logical name not in contract: {name}',
                'details': {'logical_name': name, 'script': script_name},
                'recommendation': f'Add logical name {name} to contract or update script'
            })
        
        return issues
    
    def _validate_env_var_usage(self, analysis: Dict[str, Any], contract: Dict[str, Any], script_name: str) -> List[Dict[str, Any]]:
        """Validate that script environment variable usage matches contract."""
        issues = []
        
        # Get contract environment variables
        contract_env_vars = set()
        env_config = contract.get('environment_variables', {})
        
        for var_name in env_config.get('required', []):
            contract_env_vars.add(var_name)
        for var_name in env_config.get('optional', []):
            contract_env_vars.add(var_name)
        
        # Get script environment variables
        script_env_vars = set()
        for env_access in analysis.get('env_var_accesses', []):
            script_env_vars.add(env_access.variable_name)
        
        # Check for undeclared environment variables
        undeclared_vars = script_env_vars - contract_env_vars
        for var_name in undeclared_vars:
            issues.append({
                'severity': 'ERROR',
                'category': 'environment_variables',
                'message': f'Script accesses undeclared environment variable: {var_name}',
                'details': {'variable': var_name, 'script': script_name},
                'recommendation': f'Add {var_name} to contract environment_variables'
            })
        
        # Check for required variables not accessed
        required_vars = set(env_config.get('required', []))
        missing_required = required_vars - script_env_vars
        for var_name in missing_required:
            issues.append({
                'severity': 'ERROR',
                'category': 'environment_variables',
                'message': f'Script does not access required environment variable: {var_name}',
                'details': {'variable': var_name, 'script': script_name},
                'recommendation': f'Access required environment variable {var_name} in script'
            })
        
        # Check for proper default handling of optional variables
        optional_vars = set(env_config.get('optional', []))
        for env_access in analysis.get('env_var_accesses', []):
            if env_access.variable_name in optional_vars and not env_access.has_default:
                issues.append({
                    'severity': 'WARNING',
                    'category': 'environment_variables',
                    'message': f'Optional environment variable accessed without default: {env_access.variable_name}',
                    'details': {
                        'variable': env_access.variable_name,
                        'line': env_access.line_number,
                        'script': script_name
                    },
                    'recommendation': f'Provide default value when accessing optional variable {env_access.variable_name}'
                })
        
        return issues
    
    def _validate_argument_usage(self, analysis: Dict[str, Any], contract: Dict[str, Any], script_name: str) -> List[Dict[str, Any]]:
        """Validate that script argument definitions match contract expectations."""
        issues = []
        
        # Get contract arguments
        contract_args = contract.get('arguments', {})
        expected_args = set(contract_args.keys())
        
        # Get script arguments
        script_args = set()
        for arg_def in analysis.get('argument_definitions', []):
            script_args.add(arg_def.argument_name)
        
        # Check for missing arguments
        missing_args = expected_args - script_args
        for arg_name in missing_args:
            issues.append({
                'severity': 'ERROR',
                'category': 'arguments',
                'message': f'Contract declares argument not defined in script: {arg_name}',
                'details': {'argument': arg_name, 'script': script_name},
                'recommendation': f'Add argument parser for {arg_name} in script'
            })
        
        # Check for extra arguments
        extra_args = script_args - expected_args
        for arg_name in extra_args:
            issues.append({
                'severity': 'WARNING',
                'category': 'arguments',
                'message': f'Script defines argument not in contract: {arg_name}',
                'details': {'argument': arg_name, 'script': script_name},
                'recommendation': f'Add {arg_name} to contract arguments or remove from script'
            })
        
        # Validate argument properties
        script_args_dict = {}
        for arg_def in analysis.get('argument_definitions', []):
            script_args_dict[arg_def.argument_name] = arg_def
        
        for arg_name, contract_spec in contract_args.items():
            if arg_name in script_args_dict:
                script_arg = script_args_dict[arg_name]
                
                # Check required vs optional
                contract_required = contract_spec.get('required', False)
                script_required = script_arg.is_required
                
                if contract_required and not script_required:
                    issues.append({
                        'severity': 'ERROR',
                        'category': 'arguments',
                        'message': f'Contract requires argument {arg_name} but script makes it optional',
                        'details': {'argument': arg_name, 'script': script_name},
                        'recommendation': f'Make argument {arg_name} required in script'
                    })
                
                # Check type consistency
                contract_type = contract_spec.get('type')
                script_type = script_arg.argument_type
                
                if contract_type and script_type and contract_type != script_type:
                    issues.append({
                        'severity': 'WARNING',
                        'category': 'arguments',
                        'message': f'Argument {arg_name} type mismatch: contract={contract_type}, script={script_type}',
                        'details': {
                            'argument': arg_name,
                            'contract_type': contract_type,
                            'script_type': script_type,
                            'script': script_name
                        },
                        'recommendation': f'Align argument {arg_name} type between contract and script'
                    })
        
        return issues
    
    def _validate_file_operations(self, analysis: Dict[str, Any], contract: Dict[str, Any], script_name: str) -> List[Dict[str, Any]]:
        """Validate that script file operations align with contract inputs/outputs."""
        issues = []
        
        # Get contract file specifications
        contract_inputs = contract.get('inputs', {})
        contract_outputs = contract.get('outputs', {})
        
        # Collect expected read/write operations
        expected_reads = set()
        expected_writes = set()
        
        for input_spec in contract_inputs.values():
            if 'path' in input_spec:
                expected_reads.add(normalize_path(input_spec['path']))
        
        for output_spec in contract_outputs.values():
            if 'path' in output_spec:
                expected_writes.add(normalize_path(output_spec['path']))
        
        # Get script file operations
        script_reads = set()
        script_writes = set()
        
        for file_op in analysis.get('file_operations', []):
            normalized_path = normalize_path(file_op.file_path)
            
            if file_op.operation_type == 'read':
                script_reads.add(normalized_path)
            elif file_op.operation_type == 'write':
                script_writes.add(normalized_path)
        
        # Check for reads not declared as inputs
        undeclared_reads = script_reads - expected_reads
        for path in undeclared_reads:
            if is_sagemaker_path(path):
                issues.append({
                    'severity': 'WARNING',
                    'category': 'file_operations',
                    'message': f'Script reads from path not declared as input: {path}',
                    'details': {'path': path, 'operation': 'read', 'script': script_name},
                    'recommendation': f'Add {path} to contract inputs'
                })
        
        # Check for writes not declared as outputs
        undeclared_writes = script_writes - expected_writes
        for path in undeclared_writes:
            if is_sagemaker_path(path):
                issues.append({
                    'severity': 'WARNING',
                    'category': 'file_operations',
                    'message': f'Script writes to path not declared as output: {path}',
                    'details': {'path': path, 'operation': 'write', 'script': script_name},
                    'recommendation': f'Add {path} to contract outputs'
                })
        
        # Check for declared inputs not read
        unread_inputs = expected_reads - script_reads
        for path in unread_inputs:
            issues.append({
                'severity': 'INFO',
                'category': 'file_operations',
                'message': f'Contract declares input not read by script: {path}',
                'details': {'path': path, 'operation': 'read', 'script': script_name},
                'recommendation': f'Either read {path} in script or remove from contract inputs'
            })
        
        # Check for declared outputs not written
        unwritten_outputs = expected_writes - script_writes
        for path in unwritten_outputs:
            issues.append({
                'severity': 'WARNING',
                'category': 'file_operations',
                'message': f'Contract declares output not written by script: {path}',
                'details': {'path': path, 'operation': 'write', 'script': script_name},
                'recommendation': f'Either write to {path} in script or remove from contract outputs'
            })
        
        return issues
    
    def _build_entry_point_mapping(self) -> Dict[str, str]:
        """
        Build a mapping from entry_point values to contract file names.
        
        Returns:
            Dictionary mapping entry_point (script filename) to contract filename
        """
        mapping = {}
        
        if not self.contracts_dir.exists():
            return mapping
        
        # Scan all contract files
        for contract_file in self.contracts_dir.glob("*_contract.py"):
            if contract_file.name.startswith('__'):
                continue
                
            try:
                # Extract entry_point from contract
                entry_point = self._extract_entry_point_from_contract(contract_file)
                if entry_point:
                    mapping[entry_point] = contract_file.name
            except Exception:
                # Skip contracts that can't be loaded
                continue
        
        return mapping
    
    def _extract_entry_point_from_contract(self, contract_path: Path) -> Optional[str]:
        """
        Extract the entry_point value from a contract file.
        
        Args:
            contract_path: Path to the contract file
            
        Returns:
            Entry point value or None if not found
        """
        try:
            # Add the project root to sys.path temporarily
            project_root = str(contract_path.parent.parent.parent.parent)
            src_root = str(contract_path.parent.parent.parent)
            contract_dir = str(contract_path.parent)
            
            paths_to_add = [project_root, src_root, contract_dir]
            added_paths = []
            
            for path in paths_to_add:
                if path not in sys.path:
                    sys.path.insert(0, path)
                    added_paths.append(path)
            
            try:
                # Load the module
                spec = importlib.util.spec_from_file_location(
                    f"contract_{contract_path.stem}", contract_path
                )
                if spec is None or spec.loader is None:
                    return None
                
                module = importlib.util.module_from_spec(spec)
                module.__package__ = 'cursus.steps.contracts'
                spec.loader.exec_module(module)
                
                # Look for contract objects and extract entry_point
                for attr_name in dir(module):
                    if attr_name.endswith('_CONTRACT') or attr_name == 'CONTRACT':
                        contract_obj = getattr(module, attr_name)
                        if hasattr(contract_obj, 'entry_point'):
                            return contract_obj.entry_point
                
                return None
                
            finally:
                # Clean up sys.path
                for path in added_paths:
                    if path in sys.path:
                        sys.path.remove(path)
                        
        except Exception:
            return None
    
    def _discover_scripts(self) -> List[str]:
        """Discover all Python scripts in the scripts directory."""
        scripts = []
        
        if self.scripts_dir.exists():
            for script_file in self.scripts_dir.glob("*.py"):
                if not script_file.name.startswith('__'):
                    scripts.append(script_file.stem)
        
        return sorted(scripts)
    
    def get_validation_summary(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of validation results."""
        total_scripts = len(results)
        passed_scripts = sum(1 for result in results.values() if result.get('passed', False))
        
        all_issues = []
        for result in results.values():
            all_issues.extend(result.get('issues', []))
        
        issue_counts = {
            'CRITICAL': sum(1 for issue in all_issues if issue.get('severity') == 'CRITICAL'),
            'ERROR': sum(1 for issue in all_issues if issue.get('severity') == 'ERROR'),
            'WARNING': sum(1 for issue in all_issues if issue.get('severity') == 'WARNING'),
            'INFO': sum(1 for issue in all_issues if issue.get('severity') == 'INFO')
        }
        
        return {
            'total_scripts': total_scripts,
            'passed_scripts': passed_scripts,
            'failed_scripts': total_scripts - passed_scripts,
            'pass_rate': (passed_scripts / total_scripts * 100) if total_scripts > 0 else 0,
            'total_issues': len(all_issues),
            'issue_counts': issue_counts,
            'is_passing': issue_counts['CRITICAL'] == 0 and issue_counts['ERROR'] == 0
        }
