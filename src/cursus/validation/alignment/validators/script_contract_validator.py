"""
Script-Contract Validator Module

Contains the core validation logic for script-contract alignment.
Handles path usage, environment variables, arguments, and file operations validation.
"""

from typing import Dict, Any, List, Optional, Set
from pathlib import Path

from ..alignment_utils import (
    normalize_path, extract_logical_name_from_path, is_sagemaker_path
)


class ScriptContractValidator:
    """
    Handles core validation logic for script-contract alignment.
    
    Provides methods for:
    - Path usage validation
    - Environment variable validation
    - Argument validation
    - File operations validation
    """
    
    def validate_path_usage(self, analysis: Dict[str, Any], contract: Dict[str, Any], script_name: str) -> List[Dict[str, Any]]:
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
        
        # Check for logical name consistency using contract mappings
        # This fixes the critical issue of incorrect path-based logical name extraction
        script_logical_names = set()
        contract_logical_names = set()
        
        # Build contract logical names
        for input_name in contract_inputs.keys():
            contract_logical_names.add(input_name)
        for output_name in contract_outputs.keys():
            contract_logical_names.add(output_name)
        
        # Resolve logical names from script paths using contract mappings
        for path in script_paths:
            logical_name = self._resolve_logical_name_from_contract(path, contract)
            if logical_name:
                script_logical_names.add(logical_name)
        
        # Check for logical name mismatches - only flag if path is used but not in contract
        for path in script_paths:
            if is_sagemaker_path(path):
                logical_name = self._resolve_logical_name_from_contract(path, contract)
                if logical_name is None:
                    # Path is used but not mapped to any contract logical name
                    fallback_name = extract_logical_name_from_path(path)
                    issues.append({
                        'severity': 'WARNING',
                        'category': 'logical_names',
                        'message': f'Script uses path not mapped to contract logical name: {path}',
                        'details': {
                            'path': path, 
                            'inferred_logical_name': fallback_name,
                            'script': script_name
                        },
                        'recommendation': f'Add path {path} to contract inputs/outputs with appropriate logical name'
                    })
        
        return issues
    
    def validate_env_var_usage(self, analysis: Dict[str, Any], contract: Dict[str, Any], script_name: str) -> List[Dict[str, Any]]:
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
    
    def validate_argument_usage(self, analysis: Dict[str, Any], contract: Dict[str, Any], script_name: str, builder_args: Set[str] = None) -> List[Dict[str, Any]]:
        """Validate that script argument definitions match contract expectations."""
        issues = []
        
        if builder_args is None:
            builder_args = set()
        
        # Get contract arguments
        contract_args = contract.get('arguments', {})
        
        # Get script arguments
        script_args = {}
        for arg_def in analysis.get('argument_definitions', []):
            script_args[arg_def.argument_name] = arg_def
        
        # Normalize argument names for argparse hyphen-to-underscore conversion
        # Contract uses CLI convention (hyphens), script uses Python convention (underscores)
        normalized_contract_args = {}
        for contract_arg_name, contract_spec in contract_args.items():
            # Convert contract argument name (with hyphens) to Python attribute name (with underscores)
            python_arg_name = contract_arg_name.replace('-', '_')
            normalized_contract_args[python_arg_name] = {
                'contract_name': contract_arg_name,  # Keep original for error messages
                'spec': contract_spec
            }
        
        expected_args = set(normalized_contract_args.keys())
        actual_script_args = set(script_args.keys())
        
        # Check for missing arguments
        missing_args = expected_args - actual_script_args
        for python_arg_name in missing_args:
            contract_arg_name = normalized_contract_args[python_arg_name]['contract_name']
            issues.append({
                'severity': 'ERROR',
                'category': 'arguments',
                'message': f'Contract declares argument not defined in script: {contract_arg_name} (should be accessed as args.{python_arg_name})',
                'details': {
                    'contract_argument': contract_arg_name,
                    'python_attribute': python_arg_name,
                    'script': script_name
                },
                'recommendation': f'Add argument parser for --{contract_arg_name} in script (accessed as args.{python_arg_name})'
            })
        
        # Enhanced check for extra arguments - check builder before declaring failure
        script_cli_args = set()
        for script_arg_name in actual_script_args:
            # Convert Python attribute name back to CLI argument name
            cli_arg_name = script_arg_name.replace('_', '-')
            script_cli_args.add(cli_arg_name)
        
        contract_cli_args = set(contract_args.keys())
        extra_cli_args = script_cli_args - contract_cli_args
        
        for cli_arg_name in extra_cli_args:
            python_arg_name = cli_arg_name.replace('-', '_')
            
            # Check if this argument is provided by the builder
            # Builder args are returned as Python attribute names (underscores), so compare with python_arg_name
            if python_arg_name in builder_args:
                # Argument is provided by builder - this is expected for config-driven arguments
                issues.append({
                    'severity': 'INFO',
                    'category': 'arguments',
                    'message': f'Script defines config-driven argument provided by builder: --{cli_arg_name} (accessed as args.{python_arg_name})',
                    'details': {
                        'cli_argument': cli_arg_name,
                        'python_attribute': python_arg_name,
                        'script': script_name,
                        'source': 'builder'
                    },
                    'recommendation': f'Argument --{cli_arg_name} is provided by builder - no action needed'
                })
            else:
                # Argument is not in contract or builder - this is a real issue
                issues.append({
                    'severity': 'WARNING',
                    'category': 'arguments',
                    'message': f'Script defines argument not in contract: --{cli_arg_name} (accessed as args.{python_arg_name})',
                    'details': {
                        'cli_argument': cli_arg_name,
                        'python_attribute': python_arg_name,
                        'script': script_name
                    },
                    'recommendation': f'Add --{cli_arg_name} to contract arguments or remove from script'
                })
        
        # Validate argument properties using normalized names
        for contract_arg_name, contract_spec in contract_args.items():
            python_arg_name = contract_arg_name.replace('-', '_')
            
            if python_arg_name in script_args:
                script_arg = script_args[python_arg_name]
                
                # Check required vs optional
                contract_required = contract_spec.get('required', False)
                script_required = script_arg.is_required
                
                if contract_required and not script_required:
                    issues.append({
                        'severity': 'ERROR',
                        'category': 'arguments',
                        'message': f'Contract requires argument --{contract_arg_name} but script makes it optional (args.{python_arg_name})',
                        'details': {
                            'contract_argument': contract_arg_name,
                            'python_attribute': python_arg_name,
                            'script': script_name
                        },
                        'recommendation': f'Make argument --{contract_arg_name} required in script'
                    })
                
                # Check type consistency
                contract_type = contract_spec.get('type')
                script_type = script_arg.argument_type
                
                if contract_type and script_type and contract_type != script_type:
                    issues.append({
                        'severity': 'WARNING',
                        'category': 'arguments',
                        'message': f'Argument --{contract_arg_name} type mismatch: contract={contract_type}, script={script_type} (accessed as args.{python_arg_name})',
                        'details': {
                            'contract_argument': contract_arg_name,
                            'python_attribute': python_arg_name,
                            'contract_type': contract_type,
                            'script_type': script_type,
                            'script': script_name
                        },
                        'recommendation': f'Align argument --{contract_arg_name} type between contract and script'
                    })
        
        return issues
    
    def validate_file_operations(self, analysis: Dict[str, Any], contract: Dict[str, Any], script_name: str) -> List[Dict[str, Any]]:
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
        
        # Get script file operations with enhanced detection
        script_reads = set()
        script_writes = set()
        
        # Process detected file operations
        for file_op in analysis.get('file_operations', []):
            normalized_path = normalize_path(file_op.file_path)
            
            if file_op.operation_type == 'read':
                script_reads.add(normalized_path)
            elif file_op.operation_type == 'write':
                script_writes.add(normalized_path)
        
        # Enhanced file operation detection from path references
        # This addresses the critical issue where file operations are missed
        script_reads_enhanced, script_writes_enhanced = self._detect_file_operations_from_paths(
            analysis, contract_inputs, contract_outputs
        )
        script_reads.update(script_reads_enhanced)
        script_writes.update(script_writes_enhanced)
        
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
        
        # Check for declared inputs not read (only if no file operations detected at all)
        if not script_reads and not script_writes:
            # If no file operations detected, this is likely a detection issue, not a real problem
            issues.append({
                'severity': 'INFO',
                'category': 'file_operations',
                'message': f'No file operations detected - this may indicate incomplete static analysis',
                'details': {'script': script_name},
                'recommendation': 'Review script for file operations that may not be detected by static analysis'
            })
        else:
            # Only flag unread inputs if we detected some file operations
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
    
    def _detect_file_operations_from_paths(self, analysis: Dict[str, Any], contract_inputs: Dict[str, Any], contract_outputs: Dict[str, Any]) -> tuple[set, set]:
        """
        Enhanced file operation detection from path references and context.
        
        This addresses the critical issue where basic file operation detection
        misses tarfile, shutil, pathlib, and framework-specific operations.
        """
        script_reads = set()
        script_writes = set()
        
        # Get path references from analysis
        path_references = analysis.get('path_references', [])
        
        # Analyze path usage context to infer file operations
        for path_ref in path_references:
            normalized_path = normalize_path(path_ref.path)
            context = getattr(path_ref, 'context', '').lower()
            
            # Infer operation type from context
            if any(keyword in context for keyword in [
                'read', 'load', 'open', 'extract', 'copy', 'move', 'glob', 'listdir',
                'tarfile.open', 'pd.read', 'json.load', 'pickle.load', 'np.load',
                'cv2.imread', 'PIL.Image.open', 'torch.load', 'joblib.load'
            ]):
                # Check if this path matches a contract input
                for input_spec in contract_inputs.values():
                    if 'path' in input_spec and normalize_path(input_spec['path']) == normalized_path:
                        script_reads.add(normalized_path)
                        break
            
            if any(keyword in context for keyword in [
                'write', 'save', 'dump', 'create', 'mkdir', 'copy', 'move',
                'tarfile.open', 'pd.to_', 'json.dump', 'pickle.dump', 'np.save',
                'cv2.imwrite', 'torch.save', 'joblib.dump'
            ]):
                # Check if this path matches a contract output
                for output_spec in contract_outputs.values():
                    if 'path' in output_spec and normalize_path(output_spec['path']) == normalized_path:
                        script_writes.add(normalized_path)
                        break
        
        # Additional heuristic: if a path appears in contract inputs/outputs and is referenced in script,
        # assume it's being used for its intended purpose
        for input_spec in contract_inputs.values():
            if 'path' in input_spec:
                contract_path = normalize_path(input_spec['path'])
                for path_ref in path_references:
                    if normalize_path(path_ref.path) == contract_path:
                        script_reads.add(contract_path)
                        break
        
        for output_spec in contract_outputs.values():
            if 'path' in output_spec:
                contract_path = normalize_path(output_spec['path'])
                for path_ref in path_references:
                    if normalize_path(path_ref.path) == contract_path:
                        script_writes.add(contract_path)
                        break
        
        return script_reads, script_writes
    
    def _resolve_logical_name_from_contract(self, path: str, contract: Dict[str, Any]) -> Optional[str]:
        """
        Resolve logical name from contract mappings instead of path parsing.
        
        This fixes the critical issue where logical names were incorrectly extracted
        from path patterns instead of using the actual contract mappings.
        
        Args:
            path: The file path to resolve
            contract: The contract dictionary
            
        Returns:
            Logical name if found in contract, None otherwise
        """
        normalized_path = normalize_path(path)
        
        # Check contract inputs
        for logical_name, input_spec in contract.get('inputs', {}).items():
            if 'path' in input_spec:
                if normalize_path(input_spec['path']) == normalized_path:
                    return logical_name
        
        # Check contract outputs
        for logical_name, output_spec in contract.get('outputs', {}).items():
            if 'path' in output_spec:
                if normalize_path(output_spec['path']) == normalized_path:
                    return logical_name
        
        return None  # Only return None if truly not in contract
