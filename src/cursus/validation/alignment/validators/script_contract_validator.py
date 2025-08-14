"""
Script-Contract Validator Module

Contains the core validation logic for script-contract alignment.
Handles path usage, environment variables, arguments, and file operations validation.
"""

import os
from typing import Dict, Any, List, Optional, Set
from pathlib import Path

from ..alignment_utils import (
    normalize_path, extract_logical_name_from_path, is_sagemaker_path,
    detect_step_type_from_registry, detect_framework_from_imports
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
        """
        Validate that script path usage matches contract declarations.
        
        Enhanced to handle three scenarios:
        1. Contract file path + Script uses file path → Direct match
        2. Contract file path + Script uses directory path → Parent-child relationship  
        3. Contract directory path + Script uses directory path → Direct match
        """
        issues = []
        
        # Get contract paths
        contract_inputs = contract.get('inputs', {})
        contract_outputs = contract.get('outputs', {})
        
        # Extract expected paths from contract with type detection
        contract_file_paths = set()
        contract_dir_paths = set()
        
        for input_spec in contract_inputs.values():
            if 'path' in input_spec:
                path = normalize_path(input_spec['path'])
                if self._is_file_path(path):
                    contract_file_paths.add(path)
                else:
                    contract_dir_paths.add(path)
        
        for output_spec in contract_outputs.values():
            if 'path' in output_spec:
                path = normalize_path(output_spec['path'])
                if self._is_file_path(path):
                    contract_file_paths.add(path)
                else:
                    contract_dir_paths.add(path)
        
        # Get script paths with construction pattern detection
        script_paths = set()
        script_path_constructions = {}  # path -> construction info
        
        for path_ref in analysis.get('path_references', []):
            normalized_path = normalize_path(path_ref.path)
            script_paths.add(normalized_path)
            
            # Detect path construction patterns
            context = getattr(path_ref, 'context', '').lower()
            if 'os.path.join' in context:
                script_path_constructions[normalized_path] = {
                    'method': 'os.path.join',
                    'context': context,
                    'line': getattr(path_ref, 'line_number', None)
                }
        
        # Enhanced validation with three scenarios
        validated_paths = set()
        
        # Scenario 1 & 3: Direct path matching (file-to-file, dir-to-dir)
        direct_matches = script_paths.intersection(contract_file_paths.union(contract_dir_paths))
        validated_paths.update(direct_matches)
        
        # Scenario 2: Parent-child relationship validation (contract file, script directory)
        for contract_file_path in contract_file_paths:
            contract_dir = os.path.dirname(contract_file_path)
            contract_filename = os.path.basename(contract_file_path)
            
            for script_path in script_paths:
                if normalize_path(script_path) == normalize_path(contract_dir):
                    # Script uses parent directory of contract file path
                    # Check if script constructs the file path
                    if self._script_constructs_file_path(analysis, contract_dir, contract_filename):
                        validated_paths.add(script_path)
                        validated_paths.add(contract_file_path)  # Mark contract path as validated too
                        
                        issues.append({
                            'severity': 'INFO',
                            'category': 'path_usage',
                            'message': f'Script correctly uses parent directory to construct file path: {script_path} → {contract_file_path}',
                            'details': {
                                'script_path': script_path,
                                'contract_path': contract_file_path,
                                'construction_method': 'os.path.join',
                                'script': script_name
                            },
                            'recommendation': 'Path usage pattern is correct - no action needed'
                        })
        
        # Check for undeclared paths (not validated by any scenario)
        undeclared_paths = script_paths - validated_paths
        for path in undeclared_paths:
            if is_sagemaker_path(path):
                issues.append({
                    'severity': 'ERROR',
                    'category': 'path_usage',
                    'message': f'Script uses undeclared SageMaker path: {path}',
                    'details': {'path': path, 'script': script_name},
                    'recommendation': f'Add path {path} to contract inputs or outputs'
                })
        
        # Check for unused contract paths (only if not validated by parent-child relationship)
        all_contract_paths = contract_file_paths.union(contract_dir_paths)
        unused_paths = all_contract_paths - validated_paths
        for path in unused_paths:
            issues.append({
                'severity': 'WARNING',
                'category': 'path_usage',
                'message': f'Contract declares path not used in script: {path}',
                'details': {'path': path, 'script': script_name},
                'recommendation': f'Either use path {path} in script or remove from contract'
            })
        
        # Check for logical name consistency using contract mappings
        for path in script_paths:
            if is_sagemaker_path(path) and path in validated_paths:
                logical_name = self._resolve_logical_name_from_contract(path, contract)
                if logical_name is None:
                    # Check if this path is a parent directory of a contract file path
                    parent_logical_name = self._resolve_parent_logical_name_from_contract(path, contract)
                    if parent_logical_name is None:
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
    
    def _is_file_path(self, path: str) -> bool:
        """
        Determine if a path represents a file or directory.
        
        Args:
            path: The path to analyze
            
        Returns:
            True if path appears to be a file, False if directory
        """
        # Check for common file extensions
        file_extensions = [
            '.json', '.csv', '.parquet', '.txt', '.pkl', '.bst', '.jpg', '.png',
            '.py', '.yaml', '.yml', '.xml', '.tar', '.gz', '.zip', '.log'
        ]
        
        normalized_path = normalize_path(path)
        
        # If path has a file extension, it's likely a file
        for ext in file_extensions:
            if normalized_path.lower().endswith(ext):
                return True
        
        # If path ends with a slash, it's definitely a directory
        if normalized_path.endswith('/'):
            return False
        
        # Check for common directory patterns
        directory_patterns = [
            '/opt/ml/input/data',
            '/opt/ml/model',
            '/opt/ml/output/data',
            '/opt/ml/processing/input',
            '/opt/ml/processing/output'
        ]
        
        for pattern in directory_patterns:
            if normalized_path == pattern or normalized_path.startswith(pattern + '/'):
                # If it's exactly a known directory pattern, it's a directory
                if normalized_path == pattern:
                    return False
                # If it extends beyond the pattern, check if it looks like a file
                remainder = normalized_path[len(pattern):].strip('/')
                if '.' in remainder and not '/' in remainder:
                    return True  # Single component with dot - likely a file
        
        # Default heuristic: if it contains a dot in the last component, it's likely a file
        last_component = os.path.basename(normalized_path)
        return '.' in last_component
    
    def _script_constructs_file_path(self, analysis: Dict[str, Any], directory_path: str, filename: str) -> bool:
        """
        Check if the script constructs a file path from directory + filename.
        
        Args:
            analysis: Script analysis results
            directory_path: The directory path used by script
            filename: The filename that should be constructed
            
        Returns:
            True if script constructs the file path
        """
        # Look for os.path.join patterns in the script
        path_references = analysis.get('path_references', [])
        
        for path_ref in path_references:
            context = getattr(path_ref, 'context', '').lower()
            
            # Check for os.path.join with the directory and filename
            if 'os.path.join' in context:
                # Look for patterns like os.path.join(dir, "filename")
                if directory_path.split('/')[-1] in context and filename in context:
                    return True
                
                # Look for patterns where config directory is joined with hyperparameters.json
                if 'config' in context and filename in context:
                    return True
        
        # Also check if the script has logic to handle both file and directory paths
        # This is common in well-written scripts
        for path_ref in path_references:
            context = getattr(path_ref, 'context', '').lower()
            
            # Look for conditional logic that appends filename to directory
            if any(pattern in context for pattern in [
                'endswith', 'if not', 'append', 'join', 'dirname', 'basename'
            ]):
                if filename in context:
                    return True
        
        return False
    
    def _resolve_parent_logical_name_from_contract(self, path: str, contract: Dict[str, Any]) -> Optional[str]:
        """
        Resolve logical name for a path that might be a parent directory of a contract file path.
        
        Args:
            path: The directory path to resolve
            contract: The contract dictionary
            
        Returns:
            Logical name if path is parent of a contract file path, None otherwise
        """
        normalized_path = normalize_path(path)
        
        # Check if this path is a parent directory of any contract file paths
        for logical_name, input_spec in contract.get('inputs', {}).items():
            if 'path' in input_spec:
                contract_path = normalize_path(input_spec['path'])
                contract_dir = os.path.dirname(contract_path)
                
                if normalized_path == contract_dir:
                    return logical_name
        
        for logical_name, output_spec in contract.get('outputs', {}).items():
            if 'path' in output_spec:
                contract_path = normalize_path(output_spec['path'])
                contract_dir = os.path.dirname(contract_path)
                
                if normalized_path == contract_dir:
                    return logical_name
        
        return None
    
    def validate_step_type_specific(self, analysis: Dict[str, Any], contract: Dict[str, Any], script_name: str) -> List[Dict[str, Any]]:
        """
        Phase 2 Enhancement: Add step type-specific validation.
        
        Args:
            analysis: Script analysis results including step type and framework
            contract: Contract dictionary
            script_name: Name of the script
            
        Returns:
            List of step type-specific validation issues
        """
        issues = []
        
        # Get step type and framework from analysis
        step_type = analysis.get('step_type')
        framework = analysis.get('framework')
        step_type_patterns = analysis.get('step_type_patterns', {})
        
        if step_type == "Training":
            issues.extend(self._validate_training_step_specific(analysis, contract, script_name, framework, step_type_patterns))
        elif step_type == "Processing":
            issues.extend(self._validate_processing_step_specific(analysis, contract, script_name, framework))
        
        return issues
    
    def _validate_training_step_specific(self, analysis: Dict[str, Any], contract: Dict[str, Any], script_name: str, framework: Optional[str], patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Validate training-specific requirements.
        
        Args:
            analysis: Script analysis results
            contract: Contract dictionary
            script_name: Name of the training script
            framework: Detected framework
            patterns: Detected training patterns
            
        Returns:
            List of training-specific validation issues
        """
        issues = []
        
        # Check for model output path in contract
        contract_outputs = contract.get('outputs', {})
        has_model_output = any(
            '/opt/ml/model' in output_spec.get('path', '') 
            for output_spec in contract_outputs.values()
        )
        
        if not has_model_output:
            issues.append({
                'severity': 'WARNING',
                'category': 'training_contract_validation',
                'message': 'Training script contract should declare model output path',
                'details': {
                    'script': script_name,
                    'step_type': 'Training',
                    'expected_path': '/opt/ml/model/'
                },
                'recommendation': 'Add model output path (/opt/ml/model/) to contract outputs'
            })
        
        # Check for hyperparameter input path in contract
        contract_inputs = contract.get('inputs', {})
        has_hyperparameter_input = any(
            '/opt/ml/input/data/config' in input_spec.get('path', '') 
            for input_spec in contract_inputs.values()
        )
        
        if not has_hyperparameter_input:
            issues.append({
                'severity': 'INFO',
                'category': 'training_contract_validation',
                'message': 'Training script contract should declare hyperparameter input path',
                'details': {
                    'script': script_name,
                    'step_type': 'Training',
                    'expected_path': '/opt/ml/input/data/config/'
                },
                'recommendation': 'Add hyperparameter input path (/opt/ml/input/data/config/) to contract inputs'
            })
        
        # Framework-specific validation
        if framework == 'xgboost':
            issues.extend(self._validate_xgboost_training_contract(analysis, contract, script_name, patterns))
        
        return issues
    
    def _validate_xgboost_training_contract(self, analysis: Dict[str, Any], contract: Dict[str, Any], script_name: str, patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Validate XGBoost-specific training contract requirements.
        
        Args:
            analysis: Script analysis results
            contract: Contract dictionary
            script_name: Name of the script
            patterns: Detected XGBoost patterns
            
        Returns:
            List of XGBoost-specific validation issues
        """
        issues = []
        
        # Check for XGBoost framework requirements in contract
        framework_requirements = contract.get('framework_requirements', {})
        xgboost_requirements = framework_requirements.get('xgboost', {})
        
        if not xgboost_requirements:
            issues.append({
                'severity': 'INFO',
                'category': 'xgboost_contract_validation',
                'message': 'XGBoost training script should declare framework requirements',
                'details': {
                    'script': script_name,
                    'framework': 'xgboost',
                    'step_type': 'Training'
                },
                'recommendation': 'Add xgboost framework requirements to contract'
            })
        
        # Check for training data input paths
        contract_inputs = contract.get('inputs', {})
        has_training_data = any(
            '/opt/ml/input/data/train' in input_spec.get('path', '') 
            for input_spec in contract_inputs.values()
        )
        
        if not has_training_data:
            issues.append({
                'severity': 'WARNING',
                'category': 'xgboost_contract_validation',
                'message': 'XGBoost training script should declare training data input path',
                'details': {
                    'script': script_name,
                    'framework': 'xgboost',
                    'expected_path': '/opt/ml/input/data/train/'
                },
                'recommendation': 'Add training data input path (/opt/ml/input/data/train/) to contract inputs'
            })
        
        return issues
    
    def _validate_processing_step_specific(self, analysis: Dict[str, Any], contract: Dict[str, Any], script_name: str, framework: Optional[str]) -> List[Dict[str, Any]]:
        """
        Validate processing-specific requirements.
        
        Args:
            analysis: Script analysis results
            contract: Contract dictionary
            script_name: Name of the processing script
            framework: Detected framework
            
        Returns:
            List of processing-specific validation issues
        """
        issues = []
        
        # Processing scripts typically have input/output data paths
        contract_inputs = contract.get('inputs', {})
        contract_outputs = contract.get('outputs', {})
        
        if not contract_inputs:
            issues.append({
                'severity': 'WARNING',
                'category': 'processing_contract_validation',
                'message': 'Processing script should declare input data paths',
                'details': {
                    'script': script_name,
                    'step_type': 'Processing'
                },
                'recommendation': 'Add input data paths to contract inputs'
            })
        
        if not contract_outputs:
            issues.append({
                'severity': 'WARNING',
                'category': 'processing_contract_validation',
                'message': 'Processing script should declare output data paths',
                'details': {
                    'script': script_name,
                    'step_type': 'Processing'
                },
                'recommendation': 'Add output data paths to contract outputs'
            })
        
        # Framework-specific recommendations
        if framework:
            issues.append({
                'severity': 'INFO',
                'category': 'processing_framework_validation',
                'message': f'Processing script uses {framework} framework',
                'details': {
                    'script': script_name,
                    'step_type': 'Processing',
                    'framework': framework
                },
                'recommendation': f'Ensure {framework} dependencies are properly specified in contract'
            })
        
        return issues
