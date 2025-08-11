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
from .static_analysis.builder_analyzer import extract_builder_arguments
from .alignment_utils import (
    SeverityLevel, create_alignment_issue, normalize_path,
    extract_logical_name_from_path, is_sagemaker_path,
    FlexibleFileResolver
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
    
    def __init__(self, scripts_dir: str, contracts_dir: str, builders_dir: Optional[str] = None):
        """
        Initialize the script-contract alignment tester.
        
        Args:
            scripts_dir: Directory containing processing scripts
            contracts_dir: Directory containing script contracts
            builders_dir: Optional directory containing step builders for enhanced validation
        """
        self.scripts_dir = Path(scripts_dir)
        self.contracts_dir = Path(contracts_dir)
        self.builders_dir = Path(builders_dir) if builders_dir else None
        
        # Initialize FlexibleFileResolver for robust file discovery
        base_directories = {
            'contracts': str(self.contracts_dir),
            'builders': str(self.builders_dir) if self.builders_dir else '',
            'scripts': str(self.scripts_dir)
        }
        self.file_resolver = FlexibleFileResolver(base_directories)
        
        # Build entry_point to contract file mapping (kept as fallback)
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
        
        # Hybrid approach: Try entry_point mapping first, then FlexibleFileResolver as fallback
        contract_file_path = self._find_contract_file_hybrid(script_name)
        
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
        
        if not contract_file_path:
            return {
                'passed': False,
                'issues': [{
                    'severity': 'ERROR',
                    'category': 'missing_contract',
                    'message': f'Contract file not found for script: {script_name}',
                    'details': {
                        'script': script_name,
                        'searched_methods': [
                            'Entry point mapping from contract files',
                            'FlexibleFileResolver pattern matching',
                            f'Naming convention: {script_name}_contract.py'
                        ]
                    },
                    'recommendation': f'Create contract file for {script_name} or check naming patterns'
                }]
            }
        
        contract_path = Path(contract_file_path)
        if not contract_path.exists():
            return {
                'passed': False,
                'issues': [{
                    'severity': 'ERROR',
                    'category': 'missing_contract',
                    'message': f'Contract file not found: {contract_path}',
                    'recommendation': f'Create contract file {contract_path.name}'
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
                        'required': default_value is None
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
        
        # Get builder arguments if builders directory is available
        builder_args = set()
        if self.builders_dir:
            try:
                builder_args = extract_builder_arguments(script_name, str(self.builders_dir))
            except Exception as e:
                # Log warning but continue validation
                pass
        
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
    
    def _find_contract_file_hybrid(self, script_name: str) -> Optional[str]:
        """
        Hybrid approach to find contract file: try entry_point mapping first, then FlexibleFileResolver as fallback.
        
        Args:
            script_name: Name of the script to find contract for
            
        Returns:
            Path to contract file or None if not found
        """
        script_filename = f"{script_name}.py"
        
        # Method 1: Try entry_point mapping (authoritative source)
        if script_filename in self._entry_point_to_contract:
            contract_filename = self._entry_point_to_contract[script_filename]
            contract_path = self.contracts_dir / contract_filename
            if contract_path.exists():
                return str(contract_path)
        
        # Method 2: Try FlexibleFileResolver as fallback (pattern matching)
        flexible_path = self.file_resolver.find_contract_file(script_name)
        if flexible_path and Path(flexible_path).exists():
            return flexible_path
        
        # Method 3: Try naming convention as final fallback
        conventional_path = self.contracts_dir / f"{script_name}_contract.py"
        if conventional_path.exists():
            return str(conventional_path)
        
        return None
    
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
