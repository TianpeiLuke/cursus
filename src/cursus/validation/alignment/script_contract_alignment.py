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
from .testability_validator import TestabilityPatternValidator
from .alignment_utils import (
    FlexibleFileResolver, detect_step_type_from_registry, 
    detect_framework_from_imports, create_step_type_aware_alignment_issue,
    SeverityLevel, normalize_path
)
from .loaders import ContractLoader
from .validators import ScriptContractValidator
from .framework_patterns import detect_training_patterns, detect_xgboost_patterns


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
        
        # Initialize testability validator
        self.testability_validator = TestabilityPatternValidator()
        
        # Initialize contract loader and validator
        self.contract_loader = ContractLoader(str(self.contracts_dir))
        self.script_validator = ScriptContractValidator()
        
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
        
        # Get builder arguments if builders directory is available
        builder_args = set()
        if self.builders_dir:
            try:
                builder_args = extract_builder_arguments(script_name, str(self.builders_dir))
            except Exception as e:
                # Log warning but continue validation
                pass
        
        # Validate path usage
        path_issues = self.script_validator.validate_path_usage(analysis, contract, script_name)
        issues.extend(path_issues)
        
        # Validate environment variable usage
        env_issues = self.script_validator.validate_env_var_usage(analysis, contract, script_name)
        issues.extend(env_issues)
        
        # Validate argument usage
        arg_issues = self.script_validator.validate_argument_usage(analysis, contract, script_name, builder_args)
        issues.extend(arg_issues)
        
        # Validate file operations
        file_issues = self.script_validator.validate_file_operations(analysis, contract, script_name)
        issues.extend(file_issues)
        
        # Validate script testability patterns
        try:
            testability_issues = self.testability_validator.validate_script_testability(
                str(script_path), analyzer.ast_tree
            )
            # Convert AlignmentIssue objects to dictionary format for consistency
            for issue in testability_issues:
                issues.append({
                    'severity': issue.level.value,
                    'category': issue.category,
                    'message': issue.message,
                    'details': issue.details,
                    'recommendation': issue.recommendation
                })
        except Exception as e:
            # If testability validation fails, add a warning but don't fail the entire validation
            issues.append({
                'severity': 'WARNING',
                'category': 'testability_validation_error',
                'message': f'Failed to validate script testability: {str(e)}',
                'details': {'script': script_name, 'error': str(e)},
                'recommendation': 'Check script syntax and structure for testability validation'
            })
        
        # Phase 2 Enhancement: Add step type-specific validation
        try:
            step_type_issues = self._enhance_with_step_type_validation(script_name, analysis, contract)
            issues.extend(step_type_issues)
        except Exception as e:
            # Step type enhancement is optional, don't fail validation if it fails
            issues.append({
                'severity': 'WARNING',
                'category': 'step_type_enhancement_error',
                'message': f'Failed to apply step type enhancements: {str(e)}',
                'details': {'script': script_name, 'error': str(e)},
                'recommendation': 'Check step type detection and framework patterns'
            })
        
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
    
    def _enhance_with_step_type_validation(self, script_name: str, analysis: Dict[str, Any], contract: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Phase 2 Enhancement: Add step type-specific validation to existing results.
        
        Args:
            script_name: Name of the script being validated
            analysis: Script analysis results
            contract: Contract dictionary
            
        Returns:
            List of additional validation issues
        """
        additional_issues = []
        
        # Detect step type from registry
        step_type = detect_step_type_from_registry(script_name)
        
        # Detect framework from imports
        framework = None
        if 'imports' in analysis:
            framework = detect_framework_from_imports(analysis['imports'])
        
        # Add step type-specific validation
        if step_type == "Training":
            additional_issues.extend(self._validate_training_specific(script_name, analysis, contract, framework))
        elif step_type == "Processing":
            # Processing validation is already comprehensive, but we can add framework-specific checks
            additional_issues.extend(self._validate_processing_framework_specific(script_name, analysis, contract, framework))
        
        return additional_issues
    
    def _validate_training_specific(self, script_name: str, analysis: Dict[str, Any], contract: Dict[str, Any], framework: Optional[str]) -> List[Dict[str, Any]]:
        """
        Add training-specific validation using existing patterns.
        
        Args:
            script_name: Name of the training script
            analysis: Script analysis results
            contract: Contract dictionary
            framework: Detected framework (xgboost, pytorch, etc.)
            
        Returns:
            List of training-specific validation issues
        """
        issues = []
        
        # Get script content for pattern analysis
        script_path = self.scripts_dir / f"{script_name}.py"
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                script_content = f.read()
        except Exception:
            return issues  # Can't analyze patterns without script content
        
        # Detect training patterns
        training_patterns = detect_training_patterns(script_content)
        
        # Check for training loop patterns
        if not training_patterns.get('training_loop_patterns'):
            issues.append({
                'severity': 'WARNING',
                'category': 'training_pattern_missing',
                'message': 'Training script should contain model training logic',
                'details': {
                    'script': script_name,
                    'step_type': 'Training',
                    'expected_patterns': ['model.fit()', 'xgb.train()', 'training loop']
                },
                'recommendation': 'Add model training logic such as model.fit() or xgb.train()'
            })
        
        # Check for model saving patterns
        if not training_patterns.get('model_saving_patterns'):
            issues.append({
                'severity': 'WARNING',
                'category': 'training_model_saving_missing',
                'message': 'Training script should save model artifacts',
                'details': {
                    'script': script_name,
                    'step_type': 'Training',
                    'expected_paths': ['/opt/ml/model/']
                },
                'recommendation': 'Add model saving to /opt/ml/model/ directory'
            })
        
        # Check for hyperparameter loading patterns
        if not training_patterns.get('hyperparameter_loading_patterns'):
            issues.append({
                'severity': 'INFO',
                'category': 'training_hyperparameter_loading_missing',
                'message': 'Training script should load hyperparameters from file',
                'details': {
                    'script': script_name,
                    'step_type': 'Training',
                    'expected_paths': ['/opt/ml/input/data/config/']
                },
                'recommendation': 'Add hyperparameter loading from /opt/ml/input/data/config/'
            })
        
        # Framework-specific validation
        if framework == 'xgboost':
            xgb_issues = self._validate_xgboost_training_patterns(script_name, script_content)
            issues.extend(xgb_issues)
        
        return issues
    
    def _validate_xgboost_training_patterns(self, script_name: str, script_content: str) -> List[Dict[str, Any]]:
        """
        Validate XGBoost-specific training patterns.
        
        Args:
            script_name: Name of the script
            script_content: Content of the script
            
        Returns:
            List of XGBoost-specific validation issues
        """
        issues = []
        
        # Detect XGBoost patterns
        xgb_patterns = detect_xgboost_patterns(script_content)
        
        # Check for XGBoost imports
        if not xgb_patterns.get('xgboost_imports'):
            issues.append({
                'severity': 'ERROR',
                'category': 'xgboost_import_missing',
                'message': 'XGBoost training script should import xgboost',
                'details': {
                    'script': script_name,
                    'framework': 'xgboost',
                    'expected_imports': ['import xgboost as xgb', 'from xgboost import']
                },
                'recommendation': 'Add XGBoost import: import xgboost as xgb'
            })
        
        # Check for DMatrix usage
        if not xgb_patterns.get('dmatrix_patterns'):
            issues.append({
                'severity': 'WARNING',
                'category': 'xgboost_dmatrix_missing',
                'message': 'XGBoost training should use DMatrix for data handling',
                'details': {
                    'script': script_name,
                    'framework': 'xgboost',
                    'expected_patterns': ['xgb.DMatrix()', 'xgboost.DMatrix()']
                },
                'recommendation': 'Use xgb.DMatrix() for efficient data handling'
            })
        
        # Check for XGBoost training calls
        if not xgb_patterns.get('xgboost_training'):
            issues.append({
                'severity': 'WARNING',
                'category': 'xgboost_training_missing',
                'message': 'XGBoost training script should call xgb.train() or use XGBoost estimators',
                'details': {
                    'script': script_name,
                    'framework': 'xgboost',
                    'expected_patterns': ['xgb.train()', 'XGBClassifier()', 'XGBRegressor()']
                },
                'recommendation': 'Add XGBoost training call: xgb.train() or use XGBClassifier/XGBRegressor'
            })
        
        return issues
    
    def _validate_processing_framework_specific(self, script_name: str, analysis: Dict[str, Any], contract: Dict[str, Any], framework: Optional[str]) -> List[Dict[str, Any]]:
        """
        Add framework-specific validation for processing scripts.
        
        Args:
            script_name: Name of the processing script
            analysis: Script analysis results
            contract: Contract dictionary
            framework: Detected framework
            
        Returns:
            List of framework-specific validation issues
        """
        issues = []
        
        # For processing scripts, we mainly add informational context
        if framework:
            issues.append({
                'severity': 'INFO',
                'category': 'framework_detected',
                'message': f'Processing script uses {framework} framework',
                'details': {
                    'script': script_name,
                    'step_type': 'Processing',
                    'framework': framework
                },
                'recommendation': f'Ensure {framework} dependencies are properly specified'
            })
        
        return issues

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
