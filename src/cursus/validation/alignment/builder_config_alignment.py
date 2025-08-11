"""
Builder ↔ Configuration Alignment Tester

Validates alignment between step builders and their configuration requirements.
Ensures builders properly handle configuration fields and validation.
"""

import os
import sys
import json
import ast
import importlib.util
from typing import Dict, List, Any, Optional, Set
from pathlib import Path

from .alignment_utils import FlexibleFileResolver
from ...steps.registry.step_names import CONFIG_STEP_REGISTRY, STEP_NAMES, get_step_name_from_spec_type


class BuilderConfigurationAlignmentTester:
    """
    Tests alignment between step builders and configuration requirements.
    
    Validates:
    - Configuration fields are properly handled
    - Required fields are validated
    - Default values are consistent
    - Configuration schema matches usage
    """
    
    def __init__(self, builders_dir: str, configs_dir: str):
        """
        Initialize the builder-configuration alignment tester.
        
        Args:
            builders_dir: Directory containing step builders
            configs_dir: Directory containing step configurations
        """
        self.builders_dir = Path(builders_dir)
        self.configs_dir = Path(configs_dir)
        
        # Initialize the flexible file resolver with ALL required base directories
        # FlexibleFileResolver expects 4 base directories: contracts, specs, builders, configs
        base_directories = {
            'contracts': str(self.builders_dir.parent / 'contracts'),
            'specs': str(self.builders_dir.parent / 'specs'),
            'builders': str(self.builders_dir),
            'configs': str(self.configs_dir)
        }
        
        self.file_resolver = FlexibleFileResolver(base_directories)
        
        # Add the project root to Python path for imports
        project_root = self.builders_dir.parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
    
    def validate_all_builders(self, target_scripts: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Validate alignment for all builders or specified target scripts.
        
        Args:
            target_scripts: Specific scripts to validate (None for all)
            
        Returns:
            Dictionary mapping builder names to validation results
        """
        results = {}
        
        # Discover builders to validate
        if target_scripts:
            builders_to_validate = target_scripts
        else:
            builders_to_validate = self._discover_builders()
        
        for builder_name in builders_to_validate:
            try:
                result = self.validate_builder(builder_name)
                results[builder_name] = result
            except Exception as e:
                results[builder_name] = {
                    'passed': False,
                    'error': str(e),
                    'issues': [{
                        'severity': 'CRITICAL',
                        'category': 'validation_error',
                        'message': f'Failed to validate builder {builder_name}: {str(e)}'
                    }]
                }
        
        return results
    
    def validate_builder(self, builder_name: str) -> Dict[str, Any]:
        """
        Validate alignment for a specific builder.
        
        Args:
            builder_name: Name of the builder to validate
            
        Returns:
            Validation result dictionary
        """
        # Use hybrid approach for builder file resolution
        builder_path_str = self._find_builder_file_hybrid(builder_name)
        
        # Check if builder file exists
        if not builder_path_str:
            return {
                'passed': False,
                'issues': [{
                    'severity': 'CRITICAL',
                    'category': 'missing_file',
                    'message': f'Builder file not found for {builder_name}',
                    'details': {
                        'searched_patterns': [
                            f'builder_{builder_name}_step.py',
                            'FlexibleFileResolver patterns',
                            'Fuzzy matching'
                        ],
                        'search_directory': str(self.builders_dir)
                    },
                    'recommendation': f'Create builder file builder_{builder_name}_step.py'
                }]
            }
        
        builder_path = Path(builder_path_str)
        
        # Use hybrid approach for config file resolution
        config_path_str = self._find_config_file_hybrid(builder_name)
        
        # Check if config file exists
        if not config_path_str:
            # Get detailed diagnostics from FlexibleFileResolver
            available_files_report = self.file_resolver.get_available_files_report()
            config_report = available_files_report.get('configs', {})
            
            return {
                'passed': False,
                'issues': [{
                    'severity': 'ERROR',
                    'category': 'missing_configuration',
                    'message': f'Configuration file not found for {builder_name}',
                    'details': {
                        'builder_name': builder_name,
                        'search_directory': str(self.configs_dir),
                        'available_config_files': config_report.get('discovered_files', []),
                        'available_base_names': config_report.get('base_names', []),
                        'total_configs_found': config_report.get('count', 0),
                        'resolver_strategies': [
                            'Exact match',
                            'Normalized matching (preprocess↔preprocessing, eval↔evaluation, xgb↔xgboost)',
                            'Fuzzy matching (80% similarity threshold)'
                        ]
                    },
                    'recommendation': f'Check if config file exists with correct naming pattern, or create config_{builder_name}_step.py'
                }]
            }
        
        config_path = Path(config_path_str)
        
        # Load configuration from Python file
        try:
            config_analysis = self._load_config_from_python(config_path, builder_name)
        except Exception as e:
            return {
                'passed': False,
                'issues': [{
                    'severity': 'CRITICAL',
                    'category': 'config_load_error',
                    'message': f'Failed to load configuration: {str(e)}',
                    'recommendation': 'Fix Python syntax or configuration structure in config file'
                }]
            }
        
        # Analyze builder code
        try:
            with open(builder_path, 'r') as f:
                builder_content = f.read()
            
            builder_ast = ast.parse(builder_content)
            builder_analysis = self._analyze_builder_code(builder_ast, builder_content)
        except Exception as e:
            return {
                'passed': False,
                'issues': [{
                    'severity': 'CRITICAL',
                    'category': 'builder_analysis_error',
                    'message': f'Failed to analyze builder: {str(e)}',
                    'recommendation': 'Fix syntax errors in builder file'
                }]
            }
        
        # Perform alignment validation
        issues = []
        
        # Validate configuration field handling
        config_issues = self._validate_configuration_fields(builder_analysis, config_analysis, builder_name)
        issues.extend(config_issues)
        
        # Validate required field validation
        validation_issues = self._validate_required_fields(builder_analysis, config_analysis, builder_name)
        issues.extend(validation_issues)
        
        # Validate configuration import
        import_issues = self._validate_config_import(builder_analysis, config_analysis, builder_name)
        issues.extend(import_issues)
        
        # Determine overall pass/fail status
        has_critical_or_error = any(
            issue['severity'] in ['CRITICAL', 'ERROR'] for issue in issues
        )
        
        return {
            'passed': not has_critical_or_error,
            'issues': issues,
            'builder_analysis': builder_analysis,
            'config_analysis': config_analysis
        }
    
    def _analyze_builder_code(self, builder_ast: ast.AST, builder_content: str) -> Dict[str, Any]:
        """Analyze builder code to extract configuration usage patterns."""
        analysis = {
            'config_accesses': [],
            'validation_calls': [],
            'default_assignments': [],
            'class_definitions': [],
            'method_definitions': []
        }
        
        class BuilderVisitor(ast.NodeVisitor):
            def visit_Attribute(self, node):
                # Look for config.field_name accesses
                if (isinstance(node.value, ast.Name) and 
                    node.value.id == 'config'):
                    analysis['config_accesses'].append({
                        'field_name': node.attr,
                        'line_number': node.lineno
                    })
                self.generic_visit(node)
            
            def visit_Call(self, node):
                # Look for validation method calls
                if (isinstance(node.func, ast.Attribute) and
                    node.func.attr in ['validate', 'require', 'check']):
                    analysis['validation_calls'].append({
                        'method': node.func.attr,
                        'line_number': node.lineno
                    })
                self.generic_visit(node)
            
            def visit_Assign(self, node):
                # Look for default value assignments
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        analysis['default_assignments'].append({
                            'field_name': target.attr,
                            'line_number': node.lineno
                        })
                self.generic_visit(node)
            
            def visit_ClassDef(self, node):
                analysis['class_definitions'].append({
                    'class_name': node.name,
                    'line_number': node.lineno
                })
                self.generic_visit(node)
            
            def visit_FunctionDef(self, node):
                analysis['method_definitions'].append({
                    'method_name': node.name,
                    'line_number': node.lineno
                })
                self.generic_visit(node)
        
        visitor = BuilderVisitor()
        visitor.visit(builder_ast)
        
        return analysis
    
    def _validate_configuration_fields(self, builder_analysis: Dict[str, Any], specification: Dict[str, Any], builder_name: str) -> List[Dict[str, Any]]:
        """Validate that builder properly handles configuration fields."""
        issues = []
        
        # Get configuration schema from specification
        config_schema = specification.get('configuration', {})
        required_fields = set(config_schema.get('required', []))
        optional_fields = set(config_schema.get('optional', []))
        all_spec_fields = required_fields | optional_fields
        
        # Get fields accessed in builder
        accessed_fields = set()
        for access in builder_analysis.get('config_accesses', []):
            accessed_fields.add(access['field_name'])
        
        # Check for accessed fields not in specification
        undeclared_fields = accessed_fields - all_spec_fields
        for field_name in undeclared_fields:
            issues.append({
                'severity': 'ERROR',
                'category': 'configuration_fields',
                'message': f'Builder accesses undeclared configuration field: {field_name}',
                'details': {'field_name': field_name, 'builder': builder_name},
                'recommendation': f'Add {field_name} to specification configuration schema'
            })
        
        # Check for required fields not accessed
        unaccessed_required = required_fields - accessed_fields
        for field_name in unaccessed_required:
            issues.append({
                'severity': 'WARNING',
                'category': 'configuration_fields',
                'message': f'Required configuration field not accessed in builder: {field_name}',
                'details': {'field_name': field_name, 'builder': builder_name},
                'recommendation': f'Access required field {field_name} in builder or remove from specification'
            })
        
        return issues
    
    def _validate_required_fields(self, builder_analysis: Dict[str, Any], specification: Dict[str, Any], builder_name: str) -> List[Dict[str, Any]]:
        """Validate that builder properly validates required fields."""
        issues = []
        
        config_schema = specification.get('configuration', {})
        required_fields = set(config_schema.get('required', []))
        
        # Check if builder has validation logic
        has_validation = len(builder_analysis.get('validation_calls', [])) > 0
        
        if required_fields and not has_validation:
            issues.append({
                'severity': 'WARNING',
                'category': 'required_field_validation',
                'message': f'Builder has required fields but no validation logic detected',
                'details': {'required_fields': list(required_fields), 'builder': builder_name},
                'recommendation': 'Add validation logic for required configuration fields'
            })
        
        return issues
    
    def _validate_default_values(self, builder_analysis: Dict[str, Any], specification: Dict[str, Any], builder_name: str) -> List[Dict[str, Any]]:
        """Validate that default values are consistent between builder and specification."""
        issues = []
        
        config_schema = specification.get('configuration', {})
        spec_defaults = {}
        
        # Extract default values from specification
        for field_name, field_spec in config_schema.get('fields', {}).items():
            if 'default' in field_spec:
                spec_defaults[field_name] = field_spec['default']
        
        # Get default assignments from builder
        builder_defaults = set()
        for assignment in builder_analysis.get('default_assignments', []):
            builder_defaults.add(assignment['field_name'])
        
        # Check for specification defaults not handled in builder
        for field_name, default_value in spec_defaults.items():
            if field_name not in builder_defaults:
                issues.append({
                    'severity': 'INFO',
                    'category': 'default_values',
                    'message': f'Specification defines default for {field_name} but builder does not set it',
                    'details': {
                        'field_name': field_name,
                        'spec_default': default_value,
                        'builder': builder_name
                    },
                    'recommendation': f'Consider setting default value for {field_name} in builder'
                })
        
        return issues
    
    def _load_config_from_python(self, config_path: Path, builder_name: str) -> Dict[str, Any]:
        """Load configuration from Python file."""
        try:
            # Try to import the module directly
            import sys
            module_name = f"config_{builder_name}_step"
            
            # Add both the configs directory and the project root to sys.path temporarily
            configs_dir_str = str(self.configs_dir)
            project_root_str = str(self.configs_dir.parent.parent.parent)
            
            paths_to_add = []
            if configs_dir_str not in sys.path:
                sys.path.insert(0, configs_dir_str)
                paths_to_add.append(configs_dir_str)
            if project_root_str not in sys.path:
                sys.path.insert(0, project_root_str)
                paths_to_add.append(project_root_str)
            
            try:
                spec = importlib.util.spec_from_file_location(module_name, config_path)
                module = importlib.util.module_from_spec(spec)
                
                # Set up the module's __package__ to help with relative imports
                module.__package__ = 'src.cursus.steps.configs'
                
                sys.modules[module_name] = module  # Add to sys.modules to help with relative imports
                spec.loader.exec_module(module)
            finally:
                # Clean up sys.path
                for path in paths_to_add:
                    if path in sys.path:
                        sys.path.remove(path)
            
            # Look for configuration class (CamelCase name)
            possible_names = [
                f"{builder_name.title().replace('_', '')}Config",
                f"{''.join(word.capitalize() for word in builder_name.split('_'))}Config",
                f"{''.join(word.capitalize() for word in builder_name.split('_'))}StepConfig",  # StepConfig pattern
                f"CurrencyConversionConfig",  # Specific case
                f"DummyTrainingConfig",       # Specific case
                f"BatchTransformStepConfig"   # Specific case
            ]
            
            config_class = None
            config_class_name = None
            
            for name in possible_names:
                if hasattr(module, name):
                    config_class = getattr(module, name)
                    config_class_name = name
                    break
            
            if config_class is None:
                # List all classes in the module for debugging
                classes = [name for name in dir(module) if isinstance(getattr(module, name), type)]
                raise ValueError(f"Configuration class not found in {config_path}. Available classes: {classes}")
            
            # Analyze the configuration class
            return self._analyze_config_class(config_class, config_class_name)
            
        except Exception as e:
            # Return a simplified analysis if we can't load the module
            return {
                'class_name': f"{builder_name}Config",
                'fields': {},
                'required_fields': set(),
                'optional_fields': set(),
                'default_values': {},
                'load_error': str(e)
            }
    
    def _analyze_config_class(self, config_class, class_name: str) -> Dict[str, Any]:
        """Analyze configuration class to extract field information."""
        analysis = {
            'class_name': class_name,
            'fields': {},
            'required_fields': set(),
            'optional_fields': set(),
            'default_values': {}
        }
        
        # Get class annotations (type hints)
        if hasattr(config_class, '__annotations__'):
            for field_name, field_type in config_class.__annotations__.items():
                # Determine if field is optional based on type annotation
                is_optional = self._is_optional_field(field_type, field_name, config_class)
                
                analysis['fields'][field_name] = {
                    'type': str(field_type),
                    'required': not is_optional
                }
                
                if is_optional:
                    analysis['optional_fields'].add(field_name)
                else:
                    analysis['required_fields'].add(field_name)
        
        # Check for default values in class definition and Pydantic Field definitions
        for attr_name in dir(config_class):
            if not attr_name.startswith('_'):
                attr_value = getattr(config_class, attr_name, None)
                if not callable(attr_value):
                    analysis['default_values'][attr_name] = attr_value
                    # If a field has a default value, it's optional
                    if attr_name in analysis['required_fields']:
                        analysis['required_fields'].remove(attr_name)
                        analysis['optional_fields'].add(attr_name)
                        if attr_name in analysis['fields']:
                            analysis['fields'][attr_name]['required'] = False
        
        return analysis
    
    def _is_optional_field(self, field_type, field_name: str, config_class) -> bool:
        """
        Determine if a field is optional based on its type annotation and Field definition.
        
        Args:
            field_type: The type annotation for the field
            field_name: Name of the field
            config_class: The configuration class
            
        Returns:
            True if the field is optional, False if required
        """
        import typing
        
        # First priority: Check Pydantic field info for definitive answer
        try:
            if hasattr(config_class, 'model_fields'):
                # Pydantic v2 style
                field_info = config_class.model_fields.get(field_name)
                if field_info and hasattr(field_info, 'is_required'):
                    # Use Pydantic's own determination of required/optional
                    return not field_info.is_required()
            elif hasattr(config_class, '__fields__'):
                # Pydantic v1 style
                field_info = config_class.__fields__.get(field_name)
                if field_info and hasattr(field_info, 'required'):
                    return not field_info.required
        except Exception:
            # If Pydantic info fails, fall back to other methods
            pass
        
        # Second priority: Check for Optional[Type] or Union[Type, None] patterns in type annotation
        type_str = str(field_type)
        if 'Optional[' in type_str or 'Union[' in type_str:
            # Handle Optional[Type] which is Union[Type, None]
            if hasattr(typing, 'get_origin') and hasattr(typing, 'get_args'):
                origin = typing.get_origin(field_type)
                if origin is typing.Union:
                    args = typing.get_args(field_type)
                    # Check if None is one of the union types
                    if type(None) in args:
                        return True
            # Fallback string-based check
            elif 'NoneType' in type_str or ', None' in type_str or 'None]' in type_str:
                return True
        
        # Third priority: Check if the field has a class-level default value
        if hasattr(config_class, field_name):
            default_value = getattr(config_class, field_name)
            # If it's not a callable (method) and not a Field descriptor, it's a default
            if not callable(default_value):
                return True
        
        # If none of the above, assume required
        return False
    
    def _validate_config_import(self, builder_analysis: Dict[str, Any], config_analysis: Dict[str, Any], builder_name: str) -> List[Dict[str, Any]]:
        """Validate that builder properly imports and uses configuration."""
        issues = []
        
        # Check if builder imports the configuration class
        config_class_name = config_analysis.get('class_name', '')
        
        # Look for import statements in builder (this is a simplified check)
        # In a real implementation, we'd parse import statements from the AST
        has_config_import = any(
            class_def['class_name'] == config_class_name.replace('Config', 'StepBuilder')
            for class_def in builder_analysis.get('class_definitions', [])
        )
        
        if not has_config_import:
            issues.append({
                'severity': 'INFO',
                'category': 'config_import',
                'message': f'Builder may not be properly importing configuration class {config_class_name}',
                'details': {'config_class': config_class_name, 'builder': builder_name},
                'recommendation': f'Ensure builder imports and uses {config_class_name}'
            })
        
        return issues

    def _validate_configuration_fields(self, builder_analysis: Dict[str, Any], config_analysis: Dict[str, Any], builder_name: str) -> List[Dict[str, Any]]:
        """Validate that builder properly handles configuration fields."""
        issues = []
        
        # Get configuration fields from analysis
        config_fields = set(config_analysis.get('fields', {}).keys())
        required_fields = config_analysis.get('required_fields', set())
        
        # Get fields accessed in builder
        accessed_fields = set()
        for access in builder_analysis.get('config_accesses', []):
            accessed_fields.add(access['field_name'])
        
        # Apply pattern-aware filtering to reduce false positives
        filtered_issues = []
        
        # Check for accessed fields not in configuration
        undeclared_fields = accessed_fields - config_fields
        for field_name in undeclared_fields:
            # Apply architectural pattern recognition
            if not self._is_acceptable_pattern(field_name, builder_name, 'undeclared_access'):
                filtered_issues.append({
                    'severity': 'ERROR',
                    'category': 'configuration_fields',
                    'message': f'Builder accesses undeclared configuration field: {field_name}',
                    'details': {'field_name': field_name, 'builder': builder_name},
                    'recommendation': f'Add {field_name} to configuration class or remove from builder'
                })
        
        # Check for required fields not accessed
        unaccessed_required = required_fields - accessed_fields
        for field_name in unaccessed_required:
            # Apply pattern-aware filtering
            if not self._is_acceptable_pattern(field_name, builder_name, 'unaccessed_required'):
                filtered_issues.append({
                    'severity': 'WARNING',
                    'category': 'configuration_fields',
                    'message': f'Required configuration field not accessed in builder: {field_name}',
                    'details': {'field_name': field_name, 'builder': builder_name},
                    'recommendation': f'Access required field {field_name} in builder or make it optional'
                })
        
        return filtered_issues
    
    def _is_acceptable_pattern(self, field_name: str, builder_name: str, issue_type: str) -> bool:
        """
        Determine if a configuration field issue represents an acceptable architectural pattern.
        
        Args:
            field_name: Name of the configuration field
            builder_name: Name of the builder
            issue_type: Type of issue ('undeclared_access', 'unaccessed_required')
            
        Returns:
            True if this is an acceptable pattern (should be filtered out)
        """
        # Common acceptable patterns that should not be flagged as errors
        
        # Pattern 1: Framework-provided fields
        framework_fields = {
            'logger', 'session', 'context', 'environment', 'metadata',
            'step_name', 'step_type', 'execution_id', 'pipeline_id'
        }
        
        if field_name in framework_fields:
            return True
        
        # Pattern 2: Inherited configuration fields
        inherited_patterns = [
            'base_', 'parent_', 'super_', 'common_', 'shared_'
        ]
        
        if any(field_name.startswith(pattern) for pattern in inherited_patterns):
            return True
        
        # Pattern 3: Dynamic configuration fields (runtime-determined)
        dynamic_patterns = [
            'dynamic_', 'runtime_', 'computed_', 'derived_', 'auto_'
        ]
        
        if any(field_name.startswith(pattern) for pattern in dynamic_patterns):
            return True
        
        # Pattern 4: Optional convenience fields that may not be accessed
        if issue_type == 'unaccessed_required':
            convenience_fields = {
                'debug', 'verbose', 'dry_run', 'test_mode', 'profile',
                'cache_enabled', 'parallel_enabled', 'retry_count'
            }
            
            if field_name in convenience_fields:
                return True
        
        # Pattern 5: Builder-specific patterns
        builder_specific_patterns = self._get_builder_specific_patterns(builder_name)
        
        if field_name in builder_specific_patterns.get('acceptable_undeclared', set()):
            return issue_type == 'undeclared_access'
        
        if field_name in builder_specific_patterns.get('acceptable_unaccessed', set()):
            return issue_type == 'unaccessed_required'
        
        # Pattern 6: Configuration fields that follow naming conventions
        if issue_type == 'undeclared_access':
            # Fields that follow standard naming conventions are likely legitimate
            standard_suffixes = ['_config', '_settings', '_params', '_options']
            if any(field_name.endswith(suffix) for suffix in standard_suffixes):
                return True
        
        return False
    
    def _get_builder_specific_patterns(self, builder_name: str) -> Dict[str, Set[str]]:
        """
        Get builder-specific acceptable patterns.
        
        Args:
            builder_name: Name of the builder
            
        Returns:
            Dictionary with 'acceptable_undeclared' and 'acceptable_unaccessed' sets
        """
        patterns = {
            'acceptable_undeclared': set(),
            'acceptable_unaccessed': set()
        }
        
        # Training builders often access framework-provided fields
        if 'training' in builder_name.lower():
            patterns['acceptable_undeclared'].update({
                'model_dir', 'output_dir', 'checkpoint_dir',
                'num_gpus', 'distributed', 'local_rank'
            })
        
        # Processing builders often have optional monitoring fields
        if 'processing' in builder_name.lower() or 'transform' in builder_name.lower():
            patterns['acceptable_unaccessed'].update({
                'monitoring_enabled', 'metrics_enabled', 'profiling_enabled'
            })
        
        # Evaluation builders often have optional comparison fields
        if 'evaluation' in builder_name.lower() or 'validation' in builder_name.lower():
            patterns['acceptable_unaccessed'].update({
                'baseline_model', 'comparison_metrics', 'threshold_config'
            })
        
        # Model builders often access model-specific fields
        if 'model' in builder_name.lower():
            patterns['acceptable_undeclared'].update({
                'model_name', 'model_version', 'model_artifacts'
            })
        
        return patterns

    def _validate_required_fields(self, builder_analysis: Dict[str, Any], config_analysis: Dict[str, Any], builder_name: str) -> List[Dict[str, Any]]:
        """Validate that builder properly validates required fields."""
        issues = []
        
        required_fields = config_analysis.get('required_fields', set())
        
        # Check if builder has validation logic
        has_validation = len(builder_analysis.get('validation_calls', [])) > 0
        
        if required_fields and not has_validation:
            issues.append({
                'severity': 'INFO',
                'category': 'required_field_validation',
                'message': f'Builder has required fields but no explicit validation logic detected',
                'details': {'required_fields': list(required_fields), 'builder': builder_name},
                'recommendation': 'Consider adding explicit validation logic for required configuration fields'
            })
        
        return issues

    def _find_builder_file_hybrid(self, builder_name: str) -> Optional[str]:
        """
        Hybrid builder file resolution with multiple fallback strategies.
        
        Priority:
        1. Standard pattern: builder_{builder_name}_step.py
        2. FlexibleFileResolver patterns (includes fuzzy matching)
        
        Args:
            builder_name: Name of the builder to find
            
        Returns:
            Path to the builder file or None if not found
        """
        # Strategy 1: Try standard naming convention first
        standard_path = self.builders_dir / f"builder_{builder_name}_step.py"
        if standard_path.exists():
            return str(standard_path)
        
        # Strategy 2: Use FlexibleFileResolver for known patterns and fuzzy matching
        flexible_path = self.file_resolver.find_builder_file(builder_name)
        if flexible_path and Path(flexible_path).exists():
            return flexible_path
        
        # Strategy 3: Return None if nothing found
        return None
    
    def _get_canonical_step_name(self, script_name: str) -> str:
        """
        Convert script name to canonical step name using production registry logic.
        
        This uses the same approach as Level-3 validator to ensure consistency
        with the production system's mapping logic.
        
        Args:
            script_name: Script name (e.g., 'mims_package', 'model_evaluation_xgb')
            
        Returns:
            Canonical step name (e.g., 'Package', 'XGBoostModelEval')
        """
        # Convert script name to spec_type format (same as Level-3)
        parts = script_name.split('_')
        
        # Handle job type variants
        job_type_suffixes = ['training', 'validation', 'testing', 'calibration']
        job_type = None
        base_parts = parts
        
        if len(parts) > 1 and parts[-1] in job_type_suffixes:
            job_type = parts[-1]
            base_parts = parts[:-1]
        
        # Convert to PascalCase for spec_type
        spec_type_base = ''.join(word.capitalize() for word in base_parts)
        
        if job_type:
            spec_type = f"{spec_type_base}_{job_type.capitalize()}"
        else:
            spec_type = spec_type_base
        
        # Use production function to get canonical name (strips job type suffix)
        try:
            canonical_name = get_step_name_from_spec_type(spec_type)
            return canonical_name
        except Exception:
            # Fallback: return the base spec_type without job type suffix
            return spec_type_base
    
    def _get_config_name_from_canonical(self, canonical_name: str) -> str:
        """
        Get config file base name from canonical step name using production registry.
        
        Uses the STEP_NAMES registry to find the config class name,
        then derives the config file name from that.
        
        Args:
            canonical_name: Canonical step name (e.g., 'Package', 'XGBoostModelEval')
            
        Returns:
            Config file base name (e.g., 'package', 'model_eval_step_xgboost')
        """
        # Get config class name from STEP_NAMES registry
        if canonical_name in STEP_NAMES:
            config_class = STEP_NAMES[canonical_name]['config_class']
            
            # Convert config class name to file name
            # e.g., 'PackageConfig' -> 'package'
            # e.g., 'XGBoostModelEvalConfig' -> 'model_eval_step_xgboost'
            
            # Remove 'Config' suffix
            if config_class.endswith('Config'):
                base_name = config_class[:-6]  # Remove 'Config'
            else:
                base_name = config_class
            
            # Convert CamelCase to snake_case
            import re
            snake_case = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', base_name)
            snake_case = re.sub('([a-z0-9])([A-Z])', r'\1_\2', snake_case).lower()
            return snake_case
        
        # Fallback if not in registry
        import re
        snake_case = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', canonical_name)
        snake_case = re.sub('([a-z0-9])([A-Z])', r'\1_\2', snake_case).lower()
        return snake_case

    def _find_config_file_hybrid(self, builder_name: str) -> Optional[str]:
        """
        Hybrid config file resolution with production registry integration.
        
        Priority:
        1. Production registry mapping: script_name -> canonical_name -> config_name
        2. Standard pattern: config_{builder_name}_step.py
        3. FlexibleFileResolver patterns (includes fuzzy matching)
        
        Args:
            builder_name: Name of the builder to find config for
            
        Returns:
            Path to the config file or None if not found
        """
        # Strategy 1: Use production registry mapping
        try:
            canonical_name = self._get_canonical_step_name(builder_name)
            config_base_name = self._get_config_name_from_canonical(canonical_name)
            registry_path = self.configs_dir / f"config_{config_base_name}_step.py"
            if registry_path.exists():
                return str(registry_path)
        except Exception:
            # Continue with other strategies if registry mapping fails
            pass
        
        # Strategy 2: Try standard naming convention
        standard_path = self.configs_dir / f"config_{builder_name}_step.py"
        if standard_path.exists():
            return str(standard_path)
        
        # Strategy 3: Use FlexibleFileResolver for known patterns and fuzzy matching
        flexible_path = self.file_resolver.find_config_file(builder_name)
        if flexible_path and Path(flexible_path).exists():
            return flexible_path
        
        # Strategy 4: Return None if nothing found
        return None

    def _discover_builders(self) -> List[str]:
        """Discover all builder files in the builders directory."""
        builders = []
        
        if self.builders_dir.exists():
            for builder_file in self.builders_dir.glob("builder_*_step.py"):
                if not builder_file.name.startswith('__'):
                    # Extract builder name from builder_*_step.py pattern
                    stem = builder_file.stem
                    if stem.startswith('builder_') and stem.endswith('_step'):
                        builder_name = stem[8:-5]  # Remove 'builder_' prefix and '_step' suffix
                        builders.append(builder_name)
        
        return sorted(builders)
