"""
Builder ↔ Configuration Alignment Tester

Validates alignment between step builders and their configuration requirements.
Ensures builders properly handle configuration fields and validation.
"""

import sys
import ast
import importlib.util
from typing import Dict, List, Any, Optional
from pathlib import Path

from .analyzers import ConfigurationAnalyzer, BuilderCodeAnalyzer
from .patterns import PatternRecognizer, HybridFileResolver
from .alignment_utils import FlexibleFileResolver
from ...registry.step_names import STEP_NAMES, get_step_name_from_spec_type


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
        
        # Initialize base directories for all resolvers
        base_directories = {
            'contracts': str(self.builders_dir.parent / 'contracts'),
            'specs': str(self.builders_dir.parent / 'specs'),
            'builders': str(self.builders_dir),
            'configs': str(self.configs_dir)
        }
        
        # Initialize extracted components
        self.config_analyzer = ConfigurationAnalyzer(str(self.configs_dir))
        self.builder_analyzer = BuilderCodeAnalyzer()
        self.pattern_recognizer = PatternRecognizer()
        self.file_resolver = HybridFileResolver(base_directories)
        
        # Keep FlexibleFileResolver for backward compatibility
        self.flexible_resolver = FlexibleFileResolver(base_directories)
        
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
            available_files_report = self.flexible_resolver.get_available_files_report()
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
        
        # Load configuration using extracted component
        try:
            config_analysis = self.config_analyzer.load_config_from_python(config_path, builder_name)
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
        
        # Analyze builder code using extracted component
        try:
            builder_analysis = self.builder_analyzer.analyze_builder_file(builder_path)
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
        
        # Get configuration fields from analysis (now includes inherited fields)
        config_fields = set(config_analysis.get('fields', {}).keys())
        # Handle both list and set types for required_fields
        required_fields_raw = config_analysis.get('required_fields', [])
        if isinstance(required_fields_raw, list):
            required_fields = set(required_fields_raw)
        else:
            required_fields = set(required_fields_raw)
        
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
        
        Uses the extracted PatternRecognizer component for consistent pattern recognition.
        
        Args:
            field_name: Name of the configuration field
            builder_name: Name of the builder
            issue_type: Type of issue ('undeclared_access', 'unaccessed_required')
            
        Returns:
            True if this is an acceptable pattern (should be filtered out)
        """
        return self.pattern_recognizer.is_acceptable_pattern(field_name, builder_name, issue_type)

    def _validate_required_fields(self, builder_analysis: Dict[str, Any], config_analysis: Dict[str, Any], builder_name: str) -> List[Dict[str, Any]]:
        """Validate that builder properly validates required fields."""
        issues = []
        
        required_fields = set(config_analysis.get('required_fields', []))
        
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
