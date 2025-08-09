"""
Builder ↔ Configuration Alignment Tester

Validates alignment between step builders and their configuration requirements.
Ensures builders properly handle configuration fields and validation.
"""

import os
import json
import ast
from typing import Dict, List, Any, Optional, Set
from pathlib import Path


class BuilderConfigurationAlignmentTester:
    """
    Tests alignment between step builders and configuration requirements.
    
    Validates:
    - Configuration fields are properly handled
    - Required fields are validated
    - Default values are consistent
    - Configuration schema matches usage
    """
    
    def __init__(self, builders_dir: str, specs_dir: str):
        """
        Initialize the builder-configuration alignment tester.
        
        Args:
            builders_dir: Directory containing step builders
            specs_dir: Directory containing step specifications
        """
        self.builders_dir = Path(builders_dir)
        self.specs_dir = Path(specs_dir)
    
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
        builder_path = self.builders_dir / f"{builder_name}_builder.py"
        spec_path = self.specs_dir / f"{builder_name}_spec.json"
        
        # Check if files exist
        if not builder_path.exists():
            return {
                'passed': False,
                'issues': [{
                    'severity': 'CRITICAL',
                    'category': 'missing_file',
                    'message': f'Builder file not found: {builder_path}',
                    'recommendation': f'Create the builder file {builder_name}_builder.py'
                }]
            }
        
        if not spec_path.exists():
            return {
                'passed': False,
                'issues': [{
                    'severity': 'ERROR',
                    'category': 'missing_specification',
                    'message': f'Specification file not found: {spec_path}',
                    'recommendation': f'Create specification file {builder_name}_spec.json'
                }]
            }
        
        # Load specification
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
        config_issues = self._validate_configuration_fields(builder_analysis, specification, builder_name)
        issues.extend(config_issues)
        
        # Validate required field validation
        validation_issues = self._validate_required_fields(builder_analysis, specification, builder_name)
        issues.extend(validation_issues)
        
        # Validate default values
        default_issues = self._validate_default_values(builder_analysis, specification, builder_name)
        issues.extend(default_issues)
        
        # Determine overall pass/fail status
        has_critical_or_error = any(
            issue['severity'] in ['CRITICAL', 'ERROR'] for issue in issues
        )
        
        return {
            'passed': not has_critical_or_error,
            'issues': issues,
            'builder_analysis': builder_analysis,
            'specification': specification
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
    
    def _discover_builders(self) -> List[str]:
        """Discover all builder files in the builders directory."""
        builders = []
        
        if self.builders_dir.exists():
            for builder_file in self.builders_dir.glob("*_builder.py"):
                builder_name = builder_file.stem.replace('_builder', '')
                builders.append(builder_name)
        
        return sorted(builders)
