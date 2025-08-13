"""
Script Testability Pattern Validator

Validates that scripts follow the testability refactoring pattern as outlined
in the Script Testability Implementation Guide. This ensures scripts can be
tested both locally and in containers by separating execution environment
concerns from core functionality.
"""

import ast
import os
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path

from .alignment_utils import (
    SeverityLevel, AlignmentIssue, create_alignment_issue, AlignmentLevel
)


class TestabilityPatternValidator:
    """
    Validates script compliance with testability refactoring patterns.
    
    Checks for:
    - Main function signature with testability parameters
    - Environment access patterns (parameter-based vs direct)
    - Entry point structure and environment collection
    - Helper function compliance with parameter passing
    """
    
    def __init__(self):
        """Initialize the testability pattern validator."""
        self.testability_parameters = {
            'input_paths', 'output_paths', 'environ_vars', 'job_args'
        }
        
        # Allow flexible parameter names for job arguments
        self.job_args_aliases = {'job_args', 'args', 'arguments', 'parsed_args'}
        
        # Patterns that indicate direct environment access (anti-pattern)
        self.direct_env_patterns = {
            'os.environ',
            'os.getenv',
            'environ.get'
        }
        
        # Acceptable environment access patterns in entry points
        self.entry_point_env_patterns = {
            'os.environ.get',
            'os.getenv'
        }
    
    def validate_script_testability(self, script_path: str, ast_tree: ast.AST) -> List[AlignmentIssue]:
        """
        Validate script testability compliance.
        
        Args:
            script_path: Path to the script being validated
            ast_tree: Parsed AST of the script
            
        Returns:
            List of testability validation issues
        """
        issues = []
        script_name = Path(script_path).stem
        
        # Extract script structure information
        script_info = self._analyze_script_structure(ast_tree)
        
        # Validate main function signature
        main_issues = self._validate_main_function_signature(script_info, script_name)
        issues.extend(main_issues)
        
        # Validate environment access patterns
        env_issues = self._validate_environment_access_pattern(script_info, script_name)
        issues.extend(env_issues)
        
        # Validate parameter usage patterns
        param_issues = self._validate_parameter_usage(script_info, script_name)
        issues.extend(param_issues)
        
        # Validate entry point structure
        entry_issues = self._validate_entry_point_structure(script_info, script_name)
        issues.extend(entry_issues)
        
        # Validate helper function compliance
        helper_issues = self._validate_helper_function_compliance(script_info, script_name)
        issues.extend(helper_issues)
        
        return issues
    
    def _analyze_script_structure(self, ast_tree: ast.AST) -> Dict[str, Any]:
        """
        Analyze script structure to extract testability-relevant information.
        
        Args:
            ast_tree: Parsed AST of the script
            
        Returns:
            Dictionary containing script structure information
        """
        analyzer = TestabilityStructureAnalyzer()
        analyzer.visit(ast_tree)
        
        return {
            'main_function': analyzer.main_function,
            'main_block': analyzer.main_block,
            'functions': analyzer.functions,
            'env_accesses': analyzer.env_accesses,
            'direct_path_accesses': analyzer.direct_path_accesses,
            'parameter_usages': analyzer.parameter_usages,
            'has_container_detection': analyzer.has_container_detection,
            'has_testability_structure': analyzer.has_testability_structure
        }
    
    def _validate_main_function_signature(self, script_info: Dict[str, Any], script_name: str) -> List[AlignmentIssue]:
        """
        Validate that main function has proper testability signature.
        
        Args:
            script_info: Script structure information
            script_name: Name of the script
            
        Returns:
            List of validation issues
        """
        issues = []
        main_function = script_info.get('main_function')
        
        if not main_function:
            issues.append(create_alignment_issue(
                level=SeverityLevel.WARNING,
                category="testability_main_function",
                message="No main function found - consider adding one for better testability",
                details={'script': script_name},
                recommendation="Add a main function that accepts testability parameters: main(input_paths, output_paths, environ_vars, job_args)",
                alignment_level=AlignmentLevel.SCRIPT_CONTRACT
            ))
            return issues
        
        # Check if main function has testability parameters (with flexible job_args matching)
        main_params = set(main_function.get('parameters', []))
        
        # Create a flexible version of testability parameters that accepts job_args aliases
        flexible_testability_params = {'input_paths', 'output_paths', 'environ_vars'}
        has_job_args = bool(main_params & self.job_args_aliases)
        
        if has_job_args:
            flexible_testability_params.add('job_args')  # Use canonical name for comparison
        
        missing_params = flexible_testability_params - main_params
        if has_job_args and 'job_args' in missing_params:
            missing_params.remove('job_args')  # Remove if we have an alias
        
        if missing_params:
            if len(missing_params) == len(self.testability_parameters):
                # No testability parameters at all
                issues.append(create_alignment_issue(
                    level=SeverityLevel.ERROR,
                    category="testability_main_signature",
                    message="Main function does not follow testability pattern - missing all testability parameters",
                    details={
                        'script': script_name,
                        'current_parameters': list(main_params),
                        'required_parameters': list(self.testability_parameters)
                    },
                    recommendation="Refactor main function to accept: main(input_paths, output_paths, environ_vars, job_args)",
                    alignment_level=AlignmentLevel.SCRIPT_CONTRACT
                ))
            else:
                # Partial testability parameters
                issues.append(create_alignment_issue(
                    level=SeverityLevel.WARNING,
                    category="testability_main_signature",
                    message=f"Main function missing testability parameters: {', '.join(missing_params)}",
                    details={
                        'script': script_name,
                        'missing_parameters': list(missing_params),
                        'current_parameters': list(main_params)
                    },
                    recommendation=f"Add missing parameters to main function: {', '.join(missing_params)}",
                    alignment_level=AlignmentLevel.SCRIPT_CONTRACT
                ))
        else:
            # All testability parameters present - this is good!
            issues.append(create_alignment_issue(
                level=SeverityLevel.INFO,
                category="testability_compliance",
                message="Main function follows testability pattern with all required parameters",
                details={
                    'script': script_name,
                    'testability_parameters': list(main_params & self.testability_parameters)
                },
                recommendation="No action needed - script follows testability best practices",
                alignment_level=AlignmentLevel.SCRIPT_CONTRACT
            ))
        
        return issues
    
    def _validate_environment_access_pattern(self, script_info: Dict[str, Any], script_name: str) -> List[AlignmentIssue]:
        """
        Validate environment variable access patterns.
        
        Args:
            script_info: Script structure information
            script_name: Name of the script
            
        Returns:
            List of validation issues
        """
        issues = []
        env_accesses = script_info.get('env_accesses', [])
        main_function = script_info.get('main_function')
        main_block = script_info.get('main_block')
        
        # Check for direct environment access in main function
        main_env_accesses = [
            access for access in env_accesses 
            if access.get('in_main_function', False)
        ]
        
        if main_env_accesses and main_function and main_function.get('has_testability_params', False):
            issues.append(create_alignment_issue(
                level=SeverityLevel.ERROR,
                category="testability_env_access",
                message="Main function uses direct environment access despite having testability parameters",
                details={
                    'script': script_name,
                    'direct_accesses': [access['variable'] for access in main_env_accesses],
                    'line_numbers': [access['line_number'] for access in main_env_accesses]
                },
                recommendation="Use environ_vars parameter instead of direct os.environ access in main function",
                alignment_level=AlignmentLevel.SCRIPT_CONTRACT
            ))
        
        # Check for direct environment access in helper functions
        helper_env_accesses = [
            access for access in env_accesses 
            if not access.get('in_main_function', False) and not access.get('in_main_block', False)
        ]
        
        if helper_env_accesses:
            issues.append(create_alignment_issue(
                level=SeverityLevel.WARNING,
                category="testability_env_access",
                message="Helper functions use direct environment access - consider parameter passing",
                details={
                    'script': script_name,
                    'helper_accesses': [
                        {
                            'function': access.get('function_name', 'unknown'),
                            'variable': access['variable'],
                            'line_number': access['line_number']
                        }
                        for access in helper_env_accesses
                    ]
                },
                recommendation="Pass environment variables as parameters to helper functions instead of direct access",
                alignment_level=AlignmentLevel.SCRIPT_CONTRACT
            ))
        
        # Check for proper environment collection in entry point
        entry_env_accesses = [
            access for access in env_accesses 
            if access.get('in_main_block', False)
        ]
        
        if not entry_env_accesses and main_function and main_function.get('has_testability_params', False):
            issues.append(create_alignment_issue(
                level=SeverityLevel.WARNING,
                category="testability_entry_point",
                message="Main function expects environ_vars parameter but no environment collection found in entry point",
                details={'script': script_name},
                recommendation="Add environment variable collection in __main__ block to pass to main function",
                alignment_level=AlignmentLevel.SCRIPT_CONTRACT
            ))
        
        return issues
    
    def _validate_parameter_usage(self, script_info: Dict[str, Any], script_name: str) -> List[AlignmentIssue]:
        """
        Validate parameter usage patterns in the script.
        
        Args:
            script_info: Script structure information
            script_name: Name of the script
            
        Returns:
            List of validation issues
        """
        issues = []
        parameter_usages = script_info.get('parameter_usages', [])
        main_function = script_info.get('main_function')
        
        if not main_function or not main_function.get('has_testability_params', False):
            return issues  # Skip if no testability parameters
        
        # Check if testability parameters are actually used
        used_params = set()
        for usage in parameter_usages:
            if usage.get('parameter') in self.testability_parameters:
                used_params.add(usage['parameter'])
        
        unused_params = self.testability_parameters - used_params
        if unused_params:
            issues.append(create_alignment_issue(
                level=SeverityLevel.WARNING,
                category="testability_parameter_usage",
                message=f"Testability parameters defined but not used: {', '.join(unused_params)}",
                details={
                    'script': script_name,
                    'unused_parameters': list(unused_params),
                    'used_parameters': list(used_params)
                },
                recommendation="Either use the testability parameters or remove them from function signature",
                alignment_level=AlignmentLevel.SCRIPT_CONTRACT
            ))
        
        # Check for proper parameter access patterns
        for usage in parameter_usages:
            if usage.get('parameter') in self.testability_parameters:
                access_pattern = usage.get('access_pattern')
                if access_pattern and not self._is_valid_parameter_access(access_pattern):
                    issues.append(create_alignment_issue(
                        level=SeverityLevel.INFO,
                        category="testability_parameter_access",
                        message=f"Consider using dictionary-style access for {usage['parameter']}",
                        details={
                            'script': script_name,
                            'parameter': usage['parameter'],
                            'current_pattern': access_pattern,
                            'line_number': usage.get('line_number')
                        },
                        recommendation=f"Use {usage['parameter']}['key'] for accessing nested values",
                        alignment_level=AlignmentLevel.SCRIPT_CONTRACT
                    ))
        
        return issues
    
    def _validate_entry_point_structure(self, script_info: Dict[str, Any], script_name: str) -> List[AlignmentIssue]:
        """
        Validate entry point structure for testability compliance.
        
        Args:
            script_info: Script structure information
            script_name: Name of the script
            
        Returns:
            List of validation issues
        """
        issues = []
        main_block = script_info.get('main_block')
        main_function = script_info.get('main_function')
        
        if not main_block:
            if main_function:
                issues.append(create_alignment_issue(
                    level=SeverityLevel.WARNING,
                    category="testability_entry_point",
                    message="Main function exists but no __main__ block found",
                    details={'script': script_name},
                    recommendation="Add if __name__ == '__main__': block to create proper entry point",
                    alignment_level=AlignmentLevel.SCRIPT_CONTRACT
                ))
            return issues
        
        # Check if main block calls main function
        if main_function and not main_block.get('calls_main_function', False):
            issues.append(create_alignment_issue(
                level=SeverityLevel.ERROR,
                category="testability_entry_point",
                message="Main block exists but does not call main function",
                details={'script': script_name},
                recommendation="Call main function from __main__ block with collected parameters",
                alignment_level=AlignmentLevel.SCRIPT_CONTRACT
            ))
        
        # Check for proper parameter collection in main block
        if main_function and main_function.get('has_testability_params', False):
            collected_params = set(main_block.get('collected_parameters', []))
            missing_collection = self.testability_parameters - collected_params
            
            if missing_collection:
                issues.append(create_alignment_issue(
                    level=SeverityLevel.WARNING,
                    category="testability_entry_point",
                    message=f"Main block missing parameter collection for: {', '.join(missing_collection)}",
                    details={
                        'script': script_name,
                        'missing_collection': list(missing_collection),
                        'collected_parameters': list(collected_params)
                    },
                    recommendation="Collect all testability parameters in __main__ block before calling main function",
                    alignment_level=AlignmentLevel.SCRIPT_CONTRACT
                ))
        
        # Check for container detection (optional but good practice)
        has_container_detection = script_info.get('has_container_detection', False)
        if not has_container_detection:
            issues.append(create_alignment_issue(
                level=SeverityLevel.INFO,
                category="testability_container_support",
                message="No container detection found - consider adding hybrid mode support",
                details={'script': script_name},
                recommendation="Add container detection to support both local and container execution",
                alignment_level=AlignmentLevel.SCRIPT_CONTRACT
            ))
        
        return issues
    
    def _validate_helper_function_compliance(self, script_info: Dict[str, Any], script_name: str) -> List[AlignmentIssue]:
        """
        Validate helper function compliance with testability patterns.
        
        Args:
            script_info: Script structure information
            script_name: Name of the script
            
        Returns:
            List of validation issues
        """
        issues = []
        functions = script_info.get('functions', [])
        env_accesses = script_info.get('env_accesses', [])
        
        # Find helper functions that access environment directly
        helper_functions_with_env = {}
        for access in env_accesses:
            if not access.get('in_main_function', False) and not access.get('in_main_block', False):
                func_name = access.get('function_name', 'unknown')
                if func_name not in helper_functions_with_env:
                    helper_functions_with_env[func_name] = []
                helper_functions_with_env[func_name].append(access)
        
        for func_name, accesses in helper_functions_with_env.items():
            if func_name != 'main':  # Skip main function (handled separately)
                issues.append(create_alignment_issue(
                    level=SeverityLevel.WARNING,
                    category="testability_helper_functions",
                    message=f"Helper function '{func_name}' accesses environment directly",
                    details={
                        'script': script_name,
                        'function': func_name,
                        'env_variables': [access['variable'] for access in accesses],
                        'line_numbers': [access['line_number'] for access in accesses]
                    },
                    recommendation=f"Refactor '{func_name}' to accept environment variables as parameters",
                    alignment_level=AlignmentLevel.SCRIPT_CONTRACT
                ))
        
        return issues
    
    def _is_valid_parameter_access(self, access_pattern: str) -> bool:
        """
        Check if parameter access pattern is valid for testability.
        
        Args:
            access_pattern: The access pattern string
            
        Returns:
            True if the access pattern is valid
        """
        # Valid patterns include dictionary access, method calls, etc.
        valid_patterns = [
            '.get(',
            '[',
            '.keys(',
            '.values(',
            '.items('
        ]
        
        return any(pattern in access_pattern for pattern in valid_patterns)


class TestabilityStructureAnalyzer(ast.NodeVisitor):
    """
    AST visitor to analyze script structure for testability patterns.
    """
    
    def __init__(self):
        """Initialize the structure analyzer."""
        self.main_function = None
        self.main_block = None
        self.functions = []
        self.env_accesses = []
        self.direct_path_accesses = []
        self.parameter_usages = []
        self.has_container_detection = False
        self.has_testability_structure = False
        
        # Context tracking
        self.current_function = None
        self.in_main_block = False
        self.function_stack = []
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definitions."""
        self.function_stack.append(self.current_function)
        self.current_function = node.name
        
        function_info = {
            'name': node.name,
            'line_number': node.lineno,
            'parameters': [arg.arg for arg in node.args.args],
            'has_testability_params': False
        }
        
        # Check for testability parameters (with flexible job_args matching)
        testability_params = {'input_paths', 'output_paths', 'environ_vars'}
        job_args_aliases = {'job_args', 'args', 'arguments', 'parsed_args'}
        function_params = set(function_info['parameters'])
        
        has_core_params = testability_params.issubset(function_params)
        has_job_args = bool(function_params & job_args_aliases)
        
        if has_core_params and has_job_args:
            function_info['has_testability_params'] = True
            self.has_testability_structure = True
        
        if node.name == 'main':
            self.main_function = function_info
        
        self.functions.append(function_info)
        
        # Visit function body
        self.generic_visit(node)
        
        self.current_function = self.function_stack.pop()
    
    def visit_If(self, node: ast.If):
        """Visit if statements to detect __main__ blocks."""
        # Check for if __name__ == '__main__':
        if (isinstance(node.test, ast.Compare) and
            isinstance(node.test.left, ast.Name) and
            node.test.left.id == '__name__' and
            len(node.test.ops) == 1 and
            isinstance(node.test.ops[0], ast.Eq) and
            len(node.test.comparators) == 1):
            
            comparator = node.test.comparators[0]
            if isinstance(comparator, (ast.Str, ast.Constant)):
                value = comparator.s if isinstance(comparator, ast.Str) else comparator.value
                if value == '__main__':
                    self.in_main_block = True
                    self.main_block = {
                        'line_number': node.lineno,
                        'calls_main_function': False,
                        'collected_parameters': []
                    }
                    
                    # Analyze main block content
                    self._analyze_main_block(node.body)
                    self.in_main_block = False
        
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call):
        """Visit function calls."""
        # Check for main function calls (flexible detection)
        if self.in_main_block and self.main_block:
            if isinstance(node.func, ast.Name) and node.func.id == 'main':
                self.main_block['calls_main_function'] = True
        
        # Check for container detection patterns
        if self._is_container_detection_call(node):
            self.has_container_detection = True
        
        # Check for environment variable access
        self._check_env_access_call(node)
        
        self.generic_visit(node)
    
    def visit_Subscript(self, node: ast.Subscript):
        """Visit subscript operations (e.g., dict[key])."""
        # Check for os.environ access
        if (isinstance(node.value, ast.Attribute) and
            isinstance(node.value.value, ast.Name) and
            node.value.value.id == 'os' and
            node.value.attr == 'environ'):
            
            var_name = self._extract_string_value(node.slice)
            if var_name:
                self._record_env_access(var_name, node.lineno, 'os.environ')
        
        # Check for parameter usage (including job_args aliases)
        if isinstance(node.value, ast.Name):
            param_name = node.value.id
            testability_params = {'input_paths', 'output_paths', 'environ_vars'}
            job_args_aliases = {'job_args', 'args', 'arguments', 'parsed_args'}
            
            if param_name in testability_params or param_name in job_args_aliases:
                key = self._extract_string_value(node.slice)
                # Normalize job_args aliases to 'job_args' for consistency
                normalized_param = 'job_args' if param_name in job_args_aliases else param_name
                self.parameter_usages.append({
                    'parameter': normalized_param,
                    'original_parameter': param_name,
                    'key': key,
                    'line_number': node.lineno,
                    'access_pattern': f"{param_name}['{key}']" if key else f"{param_name}[...]",
                    'function_name': self.current_function
                })
        
        self.generic_visit(node)
    
    def visit_Attribute(self, node: ast.Attribute):
        """Visit attribute access."""
        # Check for parameter method calls (including job_args aliases)
        if isinstance(node.value, ast.Name):
            param_name = node.value.id
            testability_params = {'input_paths', 'output_paths', 'environ_vars'}
            job_args_aliases = {'job_args', 'args', 'arguments', 'parsed_args'}
            
            if param_name in testability_params or param_name in job_args_aliases:
                # Normalize job_args aliases to 'job_args' for consistency
                normalized_param = 'job_args' if param_name in job_args_aliases else param_name
                self.parameter_usages.append({
                    'parameter': normalized_param,
                    'original_parameter': param_name,
                    'attribute': node.attr,
                    'line_number': node.lineno,
                    'access_pattern': f"{param_name}.{node.attr}",
                    'function_name': self.current_function
                })
        
        self.generic_visit(node)
    
    def _analyze_main_block(self, body: List[ast.stmt]):
        """Analyze the content of the __main__ block."""
        for stmt in body:
            if isinstance(stmt, ast.Assign):
                # Look for parameter collection patterns
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        if var_name in {'input_paths', 'output_paths', 'environ_vars', 'job_args', 'args'}:
                            # Normalize args to job_args for consistency
                            normalized_name = 'job_args' if var_name == 'args' else var_name
                            if normalized_name not in self.main_block['collected_parameters']:
                                self.main_block['collected_parameters'].append(normalized_name)
                
                # Check for main function calls in assignment statements (e.g., result = main(...))
                if isinstance(stmt.value, ast.Call):
                    if isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == 'main':
                        self.main_block['calls_main_function'] = True
                        
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                # Check for main function calls in expression statements
                if isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == 'main':
                    self.main_block['calls_main_function'] = True
                    
            elif isinstance(stmt, ast.Try):
                # Recursively analyze try blocks
                self._analyze_main_block(stmt.body)
                for handler in stmt.handlers:
                    self._analyze_main_block(handler.body)
                if stmt.orelse:
                    self._analyze_main_block(stmt.orelse)
                if stmt.finalbody:
                    self._analyze_main_block(stmt.finalbody)
                    
            elif isinstance(stmt, ast.If):
                # Recursively analyze if blocks
                self._analyze_main_block(stmt.body)
                if stmt.orelse:
                    self._analyze_main_block(stmt.orelse)
                    
            elif isinstance(stmt, ast.For) or isinstance(stmt, ast.While):
                # Recursively analyze loop blocks
                self._analyze_main_block(stmt.body)
                if hasattr(stmt, 'orelse') and stmt.orelse:
                    self._analyze_main_block(stmt.orelse)
                    
            elif isinstance(stmt, ast.With):
                # Recursively analyze with blocks
                self._analyze_main_block(stmt.body)
    
    def _check_env_access_call(self, node: ast.Call):
        """Check for environment variable access in function calls."""
        if isinstance(node.func, ast.Attribute):
            # os.getenv() calls
            if (isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'os' and
                node.func.attr == 'getenv'):
                
                if node.args:
                    var_name = self._extract_string_value(node.args[0])
                    if var_name:
                        self._record_env_access(var_name, node.lineno, 'os.getenv')
            
            # os.environ.get() calls
            elif (isinstance(node.func.value, ast.Attribute) and
                  isinstance(node.func.value.value, ast.Name) and
                  node.func.value.value.id == 'os' and
                  node.func.value.attr == 'environ' and
                  node.func.attr == 'get'):
                
                if node.args:
                    var_name = self._extract_string_value(node.args[0])
                    if var_name:
                        self._record_env_access(var_name, node.lineno, 'os.environ.get')
    
    def _record_env_access(self, var_name: str, line_number: int, method: str):
        """Record environment variable access."""
        self.env_accesses.append({
            'variable': var_name,
            'line_number': line_number,
            'method': method,
            'function_name': self.current_function,
            'in_main_function': self.current_function == 'main',
            'in_main_block': self.in_main_block
        })
    
    def _is_container_detection_call(self, node: ast.Call) -> bool:
        """Check if this is a container detection function call."""
        # Look for common container detection patterns
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in {'is_running_in_container', 'detect_container', 'check_container_mode'}:
                return True
        
        # Look for os.path.exists('/.dockerenv')
        if (isinstance(node.func, ast.Attribute) and
            isinstance(node.func.value, ast.Attribute) and
            isinstance(node.func.value.value, ast.Name) and
            node.func.value.value.id == 'os' and
            node.func.value.attr == 'path' and
            node.func.attr == 'exists'):
            
            if node.args:
                path = self._extract_string_value(node.args[0])
                if path == '/.dockerenv':
                    return True
        
        return False
    
    def _extract_string_value(self, node) -> Optional[str]:
        """Extract string value from AST node."""
        if isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        elif hasattr(node, 'value') and isinstance(node.value, ast.Str):
            return node.value.s
        elif hasattr(node, 'value') and isinstance(node.value, ast.Constant):
            return node.value.value if isinstance(node.value.value, str) else None
        return None
