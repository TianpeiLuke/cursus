"""
Contract-Focused Script Analyzer

Analyzes Python scripts for contract alignment validation.
Focuses on main function signature and parameter usage patterns.

Based on analysis of actual scripts:
- currency_conversion.py
- xgboost_training.py
"""

import ast
from typing import Dict, List, Any, Optional
from pathlib import Path


class ScriptAnalyzer:
    """
    Contract alignment focused script analyzer.
    
    Validates:
    - Main function signature compliance
    - Parameter usage patterns (input_paths, output_paths, environ_vars, job_args)
    - Contract alignment validation
    """
    
    def __init__(self, script_path: str):
        self.script_path = script_path
        self.script_content = self._read_script()
        self.ast_tree = self._parse_script()
    
    def _read_script(self) -> str:
        """Read script content from file."""
        with open(self.script_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _parse_script(self) -> ast.AST:
        """Parse script content into AST."""
        return ast.parse(self.script_content)
    
    def validate_main_function_signature(self) -> Dict[str, Any]:
        """
        Validate main function has correct signature.
        
        Expected signature:
        def main(input_paths: Dict[str, str], output_paths: Dict[str, str], 
                 environ_vars: Dict[str, str], job_args: argparse.Namespace) -> Any
        """
        main_function = self._find_main_function()
        if not main_function:
            return {
                "has_main": False,
                "issues": ["No main function found"],
                "signature_valid": False
            }
        
        # Check parameter names and types
        expected_params = ["input_paths", "output_paths", "environ_vars", "job_args"]
        actual_params = self._extract_function_parameters(main_function)
        
        signature_valid = self._validate_signature(expected_params, actual_params)
        issues = self._get_signature_issues(expected_params, actual_params)
        
        return {
            "has_main": True,
            "signature_valid": signature_valid,
            "actual_params": actual_params,
            "expected_params": expected_params,
            "issues": issues
        }
    
    def extract_parameter_usage(self) -> Dict[str, List[str]]:
        """
        Extract how script uses main function parameters.
        
        Returns:
            Dictionary with parameter usage patterns:
            - input_paths_keys: Keys used in input_paths["key"] or input_paths.get("key")
            - output_paths_keys: Keys used in output_paths["key"] or output_paths.get("key")
            - environ_vars_keys: Keys used in environ_vars.get("key")
            - job_args_attrs: Attributes used in job_args.attribute
        """
        main_function = self._find_main_function()
        if not main_function:
            return {
                "input_paths_keys": [],
                "output_paths_keys": [],
                "environ_vars_keys": [],
                "job_args_attrs": []
            }
        
        return {
            "input_paths_keys": self._find_parameter_usage(main_function, "input_paths"),
            "output_paths_keys": self._find_parameter_usage(main_function, "output_paths"),
            "environ_vars_keys": self._find_parameter_usage(main_function, "environ_vars"),
            "job_args_attrs": self._find_parameter_usage(main_function, "job_args")
        }
    
    def validate_contract_alignment(self, contract: Dict) -> List[Dict]:
        """
        Validate script usage aligns with contract declarations.
        
        Args:
            contract: Contract dictionary with expected_input_paths, expected_output_paths, etc.
            
        Returns:
            List of validation issues
        """
        issues = []
        parameter_usage = self.extract_parameter_usage()
        
        # Validate input paths alignment
        script_input_keys = parameter_usage.get("input_paths_keys", [])
        contract_input_keys = list(contract.get("expected_input_paths", {}).keys())
        
        for key in script_input_keys:
            if key not in contract_input_keys:
                issues.append({
                    "severity": "ERROR",
                    "category": "undeclared_input_path",
                    "message": f"Script uses input_paths['{key}'] but contract doesn't declare it",
                    "recommendation": f"Add '{key}' to contract expected_input_paths"
                })
        
        # Validate output paths alignment
        script_output_keys = parameter_usage.get("output_paths_keys", [])
        contract_output_keys = list(contract.get("expected_output_paths", {}).keys())
        
        for key in script_output_keys:
            if key not in contract_output_keys:
                issues.append({
                    "severity": "ERROR",
                    "category": "undeclared_output_path",
                    "message": f"Script uses output_paths['{key}'] but contract doesn't declare it",
                    "recommendation": f"Add '{key}' to contract expected_output_paths"
                })
        
        # Validate environment variables alignment
        script_env_keys = parameter_usage.get("environ_vars_keys", [])
        contract_required_env = contract.get("required_env_vars", [])
        contract_optional_env = list(contract.get("optional_env_vars", {}).keys())
        contract_all_env = contract_required_env + contract_optional_env
        
        for key in script_env_keys:
            if key not in contract_all_env:
                issues.append({
                    "severity": "WARNING",
                    "category": "undeclared_env_var",
                    "message": f"Script uses environ_vars.get('{key}') but contract doesn't declare it",
                    "recommendation": f"Add '{key}' to contract required_env_vars or optional_env_vars"
                })
        
        # Validate job arguments alignment
        script_job_attrs = parameter_usage.get("job_args_attrs", [])
        contract_args = list(contract.get("expected_arguments", {}).keys())
        
        for attr in script_job_attrs:
            # Convert job_args.attr to --attr format for comparison
            arg_name = attr.replace('_', '-')
            if arg_name not in contract_args:
                issues.append({
                    "severity": "WARNING",
                    "category": "undeclared_job_arg",
                    "message": f"Script uses job_args.{attr} but contract doesn't declare --{arg_name}",
                    "recommendation": f"Add '--{arg_name}' to contract expected_arguments"
                })
        
        return issues
    
    def _find_main_function(self) -> Optional[ast.FunctionDef]:
        """Find main function in AST."""
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.FunctionDef) and node.name == "main":
                return node
        return None
    
    def _extract_function_parameters(self, func_node: ast.FunctionDef) -> List[str]:
        """Extract parameter names from function definition."""
        return [arg.arg for arg in func_node.args.args]
    
    def _validate_signature(self, expected: List[str], actual: List[str]) -> bool:
        """Validate function signature matches expected parameters."""
        return expected == actual
    
    def _get_signature_issues(self, expected: List[str], actual: List[str]) -> List[str]:
        """Get list of signature validation issues."""
        issues = []
        if len(actual) != len(expected):
            issues.append(f"Expected {len(expected)} parameters, got {len(actual)}")
        
        for i, (exp, act) in enumerate(zip(expected, actual)):
            if exp != act:
                issues.append(f"Parameter {i+1}: expected '{exp}', got '{act}'")
        
        return issues
    
    def _find_parameter_usage(self, func_node: ast.FunctionDef, param_name: str) -> List[str]:
        """Find usage patterns for a specific parameter."""
        usage_keys = []
        
        # First, collect all string literals that might be used as keys
        potential_keys = self._collect_string_literals(func_node)
        
        for node in ast.walk(func_node):
            # Look for param_name["key"] or param_name.get("key") patterns
            if isinstance(node, ast.Subscript):
                if (isinstance(node.value, ast.Name) and 
                    node.value.id == param_name):
                    # Handle direct string literals (modernized for Python 3.8+)
                    if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                        key = node.slice.value
                        if key not in usage_keys:
                            usage_keys.append(key)
                    # Handle variable subscripts - check if we can find the variable's value
                    elif isinstance(node.slice, ast.Name):
                        # Look for patterns like: for key in ["train", "validation"]: ... param_name[key]
                        var_name = node.slice.id
                        keys_from_loops = self._find_keys_from_loops(func_node, var_name, potential_keys)
                        for key in keys_from_loops:
                            if key not in usage_keys:
                                usage_keys.append(key)
            
            elif isinstance(node, ast.Call):
                # Look for param_name.get("key") patterns
                if (isinstance(node.func, ast.Attribute) and
                    isinstance(node.func.value, ast.Name) and
                    node.func.value.id == param_name and
                    node.func.attr == "get" and
                    node.args and
                    isinstance(node.args[0], ast.Constant) and
                    isinstance(node.args[0].value, str)):
                    key = node.args[0].value
                    if key not in usage_keys:
                        usage_keys.append(key)
            
            elif isinstance(node, ast.Attribute):
                # Look for job_args.attribute patterns
                if (param_name == "job_args" and
                    isinstance(node.value, ast.Name) and
                    node.value.id == param_name and
                    node.attr not in usage_keys):
                    usage_keys.append(node.attr)
        
        return usage_keys
    
    def _collect_string_literals(self, func_node: ast.FunctionDef) -> List[str]:
        """Collect all string literals in the function that could be used as keys."""
        string_literals = []
        
        for node in ast.walk(func_node):
            # Only use ast.Constant for Python 3.8+ compatibility
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                string_literals.append(node.value)
        
        return string_literals
    
    def _find_keys_from_loops(self, func_node: ast.FunctionDef, var_name: str, potential_keys: List[str]) -> List[str]:
        """Find keys that might be assigned to a variable in loops or assignments."""
        keys = []
        
        for node in ast.walk(func_node):
            # Look for: for var_name in ["key1", "key2", ...]:
            if isinstance(node, ast.For):
                if (isinstance(node.target, ast.Name) and 
                    node.target.id == var_name and
                    isinstance(node.iter, (ast.List, ast.Tuple))):
                    for elt in node.iter.elts:
                        # Only use ast.Constant for Python 3.8+ compatibility
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            keys.append(elt.value)
            
            # Look for: var_name = "key" or similar assignments
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if (isinstance(target, ast.Name) and 
                        target.id == var_name):
                        # Only use ast.Constant for Python 3.8+ compatibility
                        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                            keys.append(node.value.value)
        
        return keys
