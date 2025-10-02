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
        
        for node in ast.walk(func_node):
            # Look for param_name["key"] or param_name.get("key") patterns
            if isinstance(node, ast.Subscript):
                if (isinstance(node.value, ast.Name) and 
                    node.value.id == param_name and
                    isinstance(node.slice, (ast.Str, ast.Constant))):
                    key = node.slice.s if isinstance(node.slice, ast.Str) else node.slice.value
                    if isinstance(key, str) and key not in usage_keys:
                        usage_keys.append(key)
            
            elif isinstance(node, ast.Call):
                # Look for param_name.get("key") patterns
                if (isinstance(node.func, ast.Attribute) and
                    isinstance(node.func.value, ast.Name) and
                    node.func.value.id == param_name and
                    node.func.attr == "get" and
                    node.args and
                    isinstance(node.args[0], (ast.Str, ast.Constant))):
                    key = node.args[0].s if isinstance(node.args[0], ast.Str) else node.args[0].value
                    if isinstance(key, str) and key not in usage_keys:
                        usage_keys.append(key)
            
            elif isinstance(node, ast.Attribute):
                # Look for job_args.attribute patterns
                if (param_name == "job_args" and
                    isinstance(node.value, ast.Name) and
                    node.value.id == param_name and
                    node.attr not in usage_keys):
                    usage_keys.append(node.attr)
        
        return usage_keys
