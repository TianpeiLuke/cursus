"""
Builder Analyzer for Step Builder Argument Extraction

Analyzes step builder classes to extract arguments from _get_job_arguments() methods.
This enables validation of config-driven arguments that are provided by builders
but may not be declared in script contracts.
"""

import ast
import re
from typing import Set, List, Dict, Any, Optional
from pathlib import Path


class BuilderArgumentExtractor:
    """
    Extracts command-line arguments from step builder _get_job_arguments() methods.
    
    This class uses AST parsing to analyze builder Python files and extract
    the arguments that builders pass to scripts via the _get_job_arguments() method.
    """
    
    def __init__(self, builder_file_path: str):
        """
        Initialize the builder argument extractor.
        
        Args:
            builder_file_path: Path to the step builder Python file
        """
        self.builder_file_path = Path(builder_file_path)
        self.builder_ast = None
        self._parse_builder_file()
    
    def _parse_builder_file(self) -> None:
        """Parse the builder file into an AST."""
        try:
            with open(self.builder_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.builder_ast = ast.parse(content)
        except Exception as e:
            raise ValueError(f"Failed to parse builder file {self.builder_file_path}: {e}")
    
    def extract_job_arguments(self) -> Set[str]:
        """
        Extract command-line arguments from the _get_job_arguments() method.
        
        Returns:
            Set of argument names (without -- prefix) that the builder provides
        """
        if not self.builder_ast:
            return set()
        
        # Find the _get_job_arguments method
        job_args_method = self._find_job_arguments_method()
        if not job_args_method:
            return set()
        
        # Extract arguments from the method
        arguments = self._extract_arguments_from_method(job_args_method)
        return arguments
    
    def _find_job_arguments_method(self) -> Optional[ast.FunctionDef]:
        """Find the _get_job_arguments method in the AST."""
        for node in ast.walk(self.builder_ast):
            if (isinstance(node, ast.FunctionDef) and 
                node.name == '_get_job_arguments'):
                return node
        return None
    
    def _extract_arguments_from_method(self, method_node: ast.FunctionDef) -> Set[str]:
        """
        Extract argument names from the _get_job_arguments method.
        
        Args:
            method_node: AST node for the _get_job_arguments method
            
        Returns:
            Set of argument names (without -- prefix)
        """
        arguments = set()
        
        # Walk through all nodes in the method
        for node in ast.walk(method_node):
            # Look for string literals that start with "--"
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.startswith('--'):
                    arg_name = node.value[2:]  # Remove -- prefix
                    arguments.add(arg_name)
            
            # Handle older Python versions with ast.Str
            elif isinstance(node, ast.Str) and node.s.startswith('--'):
                arg_name = node.s[2:]  # Remove -- prefix
                arguments.add(arg_name)
            
            # Look for list literals containing argument strings
            elif isinstance(node, ast.List):
                for element in node.elts:
                    if isinstance(element, ast.Constant) and isinstance(element.value, str):
                        if element.value.startswith('--'):
                            arg_name = element.value[2:]
                            arguments.add(arg_name)
                    elif isinstance(element, ast.Str) and element.s.startswith('--'):
                        arg_name = element.s[2:]
                        arguments.add(arg_name)
        
        return arguments
    
    def get_method_source(self) -> Optional[str]:
        """
        Get the source code of the _get_job_arguments method for debugging.
        
        Returns:
            Source code string or None if method not found
        """
        method_node = self._find_job_arguments_method()
        if not method_node:
            return None
        
        try:
            # Read the original file to extract method source
            with open(self.builder_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Extract method lines (approximate)
            start_line = method_node.lineno - 1
            end_line = method_node.end_lineno if hasattr(method_node, 'end_lineno') else start_line + 10
            
            method_lines = lines[start_line:end_line]
            return ''.join(method_lines)
        except Exception:
            return None


class BuilderRegistry:
    """
    Registry for mapping script names to their corresponding step builders.
    
    This class helps find the appropriate builder file for a given script
    to enable builder argument extraction during validation.
    """
    
    def __init__(self, builders_dir: str):
        """
        Initialize the builder registry.
        
        Args:
            builders_dir: Directory containing step builder files
        """
        self.builders_dir = Path(builders_dir)
        self._script_to_builder_mapping = self._build_script_mapping()
    
    def _build_script_mapping(self) -> Dict[str, str]:
        """
        Build mapping from script names to builder file paths.
        
        Returns:
            Dictionary mapping script names to builder file paths
        """
        mapping = {}
        
        if not self.builders_dir.exists():
            return mapping
        
        # Scan builder files and extract script associations
        for builder_file in self.builders_dir.glob("builder_*.py"):
            if builder_file.name.startswith('__'):
                continue
            
            try:
                script_names = self._extract_script_names_from_builder(builder_file)
                for script_name in script_names:
                    mapping[script_name] = str(builder_file)
            except Exception:
                # Skip builders that can't be analyzed
                continue
        
        return mapping
    
    def _extract_script_names_from_builder(self, builder_file: Path) -> List[str]:
        """
        Extract associated script names from a builder file.
        
        This uses heuristics based on:
        1. Builder file naming conventions (builder_<script_name>_step.py)
        2. Config class imports and usage
        3. Entry point references in the code
        4. Common naming variations (preprocess/preprocessing, eval/evaluation, etc.)
        
        Args:
            builder_file: Path to the builder file
            
        Returns:
            List of associated script names
        """
        script_names = []
        
        # Extract from filename pattern
        filename = builder_file.stem  # Remove .py extension
        if filename.startswith('builder_') and filename.endswith('_step'):
            # Extract middle part: builder_<script_name>_step -> <script_name>
            middle_part = filename[8:-5]  # Remove 'builder_' and '_step'
            script_names.append(middle_part)
            
            # Add common naming variations
            script_names.extend(self._generate_name_variations(middle_part))
        
        # Additional extraction from file content (config imports, etc.)
        try:
            with open(builder_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for config imports that might indicate script association
            config_import_pattern = r'from\s+\.\.configs\.config_(\w+)_step\s+import'
            matches = re.findall(config_import_pattern, content)
            script_names.extend(matches)
            
            # Look for entry_point references
            entry_point_pattern = r'entry_point.*["\'](\w+)\.py["\']'
            matches = re.findall(entry_point_pattern, content)
            for match in matches:
                script_names.append(match)
                
        except Exception:
            pass
        
        # Remove duplicates and return
        return list(set(script_names))
    
    def _generate_name_variations(self, name: str) -> List[str]:
        """
        Generate common naming variations for a script name.
        
        Args:
            name: Original script name
            
        Returns:
            List of possible naming variations
        """
        variations = []
        
        # Handle specific common variations
        # Use more precise matching to avoid substring conflicts
        if 'preprocessing' in name:
            variations.append(name.replace('preprocessing', 'preprocess'))
        if 'preprocess' in name and 'preprocessing' not in name:
            variations.append(name.replace('preprocess', 'preprocessing'))
        
        if 'evaluation' in name:
            variations.append(name.replace('evaluation', 'eval'))
        if 'eval' in name and 'evaluation' not in name:
            variations.append(name.replace('eval', 'evaluation'))
        
        if 'xgboost' in name:
            variations.append(name.replace('xgboost', 'xgb'))
        if 'xgb' in name and 'xgboost' not in name:
            variations.append(name.replace('xgb', 'xgboost'))
        
        return variations
    
    def get_builder_for_script(self, script_name: str) -> Optional[str]:
        """
        Get the builder file path for a given script name.
        
        Args:
            script_name: Name of the script
            
        Returns:
            Path to the builder file or None if not found
        """
        return self._script_to_builder_mapping.get(script_name)
    
    def get_all_mappings(self) -> Dict[str, str]:
        """Get all script-to-builder mappings."""
        return self._script_to_builder_mapping.copy()


def extract_builder_arguments(script_name: str, builders_dir: str) -> Set[str]:
    """
    Convenience function to extract builder arguments for a script.
    
    Args:
        script_name: Name of the script
        builders_dir: Directory containing builder files
        
    Returns:
        Set of argument names provided by the builder
    """
    registry = BuilderRegistry(builders_dir)
    builder_file = registry.get_builder_for_script(script_name)
    
    if not builder_file:
        return set()
    
    try:
        extractor = BuilderArgumentExtractor(builder_file)
        return extractor.extract_job_arguments()
    except Exception:
        return set()
