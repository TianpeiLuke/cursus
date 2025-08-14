"""
Script analyzer for extracting usage patterns from Python scripts.

Uses AST parsing to identify path references, environment variable access,
import statements, and argument parsing patterns in processing scripts.
"""

import ast
import os
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

from ..alignment_utils import (
    PathReference, EnvVarAccess, ImportStatement, ArgumentDefinition,
    PathConstruction, FileOperation, detect_step_type_from_registry,
    detect_framework_from_imports
)
from ..framework_patterns import detect_training_patterns


class ScriptAnalyzer:
    """
    Analyzes Python script source code to extract usage patterns.
    
    Uses AST parsing to identify:
    - Path references and construction
    - Environment variable access
    - Import statements
    - Function definitions and calls
    - Argument parsing patterns
    """
    
    def __init__(self, script_path: str):
        """
        Initialize the script analyzer.
        
        Args:
            script_path: Path to the Python script to analyze
        """
        self.script_path = script_path
        self.script_content = self._read_script()
        self.ast_tree = self._parse_script()
        self.lines = self.script_content.splitlines()
        
    def _read_script(self) -> str:
        """Read the script content from file."""
        try:
            with open(self.script_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Could not read script {self.script_path}: {e}")
    
    def _parse_script(self) -> ast.AST:
        """Parse the script content into an AST."""
        try:
            return ast.parse(self.script_content)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in script {self.script_path}: {e}")
    
    def _get_line_context(self, line_number: int, context_lines: int = 2) -> str:
        """Get context around a specific line number."""
        start = max(0, line_number - context_lines - 1)
        end = min(len(self.lines), line_number + context_lines)
        context_lines_list = self.lines[start:end]
        
        # Mark the target line
        target_index = line_number - start - 1
        if 0 <= target_index < len(context_lines_list):
            context_lines_list[target_index] = f">>> {context_lines_list[target_index]}"
        
        return '\n'.join(context_lines_list)
    
    def extract_path_references(self) -> List[PathReference]:
        """Extract all path references from the script."""
        path_references = []
        
        class PathVisitor(ast.NodeVisitor):
            def __init__(self, analyzer):
                self.analyzer = analyzer
                
            def visit_Str(self, node):
                # Handle string literals (Python < 3.8)
                self._check_string_for_path(node.s, node.lineno)
                self.generic_visit(node)
                
            def visit_Constant(self, node):
                # Handle string constants (Python >= 3.8)
                if isinstance(node.value, str):
                    self._check_string_for_path(node.value, node.lineno)
                self.generic_visit(node)
            
            def visit_Call(self, node):
                # Check for path construction calls
                self._check_path_construction(node)
                self.generic_visit(node)
            
            def _check_string_for_path(self, string_value: str, line_number: int):
                # Check if string looks like a path
                if self._looks_like_path(string_value):
                    context = self.analyzer._get_line_context(line_number)
                    path_ref = PathReference(
                        path=string_value,
                        line_number=line_number,
                        context=context,
                        is_hardcoded=True
                    )
                    path_references.append(path_ref)
            
            def _check_path_construction(self, node):
                # Check for os.path.join, pathlib operations, etc.
                if isinstance(node.func, ast.Attribute):
                    if (isinstance(node.func.value, ast.Attribute) and
                        isinstance(node.func.value.value, ast.Name) and
                        node.func.value.value.id == 'os' and
                        node.func.value.attr == 'path' and
                        node.func.attr == 'join'):
                        
                        # Extract path parts from os.path.join
                        parts = []
                        for arg in node.args:
                            if isinstance(arg, (ast.Str, ast.Constant)):
                                value = arg.s if isinstance(arg, ast.Str) else arg.value
                                if isinstance(value, str):
                                    parts.append(value)
                        
                        if parts:
                            constructed_path = os.path.join(*parts)
                            context = self.analyzer._get_line_context(node.lineno)
                            path_ref = PathReference(
                                path=constructed_path,
                                line_number=node.lineno,
                                context=context,
                                is_hardcoded=False,
                                construction_method='os.path.join'
                            )
                            path_references.append(path_ref)
            
            def _looks_like_path(self, string_value: str) -> bool:
                # Heuristics to identify path-like strings
                path_indicators = [
                    '/opt/ml/',
                    '/tmp/',
                    '/var/',
                    '/home/',
                    '/usr/',
                    './',
                    '../',
                    '\\',  # Windows paths
                ]
                
                # Check for common path patterns
                for indicator in path_indicators:
                    if indicator in string_value:
                        return True
                
                # Check for file extensions
                if '.' in string_value and len(string_value.split('.')[-1]) <= 5:
                    return True
                
                # Check for multiple path separators
                if string_value.count('/') > 1 or string_value.count('\\') > 1:
                    return True
                
                return False
        
        visitor = PathVisitor(self)
        visitor.visit(self.ast_tree)
        
        return path_references
    
    def extract_env_var_access(self) -> List[EnvVarAccess]:
        """Extract all environment variable access patterns."""
        env_var_accesses = []
        
        class EnvVarVisitor(ast.NodeVisitor):
            def __init__(self, analyzer):
                self.analyzer = analyzer
                
            def visit_Subscript(self, node):
                # Check for os.environ['VAR'] pattern
                if (isinstance(node.value, ast.Attribute) and
                    isinstance(node.value.value, ast.Name) and
                    node.value.value.id == 'os' and
                    node.value.attr == 'environ'):
                    
                    var_name = self._extract_string_value(node.slice)
                    if var_name:
                        context = self.analyzer._get_line_context(node.lineno)
                        env_access = EnvVarAccess(
                            variable_name=var_name,
                            line_number=node.lineno,
                            context=context,
                            access_method='os.environ'
                        )
                        env_var_accesses.append(env_access)
                
                self.generic_visit(node)
            
            def visit_Call(self, node):
                # Check for os.getenv() and os.environ.get() patterns
                if isinstance(node.func, ast.Attribute):
                    if (isinstance(node.func.value, ast.Name) and
                        node.func.value.id == 'os' and
                        node.func.attr == 'getenv'):
                        
                        # os.getenv('VAR', 'default')
                        if node.args:
                            var_name = self._extract_string_value(node.args[0])
                            if var_name:
                                has_default = len(node.args) > 1
                                default_value = None
                                if has_default:
                                    default_value = self._extract_string_value(node.args[1])
                                
                                context = self.analyzer._get_line_context(node.lineno)
                                env_access = EnvVarAccess(
                                    variable_name=var_name,
                                    line_number=node.lineno,
                                    context=context,
                                    access_method='os.getenv',
                                    has_default=has_default,
                                    default_value=default_value
                                )
                                env_var_accesses.append(env_access)
                    
                    elif (isinstance(node.func.value, ast.Attribute) and
                          isinstance(node.func.value.value, ast.Name) and
                          node.func.value.value.id == 'os' and
                          node.func.value.attr == 'environ' and
                          node.func.attr == 'get'):
                        
                        # os.environ.get('VAR', 'default')
                        if node.args:
                            var_name = self._extract_string_value(node.args[0])
                            if var_name:
                                has_default = len(node.args) > 1
                                default_value = None
                                if has_default:
                                    default_value = self._extract_string_value(node.args[1])
                                
                                context = self.analyzer._get_line_context(node.lineno)
                                env_access = EnvVarAccess(
                                    variable_name=var_name,
                                    line_number=node.lineno,
                                    context=context,
                                    access_method='os.environ.get',
                                    has_default=has_default,
                                    default_value=default_value
                                )
                                env_var_accesses.append(env_access)
                
                self.generic_visit(node)
            
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
        
        visitor = EnvVarVisitor(self)
        visitor.visit(self.ast_tree)
        
        return env_var_accesses
    
    def extract_imports(self) -> List[ImportStatement]:
        """Extract all import statements."""
        imports = []
        
        class ImportVisitor(ast.NodeVisitor):
            def visit_Import(self, node):
                for alias in node.names:
                    import_stmt = ImportStatement(
                        module_name=alias.name,
                        import_alias=alias.asname,
                        line_number=node.lineno,
                        is_from_import=False
                    )
                    imports.append(import_stmt)
            
            def visit_ImportFrom(self, node):
                if node.module:
                    imported_items = [alias.name for alias in node.names]
                    import_stmt = ImportStatement(
                        module_name=node.module,
                        import_alias=None,
                        line_number=node.lineno,
                        is_from_import=True,
                        imported_items=imported_items
                    )
                    imports.append(import_stmt)
        
        visitor = ImportVisitor()
        visitor.visit(self.ast_tree)
        
        return imports
    
    def extract_argument_definitions(self) -> List[ArgumentDefinition]:
        """Extract command-line argument definitions."""
        arguments = []
        
        class ArgumentVisitor(ast.NodeVisitor):
            def __init__(self, analyzer):
                self.analyzer = analyzer
                self.in_argparse_context = False
                
            def visit_Call(self, node):
                # Look for add_argument calls
                if (isinstance(node.func, ast.Attribute) and
                    node.func.attr == 'add_argument'):
                    
                    # Extract argument name
                    if node.args and isinstance(node.args[0], (ast.Str, ast.Constant)):
                        arg_name_raw = (node.args[0].s if isinstance(node.args[0], ast.Str) 
                                       else node.args[0].value)
                        
                        if isinstance(arg_name_raw, str):
                            # Remove leading dashes and convert to underscore format
                            arg_name = arg_name_raw.lstrip('-').replace('-', '_')
                            
                            # Extract argument properties from keywords
                            is_required = False
                            has_default = False
                            default_value = None
                            argument_type = None
                            choices = None
                            
                            for keyword in node.keywords:
                                if keyword.arg == 'required':
                                    is_required = self._extract_bool_value(keyword.value)
                                elif keyword.arg == 'default':
                                    has_default = True
                                    default_value = self._extract_value(keyword.value)
                                elif keyword.arg == 'type':
                                    argument_type = self._extract_type_name(keyword.value)
                                elif keyword.arg == 'choices':
                                    choices = self._extract_choices(keyword.value)
                            
                            context = self.analyzer._get_line_context(node.lineno)
                            arg_def = ArgumentDefinition(
                                argument_name=arg_name,
                                line_number=node.lineno,
                                is_required=is_required,
                                has_default=has_default,
                                default_value=default_value,
                                argument_type=argument_type,
                                choices=choices
                            )
                            arguments.append(arg_def)
                
                self.generic_visit(node)
            
            def _extract_bool_value(self, node) -> bool:
                """Extract boolean value from AST node."""
                if isinstance(node, ast.Constant):
                    return bool(node.value)
                elif isinstance(node, ast.NameConstant):  # Python < 3.8
                    return bool(node.value)
                return False
            
            def _extract_value(self, node) -> Any:
                """Extract value from AST node."""
                if isinstance(node, ast.Constant):
                    return node.value
                elif isinstance(node, ast.Str):
                    return node.s
                elif isinstance(node, ast.Num):
                    return node.n
                elif isinstance(node, ast.NameConstant):
                    return node.value
                return None
            
            def _extract_type_name(self, node) -> Optional[str]:
                """Extract type name from AST node."""
                if isinstance(node, ast.Name):
                    return node.id
                return None
            
            def _extract_choices(self, node) -> Optional[List[str]]:
                """Extract choices list from AST node."""
                if isinstance(node, ast.List):
                    choices = []
                    for elt in node.elts:
                        if isinstance(elt, (ast.Str, ast.Constant)):
                            value = elt.s if isinstance(elt, ast.Str) else elt.value
                            if isinstance(value, str):
                                choices.append(value)
                    return choices if choices else None
                return None
        
        visitor = ArgumentVisitor(self)
        visitor.visit(self.ast_tree)
        
        return arguments
    
    def extract_file_operations(self) -> List[FileOperation]:
        """Extract file operations (open, read, write) from the script."""
        file_operations = []
        
        class FileOpVisitor(ast.NodeVisitor):
            def __init__(self, analyzer):
                self.analyzer = analyzer
                
            def visit_Call(self, node):
                # Check for standard open() calls
                if isinstance(node.func, ast.Name) and node.func.id == 'open':
                    self._handle_open_call(node)
                
                # Check for tarfile operations
                elif isinstance(node.func, ast.Attribute):
                    self._handle_attribute_call(node)
                
                self.generic_visit(node)
            
            def _handle_open_call(self, node):
                """Handle standard open() calls."""
                if node.args:
                    file_path = self._extract_string_value(node.args[0])
                    mode = 'r'  # default mode
                    
                    if len(node.args) > 1:
                        mode_value = self._extract_string_value(node.args[1])
                        if mode_value:
                            mode = mode_value
                    
                    # Check keywords for mode
                    for keyword in node.keywords:
                        if keyword.arg == 'mode':
                            mode_value = self._extract_string_value(keyword.value)
                            if mode_value:
                                mode = mode_value
                    
                    if file_path:
                        operation_type = self._determine_operation_type(mode)
                        context = self.analyzer._get_line_context(node.lineno)
                        
                        file_op = FileOperation(
                            file_path=file_path,
                            operation_type=operation_type,
                            line_number=node.lineno,
                            context=context,
                            mode=mode,
                            method='open'
                        )
                        file_operations.append(file_op)
            
            def _handle_attribute_call(self, node):
                """Handle attribute-based file operations."""
                # tarfile.open() calls
                if (isinstance(node.func.value, ast.Name) and 
                    node.func.value.id == 'tarfile' and 
                    node.func.attr == 'open'):
                    
                    if node.args:
                        file_path = self._extract_string_value(node.args[0])
                        mode = 'r'
                        
                        if len(node.args) > 1:
                            mode_value = self._extract_string_value(node.args[1])
                            if mode_value:
                                mode = mode_value
                        
                        if file_path:
                            operation_type = 'read' if 'r' in mode else 'write'
                            context = self.analyzer._get_line_context(node.lineno)
                            
                            file_op = FileOperation(
                                file_path=file_path,
                                operation_type=operation_type,
                                line_number=node.lineno,
                                context=context,
                                mode=mode,
                                method='tarfile.open'
                            )
                            file_operations.append(file_op)
                
                # shutil operations
                elif (isinstance(node.func.value, ast.Name) and 
                      node.func.value.id == 'shutil'):
                    
                    if node.func.attr in ['copy', 'copy2', 'copyfile', 'move']:
                        # These operations read from source and write to destination
                        if len(node.args) >= 2:
                            src_path = self._extract_string_value(node.args[0])
                            dst_path = self._extract_string_value(node.args[1])
                            context = self.analyzer._get_line_context(node.lineno)
                            
                            if src_path:
                                file_op = FileOperation(
                                    file_path=src_path,
                                    operation_type='read',
                                    line_number=node.lineno,
                                    context=context,
                                    method=f'shutil.{node.func.attr}'
                                )
                                file_operations.append(file_op)
                            
                            if dst_path:
                                file_op = FileOperation(
                                    file_path=dst_path,
                                    operation_type='write',
                                    line_number=node.lineno,
                                    context=context,
                                    method=f'shutil.{node.func.attr}'
                                )
                                file_operations.append(file_op)
                
                # pathlib Path operations
                elif (hasattr(node.func, 'attr') and 
                      node.func.attr in ['mkdir', 'write_text', 'write_bytes', 'read_text', 'read_bytes']):
                    
                    # Try to extract the path from the object
                    path_obj = self._extract_path_from_pathlib(node.func.value)
                    if path_obj:
                        operation_type = 'write' if 'write' in node.func.attr or 'mkdir' in node.func.attr else 'read'
                        context = self.analyzer._get_line_context(node.lineno)
                        
                        file_op = FileOperation(
                            file_path=path_obj,
                            operation_type=operation_type,
                            line_number=node.lineno,
                            context=context,
                            method=f'pathlib.{node.func.attr}'
                        )
                        file_operations.append(file_op)
                
                # pandas operations
                elif (isinstance(node.func.value, ast.Name) and 
                      node.func.value.id == 'pd' and 
                      node.func.attr in ['read_csv', 'read_json', 'read_parquet']):
                    
                    if node.args:
                        file_path = self._extract_string_value(node.args[0])
                        if file_path:
                            context = self.analyzer._get_line_context(node.lineno)
                            file_op = FileOperation(
                                file_path=file_path,
                                operation_type='read',
                                line_number=node.lineno,
                                context=context,
                                method=f'pandas.{node.func.attr}'
                            )
                            file_operations.append(file_op)
                
                # DataFrame.to_* operations
                elif (hasattr(node.func, 'attr') and 
                      node.func.attr.startswith('to_') and 
                      node.func.attr in ['to_csv', 'to_json', 'to_parquet']):
                    
                    if node.args:
                        file_path = self._extract_string_value(node.args[0])
                        if file_path:
                            context = self.analyzer._get_line_context(node.lineno)
                            file_op = FileOperation(
                                file_path=file_path,
                                operation_type='write',
                                line_number=node.lineno,
                                context=context,
                                method=f'dataframe.{node.func.attr}'
                            )
                            file_operations.append(file_op)
                
                # pickle operations
                elif (isinstance(node.func.value, ast.Name) and 
                      node.func.value.id in ['pkl', 'pickle'] and 
                      node.func.attr in ['load', 'dump']):
                    
                    # pkl.load(f) or pkl.dump(obj, f)
                    file_arg_index = 0 if node.func.attr == 'load' else 1
                    if len(node.args) > file_arg_index:
                        # Note: pickle operations typically use file objects, not paths
                        # But we can still track the operation
                        context = self.analyzer._get_line_context(node.lineno)
                        operation_type = 'read' if node.func.attr == 'load' else 'write'
                        file_op = FileOperation(
                            file_path='<file_object>',  # Placeholder for file object
                            operation_type=operation_type,
                            line_number=node.lineno,
                            context=context,
                            method=f'pickle.{node.func.attr}'
                        )
                        file_operations.append(file_op)
                
                # json operations
                elif (isinstance(node.func.value, ast.Name) and 
                      node.func.value.id == 'json' and 
                      node.func.attr in ['load', 'dump']):
                    
                    # json.load(f) or json.dump(obj, f)
                    file_arg_index = 0 if node.func.attr == 'load' else 1
                    if len(node.args) > file_arg_index:
                        context = self.analyzer._get_line_context(node.lineno)
                        operation_type = 'read' if node.func.attr == 'load' else 'write'
                        file_op = FileOperation(
                            file_path='<file_object>',  # Placeholder for file object
                            operation_type=operation_type,
                            line_number=node.lineno,
                            context=context,
                            method=f'json.{node.func.attr}'
                        )
                        file_operations.append(file_op)
                
                # XGBoost model operations
                elif (hasattr(node.func, 'attr') and 
                      node.func.attr in ['load_model', 'save_model']):
                    
                    if node.args:
                        file_path = self._extract_string_value(node.args[0])
                        if file_path:
                            context = self.analyzer._get_line_context(node.lineno)
                            operation_type = 'read' if 'load' in node.func.attr else 'write'
                            file_op = FileOperation(
                                file_path=file_path,
                                operation_type=operation_type,
                                line_number=node.lineno,
                                context=context,
                                method=f'model.{node.func.attr}'
                            )
                            file_operations.append(file_op)
                
                # matplotlib/pyplot save operations
                elif (isinstance(node.func.value, ast.Name) and 
                      node.func.value.id == 'plt' and 
                      node.func.attr == 'savefig'):
                    
                    if node.args:
                        file_path = self._extract_string_value(node.args[0])
                        if file_path:
                            context = self.analyzer._get_line_context(node.lineno)
                            file_op = FileOperation(
                                file_path=file_path,
                                operation_type='write',
                                line_number=node.lineno,
                                context=context,
                                method='matplotlib.savefig'
                            )
                            file_operations.append(file_op)
                
                # Path.glob() operations (directory traversal)
                elif (hasattr(node.func, 'attr') and 
                      node.func.attr == 'glob'):
                    
                    # Extract the path from the Path object
                    path_obj = self._extract_path_from_pathlib(node.func.value)
                    if path_obj:
                        context = self.analyzer._get_line_context(node.lineno)
                        file_op = FileOperation(
                            file_path=path_obj,
                            operation_type='read',
                            line_number=node.lineno,
                            context=context,
                            method='pathlib.glob'
                        )
                        file_operations.append(file_op)
            
            def _extract_path_from_pathlib(self, node) -> Optional[str]:
                """Extract path string from pathlib Path construction."""
                # Handle Path(string) construction
                if (isinstance(node, ast.Call) and 
                    isinstance(node.func, ast.Name) and 
                    node.func.id == 'Path' and 
                    node.args):
                    return self._extract_string_value(node.args[0])
                
                # Handle path / "subpath" operations
                elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                    left_path = self._extract_path_from_pathlib(node.left)
                    right_path = self._extract_string_value(node.right)
                    if left_path and right_path:
                        return f"{left_path}/{right_path}"
                
                return None
            
            def _extract_string_value(self, node) -> Optional[str]:
                """Extract string value from AST node."""
                if isinstance(node, ast.Str):
                    return node.s
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    return node.value
                return None
            
            def _determine_operation_type(self, mode: str) -> str:
                """Determine operation type from file mode."""
                if 'w' in mode or 'a' in mode:
                    return 'write'
                elif 'r' in mode:
                    return 'read'
                else:
                    return 'unknown'
        
        visitor = FileOpVisitor(self)
        visitor.visit(self.ast_tree)
        
        return file_operations
    
    def get_all_analysis_results(self) -> Dict[str, Any]:
        """Get comprehensive analysis results for the script."""
        # Extract basic analysis results
        imports = self.extract_imports()
        
        # Phase 2 Enhancement: Add step type and framework detection
        script_name = Path(self.script_path).stem
        step_type = detect_step_type_from_registry(script_name)
        framework = detect_framework_from_imports(imports)
        
        # Add step type-specific patterns if this is a training script
        step_type_patterns = {}
        if step_type == "Training":
            step_type_patterns = self._detect_training_patterns()
        
        return {
            'script_path': self.script_path,
            'path_references': self.extract_path_references(),
            'env_var_accesses': self.extract_env_var_access(),
            'imports': imports,
            'argument_definitions': self.extract_argument_definitions(),
            'file_operations': self.extract_file_operations(),
            # Phase 2 Enhancement: Step type awareness
            'step_type': step_type,
            'framework': framework,
            'step_type_patterns': step_type_patterns
        }
    
    def _detect_training_patterns(self) -> Dict[str, Any]:
        """
        Phase 2 Enhancement: Detect training-specific patterns using existing pattern recognition.
        
        Returns:
            Dictionary containing detected training patterns
        """
        try:
            # Use the framework patterns detection
            training_patterns = detect_training_patterns(self.script_content)
            return training_patterns
        except Exception as e:
            # Pattern detection is optional, return empty dict if it fails
            return {'error': f'Training pattern detection failed: {str(e)}'}
    
    def has_main_function(self) -> bool:
        """Check if the script has a main function."""
        class MainFunctionVisitor(ast.NodeVisitor):
            def __init__(self):
                self.has_main = False
                
            def visit_FunctionDef(self, node):
                if node.name == 'main':
                    self.has_main = True
                self.generic_visit(node)
        
        visitor = MainFunctionVisitor()
        visitor.visit(self.ast_tree)
        return visitor.has_main
    
    def has_main_block(self) -> bool:
        """Check if the script has a __main__ block."""
        class MainBlockVisitor(ast.NodeVisitor):
            def __init__(self):
                self.has_main_block = False
                
            def visit_If(self, node):
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
                            self.has_main_block = True
                
                self.generic_visit(node)
        
        visitor = MainBlockVisitor()
        visitor.visit(self.ast_tree)
        return visitor.has_main_block
