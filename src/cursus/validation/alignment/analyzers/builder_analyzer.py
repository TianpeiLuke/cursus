"""
Builder Code Analysis Engine

Analyzes builder code using AST parsing to extract configuration usage patterns,
validation calls, and other architectural patterns.
"""

import ast
from typing import Dict, List, Any
from pathlib import Path


class BuilderCodeAnalyzer:
    """
    Analyzes builder code to extract configuration usage patterns and architectural information.
    
    Uses AST parsing to identify:
    - Configuration field accesses
    - Validation method calls
    - Default value assignments
    - Class and method definitions
    """
    
    def analyze_builder_file(self, builder_path: Path) -> Dict[str, Any]:
        """
        Analyze builder file to extract code patterns.
        
        Args:
            builder_path: Path to the builder file
            
        Returns:
            Dictionary containing builder analysis results
        """
        try:
            with open(builder_path, 'r') as f:
                builder_content = f.read()
            
            builder_ast = ast.parse(builder_content)
            return self.analyze_builder_code(builder_ast, builder_content)
        except Exception as e:
            return {
                'error': str(e),
                'config_accesses': [],
                'validation_calls': [],
                'default_assignments': [],
                'class_definitions': [],
                'method_definitions': []
            }
    
    def analyze_builder_code(self, builder_ast: ast.AST, builder_content: str) -> Dict[str, Any]:
        """
        Analyze builder AST to extract configuration usage patterns.
        
        Args:
            builder_ast: Parsed AST of the builder code
            builder_content: Raw builder code content
            
        Returns:
            Dictionary containing analysis results
        """
        analysis = {
            'config_accesses': [],
            'validation_calls': [],
            'default_assignments': [],
            'class_definitions': [],
            'method_definitions': [],
            'import_statements': [],
            'config_class_usage': []
        }
        
        visitor = BuilderVisitor(analysis)
        visitor.visit(builder_ast)
        
        return analysis


class BuilderVisitor(ast.NodeVisitor):
    """AST visitor for analyzing builder code patterns."""
    
    def __init__(self, analysis: Dict[str, Any]):
        """
        Initialize the visitor with analysis dictionary.
        
        Args:
            analysis: Dictionary to store analysis results
        """
        self.analysis = analysis
        self.method_calls = set()  # Track method calls to exclude from field accesses
    
    def visit_Call(self, node):
        """Visit function/method call nodes."""
        # Track method calls on config objects to exclude from field access detection
        if isinstance(node.func, ast.Attribute):
            # Check for config.method() calls
            if (isinstance(node.func.value, ast.Name) and 
                node.func.value.id == 'config'):
                self.method_calls.add((node.func.attr, node.lineno))
            # Check for self.config.method() calls
            elif (isinstance(node.func.value, ast.Attribute) and
                  isinstance(node.func.value.value, ast.Name) and
                  node.func.value.value.id == 'self' and
                  node.func.value.attr == 'config'):
                self.method_calls.add((node.func.attr, node.lineno))
            
            # Look for validation method calls
            if node.func.attr in ['validate', 'require', 'check', 'assert_required']:
                self.analysis['validation_calls'].append({
                    'method': node.func.attr,
                    'line_number': node.lineno,
                    'args': len(node.args),
                    'context': self._get_context(node)
                })
        
        self.generic_visit(node)
    
    def visit_Attribute(self, node):
        """Visit attribute access nodes (e.g., config.field_name or self.config.field_name)."""
        # Look for config.field_name accesses
        if (isinstance(node.value, ast.Name) and 
            node.value.id == 'config'):
            # Only record as field access if it's not a method call
            if (node.attr, node.lineno) not in self.method_calls:
                self.analysis['config_accesses'].append({
                    'field_name': node.attr,
                    'line_number': node.lineno,
                    'context': self._get_context(node)
                })
        # Look for self.config.field_name accesses
        elif (isinstance(node.value, ast.Attribute) and
              isinstance(node.value.value, ast.Name) and
              node.value.value.id == 'self' and
              node.value.attr == 'config'):
            # Only record as field access if it's not a method call
            if (node.attr, node.lineno) not in self.method_calls:
                self.analysis['config_accesses'].append({
                    'field_name': node.attr,
                    'line_number': node.lineno,
                    'context': self._get_context(node)
                })
        
        self.generic_visit(node)
    
    def visit_Assign(self, node):
        """Visit assignment nodes."""
        # Look for default value assignments
        for target in node.targets:
            if isinstance(target, ast.Attribute):
                self.analysis['default_assignments'].append({
                    'field_name': target.attr,
                    'line_number': node.lineno,
                    'target_type': type(target.value).__name__,
                    'context': self._get_context(node)
                })
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        """Visit class definition nodes."""
        self.analysis['class_definitions'].append({
            'class_name': node.name,
            'line_number': node.lineno,
            'base_classes': [self._get_name(base) for base in node.bases],
            'decorators': [self._get_name(dec) for dec in node.decorator_list]
        })
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        """Visit function/method definition nodes."""
        self.analysis['method_definitions'].append({
            'method_name': node.name,
            'line_number': node.lineno,
            'args': [arg.arg for arg in node.args.args],
            'decorators': [self._get_name(dec) for dec in node.decorator_list],
            'is_async': False
        })
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node):
        """Visit async function/method definition nodes."""
        self.analysis['method_definitions'].append({
            'method_name': node.name,
            'line_number': node.lineno,
            'args': [arg.arg for arg in node.args.args],
            'decorators': [self._get_name(dec) for dec in node.decorator_list],
            'is_async': True
        })
        self.generic_visit(node)
    
    def visit_Import(self, node):
        """Visit import statements."""
        for alias in node.names:
            self.analysis['import_statements'].append({
                'type': 'import',
                'module': alias.name,
                'alias': alias.asname,
                'line_number': node.lineno
            })
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Visit from...import statements."""
        for alias in node.names:
            self.analysis['import_statements'].append({
                'type': 'from_import',
                'module': node.module,
                'name': alias.name,
                'alias': alias.asname,
                'line_number': node.lineno
            })
        self.generic_visit(node)
    
    def _get_context(self, node) -> str:
        """
        Get context information for a node (e.g., which method it's in).
        
        Args:
            node: AST node
            
        Returns:
            Context string
        """
        # This is a simplified context extraction
        # In a more sophisticated implementation, we'd track the current
        # class and method context as we traverse the AST
        return f"line_{node.lineno}"
    
    def _get_name(self, node) -> str:
        """
        Extract name from various AST node types.
        
        Args:
            node: AST node
            
        Returns:
            Name string or node type if name cannot be extracted
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        else:
            return type(node).__name__


class BuilderPatternAnalyzer:
    """
    Analyzes builder patterns and architectural compliance.
    
    Provides higher-level analysis of builder code patterns beyond basic AST parsing.
    """
    
    def __init__(self):
        """Initialize the pattern analyzer."""
        self.code_analyzer = BuilderCodeAnalyzer()
    
    def analyze_configuration_usage(self, builder_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze how configuration is used in the builder.
        
        Args:
            builder_analysis: Result from BuilderCodeAnalyzer
            
        Returns:
            Configuration usage analysis
        """
        config_accesses = builder_analysis.get('config_accesses', [])
        
        # Group accesses by field name
        field_usage = {}
        for access in config_accesses:
            field_name = access['field_name']
            if field_name not in field_usage:
                field_usage[field_name] = []
            field_usage[field_name].append(access)
        
        # Analyze usage patterns
        usage_patterns = {}
        for field_name, accesses in field_usage.items():
            usage_patterns[field_name] = {
                'access_count': len(accesses),
                'first_access_line': min(access['line_number'] for access in accesses),
                'last_access_line': max(access['line_number'] for access in accesses),
                'contexts': [access['context'] for access in accesses]
            }
        
        return {
            'accessed_fields': set(field_usage.keys()),
            'field_usage': field_usage,
            'usage_patterns': usage_patterns,
            'total_config_accesses': len(config_accesses)
        }
    
    def analyze_validation_patterns(self, builder_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze validation patterns in the builder.
        
        Args:
            builder_analysis: Result from BuilderCodeAnalyzer
            
        Returns:
            Validation pattern analysis
        """
        validation_calls = builder_analysis.get('validation_calls', [])
        
        validation_methods = {}
        for call in validation_calls:
            method = call['method']
            if method not in validation_methods:
                validation_methods[method] = []
            validation_methods[method].append(call)
        
        return {
            'has_validation': len(validation_calls) > 0,
            'validation_methods': validation_methods,
            'validation_call_count': len(validation_calls),
            'validation_lines': [call['line_number'] for call in validation_calls]
        }
    
    def analyze_import_patterns(self, builder_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze import patterns to detect configuration class usage.
        
        Args:
            builder_analysis: Result from BuilderCodeAnalyzer
            
        Returns:
            Import pattern analysis
        """
        import_statements = builder_analysis.get('import_statements', [])
        
        config_imports = []
        for stmt in import_statements:
            if 'config' in stmt.get('module', '').lower() or 'config' in stmt.get('name', '').lower():
                config_imports.append(stmt)
        
        return {
            'total_imports': len(import_statements),
            'config_imports': config_imports,
            'has_config_import': len(config_imports) > 0,
            'import_modules': [stmt.get('module') for stmt in import_statements if stmt.get('module')]
        }
