#!/usr/bin/env python3
"""
Script to Notebook Converter - Fixed Version

This utility converts a Python script to a properly formatted Jupyter notebook 
with correct JSON structure, line breaks, and cell formatting.
"""

import json
import ast
import re
from pathlib import Path
from typing import List, Dict, Any

class ScriptToNotebookConverter:
    """Converts Python scripts to properly formatted Jupyter notebooks."""
    
    def __init__(self):
        self.notebook_template = {
            "cells": [],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.8.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
    
    def create_markdown_cell(self, content: str) -> Dict[str, Any]:
        """Create a properly formatted markdown cell."""
        # Split content into lines and ensure each line ends with \n except the last
        lines = content.split('\n')
        source_lines = []
        for i, line in enumerate(lines):
            if i == len(lines) - 1:
                # Last line doesn't need \n
                source_lines.append(line)
            else:
                # All other lines need \n
                source_lines.append(line + '\n')
        
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source_lines
        }
    
    def create_code_cell(self, content: str) -> Dict[str, Any]:
        """Create a properly formatted code cell."""
        # Split content into lines and ensure each line ends with \n except the last
        lines = content.split('\n')
        source_lines = []
        for i, line in enumerate(lines):
            if i == len(lines) - 1:
                # Last line doesn't need \n
                source_lines.append(line)
            else:
                # All other lines need \n
                source_lines.append(line + '\n')
        
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source_lines
        }
    
    def extract_functions_and_classes(self, content: str) -> Dict[str, Any]:
        """Extract functions and classes from the script content."""
        tree = ast.parse(content)
        
        functions = {}
        classes = {}
        imports = []
        other_code = []
        main_content = []
        function_calls = set()
        
        # Extract top-level nodes
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(ast.get_source_segment(content, node))
            elif isinstance(node, ast.FunctionDef):
                func_code = ast.get_source_segment(content, node)
                if node.name == 'main':
                    # Extract main function body for interactive execution
                    main_content = self._extract_main_function_body(node, content)
                else:
                    functions[node.name] = func_code
            elif isinstance(node, ast.ClassDef):
                class_code = ast.get_source_segment(content, node)
                classes[node.name] = class_code
            elif isinstance(node, ast.If) and self._is_main_guard(node):
                # Handle if __name__ == "__main__": block
                if_main_code = self._extract_if_main_body(node, content)
                if if_main_code:
                    main_content.extend(if_main_code)
            else:
                # Other code (assignments, calls, etc.)
                code_segment = ast.get_source_segment(content, node)
                if code_segment and not code_segment.startswith('#!'):
                    other_code.append(code_segment)
                    # Extract function calls from this code
                    self._extract_function_calls(node, function_calls)
        
        return {
            'imports': imports,
            'functions': functions,
            'classes': classes,
            'other_code': other_code,
            'main_content': main_content,
            'function_calls': function_calls
        }
    
    def _is_main_guard(self, node):
        """Check if this is an if __name__ == "__main__": guard."""
        if not isinstance(node, ast.If):
            return False
        
        # Check if the test is __name__ == "__main__"
        test = node.test
        if isinstance(test, ast.Compare):
            if (isinstance(test.left, ast.Name) and test.left.id == '__name__' and
                len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq) and
                len(test.comparators) == 1 and isinstance(test.comparators[0], ast.Constant) and
                test.comparators[0].value == "__main__"):
                return True
        return False
    
    def _extract_main_function_body(self, main_node, content):
        """Extract the body of the main function for interactive execution."""
        main_body = []
        for stmt in main_node.body:
            # Skip return statements since they're invalid outside a function
            if isinstance(stmt, ast.Return):
                continue
                
            stmt_code = ast.get_source_segment(content, stmt)
            if stmt_code:
                # Remove return statements from the code (including nested ones)
                stmt_code = self._remove_return_statements(stmt_code)
                
                # Remove one level of indentation
                lines = stmt_code.split('\n')
                dedented_lines = []
                for line in lines:
                    if line.strip():  # Non-empty line
                        if line.startswith('    '):
                            dedented_lines.append(line[4:])  # Remove 4 spaces
                        else:
                            dedented_lines.append(line)
                    else:
                        dedented_lines.append(line)
                main_body.append('\n'.join(dedented_lines))
        return main_body
    
    def _remove_return_statements(self, code):
        """Remove return statements from code string."""
        lines = code.split('\n')
        filtered_lines = []
        for line in lines:
            # Skip lines that are just return statements
            stripped = line.strip()
            if stripped == 'return' or stripped.startswith('return '):
                continue
            filtered_lines.append(line)
        return '\n'.join(filtered_lines)
    
    def _extract_if_main_body(self, if_node, content):
        """Extract the body of if __name__ == '__main__': block."""
        if_main_body = []
        for stmt in if_node.body:
            stmt_code = ast.get_source_segment(content, stmt)
            if stmt_code:
                # Skip main() function calls since we're inlining the main function
                if 'main()' in stmt_code:
                    continue
                    
                # Remove one level of indentation
                lines = stmt_code.split('\n')
                dedented_lines = []
                for line in lines:
                    if line.strip():  # Non-empty line
                        if line.startswith('    '):
                            dedented_lines.append(line[4:])  # Remove 4 spaces
                        else:
                            dedented_lines.append(line)
                    else:
                        dedented_lines.append(line)
                if_main_body.append('\n'.join(dedented_lines))
        return if_main_body
    
    def _extract_function_calls(self, node, function_calls):
        """Extract function calls from an AST node."""
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                function_calls.add(child.func.id)
    
    def analyze_dependencies(self, components: Dict[str, Any]) -> Dict[str, List[str]]:
        """Analyze function call dependencies to determine proper ordering."""
        dependencies = {}
        
        # Analyze each function's dependencies
        for func_name, func_code in components['functions'].items():
            func_calls = set()
            try:
                func_tree = ast.parse(func_code)
                self._extract_function_calls(func_tree, func_calls)
                # Only keep dependencies that are defined functions in our script
                dependencies[func_name] = [call for call in func_calls if call in components['functions']]
            except:
                dependencies[func_name] = []
        
        # Analyze other code dependencies
        other_code_calls = set()
        for code_segment in components['other_code']:
            try:
                code_tree = ast.parse(code_segment)
                self._extract_function_calls(code_tree, other_code_calls)
            except:
                pass
        
        dependencies['__other_code__'] = [call for call in other_code_calls if call in components['functions']]
        
        return dependencies
    
    def topological_sort(self, dependencies: Dict[str, List[str]]) -> List[str]:
        """Perform topological sort to determine proper function definition order."""
        # Create a copy of dependencies for manipulation
        deps = dependencies.copy()
        result = []
        
        # Keep track of functions with no dependencies
        no_deps = [func for func, deps_list in deps.items() if not deps_list and func != '__other_code__']
        
        while no_deps:
            # Take a function with no dependencies
            current = no_deps.pop(0)
            result.append(current)
            
            # Remove this function from all dependency lists
            for func in list(deps.keys()):
                if current in deps[func]:
                    deps[func].remove(current)
                    # If this function now has no dependencies, add it to no_deps
                    if not deps[func] and func not in result and func not in no_deps and func != '__other_code__':
                        no_deps.append(func)
        
        # Add any remaining functions (in case of circular dependencies)
        remaining = [func for func in dependencies.keys() if func not in result and func != '__other_code__']
        result.extend(remaining)
        
        return result
    
    def _generate_notebook_title_and_description(self, script_path: Path, content: str) -> str:
        """Generate appropriate title and description based on script content."""
        script_name = script_path.stem
        
        # Extract docstring from script if available
        try:
            tree = ast.parse(content)
            docstring = ast.get_docstring(tree)
        except:
            docstring = None
        
        # Generate title based on script name
        if 'end_to_end' in script_name.lower():
            title = "# XGBoost Pipeline - End-to-End Testing"
            description = """This notebook performs comprehensive end-to-end testing of the complete XGBoost pipeline.

**Pipeline Components:**
1. Pipeline DAG validation and dependency resolution
2. Sequential step execution with proper data flow
3. Error handling and rollback mechanisms
4. Output verification and result analysis

**This notebook covers:**
- Complete pipeline execution from start to finish
- Dependency validation and execution ordering
- Data flow verification between steps
- Comprehensive result analysis and reporting"""
        
        elif 'performance' in script_name.lower():
            title = "# XGBoost Pipeline - Performance Analysis"
            description = """This notebook analyzes the performance characteristics of the XGBoost pipeline execution.

**Analysis Components:**
1. Execution time analysis across all pipeline steps
2. Resource utilization and bottleneck identification
3. Performance metrics and trend analysis
4. Optimization recommendations

**This notebook covers:**
- Step-by-step performance profiling
- Resource usage analysis
- Performance visualization and reporting
- Optimization insights and recommendations"""
        
        elif 'step_testing' in script_name.lower() or 'individual' in script_name.lower():
            title = "# XGBoost Pipeline - Individual Step Testing"
            description = """This notebook tests each pipeline step individually to ensure they work correctly in isolation.

**Pipeline Steps:**
1. XGBoost Training
2. XGBoost Model Evaluation
3. Model Calibration

**This notebook covers:**
- Mock step tester implementation
- Individual step testing functions
- Step validation and error handling
- Output verification and data flow checks"""
        
        else:
            # Generic title based on script name
            title = f"# {script_name.replace('_', ' ').title()}"
            if docstring:
                description = f"This notebook is generated from `{script_path.name}`.\n\n{docstring}"
            else:
                description = f"This notebook is generated from `{script_path.name}` and contains the converted script functionality."
        
        return f"{title}\n\n{description}"
    
    def convert_script_to_notebook(self, script_path: Path, output_path: Path):
        """Convert a Python script to a properly formatted Jupyter notebook."""
        print(f"Converting {script_path} to {output_path}")
        
        # Read the script
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Extract components
        components = self.extract_functions_and_classes(content)
        
        # Analyze dependencies and determine proper ordering
        dependencies = self.analyze_dependencies(components)
        function_order = self.topological_sort(dependencies)
        
        print(f"Function dependency analysis:")
        for func, deps in dependencies.items():
            if deps:
                print(f"  {func} depends on: {deps}")
        print(f"Determined function order: {function_order}")
        
        # Create notebook
        notebook = self.notebook_template.copy()
        cells = []
        
        # 1. Title cell - dynamically generated
        title_content = self._generate_notebook_title_and_description(script_path, content)
        cells.append(self.create_markdown_cell(title_content))
        
        # 2. Setup and Imports section
        cells.append(self.create_markdown_cell("## 1. Setup and Imports"))
        
        # Add imports
        if components['imports']:
            imports_code = '\n'.join(components['imports'])
            cells.append(self.create_code_cell(imports_code))
        
        # 3. Function and Class Definitions section
        cells.append(self.create_markdown_cell("## 2. Function and Class Definitions"))
        
        # Add classes first (they typically don't depend on functions)
        for class_name, class_code in components['classes'].items():
            cells.append(self.create_code_cell(class_code))
        
        # Add functions in dependency order
        for func_name in function_order:
            if func_name in components['functions']:
                cells.append(self.create_code_cell(components['functions'][func_name]))
        
        # 4. Execution section
        cells.append(self.create_markdown_cell("## 3. Script Execution"))
        
        # Add main content (from main() function and if __name__ == "__main__") for interactive execution
        if components['main_content']:
            # Combine all main content into a single cell
            combined_main_content = '\n\n'.join(code for code in components['main_content'] if code.strip())
            if combined_main_content.strip():
                cells.append(self.create_code_cell(combined_main_content))
        
        # Add other code (execution code) after all definitions
        if components['other_code']:
            execution_code = '\n\n'.join(components['other_code'])
            cells.append(self.create_code_cell(execution_code))
        
        # Set cells in notebook
        notebook['cells'] = cells
        
        # Write notebook with proper JSON formatting
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Successfully created notebook: {output_path}")
        print(f"✓ Created {len(cells)} cells with proper formatting")
        print(f"✓ All function definitions placed before execution code")

def main():
    """Main function to convert the script."""
    import sys
    
    converter = ScriptToNotebookConverter()
    
    # Allow command line arguments for script and notebook names
    if len(sys.argv) >= 3:
        script_path = Path(sys.argv[1])
        notebook_path = Path(sys.argv[2])
    else:
        # Default to step_testing_script.py
        script_path = Path('step_testing_script.py')
        notebook_path = Path('03_individual_step_testing.ipynb')
    
    if not script_path.exists():
        print(f"Error: Script file not found: {script_path}")
        return
    
    try:
        converter.convert_script_to_notebook(script_path, notebook_path)
        print("Conversion completed successfully!")
        print("The notebook now has proper JSON formatting with correct line breaks.")
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
