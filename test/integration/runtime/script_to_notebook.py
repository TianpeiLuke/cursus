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
        
        # Extract top-level nodes
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(ast.get_source_segment(content, node))
            elif isinstance(node, ast.FunctionDef):
                func_code = ast.get_source_segment(content, node)
                functions[node.name] = func_code
            elif isinstance(node, ast.ClassDef):
                class_code = ast.get_source_segment(content, node)
                classes[node.name] = class_code
            else:
                # Other code (assignments, calls, etc.)
                code_segment = ast.get_source_segment(content, node)
                if code_segment and not code_segment.startswith('#!'):
                    other_code.append(code_segment)
        
        return {
            'imports': imports,
            'functions': functions,
            'classes': classes,
            'other_code': other_code
        }
    
    def convert_script_to_notebook(self, script_path: Path, output_path: Path):
        """Convert a Python script to a properly formatted Jupyter notebook."""
        print(f"Converting {script_path} to {output_path}")
        
        # Read the script
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Extract components
        components = self.extract_functions_and_classes(content)
        
        # Create notebook
        notebook = self.notebook_template.copy()
        cells = []
        
        # 1. Title cell
        title_content = """# XGBoost Pipeline Test - Individual Step Testing

This notebook tests each pipeline step individually to ensure they work correctly in isolation.

**Pipeline Steps:**
1. XGBoost Training
2. XGBoost Model Evaluation
3. Model Calibration

**This notebook covers:**
- Mock step tester implementation
- Individual step testing functions
- Step validation and error handling
- Output verification and data flow checks"""
        
        cells.append(self.create_markdown_cell(title_content))
        
        # 2. Setup and Imports section
        cells.append(self.create_markdown_cell("## 1. Setup and Imports"))
        
        # Combine imports and setup call
        imports_code = '\n'.join(components['imports'])
        setup_code = """
# Setup environment
cursus_available = setup_environment()"""
        
        combined_setup = imports_code + setup_code
        cells.append(self.create_code_cell(combined_setup))
        
        # Add setup_environment function
        if 'setup_environment' in components['functions']:
            cells.append(self.create_code_cell(components['functions']['setup_environment']))
        
        # 3. Load Configuration section
        cells.append(self.create_markdown_cell("## 2. Load Configuration and Validate Environment"))
        if 'load_configuration' in components['functions']:
            cells.append(self.create_code_cell(components['functions']['load_configuration']))
        
        # 4. Mock Step Tester section
        cells.append(self.create_markdown_cell("## 3. Mock Step Tester Implementation"))
        if 'MockStepTester' in components['classes']:
            cells.append(self.create_code_cell(components['classes']['MockStepTester']))
        
        # 5. Run Individual Step Tests section
        cells.append(self.create_markdown_cell("## 4. Run Individual Step Tests"))
        
        # Add the test runner code
        test_runner_code = """# Load configuration and run tests
config_data = load_configuration()

if config_data['missing_files']:
    print("Cannot proceed with missing required files!")
else:
    print("\\n✓ All required files are available for testing")
    
    # Initialize and run step tests
    if config_data['step_configs']:
        step_tester = run_individual_step_tests(config_data)
        
        # Generate summary and verify files
        generate_test_summary(step_tester, config_data)
        verify_output_files(config_data['directories'])
    else:
        print("Cannot run tests without step configurations!")"""
        
        cells.append(self.create_code_cell(test_runner_code))
        
        # Add run_individual_step_tests function
        if 'run_individual_step_tests' in components['functions']:
            cells.append(self.create_code_cell(components['functions']['run_individual_step_tests']))
        
        # 6. Test Results Summary section
        cells.append(self.create_markdown_cell("## 5. Test Results Summary"))
        if 'generate_test_summary' in components['functions']:
            cells.append(self.create_code_cell(components['functions']['generate_test_summary']))
        
        # 7. Output File Verification section
        cells.append(self.create_markdown_cell("## 6. Output File Verification"))
        if 'verify_output_files' in components['functions']:
            cells.append(self.create_code_cell(components['functions']['verify_output_files']))
        
        # Set cells in notebook
        notebook['cells'] = cells
        
        # Write notebook with proper JSON formatting
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Successfully created notebook: {output_path}")
        print(f"✓ Created {len(cells)} cells with proper formatting")

def main():
    """Main function to convert the script."""
    converter = ScriptToNotebookConverter()
    
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
