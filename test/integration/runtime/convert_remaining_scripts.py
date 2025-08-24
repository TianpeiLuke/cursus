#!/usr/bin/env python3
"""
Convert Remaining Scripts to Notebooks

This script converts the end-to-end pipeline and performance analysis scripts
to properly formatted Jupyter notebooks.
"""

import json
import ast
from pathlib import Path
from typing import List, Dict, Any

class CustomScriptConverter:
    """Custom converter for specific scripts."""
    
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
        lines = content.split('\n')
        source_lines = []
        for i, line in enumerate(lines):
            if i == len(lines) - 1:
                source_lines.append(line)
            else:
                source_lines.append(line + '\n')
        
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source_lines
        }
    
    def create_code_cell(self, content: str) -> Dict[str, Any]:
        """Create a properly formatted code cell."""
        lines = content.split('\n')
        source_lines = []
        for i, line in enumerate(lines):
            if i == len(lines) - 1:
                source_lines.append(line)
            else:
                source_lines.append(line + '\n')
        
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source_lines
        }
    
    def convert_end_to_end_script(self):
        """Convert end-to-end pipeline script to notebook."""
        script_path = Path('end_to_end_pipeline_script.py')
        notebook_path = Path('04_end_to_end_pipeline_test.ipynb')
        
        print(f"Converting {script_path} to {notebook_path}")
        
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Parse the script
        tree = ast.parse(content)
        
        # Extract components
        functions = {}
        classes = {}
        imports = []
        
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(ast.get_source_segment(content, node))
            elif isinstance(node, ast.FunctionDef):
                func_code = ast.get_source_segment(content, node)
                functions[node.name] = func_code
            elif isinstance(node, ast.ClassDef):
                class_code = ast.get_source_segment(content, node)
                classes[node.name] = class_code
        
        # Create notebook
        notebook = self.notebook_template.copy()
        cells = []
        
        # Title
        title_content = """# XGBoost Pipeline Test - End-to-End Pipeline Execution

This notebook tests the complete XGBoost 3-step pipeline execution with proper dependency validation, data flow verification, and error handling.

**Pipeline Steps:**
1. XGBoost Training
2. XGBoost Model Evaluation
3. Model Calibration

**This notebook covers:**
- Complete pipeline execution
- Dependency validation
- Data flow verification
- Error handling and recovery
- Comprehensive result reporting"""
        
        cells.append(self.create_markdown_cell(title_content))
        
        # Setup section
        cells.append(self.create_markdown_cell("## 1. Setup and Environment"))
        
        imports_code = '\n'.join(imports)
        setup_code = """
# Setup environment
cursus_available = setup_environment()"""
        
        combined_setup = imports_code + setup_code
        cells.append(self.create_code_cell(combined_setup))
        
        if 'setup_environment' in functions:
            cells.append(self.create_code_cell(functions['setup_environment']))
        
        # Configuration section
        cells.append(self.create_markdown_cell("## 2. Load Pipeline Configuration"))
        if 'load_pipeline_configuration' in functions:
            cells.append(self.create_code_cell(functions['load_pipeline_configuration']))
        
        # Pipeline executor section
        cells.append(self.create_markdown_cell("## 3. End-to-End Pipeline Executor"))
        if 'EndToEndPipelineExecutor' in classes:
            cells.append(self.create_code_cell(classes['EndToEndPipelineExecutor']))
        
        # Execution section
        cells.append(self.create_markdown_cell("## 4. Run End-to-End Pipeline Test"))
        
        test_runner_code = """# Run the complete end-to-end pipeline test
success = run_end_to_end_pipeline_test()

# Verify outputs
outputs_verified = verify_pipeline_outputs()

# Display final results
if success and outputs_verified:
    print("🎉 End-to-end pipeline test PASSED!")
else:
    print("⚠ End-to-end pipeline test had issues!")"""
        
        cells.append(self.create_code_cell(test_runner_code))
        
        if 'run_end_to_end_pipeline_test' in functions:
            cells.append(self.create_code_cell(functions['run_end_to_end_pipeline_test']))
        
        # Results section
        cells.append(self.create_markdown_cell("## 5. Generate Pipeline Results"))
        if 'generate_pipeline_results' in functions:
            cells.append(self.create_code_cell(functions['generate_pipeline_results']))
        
        # Verification section
        cells.append(self.create_markdown_cell("## 6. Verify Pipeline Outputs"))
        if 'verify_pipeline_outputs' in functions:
            cells.append(self.create_code_cell(functions['verify_pipeline_outputs']))
        
        # Set cells and save
        notebook['cells'] = cells
        
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Successfully created notebook: {notebook_path}")
        print(f"✓ Created {len(cells)} cells with proper formatting")
    
    def convert_performance_analysis_script(self):
        """Convert performance analysis script to notebook."""
        script_path = Path('performance_analysis_script.py')
        notebook_path = Path('05_performance_analysis.ipynb')
        
        print(f"Converting {script_path} to {notebook_path}")
        
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Parse the script
        tree = ast.parse(content)
        
        # Extract components
        functions = {}
        classes = {}
        imports = []
        
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(ast.get_source_segment(content, node))
            elif isinstance(node, ast.FunctionDef):
                func_code = ast.get_source_segment(content, node)
                functions[node.name] = func_code
            elif isinstance(node, ast.ClassDef):
                class_code = ast.get_source_segment(content, node)
                classes[node.name] = class_code
        
        # Create notebook
        notebook = self.notebook_template.copy()
        cells = []
        
        # Title
        title_content = """# XGBoost Pipeline Test - Performance Analysis

This notebook analyzes the performance of the XGBoost 3-step pipeline execution, generates visualizations, and provides comprehensive reporting.

**Analysis Components:**
- Execution time analysis
- Success rate analysis
- Data flow analysis
- Performance visualizations
- Comprehensive reporting

**This notebook covers:**
- Performance metrics collection
- Visualization generation
- Comparative analysis
- Final reporting and insights"""
        
        cells.append(self.create_markdown_cell(title_content))
        
        # Setup section
        cells.append(self.create_markdown_cell("## 1. Setup and Environment"))
        
        imports_code = '\n'.join(imports)
        setup_code = """
# Setup environment
cursus_available = setup_environment()"""
        
        combined_setup = imports_code + setup_code
        cells.append(self.create_code_cell(combined_setup))
        
        if 'setup_environment' in functions:
            cells.append(self.create_code_cell(functions['setup_environment']))
        
        # Load results section
        cells.append(self.create_markdown_cell("## 2. Load Test Results"))
        if 'load_test_results' in functions:
            cells.append(self.create_code_cell(functions['load_test_results']))
        
        # Performance analyzer section
        cells.append(self.create_markdown_cell("## 3. Performance Analyzer"))
        if 'PerformanceAnalyzer' in classes:
            cells.append(self.create_code_cell(classes['PerformanceAnalyzer']))
        
        # Analysis execution section
        cells.append(self.create_markdown_cell("## 4. Run Performance Analysis"))
        
        analysis_runner_code = """# Run comprehensive performance analysis
success = run_performance_analysis()

if success:
    print("🎉 Performance analysis completed successfully!")
    print("📊 Visualizations generated!")
    print("📋 Reports saved!")
else:
    print("⚠ Performance analysis had issues!")"""
        
        cells.append(self.create_code_cell(analysis_runner_code))
        
        if 'run_performance_analysis' in functions:
            cells.append(self.create_code_cell(functions['run_performance_analysis']))
        
        # Results saving section
        cells.append(self.create_markdown_cell("## 5. Save Analysis Results"))
        if 'save_analysis_results' in functions:
            cells.append(self.create_code_cell(functions['save_analysis_results']))
        
        # Final report section
        cells.append(self.create_markdown_cell("## 6. Generate Final Report"))
        if 'generate_final_report' in functions:
            cells.append(self.create_code_cell(functions['generate_final_report']))
        
        # Set cells and save
        notebook['cells'] = cells
        
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Successfully created notebook: {notebook_path}")
        print(f"✓ Created {len(cells)} cells with proper formatting")

def main():
    """Main conversion function."""
    converter = CustomScriptConverter()
    
    print("Converting remaining scripts to notebooks...")
    print("=" * 50)
    
    # Convert end-to-end pipeline script
    try:
        converter.convert_end_to_end_script()
    except Exception as e:
        print(f"Error converting end-to-end script: {e}")
    
    print()
    
    # Convert performance analysis script
    try:
        converter.convert_performance_analysis_script()
    except Exception as e:
        print(f"Error converting performance analysis script: {e}")
    
    print("\nConversion completed!")

if __name__ == "__main__":
    main()
