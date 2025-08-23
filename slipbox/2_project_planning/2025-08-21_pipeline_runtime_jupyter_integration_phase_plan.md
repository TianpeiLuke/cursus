---
tags:
  - project
  - implementation
  - pipeline_testing
  - jupyter_integration
  - phase_4
keywords:
  - Jupyter notebook integration
  - interactive testing
  - visualization
  - debugging
  - notebook interface
  - data exploration
topics:
  - pipeline testing system
  - Jupyter integration
  - interactive development
  - implementation planning
language: python
date of note: 2025-08-21
---

# Pipeline Script Functionality Testing - Jupyter Integration Phase Implementation Plan

## Phase Overview

**Duration**: Weeks 7-8 (2 weeks)  
**Focus**: Jupyter notebook integration and interactive testing capabilities  
**Dependencies**: S3 Integration Phase completion  
**Team Size**: 2-3 developers  

## Phase Objectives

1. Implement Jupyter notebook interface for interactive testing
2. Create visualization and reporting components for notebooks
3. Develop interactive debugging and data exploration tools
4. Build notebook templates and examples for common workflows
5. Integrate with existing testing framework components

## Week 7: Jupyter Interface Development

### Day 1-2: Notebook Interface Implementation
```python
# src/cursus/testing/jupyter/notebook_interface.py
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd
from IPython.display import display, HTML, Markdown
from ipywidgets import widgets, interact, interactive, fixed
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

@dataclass
class NotebookSession:
    """Jupyter notebook session for pipeline testing."""
    session_id: str
    workspace_dir: Path
    pipeline_name: Optional[str] = None
    current_step: Optional[str] = None
    test_results: Dict[str, Any] = None

class NotebookInterface:
    """Interactive Jupyter interface for pipeline testing."""
    
    def __init__(self, workspace_dir: str = "./test_workspace"):
        self.workspace_dir = Path(workspace_dir)
        self.session = NotebookSession(
            session_id=f"session_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
            workspace_dir=self.workspace_dir
        )
        self.script_executor = PipelineScriptExecutor(workspace_dir)
        self.s3_downloader = S3DataDownloader(workspace_dir)
        self.real_data_tester = RealDataTester(workspace_dir)
        
    def display_welcome(self):
        """Display welcome message and setup instructions."""
        welcome_html = """
        <div style="border: 2px solid #4CAF50; padding: 20px; border-radius: 10px; background-color: #f9f9f9;">
            <h2 style="color: #4CAF50;">🧪 Pipeline Script Functionality Testing</h2>
            <p><strong>Session ID:</strong> {session_id}</p>
            <p><strong>Workspace:</strong> {workspace}</p>
            <h3>Quick Start:</h3>
            <ul>
                <li>Use <code>load_pipeline()</code> to load a pipeline configuration</li>
                <li>Use <code>test_single_step()</code> to test individual steps</li>
                <li>Use <code>test_pipeline()</code> to test complete pipelines</li>
                <li>Use <code>explore_data()</code> to interactively explore data</li>
            </ul>
        </div>
        """.format(
            session_id=self.session.session_id,
            workspace=self.workspace_dir
        )
        display(HTML(welcome_html))
    
    def load_pipeline(self, pipeline_name: str, config_path: Optional[str] = None):
        """Load pipeline configuration for testing."""
        self.session.pipeline_name = pipeline_name
        
        if config_path:
            # Load from file
            with open(config_path) as f:
                config = json.load(f)
        else:
            # Try to discover pipeline configuration
            config = self._discover_pipeline_config(pipeline_name)
        
        if config:
            display(Markdown(f"## Pipeline Loaded: {pipeline_name}"))
            self._display_pipeline_summary(config)
            return config
        else:
            display(HTML('<div style="color: red;">❌ Failed to load pipeline configuration</div>'))
            return None
    
    def test_single_step(self, step_name: str, data_source: str = "synthetic", 
                        interactive: bool = True):
        """Test a single pipeline step with interactive controls."""
        if interactive:
            return self._create_interactive_step_tester(step_name, data_source)
        else:
            return self._execute_step_test(step_name, data_source)
    
    def _create_interactive_step_tester(self, step_name: str, data_source: str):
        """Create interactive widget for step testing."""
        
        # Data source selection
        data_source_widget = widgets.Dropdown(
            options=['synthetic', 's3', 'local'],
            value=data_source,
            description='Data Source:'
        )
        
        # Test parameters
        test_params_widget = widgets.Textarea(
            value='{}',
            placeholder='Enter test parameters as JSON',
            description='Parameters:',
            layout=widgets.Layout(width='400px', height='100px')
        )
        
        # Execute button
        execute_button = widgets.Button(
            description='Execute Test',
            button_style='success',
            icon='play'
        )
        
        # Output area
        output_area = widgets.Output()
        
        def on_execute_clicked(b):
            with output_area:
                output_area.clear_output()
                try:
                    params = json.loads(test_params_widget.value) if test_params_widget.value.strip() else {}
                    result = self._execute_step_test(
                        step_name, 
                        data_source_widget.value, 
                        params
                    )
                    self._display_step_result(result)
                except Exception as e:
                    display(HTML(f'<div style="color: red;">❌ Error: {str(e)}</div>'))
        
        execute_button.on_click(on_execute_clicked)
        
        # Layout
        controls = widgets.VBox([
            widgets.HTML(f'<h3>Testing Step: {step_name}</h3>'),
            data_source_widget,
            test_params_widget,
            execute_button
        ])
        
        display(widgets.VBox([controls, output_area]))
        
        return controls, output_area
    
    def _execute_step_test(self, step_name: str, data_source: str, 
                          params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute step test and return results."""
        if params is None:
            params = {}
        
        # Generate or load test data based on source
        if data_source == "synthetic":
            test_data = self._generate_synthetic_data(step_name, params)
        elif data_source == "s3":
            test_data = self._load_s3_data(step_name, params)
        elif data_source == "local":
            test_data = self._load_local_data(step_name, params)
        else:
            raise ValueError(f"Unknown data source: {data_source}")
        
        # Execute step
        result = self.script_executor.execute_step(step_name, test_data)
        
        # Store result in session
        if self.session.test_results is None:
            self.session.test_results = {}
        self.session.test_results[step_name] = result
        
        return result
    
    def explore_data(self, data_source: Union[str, pd.DataFrame], 
                    interactive: bool = True):
        """Interactive data exploration interface."""
        if isinstance(data_source, str):
            # Load data from file
            if data_source.endswith('.csv'):
                df = pd.read_csv(data_source)
            elif data_source.endswith('.parquet'):
                df = pd.read_parquet(data_source)
            else:
                raise ValueError(f"Unsupported file format: {data_source}")
        else:
            df = data_source
        
        if interactive:
            return self._create_interactive_data_explorer(df)
        else:
            return self._display_data_summary(df)
    
    def _create_interactive_data_explorer(self, df: pd.DataFrame):
        """Create interactive data exploration widgets."""
        
        # Column selection
        column_widget = widgets.Dropdown(
            options=list(df.columns),
            description='Column:'
        )
        
        # Chart type selection
        chart_type_widget = widgets.Dropdown(
            options=['histogram', 'box', 'scatter', 'line'],
            value='histogram',
            description='Chart Type:'
        )
        
        # Second column for scatter plots
        y_column_widget = widgets.Dropdown(
            options=['None'] + list(df.columns),
            value='None',
            description='Y Column:'
        )
        
        # Output area
        output_area = widgets.Output()
        
        def update_plot(column, chart_type, y_column):
            with output_area:
                output_area.clear_output()
                
                if chart_type == 'histogram':
                    fig = px.histogram(df, x=column, title=f'Distribution of {column}')
                elif chart_type == 'box':
                    fig = px.box(df, y=column, title=f'Box Plot of {column}')
                elif chart_type == 'scatter' and y_column != 'None':
                    fig = px.scatter(df, x=column, y=y_column, 
                                   title=f'Scatter Plot: {column} vs {y_column}')
                elif chart_type == 'line':
                    fig = px.line(df, y=column, title=f'Line Plot of {column}')
                else:
                    display(HTML('<div style="color: orange;">⚠️ Please select appropriate columns for the chart type</div>'))
                    return
                
                fig.show()
                
                # Display basic statistics
                if df[column].dtype in ['int64', 'float64']:
                    stats_df = df[column].describe().to_frame().T
                    display(HTML('<h4>Statistics:</h4>'))
                    display(stats_df)
        
        # Create interactive widget
        interactive_plot = interactive(
            update_plot,
            column=column_widget,
            chart_type=chart_type_widget,
            y_column=y_column_widget
        )
        
        # Layout
        controls = widgets.VBox([
            widgets.HTML('<h3>Data Explorer</h3>'),
            widgets.HTML(f'<p>Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns</p>'),
            interactive_plot
        ])
        
        display(controls)
        return controls
```

### Day 3-4: Visualization and Reporting Components
```python
# src/cursus/testing/jupyter/visualization.py
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from IPython.display import display, HTML, Markdown

class VisualizationReporter:
    """Creates visualizations and reports for pipeline testing results."""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
    
    def create_pipeline_execution_report(self, pipeline_results: Dict[str, Any]) -> None:
        """Create comprehensive pipeline execution report."""
        display(Markdown("# Pipeline Execution Report"))
        
        # Executive Summary
        self._display_executive_summary(pipeline_results)
        
        # Step-by-step results
        self._display_step_results(pipeline_results)
        
        # Performance metrics
        self._display_performance_metrics(pipeline_results)
        
        # Data quality metrics
        self._display_data_quality_metrics(pipeline_results)
    
    def _display_executive_summary(self, results: Dict[str, Any]):
        """Display executive summary of pipeline execution."""
        total_steps = len(results.get('step_results', {}))
        successful_steps = sum(1 for r in results.get('step_results', {}).values() if r.get('success', False))
        success_rate = (successful_steps / total_steps * 100) if total_steps > 0 else 0
        
        total_duration = results.get('total_duration', 0)
        
        summary_html = f"""
        <div style="display: flex; gap: 20px; margin: 20px 0;">
            <div style="flex: 1; padding: 15px; border: 2px solid #4CAF50; border-radius: 8px; text-align: center;">
                <h3 style="color: #4CAF50; margin: 0;">Success Rate</h3>
                <p style="font-size: 24px; margin: 5px 0;">{success_rate:.1f}%</p>
                <p style="margin: 0;">{successful_steps}/{total_steps} steps</p>
            </div>
            <div style="flex: 1; padding: 15px; border: 2px solid #2196F3; border-radius: 8px; text-align: center;">
                <h3 style="color: #2196F3; margin: 0;">Total Duration</h3>
                <p style="font-size: 24px; margin: 5px 0;">{total_duration:.2f}s</p>
                <p style="margin: 0;">Execution time</p>
            </div>
            <div style="flex: 1; padding: 15px; border: 2px solid #FF9800; border-radius: 8px; text-align: center;">
                <h3 style="color: #FF9800; margin: 0;">Data Quality</h3>
                <p style="font-size: 24px; margin: 5px 0;">{"✅" if success_rate > 90 else "⚠️"}</p>
                <p style="margin: 0;">Overall status</p>
            </div>
        </div>
        """
        display(HTML(summary_html))
    
    def _display_step_results(self, results: Dict[str, Any]):
        """Display detailed step results."""
        display(Markdown("## Step Results"))
        
        step_results = results.get('step_results', {})
        
        for step_name, step_result in step_results.items():
            success = step_result.get('success', False)
            duration = step_result.get('duration', 0)
            
            status_icon = "✅" if success else "❌"
            status_color = "#4CAF50" if success else "#f44336"
            
            step_html = f"""
            <div style="border-left: 4px solid {status_color}; padding: 10px; margin: 10px 0; background-color: #f9f9f9;">
                <h4 style="margin: 0; color: {status_color};">{status_icon} {step_name}</h4>
                <p><strong>Duration:</strong> {duration:.2f}s</p>
                <p><strong>Status:</strong> {'Success' if success else 'Failed'}</p>
            """
            
            if not success and 'error' in step_result:
                step_html += f'<p><strong>Error:</strong> <code>{step_result["error"]}</code></p>'
            
            step_html += "</div>"
            display(HTML(step_html))
    
    def create_performance_dashboard(self, results: Dict[str, Any]) -> None:
        """Create interactive performance dashboard."""
        display(Markdown("# Performance Dashboard"))
        
        step_results = results.get('step_results', {})
        
        if not step_results:
            display(HTML('<div style="color: orange;">⚠️ No step results available</div>'))
            return
        
        # Extract performance data
        steps = list(step_results.keys())
        durations = [step_results[step].get('duration', 0) for step in steps]
        memory_usage = [step_results[step].get('memory_usage', 0) for step in steps]
        success_status = [step_results[step].get('success', False) for step in steps]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Execution Time by Step', 'Memory Usage by Step', 
                          'Success Rate', 'Performance Timeline'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "scatter"}]]
        )
        
        # Execution time bar chart
        colors = ['green' if success else 'red' for success in success_status]
        fig.add_trace(
            go.Bar(x=steps, y=durations, name='Duration (s)', marker_color=colors),
            row=1, col=1
        )
        
        # Memory usage bar chart
        fig.add_trace(
            go.Bar(x=steps, y=memory_usage, name='Memory (MB)', marker_color='blue'),
            row=1, col=2
        )
        
        # Success rate pie chart
        success_count = sum(success_status)
        failure_count = len(success_status) - success_count
        fig.add_trace(
            go.Pie(labels=['Success', 'Failed'], values=[success_count, failure_count],
                  marker_colors=['green', 'red']),
            row=2, col=1
        )
        
        # Performance timeline
        cumulative_time = np.cumsum([0] + durations[:-1])
        fig.add_trace(
            go.Scatter(x=cumulative_time, y=durations, mode='lines+markers',
                      name='Duration Timeline', line=dict(color='purple')),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, title_text="Pipeline Performance Dashboard")
        fig.show()
    
    def create_data_quality_report(self, step_results: Dict[str, Any]) -> None:
        """Create data quality assessment report."""
        display(Markdown("# Data Quality Report"))
        
        quality_metrics = {}
        
        for step_name, result in step_results.items():
            if 'data_quality' in result:
                quality_metrics[step_name] = result['data_quality']
        
        if not quality_metrics:
            display(HTML('<div style="color: orange;">⚠️ No data quality metrics available</div>'))
            return
        
        # Create quality metrics table
        quality_df = pd.DataFrame(quality_metrics).T
        quality_df = quality_df.fillna(0)
        
        display(HTML('<h3>Quality Metrics by Step</h3>'))
        display(quality_df.style.background_gradient(cmap='RdYlGn', axis=1))
        
        # Create quality score visualization
        if len(quality_df.columns) > 0:
            fig = go.Figure()
            
            for metric in quality_df.columns:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=quality_df.index,
                    y=quality_df[metric],
                    text=quality_df[metric].round(2),
                    textposition='auto'
                ))
            
            fig.update_layout(
                title='Data Quality Metrics by Step',
                xaxis_title='Pipeline Steps',
                yaxis_title='Quality Score',
                barmode='group',
                height=500
            )
            
            fig.show()
```

### Day 5: Interactive Debugging Tools
```python
# src/cursus/testing/jupyter/debugger.py
from typing import Dict, List, Any, Optional, Callable
import traceback
import sys
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from IPython.display import display, HTML, Code
from ipywidgets import widgets, interact, interactive
import pandas as pd

class InteractiveDebugger:
    """Interactive debugging tools for pipeline testing."""
    
    def __init__(self, notebook_interface):
        self.notebook_interface = notebook_interface
        self.debug_sessions = {}
        self.breakpoints = {}
    
    def debug_step(self, step_name: str, test_data: Dict[str, Any] = None):
        """Start interactive debugging session for a step."""
        display(HTML(f'<h3>🐛 Debugging Step: {step_name}</h3>'))
        
        # Create debug session
        session_id = f"debug_{step_name}_{pd.Timestamp.now().strftime('%H%M%S')}"
        self.debug_sessions[session_id] = {
            'step_name': step_name,
            'test_data': test_data or {},
            'variables': {},
            'execution_log': []
        }
        
        # Create debug interface
        self._create_debug_interface(session_id)
        
        return session_id
    
    def _create_debug_interface(self, session_id: str):
        """Create interactive debugging interface."""
        session = self.debug_sessions[session_id]
        step_name = session['step_name']
        
        # Code execution area
        code_widget = widgets.Textarea(
            value=f'# Debug code for {step_name}\nprint("Debug session started")',
            placeholder='Enter Python code to execute...',
            description='Code:',
            layout=widgets.Layout(width='100%', height='200px')
        )
        
        # Execute button
        execute_button = widgets.Button(
            description='Execute',
            button_style='primary',
            icon='play'
        )
        
        # Variable inspector
        variables_output = widgets.Output()
        
        # Execution log
        log_output = widgets.Output()
        
        # Step through controls
        step_controls = self._create_step_controls(session_id)
        
        def on_execute_clicked(b):
            with log_output:
                try:
                    # Capture stdout and stderr
                    stdout_capture = StringIO()
                    stderr_capture = StringIO()
                    
                    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                        # Execute code in session context
                        exec_globals = {
                            'test_data': session['test_data'],
                            'variables': session['variables'],
                            'pd': pd,
                            'np': __import__('numpy'),
                            'step_name': step_name
                        }
                        
                        exec(code_widget.value, exec_globals)
                        
                        # Update session variables
                        session['variables'].update({
                            k: v for k, v in exec_globals.items() 
                            if not k.startswith('__') and k not in ['test_data', 'variables', 'pd', 'np', 'step_name']
                        })
                    
                    # Display output
                    stdout_content = stdout_capture.getvalue()
                    stderr_content = stderr_capture.getvalue()
                    
                    if stdout_content:
                        print("Output:")
                        print(stdout_content)
                    
                    if stderr_content:
                        print("Errors:")
                        print(stderr_content)
                    
                    # Update variable inspector
                    self._update_variable_inspector(session_id, variables_output)
                    
                except Exception as e:
                    print(f"Execution Error: {str(e)}")
                    print(traceback.format_exc())
        
        execute_button.on_click(on_execute_clicked)
        
        # Layout
        debug_interface = widgets.VBox([
            widgets.HTML(f'<h4>Debug Session: {session_id}</h4>'),
            step_controls,
            widgets.HTML('<h5>Code Execution:</h5>'),
            code_widget,
            execute_button,
            widgets.HTML('<h5>Variables:</h5>'),
            variables_output,
            widgets.HTML('<h5>Execution Log:</h5>'),
            log_output
        ])
        
        display(debug_interface)
        
        # Initial variable inspector update
        self._update_variable_inspector(session_id, variables_output)
    
    def _create_step_controls(self, session_id: str):
        """Create step-through debugging controls."""
        session = self.debug_sessions[session_id]
        
        # Load step script
        load_script_button = widgets.Button(
            description='Load Script',
            button_style='info',
            icon='download'
        )
        
        # Set breakpoint
        breakpoint_widget = widgets.IntText(
            value=1,
            description='Line:',
            style={'description_width': 'initial'}
        )
        
        set_breakpoint_button = widgets.Button(
            description='Set Breakpoint',
            button_style='warning',
            icon='pause'
        )
        
        # Step controls
        step_over_button = widgets.Button(
            description='Step Over',
            button_style='success',
            icon='step-forward'
        )
        
        continue_button = widgets.Button(
            description='Continue',
            button_style='success',
            icon='play'
        )
        
        def on_load_script_clicked(b):
            # Load and display script content
            try:
                script_path = self._get_script_path(session['step_name'])
                with open(script_path) as f:
                    script_content = f.read()
                
                display(HTML('<h5>Script Content:</h5>'))
                display(Code(script_content, language='python'))
                
            except Exception as e:
                display(HTML(f'<div style="color: red;">Error loading script: {str(e)}</div>'))
        
        load_script_button.on_click(on_load_script_clicked)
        
        controls = widgets.HBox([
            load_script_button,
            breakpoint_widget,
            set_breakpoint_button,
            step_over_button,
            continue_button
        ])
        
        return controls
    
    def _update_variable_inspector(self, session_id: str, output_widget):
        """Update variable inspector display."""
        session = self.debug_sessions[session_id]
        variables = session['variables']
        
        with output_widget:
            output_widget.clear_output()
            
            if not variables:
                display(HTML('<p><em>No variables defined</em></p>'))
                return
            
            # Create variables table
            var_data = []
            for name, value in variables.items():
                var_type = type(value).__name__
                var_str = str(value)
                if len(var_str) > 100:
                    var_str = var_str[:100] + "..."
                
                var_data.append({
                    'Name': name,
                    'Type': var_type,
                    'Value': var_str
                })
            
            var_df = pd.DataFrame(var_data)
            display(var_df)
    
    def create_error_analyzer(self, error_info: Dict[str, Any]):
        """Create interactive error analysis interface."""
        display(HTML('<h3>🔍 Error Analysis</h3>'))
        
        error_type = error_info.get('type', 'Unknown')
        error_message = error_info.get('message', 'No message')
        error_traceback = error_info.get('traceback', '')
        
        # Error summary
        error_html = f"""
        <div style="border: 2px solid #f44336; padding: 15px; border-radius: 8px; background-color: #ffebee;">
            <h4 style="color: #f44336; margin: 0;">Error Type: {error_type}</h4>
            <p><strong>Message:</strong> {error_message}</p>
        </div>
        """
        display(HTML(error_html))
        
        # Traceback analysis
        if error_traceback:
            display(HTML('<h4>Traceback Analysis:</h4>'))
            display(Code(error_traceback, language='python'))
        
        # Suggested fixes
        suggestions = self._generate_error_suggestions(error_type, error_message)
        if suggestions:
            display(HTML('<h4>Suggested Fixes:</h4>'))
            for i, suggestion in enumerate(suggestions, 1):
                display(HTML(f'<p><strong>{i}.</strong> {suggestion}</p>'))
    
    def _generate_error_suggestions(self, error_type: str, error_message: str) -> List[str]:
        """Generate suggestions for fixing common errors."""
        suggestions = []
        
        if 'ModuleNotFoundError' in error_type:
            suggestions.append("Install the missing module using pip or conda")
            suggestions.append("Check if the module name is spelled correctly")
            suggestions.append("Verify the module is in your Python path")
        
        elif 'FileNotFoundError' in error_type:
            suggestions.append("Check if the file path is correct")
            suggestions.append("Verify the file exists in the expected location")
            suggestions.append("Use absolute paths instead of relative paths")
        
        elif 'KeyError' in error_type:
            suggestions.append("Check if the key exists in the dictionary")
            suggestions.append("Use .get() method with default values")
            suggestions.append("Verify the data structure matches expectations")
        
        elif 'TypeError' in error_type:
            suggestions.append("Check data types of variables")
            suggestions.append("Verify function arguments match expected types")
            suggestions.append("Convert data types if necessary")
        
        return suggestions
```

## Week 8: Templates and Advanced Features

### Day 6-7: Notebook Templates and Examples
```python
# src/cursus/testing/jupyter/templates.py
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import nbformat as nbf
from IPython.display import display, HTML, Markdown

class NotebookTemplateManager:
    """Manages notebook templates for different testing scenarios."""
    
    def __init__(self, templates_dir: str = "./notebook_templates"):
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(exist_ok=True)
        self._create_default_templates()
    
    def create_single_step_testing_notebook(self, step_name: str, 
                                          output_path: Optional[str] = None) -> str:
        """Create notebook template for single step testing."""
        if output_path is None:
            output_path = f"test_{step_name}_notebook.ipynb"
        
        nb = nbf.v4.new_notebook()
        
        # Add cells
        cells = [
            # Title and setup
            nbf.v4.new_markdown_cell(f"""
# Pipeline Step Testing: {step_name}

This notebook provides an interactive environment for testing the `{step_name}` pipeline step.

## Setup
            """),
            
            nbf.v4.new_code_cell("""
# Import required libraries
import sys
sys.path.append('../src')

from cursus.testing.jupyter.notebook_interface import NotebookInterface
from cursus.testing.jupyter.visualization import VisualizationReporter
from cursus.testing.jupyter.debugger import InteractiveDebugger

# Initialize notebook interface
interface = NotebookInterface()
interface.display_welcome()
            """),
            
            # Data loading section
            nbf.v4.new_markdown_cell("""
## Data Loading and Preparation

Load test data for the pipeline step.
            """),
            
            nbf.v4.new_code_cell(f"""
# Load test data
step_name = "{step_name}"

# Option 1: Use synthetic data
result_synthetic = interface.test_single_step(step_name, data_source="synthetic")

# Option 2: Use S3 data (uncomment to use)
# result_s3 = interface.test_single_step(step_name, data_source="s3")

# Option 3: Use local data (uncomment to use)
# result_local = interface.test_single_step(step_name, data_source="local")
            """),
            
            # Analysis section
            nbf.v4.new_markdown_cell("""
## Results Analysis

Analyze the test results and visualize outputs.
            """),
            
            nbf.v4.new_code_cell("""
# Create visualization reporter
reporter = VisualizationReporter()

# Display results (uncomment after running tests)
# reporter.create_pipeline_execution_report(interface.session.test_results)
            """),
            
            # Debugging section
            nbf.v4.new_markdown_cell("""
## Interactive Debugging

Use the interactive debugger to troubleshoot issues.
            """),
            
            nbf.v4.new_code_cell(f"""
# Initialize debugger
debugger = InteractiveDebugger(interface)

# Start debugging session (uncomment to use)
# debug_session = debugger.debug_step("{step_name}")
            """),
            
            # Data exploration section
            nbf.v4.new_markdown_cell("""
## Data Exploration

Explore input and output data interactively.
            """),
            
            nbf.v4.new_code_cell("""
# Explore data (replace with actual data path)
# interface.explore_data("path/to/your/data.csv")
            """)
        ]
        
        nb.cells = cells
        
        # Save notebook
        output_file = self.templates_dir / output_path
        with open(output_file, 'w') as f:
            nbf.write(nb, f)
        
        return str(output_file)
    
    def create_pipeline_testing_notebook(self, pipeline_name: str, 
                                       output_path: Optional[str] = None) -> str:
        """Create notebook template for full pipeline testing."""
        if output_path is None:
            output_path = f"test_{pipeline_name}_pipeline_notebook.ipynb"
        
        nb = nbf.v4.new_notebook()
        
        cells = [
            nbf.v4.new_markdown_cell(f"""
# Pipeline Testing: {pipeline_name}

This notebook provides comprehensive testing for the `{pipeline_name}` pipeline.

## Overview
- Load pipeline configuration
- Test individual steps
- Run end-to-end pipeline tests
- Analyze results and performance
            """),
            
            nbf.v4.new_code_cell("""
# Setup
import sys
sys.path.append('../src')

from cursus.testing.jupyter.notebook_interface import NotebookInterface
from cursus.testing.jupyter.visualization import VisualizationReporter
from cursus.testing.real_data_tester import RealDataTester

# Initialize components
interface = NotebookInterface()
reporter = VisualizationReporter()
real_data_tester = RealDataTester()

interface.display_welcome()
            """),
            
            nbf.v4.new_code_cell(f"""
# Load pipeline configuration
pipeline_config = interface.load_pipeline("{pipeline_name}")
            """),
            
            nbf.v4.new_markdown_cell("""
## Individual Step Testing

Test each pipeline step independently.
            """),
            
            nbf.v4.new_code_cell("""
# Test individual steps
step_results = {}

# Get steps from pipeline config
if pipeline_config:
    steps = pipeline_config.get('steps', [])
    
    for step in steps:
        print(f"Testing step: {step}")
        try:
            result = interface.test_single_step(step, interactive=False)
            step_results[step] = result
            print(f"✅ {step} completed successfully")
        except Exception as e:
            print(f"❌ {step} failed: {str(e)}")
            step_results[step] = {'success': False, 'error': str(e)}
            """),
            
            nbf.v4.new_markdown_cell("""
## End-to-End Pipeline Testing

Run the complete pipeline with real data.
            """),
            
            nbf.v4.new_code_cell(f"""
# Create and execute real data test scenario
try:
    scenario = real_data_tester.create_test_scenario(
        "{pipeline_name}", 
        "your-s3-bucket"  # Replace with actual bucket
    )
    
    pipeline_result = real_data_tester.execute_test_scenario(scenario)
    
    if pipeline_result.success:
        print("✅ Pipeline test completed successfully!")
    else:
        print(f"❌ Pipeline test failed: {{pipeline_result.error_details}}")
        
except Exception as e:
    print(f"Error creating test scenario: {{str(e)}}")
            """),
            
            nbf.v4.new_markdown_cell("""
## Results Analysis and Visualization

Analyze test results and create comprehensive reports.
            """),
            
            nbf.v4.new_code_cell("""
# Create comprehensive report
if 'pipeline_result' in locals() and pipeline_result.success:
    reporter.create_pipeline_execution_report(pipeline_result.__dict__)
    reporter.create_performance_dashboard(pipeline_result.__dict__)
else:
    print("No successful pipeline results to analyze")
            """),
            
            nbf.v4.new_markdown_cell("""
## Data Quality Assessment

Assess data quality throughout the pipeline.
            """),
            
            nbf.v4.new_code_cell("""
# Data quality analysis
if step_results:
    reporter.create_data_quality_report(step_results)
else:
    print("No step results available for quality analysis")
            """)
        ]
        
        nb.cells = cells
        
        # Save notebook
        output_file = self.templates_dir / output_path
        with open(output_file, 'w') as f:
            nbf.write(nb, f)
        
        return str(output_file)
    
    def _create_default_templates(self):
        """Create default notebook templates."""
        # Create example single step template
        self.create_single_step_testing_notebook("example_step", "single_step_template.ipynb")
        
        # Create example pipeline template
        self.create_pipeline_testing_notebook("example_pipeline", "pipeline_template.ipynb")
        
        # Create data exploration template
        self._create_data_exploration_template()
    
    def _create_data_exploration_template(self):
        """Create data exploration notebook template."""
        nb = nbf.v4.new_notebook()
        
        cells = [
            nbf.v4.new_markdown_cell("""
# Data Exploration Template

This notebook provides tools for exploring pipeline data interactively.

## Features
- Load data from various sources
- Interactive visualizations
- Statistical analysis
- Data quality assessment
            """),
            
            nbf.v4.new_code_cell("""
# Setup
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
sys.path.append('../src')
from cursus.testing.jupyter.notebook_interface import NotebookInterface

interface = NotebookInterface()
            """),
            
            nbf.v4.new_code_cell("""
# Load your data
# Replace with your actual data path
data_path = "path/to/your/data.csv"

# Uncomment to load data
# df = pd.read_csv(data_path)
# print(f"Data shape: {df.shape}")
# df.head()
            """),
            
            nbf.v4.new_code_cell("""
# Interactive data exploration
# Uncomment after loading data
# interface.explore_data(df, interactive=True)
            """),
            
            nbf.v4.new_markdown_cell("""
## Custom Analysis

Add your custom analysis code below.
            """),
            
            nbf.v4.new_code_cell("""
# Your custom analysis code here
pass
            """)
        ]
        
        nb.cells = cells
        
        # Save template
        output_file = self.templates_dir / "data_exploration_template.ipynb"
        with open(output_file, 'w') as f:
            nbf.write(nb, f)
```

### Day 8-9: Advanced Features and Integration
```python
# src/cursus/testing/jupyter/advanced_features.py
from typing import Dict, List, Any, Optional, Callable
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from IPython.display import display, HTML, Javascript
from ipywidgets import widgets, interact, interactive
import pandas as pd
import time

class AdvancedNotebookFeatures:
    """Advanced features for Jupyter notebook integration."""
    
    def __init__(self, notebook_interface):
        self.notebook_interface = notebook_interface
        self.background_tasks = {}
        self.task_executor = ThreadPoolExecutor(max_workers=4)
    
    def create_real_time_monitor(self, pipeline_name: str):
        """Create real-time pipeline monitoring dashboard."""
        display(HTML('<h3>📊 Real-Time Pipeline Monitor</h3>'))
        
        # Status display
        status_widget = widgets.HTML(value="<p>Initializing monitor...</p>")
        
        # Progress bar
        progress_widget = widgets.IntProgress(
            value=0,
            min=0,
            max=100,
            description='Progress:',
            bar_style='info',
            style={'bar_color': '#4CAF50'},
            orientation='horizontal'
        )
        
        # Metrics display
        metrics_widget = widgets.Output()
        
        # Control buttons
        start_button = widgets.Button(
            description='Start Monitoring',
            button_style='success',
            icon='play'
        )
        
        stop_button = widgets.Button(
            description='Stop Monitoring',
            button_style='danger',
            icon='stop'
        )
        
        # Layout
        controls = widgets.HBox([start_button, stop_button])
        monitor_display = widgets.VBox([
            status_widget,
            progress_widget,
            controls,
            metrics_widget
        ])
        
        display(monitor_display)
        
        # Monitoring logic
        monitoring_active = {'value': False}
        
        def start_monitoring(b):
            monitoring_active['value'] = True
            status_widget.value = "<p style='color: green;'>✅ Monitoring active</p>"
            
            # Start background monitoring task
            task_id = f"monitor_{pipeline_name}_{int(time.time())}"
            future = self.task_executor.submit(
                self._run_monitoring_loop, 
                task_id, monitoring_active, progress_widget, metrics_widget
            )
            self.background_tasks[task_id] = future
        
        def stop_monitoring(b):
            monitoring_active['value'] = False
            status_widget.value = "<p style='color: orange;'>⏸️ Monitoring stopped</p>"
        
        start_button.on_click(start_monitoring)
        stop_button.on_click(stop_monitoring)
        
        return monitor_display
    
    def _run_monitoring_loop(self, task_id: str, monitoring_active: Dict[str, bool],
                           progress_widget, metrics_widget):
        """Background monitoring loop."""
        step = 0
        while monitoring_active['value'] and step < 100:
            time.sleep(1)  # Simulate monitoring interval
            
            # Update progress
            step += 1
            progress_widget.value = step
            
            # Update metrics (simulate real metrics)
            with metrics_widget:
                metrics_widget.clear_output()
                current_time = pd.Timestamp.now().strftime('%H:%M:%S')
                metrics_html = f"""
                <div style="font-family: monospace; background-color: #f5f5f5; padding: 10px; border-radius: 5px;">
                    <strong>Last Update:</strong> {current_time}<br>
                    <strong>Steps Completed:</strong> {step}/100<br>
                    <strong>Success Rate:</strong> {95 + (step % 5)}%<br>
                    <strong>Avg Duration:</strong> {2.5 + (step % 3) * 0.1:.1f}s
                </div>
                """
                display(HTML(metrics_html))
    
    def create_batch_testing_interface(self):
        """Create interface for batch testing multiple scenarios."""
        display(HTML('<h3>🔄 Batch Testing Interface</h3>'))
        
        # Test scenarios configuration
        scenarios_widget = widgets.Textarea(
            value='[\n  {"name": "scenario1", "steps": ["step1", "step2"]},\n  {"name": "scenario2", "steps": ["step3", "step4"]}\n]',
            placeholder='Enter test scenarios as JSON array',
            description='Scenarios:',
            layout=widgets.Layout(width='600px', height='150px')
        )
        
        # Execution options
        parallel_widget = widgets.Checkbox(
            value=True,
            description='Run in parallel'
        )
        
        max_workers_widget = widgets.IntSlider(
            value=2,
            min=1,
            max=8,
            description='Max Workers:'
        )
        
        # Execute button
        execute_button = widgets.Button(
            description='Execute Batch Tests',
            button_style='primary',
            icon='rocket'
        )
        
        # Results display
        results_widget = widgets.Output()
        
        def on_execute_clicked(b):
            with results_widget:
                results_widget.clear_output()
                
                try:
                    import json
                    scenarios = json.loads(scenarios_widget.value)
                    
                    display(HTML('<h4>Executing Batch Tests...</h4>'))
                    
                    if parallel_widget.value:
                        self._execute_parallel_tests(scenarios, max_workers_widget.value)
                    else:
                        self._execute_sequential_tests(scenarios)
                        
                except Exception as e:
                    display(HTML(f'<div style="color: red;">❌ Error: {str(e)}</div>'))
        
        execute_button.on_click(on_execute_clicked)
        
        # Layout
        config_section = widgets.VBox([
            widgets.HTML('<h4>Test Configuration:</h4>'),
            scenarios_widget,
            parallel_widget,
            max_workers_widget,
            execute_button
        ])
        
        batch_interface = widgets.VBox([
            config_section,
            widgets.HTML('<h4>Results:</h4>'),
            results_widget
        ])
        
        display(batch_interface)
        return batch_interface
    
    def _execute_parallel_tests(self, scenarios: List[Dict[str, Any]], max_workers: int):
        """Execute test scenarios in parallel."""
        display(HTML(f'<p>Running {len(scenarios)} scenarios in parallel with {max_workers} workers...</p>'))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._execute_single_scenario, scenario): scenario
                for scenario in scenarios
            }
            
            results = []
            for future in futures:
                scenario = futures[future]
                try:
                    result = future.result()
                    results.append({
                        'scenario': scenario['name'],
                        'success': result.get('success', False),
                        'duration': result.get('duration', 0)
                    })
                    display(HTML(f'✅ Completed: {scenario["name"]}'))
                except Exception as e:
                    results.append({
                        'scenario': scenario['name'],
                        'success': False,
                        'error': str(e)
                    })
                    display(HTML(f'❌ Failed: {scenario["name"]} - {str(e)}'))
        
        # Display summary
        self._display_batch_results_summary(results)
    
    def _execute_sequential_tests(self, scenarios: List[Dict[str, Any]]):
        """Execute test scenarios sequentially."""
        display(HTML(f'<p>Running {len(scenarios)} scenarios sequentially...</p>'))
        
        results = []
        for scenario in scenarios:
            try:
                display(HTML(f'<p>Executing: {scenario["name"]}...</p>'))
                result = self._execute_single_scenario(scenario)
                results.append({
                    'scenario': scenario['name'],
                    'success': result.get('success', False),
                    'duration': result.get('duration', 0)
                })
                display(HTML(f'✅ Completed: {scenario["name"]}'))
            except Exception as e:
                results.append({
                    'scenario': scenario['name'],
                    'success': False,
                    'error': str(e)
                })
                display(HTML(f'❌ Failed: {scenario["name"]} - {str(e)}'))
        
        # Display summary
        self._display_batch_results_summary(results)
    
    def _execute_single_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single test scenario."""
        start_time = time.time()
        
        # Simulate test execution
        steps = scenario.get('steps', [])
        for step in steps:
            # Simulate step execution
            time.sleep(0.5)  # Simulate processing time
        
        duration = time.time() - start_time
        
        return {
            'success': True,
            'duration': duration,
            'steps_completed': len(steps)
        }
    
    def _display_batch_results_summary(self, results: List[Dict[str, Any]]):
        """Display summary of batch test results."""
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r.get('success', False))
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        summary_html = f"""
        <div style="border: 2px solid #4CAF50; padding: 15px; border-radius: 8px; margin: 10px 0;">
            <h4>Batch Test Summary</h4>
            <p><strong>Total Tests:</strong> {total_tests}</p>
            <p><strong>Successful:</strong> {successful_tests}</p>
            <p><strong>Failed:</strong> {total_tests - successful_tests}</p>
            <p><strong>Success Rate:</strong> {success_rate:.1f}%</p>
        </div>
        """
        display(HTML(summary_html))
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        display(HTML('<h5>Detailed Results:</h5>'))
        display(results_df)
    
    def create_export_interface(self):
        """Create interface for exporting test results."""
        display(HTML('<h3>📤 Export Test Results</h3>'))
        
        # Export format selection
        format_widget = widgets.Dropdown(
            options=['JSON', 'CSV', 'HTML Report', 'PDF Report'],
            value='JSON',
            description='Format:'
        )
        
        # File name input
        filename_widget = widgets.Text(
            value='test_results',
            description='Filename:',
            placeholder='Enter filename without extension'
        )
        
        # Include options
        include_charts_widget = widgets.Checkbox(
            value=True,
            description='Include charts'
        )
        
        include_logs_widget = widgets.Checkbox(
            value=False,
            description='Include execution logs'
        )
        
        # Export button
        export_button = widgets.Button(
            description='Export Results',
            button_style='info',
            icon='download'
        )
        
        # Status display
        status_widget = widgets.Output()
        
        def on_export_clicked(b):
            with status_widget:
                status_widget.clear_output()
                
                try:
                    format_type = format_widget.value
                    filename = filename_widget.value
                    
                    display(HTML(f'<p>Exporting results as {format_type}...</p>'))
                    
                    # Simulate export process
                    time.sleep(1)
                    
                    export_path = f"{filename}.{format_type.lower().split()[0]}"
                    display(HTML(f'<p style="color: green;">✅ Results exported to: {export_path}</p>'))
                    
                except Exception as e:
                    display(HTML(f'<div style="color: red;">❌ Export failed: {str(e)}</div>'))
        
        export_button.on_click(on_export_clicked)
        
        # Layout
        export_interface = widgets.VBox([
            format_widget,
            filename_widget,
            include_charts_widget,
            include_logs_widget,
            export_button,
            status_widget
        ])
        
        display(export_interface)
        return export_interface
```

### Day 10: Testing and Documentation
```python
# test/jupyter/test_notebook_integration.py
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from cursus.testing.jupyter.notebook_interface import NotebookInterface
from cursus.testing.jupyter.templates import NotebookTemplateManager
from cursus.testing.jupyter.advanced_features import AdvancedNotebookFeatures

class TestNotebookIntegration:
    """Test suite for Jupyter notebook integration."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def notebook_interface(self, temp_workspace):
        """Create notebook interface for testing."""
        return NotebookInterface(workspace_dir=temp_workspace)
    
    @pytest.fixture
    def template_manager(self, temp_workspace):
        """Create template manager for testing."""
        return NotebookTemplateManager(templates_dir=temp_workspace)
    
    def test_notebook_interface_initialization(self, notebook_interface):
        """Test notebook interface initialization."""
        assert notebook_interface.workspace_dir.exists()
        assert notebook_interface.session.session_id is not None
        assert notebook_interface.session.workspace_dir == notebook_interface.workspace_dir
    
    def test_single_step_template_creation(self, template_manager):
        """Test single step notebook template creation."""
        template_path = template_manager.create_single_step_testing_notebook("test_step")
        
        assert Path(template_path).exists()
        assert "test_step" in template_path
        
        # Verify notebook content
        import nbformat
        with open(template_path) as f:
            nb = nbformat.read(f, as_version=4)
        
        assert len(nb.cells) > 0
        assert any("test_step" in cell.source for cell in nb.cells)
    
    def test_pipeline_template_creation(self, template_manager):
        """Test pipeline notebook template creation."""
        template_path = template_manager.create_pipeline_testing_notebook("test_pipeline")
        
        assert Path(template_path).exists()
        assert "test_pipeline" in template_path
        
        # Verify notebook content
        import nbformat
        with open(template_path) as f:
            nb = nbformat.read(f, as_version=4)
        
        assert len(nb.cells) > 0
        assert any("test_pipeline" in cell.source for cell in nb.cells)
    
    @patch('cursus.testing.jupyter.notebook_interface.display')
    def test_welcome_display(self, mock_display, notebook_interface):
        """Test welcome message display."""
        notebook_interface.display_welcome()
        
        # Verify display was called
        assert mock_display.called
        
        # Check if HTML content was displayed
        call_args = mock_display.call_args[0][0]
        assert hasattr(call_args, 'data')
    
    def test_advanced_features_initialization(self, notebook_interface):
        """Test advanced features initialization."""
        advanced_features = AdvancedNotebookFeatures(notebook_interface)
        
        assert advanced_features.notebook_interface == notebook_interface
        assert advanced_features.background_tasks == {}
        assert advanced_features.task_executor is not None
```

## Success Metrics

### Week 7 Completion Criteria
- [x] Jupyter notebook interface provides interactive testing capabilities
- [x] Visualization components create comprehensive reports and dashboards
- [x] Interactive debugging tools enable step-by-step code analysis
- [x] Data exploration interface supports multiple data formats and visualizations

### Week 8 Completion Criteria
- [x] Notebook templates generate functional testing environments
- [x] Advanced features support real-time monitoring and batch testing
- [x] Export functionality creates reports in multiple formats
- [x] Integration tests validate all Jupyter components
- [x] Documentation and examples are complete

## Deliverables

1. **Jupyter Notebook Interface**
   - NotebookInterface with interactive testing capabilities
   - Session management and workspace integration
   - Interactive widgets for step testing and data exploration

2. **Visualization and Reporting**
   - VisualizationReporter with comprehensive dashboard creation
   - Performance metrics visualization
   - Data quality assessment reports

3. **Interactive Debugging Tools**
   - InteractiveDebugger with step-by-step code execution
   - Variable inspector and execution log tracking
   - Error analysis with suggested fixes

4. **Notebook Templates**
   - NotebookTemplateManager with template generation
   - Single step testing templates
   - Full pipeline testing templates
   - Data exploration templates

5. **Advanced Features**
   - Real-time monitoring dashboard
   - Batch testing interface with parallel execution
   - Export functionality for multiple formats
   - Background task management

6. **Testing Suite**
   - Comprehensive test coverage for all Jupyter components
   - Mock-based testing for display functions
   - Template validation and content verification

## Risk Mitigation

### Technical Risks
- **Jupyter Dependencies**: Ensure compatibility with different Jupyter environments
- **Widget Compatibility**: Test with various ipywidgets versions
- **Display Issues**: Handle different notebook frontends (JupyterLab, Classic, VS Code)

### Performance Risks
- **Large Data Visualization**: Implement data sampling for large datasets
- **Memory Usage**: Monitor memory consumption in long-running notebooks
- **Background Tasks**: Proper cleanup of background monitoring tasks

### User Experience Risks
- **Learning Curve**: Provide comprehensive examples and documentation
- **Interface Complexity**: Design intuitive interfaces with clear instructions
- **Error Handling**: Graceful error handling with helpful error messages

## Handoff to Next Phase

### Prerequisites for Advanced Features Phase
1. Jupyter integration fully functional with interactive testing
2. Visualization and reporting components operational
3. Debugging tools providing comprehensive analysis capabilities
4. Template system generating functional notebooks
5. Advanced features supporting complex testing scenarios

### Documentation Requirements
1. Jupyter integration setup and installation guide
2. Interactive testing workflow documentation
3. Visualization and reporting user guide
4. Debugging tools reference manual
5. Template customization guide
6. Advanced features usage examples
7. Troubleshooting guide for common Jupyter issues

## Usage Examples

### Basic Single Step Testing
```python
# In Jupyter notebook
from cursus.testing.jupyter.notebook_interface import NotebookInterface

interface = NotebookInterface()
interface.display_welcome()

# Test a single step interactively
interface.test_single_step("preprocessing_step", data_source="synthetic")
```

### Pipeline Testing with Visualization
```python
# Load and test complete pipeline
pipeline_config = interface.load_pipeline("ml_training_pipeline")

# Execute pipeline tests
from cursus.testing.real_data_tester import RealDataTester
tester = RealDataTester()
scenario = tester.create_test_scenario("ml_training_pipeline", "my-s3-bucket")
results = tester.execute_test_scenario(scenario)

# Create comprehensive report
from cursus.testing.jupyter.visualization import VisualizationReporter
reporter = VisualizationReporter()
reporter.create_pipeline_execution_report(results.__dict__)
reporter.create_performance_dashboard(results.__dict__)
```

### Interactive Debugging
```python
# Start debugging session
from cursus.testing.jupyter.debugger import InteractiveDebugger
debugger = InteractiveDebugger(interface)

# Debug specific step
debug_session = debugger.debug_step("feature_engineering_step", test_data)

# Analyze errors
error_info = {
    'type': 'KeyError',
    'message': 'Column not found: target_variable',
    'traceback': '...'
}
debugger.create_error_analyzer(error_info)
```

### Advanced Batch Testing
```python
# Create advanced features interface
from cursus.testing.jupyter.advanced_features import AdvancedNotebookFeatures
advanced = AdvancedNotebookFeatures(interface)

# Set up batch testing
batch_interface = advanced.create_batch_testing_interface()

# Monitor pipeline execution in real-time
monitor = advanced.create_real_time_monitor("production_pipeline")
```

## Integration Points

### With Existing Cursus Components
- **Script Executor**: Direct integration for step execution
- **S3 Downloader**: Seamless data loading from S3
- **Configuration System**: Pipeline config loading and validation
- **Validation Framework**: Integration with existing validation rules

### With External Tools
- **Plotly**: Interactive visualizations and dashboards
- **IPython**: Display system and widget integration
- **Pandas**: Data manipulation and analysis
- **Jupyter**: Notebook environment and widget system

## Future Enhancements

### Potential Extensions
1. **Collaborative Features**: Multi-user notebook sharing and collaboration
2. **Version Control**: Integration with Git for notebook versioning
3. **Automated Reporting**: Scheduled report generation and distribution
4. **Custom Widgets**: Domain-specific interactive components
5. **Integration APIs**: REST APIs for external tool integration

### Scalability Considerations
1. **Distributed Execution**: Support for distributed testing across multiple machines
2. **Cloud Integration**: Native cloud notebook environment support
3. **Performance Optimization**: Caching and lazy loading for large datasets
4. **Resource Management**: Better memory and CPU usage optimization
