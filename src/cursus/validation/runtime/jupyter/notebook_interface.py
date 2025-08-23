"""Interactive Jupyter notebook interface for pipeline testing."""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from pathlib import Path
import json
import pandas as pd
from datetime import datetime

try:
    from IPython.display import display, HTML, Markdown
    from ipywidgets import widgets, interact, interactive, fixed
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    JUPYTER_AVAILABLE = True
except ImportError:
    # Graceful fallback when Jupyter dependencies are not available
    JUPYTER_AVAILABLE = False
    display = lambda x: print(str(x))
    HTML = lambda x: x
    Markdown = lambda x: x
    widgets = None

from ..core.pipeline_script_executor import PipelineScriptExecutor
from ..integration.s3_data_downloader import S3DataDownloader
from ..integration.real_data_tester import RealDataTester


class NotebookSession(BaseModel):
    """Jupyter notebook session for pipeline testing."""
    session_id: str
    workspace_dir: Path
    pipeline_name: Optional[str] = None
    current_step: Optional[str] = None
    test_results: Optional[Dict[str, Any]] = None


class NotebookInterface:
    """Interactive Jupyter interface for pipeline testing."""
    
    def __init__(self, workspace_dir: str = "./test_workspace"):
        """Initialize notebook interface with workspace directory."""
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = NotebookSession(
            session_id=f"session_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
            workspace_dir=self.workspace_dir
        )
        
        # Initialize core components
        self.script_executor = PipelineScriptExecutor(workspace_dir)
        self.s3_downloader = S3DataDownloader(workspace_dir)
        self.real_data_tester = RealDataTester(workspace_dir)
        
        if not JUPYTER_AVAILABLE:
            print("Warning: Jupyter dependencies not available. Some features may be limited.")
    
    def display_welcome(self):
        """Display welcome message and setup instructions."""
        welcome_html = f"""
        <div style="border: 2px solid #4CAF50; padding: 20px; border-radius: 10px; background-color: #f9f9f9;">
            <h2 style="color: #4CAF50;">🧪 Pipeline Script Functionality Testing</h2>
            <p><strong>Session ID:</strong> {self.session.session_id}</p>
            <p><strong>Workspace:</strong> {self.workspace_dir}</p>
            <h3>Quick Start:</h3>
            <ul>
                <li>Use <code>load_pipeline()</code> to load a pipeline configuration</li>
                <li>Use <code>test_single_step()</code> to test individual steps</li>
                <li>Use <code>test_pipeline()</code> to test complete pipelines</li>
                <li>Use <code>explore_data()</code> to interactively explore data</li>
            </ul>
            <h3>Available Data Sources:</h3>
            <ul>
                <li><strong>synthetic</strong>: Generated test data for development</li>
                <li><strong>s3</strong>: Real pipeline data from S3 buckets</li>
                <li><strong>local</strong>: Local files for testing</li>
            </ul>
        </div>
        """
        display(HTML(welcome_html))
    
    def load_pipeline(self, pipeline_name: str, config_path: Optional[str] = None):
        """Load pipeline configuration for testing."""
        self.session.pipeline_name = pipeline_name
        
        try:
            if config_path:
                # Load from file
                with open(config_path) as f:
                    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                        import yaml
                        config = yaml.safe_load(f)
                    else:
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
                
        except Exception as e:
            display(HTML(f'<div style="color: red;">❌ Error loading pipeline: {str(e)}</div>'))
            return None
    
    def test_single_step(self, step_name: str, data_source: str = "synthetic", 
                        interactive: bool = True):
        """Test a single pipeline step with interactive controls."""
        if not JUPYTER_AVAILABLE or not interactive:
            return self._execute_step_test(step_name, data_source)
        
        return self._create_interactive_step_tester(step_name, data_source)
    
    def _create_interactive_step_tester(self, step_name: str, data_source: str):
        """Create interactive widget for step testing."""
        if not widgets:
            print("Interactive widgets not available. Running in non-interactive mode.")
            return self._execute_step_test(step_name, data_source)
        
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
        
        try:
            # Execute step using the script executor
            result = self.script_executor.test_script_isolation(step_name, data_source)
            
            # Store result in session
            if self.session.test_results is None:
                self.session.test_results = {}
            self.session.test_results[step_name] = result.model_dump()
            
            return result.model_dump()
            
        except Exception as e:
            error_result = {
                'step_name': step_name,
                'success': False,
                'error': str(e),
                'data_source': data_source,
                'params': params
            }
            
            if self.session.test_results is None:
                self.session.test_results = {}
            self.session.test_results[step_name] = error_result
            
            return error_result
    
    def _display_step_result(self, result: Dict[str, Any]):
        """Display step test result with formatting."""
        success = result.get('success', False)
        step_name = result.get('script_name', result.get('step_name', 'Unknown'))
        
        if success:
            status_html = f"""
            <div style="border-left: 4px solid #4CAF50; padding: 10px; margin: 10px 0; background-color: #f9f9f9;">
                <h4 style="margin: 0; color: #4CAF50;">✅ {step_name}</h4>
                <p><strong>Status:</strong> Success</p>
                <p><strong>Execution Time:</strong> {result.get('execution_time', 0):.2f}s</p>
                <p><strong>Memory Usage:</strong> {result.get('memory_usage', 0)} MB</p>
            """
        else:
            status_html = f"""
            <div style="border-left: 4px solid #f44336; padding: 10px; margin: 10px 0; background-color: #ffebee;">
                <h4 style="margin: 0; color: #f44336;">❌ {step_name}</h4>
                <p><strong>Status:</strong> Failed</p>
                <p><strong>Error:</strong> {result.get('error_message', result.get('error', 'Unknown error'))}</p>
            """
        
        # Add recommendations if available
        recommendations = result.get('recommendations', [])
        if recommendations:
            status_html += "<p><strong>Recommendations:</strong></p><ul>"
            for rec in recommendations:
                status_html += f"<li>{rec}</li>"
            status_html += "</ul>"
        
        status_html += "</div>"
        display(HTML(status_html))
    
    def explore_data(self, data_source: Union[str, pd.DataFrame], 
                    interactive: bool = True):
        """Interactive data exploration interface."""
        try:
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
            
            if not JUPYTER_AVAILABLE or not interactive:
                return self._display_data_summary(df)
            
            return self._create_interactive_data_explorer(df)
            
        except Exception as e:
            display(HTML(f'<div style="color: red;">❌ Error loading data: {str(e)}</div>'))
            return None
    
    def _create_interactive_data_explorer(self, df: pd.DataFrame):
        """Create interactive data exploration widgets."""
        if not widgets:
            return self._display_data_summary(df)
        
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
                
                try:
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
                        
                except Exception as e:
                    display(HTML(f'<div style="color: red;">❌ Error creating plot: {str(e)}</div>'))
        
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
    
    def _display_data_summary(self, df: pd.DataFrame):
        """Display basic data summary without interactive widgets."""
        summary_html = f"""
        <div style="border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px;">
            <h3>Data Summary</h3>
            <p><strong>Shape:</strong> {df.shape[0]} rows × {df.shape[1]} columns</p>
            <p><strong>Columns:</strong> {', '.join(df.columns.tolist())}</p>
        </div>
        """
        display(HTML(summary_html))
        display(df.head())
        display(df.describe())
        return df
    
    def _discover_pipeline_config(self, pipeline_name: str) -> Optional[Dict[str, Any]]:
        """Try to discover pipeline configuration from common locations."""
        possible_paths = [
            f"pipelines/{pipeline_name}.yaml",
            f"pipelines/{pipeline_name}.yml",
            f"pipelines/{pipeline_name}.json",
            f"configs/{pipeline_name}.yaml",
            f"configs/{pipeline_name}.yml",
            f"configs/{pipeline_name}.json",
            f"{pipeline_name}.yaml",
            f"{pipeline_name}.yml",
            f"{pipeline_name}.json"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                try:
                    with open(path) as f:
                        if path.endswith('.yaml') or path.endswith('.yml'):
                            import yaml
                            return yaml.safe_load(f)
                        else:
                            return json.load(f)
                except Exception:
                    continue
        
        return None
    
    def _display_pipeline_summary(self, config: Dict[str, Any]):
        """Display pipeline configuration summary."""
        steps = config.get('steps', {})
        if isinstance(steps, list):
            step_names = steps
        else:
            step_names = list(steps.keys())
        
        summary_html = f"""
        <div style="border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px;">
            <h3>Pipeline Configuration</h3>
            <p><strong>Steps:</strong> {len(step_names)}</p>
            <ul>
        """
        
        for step in step_names[:10]:  # Show first 10 steps
            summary_html += f"<li>{step}</li>"
        
        if len(step_names) > 10:
            summary_html += f"<li>... and {len(step_names) - 10} more</li>"
        
        summary_html += "</ul></div>"
        display(HTML(summary_html))
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information."""
        return {
            'session_id': self.session.session_id,
            'workspace_dir': str(self.session.workspace_dir),
            'pipeline_name': self.session.pipeline_name,
            'current_step': self.session.current_step,
            'test_results_count': len(self.session.test_results or {}),
            'jupyter_available': JUPYTER_AVAILABLE
        }
