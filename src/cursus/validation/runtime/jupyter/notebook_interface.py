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
from ....workspace.core import WorkspaceComponentRegistry


class NotebookSession(BaseModel):
    """Jupyter notebook session for pipeline testing."""
    session_id: str
    workspace_dir: Path
    pipeline_name: Optional[str] = None
    current_step: Optional[str] = None
    test_results: Optional[Dict[str, Any]] = None
    workspace_root: Optional[str] = None
    selected_developer: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    
    @property
    def workspace_path(self):
        """Alias for workspace_dir for backward compatibility."""
        return self.workspace_dir


class NotebookInterface:
    """Interactive Jupyter interface for pipeline testing with workspace awareness."""
    
    def __init__(self, workspace_dir: str = "./development/projects/project_alpha", workspace_root: str = None):
        """Initialize notebook interface with workspace directory and optional workspace root."""
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = NotebookSession(
            session_id=f"session_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
            workspace_dir=self.workspace_dir,
            workspace_root=workspace_root
        )
        
        # Initialize core components with workspace awareness
        self.script_executor = PipelineScriptExecutor(workspace_dir, workspace_root)
        self.s3_downloader = S3DataDownloader(workspace_dir)
        self.real_data_tester = RealDataTester(workspace_dir)
        
        # Phase 5: Workspace component registry for developer discovery
        self.workspace_registry = None
        if workspace_root:
            self.workspace_registry = WorkspaceComponentRegistry(workspace_root)
        
        if not JUPYTER_AVAILABLE:
            print("Warning: Jupyter dependencies not available. Some features may be limited.")
    
    @property
    def executor(self):
        """Alias for script_executor for backward compatibility."""
        return self.script_executor
    
    def display_welcome(self):
        """Display welcome message and setup instructions."""
        workspace_info = ""
        if self.session.workspace_root:
            workspace_info = f"""
            <p><strong>Workspace Root:</strong> {self.session.workspace_root}</p>
            <p><strong>Selected Developer:</strong> {self.session.selected_developer or 'None'}</p>
            """
        
        welcome_html = f"""
        <div style="border: 2px solid #4CAF50; padding: 20px; border-radius: 10px; background-color: #f9f9f9;">
            <h2 style="color: #4CAF50;">🧪 Pipeline Script Functionality Testing</h2>
            <p><strong>Session ID:</strong> {self.session.session_id}</p>
            <p><strong>Workspace:</strong> {self.workspace_dir}</p>
            {workspace_info}
            <h3>Quick Start:</h3>
            <ul>
                <li>Use <code>select_workspace_developer()</code> to choose a developer workspace</li>
                <li>Use <code>load_pipeline()</code> to load a pipeline configuration</li>
                <li>Use <code>test_single_step()</code> to test individual steps</li>
                <li>Use <code>test_pipeline()</code> to test complete pipelines</li>
                <li>Use <code>explore_data()</code> to interactively explore data</li>
            </ul>
            <h3>Available Data Sources:</h3>
            <ul>
                <li><strong>synthetic</strong>: Generated test data for development</li>
                <li><strong>s3</strong>: Real pipeline data from S3 buckets</li>
                <li><strong>local</strong>: Local files for testing (workspace-aware)</li>
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
                        developer_id: str = None, interactive: bool = True):
        """Test a single pipeline step with interactive controls and workspace context."""
        if not JUPYTER_AVAILABLE or not interactive:
            return self._execute_step_test(step_name, data_source, developer_id=developer_id)
        
        return self._create_interactive_step_tester(step_name, data_source, developer_id)
    
    def _create_interactive_step_tester(self, step_name: str, data_source: str, developer_id: str = None):
        """Create interactive widget for step testing with workspace context."""
        if not widgets:
            print("Interactive widgets not available. Running in non-interactive mode.")
            return self._execute_step_test(step_name, data_source, developer_id=developer_id)
        
        # Data source selection
        data_source_widget = widgets.Dropdown(
            options=['synthetic', 's3', 'local'],
            value=data_source,
            description='Data Source:'
        )
        
        # Phase 5: Developer selection widget
        developer_options = ['None']
        if self.workspace_registry:
            try:
                components = self.workspace_registry.discover_components()
                developers = components['summary'].get('developers', [])
                developer_options.extend(developers)
            except Exception as e:
                print(f"Warning: Could not discover developers: {e}")
        
        developer_widget = widgets.Dropdown(
            options=developer_options,
            value=developer_id or self.session.selected_developer or 'None',
            description='Developer:'
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
                    selected_developer = developer_widget.value if developer_widget.value != 'None' else None
                    result = self._execute_step_test(
                        step_name, 
                        data_source_widget.value, 
                        developer_id=selected_developer,
                        params=params
                    )
                    self._display_step_result(result)
                except Exception as e:
                    display(HTML(f'<div style="color: red;">❌ Error: {str(e)}</div>'))
        
        execute_button.on_click(on_execute_clicked)
        
        # Layout
        controls = widgets.VBox([
            widgets.HTML(f'<h3>Testing Step: {step_name}</h3>'),
            data_source_widget,
            developer_widget,
            test_params_widget,
            execute_button
        ])
        
        display(widgets.VBox([controls, output_area]))
        
        return controls, output_area
    
    def _execute_step_test(self, step_name: str, data_source: str, 
                          developer_id: str = None, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute step test and return results with workspace context."""
        if params is None:
            params = {}
        
        try:
            # Execute step using the script executor with workspace context
            result = self.script_executor.test_script_isolation(
                step_name, 
                data_source, 
                developer_id=developer_id
            )
            
            # Store result in session with workspace context
            if self.session.test_results is None:
                self.session.test_results = {}
            
            result_dict = result.model_dump()
            result_dict['developer_id'] = developer_id
            result_dict['workspace_context'] = {
                'workspace_root': self.session.workspace_root,
                'selected_developer': developer_id
            }
            
            self.session.test_results[step_name] = result_dict
            
            return result_dict
            
        except Exception as e:
            error_result = {
                'step_name': step_name,
                'success': False,
                'error': str(e),
                'data_source': data_source,
                'developer_id': developer_id,
                'params': params,
                'workspace_context': {
                    'workspace_root': self.session.workspace_root,
                    'selected_developer': developer_id
                }
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
            'workspace_root': self.session.workspace_root,
            'selected_developer': self.session.selected_developer,
            'pipeline_name': self.session.pipeline_name,
            'current_step': self.session.current_step,
            'test_results_count': len(self.session.test_results or {}),
            'jupyter_available': JUPYTER_AVAILABLE
        }
    
    def select_workspace_developer(self, developer_id: str = None, interactive: bool = True):
        """Select a developer workspace for testing."""
        if not self.workspace_registry:
            display(HTML('<div style="color: orange;">⚠️ No workspace context available</div>'))
            return None
        
        if not JUPYTER_AVAILABLE or not interactive:
            return self._set_developer(developer_id)
        
        return self._create_developer_selector()
    
    def _create_developer_selector(self):
        """Create interactive developer selection widget."""
        if not widgets:
            print("Interactive widgets not available.")
            return None
        
        try:
            # Discover available developers
            components = self.workspace_registry.discover_components()
            developers = components['summary'].get('developers', [])
            
            if not developers:
                display(HTML('<div style="color: orange;">⚠️ No developers found in workspace</div>'))
                return None
            
            # Developer selection widget
            developer_widget = widgets.Dropdown(
                options=['None'] + developers,
                value=self.session.selected_developer or 'None',
                description='Developer:'
            )
            
            # Info display area
            info_area = widgets.Output()
            
            # Select button
            select_button = widgets.Button(
                description='Select Developer',
                button_style='primary',
                icon='user'
            )
            
            def on_developer_change(change):
                with info_area:
                    info_area.clear_output()
                    if change['new'] != 'None':
                        self._display_developer_info(change['new'])
            
            def on_select_clicked(b):
                selected_dev = developer_widget.value if developer_widget.value != 'None' else None
                self._set_developer(selected_dev)
                display(HTML(f'<div style="color: green;">✅ Selected developer: {selected_dev or "None"}</div>'))
            
            developer_widget.observe(on_developer_change, names='value')
            select_button.on_click(on_select_clicked)
            
            # Layout
            controls = widgets.VBox([
                widgets.HTML('<h3>Select Developer Workspace</h3>'),
                developer_widget,
                select_button,
                info_area
            ])
            
            display(controls)
            return controls
            
        except Exception as e:
            display(HTML(f'<div style="color: red;">❌ Error creating developer selector: {str(e)}</div>'))
            return None
    
    def _set_developer(self, developer_id: str):
        """Set the selected developer for the session."""
        self.session.selected_developer = developer_id
        if developer_id:
            display(HTML(f'<div style="color: green;">✅ Selected developer: {developer_id}</div>'))
        else:
            display(HTML('<div style="color: blue;">ℹ️ No developer selected (using general workspace)</div>'))
        return developer_id
    
    def _display_developer_info(self, developer_id: str):
        """Display information about a developer's workspace."""
        try:
            components = self.workspace_registry.discover_components(developer_id)
            
            info_html = f"""
            <div style="border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 5px;">
                <h4>Developer: {developer_id}</h4>
                <p><strong>Scripts:</strong> {len(components.get('scripts', {}))}</p>
                <p><strong>Builders:</strong> {len(components.get('builders', {}))}</p>
                <p><strong>Configs:</strong> {len(components.get('configs', {}))}</p>
                <p><strong>Contracts:</strong> {len(components.get('contracts', {}))}</p>
                <p><strong>Specs:</strong> {len(components.get('specs', {}))}</p>
            </div>
            """
            display(HTML(info_html))
            
        except Exception as e:
            display(HTML(f'<div style="color: red;">❌ Error getting developer info: {str(e)}</div>'))
    
    def list_workspace_components(self, developer_id: str = None):
        """List available workspace components."""
        if not self.workspace_registry:
            display(HTML('<div style="color: orange;">⚠️ No workspace context available</div>'))
            return None
        
        try:
            components = self.workspace_registry.discover_components(developer_id)
            
            summary_html = f"""
            <div style="border: 2px solid #2196F3; padding: 15px; margin: 10px 0; border-radius: 5px;">
                <h3>Workspace Components{f" - {developer_id}" if developer_id else ""}</h3>
                <p><strong>Total Components:</strong> {components['summary']['total_components']}</p>
                <p><strong>Developers:</strong> {len(components['summary']['developers'])}</p>
                <p><strong>Step Types:</strong> {len(components['summary']['step_types'])}</p>
            </div>
            """
            display(HTML(summary_html))
            
            # Display component details
            for component_type in ['scripts', 'builders', 'configs', 'contracts', 'specs']:
                component_data = components.get(component_type, {})
                if component_data:
                    display(HTML(f'<h4>{component_type.title()} ({len(component_data)})</h4>'))
                    
                    # Create DataFrame for better display
                    rows = []
                    for key, info in component_data.items():
                        rows.append({
                            'Key': key,
                            'Developer': info.get('developer_id', 'N/A'),
                            'Step Name': info.get('step_name', 'N/A'),
                            'Type': info.get('step_type', info.get('class_name', 'N/A'))
                        })
                    
                    if rows:
                        df = pd.DataFrame(rows)
                        display(df)
            
            return components
            
        except Exception as e:
            display(HTML(f'<div style="color: red;">❌ Error listing components: {str(e)}</div>'))
            return None
