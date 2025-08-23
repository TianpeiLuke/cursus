"""
Visualization Reporter for Pipeline Runtime Testing

This module provides comprehensive visualization capabilities for pipeline testing results,
performance metrics, and data quality assessments using Plotly for interactive dashboards.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import json
from datetime import datetime, timedelta
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    from IPython.display import display, HTML
    import ipywidgets as widgets
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False
    # Create mock classes for graceful fallback
    class go:
        class Figure: pass
        class Scatter: pass
        class Bar: pass
        class Heatmap: pass
    class px: pass
    class widgets:
        class Output: pass
        class VBox: pass
        class HBox: pass


class VisualizationConfig(BaseModel):
    """Configuration for visualization settings"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    theme: str = Field(default="plotly_white", description="Plotly theme")
    color_palette: List[str] = Field(
        default=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
        description="Color palette for charts"
    )
    figure_width: int = Field(default=1000, description="Default figure width")
    figure_height: int = Field(default=600, description="Default figure height")
    show_grid: bool = Field(default=True, description="Show grid in charts")
    interactive: bool = Field(default=True, description="Enable interactive features")


class TestResultMetrics(BaseModel):
    """Metrics extracted from test results"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    step_name: str
    execution_time: float
    memory_usage: Optional[float] = None
    success: bool
    error_message: Optional[str] = None
    data_size: Optional[int] = None
    timestamp: datetime
    test_type: str  # 'synthetic' or 'real'


class VisualizationReporter:
    """
    Comprehensive visualization reporter for pipeline testing results
    
    Provides interactive dashboards, performance metrics visualization,
    and data quality assessment reports using Plotly.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize the visualization reporter"""
        if not JUPYTER_AVAILABLE:
            print("Warning: Jupyter dependencies not available. Visualization features disabled.")
            return
            
        self.config = config or VisualizationConfig()
        self.test_results: List[TestResultMetrics] = []
        self.performance_data: Dict[str, Any] = {}
        self.data_quality_metrics: Dict[str, Any] = {}
        
        # Set up Plotly theme
        self._setup_plotly_theme()
    
    def _setup_plotly_theme(self):
        """Configure Plotly theme and defaults"""
        if not JUPYTER_AVAILABLE:
            return
            
        # Configure default template
        import plotly.io as pio
        pio.templates.default = self.config.theme
    
    def add_test_result(self, result: TestResultMetrics):
        """Add a test result for visualization"""
        self.test_results.append(result)
    
    def add_performance_data(self, step_name: str, metrics: Dict[str, Any]):
        """Add performance metrics for a step"""
        self.performance_data[step_name] = metrics
    
    def add_data_quality_metrics(self, step_name: str, metrics: Dict[str, Any]):
        """Add data quality metrics for a step"""
        self.data_quality_metrics[step_name] = metrics
    
    def create_execution_timeline(self) -> Optional[go.Figure]:
        """Create an interactive timeline of test executions"""
        if not JUPYTER_AVAILABLE or not self.test_results:
            return None
        
        # Prepare data for timeline
        df_data = []
        for result in self.test_results:
            df_data.append({
                'step_name': result.step_name,
                'start_time': result.timestamp,
                'end_time': result.timestamp + timedelta(seconds=result.execution_time),
                'execution_time': result.execution_time,
                'success': result.success,
                'test_type': result.test_type,
                'error_message': result.error_message or 'Success'
            })
        
        df = pd.DataFrame(df_data)
        
        # Create Gantt-style timeline
        fig = go.Figure()
        
        # Add bars for each execution
        for i, row in df.iterrows():
            color = self.config.color_palette[0] if row['success'] else '#d62728'
            fig.add_trace(go.Scatter(
                x=[row['start_time'], row['end_time']],
                y=[row['step_name'], row['step_name']],
                mode='lines+markers',
                line=dict(color=color, width=10),
                name=f"{row['step_name']} ({row['test_type']})",
                hovertemplate=(
                    f"<b>{row['step_name']}</b><br>"
                    f"Type: {row['test_type']}<br>"
                    f"Duration: {row['execution_time']:.2f}s<br>"
                    f"Status: {'Success' if row['success'] else 'Failed'}<br>"
                    f"Details: {row['error_message']}<br>"
                    "<extra></extra>"
                ),
                showlegend=False
            ))
        
        fig.update_layout(
            title="Pipeline Test Execution Timeline",
            xaxis_title="Time",
            yaxis_title="Pipeline Steps",
            width=self.config.figure_width,
            height=self.config.figure_height,
            showlegend=False
        )
        
        return fig
    
    def create_performance_dashboard(self) -> Optional[go.Figure]:
        """Create a comprehensive performance metrics dashboard"""
        if not JUPYTER_AVAILABLE or not self.test_results:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Execution Time by Step',
                'Success Rate by Test Type',
                'Memory Usage Distribution',
                'Data Size vs Execution Time'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Prepare data
        df_data = []
        for result in self.test_results:
            df_data.append({
                'step_name': result.step_name,
                'execution_time': result.execution_time,
                'memory_usage': result.memory_usage or 0,
                'success': result.success,
                'test_type': result.test_type,
                'data_size': result.data_size or 0
            })
        
        df = pd.DataFrame(df_data)
        
        # 1. Execution Time by Step
        step_times = df.groupby('step_name')['execution_time'].mean().reset_index()
        fig.add_trace(
            go.Bar(
                x=step_times['step_name'],
                y=step_times['execution_time'],
                name='Avg Execution Time',
                marker_color=self.config.color_palette[0]
            ),
            row=1, col=1
        )
        
        # 2. Success Rate by Test Type
        success_rates = df.groupby('test_type')['success'].mean().reset_index()
        success_rates['success_rate'] = success_rates['success'] * 100
        fig.add_trace(
            go.Bar(
                x=success_rates['test_type'],
                y=success_rates['success_rate'],
                name='Success Rate %',
                marker_color=self.config.color_palette[1]
            ),
            row=1, col=2
        )
        
        # 3. Memory Usage Distribution
        memory_data = df[df['memory_usage'] > 0]['memory_usage']
        if not memory_data.empty:
            fig.add_trace(
                go.Histogram(
                    x=memory_data,
                    name='Memory Usage',
                    marker_color=self.config.color_palette[2]
                ),
                row=2, col=1
            )
        
        # 4. Data Size vs Execution Time
        size_time_data = df[(df['data_size'] > 0) & (df['execution_time'] > 0)]
        if not size_time_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=size_time_data['data_size'],
                    y=size_time_data['execution_time'],
                    mode='markers',
                    name='Size vs Time',
                    marker=dict(
                        color=self.config.color_palette[3],
                        size=8,
                        opacity=0.7
                    ),
                    text=size_time_data['step_name'],
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Data Size: %{x}<br>"
                        "Execution Time: %{y:.2f}s<br>"
                        "<extra></extra>"
                    )
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Performance Metrics Dashboard",
            width=self.config.figure_width,
            height=self.config.figure_height,
            showlegend=False
        )
        
        return fig
    
    def create_data_quality_report(self) -> Optional[go.Figure]:
        """Create data quality assessment visualization"""
        if not JUPYTER_AVAILABLE or not self.data_quality_metrics:
            return None
        
        # Create subplots for different quality metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Data Completeness',
                'Data Validity',
                'Schema Compliance',
                'Quality Score Trends'
            )
        )
        
        steps = list(self.data_quality_metrics.keys())
        
        # 1. Data Completeness
        completeness_scores = [
            self.data_quality_metrics[step].get('completeness', 0) * 100
            for step in steps
        ]
        fig.add_trace(
            go.Bar(
                x=steps,
                y=completeness_scores,
                name='Completeness %',
                marker_color=self.config.color_palette[0]
            ),
            row=1, col=1
        )
        
        # 2. Data Validity
        validity_scores = [
            self.data_quality_metrics[step].get('validity', 0) * 100
            for step in steps
        ]
        fig.add_trace(
            go.Bar(
                x=steps,
                y=validity_scores,
                name='Validity %',
                marker_color=self.config.color_palette[1]
            ),
            row=1, col=2
        )
        
        # 3. Schema Compliance
        compliance_scores = [
            self.data_quality_metrics[step].get('schema_compliance', 0) * 100
            for step in steps
        ]
        fig.add_trace(
            go.Bar(
                x=steps,
                y=compliance_scores,
                name='Schema Compliance %',
                marker_color=self.config.color_palette[2]
            ),
            row=2, col=1
        )
        
        # 4. Overall Quality Score Trends
        overall_scores = [
            self.data_quality_metrics[step].get('overall_score', 0) * 100
            for step in steps
        ]
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=overall_scores,
                mode='lines+markers',
                name='Overall Quality',
                line=dict(color=self.config.color_palette[3], width=3),
                marker=dict(size=8)
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Data Quality Assessment Report",
            width=self.config.figure_width,
            height=self.config.figure_height,
            showlegend=False
        )
        
        return fig
    
    def create_comparison_chart(self, 
                              metric: str = 'execution_time',
                              group_by: str = 'test_type') -> Optional[go.Figure]:
        """Create comparison charts for different metrics"""
        if not JUPYTER_AVAILABLE or not self.test_results:
            return None
        
        # Prepare data
        df_data = []
        for result in self.test_results:
            df_data.append({
                'step_name': result.step_name,
                'execution_time': result.execution_time,
                'memory_usage': result.memory_usage or 0,
                'success': 1 if result.success else 0,
                'test_type': result.test_type,
                'data_size': result.data_size or 0
            })
        
        df = pd.DataFrame(df_data)
        
        if group_by == 'test_type':
            fig = px.box(
                df, 
                x='test_type', 
                y=metric,
                color='test_type',
                title=f'{metric.replace("_", " ").title()} Comparison by Test Type',
                color_discrete_sequence=self.config.color_palette
            )
        else:
            fig = px.box(
                df, 
                x='step_name', 
                y=metric,
                color='step_name',
                title=f'{metric.replace("_", " ").title()} Comparison by Step',
                color_discrete_sequence=self.config.color_palette
            )
        
        fig.update_layout(
            width=self.config.figure_width,
            height=self.config.figure_height
        )
        
        return fig
    
    def create_interactive_dashboard(self) -> Optional[widgets.VBox]:
        """Create an interactive dashboard with controls"""
        if not JUPYTER_AVAILABLE:
            return None
        
        # Create output widget for displaying charts
        output = widgets.Output()
        
        # Create control widgets
        chart_type = widgets.Dropdown(
            options=[
                ('Execution Timeline', 'timeline'),
                ('Performance Dashboard', 'performance'),
                ('Data Quality Report', 'quality'),
                ('Comparison Chart', 'comparison')
            ],
            value='timeline',
            description='Chart Type:'
        )
        
        metric_selector = widgets.Dropdown(
            options=[
                ('Execution Time', 'execution_time'),
                ('Memory Usage', 'memory_usage'),
                ('Success Rate', 'success'),
                ('Data Size', 'data_size')
            ],
            value='execution_time',
            description='Metric:'
        )
        
        group_selector = widgets.Dropdown(
            options=[
                ('Test Type', 'test_type'),
                ('Step Name', 'step_name')
            ],
            value='test_type',
            description='Group By:'
        )
        
        refresh_button = widgets.Button(
            description='Refresh Chart',
            button_style='primary'
        )
        
        def update_chart(*args):
            """Update the displayed chart based on selections"""
            with output:
                output.clear_output()
                
                chart_value = chart_type.value
                
                if chart_value == 'timeline':
                    fig = self.create_execution_timeline()
                elif chart_value == 'performance':
                    fig = self.create_performance_dashboard()
                elif chart_value == 'quality':
                    fig = self.create_data_quality_report()
                elif chart_value == 'comparison':
                    fig = self.create_comparison_chart(
                        metric=metric_selector.value,
                        group_by=group_selector.value
                    )
                else:
                    fig = None
                
                if fig:
                    display(fig)
                else:
                    print("No data available for visualization")
        
        # Set up event handlers
        chart_type.observe(update_chart, names='value')
        metric_selector.observe(update_chart, names='value')
        group_selector.observe(update_chart, names='value')
        refresh_button.on_click(update_chart)
        
        # Initial chart display
        update_chart()
        
        # Create layout
        controls = widgets.HBox([chart_type, metric_selector, group_selector, refresh_button])
        dashboard = widgets.VBox([controls, output])
        
        return dashboard
    
    def export_report(self, output_path: Path, format: str = 'html'):
        """Export visualization report to file"""
        if not JUPYTER_AVAILABLE:
            print("Cannot export report: Jupyter dependencies not available")
            return
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'html':
            # Create comprehensive HTML report
            html_content = self._generate_html_report()
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        elif format.lower() == 'json':
            # Export raw data as JSON
            report_data = {
                'test_results': [result.model_dump() for result in self.test_results],
                'performance_data': self.performance_data,
                'data_quality_metrics': self.data_quality_metrics,
                'generated_at': datetime.now().isoformat()
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, default=str)
        
        print(f"Report exported to: {output_path}")
    
    def _generate_html_report(self) -> str:
        """Generate comprehensive HTML report"""
        html_parts = [
            "<html><head><title>Pipeline Testing Report</title>",
            "<style>body { font-family: Arial, sans-serif; margin: 20px; }</style>",
            "</head><body>",
            "<h1>Pipeline Runtime Testing Report</h1>",
            f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        ]
        
        # Add charts
        charts = [
            ('Execution Timeline', self.create_execution_timeline()),
            ('Performance Dashboard', self.create_performance_dashboard()),
            ('Data Quality Report', self.create_data_quality_report())
        ]
        
        for title, fig in charts:
            if fig:
                html_parts.append(f"<h2>{title}</h2>")
                html_parts.append(fig.to_html(include_plotlyjs='cdn'))
        
        html_parts.append("</body></html>")
        return '\n'.join(html_parts)
    
    def display_summary(self):
        """Display a summary of current test results"""
        if not self.test_results:
            print("No test results available")
            return
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.success)
        success_rate = (successful_tests / total_tests) * 100
        
        avg_execution_time = sum(r.execution_time for r in self.test_results) / total_tests
        
        synthetic_tests = sum(1 for r in self.test_results if r.test_type == 'synthetic')
        real_tests = sum(1 for r in self.test_results if r.test_type == 'real')
        
        print("=== Pipeline Testing Summary ===")
        print(f"Total Tests: {total_tests}")
        print(f"Successful Tests: {successful_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Average Execution Time: {avg_execution_time:.2f}s")
        print(f"Synthetic Tests: {synthetic_tests}")
        print(f"Real Data Tests: {real_tests}")
        print(f"Unique Steps Tested: {len(set(r.step_name for r in self.test_results))}")
        
        if JUPYTER_AVAILABLE:
            print("\nUse create_interactive_dashboard() for detailed visualizations")
