"""
Jupyter Notebook Widget for Cradle Data Load Configuration

This module provides a Jupyter widget interface for the Cradle Data Load Config UI
that can be used directly in notebooks to replace manual configuration blocks.
"""

import ipywidgets as widgets
from IPython.display import display, HTML, Javascript
import json
import requests
from typing import Optional, Dict, Any
import asyncio
import threading
import time
from pathlib import Path
import uuid
import weakref

from ...steps.configs.config_cradle_data_loading_step import CradleDataLoadingConfig

class CradleConfigWidget:
    """
    Jupyter widget for Cradle Data Load Configuration with workflow integration.
    
    This widget provides an embedded UI for configuring Cradle data loading
    that can be used directly in Jupyter notebooks with full workflow context support.
    """
    
    def __init__(self, 
                 base_config=None,
                 job_type: str = "training",
                 width: str = "100%", 
                 height: str = "800px",
                 server_port: int = 8001,
                 workflow_context: Optional[Dict[str, Any]] = None,
                 embedded_mode: bool = False):
        """
        Initialize the Cradle Config Widget with workflow integration.
        
        Args:
            base_config: Base pipeline configuration object
            job_type: Type of job (training, validation, testing, calibration)
            width: Widget width
            height: Widget height
            server_port: Port where the UI server is running
            workflow_context: Workflow context from DAG analysis and step structure
            embedded_mode: Whether widget is embedded in a parent workflow (NEW)
        """
        self.base_config = base_config
        self.job_type = job_type
        self.width = width
        self.height = height
        self.server_port = server_port
        self.workflow_context = workflow_context or {}
        self.embedded_mode = embedded_mode  # NEW: Embedded mode flag
        self.config_result = None
        self.server_url = f"http://localhost:{server_port}"
        self.completion_callback = None  # NEW: Callback for embedded mode
        
        # Unique identifier for this widget instance
        self.widget_id = str(uuid.uuid4())
        
        # Initialize field categorization using 3-tier system
        self.field_categories = self._initialize_field_categories()
        
        # Resolve inherited values from full workflow chain
        self.inherited_values = self._resolve_workflow_inherited_values()
        
        # Discover available fields from DAG analysis
        self.available_fields = self._discover_workflow_fields()
        
        # Create the widget components
        self._create_widgets()
    
    def _initialize_field_categories(self) -> Dict[str, List[str]]:
        """Initialize field categorization using 3-tier system."""
        try:
            # If we have a base config, use its categorization method
            if self.base_config and hasattr(self.base_config, 'categorize_fields'):
                return self.base_config.categorize_fields()
            else:
                # Fallback to manual categorization for CradleDataLoadingConfig
                return self._manual_field_categorization()
                
        except Exception as e:
            # Fallback to basic categorization
            return {
                "essential": ["job_type", "data_sources", "transform_sql", "output_schema"],
                "system": ["cradle_account", "cluster_type", "output_format"],
                "derived": []
            }
    
    def _manual_field_categorization(self) -> Dict[str, List[str]]:
        """Manually categorize fields for CradleDataLoadingConfig."""
        return {
            "essential": [  # Tier 1: Required, user must provide
                "job_type",
                "data_sources_spec",
                "transform_spec", 
                "output_spec"
            ],
            "system": [     # Tier 2: Optional with defaults
                "cradle_job_spec",
                "job_split_options",
                "output_format",
                "cluster_type"
            ],
            "derived": []   # Tier 3: Hidden from UI (computed)
        }
    
    def _resolve_workflow_inherited_values(self) -> Dict[str, Any]:
        """Resolve inherited values from full workflow chain."""
        inherited_values = {}
        
        # Start with base config values if provided
        if self.base_config:
            try:
                # Extract standard base config fields
                base_fields = ['author', 'bucket', 'role', 'region', 'service_name', 
                              'pipeline_version', 'project_root_folder']
                for field in base_fields:
                    if hasattr(self.base_config, field):
                        inherited_values[field] = getattr(self.base_config, field)
                        
            except Exception as e:
                print(f"Warning: Could not extract base config values: {e}")
        
        # Apply workflow context inheritance if available
        if self.workflow_context:
            workflow_inherited = self._get_workflow_inherited_values()
            inherited_values.update(workflow_inherited)
        
        # Set Cradle-specific defaults
        cradle_defaults = {
            'job_type': self.job_type,
            'output_format': 'PARQUET',
            'cluster_type': 'STANDARD',
            'cradle_account': 'default'
        }
        
        # Only set defaults for fields not already inherited
        for key, default_value in cradle_defaults.items():
            if key not in inherited_values:
                inherited_values[key] = default_value
        
        return inherited_values
    
    def _get_workflow_inherited_values(self) -> Dict[str, Any]:
        """Get inherited values from workflow context."""
        workflow_values = {}
        
        # Extract values from workflow context
        if 'inheritance_chain' in self.workflow_context:
            inheritance_chain = self.workflow_context['inheritance_chain']
            for config_data in inheritance_chain:
                if isinstance(config_data, dict):
                    # Extract relevant Cradle fields
                    for key, value in config_data.items():
                        if key in ['job_type', 'output_format', 'cluster_type', 'cradle_account']:
                            workflow_values[key] = value
        
        # Extract step information from DAG analysis if available
        if 'dag_analysis' in self.workflow_context:
            dag_analysis = self.workflow_context['dag_analysis']
            if 'step_context' in dag_analysis:
                step_context = dag_analysis['step_context']
                # Use step context to determine job type or other parameters
                if 'job_type' in step_context:
                    workflow_values['job_type'] = step_context['job_type']
        
        return workflow_values
    
    def _discover_workflow_fields(self) -> Dict[str, List[str]]:
        """Discover available fields from workflow context."""
        available_fields = {
            'data_sources': [],
            'transform_fields': [],
            'output_fields': []
        }
        
        # Try to get fields from workflow context first
        if self.workflow_context and 'dag_analysis' in self.workflow_context:
            dag_analysis = self.workflow_context['dag_analysis']
            if 'discovered_fields' in dag_analysis:
                discovered_fields = dag_analysis['discovered_fields']
                available_fields.update(discovered_fields)
                return available_fields
        
        # Fallback to example fields for demonstration
        available_fields['data_sources'] = ['mds', 'edx', 'andes']
        available_fields['transform_fields'] = ['customer_id', 'transaction_amount', 'timestamp']
        available_fields['output_fields'] = ['processed_data', 'features', 'labels']
        
        return available_fields
        
    def _create_widgets(self):
        """Create the widget components."""
        # Status display with proper layout for text wrapping
        self.status_output = widgets.Output(
            layout=widgets.Layout(
                width='100%',
                max_height='400px',
                overflow='auto',
                border='1px solid #ddd',
                padding='10px'
            )
        )
        
        # Extract base config values if provided
        base_config_params = self._extract_base_config_params()
        
        # Build URL with base config parameters
        iframe_url = self.server_url
        if base_config_params:
            # Convert params to URL query string
            param_pairs = []
            for key, value in base_config_params.items():
                if value is not None:
                    param_pairs.append(f"{key}={value}")
            
            if param_pairs:
                iframe_url += "?" + "&".join(param_pairs)
        
        # Main iframe for the UI
        self.iframe = widgets.HTML(
            value=f'''
            <iframe 
                src="{iframe_url}" 
                width="{self.width}" 
                height="{self.height}"
                style="border: 1px solid #ccc; border-radius: 4px;"
                id="cradle-config-iframe-{self.widget_id}">
            </iframe>
            '''
        )
        
        # Layout - No buttons needed, just iframe and status
        self.widget = widgets.VBox([
            widgets.HTML(f"<h3>Cradle Data Load Configuration (ID: {self.widget_id[:8]})</h3>"),
            self.iframe,
            self.status_output
        ])
    
    def _extract_base_config_params(self):
        """Extract parameters from base_config to pre-populate the form."""
        if not self.base_config:
            return {}
        
        params = {}
        
        try:
            # Extract BasePipelineConfig fields
            if hasattr(self.base_config, 'author'):
                params['author'] = self.base_config.author
            if hasattr(self.base_config, 'bucket'):
                params['bucket'] = self.base_config.bucket
            if hasattr(self.base_config, 'role'):
                params['role'] = self.base_config.role
            if hasattr(self.base_config, 'region'):
                params['region'] = self.base_config.region
            if hasattr(self.base_config, 'service_name'):
                params['service_name'] = self.base_config.service_name
            if hasattr(self.base_config, 'pipeline_version'):
                params['pipeline_version'] = self.base_config.pipeline_version
            if hasattr(self.base_config, 'project_root_folder'):
                params['project_root_folder'] = self.base_config.project_root_folder
            
            # Set job type
            params['job_type'] = self.job_type
            
            # Set default save location - absolute path to where notebook is running
            import os
            notebook_dir = os.getcwd()  # Get the current working directory where notebook is running
            params['save_location'] = os.path.join(notebook_dir, f"cradle_data_load_config_{self.job_type.lower()}.json")
            
        except Exception as e:
            # If there's any error extracting config, log it but don't fail
            with self.status_output:
                print(f"⚠️ Warning: Could not extract some base config values: {str(e)}")
        
        return params
    
    def set_completion_callback(self, callback):
        """Set callback for when 4-step wizard completes (for embedded mode)."""
        self.completion_callback = callback
    
    def get_config(self) -> Optional[CradleDataLoadingConfig]:
        """
        Get the generated configuration object.
        
        Returns:
            CradleDataLoadingConfig object if available, None otherwise
        """
        return self.config_result
    
    def _create_config_instance_from_ui(self) -> CradleDataLoadingConfig:
        """
        Create config instance from UI data (placeholder implementation).
        
        This method needs to be implemented based on how the cradle UI stores its data.
        For now, it creates a basic instance with inherited values.
        """
        # TODO: Extract actual data from the cradle UI
        # This is a placeholder implementation
        config_data = self.inherited_values.copy()
        
        # Add some default values for required fields
        config_data.update({
            'job_type': self.job_type,
            'data_sources_spec': {'sources': self.available_fields.get('data_sources', [])},
            'transform_spec': {'fields': self.available_fields.get('transform_fields', [])},
            'output_spec': {'format': 'PARQUET', 'fields': self.available_fields.get('output_fields', [])}
        })
        
        return CradleDataLoadingConfig(**config_data)
    
    def _notify_completion(self, config_instance):
        """Notify parent workflow that configuration is complete."""
        self.config_result = config_instance
        if self.completion_callback:
            self.completion_callback(config_instance)
    
    def display(self):
        """Display the widget."""
        display(self.widget)
        
        # Display different instructions based on mode
        if self.embedded_mode:
            self._display_embedded_instructions()
        else:
            self._display_standalone_instructions()
    
    def _display_embedded_instructions(self):
        """Display instructions for embedded mode."""
        display(HTML("""
        <div style="background-color: #f0f9ff; border: 1px solid #0ea5e9; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <h4 style="color: #0c4a6e; margin-bottom: 10px;">📝 Embedded Cradle Configuration:</h4>
            <ol style="color: #0c4a6e; line-height: 1.6; margin: 0; padding-left: 20px;">
                <li style="margin-bottom: 8px;">Complete the 4-step configuration in the UI above</li>
                <li style="margin-bottom: 8px;">Click <strong>"Finish"</strong> in the UI - configuration will be collected automatically</li>
                <li style="margin-bottom: 8px;">Continue to next step in the main workflow</li>
                <li style="margin-bottom: 0;">All configurations will be saved together at the end</li>
            </ol>
            <div style="background-color: #dbeafe; padding: 10px; border-radius: 4px; margin-top: 10px;">
                <strong>✨ Workflow Mode:</strong> This configuration will be included in the unified pipeline export.
            </div>
        </div>
        """))
    
    def _display_standalone_instructions(self):
        """Display instructions for standalone mode."""
        display(HTML("""
        <div style="background-color: #f8f9fa; border: 1px solid #dee2e6; padding: 20px; border-radius: 8px; margin: 15px 0; color: #212529;">
            <h4 style="color: #495057; margin-bottom: 15px; font-weight: 600;">📝 How to Use:</h4>
            <ol style="color: #495057; line-height: 1.6; margin: 0; padding-left: 20px;">
                <li style="margin-bottom: 8px;">Complete the 4-step configuration in the UI above</li>
                <li style="margin-bottom: 8px;">In Step 4, specify the save location for your configuration file</li>
                <li style="margin-bottom: 8px;">Click <strong>"Finish"</strong> in the UI - the configuration will be automatically saved</li>
                <li style="margin-bottom: 8px;">Load the saved configuration: <code style="background-color: #e9ecef; color: #495057; padding: 2px 4px; border-radius: 3px; font-size: 0.9em;">load_cradle_config_from_json('your_file.json')</code></li>
                <li style="margin-bottom: 0;">Add to your config list: <code style="background-color: #e9ecef; color: #495057; padding: 2px 4px; border-radius: 3px; font-size: 0.9em;">config_list.append(config)</code></li>
            </ol>
            <div style="background-color: #d1ecf1; border: 1px solid #bee5eb; padding: 10px; border-radius: 4px; margin-top: 15px;">
                <strong>✨ Simplified:</strong> No buttons needed! Just complete the UI and click "Finish" - the configuration will be automatically saved to your specified location.
            </div>
        </div>
        """))


def create_cradle_config_widget(base_config=None, 
                               job_type: str = "training",
                               width: str = "100%", 
                               height: str = "800px",
                               server_port: int = 8001,
                               workflow_context: Optional[Dict[str, Any]] = None) -> CradleConfigWidget:
    """
    Create a Cradle Configuration Widget for Jupyter notebooks with workflow integration.
    
    Args:
        base_config: Base pipeline configuration object
        job_type: Type of job (training, validation, testing, calibration)
        width: Widget width
        height: Widget height
        server_port: Port where the UI server is running
        workflow_context: Workflow context from DAG analysis and step structure
        
    Returns:
        CradleConfigWidget instance
        
    Example:
        ```python
        # Basic usage (backward compatible):
        cradle_widget = create_cradle_config_widget(
            base_config=base_config,
            job_type="training"
        )
        cradle_widget.display()
        
        # Enhanced usage with workflow context:
        cradle_widget = create_cradle_config_widget(
            base_config=base_config,
            job_type="training",
            workflow_context=workflow_context
        )
        cradle_widget.display()
        
        # The configuration will be automatically saved when you click "Finish" in the UI
        # Load it afterwards:
        # config = load_cradle_config_from_json('your_file.json')
        # config_list.append(config)
        ```
    """
    return CradleConfigWidget(
        base_config=base_config,
        job_type=job_type,
        width=width,
        height=height,
        server_port=server_port,
        workflow_context=workflow_context
    )


# Enhanced widget with server management
class CradleConfigWidgetWithServer(CradleConfigWidget):
    """Enhanced Cradle Config Widget that can start/stop its own server."""
    
    def __init__(self, *args, **kwargs):
        self.server_process = None
        super().__init__(*args, **kwargs)
    
    def start_server(self):
        """Start the UI server if not already running."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=2)
            if response.status_code == 200:
                with self.status_output:
                    print("✅ Server is already running")
                return True
        except requests.exceptions.RequestException:
            pass
        
        try:
            import subprocess
            import sys
            
            cmd = [
                sys.executable, "-m", "uvicorn",
                "cursus.api.cradle_ui.app:app",
                "--host", "0.0.0.0",
                "--port", str(self.server_port),
                "--reload"
            ]
            
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            time.sleep(3)
            
            try:
                response = requests.get(f"{self.server_url}/health", timeout=5)
                if response.status_code == 200:
                    with self.status_output:
                        print(f"✅ Server started successfully on port {self.server_port}")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            with self.status_output:
                print(f"❌ Failed to start server on port {self.server_port}")
            return False
            
        except Exception as e:
            with self.status_output:
                print(f"❌ Error starting server: {str(e)}")
            return False
    
    def stop_server(self):
        """Stop the UI server."""
        if self.server_process:
            self.server_process.terminate()
            self.server_process = None
            with self.status_output:
                print("🛑 Server stopped")
    
    def display(self):
        """Display the widget and start server if needed."""
        if not self.start_server():
            display(HTML("""
            <div style="background-color: #ffe6e6; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <h4>⚠️ Server Not Available</h4>
                <p>The Cradle UI server is not running. Please start it manually:</p>
                <code>cd src/cursus/api/cradle_ui && uvicorn app:app --host 0.0.0.0 --port 8001 --reload</code>
            </div>
            """))
            return
        
        super().display()
