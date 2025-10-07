"""
Jupyter Notebook Widget for Universal Configuration Management

This module provides a Jupyter widget interface for the Universal Config UI
that can be used directly in notebooks to replace manual configuration blocks.
"""

import ipywidgets as widgets
from IPython.display import display, HTML, Javascript
import json
import requests
from typing import Optional, Dict, Any, List, Union
import asyncio
import threading
import time
from pathlib import Path
import uuid
import weakref

from ...core.base.config_base import BasePipelineConfig


class UniversalConfigWidget:
    """
    Jupyter widget for Universal Configuration Management.
    
    This widget provides an embedded UI for configuring any configuration type
    that can be used directly in Jupyter notebooks.
    """
    
    def __init__(self, 
                 config_class_name: str,
                 base_config=None,
                 width: str = "100%", 
                 height: str = "800px",
                 server_port: int = 8003):
        """
        Initialize the Universal Config Widget.
        
        Args:
            config_class_name: Name of the configuration class to create
            base_config: Base pipeline configuration object
            width: Widget width
            height: Widget height
            server_port: Port where the UI server is running
        """
        self.config_class_name = config_class_name
        self.base_config = base_config
        self.width = width
        self.height = height
        self.server_port = server_port
        self.config_result = None
        self.server_url = f"http://localhost:{server_port}"
        
        # Unique identifier for this widget instance
        self.widget_id = str(uuid.uuid4())
        
        # Create the widget components
        self._create_widgets()
        
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
        iframe_url = f"{self.server_url}/config-ui"
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
                id="config-ui-iframe-{self.widget_id}">
            </iframe>
            '''
        )
        
        # Get Configuration button
        self.get_config_button = widgets.Button(
            description="Get Configuration",
            button_style='success',
            layout=widgets.Layout(width='200px'),
            disabled=True
        )
        self.get_config_button.on_click(self._on_get_config_clicked)
        
        # Clear Configuration button
        self.clear_config_button = widgets.Button(
            description="Clear Configuration",
            button_style='warning',
            layout=widgets.Layout(width='200px')
        )
        self.clear_config_button.on_click(self._on_clear_config_clicked)
        
        # Button layout
        button_box = widgets.HBox([
            self.get_config_button,
            self.clear_config_button
        ])
        
        # Layout - iframe, buttons, and status
        self.widget = widgets.VBox([
            widgets.HTML(f"<h3>Universal Configuration: {self.config_class_name} (ID: {self.widget_id[:8]})</h3>"),
            self.iframe,
            button_box,
            self.status_output
        ])
        
        # Start polling for configuration availability
        self._start_config_polling()
    
    def _extract_base_config_params(self):
        """Extract parameters from base_config to pre-populate the form."""
        if not self.base_config:
            return {'config_class_name': self.config_class_name}
        
        params = {'config_class_name': self.config_class_name}
        
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
            
            # Convert base_config to JSON for API
            if hasattr(self.base_config, 'model_dump'):
                params['base_config'] = json.dumps(self.base_config.model_dump())
            
        except Exception as e:
            # If there's any error extracting config, log it but don't fail
            with self.status_output:
                print(f"⚠️ Warning: Could not extract some base config values: {str(e)}")
        
        return params
    
    def _start_config_polling(self):
        """Start polling for configuration availability."""
        def poll_config():
            while True:
                try:
                    response = requests.get(f"{self.server_url}/api/config-ui/get-latest-config", timeout=2)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('config_type') == self.config_class_name:
                            self.get_config_button.disabled = False
                            with self.status_output:
                                print(f"✅ Configuration ready for {self.config_class_name}")
                            break
                    elif response.status_code == 404:
                        # No config available yet
                        pass
                except requests.exceptions.RequestException:
                    # Server not available or other error
                    pass
                
                time.sleep(2)  # Poll every 2 seconds
        
        # Start polling in background thread
        polling_thread = threading.Thread(target=poll_config, daemon=True)
        polling_thread.start()
    
    def _on_get_config_clicked(self, button):
        """Handle get configuration button click."""
        try:
            response = requests.get(f"{self.server_url}/api/config-ui/get-latest-config", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('config_type') == self.config_class_name:
                    self.config_result = data.get('config')
                    with self.status_output:
                        print(f"✅ Configuration retrieved successfully!")
                        print(f"Configuration type: {data.get('config_type')}")
                        print(f"Fields: {len(self.config_result) if self.config_result else 0}")
                        print(f"Timestamp: {data.get('timestamp')}")
                        print("\n📋 Use widget.get_config() to access the configuration object")
                else:
                    with self.status_output:
                        print(f"❌ Configuration type mismatch. Expected: {self.config_class_name}, Got: {data.get('config_type')}")
            else:
                with self.status_output:
                    print(f"❌ Failed to retrieve configuration: HTTP {response.status_code}")
        except Exception as e:
            with self.status_output:
                print(f"❌ Error retrieving configuration: {str(e)}")
    
    def _on_clear_config_clicked(self, button):
        """Handle clear configuration button click."""
        try:
            response = requests.post(f"{self.server_url}/api/config-ui/clear-config", timeout=5)
            if response.status_code == 200:
                self.config_result = None
                self.get_config_button.disabled = True
                with self.status_output:
                    print("🗑️ Configuration cleared")
            else:
                with self.status_output:
                    print(f"❌ Failed to clear configuration: HTTP {response.status_code}")
        except Exception as e:
            with self.status_output:
                print(f"❌ Error clearing configuration: {str(e)}")
    
    def get_config(self) -> Optional[Dict[str, Any]]:
        """
        Get the generated configuration object.
        
        Returns:
            Configuration dictionary if available, None otherwise
        """
        return self.config_result
    
    def display(self):
        """Display the widget."""
        display(self.widget)
        
        # Display usage instructions
        display(HTML(f"""
        <div style="background-color: #f8f9fa; border: 1px solid #dee2e6; padding: 20px; border-radius: 8px; margin: 15px 0; color: #212529;">
            <h4 style="color: #495057; margin-bottom: 15px; font-weight: 600;">📝 How to Use:</h4>
            <ol style="color: #495057; line-height: 1.6; margin: 0; padding-left: 20px;">
                <li style="margin-bottom: 8px;">Complete the configuration form in the UI above for <strong>{self.config_class_name}</strong></li>
                <li style="margin-bottom: 8px;">Click <strong>"Save Configuration"</strong> in the UI</li>
                <li style="margin-bottom: 8px;">Click <strong>"Get Configuration"</strong> button below (will be enabled when ready)</li>
                <li style="margin-bottom: 8px;">Access the configuration: <code style="background-color: #e9ecef; color: #495057; padding: 2px 4px; border-radius: 3px; font-size: 0.9em;">config = widget.get_config()</code></li>
                <li style="margin-bottom: 0;">Create config instance: <code style="background-color: #e9ecef; color: #495057; padding: 2px 4px; border-radius: 3px; font-size: 0.9em;">config_instance = {self.config_class_name}(**config)</code></li>
            </ol>
            <div style="background-color: #d1ecf1; border: 1px solid #bee5eb; padding: 10px; border-radius: 4px; margin-top: 15px;">
                <strong>✨ Enhanced Features:</strong> Real-time validation, field-specific error messages, auto-scroll to errors, and comprehensive Pydantic validation support.
            </div>
        </div>
        """))


class PipelineConfigWidget:
    """
    Jupyter widget for Pipeline Configuration Management.
    
    This widget provides a multi-step wizard for configuring complete pipelines
    that can be used directly in Jupyter notebooks.
    """
    
    def __init__(self, 
                 dag_definition: Dict[str, Any],
                 base_config=None,
                 processing_config=None,
                 width: str = "100%", 
                 height: str = "900px",
                 server_port: int = 8003):
        """
        Initialize the Pipeline Config Widget.
        
        Args:
            dag_definition: Pipeline DAG definition
            base_config: Base pipeline configuration object
            processing_config: Processing configuration object
            width: Widget width
            height: Widget height
            server_port: Port where the UI server is running
        """
        self.dag_definition = dag_definition
        self.base_config = base_config
        self.processing_config = processing_config
        self.width = width
        self.height = height
        self.server_port = server_port
        self.config_list = None
        self.server_url = f"http://localhost:{server_port}"
        
        # Unique identifier for this widget instance
        self.widget_id = str(uuid.uuid4())
        
        # Create the widget components
        self._create_widgets()
        
    def _create_widgets(self):
        """Create the widget components."""
        # Status display
        self.status_output = widgets.Output(
            layout=widgets.Layout(
                width='100%',
                max_height='400px',
                overflow='auto',
                border='1px solid #ddd',
                padding='10px'
            )
        )
        
        # Build URL for pipeline wizard
        iframe_url = f"{self.server_url}/config-ui"  # Will be enhanced for pipeline mode
        
        # Main iframe for the UI
        self.iframe = widgets.HTML(
            value=f'''
            <iframe 
                src="{iframe_url}" 
                width="{self.width}" 
                height="{self.height}"
                style="border: 1px solid #ccc; border-radius: 4px;"
                id="pipeline-config-iframe-{self.widget_id}">
            </iframe>
            '''
        )
        
        # Get Configuration List button
        self.get_configs_button = widgets.Button(
            description="Get Configuration List",
            button_style='success',
            layout=widgets.Layout(width='200px'),
            disabled=True
        )
        self.get_configs_button.on_click(self._on_get_configs_clicked)
        
        # Clear Configurations button
        self.clear_configs_button = widgets.Button(
            description="Clear Configurations",
            button_style='warning',
            layout=widgets.Layout(width='200px')
        )
        self.clear_configs_button.on_click(self._on_clear_configs_clicked)
        
        # Button layout
        button_box = widgets.HBox([
            self.get_configs_button,
            self.clear_configs_button
        ])
        
        # Layout
        self.widget = widgets.VBox([
            widgets.HTML(f"<h3>Pipeline Configuration Wizard (ID: {self.widget_id[:8]})</h3>"),
            self.iframe,
            button_box,
            self.status_output
        ])
    
    def _on_get_configs_clicked(self, button):
        """Handle get configuration list button click."""
        try:
            response = requests.get(f"{self.server_url}/api/config-ui/get-latest-config", timeout=5)
            if response.status_code == 200:
                data = response.json()
                # For pipeline wizard, we expect a list of configurations
                self.config_list = data.get('config_list', [])
                with self.status_output:
                    print(f"✅ Configuration list retrieved successfully!")
                    print(f"Number of configurations: {len(self.config_list)}")
                    print(f"Timestamp: {data.get('timestamp')}")
                    print("\n📋 Use widget.get_config_list() to access the configuration list")
            else:
                with self.status_output:
                    print(f"❌ Failed to retrieve configuration list: HTTP {response.status_code}")
        except Exception as e:
            with self.status_output:
                print(f"❌ Error retrieving configuration list: {str(e)}")
    
    def _on_clear_configs_clicked(self, button):
        """Handle clear configurations button click."""
        try:
            response = requests.post(f"{self.server_url}/api/config-ui/clear-config", timeout=5)
            if response.status_code == 200:
                self.config_list = None
                self.get_configs_button.disabled = True
                with self.status_output:
                    print("🗑️ Configuration list cleared")
            else:
                with self.status_output:
                    print(f"❌ Failed to clear configuration list: HTTP {response.status_code}")
        except Exception as e:
            with self.status_output:
                print(f"❌ Error clearing configuration list: {str(e)}")
    
    def get_config_list(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get the generated configuration list.
        
        Returns:
            List of configuration dictionaries if available, None otherwise
        """
        return self.config_list
    
    def display(self):
        """Display the widget."""
        display(self.widget)
        
        # Display usage instructions
        display(HTML("""
        <div style="background-color: #f8f9fa; border: 1px solid #dee2e6; padding: 20px; border-radius: 8px; margin: 15px 0; color: #212529;">
            <h4 style="color: #495057; margin-bottom: 15px; font-weight: 600;">📝 How to Use Pipeline Wizard:</h4>
            <ol style="color: #495057; line-height: 1.6; margin: 0; padding-left: 20px;">
                <li style="margin-bottom: 8px;">Complete all steps in the multi-step wizard above</li>
                <li style="margin-bottom: 8px;">Each step configures a different component of your pipeline</li>
                <li style="margin-bottom: 8px;">Click <strong>"Finish"</strong> in the final step</li>
                <li style="margin-bottom: 8px;">Click <strong>"Get Configuration List"</strong> button below</li>
                <li style="margin-bottom: 0;">Use the config list: <code style="background-color: #e9ecef; color: #495057; padding: 2px 4px; border-radius: 3px; font-size: 0.9em;">config_list = widget.get_config_list()</code></li>
            </ol>
            <div style="background-color: #d4edda; border: 1px solid #c3e6cb; padding: 10px; border-radius: 4px; margin-top: 15px;">
                <strong>🎯 Pipeline Ready:</strong> The configuration list will be in the correct order for <code>merge_and_save_configs(config_list, 'your_config.json')</code>
            </div>
        </div>
        """))


# Factory functions for easy widget creation
def create_config_widget(config_class_name: str,
                        base_config=None,
                        width: str = "100%", 
                        height: str = "800px",
                        server_port: int = 8003) -> UniversalConfigWidget:
    """
    Create a Universal Configuration Widget for Jupyter notebooks.
    
    Args:
        config_class_name: Name of the configuration class to create
        base_config: Base pipeline configuration object
        width: Widget width
        height: Widget height
        server_port: Port where the UI server is running
        
    Returns:
        UniversalConfigWidget instance
        
    Example:
        ```python
        # Replace manual configuration blocks with:
        widget = create_config_widget(
            config_class_name="ProcessingStepConfigBase",
            base_config=base_config
        )
        widget.display()
        
        # After completing the UI:
        config_data = widget.get_config()
        config_instance = ProcessingStepConfigBase(**config_data)
        config_list.append(config_instance)
        ```
    """
    return UniversalConfigWidget(
        config_class_name=config_class_name,
        base_config=base_config,
        width=width,
        height=height,
        server_port=server_port
    )


def create_pipeline_config_widget(dag_definition: Dict[str, Any],
                                 base_config=None,
                                 processing_config=None,
                                 width: str = "100%", 
                                 height: str = "900px",
                                 server_port: int = 8003) -> PipelineConfigWidget:
    """
    Create a Pipeline Configuration Widget for Jupyter notebooks.
    
    Args:
        dag_definition: Pipeline DAG definition
        base_config: Base pipeline configuration object
        processing_config: Processing configuration object
        width: Widget width
        height: Widget height
        server_port: Port where the UI server is running
        
    Returns:
        PipelineConfigWidget instance
        
    Example:
        ```python
        # Replace entire manual configuration workflow with:
        pipeline_widget = create_pipeline_config_widget(
            dag_definition=create_xgboost_complete_e2e_dag(),
            base_config=base_config,
            processing_config=processing_step_config
        )
        pipeline_widget.display()
        
        # After completing all steps:
        config_list = pipeline_widget.get_config_list()
        merged_config = merge_and_save_configs(config_list, 'config_NA_xgboost_AtoZ.json')
        ```
    """
    return PipelineConfigWidget(
        dag_definition=dag_definition,
        base_config=base_config,
        processing_config=processing_config,
        width=width,
        height=height,
        server_port=server_port
    )


# Enhanced widget with server management
class UniversalConfigWidgetWithServer(UniversalConfigWidget):
    """Enhanced Universal Config Widget that can start/stop its own server."""
    
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
                sys.executable, "-m", "src.cursus.api.config_ui.run_server",
                "--host", "0.0.0.0",
                "--port", str(self.server_port)
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
            display(HTML(f"""
            <div style="background-color: #ffe6e6; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <h4>⚠️ Server Not Available</h4>
                <p>The Universal Config UI server is not running. Please start it manually:</p>
                <code>python -m src.cursus.api.config_ui.run_server --port {self.server_port}</code>
            </div>
            """))
            return
        
        super().display()
