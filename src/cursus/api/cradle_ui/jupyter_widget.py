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

from ...steps.configs.config_cradle_data_loading_step import CradleDataLoadConfig

class CradleConfigWidget:
    """
    Jupyter widget for Cradle Data Load Configuration.
    
    This widget provides an embedded UI for configuring Cradle data loading
    that can be used directly in Jupyter notebooks.
    """
    
    def __init__(self, 
                 base_config=None,
                 job_type: str = "training",
                 width: str = "100%", 
                 height: str = "800px",
                 server_port: int = 8001):
        """
        Initialize the Cradle Config Widget.
        
        Args:
            base_config: Base pipeline configuration object
            job_type: Type of job (training, validation, testing, calibration)
            width: Widget width
            height: Widget height
            server_port: Port where the UI server is running
        """
        self.base_config = base_config
        self.job_type = job_type
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
    
    def get_config(self) -> Optional[CradleDataLoadConfig]:
        """
        Get the generated configuration object.
        
        Returns:
            CradleDataLoadConfig object if available, None otherwise
        """
        return self.config_result
    
    def display(self):
        """Display the widget."""
        display(self.widget)
        
        # Display usage instructions
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
                               server_port: int = 8001) -> CradleConfigWidget:
    """
    Create a Cradle Configuration Widget for Jupyter notebooks.
    
    Args:
        base_config: Base pipeline configuration object
        job_type: Type of job (training, validation, testing, calibration)
        width: Widget width
        height: Widget height
        server_port: Port where the UI server is running
        
    Returns:
        CradleConfigWidget instance
        
    Example:
        ```python
        # Replace the manual configuration block with:
        cradle_widget = create_cradle_config_widget(
            base_config=base_config,
            job_type="training"
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
        server_port=server_port
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
