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
import os
import re

# Handle both relative and absolute imports using centralized path setup
try:
    # Try relative imports first (when run as module)
    from ....core.base.config_base import BasePipelineConfig
except ImportError:
    # Fallback: Set up cursus path and use absolute imports
    import sys
    from pathlib import Path
    
    # Add the core directory to path for import_utils
    current_dir = Path(__file__).parent
    core_dir = current_dir.parent / 'core'
    if str(core_dir) not in sys.path:
        sys.path.insert(0, str(core_dir))
    
    from ..core.import_utils import ensure_cursus_path
    ensure_cursus_path()
    
    from cursus.core.base.config_base import BasePipelineConfig


# Shared utilities and constants
class WidgetUtils:
    """Shared utility functions for all widgets."""
    
    @staticmethod
    def create_status_output(max_height: str = '300px') -> widgets.Output:
        """Create a standardized status output widget."""
        return widgets.Output(
            layout=widgets.Layout(
                width='100%',
                max_height=max_height,
                overflow='auto',
                border='1px solid #ddd',
                padding='10px'
            )
        )
    
    @staticmethod
    def create_iframe(url: str, width: str, height: str, widget_id: str) -> widgets.HTML:
        """Create a standardized iframe widget."""
        return widgets.HTML(
            value=f'''
            <iframe 
                src="{url}" 
                width="{width}" 
                height="{height}"
                style="border: 1px solid #ccc; border-radius: 4px;"
                id="config-ui-iframe-{widget_id}">
            </iframe>
            '''
        )
    
    @staticmethod
    def create_button(description: str, button_style: str, width: str = '200px', 
                     height: str = None, tooltip: str = None) -> widgets.Button:
        """Create a standardized button widget."""
        layout = widgets.Layout(width=width)
        if height:
            layout.height = height
            
        button = widgets.Button(
            description=description,
            button_style=button_style,
            layout=layout
        )
        
        if tooltip:
            button.tooltip = tooltip
            
        return button
    
    @staticmethod
    def extract_base_config_params(base_config, config_class_name: str = None) -> Dict[str, Any]:
        """Extract parameters from base_config to pre-populate forms."""
        if not base_config:
            params = {}
            if config_class_name:
                params['config_class_name'] = config_class_name
            return params
        
        params = {}
        if config_class_name:
            params['config_class_name'] = config_class_name
        
        # Extract common BasePipelineConfig fields
        field_names = ['author', 'bucket', 'role', 'region', 'service_name', 
                      'pipeline_version', 'project_root_folder']
        
        for field_name in field_names:
            if hasattr(base_config, field_name):
                value = getattr(base_config, field_name)
                if value is not None:
                    params[field_name] = value
        
        # Convert base_config to JSON for API
        if hasattr(base_config, 'model_dump'):
            params['base_config'] = json.dumps(base_config.model_dump())
        
        return params
    
    @staticmethod
    def build_url_with_params(base_url: str, params: Dict[str, Any]) -> str:
        """Build URL with query parameters."""
        if not params:
            return base_url
            
        param_pairs = []
        for key, value in params.items():
            if value is not None:
                param_pairs.append(f"{key}={value}")
        
        if param_pairs:
            return f"{base_url}?{'&'.join(param_pairs)}"
        return base_url
    
    @staticmethod
    def handle_api_response(response: requests.Response, status_output: widgets.Output, 
                           success_callback=None, error_callback=None):
        """Handle API response with standardized error handling."""
        try:
            if response.status_code == 200:
                data = response.json()
                if success_callback:
                    success_callback(data)
                return data
            else:
                error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                error_msg = error_data.get('detail', f'HTTP {response.status_code}')
                with status_output:
                    print(f"❌ Request failed: {error_msg}")
                if error_callback:
                    error_callback(error_msg)
                return None
        except Exception as e:
            with status_output:
                print(f"❌ Error processing response: {str(e)}")
            if error_callback:
                error_callback(str(e))
            return None


class BaseConfigWidget:
    """Base class for all configuration widgets with common functionality."""
    
    def __init__(self, width: str = "100%", height: str = "800px", server_port: int = 8003):
        """Initialize base widget properties."""
        self.width = width
        self.height = height
        self.server_port = server_port
        self.server_url = f"http://localhost:{server_port}"
        self.widget_id = str(uuid.uuid4())
        self.status_output = WidgetUtils.create_status_output()
        self.server_process = None
    
    def _make_api_request(self, method: str, endpoint: str, json_data: Dict = None, 
                         timeout: int = 5) -> Optional[Dict[str, Any]]:
        """Make API request with standardized error handling."""
        try:
            url = f"{self.server_url}{endpoint}"
            if method.upper() == 'GET':
                response = requests.get(url, timeout=timeout)
            elif method.upper() == 'POST':
                response = requests.post(url, json=json_data, timeout=timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            return WidgetUtils.handle_api_response(response, self.status_output)
        except requests.exceptions.RequestException as e:
            with self.status_output:
                print(f"❌ Network error: {str(e)}")
            return None
    
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
            
            # Try multiple command approaches for both pip-installed and development setups
            commands_to_try = [
                # Approach 1: Use module execution (works for pip-installed packages)
                [sys.executable, "-m", "cursus.api.config_ui.start_server", "--host", "0.0.0.0", "--port", str(self.server_port)],
                # Approach 2: Direct script execution (works for development setups)
                [sys.executable, "src/cursus/api/config_ui/start_server.py", "--host", "0.0.0.0", "--port", str(self.server_port)]
            ]
            
            server_started = False
            for cmd in commands_to_try:
                try:
                    self.server_process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    server_started = True
                    break
                except Exception as e:
                    continue
            
            if not server_started:
                # Fallback: try with current working directory context
                try:
                    self.server_process = subprocess.Popen(
                        [sys.executable, "src/cursus/api/config_ui/start_server.py", "--host", "0.0.0.0", "--port", str(self.server_port)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd=str(Path.cwd())
                    )
                except Exception as e:
                    with self.status_output:
                        print(f"❌ Failed to start server with all approaches: {e}")
                    return False
            
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
        """Display the widget - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement display method")


class UniversalConfigWidget(BaseConfigWidget):
    """Jupyter widget for Universal Configuration Management."""
    
    def __init__(self, config_class_name: str, base_config=None, **kwargs):
        """Initialize the Universal Config Widget."""
        super().__init__(**kwargs)
        self.config_class_name = config_class_name
        self.base_config = base_config
        self.config_result = None
        self._create_widgets()
        
    def _create_widgets(self):
        """Create the widget components."""
        # Extract base config values and build URL
        base_config_params = WidgetUtils.extract_base_config_params(
            self.base_config, self.config_class_name
        )
        iframe_url = WidgetUtils.build_url_with_params(
            f"{self.server_url}/config-ui", base_config_params
        )
        
        # Create iframe
        self.iframe = WidgetUtils.create_iframe(iframe_url, self.width, self.height, self.widget_id)
        
        # Create buttons
        self.get_config_button = WidgetUtils.create_button(
            "Get Configuration", 'success', tooltip="Get the generated configuration"
        )
        self.get_config_button.disabled = True
        self.get_config_button.on_click(self._on_get_config_clicked)
        
        self.clear_config_button = WidgetUtils.create_button(
            "Clear Configuration", 'warning', tooltip="Clear the current configuration"
        )
        self.clear_config_button.on_click(self._on_clear_config_clicked)
        
        # Layout
        button_box = widgets.HBox([self.get_config_button, self.clear_config_button])
        self.widget = widgets.VBox([
            widgets.HTML(f"<h3>Universal Configuration: {self.config_class_name} (ID: {self.widget_id[:8]})</h3>"),
            self.iframe,
            button_box,
            self.status_output
        ])
        
        # Don't start automatic polling - let user manually check
        # Initial status message will be shown only once when widget is displayed
    
    def _start_config_polling(self):
        """Start polling for configuration availability (disabled by default)."""
        # Polling disabled to prevent continuous 404 errors
        # Users should manually click "Get Configuration" after completing the form
        pass
    
    def _on_get_config_clicked(self, button):
        """Handle get configuration button click."""
        data = self._make_api_request('GET', '/api/config-ui/get-latest-config')
        if data and data.get('config_type') == self.config_class_name:
            self.config_result = data.get('config')
            with self.status_output:
                print(f"✅ Configuration retrieved successfully!")
                print(f"Configuration type: {data.get('config_type')}")
                print(f"Fields: {len(self.config_result) if self.config_result else 0}")
                print(f"Timestamp: {data.get('timestamp')}")
                print("\n📋 Use widget.get_config() to access the configuration object")
        elif data:
            with self.status_output:
                print(f"❌ Configuration type mismatch. Expected: {self.config_class_name}, Got: {data.get('config_type')}")
    
    def _on_clear_config_clicked(self, button):
        """Handle clear configuration button click."""
        data = self._make_api_request('POST', '/api/config-ui/clear-config')
        if data:
            self.config_result = None
            self.get_config_button.disabled = True
            with self.status_output:
                print("🗑️ Configuration cleared")
    
    def get_config(self) -> Optional[Dict[str, Any]]:
        """Get the generated configuration object."""
        return self.config_result
    
    def display(self):
        """Display the widget."""
        display(self.widget)
        
        display(HTML(f"""
        <div style="background-color: #f8f9fa; border: 1px solid #dee2e6; padding: 20px; border-radius: 8px; margin: 15px 0; color: #212529;">
            <div style="background-color: #d1ecf1; border: 1px solid #bee5eb; padding: 15px; border-radius: 6px; margin-bottom: 15px;">
                <strong>💡 Quick Start:</strong> Complete the configuration form above, then click 'Get Configuration' to retrieve it.
            </div>
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


class CompleteConfigUIWidget(BaseConfigWidget):
    """Complete Configuration UI Widget that offers the SAME experience as the web app."""
    
    def __init__(self, **kwargs):
        """Initialize the Complete Config UI Widget."""
        super().__init__(height="900px", **kwargs)
        self.merged_result = None
        self._create_widgets()
    
    def _create_widgets(self):
        """Create the complete config UI widget components."""
        # Create iframe with complete web interface
        iframe_url = f"{self.server_url}/config-ui"
        self.iframe = WidgetUtils.create_iframe(iframe_url, self.width, self.height, self.widget_id)
        
        # Create buttons
        self.get_merged_button = WidgetUtils.create_button(
            "📥 Get Merged Config", 'success', 
            tooltip="Get the merged configuration after using Save All Merged in the UI above"
        )
        self.get_merged_button.on_click(self._on_get_merged_clicked)
        
        self.refresh_button = WidgetUtils.create_button(
            "🔄 Refresh UI", 'info', tooltip="Refresh the configuration UI"
        )
        self.refresh_button.on_click(self._on_refresh_clicked)
        
        # Layout
        button_box = widgets.HBox([self.get_merged_button, self.refresh_button])
        self.widget = widgets.VBox([
            widgets.HTML(f"<h3>🎯 Complete Configuration UI - Same as Web App (ID: {self.widget_id[:8]})</h3>"),
            self.iframe,
            button_box,
            self.status_output
        ])
    
    def _on_get_merged_clicked(self, button):
        """Handle get merged configuration button click."""
        data = self._make_api_request('GET', '/api/config-ui/get-latest-merged-config')
        if data:
            self.merged_result = data
            with self.status_output:
                print(f"✅ Merged configuration retrieved successfully!")
                print(f"📄 Filename: {data.get('filename', 'N/A')}")
                print(f"📊 Configurations merged: {len(data.get('session_configs', {}))}")
                print(f"🔗 Download URL: {data.get('download_url', 'N/A')}")
                print(f"⏰ Timestamp: {data.get('timestamp', 'N/A')}")
                print("\n📋 Use widget.get_merged_result() to access the merged configuration")
        elif data is None:
            # Check if it's a 404 (no config available)
            with self.status_output:
                print("ℹ️ No merged configuration available yet.")
                print("💡 Use the 'Save All Merged' button in the UI above first.")
    
    def _on_refresh_clicked(self, button):
        """Handle refresh UI button click."""
        iframe_url = f"{self.server_url}/config-ui?refresh={int(time.time())}"
        self.iframe.value = f'''
        <iframe 
            src="{iframe_url}" 
            width="{self.width}" 
            height="{self.height}"
            style="border: 1px solid #ccc; border-radius: 4px;"
            id="complete-config-ui-iframe-{self.widget_id}">
        </iframe>
        '''
        with self.status_output:
            print("🔄 Configuration UI refreshed")
    
    def get_merged_result(self) -> Optional[Dict[str, Any]]:
        """Get the last merged configuration result."""
        return self.merged_result
    
    def display(self):
        """Display the complete config UI widget."""
        display(self.widget)
        display(HTML("""
        <div style="background-color: #f8f9fa; border: 1px solid #dee2e6; padding: 20px; border-radius: 8px; margin: 15px 0; color: #212529;">
            <h4 style="color: #495057; margin-bottom: 15px; font-weight: 600;">🎯 Complete Configuration UI - Same Experience as Web App</h4>
            
            <div style="background-color: #d1ecf1; border: 1px solid #bee5eb; padding: 15px; border-radius: 6px; margin-bottom: 15px;">
                <h5 style="color: #0c5460; margin-bottom: 10px;">✨ What You Get (Identical to Web App):</h5>
                <ul style="color: #0c5460; margin: 0; padding-left: 20px; line-height: 1.6;">
                    <li><strong>Multi-Configuration Forms:</strong> BasePipelineConfig, ProcessingStepConfigBase, ModelWikiGeneratorConfig, PayloadConfig, etc.</li>
                    <li><strong>Real-time Validation:</strong> Field-specific error messages and auto-scroll to errors</li>
                    <li><strong>3-Tier Field Organization:</strong> Required, Processing, Optional sections</li>
                    <li><strong>Smart Save All Merged:</strong> Intelligent filename generation and location options</li>
                    <li><strong>Current Directory Saving:</strong> Perfect for Jupyter notebook workflows</li>
                </ul>
            </div>
            
            <h5 style="color: #495057; margin-bottom: 10px;">📝 How to Use (Same as Web App):</h5>
            <ol style="color: #495057; line-height: 1.6; margin: 0; padding-left: 20px;">
                <li style="margin-bottom: 8px;"><strong>Fill Configuration Forms:</strong> Complete the forms in the UI above</li>
                <li style="margin-bottom: 8px;"><strong>Real-time Validation:</strong> See field-specific errors and auto-scroll to problems</li>
                <li style="margin-bottom: 8px;"><strong>Click "Save All Merged":</strong> Use the enhanced save dialog with smart filename defaults</li>
                <li style="margin-bottom: 8px;"><strong>Smart Filename:</strong> Automatically generates config_{service_name}_{region}.json</li>
                <li style="margin-bottom: 8px;"><strong>Choose Current Directory:</strong> File saves where your Jupyter notebook runs</li>
                <li style="margin-bottom: 0;"><strong>Immediate Use:</strong> <code style="background-color: #e9ecef; color: #495057; padding: 2px 4px; border-radius: 3px; font-size: 0.9em;">load_configs('config_file.json')</code></li>
            </ol>
            
            <div style="background-color: #d4edda; border: 1px solid #c3e6cb; padding: 15px; border-radius: 6px; margin-top: 15px;">
                <h5 style="color: #155724; margin-bottom: 10px;">🎉 Perfect Integration with demo_config.ipynb:</h5>
                <p style="color: #155724; margin: 0; line-height: 1.6;">
                    This widget provides the <strong>exact same experience</strong> as the web app, but embedded in Jupyter. 
                    Fill forms → Save All Merged → File appears in current directory → Ready for pipeline execution!
                </p>
            </div>
            
            <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 4px; margin-top: 10px;">
                <strong>💡 Pro Tip:</strong> After using "Save All Merged" in the UI above, click "Get Merged Config" below to access the result programmatically.
            </div>
        </div>
        """))


class EnhancedSaveAllMergedWidget(BaseConfigWidget):
    """Enhanced Jupyter widget for Save All Merged functionality with smart filename defaults."""
    
    def __init__(self, session_configs: Dict[str, Dict[str, Any]] = None, **kwargs):
        """Initialize the Enhanced Save All Merged Widget."""
        super().__init__(height="600px", **kwargs)
        self.session_configs = session_configs or {}
        self.merged_result = None
        self._save_in_progress = False
        self._create_widgets()
    
    def _create_widgets(self):
        """Create the enhanced save all merged widget components."""
        # Smart filename generation
        smart_filename = self._generate_smart_filename()
        
        # Filename input
        self.filename_input = widgets.Text(
            value=smart_filename,
            placeholder="config_service_region.json",
            description="📄 Filename:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        # Save location dropdown
        self.location_dropdown = widgets.Dropdown(
            options=[
                ('📂 Current Directory (where Jupyter notebook runs)', 'current'),
                ('⬇️ Downloads Folder', 'downloads'),
                ('📁 Custom Location (browser default)', 'custom')
            ],
            value='current',
            description="📁 Save Location:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        # Preview display
        self.preview_output = widgets.HTML(
            value=self._generate_preview_text(),
            layout=widgets.Layout(
                width='100%',
                border='1px solid #e0e0e0',
                padding='10px',
                margin='10px 0'
            )
        )
        
        # Bind events for real-time preview updates
        self.filename_input.observe(self._update_preview, names='value')
        self.location_dropdown.observe(self._update_preview, names='value')
        
        # Create buttons
        self.save_merged_button = WidgetUtils.create_button(
            "💾 Save All Merged", 'success', height='40px',
            tooltip="Create unified hierarchical JSON configuration"
        )
        self.save_merged_button.on_click(self._on_save_merged_clicked)
        
        self.add_config_button = WidgetUtils.create_button(
            "➕ Add Configuration", 'info', tooltip="Add a configuration to the merge list"
        )
        self.add_config_button.on_click(self._on_add_config_clicked)
        
        self.clear_all_button = WidgetUtils.create_button(
            "🗑️ Clear All", 'warning'
        )
        self.clear_all_button.on_click(self._on_clear_all_clicked)
        
        # Configuration summary
        self.config_summary = widgets.HTML(
            value=self._generate_config_summary(),
            layout=widgets.Layout(width='100%', margin='10px 0')
        )
        
        # Layout
        button_box = widgets.HBox([
            self.save_merged_button,
            self.add_config_button,
            self.clear_all_button
        ])
        
        save_options_box = widgets.VBox([
            widgets.HTML("<h4>💾 Save Configuration File</h4>"),
            self.filename_input,
            self.location_dropdown,
            widgets.HTML("<h4>💡 Save Preview:</h4>"),
            self.preview_output
        ])
        
        self.widget = widgets.VBox([
            widgets.HTML(f"<h3>Enhanced Save All Merged (ID: {self.widget_id[:8]})</h3>"),
            self.config_summary,
            save_options_box,
            button_box,
            self.status_output
        ])
    
    def _generate_smart_filename(self) -> str:
        """Generate smart default filename based on configuration data."""
        service_name = 'pipeline'
        region = 'default'
        
        # Extract service_name and region from session configs
        for config_name, config_data in self.session_configs.items():
            if isinstance(config_data, dict):
                if 'service_name' in config_data and config_data['service_name']:
                    service_name = config_data['service_name']
                if 'region' in config_data and config_data['region']:
                    region = config_data['region']
                break
        
        # Clean up for filename safety
        service_name = re.sub(r'[^a-zA-Z0-9_-]', '_', str(service_name))
        region = re.sub(r'[^a-zA-Z0-9_-]', '_', str(region))
        
        return f"config_{service_name}_{region}.json"
    
    def _generate_preview_text(self) -> str:
        """Generate preview text for save location and filename."""
        filename = self.filename_input.value if hasattr(self, 'filename_input') else 'config.json'
        location = self.location_dropdown.value if hasattr(self, 'location_dropdown') else 'current'
        
        location_text = {
            'current': 'current directory',
            'downloads': 'Downloads folder',
            'custom': 'browser default location'
        }.get(location, 'current directory')
        
        return f"""
        <div style="background-color: #f0f9ff; border: 1px solid #0ea5e9; border-radius: 6px; padding: 10px;">
            <strong>Will save as:</strong> <span style="color: #0ea5e9; font-weight: 600;">{filename}</span> in {location_text}
        </div>
        """
    
    def _update_preview(self, change):
        """Update preview text when filename or location changes."""
        self.preview_output.value = self._generate_preview_text()
    
    def _generate_config_summary(self) -> str:
        """Generate HTML summary of current configurations."""
        if not self.session_configs:
            return """
            <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 6px; padding: 15px; margin: 10px 0;">
                <h4 style="color: #856404; margin-bottom: 10px;">📋 No Configurations Added</h4>
                <p style="color: #856404; margin: 0;">Add configurations using the "Add Configuration" button or use <code>widget.add_config(name, config_data)</code></p>
            </div>
            """
        
        config_items = []
        for config_name, config_data in self.session_configs.items():
            field_count = len(config_data) if isinstance(config_data, dict) else 0
            config_items.append(f"<li><strong>{config_name}</strong>: {field_count} fields</li>")
        
        return f"""
        <div style="background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 6px; padding: 15px; margin: 10px 0;">
            <h4 style="color: #155724; margin-bottom: 10px;">🎯 Ready to Merge ({len(self.session_configs)} configurations)</h4>
            <ul style="color: #155724; margin: 0; padding-left: 20px;">
                {''.join(config_items)}
            </ul>
        </div>
        """
    
    def _on_save_merged_clicked(self, button):
        """Handle save all merged button click with enhanced user experience and deduplication."""
        # Deduplication protection
        if self._save_in_progress:
            with self.status_output:
                print("⏳ Save operation already in progress. Please wait...")
            return
        
        if not self.session_configs:
            with self.status_output:
                print("❌ No configurations to merge. Add configurations first.")
            return
        
        # Mark save operation as in progress
        self._save_in_progress = True
        self.save_merged_button.disabled = True
        self.save_merged_button.description = "⏳ Saving..."
        
        filename = self.filename_input.value or 'config.json'
        location = self.location_dropdown.value
        
        try:
            with self.status_output:
                print(f"🔄 Creating unified configuration file...")
                print(f"📁 Filename: {filename}")
                print(f"📂 Location: {location}")
            
            # Call the merge API
            data = self._make_api_request('POST', '/api/config-ui/merge-and-save-configs', {
                'session_configs': self.session_configs,
                'filename': filename,
                'workspace_dirs': None
            }, timeout=10)
            
            if data:
                self.merged_result = data
                self._save_file_with_location(data, filename, location)
                
                with self.status_output:
                    print(f"✅ Successfully merged {len(self.session_configs)} configurations!")
                    print(f"📄 Generated: {data.get('filename', filename)}")
                    print(f"🔗 Download URL: {data.get('download_url', 'N/A')}")
                    print(f"📊 Structure: {len(data.get('merged_config', {}).get('shared', {}))} shared fields")
                    print("\n🎉 Configuration ready for pipeline execution!")
                    
        finally:
            # Reset button state regardless of success or failure
            self._save_in_progress = False
            self.save_merged_button.disabled = False
            self.save_merged_button.description = "💾 Save All Merged"
    
    def _save_file_with_location(self, result, filename, location):
        """Save file based on user's location preference."""
        try:
            # Get the configuration data
            if 'download_url' in result:
                download_response = requests.get(f"{self.server_url}{result['download_url']}", timeout=10)
                if download_response.status_code == 200:
                    config_data = download_response.json()
                else:
                    config_data = result.get('merged_config', {})
            else:
                config_data = result.get('merged_config', {})
            
            # Determine save path based on location
            if location == 'current':
                save_path = Path.cwd() / filename
            elif location == 'downloads':
                downloads_path = Path.home() / 'Downloads'
                downloads_path.mkdir(exist_ok=True)
                save_path = downloads_path / filename
            else:  # custom
                save_path = Path.cwd() / filename
            
            # Write the file
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            with self.status_output:
                print(f"💾 File saved successfully: {save_path}")
                print(f"📁 Full path: {save_path.absolute()}")
                print(f"\n🚀 Usage in your notebook:")
                print(f"from cursus.core.config_fields import load_configs")
                print(f"config_list = load_configs('{filename}')")
                print(f"# Now ready for pipeline execution!")
                
        except Exception as e:
            with self.status_output:
                print(f"❌ File save error: {str(e)}")
                print(f"💡 You can still download via the API endpoint")
    
    def _on_add_config_clicked(self, button):
        """Handle add configuration button click."""
        with self.status_output:
            print("💡 To add configurations programmatically:")
            print("widget.add_config('ConfigName', config_data)")
            print("widget.add_config('BasePipelineConfig', base_config.model_dump())")
            print("widget.add_config('ProcessingStepConfigBase', processing_config.model_dump())")
    
    def _on_clear_all_clicked(self, button):
        """Handle clear all configurations button click."""
        self.session_configs.clear()
        self.merged_result = None
        self.config_summary.value = self._generate_config_summary()
        self.filename_input.value = self._generate_smart_filename()
        
        with self.status_output:
            print("🗑️ All configurations cleared")
    
    def add_config(self, config_name: str, config_data: Dict[str, Any]):
        """Add a configuration to the merge list."""
        self.session_configs[config_name] = config_data
        self.config_summary.value = self._generate_config_summary()
        self.filename_input.value = self._generate_smart_filename()
        
        with self.status_output:
            print(f"✅ Added {config_name} with {len(config_data) if isinstance(config_data, dict) else 0} fields")
    
    def remove_config(self, config_name: str):
        """Remove a configuration from the merge list."""
        if config_name in self.session_configs:
            del self.session_configs[config_name]
            self.config_summary.value = self._generate_config_summary()
            self.filename_input.value = self._generate_smart_filename()
            
            with self.status_output:
                print(f"🗑️ Removed {config_name}")
        else:
            with self.status_output:
                print(f"❌ Configuration {config_name} not found")
    
    def get_merged_result(self) -> Optional[Dict[str, Any]]:
        """Get the last merged result."""
        return self.merged_result
    
    def display(self):
        """Display the enhanced save all merged widget."""
        display(self.widget)
        display(HTML("""
        <div style="background-color: #f8f9fa; border: 1px solid #dee2e6; padding: 20px; border-radius: 8px; margin: 15px 0; color: #212529;">
            <h4 style="color: #495057; margin-bottom: 15px; font-weight: 600;">📝 Enhanced Save All Merged Usage:</h4>
            <ol style="color: #495057; line-height: 1.6; margin: 0; padding-left: 20px;">
                <li style="margin-bottom: 8px;">Add configurations: <code style="background-color: #e9ecef; color: #495057; padding: 2px 4px; border-radius: 3px; font-size: 0.9em;">widget.add_config('BasePipelineConfig', base_config.model_dump())</code></li>
                <li style="margin-bottom: 8px;">Smart filename will be generated automatically: <strong>config_{service_name}_{region}.json</strong></li>
                <li style="margin-bottom: 8px;">Choose save location (current directory recommended for Jupyter)</li>
                <li style="margin-bottom: 8px;">Click <strong>"Save All Merged"</strong> to create unified configuration</li>
                <li style="margin-bottom: 0;">File will be saved directly where you can use it immediately</li>
            </ol>
            <div style="background-color: #d1ecf1; border: 1px solid #bee5eb; padding: 10px; border-radius: 4px; margin-top: 15px;">
                <strong>✨ Enhanced Features:</strong> Smart filename defaults, current directory saving (Jupyter-friendly), real-time preview, and seamless integration with demo_config.ipynb workflow.
            </div>
        </div>
        """))


# Factory functions for easy widget creation
def create_config_widget(config_class_name: str, base_config=None, **kwargs) -> UniversalConfigWidget:
    """Create a Universal Configuration Widget for Jupyter notebooks."""
    return UniversalConfigWidget(config_class_name=config_class_name, base_config=base_config, **kwargs)


def create_complete_config_ui_widget(**kwargs) -> CompleteConfigUIWidget:
    """Create a Complete Configuration UI Widget that offers the SAME experience as the web app."""
    return CompleteConfigUIWidget(**kwargs)


def create_enhanced_save_all_merged_widget(session_configs: Dict[str, Dict[str, Any]] = None, **kwargs) -> EnhancedSaveAllMergedWidget:
    """Create an Enhanced Save All Merged Widget for Jupyter notebooks."""
    return EnhancedSaveAllMergedWidget(session_configs=session_configs, **kwargs)
