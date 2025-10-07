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
        
        # Create the widget components
        self._create_widgets()
        
        # Start checking for configuration availability
        self._start_config_checker()
        
    def _create_widgets(self):
        """Create the widget components."""
        # Status display
        self.status_output = widgets.Output()
        
        # Configuration result display
        self.result_output = widgets.Output()
        
        # Main iframe for the UI
        self.iframe = widgets.HTML(
            value=f'''
            <iframe 
                src="{self.server_url}" 
                width="{self.width}" 
                height="{self.height}"
                style="border: 1px solid #ccc; border-radius: 4px;"
                id="cradle-config-iframe">
            </iframe>
            '''
        )
        
        # Control buttons
        self.get_config_btn = widgets.Button(
            description="Get Configuration",
            button_style='',  # Start with default style (greyed out)
            tooltip="Complete the configuration and click 'Finish' first",
            disabled=True  # Start disabled
        )
        self.get_config_btn.on_click(self._on_get_config_clicked)
        
        self.clear_btn = widgets.Button(
            description="Clear Results",
            button_style='info',
            tooltip="Clear the configuration results"
        )
        self.clear_btn.on_click(self._on_clear_clicked)
        
        # Layout
        self.widget = widgets.VBox([
            widgets.HTML("<h3>Cradle Data Load Configuration</h3>"),
            self.iframe,
            widgets.HBox([self.get_config_btn, self.clear_btn]),
            self.status_output,
            self.result_output
        ])
    
    def _handle_config_result(self, config_data: Dict[str, Any]):
        """Handle configuration result from the UI."""
        try:
            # Convert the config data to a CradleDataLoadConfig object
            self.config_result = CradleDataLoadConfig(**config_data)
            
            with self.status_output:
                self.status_output.clear_output()
                print("✅ Configuration generated successfully!")
                print(f"Job Type: {self.config_result.job_type}")
                print(f"Data Sources: {len(self.config_result.data_sources_spec.data_sources)}")
                
        except Exception as e:
            with self.status_output:
                self.status_output.clear_output()
                print(f"❌ Error processing configuration: {str(e)}")
    
    def _on_get_config_clicked(self, button):
        """Handle get configuration button click."""
        with self.status_output:
            self.status_output.clear_output()
            print("🔄 Attempting to retrieve configuration from server...")
        
        try:
            # Try to get the latest configuration from the server
            response = requests.get(f"{self.server_url}/api/cradle-ui/get-latest-config", timeout=10)
            
            if response.status_code == 200:
                config_data = response.json()
                
                # Prompt user for save location
                from tkinter import filedialog, messagebox
                import tkinter as tk
                
                # Create a temporary root window (hidden)
                root = tk.Tk()
                root.withdraw()
                
                # Ask user where to save the configuration
                file_path = filedialog.asksaveasfilename(
                    title="Save Cradle Configuration",
                    defaultextension=".json",
                    filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                    initialfile=f"cradle_config_{self.job_type}.json"
                )
                
                if file_path:
                    # Save configuration to JSON file
                    with open(file_path, 'w') as f:
                        json.dump(config_data, f, indent=2, default=str)
                    
                    with self.status_output:
                        self.status_output.clear_output()
                        print("✅ Configuration saved successfully!")
                        print("=" * 50)
                        print(f"📁 Saved to: {file_path}")
                        print()
                        print("🔧 To load this configuration in your notebook:")
                        print("```python")
                        print("from cursus.api.cradle_ui.utils.config_loader import load_cradle_config_from_json")
                        print()
                        print(f"# Load the configuration")
                        print(f"config = load_cradle_config_from_json('{file_path}')")
                        print("config_list.append(config)")
                        print("```")
                        print()
                        print("💡 The configuration is now ready to use in your pipeline!")
                    
                    # Store the config for immediate use using proper loader
                    from .utils.config_loader import _reconstruct_cradle_config
                    self.config_result = _reconstruct_cradle_config(config_data)
                    
                else:
                    with self.status_output:
                        self.status_output.clear_output()
                        print("⚠️ Save cancelled. Configuration not saved.")
                
                root.destroy()
                
            elif response.status_code == 404:
                with self.status_output:
                    self.status_output.clear_output()
                    print("⚠️ No configuration found on server.")
                    print("Please complete the configuration in the UI above and click 'Finish' first.")
                    
            else:
                with self.status_output:
                    self.status_output.clear_output()
                    print(f"❌ Server error: {response.status_code}")
                    print("Please try again or check the server logs.")
                    
        except requests.exceptions.RequestException as e:
            with self.status_output:
                self.status_output.clear_output()
                print("❌ Could not connect to server.")
                print(f"Error: {str(e)}")
                print("Please ensure the Cradle UI server is running.")
                
        except ImportError:
            with self.status_output:
                self.status_output.clear_output()
                print("⚠️ tkinter not available for file dialog.")
                print("Please specify a file path manually:")
                print()
                print("```python")
                print("import requests")
                print("import json")
                print("from cursus.steps.configs.config_cradle_data_loading_step import CradleDataLoadConfig")
                print()
                print(f"# Get config from server")
                print(f"response = requests.get('{self.server_url}/api/cradle-ui/get-latest-config')")
                print("config_data = response.json()")
                print()
                print("# Save to file")
                print("with open('cradle_config.json', 'w') as f:")
                print("    json.dump(config_data, f, indent=2, default=str)")
                print()
                print("# Load as config object")
                print("config = CradleDataLoadConfig(**config_data)")
                print("config_list.append(config)")
                print("```")
                
        except Exception as e:
            with self.status_output:
                self.status_output.clear_output()
                print(f"❌ Unexpected error: {str(e)}")
    
    def _start_config_checker(self):
        """Start a background thread to check for configuration availability."""
        self._checker_running = True
        self._checker_thread = threading.Thread(target=self._config_checker_loop, daemon=True)
        self._checker_thread.start()
    
    def _config_checker_loop(self):
        """Background loop to check for configuration availability."""
        while self._checker_running:
            try:
                # Check if configuration is available on server
                response = requests.get(f"{self.server_url}/api/cradle-ui/get-latest-config", timeout=2)
                
                if response.status_code == 200:
                    # Configuration is available - enable the button
                    if self.get_config_btn.disabled:
                        self.get_config_btn.disabled = False
                        self.get_config_btn.button_style = 'success'
                        self.get_config_btn.tooltip = "Configuration ready! Click to save to JSON file"
                        
                        # Show notification in status output
                        with self.status_output:
                            self.status_output.clear_output()
                            print("✅ Configuration is ready! You can now click 'Get Configuration' to save it.")
                            
                elif response.status_code == 404:
                    # No configuration available - keep button disabled
                    if not self.get_config_btn.disabled:
                        self.get_config_btn.disabled = True
                        self.get_config_btn.button_style = ''
                        self.get_config_btn.tooltip = "Complete the configuration and click 'Finish' first"
                        
            except requests.exceptions.RequestException:
                # Server not available or other network error - keep button disabled
                if not self.get_config_btn.disabled:
                    self.get_config_btn.disabled = True
                    self.get_config_btn.button_style = ''
                    self.get_config_btn.tooltip = "Server not available or configuration not ready"
            
            # Wait before next check
            time.sleep(3)  # Check every 3 seconds
    
    def _stop_config_checker(self):
        """Stop the configuration checker thread."""
        self._checker_running = False
        if hasattr(self, '_checker_thread'):
            self._checker_thread.join(timeout=1)
    
    def _on_clear_clicked(self, button):
        """Handle clear button click."""
        with self.status_output:
            self.status_output.clear_output()
        with self.result_output:
            self.result_output.clear_output()
        self.config_result = None
        
        # Reset button state
        self.get_config_btn.disabled = True
        self.get_config_btn.button_style = ''
        self.get_config_btn.tooltip = "Complete the configuration and click 'Finish' first"
        
        print("🧹 Results cleared.")
    
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
        <div style="background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <h4>📝 How to Use:</h4>
            <ol>
                <li>Complete the 4-step configuration in the UI above</li>
                <li>Click "Finish" in the UI to generate the configuration</li>
                <li>Click "Get Configuration" button below to see the results</li>
                <li>Use <code>cradle_widget.get_config()</code> to get the config object</li>
                <li>Add to your config list: <code>config_list.append(cradle_widget.get_config())</code></li>
            </ol>
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
        
        # After completing the UI configuration:
        training_cradle_data_load_config = cradle_widget.get_config()
        config_list.append(training_cradle_data_load_config)
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
    """
    Enhanced Cradle Config Widget that can start/stop its own server.
    """
    
    def __init__(self, *args, **kwargs):
        self.server_process = None
        super().__init__(*args, **kwargs)
    
    def start_server(self):
        """Start the UI server if not already running."""
        try:
            # Check if server is already running
            response = requests.get(f"{self.server_url}/health", timeout=2)
            if response.status_code == 200:
                with self.status_output:
                    print("✅ Server is already running")
                return True
        except requests.exceptions.RequestException:
            pass
        
        # Start the server
        try:
            import subprocess
            import sys
            from pathlib import Path
            
            # Get the path to the app module
            app_path = Path(__file__).parent / "app.py"
            
            # Start uvicorn server
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
            
            # Wait a moment for server to start
            time.sleep(3)
            
            # Check if server started successfully
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
        # Try to start server
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
