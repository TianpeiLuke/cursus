"""
Multi-Step Wizard and Universal Configuration Widgets

Provides Jupyter widget implementations for universal configuration management
including multi-step pipeline configuration wizards.
"""

import logging
from typing import Any, Dict, List, Optional, Union
import ipywidgets as widgets
from IPython.display import display, clear_output
import json

from ...core.base.config_base import BasePipelineConfig
from ...steps.configs.config_processing_step_base import ProcessingStepConfigBase

logger = logging.getLogger(__name__)


class UniversalConfigWidget:
    """Universal configuration widget for any config type."""
    
    def __init__(self, form_data: Dict[str, Any]):
        """
        Initialize universal configuration widget.
        
        Args:
            form_data: Form data containing config class, fields, values, etc.
        """
        self.form_data = form_data
        self.config_class = form_data["config_class"]
        self.config_class_name = form_data["config_class_name"]
        self.fields = form_data["fields"]
        self.values = form_data["values"]
        self.pre_populated_instance = form_data.get("pre_populated_instance")
        
        self.widgets = {}
        self.config_instance = None
        self.output = widgets.Output()
        
        logger.info(f"UniversalConfigWidget initialized for {self.config_class_name}")
    
    def display(self):
        """Display the configuration form."""
        with self.output:
            clear_output(wait=True)
            
            # Create title
            title = widgets.HTML(f"<h3>Configure {self.config_class_name}</h3>")
            display(title)
            
            # Create form widgets
            form_widgets = []
            
            for field in self.fields:
                field_name = field["name"]
                field_type = field["type"]
                required = field["required"]
                description = field["description"]
                
                # Get current value
                current_value = self.values.get(field_name, "")
                
                # Create appropriate widget based on field type
                if field_type == "text":
                    widget = widgets.Text(
                        value=str(current_value) if current_value is not None else "",
                        description=f"{field_name}{'*' if required else ''}:",
                        style={'description_width': 'initial'},
                        layout=widgets.Layout(width='500px')
                    )
                elif field_type == "number":
                    widget = widgets.FloatText(
                        value=float(current_value) if current_value and str(current_value).replace('.', '').replace('-', '').isdigit() else 0.0,
                        description=f"{field_name}{'*' if required else ''}:",
                        style={'description_width': 'initial'},
                        layout=widgets.Layout(width='300px')
                    )
                elif field_type == "checkbox":
                    widget = widgets.Checkbox(
                        value=bool(current_value) if current_value is not None else False,
                        description=f"{field_name}{'*' if required else ''}:",
                        style={'description_width': 'initial'}
                    )
                elif field_type == "list":
                    widget = widgets.Textarea(
                        value=json.dumps(current_value) if isinstance(current_value, list) else str(current_value) if current_value else "[]",
                        description=f"{field_name}{'*' if required else ''}:",
                        placeholder="Enter JSON list, e.g., [\"item1\", \"item2\"]",
                        style={'description_width': 'initial'},
                        layout=widgets.Layout(width='500px', height='80px')
                    )
                elif field_type == "keyvalue":
                    widget = widgets.Textarea(
                        value=json.dumps(current_value, indent=2) if isinstance(current_value, dict) else str(current_value) if current_value else "{}",
                        description=f"{field_name}{'*' if required else ''}:",
                        placeholder="Enter JSON object, e.g., {\"key\": \"value\"}",
                        style={'description_width': 'initial'},
                        layout=widgets.Layout(width='500px', height='100px')
                    )
                else:
                    # Default to text
                    widget = widgets.Text(
                        value=str(current_value) if current_value is not None else "",
                        description=f"{field_name}{'*' if required else ''}:",
                        style={'description_width': 'initial'},
                        layout=widgets.Layout(width='500px')
                    )
                
                self.widgets[field_name] = widget
                
                # Add description if available
                if description:
                    desc_widget = widgets.HTML(f"<small><i>{description}</i></small>")
                    form_widgets.extend([widget, desc_widget])
                else:
                    form_widgets.append(widget)
            
            # Create buttons
            save_button = widgets.Button(
                description="Save Configuration",
                button_style='success',
                layout=widgets.Layout(width='200px')
            )
            cancel_button = widgets.Button(
                description="Cancel",
                button_style='',
                layout=widgets.Layout(width='100px')
            )
            
            save_button.on_click(self._on_save_clicked)
            cancel_button.on_click(self._on_cancel_clicked)
            
            button_box = widgets.HBox([save_button, cancel_button])
            
            # Display all widgets
            all_widgets = form_widgets + [button_box]
            form_box = widgets.VBox(all_widgets)
            display(form_box)
        
        display(self.output)
    
    def _on_save_clicked(self, button):
        """Handle save button click."""
        try:
            # Collect form data
            form_data = {}
            for field_name, widget in self.widgets.items():
                value = widget.value
                
                # Convert values based on field type
                field_info = next((f for f in self.fields if f["name"] == field_name), None)
                if field_info:
                    field_type = field_info["type"]
                    
                    if field_type == "list":
                        try:
                            value = json.loads(value) if isinstance(value, str) else value
                        except json.JSONDecodeError:
                            value = []
                    elif field_type == "keyvalue":
                        try:
                            value = json.loads(value) if isinstance(value, str) else value
                        except json.JSONDecodeError:
                            value = {}
                    elif field_type == "number":
                        value = float(value) if value != "" else 0.0
                
                form_data[field_name] = value
            
            # Create configuration instance
            self.config_instance = self.config_class(**form_data)
            
            with self.output:
                clear_output(wait=True)
                success_msg = widgets.HTML(
                    f"<div style='color: green; font-weight: bold;'>✓ Configuration saved successfully!</div>"
                    f"<p>Configuration type: {self.config_class_name}</p>"
                )
                display(success_msg)
            
            logger.info(f"Configuration saved successfully: {self.config_class_name}")
            
        except Exception as e:
            with self.output:
                clear_output(wait=True)
                error_msg = widgets.HTML(
                    f"<div style='color: red; font-weight: bold;'>✗ Error saving configuration:</div>"
                    f"<p>{str(e)}</p>"
                )
                display(error_msg)
            
            logger.error(f"Error saving configuration: {e}")
    
    def _on_cancel_clicked(self, button):
        """Handle cancel button click."""
        with self.output:
            clear_output(wait=True)
            cancel_msg = widgets.HTML("<div style='color: orange;'>Configuration cancelled.</div>")
            display(cancel_msg)
    
    def get_config(self) -> Optional[BasePipelineConfig]:
        """
        Get the saved configuration instance.
        
        Returns:
            Configuration instance if saved, None otherwise
        """
        return self.config_instance


class MultiStepWizard:
    """Multi-step pipeline configuration wizard."""
    
    def __init__(self, 
                 steps: List[Dict[str, Any]], 
                 base_config: Optional[BasePipelineConfig] = None,
                 processing_config: Optional[ProcessingStepConfigBase] = None):
        """
        Initialize multi-step wizard.
        
        Args:
            steps: List of step definitions
            base_config: Base pipeline configuration
            processing_config: Processing configuration
        """
        self.steps = steps
        self.base_config = base_config
        self.processing_config = processing_config
        self.completed_configs = {}  # Store completed configurations
        self.current_step = 0
        self.step_widgets = {}
        
        self.output = widgets.Output()
        self.navigation_output = widgets.Output()
        
        logger.info(f"MultiStepWizard initialized with {len(steps)} steps")
    
    def display(self):
        """Display the multi-step wizard interface."""
        with self.navigation_output:
            clear_output(wait=True)
            self._display_navigation()
        
        with self.output:
            clear_output(wait=True)
            self._display_current_step()
        
        # Display navigation and main content
        display(self.navigation_output)
        display(self.output)
    
    def _display_navigation(self):
        """Display navigation controls."""
        # Progress indicator
        progress_html = "<div style='margin-bottom: 20px;'>"
        progress_html += f"<h3>Pipeline Configuration Wizard</h3>"
        progress_html += f"<p>Step {self.current_step + 1} of {len(self.steps)}</p>"
        progress_html += "<div style='background-color: #f0f0f0; height: 20px; border-radius: 10px;'>"
        progress_percent = ((self.current_step + 1) / len(self.steps)) * 100
        progress_html += f"<div style='background-color: #4CAF50; height: 100%; width: {progress_percent}%; border-radius: 10px;'></div>"
        progress_html += "</div></div>"
        
        progress_widget = widgets.HTML(progress_html)
        display(progress_widget)
        
        # Navigation buttons
        prev_button = widgets.Button(
            description="← Previous",
            disabled=(self.current_step == 0),
            layout=widgets.Layout(width='100px')
        )
        next_button = widgets.Button(
            description="Next →",
            button_style='primary',
            disabled=(self.current_step == len(self.steps) - 1),
            layout=widgets.Layout(width='100px')
        )
        finish_button = widgets.Button(
            description="Finish",
            button_style='success',
            disabled=(self.current_step != len(self.steps) - 1),
            layout=widgets.Layout(width='100px')
        )
        
        prev_button.on_click(self._on_prev_clicked)
        next_button.on_click(self._on_next_clicked)
        finish_button.on_click(self._on_finish_clicked)
        
        nav_box = widgets.HBox([prev_button, next_button, finish_button])
        display(nav_box)
    
    def _display_current_step(self):
        """Display the current step."""
        if self.current_step >= len(self.steps):
            return
        
        step = self.steps[self.current_step]
        step_title = step["title"]
        config_class = step["config_class"]
        config_class_name = step["config_class_name"]
        
        # Create step widget if not exists
        if self.current_step not in self.step_widgets:
            # Prepare form data
            form_data = {
                "config_class": config_class,
                "config_class_name": config_class_name,
                "fields": self._get_step_fields(step),
                "values": self._get_step_values(step),
                "pre_populated_instance": step.get("pre_populated")
            }
            
            self.step_widgets[self.current_step] = UniversalConfigWidget(form_data)
        
        # Display step
        step_widget = self.step_widgets[self.current_step]
        step_widget.display()
    
    def _get_step_fields(self, step: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get form fields for a step."""
        config_class = step["config_class"]
        config_class_name = step["config_class_name"]
        
        # Check if there's a specialized component for this config type
        from .specialized_widgets import SpecializedComponentRegistry
        registry = SpecializedComponentRegistry()
        
        if registry.has_specialized_component(config_class_name):
            # For specialized components, return minimal fields since they handle their own UI
            return [{"name": "specialized_component", "type": "specialized", "required": False, "description": f"Uses specialized {config_class_name} widget"}]
        
        # Use UniversalConfigCore to get fields for standard components
        from .core import UniversalConfigCore
        core = UniversalConfigCore()
        return core._get_form_fields(config_class)
    
    def _get_step_values(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Get pre-populated values for a step."""
        # Check for pre-populated instance
        if "pre_populated" in step and step["pre_populated"]:
            instance = step["pre_populated"]
            if hasattr(instance, 'model_dump'):
                return instance.model_dump()
            else:
                return {}
        
        # Check for pre-populated data
        if "pre_populated_data" in step and step["pre_populated_data"]:
            return step["pre_populated_data"]
        
        # Try to create from base config
        if "base_config" in step and step["base_config"]:
            config_class = step["config_class"]
            base_config = step["base_config"]
            
            if hasattr(config_class, 'from_base_config'):
                try:
                    instance = config_class.from_base_config(base_config)
                    if hasattr(instance, 'model_dump'):
                        return instance.model_dump()
                except Exception as e:
                    logger.warning(f"Failed to create from base config: {e}")
        
        return {}
    
    def _on_prev_clicked(self, button):
        """Handle previous button click."""
        if self.current_step > 0:
            self.current_step -= 1
            self.display()
    
    def _on_next_clicked(self, button):
        """Handle next button click."""
        # Save current step
        if self._save_current_step():
            if self.current_step < len(self.steps) - 1:
                self.current_step += 1
                self.display()
    
    def _on_finish_clicked(self, button):
        """Handle finish button click."""
        # Save current step and finish
        if self._save_current_step():
            with self.output:
                clear_output(wait=True)
                
                # Show completion message
                completion_html = """
                <div style='text-align: center; padding: 20px;'>
                    <h2 style='color: green;'>✓ Pipeline Configuration Complete!</h2>
                    <p>All configuration steps have been completed successfully.</p>
                    <p>Use <code>get_completed_configs()</code> to retrieve the configuration list.</p>
                </div>
                """
                completion_widget = widgets.HTML(completion_html)
                display(completion_widget)
            
            logger.info("Pipeline configuration wizard completed successfully")
    
    def _save_current_step(self) -> bool:
        """Save the current step configuration."""
        if self.current_step not in self.step_widgets:
            return True
        
        step_widget = self.step_widgets[self.current_step]
        step = self.steps[self.current_step]
        
        # Trigger save on the widget
        try:
            # Get form data from widget
            form_data = {}
            for field_name, widget in step_widget.widgets.items():
                value = widget.value
                
                # Convert values based on field type
                field_info = next((f for f in step_widget.fields if f["name"] == field_name), None)
                if field_info:
                    field_type = field_info["type"]
                    
                    if field_type == "list":
                        try:
                            value = json.loads(value) if isinstance(value, str) else value
                        except json.JSONDecodeError:
                            value = []
                    elif field_type == "keyvalue":
                        try:
                            value = json.loads(value) if isinstance(value, str) else value
                        except json.JSONDecodeError:
                            value = {}
                    elif field_type == "number":
                        value = float(value) if value != "" else 0.0
                
                form_data[field_name] = value
            
            # Create configuration instance
            config_class = step["config_class"]
            config_instance = config_class(**form_data)
            
            # Store completed configuration
            step_key = step["title"]
            self.completed_configs[step_key] = config_instance
            
            logger.info(f"Step '{step_key}' saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving step: {e}")
            return False
    
    def get_completed_configs(self) -> List[BasePipelineConfig]:
        """
        Return list of completed configurations after user finishes all steps.
        
        Returns:
            List of configuration instances in the same order as demo_config.ipynb
        """
        if not self._all_steps_completed():
            raise ValueError("Not all required configurations have been completed")
        
        # Return configurations in the correct order for merge_and_save_configs
        config_list = []
        
        # Add base configurations first (matching demo_config.ipynb order)
        if 'Base Pipeline Configuration' in self.completed_configs:
            config_list.append(self.completed_configs['Base Pipeline Configuration'])
        
        if 'Processing Configuration' in self.completed_configs:
            config_list.append(self.completed_configs['Processing Configuration'])
        
        # Add step-specific configurations in dependency order
        for step_name in self.get_dependency_ordered_steps():
            if step_name in self.completed_configs:
                config_list.append(self.completed_configs[step_name])
        
        logger.info(f"Returning {len(config_list)} completed configurations")
        return config_list
    
    def _all_steps_completed(self) -> bool:
        """Check if all required steps have been completed."""
        required_steps = [step['title'] for step in self.steps if step.get('required', True)]
        completed_steps = list(self.completed_configs.keys())
        return all(step in completed_steps for step in required_steps)
    
    def get_dependency_ordered_steps(self) -> List[str]:
        """Return step names in dependency order for proper config_list ordering."""
        # Use step order from wizard (already in dependency order)
        ordered_steps = []
        for step in self.steps:
            step_title = step["title"]
            if step_title not in ['Base Pipeline Configuration', 'Processing Configuration']:
                ordered_steps.append(step_title)
        
        return ordered_steps
