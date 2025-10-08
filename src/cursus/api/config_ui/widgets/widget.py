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

# Suppress logger messages in widget output
logging.getLogger('cursus.api.config_ui').setLevel(logging.ERROR)
logging.getLogger('cursus.core').setLevel(logging.ERROR)
logging.getLogger('cursus.step_catalog').setLevel(logging.ERROR)
logging.getLogger('cursus.step_catalog.step_catalog').setLevel(logging.ERROR)
logging.getLogger('cursus.step_catalog.builder_discovery').setLevel(logging.ERROR)
logging.getLogger('cursus.step_catalog.config_discovery').setLevel(logging.ERROR)
# Suppress all cursus-related loggers
logging.getLogger('cursus').setLevel(logging.ERROR)

# Handle both relative and absolute imports using centralized path setup
try:
    # Try relative imports first (when run as module)
    from ....core.base.config_base import BasePipelineConfig
    from ....steps.configs.config_processing_step_base import ProcessingStepConfigBase
except ImportError:
    # Fallback: Set up cursus path and use absolute imports
    import sys
    from pathlib import Path
    
    # Add the core directory to path for import_utils
    current_dir = Path(__file__).parent
    core_dir = current_dir.parent / 'core'
    if str(core_dir) not in sys.path:
        sys.path.insert(0, str(core_dir))
    
    from core.import_utils import ensure_cursus_path
    ensure_cursus_path()
    
    from cursus.core.base.config_base import BasePipelineConfig
    from cursus.steps.configs.config_processing_step_base import ProcessingStepConfigBase

logger = logging.getLogger(__name__)


class UniversalConfigWidget:
    """Universal configuration widget for any config type."""
    
    def __init__(self, form_data: Dict[str, Any], is_final_step: bool = True):
        """
        Initialize universal configuration widget.
        
        Args:
            form_data: Form data containing config class, fields, values, etc.
            is_final_step: Whether this is the final step in a multi-step wizard
        """
        self.form_data = form_data
        self.config_class = form_data["config_class"]
        self.config_class_name = form_data["config_class_name"]
        self.fields = form_data["fields"]
        self.values = form_data["values"]
        self.pre_populated_instance = form_data.get("pre_populated_instance")
        self.is_final_step = is_final_step
        
        self.widgets = {}
        self.config_instance = None
        self.output = widgets.Output()
        
        logger.info(f"UniversalConfigWidget initialized for {self.config_class_name}")
    
    def display(self):
        """Display the configuration form with 4-tier field categorization including inheritance."""
        with self.output:
            clear_output(wait=True)
            
            # Create modern title with emoji
            title_html = f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
                <h2 style='margin: 0; display: flex; align-items: center;'>
                    ⚙️ {self.config_class_name}
                    <span style='margin-left: auto; font-size: 14px; opacity: 0.8;'>Configuration</span>
                </h2>
            </div>
            """
            title = widgets.HTML(title_html)
            display(title)
            
            # Enhanced 4-tier field categorization with inheritance
            inherited_fields = [f for f in self.fields if f.get('tier') == 'inherited']
            essential_fields = [f for f in self.fields if f.get('tier') == 'essential' or (f.get('required', False) and f.get('tier') != 'inherited')]
            system_fields = [f for f in self.fields if f.get('tier') == 'system' or (not f.get('required', False) and f.get('tier') not in ['essential', 'inherited'])]
            
            form_sections = []
            
            # NEW: Inherited Fields Section (Tier 3) - Smart Default Value Inheritance ⭐
            if inherited_fields:
                inherited_section = self._create_field_section(
                    "💾 Inherited Fields (Tier 3) - Smart Defaults",
                    inherited_fields,
                    "linear-gradient(135deg, #f0f8ff 0%, #e0f2fe 100%)",
                    "#007bff",
                    "Auto-filled from parent configurations - can be overridden if needed"
                )
                form_sections.append(inherited_section)
            
            # Essential Fields Section (Tier 1)
            if essential_fields:
                essential_section = self._create_field_section(
                    "🔥 Essential User Inputs (Tier 1)",
                    essential_fields,
                    "linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)",
                    "#f59e0b",
                    "Required fields that must be filled by user"
                )
                form_sections.append(essential_section)
            
            # System Fields Section (Tier 2)
            if system_fields:
                system_section = self._create_field_section(
                    "⚙️ System Inputs (Tier 2)",
                    system_fields,
                    "linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%)",
                    "#3b82f6",
                    "Optional fields with defaults, user-modifiable"
                )
                form_sections.append(system_section)
            
            # Legacy Inherited Configuration Display (for backward compatibility)
            legacy_inherited_section = self._create_inherited_section()
            if legacy_inherited_section:
                form_sections.append(legacy_inherited_section)
            
            # Create action buttons with modern styling
            button_section = self._create_action_buttons()
            form_sections.append(button_section)
            
            # Display all sections
            form_box = widgets.VBox(form_sections, layout=widgets.Layout(padding='10px'))
            display(form_box)
        
        display(self.output)
    
    def _create_field_section(self, title: str, fields: List[Dict], bg_gradient: str, border_color: str, description: str) -> widgets.Widget:
        """Create a modern field section with tier-specific styling."""
        # Section header
        header_html = f"""
        <div style='background: {bg_gradient}; 
                    border-left: 4px solid {border_color}; 
                    padding: 12px; border-radius: 8px 8px 0 0; margin-bottom: 0;'>
            <h4 style='margin: 0; color: #1f2937; display: flex; align-items: center;'>
                {title}
            </h4>
            <p style='margin: 5px 0 0 0; font-size: 12px; color: #6b7280; font-style: italic;'>
                {description}
            </p>
        </div>
        """
        header = widgets.HTML(header_html)
        
        # Create field widgets in a grid-like layout
        field_rows = []
        
        for i, field in enumerate(fields):
            field_widget_data = self._create_enhanced_field_widget(field)
            
            # Add to widgets dict for later access
            self.widgets[field["name"]] = field_widget_data["widget"]
            
            # Add the container (which includes widget + description if present)
            field_rows.append(field_widget_data["container"])
        
        # Create field container with modern styling
        if field_rows:
            field_container = widgets.VBox(
                field_rows, 
                layout=widgets.Layout(
                    padding='20px',
                    background='white',
                    border='1px solid #e5e7eb',
                    border_top='none',
                    border_radius='0 0 8px 8px'
                )
            )
            
            # Combine header and fields
            section = widgets.VBox([header, field_container], layout=widgets.Layout(margin='0 0 20px 0'))
        else:
            # Just header if no fields
            section = widgets.VBox([header], layout=widgets.Layout(margin='0 0 20px 0'))
        
        return section
    

    def _create_enhanced_field_widget(self, field: Dict) -> Dict:
        """Create an enhanced field widget with modern styling and emoji icons."""
        field_name = field["name"]
        field_type = field["type"]
        required = field.get("required", False)
        tier = field.get("tier", "system")
        description = field.get("description", "")
        
        # Get current value
        current_value = self.values.get(field_name, field.get("default", ""))
        
        # Get emoji icon for field
        emoji_icon = self._get_field_emoji(field_name)
        
        # Create field label with emoji and styling
        label_style = "font-weight: 600; color: #374151;" if required else "color: #6b7280;"
        required_indicator = " *" if required else ""
        
        # Create appropriate widget based on field type with SIMPLIFIED approach
        if field_type == "text":
            widget = widgets.Text(
                value=str(current_value) if current_value is not None else "",
                description=f"{emoji_icon} {field_name}{required_indicator}:",
                style={'description_width': '200px'},
                layout=widgets.Layout(width='500px', margin='5px 0')
            )
        elif field_type == "number":
            widget = widgets.FloatText(
                value=float(current_value) if current_value and str(current_value).replace('.', '').replace('-', '').isdigit() else (field.get("default", 0.0) or 0.0),
                description=f"{emoji_icon} {field_name}{required_indicator}:",
                style={'description_width': '200px'},
                layout=widgets.Layout(width='300px', margin='5px 0')
            )
        elif field_type == "checkbox":
            widget = widgets.Checkbox(
                value=bool(current_value) if current_value is not None else bool(field.get("default", False)),
                description=f"{emoji_icon} {field_name}{required_indicator}:",
                style={'description_width': '200px'},
                layout=widgets.Layout(margin='5px 0')
            )
        elif field_type == "list":
            widget = widgets.Textarea(
                value=json.dumps(current_value) if isinstance(current_value, list) else str(current_value) if current_value else "[]",
                description=f"{emoji_icon} {field_name}{required_indicator}:",
                placeholder="Enter JSON list, e.g., [\"item1\", \"item2\"]",
                style={'description_width': '200px'},
                layout=widgets.Layout(width='500px', height='80px', margin='5px 0')
            )
        elif field_type == "keyvalue":
            widget = widgets.Textarea(
                value=json.dumps(current_value, indent=2) if isinstance(current_value, dict) else str(current_value) if current_value else "{}",
                description=f"{emoji_icon} {field_name}{required_indicator}:",
                placeholder="Enter JSON object, e.g., {\"key\": \"value\"}",
                style={'description_width': '200px'},
                layout=widgets.Layout(width='500px', height='100px', margin='5px 0')
            )
        elif field_type == "specialized":
            # Create specialized configuration interface
            return self._create_specialized_field_widget(field)
        else:
            # Default to text
            widget = widgets.Text(
                value=str(current_value) if current_value is not None else "",
                description=f"{emoji_icon} {field_name}{required_indicator}:",
                style={'description_width': '200px'},
                layout=widgets.Layout(width='500px', margin='5px 0')
            )
        
        # Add description if available
        if description:
            desc_html = f"<div style='margin-left: 210px; margin-top: -5px; margin-bottom: 10px; font-size: 11px; color: #6b7280; font-style: italic;'>{description}</div>"
            desc_widget = widgets.HTML(desc_html)
            container = widgets.VBox([widget, desc_widget])
            return {"widget": widget, "description": desc_widget, "container": container}
        else:
            return {"widget": widget, "container": widget}
    
    def _create_specialized_field_widget(self, field: Dict) -> Dict:
        """Create a specialized configuration interface widget."""
        config_class_name = field.get("config_class_name", "Unknown")
        icon = field.get("icon", "🎛️")
        complexity = field.get("complexity", "advanced")
        description = field.get("description", "Specialized configuration interface")
        features = field.get("features", [])
        
        # Create complexity badge
        complexity_colors = {
            "basic": "#10b981",
            "intermediate": "#f59e0b", 
            "advanced": "#ef4444"
        }
        complexity_color = complexity_colors.get(complexity, "#6b7280")
        
        # Create features list
        features_html = ""
        if features:
            features_html = "<br>".join([f"    {feature}" for feature in features])
        
        # Create specialized interface display
        specialized_html = f"""
        <div style='background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                    border: 2px solid #0ea5e9; border-radius: 12px; padding: 20px; margin: 15px 0;
                    box-shadow: 0 4px 12px rgba(14, 165, 233, 0.15);'>
            <div style='display: flex; align-items: center; margin-bottom: 15px;'>
                <div style='font-size: 32px; margin-right: 15px;'>{icon}</div>
                <div style='flex: 1;'>
                    <h3 style='margin: 0; color: #0c4a6e; font-size: 20px;'>Specialized Configuration</h3>
                    <div style='display: flex; align-items: center; margin-top: 5px;'>
                        <span style='background: {complexity_color}; color: white; padding: 2px 8px; 
                                     border-radius: 12px; font-size: 11px; font-weight: bold; text-transform: uppercase;'>
                            {complexity}
                        </span>
                        <span style='margin-left: 10px; color: #0c4a6e; font-weight: 600;'>{config_class_name}</span>
                    </div>
                </div>
            </div>
            
            <p style='margin: 0 0 15px 0; color: #0c4a6e; font-size: 14px; line-height: 1.5;'>
                {description}
            </p>
            
            <div style='background: rgba(255, 255, 255, 0.7); border-radius: 8px; padding: 15px; margin-bottom: 15px;'>
                <h4 style='margin: 0 0 10px 0; color: #0c4a6e; font-size: 14px;'>✨ Features:</h4>
                <div style='color: #0c4a6e; font-size: 13px; line-height: 1.6;'>
                    {features_html}
                </div>
            </div>
            
            <div style='text-align: center;'>
                <button style='background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%); 
                               color: white; border: none; padding: 12px 24px; border-radius: 8px; 
                               font-weight: 600; cursor: pointer; font-size: 14px;
                               box-shadow: 0 2px 8px rgba(14, 165, 233, 0.3);
                               transition: all 0.3s ease;'
                        onmouseover='this.style.transform="translateY(-2px)"; this.style.boxShadow="0 4px 12px rgba(14, 165, 233, 0.4)";'
                        onmouseout='this.style.transform="translateY(0px)"; this.style.boxShadow="0 2px 8px rgba(14, 165, 233, 0.3)";'>
                    {icon} Open {config_class_name} Wizard
                </button>
                <p style='margin: 10px 0 0 0; font-size: 11px; color: #6b7280; font-style: italic;'>
                    Base configuration will be pre-filled automatically
                </p>
            </div>
        </div>
        """
        
        # Create a dummy widget for form compatibility
        dummy_widget = widgets.HTML(value="specialized_widget_placeholder")
        
        specialized_display = widgets.HTML(specialized_html)
        
        return {
            "widget": dummy_widget,
            "container": specialized_display
        }
    
    def _get_field_emoji(self, field_name: str) -> str:
        """Get appropriate emoji icon for field name."""
        emoji_map = {
            "author": "👤", "bucket": "🪣", "role": "🔐", "region": "🌍",
            "service_name": "🎯", "pipeline_version": "📅", "project_root_folder": "📁",
            "model_class": "🤖", "instance_type": "🖥️", "volume_size": "💾",
            "processing_source_dir": "📂", "entry_point": "🎯", "job_type": "🏷️",
            "label_name": "🎯", "output_schema": "📊", "output_format": "📄",
            "cluster_type": "⚙️", "cradle_account": "🔐", "transform_sql": "🔄",
            "num_round": "🔢", "max_depth": "📏", "learning_rate": "📈",
            "lr": "📈", "batch_size": "📦", "max_epochs": "🔄", "device": "💻",
            "optimizer": "⚡", "metric_choices": "📊"
        }
        return emoji_map.get(field_name.lower(), "⚙️")
    
    def _create_inherited_section(self) -> Optional[widgets.Widget]:
        """Create inherited configuration display section."""
        if not hasattr(self, 'pre_populated_instance') or not self.pre_populated_instance:
            return None
        
        # Extract inherited values
        inherited_values = {}
        if hasattr(self.pre_populated_instance, 'model_dump'):
            inherited_values = self.pre_populated_instance.model_dump()
        elif hasattr(self.pre_populated_instance, '__dict__'):
            inherited_values = self.pre_populated_instance.__dict__
        
        if not inherited_values:
            return None
        
        # Create inherited fields display
        inherited_items = []
        for key, value in inherited_values.items():
            if not key.startswith('_') and value is not None:
                emoji = self._get_field_emoji(key)
                inherited_items.append(f"• {emoji} {key}: {value}")
        
        if not inherited_items:
            return None
        
        inherited_html = f"""
        <div style='background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); 
                    border-left: 4px solid #8b5cf6; 
                    padding: 15px; border-radius: 8px; margin: 20px 0;'>
            <h4 style='margin: 0 0 10px 0; color: #1f2937; display: flex; align-items: center;'>
                💾 Inherited Configuration
            </h4>
            <p style='margin: 0 0 10px 0; font-size: 12px; color: #6b7280; font-style: italic;'>
                Auto-filled from parent configuration
            </p>
            <div style='font-size: 13px; color: #4c1d95; line-height: 1.6;'>
                {' <br>'.join(inherited_items[:6])}
                {' <br><em>... and more</em>' if len(inherited_items) > 6 else ''}
            </div>
        </div>
        """
        
        return widgets.HTML(inherited_html)
    
    def _create_action_buttons(self) -> widgets.Widget:
        """Create modern action buttons - conditionally show save button only on final step."""
        if self.is_final_step:
            # Final step: Show save button
            save_button = widgets.Button(
                description="💾 Complete Configuration",
                button_style='success',
                layout=widgets.Layout(width='220px', height='40px')
            )
            cancel_button = widgets.Button(
                description="❌ Cancel",
                button_style='',
                layout=widgets.Layout(width='120px', height='40px')
            )
            
            save_button.on_click(self._on_save_clicked)
            cancel_button.on_click(self._on_cancel_clicked)
            
            button_box = widgets.HBox(
                [save_button, cancel_button], 
                layout=widgets.Layout(justify_content='center', margin='20px 0')
            )
            
            return button_box
        else:
            # Intermediate step: Show guidance instead of save button
            guidance_html = f"""
            <div style='background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                        border: 2px solid #0ea5e9; border-radius: 12px; padding: 20px; margin: 20px 0;
                        text-align: center;'>
                <h4 style='margin: 0 0 10px 0; color: #0c4a6e; display: flex; align-items: center; justify-content: center;'>
                    📋 Step {self.config_class_name}
                </h4>
                <p style='margin: 0 0 15px 0; color: #0c4a6e; font-size: 14px;'>
                    Fill in the fields above and use the <strong>"Next →"</strong> button to continue to the next step.
                </p>
                <div style='background: rgba(255, 255, 255, 0.7); border-radius: 8px; padding: 12px; margin: 10px 0;'>
                    <p style='margin: 0; color: #0369a1; font-size: 13px; font-style: italic;'>
                        💡 Your configuration will be automatically saved when you click "Next"
                    </p>
                </div>
                <div style='color: #0284c7; font-size: 12px; margin-top: 10px;'>
                    ⬆️ Use the navigation buttons above to move between steps
                </div>
            </div>
            """
            
            return widgets.HTML(guidance_html)
    
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
    """Multi-step pipeline configuration wizard with Smart Default Value Inheritance support."""
    
    def __init__(self, 
                 steps: List[Dict[str, Any]], 
                 base_config: Optional[BasePipelineConfig] = None,
                 processing_config: Optional[ProcessingStepConfigBase] = None,
                 enable_inheritance: bool = True):
        """
        Initialize multi-step wizard with Smart Default Value Inheritance support.
        
        Args:
            steps: List of step definitions
            base_config: Base pipeline configuration
            processing_config: Processing configuration
            enable_inheritance: Enable smart inheritance features (NEW)
        """
        self.steps = steps
        self.base_config = base_config
        self.processing_config = processing_config
        self.enable_inheritance = enable_inheritance  # NEW: Inheritance support
        self.completed_configs = {}  # Store completed configurations
        self.current_step = 0
        self.step_widgets = {}
        
        self.output = widgets.Output()
        self.navigation_output = widgets.Output()
        
        # NEW: Initialize completed configs for inheritance
        if self.enable_inheritance:
            if base_config:
                self.completed_configs["BasePipelineConfig"] = base_config
            if processing_config:
                self.completed_configs["ProcessingStepConfigBase"] = processing_config
        
        logger.info(f"MultiStepWizard initialized with {len(steps)} steps, inheritance={'enabled' if enable_inheritance else 'disabled'}")
    
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
        """Display enhanced navigation controls with detailed step visualization."""
        # Get current step info
        current_step_info = self.steps[self.current_step] if self.current_step < len(self.steps) else {"title": "Complete"}
        
        # Enhanced progress indicator with step details
        progress_percent = ((self.current_step + 1) / len(self.steps)) * 100
        
        # Create detailed step indicators with titles
        step_indicators = []
        step_details = []
        for i, step in enumerate(self.steps):
            if i < self.current_step:
                # Completed step
                step_indicators.append("●")
                step_details.append(f"✅ {step['title']}")
            elif i == self.current_step:
                # Current step
                step_indicators.append("●")
                step_details.append(f"🔄 {step['title']} (Current)")
            else:
                # Future step
                step_indicators.append("○")
                step_details.append(f"⏳ {step['title']}")
        
        # Create step overview section
        step_overview_html = f"""
        <div style='background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                    border: 1px solid #0ea5e9; border-radius: 8px; padding: 15px; margin-bottom: 15px;'>
            <h4 style='margin: 0 0 10px 0; color: #0c4a6e;'>📋 Configuration Workflow Overview</h4>
            <div style='font-size: 13px; line-height: 1.6;'>
                {' <br>'.join(step_details)}
            </div>
        </div>
        """
        
        progress_html = f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 20px; border-radius: 12px; margin-bottom: 20px;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;'>
                <h2 style='margin: 0; font-size: 24px;'>🎯 Pipeline Configuration Wizard</h2>
                <div style='font-size: 14px; opacity: 0.9;'>
                    Step {self.current_step + 1} of {len(self.steps)}
                </div>
            </div>
            
            <div style='margin-bottom: 15px;'>
                <h3 style='margin: 0; font-size: 18px; opacity: 0.95;'>{current_step_info["title"]}</h3>
                <p style='margin: 5px 0 0 0; font-size: 14px; opacity: 0.8;'>
                    {current_step_info.get("description", "Configure the settings for this step")}
                </p>
            </div>
            
            <div style='margin-bottom: 15px;'>
                <div style='background: rgba(255, 255, 255, 0.2); height: 12px; border-radius: 6px; overflow: hidden;'>
                    <div style='background: linear-gradient(90deg, #10b981 0%, #059669 100%); height: 100%; width: {progress_percent}%; 
                                border-radius: 6px; transition: width 0.5s ease; box-shadow: 0 2px 4px rgba(16, 185, 129, 0.3);'></div>
                </div>
            </div>
            
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div style='font-size: 14px; opacity: 0.8; letter-spacing: 2px;'>
                    Progress: {' '.join(step_indicators)} ({self.current_step + 1}/{len(self.steps)})
                </div>
                <div style='font-size: 12px; opacity: 0.7;'>
                    {progress_percent:.0f}% Complete
                </div>
            </div>
        </div>
        """
        
        # Display step overview and progress
        overview_widget = widgets.HTML(step_overview_html)
        progress_widget = widgets.HTML(progress_html)
        display(overview_widget)
        display(progress_widget)
        
        # Enhanced navigation buttons with step context
        prev_button = widgets.Button(
            description="← Previous",
            disabled=(self.current_step == 0),
            layout=widgets.Layout(width='140px', height='45px'),
            style={'button_color': '#6b7280' if self.current_step == 0 else '#374151'},
            tooltip=f"Go back to: {self.steps[self.current_step - 1]['title'] if self.current_step > 0 else 'N/A'}"
        )
        
        next_button = widgets.Button(
            description="Next →",
            button_style='primary',
            disabled=(self.current_step == len(self.steps) - 1),
            layout=widgets.Layout(width='140px', height='45px'),
            tooltip=f"Continue to: {self.steps[self.current_step + 1]['title'] if self.current_step < len(self.steps) - 1 else 'N/A'}"
        )
        
        finish_button = widgets.Button(
            description="🎉 Complete Workflow",
            button_style='success',
            disabled=(self.current_step != len(self.steps) - 1),
            layout=widgets.Layout(width='180px', height='45px'),
            tooltip="Finish configuration and generate config_list"
        )
        
        prev_button.on_click(self._on_prev_clicked)
        next_button.on_click(self._on_next_clicked)
        finish_button.on_click(self._on_finish_clicked)
        
        # Create enhanced navigation container
        nav_box = widgets.HBox(
            [prev_button, next_button, finish_button], 
            layout=widgets.Layout(
                justify_content='center', 
                margin='15px 0',
                padding='20px',
                border='2px solid #e2e8f0',
                border_radius='12px',
                background='linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)'
            )
        )
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
            
            # Determine if this is the final step
            is_final_step = (self.current_step == len(self.steps) - 1)
            
            self.step_widgets[self.current_step] = UniversalConfigWidget(form_data, is_final_step=is_final_step)
        
        # Display step
        step_widget = self.step_widgets[self.current_step]
        step_widget.display()
    
    def _get_step_fields(self, step: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get form fields for a step with Smart Default Value Inheritance support."""
        config_class = step["config_class"]
        config_class_name = step["config_class_name"]
        
        # Check if there's a specialized component for this config type
        try:
            from .specialized_widgets import SpecializedComponentRegistry
            registry = SpecializedComponentRegistry()
            
            if registry.has_specialized_component(config_class_name):
                # For specialized components, create a visual interface description
                spec_info = registry.SPECIALIZED_COMPONENTS[config_class_name]
                return [{
                    "name": "specialized_component", 
                    "type": "specialized", 
                    "required": False, 
                    "description": spec_info["description"],
                    "features": spec_info["features"],
                    "icon": spec_info["icon"],
                    "complexity": spec_info["complexity"],
                    "config_class_name": config_class_name
                }]
        except ImportError:
            # Specialized widgets not available, continue with standard processing
            pass
        
        # Use UniversalConfigCore with Smart Default Value Inheritance
        from ..core.core import UniversalConfigCore
        core = UniversalConfigCore()
        
        # NEW: Use inheritance-aware field generation if inheritance is enabled
        if self.enable_inheritance and hasattr(step, 'inheritance_analysis'):
            # Use the enhanced inheritance-aware method
            return core.get_inheritance_aware_form_fields(
                config_class_name, 
                step['inheritance_analysis']
            )
        elif self.enable_inheritance:
            # Create inheritance analysis on-the-fly using completed configs
            inheritance_analysis = self._create_inheritance_analysis(config_class_name)
            return core.get_inheritance_aware_form_fields(
                config_class_name, 
                inheritance_analysis
            )
        else:
            # Fallback to standard field generation
            return core._get_form_fields(config_class)
    
    def _create_inheritance_analysis(self, config_class_name: str) -> Dict[str, Any]:
        """Create inheritance analysis on-the-fly using StepCatalog and completed configs."""
        try:
            # Use UniversalConfigCore's step_catalog for inheritance analysis
            from ..core.core import UniversalConfigCore
            core = UniversalConfigCore()
            
            if core.step_catalog:
                # Get parent class and values using StepCatalog methods
                parent_class = core.step_catalog.get_immediate_parent_config_class(config_class_name)
                parent_values = core.step_catalog.extract_parent_values_for_inheritance(
                    config_class_name, self.completed_configs
                )
                
                return {
                    'inheritance_enabled': True,
                    'immediate_parent': parent_class,
                    'parent_values': parent_values,
                    'total_inherited_fields': len(parent_values)
                }
            else:
                # StepCatalog not available
                return {
                    'inheritance_enabled': False,
                    'immediate_parent': None,
                    'parent_values': {},
                    'total_inherited_fields': 0
                }
                
        except Exception as e:
            logger.warning(f"Failed to create inheritance analysis for {config_class_name}: {e}")
            return {
                'inheritance_enabled': False,
                'immediate_parent': None,
                'parent_values': {},
                'total_inherited_fields': 0,
                'error': str(e)
            }
    
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
            # Update navigation and current step without full redisplay
            self._update_navigation_and_step()
    
    def _on_next_clicked(self, button):
        """Handle next button click."""
        # Save current step
        if self._save_current_step():
            if self.current_step < len(self.steps) - 1:
                self.current_step += 1
                # Update navigation and current step without full redisplay
                self._update_navigation_and_step()
    
    def _update_navigation_and_step(self):
        """Update navigation and current step without full widget recreation."""
        # Update navigation display
        with self.navigation_output:
            clear_output(wait=True)
            self._display_navigation()
        
        # Update current step display
        with self.output:
            clear_output(wait=True)
            self._display_current_step()
    
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
