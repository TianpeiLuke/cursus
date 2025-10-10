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

# TEMPORARILY ENABLE LOGGING FOR DEBUGGING
logging.getLogger('cursus.api.config_ui').setLevel(logging.INFO)
logging.getLogger('cursus.core').setLevel(logging.INFO)
logging.getLogger('cursus.step_catalog').setLevel(logging.ERROR)
logging.getLogger('cursus.step_catalog.step_catalog').setLevel(logging.ERROR)
logging.getLogger('cursus.step_catalog.builder_discovery').setLevel(logging.ERROR)
logging.getLogger('cursus.step_catalog.config_discovery').setLevel(logging.ERROR)
# Suppress all cursus-related loggers except config_ui
logging.getLogger('cursus').setLevel(logging.ERROR)

# Handle both relative and absolute imports using centralized path setup
try:
    # Try relative imports first (when run as module)
    from ....core.base.config_base import BasePipelineConfig
    from ....steps.configs.config_processing_step_base import ProcessingStepConfigBase
    from ..core.data_sources_manager import DataSourcesManager
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
    
    def __init__(self, form_data: Dict[str, Any], is_final_step: bool = True, config_core=None):
        """
        Initialize universal configuration widget.
        
        Args:
            form_data: Form data containing config class, fields, values, etc.
            is_final_step: Whether this is the final step in a multi-step wizard
            config_core: UniversalConfigCore instance for field definitions (CRITICAL FIX)
        """
        self.form_data = form_data
        self.config_class = form_data["config_class"]
        self.config_class_name = form_data["config_class_name"]
        self.fields = form_data["fields"]
        self.values = form_data["values"]
        self.pre_populated_instance = form_data.get("pre_populated_instance")
        self.is_final_step = is_final_step
        
        # CRITICAL FIX: Store config_core for field definitions
        self.config_core = config_core
        
        self.widgets = {}
        self.config_instance = None
        self.output = widgets.Output()
        
        # ROBUST SOLUTION: Display state management
        self._is_rendered = False  # Track if content has been rendered
        self._is_displayed = False  # Track if output has been displayed
        
        logger.info(f"UniversalConfigWidget initialized for {self.config_class_name} with config_core: {config_core is not None}")
    
    def render(self):
        """ROBUST SOLUTION: Render widget content (internal method)."""
        if self._is_rendered:
            return  # Already rendered, skip
        
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
            
            # Check if this is CradleDataLoadingConfig for sub-config grouping
            if self.config_class_name == "CradleDataLoadingConfig":
                form_sections = self._create_field_sections_by_subconfig()
            else:
                # Enhanced 4-tier field categorization with inheritance for other configs
                inherited_fields = [f for f in self.fields if f.get('tier') == 'inherited']
                essential_fields = [f for f in self.fields if f.get('tier') == 'essential' or (f.get('required', False) and f.get('tier') != 'inherited')]
                system_fields = [f for f in self.fields if f.get('tier') == 'system' or (not f.get('required', False) and f.get('tier') not in ['essential', 'inherited'])]
                
                form_sections = self._create_field_sections_by_tier(inherited_fields, essential_fields, system_fields)
            
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
        
        self._is_rendered = True  # Mark as rendered
        logger.debug(f"Widget content rendered for {self.config_class_name}")
    
    def display(self):
        """ROBUST SOLUTION: Safe display method - ensures content is rendered but not duplicated."""
        # Always render content first (idempotent)
        self.render()
        
        # Return the output widget for external display
        # This allows the caller to decide when/how to display it
        return self.output
    
    def show(self):
        """ROBUST SOLUTION: Force display the widget output (for standalone use)."""
        if self._is_displayed:
            logger.debug(f"Widget already displayed for {self.config_class_name}, skipping duplicate display")
            return
        
        # Ensure content is rendered
        self.render()
        
        # ROBUST SOLUTION: Enhanced display for VS Code compatibility
        try:
            # Primary display method
            display(self.output)
            self._is_displayed = True
            logger.debug(f"Widget displayed for {self.config_class_name}")
        except Exception as e:
            logger.error(f"Error displaying widget: {e}")
            # Fallback: Try to display content directly
            try:
                with self.output:
                    pass  # Content already rendered
                display(self.output)
                self._is_displayed = True
                logger.debug(f"Fallback display successful for {self.config_class_name}")
            except Exception as e2:
                logger.error(f"Fallback display also failed: {e2}")
    
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
        
        # Create appropriate widget based on field type with ENHANCED support for new types
        if field_type == "text":
            widget = widgets.Text(
                value=str(current_value) if current_value is not None else "",
                description=f"{emoji_icon} {field_name}{required_indicator}:",
                style={'description_width': '200px'},
                layout=widgets.Layout(width='500px', margin='5px 0')
            )
        elif field_type == "datetime":
            # NEW: Enhanced datetime field widget
            widget = widgets.Text(
                value=str(current_value) if current_value else "",
                placeholder=field.get("placeholder", "YYYY-MM-DDTHH:MM:SS"),
                description=f"{emoji_icon} {field_name}{required_indicator}:",
                style={'description_width': '200px'},
                layout=widgets.Layout(width='400px', margin='5px 0')
            )
        elif field_type == "code_editor":
            # NEW: Enhanced code editor field widget (textarea with SQL syntax)
            # Special handling for transform_sql to give it a much larger window
            if field_name == "transform_sql":
                height = '300px'  # Much larger for SQL editing
                width = '900px'   # Wider for SQL queries
                placeholder = "Enter your SQL transformation query here...\n\nExample:\nSELECT objectId, transactionDate, is_abuse\nFROM input_data\nWHERE transactionDate >= '2025-01-01'\nAND is_abuse IS NOT NULL"
            else:
                height = field.get('height', '150px')
                width = '800px'
                placeholder = f"Enter {field.get('language', 'code')}..."
            
            widget = widgets.Textarea(
                value=str(current_value) if current_value else field.get("default", ""),
                placeholder=placeholder,
                description=f"{emoji_icon} {field_name}{required_indicator}:",
                style={'description_width': '200px'},
                layout=widgets.Layout(width=width, height=height, margin='5px 0')
            )
        elif field_type == "tag_list":
            # NEW: Enhanced tag list field widget (comma-separated values)
            if isinstance(current_value, list):
                value_str = ", ".join(str(item) for item in current_value)
            else:
                value_str = str(current_value) if current_value else ""
            widget = widgets.Text(
                value=value_str,
                placeholder="Enter comma-separated values",
                description=f"{emoji_icon} {field_name}{required_indicator}:",
                style={'description_width': '200px'},
                layout=widgets.Layout(width='600px', margin='5px 0')
            )
        elif field_type == "radio":
            # NEW: Enhanced radio button field widget
            options = field.get("options", [])
            default_value = field.get("default")
            selected_value = current_value if current_value in options else default_value
            widget = widgets.RadioButtons(
                options=options,
                value=selected_value if selected_value in options else None,
                description=f"{emoji_icon} {field_name}{required_indicator}:",
                style={'description_width': '200px'},
                layout=widgets.Layout(margin='10px 0')
            )
        elif field_type == "dropdown":
            # NEW: Enhanced dropdown field widget
            options = field.get("options", [])
            default_value = field.get("default")
            selected_value = current_value if current_value in options else default_value
            widget = widgets.Dropdown(
                options=options,
                value=selected_value if selected_value in options else (options[0] if options else None),
                description=f"{emoji_icon} {field_name}{required_indicator}:",
                style={'description_width': '200px'},
                layout=widgets.Layout(width='300px', margin='5px 0')
            )
        elif field_type == "textarea":
            # NEW: Enhanced textarea field widget
            widget = widgets.Textarea(
                value=str(current_value) if current_value else field.get("default", ""),
                description=f"{emoji_icon} {field_name}{required_indicator}:",
                placeholder=field.get("placeholder", "Enter text..."),
                style={'description_width': '200px'},
                layout=widgets.Layout(width='600px', height='100px', margin='5px 0')
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
        elif field_type == "dynamic_data_sources":
            # NEW: Create dynamic data sources widget
            return self._create_dynamic_data_sources_widget(field)
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
        
        # CRITICAL: Create the actual specialized widget instance
        try:
            from .specialized_widgets import SpecializedComponentRegistry
            registry = SpecializedComponentRegistry()
            
            # Create the specialized widget with base config pre-population
            base_config = self.values  # Pass current form values as base config
            specialized_widget = registry.create_specialized_widget(
                config_class_name,
                base_config=base_config,
                completion_callback=self._on_specialized_widget_complete
            )
            
            if specialized_widget:
                logger.info(f"Created specialized widget for {config_class_name}")
                return {
                    "widget": specialized_widget,
                    "container": specialized_widget.display() if hasattr(specialized_widget, 'display') else specialized_widget
                }
            else:
                logger.warning(f"Failed to create specialized widget for {config_class_name}")
                
        except Exception as e:
            logger.error(f"Error creating specialized widget for {config_class_name}: {e}")
        
        # Fallback: Create visual placeholder if specialized widget creation fails
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
            
            <div style='background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 15px; margin-bottom: 15px;'>
                <h4 style='margin: 0 0 10px 0; color: #dc2626; font-size: 14px;'>⚠️ Widget Creation Failed</h4>
                <p style='margin: 0; color: #dc2626; font-size: 13px;'>
                    Could not create specialized widget for {config_class_name}. Please check the implementation.
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
    
    def _create_dynamic_data_sources_widget(self, field: Dict) -> Dict:
        """Create dynamic data sources widget section."""
        
        # Initialize data sources manager with discovery-based field templates
        initial_data = self.values.get("data_sources", [])
        
        # Pass the config core if available for discovery
        config_core = getattr(self, 'config_core', None)
        data_sources_manager = DataSourcesManager(initial_data, config_core=config_core)
        
        # Create section container with styling
        section_html = """
        <div style='margin: 10px 0;'>
            <h5 style='margin: 10px 0; color: #374151; display: flex; align-items: center;'>
                📊 Data Sources (Dynamic List)
            </h5>
            <p style='margin: 5px 0 15px 0; font-size: 12px; color: #6b7280; font-style: italic;'>
                Configure one or more data sources for your job. Click "Add Data Source" to add additional sources.
            </p>
        </div>
        """
        
        section_header = widgets.HTML(section_html)
        section_container = widgets.VBox([
            section_header,
            data_sources_manager.container
        ])
        
        return {
            "widget": data_sources_manager,  # Store manager for data collection
            "container": section_container
        }
    
    def _on_specialized_widget_complete(self, config_instance):
        """Handle completion of specialized widget configuration."""
        logger.info(f"Specialized widget completed with config: {type(config_instance)}")
        # Store the completed config for later retrieval
        self.config_instance = config_instance
    
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
    
    def _create_field_sections_by_subconfig(self) -> List[widgets.Widget]:
        """Create field sections organized by sub-config blocks for CradleDataLoadingConfig."""
        try:
            from ..core.field_definitions import get_cradle_fields_by_sub_config, get_sub_config_section_metadata
            
            # CRITICAL FIX: Pass the config_core instance to get the correct field definitions
            field_blocks = get_cradle_fields_by_sub_config(config_core=self.config_core)
            section_metadata = get_sub_config_section_metadata()
            
            logger.info(f"Using config_core: {self.config_core is not None} for field definitions")
            
            form_sections = []
            
            # Create sections in the requested order:
            # 1. Data Sources Specification 
            # 2. Transform Specification 
            # 3. Output Specification
            # 4. Cradle Job Specification
            # 5. Inherited Configuration
            section_order = ["data_sources_spec", "transform_spec", "output_spec", "cradle_job_spec", "inherited"]
            
            for section_name in section_order:
                if section_name in field_blocks and field_blocks[section_name]:
                    fields = field_blocks[section_name]
                    metadata = section_metadata.get(section_name, {})
                    
                    # Create section with metadata styling
                    section = self._create_field_section(
                        metadata.get("title", f"{section_name.title()} Configuration"),
                        fields,
                        metadata.get("bg_gradient", "linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%)"),
                        metadata.get("border_color", "#9ca3af"),
                        metadata.get("description", f"Configure {section_name} settings")
                    )
                    form_sections.append(section)
            
            # Add root fields if they exist (job_type, etc.)
            if "root" in field_blocks and field_blocks["root"]:
                root_fields = field_blocks["root"]
                root_metadata = section_metadata.get("root", {})
                
                root_section = self._create_field_section(
                    root_metadata.get("title", "🎯 Job Configuration"),
                    root_fields,
                    root_metadata.get("bg_gradient", "linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)"),
                    root_metadata.get("border_color", "#f59e0b"),
                    root_metadata.get("description", "Select job type and advanced options")
                )
                # Insert root section at the beginning
                form_sections.insert(0, root_section)
            
            logger.info(f"Created {len(form_sections)} sub-config sections for CradleDataLoadingConfig")
            return form_sections
            
        except Exception as e:
            logger.error(f"Error creating sub-config sections: {e}")
            # Fallback to tier-based organization
            return self._create_field_sections_by_tier_fallback()
    
    def _create_field_sections_by_tier(self, inherited_fields: List[Dict], essential_fields: List[Dict], system_fields: List[Dict]) -> List[widgets.Widget]:
        """Create field sections organized by tier for non-CradleDataLoadingConfig."""
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
        
        return form_sections
    
    def _create_field_sections_by_tier_fallback(self) -> List[widgets.Widget]:
        """Fallback method to create sections by tier when sub-config organization fails."""
        # Enhanced 4-tier field categorization with inheritance for fallback
        inherited_fields = [f for f in self.fields if f.get('tier') == 'inherited']
        essential_fields = [f for f in self.fields if f.get('tier') == 'essential' or (f.get('required', False) and f.get('tier') != 'inherited')]
        system_fields = [f for f in self.fields if f.get('tier') == 'system' or (not f.get('required', False) and f.get('tier') not in ['essential', 'inherited'])]
        
        return self._create_field_sections_by_tier(inherited_fields, essential_fields, system_fields)

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
        # If config_instance already exists, return it
        if self.config_instance:
            return self.config_instance
        
        # If no config_instance, try to create it from current widget values
        try:
            return self._create_config_from_current_values()
        except Exception as e:
            logger.error(f"Error creating config from current values: {e}")
            return None
    
    def _create_config_from_current_values(self) -> Optional[BasePipelineConfig]:
        """Create configuration instance from current widget values."""
        try:
            # Collect form data from current widget values
            form_data = {}
            for field_name, widget in self.widgets.items():
                # Handle special widget types
                if field_name == "data_sources" and hasattr(widget, 'get_all_data_sources'):
                    # Collect multiple data sources from DataSourcesManager
                    data_sources_list = widget.get_all_data_sources()
                    form_data[field_name] = data_sources_list
                    continue
                
                value = widget.value
                
                # Convert values based on field type
                field_info = next((f for f in self.fields if f["name"] == field_name), None)
                if field_info:
                    field_type = field_info["type"]
                    
                    if field_type == "tag_list":
                        # Convert comma-separated string back to list
                        if isinstance(value, str):
                            value = [item.strip() for item in value.split(",") if item.strip()]
                        elif not isinstance(value, list):
                            value = []
                    elif field_type == "list":
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
                        try:
                            value = float(value) if value != "" else field_info.get("default", 0.0)
                        except (ValueError, TypeError):
                            value = field_info.get("default", 0.0)
                    elif field_type == "checkbox":
                        value = bool(value)
                
                form_data[field_name] = value
            
            # Create configuration instance
            config_instance = self.config_class(**form_data)
            
            # Cache the instance for future calls
            self.config_instance = config_instance
            
            logger.info(f"Created config instance from current values: {self.config_class_name}")
            return config_instance
            
        except Exception as e:
            logger.error(f"Error creating config from current values: {e}")
            return None


class MultiStepWizard:
    """Multi-step pipeline configuration wizard with Smart Default Value Inheritance support."""
    
    def __init__(self, 
                 steps: List[Dict[str, Any]], 
                 base_config: Optional[BasePipelineConfig] = None,
                 processing_config: Optional[ProcessingStepConfigBase] = None,
                 enable_inheritance: bool = True,
                 core: Optional['UniversalConfigCore'] = None):
        """
        Initialize multi-step wizard with Smart Default Value Inheritance support.
        
        Args:
            steps: List of step definitions
            base_config: Base pipeline configuration
            processing_config: Processing configuration
            enable_inheritance: Enable smart inheritance features (NEW)
            core: UniversalConfigCore instance to use (CRITICAL FIX)
        """
        self.steps = steps
        self.base_config = base_config
        self.processing_config = processing_config
        self.enable_inheritance = enable_inheritance  # NEW: Inheritance support
        self.completed_configs = {}  # Store completed configurations
        self.current_step = 0
        self.step_widgets = {}
        
        # CRITICAL FIX: Store the core instance to avoid creating new ones
        self.core = core
        
        self.output = widgets.Output()
        self.navigation_output = widgets.Output()
        
        # NEW: Navigation control for nested wizards
        self.navigation_disabled = False
        self.next_button = None
        self.prev_button = None
        self.finish_button = None
        
        # NEW: Initialize completed configs for inheritance
        if self.enable_inheritance:
            if base_config:
                self.completed_configs["BasePipelineConfig"] = base_config
            if processing_config:
                self.completed_configs["ProcessingStepConfigBase"] = processing_config
        
        logger.info(f"MultiStepWizard initialized with {len(steps)} steps, inheritance={'enabled' if enable_inheritance else 'disabled'}")
    
    def display(self):
        """HOLISTIC SOLUTION: Display using widget assignment to avoid nested display() calls."""
        # STEP 1: Create navigation widgets directly (not in output widget)
        self.navigation_widgets = self._create_navigation_widgets_direct()
        
        # STEP 2: Populate main output with content
        with self.output:
            clear_output(wait=True)
            self._display_current_step()
        
        # STEP 3: Create a single container with direct navigation widgets and main output
        # This avoids nested display() calls that cause duplication
        self._main_container = widgets.VBox([
            self.navigation_widgets,  # Direct widget assignment - no display() call
            self.output
        ], layout=widgets.Layout(width='100%'))
        
        # STEP 4: Display the container widget directly
        display(self._main_container)
    
    def _create_navigation_widgets_direct(self):
        """Create navigation widgets directly without using Output widgets."""
        # Get current step info
        current_step_info = self.steps[self.current_step] if self.current_step < len(self.steps) else {"title": "Complete"}
        
        # Enhanced progress indicator
        progress_percent = ((self.current_step + 1) / len(self.steps)) * 100
        
        # Create step indicators
        step_indicators = []
        step_details = []
        for i, step in enumerate(self.steps):
            if i < self.current_step:
                step_indicators.append("●")
                step_details.append(f"✅ {step['title']}")
            elif i == self.current_step:
                step_indicators.append("●")
                step_details.append(f"🔄 {step['title']} (Current)")
            else:
                step_indicators.append("○")
                step_details.append(f"⏳ {step['title']}")
        
        # Create overview HTML
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
        
        # Create HTML widgets
        overview_widget = widgets.HTML(step_overview_html)
        progress_widget = widgets.HTML(progress_html)
        
        # Create navigation buttons with enhanced styling
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
        
        # Attach event handlers
        prev_button.on_click(self._on_prev_clicked)
        next_button.on_click(self._on_next_clicked)
        finish_button.on_click(self._on_finish_clicked)
        
        # Store button references for navigation control
        self.prev_button = prev_button
        self.next_button = next_button
        self.finish_button = finish_button
        
        # Apply navigation disabled state if needed
        if self.navigation_disabled:
            self.prev_button.disabled = True
            self.next_button.disabled = True
            self.finish_button.disabled = True
        
        # Create navigation container
        nav_box = widgets.HBox(
            [prev_button, next_button, finish_button], 
            layout=widgets.Layout(
                justify_content='center', 
                margin='15px 0',
                padding='20px',
                border='2px solid #e2e8f0',
                border_radius='12px'
            )
        )
        
        # Return complete navigation section as VBox
        return widgets.VBox([
            overview_widget,
            progress_widget,
            nav_box
        ])
    
    def _ensure_vscode_widget_display(self, widget):
        """Ensure proper widget display in VS Code Jupyter extension."""
        try:
            # Force widget model creation and synchronization
            if hasattr(widget, '_model_id') and widget._model_id is None:
                widget._model_id = widget._gen_model_id()
            
            # Ensure all child widgets are properly initialized
            def _init_widget_recursive(w):
                if hasattr(w, 'children'):
                    for child in w.children:
                        if hasattr(child, '_model_id') and child._model_id is None:
                            child._model_id = child._gen_model_id()
                        _init_widget_recursive(child)
            
            _init_widget_recursive(widget)
            
            # Add VS Code specific display hints
            from IPython.display import display, HTML, Javascript
            
            # Display JavaScript to ensure widget rendering
            display(Javascript("""
            // VS Code Jupyter Widget Display Enhancement
            (function() {
                console.log('🔧 Ensuring VS Code widget compatibility...');
                
                // Force widget manager to render widgets
                if (window.Jupyter && window.Jupyter.notebook) {
                    // Classic Jupyter
                    console.log('📝 Classic Jupyter detected');
                } else if (window.requirejs) {
                    // VS Code or JupyterLab
                    console.log('🆚 VS Code/JupyterLab detected');
                    
                    // Ensure widget manager is loaded
                    setTimeout(function() {
                        const widgets = document.querySelectorAll('.widget-area, .jp-OutputArea-child');
                        console.log(`Found ${widgets.length} widget areas`);
                        
                        // Force re-render of widget areas
                        widgets.forEach(function(widget, index) {
                            if (widget.style.display === 'none') {
                                widget.style.display = 'block';
                                console.log(`Showed hidden widget ${index}`);
                            }
                        });
                    }, 100);
                }
                
                console.log('✅ Widget compatibility check complete');
            })();
            """))
            
        except Exception as e:
            logger.warning(f"Widget display enhancement failed: {e}")
    
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
        
        # NEW: Store button references for navigation control
        self.prev_button = prev_button
        self.next_button = next_button
        self.finish_button = finish_button
        
        # NEW: Apply navigation disabled state if needed
        if self.navigation_disabled:
            self.prev_button.disabled = True
            self.next_button.disabled = True
            self.finish_button.disabled = True
        
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
        """ROBUST SOLUTION: Display the current step using safe display methods."""
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
        
        # BALANCED FIX: Render widget and display its output widget (not content) to avoid duplication
        # This displays the widget's output container, not the individual content elements
        step_widget = self.step_widgets[self.current_step]
        
        # Ensure the widget is rendered (idempotent operation)
        step_widget.render()
        
        # Display the widget's output container - this shows the content without duplication
        # The output widget contains all the rendered content as a single unit
        display(step_widget.output)
    
    def _get_step_fields(self, step: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get form fields for a step with Smart Default Value Inheritance support."""
        config_class = step["config_class"]
        config_class_name = step["config_class_name"]
        
        logger.info(f"🔍 Getting step fields for {config_class_name}")
        
        # Check if there's a specialized component for this config type
        try:
            from .specialized_widgets import SpecializedComponentRegistry
            registry = SpecializedComponentRegistry()
            
            logger.info(f"🔍 Checking if {config_class_name} has specialized component...")
            if registry.has_specialized_component(config_class_name):
                logger.info(f"❌ {config_class_name} has specialized component - this should NOT happen after our fix!")
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
            else:
                logger.info(f"✅ {config_class_name} does NOT have specialized component - will use comprehensive fields")
        except ImportError as e:
            logger.info(f"⚠️ Specialized widgets not available: {e}, continuing with standard processing")
        
        # CRITICAL FIX: Use the core instance passed to constructor instead of creating new one
        if self.core:
            core = self.core
            logger.info(f"🔍 Using provided UniversalConfigCore instance to get fields for {config_class_name}")
        else:
            # Fallback: create new instance if none provided (backward compatibility)
            from ..core.core import UniversalConfigCore
            core = UniversalConfigCore()
            logger.warning(f"🔍 No core instance provided, creating new UniversalConfigCore for {config_class_name}")
        
        # NEW: Use inheritance-aware field generation if inheritance is enabled
        if self.enable_inheritance and hasattr(step, 'inheritance_analysis'):
            logger.info(f"🔍 Using inheritance-aware field generation with existing analysis")
            # Use the enhanced inheritance-aware method
            fields = core.get_inheritance_aware_form_fields(
                config_class_name, 
                step['inheritance_analysis']
            )
        elif self.enable_inheritance:
            logger.info(f"🔍 Creating inheritance analysis on-the-fly for {config_class_name}")
            # Create inheritance analysis on-the-fly using completed configs
            inheritance_analysis = self._create_inheritance_analysis(config_class_name)
            fields = core.get_inheritance_aware_form_fields(
                config_class_name, 
                inheritance_analysis
            )
        else:
            logger.info(f"🔍 Using standard field generation for {config_class_name}")
            # Fallback to standard field generation
            fields = core._get_form_fields(config_class)
        
        logger.info(f"📊 Got {len(fields)} fields for {config_class_name}")
        if len(fields) > 0:
            logger.info(f"📋 First few fields: {[f['name'] for f in fields[:5]]}")
        
        return fields
    
    def _find_base_config(self) -> Optional[Any]:
        """Find base config under any of the possible keys - robust key resolution."""
        possible_keys = [
            "BasePipelineConfig",           # Class name
            "Base Configuration",           # Step title variant 1
            "Base Pipeline Configuration",  # Step title variant 2
            "Base Config",                  # Step title variant 3
        ]
        
        for key in possible_keys:
            if key in self.completed_configs:
                logger.debug(f"Found base config under key: '{key}'")
                return self.completed_configs[key]
        
        logger.warning(f"Base config not found under any of these keys: {possible_keys}")
        logger.debug(f"Available keys in completed_configs: {list(self.completed_configs.keys())}")
        return None
    
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
        """Get pre-populated values for a step with robust base config lookup and universal inherited field population."""
        step_index = self.steps.index(step) if step in self.steps else -1
        config_class_name = step["config_class_name"]
        
        logger.info(f"🔍 Getting step values for {config_class_name} (step {step_index + 1})")
        
        # Auto-fill first step with base_config values if it's BasePipelineConfig
        if (step_index == 0 and 
            self.base_config and 
            config_class_name == "BasePipelineConfig"):
            
            logger.info(f"Auto-filling first step with base_config values: {type(self.base_config)}")
            
            # Extract values from base_config
            if hasattr(self.base_config, 'model_dump'):
                base_values = self.base_config.model_dump()
                logger.debug(f"Extracted {len(base_values)} values from base_config using model_dump()")
                return base_values
            elif hasattr(self.base_config, '__dict__'):
                base_values = {k: v for k, v in self.base_config.__dict__.items() 
                              if not k.startswith('_') and v is not None}
                logger.debug(f"Extracted {len(base_values)} values from base_config using __dict__")
                return base_values
            else:
                logger.warning("base_config has no model_dump() or __dict__ method")
        
        # UNIVERSAL INHERITED FIELD POPULATION: Use robust base config lookup for any step with inherited fields
        base_config_instance = self._find_base_config()
        if base_config_instance:
            logger.info(f"Found base config for {config_class_name} - extracting inherited values")
            
            # Extract inherited values from base config
            inherited_values = {}
            if hasattr(base_config_instance, 'model_dump'):
                inherited_values = base_config_instance.model_dump()
                logger.debug(f"Extracted {len(inherited_values)} inherited values using model_dump()")
            elif hasattr(base_config_instance, '__dict__'):
                inherited_values = {k: v for k, v in base_config_instance.__dict__.items() 
                                  if not k.startswith('_') and v is not None}
                logger.debug(f"Extracted {len(inherited_values)} inherited values using __dict__")
            
            # Add step-specific defaults for CradleDataLoadingConfig
            if config_class_name == "CradleDataLoadingConfig":
                step_values = inherited_values.copy()
                step_values.update({
                    "job_type": "training",
                    "start_date": "2025-01-01T00:00:00",
                    "end_date": "2025-04-17T00:00:00",
                    "transform_sql": "SELECT * FROM input_data",
                    "output_schema": ["objectId", "transactionDate", "is_abuse"],
                    "output_format": "PARQUET",
                    "cradle_account": "Buyer-Abuse-RnD-Dev",
                    "cluster_type": "STANDARD"
                })
                logger.info(f"Auto-filled {config_class_name} with {len(step_values)} values including {len(inherited_values)} inherited values")
                return step_values
            else:
                # For other config types, just return inherited values
                logger.info(f"Auto-filled {config_class_name} with {len(inherited_values)} inherited values")
                return inherited_values
        else:
            logger.debug(f"No base config found for {config_class_name} - checking other sources")
        
        # Check for pre-populated instance
        if "pre_populated" in step and step["pre_populated"]:
            instance = step["pre_populated"]
            if hasattr(instance, 'model_dump'):
                values = instance.model_dump()
                logger.debug(f"Using pre-populated instance with {len(values)} values")
                return values
            else:
                return {}
        
        # Check for pre-populated data
        if "pre_populated_data" in step and step["pre_populated_data"]:
            values = step["pre_populated_data"]
            logger.debug(f"Using pre-populated data with {len(values)} values")
            return values
        
        # Try to create from base config
        if "base_config" in step and step["base_config"]:
            config_class = step["config_class"]
            base_config = step["base_config"]
            
            if hasattr(config_class, 'from_base_config'):
                try:
                    instance = config_class.from_base_config(base_config)
                    if hasattr(instance, 'model_dump'):
                        values = instance.model_dump()
                        logger.debug(f"Created from base config with {len(values)} values")
                        return values
                except Exception as e:
                    logger.warning(f"Failed to create from base config: {e}")
        
        logger.debug(f"No values found for {config_class_name} - returning empty dict")
        return {}
    
    def _on_prev_clicked(self, button):
        """Handle previous button click with state preservation."""
        # ENHANCED: Save current step before going back
        if self._save_current_step():  # ✅ Save current changes
            if self.current_step > 0:
                self.current_step -= 1
                # Update navigation and current step without full redisplay
                self._update_navigation_and_step()
        else:
            # Show validation error if save fails
            self._show_validation_error("Please fix validation errors before navigating")
    
    def _on_next_clicked(self, button):
        """Handle next button click with detailed logging."""
        logger.info(f"🔘 Next button clicked - Current step: {self.current_step}")
        
        # Get current step info for logging
        if self.current_step < len(self.steps):
            current_step_info = self.steps[self.current_step]
            logger.info(f"🔘 Current step details: {current_step_info['title']} ({current_step_info['config_class_name']})")
        
        # Save current step with detailed logging
        logger.info(f"🔘 Attempting to save current step {self.current_step}...")
        save_result = self._save_current_step()
        logger.info(f"🔘 Save result for step {self.current_step}: {save_result}")
        
        if save_result:
            logger.info(f"✅ Step {self.current_step} saved successfully")
            
            if self.current_step < len(self.steps) - 1:
                old_step = self.current_step
                self.current_step += 1
                
                # Get next step info for logging
                next_step_info = self.steps[self.current_step]
                logger.info(f"🔘 Navigating from step {old_step} to step {self.current_step}")
                logger.info(f"🔘 Next step details: {next_step_info['title']} ({next_step_info['config_class_name']})")
                
                # Update navigation and current step without full redisplay
                logger.info(f"🔘 Calling _update_navigation_and_step()...")
                self._update_navigation_and_step()
                logger.info(f"✅ Navigation update completed for step {self.current_step}")
            else:
                logger.info(f"🔘 Already at final step {self.current_step}, no navigation needed")
        else:
            logger.error(f"❌ Failed to save step {self.current_step}, navigation blocked")
            
            # Get detailed error info
            if self.current_step < len(self.steps):
                failed_step_info = self.steps[self.current_step]
                logger.error(f"❌ Failed step details: {failed_step_info['title']} ({failed_step_info['config_class_name']})")
                
                # Check if widget exists and can provide more info
                if self.current_step in self.step_widgets:
                    step_widget = self.step_widgets[self.current_step]
                    logger.error(f"❌ Widget exists for failed step: {type(step_widget)}")
                    
                    # Try to get config to see what fails
                    try:
                        config_instance = step_widget.get_config()
                        if config_instance:
                            logger.error(f"❌ Widget has config instance: {type(config_instance)}")
                        else:
                            logger.error(f"❌ Widget get_config() returned None - this is the issue!")
                    except Exception as e:
                        logger.error(f"❌ Widget get_config() failed: {e}")
                else:
                    logger.error(f"❌ No widget found for failed step {self.current_step}")
    
    def _update_navigation_and_step(self):
        """HOLISTIC SOLUTION: Update navigation and step using widget replacement - ZERO display() calls."""
        # STEP 1: Create new navigation widgets (no display() calls)
        new_navigation_widgets = self._create_navigation_widgets_direct()
        
        # STEP 2: Replace the navigation widgets in the container
        # This should always work since display() creates _main_container
        if hasattr(self, '_main_container') and self._main_container:
            # Update the container's children with new navigation widgets
            self._main_container.children = (new_navigation_widgets, self.output)
        else:
            # This should never happen - log error but don't call display()
            logger.error("_main_container not found - navigation update failed")
            return
        
        # STEP 3: Update current step display (this is safe as it's in self.output)
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
        """Save the current step configuration with enhanced data transformation and ValidationService integration."""
        logger.info(f"🔍 _save_current_step called for step {self.current_step}")
        
        if self.current_step not in self.step_widgets:
            logger.info(f"✅ No widget for step {self.current_step}, returning True")
            return True
        
        step_widget = self.step_widgets[self.current_step]
        step = self.steps[self.current_step]
        
        logger.info(f"🔍 Saving step {self.current_step}: {step['title']} ({step['config_class_name']})")
        
        try:
            # Check if this is a specialized widget step
            if hasattr(step_widget, 'widgets') and 'specialized_component' in step_widget.widgets:
                # ENHANCED: Handle specialized widget config collection
                specialized_widget = step_widget.widgets['specialized_component']
                
                # For cradle widgets, get the config object
                if hasattr(specialized_widget, 'get_config'):
                    config_instance = specialized_widget.get_config()
                    if config_instance:
                        step_key = step["title"]
                        config_class_name = step["config_class_name"]
                        
                        self.completed_configs[step_key] = config_instance
                        self.completed_configs[config_class_name] = config_instance
                        
                        logger.info(f"Collected specialized config for '{step_key}'")
                        return True
                    else:
                        logger.warning(f"Specialized widget has no config available for '{step['title']}'")
                        return False
                else:
                    logger.warning(f"Specialized widget does not support get_config() for '{step['title']}'")
                    return False
            else:
                # ENHANCED: Handle standard widget form data collection with new field types and dynamic data sources
                form_data = {}
                for field_name, widget in step_widget.widgets.items():
                    # PHASE 2 ENHANCEMENT: Special handling for dynamic data sources
                    if field_name == "data_sources" and hasattr(widget, 'get_all_data_sources'):
                        # Collect multiple data sources from DataSourcesManager
                        data_sources_list = widget.get_all_data_sources()
                        form_data[field_name] = data_sources_list
                        logger.info(f"Collected {len(data_sources_list)} data sources from DataSourcesManager")
                        continue
                    
                    value = widget.value
                    
                    # Handle special field types with enhanced conversion
                    field_info = next((f for f in step_widget.fields if f["name"] == field_name), None)
                    if field_info:
                        field_type = field_info["type"]
                        
                        if field_type == "tag_list":
                            # Convert comma-separated string back to list
                            if isinstance(value, str):
                                value = [item.strip() for item in value.split(",") if item.strip()]
                            elif not isinstance(value, list):
                                value = []
                        elif field_type == "radio":
                            # Radio button value is already correct
                            pass
                        elif field_type == "datetime":
                            # Keep as string, validation happens in config creation
                            value = str(value) if value else ""
                        elif field_type == "code_editor":
                            # Keep as string for SQL code
                            value = str(value) if value else ""
                        elif field_type == "textarea":
                            # Keep as string
                            value = str(value) if value else ""
                        elif field_type == "dropdown":
                            # Dropdown value is already correct
                            pass
                        elif field_type == "list":
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
                            try:
                                value = float(value) if value != "" else field_info.get("default", 0.0)
                            except (ValueError, TypeError):
                                value = field_info.get("default", 0.0)
                        elif field_type == "checkbox":
                            value = bool(value)
                    
                    form_data[field_name] = value
                
                # Enhanced config creation with ValidationService integration
                config_class = step["config_class"]
                config_class_name = step["config_class_name"]
                
                if config_class_name == "CradleDataLoadingConfig":
                    # Transform flat form data to nested ui_data structure for ValidationService
                    ui_data = self._transform_cradle_form_data(form_data)
                    
                    # REUSE ORIGINAL VALIDATION AND CONFIG BUILDING LOGIC
                    try:
                        from ...cradle_ui.services.validation_service import ValidationService
                        validation_service = ValidationService()
                        config_instance = validation_service.build_final_config(ui_data)
                        logger.info(f"Created CradleDataLoadingConfig using ValidationService with {len(ui_data)} ui_data fields")
                    except ImportError as e:
                        logger.warning(f"ValidationService not available: {e}, falling back to direct config creation")
                        # Fallback: Create config directly (less robust but functional)
                        config_instance = config_class(**form_data)
                    except Exception as e:
                        logger.error(f"ValidationService failed: {e}, falling back to direct config creation")
                        # Fallback: Create config directly
                        config_instance = config_class(**form_data)
                else:
                    # Standard config creation for other classes
                    config_instance = config_class(**form_data)
                
                # Store completed configuration with BOTH step title and class name for inheritance
                step_key = step["title"]
                
                self.completed_configs[step_key] = config_instance
                self.completed_configs[config_class_name] = config_instance  # CRITICAL: Add class name mapping
                
                # CRITICAL: Update base_config and processing_config references for inheritance
                if config_class_name == "BasePipelineConfig":
                    self.base_config = config_instance
                    logger.info("Updated base_config reference for inheritance")
                elif config_class_name == "ProcessingStepConfigBase":
                    self.processing_config = config_instance
                    logger.info("Updated processing_config reference for inheritance")
                
                logger.info(f"Step '{step_key}' saved successfully with enhanced data transformation")
                logger.debug(f"Available configs for inheritance: {list(self.completed_configs.keys())}")
                return True
            
        except Exception as e:
            logger.error(f"Error saving step: {e}")
            return False
    
    def _transform_cradle_form_data(self, form_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        PHASE 2 ENHANCEMENT: Transform form data with multiple data sources support.
        
        This creates the exact same nested structure that the original cradle_ui expects,
        ensuring 100% compatibility with the proven config building logic, but now
        supports multiple data sources from DataSourcesManager.
        
        Args:
            form_data: Form data from single-page form with multiple data sources
            
        Returns:
            Nested ui_data structure compatible with ValidationService
        """
        logger.debug(f"Transforming cradle form data with {len(form_data)} fields")
        
        # PHASE 2 ENHANCEMENT: Process multiple data sources from DataSourcesManager
        data_sources_list = form_data.get("data_sources", [])
        logger.info(f"Processing {len(data_sources_list)} data sources for transformation")
        
        # Transform each data source to the expected format
        transformed_data_sources = []
        for i, source_data in enumerate(data_sources_list):
            data_source_type = source_data.get("data_source_type", "MDS")
            logger.debug(f"Transforming data source {i+1}: {data_source_type}")
            
            # Create type-specific properties based on actual config structure
            if data_source_type == "MDS":
                data_source_properties = {
                    "mds_data_source_properties": {
                        "service_name": source_data.get("service_name", "AtoZ"),
                        "region": source_data.get("region", "NA"),
                        "output_schema": source_data.get("output_schema", []),
                        "org_id": source_data.get("org_id", 0),
                        "use_hourly_edx_data_set": source_data.get("use_hourly_edx_data_set", False)
                    }
                }
            elif data_source_type == "EDX":
                data_source_properties = {
                    "edx_data_source_properties": {
                        "edx_provider": source_data.get("edx_provider", ""),
                        "edx_subject": source_data.get("edx_subject", ""),
                        "edx_dataset": source_data.get("edx_dataset", ""),
                        "edx_manifest_key": source_data.get("edx_manifest_key", ""),
                        "schema_overrides": source_data.get("schema_overrides", [])
                    }
                }
            elif data_source_type == "ANDES":
                data_source_properties = {
                    "andes_data_source_properties": {
                        "provider": source_data.get("provider", ""),
                        "table_name": source_data.get("table_name", ""),
                        "andes3_enabled": source_data.get("andes3_enabled", True)
                    }
                }
            else:
                # Default to MDS if type is unknown
                logger.warning(f"Unknown data source type: {data_source_type}, defaulting to MDS")
                data_source_properties = {
                    "mds_data_source_properties": {
                        "service_name": "AtoZ",
                        "region": "NA",
                        "output_schema": ["objectId", "transactionDate"],
                        "org_id": 0,
                        "use_hourly_edx_data_set": False
                    }
                }
            
            # Create data source config
            data_source_config = {
                "data_source_name": source_data.get("data_source_name", f"RAW_{data_source_type}_NA"),
                "data_source_type": data_source_type,
                **data_source_properties
            }
            transformed_data_sources.append(data_source_config)
        
        # Fallback: If no data sources provided, create a default MDS source
        if not transformed_data_sources:
            logger.warning("No data sources found, creating default MDS data source")
            default_data_source = {
                "data_source_name": "RAW_MDS_NA",
                "data_source_type": "MDS",
                "mds_data_source_properties": {
                    "service_name": "AtoZ",
                    "region": "NA",
                    "output_schema": ["objectId", "transactionDate"],
                    "org_id": 0,
                    "use_hourly_edx_data_set": False
                }
            }
            transformed_data_sources.append(default_data_source)
        
        # Create ui_data structure that matches ValidationService.build_final_config() expectations
        ui_data = {
            # Root level fields (BasePipelineConfig)
            "job_type": form_data.get("job_type", "training"),
            "author": form_data.get("author", "test-user"),
            "bucket": form_data.get("bucket", "test-bucket"),
            "role": form_data.get("role", "arn:aws:iam::123456789012:role/test-role"),
            "region": form_data.get("region", "NA"),
            "service_name": form_data.get("service_name", "test-service"),
            "pipeline_version": form_data.get("pipeline_version", "1.0.0"),
            "project_root_folder": form_data.get("project_root_folder", "test-project"),
            
            # LEVEL 3: Nested specification structures (exact match with ValidationService expectations)
            "data_sources_spec": {
                "start_date": form_data.get("start_date", "2025-01-01T00:00:00"),
                "end_date": form_data.get("end_date", "2025-04-17T00:00:00"),
                "data_sources": transformed_data_sources  # PHASE 2: Multiple data sources support
            },
            
            "transform_spec": {
                "transform_sql": form_data.get("transform_sql", "SELECT * FROM input_data"),
                "job_split_options": {
                    "split_job": form_data.get("split_job", False),
                    "days_per_split": form_data.get("days_per_split", 7),
                    "merge_sql": form_data.get("merge_sql", "SELECT * FROM INPUT") if form_data.get("split_job") else None
                }
            },
            
            "output_spec": {
                "output_schema": form_data.get("output_schema", ["objectId", "transactionDate", "is_abuse"]),
                "pipeline_s3_loc": f"s3://{form_data.get('bucket', 'test-bucket')}/{form_data.get('project_root_folder', 'test-project')}",
                "output_format": form_data.get("output_format", "PARQUET"),
                "output_save_mode": form_data.get("output_save_mode", "ERRORIFEXISTS"),
                "output_file_count": form_data.get("output_file_count", 0),
                "keep_dot_in_output_schema": form_data.get("keep_dot_in_output_schema", False),
                "include_header_in_s3_output": form_data.get("include_header_in_s3_output", True)
            },
            
            "cradle_job_spec": {
                "cradle_account": form_data.get("cradle_account", "Buyer-Abuse-RnD-Dev"),
                "cluster_type": form_data.get("cluster_type", "STANDARD"),
                "extra_spark_job_arguments": form_data.get("extra_spark_job_arguments", ""),
                "job_retry_count": form_data.get("job_retry_count", 1)
            }
        }
        
        # Add optional fields if present
        if form_data.get("s3_input_override"):
            ui_data["s3_input_override"] = form_data["s3_input_override"]
        
        logger.info(f"Transformed ui_data structure with {len(transformed_data_sources)} data sources")
        return ui_data
    
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
    
    def _show_validation_error(self, message: str):
        """Show validation error with professional styling."""
        error_html = f"""
        <div style='background: #fef2f2; border: 1px solid #fecaca; color: #dc2626; 
                    padding: 12px; border-radius: 8px; margin: 10px 0;'>
            <strong>⚠️ Validation Error:</strong> {message}
        </div>
        """
        with self.output:
            display(widgets.HTML(error_html))
    
    def _handle_navigation_control(self, action: str):
        """Handle navigation control from nested wizards."""
        if action == 'disable_navigation':
            self.navigation_disabled = True
            if self.prev_button:
                self.prev_button.disabled = True
            if self.next_button:
                self.next_button.disabled = True
            if self.finish_button:
                self.finish_button.disabled = True
            logger.debug("Navigation disabled by nested wizard")
        elif action == 'enable_navigation':
            self.navigation_disabled = False
            if self.prev_button:
                self.prev_button.disabled = (self.current_step == 0)
            if self.next_button:
                self.next_button.disabled = (self.current_step == len(self.steps) - 1)
            if self.finish_button:
                self.finish_button.disabled = (self.current_step != len(self.steps) - 1)
            logger.debug("Navigation enabled by nested wizard")
    
    def _setup_specialized_widget_callbacks(self, step_widget, config_class_name: str):
        """Set up callbacks for specialized widgets to control navigation."""
        if config_class_name == "CradleDataLoadingConfig":
            # Check if the widget has specialized components
            if hasattr(step_widget, 'widgets') and 'specialized_component' in step_widget.widgets:
                specialized_widget = step_widget.widgets['specialized_component']
                
                # Set up navigation callback if the specialized widget supports it
                if hasattr(specialized_widget, 'set_navigation_callback'):
                    specialized_widget.set_navigation_callback(self._handle_navigation_control)
                    logger.debug(f"Set up navigation callback for {config_class_name}")
