"""
Dynamic Data Sources Manager for Cradle Data Loading Configuration

Provides dynamic add/remove functionality for multiple data sources with type-specific fields.
Uses discovery-based field templates from config classes for maximum maintainability.
"""

import logging
from typing import Any, Dict, List, Optional
import ipywidgets as widgets
from IPython.display import display, clear_output

logger = logging.getLogger(__name__)


class DataSourcesManager:
    """Manages dynamic data sources with add/remove functionality using discovery-based field templates."""
    
    def __init__(self, initial_data_sources=None, config_core=None):
        # Use UniversalConfigCore's discovery system
        self.config_core = config_core or self._create_config_core()
        
        # Get data source config classes via discovery
        all_config_classes = self.config_core.discover_config_classes()
        self.data_source_config_classes = {
            "MDS": all_config_classes.get("MdsDataSourceConfig"),
            "EDX": all_config_classes.get("EdxDataSourceConfig"),
            "ANDES": all_config_classes.get("AndesDataSourceConfig")
        }
        
        # Generate field templates dynamically using existing field discovery
        self.field_templates = self._generate_field_templates_dynamically()
        
        self.data_sources = initial_data_sources or [self._create_default_data_source()]
        self.container = widgets.VBox()
        self.data_source_widgets = []
        self._render_data_sources()
    
    def _create_config_core(self):
        """Create UniversalConfigCore instance for discovery."""
        from .core import UniversalConfigCore
        return UniversalConfigCore()
    
    def _generate_field_templates_dynamically(self) -> Dict[str, Dict]:
        """Generate field templates using UniversalConfigCore's field discovery."""
        templates = {}
        
        for source_type, config_class in self.data_source_config_classes.items():
            if config_class:
                # Use existing _get_form_fields method
                fields = self.config_core._get_form_fields(config_class)
                templates[source_type] = self._convert_fields_to_template(fields)
                logger.info(f"Generated {len(fields)} fields for {source_type} data source using discovery")
            else:
                # Fallback template if config class not found
                templates[source_type] = self._create_fallback_template(source_type)
                logger.warning(f"Config class not found for {source_type}, using fallback template")
        
        return templates
    
    def _convert_fields_to_template(self, fields: List[Dict]) -> Dict:
        """Convert field definitions to template format."""
        template = {
            "required_fields": [],
            "optional_fields": [],
            "field_definitions": {}
        }
        
        for field in fields:
            field_name = field["name"]
            if field.get("required", False):
                template["required_fields"].append(field_name)
            else:
                template["optional_fields"].append(field_name)
            
            template["field_definitions"][field_name] = {
                "type": field.get("type", "text"),
                "default": field.get("default"),
                "options": field.get("options"),
                "placeholder": field.get("placeholder"),
                "tier": field.get("tier", "essential" if field.get("required") else "system")
            }
        
        return template
    
    def _create_fallback_template(self, source_type: str) -> Dict:
        """Create fallback template if config class discovery fails."""
        fallback_templates = {
            "MDS": {
                "required_fields": ["data_source_name", "service_name", "region"],
                "optional_fields": ["org_id"],
                "field_definitions": {
                    "data_source_name": {"type": "text", "default": "RAW_MDS_NA"},
                    "service_name": {"type": "text", "default": "AtoZ"},
                    "region": {"type": "dropdown", "options": ["NA", "EU", "FE"], "default": "NA"},
                    "org_id": {"type": "number", "default": 0}
                }
            },
            "EDX": {
                "required_fields": ["data_source_name", "edx_provider", "edx_subject"],
                "optional_fields": [],
                "field_definitions": {
                    "data_source_name": {"type": "text", "default": "RAW_EDX_EU"},
                    "edx_provider": {"type": "text", "default": ""},
                    "edx_subject": {"type": "text", "default": ""}
                }
            },
            "ANDES": {
                "required_fields": ["data_source_name", "provider", "table_name"],
                "optional_fields": [],
                "field_definitions": {
                    "data_source_name": {"type": "text", "default": "RAW_ANDES_NA"},
                    "provider": {"type": "text", "default": ""},
                    "table_name": {"type": "text", "default": ""}
                }
            }
        }
        return fallback_templates.get(source_type, fallback_templates["MDS"])
    
    def _create_default_data_source(self):
        """Create default MDS data source using discovered field template."""
        mds_template = self.field_templates.get("MDS", {})
        field_definitions = mds_template.get("field_definitions", {})
        
        default_source = {"data_source_type": "MDS"}
        for field_name, field_def in field_definitions.items():
            default_source[field_name] = field_def.get("default")
        
        return default_source
    
    def _create_data_source_template(self, source_type):
        """Create data source template with type-specific defaults using discovered fields."""
        template = self.field_templates.get(source_type, {})
        field_definitions = template.get("field_definitions", {})
        
        new_source = {"data_source_type": source_type}
        for field_name, field_def in field_definitions.items():
            new_source[field_name] = field_def.get("default")
        
        return new_source
    
    def add_data_source(self, source_type="MDS"):
        """Add new data source with type-specific default values using discovered templates."""
        new_source = self._create_data_source_template(source_type)
        self.data_sources.append(new_source)
        self._refresh_ui()
        logger.info(f"Added new {source_type} data source")
    
    def remove_data_source(self, index):
        """Remove data source at index (minimum 1 data source required)."""
        if len(self.data_sources) > 1:
            removed_type = self.data_sources[index].get("data_source_type", "Unknown")
            self.data_sources.pop(index)
            self._refresh_ui()
            logger.info(f"Removed {removed_type} data source at index {index}")
        else:
            logger.warning("Cannot remove last data source - minimum 1 required")
    
    def _refresh_ui(self):
        """Refresh the entire data sources UI."""
        self._render_data_sources()
    
    def _render_data_sources(self):
        """Render all data sources with add/remove functionality."""
        self.data_source_widgets = []
        
        # Clear container children instead of using context manager
        self.container.children = []
        
        # Create new children list
        new_children = []
        
        # Render each data source
        for i, source_data in enumerate(self.data_sources):
            widget_group = self._create_data_source_widget(source_data, i)
            self.data_source_widgets.append(widget_group)
            new_children.append(widget_group["widget"])
        
        # Add data source button
        add_button = widgets.Button(
            description="+ Add Data Source",
            button_style='info',
            layout=widgets.Layout(width='150px', margin='10px 0')
        )
        
        def on_add_click(button):
            self.add_data_source()
        
        add_button.on_click(on_add_click)
        new_children.append(add_button)
        
        # Update container children
        self.container.children = new_children
    
    def _create_data_source_widget(self, source_data, index):
        """Create widget for a single data source with type-specific fields."""
        source_type = source_data.get("data_source_type", "MDS")
        template = self.field_templates[source_type]
        
        # Header with type selector and remove button
        header_html = f"""
        <div style='background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%); 
                    border: 1px solid #d1d5db; border-radius: 8px 8px 0 0; 
                    padding: 12px; display: flex; justify-content: space-between; align-items: center;'>
            <h4 style='margin: 0; color: #374151;'>📊 Data Source {index + 1}</h4>
            <div style='font-size: 12px; color: #6b7280;'>Type: {source_type}</div>
        </div>
        """
        header_widget = widgets.HTML(header_html)
        
        # Type selector dropdown
        type_dropdown = widgets.Dropdown(
            options=["MDS", "EDX", "ANDES"],
            value=source_type,
            description="Type:",
            style={'description_width': '60px'},
            layout=widgets.Layout(width='150px')
        )
        
        # Remove button (disabled if only one data source)
        remove_button = widgets.Button(
            description="Remove",
            button_style='danger',
            layout=widgets.Layout(width='80px'),
            disabled=(len(self.data_sources) <= 1)
        )
        
        # Type-specific fields container
        fields_container = widgets.VBox()
        
        # Create type-specific fields
        field_widgets = {}
        for field_name, field_def in template["field_definitions"].items():
            field_widget = self._create_field_widget(field_name, field_def, source_data.get(field_name))
            field_widgets[field_name] = field_widget
            fields_container.children += (field_widget,)
        
        # Event handlers
        def on_type_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                # Update data source type and refresh fields
                self.data_sources[index]["data_source_type"] = change['new']
                self._refresh_single_data_source(index)
        
        def on_remove_click(button):
            self.remove_data_source(index)
        
        type_dropdown.observe(on_type_change)
        remove_button.on_click(on_remove_click)
        
        # Controls row
        controls = widgets.HBox([
            type_dropdown,
            remove_button
        ], layout=widgets.Layout(padding='10px'))
        
        # Complete data source widget
        data_source_widget = widgets.VBox([
            header_widget,
            controls,
            fields_container
        ], layout=widgets.Layout(
            border='1px solid #d1d5db',
            border_radius='8px',
            margin='10px 0'
        ))
        
        return {
            "widget": data_source_widget,
            "type_dropdown": type_dropdown,
            "remove_button": remove_button,
            "field_widgets": field_widgets,
            "fields_container": fields_container
        }
    
    def _create_field_widget(self, field_name, field_def, current_value):
        """Create individual field widget based on field definition."""
        field_type = field_def["type"]
        default_value = current_value if current_value is not None else field_def.get("default")
        
        if field_type == "text":
            return widgets.Text(
                value=str(default_value) if default_value else "",
                description=f"{field_name}:",
                placeholder=field_def.get("placeholder") or "",
                style={'description_width': '120px'},
                layout=widgets.Layout(width='300px', margin='5px 0')
            )
        elif field_type == "dropdown":
            options = field_def.get("options", [""])
            return widgets.Dropdown(
                options=options,
                value=default_value if default_value in options else (options[0] if options else ""),
                description=f"{field_name}:",
                style={'description_width': '120px'},
                layout=widgets.Layout(width='200px', margin='5px 0')
            )
        elif field_type == "tag_list" or field_type == "schema_list":
            # Handle both tag_list and schema_list as comma-separated text for now
            if isinstance(default_value, list):
                if default_value and isinstance(default_value[0], dict):
                    # Schema list - convert to readable format
                    value_str = ", ".join([f"{item.get('field_name', '')}:{item.get('field_type', '')}" for item in default_value])
                else:
                    # Tag list - simple join
                    value_str = ", ".join(default_value)
            else:
                value_str = str(default_value) if default_value else ""
            
            return widgets.Text(
                value=value_str,
                description=f"{field_name}:",
                placeholder="Enter comma-separated values" if field_type == "tag_list" else "field_name:type, ...",
                style={'description_width': '120px'},
                layout=widgets.Layout(width='400px', margin='5px 0')
            )
        elif field_type == "number":
            return widgets.FloatText(
                value=float(default_value) if default_value else 0.0,
                description=f"{field_name}:",
                style={'description_width': '120px'},
                layout=widgets.Layout(width='150px', margin='5px 0')
            )
        elif field_type == "checkbox":
            return widgets.Checkbox(
                value=bool(default_value),
                description=f"{field_name}:",
                style={'description_width': '120px'},
                layout=widgets.Layout(margin='5px 0')
            )
        else:
            # Default to text
            return widgets.Text(
                value=str(default_value) if default_value else "",
                description=f"{field_name}:",
                style={'description_width': '120px'},
                layout=widgets.Layout(width='300px', margin='5px 0')
            )
    
    def _collect_data_source_data(self, widget_group, index):
        """Collect data from a single data source widget group."""
        source_data = {}
        
        # Get data source type
        source_data["data_source_type"] = widget_group["type_dropdown"].value
        
        # Get field values
        template = self.field_templates[source_data["data_source_type"]]
        for field_name, field_widget in widget_group["field_widgets"].items():
            value = field_widget.value
            
            # Convert field values based on type
            field_def = template["field_definitions"][field_name]
            field_type = field_def["type"]
            
            if field_type == "tag_list" and isinstance(value, str):
                value = [item.strip() for item in value.split(",") if item.strip()]
            elif field_type == "schema_list" and isinstance(value, str):
                # Convert "field_name:type, ..." back to list of dicts
                schema_items = []
                for item in value.split(","):
                    item = item.strip()
                    if ":" in item:
                        field_name_part, field_type_part = item.split(":", 1)
                        schema_items.append({
                            "field_name": field_name_part.strip(),
                            "field_type": field_type_part.strip()
                        })
                value = schema_items
            
            source_data[field_name] = value
        
        return source_data
    
    def get_all_data_sources(self):
        """Collect data from all data source widgets."""
        collected_data = []
        for i, widget_group in enumerate(self.data_source_widgets):
            source_data = self._collect_data_source_data(widget_group, i)
            collected_data.append(source_data)
        return collected_data
    
    def _refresh_single_data_source(self, index):
        """Refresh a single data source when type changes."""
        # Update the data source data
        new_type = self.data_sources[index]["data_source_type"]
        self.data_sources[index] = self._create_data_source_template(new_type)
        self.data_sources[index]["data_source_name"] = f"RAW_{new_type}_NA"
        
        # Refresh entire UI (simpler than partial refresh)
        self._refresh_ui()
        logger.info(f"Refreshed data source {index} to type {new_type}")
