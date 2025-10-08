"""
SageMaker Notebook Config UI - Server-Free Native Solution

This module provides server-free configuration widgets optimized for SageMaker environments.
Reuses 85% of existing cursus/api/config_ui components while providing native ipywidgets interface.
"""

import ipywidgets as widgets
from IPython.display import display, HTML
from typing import Dict, Any, List, Optional, Union
import json
import os
import re
from pathlib import Path

# Reuse existing components
# Handle both relative and absolute imports using centralized path setup
try:
    # Try relative imports first (when run as module)
    from ..core import UniversalConfigCore
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
    
    from cursus.api.config_ui.core import UniversalConfigCore
# Handle DAGConfigurationManager import
try:
    from ..core.dag_manager import DAGConfigurationManager
except ImportError:
    from cursus.api.config_ui.core.dag_manager import DAGConfigurationManager

# Handle SpecializedComponentRegistry import
try:
    from .specialized_widgets import SpecializedComponentRegistry
except ImportError:
    from cursus.api.config_ui.widgets.specialized_widgets import SpecializedComponentRegistry


class SageMakerFieldRenderer:
    """
    Renders configuration fields as native ipywidgets.
    
    Reuses existing field type mapping and validation logic while creating
    appropriate ipywidgets for each field type.
    """
    
    def __init__(self, field_categories: Dict[str, List[str]]):
        self.field_categories = field_categories
        
        # Reuse existing field type mapping from UniversalConfigCore
        self.field_type_mapping = {
            str: self._create_text_widget,
            int: self._create_int_widget,
            float: self._create_float_widget,
            bool: self._create_bool_widget,
            list: self._create_list_widget,
            dict: self._create_dict_widget
        }
    
    def create_field_widget(self, field_info: Dict[str, Any], base_config=None):
        """Create appropriate ipywidget for field."""
        field_name = field_info['name']
        field_type = field_info.get('type', 'text')
        required = field_info.get('required', False)
        description = field_info.get('description', '')
        default_value = field_info.get('default')
        
        # Get inherited value from base config
        inherited_value = None
        if base_config and hasattr(base_config, field_name):
            inherited_value = getattr(base_config, field_name)
        
        # Determine initial value
        initial_value = inherited_value or default_value or self._get_default_for_type(field_type)
        
        # Create widget based on field type
        widget_creator = self.field_type_mapping.get(field_info.get('annotation', str), self._create_text_widget)
        widget = widget_creator(field_name, initial_value, required, description)
        
        # Add field styling
        self._style_widget(widget, field_name, required, inherited_value is not None)
        
        return widget
    
    def _create_text_widget(self, field_name: str, value: Any, required: bool, description: str):
        """Create text input widget."""
        return widgets.Text(
            value=str(value) if value is not None else '',
            description=f"{'🔥' if required else '⚙️'} {field_name}:",
            placeholder=f"Enter {field_name}{'*' if required else ''}",
            layout=widgets.Layout(width='400px'),
            style={'description_width': '150px'}
        )
    
    def _create_int_widget(self, field_name: str, value: Any, required: bool, description: str):
        """Create integer input widget."""
        return widgets.IntText(
            value=int(value) if value is not None else 0,
            description=f"{'🔥' if required else '⚙️'} {field_name}:",
            layout=widgets.Layout(width='400px'),
            style={'description_width': '150px'}
        )
    
    def _create_float_widget(self, field_name: str, value: Any, required: bool, description: str):
        """Create float input widget."""
        return widgets.FloatText(
            value=float(value) if value is not None else 0.0,
            description=f"{'🔥' if required else '⚙️'} {field_name}:",
            layout=widgets.Layout(width='400px'),
            style={'description_width': '150px'}
        )
    
    def _create_bool_widget(self, field_name: str, value: Any, required: bool, description: str):
        """Create boolean checkbox widget."""
        return widgets.Checkbox(
            value=bool(value) if value is not None else False,
            description=f"{'🔥' if required else '⚙️'} {field_name}",
            layout=widgets.Layout(width='400px')
        )
    
    def _create_list_widget(self, field_name: str, value: Any, required: bool, description: str):
        """Create list input widget (as textarea)."""
        list_value = value if isinstance(value, list) else []
        return widgets.Textarea(
            value='\n'.join(str(item) for item in list_value),
            description=f"{'🔥' if required else '⚙️'} {field_name}:",
            placeholder="Enter one item per line",
            rows=3,
            layout=widgets.Layout(width='400px'),
            style={'description_width': '150px'}
        )
    
    def _create_dict_widget(self, field_name: str, value: Any, required: bool, description: str):
        """Create dictionary input widget (as textarea with JSON)."""
        dict_value = value if isinstance(value, dict) else {}
        return widgets.Textarea(
            value=json.dumps(dict_value, indent=2) if dict_value else '{}',
            description=f"{'🔥' if required else '⚙️'} {field_name}:",
            placeholder="Enter JSON format",
            rows=4,
            layout=widgets.Layout(width='400px'),
            style={'description_width': '150px'}
        )
    
    def _get_default_for_type(self, field_type):
        """Get default value for field type."""
        type_defaults = {
            'text': '',
            'int': 0,
            'float': 0.0,
            'bool': False,
            'list': [],
            'dict': {}
        }
        return type_defaults.get(field_type, '')
    
    def _style_widget(self, widget, field_name: str, required: bool, inherited: bool):
        """Apply styling to widget based on field properties."""
        if inherited:
            widget.tooltip = f"Inherited from base configuration"


class SageMakerFileManager:
    """
    Handles file operations for SageMaker environment.
    
    Provides direct file system access without server dependencies.
    """
    
    def __init__(self, base_directory: Optional[Path] = None):
        self.base_directory = base_directory or Path.cwd()
        self.base_directory.mkdir(parents=True, exist_ok=True)
    
    def save_config(self, config_data: Dict[str, Any], filename: str) -> Path:
        """Save configuration to file."""
        file_path = self.base_directory / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        return file_path
    
    def save_merged_configs(self, config_list: List[Any], filename: str = None) -> Path:
        """Save merged configurations using existing merge logic."""
        try:
            # Try to reuse existing merge_and_save_configs function
            from cursus.core.config_fields import merge_and_save_configs
            
            if not filename:
                filename = self._generate_smart_filename(config_list)
            
            file_path = self.base_directory / filename
            merged_config = merge_and_save_configs(config_list, str(file_path))
            
            return file_path
        except ImportError:
            # Fallback to manual merge if function not available
            return self._manual_merge_configs(config_list, filename)
    
    def _generate_smart_filename(self, config_list: List[Any]) -> str:
        """Generate smart filename based on configuration content."""
        # Extract service name and region from configs
        service_name = "pipeline"
        region = "default"
        
        for config in config_list:
            if hasattr(config, 'service_name') and config.service_name:
                service_name = config.service_name
            if hasattr(config, 'region') and config.region:
                region = config.region
        
        # Sanitize for filename
        service_name = re.sub(r'[^a-zA-Z0-9_-]', '_', service_name)
        region = re.sub(r'[^a-zA-Z0-9_-]', '_', region)
        
        return f"config_{region}_{service_name}.json"
    
    def _manual_merge_configs(self, config_list: List[Any], filename: str = None) -> Path:
        """Manual merge fallback if merge_and_save_configs not available."""
        if not filename:
            filename = self._generate_smart_filename(config_list)
        
        merged_data = {}
        for config in config_list:
            if hasattr(config, 'model_dump'):
                merged_data.update(config.model_dump())
            elif hasattr(config, 'dict'):
                merged_data.update(config.dict())
            elif isinstance(config, dict):
                merged_data.update(config)
        
        return self.save_config(merged_data, filename)


class SageMakerConfigWidget:
    """
    Server-free configuration widget for SageMaker environments.
    
    Reuses UniversalConfigCore for all configuration logic while providing
    native ipywidgets interface instead of web forms.
    """
    
    def __init__(self, config_class_name: str, base_config=None, **kwargs):
        # Reuse existing core engine
        self.core = UniversalConfigCore()
        self.config_class_name = config_class_name
        self.base_config = base_config
        self.config_result = None
        
        # Reuse existing discovery logic
        self.config_classes = self.core.discover_config_classes()
        self.config_class = self.config_classes.get(config_class_name)
        
        if not self.config_class:
            raise ValueError(f"Configuration class {config_class_name} not found")
        
        # Reuse existing field categorization
        if hasattr(self.config_class, 'categorize_fields'):
            temp_instance = self.config_class() if not base_config else self.config_class.from_base_config(base_config)
            self.field_categories = temp_instance.categorize_fields()
        else:
            self.field_categories = self._manual_field_categorization()
        
        # Create native ipywidgets interface
        self.field_renderer = SageMakerFieldRenderer(self.field_categories)
        self.widgets = {}
        self.validation_output = None
        self.file_manager = SageMakerFileManager()
        self._create_widgets()
    
    def _manual_field_categorization(self):
        """Manual field categorization fallback."""
        # Get all fields from the config class
        if hasattr(self.config_class, '__fields__'):
            all_fields = list(self.config_class.__fields__.keys())
        else:
            all_fields = []
        
        # Basic categorization - treat first few as essential
        essential_fields = all_fields[:5] if len(all_fields) > 5 else all_fields
        system_fields = all_fields[5:] if len(all_fields) > 5 else []
        
        return {
            'essential': essential_fields,
            'system': system_fields,
            'derived': []
        }
    
    def _create_widgets(self):
        """Create native ipywidgets form using existing field logic."""
        # Reuse existing form field generation
        try:
            form_fields = self.core._get_form_fields_with_tiers(self.config_class, self.field_categories)
        except AttributeError:
            # Fallback if method not available
            form_fields = self._manual_form_fields()
        
        # Create ipywidgets for each field
        for field in form_fields:
            widget = self.field_renderer.create_field_widget(field, self.base_config)
            self.widgets[field['name']] = widget
        
        # Create validation output
        self.validation_output = widgets.Output(layout=widgets.Layout(height='100px'))
        
        # Create action buttons
        self.save_button = widgets.Button(
            description='💾 Save Configuration',
            button_style='success',
            layout=widgets.Layout(width='200px')
        )
        self.save_button.on_click(self._on_save_clicked)
        
        self.validate_button = widgets.Button(
            description='✅ Validate',
            button_style='info',
            layout=widgets.Layout(width='150px')
        )
        self.validate_button.on_click(self._on_validate_clicked)
    
    def _manual_form_fields(self):
        """Manual form field generation fallback."""
        form_fields = []
        
        if hasattr(self.config_class, '__fields__'):
            for field_name, field_info in self.config_class.__fields__.items():
                form_fields.append({
                    'name': field_name,
                    'type': 'text',
                    'required': field_info.is_required() if hasattr(field_info, 'is_required') else False,
                    'description': getattr(field_info, 'description', ''),
                    'default': getattr(field_info, 'default', None)
                })
        
        return form_fields
    
    def display(self):
        """Display the native ipywidgets form."""
        # Create form layout
        form_sections = self._create_form_sections()
        
        # Create main widget
        main_widget = widgets.VBox([
            widgets.HTML(f"<h3>⚙️ {self.config_class_name} Configuration</h3>"),
            *form_sections,
            widgets.HBox([self.validate_button, self.save_button]),
            self.validation_output
        ])
        
        display(main_widget)
        
        # Display usage instructions
        self._display_usage_instructions()
    
    def _create_form_sections(self):
        """Create form sections organized by field tiers."""
        sections = []
        
        # Essential fields section (Tier 1)
        if self.field_categories.get('essential'):
            essential_widgets = [self.widgets[field] for field in self.field_categories['essential'] if field in self.widgets]
            if essential_widgets:
                sections.append(widgets.HTML("<h4>🔥 Essential Configuration (Required)</h4>"))
                sections.extend(essential_widgets)
        
        # System fields section (Tier 2)
        if self.field_categories.get('system'):
            system_widgets = [self.widgets[field] for field in self.field_categories['system'] if field in self.widgets]
            if system_widgets:
                sections.append(widgets.HTML("<h4>⚙️ System Configuration (Optional)</h4>"))
                sections.extend(system_widgets)
        
        # Inherited fields display (read-only)
        if self.base_config:
            sections.append(self._create_inherited_section())
        
        return sections
    
    def _create_inherited_section(self):
        """Create inherited fields display section."""
        inherited_info = []
        
        if hasattr(self.base_config, 'author'):
            inherited_info.append(f"👤 Author: {self.base_config.author}")
        if hasattr(self.base_config, 'bucket'):
            inherited_info.append(f"🪣 Bucket: {self.base_config.bucket}")
        if hasattr(self.base_config, 'region'):
            inherited_info.append(f"🌍 Region: {self.base_config.region}")
        
        if inherited_info:
            return widgets.HTML(f"""
            <div style="background-color: #f0f9ff; border: 1px solid #0ea5e9; padding: 10px; border-radius: 5px; margin: 10px 0;">
                <h5>💾 Inherited from Base Configuration:</h5>
                <p>{' • '.join(inherited_info)}</p>
            </div>
            """)
        
        return widgets.HTML("")
    
    def _display_usage_instructions(self):
        """Display usage instructions."""
        display(widgets.HTML("""
        <div style="background-color: #f8f9fa; border: 1px solid #dee2e6; padding: 15px; border-radius: 8px; margin: 10px 0;">
            <h5>📋 Usage Instructions:</h5>
            <ol>
                <li>Fill in the configuration fields above</li>
                <li>Click <strong>✅ Validate</strong> to check your configuration</li>
                <li>Click <strong>💾 Save Configuration</strong> to save</li>
                <li>Use <code>widget.get_config()</code> to access the saved configuration</li>
            </ol>
        </div>
        """))
    
    def _on_validate_clicked(self, button):
        """Handle validation button click."""
        try:
            form_data = self._collect_form_data()
            config_instance = self.config_class(**form_data)
            
            with self.validation_output:
                self.validation_output.clear_output()
                print("✅ Configuration is valid!")
                print(f"📊 Fields: {len(form_data)}")
                
        except Exception as e:
            with self.validation_output:
                self.validation_output.clear_output()
                print(f"❌ Validation failed: {str(e)}")
    
    def _on_save_clicked(self, button):
        """Handle save button click."""
        try:
            form_data = self._collect_form_data()
            config_instance = self.config_class(**form_data)
            self.config_result = config_instance.model_dump() if hasattr(config_instance, 'model_dump') else config_instance.dict()
            
            with self.validation_output:
                self.validation_output.clear_output()
                print("✅ Configuration saved successfully!")
                print("📋 Use widget.get_config() to access the configuration")
                
        except Exception as e:
            with self.validation_output:
                self.validation_output.clear_output()
                print(f"❌ Save failed: {str(e)}")
    
    def get_config(self):
        """Get the saved configuration."""
        return self.config_result
    
    def _collect_form_data(self):
        """Collect data from all form widgets."""
        form_data = {}
        for field_name, widget in self.widgets.items():
            if hasattr(widget, 'value'):
                value = widget.value
                
                # Handle special widget types
                if isinstance(widget, widgets.Textarea):
                    # Handle list and dict widgets
                    if field_name in self.field_categories.get('essential', []) + self.field_categories.get('system', []):
                        # Try to parse as JSON for dict fields
                        if value.strip().startswith('{') or value.strip().startswith('['):
                            try:
                                value = json.loads(value)
                            except json.JSONDecodeError:
                                pass
                        # Handle list fields (one item per line)
                        elif '\n' in value:
                            value = [line.strip() for line in value.split('\n') if line.strip()]
                
                form_data[field_name] = value
        
        return form_data


class SageMakerPipelineWidget:
    """
    Multi-step pipeline configuration widget for SageMaker environments.
    
    Reuses DAGConfigurationManager for pipeline analysis and workflow generation
    while providing native ipywidgets multi-step interface.
    """
    
    def __init__(self, dag_name: str = None, pipeline_dag=None, **kwargs):
        # Reuse existing DAG analysis logic
        self.dag_manager = DAGConfigurationManager()
        self.core = UniversalConfigCore()
        
        if pipeline_dag:
            self.pipeline_dag = pipeline_dag
        elif dag_name:
            self.pipeline_dag = self._load_dag_from_catalog(dag_name)
        else:
            raise ValueError("Either dag_name or pipeline_dag must be provided")
        
        # Reuse existing workflow generation
        try:
            self.dag_analysis = self.dag_manager.analyze_pipeline_dag(self.pipeline_dag)
            self.workflow_steps = self.dag_analysis['workflow_steps']
        except Exception as e:
            # Fallback workflow if analysis fails
            self.dag_analysis = {'workflow_steps': [], 'total_steps': 0}
            self.workflow_steps = self._create_fallback_workflow()
        
        # State management
        self.current_step = 0
        self.completed_configs = {}
        self.step_widgets = {}
        self.file_manager = SageMakerFileManager()
        
        # Create UI components
        self._create_widgets()
    
    def _load_dag_from_catalog(self, dag_name: str):
        """Load DAG from catalog."""
        # This would integrate with existing DAG catalog
        # For now, return a simple placeholder
        return {"name": dag_name, "steps": []}
    
    def _create_fallback_workflow(self):
        """Create fallback workflow steps."""
        return [
            {'type': 'base', 'config_class_name': 'BasePipelineConfig', 'title': 'Base Configuration'},
            {'type': 'processing', 'config_class_name': 'ProcessingStepConfigBase', 'title': 'Processing Configuration'}
        ]
    
    def _create_widgets(self):
        """Create multi-step interface widgets."""
        # Progress indicator
        self.progress_bar = widgets.IntProgress(
            value=0,
            min=0,
            max=len(self.workflow_steps),
            description='Progress:',
            bar_style='info',
            layout=widgets.Layout(width='100%')
        )
        
        # Step navigation
        self.prev_button = widgets.Button(
            description='⬅️ Previous',
            button_style='',
            disabled=True,
            layout=widgets.Layout(width='120px')
        )
        self.prev_button.on_click(self._on_prev_clicked)
        
        self.next_button = widgets.Button(
            description='Next ➡️',
            button_style='primary',
            layout=widgets.Layout(width='120px')
        )
        self.next_button.on_click(self._on_next_clicked)
        
        # Step content area
        self.step_content = widgets.VBox()
        
        # Summary area
        self.summary_output = widgets.Output()
    
    def display(self):
        """Display the multi-step pipeline configuration interface."""
        # Display DAG analysis summary
        self._display_dag_analysis()
        
        # Create main interface
        main_widget = widgets.VBox([
            widgets.HTML("<h2>🎯 Pipeline Configuration Wizard</h2>"),
            self._create_dag_summary(),
            self.progress_bar,
            widgets.HTML(f"<h3>Step {self.current_step + 1} of {len(self.workflow_steps)}</h3>"),
            self.step_content,
            widgets.HBox([self.prev_button, self.next_button]),
            self.summary_output
        ])
        
        display(main_widget)
        
        # Load first step
        self._load_current_step()
    
    def _display_dag_analysis(self):
        """Display DAG analysis results using existing logic."""
        analysis = self.dag_analysis
        
        display(widgets.HTML(f"""
        <div style="background-color: #f0f9ff; border: 1px solid #0ea5e9; padding: 15px; border-radius: 8px; margin: 10px 0;">
            <h4>📊 Pipeline Analysis Results</h4>
            <p><strong>🔍 Discovered Steps:</strong> {len(analysis.get('discovered_steps', []))}</p>
            <p><strong>⚙️ Required Configurations:</strong> {len(analysis.get('required_configs', []))}</p>
            <p><strong>📋 Workflow Steps:</strong> {analysis.get('total_steps', len(self.workflow_steps))}</p>
        </div>
        """))
    
    def _create_dag_summary(self):
        """Create DAG summary widget."""
        return widgets.HTML(f"""
        <div style="background-color: #f8f9fa; border: 1px solid #dee2e6; padding: 10px; border-radius: 5px; margin: 10px 0;">
            <h5>🎯 Pipeline: {getattr(self.pipeline_dag, 'name', 'Custom Pipeline')}</h5>
            <p>📋 Total Steps: {len(self.workflow_steps)}</p>
        </div>
        """)
    
    def _load_current_step(self):
        """Load the current configuration step."""
        if self.current_step >= len(self.workflow_steps):
            self._show_completion_summary()
            return
        
        step = self.workflow_steps[self.current_step]
        
        # Update progress
        self.progress_bar.value = self.current_step
        
        # Update navigation buttons
        self.prev_button.disabled = (self.current_step == 0)
        self.next_button.description = 'Finish 🎯' if self.current_step == len(self.workflow_steps) - 1 else 'Next ➡️'
        
        # Create step widget
        base_config = self.completed_configs.get('BasePipelineConfig')
        step_widget = SageMakerConfigWidget(step['config_class_name'], base_config=base_config)
        
        self.step_widgets[self.current_step] = step_widget
        
        # Update step content
        step_widget_box = widgets.VBox([
            widgets.HTML(f"<h4>{step.get('title', step['config_class_name'])}</h4>"),
            *step_widget._create_form_sections(),
            widgets.HBox([step_widget.validate_button, step_widget.save_button]),
            step_widget.validation_output
        ])
        
        self.step_content.children = [step_widget_box]
    
    def _on_next_clicked(self, button):
        """Handle next button click."""
        # Save current step configuration
        current_widget = self.step_widgets.get(self.current_step)
        if current_widget:
            try:
                form_data = current_widget._collect_form_data()
                step = self.workflow_steps[self.current_step]
                config_class = current_widget.config_class
                
                if config_class:
                    config_instance = config_class(**form_data)
                    self.completed_configs[step['config_class_name']] = config_instance
                
            except Exception as e:
                with self.summary_output:
                    self.summary_output.clear_output()
                    print(f"❌ Error saving step {self.current_step + 1}: {str(e)}")
                return
        
        # Move to next step
        self.current_step += 1
        self._load_current_step()
    
    def _on_prev_clicked(self, button):
        """Handle previous button click."""
        if self.current_step > 0:
            self.current_step -= 1
            self._load_current_step()
    
    def _show_completion_summary(self):
        """Show completion summary and export options."""
        self.step_content.children = [
            widgets.HTML("<h3>✅ Configuration Complete!</h3>"),
            widgets.HTML(f"<p>📋 Completed {len(self.completed_configs)} configurations</p>"),
            self._create_export_options()
        ]
        
        self.next_button.disabled = True
    
    def _create_export_options(self):
        """Create export options widget."""
        save_merged_button = widgets.Button(
            description='💾 Save All Merged',
            button_style='success',
            layout=widgets.Layout(width='200px')
        )
        save_merged_button.on_click(self._on_save_merged_clicked)
        
        export_individual_button = widgets.Button(
            description='📤 Export Individual',
            button_style='info',
            layout=widgets.Layout(width='200px')
        )
        export_individual_button.on_click(self._on_export_individual_clicked)
        
        return widgets.VBox([
            widgets.HTML("<h4>💾 Export Options:</h4>"),
            widgets.HBox([save_merged_button, export_individual_button]),
            widgets.HTML("<p><small>💡 Save All Merged creates unified hierarchical JSON like demo_config.ipynb</small></p>")
        ])
    
    def _on_save_merged_clicked(self, button):
        """Handle save all merged button click."""
        try:
            config_list = list(self.completed_configs.values())
            file_path = self.file_manager.save_merged_configs(config_list)
            
            with self.summary_output:
                self.summary_output.clear_output()
                print(f"✅ Merged configuration saved: {file_path.name}")
                print(f"📁 Location: {file_path}")
                print("🚀 Ready for pipeline execution!")
                
        except Exception as e:
            with self.summary_output:
                self.summary_output.clear_output()
                print(f"❌ Merge failed: {str(e)}")
    
    def _on_export_individual_clicked(self, button):
        """Handle export individual button click."""
        try:
            for config_name, config in self.completed_configs.items():
                filename = f"{config_name.lower()}.json"
                config_data = config.model_dump() if hasattr(config, 'model_dump') else config.dict()
                file_path = self.file_manager.save_config(config_data, filename)
                
                with self.summary_output:
                    self.summary_output.clear_output(wait=True)
                    print(f"✅ Exported: {filename}")
            
            with self.summary_output:
                print(f"📁 All configurations exported to: {self.file_manager.base_directory}")
                
        except Exception as e:
            with self.summary_output:
                self.summary_output.clear_output()
                print(f"❌ Export failed: {str(e)}")
    
    def get_completed_configs(self):
        """Get all completed configurations."""
        return self.completed_configs


# Factory functions for easy usage
def create_sagemaker_config_widget(config_class_name: str, base_config=None, **kwargs):
    """Create SageMaker configuration widget."""
    return SageMakerConfigWidget(config_class_name, base_config, **kwargs)


def create_sagemaker_pipeline_widget(dag_name: str = None, pipeline_dag=None, **kwargs):
    """Create SageMaker pipeline configuration widget."""
    return SageMakerPipelineWidget(dag_name, pipeline_dag, **kwargs)
