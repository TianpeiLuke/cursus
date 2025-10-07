"""
Specialized widgets for complex configuration types

This module provides specialized UI components for configuration types that require
more complex interfaces than the standard form-based approach.
"""

import logging
from typing import Any, Dict, List, Optional, Union, Type
import ipywidgets as widgets
from IPython.display import display, clear_output
import json

from ...core.base.hyperparameters_base import ModelHyperparameters
from ...steps.hyperparams.hyperparameters_xgboost import XGBoostModelHyperparameters
from ..cradle_ui.services.config_builder import ConfigBuilderService
from ...steps.configs.config_cradle_data_loading_step import CradleDataLoadConfig

logger = logging.getLogger(__name__)


class HyperparametersConfigWidget:
    """Specialized widget for model hyperparameters configuration."""
    
    def __init__(self, 
                 base_hyperparameter: Optional[ModelHyperparameters] = None,
                 hyperparameter_class: Type[ModelHyperparameters] = XGBoostModelHyperparameters):
        """
        Initialize hyperparameters configuration widget.
        
        Args:
            base_hyperparameter: Base hyperparameter instance to inherit from
            hyperparameter_class: Target hyperparameter class to create
        """
        self.base_hyperparameter = base_hyperparameter
        self.hyperparameter_class = hyperparameter_class
        self.field_widgets = {}
        self.config_instance = None
        self.output = widgets.Output()
        
        # Initialize with base values if provided
        self.current_values = {}
        if base_hyperparameter:
            self.current_values = base_hyperparameter.get_public_init_fields()
        
        logger.info(f"HyperparametersConfigWidget initialized for {hyperparameter_class.__name__}")
    
    def display(self):
        """Display the hyperparameters configuration interface."""
        with self.output:
            clear_output(wait=True)
            
            # Create title
            title = widgets.HTML(f"<h3>Configure {self.hyperparameter_class.__name__}</h3>")
            display(title)
            
            # Create tabbed interface
            tab_children = []
            tab_titles = []
            
            # Tab 1: Field Lists Management
            field_lists_tab = self._create_field_lists_tab()
            tab_children.append(field_lists_tab)
            tab_titles.append("Field Lists")
            
            # Tab 2: Model Parameters
            model_params_tab = self._create_model_parameters_tab()
            tab_children.append(model_params_tab)
            tab_titles.append("Model Parameters")
            
            # Tab 3: Advanced Parameters (if XGBoost)
            if self.hyperparameter_class == XGBoostModelHyperparameters:
                advanced_tab = self._create_xgboost_advanced_tab()
                tab_children.append(advanced_tab)
                tab_titles.append("XGBoost Advanced")
            
            # Create tab widget
            tabs = widgets.Tab(children=tab_children)
            for i, title in enumerate(tab_titles):
                tabs.set_title(i, title)
            
            display(tabs)
            
            # Create action buttons
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
            display(button_box)
        
        display(self.output)
    
    def _create_field_lists_tab(self) -> widgets.Widget:
        """Create the field lists management tab."""
        # Full field list editor
        full_field_list = self.current_values.get('full_field_list', [])
        self.field_widgets['full_field_list'] = widgets.Textarea(
            value='\n'.join(full_field_list),
            description='Full Field List:',
            placeholder='Enter field names, one per line',
            layout=widgets.Layout(width='500px', height='150px'),
            style={'description_width': 'initial'}
        )
        
        # Tabular fields multi-select
        tab_field_list = self.current_values.get('tab_field_list', [])
        self.field_widgets['tab_field_list'] = widgets.SelectMultiple(
            options=full_field_list,
            value=tab_field_list,
            description='Tabular Fields:',
            layout=widgets.Layout(width='500px', height='120px'),
            style={'description_width': 'initial'}
        )
        
        # Categorical fields multi-select
        cat_field_list = self.current_values.get('cat_field_list', [])
        self.field_widgets['cat_field_list'] = widgets.SelectMultiple(
            options=full_field_list,
            value=cat_field_list,
            description='Categorical Fields:',
            layout=widgets.Layout(width='500px', height='120px'),
            style={'description_width': 'initial'}
        )
        
        # ID and Label fields
        self.field_widgets['id_name'] = widgets.Dropdown(
            options=full_field_list,
            value=self.current_values.get('id_name', full_field_list[0] if full_field_list else None),
            description='ID Field:',
            style={'description_width': 'initial'}
        )
        
        self.field_widgets['label_name'] = widgets.Dropdown(
            options=full_field_list,
            value=self.current_values.get('label_name', full_field_list[-1] if full_field_list else None),
            description='Label Field:',
            style={'description_width': 'initial'}
        )
        
        # Multiclass categories
        multiclass_categories = self.current_values.get('multiclass_categories', [0, 1])
        self.field_widgets['multiclass_categories'] = widgets.Textarea(
            value=json.dumps(multiclass_categories),
            description='Categories:',
            placeholder='Enter JSON list, e.g., [0, 1] or ["class1", "class2"]',
            layout=widgets.Layout(width='500px', height='80px'),
            style={'description_width': 'initial'}
        )
        
        # Update field options when full_field_list changes
        def update_field_options(change):
            field_names = [name.strip() for name in change['new'].split('\n') if name.strip()]
            
            # Update dropdown options
            self.field_widgets['tab_field_list'].options = field_names
            self.field_widgets['cat_field_list'].options = field_names
            self.field_widgets['id_name'].options = field_names
            self.field_widgets['label_name'].options = field_names
            
            # Set default values if not already set
            if field_names:
                if not self.field_widgets['id_name'].value:
                    self.field_widgets['id_name'].value = field_names[0]
                if not self.field_widgets['label_name'].value:
                    self.field_widgets['label_name'].value = field_names[-1]
        
        self.field_widgets['full_field_list'].observe(update_field_options, names='value')
        
        # Create validation info
        validation_info = widgets.HTML("""
        <div style="background-color: #e7f3ff; padding: 10px; border-radius: 5px; margin: 10px 0;">
            <strong>Field Validation Rules:</strong>
            <ul>
                <li>Tabular fields must be a subset of full field list</li>
                <li>Categorical fields must be a subset of full field list</li>
                <li>ID and Label fields must be in full field list</li>
                <li>Categories should match your target variable values</li>
            </ul>
        </div>
        """)
        
        return widgets.VBox([
            validation_info,
            self.field_widgets['full_field_list'],
            widgets.HBox([
                self.field_widgets['tab_field_list'],
                self.field_widgets['cat_field_list']
            ]),
            widgets.HBox([
                self.field_widgets['id_name'],
                self.field_widgets['label_name']
            ]),
            self.field_widgets['multiclass_categories']
        ])
    
    def _create_model_parameters_tab(self) -> widgets.Widget:
        """Create the model parameters tab."""
        # Essential model parameters
        self.field_widgets['model_class'] = widgets.Text(
            value=self.current_values.get('model_class', 'base_model'),
            description='Model Class:',
            style={'description_width': 'initial'}
        )
        
        self.field_widgets['lr'] = widgets.FloatText(
            value=self.current_values.get('lr', 3e-05),
            description='Learning Rate:',
            style={'description_width': 'initial'}
        )
        
        self.field_widgets['batch_size'] = widgets.IntText(
            value=self.current_values.get('batch_size', 2),
            description='Batch Size:',
            style={'description_width': 'initial'}
        )
        
        self.field_widgets['max_epochs'] = widgets.IntText(
            value=self.current_values.get('max_epochs', 3),
            description='Max Epochs:',
            style={'description_width': 'initial'}
        )
        
        self.field_widgets['device'] = widgets.IntText(
            value=self.current_values.get('device', -1),
            description='Device (-1 for CPU):',
            style={'description_width': 'initial'}
        )
        
        self.field_widgets['optimizer'] = widgets.Dropdown(
            options=['SGD', 'Adam', 'AdamW', 'RMSprop'],
            value=self.current_values.get('optimizer', 'SGD'),
            description='Optimizer:',
            style={'description_width': 'initial'}
        )
        
        # Metrics selection
        available_metrics = ['f1_score', 'auroc', 'accuracy', 'precision', 'recall']
        current_metrics = self.current_values.get('metric_choices', ['f1_score', 'auroc'])
        self.field_widgets['metric_choices'] = widgets.SelectMultiple(
            options=available_metrics,
            value=current_metrics,
            description='Metrics:',
            layout=widgets.Layout(height='120px'),
            style={'description_width': 'initial'}
        )
        
        return widgets.VBox([
            widgets.HBox([
                self.field_widgets['model_class'],
                self.field_widgets['optimizer']
            ]),
            widgets.HBox([
                self.field_widgets['lr'],
                self.field_widgets['batch_size']
            ]),
            widgets.HBox([
                self.field_widgets['max_epochs'],
                self.field_widgets['device']
            ]),
            self.field_widgets['metric_choices']
        ])
    
    def _create_xgboost_advanced_tab(self) -> widgets.Widget:
        """Create XGBoost advanced parameters tab."""
        # XGBoost specific parameters
        self.field_widgets['num_round'] = widgets.IntText(
            value=self.current_values.get('num_round', 100),
            description='Num Rounds:',
            style={'description_width': 'initial'}
        )
        
        self.field_widgets['max_depth'] = widgets.IntText(
            value=self.current_values.get('max_depth', 6),
            description='Max Depth:',
            style={'description_width': 'initial'}
        )
        
        self.field_widgets['min_child_weight'] = widgets.FloatText(
            value=self.current_values.get('min_child_weight', 1.0),
            description='Min Child Weight:',
            style={'description_width': 'initial'}
        )
        
        self.field_widgets['gamma'] = widgets.FloatText(
            value=self.current_values.get('gamma', 0.0),
            description='Gamma:',
            style={'description_width': 'initial'}
        )
        
        self.field_widgets['alpha'] = widgets.FloatText(
            value=self.current_values.get('alpha', 0.0),
            description='Alpha (L1):',
            style={'description_width': 'initial'}
        )
        
        self.field_widgets['lambda'] = widgets.FloatText(
            value=self.current_values.get('lambda', 1.0),
            description='Lambda (L2):',
            style={'description_width': 'initial'}
        )
        
        self.field_widgets['tree_method'] = widgets.Dropdown(
            options=['auto', 'exact', 'approx', 'hist', 'gpu_hist'],
            value=self.current_values.get('tree_method', 'auto'),
            description='Tree Method:',
            style={'description_width': 'initial'}
        )
        
        # Collapsible advanced section
        advanced_params = widgets.VBox([
            widgets.HTML("<h4>Advanced XGBoost Parameters</h4>"),
            widgets.HBox([
                self.field_widgets['gamma'],
                self.field_widgets['alpha']
            ]),
            widgets.HBox([
                self.field_widgets['lambda'],
                self.field_widgets['tree_method']
            ])
        ])
        
        return widgets.VBox([
            widgets.HTML("<h4>Essential XGBoost Parameters</h4>"),
            widgets.HBox([
                self.field_widgets['num_round'],
                self.field_widgets['max_depth']
            ]),
            self.field_widgets['min_child_weight'],
            advanced_params
        ])
    
    def _on_save_clicked(self, button):
        """Handle save button click."""
        try:
            # Collect all form data
            form_data = {}
            
            for field_name, widget in self.field_widgets.items():
                value = widget.value
                
                # Handle special field types
                if field_name == 'full_field_list':
                    value = [name.strip() for name in value.split('\n') if name.strip()]
                elif field_name == 'multiclass_categories':
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        value = [0, 1]  # Default binary
                elif field_name in ['tab_field_list', 'cat_field_list', 'metric_choices']:
                    value = list(value)  # Convert tuple to list
                
                form_data[field_name] = value
            
            # Validate field consistency
            self._validate_field_consistency(form_data)
            
            # Create hyperparameter instance
            if self.base_hyperparameter:
                self.config_instance = self.hyperparameter_class.from_base_hyperparam(
                    self.base_hyperparameter, **form_data
                )
            else:
                self.config_instance = self.hyperparameter_class(**form_data)
            
            with self.output:
                clear_output(wait=True)
                success_msg = widgets.HTML(
                    f"<div style='color: green; font-weight: bold;'>✓ Hyperparameters saved successfully!</div>"
                    f"<p>Configuration type: {self.hyperparameter_class.__name__}</p>"
                    f"<p>Fields: {len(form_data['full_field_list'])} total, "
                    f"{len(form_data['tab_field_list'])} tabular, "
                    f"{len(form_data['cat_field_list'])} categorical</p>"
                )
                display(success_msg)
            
            logger.info(f"Hyperparameters saved successfully: {self.hyperparameter_class.__name__}")
            
        except Exception as e:
            with self.output:
                clear_output(wait=True)
                error_msg = widgets.HTML(
                    f"<div style='color: red; font-weight: bold;'>✗ Error saving hyperparameters:</div>"
                    f"<p>{str(e)}</p>"
                )
                display(error_msg)
            
            logger.error(f"Error saving hyperparameters: {e}")
    
    def _validate_field_consistency(self, form_data: Dict[str, Any]) -> None:
        """Validate field list consistency and relationships."""
        full_fields = set(form_data['full_field_list'])
        tab_fields = set(form_data['tab_field_list'])
        cat_fields = set(form_data['cat_field_list'])
        
        # Validate subset relationships
        if not tab_fields.issubset(full_fields):
            raise ValueError("Tabular fields must be a subset of full field list")
        
        if not cat_fields.issubset(full_fields):
            raise ValueError("Categorical fields must be a subset of full field list")
        
        # Validate ID and label fields
        if form_data['id_name'] not in full_fields:
            raise ValueError("ID field must be in full field list")
        
        if form_data['label_name'] not in full_fields:
            raise ValueError("Label field must be in full field list")
        
        # Validate categories
        if len(form_data['multiclass_categories']) < 2:
            raise ValueError("Must have at least 2 categories")
    
    def _on_cancel_clicked(self, button):
        """Handle cancel button click."""
        with self.output:
            clear_output(wait=True)
            cancel_msg = widgets.HTML("<div style='color: orange;'>Hyperparameters configuration cancelled.</div>")
            display(cancel_msg)
    
    def get_config(self) -> Optional[ModelHyperparameters]:
        """Get the saved hyperparameters configuration."""
        return self.config_instance


class SpecializedComponentRegistry:
    """Registry for specialized UI components with enhanced visual integration."""
    
    SPECIALIZED_COMPONENTS = {
        "CradleDataLoadConfig": {
            "component_class": "CradleConfigWidget",
            "module": "cursus.api.cradle_ui.jupyter_widget",
            "preserve_existing_ui": True,
            "description": "Advanced data loading configuration with 5-step wizard interface",
            "features": [
                "1️⃣ Data Sources Configuration",
                "2️⃣ Transform Specification", 
                "3️⃣ Output Configuration",
                "4️⃣ Cradle Job Settings",
                "5️⃣ Validation & Preview"
            ],
            "icon": "🎛️",
            "complexity": "advanced"
        },
        "ModelHyperparameters": {
            "component_class": "HyperparametersConfigWidget",
            "module": "cursus.api.config_ui.specialized_widgets",
            "preserve_existing_ui": False,
            "description": "Comprehensive hyperparameter configuration with field management",
            "features": [
                "📊 Dynamic field list editor",
                "🎯 Feature selection interface",
                "⚙️ Model parameter tuning",
                "📈 Performance metrics selection"
            ],
            "icon": "🧠",
            "complexity": "intermediate"
        },
        "XGBoostModelHyperparameters": {
            "component_class": "HyperparametersConfigWidget",
            "module": "cursus.api.config_ui.specialized_widgets",
            "preserve_existing_ui": False,
            "description": "XGBoost-specific hyperparameter configuration with advanced options",
            "features": [
                "🌳 Tree-specific parameters",
                "📊 Boosting configuration",
                "🎯 Regularization settings",
                "⚡ Performance optimization"
            ],
            "icon": "🚀",
            "complexity": "advanced"
        }
    }
    
    def get_specialized_component(self, config_class_name: str) -> Optional[Type]:
        """Get specialized component for configuration class."""
        if config_class_name in self.SPECIALIZED_COMPONENTS:
            spec = self.SPECIALIZED_COMPONENTS[config_class_name]
            try:
                import importlib
                module = importlib.import_module(spec["module"])
                return getattr(module, spec["component_class"])
            except (ImportError, AttributeError) as e:
                logger.warning(f"Could not load specialized component {spec['component_class']}: {e}")
                return None
        return None
    
    def has_specialized_component(self, config_class_name: str) -> bool:
        """Check if a specialized component exists for the given config class."""
        return config_class_name in self.SPECIALIZED_COMPONENTS
    
    def create_specialized_widget(self, 
                                config_class_name: str, 
                                base_config=None, 
                                **kwargs) -> Optional[Any]:
        """Create a specialized widget instance."""
        component_class = self.get_specialized_component(config_class_name)
        if component_class:
            try:
                if config_class_name == "CradleDataLoadConfig":
                    # Use existing Cradle UI widget
                    return component_class(base_config=base_config, **kwargs)
                elif config_class_name in ["ModelHyperparameters", "XGBoostModelHyperparameters"]:
                    # Use hyperparameters widget
                    from ...core.base.hyperparameters_base import ModelHyperparameters
                    from ...steps.hyperparams.hyperparameters_xgboost import XGBoostModelHyperparameters
                    
                    hyperparameter_class = XGBoostModelHyperparameters if config_class_name == "XGBoostModelHyperparameters" else ModelHyperparameters
                    return component_class(
                        base_hyperparameter=base_config if isinstance(base_config, ModelHyperparameters) else None,
                        hyperparameter_class=hyperparameter_class
                    )
                else:
                    return component_class(base_config=base_config, **kwargs)
            except Exception as e:
                logger.error(f"Error creating specialized widget for {config_class_name}: {e}")
                return None
        return None


# Factory function for creating specialized widgets
def create_specialized_widget(config_class_name: str, base_config=None, **kwargs):
    """
    Factory function to create specialized widgets.
    
    Args:
        config_class_name: Name of the configuration class
        base_config: Base configuration instance
        **kwargs: Additional arguments for widget creation
        
    Returns:
        Specialized widget instance or None if not available
    """
    registry = SpecializedComponentRegistry()
    return registry.create_specialized_widget(config_class_name, base_config, **kwargs)
