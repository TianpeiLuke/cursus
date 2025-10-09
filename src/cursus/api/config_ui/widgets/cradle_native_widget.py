"""
SageMaker Native Cradle Configuration Widget

This module provides a pure Jupyter widget implementation that replicates the exact
UX/UI of the original cradle data loading configuration wizard. It runs entirely 
within SageMaker notebooks without requiring a separate server.
"""

import logging
from typing import Any, Dict, List, Optional, Callable
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import json

# Import the configuration classes and services
try:
    from ....steps.configs.config_cradle_data_loading_step import (
        CradleDataLoadingConfig,
        DataSourcesSpecificationConfig,
        DataSourceConfig,
        MdsDataSourceConfig,
        EdxDataSourceConfig,
        AndesDataSourceConfig,
        TransformSpecificationConfig,
        JobSplitOptionsConfig,
        OutputSpecificationConfig,
        CradleJobSpecificationConfig
    )
    from ...cradle_ui.services.validation_service import ValidationService
    from ...cradle_ui.services.config_builder import ConfigBuilderService
except ImportError:
    # Fallback imports for development
    pass

logger = logging.getLogger(__name__)


class CradleNativeWidget:
    """
    SageMaker native implementation that replicates the exact cradle UI experience.
    
    This widget provides the same 4-step configuration wizard as the original
    cradle UI but runs entirely within Jupyter widgets without requiring a server.
    """
    
    def __init__(self, 
                 base_config: Optional[Dict[str, Any]] = None,
                 embedded_mode: bool = False,
                 completion_callback: Optional[Callable] = None):
        """
        Initialize the SageMaker native cradle widget.
        
        Args:
            base_config: Base configuration values to inherit
            embedded_mode: Whether running in embedded mode within enhanced widget
            completion_callback: Callback to call when configuration is complete
        """
        self.base_config = base_config or {}
        self.embedded_mode = embedded_mode
        self.completion_callback = completion_callback
        
        # Initialize services
        try:
            self.validation_service = ValidationService()
            self.config_builder = ConfigBuilderService()
        except:
            self.validation_service = None
            self.config_builder = None
        
        # Widget state
        self.current_step = 1
        self.total_steps = 5  # 4 config steps + 1 completion step
        self.steps_data = {}
        self.completed_config = None
        
        # UI components
        self.main_container = None
        self.wizard_content = None
        self.navigation_container = None
        
        # Step widgets storage
        self.step_widgets = {}
        
        logger.info(f"CradleNativeWidget initialized, embedded_mode={embedded_mode}")
    
    def display(self):
        """Display the cradle configuration wizard with exact original styling."""
        # Create the main wizard structure
        self._create_wizard_structure()
        
        # Initialize with step 1
        self._show_step(1)
        
        # Display the main container
        display(self.main_container)
    
    def _create_wizard_structure(self):
        """Create the main wizard structure matching the original HTML."""
        # Wizard header
        header_html = """
        <div style='background: #2563eb; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0;'>
            <h1 style='font-size: 24px; margin: 0 0 8px 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>
                Cradle Data Load Configuration
            </h1>
            <p style='margin: 0; opacity: 0.9; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>
                Create your data loading configuration step by step
            </p>
        </div>
        """
        
        # Progress indicator (will be updated dynamically)
        self.progress_container = widgets.HTML()
        
        # Wizard content container
        self.wizard_content = widgets.Output()
        
        # Navigation container
        self.navigation_container = widgets.HTML()
        
        # Main container with exact styling
        container_style = """
        <style>
        .wizard-container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        </style>
        """
        
        # Combine all components
        self.main_container = widgets.VBox([
            widgets.HTML(container_style),
            widgets.HTML(header_html),
            self.progress_container,
            self.wizard_content,
            self.navigation_container
        ], layout=widgets.Layout(
            max_width='1200px',
            margin='0 auto',
            background='white',
            border_radius='8px',
            box_shadow='0 2px 10px rgba(0,0,0,0.1)'
        ))
    
    def _update_progress_indicator(self):
        """Update the progress indicator to match original styling."""
        progress_html = """
        <div style='display: flex; justify-content: space-between; padding: 20px; background: #f8fafc; border-bottom: 1px solid #e2e8f0;'>
        """
        
        steps = [
            {"num": 1, "label": "Data Sources"},
            {"num": 2, "label": "Transform"},
            {"num": 3, "label": "Output"},
            {"num": 4, "label": "Job Config"}
        ]
        
        for i, step in enumerate(steps):
            step_num = step["num"]
            
            # Determine step state
            if step_num < self.current_step:
                circle_bg = "#10b981"
                circle_color = "white"
                label_color = "#10b981"
                label_weight = "600"
                step_class = "completed"
            elif step_num == self.current_step:
                circle_bg = "#2563eb"
                circle_color = "white"
                label_color = "#2563eb"
                label_weight = "600"
                step_class = "active"
            else:
                circle_bg = "#e2e8f0"
                circle_color = "#64748b"
                label_color = "#64748b"
                label_weight = "normal"
                step_class = "inactive"
            
            # Add connector line (except for last step)
            connector = ""
            if i < len(steps) - 1:
                if step_num < self.current_step:
                    connector_color = "#10b981"
                elif step_num == self.current_step:
                    connector_color = "#2563eb"
                else:
                    connector_color = "#e2e8f0"
                
                connector = f"""
                <div style='position: absolute; top: 15px; right: -50%; width: 100%; height: 2px; 
                           background: {connector_color}; z-index: 1;'></div>
                """
            
            progress_html += f"""
            <div style='flex: 1; text-align: center; position: relative;'>
                {connector}
                <div style='width: 30px; height: 30px; border-radius: 50%; background: {circle_bg}; 
                           color: {circle_color}; display: flex; align-items: center; justify-content: center; 
                           margin: 0 auto 8px; font-weight: 600; position: relative; z-index: 2;'>
                    {step_num}
                </div>
                <div style='font-size: 12px; color: {label_color}; font-weight: {label_weight};'>
                    {step["label"]}
                </div>
            </div>
            """
        
        progress_html += "</div>"
        self.progress_container.value = progress_html
    
    def _update_navigation(self):
        """Update navigation buttons to match original styling."""
        # Back button
        back_style = "display: none;" if self.current_step == 1 else "display: block;"
        
        # Next/Finish button
        if self.current_step < self.total_steps:
            action_button = f"""
            <button onclick="window.cradle_widget.next_step()" 
                    style="padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; 
                           font-size: 14px; font-weight: 500; background: #2563eb; color: white;">
                Next
            </button>
            """
        else:
            action_button = f"""
            <button onclick="window.cradle_widget.finish_wizard()" 
                    style="padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; 
                           font-size: 14px; font-weight: 500; background: #2563eb; color: white;">
                Finish
            </button>
            """
        
        navigation_html = f"""
        <div style='display: flex; justify-content: space-between; padding: 20px 30px; 
                    background: #f8fafc; border-top: 1px solid #e2e8f0;'>
            <div style='display: flex; gap: 10px;'>
                <button onclick="window.cradle_widget.previous_step()" 
                        style="padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; 
                               font-size: 14px; font-weight: 500; background: #6b7280; color: white; {back_style}">
                    Back
                </button>
                <button onclick="window.cradle_widget.cancel_wizard()" 
                        style="padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; 
                               font-size: 14px; font-weight: 500; background: #6b7280; color: white;">
                    Cancel
                </button>
            </div>
            <div>
                {action_button}
            </div>
        </div>
        """
        
        self.navigation_container.value = navigation_html
    
    def _show_step(self, step_num: int):
        """Show the specified step with exact original styling."""
        self.current_step = step_num
        
        # Update progress indicator
        self._update_progress_indicator()
        
        # Update navigation
        self._update_navigation()
        
        # Show step content
        with self.wizard_content:
            clear_output(wait=True)
            
            if step_num == 1:
                self._show_step1_data_sources()
            elif step_num == 2:
                self._show_step2_transform()
            elif step_num == 3:
                self._show_step3_output()
            elif step_num == 4:
                self._show_step4_cradle_job()
            elif step_num == 5:
                self._show_step5_completion()
    
    def _show_step1_data_sources(self):
        """Show Step 1: Data Sources configuration with exact original styling."""
        # Create step content matching original HTML structure
        step_html = """
        <div style='padding: 30px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>
            <h2 style='margin: 0 0 10px 0; font-size: 20px; color: #1f2937;'>Step 1: Data Sources Configuration</h2>
            <p style='margin-bottom: 30px; color: #6b7280; font-size: 14px;'>
                Configure the project settings, time range and data sources for your job.
            </p>
            
            <h3 style='margin-bottom: 15px; color: #374151; border-bottom: 1px solid #e2e8f0; padding-bottom: 5px; font-size: 16px;'>
                Project Configuration
            </h3>
        </div>
        """
        
        display(HTML(step_html))
        
        # Create form widgets with exact original styling
        self._create_step1_widgets()
    
    def _create_step1_widgets(self):
        """Create Step 1 form widgets matching original styling."""
        # CRITICAL FIX: Initialize step_widgets[1] BEFORE calling _create_data_source_block()
        # This prevents KeyError: 1 when _create_data_source_block tries to update step_widgets[1]
        self.step_widgets[1] = {}
        
        # Project Configuration Section
        author_widget = widgets.Text(
            value=self.base_config.get('author', 'test-user'),
            placeholder='e.g., john-doe',
            description='Author *:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='400px', margin='5px 0')
        )
        
        bucket_widget = widgets.Text(
            value=self.base_config.get('bucket', 'test-bucket'),
            placeholder='e.g., my-pipeline-bucket',
            description='S3 Bucket *:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='400px', margin='5px 0')
        )
        
        role_widget = widgets.Text(
            value=self.base_config.get('role', 'arn:aws:iam::123456789012:role/test-role'),
            placeholder='arn:aws:iam::123456789012:role/MyRole',
            description='IAM Role *:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='600px', margin='5px 0')
        )
        
        region_widget = widgets.Dropdown(
            options=['NA', 'EU', 'FE'],
            value=self.base_config.get('region', 'NA'),
            description='Pipeline Region *:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='300px', margin='5px 0')
        )
        
        service_name_widget = widgets.Text(
            value=self.base_config.get('service_name', 'test-service'),
            placeholder='e.g., my-service',
            description='Service Name *:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='400px', margin='5px 0')
        )
        
        pipeline_version_widget = widgets.Text(
            value=self.base_config.get('pipeline_version', '1.0.0'),
            placeholder='e.g., 1.0.0',
            description='Pipeline Version *:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='300px', margin='5px 0')
        )
        
        project_root_widget = widgets.Text(
            value=self.base_config.get('project_root_folder', 'test-project'),
            placeholder='e.g., my-project',
            description='Project Root Folder *:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='500px', margin='5px 0')
        )
        
        # Time Range Section
        time_range_html = """
        <h3 style='margin: 30px 0 15px; color: #374151; border-bottom: 1px solid #e2e8f0; 
                   padding-bottom: 5px; font-size: 16px;'>Time Range</h3>
        """
        display(HTML(time_range_html))
        
        start_date_widget = widgets.Text(
            value='2025-01-01T00:00:00',
            placeholder='YYYY-MM-DDTHH:MM:SS',
            description='Start Date *:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='400px', margin='5px 0')
        )
        
        end_date_widget = widgets.Text(
            value='2025-04-17T00:00:00',
            placeholder='YYYY-MM-DDTHH:MM:SS',
            description='End Date *:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='400px', margin='5px 0')
        )
        
        # Store base widgets first
        self.step_widgets[1].update({
            'author': author_widget,
            'bucket': bucket_widget,
            'role': role_widget,
            'region': region_widget,
            'service_name': service_name_widget,
            'pipeline_version': pipeline_version_widget,
            'project_root_folder': project_root_widget,
            'start_date': start_date_widget,
            'end_date': end_date_widget
        })
        
        # Data Sources Section
        data_sources_html = """
        <h3 style='margin: 30px 0 15px; color: #374151; border-bottom: 1px solid #e2e8f0; 
                   padding-bottom: 5px; font-size: 16px;'>Data Sources</h3>
        """
        display(HTML(data_sources_html))
        
        # Now create data source block (this will update step_widgets[1])
        self._create_data_source_block()
        
        # Display widgets in grid layout
        project_row1 = widgets.HBox([author_widget, bucket_widget])
        project_row2 = widgets.HBox([role_widget])
        project_row3 = widgets.HBox([region_widget, service_name_widget])
        project_row4 = widgets.HBox([pipeline_version_widget, project_root_widget])
        
        time_row = widgets.HBox([start_date_widget, end_date_widget])
        
        display(widgets.VBox([project_row1, project_row2, project_row3, project_row4, time_row]))
    
    def _create_data_source_block(self):
        """Create a data source configuration block matching original styling."""
        # Data source block with original styling
        block_html = """
        <div style='border: 1px solid #e2e8f0; border-radius: 6px; padding: 20px; margin-bottom: 20px; 
                    background: #f8fafc;'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;'>
                <h3 style='color: #374151; font-size: 16px; margin: 0;'>Data Source 1</h3>
                <button style='padding: 4px 8px; background: #dc2626; color: white; border: none; 
                               border-radius: 4px; font-size: 12px; cursor: pointer;'>Remove</button>
            </div>
        </div>
        """
        display(HTML(block_html))
        
        # Data source configuration widgets
        source_name_widget = widgets.Text(
            value='RAW_MDS_NA',
            placeholder='e.g., RAW_MDS_NA',
            description='Source Name *:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='400px', margin='5px 0')
        )
        
        source_type_widget = widgets.Dropdown(
            options=['MDS', 'EDX', 'ANDES'],
            value='MDS',
            description='Source Type *:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='300px', margin='5px 0')
        )
        
        # MDS Configuration
        mds_html = """
        <h4 style='margin: 15px 0 10px; color: #374151; font-size: 14px;'>MDS Configuration</h4>
        """
        display(HTML(mds_html))
        
        mds_service_widget = widgets.Text(
            value='AtoZ',
            placeholder='e.g., AtoZ',
            description='Service Name *:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='300px', margin='5px 0')
        )
        
        mds_region_widget = widgets.Dropdown(
            options=['NA', 'EU', 'FE'],
            value='NA',
            description='Region *:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='200px', margin='5px 0')
        )
        
        # Output Schema
        schema_html = """
        <div style='margin: 15px 0;'>
            <label style='display: block; margin-bottom: 5px; font-weight: 500; color: #374151;'>
                Output Schema *
            </label>
            <div style='border: 1px solid #d1d5db; border-radius: 4px; max-height: 150px; overflow-y: auto;'>
                <div style='display: flex; justify-content: space-between; align-items: center; 
                           padding: 8px 12px; border-bottom: 1px solid #e5e7eb;'>
                    <span>objectId</span>
                    <button style='padding: 2px 6px; background: #dc2626; color: white; border: none; 
                                   border-radius: 3px; font-size: 11px; cursor: pointer;'>Remove</button>
                </div>
                <div style='display: flex; justify-content: space-between; align-items: center; 
                           padding: 8px 12px;'>
                    <span>transactionDate</span>
                    <button style='padding: 2px 6px; background: #dc2626; color: white; border: none; 
                                   border-radius: 3px; font-size: 11px; cursor: pointer;'>Remove</button>
                </div>
            </div>
        </div>
        """
        display(HTML(schema_html))
        
        # Add field input
        add_field_widget = widgets.Text(
            placeholder='Field name',
            layout=widgets.Layout(width='300px', margin='5px 0')
        )
        
        add_button = widgets.Button(
            description='Add Field',
            button_style='',
            layout=widgets.Layout(width='100px', margin='5px 0')
        )
        
        # Store data source widgets
        self.step_widgets[1].update({
            'source_name': source_name_widget,
            'source_type': source_type_widget,
            'mds_service': mds_service_widget,
            'mds_region': mds_region_widget,
            'add_field': add_field_widget,
            'add_button': add_button
        })
        
        # Display data source widgets
        source_row = widgets.HBox([source_name_widget, source_type_widget])
        mds_row = widgets.HBox([mds_service_widget, mds_region_widget])
        add_field_row = widgets.HBox([add_field_widget, add_button])
        
        display(widgets.VBox([source_row, mds_row, add_field_row]))
    
    def _show_step2_transform(self):
        """Show Step 2: Transform configuration with exact original styling."""
        step_html = """
        <div style='padding: 30px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>
            <h2 style='margin: 0 0 10px 0; font-size: 20px; color: #1f2937;'>Step 2: Transform Configuration</h2>
            <p style='margin-bottom: 30px; color: #6b7280; font-size: 14px;'>
                Configure the SQL transformation and job splitting options.
            </p>
        </div>
        """
        display(HTML(step_html))
        
        # SQL Transformation
        transform_sql_widget = widgets.Textarea(
            value="""SELECT
  mds.objectId,
  mds.transactionDate,
  edx.is_abuse
FROM mds_source mds
JOIN edx_source edx ON mds.objectId = edx.order_id""",
            placeholder='Enter your SQL transformation query...',
            description='SQL Transformation *:',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='800px', height='200px', margin='10px 0')
        )
        
        # Job Splitting Options
        job_splitting_html = """
        <div style='margin: 30px 0 15px;'>
            <label style='display: flex; align-items: center; gap: 8px; font-weight: 500; color: #374151;'>
                <input type='checkbox' id='enableJobSplitting' style='margin: 0;'>
                Enable Job Splitting
            </label>
        </div>
        """
        display(HTML(job_splitting_html))
        
        enable_splitting_widget = widgets.Checkbox(
            value=False,
            description='Enable Job Splitting',
            layout=widgets.Layout(margin='10px 0')
        )
        
        days_per_split_widget = widgets.IntText(
            value=7,
            description='Days per Split:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='200px', margin='5px 0')
        )
        
        merge_sql_widget = widgets.Textarea(
            value='SELECT * FROM INPUT',
            placeholder='SELECT * FROM INPUT',
            description='Merge SQL *:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='600px', height='80px', margin='5px 0')
        )
        
        # Store widgets
        self.step_widgets[2] = {
            'transform_sql': transform_sql_widget,
            'enable_splitting': enable_splitting_widget,
            'days_per_split': days_per_split_widget,
            'merge_sql': merge_sql_widget
        }
        
        # Display widgets
        splitting_row = widgets.HBox([days_per_split_widget, merge_sql_widget])
        display(widgets.VBox([transform_sql_widget, enable_splitting_widget, splitting_row]))
    
    def _show_step3_output(self):
        """Show Step 3: Output configuration with exact original styling."""
        step_html = """
        <div style='padding: 30px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>
            <h2 style='margin: 0 0 10px 0; font-size: 20px; color: #1f2937;'>Step 3: Output Configuration</h2>
            <p style='margin-bottom: 30px; color: #6b7280; font-size: 14px;'>
                Configure the output schema and format options.
            </p>
        </div>
        """
        display(HTML(step_html))
        
        # Output Schema
        schema_html = """
        <div style='margin-bottom: 20px;'>
            <label style='display: block; margin-bottom: 5px; font-weight: 500; color: #374151;'>
                Output Schema *
            </label>
            <div style='border: 1px solid #d1d5db; border-radius: 4px; max-height: 150px; overflow-y: auto;'>
                <div style='display: flex; justify-content: space-between; align-items: center; 
                           padding: 8px 12px; border-bottom: 1px solid #e5e7eb;'>
                    <span>objectId</span>
                    <button style='padding: 2px 6px; background: #dc2626; color: white; border: none; 
                                   border-radius: 3px; font-size: 11px; cursor: pointer;'>Remove</button>
                </div>
                <div style='display: flex; justify-content: space-between; align-items: center; 
                           padding: 8px 12px; border-bottom: 1px solid #e5e7eb;'>
                    <span>transactionDate</span>
                    <button style='padding: 2px 6px; background: #dc2626; color: white; border: none; 
                                   border-radius: 3px; font-size: 11px; cursor: pointer;'>Remove</button>
                </div>
                <div style='display: flex; justify-content: space-between; align-items: center; 
                           padding: 8px 12px;'>
                    <span>is_abuse</span>
                    <button style='padding: 2px 6px; background: #dc2626; color: white; border: none; 
                                   border-radius: 3px; font-size: 11px; cursor: pointer;'>Remove</button>
                </div>
            </div>
        </div>
        """
        display(HTML(schema_html))
        
        # Output format and options
        output_format_widget = widgets.Dropdown(
            options=['PARQUET', 'CSV', 'JSON', 'ION', 'UNESCAPED_TSV'],
            value='PARQUET',
            description='Output Format:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='300px', margin='5px 0')
        )
        
        save_mode_widget = widgets.Dropdown(
            options=['ERRORIFEXISTS', 'OVERWRITE', 'APPEND', 'IGNORE'],
            value='ERRORIFEXISTS',
            description='Save Mode:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='300px', margin='5px 0')
        )
        
        file_count_widget = widgets.IntText(
            value=0,
            description='Output File Count:',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='250px', margin='5px 0')
        )
        
        keep_dots_widget = widgets.Checkbox(
            value=False,
            description='Keep dots in output schema',
            layout=widgets.Layout(margin='10px 0')
        )
        
        include_header_widget = widgets.Checkbox(
            value=True,
            description='Include header in S3 output',
            layout=widgets.Layout(margin='10px 0')
        )
        
        # Store widgets
        self.step_widgets[3] = {
            'output_format': output_format_widget,
            'save_mode': save_mode_widget,
            'file_count': file_count_widget,
            'keep_dots': keep_dots_widget,
            'include_header': include_header_widget
        }
        
        # Display widgets
        format_row = widgets.HBox([output_format_widget, save_mode_widget])
        options_row = widgets.VBox([file_count_widget, keep_dots_widget, include_header_widget])
        display(widgets.VBox([format_row, options_row]))
    
    def _show_step4_cradle_job(self):
        """Show Step 4: Cradle Job configuration with exact original styling."""
        step_html = """
        <div style='padding: 30px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>
            <h2 style='margin: 0 0 10px 0; font-size: 20px; color: #1f2937;'>Step 4: Cradle Job Configuration</h2>
            <p style='margin-bottom: 30px; color: #6b7280; font-size: 14px;'>
                Configure the cluster and job execution settings.
            </p>
        </div>
        """
        display(HTML(step_html))
        
        # Cradle job settings
        cradle_account_widget = widgets.Text(
            value='Buyer-Abuse-RnD-Dev',
            placeholder='e.g., Buyer-Abuse-RnD-Dev',
            description='Cradle Account *:',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='500px', margin='5px 0')
        )
        
        cluster_type_widget = widgets.Dropdown(
            options=['STANDARD', 'SMALL', 'MEDIUM', 'LARGE'],
            value='STANDARD',
            description='Cluster Type:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='300px', margin='5px 0')
        )
        
        retry_count_widget = widgets.IntText(
            value=1,
            description='Job Retry Count:',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='250px', margin='5px 0')
        )
        
        extra_args_widget = widgets.Textarea(
            value='',
            placeholder='Additional Spark driver options',
            description='Extra Spark Arguments:',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='600px', height='80px', margin='10px 0')
        )
        
        # Store widgets
        self.step_widgets[4] = {
            'cradle_account': cradle_account_widget,
            'cluster_type': cluster_type_widget,
            'retry_count': retry_count_widget,
            'extra_args': extra_args_widget
        }
        
        # Display widgets
        settings_row = widgets.HBox([cluster_type_widget, retry_count_widget])
        display(widgets.VBox([cradle_account_widget, settings_row, extra_args_widget]))
    
    def _show_step5_completion(self):
        """Show Step 5: Completion with job type selection and summary."""
        step_html = """
        <div style='padding: 30px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>
            <h2 style='margin: 0 0 10px 0; font-size: 20px; color: #1f2937;'>Complete Configuration</h2>
            <p style='margin-bottom: 30px; color: #6b7280; font-size: 14px;'>
                Review your configuration and select the job type.
            </p>
        </div>
        """
        display(HTML(step_html))
        
        # Job Type Selection
        job_type_widget = widgets.RadioButtons(
            options=['training', 'validation', 'testing', 'calibration'],
            value='training',
            description='Job Type *:',
            style={'description_width': '120px'},
            layout=widgets.Layout(margin='10px 0')
        )
        
        # Configuration Summary
        summary_html = """
        <div style='background: #f0f9ff; border: 1px solid #0ea5e9; border-radius: 6px; 
                    padding: 20px; margin: 20px 0;'>
            <h3 style='color: #0c4a6e; margin: 0 0 15px 0;'>Configuration Summary</h3>
            <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                <span style='font-weight: 500; color: #374151;'>Data Sources:</span>
                <span style='color: #6b7280;'>1 configured</span>
            </div>
            <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                <span style='font-weight: 500; color: #374151;'>Time Range:</span>
                <span style='color: #6b7280;'>2025-01-01 to 2025-04-17</span>
            </div>
            <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                <span style='font-weight: 500; color: #374151;'>Transform:</span>
                <span style='color: #6b7280;'>Custom SQL provided</span>
            </div>
            <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                <span style='font-weight: 500; color: #374151;'>Output:</span>
                <span style='color: #6b7280;'>PARQUET format, 3 fields</span>
            </div>
            <div style='display: flex; justify-content: space-between;'>
                <span style='font-weight: 500; color: #374151;'>Cluster:</span>
                <span style='color: #6b7280;'>STANDARD</span>
            </div>
        </div>
        """
        display(HTML(summary_html))
        
        # Store widgets
        self.step_widgets[5] = {
            'job_type': job_type_widget
        }
        
        display(job_type_widget)
    
    def next_step(self):
        """Handle next step navigation."""
        if self.current_step < self.total_steps:
            self._save_current_step_data()
            self._show_step(self.current_step + 1)
    
    def previous_step(self):
        """Handle previous step navigation."""
        if self.current_step > 1:
            self._save_current_step_data()
            self._show_step(self.current_step - 1)
    
    def cancel_wizard(self):
        """Handle wizard cancellation."""
        with self.wizard_content:
            clear_output(wait=True)
            cancel_html = """
            <div style='padding: 50px; text-align: center;'>
                <h2 style='color: #dc2626;'>Configuration Cancelled</h2>
                <p>The configuration wizard has been cancelled.</p>
            </div>
            """
            display(HTML(cancel_html))
    
    def finish_wizard(self):
        """Handle wizard completion."""
        self._save_current_step_data()
        self._create_final_config()
        
        if self.completed_config:
            with self.wizard_content:
                clear_output(wait=True)
                success_html = """
                <div style='padding: 50px; text-align: center;'>
                    <div style='background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                                color: white; padding: 20px; border-radius: 12px; 
                                box-shadow: 0 4px 20px rgba(16, 185, 129, 0.3);'>
                        <h2 style='margin: 0 0 10px 0;'>🎉 Configuration Complete!</h2>
                        <p style='margin: 0; font-size: 14px; opacity: 0.9;'>
                            Your cradle data loading configuration has been successfully created.
                        </p>
                    </div>
                </div>
                """
                display(HTML(success_html))
            
            # Call completion callback if in embedded mode
            if self.embedded_mode and self.completion_callback:
                self.completion_callback(self.completed_config)
    
    def _save_current_step_data(self):
        """Save current step data from widgets."""
        if self.current_step in self.step_widgets:
            step_data = {}
            for key, widget in self.step_widgets[self.current_step].items():
                step_data[key] = widget.value
            self.steps_data[self.current_step] = step_data
    
    def _create_final_config(self):
        """Create the final CradleDataLoadingConfig from collected data."""
        try:
            # Build UI data structure matching ValidationService expectations
            ui_data = {
                # Base config from step 1
                'author': self.steps_data.get(1, {}).get('author', 'sagemaker-user'),
                'bucket': self.steps_data.get(1, {}).get('bucket', 'my-sagemaker-bucket'),
                'role': self.steps_data.get(1, {}).get('role', 'arn:aws:iam::123456789012:role/SageMakerRole'),
                'region': self.steps_data.get(1, {}).get('region', 'us-east-1'),
                'service_name': self.steps_data.get(1, {}).get('service_name', 'cursus-pipeline'),
                'pipeline_version': self.steps_data.get(1, {}).get('pipeline_version', '1.0.0'),
                'project_root_folder': self.steps_data.get(1, {}).get('project_root_folder', '/opt/ml/code'),
                'job_type': self.steps_data.get(5, {}).get('job_type', 'training'),
                
                # Step data
                'data_sources_spec': {
                    'start_date': self.steps_data.get(1, {}).get('start_date', '2025-01-01T00:00:00'),
                    'end_date': self.steps_data.get(1, {}).get('end_date', '2025-04-17T00:00:00'),
                    'data_sources': [{
                        'data_source_name': self.steps_data.get(1, {}).get('source_name', 'RAW_MDS_NA'),
                        'data_source_type': self.steps_data.get(1, {}).get('source_type', 'MDS'),
                        'mds_data_source_properties': {
                            'mds_table_name': self.steps_data.get(1, {}).get('mds_service', 'AtoZ'),
                            'mds_database_name': self.steps_data.get(1, {}).get('mds_region', 'NA')
                        }
                    }]
                },
                'transform_spec': {
                    'transform_sql': self.steps_data.get(2, {}).get('transform_sql', 'SELECT * FROM input_data'),
                    'job_split_options': {
                        'split_job': self.steps_data.get(2, {}).get('enable_splitting', False),
                        'days_per_split': self.steps_data.get(2, {}).get('days_per_split', 7),
                        'merge_sql': self.steps_data.get(2, {}).get('merge_sql', 'SELECT * FROM INPUT')
                    }
                },
                'output_spec': {
                    'output_schema': ['objectId', 'transactionDate', 'is_abuse'],
                    'output_format': self.steps_data.get(3, {}).get('output_format', 'PARQUET'),
                    'output_save_mode': self.steps_data.get(3, {}).get('save_mode', 'ERRORIFEXISTS'),
                    'output_file_count': self.steps_data.get(3, {}).get('file_count', 0),
                    'keep_dot_in_output_schema': self.steps_data.get(3, {}).get('keep_dots', False),
                    'include_header_in_s3_output': self.steps_data.get(3, {}).get('include_header', True)
                },
                'cradle_job_spec': {
                    'cradle_account': self.steps_data.get(4, {}).get('cradle_account', 'Buyer-Abuse-RnD-Dev'),
                    'cluster_type': self.steps_data.get(4, {}).get('cluster_type', 'STANDARD'),
                    'job_retry_count': self.steps_data.get(4, {}).get('retry_count', 1),
                    'extra_spark_job_arguments': self.steps_data.get(4, {}).get('extra_args', '')
                }
            }
            
            # Use validation service to build config if available
            if self.validation_service:
                self.completed_config = self.validation_service.build_final_config(ui_data)
            else:
                # Fallback: create basic config
                self.completed_config = CradleDataLoadingConfig(**ui_data)
                
        except Exception as e:
            logger.error(f"Error creating final config: {e}")
            self.completed_config = None
    
    def get_config(self) -> Optional[CradleDataLoadingConfig]:
        """Get the completed configuration."""
        return self.completed_config
    
    def set_navigation_callback(self, callback: Callable):
        """Set callback for navigation control (for embedded mode)."""
        # This method allows the parent wizard to control navigation
        # Implementation would depend on specific navigation control needs
        pass
