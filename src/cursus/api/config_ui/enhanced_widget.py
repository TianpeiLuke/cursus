"""
Enhanced Pipeline Configuration Widget

Single enhanced entry point that leverages 100% of existing infrastructure
to provide the complete enhanced UX for SageMaker native environments.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display, HTML, Javascript
import json
import re

from ...core.base.config_base import BasePipelineConfig
from ...steps.configs.config_processing_step_base import ProcessingStepConfigBase
from .core.core import UniversalConfigCore
# DAGConfigurationManager functionality replaced by direct factory imports
from ..factory import DAGConfigFactory, ConfigClassMapper

logger = logging.getLogger(__name__)


class EnhancedPipelineConfigWidget:
    """
    Enhanced Pipeline Configuration Widget - Single Enhanced Entry Point
    
    Leverages 100% of existing infrastructure to provide complete enhanced UX:
    - DAG-driven configuration discovery (existing)
    - Multi-step wizard with professional UX (existing) 
    - 3-tier field categorization (existing)
    - Specialized component integration (existing)
    - Progress tracking and workflow management (existing)
    - Save All Merged functionality (existing)
    
    This class is primarily a convenience wrapper with SageMaker optimizations.
    """
    
    def __init__(self, workspace_dirs: Optional[List[Union[str, Path]]] = None):
        """
        Initialize enhanced widget with existing infrastructure.
        
        Args:
            workspace_dirs: Optional workspace directories for step catalog
        """
        # Direct reuse of existing core infrastructure (100% reuse)
        self.core = UniversalConfigCore(workspace_dirs=workspace_dirs)
        self.dag_manager = DAGConfigurationManager(self.core)
        
        # SageMaker-specific enhancements
        self.sagemaker_optimizations = SageMakerOptimizations()
        
        logger.info("EnhancedPipelineConfigWidget initialized with existing infrastructure")
    
    def create_dag_driven_wizard(self, 
                                pipeline_dag: Any, 
                                base_config: BasePipelineConfig,
                                processing_config: Optional[ProcessingStepConfigBase] = None,
                                **kwargs) -> 'EnhancedMultiStepWizard':
        """
        Create DAG-driven wizard using existing infrastructure (95% reuse).
        
        Args:
            pipeline_dag: Pipeline DAG definition
            base_config: Base pipeline configuration
            processing_config: Optional processing configuration
            **kwargs: Additional arguments
            
        Returns:
            Enhanced MultiStepWizard with SageMaker optimizations
        """
        logger.info("Creating DAG-driven wizard using existing infrastructure")
        
        # Use existing create_dag_driven_widget method (100% reuse)
        wizard = self.dag_manager.create_dag_driven_widget(
            pipeline_dag=pipeline_dag,
            base_config=base_config,
            processing_config=processing_config
        )
        
        # Wrap with enhanced functionality (5% new code)
        enhanced_wizard = EnhancedMultiStepWizard(wizard, self.sagemaker_optimizations)
        
        logger.info(f"Enhanced wizard created with {len(wizard.steps)} steps")
        return enhanced_wizard
    
    def analyze_pipeline_dag(self, pipeline_dag: Any) -> Dict[str, Any]:
        """
        Analyze pipeline DAG using existing infrastructure (100% reuse).
        
        Args:
            pipeline_dag: Pipeline DAG to analyze
            
        Returns:
            DAG analysis results with enhanced summary
        """
        # Use existing analyze_pipeline_dag method (100% reuse)
        analysis_result = self.dag_manager.analyze_pipeline_dag(pipeline_dag)
        
        # Add enhanced summary for better UX
        analysis_result["enhanced_summary"] = self._create_enhanced_summary(analysis_result)
        
        return analysis_result
    
    def _create_enhanced_summary(self, analysis_result: Dict[str, Any]) -> str:
        """Create enhanced human-readable summary of DAG analysis."""
        summary = analysis_result.get("analysis_summary", {})
        
        return f"""
🎯 Enhanced Pipeline Analysis Results:

📊 Pipeline Overview:
   • Discovered Steps: {summary.get('total_dag_nodes', 0)} pipeline steps
   • Required Configurations: {summary.get('required_configs', 0)} (only these will be shown)
   • Workflow Steps: {summary.get('workflow_steps', 0)} total configuration steps
   • Specialized Configurations: {summary.get('specialized_configs', 0)} with custom interfaces

🔍 Configuration Workflow:
   Step 1: Base Configuration (Essential user inputs)
   Step 2: Processing Configuration (System-level settings)
   Steps 3+: {summary.get('required_configs', 0)} Step-specific configurations

✨ Enhanced Features:
   • 3-tier field categorization (Essential/System/Hidden)
   • Specialized component integration (Cradle UI, Hyperparameters)
   • Progress tracking with visual indicators
   • Save All Merged functionality for demo_config.ipynb compatibility

❌ Hidden: {summary.get('hidden_configs', 0)} other config types not needed for this pipeline
        """.strip()


class EnhancedMultiStepWizard:
    """
    Enhanced wrapper around existing MultiStepWizard with SageMaker optimizations.
    
    Provides 100% of existing functionality plus SageMaker-specific enhancements.
    """
    
    def __init__(self, base_wizard, sagemaker_optimizations):
        """
        Initialize enhanced wizard wrapper.
        
        Args:
            base_wizard: Existing MultiStepWizard instance
            sagemaker_optimizations: SageMaker optimization utilities
        """
        self.base_wizard = base_wizard
        self.sagemaker_opts = sagemaker_optimizations
        
        # Expose all base wizard attributes and methods
        self.steps = base_wizard.steps
        self.completed_configs = base_wizard.completed_configs
        self.current_step = base_wizard.current_step
        
        logger.info("EnhancedMultiStepWizard wrapper initialized")
    
    def _find_base_config(self):
        """Delegate to base wizard's _find_base_config method."""
        return self.base_wizard._find_base_config()
    
    def _get_step_fields(self, step):
        """Delegate to base wizard's _get_step_fields method."""
        return self.base_wizard._get_step_fields(step)
    
    def _get_step_values(self, step):
        """Delegate to base wizard's _get_step_values method."""
        return self.base_wizard._get_step_values(step)
    
    def _update_navigation_and_step(self):
        """Update navigation and step display for enhanced wizard."""
        # Force state synchronization first
        self._force_sync_state()
        
        # Update the current step display in the base wizard's output
        try:
            with self.base_wizard.output:
                from IPython.display import clear_output
                clear_output(wait=True)
                self.base_wizard._display_current_step()
            logger.debug("Enhanced wizard step display updated successfully")
        except Exception as e:
            logger.error(f"Error updating enhanced wizard step display: {e}")
    
    def display(self):
        """ENHANCED SOLUTION: Display with proper button handler override."""
        # Check if already displayed to prevent duplication
        if hasattr(self, '_display_called') and self._display_called:
            logger.debug("Display already called, skipping to prevent duplication")
            return
        
        # Apply SageMaker clipboard optimizations (silent)
        self.sagemaker_opts.enhance_clipboard_support()
        
        # Mark as displayed to prevent future duplications
        self._display_called = True
        
        # ENHANCED SOLUTION: Display base wizard then override button handlers
        try:
            logger.debug("About to call base wizard display()...")
            self.base_wizard.display()
            logger.debug("Base wizard display() called successfully")
            
            # CRITICAL FIX: Override button handlers to use enhanced wizard methods
            logger.debug("About to call _override_button_handlers()...")
            try:
                self._override_button_handlers()
                logger.debug("Button handlers overridden to use enhanced wizard methods")
            except Exception as e:
                logger.error(f"Failed to override button handlers: {e}")
                import traceback
                traceback.print_exc()
            
        except Exception as e:
            logger.error(f"Error displaying enhanced wizard: {e}")
            # Reset flag on error so user can try again
            self._display_called = False
            # Fallback: Try to display components individually
            try:
                display(self.base_wizard.navigation_output)
                display(self.base_wizard.output)
                logger.debug("Fallback display successful")
                self._display_called = True  # Mark as successful
            except Exception as e2:
                logger.error(f"Fallback display also failed: {e2}")
                self._display_called = False  # Reset on complete failure
    
    def _override_button_handlers(self):
        """CRITICAL FIX: Override button handlers to use enhanced wizard methods."""
        try:
            # Find the navigation buttons in the main container
            main_container = getattr(self.base_wizard, '_main_container', None)
            if not main_container:
                logger.warning("No _main_container found, searching for buttons in navigation widgets")
                # Try to find buttons in navigation widgets
                self._find_and_override_buttons_in_widgets()
                return
            
            # Search for buttons in the main container
            buttons_found = self._find_and_override_buttons_recursive(main_container)
            
            if buttons_found:
                logger.debug(f"Successfully overridden {buttons_found} button handlers")
            else:
                logger.warning("No buttons found to override")
                
        except Exception as e:
            logger.error(f"Error overriding button handlers: {e}")
    
    def _find_and_override_buttons_in_widgets(self):
        """Find and override buttons in navigation widgets."""
        try:
            # Check if base wizard has navigation widgets created by display()
            if hasattr(self.base_wizard, 'navigation_widgets'):
                self._find_and_override_buttons_recursive(self.base_wizard.navigation_widgets)
                return
            
            # Fallback: Look for button attributes directly
            for attr_name in ['next_button', 'prev_button', 'finish_button']:
                if hasattr(self.base_wizard, attr_name):
                    button = getattr(self.base_wizard, attr_name)
                    if button and hasattr(button, '_click_handlers'):
                        self._override_single_button(button, attr_name)
                        
        except Exception as e:
            logger.error(f"Error finding buttons in widgets: {e}")
    
    def _find_and_override_buttons_recursive(self, widget):
        """Recursively find and override button handlers in widget tree."""
        buttons_found = 0
        
        try:
            # Check if this widget is a button
            if hasattr(widget, 'description') and hasattr(widget, '_click_handlers'):
                description = getattr(widget, 'description', '')
                
                if 'Next' in description:
                    self._override_single_button(widget, 'next')
                    buttons_found += 1
                elif 'Previous' in description:
                    self._override_single_button(widget, 'prev')
                    buttons_found += 1
                elif 'Complete Workflow' in description:
                    self._override_single_button(widget, 'finish')
                    buttons_found += 1
            
            # Recursively search children
            if hasattr(widget, 'children'):
                for child in widget.children:
                    buttons_found += self._find_and_override_buttons_recursive(child)
                    
        except Exception as e:
            logger.debug(f"Error in recursive button search: {e}")
        
        return buttons_found
    
    def _override_single_button(self, button, button_type):
        """Override a single button's click handler."""
        try:
            # Clear existing handlers
            button._click_handlers.callbacks.clear()
            
            # Add enhanced wizard handler based on button type
            if button_type == 'next':
                button.on_click(self._on_next_clicked)
                logger.debug("Next button handler overridden to use enhanced wizard method")
            elif button_type == 'prev':
                button.on_click(self._on_prev_clicked)
                logger.debug("Previous button handler overridden to use enhanced wizard method")
            elif button_type == 'finish':
                button.on_click(self._on_finish_clicked)
                logger.debug("Finish button handler overridden to use enhanced wizard method")
                
        except Exception as e:
            logger.error(f"Error overriding {button_type} button handler: {e}")
    
    # Ensure all navigation methods are properly delegated
    def _display_navigation(self):
        """Delegate navigation display to base wizard."""
        return self.base_wizard._display_navigation()
    
    def _display_current_step(self):
        """Delegate current step display to base wizard."""
        return self.base_wizard._display_current_step()
    
    def _on_prev_clicked(self, button):
        """Delegate previous button click to base wizard with state sync."""
        result = self.base_wizard._on_prev_clicked(button)
        # Force state synchronization after navigation
        self._force_sync_state()
        return result
    
    def _on_next_clicked(self, button):
        """Delegate next button click to base wizard with detailed logging and state sync."""
        logger.info(f"🔘 ENHANCED: Next button clicked - Enhanced step: {self.current_step}, Base step: {self.base_wizard.current_step}")
        
        # Get current step info for logging
        if self.current_step < len(self.steps):
            current_step_info = self.steps[self.current_step]
            logger.info(f"🔘 ENHANCED: Current step details: {current_step_info['title']} ({current_step_info['config_class_name']})")
        
        # Log the delegation
        logger.info(f"🔘 ENHANCED: Delegating to base wizard _on_next_clicked...")
        
        # Delegate to base wizard
        result = self.base_wizard._on_next_clicked(button)
        
        logger.info(f"🔘 ENHANCED: Base wizard returned: {result}")
        logger.info(f"🔘 ENHANCED: After delegation - Enhanced step: {self.current_step}, Base step: {self.base_wizard.current_step}")
        
        # Force state synchronization after navigation
        logger.info(f"🔘 ENHANCED: Calling _force_sync_state()...")
        self._force_sync_state()
        
        logger.info(f"🔘 ENHANCED: After sync - Enhanced step: {self.current_step}, Base step: {self.base_wizard.current_step}")
        
        return result
    
    def _on_finish_clicked(self, button):
        """Delegate finish button click to base wizard with state sync."""
        result = self.base_wizard._on_finish_clicked(button)
        # Force state synchronization after navigation
        self._force_sync_state()
        return result
    
    def _force_sync_state(self):
        """Force synchronization of all state from base wizard."""
        # Sync all state attributes
        self.current_step = self.base_wizard.current_step
        self.completed_configs = self.base_wizard.completed_configs
        self.steps = self.base_wizard.steps
        
        # Also sync any other attributes that might have changed
        if hasattr(self.base_wizard, 'step_widgets'):
            self.step_widgets = self.base_wizard.step_widgets
        
        logger.debug(f"Force state sync: current_step={self.current_step}, base_step={self.base_wizard.current_step}")
    
    def _display_enhanced_welcome(self):
        """Display enhanced welcome message for SageMaker users."""
        welcome_html = """
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 20px; border-radius: 12px; margin-bottom: 20px;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);'>
            <h2 style='margin: 0 0 10px 0; display: flex; align-items: center;'>
                🚀 Enhanced Pipeline Configuration Wizard
                <span style='margin-left: auto; font-size: 14px; opacity: 0.8;'>SageMaker Native</span>
            </h2>
            <p style='margin: 0; opacity: 0.9; font-size: 14px;'>
                Complete DAG-driven configuration with 3-tier field categorization, 
                specialized components, and Save All Merged functionality.
            </p>
        </div>
        """
        display(HTML(welcome_html))
    
    def _display_sagemaker_help(self):
        """Display SageMaker-specific help and tips."""
        help_html = """
        <div style='background: #f0f9ff; border: 1px solid #0ea5e9; border-radius: 8px; 
                    padding: 15px; margin: 15px 0;'>
            <h4 style='margin: 0 0 10px 0; color: #0c4a6e;'>💡 SageMaker Tips:</h4>
            <ul style='margin: 0; color: #0c4a6e; font-size: 13px; line-height: 1.6;'>
                <li><strong>Clipboard:</strong> Enhanced copy/paste support for SageMaker environment</li>
                <li><strong>Offline Mode:</strong> All functionality works without network dependencies</li>
                <li><strong>File Saving:</strong> Configurations save directly to SageMaker filesystem</li>
                <li><strong>Integration:</strong> Perfect compatibility with demo_config.ipynb workflow</li>
            </ul>
        </div>
        """
        display(HTML(help_html))
    
    
    def get_completed_configs(self) -> List[BasePipelineConfig]:
        """
        Get completed configurations using existing method (100% reuse).
        
        Returns:
            List of configuration instances in demo_config.ipynb order
        """
        return self.base_wizard.get_completed_configs()
    
    def save_all_merged(self, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Enhanced Save All Merged functionality with smart filename generation.
        
        Args:
            filename: Optional custom filename
            
        Returns:
            Merge results with enhanced metadata
        """
        # Generate smart filename if not provided
        if not filename:
            filename = self.sagemaker_opts.generate_smart_filename(self.completed_configs)
        
        # Use existing merge_and_save_configs functionality
        try:
            from ...steps.configs import merge_and_save_configs
            
            config_list = self.get_completed_configs()
            merged_config_result = merge_and_save_configs(
                config_list=config_list,
                output_file=filename
            )
            
            # Enhanced result with metadata
            # The function returns a dict, so we need to handle the filename properly
            from pathlib import Path
            file_path = Path(filename)
            
            result = {
                "success": True,
                "filename": file_path.name,
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
                "config_count": len(config_list),
                "sagemaker_optimized": True
            }
            
            # Display enhanced success message
            self._display_merge_success(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Save All Merged failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _display_merge_success(self, result: Dict[str, Any]):
        """Display enhanced merge success message."""
        success_html = f"""
        <div style='background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); 
                    border: 2px solid #10b981; border-radius: 12px; padding: 20px; margin: 20px 0;'>
            <h3 style='margin: 0 0 15px 0; color: #065f46; display: flex; align-items: center;'>
                ✅ Save All Merged - Configuration Export Complete
            </h3>
            
            <div style='background: white; border-radius: 8px; padding: 15px; margin-bottom: 15px;'>
                <h4 style='margin: 0 0 10px 0; color: #065f46;'>📁 Generated File:</h4>
                <div style='font-family: monospace; background: #f3f4f6; padding: 8px; border-radius: 4px;'>
                    📄 {result['filename']}
                </div>
                <div style='margin-top: 8px; color: #6b7280; font-size: 0.9em;'>
                    📊 {result['config_count']} configurations merged • 
                    💾 {self._format_file_size(result['file_size'])} • 
                    🚀 SageMaker optimized
                </div>
            </div>
            
            <div style='text-align: center;'>
                <p style='margin: 0; color: #065f46; font-weight: 600;'>
                    ✨ Ready for use with demo_config.ipynb workflow patterns!
                </p>
            </div>
        </div>
        """
        display(HTML(success_html))
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
    


class SageMakerOptimizations:
    """SageMaker-specific optimizations and enhancements."""
    
    def enhance_clipboard_support(self):
        """Enhanced clipboard support - completely silent (no display calls at all)."""
        # FIXED: Make clipboard support completely silent to avoid any display calls
        # Skip clipboard support entirely to eliminate duplicate displays
        # Users can still copy/paste normally, just without enhanced feedback
        logger.debug("Clipboard support skipped to prevent duplicate displays")
        pass
    
    def generate_smart_filename(self, completed_configs: Dict[str, Any]) -> str:
        """
        Generate smart filename based on configuration data.
        
        Args:
            completed_configs: Dictionary of completed configurations
            
        Returns:
            Smart filename in format: config_{service_name}_{region}.json
        """
        # Extract service name and region from base config
        service_name = "pipeline"
        region = "us-east-1"
        
        # Try to extract from completed configs
        for config_name, config_instance in completed_configs.items():
            if hasattr(config_instance, 'service_name') and config_instance.service_name:
                service_name = config_instance.service_name
            if hasattr(config_instance, 'region') and config_instance.region:
                region = config_instance.region
            
            # Break after finding base config values
            if service_name != "pipeline" and region != "us-east-1":
                break
        
        # Sanitize for filename safety
        safe_service = re.sub(r'[^\w\-_]', '_', str(service_name))
        safe_region = re.sub(r'[^\w\-_]', '_', str(region))
        
        return f"config_{safe_service}_{safe_region}.json"
    
    def create_enhanced_file_save_dialog(self, default_filename: str) -> widgets.Widget:
        """
        Create enhanced file save dialog with smart defaults.
        
        Args:
            default_filename: Default filename to use
            
        Returns:
            File save dialog widget
        """
        filename_input = widgets.Text(
            value=default_filename,
            description="📄 Filename:",
            style={'description_width': '120px'},
            layout=widgets.Layout(width='400px')
        )
        
        location_dropdown = widgets.Dropdown(
            options=[
                ('📂 Current Directory (Jupyter notebook location)', 'current'),
                ('⬇️ Downloads Folder', 'downloads'), 
                ('📁 Custom Location (browser default)', 'custom')
            ],
            value='current',
            description="📁 Save Location:",
            style={'description_width': '120px'},
            layout=widgets.Layout(width='400px')
        )
        
        preview_html = HTML(
            value=f"<div style='background: #e3f2fd; padding: 10px; border-radius: 4px; margin-top: 10px;'>"
                  f"<strong>� Save Preview:</strong> Will save as <strong>{default_filename}</strong> in current directory</div>"
        )
        
        # Update preview when inputs change
        def update_preview(change):
            location_text = {
                'current': 'current directory',
                'downloads': 'downloads folder',
                'custom': 'browser default location'
            }.get(location_dropdown.value, 'current directory')
            
            preview_html.value = (
                f"<div style='background: #e3f2fd; padding: 10px; border-radius: 4px; margin-top: 10px;'>"
                f"<strong>💡 Save Preview:</strong> Will save as <strong>{filename_input.value}</strong> in {location_text}</div>"
            )
        
        filename_input.observe(update_preview, names='value')
        location_dropdown.observe(update_preview, names='value')
        
        return widgets.VBox([
            widgets.HTML("<h4>💾 Enhanced File Save Options</h4>"),
            filename_input,
            location_dropdown,
            preview_html
        ])


# Factory Functions - Main Entry Points

def create_enhanced_pipeline_widget(pipeline_dag: Any, 
                                   base_config: BasePipelineConfig,
                                   processing_config: Optional[ProcessingStepConfigBase] = None,
                                   workspace_dirs: Optional[List[Union[str, Path]]] = None,
                                   **kwargs) -> EnhancedMultiStepWizard:
    """
    Factory function that creates enhanced pipeline widget.
    
    This is the main entry point for users wanting the complete enhanced UX.
    
    Args:
        pipeline_dag: Pipeline DAG definition
        base_config: Base pipeline configuration
        processing_config: Optional processing configuration
        workspace_dirs: Optional workspace directories for step catalog
        **kwargs: Additional arguments
        
    Returns:
        Enhanced MultiStepWizard with complete UX
        
    Example:
        >>> from cursus.api.config_ui.enhanced_widget import create_enhanced_pipeline_widget
        >>> from cursus.pipeline_catalog.shared_dags import create_xgboost_complete_e2e_dag
        >>> 
        >>> # Create base config
        >>> base_config = BasePipelineConfig(
        ...     author="user",
        ...     bucket="my-bucket",
        ...     role="arn:aws:iam::123456789012:role/SageMakerRole",
        ...     region="us-east-1"
        ... )
        >>> 
        >>> # Create DAG
        >>> dag = create_xgboost_complete_e2e_dag()
        >>> 
        >>> # Create enhanced widget
        >>> wizard = create_enhanced_pipeline_widget(dag, base_config)
        >>> wizard.display()  # Shows complete multi-step wizard
        >>> 
        >>> # Get results
        >>> config_list = wizard.get_completed_configs()
        >>> merge_result = wizard.save_all_merged()
    """
    enhanced_widget = EnhancedPipelineConfigWidget(workspace_dirs=workspace_dirs)
    return enhanced_widget.create_dag_driven_wizard(
        pipeline_dag=pipeline_dag,
        base_config=base_config,
        processing_config=processing_config,
        **kwargs
    )


def analyze_enhanced_pipeline_dag(pipeline_dag: Any, 
                                 workspace_dirs: Optional[List[Union[str, Path]]] = None) -> Dict[str, Any]:
    """
    Factory function to analyze pipeline DAG with enhanced summary.
    
    Args:
        pipeline_dag: Pipeline DAG to analyze
        workspace_dirs: Optional workspace directories for step catalog
        
    Returns:
        Enhanced DAG analysis results
        
    Example:
        >>> from cursus.api.config_ui.enhanced_widget import analyze_enhanced_pipeline_dag
        >>> 
        >>> analysis = analyze_enhanced_pipeline_dag(my_dag)
        >>> print(analysis["enhanced_summary"])
    """
    enhanced_widget = EnhancedPipelineConfigWidget(workspace_dirs=workspace_dirs)
    return enhanced_widget.analyze_pipeline_dag(pipeline_dag)


# Convenience function for direct usage of existing infrastructure
def create_pipeline_config_widget_direct(pipeline_dag: Any, 
                                        base_config: BasePipelineConfig,
                                        processing_config: Optional[ProcessingStepConfigBase] = None,
                                        workspace_dirs: Optional[List[Union[str, Path]]] = None):
    """
    Direct usage of existing infrastructure without wrapper (100% existing code).
    
    This function demonstrates that users can get the complete enhanced UX
    using existing infrastructure with zero new code.
    
    Args:
        pipeline_dag: Pipeline DAG definition
        base_config: Base pipeline configuration
        processing_config: Optional processing configuration
        workspace_dirs: Optional workspace directories
        
    Returns:
        Existing MultiStepWizard with complete functionality
        
    Example:
        >>> # This provides the same UX as enhanced widget using existing code
        >>> wizard = create_pipeline_config_widget_direct(dag, base_config)
        >>> wizard.display()  # Complete multi-step wizard
        >>> config_list = wizard.get_completed_configs()  # demo_config.ipynb order
    """
    from .core.dag_manager import create_pipeline_config_widget
    
    return create_pipeline_config_widget(
        pipeline_dag=pipeline_dag,
        base_config=base_config,
        processing_config=processing_config,
        workspace_dirs=workspace_dirs
    )
