"""
DAG Configuration Manager

Manages PipelineDAG-driven configuration discovery and UI generation.
Provides intelligent analysis of pipeline DAGs to determine required configurations
and create appropriate workflow structures.
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union
from pathlib import Path

# Handle both relative and absolute imports using centralized path setup
try:
    # Try relative imports first (when run as module)
    from ....core.base.config_base import BasePipelineConfig
    from ....steps.configs.config_processing_step_base import ProcessingStepConfigBase
except ImportError:
    # Fallback: Set up cursus path and use absolute imports
    from .import_utils import ensure_cursus_path
    ensure_cursus_path()
    
    from cursus.core.base.config_base import BasePipelineConfig
    from cursus.steps.configs.config_processing_step_base import ProcessingStepConfigBase

logger = logging.getLogger(__name__)


class DAGConfigurationManager:
    """Manages PipelineDAG-driven configuration discovery and UI generation."""
    
    def __init__(self, universal_config_core):
        """
        Initialize DAG configuration manager.
        
        Args:
            universal_config_core: UniversalConfigCore instance for config operations
        """
        self.core = universal_config_core
        self.step_catalog = universal_config_core.step_catalog
        self.config_resolver = None
        
        # Try to initialize config resolver
        try:
            from ....step_catalog.adapters.config_resolver import StepConfigResolverAdapter
            self.config_resolver = StepConfigResolverAdapter()
        except ImportError:
            try:
                # Fallback: use absolute import with path setup
                from .import_utils import ensure_cursus_path
                ensure_cursus_path()
                from cursus.step_catalog.adapters.config_resolver import StepConfigResolverAdapter
                self.config_resolver = StepConfigResolverAdapter()
            except ImportError as e:
                logger.warning(f"StepConfigResolverAdapter not available: {e}")
        
        logger.info("DAGConfigurationManager initialized")
    
    def analyze_pipeline_dag(self, pipeline_dag: Any) -> Dict[str, Any]:
        """
        Analyze PipelineDAG to discover required configuration classes.
        
        Compatible with cursus.api.dag.base_dag.PipelineDAG where:
        - nodes is a list of strings (step names)
        - dependencies are managed through DAG structure
        
        Args:
            pipeline_dag: The pipeline DAG to analyze (PipelineDAG instance)
            
        Returns:
            Dict containing discovered steps, required configs, and workflow structure
        """
        logger.info("Analyzing pipeline DAG for configuration discovery")
        
        # Extract step names from DAG nodes
        discovered_steps = []
        dag_nodes = []
        
        if hasattr(pipeline_dag, 'nodes'):
            for node in pipeline_dag.nodes:
                # Handle both PipelineDAG (strings) and mock DAG nodes (objects)
                if isinstance(node, str):
                    # Real PipelineDAG node - just a string step name
                    node_name = node
                    step_type = self._infer_step_type_from_name(node_name)
                    dependencies = pipeline_dag.get_dependencies(node_name) if hasattr(pipeline_dag, 'get_dependencies') else []
                else:
                    # Mock DAG node with attributes (for testing/demo)
                    node_name = getattr(node, 'name', str(node))
                    step_type = getattr(node, 'step_type', self._infer_step_type_from_name(node_name))
                    dependencies = getattr(node, 'dependencies', [])
                
                dag_nodes.append(node_name)
                
                discovered_steps.append({
                    "step_name": node_name,
                    "step_type": step_type,
                    "dependencies": dependencies
                })
        else:
            logger.warning("Pipeline DAG does not have 'nodes' attribute")
        
        logger.info(f"Discovered {len(discovered_steps)} steps from DAG")
        
        # Discover required configuration classes
        required_configs = self.core._discover_required_config_classes(dag_nodes, self.config_resolver)
        
        # Create workflow structure
        workflow_steps = self.core._create_workflow_structure(required_configs)
        
        # Count total available configs for comparison
        total_configs = len(self.core.discover_config_classes())
        hidden_configs_count = total_configs - len(required_configs)
        
        analysis_result = {
            "discovered_steps": discovered_steps,
            "required_configs": required_configs,
            "workflow_steps": workflow_steps,
            "total_steps": len(workflow_steps),
            "hidden_configs_count": hidden_configs_count,
            "dag_nodes": dag_nodes,
            "analysis_summary": {
                "total_dag_nodes": len(dag_nodes),
                "required_configs": len(required_configs),
                "workflow_steps": len(workflow_steps),
                "hidden_configs": hidden_configs_count,
                "specialized_configs": len([c for c in required_configs if c.get("is_specialized", False)])
            }
        }
        
        logger.info(f"DAG analysis complete: {len(required_configs)} required configs, "
                   f"{len(workflow_steps)} workflow steps, {hidden_configs_count} configs hidden")
        
        return analysis_result
    
    def _infer_step_type_from_name(self, step_name: str) -> str:
        """
        Infer SageMaker step type from step name using cursus registry helper functions.
        
        Args:
            step_name: Name of the pipeline step (may include job type variants)
            
        Returns:
            SageMaker step type from registry or step name as fallback
        """
        try:
            from ....registry.step_names import get_canonical_name_from_file_name, get_sagemaker_step_type
            
            # get_canonical_name_from_file_name handles job type suffix removal automatically
            canonical_name = get_canonical_name_from_file_name(step_name)
            return get_sagemaker_step_type(canonical_name)
                
        except ImportError:
            try:
                # Fallback to absolute import
                from .import_utils import ensure_cursus_path
                ensure_cursus_path()
                from cursus.registry.step_names import get_canonical_name_from_file_name, get_sagemaker_step_type
                
                canonical_name = get_canonical_name_from_file_name(step_name)
                return get_sagemaker_step_type(canonical_name)
                    
            except (ImportError, ValueError):
                # Just return the step name as fallback
                return step_name
    
    def create_dag_driven_widget(self, 
                                pipeline_dag: Any, 
                                base_config: BasePipelineConfig,
                                processing_config: Optional[ProcessingStepConfigBase] = None) -> 'MultiStepWizard':
        """
        Create DAG-driven configuration widget with analysis.
        
        Args:
            pipeline_dag: Pipeline DAG to analyze
            base_config: Base configuration
            processing_config: Optional processing configuration
            
        Returns:
            MultiStepWizard configured for the DAG
        """
        # Analyze DAG first
        analysis_result = self.analyze_pipeline_dag(pipeline_dag)
        
        # Create enhanced workflow steps with DAG context
        enhanced_steps = []
        for step in analysis_result["workflow_steps"]:
            enhanced_step = step.copy()
            enhanced_step["dag_analysis"] = analysis_result["analysis_summary"]
            enhanced_step["description"] = self._get_step_description(step)
            enhanced_steps.append(enhanced_step)
        
        # Import here to avoid circular imports
        from ..widgets.widget import MultiStepWizard
        wizard = MultiStepWizard(enhanced_steps, base_config=base_config, processing_config=processing_config)
        
        # Add DAG analysis context to wizard
        wizard.dag_analysis = analysis_result
        
        logger.info(f"Created DAG-driven wizard with {len(enhanced_steps)} steps")
        return wizard
    
    def _get_step_description(self, step: Dict[str, Any]) -> str:
        """Get descriptive text for a workflow step."""
        step_type = step.get("type", "unknown")
        step_title = step.get("title", "Configuration")
        
        descriptions = {
            "base": "Configure common pipeline settings that will be inherited by all other steps",
            "processing": "Configure processing-specific settings for steps that require compute resources",
            "specific": f"Configure settings specific to the {step_title} step in your pipeline"
        }
        
        if step.get("is_specialized", False):
            return f"Configure {step_title} using a specialized interface with advanced options"
        
        return descriptions.get(step_type, f"Configure settings for {step_title}")
    
    def get_dag_analysis_summary(self, pipeline_dag: Any) -> str:
        """
        Get a human-readable summary of DAG analysis.
        
        Args:
            pipeline_dag: Pipeline DAG to analyze
            
        Returns:
            Formatted summary string
        """
        analysis = self.analyze_pipeline_dag(pipeline_dag)
        summary = analysis["analysis_summary"]
        
        summary_text = f"""
📊 Pipeline Analysis Results:

🔍 Discovered Pipeline Steps: {summary['total_dag_nodes']}
⚙️ Required Configurations: {summary['required_configs']} (Only these will be shown)
📋 Configuration Workflow: {summary['workflow_steps']} steps total
❌ Hidden Configurations: {summary['hidden_configs']} other config types not needed
🎛️ Specialized Configurations: {summary['specialized_configs']} requiring custom interfaces

Workflow Structure: Base Config → Processing Config → {summary['required_configs']} Specific Configs
        """
        
        return summary_text.strip()


# Factory functions for DAG-driven configuration
def create_pipeline_config_widget(pipeline_dag: Any, 
                                 base_config: BasePipelineConfig,
                                 processing_config: Optional[ProcessingStepConfigBase] = None,
                                 workspace_dirs: Optional[List[Union[str, Path]]] = None) -> 'MultiStepWizard':
    """
    Factory function for DAG-driven pipeline configuration widgets.
    
    Args:
        pipeline_dag: Pipeline DAG definition
        base_config: Base pipeline configuration
        processing_config: Optional processing configuration
        workspace_dirs: Optional workspace directories for step catalog
        
    Returns:
        MultiStepWizard instance
    """
    from .core import UniversalConfigCore
    core = UniversalConfigCore(workspace_dirs=workspace_dirs)
    dag_manager = DAGConfigurationManager(core)
    return dag_manager.create_dag_driven_widget(pipeline_dag, base_config, processing_config)


def analyze_pipeline_dag(pipeline_dag: Any, 
                        workspace_dirs: Optional[List[Union[str, Path]]] = None) -> Dict[str, Any]:
    """
    Factory function to analyze pipeline DAG for configuration discovery.
    
    Args:
        pipeline_dag: Pipeline DAG to analyze
        workspace_dirs: Optional workspace directories for step catalog
        
    Returns:
        DAG analysis results
    """
    from .core import UniversalConfigCore
    core = UniversalConfigCore(workspace_dirs=workspace_dirs)
    dag_manager = DAGConfigurationManager(core)
    return dag_manager.analyze_pipeline_dag(pipeline_dag)
