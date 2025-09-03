"""
Workspace-aware pipeline assembler.

This module extends the PipelineAssembler to support workspace components
while maintaining full backward compatibility with existing functionality.
"""

from typing import Dict, List, Any, Optional, Type
from pathlib import Path
import logging
import time

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession

from ...core.assembler.pipeline_assembler import PipelineAssembler
from ...core.base import BasePipelineConfig, StepBuilderBase
from ...api.dag.base_dag import PipelineDAG
from ...api.dag.workspace_dag import WorkspaceAwareDAG
from .config import WorkspacePipelineDefinition, WorkspaceStepDefinition
from .registry import WorkspaceComponentRegistry

logger = logging.getLogger(__name__)


class WorkspacePipelineAssembler(PipelineAssembler):
    """Pipeline assembler with workspace component support."""
    
    def __init__(
        self,
        workspace_root: str,
        workspace_manager: Optional['WorkspaceManager'] = None,
        dag: Optional[PipelineDAG] = None,
        config_map: Optional[Dict[str, BasePipelineConfig]] = None,
        step_builder_map: Optional[Dict[str, Type[StepBuilderBase]]] = None,
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        pipeline_parameters: Optional[List] = None,
        notebook_root: Optional[Path] = None,
        **kwargs
    ):
        """
        Initialize workspace pipeline assembler with Phase 1 consolidated manager integration.
        
        Args:
            workspace_root: Root path of the workspace
            workspace_manager: Optional consolidated WorkspaceManager instance (Phase 1 integration)
            dag: Optional PipelineDAG instance
            config_map: Optional mapping from step name to config instance
            step_builder_map: Optional mapping from step type to builder class
            sagemaker_session: Optional SageMaker session
            role: Optional IAM role
            pipeline_parameters: Optional pipeline parameters
            notebook_root: Optional notebook root directory
            **kwargs: Additional arguments passed to parent constructor
        """
        self.workspace_root = workspace_root
        
        # PHASE 2 OPTIMIZATION: Integrate with Phase 1 consolidated managers
        if workspace_manager:
            self.workspace_manager = workspace_manager
        else:
            # Runtime import to avoid circular imports
            from .manager import WorkspaceManager
            self.workspace_manager = WorkspaceManager(workspace_root)
        
        # Enhanced component registry using consolidated discovery manager
        self.workspace_registry = WorkspaceComponentRegistry(
            workspace_root, 
            discovery_manager=self.workspace_manager.discovery_manager
        )
        
        # Access to specialized managers for enhanced functionality
        self.lifecycle_manager = self.workspace_manager.lifecycle_manager
        self.isolation_manager = self.workspace_manager.isolation_manager
        self.integration_manager = self.workspace_manager.integration_manager
        
        # Initialize parent with empty maps if not provided
        # We'll populate them from workspace components
        super().__init__(
            dag=dag or PipelineDAG(),
            config_map=config_map or {},
            step_builder_map=step_builder_map or {},
            sagemaker_session=sagemaker_session,
            role=role,
            pipeline_parameters=pipeline_parameters,
            notebook_root=notebook_root,
            **kwargs
        )
        
        logger.info(f"Initialized enhanced workspace pipeline assembler with Phase 1 integration for: {workspace_root}")
    
    def _resolve_workspace_configs(self, workspace_config: WorkspacePipelineDefinition) -> Dict[str, BasePipelineConfig]:
        """
        Resolve workspace step configurations to BasePipelineConfig instances.
        
        Args:
            workspace_config: WorkspacePipelineDefinition instance
            
        Returns:
            Dictionary mapping step names to config instances
        """
        logger.info(f"Resolving workspace configs for {len(workspace_config.steps)} steps")
        config_map = {}
        
        for step in workspace_config.steps:
            try:
                # Try to find config class in workspace
                config_class = self.workspace_registry.find_config_class(
                    step.step_name, 
                    step.developer_id
                )
                
                if config_class:
                    # Create config instance from workspace config data
                    config_instance = config_class(**step.config_data)
                    config_map[step.step_name] = config_instance
                    logger.debug(f"Resolved config for {step.step_name} using {config_class.__name__}")
                else:
                    # Fallback: create a generic config wrapper
                    logger.warning(f"Config class not found for {step.step_name}, using generic wrapper")
                    config_map[step.step_name] = step.config_data
                    
            except Exception as e:
                logger.error(f"Error resolving config for {step.step_name}: {e}")
                # Use raw config data as fallback
                config_map[step.step_name] = step.config_data
        
        logger.info(f"Resolved {len(config_map)} workspace configs")
        return config_map
    
    def _resolve_workspace_builders(self, workspace_config: WorkspacePipelineDefinition) -> Dict[str, Type[StepBuilderBase]]:
        """
        Resolve workspace step builders.
        
        Args:
            workspace_config: WorkspacePipelineConfig instance
            
        Returns:
            Dictionary mapping step types to builder classes
        """
        logger.info(f"Resolving workspace builders for {len(workspace_config.steps)} steps")
        builder_map = {}
        
        for step in workspace_config.steps:
            try:
                # Try to find builder class in workspace
                builder_class = self.workspace_registry.find_builder_class(
                    step.step_name,
                    step.developer_id
                )
                
                if builder_class:
                    builder_map[step.step_type] = builder_class
                    logger.debug(f"Resolved builder for {step.step_type} using {builder_class.__name__}")
                else:
                    logger.error(f"Builder class not found for {step.step_name} (type: {step.step_type})")
                    
            except Exception as e:
                logger.error(f"Error resolving builder for {step.step_name}: {e}")
        
        logger.info(f"Resolved {len(builder_map)} workspace builders")
        return builder_map
    
    def validate_workspace_components(self, workspace_config: WorkspacePipelineDefinition) -> Dict[str, Any]:
        """
        Validate workspace component availability and compatibility.
        
        Args:
            workspace_config: WorkspacePipelineConfig instance
            
        Returns:
            Validation result dictionary
        """
        logger.info("Validating workspace components")
        start_time = time.time()
        
        # Use registry to validate component availability
        validation_result = self.workspace_registry.validate_component_availability(workspace_config)
        
        # Additional validation for workspace-specific requirements
        validation_result['workspace_validation'] = {
            'dependency_validation': workspace_config.validate_workspace_dependencies(),
            'developer_consistency': self._validate_developer_consistency(workspace_config),
            'step_type_consistency': self._validate_step_type_consistency(workspace_config)
        }
        
        # Overall validation status
        workspace_valid = all([
            validation_result['workspace_validation']['dependency_validation']['valid'],
            validation_result['workspace_validation']['developer_consistency']['valid'],
            validation_result['workspace_validation']['step_type_consistency']['valid']
        ])
        
        validation_result['workspace_valid'] = workspace_valid
        validation_result['overall_valid'] = validation_result['valid'] and workspace_valid
        
        elapsed_time = time.time() - start_time
        logger.info(f"Workspace component validation completed in {elapsed_time:.2f}s")
        
        return validation_result
    
    def _validate_developer_consistency(self, workspace_config: WorkspacePipelineDefinition) -> Dict[str, Any]:
        """Validate developer consistency across workspace."""
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'developer_stats': {}
        }
        
        try:
            # Get developer statistics
            developers = workspace_config.get_developers()
            for dev_id in developers:
                dev_steps = workspace_config.get_steps_by_developer(dev_id)
                result['developer_stats'][dev_id] = {
                    'step_count': len(dev_steps),
                    'step_types': list(set(step.step_type for step in dev_steps))
                }
            
            # Check for potential issues
            if len(developers) > 10:
                result['warnings'].append(f"Large number of developers ({len(developers)}) may impact performance")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Developer consistency validation failed: {e}")
        
        return result
    
    def _validate_step_type_consistency(self, workspace_config: WorkspacePipelineDefinition) -> Dict[str, Any]:
        """Validate step type consistency."""
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'step_type_stats': {}
        }
        
        try:
            # Collect step type statistics
            step_types = {}
            for step in workspace_config.steps:
                if step.step_type not in step_types:
                    step_types[step.step_type] = []
                step_types[step.step_type].append({
                    'step_name': step.step_name,
                    'developer_id': step.developer_id
                })
            
            result['step_type_stats'] = step_types
            
            # Check for potential issues
            for step_type, instances in step_types.items():
                if len(instances) > 5:
                    result['warnings'].append(
                        f"Step type '{step_type}' has many instances ({len(instances)})"
                    )
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Step type consistency validation failed: {e}")
        
        return result
    
    def assemble_workspace_pipeline(self, workspace_config: WorkspacePipelineDefinition) -> Pipeline:
        """
        Assemble pipeline from workspace configuration.
        
        Args:
            workspace_config: WorkspacePipelineConfig instance
            
        Returns:
            Assembled SageMaker Pipeline
        """
        logger.info(f"Assembling workspace pipeline: {workspace_config.pipeline_name}")
        start_time = time.time()
        
        try:
            # Validate workspace components first
            validation_result = self.validate_workspace_components(workspace_config)
            if not validation_result['overall_valid']:
                error_msg = f"Workspace component validation failed: {validation_result}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Resolve workspace configurations and builders
            config_map = self._resolve_workspace_configs(workspace_config)
            builder_map = self._resolve_workspace_builders(workspace_config)
            
            # Create DAG from workspace configuration
            dag = self._create_dag_from_workspace_config(workspace_config)
            
            # Update assembler with resolved components
            self.config_map = config_map
            self.step_builder_map = builder_map
            self.dag = dag
            
            # Re-initialize step builders with new components
            self._initialize_step_builders()
            
            # Generate pipeline using parent functionality
            pipeline = self.generate_pipeline(workspace_config.pipeline_name)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Assembled workspace pipeline in {elapsed_time:.2f}s")
            
            return pipeline
            
        except Exception as e:
            logger.error(f"Error assembling workspace pipeline: {e}")
            raise ValueError(f"Failed to assemble workspace pipeline: {e}") from e
    
    def _create_dag_from_workspace_config(self, workspace_config: WorkspacePipelineDefinition) -> PipelineDAG:
        """Create DAG from workspace configuration."""
        dag = PipelineDAG()
        
        # Add all steps as nodes
        for step in workspace_config.steps:
            dag.add_node(step.step_name)
        
        # Add edges based on dependencies
        for step in workspace_config.steps:
            for dependency in step.dependencies:
                dag.add_edge(dependency, step.step_name)
        
        logger.info(f"Created DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges")
        return dag
    
    @classmethod
    def from_workspace_config(
        cls,
        workspace_config: WorkspacePipelineDefinition,
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        **kwargs
    ) -> "WorkspacePipelineAssembler":
        """
        Create assembler from workspace configuration.
        
        Args:
            workspace_config: WorkspacePipelineConfig instance
            sagemaker_session: Optional SageMaker session
            role: Optional IAM role
            **kwargs: Additional arguments
            
        Returns:
            Configured WorkspacePipelineAssembler instance
        """
        assembler = cls(
            workspace_root=workspace_config.workspace_root,
            sagemaker_session=sagemaker_session,
            role=role,
            **kwargs
        )
        
        return assembler
    
    @classmethod
    def from_workspace_config_file(
        cls,
        config_file_path: str,
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        **kwargs
    ) -> "WorkspacePipelineAssembler":
        """
        Create assembler from workspace configuration file.
        
        Args:
            config_file_path: Path to workspace configuration file (JSON or YAML)
            sagemaker_session: Optional SageMaker session
            role: Optional IAM role
            **kwargs: Additional arguments
            
        Returns:
            Configured WorkspacePipelineAssembler instance
        """
        # Load workspace configuration
        if config_file_path.endswith('.json'):
            workspace_config = WorkspacePipelineDefinition.from_json_file(config_file_path)
        elif config_file_path.endswith(('.yaml', '.yml')):
            workspace_config = WorkspacePipelineDefinition.from_yaml_file(config_file_path)
        else:
            raise ValueError(f"Unsupported config file format: {config_file_path}")
        
        return cls.from_workspace_config(
            workspace_config=workspace_config,
            sagemaker_session=sagemaker_session,
            role=role,
            **kwargs
        )
    
    def get_workspace_summary(self) -> Dict[str, Any]:
        """Get summary of workspace components and assembly status."""
        return {
            'workspace_root': self.workspace_root,
            'registry_summary': self.workspace_registry.get_workspace_summary(),
            'assembly_status': {
                'dag_nodes': len(self.dag.nodes) if self.dag else 0,
                'dag_edges': len(self.dag.edges) if self.dag else 0,
                'config_count': len(self.config_map),
                'builder_count': len(self.step_builder_map),
                'step_instances': len(self.step_instances)
            }
        }
    
    def preview_workspace_assembly(self, workspace_config: WorkspacePipelineDefinition) -> Dict[str, Any]:
        """
        Preview workspace assembly without actually building the pipeline.
        
        Args:
            workspace_config: WorkspacePipelineConfig instance
            
        Returns:
            Preview information dictionary
        """
        logger.info("Previewing workspace assembly")
        
        preview = {
            'workspace_config': {
                'pipeline_name': workspace_config.pipeline_name,
                'step_count': len(workspace_config.steps),
                'developers': workspace_config.get_developers()
            },
            'component_resolution': {},
            'validation_results': {},
            'assembly_plan': {}
        }
        
        try:
            # Validate components
            validation_result = self.validate_workspace_components(workspace_config)
            preview['validation_results'] = validation_result
            
            # Preview component resolution
            for step in workspace_config.steps:
                step_key = f"{step.developer_id}:{step.step_name}"
                
                builder_class = self.workspace_registry.find_builder_class(
                    step.step_name, step.developer_id
                )
                config_class = self.workspace_registry.find_config_class(
                    step.step_name, step.developer_id
                )
                
                preview['component_resolution'][step_key] = {
                    'step_name': step.step_name,
                    'step_type': step.step_type,
                    'developer_id': step.developer_id,
                    'builder_available': builder_class is not None,
                    'builder_class': builder_class.__name__ if builder_class else None,
                    'config_available': config_class is not None,
                    'config_class': config_class.__name__ if config_class else None,
                    'dependencies': step.dependencies
                }
            
            # Create assembly plan
            dag = self._create_dag_from_workspace_config(workspace_config)
            try:
                build_order = dag.topological_sort()
                preview['assembly_plan'] = {
                    'build_order': build_order,
                    'dag_valid': True,
                    'total_steps': len(build_order)
                }
            except ValueError as e:
                preview['assembly_plan'] = {
                    'build_order': [],
                    'dag_valid': False,
                    'error': str(e)
                }
            
        except Exception as e:
            preview['error'] = str(e)
            logger.error(f"Error previewing workspace assembly: {e}")
        
        return preview
