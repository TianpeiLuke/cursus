"""
Workspace DAG compiler for compiling workspace-aware DAGs to pipelines.

This module extends the PipelineDAGCompiler to support workspace components
and provides workspace-specific compilation capabilities.
"""

from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
import time

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession

from ...core.compiler.dag_compiler import PipelineDAGCompiler
from ...api.dag.workspace_dag import WorkspaceAwareDAG
from .config import WorkspacePipelineDefinition
from .assembler import WorkspacePipelineAssembler
from .registry import WorkspaceComponentRegistry

logger = logging.getLogger(__name__)


class WorkspaceDAGCompiler(PipelineDAGCompiler):
    """DAG compiler with workspace component support."""
    
    def __init__(
        self,
        workspace_root: str,
        workspace_manager: Optional['WorkspaceManager'] = None,
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize workspace DAG compiler with Phase 1 consolidated manager integration.
        
        Args:
            workspace_root: Root path of the workspace
            workspace_manager: Optional consolidated WorkspaceManager instance (Phase 1 integration)
            sagemaker_session: Optional SageMaker session
            role: Optional IAM role
            **kwargs: Additional arguments for parent constructor
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
        
        # Initialize parent with dummy config path (we'll override the compilation logic)
        super().__init__(
            config_path="",  # Not used for workspace compilation
            sagemaker_session=sagemaker_session,
            role=role,
            **kwargs
        )
        
        logger.info(f"Initialized enhanced workspace DAG compiler with Phase 1 integration for: {workspace_root}")
    
    def compile_workspace_dag(
        self,
        workspace_dag: WorkspaceAwareDAG,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[Pipeline, Dict]:
        """
        Compile workspace DAG to executable pipeline.
        
        Args:
            workspace_dag: WorkspaceAwareDAG instance to compile
            config: Optional additional configuration
            
        Returns:
            Tuple of (Pipeline, compilation_metadata)
        """
        logger.info(f"Compiling workspace DAG with {len(workspace_dag.workspace_steps)} steps")
        start_time = time.time()
        
        try:
            # Convert workspace DAG to workspace pipeline config
            pipeline_config_dict = workspace_dag.to_workspace_pipeline_config("workspace_pipeline")
            workspace_config = WorkspacePipelineDefinition(**pipeline_config_dict)
            
            # Create workspace pipeline assembler
            assembler = WorkspacePipelineAssembler(
                workspace_root=self.workspace_root,
                sagemaker_session=self.sagemaker_session,
                role=self.role
            )
            
            # Assemble pipeline
            pipeline = assembler.assemble_workspace_pipeline(workspace_config)
            
            # Generate compilation metadata
            compilation_metadata = {
                'workspace_root': self.workspace_root,
                'compilation_time': time.time() - start_time,
                'step_count': len(workspace_dag.workspace_steps),
                'developer_count': len(workspace_dag.get_developers()),
                'dependency_validation': workspace_dag.validate_workspace_dependencies(),
                'complexity_analysis': workspace_dag.analyze_workspace_complexity(),
                'assembler_summary': assembler.get_workspace_summary()
            }
            
            elapsed_time = time.time() - start_time
            logger.info(f"Compiled workspace DAG to pipeline in {elapsed_time:.2f}s")
            
            return pipeline, compilation_metadata
            
        except Exception as e:
            logger.error(f"Error compiling workspace DAG: {e}")
            raise ValueError(f"Failed to compile workspace DAG: {e}") from e
    
    def preview_workspace_resolution(self, workspace_dag: WorkspaceAwareDAG) -> Dict[str, Any]:
        """
        Preview how workspace DAG will be resolved to components.
        
        Args:
            workspace_dag: WorkspaceAwareDAG instance to preview
            
        Returns:
            Preview information dictionary
        """
        logger.info("Previewing workspace DAG resolution")
        
        preview = {
            'dag_summary': workspace_dag.get_workspace_summary(),
            'component_resolution': {},
            'validation_results': {},
            'compilation_feasibility': {}
        }
        
        try:
            # Convert to workspace config for validation
            pipeline_config_dict = workspace_dag.to_workspace_pipeline_config("preview_pipeline")
            workspace_config = WorkspacePipelineDefinition(**pipeline_config_dict)
            
            # Create assembler for preview
            assembler = WorkspacePipelineAssembler(
                workspace_root=self.workspace_root,
                sagemaker_session=self.sagemaker_session,
                role=self.role
            )
            
            # Get preview from assembler
            assembly_preview = assembler.preview_workspace_assembly(workspace_config)
            preview.update(assembly_preview)
            
            # Add compilation-specific analysis
            preview['compilation_feasibility'] = {
                'can_compile': assembly_preview['validation_results'].get('overall_valid', False),
                'blocking_issues': [],
                'warnings': [],
                'estimated_compilation_time': self._estimate_compilation_time(workspace_dag)
            }
            
            # Identify blocking issues
            validation_results = assembly_preview.get('validation_results', {})
            if not validation_results.get('valid', True):
                preview['compilation_feasibility']['blocking_issues'].extend(
                    validation_results.get('missing_components', [])
                )
            
            workspace_validation = validation_results.get('workspace_validation', {})
            for validation_type, result in workspace_validation.items():
                if not result.get('valid', True):
                    preview['compilation_feasibility']['blocking_issues'].extend(
                        result.get('errors', [])
                    )
                preview['compilation_feasibility']['warnings'].extend(
                    result.get('warnings', [])
                )
            
        except Exception as e:
            preview['error'] = str(e)
            logger.error(f"Error previewing workspace resolution: {e}")
        
        return preview
    
    def _estimate_compilation_time(self, workspace_dag: WorkspaceAwareDAG) -> float:
        """Estimate compilation time based on DAG complexity."""
        base_time = 5.0  # Base compilation time in seconds
        
        # Add time based on number of steps
        step_time = len(workspace_dag.workspace_steps) * 0.5
        
        # Add time based on number of developers (cross-workspace complexity)
        developer_time = len(workspace_dag.get_developers()) * 1.0
        
        # Add time based on dependency complexity
        dependency_count = sum(
            len(step['dependencies']) 
            for step in workspace_dag.workspace_steps.values()
        )
        dependency_time = dependency_count * 0.2
        
        return base_time + step_time + developer_time + dependency_time
    
    def validate_workspace_components(self, workspace_dag: WorkspaceAwareDAG) -> Dict[str, Any]:
        """
        Validate workspace component availability for compilation.
        
        Args:
            workspace_dag: WorkspaceAwareDAG instance to validate
            
        Returns:
            Validation result dictionary
        """
        logger.info("Validating workspace components for compilation")
        
        try:
            # Convert to workspace config for validation
            pipeline_config_dict = workspace_dag.to_workspace_pipeline_config("validation_pipeline")
            workspace_config = WorkspacePipelineDefinition(**pipeline_config_dict)
            
            # Use registry to validate components
            validation_result = self.workspace_registry.validate_component_availability(workspace_config)
            
            # Add DAG-specific validation
            dag_validation = workspace_dag.validate_workspace_dependencies()
            validation_result['dag_validation'] = dag_validation
            
            # Overall validation status
            validation_result['compilation_ready'] = (
                validation_result['valid'] and 
                dag_validation['valid']
            )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating workspace components: {e}")
            return {
                'valid': False,
                'compilation_ready': False,
                'error': str(e)
            }
    
    def generate_compilation_report(self, workspace_dag: WorkspaceAwareDAG) -> Dict[str, Any]:
        """
        Generate comprehensive compilation report for workspace DAG.
        
        Args:
            workspace_dag: WorkspaceAwareDAG instance to analyze
            
        Returns:
            Compilation report dictionary
        """
        logger.info("Generating workspace DAG compilation report")
        
        report = {
            'dag_analysis': workspace_dag.get_workspace_summary(),
            'complexity_analysis': workspace_dag.analyze_workspace_complexity(),
            'component_validation': self.validate_workspace_components(workspace_dag),
            'compilation_preview': self.preview_workspace_resolution(workspace_dag),
            'recommendations': [],
            'estimated_resources': {}
        }
        
        try:
            # Generate recommendations based on analysis
            complexity = report['complexity_analysis']
            
            if complexity['basic_metrics']['developer_count'] > 5:
                report['recommendations'].append(
                    "Consider splitting large workspace into smaller groups for better maintainability"
                )
            
            if complexity['complexity_metrics'].get('avg_dependencies_per_step', 0) > 3:
                report['recommendations'].append(
                    "High dependency complexity detected - consider simplifying step relationships"
                )
            
            # Estimate resource requirements
            step_count = complexity['basic_metrics']['node_count']
            report['estimated_resources'] = {
                'compilation_time_seconds': self._estimate_compilation_time(workspace_dag),
                'memory_usage_mb': step_count * 50 + 200,  # Rough estimate
                'storage_requirements': {
                    'config_files': f"{step_count * 2}KB",
                    'metadata': f"{step_count * 5}KB"
                }
            }
            
            # Add validation summary
            validation = report['component_validation']
            report['validation_summary'] = {
                'ready_for_compilation': validation.get('compilation_ready', False),
                'missing_components': len(validation.get('missing_components', [])),
                'available_components': len(validation.get('available_components', [])),
                'warnings': len(validation.get('warnings', []))
            }
            
        except Exception as e:
            report['error'] = str(e)
            logger.error(f"Error generating compilation report: {e}")
        
        return report
    
    @classmethod
    def from_workspace_config(
        cls,
        workspace_config: WorkspacePipelineDefinition,
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        **kwargs
    ) -> "WorkspaceDAGCompiler":
        """
        Create compiler from workspace configuration.
        
        Args:
            workspace_config: WorkspacePipelineDefinition instance
            sagemaker_session: Optional SageMaker session
            role: Optional IAM role
            **kwargs: Additional arguments
            
        Returns:
            Configured WorkspaceDAGCompiler instance
        """
        return cls(
            workspace_root=workspace_config.workspace_root,
            sagemaker_session=sagemaker_session,
            role=role,
            **kwargs
        )
    
    @classmethod
    def from_workspace_dag(
        cls,
        workspace_dag: WorkspaceAwareDAG,
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        **kwargs
    ) -> "WorkspaceDAGCompiler":
        """
        Create compiler from workspace DAG.
        
        Args:
            workspace_dag: WorkspaceAwareDAG instance
            sagemaker_session: Optional SageMaker session
            role: Optional IAM role
            **kwargs: Additional arguments
            
        Returns:
            Configured WorkspaceDAGCompiler instance
        """
        return cls(
            workspace_root=workspace_dag.workspace_root,
            sagemaker_session=sagemaker_session,
            role=role,
            **kwargs
        )
    
    def compile_with_detailed_report(
        self,
        workspace_dag: WorkspaceAwareDAG,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[Pipeline, Dict[str, Any]]:
        """
        Compile workspace DAG with detailed compilation report.
        
        Args:
            workspace_dag: WorkspaceAwareDAG instance to compile
            config: Optional additional configuration
            
        Returns:
            Tuple of (Pipeline, detailed_report)
        """
        logger.info("Compiling workspace DAG with detailed reporting")
        
        # Generate pre-compilation report
        pre_report = self.generate_compilation_report(workspace_dag)
        
        # Check if compilation is feasible
        if not pre_report['component_validation'].get('compilation_ready', False):
            raise ValueError(
                f"Workspace DAG is not ready for compilation: "
                f"{pre_report['component_validation']}"
            )
        
        # Compile the DAG
        pipeline, compilation_metadata = self.compile_workspace_dag(workspace_dag, config)
        
        # Create detailed report
        detailed_report = {
            'pre_compilation_analysis': pre_report,
            'compilation_metadata': compilation_metadata,
            'pipeline_info': {
                'name': pipeline.name,
                'step_count': len(pipeline.steps),
                'parameters': [p.name for p in pipeline.parameters] if pipeline.parameters else []
            },
            'success': True,
            'compilation_summary': {
                'total_time': compilation_metadata['compilation_time'],
                'steps_compiled': compilation_metadata['step_count'],
                'developers_involved': compilation_metadata['developer_count'],
                'validation_passed': True
            }
        }
        
        logger.info("Workspace DAG compilation with detailed reporting completed successfully")
        return pipeline, detailed_report
    
    def get_workspace_summary(self) -> Dict[str, Any]:
        """Get summary of workspace compiler capabilities."""
        return {
            'workspace_root': self.workspace_root,
            'registry_summary': self.workspace_registry.get_workspace_summary(),
            'compiler_capabilities': {
                'supports_workspace_dags': True,
                'supports_cross_workspace_dependencies': True,
                'supports_component_validation': True,
                'supports_compilation_preview': True,
                'supports_detailed_reporting': True
            }
        }
