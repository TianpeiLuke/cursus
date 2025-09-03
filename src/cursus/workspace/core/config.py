"""
Workspace configuration models using Pydantic V2.

This module provides Pydantic V2 models for workspace step definitions and
pipeline configurations, enabling workspace-aware pipeline assembly.
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Dict, List, Any, Optional
import json
import yaml
from pathlib import Path

from ...core.base import BasePipelineConfig


class WorkspaceStepDefinition(BaseModel):
    """Pydantic V2 model for workspace step definitions."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
        frozen=False
    )
    
    step_name: str = Field(..., description="Name of the step")
    developer_id: str = Field(..., description="Developer workspace identifier")
    step_type: str = Field(..., description="Type of the step (e.g., 'XGBoostTraining')")
    config_data: Dict[str, Any] = Field(..., description="Step configuration data")
    workspace_root: str = Field(..., description="Root path of the workspace")
    dependencies: List[str] = Field(default_factory=list, description="List of step dependencies")
    
    @field_validator('step_name')
    @classmethod
    def validate_step_name(cls, v):
        """Validate step name format."""
        if not v or not isinstance(v, str):
            raise ValueError("step_name must be a non-empty string")
        return v
    
    @field_validator('developer_id')
    @classmethod
    def validate_developer_id(cls, v):
        """Validate developer ID format."""
        if not v or not isinstance(v, str):
            raise ValueError("developer_id must be a non-empty string")
        return v
    
    @field_validator('step_type')
    @classmethod
    def validate_step_type(cls, v):
        """Validate step type format."""
        if not v or not isinstance(v, str):
            raise ValueError("step_type must be a non-empty string")
        return v
    
    @field_validator('workspace_root')
    @classmethod
    def validate_workspace_root(cls, v):
        """Validate workspace root path."""
        if not v or not isinstance(v, str):
            raise ValueError("workspace_root must be a non-empty string")
        return v
    
    def to_config_instance(self) -> BasePipelineConfig:
        """Convert to a BasePipelineConfig instance."""
        # This would need to be implemented based on the specific config type
        # For now, return a generic representation
        return self.config_data
    
    def get_workspace_path(self, relative_path: str = "") -> str:
        """Get a path relative to the workspace root."""
        if relative_path:
            return str(Path(self.workspace_root) / relative_path)
        return self.workspace_root
    
    def validate_with_workspace_manager(self, workspace_manager: 'WorkspaceManager') -> Dict[str, Any]:
        """
        Enhanced validation using consolidated workspace manager (Phase 2 optimization).
        
        Args:
            workspace_manager: Consolidated WorkspaceManager instance
            
        Returns:
            Validation result dictionary
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'validations': {}
        }
        
        try:
            # Validate using isolation manager
            isolation_result = workspace_manager.isolation_manager.validate_step_definition(self)
            validation_result['validations']['isolation'] = isolation_result
            if not isolation_result.get('valid', True):
                validation_result['valid'] = False
                validation_result['errors'].extend(isolation_result.get('errors', []))
            
            # Validate using lifecycle manager
            lifecycle_result = workspace_manager.lifecycle_manager.validate_step_lifecycle(self)
            validation_result['validations']['lifecycle'] = lifecycle_result
            if not lifecycle_result.get('valid', True):
                validation_result['valid'] = False
                validation_result['errors'].extend(lifecycle_result.get('errors', []))
            
            # Add warnings from all validations
            for validation_type, result in validation_result['validations'].items():
                validation_result['warnings'].extend(result.get('warnings', []))
                
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation failed: {e}")
        
        return validation_result
    
    def resolve_dependencies(self, workspace_manager: 'WorkspaceManager') -> Dict[str, Any]:
        """
        Enhanced dependency resolution using discovery manager (Phase 2 optimization).
        
        Args:
            workspace_manager: Consolidated WorkspaceManager instance
            
        Returns:
            Dependency resolution result
        """
        try:
            return workspace_manager.discovery_manager.resolve_step_dependencies(self)
        except Exception as e:
            return {
                'valid': False,
                'error': f"Dependency resolution failed: {e}",
                'dependencies': self.dependencies
            }


class WorkspacePipelineDefinition(BaseModel):
    """Pydantic V2 model for workspace pipeline definition."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
        frozen=False
    )
    
    pipeline_name: str = Field(..., description="Name of the pipeline")
    workspace_root: str = Field(..., description="Root path of the workspace")
    steps: List[WorkspaceStepDefinition] = Field(..., description="List of pipeline steps")
    global_config: Dict[str, Any] = Field(default_factory=dict, description="Global pipeline configuration")
    
    @field_validator('pipeline_name')
    @classmethod
    def validate_pipeline_name(cls, v):
        """Validate pipeline name format."""
        if not v or not isinstance(v, str):
            raise ValueError("pipeline_name must be a non-empty string")
        return v
    
    @field_validator('workspace_root')
    @classmethod
    def validate_workspace_root(cls, v):
        """Validate workspace root path."""
        if not v or not isinstance(v, str):
            raise ValueError("workspace_root must be a non-empty string")
        return v
    
    @field_validator('steps')
    @classmethod
    def validate_steps(cls, v):
        """Validate steps list."""
        if not v:
            raise ValueError("steps list cannot be empty")
        
        # Check for duplicate step names
        step_names = [step.step_name for step in v]
        if len(step_names) != len(set(step_names)):
            raise ValueError("Duplicate step names found in pipeline")
        
        return v
    
    def validate_workspace_dependencies(self) -> Dict[str, Any]:
        """Validate workspace dependencies and references."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'dependency_graph': {}
        }
        
        step_names = {step.step_name for step in self.steps}
        
        for step in self.steps:
            validation_result['dependency_graph'][step.step_name] = step.dependencies
            
            # Check if all dependencies exist in the pipeline
            for dep in step.dependencies:
                if dep not in step_names:
                    validation_result['valid'] = False
                    validation_result['errors'].append(
                        f"Step '{step.step_name}' depends on '{dep}' which is not defined in the pipeline"
                    )
        
        # Check for circular dependencies
        if self._has_circular_dependencies():
            validation_result['valid'] = False
            validation_result['errors'].append("Circular dependencies detected in pipeline")
        
        return validation_result
    
    def _has_circular_dependencies(self) -> bool:
        """Check for circular dependencies using DFS."""
        # Build adjacency list
        graph = {}
        for step in self.steps:
            graph[step.step_name] = step.dependencies
        
        # Track visited nodes and recursion stack
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if has_cycle(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        # Check each node
        for step_name in graph:
            if step_name not in visited:
                if has_cycle(step_name):
                    return True
        
        return False
    
    def to_pipeline_config(self) -> Dict[str, Any]:
        """Convert to standard pipeline configuration format."""
        config = {
            'pipeline_name': self.pipeline_name,
            'workspace_root': self.workspace_root,
            'global_config': self.global_config,
            'steps': {}
        }
        
        for step in self.steps:
            config['steps'][step.step_name] = {
                'step_type': step.step_type,
                'developer_id': step.developer_id,
                'config_data': step.config_data,
                'dependencies': step.dependencies
            }
        
        return config
    
    def get_developers(self) -> List[str]:
        """Get list of unique developers in the pipeline."""
        return list(set(step.developer_id for step in self.steps))
    
    def get_steps_by_developer(self, developer_id: str) -> List[WorkspaceStepDefinition]:
        """Get all steps for a specific developer."""
        return [step for step in self.steps if step.developer_id == developer_id]
    
    def get_step_by_name(self, step_name: str) -> Optional[WorkspaceStepDefinition]:
        """Get a step by its name."""
        for step in self.steps:
            if step.step_name == step_name:
                return step
        return None
    
    @classmethod
    def from_json_file(cls, file_path: str) -> 'WorkspacePipelineDefinition':
        """Load configuration from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def from_yaml_file(cls, file_path: str) -> 'WorkspacePipelineDefinition':
        """Load configuration from YAML file."""
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_json_file(self, file_path: str, indent: int = 2) -> None:
        """Save configuration to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.model_dump(), f, indent=indent)
    
    def to_yaml_file(self, file_path: str) -> None:
        """Save configuration to YAML file."""
        with open(file_path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
    
    def validate_with_consolidated_managers(self, workspace_manager: 'WorkspaceManager') -> Dict[str, Any]:
        """
        Comprehensive validation using all consolidated managers (Phase 2 optimization).
        
        Args:
            workspace_manager: Consolidated WorkspaceManager instance
            
        Returns:
            Comprehensive validation result
        """
        validation_results = {
            'overall_valid': True,
            'validations': {},
            'errors': [],
            'warnings': [],
            'summary': {}
        }
        
        try:
            # Lifecycle validation
            lifecycle_validation = workspace_manager.lifecycle_manager.validate_pipeline_lifecycle(self)
            validation_results['validations']['lifecycle'] = lifecycle_validation
            
            # Isolation validation
            isolation_validation = workspace_manager.isolation_manager.validate_pipeline_isolation(self)
            validation_results['validations']['isolation'] = isolation_validation
            
            # Discovery validation (dependency resolution)
            discovery_validation = workspace_manager.discovery_manager.validate_pipeline_dependencies(self)
            validation_results['validations']['discovery'] = discovery_validation
            
            # Integration validation
            integration_validation = workspace_manager.integration_manager.validate_pipeline_integration(self)
            validation_results['validations']['integration'] = integration_validation
            
            # Combine results
            all_valid = True
            total_errors = []
            total_warnings = []
            
            for validation_type, result in validation_results['validations'].items():
                if not result.get('valid', True):
                    all_valid = False
                total_errors.extend(result.get('errors', []))
                total_warnings.extend(result.get('warnings', []))
            
            validation_results['overall_valid'] = all_valid
            validation_results['errors'] = total_errors
            validation_results['warnings'] = total_warnings
            
            # Create summary
            validation_results['summary'] = {
                'total_validations': len(validation_results['validations']),
                'passed_validations': sum(1 for v in validation_results['validations'].values() if v.get('valid', True)),
                'total_errors': len(total_errors),
                'total_warnings': len(total_warnings),
                'pipeline_ready': all_valid and len(total_errors) == 0
            }
            
        except Exception as e:
            validation_results['overall_valid'] = False
            validation_results['errors'].append(f"Validation framework error: {e}")
            validation_results['summary'] = {'error': str(e)}
        
        return validation_results
    
    def resolve_cross_workspace_dependencies(self, workspace_manager: 'WorkspaceManager') -> Dict[str, Any]:
        """
        Enhanced cross-workspace dependency resolution (Phase 2 optimization).
        
        Args:
            workspace_manager: Consolidated WorkspaceManager instance
            
        Returns:
            Cross-workspace dependency resolution result
        """
        try:
            return workspace_manager.discovery_manager.resolve_cross_workspace_dependencies(self)
        except Exception as e:
            return {
                'valid': False,
                'error': f"Cross-workspace dependency resolution failed: {e}",
                'dependencies': {step.step_name: step.dependencies for step in self.steps}
            }
    
    def prepare_for_integration(self, workspace_manager: 'WorkspaceManager') -> Dict[str, Any]:
        """
        Prepare pipeline for integration staging (Phase 2 optimization).
        
        Args:
            workspace_manager: Consolidated WorkspaceManager instance
            
        Returns:
            Integration preparation result
        """
        try:
            return workspace_manager.integration_manager.prepare_pipeline_for_integration(self)
        except Exception as e:
            return {
                'ready': False,
                'error': f"Integration preparation failed: {e}",
                'pipeline_name': self.pipeline_name
            }
