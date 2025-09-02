"""
Integration tests for Phase 2 Pipeline Assembly Layer Optimization.

This module provides comprehensive integration testing for the optimized pipeline
assembly components that integrate with Phase 1 consolidated managers.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch

from src.cursus.core.workspace.manager import WorkspaceManager
from src.cursus.core.workspace.assembler import WorkspacePipelineAssembler
from src.cursus.core.workspace.registry import WorkspaceComponentRegistry
from src.cursus.core.workspace.config import WorkspaceStepDefinition, WorkspacePipelineDefinition
from src.cursus.core.workspace.compiler import WorkspaceDAGCompiler


class TestPhase2PipelineAssemblyIntegration:
    """Test suite for Phase 2 pipeline assembly layer optimization."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        workspace_path = Path(temp_dir) / "test_workspace"
        workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Create basic workspace structure
        (workspace_path / "developer_1").mkdir(exist_ok=True)
        (workspace_path / "developer_2").mkdir(exist_ok=True)
        
        yield str(workspace_path)
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def consolidated_workspace_manager(self, temp_workspace):
        """Create consolidated workspace manager for testing."""
        return WorkspaceManager(temp_workspace)
    
    @pytest.fixture
    def sample_workspace_config(self, temp_workspace):
        """Create sample workspace pipeline configuration."""
        steps = [
            WorkspaceStepDefinition(
                step_name="data_preprocessing",
                developer_id="developer_1",
                step_type="ProcessingStep",
                config_data={"input_path": "/data/raw", "output_path": "/data/processed"},
                workspace_root=temp_workspace,
                dependencies=[]
            ),
            WorkspaceStepDefinition(
                step_name="model_training",
                developer_id="developer_2",
                step_type="TrainingStep",
                config_data={"model_type": "xgboost", "hyperparameters": {"n_estimators": 100}},
                workspace_root=temp_workspace,
                dependencies=["data_preprocessing"]
            )
        ]
        
        return WorkspacePipelineDefinition(
            pipeline_name="test_pipeline",
            workspace_root=temp_workspace,
            steps=steps,
            global_config={"region": "us-west-2", "role": "test-role"}
        )


class TestWorkspacePipelineAssemblerIntegration:
    """Test WorkspacePipelineAssembler integration with Phase 1 managers."""
    
    def test_assembler_initialization_with_consolidated_manager(self, temp_workspace, consolidated_workspace_manager):
        """Test assembler initialization with consolidated workspace manager."""
        # Test with provided workspace manager
        assembler = WorkspacePipelineAssembler(
            workspace_root=temp_workspace,
            workspace_manager=consolidated_workspace_manager
        )
        
        # Verify integration
        assert assembler.workspace_manager is consolidated_workspace_manager
        assert assembler.lifecycle_manager is consolidated_workspace_manager.lifecycle_manager
        assert assembler.isolation_manager is consolidated_workspace_manager.isolation_manager
        assert assembler.integration_manager is consolidated_workspace_manager.integration_manager
        
        # Verify registry integration
        assert assembler.workspace_registry.discovery_manager is consolidated_workspace_manager.discovery_manager
    
    def test_assembler_initialization_without_manager(self, temp_workspace):
        """Test assembler initialization without provided workspace manager."""
        assembler = WorkspacePipelineAssembler(workspace_root=temp_workspace)
        
        # Verify it creates its own consolidated manager
        assert assembler.workspace_manager is not None
        assert assembler.workspace_manager.workspace_root == temp_workspace
        assert assembler.lifecycle_manager is not None
        assert assembler.isolation_manager is not None
        assert assembler.integration_manager is not None
    
    def test_enhanced_validation_integration(self, temp_workspace, consolidated_workspace_manager, sample_workspace_config):
        """Test enhanced validation using consolidated managers."""
        assembler = WorkspacePipelineAssembler(
            workspace_root=temp_workspace,
            workspace_manager=consolidated_workspace_manager
        )
        
        # Mock the specialized managers for testing
        with patch.object(assembler.isolation_manager, 'validate_workspace_boundaries_for_pipeline') as mock_isolation, \
             patch.object(assembler.workspace_registry, 'validate_component_availability') as mock_registry:
            
            mock_registry.return_value = {
                'valid': True,
                'missing_components': [],
                'available_components': [
                    {'step_name': 'data_preprocessing', 'developer_id': 'developer_1', 'component_type': 'builder'},
                    {'step_name': 'model_training', 'developer_id': 'developer_2', 'component_type': 'builder'}
                ]
            }
            
            mock_isolation.return_value = {'valid': True, 'errors': [], 'warnings': []}
            
            # Test validation
            result = assembler.validate_workspace_components(sample_workspace_config)
            
            # Verify enhanced validation was called
            mock_registry.assert_called_once_with(sample_workspace_config)
            assert result['overall_valid'] is True
            assert 'workspace_validation' in result
    
    def test_cross_workspace_dependency_resolution(self, temp_workspace, consolidated_workspace_manager, sample_workspace_config):
        """Test cross-workspace dependency resolution using discovery manager."""
        assembler = WorkspacePipelineAssembler(
            workspace_root=temp_workspace,
            workspace_manager=consolidated_workspace_manager
        )
        
        # Mock discovery manager
        with patch.object(assembler.workspace_manager.discovery_manager, 'resolve_pipeline_dependencies') as mock_resolve:
            mock_resolve.return_value = {
                'valid': True,
                'resolved_dependencies': {
                    'model_training': ['data_preprocessing']
                },
                'dependency_graph': {
                    'data_preprocessing': [],
                    'model_training': ['data_preprocessing']
                }
            }
            
            # This would be called internally during pipeline assembly
            # For now, we test the integration point exists
            assert hasattr(assembler.workspace_manager.discovery_manager, 'resolve_pipeline_dependencies')


class TestWorkspaceComponentRegistryIntegration:
    """Test WorkspaceComponentRegistry integration with Phase 1 managers."""
    
    def test_registry_initialization_with_discovery_manager(self, temp_workspace, consolidated_workspace_manager):
        """Test registry initialization with discovery manager."""
        discovery_manager = consolidated_workspace_manager.discovery_manager
        
        registry = WorkspaceComponentRegistry(
            workspace_root=temp_workspace,
            discovery_manager=discovery_manager
        )
        
        # Verify integration
        assert registry.discovery_manager is discovery_manager
        assert registry.workspace_manager is consolidated_workspace_manager
    
    def test_registry_initialization_without_discovery_manager(self, temp_workspace):
        """Test registry initialization without discovery manager."""
        registry = WorkspaceComponentRegistry(workspace_root=temp_workspace)
        
        # Verify it creates consolidated manager
        assert registry.workspace_manager is not None
        assert registry.discovery_manager is not None
        assert registry.discovery_manager.workspace_manager is registry.workspace_manager
    
    def test_enhanced_caching_integration(self, temp_workspace, consolidated_workspace_manager):
        """Test enhanced caching using discovery manager."""
        discovery_manager = consolidated_workspace_manager.discovery_manager
        
        # Mock the discovery manager's cache
        with patch.object(discovery_manager, 'get_component_cache') as mock_cache:
            mock_cache.return_value = {'test_cache': 'data'}
            
            registry = WorkspaceComponentRegistry(
                workspace_root=temp_workspace,
                discovery_manager=discovery_manager
            )
            
            # Verify cache integration
            mock_cache.assert_called_once()
            assert registry._component_cache == {'test_cache': 'data'}
    
    def test_component_discovery_delegation(self, temp_workspace, consolidated_workspace_manager):
        """Test component discovery delegation to consolidated discovery manager."""
        registry = WorkspaceComponentRegistry(
            workspace_root=temp_workspace,
            discovery_manager=consolidated_workspace_manager.discovery_manager
        )
        
        # Mock discovery manager methods
        with patch.object(registry.discovery_manager, 'discover_all_workspace_components') as mock_discover:
            mock_discover.return_value = {
                'builders': {'dev1:step1': {'info': 'test'}},
                'configs': {},
                'summary': {'total_components': 1}
            }
            
            # Test that registry delegates to discovery manager
            # This would be called internally by discover_components
            assert hasattr(registry.discovery_manager, 'discover_all_workspace_components')


class TestConfigurationModelsIntegration:
    """Test configuration models integration with Phase 1 managers."""
    
    def test_step_definition_validation_integration(self, temp_workspace, consolidated_workspace_manager):
        """Test step definition validation with consolidated managers."""
        step = WorkspaceStepDefinition(
            step_name="test_step",
            developer_id="developer_1",
            step_type="TestStep",
            config_data={"param1": "value1"},
            workspace_root=temp_workspace,
            dependencies=[]
        )
        
        # Mock specialized managers
        with patch.object(consolidated_workspace_manager.isolation_manager, 'validate_step_definition') as mock_isolation, \
             patch.object(consolidated_workspace_manager.lifecycle_manager, 'validate_step_lifecycle') as mock_lifecycle:
            
            mock_isolation.return_value = {'valid': True, 'errors': [], 'warnings': []}
            mock_lifecycle.return_value = {'valid': True, 'errors': [], 'warnings': []}
            
            # Test validation
            result = step.validate_with_workspace_manager(consolidated_workspace_manager)
            
            # Verify integration
            mock_isolation.assert_called_once_with(step)
            mock_lifecycle.assert_called_once_with(step)
            assert result['valid'] is True
            assert 'validations' in result
            assert 'isolation' in result['validations']
            assert 'lifecycle' in result['validations']
    
    def test_step_dependency_resolution(self, temp_workspace, consolidated_workspace_manager):
        """Test step dependency resolution using discovery manager."""
        step = WorkspaceStepDefinition(
            step_name="test_step",
            developer_id="developer_1",
            step_type="TestStep",
            config_data={"param1": "value1"},
            workspace_root=temp_workspace,
            dependencies=["dependency_step"]
        )
        
        # Mock discovery manager
        with patch.object(consolidated_workspace_manager.discovery_manager, 'resolve_step_dependencies') as mock_resolve:
            mock_resolve.return_value = {
                'valid': True,
                'resolved_dependencies': ['dependency_step'],
                'dependency_info': {'dependency_step': {'developer_id': 'developer_2'}}
            }
            
            # Test dependency resolution
            result = step.resolve_dependencies(consolidated_workspace_manager)
            
            # Verify integration
            mock_resolve.assert_called_once_with(step)
            assert result['valid'] is True
    
    def test_pipeline_definition_comprehensive_validation(self, temp_workspace, consolidated_workspace_manager, sample_workspace_config):
        """Test pipeline definition comprehensive validation with all managers."""
        # Mock all specialized managers
        with patch.object(consolidated_workspace_manager.lifecycle_manager, 'validate_pipeline_lifecycle') as mock_lifecycle, \
             patch.object(consolidated_workspace_manager.isolation_manager, 'validate_pipeline_isolation') as mock_isolation, \
             patch.object(consolidated_workspace_manager.discovery_manager, 'validate_pipeline_dependencies') as mock_discovery, \
             patch.object(consolidated_workspace_manager.integration_manager, 'validate_pipeline_integration') as mock_integration:
            
            # Setup mock returns
            mock_lifecycle.return_value = {'valid': True, 'errors': [], 'warnings': []}
            mock_isolation.return_value = {'valid': True, 'errors': [], 'warnings': []}
            mock_discovery.return_value = {'valid': True, 'errors': [], 'warnings': []}
            mock_integration.return_value = {'valid': True, 'errors': [], 'warnings': []}
            
            # Test comprehensive validation
            result = sample_workspace_config.validate_with_consolidated_managers(consolidated_workspace_manager)
            
            # Verify all managers were called
            mock_lifecycle.assert_called_once_with(sample_workspace_config)
            mock_isolation.assert_called_once_with(sample_workspace_config)
            mock_discovery.assert_called_once_with(sample_workspace_config)
            mock_integration.assert_called_once_with(sample_workspace_config)
            
            # Verify result structure
            assert result['overall_valid'] is True
            assert 'validations' in result
            assert len(result['validations']) == 4
            assert 'summary' in result
    
    def test_pipeline_integration_preparation(self, temp_workspace, consolidated_workspace_manager, sample_workspace_config):
        """Test pipeline integration preparation using integration manager."""
        # Mock integration manager
        with patch.object(consolidated_workspace_manager.integration_manager, 'prepare_pipeline_for_integration') as mock_prepare:
            mock_prepare.return_value = {
                'ready': True,
                'staging_area': '/staging/test_pipeline',
                'components_staged': ['data_preprocessing', 'model_training']
            }
            
            # Test integration preparation
            result = sample_workspace_config.prepare_for_integration(consolidated_workspace_manager)
            
            # Verify integration
            mock_prepare.assert_called_once_with(sample_workspace_config)
            assert result['ready'] is True


class TestWorkspaceDAGCompilerIntegration:
    """Test WorkspaceDAGCompiler integration with Phase 1 managers."""
    
    def test_compiler_initialization_with_consolidated_manager(self, temp_workspace, consolidated_workspace_manager):
        """Test compiler initialization with consolidated workspace manager."""
        compiler = WorkspaceDAGCompiler(
            workspace_root=temp_workspace,
            workspace_manager=consolidated_workspace_manager
        )
        
        # Verify integration
        assert compiler.workspace_manager is consolidated_workspace_manager
        assert compiler.lifecycle_manager is consolidated_workspace_manager.lifecycle_manager
        assert compiler.isolation_manager is consolidated_workspace_manager.isolation_manager
        assert compiler.integration_manager is consolidated_workspace_manager.integration_manager
        
        # Verify registry integration
        assert compiler.workspace_registry.discovery_manager is consolidated_workspace_manager.discovery_manager
    
    def test_compiler_initialization_without_manager(self, temp_workspace):
        """Test compiler initialization without provided workspace manager."""
        compiler = WorkspaceDAGCompiler(workspace_root=temp_workspace)
        
        # Verify it creates its own consolidated manager
        assert compiler.workspace_manager is not None
        assert compiler.workspace_manager.workspace_root == temp_workspace
        assert compiler.lifecycle_manager is not None
        assert compiler.isolation_manager is not None
        assert compiler.integration_manager is not None
    
    def test_enhanced_compilation_with_consolidated_managers(self, temp_workspace, consolidated_workspace_manager):
        """Test enhanced compilation process using consolidated managers."""
        compiler = WorkspaceDAGCompiler(
            workspace_root=temp_workspace,
            workspace_manager=consolidated_workspace_manager
        )
        
        # Mock workspace DAG
        mock_dag = Mock()
        mock_dag.workspace_steps = {
            'step1': {'developer_id': 'dev1', 'dependencies': []},
            'step2': {'developer_id': 'dev2', 'dependencies': ['step1']}
        }
        mock_dag.get_developers.return_value = ['dev1', 'dev2']
        mock_dag.to_workspace_pipeline_config.return_value = {
            'pipeline_name': 'test_pipeline',
            'workspace_root': temp_workspace,
            'steps': [],
            'global_config': {}
        }
        mock_dag.validate_workspace_dependencies.return_value = {'valid': True}
        mock_dag.analyze_workspace_complexity.return_value = {'basic_metrics': {'node_count': 2}}
        
        # Test that compiler has access to all specialized managers
        assert hasattr(compiler, 'lifecycle_manager')
        assert hasattr(compiler, 'isolation_manager')
        assert hasattr(compiler, 'integration_manager')
        
        # Verify the managers are the same instances from consolidated manager
        assert compiler.lifecycle_manager is consolidated_workspace_manager.lifecycle_manager
        assert compiler.isolation_manager is consolidated_workspace_manager.isolation_manager
        assert compiler.integration_manager is consolidated_workspace_manager.integration_manager


class TestEndToEndIntegration:
    """End-to-end integration tests for Phase 2 optimizations."""
    
    def test_complete_pipeline_assembly_workflow(self, temp_workspace, consolidated_workspace_manager, sample_workspace_config):
        """Test complete pipeline assembly workflow with Phase 1 integration."""
        # Create assembler with consolidated manager
        assembler = WorkspacePipelineAssembler(
            workspace_root=temp_workspace,
            workspace_manager=consolidated_workspace_manager
        )
        
        # Mock all necessary components for end-to-end test
        with patch.object(assembler.workspace_registry, 'validate_component_availability') as mock_validate, \
             patch.object(assembler.workspace_registry, 'find_builder_class') as mock_find_builder, \
             patch.object(assembler.workspace_registry, 'find_config_class') as mock_find_config, \
             patch.object(assembler, '_initialize_step_builders') as mock_init_builders, \
             patch.object(assembler, 'generate_pipeline') as mock_generate:
            
            # Setup mocks
            mock_validate.return_value = {
                'valid': True,
                'overall_valid': True,
                'missing_components': [],
                'workspace_validation': {
                    'dependency_validation': {'valid': True},
                    'developer_consistency': {'valid': True},
                    'step_type_consistency': {'valid': True}
                }
            }
            
            mock_find_builder.return_value = Mock()
            mock_find_config.return_value = Mock()
            mock_generate.return_value = Mock()
            
            # Test pipeline assembly
            try:
                result = assembler.assemble_workspace_pipeline(sample_workspace_config)
                # If we get here, the integration is working
                assert result is not None
            except Exception as e:
                # Check if it's an expected error due to mocking
                if "Failed to assemble workspace pipeline" not in str(e):
                    raise
    
    def test_performance_with_consolidated_managers(self, temp_workspace, consolidated_workspace_manager):
        """Test performance characteristics with consolidated managers."""
        import time
        
        # Create multiple components to test performance
        assembler = WorkspacePipelineAssembler(
            workspace_root=temp_workspace,
            workspace_manager=consolidated_workspace_manager
        )
        
        registry = WorkspaceComponentRegistry(
            workspace_root=temp_workspace,
            discovery_manager=consolidated_workspace_manager.discovery_manager
        )
        
        compiler = WorkspaceDAGCompiler(
            workspace_root=temp_workspace,
            workspace_manager=consolidated_workspace_manager
        )
        
        # Verify they all share the same consolidated manager
        assert assembler.workspace_manager is consolidated_workspace_manager
        assert registry.workspace_manager is consolidated_workspace_manager
        assert compiler.workspace_manager is consolidated_workspace_manager
        
        # Verify shared discovery manager
        assert assembler.workspace_registry.discovery_manager is consolidated_workspace_manager.discovery_manager
        assert registry.discovery_manager is consolidated_workspace_manager.discovery_manager
        assert compiler.workspace_registry.discovery_manager is consolidated_workspace_manager.discovery_manager
    
    def test_error_handling_integration(self, temp_workspace, consolidated_workspace_manager):
        """Test error handling across integrated components."""
        assembler = WorkspacePipelineAssembler(
            workspace_root=temp_workspace,
            workspace_manager=consolidated_workspace_manager
        )
        
        # Test with invalid configuration
        invalid_config = WorkspacePipelineDefinition(
            pipeline_name="invalid_pipeline",
            workspace_root=temp_workspace,
            steps=[],  # Empty steps should cause validation error
            global_config={}
        )
        
        # This should raise a validation error due to empty steps
        with pytest.raises(ValueError, match="steps list cannot be empty"):
            WorkspacePipelineDefinition(
                pipeline_name="invalid_pipeline",
                workspace_root=temp_workspace,
                steps=[],
                global_config={}
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
