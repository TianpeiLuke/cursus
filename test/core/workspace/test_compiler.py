"""
Unit tests for workspace DAG compiler.

Tests the WorkspaceDAGCompiler for workspace-aware DAG compilation.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.cursus.core.workspace.compiler import WorkspaceDAGCompiler
from src.cursus.core.workspace.config import WorkspaceStepDefinition, WorkspacePipelineDefinition
from src.cursus.api.dag.workspace_dag import WorkspaceAwareDAG


class TestWorkspaceDAGCompiler:
    """Test cases for WorkspaceDAGCompiler."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def sample_workspace_dag(self, temp_workspace):
        """Create a sample workspace DAG."""
        dag = WorkspaceAwareDAG(workspace_root=temp_workspace)
        
        # Add workspace steps
        dag.add_workspace_step(
            step_name='preprocessing',
            developer_id='dev1',
            step_type='DataPreprocessing',
            config_data={'input_path': '/data/input'},
            dependencies=[]
        )
        
        dag.add_workspace_step(
            step_name='training',
            developer_id='dev2',
            step_type='XGBoostTraining',
            config_data={'model_params': {'max_depth': 6}},
            dependencies=['preprocessing']
        )
        
        return dag
    
    @pytest.fixture
    def mock_workspace_registry(self):
        """Create a mock workspace registry."""
        mock_registry = Mock()
        mock_registry.validate_component_availability.return_value = {
            'valid': True,
            'compilation_ready': True,
            'missing_components': [],
            'available_components': [
                {'step_name': 'preprocessing', 'component_type': 'builder'},
                {'step_name': 'training', 'component_type': 'builder'}
            ]
        }
        mock_registry.get_workspace_summary.return_value = {
            'workspace_root': '/test/workspace',
            'total_components': 4
        }
        return mock_registry
    
    @patch('src.cursus.core.workspace.compiler.WorkspaceComponentRegistry')
    def test_compiler_initialization(self, mock_registry_class, temp_workspace):
        """Test compiler initialization."""
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry
        
        compiler = WorkspaceDAGCompiler(workspace_root=temp_workspace)
        
        assert compiler.workspace_root == temp_workspace
        assert compiler.workspace_registry == mock_registry
        mock_registry_class.assert_called_once_with(temp_workspace)
    
    @patch('src.cursus.core.workspace.compiler.WorkspaceComponentRegistry')
    @patch('src.cursus.core.workspace.compiler.WorkspacePipelineAssembler')
    def test_compile_workspace_dag_success(self, mock_assembler_class, mock_registry_class, 
                                         temp_workspace, sample_workspace_dag):
        """Test successful workspace DAG compilation."""
        # Setup mocks
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry
        
        mock_assembler = Mock()
        mock_pipeline = Mock()
        mock_pipeline.name = 'test_pipeline'
        mock_assembler.assemble_workspace_pipeline.return_value = mock_pipeline
        mock_assembler.get_workspace_summary.return_value = {'test': 'summary'}
        mock_assembler_class.return_value = mock_assembler
        
        compiler = WorkspaceDAGCompiler(workspace_root=temp_workspace)
        
        # Mock DAG methods
        with patch.object(sample_workspace_dag, 'to_workspace_pipeline_config') as mock_to_config, \
             patch.object(sample_workspace_dag, 'validate_workspace_dependencies') as mock_validate_deps, \
             patch.object(sample_workspace_dag, 'analyze_workspace_complexity') as mock_analyze, \
             patch.object(sample_workspace_dag, 'get_developers') as mock_get_devs:
            
            mock_to_config.return_value = {
                'pipeline_name': 'workspace_pipeline',
                'workspace_root': temp_workspace,
                'steps': [
                    {
                        'step_name': 'test_step',
                        'developer_id': 'dev1',
                        'step_type': 'TestStep',
                        'config_data': {},
                        'workspace_root': temp_workspace
                    }
                ],
                'global_config': {}
            }
            mock_validate_deps.return_value = {'valid': True}
            mock_analyze.return_value = {'basic_metrics': {'node_count': 2}}
            mock_get_devs.return_value = ['dev1', 'dev2']
            
            pipeline, metadata = compiler.compile_workspace_dag(sample_workspace_dag)
            
            assert pipeline == mock_pipeline
            assert 'workspace_root' in metadata
            assert 'compilation_time' in metadata
            assert 'step_count' in metadata
            assert 'developer_count' in metadata
            assert metadata['step_count'] == 2
            assert metadata['developer_count'] == 2
    
    @patch('src.cursus.core.workspace.compiler.WorkspaceComponentRegistry')
    @patch('src.cursus.core.workspace.compiler.WorkspacePipelineAssembler')
    def test_compile_workspace_dag_failure(self, mock_assembler_class, mock_registry_class,
                                         temp_workspace, sample_workspace_dag):
        """Test workspace DAG compilation failure."""
        mock_registry_class.return_value = Mock()
        
        # Mock assembler to raise exception
        mock_assembler = Mock()
        mock_assembler.assemble_workspace_pipeline.side_effect = Exception("Assembly failed")
        mock_assembler_class.return_value = mock_assembler
        
        compiler = WorkspaceDAGCompiler(workspace_root=temp_workspace)
        
        with patch.object(sample_workspace_dag, 'to_workspace_pipeline_config') as mock_to_config:
            mock_to_config.return_value = {
                'pipeline_name': 'workspace_pipeline',
                'workspace_root': temp_workspace,
                'steps': [],
                'global_config': {}
            }
            
            with pytest.raises(ValueError, match="Failed to compile workspace DAG"):
                compiler.compile_workspace_dag(sample_workspace_dag)
    
    @patch('src.cursus.core.workspace.compiler.WorkspaceComponentRegistry')
    @patch('src.cursus.core.workspace.compiler.WorkspacePipelineAssembler')
    def test_preview_workspace_resolution(self, mock_assembler_class, mock_registry_class,
                                        temp_workspace, sample_workspace_dag):
        """Test previewing workspace resolution."""
        mock_registry_class.return_value = Mock()
        
        # Mock assembler preview
        mock_assembler = Mock()
        mock_assembler.preview_workspace_assembly.return_value = {
            'validation_results': {'overall_valid': True, 'valid': True},
            'component_resolution': {'dev1:preprocessing': {}, 'dev2:training': {}},
            'assembly_plan': {'build_order': ['preprocessing', 'training']}
        }
        mock_assembler_class.return_value = mock_assembler
        
        compiler = WorkspaceDAGCompiler(workspace_root=temp_workspace)
        
        with patch.object(sample_workspace_dag, 'to_workspace_pipeline_config') as mock_to_config, \
             patch.object(sample_workspace_dag, 'get_workspace_summary') as mock_summary:
            
            mock_to_config.return_value = {
                'pipeline_name': 'preview_pipeline',
                'workspace_root': temp_workspace,
                'steps': [
                    {
                        'step_name': 'test_step',
                        'developer_id': 'dev1',
                        'step_type': 'TestStep',
                        'config_data': {},
                        'workspace_root': temp_workspace
                    }
                ],
                'global_config': {}
            }
            mock_summary.return_value = {'total_steps': 2}
            
            preview = compiler.preview_workspace_resolution(sample_workspace_dag)
            
            assert 'dag_summary' in preview
            assert 'component_resolution' in preview
            assert 'validation_results' in preview
            assert 'compilation_feasibility' in preview
            assert preview['compilation_feasibility']['can_compile'] is True
    
    def test_estimate_compilation_time(self, temp_workspace, sample_workspace_dag):
        """Test compilation time estimation."""
        with patch('src.cursus.core.workspace.compiler.WorkspaceComponentRegistry'):
            compiler = WorkspaceDAGCompiler(workspace_root=temp_workspace)
            
            # Mock DAG methods
            with patch.object(sample_workspace_dag, 'get_developers') as mock_get_devs:
                mock_get_devs.return_value = ['dev1', 'dev2']
                
                estimated_time = compiler._estimate_compilation_time(sample_workspace_dag)
                
                assert isinstance(estimated_time, float)
                assert estimated_time > 0
    
    @patch('src.cursus.core.workspace.compiler.WorkspaceComponentRegistry')
    def test_validate_workspace_components(self, mock_registry_class, temp_workspace, sample_workspace_dag):
        """Test workspace component validation."""
        mock_registry = Mock()
        mock_registry.validate_component_availability.return_value = {
            'valid': True,
            'missing_components': [],
            'available_components': []
        }
        mock_registry_class.return_value = mock_registry
        
        compiler = WorkspaceDAGCompiler(workspace_root=temp_workspace)
        
        with patch.object(sample_workspace_dag, 'to_workspace_pipeline_config') as mock_to_config, \
             patch.object(sample_workspace_dag, 'validate_workspace_dependencies') as mock_validate:
            
            mock_to_config.return_value = {
                'pipeline_name': 'validation_pipeline',
                'workspace_root': temp_workspace,
                'steps': [
                    {
                        'step_name': 'test_step',
                        'developer_id': 'dev1',
                        'step_type': 'TestStep',
                        'config_data': {},
                        'workspace_root': temp_workspace
                    }
                ],
                'global_config': {}
            }
            mock_validate.return_value = {'valid': True}
            
            result = compiler.validate_workspace_components(sample_workspace_dag)
            
            assert result['valid'] is True
            assert result['compilation_ready'] is True
            assert 'dag_validation' in result
    
    @patch('src.cursus.core.workspace.compiler.WorkspaceComponentRegistry')
    def test_validate_workspace_components_failure(self, mock_registry_class, temp_workspace, sample_workspace_dag):
        """Test workspace component validation failure."""
        mock_registry_class.return_value = Mock()
        
        compiler = WorkspaceDAGCompiler(workspace_root=temp_workspace)
        
        # Mock to_workspace_pipeline_config to raise exception
        with patch.object(sample_workspace_dag, 'to_workspace_pipeline_config') as mock_to_config:
            mock_to_config.side_effect = Exception("Config conversion failed")
            
            result = compiler.validate_workspace_components(sample_workspace_dag)
            
            assert result['valid'] is False
            assert result['compilation_ready'] is False
            assert 'error' in result
    
    @patch('src.cursus.core.workspace.compiler.WorkspaceComponentRegistry')
    def test_generate_compilation_report(self, mock_registry_class, temp_workspace, sample_workspace_dag):
        """Test generating compilation report."""
        mock_registry_class.return_value = Mock()
        
        compiler = WorkspaceDAGCompiler(workspace_root=temp_workspace)
        
        # Mock all required methods
        with patch.object(sample_workspace_dag, 'get_workspace_summary') as mock_summary, \
             patch.object(sample_workspace_dag, 'analyze_workspace_complexity') as mock_analyze, \
             patch.object(compiler, 'validate_workspace_components') as mock_validate, \
             patch.object(compiler, 'preview_workspace_resolution') as mock_preview:
            
            mock_summary.return_value = {'total_steps': 2}
            mock_analyze.return_value = {
                'basic_metrics': {
                    'node_count': 2,
                    'developer_count': 2
                },
                'complexity_metrics': {
                    'avg_dependencies_per_step': 1.0
                }
            }
            mock_validate.return_value = {
                'compilation_ready': True,
                'missing_components': [],
                'available_components': [{'step': 'test'}],
                'warnings': []
            }
            mock_preview.return_value = {'test': 'preview'}
            
            report = compiler.generate_compilation_report(sample_workspace_dag)
            
            assert 'dag_analysis' in report
            assert 'complexity_analysis' in report
            assert 'component_validation' in report
            assert 'compilation_preview' in report
            assert 'recommendations' in report
            assert 'estimated_resources' in report
            assert 'validation_summary' in report
    
    @patch('src.cursus.core.workspace.compiler.WorkspaceComponentRegistry')
    def test_from_workspace_config(self, mock_registry_class, temp_workspace):
        """Test creating compiler from workspace configuration."""
        mock_registry_class.return_value = Mock()
        
        # Create a valid workspace config with at least one step
        step = WorkspaceStepDefinition(
            step_name='test_step',
            developer_id='dev1',
            step_type='TestStep',
            config_data={},
            workspace_root=temp_workspace
        )
        
        workspace_config = WorkspacePipelineDefinition(
            pipeline_name='test_pipeline',
            workspace_root=temp_workspace,
            steps=[step]
        )
        
        compiler = WorkspaceDAGCompiler.from_workspace_config(
            workspace_config=workspace_config,
            role='test-role'
        )
        
        assert isinstance(compiler, WorkspaceDAGCompiler)
        assert compiler.workspace_root == temp_workspace
    
    @patch('src.cursus.core.workspace.compiler.WorkspaceComponentRegistry')
    def test_from_workspace_dag(self, mock_registry_class, temp_workspace, sample_workspace_dag):
        """Test creating compiler from workspace DAG."""
        mock_registry_class.return_value = Mock()
        
        compiler = WorkspaceDAGCompiler.from_workspace_dag(
            workspace_dag=sample_workspace_dag,
            role='test-role'
        )
        
        assert isinstance(compiler, WorkspaceDAGCompiler)
        assert compiler.workspace_root == temp_workspace
    
    @patch('src.cursus.core.workspace.compiler.WorkspaceComponentRegistry')
    def test_compile_with_detailed_report_success(self, mock_registry_class, temp_workspace, sample_workspace_dag):
        """Test compilation with detailed report - success case."""
        mock_registry_class.return_value = Mock()
        
        compiler = WorkspaceDAGCompiler(workspace_root=temp_workspace)
        
        # Mock all required methods
        with patch.object(compiler, 'generate_compilation_report') as mock_report, \
             patch.object(compiler, 'compile_workspace_dag') as mock_compile:
            
            mock_report.return_value = {
                'component_validation': {'compilation_ready': True}
            }
            
            mock_pipeline = Mock()
            mock_pipeline.name = 'test_pipeline'
            mock_pipeline.steps = [Mock(), Mock()]
            mock_pipeline.parameters = [Mock(name='param1')]
            
            mock_compile.return_value = (mock_pipeline, {
                'compilation_time': 10.5,
                'step_count': 2,
                'developer_count': 2
            })
            
            pipeline, detailed_report = compiler.compile_with_detailed_report(sample_workspace_dag)
            
            assert pipeline == mock_pipeline
            assert 'pre_compilation_analysis' in detailed_report
            assert 'compilation_metadata' in detailed_report
            assert 'pipeline_info' in detailed_report
            assert 'success' in detailed_report
            assert 'compilation_summary' in detailed_report
            assert detailed_report['success'] is True
    
    @patch('src.cursus.core.workspace.compiler.WorkspaceComponentRegistry')
    def test_compile_with_detailed_report_not_ready(self, mock_registry_class, temp_workspace, sample_workspace_dag):
        """Test compilation with detailed report - not ready case."""
        mock_registry_class.return_value = Mock()
        
        compiler = WorkspaceDAGCompiler(workspace_root=temp_workspace)
        
        with patch.object(compiler, 'generate_compilation_report') as mock_report:
            mock_report.return_value = {
                'component_validation': {'compilation_ready': False}
            }
            
            with pytest.raises(ValueError, match="Workspace DAG is not ready for compilation"):
                compiler.compile_with_detailed_report(sample_workspace_dag)
    
    @patch('src.cursus.core.workspace.compiler.WorkspaceComponentRegistry')
    def test_get_workspace_summary(self, mock_registry_class, temp_workspace):
        """Test getting workspace summary."""
        mock_registry = Mock()
        mock_registry.get_workspace_summary.return_value = {
            'workspace_root': temp_workspace,
            'total_components': 4
        }
        mock_registry_class.return_value = mock_registry
        
        compiler = WorkspaceDAGCompiler(workspace_root=temp_workspace)
        summary = compiler.get_workspace_summary()
        
        assert summary['workspace_root'] == temp_workspace
        assert 'registry_summary' in summary
        assert 'compiler_capabilities' in summary
        
        capabilities = summary['compiler_capabilities']
        assert capabilities['supports_workspace_dags'] is True
        assert capabilities['supports_cross_workspace_dependencies'] is True
        assert capabilities['supports_component_validation'] is True
        assert capabilities['supports_compilation_preview'] is True
        assert capabilities['supports_detailed_reporting'] is True
    
    @patch('src.cursus.core.workspace.compiler.WorkspaceComponentRegistry')
    def test_preview_with_blocking_issues(self, mock_registry_class, temp_workspace, sample_workspace_dag):
        """Test preview with blocking issues identified."""
        mock_registry_class.return_value = Mock()
        
        compiler = WorkspaceDAGCompiler(workspace_root=temp_workspace)
        
        with patch('src.cursus.core.workspace.compiler.WorkspacePipelineAssembler') as mock_assembler_class:
            mock_assembler = Mock()
            mock_assembler.preview_workspace_assembly.return_value = {
                'validation_results': {
                    'overall_valid': False,
                    'valid': False,
                    'missing_components': [{'step_name': 'missing_step'}],
                    'workspace_validation': {
                        'dependency_validation': {
                            'valid': False,
                            'errors': ['Circular dependency detected'],
                            'warnings': ['High complexity']
                        }
                    }
                }
            }
            mock_assembler_class.return_value = mock_assembler
            
            with patch.object(sample_workspace_dag, 'to_workspace_pipeline_config') as mock_to_config, \
                 patch.object(sample_workspace_dag, 'get_workspace_summary') as mock_summary:
                
                mock_to_config.return_value = {
                    'pipeline_name': 'preview_pipeline',
                    'workspace_root': temp_workspace,
                    'steps': [
                        {
                            'step_name': 'test_step',
                            'developer_id': 'dev1',
                            'step_type': 'TestStep',
                            'config_data': {},
                            'workspace_root': temp_workspace
                        }
                    ],
                    'global_config': {}
                }
                mock_summary.return_value = {'total_steps': 2}
                
                preview = compiler.preview_workspace_resolution(sample_workspace_dag)
                
                assert preview['compilation_feasibility']['can_compile'] is False
                assert len(preview['compilation_feasibility']['blocking_issues']) > 0
                assert len(preview['compilation_feasibility']['warnings']) > 0


if __name__ == '__main__':
    pytest.main([__file__])
