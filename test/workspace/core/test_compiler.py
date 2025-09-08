"""
Unit tests for workspace DAG compiler.

Tests the WorkspaceDAGCompiler for workspace-aware DAG compilation.
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from cursus.workspace.core.compiler import WorkspaceDAGCompiler
from cursus.workspace.core.config import WorkspaceStepDefinition, WorkspacePipelineDefinition
from cursus.api.dag.workspace_dag import WorkspaceAwareDAG

class TestWorkspaceDAGCompiler(unittest.TestCase):
    """Test cases for WorkspaceDAGCompiler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_workspace = self.temp_dir
        
        # Create sample workspace DAG
        self.sample_workspace_dag = WorkspaceAwareDAG(workspace_root=self.temp_workspace)
        
        # Add workspace steps
        self.sample_workspace_dag.add_workspace_step(
            step_name='preprocessing',
            developer_id='dev1',
            step_type='DataPreprocessing',
            config_data={'input_path': '/data/input'},
            dependencies=[]
        )
        
        self.sample_workspace_dag.add_workspace_step(
            step_name='training',
            developer_id='dev2',
            step_type='XGBoostTraining',
            config_data={'model_params': {'max_depth': 6}},
            dependencies=['preprocessing']
        )
        
        # Create mock workspace registry
        self.mock_workspace_registry = Mock()
        self.mock_workspace_registry.validate_component_availability.return_value = {
            'valid': True,
            'compilation_ready': True,
            'missing_components': [],
            'available_components': [
                {'step_name': 'preprocessing', 'component_type': 'builder'},
                {'step_name': 'training', 'component_type': 'builder'}
            ]
        }
        self.mock_workspace_registry.get_workspace_summary.return_value = {
            'workspace_root': '/test/workspace',
            'total_components': 4
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_compiler_initialization_with_workspace_manager(self):
        """Test compiler initialization with workspace manager (Phase 2 optimization)."""
        mock_manager = Mock()
        
        compiler = WorkspaceDAGCompiler(
            workspace_root=self.temp_workspace,
            workspace_manager=mock_manager
        )
        
        self.assertEqual(compiler.workspace_root, self.temp_workspace)
        self.assertEqual(compiler.workspace_manager, mock_manager)
        self.assertIsNotNone(compiler.workspace_registry)
    
    @patch('cursus.workspace.core.manager.WorkspaceManager')
    def test_compiler_initialization_without_workspace_manager(self, mock_manager_class):
        """Test compiler initialization without workspace manager."""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        compiler = WorkspaceDAGCompiler(workspace_root=self.temp_workspace)
        
        self.assertEqual(compiler.workspace_root, self.temp_workspace)
        self.assertEqual(compiler.workspace_manager, mock_manager)
        self.assertIsNotNone(compiler.workspace_registry)
        mock_manager_class.assert_called_once_with(self.temp_workspace)
    
    @patch('cursus.workspace.core.manager.WorkspaceManager')
    def test_compile_workspace_dag_success(self, mock_manager_class):
        """Test successful workspace DAG compilation."""
        # Setup mocks
        mock_manager_class.return_value = Mock()
        
        compiler = WorkspaceDAGCompiler(workspace_root=self.temp_workspace)
        
        # Mock DAG methods and patch the assembler creation inside the compiler
        with patch.object(self.sample_workspace_dag, 'to_workspace_pipeline_config') as mock_to_config, \
             patch.object(self.sample_workspace_dag, 'validate_workspace_dependencies') as mock_validate_deps, \
             patch.object(self.sample_workspace_dag, 'analyze_workspace_complexity') as mock_analyze, \
             patch.object(self.sample_workspace_dag, 'get_developers') as mock_get_devs, \
             patch.object(compiler, 'WorkspacePipelineAssembler', create=True) as mock_assembler_class:
            
            # Setup assembler mock
            mock_assembler = Mock()
            mock_pipeline = Mock()
            mock_pipeline.name = 'test_pipeline'
            mock_assembler.assemble_workspace_pipeline.return_value = mock_pipeline
            mock_assembler.get_workspace_summary.return_value = {'test': 'summary'}
            mock_assembler_class.return_value = mock_assembler
            
            mock_to_config.return_value = {
                'pipeline_name': 'workspace_pipeline',
                'workspace_root': self.temp_workspace,
                'steps': [
                    {
                        'step_name': 'test_step',
                        'developer_id': 'dev1',
                        'step_type': 'TestStep',
                        'config_data': {},
                        'workspace_root': self.temp_workspace
                    }
                ],
                'global_config': {}
            }
            mock_validate_deps.return_value = {'valid': True}
            mock_analyze.return_value = {'basic_metrics': {'node_count': 2}}
            mock_get_devs.return_value = ['dev1', 'dev2']
            
            # Patch the assembler import at the module level where it's used
            with patch('cursus.workspace.core.compiler.WorkspacePipelineAssembler', mock_assembler_class):
                pipeline, metadata = compiler.compile_workspace_dag(self.sample_workspace_dag)
            
            self.assertEqual(pipeline, mock_pipeline)
            self.assertIn('workspace_root', metadata)
            self.assertIn('compilation_time', metadata)
            self.assertIn('step_count', metadata)
            self.assertIn('developer_count', metadata)
            self.assertEqual(metadata['step_count'], 2)
            self.assertEqual(metadata['developer_count'], 2)
    
    @patch('cursus.workspace.core.registry.WorkspaceComponentRegistry')
    @patch('cursus.workspace.core.manager.WorkspaceManager')
    @patch('cursus.workspace.core.assembler.WorkspacePipelineAssembler')
    def test_compile_workspace_dag_failure(self, mock_assembler_class, mock_manager_class, mock_registry_class):
        """Test workspace DAG compilation failure."""
        mock_registry_class.return_value = Mock()
        mock_manager_class.return_value = Mock()
        
        # Mock assembler to raise exception
        mock_assembler = Mock()
        mock_assembler.assemble_workspace_pipeline.side_effect = Exception("Assembly failed")
        mock_assembler_class.return_value = mock_assembler
        
        compiler = WorkspaceDAGCompiler(workspace_root=self.temp_workspace)
        
        with patch.object(self.sample_workspace_dag, 'to_workspace_pipeline_config') as mock_to_config:
            mock_to_config.return_value = {
                'pipeline_name': 'workspace_pipeline',
                'workspace_root': self.temp_workspace,
                'steps': [],
                'global_config': {}
            }
            
            with self.assertRaises(ValueError) as context:
                compiler.compile_workspace_dag(self.sample_workspace_dag)
            
            self.assertIn("Failed to compile workspace DAG", str(context.exception))
    
    @patch('cursus.workspace.core.manager.WorkspaceManager')
    def test_preview_workspace_resolution(self, mock_manager_class):
        """Test previewing workspace resolution."""
        mock_manager_class.return_value = Mock()
        
        compiler = WorkspaceDAGCompiler(workspace_root=self.temp_workspace)
        
        with patch.object(self.sample_workspace_dag, 'to_workspace_pipeline_config') as mock_to_config, \
             patch.object(self.sample_workspace_dag, 'get_workspace_summary') as mock_summary, \
             patch('cursus.workspace.core.compiler.WorkspacePipelineAssembler') as mock_assembler_class:
            
            # Mock assembler preview
            mock_assembler = Mock()
            mock_assembler.preview_workspace_assembly.return_value = {
                'validation_results': {'overall_valid': True, 'valid': True},
                'component_resolution': {'dev1:preprocessing': {}, 'dev2:training': {}},
                'assembly_plan': {'build_order': ['preprocessing', 'training']}
            }
            mock_assembler_class.return_value = mock_assembler
            
            mock_to_config.return_value = {
                'pipeline_name': 'preview_pipeline',
                'workspace_root': self.temp_workspace,
                'steps': [
                    {
                        'step_name': 'test_step',
                        'developer_id': 'dev1',
                        'step_type': 'TestStep',
                        'config_data': {},
                        'workspace_root': self.temp_workspace
                    }
                ],
                'global_config': {}
            }
            mock_summary.return_value = {'total_steps': 2}
            
            preview = compiler.preview_workspace_resolution(self.sample_workspace_dag)
            
            self.assertIn('dag_summary', preview)
            self.assertIn('component_resolution', preview)
            self.assertIn('validation_results', preview)
            self.assertIn('compilation_feasibility', preview)
            self.assertTrue(preview['compilation_feasibility']['can_compile'])
    
    @patch('cursus.workspace.core.registry.WorkspaceComponentRegistry')
    @patch('cursus.workspace.core.manager.WorkspaceManager')
    def test_estimate_compilation_time(self, mock_manager_class, mock_registry_class):
        """Test compilation time estimation."""
        mock_registry_class.return_value = Mock()
        mock_manager_class.return_value = Mock()
        
        compiler = WorkspaceDAGCompiler(workspace_root=self.temp_workspace)
        
        # Mock DAG methods
        with patch.object(self.sample_workspace_dag, 'get_developers') as mock_get_devs:
            mock_get_devs.return_value = ['dev1', 'dev2']
            
            estimated_time = compiler._estimate_compilation_time(self.sample_workspace_dag)
            
            self.assertIsInstance(estimated_time, float)
            self.assertGreater(estimated_time, 0)
    
    @patch('cursus.workspace.core.manager.WorkspaceManager')
    def test_validate_workspace_components(self, mock_manager_class):
        """Test workspace component validation."""
        mock_registry = Mock()
        mock_registry.validate_component_availability.return_value = {
            'valid': True,
            'missing_components': [],
            'available_components': []
        }
        mock_manager_class.return_value = Mock()
        
        compiler = WorkspaceDAGCompiler(workspace_root=self.temp_workspace)
        
        # Replace the registry instance with our mock
        compiler.workspace_registry = mock_registry
        
        with patch.object(self.sample_workspace_dag, 'to_workspace_pipeline_config') as mock_to_config, \
             patch.object(self.sample_workspace_dag, 'validate_workspace_dependencies') as mock_validate:
            
            mock_to_config.return_value = {
                'pipeline_name': 'validation_pipeline',
                'workspace_root': self.temp_workspace,
                'steps': [
                    {
                        'step_name': 'test_step',
                        'developer_id': 'dev1',
                        'step_type': 'TestStep',
                        'config_data': {},
                        'workspace_root': self.temp_workspace
                    }
                ],
                'global_config': {}
            }
            mock_validate.return_value = {'valid': True}
            
            result = compiler.validate_workspace_components(self.sample_workspace_dag)
            
            self.assertTrue(result['valid'])
            self.assertTrue(result['compilation_ready'])
            self.assertIn('dag_validation', result)
    
    @patch('cursus.workspace.core.registry.WorkspaceComponentRegistry')
    @patch('cursus.workspace.core.manager.WorkspaceManager')
    def test_validate_workspace_components_failure(self, mock_manager_class, mock_registry_class):
        """Test workspace component validation failure."""
        mock_registry_class.return_value = Mock()
        mock_manager_class.return_value = Mock()
        
        compiler = WorkspaceDAGCompiler(workspace_root=self.temp_workspace)
        
        # Mock to_workspace_pipeline_config to raise exception
        with patch.object(self.sample_workspace_dag, 'to_workspace_pipeline_config') as mock_to_config:
            mock_to_config.side_effect = Exception("Config conversion failed")
            
            result = compiler.validate_workspace_components(self.sample_workspace_dag)
            
            self.assertFalse(result['valid'])
            self.assertFalse(result['compilation_ready'])
            self.assertIn('error', result)
    
    @patch('cursus.workspace.core.registry.WorkspaceComponentRegistry')
    @patch('cursus.workspace.core.manager.WorkspaceManager')
    def test_generate_compilation_report(self, mock_manager_class, mock_registry_class):
        """Test generating compilation report."""
        mock_registry_class.return_value = Mock()
        mock_manager_class.return_value = Mock()
        
        compiler = WorkspaceDAGCompiler(workspace_root=self.temp_workspace)
        
        # Mock all required methods
        with patch.object(self.sample_workspace_dag, 'get_workspace_summary') as mock_summary, \
             patch.object(self.sample_workspace_dag, 'analyze_workspace_complexity') as mock_analyze, \
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
            
            report = compiler.generate_compilation_report(self.sample_workspace_dag)
            
            self.assertIn('dag_analysis', report)
            self.assertIn('complexity_analysis', report)
            self.assertIn('component_validation', report)
            self.assertIn('compilation_preview', report)
            self.assertIn('recommendations', report)
            self.assertIn('estimated_resources', report)
            self.assertIn('validation_summary', report)
    
    @patch('cursus.workspace.core.registry.WorkspaceComponentRegistry')
    @patch('cursus.workspace.core.manager.WorkspaceManager')
    def test_from_workspace_config(self, mock_manager_class, mock_registry_class):
        """Test creating compiler from workspace configuration."""
        mock_registry_class.return_value = Mock()
        mock_manager_class.return_value = Mock()
        
        # Create a valid workspace config with at least one step
        step = WorkspaceStepDefinition(
            step_name='test_step',
            developer_id='dev1',
            step_type='TestStep',
            config_data={},
            workspace_root=self.temp_workspace
        )
        
        workspace_config = WorkspacePipelineDefinition(
            pipeline_name='test_pipeline',
            workspace_root=self.temp_workspace,
            steps=[step]
        )
        
        compiler = WorkspaceDAGCompiler.from_workspace_config(
            workspace_config=workspace_config,
            role='test-role'
        )
        
        self.assertIsInstance(compiler, WorkspaceDAGCompiler)
        self.assertEqual(compiler.workspace_root, self.temp_workspace)
    
    @patch('cursus.workspace.core.registry.WorkspaceComponentRegistry')
    @patch('cursus.workspace.core.manager.WorkspaceManager')
    def test_from_workspace_dag(self, mock_manager_class, mock_registry_class):
        """Test creating compiler from workspace DAG."""
        mock_registry_class.return_value = Mock()
        mock_manager_class.return_value = Mock()
        
        compiler = WorkspaceDAGCompiler.from_workspace_dag(
            workspace_dag=self.sample_workspace_dag,
            role='test-role'
        )
        
        self.assertIsInstance(compiler, WorkspaceDAGCompiler)
        self.assertEqual(compiler.workspace_root, self.temp_workspace)
    
    @patch('cursus.workspace.core.registry.WorkspaceComponentRegistry')
    @patch('cursus.workspace.core.manager.WorkspaceManager')
    def test_compile_with_detailed_report_success(self, mock_manager_class, mock_registry_class):
        """Test compilation with detailed report - success case."""
        mock_registry_class.return_value = Mock()
        mock_manager_class.return_value = Mock()
        
        compiler = WorkspaceDAGCompiler(workspace_root=self.temp_workspace)
        
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
            
            pipeline, detailed_report = compiler.compile_with_detailed_report(self.sample_workspace_dag)
            
            self.assertEqual(pipeline, mock_pipeline)
            self.assertIn('pre_compilation_analysis', detailed_report)
            self.assertIn('compilation_metadata', detailed_report)
            self.assertIn('pipeline_info', detailed_report)
            self.assertIn('success', detailed_report)
            self.assertIn('compilation_summary', detailed_report)
            self.assertTrue(detailed_report['success'])
    
    @patch('cursus.workspace.core.registry.WorkspaceComponentRegistry')
    @patch('cursus.workspace.core.manager.WorkspaceManager')
    def test_compile_with_detailed_report_not_ready(self, mock_manager_class, mock_registry_class):
        """Test compilation with detailed report - not ready case."""
        mock_registry_class.return_value = Mock()
        mock_manager_class.return_value = Mock()
        
        compiler = WorkspaceDAGCompiler(workspace_root=self.temp_workspace)
        
        with patch.object(compiler, 'generate_compilation_report') as mock_report:
            mock_report.return_value = {
                'component_validation': {'compilation_ready': False}
            }
            
            with self.assertRaises(ValueError) as context:
                compiler.compile_with_detailed_report(self.sample_workspace_dag)
            
            self.assertIn("Workspace DAG is not ready for compilation", str(context.exception))
    
    @patch('cursus.workspace.core.registry.WorkspaceComponentRegistry')
    @patch('cursus.workspace.core.manager.WorkspaceManager')
    def test_get_workspace_summary(self, mock_manager_class, mock_registry_class):
        """Test getting workspace summary."""
        mock_registry = Mock()
        mock_registry.get_workspace_summary.return_value = {
            'workspace_root': self.temp_workspace,
            'total_components': 4
        }
        mock_registry_class.return_value = mock_registry
        mock_manager_class.return_value = Mock()
        
        compiler = WorkspaceDAGCompiler(workspace_root=self.temp_workspace)
        summary = compiler.get_workspace_summary()
        
        self.assertEqual(summary['workspace_root'], self.temp_workspace)
        self.assertIn('registry_summary', summary)
        self.assertIn('compiler_capabilities', summary)
        
        capabilities = summary['compiler_capabilities']
        self.assertTrue(capabilities['supports_workspace_dags'])
        self.assertTrue(capabilities['supports_cross_workspace_dependencies'])
        self.assertTrue(capabilities['supports_component_validation'])
        self.assertTrue(capabilities['supports_compilation_preview'])
        self.assertTrue(capabilities['supports_detailed_reporting'])
    
    @patch('cursus.workspace.core.registry.WorkspaceComponentRegistry')
    @patch('cursus.workspace.core.manager.WorkspaceManager')
    def test_preview_with_blocking_issues(self, mock_manager_class, mock_registry_class):
        """Test preview with blocking issues identified."""
        mock_registry_class.return_value = Mock()
        mock_manager_class.return_value = Mock()
        
        compiler = WorkspaceDAGCompiler(workspace_root=self.temp_workspace)
        
        with patch('cursus.workspace.core.compiler.WorkspacePipelineAssembler') as mock_assembler_class:
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
            
            with patch.object(self.sample_workspace_dag, 'to_workspace_pipeline_config') as mock_to_config, \
                 patch.object(self.sample_workspace_dag, 'get_workspace_summary') as mock_summary:
                
                mock_to_config.return_value = {
                    'pipeline_name': 'preview_pipeline',
                    'workspace_root': self.temp_workspace,
                    'steps': [
                        {
                            'step_name': 'test_step',
                            'developer_id': 'dev1',
                            'step_type': 'TestStep',
                            'config_data': {},
                            'workspace_root': self.temp_workspace
                        }
                    ],
                    'global_config': {}
                }
                mock_summary.return_value = {'total_steps': 2}
                
                preview = compiler.preview_workspace_resolution(self.sample_workspace_dag)
                
                self.assertFalse(preview['compilation_feasibility']['can_compile'])
                self.assertGreater(len(preview['compilation_feasibility']['blocking_issues']), 0)
                self.assertGreater(len(preview['compilation_feasibility']['warnings']), 0)

if __name__ == '__main__':
    unittest.main()
