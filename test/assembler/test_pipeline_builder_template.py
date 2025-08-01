import unittest
from unittest.mock import patch, MagicMock, ANY
from pathlib import Path
from collections import defaultdict

# Add the project root to the Python path to allow for absolute imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pipeline_builder.pipeline_builder_template import PipelineBuilderTemplate
from src.pipeline_builder.pipeline_dag import PipelineDAG
from src.pipeline_steps.config_base import BasePipelineConfig
from src.pipeline_steps.builder_step_base import StepBuilderBase


class TestPipelineBuilderTemplate(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Mock PipelineDAG
        self.mock_dag = MagicMock(spec=PipelineDAG)
        self.mock_dag.nodes = ['step1', 'step2', 'step3']
        self.mock_dag.edges = [('step1', 'step2'), ('step2', 'step3')]
        self.mock_dag.topological_sort.return_value = ['step1', 'step2', 'step3']
        self.mock_dag.get_dependencies.side_effect = lambda node: {
            'step1': [],
            'step2': ['step1'],
            'step3': ['step2']
        }[node]
        
        # Mock BasePipelineConfig
        self.mock_config_base = MagicMock(spec=BasePipelineConfig)
        self.mock_config_base.pipeline_s3_loc = "s3://test-bucket/test-pipeline"
        
        # Mock StepBuilderBase
        self.mock_builder_base = MagicMock(spec=StepBuilderBase)
        self.mock_builder_base.get_input_requirements.return_value = {
            'input1': 'Input 1 description',
            'input2': 'Input 2 description'
        }
        self.mock_builder_base.get_output_properties.return_value = {
            'output1': 'Output 1 description',
            'output2': 'Output 2 description'
        }
        self.mock_builder_base.create_step.return_value = MagicMock(name="mock_step")
        
        # Mock config_map
        self.mock_config_map = {
            'step1': MagicMock(spec=BasePipelineConfig),
            'step2': MagicMock(spec=BasePipelineConfig),
            'step3': MagicMock(spec=BasePipelineConfig)
        }
        
        # Mock step_builder_map
        self.mock_step_builder_cls = MagicMock(return_value=self.mock_builder_base)
        self.mock_step_builder_map = {
            'Step1': self.mock_step_builder_cls,
            'Step2': self.mock_step_builder_cls,
            'Step3': self.mock_step_builder_cls,
            'MagicMock': self.mock_step_builder_cls  # Add this for the test_collect_step_io_requirements
        }
        
        # Mock PipelineSession
        self.mock_session = MagicMock()
        
        # Mock role
        self.mock_role = "arn:aws:iam::123456789012:role/SageMakerRole"
        
        # Mock notebook_root
        self.mock_notebook_root = Path("/dummy/notebook/root")
        
        # Patch BasePipelineConfig.get_step_name
        self.get_step_name_patch = patch('src.pipeline_steps.config_base.BasePipelineConfig.get_step_name')
        self.mock_get_step_name = self.get_step_name_patch.start()
        self.mock_get_step_name.side_effect = lambda class_name: class_name.replace('Config', '')
        
        # Patch Pipeline
        self.pipeline_patch = patch('src.pipeline_builder.pipeline_builder_template.Pipeline')
        self.mock_pipeline_cls = self.pipeline_patch.start()
        self.mock_pipeline = MagicMock()
        self.mock_pipeline_cls.return_value = self.mock_pipeline
        
        # Create the builder instance
        self.builder = PipelineBuilderTemplate(
            dag=self.mock_dag,
            config_map=self.mock_config_map,
            step_builder_map=self.mock_step_builder_map,
            sagemaker_session=self.mock_session,
            role=self.mock_role,
            pipeline_parameters=[],
            notebook_root=self.mock_notebook_root
        )

    def tearDown(self):
        """Clean up patches after each test."""
        self.get_step_name_patch.stop()
        self.pipeline_patch.stop()

    def test_initialization(self):
        """Test that the builder initializes correctly."""
        # Verify attributes were set correctly
        self.assertEqual(self.builder.dag, self.mock_dag)
        self.assertEqual(self.builder.config_map, self.mock_config_map)
        self.assertEqual(self.builder.step_builder_map, self.mock_step_builder_map)
        self.assertEqual(self.builder.sagemaker_session, self.mock_session)
        self.assertEqual(self.builder.role, self.mock_role)
        self.assertEqual(self.builder.notebook_root, self.mock_notebook_root)
        self.assertEqual(self.builder.pipeline_parameters, [])
        
        # Verify data structures were initialized
        self.assertEqual(self.builder.step_instances, {})
        # step_builders is initialized in the constructor, so it should not be empty
        self.assertEqual(len(self.builder.step_builders), 3)
        self.assertIn('step1', self.builder.step_builders)
        self.assertIn('step2', self.builder.step_builders)
        self.assertIn('step3', self.builder.step_builders)
        self.assertEqual(self.builder.step_input_requirements, {})
        self.assertEqual(self.builder.step_output_properties, {})
        self.assertIsInstance(self.builder.step_messages, defaultdict)

    def test_collect_step_io_requirements(self):
        """Test that _collect_step_io_requirements collects input and output requirements."""
        # Call the method
        self.builder._collect_step_io_requirements()
        
        # Verify step builders were created
        self.assertEqual(len(self.builder.step_builders), 3)
        self.assertIn('step1', self.builder.step_builders)
        self.assertIn('step2', self.builder.step_builders)
        self.assertIn('step3', self.builder.step_builders)
        
        # Verify input requirements were collected
        self.assertEqual(len(self.builder.step_input_requirements), 3)
        for step_name in ['step1', 'step2', 'step3']:
            self.assertIn(step_name, self.builder.step_input_requirements)
            self.assertEqual(
                self.builder.step_input_requirements[step_name],
                {'input1': 'Input 1 description', 'input2': 'Input 2 description'}
            )
        
        # Verify output properties were collected
        self.assertEqual(len(self.builder.step_output_properties), 3)
        for step_name in ['step1', 'step2', 'step3']:
            self.assertIn(step_name, self.builder.step_output_properties)
            self.assertEqual(
                self.builder.step_output_properties[step_name],
                {'output1': 'Output 1 description', 'output2': 'Output 2 description'}
            )

    def test_propagate_messages_direct_match(self):
        """Test that _propagate_messages matches inputs to outputs by name."""
        # Set up input requirements and output properties
        self.builder.step_input_requirements = {
            'step1': {},  # No inputs for step1
            'step2': {'output1': 'Need output1'},  # step2 needs output1
            'step3': {'output2': 'Need output2'}   # step3 needs output2
        }
        
        self.builder.step_output_properties = {
            'step1': {'output1': 'Provides output1', 'output2': 'Provides output2'},
            'step2': {'output2': 'Provides output2'},
            'step3': {}  # No outputs for step3
        }
        
        # Call the method
        self.builder._propagate_messages()
        
        # Verify messages were created for direct matches
        self.assertIn('step2', self.builder.step_messages)
        self.assertIn('output1', self.builder.step_messages['step2'])
        self.assertEqual(
            self.builder.step_messages['step2']['output1'],
            {'source_step': 'step1', 'source_output': 'output1'}
        )
        
        self.assertIn('step3', self.builder.step_messages)
        self.assertIn('output2', self.builder.step_messages['step3'])
        self.assertEqual(
            self.builder.step_messages['step3']['output2'],
            {'source_step': 'step2', 'source_output': 'output2'}
        )

    def test_propagate_messages_pattern_match(self):
        """Test that _propagate_messages matches inputs to outputs by pattern."""
        # Set up input requirements and output properties
        self.builder.step_input_requirements = {
            'step1': {},  # No inputs for step1
            'step2': {'model_data': 'Need model data'},  # step2 needs model_data
            'step3': {'training_data': 'Need training data'}   # step3 needs training_data
        }
        
        self.builder.step_output_properties = {
            'step1': {'model_artifacts': 'Provides model artifacts', 'dataset': 'Provides dataset'},
            'step2': {'data_output': 'Provides data output'},
            'step3': {}  # No outputs for step3
        }
        
        # Call the method
        self.builder._propagate_messages()
        
        # Verify messages were created for pattern matches
        self.assertIn('step2', self.builder.step_messages)
        self.assertIn('model_data', self.builder.step_messages['step2'])
        self.assertEqual(
            self.builder.step_messages['step2']['model_data']['source_step'],
            'step1'
        )
        self.assertEqual(
            self.builder.step_messages['step2']['model_data']['source_output'],
            'model_artifacts'
        )
        self.assertTrue(self.builder.step_messages['step2']['model_data']['pattern_match'])
        
        self.assertIn('step3', self.builder.step_messages)
        self.assertIn('training_data', self.builder.step_messages['step3'])
        self.assertEqual(
            self.builder.step_messages['step3']['training_data']['source_step'],
            'step2'
        )
        self.assertEqual(
            self.builder.step_messages['step3']['training_data']['source_output'],
            'data_output'
        )
        self.assertTrue(self.builder.step_messages['step3']['training_data']['pattern_match'])

    def test_instantiate_step(self):
        """Test that _instantiate_step creates a step with the right inputs."""
        # Set up step instances for dependencies
        mock_step1 = MagicMock(name="step1")
        mock_step1.name = "step1"
        self.builder.step_instances = {'step1': mock_step1}
        
        # Set up step builders
        self.builder.step_builders = {
            'step2': self.mock_builder_base
        }
        
        # Set up messages
        self.builder.step_messages = {
            'step2': {
                'input1': {'source_step': 'step1', 'source_output': 'output1'}
            }
        }
        
        # Mock step1 to have output1 attribute
        mock_step1.output1 = "output1_value"
        
        # Call the method
        step = self.builder._instantiate_step('step2')
        
        # Verify the step was created with the right inputs
        self.mock_builder_base.create_step.assert_called_once()
        kwargs = self.mock_builder_base.create_step.call_args[1]
        self.assertEqual(kwargs['dependencies'], [mock_step1])
        self.assertEqual(kwargs['input1'], "output1_value")
        
        # Verify the step was returned
        self.assertEqual(step, self.mock_builder_base.create_step.return_value)

    def test_generate_pipeline(self):
        """Test that generate_pipeline creates a complete pipeline."""
        # Mock the internal methods
        with patch.object(self.builder, '_collect_step_io_requirements') as mock_collect:
            with patch.object(self.builder, '_propagate_messages') as mock_propagate:
                with patch.object(self.builder, '_instantiate_step') as mock_instantiate:
                    # Set up mock steps
                    mock_step1 = MagicMock(name="step1")
                    mock_step2 = MagicMock(name="step2")
                    mock_step3 = MagicMock(name="step3")
                    mock_instantiate.side_effect = [mock_step1, mock_step2, mock_step3]
                    
                    # Call the method
                    pipeline = self.builder.generate_pipeline("test-pipeline")
                    
                    # Verify the internal methods were called
                    mock_collect.assert_called_once()
                    mock_propagate.assert_called_once()
                    
                    # Verify _instantiate_step was called for each step
                    self.assertEqual(mock_instantiate.call_count, 3)
                    mock_instantiate.assert_any_call('step1')
                    mock_instantiate.assert_any_call('step2')
                    mock_instantiate.assert_any_call('step3')
                    
                    # Verify step instances were stored
                    self.assertEqual(len(self.builder.step_instances), 3)
                    self.assertEqual(self.builder.step_instances['step1'], mock_step1)
                    self.assertEqual(self.builder.step_instances['step2'], mock_step2)
                    self.assertEqual(self.builder.step_instances['step3'], mock_step3)
                    
                    # Verify Pipeline was created with the right parameters
                    self.mock_pipeline_cls.assert_called_once_with(
                        name="test-pipeline",
                        parameters=[],
                        steps=[mock_step1, mock_step2, mock_step3],
                        sagemaker_session=self.mock_session
                    )
                    
                    # Verify the pipeline was returned
                    self.assertEqual(pipeline, self.mock_pipeline)

    def test_add_config_inputs(self):
        """Test that _add_config_inputs adds inputs from config."""
        # Create a config with some common inputs
        config = MagicMock(spec=BasePipelineConfig)
        config.model_data = "s3://test-bucket/model.tar.gz"
        config.data_uri = "s3://test-bucket/data"
        config.input_data = None  # This should not be added
        
        # Call the method
        kwargs = {}
        self.builder._add_config_inputs(kwargs, config)
        
        # Verify inputs were added
        self.assertEqual(kwargs['model_data'], "s3://test-bucket/model.tar.gz")
        self.assertEqual(kwargs['data_uri'], "s3://test-bucket/data")
        self.assertNotIn('input_data', kwargs)

    def test_extract_common_outputs_model_artifacts(self):
        """Test that _extract_common_outputs extracts model artifacts."""
        # Create a step with model_artifacts_path
        prev_step = MagicMock()
        prev_step.model_artifacts_path = "s3://test-bucket/model.tar.gz"
        
        # Ensure training_output_path doesn't override model_artifacts_path
        # by setting it to None or removing the attribute
        if hasattr(prev_step, 'training_output_path'):
            del prev_step.training_output_path
        
        # Call the method for a normal step
        kwargs = {}
        self.builder._extract_common_outputs(kwargs, prev_step, "step_name", "NormalStep")
        
        # Verify model_data was added
        self.assertEqual(kwargs['model_data'], "s3://test-bucket/model.tar.gz")
        
        # Call the method for a packaging step
        kwargs = {}
        self.builder._extract_common_outputs(kwargs, prev_step, "step_name", "PackagingStep")
        
        # Verify model_artifacts_input_source was added
        self.assertEqual(kwargs['model_artifacts_input_source'], "s3://test-bucket/model.tar.gz")

    def test_extract_common_outputs_processing_output(self):
        """Test that _extract_common_outputs extracts processing output."""
        # Create a step with ProcessingOutputConfig
        prev_step = MagicMock()
        prev_step.properties.ProcessingOutputConfig.Outputs = [MagicMock()]
        prev_step.properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri = "s3://test-bucket/output"
        
        # Call the method for a normal step
        kwargs = {}
        self.builder._extract_common_outputs(kwargs, prev_step, "step_name", "NormalStep")
        
        # Verify processing_output was added
        self.assertEqual(kwargs['processing_output'], "s3://test-bucket/output")
        
        # Call the method for a registration step
        kwargs = {}
        self.builder._extract_common_outputs(kwargs, prev_step, "step_name", "RegistrationStep")
        
        # Verify packaging_step_output was added
        self.assertEqual(kwargs['packaging_step_output'], "s3://test-bucket/output")


if __name__ == '__main__':
    unittest.main()
