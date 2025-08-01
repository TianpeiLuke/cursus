"""
Tests for SpecificationRegistry - atomized registry functionality.
"""

import unittest
from test.pipeline_deps.test_helpers import IsolatedTestCase, reset_all_global_state
from src.pipeline_deps.specification_registry import SpecificationRegistry
from src.pipeline_deps.base_specifications import (
    StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType
)


class TestSpecificationRegistry(IsolatedTestCase):
    """Test cases for SpecificationRegistry."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Call parent setUp to reset global state
        super().setUp()
        
        self.registry = SpecificationRegistry("test_context")
        
        # Create fresh instances of the enums for each test to ensure isolation
        self.node_type_source = NodeType.SOURCE
        self.node_type_internal = NodeType.INTERNAL
        self.node_type_sink = NodeType.SINK
        self.dependency_type = DependencyType.PROCESSING_OUTPUT
        self.model_artifact_type = DependencyType.MODEL_ARTIFACTS
        
        # Create test specifications
        output_spec = OutputSpec(
            logical_name="raw_data",
            output_type=self.dependency_type,
            property_path="properties.ProcessingOutputConfig.Outputs['RawData'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Raw data output"
        )
        
        self.data_loading_spec = StepSpecification(
            step_type="DataLoadingStep",
            node_type=self.node_type_source,
            dependencies=[],
            outputs=[output_spec]
        )
        
        # Create dependency and output specs separately
        dep_spec = DependencySpec(
            logical_name="input_data",
            dependency_type=self.dependency_type,
            required=True,
            compatible_sources=["DataLoadingStep"],
            semantic_keywords=["data", "input"],
            data_type="S3Uri",
            description="Input data for preprocessing"
        )
        
        output_spec = OutputSpec(
            logical_name="processed_data",
            output_type=self.dependency_type,
            property_path="properties.ProcessingOutputConfig.Outputs['ProcessedData'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Processed data output"
        )
        
        self.preprocessing_spec = StepSpecification(
            step_type="PreprocessingStep",
            node_type=self.node_type_internal,
            dependencies=[dep_spec],
            outputs=[output_spec]
        )
        
        # Create training spec
        training_dep_spec = DependencySpec(
            logical_name="training_data",
            dependency_type=self.dependency_type,
            required=True,
            compatible_sources=["PreprocessingStep"],
            semantic_keywords=["data", "processed"],
            data_type="S3Uri",
            description="Processed data for training"
        )
        
        training_output_spec = OutputSpec(
            logical_name="model_artifacts",
            output_type=self.model_artifact_type,
            property_path="properties.ModelArtifacts.S3ModelArtifacts",
            data_type="S3Uri",
            description="Trained model artifacts"
        )
        
        self.training_spec = StepSpecification(
            step_type="TrainingStep",
            node_type=self.node_type_internal,
            dependencies=[training_dep_spec],
            outputs=[training_output_spec]
        )
        
        # Create evaluation spec (sink node with dependency but no outputs)
        eval_dep_spec = DependencySpec(
            logical_name="model_input",
            dependency_type=self.model_artifact_type,
            required=True,
            compatible_sources=["TrainingStep"],
            semantic_keywords=["model", "artifacts"],
            data_type="S3Uri",
            description="Model for evaluation"
        )
        
        self.evaluation_spec = StepSpecification(
            step_type="EvaluationStep",
            node_type=self.node_type_sink,
            dependencies=[eval_dep_spec],
            outputs=[]
        )
        
    def tearDown(self):
        """Clean up after tests."""
        # Call parent tearDown to reset global state
        super().tearDown()
    
    def test_registry_initialization(self):
        """Test registry initialization with context name."""
        registry = SpecificationRegistry("test_pipeline")
        self.assertEqual(registry.context_name, "test_pipeline")
        self.assertEqual(len(registry.list_step_names()), 0)
        self.assertEqual(len(registry.list_step_types()), 0)
    
    def test_register_specification(self):
        """Test registering step specifications."""
        # Register data loading step
        self.registry.register("data_loading", self.data_loading_spec)
        
        # Verify registration
        self.assertIn("data_loading", self.registry.list_step_names())
        self.assertIn("DataLoadingStep", self.registry.list_step_types())
        
        # Verify retrieval
        retrieved_spec = self.registry.get_specification("data_loading")
        self.assertIsNotNone(retrieved_spec)
        self.assertEqual(retrieved_spec.step_type, "DataLoadingStep")
    
    def test_register_invalid_specification(self):
        """Test registering invalid specifications raises errors."""
        with self.assertRaises(ValueError):
            self.registry.register("invalid", "not_a_specification")
    
    def test_get_specifications_by_type(self):
        """Test retrieving specifications by step type."""
        # Register multiple steps
        self.registry.register("data_loading", self.data_loading_spec)
        self.registry.register("preprocessing", self.preprocessing_spec)
        
        # Get by type
        data_loading_specs = self.registry.get_specifications_by_type("DataLoadingStep")
        self.assertEqual(len(data_loading_specs), 1)
        self.assertEqual(data_loading_specs[0].step_type, "DataLoadingStep")
        
        preprocessing_specs = self.registry.get_specifications_by_type("PreprocessingStep")
        self.assertEqual(len(preprocessing_specs), 1)
        self.assertEqual(preprocessing_specs[0].step_type, "PreprocessingStep")
        
        # Non-existent type
        nonexistent_specs = self.registry.get_specifications_by_type("NonExistentStep")
        self.assertEqual(len(nonexistent_specs), 0)
    
    def test_find_compatible_outputs(self):
        """Test finding compatible outputs for dependencies."""
        # Register steps
        self.registry.register("data_loading", self.data_loading_spec)
        self.registry.register("preprocessing", self.preprocessing_spec)
        
        # Create dependency spec to match
        dependency_spec = DependencySpec(
            logical_name="input_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["DataLoadingStep"],
            data_type="S3Uri"
        )
        
        # Find compatible outputs
        compatible = self.registry.find_compatible_outputs(dependency_spec)
        
        # Should find the data loading output
        self.assertGreater(len(compatible), 0)
        
        # Check the best match
        best_match = compatible[0]
        step_name, output_name, output_spec, score = best_match
        
        self.assertEqual(step_name, "data_loading")
        self.assertEqual(output_name, "raw_data")
        self.assertEqual(output_spec.output_type, DependencyType.PROCESSING_OUTPUT)
        self.assertGreater(score, 0.5)  # Should have good compatibility score
    
    def test_compatibility_checking(self):
        """Test internal compatibility checking logic."""
        # Register data loading step
        self.registry.register("data_loading", self.data_loading_spec)
        
        # Compatible dependency
        compatible_dep = DependencySpec(
            logical_name="data_input",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            data_type="S3Uri"
        )
        
        # Incompatible dependency (wrong type)
        incompatible_dep = DependencySpec(
            logical_name="model_input",
            dependency_type=DependencyType.MODEL_ARTIFACTS,
            data_type="S3Uri"
        )
        
        # Test compatibility
        compatible_outputs = self.registry.find_compatible_outputs(compatible_dep)
        incompatible_outputs = self.registry.find_compatible_outputs(incompatible_dep)
        
        self.assertGreater(len(compatible_outputs), 0)
        self.assertEqual(len(incompatible_outputs), 0)
    
    def test_context_isolation(self):
        """Test that different registry contexts are isolated."""
        # Create two registries with different contexts
        registry1 = SpecificationRegistry("pipeline_1")
        registry2 = SpecificationRegistry("pipeline_2")
        
        # Register different specs in each
        registry1.register("step1", self.data_loading_spec)
        registry2.register("step2", self.preprocessing_spec)
        
        # Verify isolation
        self.assertIn("step1", registry1.list_step_names())
        self.assertNotIn("step1", registry2.list_step_names())
        
        self.assertIn("step2", registry2.list_step_names())
        self.assertNotIn("step2", registry1.list_step_names())
        
        # Verify context names
        self.assertEqual(registry1.context_name, "pipeline_1")
        self.assertEqual(registry2.context_name, "pipeline_2")
    
    def test_registry_string_representation(self):
        """Test string representation of registry."""
        self.registry.register("data_loading", self.data_loading_spec)
        
        repr_str = repr(self.registry)
        self.assertIn("test_context", repr_str)
        self.assertIn("steps=1", repr_str)
    
    def test_empty_registry_operations(self):
        """Test operations on empty registry."""
        empty_registry = SpecificationRegistry("empty")
        
        # Test empty operations
        self.assertEqual(len(empty_registry.list_step_names()), 0)
        self.assertEqual(len(empty_registry.list_step_types()), 0)
        self.assertIsNone(empty_registry.get_specification("nonexistent"))
        
        # Test finding compatible outputs on empty registry
        dep_spec = DependencySpec(
            logical_name="test_dep",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            data_type="S3Uri"
        )
        compatible = empty_registry.find_compatible_outputs(dep_spec)
        self.assertEqual(len(compatible), 0)
    
    def test_compatibility_scoring_algorithm(self):
        """Test compatibility scoring algorithm in detail."""
        # Register all test specs
        self.registry.register("data_loading", self.data_loading_spec)
        self.registry.register("preprocessing", self.preprocessing_spec)
        self.registry.register("training", self.training_spec)
        
        # Test with varying compatible_sources and semantic_keywords
        
        # Case 1: Exact match on compatible_sources and some keywords
        dep_spec1 = DependencySpec(
            logical_name="processed_data_dep",
            dependency_type=self.dependency_type,
            required=True,
            compatible_sources=["PreprocessingStep"],  # Exact match
            semantic_keywords=["processed", "data"],    # Two keyword matches
            data_type="S3Uri"
        )
        
        # Case 2: No compatible_sources but keyword match
        dep_spec2 = DependencySpec(
            logical_name="data_dep",
            dependency_type=self.dependency_type,
            required=True,
            compatible_sources=[],                  # No sources specified
            semantic_keywords=["data", "processed"], # Still has keyword matches
            data_type="S3Uri"
        )
        
        # Case 3: Compatible source but no keywords
        dep_spec3 = DependencySpec(
            logical_name="source_match_dep",
            dependency_type=self.dependency_type,
            required=True,
            compatible_sources=["PreprocessingStep"],  # Source match
            semantic_keywords=[],                      # No keywords
            data_type="S3Uri"
        )
        
        # Run compatibility checks
        results1 = self.registry.find_compatible_outputs(dep_spec1)
        results2 = self.registry.find_compatible_outputs(dep_spec2)
        results3 = self.registry.find_compatible_outputs(dep_spec3)
        
        # Check scores
        if results1:
            _, _, _, score1 = results1[0]
            self.assertGreater(score1, 0.7)  # Should have high score (source + keywords)
        
        if results2:
            _, _, _, score2 = results2[0]
            self.assertGreater(score2, 0.5)  # Medium score (keywords only)
            self.assertLess(score2, score1)  # Lower than case 1
        
        if results3:
            _, _, _, score3 = results3[0]
            self.assertGreater(score3, 0.5)  # Medium score (source only)
            self.assertGreater(score1, score3)  # Lower than case 1
    
    def test_complex_pipeline_compatibility(self):
        """Test compatibility in a more complex pipeline with all step types."""
        # Register all specs to simulate a complete pipeline
        self.registry.register("data_loading", self.data_loading_spec)
        self.registry.register("preprocessing", self.preprocessing_spec)
        self.registry.register("training", self.training_spec)
        self.registry.register("evaluation", self.evaluation_spec)
        
        # Verify all step types are registered
        step_types = self.registry.list_step_types()
        self.assertEqual(len(step_types), 4)
        self.assertIn("DataLoadingStep", step_types)
        self.assertIn("PreprocessingStep", step_types)
        self.assertIn("TrainingStep", step_types)
        self.assertIn("EvaluationStep", step_types)
        
        # Test training step dependency can find preprocessing outputs
        training_dep = DependencySpec(
            logical_name="training_input",
            dependency_type=self.dependency_type,
            required=True,
            compatible_sources=["PreprocessingStep"],
            data_type="S3Uri"
        )
        
        training_matches = self.registry.find_compatible_outputs(training_dep)
        self.assertGreater(len(training_matches), 0)
        step_name, output_name, _, _ = training_matches[0]
        self.assertEqual(step_name, "preprocessing")
        self.assertEqual(output_name, "processed_data")
        
        # Test evaluation step dependency can find training outputs
        eval_dep = DependencySpec(
            logical_name="model_input",
            dependency_type=self.model_artifact_type,
            required=True,
            compatible_sources=["TrainingStep"],
            data_type="S3Uri"
        )
        
        eval_matches = self.registry.find_compatible_outputs(eval_dep)
        self.assertGreater(len(eval_matches), 0)
        step_name, output_name, _, _ = eval_matches[0]
        self.assertEqual(step_name, "training")
        self.assertEqual(output_name, "model_artifacts")
    
    def test_multiple_compatible_outputs(self):
        """Test handling of multiple compatible outputs with different scores."""
        # Create two data loading steps with similar outputs
        output_spec1 = OutputSpec(
            logical_name="training_data",
            output_type=self.dependency_type,
            property_path="properties.ProcessingOutputConfig.Outputs['TrainingData'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Training data output"
        )
        
        output_spec2 = OutputSpec(
            logical_name="validation_data",
            output_type=self.dependency_type,
            property_path="properties.ProcessingOutputConfig.Outputs['ValidationData'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Validation data output"
        )
        
        data_loading_spec1 = StepSpecification(
            step_type="DataLoadingStep",
            node_type=self.node_type_source,
            dependencies=[],
            outputs=[output_spec1]
        )
        
        data_loading_spec2 = StepSpecification(
            step_type="DataLoadingStep",
            node_type=self.node_type_source,
            dependencies=[],
            outputs=[output_spec2]
        )
        
        # Register both specs
        self.registry.register("training_data_loader", data_loading_spec1)
        self.registry.register("validation_data_loader", data_loading_spec2)
        
        # Create dependency spec that matches both outputs
        dep_spec = DependencySpec(
            logical_name="data_input",
            dependency_type=self.dependency_type,
            required=True,
            compatible_sources=["DataLoadingStep"],
            semantic_keywords=["training", "data"],  # Better match for training_data
            data_type="S3Uri"
        )
        
        # Find compatible outputs
        compatible = self.registry.find_compatible_outputs(dep_spec)
        
        # Should find both outputs
        self.assertEqual(len(compatible), 2)
        
        # First match should be training_data due to keyword match
        step_name, output_name, _, score1 = compatible[0]
        self.assertEqual(output_name, "training_data")
        
        # Second match should be validation_data with lower score
        _, second_output_name, _, score2 = compatible[1]
        self.assertEqual(second_output_name, "validation_data")
        
        # First match should have higher score
        self.assertGreater(score1, score2)
    
    def test_data_type_compatibility(self):
        """Test compatibility checking with different data types."""
        # Create outputs with different data types
        string_output = OutputSpec(
            logical_name="string_output",
            output_type=self.dependency_type,
            property_path="properties.Output.String",
            data_type="String",
            description="String output"
        )
        
        s3uri_output = OutputSpec(
            logical_name="s3_output",
            output_type=self.dependency_type,
            property_path="properties.Output.S3Uri",
            data_type="S3Uri",
            description="S3Uri output"
        )
        
        # Create specs with these outputs
        string_spec = StepSpecification(
            step_type="StringStep",
            node_type=self.node_type_source,
            dependencies=[],
            outputs=[string_output]
        )
        
        s3_spec = StepSpecification(
            step_type="S3Step",
            node_type=self.node_type_source,
            dependencies=[],
            outputs=[s3uri_output]
        )
        
        # Register specs
        self.registry.register("string_step", string_spec)
        self.registry.register("s3_step", s3_spec)
        
        # Create dependency specs with different data types
        string_dep = DependencySpec(
            logical_name="string_dep",
            dependency_type=self.dependency_type,
            data_type="String"
        )
        
        s3_dep = DependencySpec(
            logical_name="s3_dep",
            dependency_type=self.dependency_type,
            data_type="S3Uri"
        )
        
        # Test compatibility with matching data types
        string_matches = self.registry.find_compatible_outputs(string_dep)
        s3_matches = self.registry.find_compatible_outputs(s3_dep)
        
        self.assertEqual(len(string_matches), 1)
        self.assertEqual(string_matches[0][1], "string_output")
        
        self.assertEqual(len(s3_matches), 1)
        self.assertEqual(s3_matches[0][1], "s3_output")


if __name__ == '__main__':
    unittest.main()
