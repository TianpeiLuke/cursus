"""
Comprehensive tests for SpecificationRegistry functionality.

Tests the complete functionality of specification registry including:
- Registry creation and basic operations
- Registering and retrieving specifications
- Type-based queries and compatibility finding
- Context isolation and complex pipeline scenarios
- Data type compatibility and edge cases
"""

import pytest
from .test_helpers import reset_all_global_state
from cursus.core.deps.specification_registry import SpecificationRegistry
from cursus.core.base.specification_base import (
    StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType
)

class TestSpecificationRegistry:
    """Test cases for SpecificationRegistry."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        # Reset global state
        reset_all_global_state()
        
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
        
        yield
        
        # Clean up after tests
        reset_all_global_state()
    
    def test_registry_initialization(self):
        """Test registry initialization with context name."""
        registry = SpecificationRegistry("test_pipeline")
        assert registry.context_name == "test_pipeline"
        assert len(registry.list_step_names()) == 0
        assert len(registry.list_step_types()) == 0
    
    def test_register_specification(self):
        """Test registering step specifications."""
        # Register data loading step
        self.registry.register("data_loading", self.data_loading_spec)
        
        # Verify registration
        assert "data_loading" in self.registry.list_step_names()
        assert "DataLoadingStep" in self.registry.list_step_types()
        
        # Verify retrieval
        retrieved_spec = self.registry.get_specification("data_loading")
        assert retrieved_spec is not None
        assert retrieved_spec.step_type == "DataLoadingStep"
    
    def test_register_invalid_specification(self):
        """Test registering invalid specifications raises errors."""
        with pytest.raises(ValueError):
            self.registry.register("invalid", "not_a_specification")
    
    def test_get_specifications_by_type(self):
        """Test retrieving specifications by step type."""
        # Register multiple steps
        self.registry.register("data_loading", self.data_loading_spec)
        self.registry.register("preprocessing", self.preprocessing_spec)
        
        # Get by type
        data_loading_specs = self.registry.get_specifications_by_type("DataLoadingStep")
        assert len(data_loading_specs) == 1
        assert data_loading_specs[0].step_type == "DataLoadingStep"
        
        preprocessing_specs = self.registry.get_specifications_by_type("PreprocessingStep")
        assert len(preprocessing_specs) == 1
        assert preprocessing_specs[0].step_type == "PreprocessingStep"
        
        # Non-existent type
        nonexistent_specs = self.registry.get_specifications_by_type("NonExistentStep")
        assert len(nonexistent_specs) == 0
    
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
        assert len(compatible) > 0
        
        # Check the best match
        best_match = compatible[0]
        step_name, output_name, output_spec, score = best_match
        
        assert step_name == "data_loading"
        assert output_name == "raw_data"
        assert output_spec.output_type == DependencyType.PROCESSING_OUTPUT
        assert score > 0.5  # Should have good compatibility score
    
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
        
        assert len(compatible_outputs) > 0
        assert len(incompatible_outputs) == 0
    
    def test_context_isolation(self):
        """Test that different registry contexts are isolated."""
        # Create two registries with different contexts
        registry1 = SpecificationRegistry("pipeline_1")
        registry2 = SpecificationRegistry("pipeline_2")
        
        # Register different specs in each
        registry1.register("step1", self.data_loading_spec)
        registry2.register("step2", self.preprocessing_spec)
        
        # Verify isolation
        assert "step1" in registry1.list_step_names()
        assert "step1" not in registry2.list_step_names()
        
        assert "step2" in registry2.list_step_names()
        assert "step2" not in registry1.list_step_names()
        
        # Verify context names
        assert registry1.context_name == "pipeline_1"
        assert registry2.context_name == "pipeline_2"
    
    def test_registry_string_representation(self):
        """Test string representation of registry."""
        self.registry.register("data_loading", self.data_loading_spec)
        
        repr_str = repr(self.registry)
        assert "test_context" in repr_str
        assert "steps=1" in repr_str
    
    def test_empty_registry_operations(self):
        """Test operations on empty registry."""
        empty_registry = SpecificationRegistry("empty")
        
        # Test empty operations
        assert len(empty_registry.list_step_names()) == 0
        assert len(empty_registry.list_step_types()) == 0
        assert empty_registry.get_specification("nonexistent") is None
        
        # Test finding compatible outputs on empty registry
        dep_spec = DependencySpec(
            logical_name="test_dep",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            data_type="S3Uri"
        )
        compatible = empty_registry.find_compatible_outputs(dep_spec)
        assert len(compatible) == 0
    
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
            assert score1 > 0.7  # Should have high score (source + keywords)
        
        if results2:
            _, _, _, score2 = results2[0]
            assert score2 > 0.5  # Medium score (keywords only)
            assert score2 < score1  # Lower than case 1
        
        if results3:
            _, _, _, score3 = results3[0]
            assert score3 > 0.5  # Medium score (source only)
            assert score1 > score3  # Lower than case 1
    
    def test_complex_pipeline_compatibility(self):
        """Test compatibility in a more complex pipeline with all step types."""
        # Register all specs to simulate a complete pipeline
        self.registry.register("data_loading", self.data_loading_spec)
        self.registry.register("preprocessing", self.preprocessing_spec)
        self.registry.register("training", self.training_spec)
        self.registry.register("evaluation", self.evaluation_spec)
        
        # Verify all step types are registered
        step_types = self.registry.list_step_types()
        assert len(step_types) == 4
        assert "DataLoadingStep" in step_types
        assert "PreprocessingStep" in step_types
        assert "TrainingStep" in step_types
        assert "EvaluationStep" in step_types
        
        # Test training step dependency can find preprocessing outputs
        training_dep = DependencySpec(
            logical_name="training_input",
            dependency_type=self.dependency_type,
            required=True,
            compatible_sources=["PreprocessingStep"],
            data_type="S3Uri"
        )
        
        training_matches = self.registry.find_compatible_outputs(training_dep)
        assert len(training_matches) > 0
        step_name, output_name, _, _ = training_matches[0]
        assert step_name == "preprocessing"
        assert output_name == "processed_data"
        
        # Test evaluation step dependency can find training outputs
        eval_dep = DependencySpec(
            logical_name="model_input",
            dependency_type=self.model_artifact_type,
            required=True,
            compatible_sources=["TrainingStep"],
            data_type="S3Uri"
        )
        
        eval_matches = self.registry.find_compatible_outputs(eval_dep)
        assert len(eval_matches) > 0
        step_name, output_name, _, _ = eval_matches[0]
        assert step_name == "training"
        assert output_name == "model_artifacts"
    
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
        assert len(compatible) == 2
        
        # First match should be training_data due to keyword match
        step_name, output_name, _, score1 = compatible[0]
        assert output_name == "training_data"
        
        # Second match should be validation_data with lower score
        _, second_output_name, _, score2 = compatible[1]
        assert second_output_name == "validation_data"
        
        # First match should have higher score
        assert score1 > score2
    
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
        
        assert len(string_matches) == 1
        assert string_matches[0][1] == "string_output"
        
        assert len(s3_matches) == 1
        assert s3_matches[0][1] == "s3_output"

    def test_register_multiple_specifications(self):
        """Test registering multiple specifications."""
        # Register first spec
        first_name = "first_step"
        self.registry.register(first_name, self.data_loading_spec)
        
        # Create and register second spec
        second_spec = StepSpecification(
            step_type="SecondStep",
            node_type=self.node_type_source,  # SOURCE must have outputs
            dependencies=[],
            outputs=[OutputSpec(
                logical_name="second_output",
                output_type=self.dependency_type,
                property_path="properties.Output.S3Uri",
                data_type="S3Uri"
            )]
        )
        second_name = "second_step"
        self.registry.register(second_name, second_spec)
        
        # Check if both registered
        assert len(self.registry._specifications) == 2
        assert first_name in self.registry._specifications
        assert second_name in self.registry._specifications
        assert self.registry._specifications[first_name] == self.data_loading_spec
        assert self.registry._specifications[second_name] == second_spec

    def test_get_specification_detailed(self):
        """Test detailed specification retrieval."""
        # Register specification
        step_name = "test_step"
        self.registry.register(step_name, self.preprocessing_spec)
        
        # Get specification by name
        spec = self.registry.get_specification(step_name)
        
        # Check if correct
        assert spec == self.preprocessing_spec
        assert spec.step_type == "PreprocessingStep"
        assert len(spec.dependencies) == 1
        assert len(spec.outputs) == 1
        
        # Access items from dictionaries
        output_name = next(iter(spec.outputs.keys()))
        assert spec.outputs[output_name].logical_name == "processed_data"

    def test_get_specification_by_type_detailed(self):
        """Test detailed retrieval of specifications by type."""
        # Create specs with same type
        spec1 = StepSpecification(
            step_type="SharedType",
            node_type=self.node_type_source,
            dependencies=[],
            outputs=[OutputSpec(
                logical_name="output1",
                output_type=self.dependency_type,
                property_path="properties.Output1.S3Uri",
                data_type="S3Uri"
            )]
        )
        spec2 = StepSpecification(
            step_type="SharedType",  # Same type as spec1
            node_type=self.node_type_source,
            dependencies=[],
            outputs=[OutputSpec(
                logical_name="output2",
                output_type=self.dependency_type,
                property_path="properties.Output2.S3Uri",
                data_type="S3Uri"
            )]
        )
        spec3 = StepSpecification(
            step_type="UniqueType",  # Different type
            node_type=self.node_type_source,
            dependencies=[],
            outputs=[OutputSpec(
                logical_name="output3",
                output_type=self.dependency_type,
                property_path="properties.Output3.S3Uri",
                data_type="S3Uri"
            )]
        )
        
        # Register specifications
        self.registry.register("step1", spec1)
        self.registry.register("step2", spec2)
        self.registry.register("step3", spec3)
        
        # Get specifications by type
        shared_specs = self.registry.get_specifications_by_type("SharedType")
        unique_specs = self.registry.get_specifications_by_type("UniqueType")
        
        # Check results
        assert len(shared_specs) == 2  # Should be 2 specs of type "SharedType"
        assert len(unique_specs) == 1  # Should be 1 spec of type "UniqueType"

    def test_list_operations_detailed(self):
        """Test detailed listing operations."""
        # Register specifications
        self.registry.register("step1", self.data_loading_spec)
        
        other_spec = StepSpecification(
            step_type="OtherType",
            node_type=self.node_type_source,
            dependencies=[],
            outputs=[OutputSpec(
                logical_name="other_output",
                output_type=self.dependency_type,
                property_path="properties.Other.S3Uri",
                data_type="S3Uri"
            )]
        )
        self.registry.register("step2", other_spec)
        
        # Test list_step_names
        step_names = self.registry.list_step_names()
        assert len(step_names) == 2
        assert "step1" in step_names
        assert "step2" in step_names
        
        # Test list_step_types
        step_types = self.registry.list_step_types()
        assert len(step_types) == 2
        assert "DataLoadingStep" in step_types
        assert "OtherType" in step_types

    def test_find_compatible_outputs_detailed(self):
        """Test detailed compatible output finding."""
        # Register source step
        source_spec = StepSpecification(
            step_type="SourceStep",
            node_type=self.node_type_source,
            dependencies=[],
            outputs=[OutputSpec(
                logical_name="source_output",
                output_type=self.dependency_type,
                property_path="properties.Output.S3Uri",
                data_type="S3Uri"
            )]
        )
        self.registry.register("source", source_spec)
        
        # Create dependency spec to search for
        dep_spec = DependencySpec(
            logical_name="test_input",
            dependency_type=self.dependency_type,
            data_type="S3Uri",
            compatible_sources=["SourceStep"]  # Match source step type
        )
        
        # Find compatible outputs
        compatible = self.registry.find_compatible_outputs(dep_spec)
        
        # Should find the source output
        assert len(compatible) == 1
        assert compatible[0][0] == "source"  # Step name
        assert compatible[0][1] == "source_output"  # Output name
