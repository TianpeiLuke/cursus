import unittest
from src.pipeline_deps import (
    UnifiedDependencyResolver,
    SpecificationRegistry,
    StepSpecification,
    OutputSpec,
    DependencySpec,
    DependencyType,
    NodeType,
    PropertyReference,
    SemanticMatcher,
    DependencyResolutionError,
)

class TestDependencyResolver(unittest.TestCase):
    def test_dependency_resolution_with_aliases(self):
        """Test that dependency resolution uses aliases for matching."""
        registry = SpecificationRegistry()
        semantic_matcher = SemanticMatcher()
        resolver = UnifiedDependencyResolver(registry, semantic_matcher)
        
        # Create producer step specification with aliases
        producer_spec = StepSpecification(
            step_type="PreprocessingStep",
            node_type="source",
            dependencies=[],  # No dependencies for this test
            outputs=[
                OutputSpec(
                    logical_name="processed_data",
                    aliases=["training_data", "model_input"],
                    output_type=DependencyType.PROCESSING_OUTPUT,
                    property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
                    data_type="S3Uri"
                )
            ]
        )
        
        # Create consumer step specification
        consumer_spec = StepSpecification(
            step_type="TrainingStep",
            node_type="sink",
            dependencies=[
                DependencySpec(
                    logical_name="training_data",
                    dependency_type=DependencyType.PROCESSING_OUTPUT,
                    required=True,
                    compatible_sources=["PreprocessingStep"],
                    data_type="S3Uri"
                )
            ],
            outputs=[]  # No outputs needed for this test
        )
        
        # Register step specifications
        registry.register("producer", producer_spec)
        registry.register("consumer", consumer_spec)
        
        # Resolve dependencies
        resolved = resolver.resolve_step_dependencies("consumer", ["producer"])
        
        # Assert that resolution was successful
        self.assertIn("training_data", resolved)
        self.assertEqual(resolved["training_data"].step_name, "producer")
        self.assertEqual(resolved["training_data"].output_spec.logical_name, "processed_data")

    def test_weight_calculation(self):
        """Test that the weight calculation is correct."""
        registry = SpecificationRegistry()
        semantic_matcher = SemanticMatcher()
        resolver = UnifiedDependencyResolver(registry, semantic_matcher)
        
        # Create producer step specification
        producer_spec = StepSpecification(
            step_type="PreprocessingStep",
            node_type="source",
            dependencies=[],
            outputs=[
                OutputSpec(
                    logical_name="processed_data",
                    output_type=DependencyType.PROCESSING_OUTPUT,
                    property_path="properties.test",
                    data_type="S3Uri"
                )
            ]
        )
        
        # Create consumer step specification
        consumer_spec = StepSpecification(
            step_type="TrainingStep",
            node_type="sink",
            dependencies=[
                DependencySpec(
                    logical_name="training_data",
                    dependency_type=DependencyType.PROCESSING_OUTPUT,
                    data_type="S3Uri",
                    compatible_sources=["PreprocessingStep"],
                    semantic_keywords=["data"]
                )
            ],
            outputs=[]
        )
        
        # Calculate scores
        dep_spec = consumer_spec.dependencies["training_data"]
        output_spec = producer_spec.outputs["processed_data"]
        
        type_compatibility_score = 0.4  # Exact match
        data_type_score = 0.2  # Exact match
        semantic_score = resolver.semantic_matcher.calculate_similarity_with_aliases(
            dep_spec.logical_name, output_spec
        ) * 0.25
        direct_match_bonus = 0.0
        compatible_source_score = 0.1
        keyword_match_score = resolver._calculate_keyword_match(dep_spec.semantic_keywords, output_spec.logical_name) * 0.05
        
        # Expected score
        expected_score = (
            type_compatibility_score +
            data_type_score +
            semantic_score +
            direct_match_bonus +
            compatible_source_score +
            keyword_match_score
        )
        
        # Actual score
        actual_score = resolver._calculate_compatibility(dep_spec, output_spec, producer_spec)
        
        # Assert that the scores are equal
        self.assertAlmostEqual(expected_score, actual_score, places=5)

    def test_multiple_competing_candidates(self):
        """Test that resolver selects the best candidate among multiple options."""
        registry = SpecificationRegistry()
        semantic_matcher = SemanticMatcher()
        resolver = UnifiedDependencyResolver(registry, semantic_matcher)
        
        # Create three producer step specifications with varying compatibility scores
        # Producer 1: Good type match but poor semantic match
        producer1_spec = StepSpecification(
            step_type="DataLoadingStep",
            node_type="source",
            outputs=[
                OutputSpec(
                    logical_name="raw_data",
                    output_type=DependencyType.PROCESSING_OUTPUT,
                    property_path="properties.ProcessingOutputConfig.Outputs['raw_data'].S3Output.S3Uri",
                    data_type="S3Uri"
                )
            ]
        )
        
        # Producer 2: Perfect match (best candidate)
        producer2_spec = StepSpecification(
            step_type="PreprocessingStep",
            node_type="source",
            outputs=[
                OutputSpec(
                    logical_name="training_dataset",
                    aliases=["training_data"],
                    output_type=DependencyType.PROCESSING_OUTPUT,
                    property_path="properties.ProcessingOutputConfig.Outputs['training_dataset'].S3Output.S3Uri",
                    data_type="S3Uri"
                )
            ]
        )
        
        # Producer 3: Good semantic match but wrong type
        producer3_spec = StepSpecification(
            step_type="EvaluationStep",
            node_type="source",
            outputs=[
                OutputSpec(
                    logical_name="training_metrics",
                    output_type=DependencyType.CUSTOM_PROPERTY,
                    property_path="properties.CustomMetrics",
                    data_type="String"
                )
            ]
        )
        
        # Consumer step that needs training data
        consumer_spec = StepSpecification(
            step_type="TrainingStep",
            node_type="sink",
            dependencies=[
                DependencySpec(
                    logical_name="training_data",
                    dependency_type=DependencyType.PROCESSING_OUTPUT,
                    required=True,
                    compatible_sources=["DataLoadingStep", "PreprocessingStep"],
                    data_type="S3Uri"
                )
            ]
        )
        
        # Register all steps
        registry.register("data_loading", producer1_spec)
        registry.register("preprocessing", producer2_spec)
        registry.register("evaluation", producer3_spec)
        registry.register("training", consumer_spec)
        
        # Resolve dependencies
        resolved = resolver.resolve_step_dependencies("training", ["data_loading", "preprocessing", "evaluation"])
        
        # Assert that the resolver selected the best candidate (producer2)
        self.assertIn("training_data", resolved)
        self.assertEqual(resolved["training_data"].step_name, "preprocessing")
        self.assertEqual(resolved["training_data"].output_spec.logical_name, "training_dataset")

    def test_type_compatibility_matrix(self):
        """Test compatibility matrix for different dependency types."""
        registry = SpecificationRegistry()
        semantic_matcher = SemanticMatcher()
        resolver = UnifiedDependencyResolver(registry, semantic_matcher)
        
        # Test compatible types according to the matrix in dependency_resolver.py
        compatible_pairs = [
            (DependencyType.MODEL_ARTIFACTS, DependencyType.MODEL_ARTIFACTS),
            (DependencyType.TRAINING_DATA, DependencyType.PROCESSING_OUTPUT),
            (DependencyType.TRAINING_DATA, DependencyType.TRAINING_DATA),
            (DependencyType.PROCESSING_OUTPUT, DependencyType.PROCESSING_OUTPUT),
            (DependencyType.PROCESSING_OUTPUT, DependencyType.TRAINING_DATA),
            (DependencyType.HYPERPARAMETERS, DependencyType.HYPERPARAMETERS),
            (DependencyType.HYPERPARAMETERS, DependencyType.CUSTOM_PROPERTY),
            (DependencyType.PAYLOAD_SAMPLES, DependencyType.PAYLOAD_SAMPLES),
            (DependencyType.PAYLOAD_SAMPLES, DependencyType.PROCESSING_OUTPUT),
            (DependencyType.CUSTOM_PROPERTY, DependencyType.CUSTOM_PROPERTY)
        ]
        
        # Test incompatible types
        incompatible_pairs = [
            (DependencyType.MODEL_ARTIFACTS, DependencyType.PROCESSING_OUTPUT),
            (DependencyType.MODEL_ARTIFACTS, DependencyType.TRAINING_DATA),
            (DependencyType.MODEL_ARTIFACTS, DependencyType.HYPERPARAMETERS),
            (DependencyType.TRAINING_DATA, DependencyType.MODEL_ARTIFACTS),
            (DependencyType.PROCESSING_OUTPUT, DependencyType.HYPERPARAMETERS),
            (DependencyType.HYPERPARAMETERS, DependencyType.MODEL_ARTIFACTS),
            (DependencyType.PAYLOAD_SAMPLES, DependencyType.MODEL_ARTIFACTS),
            (DependencyType.CUSTOM_PROPERTY, DependencyType.MODEL_ARTIFACTS)
        ]
        
        # Assert compatible types
        for dep_type, output_type in compatible_pairs:
            self.assertTrue(
                resolver._are_types_compatible(dep_type, output_type),
                f"Types should be compatible: {dep_type.value} <- {output_type.value}"
            )
        
        # Assert incompatible types
        for dep_type, output_type in incompatible_pairs:
            self.assertFalse(
                resolver._are_types_compatible(dep_type, output_type),
                f"Types should not be compatible: {dep_type.value} <- {output_type.value}"
            )

    def test_data_type_compatibility(self):
        """Test data type compatibility rules."""
        registry = SpecificationRegistry()
        semantic_matcher = SemanticMatcher()
        resolver = UnifiedDependencyResolver(registry, semantic_matcher)
        
        # Compatible data type pairs (dep_type, output_type)
        compatible_pairs = [
            ("S3Uri", "S3Uri"),
            ("S3Uri", "String"),
            ("String", "String"),
            ("String", "S3Uri"),
            ("Integer", "Integer"),
            ("Integer", "Float"),
            ("Float", "Float"),
            ("Float", "Integer"),
            ("Boolean", "Boolean")
        ]
        
        # Incompatible data type pairs
        incompatible_pairs = [
            ("S3Uri", "Integer"),
            ("S3Uri", "Boolean"),
            ("String", "Integer"),
            ("String", "Boolean"),
            ("Integer", "Boolean"),
            ("Integer", "S3Uri"),
            ("Float", "Boolean"),
            ("Float", "String"),
            ("Boolean", "Integer"),
            ("Boolean", "Float"),
            ("Boolean", "String"),
            ("Boolean", "S3Uri")
        ]
        
        # Assert compatible data types
        for dep_type, output_type in compatible_pairs:
            self.assertTrue(
                resolver._are_data_types_compatible(dep_type, output_type),
                f"Data types should be compatible: {dep_type} <- {output_type}"
            )
        
        # Assert incompatible data types
        for dep_type, output_type in incompatible_pairs:
            self.assertFalse(
                resolver._are_data_types_compatible(dep_type, output_type),
                f"Data types should not be compatible: {dep_type} <- {output_type}"
            )

    def test_semantic_matching(self):
        """Test semantic similarity for name matching."""
        semantic_matcher = SemanticMatcher()
        
        # Test exact match
        exact_score = semantic_matcher.calculate_similarity("training_data", "training_data")
        self.assertEqual(exact_score, 1.0)
        
        # Test close match
        close_score = semantic_matcher.calculate_similarity("training_data", "training_dataset")
        self.assertGreater(close_score, 0.7)
        
        # Test partial match
        partial_score = semantic_matcher.calculate_similarity("training_data", "data")
        self.assertGreater(partial_score, 0.3)
        
        # Test synonym match based on semantic matcher's synonym dictionary
        # "data" and "dataset" should be considered related
        synonym_score = semantic_matcher.calculate_similarity("data", "dataset")
        self.assertGreater(synonym_score, 0.3)
        
        # Test unrelated terms
        unrelated_score = semantic_matcher.calculate_similarity("training_data", "model_output")
        self.assertLess(unrelated_score, 0.3)
        
        # Test with aliases
        output_spec = OutputSpec(
            logical_name="processed_data",
            aliases=["training_data", "model_input"],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri"
        )
        
        alias_score = semantic_matcher.calculate_similarity_with_aliases("training_dataset", output_spec)
        self.assertGreater(alias_score, 0.7)

    def test_required_vs_optional_dependencies(self):
        """Test resolution of required vs. optional dependencies."""
        registry = SpecificationRegistry()
        semantic_matcher = SemanticMatcher()
        resolver = UnifiedDependencyResolver(registry, semantic_matcher)
        
        # Create a producer step
        producer_spec = StepSpecification(
            step_type="PreprocessingStep",
            node_type="source",
            outputs=[
                OutputSpec(
                    logical_name="processed_data",
                    output_type=DependencyType.PROCESSING_OUTPUT,
                    property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
                    data_type="S3Uri"
                )
            ]
        )
        
        # Consumer with one required and one optional dependency
        # Use CUSTOM_PROPERTY type for validation_data to prevent matching with processed_data
        consumer_spec = StepSpecification(
            step_type="TrainingStep",
            node_type="internal",
            dependencies=[
                DependencySpec(
                    logical_name="training_data",
                    dependency_type=DependencyType.PROCESSING_OUTPUT,
                    required=True,
                    compatible_sources=["PreprocessingStep"],
                    data_type="S3Uri"
                ),
                DependencySpec(
                    logical_name="validation_data",
                    dependency_type=DependencyType.CUSTOM_PROPERTY,  # Different type to avoid matching
                    required=False,  # Optional dependency
                    compatible_sources=["ValidationStep"],
                    data_type="String"  # Different data type
                )
            ],
            outputs=[
                OutputSpec(
                    logical_name="model",
                    output_type=DependencyType.MODEL_ARTIFACTS,
                    property_path="properties.ModelArtifacts.S3ModelArtifacts",
                    data_type="S3Uri"
                )
            ]
        )
        
        # Register the steps
        registry.register("preprocessing", producer_spec)
        registry.register("training", consumer_spec)
        
        # Resolve dependencies (only required dependency will be resolved)
        resolved = resolver.resolve_step_dependencies("training", ["preprocessing"])
        
        # Assert that only the required dependency is resolved
        self.assertIn("training_data", resolved)
        self.assertEqual(resolved["training_data"].step_name, "preprocessing")
        self.assertNotIn("validation_data", resolved)  # Optional dependency not resolved

    def test_unresolvable_dependencies(self):
        """Test error handling for unresolvable dependencies."""
        registry = SpecificationRegistry()
        semantic_matcher = SemanticMatcher()
        resolver = UnifiedDependencyResolver(registry, semantic_matcher)
        
        # Create a consumer step with a required dependency
        consumer_spec = StepSpecification(
            step_type="TrainingStep",
            node_type="sink",
            dependencies=[
                DependencySpec(
                    logical_name="training_data",
                    dependency_type=DependencyType.PROCESSING_OUTPUT,
                    required=True,
                    compatible_sources=["PreprocessingStep"],
                    data_type="S3Uri"
                )
            ]
        )
        
        # Create an incompatible producer (wrong type)
        wrong_type_producer = StepSpecification(
            step_type="PreprocessingStep",
            node_type="source",
            outputs=[
                OutputSpec(
                    logical_name="processed_data",
                    output_type=DependencyType.CUSTOM_PROPERTY,  # Wrong type
                    property_path="properties.CustomProperty",
                    data_type="S3Uri"
                )
            ]
        )
        
        # Register the steps
        registry.register("training", consumer_spec)
        registry.register("wrong_producer", wrong_type_producer)
        
        # Attempt to resolve with incompatible producer
        with self.assertRaises(DependencyResolutionError) as context:
            resolver.resolve_step_dependencies("training", ["wrong_producer"])
        
        # Check that the error message mentions the unresolved dependency
        self.assertIn("training_data", str(context.exception))
        self.assertIn("unresolved required dependencies", str(context.exception).lower())

    def test_registry_isolation(self):
        """Test that different registry contexts remain isolated."""
        # Create two separate registries
        registry1 = SpecificationRegistry()
        registry2 = SpecificationRegistry()
        
        # Create resolvers for each registry
        resolver1 = UnifiedDependencyResolver(registry1, SemanticMatcher())
        resolver2 = UnifiedDependencyResolver(registry2, SemanticMatcher())
        
        # Create a specification for each registry
        spec1 = StepSpecification(
            step_type="PreprocessingStep",
            node_type="source",
            outputs=[
                OutputSpec(
                    logical_name="processed_data",
                    output_type=DependencyType.PROCESSING_OUTPUT,
                    property_path="properties.test",
                    data_type="S3Uri"
                )
            ]
        )
        
        spec2 = StepSpecification(
            step_type="ProcessingStep",  # Different name
            node_type="source",
            outputs=[
                OutputSpec(
                    logical_name="data_output",  # Different name
                    output_type=DependencyType.PROCESSING_OUTPUT,
                    property_path="properties.test",
                    data_type="S3Uri"
                )
            ]
        )
        
        # Register specs in their respective registries
        registry1.register("step1", spec1)
        registry2.register("step1", spec2)  # Same step name, different spec
        
        # Assert registries have isolated contents
        self.assertEqual(registry1.get_specification("step1").step_type, "PreprocessingStep")
        self.assertEqual(registry2.get_specification("step1").step_type, "ProcessingStep")
        
        # Assert resolvers use their own registry
        self.assertEqual(resolver1.registry.get_specification("step1").step_type, "PreprocessingStep")
        self.assertEqual(resolver2.registry.get_specification("step1").step_type, "ProcessingStep")

    def test_property_reference_functionality(self):
        """Test PropertyReference navigation and conversion."""
        # Create an output spec
        output_spec = OutputSpec(
            logical_name="processed_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
            data_type="S3Uri"
        )
        
        # Create a property reference
        prop_ref = PropertyReference(
            step_name="preprocessing",
            output_spec=output_spec
        )
        
        # Test to_sagemaker_property - implementation removes "properties." prefix
        sagemaker_prop = prop_ref.to_sagemaker_property()
        self.assertEqual(
            sagemaker_prop,
            {"Get": "Steps.preprocessing.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri"}
        )
        
        # Test string representation
        self.assertEqual(str(prop_ref), "preprocessing.processed_data")
        
        # Test parsing of property path
        path_parts = prop_ref._parse_property_path(output_spec.property_path)
        
        # Expected path parts
        expected_parts = [
            "ProcessingOutputConfig", 
            ("Outputs", "processed_data"), 
            "S3Output", 
            "S3Uri"
        ]
        
        self.assertEqual(len(path_parts), len(expected_parts))
        
        # Check first part is a string
        self.assertEqual(path_parts[0], expected_parts[0])
        
        # Check second part is a tuple with correct dict access
        self.assertIsInstance(path_parts[1], tuple)
        self.assertEqual(path_parts[1][0], expected_parts[1][0])
        self.assertEqual(path_parts[1][1], expected_parts[1][1])
        
        # Check remaining parts
        self.assertEqual(path_parts[2], expected_parts[2])
        self.assertEqual(path_parts[3], expected_parts[3])

    def test_end_to_end_resolution(self):
        """Test end-to-end resolution of a multi-step pipeline."""
        registry = SpecificationRegistry()
        semantic_matcher = SemanticMatcher()
        resolver = UnifiedDependencyResolver(registry, semantic_matcher)
        
        # Create a chain of steps: data loading → preprocessing → training → evaluation
        data_loading_spec = StepSpecification(
            step_type="DataLoadingStep",
            node_type="source",
            outputs=[
                OutputSpec(
                    logical_name="raw_data",
                    output_type=DependencyType.PROCESSING_OUTPUT,
                    property_path="properties.ProcessingOutputConfig.Outputs['raw_data'].S3Output.S3Uri",
                    data_type="S3Uri"
                )
            ]
        )
        
        preprocessing_spec = StepSpecification(
            step_type="PreprocessingStep",
            node_type="internal",
            dependencies=[
                DependencySpec(
                    logical_name="input_data",
                    dependency_type=DependencyType.PROCESSING_OUTPUT,
                    required=True,
                    compatible_sources=["DataLoadingStep"],
                    data_type="S3Uri"
                )
            ],
            outputs=[
                OutputSpec(
                    logical_name="processed_data",
                    output_type=DependencyType.PROCESSING_OUTPUT,
                    property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
                    data_type="S3Uri"
                ),
                OutputSpec(
                    logical_name="validation_data",
                    output_type=DependencyType.PROCESSING_OUTPUT,
                    property_path="properties.ProcessingOutputConfig.Outputs['validation_data'].S3Output.S3Uri",
                    data_type="S3Uri"
                )
            ]
        )
        
        training_spec = StepSpecification(
            step_type="TrainingStep",
            node_type="internal",
            dependencies=[
                DependencySpec(
                    logical_name="training_data",
                    dependency_type=DependencyType.PROCESSING_OUTPUT,
                    required=True,
                    compatible_sources=["PreprocessingStep"],
                    data_type="S3Uri"
                ),
                DependencySpec(
                    logical_name="validation_data",
                    dependency_type=DependencyType.PROCESSING_OUTPUT,
                    required=True,
                    compatible_sources=["PreprocessingStep"],
                    data_type="S3Uri"
                )
            ],
            outputs=[
                OutputSpec(
                    logical_name="model_artifacts",
                    output_type=DependencyType.MODEL_ARTIFACTS,
                    property_path="properties.ModelArtifacts.S3ModelArtifacts",
                    data_type="S3Uri"
                )
            ]
        )
        
        evaluation_spec = StepSpecification(
            step_type="EvaluationStep",
            node_type="sink",
            dependencies=[
                DependencySpec(
                    logical_name="model",
                    dependency_type=DependencyType.MODEL_ARTIFACTS,
                    required=True,
                    compatible_sources=["TrainingStep"],
                    data_type="S3Uri"
                ),
                DependencySpec(
                    logical_name="test_data",
                    dependency_type=DependencyType.PROCESSING_OUTPUT,
                    required=True,
                    compatible_sources=["PreprocessingStep"],
                    semantic_keywords=["validation", "test"],  # Added keywords to match validation_data
                    data_type="S3Uri"
                )
            ],
            outputs=[]
        )
        
        # Register all steps
        registry.register("data_loading", data_loading_spec)
        registry.register("preprocessing", preprocessing_spec)
        registry.register("training", training_spec)
        registry.register("evaluation", evaluation_spec)
        
        # Resolve all dependencies
        all_steps = ["data_loading", "preprocessing", "training", "evaluation"]
        resolved = resolver.resolve_all_dependencies(all_steps)
        
        # Check that all expected steps have dependencies resolved
        self.assertIn("preprocessing", resolved)
        self.assertIn("training", resolved)
        self.assertIn("evaluation", resolved)
        
        # Check preprocessing step dependencies
        self.assertIn("input_data", resolved["preprocessing"])
        self.assertEqual(resolved["preprocessing"]["input_data"].step_name, "data_loading")
        self.assertEqual(resolved["preprocessing"]["input_data"].output_spec.logical_name, "raw_data")
        
        # Check training step dependencies
        self.assertIn("training_data", resolved["training"])
        self.assertIn("validation_data", resolved["training"])
        self.assertEqual(resolved["training"]["training_data"].step_name, "preprocessing")
        self.assertEqual(resolved["training"]["training_data"].output_spec.logical_name, "processed_data")
        self.assertEqual(resolved["training"]["validation_data"].step_name, "preprocessing")
        self.assertEqual(resolved["training"]["validation_data"].output_spec.logical_name, "validation_data")
        
        # Check evaluation step dependencies
        self.assertIn("model", resolved["evaluation"])
        self.assertIn("test_data", resolved["evaluation"])
        self.assertEqual(resolved["evaluation"]["model"].step_name, "training")
        self.assertEqual(resolved["evaluation"]["model"].output_spec.logical_name, "model_artifacts")
        self.assertEqual(resolved["evaluation"]["test_data"].step_name, "preprocessing")
        # Given the added semantic keywords, it should now match with validation_data
        self.assertEqual(resolved["evaluation"]["test_data"].output_spec.logical_name, "validation_data")


if __name__ == '__main__':
    unittest.main()
