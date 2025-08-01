import unittest
from src.pipeline_deps import (
    SemanticMatcher, OutputSpec, DependencyType,
    UnifiedDependencyResolver, SpecificationRegistry,
    StepSpecification, DependencySpec, NodeType
)

class TestSemanticMatcher(unittest.TestCase):
    def test_calculate_similarity_with_aliases(self):
        """Test that calculate_similarity_with_aliases returns the highest similarity."""
        matcher = SemanticMatcher()
        
        # Create test output spec with aliases
        output_spec = OutputSpec(
            logical_name="processed_data",
            aliases=["training_data", "input_features", "ModelData"],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri"
        )
        
        # Test with various dependency names
        test_cases = [
            # (dep_name, expected_min_score, expected_best_match)
            ("training_data", 1.0, "training_data"),        # Exact match with alias
            ("train_data", 0.7, "training_data"),           # Close match with alias
            ("processed_data", 1.0, "processed_data"),      # Exact match with logical_name
            ("model_data", 0.5, "ModelData"),               # Close match with alias
            ("feature_data", 0.5, "input_features"),        # Moderate match with alias
            ("unrelated_name", 0.1, "processed_data")       # No good match (defaults to logical)
        ]
        
        for dep_name, expected_min_score, expected_best_match in test_cases:
            score = matcher.calculate_similarity_with_aliases(dep_name, output_spec)
            self.assertGreaterEqual(score, expected_min_score, f"Score for '{dep_name}' below expectation")

    def test_weight_calculation(self):
        """Test that the weight calculation is correct."""
        matcher = SemanticMatcher()
        
        # Test with known values
        name1 = "training_data"
        name2 = "train_data"
        
        # Calculate scores
        norm1 = matcher._normalize_name(name1)
        norm2 = matcher._normalize_name(name2)
        string_sim = matcher._calculate_string_similarity(norm1, norm2)
        token_sim = matcher._calculate_token_similarity(norm1, norm2)
        semantic_sim = matcher._calculate_semantic_similarity(norm1, norm2)
        substring_sim = matcher._calculate_substring_similarity(norm1, norm2)
        
        # Expected score
        expected_score = (
            string_sim * 0.3 +
            token_sim * 0.25 +
            semantic_sim * 0.25 +
            substring_sim * 0.2
        )
        
        # Actual score
        actual_score = matcher.calculate_similarity(name1, name2)
        
        # Assert that the scores are equal
        self.assertAlmostEqual(expected_score, actual_score, places=5)

    def test_name_normalization(self):
        """Test name normalization with different formats and stop words."""
        matcher = SemanticMatcher()
        
        # Test cases: (input, expected_normalized)
        test_cases = [
            ("training_data", "training data"),
            ("training-data", "training data"),
            ("training.data", "training data"),
            ("TrainingData", "trainingdata"),  # No spaces in camelCase after normalization
            ("the training data", "training data"),  # Stop word removal
            ("training and data", "training data"),  # Stop word removal
            ("config", "configuration"),             # Abbreviation expansion
            ("eval metrics", "evaluation metrics")   # Abbreviation expansion
        ]
        
        for input_name, expected_output in test_cases:
            normalized = matcher._normalize_name(input_name)
            self.assertEqual(normalized, expected_output, f"Failed to normalize '{input_name}'")

    def test_individual_similarity_metrics(self):
        """Test each individual similarity metric."""
        matcher = SemanticMatcher()
        
        # String similarity (sequence matching)
        string_sim = matcher._calculate_string_similarity("training data", "training dataset")
        self.assertGreaterEqual(string_sim, 0.85)
        self.assertLessEqual(string_sim, 0.95)  # Allow some flexibility in the implementation
        
        # Token similarity
        self.assertEqual(
            matcher._calculate_token_similarity("training data", "data training"),
            1.0  # Same tokens, different order = perfect match
        )
        self.assertAlmostEqual(
            matcher._calculate_token_similarity("training data", "training model"),
            0.33, places=2  # 1 common token out of 3 unique tokens
        )
        
        # Semantic similarity 
        semantic_sim = matcher._calculate_semantic_similarity("training data", "train dataset")
        self.assertGreaterEqual(semantic_sim, 0.25)  # Should have some matches due to synonyms
        self.assertLessEqual(semantic_sim, 0.8)  # But not a perfect match
        
        # Substring similarity
        substring_sim = matcher._calculate_substring_similarity("training", "training data")
        self.assertGreaterEqual(substring_sim, 0.57)  # At least 8/14 characters
        self.assertLessEqual(substring_sim, 0.65)  # Allow for implementation variations

    def test_synonym_matching(self):
        """Test synonym recognition capability."""
        matcher = SemanticMatcher()
        
        # Test synonym pairs
        self.assertTrue(matcher._are_synonyms("model", "artifact"))
        self.assertTrue(matcher._are_synonyms("data", "dataset"))
        self.assertTrue(matcher._are_synonyms("config", "parameters"))
        
        # Test non-synonym pairs
        self.assertFalse(matcher._are_synonyms("model", "data"))
        self.assertFalse(matcher._are_synonyms("training", "evaluation"))

    def test_find_best_matches(self):
        """Test finding best matches from candidates."""
        matcher = SemanticMatcher()
        
        candidates = [
            "model_output", 
            "training_dataset", 
            "processed_data",
            "validation_data", 
            "test_results"
        ]
        
        # Search for training data
        matches = matcher.find_best_matches("training_data", candidates, threshold=0.7)
        self.assertGreaterEqual(len(matches), 1)  # Should match at least "training_dataset" 
        self.assertEqual(matches[0][0], "training_dataset")  # Best match should be this
        
        # Search with low threshold
        matches_low_threshold = matcher.find_best_matches("training_data", candidates, threshold=0.3)
        self.assertGreater(len(matches_low_threshold), len(matches))  # Should match more items
        
        # Search with no matches
        no_matches = matcher.find_best_matches("completely_different", candidates, threshold=0.7)
        self.assertEqual(len(no_matches), 0)  # Should find no matches

    def test_similarity_explanation(self):
        """Test detailed similarity explanation."""
        matcher = SemanticMatcher()
        
        explanation = matcher.explain_similarity("training_data", "training_dataset")
        
        # Check explanation structure
        self.assertIn('overall_score', explanation)
        self.assertIn('normalized_names', explanation)
        self.assertIn('string_similarity', explanation)
        self.assertIn('token_similarity', explanation)
        self.assertIn('semantic_similarity', explanation)
        self.assertIn('substring_similarity', explanation)
        
        # Check explanation values
        self.assertGreater(explanation['overall_score'], 0.7)
        self.assertEqual(explanation['normalized_names'][0], "training data")

    def test_edge_cases(self):
        """Test edge cases for semantic matching."""
        matcher = SemanticMatcher()
        
        # Empty strings
        self.assertEqual(matcher.calculate_similarity("", "data"), 0.0)
        self.assertEqual(matcher.calculate_similarity("data", ""), 0.0)
        self.assertEqual(matcher.calculate_similarity("", ""), 0.0)
        
        # Identical strings
        self.assertEqual(matcher.calculate_similarity("data", "data"), 1.0)
        
        # Names with only stop words
        self.assertLess(matcher.calculate_similarity("the and of", "data"), 0.3)
        
        # Special characters
        self.assertGreater(matcher.calculate_similarity("data-set!", "dataset"), 0.4)

    def test_integration_with_dependency_resolver(self):
        """Test that SemanticMatcher works as expected with dependency resolution."""
        # Create registry and matcher
        registry = SpecificationRegistry()
        matcher = SemanticMatcher()
        resolver = UnifiedDependencyResolver(registry, matcher)
        
        # Create specifications with similar but not identical names
        producer_spec = StepSpecification(
            step_type="DataStep",
            node_type=NodeType.SOURCE,
            outputs=[
                OutputSpec(
                    logical_name="processed_data",
                    output_type=DependencyType.PROCESSING_OUTPUT,
                    property_path="properties.test"
                )
            ]
        )
        
        consumer_spec = StepSpecification(
            step_type="ModelStep",
            node_type=NodeType.SINK,
            dependencies=[
                DependencySpec(
                    logical_name="training_dataset",  # Not exact match, but semantically similar
                    dependency_type=DependencyType.PROCESSING_OUTPUT,
                    required=True
                )
            ]
        )
        
        # Register specs
        registry.register("data_step", producer_spec)
        registry.register("model_step", consumer_spec)
        
        # Resolve dependencies
        resolved = resolver.resolve_step_dependencies("model_step", ["data_step"])
        
        # Should resolve successfully despite different names
        self.assertIn("training_dataset", resolved)
        self.assertEqual(resolved["training_dataset"].step_name, "data_step")
    
    def test_direct_name_matching(self):
        """
        Test direct name matching prioritization as mentioned in dependency_resolution_improvement.md.
        
        According to the document, direct logical name matching should be preferred over
        semantic matching to improve resolution clarity.
        """
        registry = SpecificationRegistry()
        matcher = SemanticMatcher()
        resolver = UnifiedDependencyResolver(registry, matcher)
        
        # Setup a case similar to the one described in dependency_resolution_improvement.md
        # where there were conflicting matches
        
        # Create a preprocessing step with "processed_data" output
        preprocessing_spec = StepSpecification(
            step_type="PreprocessingStep",
            node_type=NodeType.SOURCE,
            outputs=[
                OutputSpec(
                    logical_name="processed_data",
                    output_type=DependencyType.PROCESSING_OUTPUT,
                    property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri"
                )
            ]
        )
        
        # Create a training step with "evaluation_output" that could semantically match with "eval_data"
        training_spec = StepSpecification(
            step_type="TrainingStep",
            node_type=NodeType.SOURCE,
            outputs=[
                OutputSpec(
                    logical_name="evaluation_output",
                    output_type=DependencyType.PROCESSING_OUTPUT,
                    property_path="properties.OutputDataConfig.S3OutputPath"
                ),
                OutputSpec(
                    logical_name="model_output",
                    output_type=DependencyType.MODEL_ARTIFACTS,
                    property_path="properties.ModelArtifacts.S3ModelArtifacts"
                )
            ]
        )
        
        # Create an evaluation step that needs processed_data
        # Using direct name matching as recommended in the improvement doc
        evaluation_spec = StepSpecification(
            step_type="EvaluationStep",
            node_type=NodeType.SINK,
            dependencies=[
                DependencySpec(
                    logical_name="processed_data",  # Direct match with preprocessing output
                    dependency_type=DependencyType.PROCESSING_OUTPUT,
                    required=True
                ),
                DependencySpec(
                    logical_name="model_input",
                    dependency_type=DependencyType.MODEL_ARTIFACTS,
                    required=True
                )
            ]
        )
        
        # Register specs
        registry.register("preprocessing", preprocessing_spec)
        registry.register("training", training_spec)
        registry.register("evaluation", evaluation_spec)
        
        # Resolve dependencies
        resolved = resolver.resolve_step_dependencies("evaluation", ["preprocessing", "training"])
        
        # Verify that "processed_data" dependency maps to preprocessing's output
        # rather than training's "evaluation_output" despite potential semantic match
        self.assertIn("processed_data", resolved)
        self.assertEqual(resolved["processed_data"].step_name, "preprocessing")
        self.assertEqual(resolved["processed_data"].output_spec.logical_name, "processed_data")
    
    def test_domain_specific_synonyms(self):
        """Test domain-specific synonym matching from the knowledge base."""
        matcher = SemanticMatcher()
        
        # Test domain-specific synonym pairs from SemanticMatcher's knowledge base
        # Group synonyms by expected similarity level
        high_similarity_pairs = [
            ("training", "train"),
            ("preprocessing", "preprocess")
        ]
        
        medium_similarity_pairs = [
            ("data", "dataset"),
            ("parameters", "hyperparameters")
        ]
        
        low_similarity_pairs = [
            ("model", "artifact"),   # These are synonyms in context but not lexically similar
            ("payload", "sample"),   # These have lower lexical similarity
            ("output", "result")     # These have lower lexical similarity
        ]
        
        # Check high similarity pairs (>0.5)
        for word1, word2 in high_similarity_pairs:
            similarity = matcher.calculate_similarity(word1, word2)
            self.assertGreater(
                similarity, 0.5, 
                f"High similarity synonyms '{word1}' and '{word2}' should have high similarity"
            )
            
        # Check medium similarity pairs (>0.4)
        for word1, word2 in medium_similarity_pairs:
            similarity = matcher.calculate_similarity(word1, word2)
            self.assertGreater(
                similarity, 0.4, 
                f"Medium similarity synonyms '{word1}' and '{word2}' should have medium similarity"
            )
            
        # Check low similarity pairs (≥0.2)
        for word1, word2 in low_similarity_pairs:
            similarity = matcher.calculate_similarity(word1, word2)
            self.assertGreaterEqual(
                similarity, 0.2, 
                f"Low similarity synonyms '{word1}' and '{word2}' should have some similarity"
            )

if __name__ == '__main__':
    unittest.main()
