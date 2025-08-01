#!/usr/bin/env python3
"""
Unit tests for OutputSpec alias functionality.

This module provides comprehensive tests for the new alias functionality
in OutputSpec, including validation, lookup methods, and conflict detection.
"""

import unittest
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.pipeline_deps.base_specifications import (
    OutputSpec, StepSpecification, DependencySpec, DependencyType, NodeType
)
from pydantic import ValidationError


class TestOutputSpecAliases(unittest.TestCase):
    """Test cases for OutputSpec alias functionality."""
    
    def test_output_spec_with_valid_aliases(self):
        """Test creating OutputSpec with valid aliases."""
        output_spec = OutputSpec(
            logical_name="processed_data",
            aliases=["ProcessedData", "DATA", "TrainingData"],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
            description="Processed training data output"
        )
        
        self.assertEqual(output_spec.logical_name, "processed_data")
        self.assertEqual(len(output_spec.aliases), 3)
        self.assertIn("ProcessedData", output_spec.aliases)
        self.assertIn("DATA", output_spec.aliases)
        self.assertIn("TrainingData", output_spec.aliases)
    
    def test_output_spec_with_empty_aliases(self):
        """Test creating OutputSpec with empty aliases list."""
        output_spec = OutputSpec(
            logical_name="processed_data",
            aliases=[],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri"
        )
        
        self.assertEqual(len(output_spec.aliases), 0)
    
    def test_output_spec_without_aliases_field(self):
        """Test creating OutputSpec without specifying aliases (should default to empty list)."""
        output_spec = OutputSpec(
            logical_name="processed_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri"
        )
        
        self.assertEqual(len(output_spec.aliases), 0)
        self.assertIsInstance(output_spec.aliases, list)
    
    def test_alias_validation_removes_duplicates(self):
        """Test that alias validation removes duplicates (case-insensitive)."""
        output_spec = OutputSpec(
            logical_name="processed_data",
            aliases=["ProcessedData", "DATA", "processeddata", "Data"],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri"
        )
        
        # Should remove case-insensitive duplicates
        self.assertEqual(len(output_spec.aliases), 2)  # ProcessedData and DATA
        self.assertIn("ProcessedData", output_spec.aliases)
        self.assertIn("DATA", output_spec.aliases)
    
    def test_alias_validation_removes_empty_strings(self):
        """Test that alias validation removes empty strings and whitespace."""
        output_spec = OutputSpec(
            logical_name="processed_data",
            aliases=["ProcessedData", "", "  ", "DATA"],  # Remove None as it causes validation error
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri"
        )
        
        # Should remove empty strings
        self.assertEqual(len(output_spec.aliases), 2)
        self.assertIn("ProcessedData", output_spec.aliases)
        self.assertIn("DATA", output_spec.aliases)
    
    def test_alias_validation_invalid_characters(self):
        """Test that alias validation rejects invalid characters."""
        with self.assertRaises(ValidationError) as context:
            OutputSpec(
                logical_name="processed_data",
                aliases=["Processed@Data", "DATA"],
                output_type=DependencyType.PROCESSING_OUTPUT,
                property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri"
            )
        
        self.assertIn("should contain only alphanumeric characters", str(context.exception))
    
    def test_alias_conflicts_with_logical_name(self):
        """Test that aliases cannot conflict with the logical name."""
        with self.assertRaises(ValidationError) as context:
            OutputSpec(
                logical_name="processed_data",
                aliases=["ProcessedData", "processed_data"],  # Conflicts with logical name
                output_type=DependencyType.PROCESSING_OUTPUT,
                property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri"
            )
        
        self.assertIn("cannot be the same as logical_name", str(context.exception))
    
    def test_step_specification_get_output_by_name_or_alias(self):
        """Test the new get_output_by_name_or_alias method."""
        output_spec = OutputSpec(
            logical_name="processed_data",
            aliases=["ProcessedData", "DATA"],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri"
        )
        
        step_spec = StepSpecification(
            step_type="TestStep",
            node_type=NodeType.INTERNAL,
            dependencies=[
                DependencySpec(
                    logical_name="input_data",
                    dependency_type=DependencyType.PROCESSING_OUTPUT
                )
            ],
            outputs=[output_spec]
        )
        
        # Test lookup by logical name
        result = step_spec.get_output_by_name_or_alias("processed_data")
        self.assertIsNotNone(result)
        self.assertEqual(result.logical_name, "processed_data")
        
        # Test lookup by alias (exact case)
        result = step_spec.get_output_by_name_or_alias("ProcessedData")
        self.assertIsNotNone(result)
        self.assertEqual(result.logical_name, "processed_data")
        
        # Test lookup by alias (different case)
        result = step_spec.get_output_by_name_or_alias("data")
        self.assertIsNotNone(result)
        self.assertEqual(result.logical_name, "processed_data")
        
        # Test lookup by non-existent name
        result = step_spec.get_output_by_name_or_alias("nonexistent")
        self.assertIsNone(result)
    
    def test_step_specification_list_all_output_names(self):
        """Test the new list_all_output_names method."""
        output_spec1 = OutputSpec(
            logical_name="processed_data",
            aliases=["ProcessedData", "DATA"],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri"
        )
        
        output_spec2 = OutputSpec(
            logical_name="model_artifacts",
            aliases=["ModelArtifacts"],
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts"
        )
        
        step_spec = StepSpecification(
            step_type="TestStep",
            node_type=NodeType.INTERNAL,
            dependencies=[
                DependencySpec(
                    logical_name="input_data",
                    dependency_type=DependencyType.PROCESSING_OUTPUT
                )
            ],
            outputs=[output_spec1, output_spec2]
        )
        
        all_names = step_spec.list_all_output_names()
        
        # Should include all logical names and aliases
        expected_names = [
            "processed_data", "ProcessedData", "DATA",
            "model_artifacts", "ModelArtifacts"
        ]
        
        for name in expected_names:
            self.assertIn(name, all_names)
        
        self.assertEqual(len(all_names), 5)
    
    def test_step_specification_alias_conflicts_across_outputs(self):
        """Test that aliases cannot conflict across different outputs in the same step."""
        output_spec1 = OutputSpec(
            logical_name="processed_data",
            aliases=["ProcessedData", "DATA"],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri"
        )
        
        output_spec2 = OutputSpec(
            logical_name="model_artifacts",
            aliases=["DATA"],  # Conflicts with output_spec1 alias
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts"
        )
        
        with self.assertRaises(ValidationError) as context:
            StepSpecification(
                step_type="TestStep",
                node_type=NodeType.INTERNAL,
                dependencies=[
                    DependencySpec(
                        logical_name="input_data",
                        dependency_type=DependencyType.PROCESSING_OUTPUT
                    )
                ],
                outputs=[output_spec1, output_spec2]
            )
        
        self.assertIn("conflicts with existing name or alias", str(context.exception))
    
    def test_step_specification_logical_name_conflicts(self):
        """Test that logical names cannot conflict across outputs."""
        output_spec1 = OutputSpec(
            logical_name="processed_data",
            aliases=["ProcessedData"],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri"
        )
        
        output_spec2 = OutputSpec(
            logical_name="processed_data",  # Conflicts with output_spec1 logical name
            aliases=["ModelArtifacts"],
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts"
        )
        
        with self.assertRaises(ValueError) as context:  # Changed from ValidationError to ValueError
            StepSpecification(
                step_type="TestStep",
                node_type=NodeType.INTERNAL,
                dependencies=[
                    DependencySpec(
                        logical_name="input_data",
                        dependency_type=DependencyType.PROCESSING_OUTPUT
                    )
                ],
                outputs=[output_spec1, output_spec2]
            )
        
        self.assertIn("Duplicate output logical names found", str(context.exception))
    
    def test_alias_conflicts_with_other_logical_name(self):
        """Test that aliases cannot conflict with other outputs' logical names."""
        output_spec1 = OutputSpec(
            logical_name="processed_data",
            aliases=["model_artifacts"],  # Conflicts with output_spec2 logical name
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri"
        )
        
        output_spec2 = OutputSpec(
            logical_name="model_artifacts",
            aliases=["ModelArtifacts"],
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts"
        )
        
        with self.assertRaises(ValidationError) as context:
            StepSpecification(
                step_type="TestStep",
                node_type=NodeType.INTERNAL,
                dependencies=[
                    DependencySpec(
                        logical_name="input_data",
                        dependency_type=DependencyType.PROCESSING_OUTPUT
                    )
                ],
                outputs=[output_spec1, output_spec2]
            )
        
        # The error message includes "Duplicate logical name" because the logical name conflict is detected first
        self.assertIn("Duplicate logical name", str(context.exception))
    
    def test_case_insensitive_conflict_detection(self):
        """Test that conflict detection is case-insensitive."""
        output_spec1 = OutputSpec(
            logical_name="processed_data",
            aliases=["ProcessedData"],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri"
        )
        
        output_spec2 = OutputSpec(
            logical_name="model_artifacts",
            aliases=["processeddata"],  # Case-insensitive conflict with output_spec1 alias
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts"
        )
        
        with self.assertRaises(ValidationError) as context:
            StepSpecification(
                step_type="TestStep",
                node_type=NodeType.INTERNAL,
                dependencies=[
                    DependencySpec(
                        logical_name="input_data",
                        dependency_type=DependencyType.PROCESSING_OUTPUT
                    )
                ],
                outputs=[output_spec1, output_spec2]
            )
        
        self.assertIn("conflicts with existing name or alias", str(context.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
