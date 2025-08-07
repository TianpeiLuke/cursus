import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import logging

# Add the project root to the Python path to allow for absolute imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.cursus.core.base.specification_base import (
    StepSpecification, OutputSpec, DependencySpec, 
    ValidationResult, AlignmentResult
)


class TestOutputSpec(unittest.TestCase):
    """Test cases for OutputSpec class."""
    
    def test_init_with_required_fields(self):
        """Test initialization with required fields."""
        output_spec = OutputSpec(
            logical_name="test_output",
            description="Test output description",
            property_path="Steps.TestStep.Properties.OutputDataConfig.S3OutputPath"
        )
        
        self.assertEqual(output_spec.logical_name, "test_output")
        self.assertEqual(output_spec.description, "Test output description")
        self.assertEqual(output_spec.property_path, "Steps.TestStep.Properties.OutputDataConfig.S3OutputPath")
        self.assertIsNone(output_spec.data_type)
        self.assertEqual(output_spec.aliases, [])
    
    def test_init_with_optional_fields(self):
        """Test initialization with optional fields."""
        output_spec = OutputSpec(
            logical_name="test_output",
            description="Test output description",
            property_path="Steps.TestStep.Properties.OutputDataConfig.S3OutputPath",
            data_type="s3_uri",
            aliases=["output_alias", "alt_output"]
        )
        
        self.assertEqual(output_spec.data_type, "s3_uri")
        self.assertEqual(output_spec.aliases, ["output_alias", "alt_output"])
    
    def test_matches_name_or_alias(self):
        """Test matching by name or alias."""
        output_spec = OutputSpec(
            logical_name="test_output",
            description="Test output description",
            property_path="Steps.TestStep.Properties.OutputDataConfig.S3OutputPath",
            aliases=["output_alias", "alt_output"]
        )
        
        # Test matching logical name
        self.assertTrue(output_spec.matches_name_or_alias("test_output"))
        
        # Test matching aliases
        self.assertTrue(output_spec.matches_name_or_alias("output_alias"))
        self.assertTrue(output_spec.matches_name_or_alias("alt_output"))
        
        # Test non-matching name
        self.assertFalse(output_spec.matches_name_or_alias("nonexistent"))


class TestDependencySpec(unittest.TestCase):
    """Test cases for DependencySpec class."""
    
    def test_init_with_required_fields(self):
        """Test initialization with required fields."""
        dep_spec = DependencySpec(
            logical_name="test_dependency",
            description="Test dependency description"
        )
        
        self.assertEqual(dep_spec.logical_name, "test_dependency")
        self.assertEqual(dep_spec.description, "Test dependency description")
        self.assertTrue(dep_spec.required)  # Default is True
        self.assertIsNone(dep_spec.data_type)
        self.assertEqual(dep_spec.aliases, [])
    
    def test_init_with_optional_fields(self):
        """Test initialization with optional fields."""
        dep_spec = DependencySpec(
            logical_name="test_dependency",
            description="Test dependency description",
            required=False,
            data_type="s3_uri",
            aliases=["dep_alias", "alt_dep"]
        )
        
        self.assertFalse(dep_spec.required)
        self.assertEqual(dep_spec.data_type, "s3_uri")
        self.assertEqual(dep_spec.aliases, ["dep_alias", "alt_dep"])
    
    def test_matches_name_or_alias(self):
        """Test matching by name or alias."""
        dep_spec = DependencySpec(
            logical_name="test_dependency",
            description="Test dependency description",
            aliases=["dep_alias", "alt_dep"]
        )
        
        # Test matching logical name
        self.assertTrue(dep_spec.matches_name_or_alias("test_dependency"))
        
        # Test matching aliases
        self.assertTrue(dep_spec.matches_name_or_alias("dep_alias"))
        self.assertTrue(dep_spec.matches_name_or_alias("alt_dep"))
        
        # Test non-matching name
        self.assertFalse(dep_spec.matches_name_or_alias("nonexistent"))


class TestValidationResult(unittest.TestCase):
    """Test cases for ValidationResult class."""
    
    def test_init_valid(self):
        """Test initialization with valid result."""
        result = ValidationResult(is_valid=True)
        
        self.assertTrue(result.is_valid)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.warnings, [])
    
    def test_init_invalid_with_errors(self):
        """Test initialization with invalid result and errors."""
        errors = ["Error 1", "Error 2"]
        warnings = ["Warning 1"]
        
        result = ValidationResult(
            is_valid=False,
            errors=errors,
            warnings=warnings
        )
        
        self.assertFalse(result.is_valid)
        self.assertEqual(result.errors, errors)
        self.assertEqual(result.warnings, warnings)
    
    def test_add_error(self):
        """Test adding errors."""
        result = ValidationResult(is_valid=True)
        
        result.add_error("Test error")
        
        self.assertFalse(result.is_valid)  # Should become invalid
        self.assertIn("Test error", result.errors)
    
    def test_add_warning(self):
        """Test adding warnings."""
        result = ValidationResult(is_valid=True)
        
        result.add_warning("Test warning")
        
        self.assertTrue(result.is_valid)  # Should remain valid
        self.assertIn("Test warning", result.warnings)


class TestAlignmentResult(unittest.TestCase):
    """Test cases for AlignmentResult class."""
    
    def test_init_valid(self):
        """Test initialization with valid alignment."""
        result = AlignmentResult(is_valid=True)
        
        self.assertTrue(result.is_valid)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.warnings, [])
        self.assertEqual(result.missing_outputs, [])
        self.assertEqual(result.missing_inputs, [])
        self.assertEqual(result.extra_outputs, [])
        self.assertEqual(result.extra_inputs, [])
    
    def test_init_invalid_with_details(self):
        """Test initialization with invalid alignment and details."""
        result = AlignmentResult(
            is_valid=False,
            errors=["Alignment error"],
            warnings=["Alignment warning"],
            missing_outputs=["missing_output"],
            missing_inputs=["missing_input"],
            extra_outputs=["extra_output"],
            extra_inputs=["extra_input"]
        )
        
        self.assertFalse(result.is_valid)
        self.assertEqual(result.errors, ["Alignment error"])
        self.assertEqual(result.warnings, ["Alignment warning"])
        self.assertEqual(result.missing_outputs, ["missing_output"])
        self.assertEqual(result.missing_inputs, ["missing_input"])
        self.assertEqual(result.extra_outputs, ["extra_output"])
        self.assertEqual(result.extra_inputs, ["extra_input"])


class TestStepSpecification(unittest.TestCase):
    """Test cases for StepSpecification class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.output_spec = OutputSpec(
            logical_name="test_output",
            description="Test output",
            property_path="Steps.TestStep.Properties.OutputDataConfig.S3OutputPath"
        )
        
        self.dependency_spec = DependencySpec(
            logical_name="test_dependency",
            description="Test dependency"
        )
        
        self.spec_data = {
            "step_type": "TestStep",
            "description": "Test step specification",
            "dependencies": {"dep1": self.dependency_spec},
            "outputs": {"out1": self.output_spec}
        }
    
    def test_init_with_required_fields(self):
        """Test initialization with required fields."""
        spec = StepSpecification(**self.spec_data)
        
        self.assertEqual(spec.step_type, "TestStep")
        self.assertEqual(spec.description, "Test step specification")
        self.assertEqual(spec.dependencies, {"dep1": self.dependency_spec})
        self.assertEqual(spec.outputs, {"out1": self.output_spec})
        self.assertIsNone(spec.script_contract)
    
    def test_init_with_script_contract(self):
        """Test initialization with script contract."""
        mock_contract = Mock()
        spec_data = self.spec_data.copy()
        spec_data["script_contract"] = mock_contract
        
        spec = StepSpecification(**spec_data)
        
        self.assertEqual(spec.script_contract, mock_contract)
    
    def test_get_output_by_name_or_alias(self):
        """Test getting output by name or alias."""
        output_with_alias = OutputSpec(
            logical_name="aliased_output",
            description="Output with alias",
            property_path="Steps.TestStep.Properties.AliasedOutput",
            aliases=["alias1", "alias2"]
        )
        
        spec_data = self.spec_data.copy()
        spec_data["outputs"] = {
            "out1": self.output_spec,
            "out2": output_with_alias
        }
        
        spec = StepSpecification(**spec_data)
        
        # Test getting by logical name
        result = spec.get_output_by_name_or_alias("test_output")
        self.assertEqual(result, self.output_spec)
        
        # Test getting by alias
        result = spec.get_output_by_name_or_alias("alias1")
        self.assertEqual(result, output_with_alias)
        
        # Test non-existent output
        result = spec.get_output_by_name_or_alias("nonexistent")
        self.assertIsNone(result)
    
    def test_get_dependency_by_name_or_alias(self):
        """Test getting dependency by name or alias."""
        dep_with_alias = DependencySpec(
            logical_name="aliased_dependency",
            description="Dependency with alias",
            aliases=["dep_alias1", "dep_alias2"]
        )
        
        spec_data = self.spec_data.copy()
        spec_data["dependencies"] = {
            "dep1": self.dependency_spec,
            "dep2": dep_with_alias
        }
        
        spec = StepSpecification(**spec_data)
        
        # Test getting by logical name
        result = spec.get_dependency_by_name_or_alias("test_dependency")
        self.assertEqual(result, self.dependency_spec)
        
        # Test getting by alias
        result = spec.get_dependency_by_name_or_alias("dep_alias1")
        self.assertEqual(result, dep_with_alias)
        
        # Test non-existent dependency
        result = spec.get_dependency_by_name_or_alias("nonexistent")
        self.assertIsNone(result)
    
    def test_validate_basic(self):
        """Test basic validation."""
        spec = StepSpecification(**self.spec_data)
        
        result = spec.validate()
        
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid)
    
    def test_validate_empty_step_type(self):
        """Test validation with empty step type."""
        spec_data = self.spec_data.copy()
        spec_data["step_type"] = ""
        
        spec = StepSpecification(**spec_data)
        result = spec.validate()
        
        self.assertFalse(result.is_valid)
        self.assertIn("Step type cannot be empty", result.errors)
    
    def test_validate_empty_description(self):
        """Test validation with empty description."""
        spec_data = self.spec_data.copy()
        spec_data["description"] = ""
        
        spec = StepSpecification(**spec_data)
        result = spec.validate()
        
        self.assertFalse(result.is_valid)
        self.assertIn("Description cannot be empty", result.errors)
    
    def test_validate_duplicate_output_names(self):
        """Test validation with duplicate output names."""
        output1 = OutputSpec(
            logical_name="duplicate_name",
            description="Output 1",
            property_path="Path1"
        )
        output2 = OutputSpec(
            logical_name="duplicate_name",
            description="Output 2",
            property_path="Path2"
        )
        
        spec_data = self.spec_data.copy()
        spec_data["outputs"] = {"out1": output1, "out2": output2}
        
        spec = StepSpecification(**spec_data)
        result = spec.validate()
        
        self.assertFalse(result.is_valid)
        self.assertTrue(any("Duplicate output logical name" in error for error in result.errors))
    
    def test_validate_duplicate_dependency_names(self):
        """Test validation with duplicate dependency names."""
        dep1 = DependencySpec(
            logical_name="duplicate_name",
            description="Dependency 1"
        )
        dep2 = DependencySpec(
            logical_name="duplicate_name",
            description="Dependency 2"
        )
        
        spec_data = self.spec_data.copy()
        spec_data["dependencies"] = {"dep1": dep1, "dep2": dep2}
        
        spec = StepSpecification(**spec_data)
        result = spec.validate()
        
        self.assertFalse(result.is_valid)
        self.assertTrue(any("Duplicate dependency logical name" in error for error in result.errors))
    
    def test_validate_contract_alignment_no_contract(self):
        """Test contract alignment validation without contract."""
        spec = StepSpecification(**self.spec_data)
        
        result = spec.validate_contract_alignment()
        
        self.assertIsInstance(result, AlignmentResult)
        self.assertFalse(result.is_valid)
        self.assertIn("No script contract available", result.errors)
    
    def test_validate_contract_alignment_with_contract(self):
        """Test contract alignment validation with contract."""
        # Mock contract
        mock_contract = Mock()
        mock_contract.required_inputs = ["test_dependency"]
        mock_contract.optional_inputs = {}
        mock_contract.required_outputs = ["test_output"]
        mock_contract.optional_outputs = {}
        
        spec_data = self.spec_data.copy()
        spec_data["script_contract"] = mock_contract
        
        spec = StepSpecification(**spec_data)
        result = spec.validate_contract_alignment()
        
        self.assertIsInstance(result, AlignmentResult)
        # Should be valid since we have matching inputs/outputs
        self.assertTrue(result.is_valid)
    
    def test_validate_contract_alignment_missing_inputs(self):
        """Test contract alignment with missing inputs."""
        # Mock contract requiring more inputs than spec provides
        mock_contract = Mock()
        mock_contract.required_inputs = ["test_dependency", "missing_input"]
        mock_contract.optional_inputs = {}
        mock_contract.required_outputs = ["test_output"]
        mock_contract.optional_outputs = {}
        
        spec_data = self.spec_data.copy()
        spec_data["script_contract"] = mock_contract
        
        spec = StepSpecification(**spec_data)
        result = spec.validate_contract_alignment()
        
        self.assertFalse(result.is_valid)
        self.assertIn("missing_input", result.missing_inputs)
    
    def test_validate_contract_alignment_missing_outputs(self):
        """Test contract alignment with missing outputs."""
        # Mock contract requiring more outputs than spec provides
        mock_contract = Mock()
        mock_contract.required_inputs = ["test_dependency"]
        mock_contract.optional_inputs = {}
        mock_contract.required_outputs = ["test_output", "missing_output"]
        mock_contract.optional_outputs = {}
        
        spec_data = self.spec_data.copy()
        spec_data["script_contract"] = mock_contract
        
        spec = StepSpecification(**spec_data)
        result = spec.validate_contract_alignment()
        
        self.assertFalse(result.is_valid)
        self.assertIn("missing_output", result.missing_outputs)
    
    def test_get_required_dependencies(self):
        """Test getting required dependencies."""
        required_dep = DependencySpec(
            logical_name="required_dep",
            description="Required dependency",
            required=True
        )
        optional_dep = DependencySpec(
            logical_name="optional_dep",
            description="Optional dependency",
            required=False
        )
        
        spec_data = self.spec_data.copy()
        spec_data["dependencies"] = {
            "req": required_dep,
            "opt": optional_dep
        }
        
        spec = StepSpecification(**spec_data)
        required_deps = spec.get_required_dependencies()
        
        self.assertEqual(len(required_deps), 1)
        self.assertEqual(required_deps[0].logical_name, "required_dep")
    
    def test_get_optional_dependencies(self):
        """Test getting optional dependencies."""
        required_dep = DependencySpec(
            logical_name="required_dep",
            description="Required dependency",
            required=True
        )
        optional_dep = DependencySpec(
            logical_name="optional_dep",
            description="Optional dependency",
            required=False
        )
        
        spec_data = self.spec_data.copy()
        spec_data["dependencies"] = {
            "req": required_dep,
            "opt": optional_dep
        }
        
        spec = StepSpecification(**spec_data)
        optional_deps = spec.get_optional_dependencies()
        
        self.assertEqual(len(optional_deps), 1)
        self.assertEqual(optional_deps[0].logical_name, "optional_dep")
    
    def test_get_all_output_names(self):
        """Test getting all output names including aliases."""
        output_with_aliases = OutputSpec(
            logical_name="main_output",
            description="Output with aliases",
            property_path="Steps.TestStep.Properties.MainOutput",
            aliases=["alias1", "alias2"]
        )
        
        spec_data = self.spec_data.copy()
        spec_data["outputs"] = {
            "out1": self.output_spec,
            "out2": output_with_aliases
        }
        
        spec = StepSpecification(**spec_data)
        all_names = spec.get_all_output_names()
        
        expected_names = {"test_output", "main_output", "alias1", "alias2"}
        self.assertEqual(set(all_names), expected_names)
    
    def test_get_all_dependency_names(self):
        """Test getting all dependency names including aliases."""
        dep_with_aliases = DependencySpec(
            logical_name="main_dependency",
            description="Dependency with aliases",
            aliases=["dep_alias1", "dep_alias2"]
        )
        
        spec_data = self.spec_data.copy()
        spec_data["dependencies"] = {
            "dep1": self.dependency_spec,
            "dep2": dep_with_aliases
        }
        
        spec = StepSpecification(**spec_data)
        all_names = spec.get_all_dependency_names()
        
        expected_names = {"test_dependency", "main_dependency", "dep_alias1", "dep_alias2"}
        self.assertEqual(set(all_names), expected_names)


if __name__ == '__main__':
    # Set up logging for tests
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
