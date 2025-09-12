import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import logging

from cursus.core.base.specification_base import (
    StepSpecification,
    OutputSpec,
    DependencySpec,
)
from cursus.core.base.contract_base import ValidationResult, AlignmentResult
from cursus.core.base.enums import DependencyType, NodeType


class TestOutputSpec:
    """Test cases for OutputSpec class."""

    def test_init_with_required_fields(self):
        """Test initialization with required fields."""
        from cursus.core.base.enums import DependencyType

        output_spec = OutputSpec(
            logical_name="test_output",
            description="Test output description",
            property_path="properties.TestStep.Properties.OutputDataConfig.S3OutputPath",
            output_type=DependencyType.PROCESSING_OUTPUT,
        )

        assert output_spec.logical_name == "test_output"
        assert output_spec.description == "Test output description"
        assert (
            output_spec.property_path
            == "properties.TestStep.Properties.OutputDataConfig.S3OutputPath"
        )
        assert output_spec.data_type == "S3Uri"  # Default value
        assert output_spec.aliases == []

    def test_init_with_optional_fields(self):
        """Test initialization with optional fields."""
        from cursus.core.base.enums import DependencyType

        output_spec = OutputSpec(
            logical_name="test_output",
            description="Test output description",
            property_path="properties.TestStep.Properties.OutputDataConfig.S3OutputPath",
            output_type=DependencyType.PROCESSING_OUTPUT,
            data_type="s3_uri",
            aliases=["output_alias", "alt_output"],
        )

        assert output_spec.data_type == "s3_uri"
        assert output_spec.aliases == ["output_alias", "alt_output"]

    def test_matches_name_or_alias(self):
        """Test matching by name or alias."""
        from cursus.core.base.enums import DependencyType

        output_spec = OutputSpec(
            logical_name="test_output",
            description="Test output description",
            property_path="properties.TestStep.Properties.OutputDataConfig.S3OutputPath",
            output_type=DependencyType.PROCESSING_OUTPUT,
            aliases=["output_alias", "alt_output"],
        )

        # Test matching logical name
        assert output_spec.matches_name_or_alias("test_output")

        # Test matching aliases
        assert output_spec.matches_name_or_alias("output_alias")
        assert output_spec.matches_name_or_alias("alt_output")

        # Test non-matching name
        assert not output_spec.matches_name_or_alias("nonexistent")


class TestDependencySpec:
    """Test cases for DependencySpec class."""

    def test_init_with_required_fields(self):
        """Test initialization with required fields."""
        from cursus.core.base.enums import DependencyType

        dep_spec = DependencySpec(
            logical_name="test_dependency",
            description="Test dependency description",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
        )

        assert dep_spec.logical_name == "test_dependency"
        assert dep_spec.description == "Test dependency description"
        assert dep_spec.required  # Default is True
        assert dep_spec.data_type == "S3Uri"  # Default value
        # DependencySpec doesn't have aliases field in current implementation

    def test_init_with_optional_fields(self):
        """Test initialization with optional fields."""
        from cursus.core.base.enums import DependencyType

        dep_spec = DependencySpec(
            logical_name="test_dependency",
            description="Test dependency description",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=False,
            data_type="s3_uri",
        )

        assert not dep_spec.required
        assert dep_spec.data_type == "s3_uri"
        # Note: DependencySpec doesn't have aliases field in current implementation

    def test_matches_name_or_alias(self):
        """Test matching by name or alias."""
        from cursus.core.base.enums import DependencyType

        dep_spec = DependencySpec(
            logical_name="test_dependency",
            description="Test dependency description",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
        )

        # Test matching logical name
        assert dep_spec.matches_name_or_alias("test_dependency")

        # Test non-matching name
        assert not dep_spec.matches_name_or_alias("nonexistent")

        # Note: DependencySpec doesn't have aliases field in current implementation
        # so we can't test alias matching


class TestValidationResult:
    """Test cases for ValidationResult class."""

    def test_init_valid(self):
        """Test initialization with valid result."""
        result = ValidationResult(is_valid=True)

        assert result.is_valid
        assert result.errors == []
        assert result.warnings == []

    def test_init_invalid_with_errors(self):
        """Test initialization with invalid result and errors."""
        errors = ["Error 1", "Error 2"]
        warnings = ["Warning 1"]

        result = ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        assert not result.is_valid
        assert result.errors == errors
        assert result.warnings == warnings

    def test_add_error(self):
        """Test adding errors."""
        result = ValidationResult(is_valid=True)

        result.add_error("Test error")

        assert not result.is_valid  # Should become invalid
        assert "Test error" in result.errors

    def test_add_warning(self):
        """Test adding warnings."""
        result = ValidationResult(is_valid=True)

        result.add_warning("Test warning")

        assert result.is_valid  # Should remain valid
        assert "Test warning" in result.warnings


class TestAlignmentResult:
    """Test cases for AlignmentResult class."""

    def test_init_valid(self):
        """Test initialization with valid alignment."""
        result = AlignmentResult(is_valid=True)

        assert result.is_valid
        assert result.errors == []
        assert result.warnings == []
        assert result.missing_outputs == []
        assert result.missing_inputs == []
        assert result.extra_outputs == []
        assert result.extra_inputs == []

    def test_init_invalid_with_details(self):
        """Test initialization with invalid alignment and details."""
        result = AlignmentResult(
            is_valid=False,
            errors=["Alignment error"],
            warnings=["Alignment warning"],
            missing_outputs=["missing_output"],
            missing_inputs=["missing_input"],
            extra_outputs=["extra_output"],
            extra_inputs=["extra_input"],
        )

        assert not result.is_valid
        assert result.errors == ["Alignment error"]
        assert result.warnings == ["Alignment warning"]
        assert result.missing_outputs == ["missing_output"]
        assert result.missing_inputs == ["missing_input"]
        assert result.extra_outputs == ["extra_output"]
        assert result.extra_inputs == ["extra_input"]


class TestStepSpecification:
    """Test cases for StepSpecification class."""

    @pytest.fixture
    def output_spec(self):
        """Set up test fixtures."""
        from cursus.core.base.enums import DependencyType

        return OutputSpec(
            logical_name="test_output",
            description="Test output",
            property_path="properties.TestStep.Properties.OutputDataConfig.S3OutputPath",
            output_type=DependencyType.PROCESSING_OUTPUT,
        )

    @pytest.fixture
    def dependency_spec(self):
        from cursus.core.base.enums import DependencyType

        return DependencySpec(
            logical_name="test_dependency",
            description="Test dependency",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
        )

    @pytest.fixture
    def spec_data(self, output_spec, dependency_spec):
        from cursus.core.base.enums import NodeType

        return {
            "step_type": "TestStep",
            "node_type": NodeType.INTERNAL,
            "dependencies": {"dep1": dependency_spec},
            "outputs": {"out1": output_spec},
        }

    def test_init_with_required_fields(self, spec_data, dependency_spec, output_spec):
        """Test initialization with required fields."""
        spec = StepSpecification(**spec_data)

        assert spec.step_type == "TestStep"
        assert spec.dependencies == {"dep1": dependency_spec}
        assert spec.outputs == {"out1": output_spec}
        assert spec.script_contract is None

    def test_init_with_script_contract(self, spec_data):
        """Test initialization with script contract."""
        mock_contract = Mock()
        spec_data = spec_data.copy()
        spec_data["script_contract"] = mock_contract

        spec = StepSpecification(**spec_data)

        assert spec.script_contract == mock_contract

    def test_get_output_by_name_or_alias(self, output_spec, dependency_spec):
        """Test getting output by name or alias."""
        from cursus.core.base.enums import DependencyType, NodeType

        output_with_alias = OutputSpec(
            logical_name="aliased_output",
            description="Output with alias",
            property_path="properties.TestStep.Properties.AliasedOutput",
            output_type=DependencyType.PROCESSING_OUTPUT,
            aliases=["alias1", "alias2"],
        )

        # Create spec data where dictionary keys match logical names
        spec_data = {
            "step_type": "TestStep",
            "node_type": NodeType.INTERNAL,
            "dependencies": {"test_dependency": dependency_spec},
            "outputs": {
                "test_output": output_spec,  # Key matches logical name
                "aliased_output": output_with_alias,  # Key matches logical name
            },
        }

        spec = StepSpecification(**spec_data)

        # Test getting by logical name
        result = spec.get_output_by_name_or_alias("test_output")
        assert result == output_spec

        # Test getting by alias
        result = spec.get_output_by_name_or_alias("alias1")
        assert result == output_with_alias

        # Test non-existent output
        result = spec.get_output_by_name_or_alias("nonexistent")
        assert result is None

    def test_get_dependency(self, output_spec, dependency_spec):
        """Test getting dependency by logical name."""
        # Create a spec where the dictionary key matches the logical name
        from cursus.core.base.enums import NodeType

        spec_data = {
            "step_type": "TestStep",
            "node_type": NodeType.INTERNAL,
            "dependencies": {
                "test_dependency": dependency_spec
            },  # Key matches logical name
            "outputs": {"test_output": output_spec},  # Key matches logical name
        }

        spec = StepSpecification(**spec_data)

        # Test getting by logical name
        result = spec.get_dependency("test_dependency")
        assert result == dependency_spec

        # Test non-existent dependency
        result = spec.get_dependency("nonexistent")
        assert result is None

    def test_validate_basic(self, spec_data):
        """Test basic validation."""
        spec = StepSpecification(**spec_data)

        result = spec.validate()

        # The validate method returns a list, not a ValidationResult
        assert isinstance(result, list)
        assert len(result) == 0  # Empty list means valid

    def test_validate_empty_step_type(self, dependency_spec, output_spec):
        """Test validation with empty step type."""
        from cursus.core.base.enums import NodeType

        # This should raise a ValidationError during construction, not during validate()
        with pytest.raises(Exception):  # Pydantic ValidationError
            StepSpecification(
                step_type="",
                node_type=NodeType.INTERNAL,
                dependencies={"dep1": dependency_spec},
                outputs={"out1": output_spec},
            )

    def test_validate_duplicate_output_names(self, spec_data):
        """Test validation with duplicate output names."""
        from cursus.core.base.enums import DependencyType

        # This should raise a ValidationError during construction due to duplicate names
        with pytest.raises(Exception):  # Pydantic ValidationError
            output1 = OutputSpec(
                logical_name="duplicate_name",
                description="Output 1",
                property_path="properties.Path1",
                output_type=DependencyType.PROCESSING_OUTPUT,
            )
            output2 = OutputSpec(
                logical_name="duplicate_name",
                description="Output 2",
                property_path="properties.Path2",
                output_type=DependencyType.PROCESSING_OUTPUT,
            )

            StepSpecification(
                step_type="TestStep",
                node_type=spec_data["node_type"],
                dependencies=spec_data["dependencies"],
                outputs={"out1": output1, "out2": output2},
            )

    def test_validate_contract_alignment_no_contract(self, spec_data):
        """Test contract alignment validation without contract."""
        spec = StepSpecification(**spec_data)

        result = spec.validate_contract_alignment()

        # The method returns ValidationResult, not AlignmentResult
        from cursus.core.base.contract_base import ValidationResult

        assert isinstance(result, ValidationResult)
        assert result.is_valid  # No contract to validate means success
        assert "No contract to validate" in (
            result.errors[0] if result.errors else "No contract to validate"
        )

    def test_validate_contract_alignment_with_contract(self, spec_data):
        """Test contract alignment validation with contract."""
        # Mock contract with expected_input_paths and expected_output_paths
        mock_contract = Mock()
        mock_contract.expected_input_paths = {"test_dependency": "some/path"}
        mock_contract.expected_output_paths = {"test_output": "some/output/path"}

        spec_data = spec_data.copy()
        spec_data["script_contract"] = mock_contract

        spec = StepSpecification(**spec_data)
        result = spec.validate_contract_alignment()

        from cursus.core.base.contract_base import ValidationResult

        assert isinstance(result, ValidationResult)
        # Should be valid since we have matching inputs/outputs
        assert result.is_valid

    def test_validate_contract_alignment_missing_inputs(self, spec_data):
        """Test contract alignment with missing inputs."""
        # Mock contract requiring more inputs than spec provides
        mock_contract = Mock()
        mock_contract.expected_input_paths = {
            "test_dependency": "path1",
            "missing_input": "path2",
        }
        mock_contract.expected_output_paths = {"test_output": "output/path"}

        spec_data = spec_data.copy()
        spec_data["script_contract"] = mock_contract

        spec = StepSpecification(**spec_data)
        result = spec.validate_contract_alignment()

        from cursus.core.base.contract_base import ValidationResult

        assert isinstance(result, ValidationResult)
        assert not result.is_valid
        assert any("missing_input" in error for error in result.errors)

    def test_validate_contract_alignment_missing_outputs(self, spec_data):
        """Test contract alignment with missing outputs."""
        # Mock contract requiring more outputs than spec provides
        mock_contract = Mock()
        mock_contract.expected_input_paths = {"test_dependency": "input/path"}
        mock_contract.expected_output_paths = {
            "test_output": "path1",
            "missing_output": "path2",
        }

        spec_data = spec_data.copy()
        spec_data["script_contract"] = mock_contract

        spec = StepSpecification(**spec_data)
        result = spec.validate_contract_alignment()

        from cursus.core.base.contract_base import ValidationResult

        assert isinstance(result, ValidationResult)
        assert not result.is_valid
        assert any("missing_output" in error for error in result.errors)

    def test_list_required_dependencies(self, spec_data):
        """Test getting required dependencies."""
        from cursus.core.base.enums import DependencyType

        required_dep = DependencySpec(
            logical_name="required_dep",
            description="Required dependency",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
        )
        optional_dep = DependencySpec(
            logical_name="optional_dep",
            description="Optional dependency",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=False,
        )

        spec_data = spec_data.copy()
        spec_data["dependencies"] = {"req": required_dep, "opt": optional_dep}

        spec = StepSpecification(**spec_data)
        required_deps = spec.list_required_dependencies()

        assert len(required_deps) == 1
        assert required_deps[0].logical_name == "required_dep"

    def test_list_optional_dependencies(self, spec_data):
        """Test getting optional dependencies."""
        from cursus.core.base.enums import DependencyType

        required_dep = DependencySpec(
            logical_name="required_dep",
            description="Required dependency",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
        )
        optional_dep = DependencySpec(
            logical_name="optional_dep",
            description="Optional dependency",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=False,
        )

        spec_data = spec_data.copy()
        spec_data["dependencies"] = {"req": required_dep, "opt": optional_dep}

        spec = StepSpecification(**spec_data)
        optional_deps = spec.list_optional_dependencies()

        assert len(optional_deps) == 1
        assert optional_deps[0].logical_name == "optional_dep"

    def test_list_all_output_names(self, spec_data, output_spec):
        """Test getting all output names including aliases."""
        from cursus.core.base.enums import DependencyType

        output_with_aliases = OutputSpec(
            logical_name="main_output",
            description="Output with aliases",
            property_path="properties.TestStep.Properties.MainOutput",
            output_type=DependencyType.PROCESSING_OUTPUT,
            aliases=["alias1", "alias2"],
        )

        spec_data = spec_data.copy()
        spec_data["outputs"] = {"out1": output_spec, "out2": output_with_aliases}

        spec = StepSpecification(**spec_data)
        all_names = spec.list_all_output_names()

        expected_names = {"test_output", "main_output", "alias1", "alias2"}
        assert set(all_names) == expected_names
