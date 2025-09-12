"""
Unit tests for ConfigurationAnalyzer

Tests the enhanced configuration analysis capabilities including:
- Property detection from base classes
- Pydantic field analysis (v1 and v2)
- Inheritance handling through MRO
- Required/optional field classification
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys


from cursus.validation.alignment.analyzers.config_analyzer import ConfigurationAnalyzer


class TestConfigurationAnalyzer:
    """Test cases for ConfigurationAnalyzer"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.analyzer = ConfigurationAnalyzer(self.temp_dir)
        yield
        # Clean up test fixtures
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_analyze_config_class_basic_annotations(self):
        """Test analysis of basic type annotations"""
        from typing import Optional, Union

        # Create a mock config class with annotations
        class MockConfig:
            __annotations__ = {
                "required_field": str,
                "optional_field": Optional[str],
                "union_field": Union[str, None],
            }

        result = self.analyzer.analyze_config_class(MockConfig, "MockConfig")

        assert result["class_name"] == "MockConfig"
        assert "required_field" in result["fields"]
        assert "optional_field" in result["fields"]
        assert "union_field" in result["fields"]

        # Check required/optional classification
        assert "required_field" in result["required_fields"]
        assert "optional_field" in result["optional_fields"]
        assert "union_field" in result["optional_fields"]

    def test_analyze_config_class_with_properties(self):
        """Test detection of properties from base classes"""

        class BaseConfig:
            @property
            def pipeline_s3_loc(self) -> str:
                return "s3://bucket/path"

            @property
            def computed_field(self) -> str:
                return "computed"

        class MockConfig(BaseConfig):
            __annotations__ = {"regular_field": str}

        result = self.analyzer.analyze_config_class(MockConfig, "MockConfig")

        # Should detect both regular fields and properties
        assert "regular_field" in result["fields"]
        assert "pipeline_s3_loc" in result["fields"]
        assert "computed_field" in result["fields"]

        # Properties should be optional
        assert "pipeline_s3_loc" in result["optional_fields"]
        assert result["fields"]["pipeline_s3_loc"]["type"] == "property"

    def test_analyze_config_class_pydantic_v2(self):
        """Test analysis of Pydantic v2 model fields"""

        # Mock Pydantic v2 style class
        class MockFieldInfo:
            def __init__(self, required=True):
                self._required = required

            def is_required(self):
                return self._required

        class MockConfig:
            __annotations__ = {"field1": str, "field2": str}

            model_fields = {
                "field1": MockFieldInfo(required=True),
                "field2": MockFieldInfo(required=False),
            }

        result = self.analyzer.analyze_config_class(MockConfig, "MockConfig")

        # Check Pydantic field info is used for required/optional
        assert "field1" in result["required_fields"]
        assert "field2" in result["optional_fields"]
        assert result["fields"]["field1"]["required"] is True
        assert result["fields"]["field2"]["required"] is False

    def test_analyze_config_class_pydantic_v1(self):
        """Test analysis of Pydantic v1 model fields"""

        # Mock Pydantic v1 style class
        class MockFieldInfo:
            def __init__(self, required=True, default=None):
                self.required = required
                self.default = default
                self.type_ = str

        class MockConfig:
            __annotations__ = {"field1": str, "field2": str}

            __fields__ = {
                "field1": MockFieldInfo(required=True),
                "field2": MockFieldInfo(required=False, default="default_value"),
            }

        result = self.analyzer.analyze_config_class(MockConfig, "MockConfig")

        # Check Pydantic v1 field info is used
        assert "field1" in result["required_fields"]
        assert "field2" in result["optional_fields"]
        assert result["default_values"]["field2"] == "default_value"

    def test_analyze_config_class_inheritance(self):
        """Test analysis handles inheritance through MRO"""

        class BaseConfig:
            __annotations__ = {"base_field": str, "inherited_field": int}

        class MiddleConfig(BaseConfig):
            __annotations__ = {"middle_field": bool}

        class MockConfig(MiddleConfig):
            __annotations__ = {"child_field": float}

        result = self.analyzer.analyze_config_class(MockConfig, "MockConfig")

        # Should detect fields from all levels of inheritance
        assert "base_field" in result["fields"]
        assert "inherited_field" in result["fields"]
        assert "middle_field" in result["fields"]
        assert "child_field" in result["fields"]

    def test_is_optional_field_optional_type(self):
        """Test detection of Optional[Type] annotations"""

        class MockConfig:
            pass

        # Test Optional[str]
        from typing import Optional

        assert (
            self.analyzer._is_optional_field(Optional[str], "test_field", MockConfig)
            is True
        )

        # Test regular str
        assert self.analyzer._is_optional_field(str, "test_field", MockConfig) is False

    def test_is_optional_field_union_with_none(self):
        """Test detection of Union[Type, None] annotations"""

        class MockConfig:
            pass

        from typing import Union

        # Test Union[str, None]
        assert (
            self.analyzer._is_optional_field(Union[str, None], "test_field", MockConfig)
            is True
        )

        # Test Union[str, int] (no None)
        assert (
            self.analyzer._is_optional_field(Union[str, int], "test_field", MockConfig)
            is False
        )

    def test_is_optional_field_with_default(self):
        """Test detection of fields with default values"""

        class MockConfig:
            test_field = "default_value"

        # Field with default should be optional
        assert self.analyzer._is_optional_field(str, "test_field", MockConfig) is True

        # Field without default should be required
        assert self.analyzer._is_optional_field(str, "other_field", MockConfig) is False

    def test_load_config_from_python_success(self):
        """Test successful loading of config from Python file"""
        # Create a temporary config file
        config_content = """
from typing import Optional
from pydantic import BaseModel

class TestConfig(BaseModel):
    required_field: str
    optional_field: Optional[str] = None
    
    @property
    def computed_field(self) -> str:
        return "computed"
"""

        config_file = Path(self.temp_dir) / "config_test_step.py"
        config_file.write_text(config_content)

        # Mock the import process
        with patch("importlib.util.spec_from_file_location") as mock_spec, patch(
            "importlib.util.module_from_spec"
        ) as mock_module, patch("sys.modules", {}):

            # Create mock module with TestConfig class
            mock_config_class = type(
                "TestConfig",
                (),
                {
                    "__annotations__": {
                        "required_field": str,
                        "optional_field": "Optional[str]",
                    }
                },
            )

            # Add property to mock class
            mock_config_class.computed_field = property(lambda self: "computed")

            mock_mod = MagicMock()
            mock_mod.TestConfig = mock_config_class
            mock_module.return_value = mock_mod

            mock_spec_obj = MagicMock()
            mock_spec_obj.loader.exec_module = MagicMock()
            mock_spec.return_value = mock_spec_obj

            result = self.analyzer.load_config_from_python(config_file, "test")

            assert result["class_name"] == "TestConfig"
            assert "required_field" in result["fields"]
            assert "optional_field" in result["fields"]
            assert "computed_field" in result["fields"]

    def test_load_config_from_python_error_handling(self):
        """Test error handling when config loading fails"""
        non_existent_file = Path(self.temp_dir) / "non_existent.py"

        result = self.analyzer.load_config_from_python(non_existent_file, "test")

        # Should return error analysis
        assert result["class_name"] == "testConfig"
        assert "load_error" in result
        assert result["fields"] == {}

    def test_get_configuration_schema(self):
        """Test conversion to standardized schema format"""
        config_analysis = {
            "required_fields": {"field1", "field2"},
            "optional_fields": {"field3", "field4"},
            "fields": {
                "field1": {"type": "str", "required": True},
                "field2": {"type": "int", "required": True},
                "field3": {"type": "bool", "required": False},
                "field4": {"type": "float", "required": False},
            },
            "default_values": {"field3": True, "field4": 1.0},
        }

        schema = self.analyzer.get_configuration_schema(config_analysis)

        # Check structure
        assert "configuration" in schema
        config_schema = schema["configuration"]

        # Check that required and optional are lists or sets (order may vary)
        assert type(config_schema["required"]) in [list, set]
        assert type(config_schema["optional"]) in [list, set]

        # Check contents (convert to sets for comparison since order doesn't matter)
        assert set(config_schema["required"]) == {"field1", "field2"}
        assert set(config_schema["optional"]) == {"field3", "field4"}

        # Check fields and defaults
        assert config_schema["fields"] == config_analysis["fields"]
        assert config_schema["defaults"] == {"field3": True, "field4": 1.0}


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
