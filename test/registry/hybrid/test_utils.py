"""
Test suite for hybrid registry utilities.

Tests RegistryLoader, StepDefinitionConverter, and RegistryValidationUtils.
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.cursus.registry.hybrid.utils import (
    RegistryLoader,
    StepDefinitionConverter,
    RegistryValidationUtils,
    RegistryErrorFormatter
)
from src.cursus.registry.exceptions import RegistryLoadError


class TestRegistryLoader(unittest.TestCase):
    """Test RegistryLoader utility class."""
    
    def test_load_registry_module_success(self):
        """Test successful registry module loading."""
        # Create temporary registry file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
STEP_NAMES = {
    "TestStep": {
        "config_class": "TestStepConfig",
        "builder_step_name": "TestStepBuilder",
        "spec_type": "TestStep",
        "sagemaker_step_type": "Processing",
        "description": "Test step"
    }
}
""")
            temp_file = f.name
        
        try:
            # Load the module
            module = RegistryLoader.load_registry_module(temp_file, "test_registry")
            
            # Verify module was loaded correctly
            self.assertTrue(hasattr(module, 'STEP_NAMES'))
            self.assertIn("TestStep", module.STEP_NAMES)
            self.assertEqual(module.STEP_NAMES["TestStep"]["config_class"], "TestStepConfig")
            
        finally:
            # Clean up
            Path(temp_file).unlink()
    
    def test_load_registry_module_file_not_found(self):
        """Test loading non-existent registry file."""
        with self.assertRaises(RegistryLoadError) as exc_info:
            RegistryLoader.load_registry_module("nonexistent.py", "test")
        
        self.assertIn("Registry file not found", str(exc_info.exception))
    
    def test_load_registry_module_invalid_python(self):
        """Test loading invalid Python file."""
        # Create temporary file with invalid Python
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("invalid python syntax !!!")
            temp_file = f.name
        
        try:
            with self.assertRaises(RegistryLoadError) as exc_info:
                RegistryLoader.load_registry_module(temp_file, "test")
            
            self.assertIn("Failed to load registry module", str(exc_info.exception))
            
        finally:
            Path(temp_file).unlink()
    
    def test_validate_registry_structure_success(self):
        """Test successful registry structure validation."""
        mock_module = Mock()
        mock_module.STEP_NAMES = {}
        mock_module.WORKSPACE_METADATA = {}
        
        # Should not raise exception
        RegistryLoader.validate_registry_structure(
            mock_module, 
            ['STEP_NAMES', 'WORKSPACE_METADATA']
        )
    
    def test_validate_registry_structure_missing_attributes(self):
        """Test registry structure validation with missing attributes."""
        mock_module = Mock()
        mock_module.STEP_NAMES = {}
        # Missing WORKSPACE_METADATA
        
        with self.assertRaises(RegistryLoadError) as exc_info:
            RegistryLoader.validate_registry_structure(
                mock_module,
                ['STEP_NAMES', 'WORKSPACE_METADATA']
            )
        
        self.assertIn("missing required attributes", str(exc_info.exception))
        self.assertIn("WORKSPACE_METADATA", str(exc_info.exception))
    
    def test_safe_get_attribute_exists(self):
        """Test safe attribute access when attribute exists."""
        mock_module = Mock()
        mock_module.test_attr = "test_value"
        
        result = RegistryLoader.safe_get_attribute(mock_module, "test_attr", "default")
        self.assertEqual(result, "test_value")
    
    def test_safe_get_attribute_missing(self):
        """Test safe attribute access when attribute is missing."""
        mock_module = Mock()
        # No test_attr
        
        result = RegistryLoader.safe_get_attribute(mock_module, "test_attr", "default")
        self.assertEqual(result, "default")
    
    def test_safe_get_attribute_no_default(self):
        """Test safe attribute access with no default."""
        mock_module = Mock()
        
        result = RegistryLoader.safe_get_attribute(mock_module, "test_attr")
        self.assertIsNone(result)


class TestStepDefinitionConverter(unittest.TestCase):
    """Test StepDefinitionConverter utility class."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_step_info = {
            "config_class": "TestStepConfig",
            "builder_step_name": "TestStepBuilder",
            "spec_type": "TestStep",
            "sagemaker_step_type": "Processing",
            "description": "Test step description"
        }
        
        self.sample_step_names = {
            "TestStep1": self.sample_step_info,
            "TestStep2": {
                "config_class": "TestStep2Config",
                "builder_step_name": "TestStep2Builder",
                "spec_type": "TestStep2",
                "sagemaker_step_type": "Training",
                "description": "Test step 2"
            }
        }
    
    @patch('src.cursus.registry.hybrid.utils.HybridStepDefinition')
    def test_from_legacy_format_basic(self, mock_hybrid_step_def):
        """Test converting from legacy format to hybrid format."""
        mock_definition = Mock()
        mock_hybrid_step_def.return_value = mock_definition
        
        result = StepDefinitionConverter.from_legacy_format(
            "TestStep", 
            self.sample_step_info,
            registry_type="core",
            workspace_id=None
        )
        
        # Verify HybridStepDefinition was called with correct parameters
        mock_hybrid_step_def.assert_called_once()
        call_args = mock_hybrid_step_def.call_args[1]
        
        self.assertEqual(call_args['name'], "TestStep")
        self.assertEqual(call_args['config_class'], "TestStepConfig")
        self.assertEqual(call_args['builder_step_name'], "TestStepBuilder")
        self.assertEqual(call_args['spec_type'], "TestStep")
        self.assertEqual(call_args['sagemaker_step_type'], "Processing")
        self.assertEqual(call_args['description'], "Test step description")
        self.assertEqual(call_args['registry_type'], "core")
        self.assertIsNone(call_args['workspace_id'])
        self.assertEqual(call_args['priority'], 100)  # default
        
        self.assertEqual(result, mock_definition)
    
    @patch('src.cursus.registry.hybrid.utils.HybridStepDefinition')
    def test_from_legacy_format_with_metadata(self, mock_hybrid_step_def):
        """Test converting with additional metadata."""
        enhanced_step_info = {
            **self.sample_step_info,
            "framework": "pytorch",
            "environment_tags": ["development", "gpu"],
            "priority": 85
        }
        
        StepDefinitionConverter.from_legacy_format(
            "TestStep",
            enhanced_step_info,
            registry_type="workspace",
            workspace_id="developer_1"
        )
        
        call_args = mock_hybrid_step_def.call_args[1]
        self.assertEqual(call_args['framework'], "pytorch")
        self.assertEqual(call_args['environment_tags'], ["development", "gpu"])
        self.assertEqual(call_args['priority'], 85)
        self.assertEqual(call_args['registry_type'], "workspace")
        self.assertEqual(call_args['workspace_id'], "developer_1")
    
    def test_to_legacy_format(self):
        """Test converting from hybrid format to legacy format."""
        mock_definition = Mock()
        mock_definition.config_class = "TestStepConfig"
        mock_definition.builder_step_name = "TestStepBuilder"
        mock_definition.spec_type = "TestStep"
        mock_definition.sagemaker_step_type = "Processing"
        mock_definition.description = "Test step description"
        
        result = StepDefinitionConverter.to_legacy_format(mock_definition)
        
        expected = {
            'config_class': "TestStepConfig",
            'builder_step_name': "TestStepBuilder",
            'spec_type': "TestStep",
            'sagemaker_step_type': "Processing",
            'description': "Test step description"
        }
        
        self.assertEqual(result, expected)
    
    @patch('src.cursus.registry.hybrid.utils.StepDefinitionConverter.from_legacy_format')
    def test_batch_convert_from_legacy(self, mock_from_legacy):
        """Test batch conversion from legacy format."""
        mock_def1 = Mock()
        mock_def2 = Mock()
        mock_from_legacy.side_effect = [mock_def1, mock_def2]
        
        result = StepDefinitionConverter.batch_convert_from_legacy(
            self.sample_step_names,
            registry_type="core",
            workspace_id=None
        )
        
        self.assertEqual(len(result), 2)
        self.assertIn("TestStep1", result)
        self.assertIn("TestStep2", result)
        self.assertEqual(result["TestStep1"], mock_def1)
        self.assertEqual(result["TestStep2"], mock_def2)
        
        # Verify from_legacy_format was called correctly
        self.assertEqual(mock_from_legacy.call_count, 2)
    
    @patch('src.cursus.registry.hybrid.utils.StepDefinitionConverter.to_legacy_format')
    def test_batch_convert_to_legacy(self, mock_to_legacy):
        """Test batch conversion to legacy format."""
        mock_def1 = Mock()
        mock_def2 = Mock()
        mock_to_legacy.side_effect = [
            {"config_class": "Config1"},
            {"config_class": "Config2"}
        ]
        
        definitions = {
            "TestStep1": mock_def1,
            "TestStep2": mock_def2
        }
        
        result = StepDefinitionConverter.batch_convert_to_legacy(definitions)
        
        self.assertEqual(len(result), 2)
        self.assertIn("TestStep1", result)
        self.assertIn("TestStep2", result)
        self.assertEqual(result["TestStep1"]["config_class"], "Config1")
        self.assertEqual(result["TestStep2"]["config_class"], "Config2")


class TestRegistryValidationUtils(unittest.TestCase):
    """Test RegistryValidationUtils utility class."""
    
    def test_validate_registry_type_valid(self):
        """Test validation of valid registry types."""
        valid_types = ['core', 'workspace', 'override']
        
        for registry_type in valid_types:
            result = RegistryValidationUtils.validate_registry_type(registry_type)
            self.assertEqual(result, registry_type)
    
    def test_validate_registry_type_invalid(self):
        """Test validation of invalid registry type."""
        with self.assertRaises(ValueError) as exc_info:
            RegistryValidationUtils.validate_registry_type("invalid_type")
        
        self.assertIn("Invalid registry_type", str(exc_info.exception))
        self.assertIn("Must be one of", str(exc_info.exception))
    
    def test_validate_step_name_valid(self):
        """Test validation of valid step names."""
        valid_names = ["TestStep", "XGBoostTraining", "test_step", "Test-Step"]
        
        for step_name in valid_names:
            result = RegistryValidationUtils.validate_step_name(step_name)
            self.assertEqual(result, step_name.strip())
    
    def test_validate_step_name_empty(self):
        """Test validation of empty step name."""
        with self.assertRaises(ValueError) as exc_info:
            RegistryValidationUtils.validate_step_name("")
        
        self.assertIn("Step name cannot be empty", str(exc_info.exception))
    
    def test_validate_step_name_invalid_characters(self):
        """Test validation of step name with invalid characters."""
        with self.assertRaises(ValueError) as exc_info:
            RegistryValidationUtils.validate_step_name("Test@Step!")
        
        self.assertIn("contains invalid characters", str(exc_info.exception))
    
    def test_validate_step_definition_completeness_valid(self):
        """Test validation of complete step definition."""
        mock_definition = Mock()
        mock_definition.config_class = "TestConfig"
        mock_definition.builder_step_name = "TestBuilder"
        mock_definition.spec_type = "TestSpec"
        mock_definition.sagemaker_step_type = "Processing"
        
        issues = RegistryValidationUtils.validate_step_definition_completeness(mock_definition)
        self.assertEqual(len(issues), 0)
    
    def test_validate_step_definition_completeness_missing_fields(self):
        """Test validation of incomplete step definition."""
        mock_definition = Mock()
        mock_definition.config_class = "TestConfig"
        mock_definition.builder_step_name = ""  # Empty
        mock_definition.spec_type = "TestSpec"
        mock_definition.sagemaker_step_type = None  # Missing
        
        issues = RegistryValidationUtils.validate_step_definition_completeness(mock_definition)
        self.assertEqual(len(issues), 2)
        self.assertTrue(any("builder_step_name" in issue for issue in issues))
        self.assertTrue(any("sagemaker_step_type" in issue for issue in issues))
    
    def test_validate_workspace_registry_structure_valid(self):
        """Test validation of valid workspace registry structure."""
        valid_registry = {
            "LOCAL_STEPS": {"Step1": {}},
            "STEP_OVERRIDES": {"Step2": {}},
            "WORKSPACE_METADATA": {}
        }
        
        issues = RegistryValidationUtils.validate_workspace_registry_structure(valid_registry)
        self.assertEqual(len(issues), 0)
    
    def test_validate_workspace_registry_structure_missing_sections(self):
        """Test validation of registry with missing sections."""
        invalid_registry = {
            "WORKSPACE_METADATA": {}
            # Missing both LOCAL_STEPS and STEP_OVERRIDES
        }
        
        issues = RegistryValidationUtils.validate_workspace_registry_structure(invalid_registry)
        self.assertEqual(len(issues), 1)
        self.assertIn("must define either LOCAL_STEPS or STEP_OVERRIDES", issues[0])
    
    def test_validate_workspace_registry_structure_invalid_types(self):
        """Test validation of registry with invalid data types."""
        invalid_registry = {
            "LOCAL_STEPS": "not_a_dict",  # Should be dict
            "STEP_OVERRIDES": ["not", "a", "dict"],  # Should be dict
        }
        
        issues = RegistryValidationUtils.validate_workspace_registry_structure(invalid_registry)
        self.assertEqual(len(issues), 2)
        self.assertTrue(any("LOCAL_STEPS must be a dictionary" in issue for issue in issues))
        self.assertTrue(any("STEP_OVERRIDES must be a dictionary" in issue for issue in issues))
    
    def test_format_registry_error_basic(self):
        """Test basic error message formatting."""
        result = RegistryValidationUtils.format_registry_error(
            "Test Context",
            "Test error message"
        )
        
        self.assertIn("Registry Error in Test Context: Test error message", result)
    
    def test_format_registry_error_with_suggestions(self):
        """Test error message formatting with suggestions."""
        suggestions = ["Try this", "Or try that", "Maybe this works"]
        
        result = RegistryValidationUtils.format_registry_error(
            "Test Context",
            "Test error message",
            suggestions
        )
        
        self.assertIn("Registry Error in Test Context: Test error message", result)
        self.assertIn("Suggestions:", result)
        self.assertIn("1. Try this", result)
        self.assertIn("2. Or try that", result)
        self.assertIn("3. Maybe this works", result)
    
    def test_validate_conflict_resolution_metadata_valid(self):
        """Test validation of valid conflict resolution metadata."""
        mock_definition = Mock()
        mock_definition.priority = 85
        mock_definition.conflict_resolution_strategy = "workspace_priority"
        mock_definition.framework = "pytorch"
        
        issues = RegistryValidationUtils.validate_conflict_resolution_metadata(mock_definition)
        self.assertEqual(len(issues), 0)
    
    def test_validate_conflict_resolution_metadata_invalid_priority(self):
        """Test validation of invalid priority."""
        mock_definition = Mock()
        mock_definition.priority = -10  # Invalid
        mock_definition.conflict_resolution_strategy = "workspace_priority"
        mock_definition.framework = "pytorch"
        
        issues = RegistryValidationUtils.validate_conflict_resolution_metadata(mock_definition)
        self.assertEqual(len(issues), 1)
        self.assertIn("Priority -10 outside valid range", issues[0])
    
    def test_validate_conflict_resolution_metadata_invalid_strategy(self):
        """Test validation of invalid resolution strategy."""
        mock_definition = Mock()
        mock_definition.priority = 85
        mock_definition.conflict_resolution_strategy = "invalid_strategy"
        mock_definition.framework = "pytorch"
        
        issues = RegistryValidationUtils.validate_conflict_resolution_metadata(mock_definition)
        self.assertEqual(len(issues), 1)
        self.assertIn("Invalid conflict resolution strategy", issues[0])
    
    def test_validate_conflict_resolution_metadata_unknown_framework(self):
        """Test validation of unknown framework."""
        mock_definition = Mock()
        mock_definition.priority = 85
        mock_definition.conflict_resolution_strategy = "workspace_priority"
        mock_definition.framework = "unknown_framework"
        
        issues = RegistryValidationUtils.validate_conflict_resolution_metadata(mock_definition)
        self.assertEqual(len(issues), 1)
        self.assertIn("Unknown framework: unknown_framework", issues[0])


class TestRegistryErrorFormatter(unittest.TestCase):
    """Test RegistryErrorFormatter utility class."""
    
    def test_format_registry_error_basic(self):
        """Test basic error formatting."""
        result = RegistryErrorFormatter.format_registry_error(
            "Loading",
            "File not found"
        )
        
        self.assertIn("Registry Error in Loading: File not found", result)
    
    def test_format_registry_error_with_suggestions(self):
        """Test error formatting with suggestions."""
        suggestions = ["Check file path", "Verify permissions"]
        
        result = RegistryErrorFormatter.format_registry_error(
            "Loading",
            "File not found",
            suggestions
        )
        
        self.assertIn("Registry Error in Loading: File not found", result)
        self.assertIn("Suggestions:", result)
        self.assertIn("1. Check file path", result)
        self.assertIn("2. Verify permissions", result)
    
    def test_format_validation_error(self):
        """Test validation error formatting."""
        result = RegistryErrorFormatter.format_validation_error(
            "TestStep",
            ["Missing config_class", "Invalid priority"]
        )
        
        self.assertIn("Validation Error for TestStep", result)
        self.assertIn("Missing config_class", result)
        self.assertIn("Invalid priority", result)
    
    def test_format_conflict_error(self):
        """Test conflict error formatting."""
        conflicts = ["developer_1", "developer_2"]
        
        result = RegistryErrorFormatter.format_conflict_error(
            "TestStep",
            conflicts
        )
        
        self.assertIn("Step Name Conflict: TestStep", result)
        self.assertIn("developer_1", result)
        self.assertIn("developer_2", result)
    
    def test_format_resolution_error(self):
        """Test resolution error formatting."""
        result = RegistryErrorFormatter.format_resolution_error(
            "TestStep",
            "No matching framework",
            ["Try different framework", "Check workspace context"]
        )
        
        self.assertIn("Resolution Error for TestStep", result)
        self.assertIn("No matching framework", result)
        self.assertIn("Try different framework", result)
        self.assertIn("Check workspace context", result)


if __name__ == "__main__":
    unittest.main()
