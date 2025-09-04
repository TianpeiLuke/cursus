"""
Test suite for hybrid registry data models.

Tests StepDefinition, ResolutionContext, StepResolutionResult, and other data models.
"""

import unittest
from unittest.mock import Mock
from pydantic import ValidationError
from typing import Dict, Any

from src.cursus.registry.hybrid.models import (
    StepDefinition,
    NamespacedStepDefinition,
    ResolutionContext,
    StepResolutionResult,
    RegistryValidationResult,
    ConflictAnalysis,
    StepComponentResolution,
    DistributedRegistryValidationResult
)


class TestStepDefinition(unittest.TestCase):
    """Test StepDefinition data model."""
    
    def test_step_definition_creation_minimal(self):
        """Test creating step definition with minimal required fields."""
        definition = StepDefinition(
            name="TestStep",
            config_class="TestStepConfig",
            builder_step_name="TestStepBuilder",
            spec_type="TestStep",
            sagemaker_step_type="Processing",
            description="Test step description",
            registry_type="core"
        )
        
        self.assertEqual(definition.name, "TestStep")
        self.assertEqual(definition.config_class, "TestStepConfig")
        self.assertEqual(definition.builder_step_name, "TestStepBuilder")
        self.assertEqual(definition.spec_type, "TestStep")
        self.assertEqual(definition.sagemaker_step_type, "Processing")
        self.assertEqual(definition.description, "Test step description")
        self.assertEqual(definition.registry_type, "core")
        self.assertIsNone(definition.workspace_id)
        self.assertEqual(definition.priority, 100)  # default
        self.assertIsNone(definition.framework)
        self.assertEqual(definition.environment_tags, [])
        self.assertEqual(definition.compatibility_tags, [])
        self.assertEqual(definition.conflict_resolution_strategy, "workspace_priority")
    
    def test_step_definition_creation_full(self):
        """Test creating step definition with all fields."""
        definition = StepDefinition(
            name="CustomStep",
            config_class="CustomStepConfig",
            builder_step_name="CustomStepBuilder",
            spec_type="CustomStep",
            sagemaker_step_type="Training",
            description="Custom training step",
            registry_type="workspace",
            workspace_id="developer_1",
            priority=85,
            framework="pytorch",
            environment_tags=["development", "gpu"],
            compatibility_tags=["experimental"],
            conflict_resolution_strategy="framework_match"
        )
        
        self.assertEqual(definition.name, "CustomStep")
        self.assertEqual(definition.registry_type, "workspace")
        self.assertEqual(definition.workspace_id, "developer_1")
        self.assertEqual(definition.priority, 85)
        self.assertEqual(definition.framework, "pytorch")
        self.assertEqual(definition.environment_tags, ["development", "gpu"])
        self.assertEqual(definition.compatibility_tags, ["experimental"])
        self.assertEqual(definition.conflict_resolution_strategy, "framework_match")
    
    def test_step_definition_validation_empty_name(self):
        """Test validation fails for empty name."""
        with self.assertRaises(ValidationError) as exc_info:
            StepDefinition(
                name="",  # Empty name
                config_class="TestStepConfig",
                builder_step_name="TestStepBuilder",
                spec_type="TestStep",
                sagemaker_step_type="Processing",
                description="Test step",
                registry_type="core"
            )
        
        self.assertIn("String should have at least 1 character", str(exc_info.exception))
    
    def test_step_definition_validation_invalid_registry_type(self):
        """Test validation fails for invalid registry type."""
        with self.assertRaises(ValidationError) as exc_info:
            StepDefinition(
                name="TestStep",
                config_class="TestStepConfig",
                builder_step_name="TestStepBuilder",
                spec_type="TestStep",
                sagemaker_step_type="Processing",
                description="Test step",
                registry_type="invalid_type"  # Invalid
            )
        
        self.assertIn("registry_type must be one of", str(exc_info.exception))
    
    def test_step_definition_to_legacy_format(self):
        """Test conversion to legacy format."""
        definition = StepDefinition(
            name="TestStep",
            config_class="TestStepConfig",
            builder_step_name="TestStepBuilder",
            spec_type="TestStep",
            sagemaker_step_type="Processing",
            description="Test step description",
            registry_type="core"
        )
        
        legacy_format = definition.to_legacy_format()
        
        expected = {
            'config_class': "TestStepConfig",
            'builder_step_name': "TestStepBuilder",
            'spec_type': "TestStep",
            'sagemaker_step_type': "Processing",
            'description': "Test step description"
        }
        
        self.assertEqual(legacy_format, expected)
    
    def test_step_definition_string_stripping(self):
        """Test that string fields are automatically stripped."""
        definition = StepDefinition(
            name="  TestStep  ",
            config_class="  TestStepConfig  ",
            builder_step_name="  TestStepBuilder  ",
            spec_type="  TestStep  ",
            sagemaker_step_type="  Processing  ",
            description="  Test step description  ",
            registry_type="core"
        )
        
        self.assertEqual(definition.name, "TestStep")
        self.assertEqual(definition.config_class, "TestStepConfig")
        self.assertEqual(definition.builder_step_name, "TestStepBuilder")
        self.assertEqual(definition.spec_type, "TestStep")
        self.assertEqual(definition.sagemaker_step_type, "Processing")
        self.assertEqual(definition.description, "Test step description")


class TestNamespacedStepDefinition(unittest.TestCase):
    """Test NamespacedStepDefinition data model."""
    
    def test_namespaced_step_definition_creation(self):
        """Test creating namespaced step definition."""
        definition = NamespacedStepDefinition(
            name="TestStep",
            config_class="TestStepConfig",
            builder_step_name="TestStepBuilder",
            spec_type="TestStep",
            sagemaker_step_type="Processing",
            description="Test step description",
            registry_type="workspace",
            namespace="developer_1",
            fully_qualified_name="developer_1.TestStep"
        )
        
        self.assertEqual(definition.name, "TestStep")
        self.assertEqual(definition.namespace, "developer_1")
        self.assertEqual(definition.fully_qualified_name, "developer_1.TestStep")
        self.assertEqual(definition.registry_type, "workspace")
    
    def test_namespaced_step_definition_validation(self):
        """Test validation of namespaced step definition."""
        with self.assertRaises(ValidationError) as exc_info:
            NamespacedStepDefinition(
                name="TestStep",
                config_class="TestStepConfig",
                builder_step_name="TestStepBuilder",
                spec_type="TestStep",
                sagemaker_step_type="Processing",
                description="Test step description",
                registry_type="workspace",
                namespace="",  # Empty namespace
                fully_qualified_name="TestStep"
            )
        
        self.assertIn("String should have at least 1 character", str(exc_info.exception))


class TestResolutionContext(unittest.TestCase):
    """Test ResolutionContext data model."""
    
    def test_resolution_context_creation_minimal(self):
        """Test creating resolution context with minimal fields."""
        context = ResolutionContext()
        
        self.assertIsNone(context.workspace_id)
        self.assertIsNone(context.preferred_framework)
        self.assertEqual(context.environment_tags, [])
        self.assertEqual(context.resolution_mode, "automatic")
    
    def test_resolution_context_creation_full(self):
        """Test creating resolution context with all fields."""
        context = ResolutionContext(
            workspace_id="developer_1",
            preferred_framework="pytorch",
            environment_tags=["development", "gpu"],
            resolution_mode="interactive"
        )
        
        self.assertEqual(context.workspace_id, "developer_1")
        self.assertEqual(context.preferred_framework, "pytorch")
        self.assertEqual(context.environment_tags, ["development", "gpu"])
        self.assertEqual(context.resolution_mode, "interactive")
    
    def test_resolution_context_validation_invalid_mode(self):
        """Test validation fails for invalid resolution mode."""
        with self.assertRaises(ValidationError) as exc_info:
            ResolutionContext(
                resolution_mode="invalid_mode"
            )
        
        self.assertIn("resolution_mode must be one of", str(exc_info.exception))
    
    def test_resolution_context_validation_valid_modes(self):
        """Test validation passes for valid resolution modes."""
        valid_modes = ['automatic', 'interactive', 'strict']
        
        for mode in valid_modes:
            context = ResolutionContext(resolution_mode=mode)
            self.assertEqual(context.resolution_mode, mode)


class TestStepResolutionResult(unittest.TestCase):
    """Test StepResolutionResult data model."""
    
    def test_step_resolution_result_success(self):
        """Test creating successful resolution result."""
        mock_definition = Mock()
        
        result = StepResolutionResult(
            step_name="TestStep",
            resolved=True,
            selected_definition=mock_definition,
            resolution_strategy="workspace_priority",
            reason="Found in current workspace",
            conflicting_definitions=[]
        )
        
        self.assertEqual(result.step_name, "TestStep")
        self.assertTrue(result.resolved)
        self.assertEqual(result.selected_definition, mock_definition)
        self.assertEqual(result.resolution_strategy, "workspace_priority")
        self.assertEqual(result.reason, "Found in current workspace")
        self.assertEqual(result.conflicting_definitions, [])
    
    def test_step_resolution_result_failure(self):
        """Test creating failed resolution result."""
        result = StepResolutionResult(
            step_name="UnknownStep",
            resolved=False,
            reason="Step not found in any registry"
        )
        
        self.assertEqual(result.step_name, "UnknownStep")
        self.assertFalse(result.resolved)
        self.assertIsNone(result.selected_definition)
        self.assertIsNone(result.resolution_strategy)
        self.assertEqual(result.reason, "Step not found in any registry")
        self.assertEqual(result.conflicting_definitions, [])
    
    def test_step_resolution_result_with_conflicts(self):
        """Test creating resolution result with conflicts."""
        mock_def1 = Mock()
        mock_def2 = Mock()
        conflicting_definitions = [mock_def1, mock_def2]
        
        result = StepResolutionResult(
            step_name="ConflictedStep",
            resolved=True,
            selected_definition=mock_def1,
            resolution_strategy="priority_based",
            reason="Selected based on priority",
            conflicting_definitions=conflicting_definitions
        )
        
        self.assertEqual(result.step_name, "ConflictedStep")
        self.assertTrue(result.resolved)
        self.assertEqual(result.selected_definition, mock_def1)
        self.assertEqual(result.conflicting_definitions, conflicting_definitions)
        self.assertEqual(len(result.conflicting_definitions), 2)


class TestRegistryValidationResult(unittest.TestCase):
    """Test RegistryValidationResult data model."""
    
    def test_registry_validation_result_success(self):
        """Test creating successful validation result."""
        result = RegistryValidationResult(
            registry_type="core",
            is_valid=True,
            step_count=17,
            issues=[],
            warnings=[]
        )
        
        self.assertEqual(result.registry_type, "core")
        self.assertTrue(result.is_valid)
        self.assertEqual(result.step_count, 17)
        self.assertEqual(result.issues, [])
        self.assertEqual(result.warnings, [])
    
    def test_registry_validation_result_with_issues(self):
        """Test creating validation result with issues."""
        issues = ["Missing config_class", "Invalid priority"]
        warnings = ["Deprecated field used"]
        
        result = RegistryValidationResult(
            registry_type="workspace",
            is_valid=False,
            step_count=5,
            issues=issues,
            warnings=warnings
        )
        
        self.assertEqual(result.registry_type, "workspace")
        self.assertFalse(result.is_valid)
        self.assertEqual(result.step_count, 5)
        self.assertEqual(result.issues, issues)
        self.assertEqual(result.warnings, warnings)


class TestConflictAnalysis(unittest.TestCase):
    """Test ConflictAnalysis data model."""
    
    def test_conflict_analysis_creation(self):
        """Test creating conflict analysis."""
        mock_def1 = Mock()
        mock_def2 = Mock()
        
        analysis = ConflictAnalysis(
            step_name="ConflictedStep",
            conflicting_definitions=[mock_def1, mock_def2],
            conflict_type="workspace_collision",
            resolution_strategies=["workspace_priority", "framework_match"],
            recommended_strategy="workspace_priority",
            impact_assessment="Low impact - isolated to development"
        )
        
        self.assertEqual(analysis.step_name, "ConflictedStep")
        self.assertEqual(analysis.conflicting_definitions, [mock_def1, mock_def2])
        self.assertEqual(analysis.conflict_type, "workspace_collision")
        self.assertEqual(analysis.resolution_strategies, ["workspace_priority", "framework_match"])
        self.assertEqual(analysis.recommended_strategy, "workspace_priority")
        self.assertEqual(analysis.impact_assessment, "Low impact - isolated to development")


class TestStepComponentResolution(unittest.TestCase):
    """Test StepComponentResolution data model."""
    
    def test_step_component_resolution_creation(self):
        """Test creating step component resolution."""
        resolution = StepComponentResolution(
            step_name="TestStep",
            config_class="TestStepConfig",
            builder_class="TestStepBuilder",
            spec_class="TestStepSpec",
            script_path="/path/to/script.py",
            contract_class="TestStepContract",
            workspace_id="developer_1",
            resolution_source="workspace_registry"
        )
        
        self.assertEqual(resolution.step_name, "TestStep")
        self.assertEqual(resolution.config_class, "TestStepConfig")
        self.assertEqual(resolution.builder_class, "TestStepBuilder")
        self.assertEqual(resolution.spec_class, "TestStepSpec")
        self.assertEqual(resolution.script_path, "/path/to/script.py")
        self.assertEqual(resolution.contract_class, "TestStepContract")
        self.assertEqual(resolution.workspace_id, "developer_1")
        self.assertEqual(resolution.resolution_source, "workspace_registry")


class TestDistributedRegistryValidationResult(unittest.TestCase):
    """Test DistributedRegistryValidationResult data model."""
    
    def test_distributed_validation_result_creation(self):
        """Test creating distributed validation result."""
        core_result = RegistryValidationResult(
            registry_type="core",
            is_valid=True,
            step_count=17,
            issues=[],
            warnings=[]
        )
        
        workspace_results = {
            "developer_1": RegistryValidationResult(
                registry_type="workspace",
                is_valid=True,
                step_count=3,
                issues=[],
                warnings=[]
            )
        }
        
        conflicts = {
            "ConflictedStep": ConflictAnalysis(
                step_name="ConflictedStep",
                conflicting_definitions=[],
                conflict_type="workspace_collision",
                resolution_strategies=["workspace_priority"],
                recommended_strategy="workspace_priority",
                impact_assessment="Low impact"
            )
        }
        
        result = DistributedRegistryValidationResult(
            overall_status="HEALTHY",
            core_registry_result=core_result,
            workspace_results=workspace_results,
            conflicts=conflicts,
            total_step_count=20,
            recommendations=["Consider renaming conflicted steps"]
        )
        
        self.assertEqual(result.overall_status, "HEALTHY")
        self.assertEqual(result.core_registry_result, core_result)
        self.assertEqual(result.workspace_results, workspace_results)
        self.assertEqual(result.conflicts, conflicts)
        self.assertEqual(result.total_step_count, 20)
        self.assertEqual(result.recommendations, ["Consider renaming conflicted steps"])


class TestModelValidation(unittest.TestCase):
    """Test Pydantic validation features across all models."""
    
    def test_step_definition_extra_fields_forbidden(self):
        """Test that extra fields are forbidden in StepDefinition."""
        with self.assertRaises(ValidationError) as exc_info:
            StepDefinition(
                name="TestStep",
                config_class="TestStepConfig",
                builder_step_name="TestStepBuilder",
                spec_type="TestStep",
                sagemaker_step_type="Processing",
                description="Test step",
                registry_type="core",
                extra_field="not_allowed"  # Extra field
            )
        
        self.assertIn("Extra inputs are not permitted", str(exc_info.exception))
    
    def test_resolution_context_extra_fields_forbidden(self):
        """Test that extra fields are forbidden in ResolutionContext."""
        with self.assertRaises(ValidationError) as exc_info:
            ResolutionContext(
                workspace_id="developer_1",
                extra_field="not_allowed"  # Extra field
            )
        
        self.assertIn("Extra inputs are not permitted", str(exc_info.exception))
    
    def test_step_resolution_result_required_fields(self):
        """Test that required fields are enforced in StepResolutionResult."""
        with self.assertRaises(ValidationError) as exc_info:
            StepResolutionResult(
                # Missing step_name and resolved
                reason="Test reason"
            )
        
        self.assertIn("Field required", str(exc_info.exception))
    
    def test_registry_validation_result_field_validation(self):
        """Test field validation in RegistryValidationResult."""
        # Test negative step count
        with self.assertRaises(ValidationError) as exc_info:
            RegistryValidationResult(
                registry_type="core",
                is_valid=True,
                step_count=-1,  # Invalid negative count
                issues=[],
                warnings=[]
            )
        
        self.assertIn("Input should be greater than or equal to 0", str(exc_info.exception))


class TestModelSerialization(unittest.TestCase):
    """Test model serialization and deserialization."""
    
    def test_step_definition_serialization(self):
        """Test StepDefinition serialization to dict."""
        definition = StepDefinition(
            name="TestStep",
            config_class="TestStepConfig",
            builder_step_name="TestStepBuilder",
            spec_type="TestStep",
            sagemaker_step_type="Processing",
            description="Test step",
            registry_type="core",
            priority=85,
            framework="pytorch"
        )
        
        data = definition.model_dump()
        
        self.assertEqual(data['name'], "TestStep")
        self.assertEqual(data['config_class'], "TestStepConfig")
        self.assertEqual(data['priority'], 85)
        self.assertEqual(data['framework'], "pytorch")
        self.assertIsNone(data['workspace_id'])
    
    def test_step_definition_deserialization(self):
        """Test StepDefinition deserialization from dict."""
        data = {
            'name': "TestStep",
            'config_class': "TestStepConfig",
            'builder_step_name': "TestStepBuilder",
            'spec_type': "TestStep",
            'sagemaker_step_type': "Processing",
            'description': "Test step",
            'registry_type': "core",
            'priority': 85,
            'framework': "pytorch"
        }
        
        definition = StepDefinition(**data)
        
        self.assertEqual(definition.name, "TestStep")
        self.assertEqual(definition.config_class, "TestStepConfig")
        self.assertEqual(definition.priority, 85)
        self.assertEqual(definition.framework, "pytorch")
    
    def test_resolution_context_json_serialization(self):
        """Test ResolutionContext JSON serialization."""
        context = ResolutionContext(
            workspace_id="developer_1",
            preferred_framework="pytorch",
            environment_tags=["development", "gpu"],
            resolution_mode="interactive"
        )
        
        json_str = context.model_dump_json()
        self.assertIn('"workspace_id":"developer_1"', json_str)
        self.assertIn('"preferred_framework":"pytorch"', json_str)
        self.assertIn('"resolution_mode":"interactive"', json_str)
    
    def test_step_resolution_result_json_deserialization(self):
        """Test StepResolutionResult JSON deserialization."""
        json_data = {
            "step_name": "TestStep",
            "resolved": True,
            "selected_definition": None,
            "resolution_strategy": "workspace_priority",
            "reason": "Found in workspace",
            "conflicting_definitions": []
        }
        
        result = StepResolutionResult(**json_data)
        
        self.assertEqual(result.step_name, "TestStep")
        self.assertTrue(result.resolved)
        self.assertEqual(result.resolution_strategy, "workspace_priority")
        self.assertEqual(result.reason, "Found in workspace")


class TestModelEquality(unittest.TestCase):
    """Test model equality and comparison."""
    
    def test_step_definition_equality(self):
        """Test StepDefinition equality comparison."""
        definition1 = StepDefinition(
            name="TestStep",
            config_class="TestStepConfig",
            builder_step_name="TestStepBuilder",
            spec_type="TestStep",
            sagemaker_step_type="Processing",
            description="Test step",
            registry_type="core"
        )
        
        definition2 = StepDefinition(
            name="TestStep",
            config_class="TestStepConfig",
            builder_step_name="TestStepBuilder",
            spec_type="TestStep",
            sagemaker_step_type="Processing",
            description="Test step",
            registry_type="core"
        )
        
        self.assertEqual(definition1, definition2)
    
    def test_step_definition_inequality(self):
        """Test StepDefinition inequality comparison."""
        definition1 = StepDefinition(
            name="TestStep1",
            config_class="TestStepConfig",
            builder_step_name="TestStepBuilder",
            spec_type="TestStep",
            sagemaker_step_type="Processing",
            description="Test step",
            registry_type="core"
        )
        
        definition2 = StepDefinition(
            name="TestStep2",  # Different name
            config_class="TestStepConfig",
            builder_step_name="TestStepBuilder",
            spec_type="TestStep",
            sagemaker_step_type="Processing",
            description="Test step",
            registry_type="core"
        )
        
        self.assertNotEqual(definition1, definition2)
    
    def test_resolution_context_equality(self):
        """Test ResolutionContext equality comparison."""
        context1 = ResolutionContext(
            workspace_id="developer_1",
            preferred_framework="pytorch"
        )
        
        context2 = ResolutionContext(
            workspace_id="developer_1",
            preferred_framework="pytorch"
        )
        
        self.assertEqual(context1, context2)


if __name__ == "__main__":
    unittest.main()
