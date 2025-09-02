"""
Unit tests for WorkspaceIntegrationManager.

This module provides comprehensive unit testing for the WorkspaceIntegrationManager
and its integration staging capabilities.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from src.cursus.core.workspace.integration import (
    WorkspaceIntegrationManager, 
    StagedComponent, 
    IntegrationPipeline,
    ComponentStatus,
    IntegrationStage
)


class TestWorkspaceIntegrationManager(unittest.TestCase):
    """Test suite for WorkspaceIntegrationManager."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_workspace = Path(self.temp_dir) / "test_workspace"
        self.temp_workspace.mkdir(parents=True, exist_ok=True)
        
        # Create mock workspace manager
        self.mock_workspace_manager = Mock()
        self.mock_workspace_manager.workspace_root = self.temp_workspace
        self.mock_workspace_manager.discovery_manager = Mock()
    
    def tearDown(self):
        """Clean up after each test method."""
        if hasattr(self, 'temp_dir') and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_integration_manager_initialization(self):
        """Test WorkspaceIntegrationManager initialization."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        self.assertIs(integration_manager.workspace_manager, self.mock_workspace_manager)
        self.assertIsInstance(integration_manager.staged_components, dict)
        self.assertIsInstance(integration_manager.integration_pipelines, dict)
        self.assertIsNotNone(integration_manager.staging_root)
    
    def test_stage_for_integration_success(self):
        """Test successful component staging for integration."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        # Create source component
        developers_dir = self.temp_workspace / "developers"
        developers_dir.mkdir(exist_ok=True)
        workspace_path = developers_dir / "developer_1"
        workspace_path.mkdir(exist_ok=True)
        
        # Create component structure
        cursus_dev_dir = workspace_path / "src" / "cursus_dev" / "steps" / "builders"
        cursus_dev_dir.mkdir(parents=True, exist_ok=True)
        component_file = cursus_dev_dir / "test_component.py"
        component_file.write_text("class TestComponent: pass")
        
        # Mock discovery manager
        self.mock_workspace_manager.discovery_manager._check_component_exists.return_value = True
        
        result = integration_manager.stage_for_integration("test_component", "developer_1")
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result["success"])
        self.assertEqual(result["component_id"], "test_component")
        self.assertEqual(result["source_workspace"], "developer_1")
    
    def test_stage_for_integration_component_not_found(self):
        """Test staging non-existent component."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        # Mock discovery manager to return False
        self.mock_workspace_manager.discovery_manager._check_component_exists.return_value = False
        
        result = integration_manager.stage_for_integration("non_existent_component", "developer_1")
        
        self.assertIsInstance(result, dict)
        self.assertFalse(result["success"])
        self.assertGreater(len(result["issues"]), 0)
        self.assertIn("Component not found", result["issues"][0])
    
    def test_validate_integration_readiness_ready(self):
        """Test integration readiness validation with ready components."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        # Create staged components
        staged_comp1 = StagedComponent("comp1", "developer_1", "builders")
        staged_comp1.status = ComponentStatus.STAGED
        staged_comp1.metadata["staging_path"] = str(self.temp_workspace / "staging1")
        Path(staged_comp1.metadata["staging_path"]).mkdir(parents=True, exist_ok=True)
        
        staged_comp2 = StagedComponent("comp2", "developer_2", "scripts")
        staged_comp2.status = ComponentStatus.STAGED
        staged_comp2.metadata["staging_path"] = str(self.temp_workspace / "staging2")
        Path(staged_comp2.metadata["staging_path"]).mkdir(parents=True, exist_ok=True)
        
        integration_manager.staged_components["developer_1:comp1"] = staged_comp1
        integration_manager.staged_components["developer_2:comp2"] = staged_comp2
        
        result = integration_manager.validate_integration_readiness(["developer_1:comp1", "developer_2:comp2"])
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result["overall_ready"])
        self.assertEqual(len(result["component_results"]), 2)
    
    def test_validate_integration_readiness_not_ready(self):
        """Test integration readiness validation with missing components."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        result = integration_manager.validate_integration_readiness(["nonexistent:comp1", "missing:comp2"])
        
        self.assertIsInstance(result, dict)
        self.assertFalse(result["overall_ready"])
        self.assertGreater(len(result["integration_issues"]), 0)
        self.assertIn("Component not found in staging", result["integration_issues"][0])
    
    def test_promote_to_production_success(self):
        """Test successful component promotion to production."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        # Create shared workspace
        shared_dir = self.temp_workspace / "shared"
        shared_dir.mkdir(exist_ok=True)
        
        # Create staged component
        staged_comp = StagedComponent("test_comp", "developer_1", "builders")
        staged_comp.status = ComponentStatus.STAGED
        staging_path = self.temp_workspace / "staging" / "test_comp"
        staging_path.mkdir(parents=True, exist_ok=True)
        
        # Create test file in staging
        test_file = staging_path / "test_comp.py"
        test_file.write_text("class TestComp: pass")
        
        staged_comp.metadata = {
            "staging_path": str(staging_path),
            "original_files": [{"destination": str(test_file)}]
        }
        
        integration_manager.staged_components["developer_1:test_comp"] = staged_comp
        
        result = integration_manager.promote_to_production("test_comp")
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result["success"])
        self.assertEqual(result["component_id"], "test_comp")
        self.assertEqual(staged_comp.status, ComponentStatus.PROMOTED)
    
    def test_promote_to_production_not_found(self):
        """Test promotion of component that's not found."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        result = integration_manager.promote_to_production("nonexistent_component")
        
        self.assertIsInstance(result, dict)
        self.assertFalse(result["success"])
        self.assertIn("Component not found in staging", result["issues"][0])
    
    def test_rollback_integration_success(self):
        """Test successful integration rollback."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        # Create staged component
        staged_comp = StagedComponent("test_comp", "developer_1", "builders")
        staged_comp.status = ComponentStatus.PROMOTED
        integration_manager.staged_components["developer_1:test_comp"] = staged_comp
        
        with patch.object(integration_manager, '_remove_from_production') as mock_remove:
            result = integration_manager.rollback_integration("test_comp")
            
            self.assertIsInstance(result, dict)
            self.assertTrue(result["success"])
            self.assertEqual(result["component_id"], "test_comp")
            self.assertEqual(staged_comp.status, ComponentStatus.ROLLED_BACK)
            mock_remove.assert_called_once()
    
    def test_rollback_integration_not_found(self):
        """Test rollback of component that's not found."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        result = integration_manager.rollback_integration("nonexistent_component")
        
        self.assertIsInstance(result, dict)
        self.assertFalse(result["success"])
        self.assertIn("Component not found", result["issues"][0])
    
    def test_get_integration_summary(self):
        """Test getting integration summary."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        # Add test data
        staged_comp1 = StagedComponent("comp1", "developer_1", "builders")
        staged_comp1.status = ComponentStatus.STAGED
        staged_comp2 = StagedComponent("comp2", "developer_2", "scripts")
        staged_comp2.status = ComponentStatus.PROMOTED
        
        integration_manager.staged_components["developer_1:comp1"] = staged_comp1
        integration_manager.staged_components["developer_2:comp2"] = staged_comp2
        
        summary = integration_manager.get_integration_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn("staged_components", summary)
        self.assertIn("component_status_distribution", summary)
        self.assertEqual(summary["staged_components"], 2)
    
    def test_get_statistics(self):
        """Test getting integration statistics."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        # Add test data
        staged_comp = StagedComponent("comp1", "developer_1", "builders")
        staged_comp.status = ComponentStatus.PROMOTED
        integration_manager.staged_components["developer_1:comp1"] = staged_comp
        
        stats = integration_manager.get_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn("integration_operations", stats)
        self.assertIn("component_statistics", stats)
        self.assertIn("integration_summary", stats)
    
    def test_staged_component_creation(self):
        """Test StagedComponent creation and methods."""
        staged_comp = StagedComponent(
            component_id="test_comp",
            source_workspace="developer_1",
            component_type="builders",
            stage="integration",
            metadata={"key": "value"}
        )
        
        self.assertEqual(staged_comp.component_id, "test_comp")
        self.assertEqual(staged_comp.source_workspace, "developer_1")
        self.assertEqual(staged_comp.component_type, "builders")
        self.assertEqual(staged_comp.stage, "integration")
        self.assertEqual(staged_comp.status, ComponentStatus.PENDING)
        self.assertEqual(staged_comp.metadata["key"], "value")
        
        # Test dictionary conversion
        comp_dict = staged_comp.to_dict()
        self.assertIsInstance(comp_dict, dict)
        self.assertEqual(comp_dict["component_id"], "test_comp")
        self.assertEqual(comp_dict["status"], "pending")
    
    def test_integration_pipeline_creation(self):
        """Test IntegrationPipeline creation and methods."""
        staged_comp1 = StagedComponent("comp1", "developer_1", "builders")
        staged_comp2 = StagedComponent("comp2", "developer_2", "scripts")
        
        pipeline = IntegrationPipeline("test_pipeline", [staged_comp1, staged_comp2])
        
        self.assertEqual(pipeline.pipeline_id, "test_pipeline")
        self.assertEqual(len(pipeline.components), 2)
        self.assertEqual(pipeline.status, "pending")
        
        # Test dictionary conversion
        pipeline_dict = pipeline.to_dict()
        self.assertIsInstance(pipeline_dict, dict)
        self.assertEqual(pipeline_dict["pipeline_id"], "test_pipeline")
        self.assertEqual(len(pipeline_dict["components"]), 2)
    
    def test_component_status_enum(self):
        """Test ComponentStatus enum values."""
        self.assertEqual(ComponentStatus.PENDING.value, "pending")
        self.assertEqual(ComponentStatus.STAGED.value, "staged")
        self.assertEqual(ComponentStatus.APPROVED.value, "approved")
        self.assertEqual(ComponentStatus.REJECTED.value, "rejected")
        self.assertEqual(ComponentStatus.PROMOTED.value, "promoted")
        self.assertEqual(ComponentStatus.ROLLED_BACK.value, "rolled_back")
    
    def test_integration_stage_enum(self):
        """Test IntegrationStage enum values."""
        self.assertEqual(IntegrationStage.DEVELOPMENT.value, "development")
        self.assertEqual(IntegrationStage.STAGING.value, "staging")
        self.assertEqual(IntegrationStage.INTEGRATION.value, "integration")
        self.assertEqual(IntegrationStage.PRODUCTION.value, "production")
    
    def test_determine_component_type(self):
        """Test component type determination."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        # Create component structure
        developers_dir = self.temp_workspace / "developers"
        developers_dir.mkdir(exist_ok=True)
        workspace_path = developers_dir / "developer_1"
        workspace_path.mkdir(exist_ok=True)
        
        # Create builder component
        builders_dir = workspace_path / "src" / "cursus_dev" / "steps" / "builders"
        builders_dir.mkdir(parents=True, exist_ok=True)
        (builders_dir / "test_builder.py").write_text("class TestBuilder: pass")
        
        component_type = integration_manager._determine_component_type("test_builder", "developer_1")
        self.assertEqual(component_type, "builders")
        
        # Test non-existent component
        component_type = integration_manager._determine_component_type("nonexistent", "developer_1")
        self.assertIsNone(component_type)
    
    def test_validate_component_syntax(self):
        """Test component syntax validation."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        # Create staging path with valid Python file
        staging_path = self.temp_workspace / "staging"
        staging_path.mkdir(exist_ok=True)
        
        valid_file = staging_path / "valid.py"
        valid_file.write_text("class ValidComponent:\n    def __init__(self):\n        pass")
        
        result = integration_manager._validate_component_syntax(staging_path)
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["issues"]), 0)
        
        # Test invalid syntax
        invalid_file = staging_path / "invalid.py"
        invalid_file.write_text("class InvalidComponent\n    def __init__(self):")  # Missing colon
        
        result = integration_manager._validate_component_syntax(staging_path)
        self.assertFalse(result["valid"])
        self.assertGreater(len(result["issues"]), 0)
    
    def test_copy_component_to_staging(self):
        """Test copying component to staging area."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        # Create source component
        developers_dir = self.temp_workspace / "developers"
        developers_dir.mkdir(exist_ok=True)
        workspace_path = developers_dir / "developer_1"
        workspace_path.mkdir(exist_ok=True)
        
        builders_dir = workspace_path / "src" / "cursus_dev" / "steps" / "builders"
        builders_dir.mkdir(parents=True, exist_ok=True)
        component_file = builders_dir / "test_comp.py"
        component_file.write_text("class TestComp: pass")
        
        # Create staging path
        staging_path = self.temp_workspace / "staging"
        staging_path.mkdir(exist_ok=True)
        
        result = integration_manager._copy_component_to_staging(
            "test_comp", "developer_1", "builders", staging_path
        )
        
        self.assertTrue(result["success"])
        self.assertGreater(len(result["copied_files"]), 0)
        self.assertTrue((staging_path / "test_comp.py").exists())
    
    def test_error_handling_no_staging_root(self):
        """Test error handling when no staging root is configured."""
        self.mock_workspace_manager.workspace_root = None
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        self.assertIsNone(integration_manager.staging_root)
        
        # Mock discovery manager to avoid the component type determination error
        self.mock_workspace_manager.discovery_manager._check_component_exists.return_value = True
        
        result = integration_manager.stage_for_integration("test_comp", "developer_1")
        self.assertFalse(result["success"])
        # The error should be about staging root, but it might come after component type determination
        self.assertGreater(len(result["issues"]), 0)


if __name__ == "__main__":
    unittest.main()
