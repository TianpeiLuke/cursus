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

from src.cursus.core.workspace.integration import WorkspaceIntegrationManager


class TestWorkspaceIntegrationManager(unittest.TestCase):
    """Test suite for WorkspaceIntegrationManager."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_workspace = str(Path(self.temp_dir) / "test_workspace")
        Path(self.temp_workspace).mkdir(parents=True, exist_ok=True)
        
        # Create mock workspace manager
        self.mock_workspace_manager = Mock()
        self.mock_workspace_manager.workspace_root = self.temp_workspace
        self.mock_workspace_manager.config_file = None
        self.mock_workspace_manager.auto_discover = True
    
    def tearDown(self):
        """Clean up after each test method."""
        if hasattr(self, 'temp_dir') and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_integration_manager_initialization(self):
        """Test WorkspaceIntegrationManager initialization."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        self.assertIs(integration_manager.workspace_manager, self.mock_workspace_manager)
        self.assertEqual(integration_manager.workspace_root, self.mock_workspace_manager.workspace_root)
        self.assertEqual(integration_manager.staging_areas, {})
        self.assertEqual(integration_manager.staged_components, {})
        self.assertEqual(integration_manager.integration_queue, [])
    
    def test_create_staging_area(self):
        """Test creating staging area."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        staging_area = integration_manager.create_staging_area("developer_1")
        
        self.assertEqual(staging_area.workspace_id, "developer_1")
        self.assertIsNotNone(staging_area.staging_path)
        self.assertIsNotNone(staging_area.created_at)
        self.assertIn("developer_1", integration_manager.staging_areas)
    
    def test_stage_component_for_integration_success(self):
        """Test successful component staging."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        # Create staging area first
        staging_area = integration_manager.create_staging_area("developer_1")
        
        # Create source component
        workspace_path = Path(self.temp_workspace) / "developer_1"
        workspace_path.mkdir(exist_ok=True)
        component_path = workspace_path / "src" / "test_component.py"
        component_path.parent.mkdir(parents=True, exist_ok=True)
        component_path.write_text("# Test component")
        
        result = integration_manager.stage_component_for_integration("test_component", "developer_1")
        
        self.assertTrue(result.success)
        self.assertEqual(result.component_id, "test_component")
        self.assertEqual(result.source_workspace, "developer_1")
        self.assertIsNotNone(result.staging_path)
    
    def test_stage_component_for_integration_missing_component(self):
        """Test staging non-existent component."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        # Create staging area
        integration_manager.create_staging_area("developer_1")
        
        result = integration_manager.stage_component_for_integration("non_existent_component", "developer_1")
        
        self.assertFalse(result.success)
        self.assertIn("not found", result.error.lower())
    
    def test_validate_integration_readiness_ready(self):
        """Test integration readiness validation with ready components."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        # Mock staged components
        integration_manager.staged_components = {
            "component_1": {
                "status": "staged",
                "validation_passed": True,
                "dependencies_resolved": True
            },
            "component_2": {
                "status": "staged", 
                "validation_passed": True,
                "dependencies_resolved": True
            }
        }
        
        report = integration_manager.validate_integration_readiness(["component_1", "component_2"])
        
        self.assertTrue(report.ready)
        self.assertEqual(len(report.ready_components), 2)
        self.assertEqual(len(report.blocking_issues), 0)
    
    def test_validate_integration_readiness_not_ready(self):
        """Test integration readiness validation with unready components."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        # Mock staged components with issues
        integration_manager.staged_components = {
            "component_1": {
                "status": "staged",
                "validation_passed": False,
                "dependencies_resolved": True
            },
            "component_2": {
                "status": "staged",
                "validation_passed": True,
                "dependencies_resolved": False
            }
        }
        
        report = integration_manager.validate_integration_readiness(["component_1", "component_2"])
        
        self.assertFalse(report.ready)
        self.assertEqual(len(report.ready_components), 0)
        self.assertGreater(len(report.blocking_issues), 0)
    
    def test_promote_to_production_success(self):
        """Test successful component promotion to production."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        # Mock staged component
        integration_manager.staged_components["component_1"] = {
            "status": "staged",
            "validation_passed": True,
            "dependencies_resolved": True,
            "staging_path": "/staging/component_1"
        }
        
        with patch('shutil.copytree') as mock_copy:
            result = integration_manager.promote_to_production("component_1")
            
            self.assertTrue(result.success)
            self.assertEqual(result.component_id, "component_1")
            self.assertIsNotNone(result.production_path)
            mock_copy.assert_called_once()
    
    def test_promote_to_production_not_ready(self):
        """Test promotion of component that's not ready."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        # Mock unready component
        integration_manager.staged_components["component_1"] = {
            "status": "staged",
            "validation_passed": False,
            "dependencies_resolved": True
        }
        
        result = integration_manager.promote_to_production("component_1")
        
        self.assertFalse(result.success)
        self.assertIn("not ready", result.error.lower())
    
    def test_validate_pipeline_integration_valid(self):
        """Test pipeline integration validation with valid pipeline."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        mock_pipeline = Mock()
        mock_pipeline.pipeline_name = "test_pipeline"
        mock_pipeline.steps = []
        
        result = integration_manager.validate_pipeline_integration(mock_pipeline)
        
        self.assertTrue(result['valid'])
        self.assertEqual(len(result['errors']), 0)
    
    def test_validate_pipeline_integration_invalid(self):
        """Test pipeline integration validation with invalid pipeline."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        mock_step = Mock()
        mock_step.step_name = "invalid_step"
        mock_step.developer_id = "developer_1"
        
        mock_pipeline = Mock()
        mock_pipeline.pipeline_name = "test_pipeline"
        mock_pipeline.steps = [mock_step]
        
        # Mock component as not staged
        integration_manager.staged_components = {}
        
        result = integration_manager.validate_pipeline_integration(mock_pipeline)
        
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['errors']), 0)
    
    def test_prepare_pipeline_for_integration(self):
        """Test preparing pipeline for integration."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        mock_step = Mock()
        mock_step.step_name = "test_step"
        mock_step.developer_id = "developer_1"
        
        mock_pipeline = Mock()
        mock_pipeline.pipeline_name = "test_pipeline"
        mock_pipeline.steps = [mock_step]
        
        with patch.object(integration_manager, 'stage_component_for_integration') as mock_stage:
            mock_stage.return_value = Mock(success=True, staging_path="/staging/test_step")
            
            result = integration_manager.prepare_pipeline_for_integration(mock_pipeline)
            
            self.assertTrue(result['ready'])
            self.assertEqual(result['pipeline_name'], "test_pipeline")
            mock_stage.assert_called_once_with("test_step", "developer_1")
    
    def test_get_staging_status(self):
        """Test getting staging status."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        # Add test data
        integration_manager.staged_components = {
            "component_1": {"status": "staged", "validation_passed": True},
            "component_2": {"status": "staging", "validation_passed": False}
        }
        integration_manager.integration_queue = ["component_3", "component_4"]
        
        status = integration_manager.get_staging_status()
        
        self.assertEqual(status.total_staged, 2)
        self.assertEqual(status.ready_for_integration, 1)
        self.assertEqual(status.queue_length, 2)
        self.assertEqual(len(status.component_details), 2)
    
    def test_cleanup_staging_area(self):
        """Test cleaning up staging area."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        # Create staging area
        staging_area = integration_manager.create_staging_area("developer_1")
        
        # Add staged component
        integration_manager.staged_components["component_1"] = {
            "workspace_id": "developer_1",
            "staging_path": "/staging/component_1"
        }
        
        with patch('shutil.rmtree') as mock_rmtree:
            result = integration_manager.cleanup_staging_area("developer_1")
            
            self.assertTrue(result.success)
            self.assertEqual(result.workspace_id, "developer_1")
            self.assertNotIn("developer_1", integration_manager.staging_areas)
            mock_rmtree.assert_called()
    
    def test_rollback_component_staging(self):
        """Test rolling back component staging."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        # Mock staged component
        integration_manager.staged_components["component_1"] = {
            "status": "staged",
            "staging_path": "/staging/component_1",
            "workspace_id": "developer_1"
        }
        
        with patch('shutil.rmtree') as mock_rmtree:
            result = integration_manager.rollback_component_staging("component_1")
            
            self.assertTrue(result.success)
            self.assertEqual(result.component_id, "component_1")
            self.assertNotIn("component_1", integration_manager.staged_components)
            mock_rmtree.assert_called_once()
    
    def test_get_integration_history(self):
        """Test getting integration history."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        # Mock integration history
        integration_manager.integration_history = [
            {
                "component_id": "component_1",
                "action": "staged",
                "timestamp": "2025-09-02T10:00:00",
                "workspace_id": "developer_1"
            },
            {
                "component_id": "component_1", 
                "action": "promoted",
                "timestamp": "2025-09-02T11:00:00",
                "workspace_id": "developer_1"
            }
        ]
        
        history = integration_manager.get_integration_history("component_1")
        
        self.assertEqual(len(history), 2)
        self.assertTrue(all(h["component_id"] == "component_1" for h in history))
    
    def test_validate_cross_workspace_compatibility(self):
        """Test cross-workspace compatibility validation."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        components = ["component_1", "component_2"]
        
        # Mock staged components from different workspaces
        integration_manager.staged_components = {
            "component_1": {"workspace_id": "developer_1", "version": "1.0"},
            "component_2": {"workspace_id": "developer_2", "version": "1.0"}
        }
        
        result = integration_manager.validate_cross_workspace_compatibility(components)
        
        self.assertTrue(result.compatible)
        self.assertEqual(len(result.conflicts), 0)
    
    def test_get_summary(self):
        """Test getting integration manager summary."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        # Add test data
        integration_manager.staging_areas = {"developer_1": Mock()}
        integration_manager.staged_components = {"component_1": {}, "component_2": {}}
        integration_manager.integration_queue = ["component_3"]
        
        summary = integration_manager.get_summary()
        
        self.assertIn('total_staging_areas', summary)
        self.assertIn('total_staged_components', summary)
        self.assertIn('integration_queue_length', summary)
        self.assertIn('workspace_root', summary)
        self.assertEqual(summary['total_staging_areas'], 1)
        self.assertEqual(summary['total_staged_components'], 2)
        self.assertEqual(summary['integration_queue_length'], 1)
    
    def test_validate_health(self):
        """Test integration manager health validation."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        health = integration_manager.validate_health()
        
        self.assertIn('healthy', health)
        self.assertIn('integration_system_functional', health)
        self.assertIn('staging_system_operational', health)
        self.assertIn('workspace_root_accessible', health)
        self.assertTrue(health['healthy'])
    
    def test_error_handling_invalid_workspace(self):
        """Test error handling for invalid workspace."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        # Try to create staging area for non-existent workspace
        staging_area = integration_manager.create_staging_area("non_existent_workspace")
        
        # Should still create staging area but may have warnings
        self.assertEqual(staging_area.workspace_id, "non_existent_workspace")
    
    def test_concurrent_staging_operations(self):
        """Test concurrent staging operations."""
        integration_manager = WorkspaceIntegrationManager(self.mock_workspace_manager)
        
        # Create multiple staging areas
        workspaces = ["developer_1", "developer_2", "developer_3"]
        staging_areas = []
        
        for workspace in workspaces:
            staging_area = integration_manager.create_staging_area(workspace)
            staging_areas.append(staging_area)
        
        # All should succeed
        self.assertEqual(len(staging_areas), 3)
        self.assertEqual(len(integration_manager.staging_areas), 3)
        self.assertTrue(all(sa.workspace_id in workspaces for sa in staging_areas))


if __name__ == "__main__":
    unittest.main()
