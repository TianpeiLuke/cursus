"""
Unit tests for WorkspaceValidationOrchestrator.

Tests comprehensive workspace validation orchestration functionality including:
- High-level orchestration for workspace validation operations
- Coordination of alignment and builder testing
- Single and multi-workspace validation capabilities
- Parallel validation support
- Comprehensive validation reporting
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from src.cursus.validation.workspace.workspace_orchestrator import WorkspaceValidationOrchestrator
from src.cursus.validation.workspace.workspace_manager import WorkspaceManager


class TestWorkspaceValidationOrchestrator(unittest.TestCase):
    """Test cases for WorkspaceValidationOrchestrator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_root = Path(self.temp_dir)
        
        # Create mock workspace structure
        self.dev1_path = self.workspace_root / "developer_1"
        self.dev2_path = self.workspace_root / "developer_2"
        
        for dev_path in [self.dev1_path, self.dev2_path]:
            for subdir in ["builders", "contracts", "scripts", "specs", "configs"]:
                (dev_path / subdir).mkdir(parents=True, exist_ok=True)
        
        # Create mock workspace manager
        self.mock_workspace_manager = Mock()
        self.mock_workspace_manager.workspace_root = self.workspace_root
        self.mock_workspace_manager.list_available_developers.return_value = [
            "developer_1", "developer_2"
        ]
        
        # Create orchestrator instance
        self.orchestrator = WorkspaceValidationOrchestrator(
            workspace_root=self.workspace_root
        )
        # Inject mock workspace manager for testing
        self.orchestrator.workspace_manager = self.mock_workspace_manager
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test proper initialization of WorkspaceValidationOrchestrator."""
        self.assertIsNotNone(self.orchestrator.workspace_manager)
        self.assertEqual(self.orchestrator.workspace_manager, self.mock_workspace_manager)
        self.assertIsNotNone(self.orchestrator.alignment_tester)
        self.assertIsNotNone(self.orchestrator.builder_tester)
    
    def test_initialization_with_custom_testers(self):
        """Test initialization with custom tester instances."""
        mock_alignment_tester = Mock()
        mock_builder_tester = Mock()
        
        orchestrator = WorkspaceValidationOrchestrator(
            workspace_root=self.workspace_root
        )
        # Inject custom testers for testing
        orchestrator.alignment_tester = mock_alignment_tester
        orchestrator.builder_tester = mock_builder_tester
        
        self.assertEqual(orchestrator.alignment_tester, mock_alignment_tester)
        self.assertEqual(orchestrator.builder_tester, mock_builder_tester)
    
    @patch('src.cursus.validation.workspace.workspace_orchestrator.WorkspaceUnifiedAlignmentTester')
    @patch('src.cursus.validation.workspace.workspace_orchestrator.WorkspaceUniversalStepBuilderTest')
    def test_validate_workspace_single_developer(self, mock_builder_class, mock_alignment_class):
        """Test validation of single developer workspace."""
        # Mock alignment tester
        mock_alignment_tester = Mock()
        mock_alignment_tester.run_workspace_validation.return_value = {
            "developer_1": {
                "level1": {"passed": True, "errors": []},
                "level2": {"passed": True, "errors": []},
                "level3": {"passed": True, "errors": []},
                "level4": {"passed": True, "errors": []}
            }
        }
        mock_alignment_class.return_value = mock_alignment_tester
        
        # Mock builder tester
        mock_builder_tester = Mock()
        mock_builder_tester.run_workspace_builder_test.return_value = {
            "developer_1": {
                "TestBuilder": {"passed": True, "errors": []}
            }
        }
        mock_builder_class.return_value = mock_builder_tester
        
        # Create new orchestrator to use mocked classes
        orchestrator = WorkspaceValidationOrchestrator(
            workspace_root=self.workspace_root
        )
        orchestrator.workspace_manager = self.mock_workspace_manager
        
        results = orchestrator.validate_workspace("developer_1")
        
        self.assertIsNotNone(results)
        # Check comprehensive structure
        self.assertEqual(results['developer_id'], 'developer_1')
        self.assertIn('results', results)
        self.assertIn('alignment', results['results'])
        self.assertIn('builders', results['results'])
        self.assertIn('summary', results)
        self.assertIn('recommendations', results)
        self.assertTrue(results['success'])
        
        # Verify testers were called correctly (called twice: once in init, once in validation)
        self.assertEqual(mock_alignment_class.call_count, 2)
        self.assertEqual(mock_builder_class.call_count, 2)
    
    @patch('src.cursus.validation.workspace.workspace_orchestrator.WorkspaceUnifiedAlignmentTester')
    @patch('src.cursus.validation.workspace.workspace_orchestrator.WorkspaceUniversalStepBuilderTest')
    def test_validate_workspace_with_failures(self, mock_builder_class, mock_alignment_class):
        """Test validation with failures in workspace."""
        # Mock alignment tester with failures
        mock_alignment_tester = Mock()
        mock_alignment_tester.run_workspace_validation.return_value = {
            "developer_1": {
                "level1": {"passed": True, "errors": []},
                "level2": {"passed": False, "errors": ["Contract mismatch"]},
                "level3": {"passed": True, "errors": []},
                "level4": {"passed": True, "errors": []}
            }
        }
        mock_alignment_class.return_value = mock_alignment_tester
        
        # Mock builder tester with failures
        mock_builder_tester = Mock()
        mock_builder_tester.run_workspace_builder_test.return_value = {
            "developer_1": {
                "TestBuilder": {"passed": False, "errors": ["Builder validation failed"]}
            }
        }
        mock_builder_class.return_value = mock_builder_tester
        
        # Create new orchestrator to use mocked classes
        orchestrator = WorkspaceValidationOrchestrator(
            workspace_root=self.workspace_root
        )
        orchestrator.workspace_manager = self.mock_workspace_manager
        
        results = orchestrator.validate_workspace("developer_1")
        
        self.assertIsNotNone(results)
        # Check comprehensive structure with failures
        self.assertEqual(results['developer_id'], 'developer_1')
        self.assertFalse(results['success'])  # Should be false due to failures
        self.assertIn('results', results)
        
        # Check alignment failures in the nested structure
        alignment_results = results['results']['alignment']
        self.assertIn('developer_1', alignment_results)
        self.assertFalse(alignment_results['developer_1']['level2']['passed'])
        
        # Check builder failures in the nested structure
        builder_results = results['results']['builders']
        self.assertIn('developer_1', builder_results)
        self.assertFalse(builder_results['developer_1']['TestBuilder']['passed'])
    
    def test_validate_workspace_invalid_developer(self):
        """Test validation with invalid developer name."""
        self.mock_workspace_manager.list_available_developers.return_value = ["developer_1"]
        
        results = self.orchestrator.validate_workspace("invalid_developer")
        
        # Should return error structure, not empty dict
        self.assertIn('error', results)
        self.assertIn('Developer workspace not found', results['error'])
        self.assertFalse(results['success'])
    
    @patch('src.cursus.validation.workspace.workspace_orchestrator.WorkspaceUnifiedAlignmentTester')
    @patch('src.cursus.validation.workspace.workspace_orchestrator.WorkspaceUniversalStepBuilderTest')
    def test_validate_all_workspaces(self, mock_builder_class, mock_alignment_class):
        """Test validation of all workspaces."""
        # Mock alignment tester
        mock_alignment_tester = Mock()
        mock_alignment_tester.run_workspace_validation.return_value = {
            "developer_1": {
                "level1": {"passed": True, "errors": []},
                "level2": {"passed": True, "errors": []},
                "level3": {"passed": True, "errors": []},
                "level4": {"passed": True, "errors": []}
            },
            "developer_2": {
                "level1": {"passed": True, "errors": []},
                "level2": {"passed": True, "errors": []},
                "level3": {"passed": True, "errors": []},
                "level4": {"passed": True, "errors": []}
            }
        }
        mock_alignment_class.return_value = mock_alignment_tester
        
        # Mock builder tester
        mock_builder_tester = Mock()
        mock_builder_tester.run_workspace_builder_test.return_value = {
            "developer_1": {
                "TestBuilder": {"passed": True, "errors": []}
            },
            "developer_2": {
                "TestBuilder": {"passed": True, "errors": []}
            }
        }
        mock_builder_class.return_value = mock_builder_tester
        
        # Create new orchestrator to use mocked classes
        orchestrator = WorkspaceValidationOrchestrator(
            workspace_root=self.workspace_root
        )
        orchestrator.workspace_manager = self.mock_workspace_manager
        
        results = orchestrator.validate_all_workspaces()
        
        self.assertIsNotNone(results)
        # Check comprehensive multi-workspace structure
        self.assertIn('results', results)
        self.assertIn('summary', results)
        self.assertIn('recommendations', results)
        self.assertIn('total_workspaces', results)
        self.assertEqual(results['total_workspaces'], 2)
        
        # Check that individual workspace results are nested under 'results'
        self.assertIn("developer_1", results['results'])
        self.assertIn("developer_2", results['results'])
        
        # Verify testers were called correctly (not with all_developers=True)
        mock_alignment_class.assert_called()
        mock_builder_class.assert_called()
    
    @patch('src.cursus.validation.workspace.workspace_orchestrator.concurrent.futures.ThreadPoolExecutor')
    @patch('src.cursus.validation.workspace.workspace_orchestrator.WorkspaceUnifiedAlignmentTester')
    @patch('src.cursus.validation.workspace.workspace_orchestrator.WorkspaceUniversalStepBuilderTest')
    def test_validate_all_workspaces_parallel(self, mock_builder_class, mock_alignment_class, mock_executor_class):
        """Test parallel validation of all workspaces."""
        # Mock executor
        mock_executor = Mock()
        mock_future1 = Mock()
        mock_future2 = Mock()
        mock_future1.result.return_value = {
            "developer_1": {"alignment": {"level1": {"passed": True, "errors": []}}}
        }
        mock_future2.result.return_value = {
            "developer_1": {"builders": {"TestBuilder": {"passed": True, "errors": []}}}
        }
        mock_executor.submit.side_effect = [mock_future1, mock_future2]
        mock_executor.__enter__.return_value = mock_executor
        mock_executor.__exit__.return_value = None
        mock_executor_class.return_value = mock_executor
        
        # Mock testers
        mock_alignment_tester = Mock()
        mock_builder_tester = Mock()
        mock_alignment_class.return_value = mock_alignment_tester
        mock_builder_class.return_value = mock_builder_tester
        
        # Create new orchestrator to use mocked classes
        orchestrator = WorkspaceValidationOrchestrator(
            workspace_root=self.workspace_root
        )
        orchestrator.workspace_manager = self.mock_workspace_manager
        
        results = orchestrator.validate_all_workspaces(parallel=True)
        
        self.assertIsNotNone(results)
        # Verify executor was used
        mock_executor_class.assert_called_once()
        self.assertEqual(mock_executor.submit.call_count, 2)
    
    def test_generate_validation_report(self):
        """Test generation of validation report."""
        # Mock validation results
        validation_results = {
            "developer_1": {
                "alignment": {
                    "level1": {"passed": True, "errors": []},
                    "level2": {"passed": False, "errors": ["Contract mismatch"]},
                    "level3": {"passed": True, "errors": []},
                    "level4": {"passed": True, "errors": []}
                },
                "builders": {
                    "TestBuilder": {"passed": True, "errors": []}
                }
            },
            "developer_2": {
                "alignment": {
                    "level1": {"passed": True, "errors": []},
                    "level2": {"passed": True, "errors": []},
                    "level3": {"passed": True, "errors": []},
                    "level4": {"passed": True, "errors": []}
                },
                "builders": {
                    "TestBuilder": {"passed": False, "errors": ["Builder validation failed"]}
                }
            }
        }
        
        report = self.orchestrator.generate_validation_report(validation_results)
        
        self.assertIsNotNone(report)
        self.assertIn("summary", report)
        self.assertIn("details", report)
        self.assertIn("recommendations", report)
        
        # Check summary
        self.assertEqual(report["summary"]["total_workspaces"], 2)
        self.assertEqual(report["summary"]["failed_workspaces"], 2)  # Both have failures
        
        # Check details
        self.assertIn("developer_1", report["details"])
        self.assertIn("developer_2", report["details"])
    
    def test_generate_validation_report_all_passed(self):
        """Test report generation when all validations pass."""
        validation_results = {
            "developer_1": {
                "alignment": {
                    "level1": {"passed": True, "errors": []},
                    "level2": {"passed": True, "errors": []},
                    "level3": {"passed": True, "errors": []},
                    "level4": {"passed": True, "errors": []}
                },
                "builders": {
                    "TestBuilder": {"passed": True, "errors": []}
                }
            }
        }
        
        report = self.orchestrator.generate_validation_report(validation_results)
        
        self.assertEqual(report["summary"]["failed_workspaces"], 0)
        self.assertEqual(len(report["recommendations"]), 0)
    
    def test_analyze_cross_workspace_dependencies(self):
        """Test cross-workspace dependency analysis."""
        validation_results = {
            "developer_1": {
                "alignment": {
                    "level3": {
                        "passed": True,
                        "errors": [],
                        "dependencies": ["step_a", "step_b"]
                    }
                }
            },
            "developer_2": {
                "alignment": {
                    "level3": {
                        "passed": True,
                        "errors": [],
                        "dependencies": ["step_b", "step_c"]
                    }
                }
            }
        }
        
        dependencies = self.orchestrator._analyze_cross_workspace_dependencies(validation_results)
        
        self.assertIsNotNone(dependencies)
        self.assertIn("shared_dependencies", dependencies)
        self.assertIn("step_b", dependencies["shared_dependencies"])
        self.assertIn("workspace_specific", dependencies)
    
    def test_generate_recommendations(self):
        """Test recommendation generation based on validation results."""
        validation_results = {
            "developer_1": {
                "alignment": {
                    "level2": {"passed": False, "errors": ["Contract mismatch"]},
                    "level4": {"passed": False, "errors": ["Config validation failed"]}
                },
                "builders": {
                    "TestBuilder": {"passed": False, "errors": ["Builder validation failed"]}
                }
            }
        }
        
        recommendations = self.orchestrator._generate_recommendations(validation_results)
        
        self.assertIsInstance(recommendations, list)
        self.assertTrue(len(recommendations) > 0)
        
        # Check that recommendations address the failures
        recommendation_text = " ".join(recommendations)
        self.assertIn("contract", recommendation_text.lower())
    
    def test_get_validation_summary(self):
        """Test validation summary generation."""
        validation_results = {
            "developer_1": {
                "alignment": {
                    "level1": {"passed": True, "errors": []},
                    "level2": {"passed": False, "errors": ["Error"]}
                },
                "builders": {
                    "TestBuilder": {"passed": True, "errors": []}
                }
            },
            "developer_2": {
                "alignment": {
                    "level1": {"passed": True, "errors": []},
                    "level2": {"passed": True, "errors": []}
                },
                "builders": {
                    "TestBuilder": {"passed": True, "errors": []}
                }
            }
        }
        
        summary = self.orchestrator._get_validation_summary(validation_results)
        
        self.assertEqual(summary["total_workspaces"], 2)
        self.assertEqual(summary["passed_workspaces"], 1)
        self.assertEqual(summary["failed_workspaces"], 1)
        self.assertIn("alignment_results", summary)
        self.assertIn("builder_results", summary)
    
    def test_error_handling_during_validation(self):
        """Test error handling when validation fails."""
        # Mock alignment tester to raise exception
        mock_alignment_tester = Mock()
        mock_alignment_tester.run_workspace_validation.side_effect = Exception("Validation error")
        
        # Mock builder tester
        mock_builder_tester = Mock()
        mock_builder_tester.run_workspace_builder_test.return_value = {}
        
        orchestrator = WorkspaceValidationOrchestrator(
            workspace_root=self.workspace_root
        )
        orchestrator.workspace_manager = self.mock_workspace_manager
        orchestrator.alignment_tester = mock_alignment_tester
        orchestrator.builder_tester = mock_builder_tester
        
        results = orchestrator.validate_workspace("developer_1")
        
        # Should handle error gracefully
        self.assertIsNotNone(results)
        # Results might be empty or contain error information
    
    @patch('src.cursus.validation.workspace.workspace_orchestrator.WorkspaceValidationOrchestrator')
    def test_validate_all_workspaces_class_method(self, mock_class):
        """Test the class method for validating all workspaces."""
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        mock_instance.validate_all_workspaces.return_value = {
            "developer_1": {"alignment": {}, "builders": {}},
            "developer_2": {"alignment": {}, "builders": {}}
        }
        
        results = WorkspaceValidationOrchestrator.validate_all_workspaces_static(
            workspace_root=self.workspace_root
        )
        
        self.assertIsNotNone(results)
        mock_class.assert_called_once_with(workspace_root=self.workspace_root)
        mock_instance.validate_all_workspaces.assert_called_once()


if __name__ == '__main__':
    unittest.main()
