"""
Tests for workspace quality monitoring functionality.
"""

import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile

from src.cursus.workspace.quality.quality_monitor import WorkspaceQualityMonitor


class TestQualityMonitor(unittest.TestCase):
    """Test cases for QualityMonitor."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_path = Path(self.temp_dir)
        self.quality_monitor = WorkspaceQualityMonitor()

    def test_quality_monitor_initialization(self):
        """Test that QualityMonitor initializes correctly."""
        self.assertIsInstance(self.quality_monitor, WorkspaceQualityMonitor)
        self.assertTrue(hasattr(self.quality_monitor, 'run_quality_assessment'))

    def test_run_quality_assessment(self):
        """Test quality assessment execution."""
        # Test basic quality assessment
        quality_report = self.quality_monitor.run_quality_assessment()
        self.assertIsNotNone(quality_report)
        self.assertTrue(hasattr(quality_report, 'overall_score'))
        self.assertTrue(hasattr(quality_report, 'overall_status'))

    def test_quality_dimensions(self):
        """Test that all quality dimensions are assessed."""
        quality_report = self.quality_monitor.run_quality_assessment()
        
        # Check for key quality dimension attributes
        expected_dimensions = [
            'robustness_reliability',
            'maintainability_extensibility', 
            'scalability_performance',
            'reusability_modularity',
            'testability_observability',
            'security_safety',
            'usability_developer_experience'
        ]
        
        for dimension in expected_dimensions:
            self.assertTrue(hasattr(quality_report, dimension))

    def test_quality_gates_validation(self):
        """Test quality gates validation."""
        # Test that quality gates are included in assessment
        quality_report = self.quality_monitor.run_quality_assessment()
        self.assertTrue(hasattr(quality_report, 'quality_gates'))
        self.assertIsInstance(quality_report.quality_gates, list)

    def test_quality_dashboard(self):
        """Test quality dashboard generation."""
        # Run assessment first
        self.quality_monitor.run_quality_assessment()
        # Test dashboard generation
        dashboard = self.quality_monitor.get_quality_dashboard()
        self.assertIsInstance(dashboard, dict)


if __name__ == '__main__':
    unittest.main()
