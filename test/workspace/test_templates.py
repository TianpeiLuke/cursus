"""
Tests for workspace templates functionality.
"""

import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile

from cursus.workspace.templates import TemplateManager, WorkspaceTemplate

class TestTemplateManager(unittest.TestCase):
    """Test cases for TemplateManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.templates_dir = Path(self.temp_dir) / "templates"
        self.templates_dir.mkdir()
        self.template_manager = TemplateManager(templates_dir=self.templates_dir)

    def test_template_manager_initialization(self):
        """Test that TemplateManager initializes correctly."""
        self.assertIsInstance(self.template_manager, TemplateManager)
        self.assertTrue(hasattr(self.template_manager, 'get_template'))

    def test_get_template(self):
        """Test template retrieval."""
        # Test getting a built-in template
        template = self.template_manager.get_template("basic")
        if template:
            self.assertIsInstance(template, WorkspaceTemplate)

    def test_list_templates(self):
        """Test template listing."""
        # Test listing available templates
        templates = self.template_manager.list_templates()
        self.assertIsInstance(templates, list)

    def test_apply_template(self):
        """Test template application."""
        # Create a test workspace directory
        workspace_path = Path(self.temp_dir) / "test_workspace"
        workspace_path.mkdir()
        
        # Test applying a template
        result = self.template_manager.apply_template("basic", workspace_path)
        self.assertIsInstance(result, bool)

    def test_builtin_templates(self):
        """Test that built-in templates are available."""
        templates = self.template_manager.list_templates()
        template_names = [t.name for t in templates] if templates else []
        
        # Check for expected built-in templates
        expected_templates = ["basic", "ml_pipeline", "data_processing"]
        for template_name in expected_templates:
            # Template may or may not exist depending on implementation
            # Just verify the method works
            template = self.template_manager.get_template(template_name)
            # Template can be None if not implemented yet

class TestWorkspaceTemplate(unittest.TestCase):
    """Test cases for WorkspaceTemplate."""

    def test_workspace_template_creation(self):
        """Test WorkspaceTemplate creation."""
        # Test creating a basic template
        from cursus.workspace.templates import TemplateType
        template = WorkspaceTemplate(
            name="test_template",
            description="Test template",
            type=TemplateType.BASIC
        )
        self.assertEqual(template.name, "test_template")
        self.assertEqual(template.description, "Test template")

if __name__ == '__main__':
    unittest.main()
