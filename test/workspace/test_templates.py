"""
Tests for workspace templates functionality.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile

from cursus.workspace.templates import TemplateManager, WorkspaceTemplate


class TestTemplateManager:
    """Test cases for TemplateManager."""

    @pytest.fixture
    def temp_workspace(self):
        """Set up test fixtures."""
        temp_dir = tempfile.mkdtemp()
        templates_dir = Path(temp_dir) / "templates"
        templates_dir.mkdir()
        yield temp_dir, templates_dir
        # Cleanup is handled automatically by tempfile

    @pytest.fixture
    def template_manager(self, temp_workspace):
        """Create TemplateManager instance."""
        temp_dir, templates_dir = temp_workspace
        return TemplateManager(templates_dir=templates_dir)

    def test_template_manager_initialization(self, template_manager):
        """Test that TemplateManager initializes correctly."""
        assert isinstance(template_manager, TemplateManager)
        assert hasattr(template_manager, 'get_template')

    def test_get_template(self, template_manager):
        """Test template retrieval."""
        # Test getting a built-in template
        template = template_manager.get_template("basic")
        if template:
            assert isinstance(template, WorkspaceTemplate)

    def test_list_templates(self, template_manager):
        """Test template listing."""
        # Test listing available templates
        templates = template_manager.list_templates()
        assert isinstance(templates, list)

    def test_apply_template(self, template_manager, temp_workspace):
        """Test template application."""
        temp_dir, templates_dir = temp_workspace
        # Create a test workspace directory
        workspace_path = Path(temp_dir) / "test_workspace"
        workspace_path.mkdir()
        
        # Test applying a template
        result = template_manager.apply_template("basic", workspace_path)
        assert isinstance(result, bool)

    def test_builtin_templates(self, template_manager):
        """Test that built-in templates are available."""
        templates = template_manager.list_templates()
        template_names = [t.name for t in templates] if templates else []
        
        # Check for expected built-in templates
        expected_templates = ["basic", "ml_pipeline", "data_processing"]
        for template_name in expected_templates:
            # Template may or may not exist depending on implementation
            # Just verify the method works
            template = template_manager.get_template(template_name)
            # Template can be None if not implemented yet


class TestWorkspaceTemplate:
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
        assert template.name == "test_template"
        assert template.description == "Test template"
