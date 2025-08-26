"""
Unit tests for templates.py module
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import tempfile
import shutil
import json
from datetime import datetime

import pytest

# Import the module under test
from src.cursus.validation.runtime.jupyter.templates import (
    NotebookTemplateManager,
    NotebookTemplate,
    JUPYTER_AVAILABLE
)


class TestNotebookTemplate(unittest.TestCase):
    """Test cases for NotebookTemplate model"""
    
    def test_notebook_template_creation(self):
        """Test NotebookTemplate creation with required fields"""
        template = NotebookTemplate(
            name="test_template",
            description="Test template description",
            category="testing"
        )
        
        self.assertEqual(template.name, "test_template")
        self.assertEqual(template.description, "Test template description")
        self.assertEqual(template.category, "testing")
        self.assertIsNone(template.template_path)
        self.assertIsNone(template.template_content)
        self.assertEqual(template.variables, {})
        self.assertEqual(template.required_imports, [])
        self.assertEqual(template.cell_templates, [])
        self.assertEqual(template.metadata, {})
        self.assertIsInstance(template.created_at, datetime)
    
    def test_notebook_template_with_optional_fields(self):
        """Test NotebookTemplate creation with optional fields"""
        template_path = Path("/tmp/template.json")
        variables = {"var1": "value1"}
        required_imports = ["import pandas as pd"]
        cell_templates = [{"cell_type": "code", "source": "print('hello')"}]
        metadata = {"author": "test"}
        
        template = NotebookTemplate(
            name="test_template",
            description="Test template description",
            category="testing",
            template_path=template_path,
            template_content="content",
            variables=variables,
            required_imports=required_imports,
            cell_templates=cell_templates,
            metadata=metadata
        )
        
        self.assertEqual(template.template_path, template_path)
        self.assertEqual(template.template_content, "content")
        self.assertEqual(template.variables, variables)
        self.assertEqual(template.required_imports, required_imports)
        self.assertEqual(template.cell_templates, cell_templates)
        self.assertEqual(template.metadata, metadata)


class TestNotebookTemplateManager(unittest.TestCase):
    """Test cases for NotebookTemplateManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.template_dir = Path(self.temp_dir) / "templates"
        self.manager = NotebookTemplateManager(self.template_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test NotebookTemplateManager initialization"""
        self.assertTrue(self.template_dir.exists())
        self.assertIsInstance(self.manager.templates, dict)
        
        # Should have built-in templates
        self.assertGreater(len(self.manager.templates), 0)
        self.assertIn("basic_testing", self.manager.templates)
        self.assertIn("data_analysis", self.manager.templates)
        self.assertIn("debugging", self.manager.templates)
    
    def test_initialization_without_jupyter(self):
        """Test initialization when Jupyter is not available"""
        with patch('src.cursus.validation.runtime.jupyter.templates.JUPYTER_AVAILABLE', False):
            manager = NotebookTemplateManager()
            self.assertIsNotNone(manager)
    
    def test_initialization_default_template_dir(self):
        """Test initialization with default template directory"""
        manager = NotebookTemplateManager()
        expected_dir = Path(__file__).parent.parent.parent.parent.parent / "src" / "cursus" / "validation" / "runtime" / "jupyter" / "templates"
        # Just check that it's a Path object, actual path may vary
        self.assertIsInstance(manager.template_dir, Path)
    
    def test_register_template(self):
        """Test registering a new template"""
        template = NotebookTemplate(
            name="custom_template",
            description="Custom test template",
            category="custom"
        )
        
        with patch('builtins.print') as mock_print:
            self.manager.register_template(template)
        
        self.assertIn("custom_template", self.manager.templates)
        self.assertEqual(self.manager.templates["custom_template"], template)
        mock_print.assert_called_with("Template 'custom_template' registered successfully")
    
    def test_list_templates(self):
        """Test listing all templates"""
        templates = self.manager.list_templates()
        
        self.assertIsInstance(templates, list)
        self.assertGreater(len(templates), 0)
        
        # Check structure of template info
        template_info = templates[0]
        expected_keys = ['name', 'description', 'category', 'created_at']
        for key in expected_keys:
            self.assertIn(key, template_info)
    
    def test_get_template_exists(self):
        """Test getting an existing template"""
        template = self.manager.get_template("basic_testing")
        
        self.assertIsNotNone(template)
        self.assertIsInstance(template, NotebookTemplate)
        self.assertEqual(template.name, "basic_testing")
    
    def test_get_template_not_exists(self):
        """Test getting a non-existent template"""
        template = self.manager.get_template("nonexistent_template")
        
        self.assertIsNone(template)
    
    def test_get_template_categories(self):
        """Test getting all template categories"""
        categories = self.manager.get_template_categories()
        
        self.assertIsInstance(categories, list)
        self.assertGreater(len(categories), 0)
        self.assertIn("testing", categories)
        self.assertIn("analysis", categories)
        self.assertIn("debugging", categories)
    
    def test_get_templates_by_category(self):
        """Test getting templates by category"""
        testing_templates = self.manager.get_templates_by_category("testing")
        
        self.assertIsInstance(testing_templates, list)
        self.assertGreater(len(testing_templates), 0)
        
        # All templates should be in testing category
        for template in testing_templates:
            self.assertEqual(template.category, "testing")
    
    def test_get_templates_by_nonexistent_category(self):
        """Test getting templates by non-existent category"""
        templates = self.manager.get_templates_by_category("nonexistent")
        
        self.assertIsInstance(templates, list)
        self.assertEqual(len(templates), 0)
    
    def test_save_template_to_file(self):
        """Test saving a template to file"""
        template = NotebookTemplate(
            name="test_template",
            description="Test template",
            category="testing"
        )
        self.manager.register_template(template)

        file_path = Path(self.temp_dir) / "test_template.json"
        
        with patch('builtins.print') as mock_print:
            self.manager.save_template_to_file("test_template", file_path)
        
        self.assertTrue(file_path.exists())
        
        # Verify file content
        with open(file_path) as f:
            data = json.load(f)
        
        self.assertEqual(data["name"], "test_template")
        self.assertEqual(data["description"], "Test template")
        self.assertEqual(data["category"], "testing")
        self.assertIn("created_at", data)
        
        mock_print.assert_called_with(f"Template saved to: {file_path}")
    
    def test_save_template_to_file_not_found(self):
        """Test saving a non-existent template to file"""
        file_path = Path(self.temp_dir) / "nonexistent.json"
        
        with patch('builtins.print') as mock_print:
            self.manager.save_template_to_file("nonexistent_template", file_path)
        
        self.assertFalse(file_path.exists())
        mock_print.assert_called_with("Template 'nonexistent_template' not found")
    
    def test_load_template_from_file(self):
        """Test loading a template from file"""
        # Create a test template file
        template_data = {
            "name": "loaded_template",
            "description": "Loaded from file",
            "category": "loaded",
            "variables": {"var1": "value1"},
            "required_imports": ["import os"],
            "cell_templates": [{"cell_type": "code", "source": "print('loaded')"}],
            "metadata": {"author": "test"},
            "created_at": datetime.now().isoformat()
        }

        file_path = Path(self.temp_dir) / "loaded_template.json"
        with open(file_path, 'w') as f:
            json.dump(template_data, f)
        
        template = self.manager.load_template_from_file(file_path)
        
        self.assertIsNotNone(template)
        self.assertIsInstance(template, NotebookTemplate)
        self.assertEqual(template.name, "loaded_template")
        self.assertEqual(template.description, "Loaded from file")
        self.assertEqual(template.category, "loaded")
        self.assertEqual(template.variables, {"var1": "value1"})
        self.assertEqual(template.required_imports, ["import os"])
        self.assertEqual(len(template.cell_templates), 1)
        self.assertEqual(template.metadata, {"author": "test"})
        
        # Should be registered in manager
        self.assertIn("loaded_template", self.manager.templates)
    
    def test_load_template_from_nonexistent_file(self):
        """Test loading a template from non-existent file"""
        file_path = Path(self.temp_dir) / "nonexistent.json"
        
        with patch('builtins.print') as mock_print:
            template = self.manager.load_template_from_file(file_path)
        
        self.assertIsNone(template)
        mock_print.assert_called()
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        self.assertTrue(any("not found" in call for call in print_calls))
    
    def test_load_template_from_invalid_file(self):
        """Test loading a template from invalid JSON file"""
        file_path = Path(self.temp_dir) / "invalid.json"
        file_path.write_text("invalid json content")
        
        with patch('builtins.print') as mock_print:
            template = self.manager.load_template_from_file(file_path)
        
        self.assertIsNone(template)
        mock_print.assert_called()
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        self.assertTrue(any("Error loading template" in call for call in print_calls))
    
    def test_create_custom_template(self):
        """Test creating a custom template"""
        cell_definitions = [
            {"cell_type": "markdown", "source": "# Custom Template"},
            {"cell_type": "code", "source": "print('custom')"}
        ]
        variables = {"custom_var": "custom_value"}
        required_imports = ["import custom_module"]
        
        template = self.manager.create_custom_template(
            name="custom_template",
            description="Custom template description",
            category="custom",
            cell_definitions=cell_definitions,
            variables=variables,
            required_imports=required_imports
        )
        
        self.assertIsInstance(template, NotebookTemplate)
        self.assertEqual(template.name, "custom_template")
        self.assertEqual(template.description, "Custom template description")
        self.assertEqual(template.category, "custom")
        self.assertEqual(template.cell_templates, cell_definitions)
        self.assertEqual(template.variables, variables)
        self.assertEqual(template.required_imports, required_imports)
        
        # Should be registered in manager
        self.assertIn("custom_template", self.manager.templates)
    
    def test_export_templates(self):
        """Test exporting all templates"""
        output_dir = Path(self.temp_dir) / "exported"
        
        with patch('builtins.print') as mock_print:
            self.manager.export_templates(output_dir)
        
        self.assertTrue(output_dir.exists())
        
        # Check that template files were created
        template_files = list(output_dir.glob("*.json"))
        self.assertGreater(len(template_files), 0)
        
        # Check that built-in templates were exported
        basic_testing_file = output_dir / "basic_testing.json"
        self.assertTrue(basic_testing_file.exists())
        
        mock_print.assert_called_with(f"All templates exported to: {output_dir}")
    
    def test_import_templates(self):
        """Test importing templates from directory"""
        # Create some template files
        import_dir = Path(self.temp_dir) / "import"
        import_dir.mkdir()
        
        template_data = {
            "name": "imported_template",
            "description": "Imported template",
            "category": "imported",
            "variables": {},
            "required_imports": [],
            "cell_templates": [],
            "metadata": {},
            "created_at": datetime.now().isoformat()
        }
        
        template_file = import_dir / "imported_template.json"
        with open(template_file, 'w') as f:
            json.dump(template_data, f)
        
        with patch('builtins.print') as mock_print:
            self.manager.import_templates(import_dir)
        
        # Should have imported the template
        self.assertIn("imported_template", self.manager.templates)
        
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        self.assertTrue(any("Imported 1 templates" in call for call in print_calls))
    
    def test_import_templates_nonexistent_dir(self):
        """Test importing templates from non-existent directory"""
        nonexistent_dir = Path(self.temp_dir) / "nonexistent"
        
        with patch('builtins.print') as mock_print:
            self.manager.import_templates(nonexistent_dir)
        
        mock_print.assert_called()
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        self.assertTrue(any("not found" in call for call in print_calls))


@unittest.skipIf(not JUPYTER_AVAILABLE, "Jupyter dependencies not available")
class TestNotebookTemplateManagerWithJupyter(unittest.TestCase):
    """Test cases that require Jupyter dependencies"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.template_dir = Path(self.temp_dir) / "templates"
        self.manager = NotebookTemplateManager(self.template_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    @patch('src.cursus.validation.runtime.jupyter.templates.nbformat')
    def test_create_notebook_from_template(self, mock_nbformat):
        """Test creating a notebook from template"""
        # Mock nbformat
        mock_nb = Mock()
        mock_cell = Mock()
        mock_nbformat.v4.new_notebook.return_value = mock_nb
        mock_nbformat.v4.new_code_cell.return_value = mock_cell
        mock_nbformat.v4.new_markdown_cell.return_value = mock_cell
        mock_nbformat.writes.return_value = "notebook_content"
        
        variables = {
            "pipeline_name": "test_pipeline",
            "pipeline_path": "./test.yaml",
            "timestamp": "2023-01-01 12:00:00"
        }
        
        result = self.manager.create_notebook_from_template(
            "basic_testing",
            variables=variables
        )
        
        self.assertEqual(result, "notebook_content")
        mock_nbformat.v4.new_notebook.assert_called_once()
        mock_nbformat.writes.assert_called_once()
    
    @patch('src.cursus.validation.runtime.jupyter.templates.nbformat')
    def test_create_notebook_from_template_with_output_path(self, mock_nbformat):
        """Test creating a notebook from template with output path"""
        # Mock nbformat
        mock_nb = Mock()
        mock_cell = Mock()
        mock_nbformat.v4.new_notebook.return_value = mock_nb
        mock_nbformat.v4.new_code_cell.return_value = mock_cell
        mock_nbformat.v4.new_markdown_cell.return_value = mock_cell
        
        output_path = Path(self.temp_dir) / "generated_notebook.ipynb"
        
        with patch('builtins.print') as mock_print:
            result = self.manager.create_notebook_from_template(
                "basic_testing",
                output_path=output_path
            )
        
        self.assertEqual(result, str(output_path))
        self.assertTrue(output_path.exists())
        mock_print.assert_called_with(f"Notebook created: {output_path}")
    
    def test_create_notebook_from_nonexistent_template(self):
        """Test creating notebook from non-existent template"""
        with patch('builtins.print') as mock_print:
            result = self.manager.create_notebook_from_template("nonexistent_template")
        
        self.assertIsNone(result)
        mock_print.assert_called_with("Template 'nonexistent_template' not found")
    
    @patch('src.cursus.validation.runtime.jupyter.templates.widgets')
    @patch('src.cursus.validation.runtime.jupyter.templates.display')
    def test_create_template_selector_widget(self, mock_display, mock_widgets):
        """Test creating template selector widget"""
        # Mock widgets
        mock_dropdown = Mock()
        mock_text = Mock()
        mock_button = Mock()
        mock_output = Mock()
        mock_vbox = Mock()
        mock_hbox = Mock()
        
        mock_widgets.Dropdown.return_value = mock_dropdown
        mock_widgets.Text.return_value = mock_text
        mock_widgets.Button.return_value = mock_button
        mock_widgets.Output.return_value = mock_output
        mock_widgets.VBox.return_value = mock_vbox
        mock_widgets.HBox.return_value = mock_hbox
        
        result = self.manager.create_template_selector_widget()
        
        self.assertIsNotNone(result)
        mock_widgets.Dropdown.assert_called()
        mock_widgets.Text.assert_called()
        mock_widgets.Button.assert_called()
        mock_widgets.Output.assert_called()
        mock_widgets.VBox.assert_called()


class TestNotebookTemplateManagerEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.template_dir = Path(self.temp_dir) / "templates"
        self.manager = NotebookTemplateManager(self.template_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_create_notebook_without_jupyter(self):
        """Test creating notebook when Jupyter is not available"""
        with patch('src.cursus.validation.runtime.jupyter.templates.JUPYTER_AVAILABLE', False):
            manager = NotebookTemplateManager()
            
            with patch('builtins.print') as mock_print:
                result = manager.create_notebook_from_template("basic_testing")
            
            self.assertIsNone(result)
            mock_print.assert_called_with("Cannot create notebook: Jupyter dependencies not available")
    
    def test_create_template_selector_widget_without_jupyter(self):
        """Test creating template selector widget when Jupyter is not available"""
        with patch('src.cursus.validation.runtime.jupyter.templates.JUPYTER_AVAILABLE', False):
            manager = NotebookTemplateManager()
            result = manager.create_template_selector_widget()
            self.assertIsNone(result)
    
    def test_save_template_creates_directory(self):
        """Test that saving template creates necessary directories"""
        template = NotebookTemplate(
            name="test_template",
            description="Test template",
            category="testing"
        )
        self.manager.register_template(template)

        # Use a nested path that doesn't exist
        nested_path = Path(self.temp_dir) / "nested" / "dir" / "template.json"
        
        self.manager.save_template_to_file("test_template", nested_path)
        
        self.assertTrue(nested_path.exists())
        self.assertTrue(nested_path.parent.exists())
    
    def test_load_template_with_missing_datetime(self):
        """Test loading template with missing created_at field"""
        template_data = {
            "name": "no_datetime_template",
            "description": "Template without datetime",
            "category": "test",
            "variables": {},
            "required_imports": [],
            "cell_templates": [],
            "metadata": {}
            # Missing created_at field
        }

        file_path = Path(self.temp_dir) / "no_datetime.json"
        with open(file_path, 'w') as f:
            json.dump(template_data, f)
        
        template = self.manager.load_template_from_file(file_path)
        
        self.assertIsNotNone(template)
        self.assertIsInstance(template.created_at, datetime)
    
    def test_create_custom_template_with_defaults(self):
        """Test creating custom template with default values"""
        cell_definitions = [{"cell_type": "code", "source": "print('test')"}]
        
        template = self.manager.create_custom_template(
            name="minimal_template",
            description="Minimal template",
            category="minimal",
            cell_definitions=cell_definitions
            # No variables or required_imports provided
        )
        
        self.assertEqual(template.variables, {})
        self.assertEqual(template.required_imports, [])
        self.assertEqual(template.cell_templates, cell_definitions)
    
    def test_import_templates_with_invalid_files(self):
        """Test importing templates with some invalid files"""
        import_dir = Path(self.temp_dir) / "mixed_import"
        import_dir.mkdir()
        
        # Create valid template file
        valid_template = {
            "name": "valid_template",
            "description": "Valid template",
            "category": "valid",
            "variables": {},
            "required_imports": [],
            "cell_templates": [],
            "metadata": {},
            "created_at": datetime.now().isoformat()
        }
        
        valid_file = import_dir / "valid.json"
        with open(valid_file, 'w') as f:
            json.dump(valid_template, f)
        
        # Create invalid template file
        invalid_file = import_dir / "invalid.json"
        invalid_file.write_text("invalid json")
        
        # Create non-JSON file (should be ignored)
        text_file = import_dir / "readme.txt"
        text_file.write_text("This is not a JSON file")
        
        with patch('builtins.print') as mock_print:
            self.manager.import_templates(import_dir)
        
        # Should have imported only the valid template
        self.assertIn("valid_template", self.manager.templates)
        
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        # Should report importing 1 template (only the valid one)
        self.assertTrue(any("Imported 1 templates" in call for call in print_calls))


if __name__ == '__main__':
    unittest.main()
