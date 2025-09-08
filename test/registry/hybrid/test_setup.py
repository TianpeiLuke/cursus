"""
Test suite for hybrid registry setup utilities.

Tests workspace creation, registry templates, and workspace initialization functions.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, mock_open

from cursus.registry.hybrid.setup import (
    create_workspace_registry,
    create_workspace_structure,
    create_workspace_documentation,
    create_example_implementations,
    validate_workspace_setup,
    copy_registry_from_developer,
    _get_registry_template,
    _get_standard_template,
    _get_minimal_template
)

class TestWorkspaceRegistryCreation(unittest.TestCase):
    """Test workspace registry creation functions."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_path = Path(self.temp_dir) / "test_workspace"
        self.workspace_path.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_workspace_registry_standard(self):
        """Test creating workspace registry with standard template."""
        registry_file = create_workspace_registry(
            str(self.workspace_path),
            "test_developer",
            "standard"
        )
        
        # Check that registry file was created
        self.assertTrue(Path(registry_file).exists())
        
        # Check directory structure
        expected_dir = self.workspace_path / "src" / "cursus_dev" / "registry"
        self.assertTrue(expected_dir.exists())
        
        # Check __init__.py was created
        init_file = expected_dir / "__init__.py"
        self.assertTrue(init_file.exists())
        
        # Check registry content
        with open(registry_file, 'r') as f:
            content = f.read()
        
        self.assertIn("LOCAL_STEPS", content)
        self.assertIn("STEP_OVERRIDES", content)
        self.assertIn("WORKSPACE_METADATA", content)
        self.assertIn("test_developer", content)
    
    def test_create_workspace_registry_minimal(self):
        """Test creating workspace registry with minimal template."""
        registry_file = create_workspace_registry(
            str(self.workspace_path),
            "minimal_dev",
            "minimal"
        )
        
        # Check that registry file was created
        self.assertTrue(Path(registry_file).exists())
        
        # Check registry content is minimal
        with open(registry_file, 'r') as f:
            content = f.read()
        
        self.assertIn("LOCAL_STEPS", content)
        self.assertIn("STEP_OVERRIDES", content)
        self.assertIn("WORKSPACE_METADATA", content)
        self.assertIn("minimal_dev", content)
        # Minimal template should be shorter
        self.assertLess(len(content), 1000)  # Arbitrary threshold
    
    def test_create_workspace_registry_invalid_developer_id(self):
        """Test creating workspace registry with invalid developer ID."""
        # Empty developer ID
        with self.assertRaises(ValueError) as exc_info:
            create_workspace_registry(
                str(self.workspace_path),
                "",
                "standard"
            )
        self.assertIn("Developer ID cannot be empty", str(exc_info.exception))
        
        # Invalid characters in developer ID
        with self.assertRaises(ValueError) as exc_info:
            create_workspace_registry(
                str(self.workspace_path),
                "invalid@dev!",
                "standard"
            )
        self.assertIn("contains invalid characters", str(exc_info.exception))
    
    def test_create_workspace_registry_valid_developer_ids(self):
        """Test creating workspace registry with various valid developer IDs."""
        valid_ids = ["dev1", "developer_2", "test-dev", "DevTeam123"]
        
        for dev_id in valid_ids:
            workspace_subdir = self.workspace_path / dev_id
            workspace_subdir.mkdir(exist_ok=True)
            
            registry_file = create_workspace_registry(
                str(workspace_subdir),
                dev_id,
                "standard"
            )
            
            self.assertTrue(Path(registry_file).exists())
            
            # Check content contains the developer ID
            with open(registry_file, 'r') as f:
                content = f.read()
            self.assertIn(dev_id, content)

class TestRegistryTemplates(unittest.TestCase):
    """Test registry template generation functions."""
    
    def test_get_registry_template_standard(self):
        """Test getting standard registry template."""
        template = _get_registry_template("test_dev", "standard")
        
        self.assertIn("LOCAL_STEPS", template)
        self.assertIn("STEP_OVERRIDES", template)
        self.assertIn("WORKSPACE_METADATA", template)
        self.assertIn("test_dev", template)
        self.assertIn("Example:", template)  # Should have examples
    
    def test_get_registry_template_minimal(self):
        """Test getting minimal registry template."""
        template = _get_registry_template("minimal_dev", "minimal")
        
        self.assertIn("LOCAL_STEPS", template)
        self.assertIn("STEP_OVERRIDES", template)
        self.assertIn("WORKSPACE_METADATA", template)
        self.assertIn("minimal_dev", template)
        # Minimal should be shorter and have fewer examples
        self.assertLess(len(template), len(_get_registry_template("minimal_dev", "standard")))
    
    def test_get_standard_template(self):
        """Test standard template generation."""
        template = _get_standard_template("standard_dev")
        
        self.assertIn("standard_dev", template)
        self.assertIn("MyCustomProcessingStep", template)  # Example step
        self.assertIn("ExperimentalTrainingStep", template)  # Example step
        self.assertIn("XGBoostTraining", template)  # Override example
        self.assertIn("version", template)
    
    def test_get_minimal_template(self):
        """Test minimal template generation."""
        template = _get_minimal_template("minimal_dev")
        
        self.assertIn("minimal_dev", template)
        self.assertIn("LOCAL_STEPS", template)
        self.assertIn("STEP_OVERRIDES", template)
        self.assertIn("WORKSPACE_METADATA", template)
        # Should not have detailed examples
        self.assertNotIn("MyCustomProcessingStep", template)

class TestWorkspaceStructure(unittest.TestCase):
    """Test workspace structure creation."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_path = Path(self.temp_dir) / "structure_test"
        self.workspace_path.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_workspace_structure(self):
        """Test creating complete workspace directory structure."""
        create_workspace_structure(str(self.workspace_path))
        
        # Check all expected directories exist
        expected_dirs = [
            "src/cursus_dev/steps/builders",
            "src/cursus_dev/steps/configs",
            "src/cursus_dev/steps/contracts",
            "src/cursus_dev/steps/scripts",
            "src/cursus_dev/steps/specs",
            "src/cursus_dev/registry",
            "test/unit",
            "test/integration",
            "validation_reports",
            "examples",
            "docs"
        ]
        
        for dir_path in expected_dirs:
            full_path = self.workspace_path / dir_path
            self.assertTrue(full_path.exists(), f"Directory {dir_path} should exist")
        
        # Check __init__.py files were created in Python packages
        python_dirs = [
            "src/cursus_dev/steps/builders",
            "src/cursus_dev/steps/configs",
            "src/cursus_dev/steps/contracts",
            "src/cursus_dev/steps/scripts",
            "src/cursus_dev/steps/specs",
            "src/cursus_dev/registry"
        ]
        
        for dir_path in python_dirs:
            init_file = self.workspace_path / dir_path / "__init__.py"
            self.assertTrue(init_file.exists(), f"__init__.py should exist in {dir_path}")

class TestWorkspaceDocumentation(unittest.TestCase):
    """Test workspace documentation creation."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_path = Path(self.temp_dir) / "doc_test"
        self.workspace_path.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_workspace_documentation(self):
        """Test creating workspace documentation."""
        registry_file = "src/cursus_dev/registry/workspace_registry.py"
        
        readme_file = create_workspace_documentation(
            self.workspace_path,
            "doc_dev",
            registry_file
        )
        
        # Check README was created
        self.assertTrue(readme_file.exists())
        self.assertEqual(readme_file.name, "README.md")
        
        # Check content
        content = readme_file.read_text()
        self.assertIn("doc_dev", content)
        self.assertIn("Directory Structure", content)
        self.assertIn("Quick Start", content)
        self.assertIn("CLI Commands", content)
        self.assertIn("Best Practices", content)
        self.assertIn(registry_file, content)

class TestExampleImplementations(unittest.TestCase):
    """Test example implementation creation."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_path = Path(self.temp_dir) / "example_test"
        self.workspace_path.mkdir(exist_ok=True)
        # Create examples directory
        (self.workspace_path / "examples").mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_example_implementations(self):
        """Test creating example implementations."""
        create_example_implementations(self.workspace_path, "example_dev")
        
        examples_dir = self.workspace_path / "examples"
        
        # Check example files were created
        config_file = examples_dir / "example_custom_step_config.py"
        builder_file = examples_dir / "example_custom_step_builder.py"
        
        self.assertTrue(config_file.exists())
        self.assertTrue(builder_file.exists())
        
        # Check content
        config_content = config_file.read_text()
        self.assertIn("example_dev", config_content)
        self.assertIn("ExampleCustomStepConfig", config_content)
        self.assertIn("BasePipelineConfig", config_content)
        
        builder_content = builder_file.read_text()
        self.assertIn("example_dev", builder_content)
        self.assertIn("ExampleCustomStepBuilder", builder_content)
        self.assertIn("StepBuilderBase", builder_content)

class TestWorkspaceValidation(unittest.TestCase):
    """Test workspace validation functions."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_path = Path(self.temp_dir) / "validation_test"
        self.workspace_path.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validate_workspace_setup_success(self):
        """Test successful workspace validation."""
        # Create required structure
        create_workspace_structure(str(self.workspace_path))
        create_workspace_registry(str(self.workspace_path), "valid_dev", "standard")
        
        # Should not raise exception
        validate_workspace_setup(str(self.workspace_path), "valid_dev")
    
    def test_validate_workspace_setup_missing_directory(self):
        """Test workspace validation with missing directory."""
        # Create partial structure (missing some directories)
        (self.workspace_path / "src/cursus_dev/registry").mkdir(parents=True, exist_ok=True)
        
        with self.assertRaises(ValueError) as exc_info:
            validate_workspace_setup(str(self.workspace_path), "invalid_dev")
        
        self.assertIn("Required directory missing", str(exc_info.exception))
    
    def test_validate_workspace_setup_missing_registry(self):
        """Test workspace validation with missing registry file."""
        # Create directory structure but no registry file
        create_workspace_structure(str(self.workspace_path))
        
        with self.assertRaises(ValueError) as exc_info:
            validate_workspace_setup(str(self.workspace_path), "no_registry_dev")
        
        self.assertIn("Registry file not created", str(exc_info.exception))
    
    def test_validate_workspace_setup_invalid_registry(self):
        """Test workspace validation with invalid registry file."""
        # Create structure and invalid registry
        create_workspace_structure(str(self.workspace_path))
        registry_file = self.workspace_path / "src/cursus_dev/registry/workspace_registry.py"
        registry_file.write_text("# Invalid registry content")
        
        with self.assertRaises(ValueError) as exc_info:
            validate_workspace_setup(str(self.workspace_path), "invalid_registry_dev")
        
        self.assertIn("Registry file missing required sections", str(exc_info.exception))

class TestRegistryCopying(unittest.TestCase):
    """Test registry copying functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_path = Path(self.temp_dir) / "copy_test"
        self.workspace_path.mkdir(exist_ok=True)
        
        # Create source registry structure
        self.source_path = Path(self.temp_dir) / "developer_workspaces/developers/source_dev/src/cursus_dev/registry"
        self.source_path.mkdir(parents=True, exist_ok=True)
        
        # Create source registry file
        self.source_registry = self.source_path / "workspace_registry.py"
        source_content = '''"""Registry for source_dev."""
LOCAL_STEPS = {
    "SourceStep": {
        "config_class": "SourceStepConfig",
        "description": "Step from source_dev"
    }
}

WORKSPACE_METADATA = {
    "developer_id": "source_dev",
    "version": "1.0.0"
}
'''
        self.source_registry.write_text(source_content)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_copy_registry_from_developer_success(self):
        """Test successful registry copying."""
        # Create the actual source directory structure that the function expects
        source_dev_path = Path(self.temp_dir) / "developer_workspaces" / "developers" / "source_dev" / "src" / "cursus_dev" / "registry"
        source_dev_path.mkdir(parents=True, exist_ok=True)
        
        # Create the source registry file
        source_registry_file = source_dev_path / "workspace_registry.py"
        source_content = '''"""Source registry."""
LOCAL_STEPS = {"TestStep": {"config_class": "TestConfig"}}
STEP_OVERRIDES = {}
WORKSPACE_METADATA = {"developer_id": "source_dev", "version": "1.0.0"}
'''
        source_registry_file.write_text(source_content)
        
        # Change to the temp directory so the relative path works
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            
            result = copy_registry_from_developer(
                str(self.workspace_path),
                "target_dev",
                "source_dev"
            )
            
            # Should return target path
            self.assertIn("workspace_registry.py", result)
            self.assertTrue(result.endswith("workspace_registry.py"))
            
            # Check that the target file was created and has correct content
            target_file = Path(result)
            self.assertTrue(target_file.exists())
            
            target_content = target_file.read_text()
            self.assertIn("target_dev", target_content)
            self.assertNotIn("source_dev", target_content)  # Should be replaced
            
        finally:
            os.chdir(original_cwd)
    
    def test_copy_registry_from_developer_source_not_found(self):
        """Test registry copying with non-existent source."""
        with self.assertRaises(ValueError) as exc_info:
            copy_registry_from_developer(
                str(self.workspace_path),
                "target_dev",
                "nonexistent_dev"
            )
        
        self.assertIn("has no registry file", str(exc_info.exception))
    
    @patch('cursus.registry.hybrid.setup.Path')
    def test_copy_registry_from_developer_read_error(self, mock_path):
        """Test registry copying with read error."""
        # Mock source exists but reading fails
        mock_source_path = mock_path.return_value
        mock_source_path.exists.return_value = True
        
        with patch('builtins.open', side_effect=IOError("Read error")):
            with self.assertRaises(ValueError) as exc_info:
                copy_registry_from_developer(
                    str(self.workspace_path),
                    "target_dev",
                    "source_dev"
                )
            
            self.assertIn("Failed to read source registry", str(exc_info.exception))

if __name__ == "__main__":
    unittest.main()
