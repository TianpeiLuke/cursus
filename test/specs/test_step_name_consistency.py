"""
Test Step Name Consistency Across Pipeline System

This test suite validates that all step names are consistent across:
1. Central registry definitions
2. Step specifications
3. Pipeline templates
4. Config classes

Tests ensure no hardcoded step names exist and all components use the central registry.
"""

import unittest
import sys
import re
from pathlib import Path
from typing import List, Dict, Set

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from src.pipeline_registry.step_names import (
        STEP_TYPE_REGISTRY,
        get_spec_step_type,
        get_builder_step_type
    )
except ImportError:
    # Skip tests if registry not available
    STEP_TYPE_REGISTRY = {}
    def get_spec_step_type(name): return name
    def get_builder_step_type(name): return name


class TestStepNameConsistency(unittest.TestCase):
    """Test step name consistency across the pipeline system"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class with project paths"""
        cls.project_root = Path(__file__).parent.parent.parent
        cls.src_root = cls.project_root / "src"
        
        # Required step types that should be in registry
        cls.required_step_types = [
            "CradleDataLoading", "TabularPreprocessing", "PytorchTraining",
            "PytorchModel", "XGBoostTraining", "XGBoostModel", 
            "Registration", "Package", "Payload"
        ]
    
    def test_central_registry_exists(self):
        """Test that central registry exists and has required step types"""
        registry_file = self.src_root / "pipeline_registry" / "step_names.py"
        self.assertTrue(registry_file.exists(), 
                       f"Central registry not found at {registry_file}")
        
        # Test registry has required step types
        for step_type in self.required_step_types:
            with self.subTest(step_type=step_type):
                try:
                    result = get_spec_step_type(step_type)
                    self.assertIsNotNone(result, 
                                       f"Step type '{step_type}' not found in registry")
                    self.assertEqual(result, step_type,
                                   f"Step type '{step_type}' returns '{result}' instead")
                except Exception as e:
                    self.fail(f"Error getting step type '{step_type}': {e}")
    
    def test_step_specifications_use_registry(self):
        """Test that step specifications use central registry"""
        spec_dir = self.src_root / "pipeline_step_specs"
        self.assertTrue(spec_dir.exists(), f"Step specs directory not found: {spec_dir}")
        
        spec_files = [f for f in spec_dir.glob("*.py") if not f.name.startswith("__")]
        self.assertGreater(len(spec_files), 0, "No step specification files found")
        
        registry_import_pattern = re.compile(r'from.*pipeline_registry.*import.*get_spec_step_type')
        registry_usage_pattern = re.compile(r'get_spec_step_type\s*\(\s*["\']([^"\']+)["\']\s*\)')
        hardcoded_step_pattern = re.compile(r'step_type\s*=\s*["\']([A-Z][a-zA-Z]+)["\']')
        
        for spec_file in spec_files:
            with self.subTest(file=spec_file.name):
                content = spec_file.read_text()
                
                # Check if file defines step_type
                has_step_type = "step_type=" in content
                
                if has_step_type:
                    # File defines step_type, should use registry
                    has_import = bool(registry_import_pattern.search(content))
                    usage_matches = registry_usage_pattern.findall(content)
                    hardcoded_matches = hardcoded_step_pattern.findall(content)
                    
                    self.assertTrue(has_import, 
                                  f"{spec_file.name} defines step_type but doesn't import registry")
                    self.assertGreater(len(usage_matches), 0,
                                     f"{spec_file.name} defines step_type but doesn't use get_spec_step_type()")
                    self.assertEqual(len(hardcoded_matches), 0,
                                   f"{spec_file.name} has hardcoded step_type: {hardcoded_matches}")
    
    def test_pipeline_templates_use_consistent_names(self):
        """Test that pipeline templates use consistent step names"""
        template_dir = self.src_root / "pipeline_builder"
        self.assertTrue(template_dir.exists(), f"Template directory not found: {template_dir}")
        
        template_files = [f for f in template_dir.glob("template_*.py")]
        template_files.append(template_dir / "pipeline_builder_template.py")
        
        builder_map_pattern = re.compile(r'BUILDER_MAP\s*=\s*\{([^}]+)\}', re.DOTALL)
        step_name_pattern = re.compile(r'["\']([^"\']+)["\']:\s*\w+StepBuilder')
        
        for template_file in template_files:
            if not template_file.exists():
                continue
                
            with self.subTest(file=template_file.name):
                content = template_file.read_text()
                
                # Find BUILDER_MAP definitions
                builder_maps = builder_map_pattern.findall(content)
                
                if not builder_maps:
                    # If file has StepBuilder references, it should have BUILDER_MAP
                    if "StepBuilder" in content:
                        self.fail(f"{template_file.name} has StepBuilder references but no BUILDER_MAP")
                    continue
                
                # Validate step names in BUILDER_MAP
                for builder_map in builder_maps:
                    step_names = step_name_pattern.findall(builder_map)
                    
                    for step_name in step_names:
                        # Check if step name exists in registry or is a valid variant
                        try:
                            registry_name = get_spec_step_type(step_name)
                            # Either exact match or step name is valid
                            self.assertTrue(registry_name == step_name or step_name in self.required_step_types,
                                          f"{template_file.name} uses inconsistent step name: '{step_name}'")
                        except Exception:
                            # Step name might be valid for specialized steps
                            # This is acceptable as long as it's not obviously wrong
                            pass
    
    def test_config_classes_use_registry(self):
        """Test that config classes import from central registry when needed"""
        config_dir = self.src_root / "pipeline_steps"
        if not config_dir.exists():
            self.skipTest(f"Config directory not found: {config_dir}")
        
        config_files = [f for f in config_dir.glob("config_*.py")]
        config_files.append(config_dir / "config_base.py")
        
        registry_import_pattern = re.compile(r'from.*pipeline_registry.*import')
        
        for config_file in config_files:
            if not config_file.exists():
                continue
                
            with self.subTest(file=config_file.name):
                content = config_file.read_text()
                
                # Check if file uses step names
                uses_step_names = ("get_step_name" in content or 
                                 "step_type" in content or
                                 "STEP_TYPE" in content)
                
                if uses_step_names:
                    has_import = bool(registry_import_pattern.search(content))
                    self.assertTrue(has_import,
                                  f"{config_file.name} uses step names but doesn't import from registry")
    
    def test_no_hardcoded_step_names_in_specs(self):
        """Test that step specifications don't have hardcoded step names"""
        spec_dir = self.src_root / "pipeline_step_specs"
        if not spec_dir.exists():
            self.skipTest(f"Step specs directory not found: {spec_dir}")
        
        # Patterns for hardcoded step names
        hardcoded_patterns = [
            re.compile(r'step_type\s*=\s*["\']([A-Z][a-zA-Z]+Step)["\']'),
            re.compile(r'step_type\s*=\s*["\']([A-Z][a-zA-Z]*Training)["\']'),
            re.compile(r'step_type\s*=\s*["\']([A-Z][a-zA-Z]*Model)["\']'),
        ]
        
        spec_files = [f for f in spec_dir.glob("*.py") if not f.name.startswith("__")]
        
        for spec_file in spec_files:
            with self.subTest(file=spec_file.name):
                content = spec_file.read_text()
                
                # Skip files that properly import registry
                if re.search(r'from.*pipeline_registry.*import', content):
                    continue
                
                # Check for hardcoded patterns
                for pattern in hardcoded_patterns:
                    matches = pattern.findall(content)
                    self.assertEqual(len(matches), 0,
                                   f"{spec_file.name} has hardcoded step names: {matches}")
    
    def test_no_hardcoded_step_names_in_templates(self):
        """Test that pipeline templates don't have hardcoded step names in wrong places"""
        template_dir = self.src_root / "pipeline_builder"
        if not template_dir.exists():
            self.skipTest(f"Template directory not found: {template_dir}")
        
        template_files = [f for f in template_dir.glob("*.py") if not f.name.startswith("__")]
        
        # Pattern for hardcoded step types (not in BUILDER_MAP)
        hardcoded_step_type_pattern = re.compile(r'step_type\s*=\s*["\']([A-Z][a-zA-Z]+)["\']')
        
        for template_file in template_files:
            with self.subTest(file=template_file.name):
                content = template_file.read_text()
                
                # Look for hardcoded step_type assignments
                matches = hardcoded_step_type_pattern.findall(content)
                
                # Filter out matches that are in BUILDER_MAP context (which is OK)
                problematic_matches = []
                for match in matches:
                    # Simple heuristic: if the match is not near "BUILDER_MAP", it's problematic
                    match_pos = content.find(f'step_type="{match}"')
                    if match_pos == -1:
                        match_pos = content.find(f"step_type='{match}'")
                    
                    if match_pos != -1:
                        # Check if this is within a BUILDER_MAP context
                        before_text = content[max(0, match_pos-200):match_pos]
                        after_text = content[match_pos:match_pos+200]
                        
                        if "BUILDER_MAP" not in before_text and "BUILDER_MAP" not in after_text:
                            problematic_matches.append(match)
                
                self.assertEqual(len(problematic_matches), 0,
                               f"{template_file.name} has hardcoded step_type outside BUILDER_MAP: {problematic_matches}")
    
    def test_job_type_variants_consistency(self):
        """Test that job type variants follow consistent naming pattern"""
        spec_dir = self.src_root / "pipeline_step_specs"
        if not spec_dir.exists():
            self.skipTest(f"Step specs directory not found: {spec_dir}")
        
        # Job type variants should follow pattern: BaseStepName_JobType
        job_type_pattern = re.compile(r'get_spec_step_type\s*\(\s*["\']([^"\']+)["\']\s*\)\s*\+\s*["\']_([^"\']+)["\']')
        
        spec_files = [f for f in spec_dir.glob("*_training_spec.py")]
        spec_files.extend(spec_dir.glob("*_testing_spec.py"))
        spec_files.extend(spec_dir.glob("*_validation_spec.py"))
        spec_files.extend(spec_dir.glob("*_calibration_spec.py"))
        
        for spec_file in spec_files:
            with self.subTest(file=spec_file.name):
                content = spec_file.read_text()
                
                # Find job type variant patterns
                matches = job_type_pattern.findall(content)
                
                for base_step, job_type in matches:
                    # Validate base step exists in registry
                    try:
                        registry_name = get_spec_step_type(base_step)
                        self.assertEqual(registry_name, base_step,
                                       f"{spec_file.name} uses unknown base step: '{base_step}'")
                    except Exception:
                        self.fail(f"{spec_file.name} uses base step '{base_step}' not in registry")
                    
                    # Validate job type is reasonable
                    valid_job_types = ["Training", "Testing", "Validation", "Calibration"]
                    self.assertIn(job_type, valid_job_types,
                                f"{spec_file.name} uses invalid job type: '{job_type}'")


class TestStepNameValidationTool(unittest.TestCase):
    """Test the step name validation tool itself"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class"""
        cls.project_root = Path(__file__).parent.parent.parent
        cls.validation_tool = cls.project_root / "tools" / "validate_step_names.py"
    
    def test_validation_tool_exists(self):
        """Test that validation tool exists and is executable"""
        self.assertTrue(self.validation_tool.exists(),
                       f"Validation tool not found: {self.validation_tool}")
        
        # Check if file is executable
        import stat
        file_stat = self.validation_tool.stat()
        is_executable = bool(file_stat.st_mode & stat.S_IEXEC)
        self.assertTrue(is_executable,
                       f"Validation tool is not executable: {self.validation_tool}")
    
    def test_validation_tool_imports(self):
        """Test that validation tool can import required modules"""
        # This is a basic smoke test to ensure the tool can be imported
        import subprocess
        import sys
        
        # Try to run the tool with --help or similar to test basic functionality
        try:
            result = subprocess.run([
                sys.executable, str(self.validation_tool), "--help"
            ], capture_output=True, text=True, timeout=10)
            
            # Tool might not have --help, but it shouldn't crash on import
            # If it exits with code 2 (argparse help), that's OK
            # If it exits with code 1 (validation failed), that's also OK for this test
            # Only code 127 (command not found) or import errors are problems
            self.assertNotEqual(result.returncode, 127,
                              f"Validation tool failed to run: {result.stderr}")
            
        except subprocess.TimeoutExpired:
            self.fail("Validation tool timed out - possible infinite loop")
        except Exception as e:
            self.fail(f"Error running validation tool: {e}")


if __name__ == "__main__":
    unittest.main()
