"""
Debug test for builder argument detection issue.

This test specifically debugs why the builder argument detection is failing
for the tabular_preprocess script and job_type argument.
"""

import unittest
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.cursus.validation.alignment.static_analysis.builder_analyzer import (
    BuilderRegistry, BuilderArgumentExtractor, extract_builder_arguments
)


class TestBuilderArgumentDebug(unittest.TestCase):
    """Debug test for builder argument detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.builders_dir = Path(self.temp_dir) / "builders"
        self.builders_dir.mkdir()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_builder_registry_mapping_debug(self):
        """Debug the BuilderRegistry mapping for tabular_preprocess."""
        print("\n🔍 DEBUG: Testing BuilderRegistry mapping...")
        
        # Create a mock tabular preprocessing builder file
        builder_content = '''"""
Tabular Preprocessing Step Builder
"""

from typing import Dict, Optional, Any, List
from pathlib import Path

from ..configs.config_tabular_preprocessing_step import TabularPreprocessingConfig
from ...core.base.builder_base import StepBuilderBase


class TabularPreprocessingStepBuilder(StepBuilderBase):
    """Builder for a Tabular Preprocessing ProcessingStep."""

    def __init__(self, config: TabularPreprocessingConfig, **kwargs):
        super().__init__(config=config, **kwargs)
        self.config: TabularPreprocessingConfig = config

    def _get_job_arguments(self) -> List[str]:
        """
        Constructs the list of command-line arguments to be passed to the processing script.
        """
        job_type = self.config.job_type
        return ["--job_type", job_type]

    def create_step(self, **kwargs):
        pass
'''
        
        builder_path = self.builders_dir / "builder_tabular_preprocessing_step.py"
        builder_path.write_text(builder_content)
        
        # Test BuilderRegistry
        registry = BuilderRegistry(str(self.builders_dir))
        mappings = registry.get_all_mappings()
        
        print(f"📋 All mappings found: {mappings}")
        
        # Check if tabular_preprocess is mapped
        builder_file = registry.get_builder_for_script("tabular_preprocess")
        print(f"🎯 Builder for 'tabular_preprocess': {builder_file}")
        
        # Check if tabular_preprocessing is mapped
        builder_file_alt = registry.get_builder_for_script("tabular_preprocessing")
        print(f"🎯 Builder for 'tabular_preprocessing': {builder_file_alt}")
        
        # Test name variations
        variations = registry._generate_name_variations("tabular_preprocessing")
        print(f"🔄 Name variations for 'tabular_preprocessing': {variations}")
        
        # Assertions
        self.assertIn("tabular_preprocess", mappings, 
                     f"Expected 'tabular_preprocess' in mappings, but got: {list(mappings.keys())}")
        self.assertIsNotNone(builder_file, 
                           "Expected to find builder for 'tabular_preprocess'")
        self.assertIn("tabular_preprocess", variations,
                     f"Expected 'tabular_preprocess' in variations, but got: {variations}")
    
    def test_builder_argument_extractor_debug(self):
        """Debug the BuilderArgumentExtractor for job_type."""
        print("\n🔍 DEBUG: Testing BuilderArgumentExtractor...")
        
        # Create a mock tabular preprocessing builder file
        builder_content = '''"""
Tabular Preprocessing Step Builder
"""

from typing import List

class TabularPreprocessingStepBuilder:
    """Builder for a Tabular Preprocessing ProcessingStep."""

    def _get_job_arguments(self) -> List[str]:
        """
        Constructs the list of command-line arguments to be passed to the processing script.
        """
        job_type = self.config.job_type
        return ["--job_type", job_type]
'''
        
        builder_path = self.builders_dir / "builder_tabular_preprocessing_step.py"
        builder_path.write_text(builder_content)
        
        # Test BuilderArgumentExtractor
        extractor = BuilderArgumentExtractor(str(builder_path))
        
        # Debug: Get method source
        method_source = extractor.get_method_source()
        print(f"📄 Method source found:\n{method_source}")
        
        # Extract arguments
        arguments = extractor.extract_job_arguments()
        print(f"🎯 Arguments extracted: {arguments}")
        
        # Assertions
        self.assertIsNotNone(method_source, "Expected to find _get_job_arguments method")
        self.assertIn("job_type", arguments, 
                     f"Expected 'job_type' in arguments, but got: {arguments}")
    
    def test_full_integration_debug(self):
        """Debug the full integration: extract_builder_arguments function."""
        print("\n🔍 DEBUG: Testing full integration...")
        
        # Create a mock tabular preprocessing builder file
        builder_content = '''"""
Tabular Preprocessing Step Builder
"""

from typing import List

class TabularPreprocessingStepBuilder:
    """Builder for a Tabular Preprocessing ProcessingStep."""

    def _get_job_arguments(self) -> List[str]:
        """
        Constructs the list of command-line arguments to be passed to the processing script.
        """
        job_type = self.config.job_type
        return ["--job_type", job_type]
'''
        
        builder_path = self.builders_dir / "builder_tabular_preprocessing_step.py"
        builder_path.write_text(builder_content)
        
        # Test the full integration function
        arguments = extract_builder_arguments("tabular_preprocess", str(self.builders_dir))
        print(f"🎯 Final result - Arguments for 'tabular_preprocess': {arguments}")
        
        # Also test with the exact builder name
        arguments_alt = extract_builder_arguments("tabular_preprocessing", str(self.builders_dir))
        print(f"🎯 Final result - Arguments for 'tabular_preprocessing': {arguments_alt}")
        
        # Assertions
        self.assertIn("job_type", arguments, 
                     f"Expected 'job_type' in arguments for 'tabular_preprocess', but got: {arguments}")
    
    def test_real_builder_file_debug(self):
        """Debug using the actual builder file from the project."""
        print("\n🔍 DEBUG: Testing with real builder file...")
        
        # Use the actual builders directory
        real_builders_dir = "src/cursus/steps/builders"
        
        # Test BuilderRegistry with real directory
        registry = BuilderRegistry(real_builders_dir)
        mappings = registry.get_all_mappings()
        
        print(f"📋 Real mappings found: {list(mappings.keys())}")
        
        # Check specific mapping
        builder_file = registry.get_builder_for_script("tabular_preprocess")
        print(f"🎯 Real builder for 'tabular_preprocess': {builder_file}")
        
        if builder_file:
            # Test argument extraction from real file
            try:
                extractor = BuilderArgumentExtractor(builder_file)
                method_source = extractor.get_method_source()
                print(f"📄 Real method source:\n{method_source}")
                
                arguments = extractor.extract_job_arguments()
                print(f"🎯 Real arguments extracted: {arguments}")
                
                # Test full integration
                full_args = extract_builder_arguments("tabular_preprocess", real_builders_dir)
                print(f"🎯 Full integration result: {full_args}")
                
            except Exception as e:
                print(f"❌ Error extracting from real builder: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("❌ No builder file found for 'tabular_preprocess'")
            
            # List all builder files for debugging
            builders_path = Path(real_builders_dir)
            if builders_path.exists():
                builder_files = list(builders_path.glob("builder_*.py"))
                print(f"📁 Available builder files: {[f.name for f in builder_files]}")
    
    def test_name_variation_logic_debug(self):
        """Debug the name variation generation logic specifically."""
        print("\n🔍 DEBUG: Testing name variation logic...")
        
        registry = BuilderRegistry(str(self.builders_dir))  # Empty dir is fine for this test
        
        # Test various name patterns
        test_cases = [
            "tabular_preprocessing",
            "tabular_preprocess", 
            "model_evaluation",
            "model_eval",
            "xgboost_training",
            "xgb_training"
        ]
        
        for name in test_cases:
            variations = registry._generate_name_variations(name)
            print(f"🔄 '{name}' → {variations}")
        
        # Specific test for our case
        variations = registry._generate_name_variations("tabular_preprocessing")
        self.assertIn("tabular_preprocess", variations,
                     f"Expected 'tabular_preprocess' in variations for 'tabular_preprocessing', got: {variations}")
    
    def test_script_name_extraction_debug(self):
        """Debug the script name extraction from builder filename."""
        print("\n🔍 DEBUG: Testing script name extraction...")
        
        # Create a mock builder file
        builder_path = self.builders_dir / "builder_tabular_preprocessing_step.py"
        builder_path.write_text("# Mock builder")
        
        registry = BuilderRegistry(str(self.builders_dir))
        
        # Test the extraction method directly
        script_names = registry._extract_script_names_from_builder(builder_path)
        print(f"📝 Script names extracted from 'builder_tabular_preprocessing_step.py': {script_names}")
        
        # Should include both the base name and variations
        expected_names = ["tabular_preprocessing", "tabular_preprocess"]
        for expected in expected_names:
            self.assertIn(expected, script_names,
                         f"Expected '{expected}' in extracted names: {script_names}")


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
