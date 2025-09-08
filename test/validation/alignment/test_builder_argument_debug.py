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
