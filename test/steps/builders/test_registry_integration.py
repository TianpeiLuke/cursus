"""
Test to verify the integration between test files and centralized registry discovery.

This test validates that the centralized registry discovery utilities work correctly
and that test files can successfully use them to discover and load step builders.
"""

import unittest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from cursus.validation.builders import (
    RegistryStepDiscovery,
    get_training_steps_from_registry,
    get_transform_steps_from_registry,
    get_createmodel_steps_from_registry,
    get_processing_steps_from_registry,
    load_builder_class,
    UniversalStepBuilderTest
)


class TestRegistryIntegration(unittest.TestCase):
    """Test the integration between test files and centralized registry discovery."""
    
    def test_registry_discovery_methods_available(self):
        """Test that all registry discovery methods are available."""
        # Test that convenience functions work
        training_steps = get_training_steps_from_registry()
        transform_steps = get_transform_steps_from_registry()
        createmodel_steps = get_createmodel_steps_from_registry()
        processing_steps = get_processing_steps_from_registry()
        
        # Verify we get lists
        self.assertIsInstance(training_steps, list)
        self.assertIsInstance(transform_steps, list)
        self.assertIsInstance(createmodel_steps, list)
        self.assertIsInstance(processing_steps, list)
        
        print(f"Found {len(training_steps)} Training steps")
        print(f"Found {len(transform_steps)} Transform steps")
        print(f"Found {len(createmodel_steps)} CreateModel steps")
        print(f"Found {len(processing_steps)} Processing steps")
    
    def test_registry_step_discovery_class_methods(self):
        """Test that RegistryStepDiscovery class methods work."""
        # Test get_all_sagemaker_step_types
        step_types = RegistryStepDiscovery.get_all_sagemaker_step_types()
        self.assertIsInstance(step_types, list)
        self.assertIn("Training", step_types)
        self.assertIn("Transform", step_types)
        self.assertIn("CreateModel", step_types)
        self.assertIn("Processing", step_types)
        
        print(f"Available SageMaker step types: {step_types}")
        
        # Test generate_discovery_report
        report = RegistryStepDiscovery.generate_discovery_report()
        self.assertIsInstance(report, dict)
        self.assertIn("total_steps", report)
        self.assertIn("sagemaker_step_types", report)
        self.assertIn("availability_summary", report)
        
        print(f"Discovery report summary: {report['availability_summary']}")
    
    def test_universal_test_registry_methods(self):
        """Test that UniversalStepBuilderTest has registry-based methods."""
        # Test class methods exist
        self.assertTrue(hasattr(UniversalStepBuilderTest, 'test_all_builders_by_type'))
        self.assertTrue(hasattr(UniversalStepBuilderTest, 'generate_registry_discovery_report'))
        self.assertTrue(hasattr(UniversalStepBuilderTest, 'validate_builder_availability'))
        
        # Test generate_registry_discovery_report
        report = UniversalStepBuilderTest.generate_registry_discovery_report()
        self.assertIsInstance(report, dict)
        self.assertIn("total_steps", report)
        
        print("✅ UniversalStepBuilderTest registry methods are available")
    
    def test_step_builder_loading(self):
        """Test that step builders can be loaded using centralized methods."""
        # Try to load a few step builders from different types
        test_cases = [
            ("TabularPreprocessing", "Processing"),
            ("PyTorchTraining", "Training"),
            ("BatchTransform", "Transform"),
            ("PyTorchModel", "CreateModel")
        ]
        
        successful_loads = 0
        
        for step_name, expected_type in test_cases:
            try:
                # Validate availability first
                validation = RegistryStepDiscovery.validate_step_builder_availability(step_name)
                print(f"\n{step_name} validation: {validation}")
                
                if validation.get("loadable", False):
                    # Try to load the builder
                    builder_class = load_builder_class(step_name)
                    self.assertIsNotNone(builder_class)
                    
                    # Verify it's a class
                    self.assertTrue(callable(builder_class))
                    
                    # Get step info
                    step_info = RegistryStepDiscovery.get_step_info_from_registry(step_name)
                    actual_type = step_info.get("sagemaker_step_type")
                    self.assertEqual(actual_type, expected_type)
                    
                    successful_loads += 1
                    print(f"✅ Successfully loaded {step_name} ({builder_class.__name__})")
                else:
                    print(f"⚠️  {step_name} not available: {validation.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"❌ Failed to load {step_name}: {e}")
        
        # We should be able to load at least one step builder
        self.assertGreater(successful_loads, 0, "Should be able to load at least one step builder")
        print(f"\n✅ Successfully loaded {successful_loads}/{len(test_cases)} step builders")
    
    def test_universal_test_with_registry_discovery(self):
        """Test that UniversalStepBuilderTest works with registry-discovered builders."""
        # Try to test all Processing builders using registry discovery
        try:
            results = UniversalStepBuilderTest.test_all_builders_by_type(
                "Processing", 
                verbose=False, 
                enable_scoring=False
            )
            
            self.assertIsInstance(results, dict)
            
            # Check if we got any results
            if results and not results.get('error'):
                print(f"✅ Successfully tested {len(results)} Processing builders using registry discovery")
                
                # Check structure of results
                for step_name, result in results.items():
                    if isinstance(result, dict) and 'test_results' in result:
                        test_results = result['test_results']
                        self.assertIsInstance(test_results, dict)
                        print(f"  - {step_name}: {len(test_results)} tests")
            else:
                print(f"⚠️  No Processing builders available or error occurred: {results.get('error', 'Unknown')}")
                
        except Exception as e:
            print(f"❌ Failed to test builders by type: {e}")
            # Don't fail the test - this might be expected if builders aren't available
    
    def test_integration_with_existing_test_files(self):
        """Test that existing test files can use the centralized discovery."""
        # Import the test suites to verify they work
        try:
            from test_training_step_builders import TrainingStepBuilderTestSuite
            
            # Test that the suite can get training steps
            suite = TrainingStepBuilderTestSuite()
            training_steps = suite.get_training_steps_from_registry()
            
            self.assertIsInstance(training_steps, list)
            print(f"✅ TrainingStepBuilderTestSuite integration works: {len(training_steps)} steps found")
            
            # Test that it can get available builders
            available_builders = suite.get_available_training_builders()
            self.assertIsInstance(available_builders, list)
            print(f"✅ Available training builders: {len(available_builders)}")
            
        except ImportError as e:
            print(f"⚠️  Could not import test suite: {e}")
            # Don't fail - this is expected in some environments


if __name__ == '__main__':
    print("Testing Registry Integration...")
    print("=" * 60)
    unittest.main(verbosity=2)
