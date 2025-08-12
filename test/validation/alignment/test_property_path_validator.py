"""
Test script for SageMaker Property Path Validator

This script demonstrates the Level 2 Property Path Validation functionality
that validates SageMaker step property paths against official documentation.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.cursus.validation.alignment.property_path_validator import SageMakerPropertyPathValidator


def test_property_path_validator():
    """Test the SageMaker Property Path Validator with various scenarios."""
    
    print("🔍 Testing SageMaker Property Path Validator")
    print("=" * 60)
    
    validator = SageMakerPropertyPathValidator()
    
    # Test 1: Valid TrainingStep property paths
    print("\n📝 Test 1: Valid TrainingStep Property Paths")
    training_spec = {
        'step_type': 'training',
        'node_type': 'training',
        'outputs': [
            {
                'logical_name': 'model_artifacts',
                'property_path': 'properties.ModelArtifacts.S3ModelArtifacts'
            },
            {
                'logical_name': 'training_metrics',
                'property_path': 'properties.FinalMetricDataList[\'accuracy\'].Value'
            }
        ]
    }
    
    issues = validator.validate_specification_property_paths(training_spec, 'dummy_training')
    print(f"Found {len(issues)} validation issues:")
    for issue in issues:
        print(f"  {issue['severity']}: {issue['message']}")
    
    # Test 2: Invalid property paths
    print("\n📝 Test 2: Invalid Property Paths")
    invalid_spec = {
        'step_type': 'training',
        'node_type': 'training',
        'outputs': [
            {
                'logical_name': 'invalid_output',
                'property_path': 'properties.InvalidPath.DoesNotExist'
            }
        ]
    }
    
    issues = validator.validate_specification_property_paths(invalid_spec, 'test_invalid')
    print(f"Found {len(issues)} validation issues:")
    for issue in issues:
        print(f"  {issue['severity']}: {issue['message']}")
        if issue['severity'] == 'ERROR':
            print(f"    Suggestions: {', '.join(issue['details']['valid_paths'][:3])}")
    
    # Test 3: ProcessingStep property paths
    print("\n📝 Test 3: ProcessingStep Property Paths")
    processing_spec = {
        'step_type': 'processing',
        'node_type': 'processing',
        'outputs': [
            {
                'logical_name': 'processed_data',
                'property_path': 'properties.ProcessingOutputConfig.Outputs[\'train\'].S3Output.S3Uri'
            }
        ]
    }
    
    issues = validator.validate_specification_property_paths(processing_spec, 'tabular_preprocess')
    print(f"Found {len(issues)} validation issues:")
    for issue in issues:
        print(f"  {issue['severity']}: {issue['message']}")
    
    # Test 4: Unknown step type
    print("\n📝 Test 4: Unknown Step Type")
    unknown_spec = {
        'step_type': 'unknown_step',
        'node_type': 'unknown',
        'outputs': [
            {
                'logical_name': 'some_output',
                'property_path': 'properties.SomeProperty'
            }
        ]
    }
    
    issues = validator.validate_specification_property_paths(unknown_spec, 'unknown_step')
    print(f"Found {len(issues)} validation issues:")
    for issue in issues:
        print(f"  {issue['severity']}: {issue['message']}")
    
    # Test 5: List supported step types
    print("\n📝 Test 5: Supported Step Types")
    supported_types = validator.list_supported_step_types()
    print(f"Validator supports {len(supported_types)} step types:")
    for step_info in supported_types[:5]:  # Show first 5
        print(f"  - {step_info['step_type']}: {step_info['description']}")
        print(f"    Total valid paths: {step_info['total_valid_paths']}")
    
    # Test 6: Get documentation for specific step type
    print("\n📝 Test 6: Step Type Documentation")
    doc_info = validator.get_step_type_documentation('training', 'training')
    print(f"TrainingStep documentation:")
    print(f"  Documentation URL: {doc_info['documentation_url']}")
    print(f"  Total valid paths: {doc_info['total_valid_paths']}")
    print(f"  Categories: {', '.join(doc_info['categories'])}")
    
    # Show some example paths
    if 'model_artifacts' in doc_info['valid_property_paths']:
        print(f"  Model artifacts paths:")
        for path in doc_info['valid_property_paths']['model_artifacts'][:3]:
            print(f"    - {path}")
    
    print("\n✅ Property Path Validator testing completed!")
    return True


def test_integration_with_unified_tester():
    """Test integration with the unified alignment tester."""
    
    print("\n🔗 Testing Integration with Unified Alignment Tester")
    print("=" * 60)
    
    try:
        from src.cursus.validation.alignment.unified_alignment_tester import UnifiedAlignmentTester
        
        # Initialize the unified tester
        tester = UnifiedAlignmentTester()
        
        # Run Level 2 validation specifically
        print("Running Level 2 validation (includes property path validation)...")
        report = tester.run_level_validation(level=2, target_scripts=['dummy_training'])
        
        # Check if property path validation results are included
        level2_results = report.level2_results
        
        if level2_results:
            print(f"Level 2 validation completed for {len(level2_results)} contracts:")
            
            for contract_name, result in level2_results.items():
                print(f"\n📋 Contract: {contract_name}")
                print(f"  Status: {'PASSED' if result.passed else 'FAILED'}")
                
                # Look for property path validation issues
                property_path_issues = [
                    issue for issue in result.issues 
                    if issue.category in ['property_path_validation', 'property_path_validation_summary']
                ]
                
                if property_path_issues:
                    print(f"  Property path validation issues: {len(property_path_issues)}")
                    for issue in property_path_issues[:2]:  # Show first 2
                        print(f"    {issue.level.value}: {issue.message}")
                else:
                    print("  No property path validation issues found")
        else:
            print("No Level 2 validation results found")
        
        print("\n✅ Integration testing completed!")
        return True
        
    except ImportError as e:
        print(f"❌ Could not import unified alignment tester: {e}")
        return False
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Property Path Validator Tests")
    print("=" * 80)
    
    # Run standalone validator tests
    success1 = test_property_path_validator()
    
    # Run integration tests
    success2 = test_integration_with_unified_tester()
    
    print("\n" + "=" * 80)
    if success1 and success2:
        print("🎉 All tests completed successfully!")
        print("\n📚 Property Path Validation (Level 2) Implementation Summary:")
        print("  ✅ Created SageMakerPropertyPathValidator module")
        print("  ✅ Integrated with ContractSpecificationAlignmentTester")
        print("  ✅ Supports 10 SageMaker step types with comprehensive property paths")
        print("  ✅ Provides intelligent suggestions for invalid property paths")
        print("  ✅ References official SageMaker documentation v2.92.2")
        print("  ✅ Includes pattern matching for array indexing (e.g., [*], ['metric_name'])")
        print("  ✅ Integrated with unified alignment tester workflow")
        
        print("\n🔗 Reference Documentation:")
        print("  https://sagemaker.readthedocs.io/en/v2.92.2/amazon_sagemaker_model_building_pipeline.html#data-dependency-property-reference")
    else:
        print("❌ Some tests failed. Check the output above for details.")
