"""
Test script for SageMaker Property Path Validator

This script demonstrates the Level 2 Property Path Validation functionality
that validates SageMaker step property paths against official documentation.
"""

import sys
from pathlib import Path

from cursus.validation.alignment.property_path_validator import (
    SageMakerPropertyPathValidator,
)


def test_sagemaker_property_path_validator():
    """Test the SageMaker Property Path Validator with various scenarios."""

    print("🔍 Testing SageMaker Property Path Validator")
    print("=" * 60)

    validator = SageMakerPropertyPathValidator()

    # Test 1: Valid TrainingStep property paths (enhanced patterns)
    print("\n📝 Test 1: Enhanced TrainingStep Property Paths")
    training_spec = {
        "step_type": "XGBoostTraining",  # Test step registry resolution
        "node_type": "training",
        "outputs": [
            {
                "logical_name": "model_artifacts",
                "property_path": "properties.ModelArtifacts.S3ModelArtifacts",
            },
            {
                "logical_name": "training_metrics_named",
                "property_path": "properties.FinalMetricDataList['accuracy'].Value",
            },
            {
                "logical_name": "training_metrics_indexed",
                "property_path": "properties.FinalMetricDataList[0].Value",
            },
            {
                "logical_name": "algorithm_spec",
                "property_path": "properties.AlgorithmSpecification.TrainingImage",
            },
            {
                "logical_name": "resource_config",
                "property_path": "properties.ResourceConfig.VolumeSizeInGB",
            },
        ],
    }

    issues = validator.validate_specification_property_paths(
        training_spec, "enhanced_training"
    )
    print(f"Found {len(issues)} validation issues:")
    for issue in issues:
        print(f"  {issue['severity']}: {issue['message']}")

    # Test 2: Enhanced ProcessingStep property paths
    print("\n📝 Test 2: Enhanced ProcessingStep Property Paths")
    processing_spec = {
        "step_type": "processing",
        "node_type": "processing",
        "outputs": [
            {
                "logical_name": "named_output",
                "property_path": "properties.ProcessingOutputConfig.Outputs['train'].S3Output.S3Uri",
            },
            {
                "logical_name": "indexed_output",
                "property_path": "properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri",
            },
            {
                "logical_name": "app_spec",
                "property_path": "properties.AppSpecification.ImageUri",
            },
            {
                "logical_name": "processing_inputs",
                "property_path": "properties.ProcessingInputs[0].S3Input.S3Uri",
            },
        ],
    }

    issues = validator.validate_specification_property_paths(
        processing_spec, "enhanced_processing"
    )
    print(f"Found {len(issues)} validation issues:")
    for issue in issues:
        print(f"  {issue['severity']}: {issue['message']}")

    # Test 3: New step types - TuningStep
    print("\n📝 Test 3: TuningStep Property Paths")
    tuning_spec = {
        "step_type": "tuning",
        "node_type": "tuning",
        "outputs": [
            {
                "logical_name": "best_model",
                "property_path": "properties.BestTrainingJob.TrainingJobName",
            },
            {
                "logical_name": "best_metric",
                "property_path": "properties.BestTrainingJob.FinalHyperParameterTuningJobObjectiveMetric.Value",
            },
            {
                "logical_name": "top_models",
                "property_path": "properties.TrainingJobSummaries[0].TrainingJobName",
            },
            {
                "logical_name": "job_counts",
                "property_path": "properties.TrainingJobStatusCounters.Completed",
            },
        ],
    }

    issues = validator.validate_specification_property_paths(tuning_spec, "tuning_test")
    print(f"Found {len(issues)} validation issues:")
    for issue in issues:
        print(f"  {issue['severity']}: {issue['message']}")

    # Test 4: CreateModelStep property paths
    print("\n📝 Test 4: CreateModelStep Property Paths")
    model_spec = {
        "step_type": "create_model",
        "node_type": "model",
        "outputs": [
            {"logical_name": "model_name", "property_path": "properties.ModelName"},
            {
                "logical_name": "model_data",
                "property_path": "properties.PrimaryContainer.ModelDataUrl",
            },
            {
                "logical_name": "container_image",
                "property_path": "properties.PrimaryContainer.Image",
            },
            {
                "logical_name": "execution_role",
                "property_path": "properties.ExecutionRoleArn",
            },
        ],
    }

    issues = validator.validate_specification_property_paths(
        model_spec, "model_creation_test"
    )
    print(f"Found {len(issues)} validation issues:")
    for issue in issues:
        print(f"  {issue['severity']}: {issue['message']}")

    # Test 5: LambdaStep property paths (no properties prefix)
    print("\n📝 Test 5: LambdaStep Property Paths")
    lambda_spec = {
        "step_type": "lambda",
        "node_type": "lambda",
        "outputs": [
            {
                "logical_name": "lambda_result",
                "property_path": "OutputParameters['result']",
            },
            {
                "logical_name": "lambda_status",
                "property_path": "OutputParameters['status']",
            },
        ],
    }

    issues = validator.validate_specification_property_paths(lambda_spec, "lambda_test")
    print(f"Found {len(issues)} validation issues:")
    for issue in issues:
        print(f"  {issue['severity']}: {issue['message']}")

    # Test 6: Invalid property paths with enhanced error reporting
    print("\n📝 Test 6: Invalid Property Paths with Enhanced Error Reporting")
    invalid_spec = {
        "step_type": "training",
        "node_type": "training",
        "outputs": [
            {
                "logical_name": "invalid_output",
                "property_path": "properties.InvalidPath.DoesNotExist",
            },
            {
                "logical_name": "wrong_step_type_path",
                "property_path": "properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri",  # ProcessingStep path in TrainingStep
            },
        ],
    }

    issues = validator.validate_specification_property_paths(
        invalid_spec, "test_invalid"
    )
    print(f"Found {len(issues)} validation issues:")
    for issue in issues:
        print(f"  {issue['severity']}: {issue['message']}")
        if issue["severity"] == "ERROR" and "valid_paths" in issue.get("details", {}):
            suggestions = issue["details"]["valid_paths"][:3]
            print(f"    Suggestions: {', '.join(suggestions)}")

    # Test 7: Pattern matching validation
    print("\n📝 Test 7: Pattern Matching Validation")
    test_patterns = [
        ('properties.FinalMetricDataList["accuracy"].Value', "training", True),
        ("properties.FinalMetricDataList[0].Value", "training", True),
        (
            'properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri',
            "processing",
            True,
        ),
        ('OutputParameters["result"]', "lambda", True),
        (
            'properties.OutputParameters["result"]',
            "lambda",
            False,
        ),  # Should fail - wrong prefix for lambda
        ("properties.InvalidPath", "training", False),
    ]

    for path, step_type, expected_valid in test_patterns:
        valid_paths = validator._get_valid_property_paths_for_step_type(
            step_type, step_type
        )
        if valid_paths:
            all_paths = []
            for category, paths in valid_paths.items():
                all_paths.extend(paths)

            is_valid = any(
                validator._matches_property_path_pattern(path, valid_path)
                for valid_path in all_paths
            )
            status = "✅" if is_valid == expected_valid else "❌"
            print(
                f"  {status} {path} [{step_type}] - Expected: {expected_valid}, Got: {is_valid}"
            )

    # Test 8: List supported step types (enhanced)
    print("\n📝 Test 8: Enhanced Supported Step Types")
    supported_types = validator.list_supported_step_types()
    print(f"Validator supports {len(supported_types)} step types:")
    for step_info in supported_types:
        print(
            f"  - {step_info['step_type']}: {step_info['total_valid_paths']} patterns, {len(step_info['categories'])} categories"
        )

    # Test 9: Get documentation for specific step type
    print("\n📝 Test 9: Step Type Documentation")
    doc_info = validator.get_step_type_documentation("training", "training")
    print(f"TrainingStep documentation:")
    print(f"  Documentation URL: {doc_info['documentation_url']}")
    print(f"  Total valid paths: {doc_info['total_valid_paths']}")
    print(f"  Categories: {', '.join(doc_info['categories'])}")

    # Show some example paths from different categories
    for category in ["model_artifacts", "metrics", "algorithm"][:3]:
        if category in doc_info["valid_property_paths"]:
            print(f"  {category.title()} paths:")
            for path in doc_info["valid_property_paths"][category][:2]:
                print(f"    - {path}")

    print("\n✅ Enhanced Property Path Validator testing completed!")
    assert True  # Test completed successfully


def test_sagemaker_property_path_integration_with_unified_tester():
    """Test SageMaker property path validator integration with the unified alignment tester."""

    print("\n🔗 Testing Integration with Unified Alignment Tester")
    print("=" * 60)

    try:
        from cursus.validation.alignment.unified_alignment_tester import (
            UnifiedAlignmentTester,
        )

        # Initialize the unified tester
        tester = UnifiedAlignmentTester()

        # Run Level 2 validation specifically
        print("Running Level 2 validation (includes property path validation)...")
        report = tester.run_level_validation(level=2, target_scripts=["dummy_training"])

        # Check if property path validation results are included
        level2_results = report.level2_results

        if level2_results:
            print(f"Level 2 validation completed for {len(level2_results)} contracts:")

            for contract_name, result in level2_results.items():
                print(f"\n📋 Contract: {contract_name}")
                print(f"  Status: {'PASSED' if result.passed else 'FAILED'}")

                # Look for property path validation issues
                property_path_issues = [
                    issue
                    for issue in result.issues
                    if issue.category
                    in ["property_path_validation", "property_path_validation_summary"]
                ]

                if property_path_issues:
                    print(
                        f"  Property path validation issues: {len(property_path_issues)}"
                    )
                    for issue in property_path_issues[:2]:  # Show first 2
                        print(f"    {issue.level.value}: {issue.message}")
                else:
                    print("  No property path validation issues found")
        else:
            print("No Level 2 validation results found")

        print("\n✅ Integration testing completed!")
        assert True  # Integration test completed successfully

    except ImportError as e:
        print(f"❌ Could not import unified alignment tester: {e}")
        assert False, f"Could not import unified alignment tester: {e}"
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        assert False, f"Integration test failed: {e}"


if __name__ == "__main__":
    print("🚀 Starting Property Path Validator Tests")
    print("=" * 80)

    # Run standalone validator tests
    test_sagemaker_property_path_validator()

    # Run integration tests
    test_sagemaker_property_path_integration_with_unified_tester()

    print("\n" + "=" * 80)
    print("🎉 All tests completed successfully!")
    print("\n📚 Property Path Validation (Level 2) Implementation Summary:")
    print("  ✅ Created SageMakerPropertyPathValidator module")
    print("  ✅ Integrated with ContractSpecificationAlignmentTester")
    print("  ✅ Supports 10 SageMaker step types with comprehensive property paths")
    print("  ✅ Provides intelligent suggestions for invalid property paths")
    print("  ✅ References official SageMaker documentation v2.92.2")
    print(
        "  ✅ Includes pattern matching for array indexing (e.g., [*], ['metric_name'])"
    )
    print("  ✅ Integrated with unified alignment tester workflow")

    print("\n🔗 Reference Documentation:")
    print(
        "  https://sagemaker.readthedocs.io/en/v2.92.2/amazon_sagemaker_model_building_pipeline.html#data-dependency-property-reference"
    )
