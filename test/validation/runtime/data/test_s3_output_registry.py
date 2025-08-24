"""Unit tests for S3OutputPathRegistry and related classes."""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import patch, MagicMock

from src.cursus.validation.runtime.data.s3_output_registry import (
    S3OutputInfo,
    ExecutionMetadata,
    S3OutputPathRegistry
)


class TestS3OutputInfo:
    """Test cases for S3OutputInfo model"""
    
    def test_s3_output_info_xgboost_training_model(self):
        """Test S3OutputInfo creation for XGBoost training model output"""
        output_info = S3OutputInfo(
            logical_name="model_output",
            s3_uri="s3://sagemaker-us-west-2-123456789012/xgboost-training-2023-12-01-10-30-45-123/output/model.tar.gz",
            property_path="Steps.XGBoostTraining.ModelArtifacts.S3ModelArtifacts",
            data_type="ModelArtifacts",
            step_name="XGBoostTraining",
            job_type="training",
            metadata={
                "container_path": "/opt/ml/model",
                "artifacts": ["xgboost_model.bst", "risk_table_map.pkl", "impute_dict.pkl", "feature_importance.json"]
            }
        )
        
        assert output_info.logical_name == "model_output"
        assert output_info.s3_uri == "s3://sagemaker-us-west-2-123456789012/xgboost-training-2023-12-01-10-30-45-123/output/model.tar.gz"
        assert output_info.property_path == "Steps.XGBoostTraining.ModelArtifacts.S3ModelArtifacts"
        assert output_info.data_type == "ModelArtifacts"
        assert output_info.step_name == "XGBoostTraining"
        assert output_info.job_type == "training"
        assert isinstance(output_info.timestamp, datetime)
        assert "artifacts" in output_info.metadata
    
    def test_s3_output_info_xgboost_training_evaluation(self):
        """Test S3OutputInfo creation for XGBoost training evaluation output"""
        output_info = S3OutputInfo(
            logical_name="evaluation_output",
            s3_uri="s3://sagemaker-us-west-2-123456789012/xgboost-training-2023-12-01-10-30-45-123/output/data",
            property_path="Steps.XGBoostTraining.ProcessingOutputConfig.Outputs.evaluation_output.S3Output.S3Uri",
            data_type="S3Uri",
            step_name="XGBoostTraining",
            job_type="training",
            metadata={
                "container_path": "/opt/ml/output/data",
                "evaluation_files": ["val.tar.gz", "test.tar.gz"]
            }
        )
        
        assert output_info.logical_name == "evaluation_output"
        assert output_info.job_type == "training"
        assert output_info.metadata["evaluation_files"] == ["val.tar.gz", "test.tar.gz"]
    
    def test_s3_output_info_package_step(self):
        """Test S3OutputInfo creation for Package step output"""
        output_info = S3OutputInfo(
            logical_name="packaged_model",
            s3_uri="s3://sagemaker-us-west-2-123456789012/package-step-2023-12-01-11-15-30-456/output/model.tar.gz",
            property_path="Steps.Package.ProcessingOutputConfig.Outputs.packaged_model.S3Output.S3Uri",
            data_type="S3Uri",
            step_name="Package",
            metadata={
                "container_path": "/opt/ml/processing/output",
                "depends_on": "XGBoostTraining",
                "input_model": "s3://sagemaker-us-west-2-123456789012/xgboost-training-2023-12-01-10-30-45-123/output/model.tar.gz"
            }
        )
        
        assert output_info.logical_name == "packaged_model"
        assert output_info.s3_uri == "s3://sagemaker-us-west-2-123456789012/package-step-2023-12-01-11-15-30-456/output/model.tar.gz"
        assert output_info.property_path == "Steps.Package.ProcessingOutputConfig.Outputs.packaged_model.S3Output.S3Uri"
        assert output_info.data_type == "S3Uri"
        assert output_info.step_name == "Package"
        assert output_info.job_type is None
        assert output_info.metadata["depends_on"] == "XGBoostTraining"
    
    def test_s3_output_info_timestamp_default(self):
        """Test that timestamp is automatically set"""
        before_creation = datetime.now()
        
        output_info = S3OutputInfo(
            logical_name="test_output",
            s3_uri="s3://bucket/test",
            property_path="Steps.Test.Output",
            data_type="S3Uri",
            step_name="test-step"
        )
        
        after_creation = datetime.now()
        
        assert before_creation <= output_info.timestamp <= after_creation
    
    def test_s3_output_info_serialization(self):
        """Test S3OutputInfo serialization to dict"""
        output_info = S3OutputInfo(
            logical_name="test_output",
            s3_uri="s3://bucket/test",
            property_path="Steps.Test.Output",
            data_type="S3Uri",
            step_name="test-step",
            job_type="testing"
        )
        
        data = output_info.model_dump()
        
        assert data["logical_name"] == "test_output"
        assert data["s3_uri"] == "s3://bucket/test"
        assert data["job_type"] == "testing"
        assert "timestamp" in data


class TestExecutionMetadata:
    """Test cases for ExecutionMetadata model"""
    
    def test_execution_metadata_creation(self):
        """Test basic ExecutionMetadata creation"""
        metadata = ExecutionMetadata()
        
        assert metadata.pipeline_name is None
        assert metadata.execution_id is None
        assert isinstance(metadata.start_time, datetime)
        assert metadata.end_time is None
        assert metadata.total_steps == 0
        assert metadata.completed_steps == 0
    
    def test_execution_metadata_with_values(self):
        """Test ExecutionMetadata creation with values"""
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=1)
        
        metadata = ExecutionMetadata(
            pipeline_name="test-pipeline",
            execution_id="exec-123",
            start_time=start_time,
            end_time=end_time,
            total_steps=5,
            completed_steps=3
        )
        
        assert metadata.pipeline_name == "test-pipeline"
        assert metadata.execution_id == "exec-123"
        assert metadata.start_time == start_time
        assert metadata.end_time == end_time
        assert metadata.total_steps == 5
        assert metadata.completed_steps == 3
    
    def test_mark_step_completed(self):
        """Test marking steps as completed"""
        metadata = ExecutionMetadata(total_steps=3, completed_steps=1)
        
        assert metadata.completed_steps == 1
        
        metadata.mark_step_completed()
        assert metadata.completed_steps == 2
        
        metadata.mark_step_completed()
        assert metadata.completed_steps == 3
    
    def test_is_complete(self):
        """Test completion status checking"""
        metadata = ExecutionMetadata(total_steps=3, completed_steps=2)
        
        assert not metadata.is_complete()
        
        metadata.mark_step_completed()
        assert metadata.is_complete()
        
        # Test edge case where completed > total
        metadata.mark_step_completed()
        assert metadata.is_complete()
    
    def test_is_complete_zero_steps(self):
        """Test completion status with zero total steps"""
        metadata = ExecutionMetadata(total_steps=0, completed_steps=0)
        assert metadata.is_complete()


class TestS3OutputPathRegistry:
    """Test cases for S3OutputPathRegistry"""
    
    def test_registry_creation(self):
        """Test basic registry creation"""
        registry = S3OutputPathRegistry()
        
        assert registry.step_outputs == {}
        assert isinstance(registry.execution_metadata, ExecutionMetadata)
    
    def test_register_step_output(self):
        """Test registering XGBoost training step output"""
        registry = S3OutputPathRegistry()
        
        output_info = S3OutputInfo(
            logical_name="model_output",
            s3_uri="s3://sagemaker-us-west-2-123456789012/xgboost-training-2023-12-01-10-30-45-123/output/model.tar.gz",
            property_path="Steps.XGBoostTraining.ModelArtifacts.S3ModelArtifacts",
            data_type="ModelArtifacts",
            step_name="XGBoostTraining"
        )
        
        registry.register_step_output("XGBoostTraining", "model_output", output_info)
        
        assert "XGBoostTraining" in registry.step_outputs
        assert "model_output" in registry.step_outputs["XGBoostTraining"]
        assert registry.step_outputs["XGBoostTraining"]["model_output"] == output_info
        assert registry.execution_metadata.completed_steps == 1
    
    def test_register_multiple_outputs_same_step(self):
        """Test registering multiple outputs for the same step"""
        registry = S3OutputPathRegistry()
        
        output1 = S3OutputInfo(
            logical_name="model_artifacts",
            s3_uri="s3://bucket/model.tar.gz",
            property_path="Steps.Training.ModelArtifacts",
            data_type="ModelArtifacts",
            step_name="training-step"
        )
        
        output2 = S3OutputInfo(
            logical_name="training_metrics",
            s3_uri="s3://bucket/metrics.json",
            property_path="Steps.Training.ProcessingOutputConfig.Outputs.metrics",
            data_type="S3Uri",
            step_name="training-step"
        )
        
        registry.register_step_output("training-step", "model_artifacts", output1)
        registry.register_step_output("training-step", "training_metrics", output2)
        
        assert len(registry.step_outputs["training-step"]) == 2
        assert registry.execution_metadata.completed_steps == 2
    
    def test_get_step_output_info(self):
        """Test retrieving step output info"""
        registry = S3OutputPathRegistry()
        
        output_info = S3OutputInfo(
            logical_name="model_artifacts",
            s3_uri="s3://bucket/model.tar.gz",
            property_path="Steps.Training.ModelArtifacts",
            data_type="ModelArtifacts",
            step_name="training-step"
        )
        
        registry.register_step_output("training-step", "model_artifacts", output_info)
        
        retrieved_info = registry.get_step_output_info("training-step", "model_artifacts")
        assert retrieved_info == output_info
        
        # Test non-existent step
        assert registry.get_step_output_info("non-existent", "model_artifacts") is None
        
        # Test non-existent output
        assert registry.get_step_output_info("training-step", "non-existent") is None
    
    def test_get_step_output_path(self):
        """Test retrieving step output S3 path"""
        registry = S3OutputPathRegistry()
        
        output_info = S3OutputInfo(
            logical_name="model_artifacts",
            s3_uri="s3://bucket/model.tar.gz",
            property_path="Steps.Training.ModelArtifacts",
            data_type="ModelArtifacts",
            step_name="training-step"
        )
        
        registry.register_step_output("training-step", "model_artifacts", output_info)
        
        s3_path = registry.get_step_output_path("training-step", "model_artifacts")
        assert s3_path == "s3://bucket/model.tar.gz"
        
        # Test non-existent output
        assert registry.get_step_output_path("training-step", "non-existent") is None
    
    def test_get_all_step_outputs(self):
        """Test retrieving all outputs for a step"""
        registry = S3OutputPathRegistry()
        
        output1 = S3OutputInfo(
            logical_name="model_artifacts",
            s3_uri="s3://bucket/model.tar.gz",
            property_path="Steps.Training.ModelArtifacts",
            data_type="ModelArtifacts",
            step_name="training-step"
        )
        
        output2 = S3OutputInfo(
            logical_name="training_metrics",
            s3_uri="s3://bucket/metrics.json",
            property_path="Steps.Training.ProcessingOutputConfig.Outputs.metrics",
            data_type="S3Uri",
            step_name="training-step"
        )
        
        registry.register_step_output("training-step", "model_artifacts", output1)
        registry.register_step_output("training-step", "training_metrics", output2)
        
        all_outputs = registry.get_all_step_outputs("training-step")
        assert len(all_outputs) == 2
        assert all_outputs["model_artifacts"] == output1
        assert all_outputs["training_metrics"] == output2
        
        # Test non-existent step
        assert registry.get_all_step_outputs("non-existent") == {}
    
    def test_list_all_steps(self):
        """Test listing all steps with outputs"""
        registry = S3OutputPathRegistry()
        
        output1 = S3OutputInfo(
            logical_name="model_artifacts",
            s3_uri="s3://bucket/model.tar.gz",
            property_path="Steps.Training.ModelArtifacts",
            data_type="ModelArtifacts",
            step_name="training-step"
        )
        
        output2 = S3OutputInfo(
            logical_name="validation_results",
            s3_uri="s3://bucket/validation.json",
            property_path="Steps.Validation.Output",
            data_type="S3Uri",
            step_name="validation-step"
        )
        
        registry.register_step_output("training-step", "model_artifacts", output1)
        registry.register_step_output("validation-step", "validation_results", output2)
        
        steps = registry.list_all_steps()
        assert len(steps) == 2
        assert "training-step" in steps
        assert "validation-step" in steps
    
    def test_get_outputs_by_data_type(self):
        """Test filtering outputs by data type"""
        registry = S3OutputPathRegistry()
        
        output1 = S3OutputInfo(
            logical_name="model_artifacts",
            s3_uri="s3://bucket/model.tar.gz",
            property_path="Steps.Training.ModelArtifacts",
            data_type="ModelArtifacts",
            step_name="training-step"
        )
        
        output2 = S3OutputInfo(
            logical_name="training_metrics",
            s3_uri="s3://bucket/metrics.json",
            property_path="Steps.Training.Output",
            data_type="S3Uri",
            step_name="training-step"
        )
        
        output3 = S3OutputInfo(
            logical_name="validation_results",
            s3_uri="s3://bucket/validation.json",
            property_path="Steps.Validation.Output",
            data_type="S3Uri",
            step_name="validation-step"
        )
        
        registry.register_step_output("training-step", "model_artifacts", output1)
        registry.register_step_output("training-step", "training_metrics", output2)
        registry.register_step_output("validation-step", "validation_results", output3)
        
        s3_uri_outputs = registry.get_outputs_by_data_type("S3Uri")
        assert len(s3_uri_outputs) == 2
        assert output2 in s3_uri_outputs
        assert output3 in s3_uri_outputs
        
        model_artifacts_outputs = registry.get_outputs_by_data_type("ModelArtifacts")
        assert len(model_artifacts_outputs) == 1
        assert output1 in model_artifacts_outputs
        
        # Test non-existent data type
        assert registry.get_outputs_by_data_type("NonExistent") == []
    
    def test_get_outputs_by_job_type(self):
        """Test filtering outputs by job type"""
        registry = S3OutputPathRegistry()
        
        output1 = S3OutputInfo(
            logical_name="training_model",
            s3_uri="s3://bucket/training_model.tar.gz",
            property_path="Steps.Training.ModelArtifacts",
            data_type="ModelArtifacts",
            step_name="training-step",
            job_type="training"
        )
        
        output2 = S3OutputInfo(
            logical_name="validation_results",
            s3_uri="s3://bucket/validation.json",
            property_path="Steps.Validation.Output",
            data_type="S3Uri",
            step_name="validation-step",
            job_type="validation"
        )
        
        output3 = S3OutputInfo(
            logical_name="test_results",
            s3_uri="s3://bucket/test.json",
            property_path="Steps.Test.Output",
            data_type="S3Uri",
            step_name="test-step",
            job_type="validation"
        )
        
        registry.register_step_output("training-step", "training_model", output1)
        registry.register_step_output("validation-step", "validation_results", output2)
        registry.register_step_output("test-step", "test_results", output3)
        
        validation_outputs = registry.get_outputs_by_job_type("validation")
        assert len(validation_outputs) == 2
        assert output2 in validation_outputs
        assert output3 in validation_outputs
        
        training_outputs = registry.get_outputs_by_job_type("training")
        assert len(training_outputs) == 1
        assert output1 in training_outputs
        
        # Test non-existent job type
        assert registry.get_outputs_by_job_type("non-existent") == []
    
    def test_resolve_property_path(self):
        """Test resolving property paths to S3 URIs"""
        registry = S3OutputPathRegistry()
        
        output1 = S3OutputInfo(
            logical_name="model_artifacts",
            s3_uri="s3://bucket/model.tar.gz",
            property_path="Steps.Training.ModelArtifacts.S3ModelArtifacts",
            data_type="ModelArtifacts",
            step_name="training-step"
        )
        
        output2 = S3OutputInfo(
            logical_name="validation_results",
            s3_uri="s3://bucket/validation.json",
            property_path="Steps.Validation.ProcessingOutputConfig.Outputs.validation",
            data_type="S3Uri",
            step_name="validation-step"
        )
        
        registry.register_step_output("training-step", "model_artifacts", output1)
        registry.register_step_output("validation-step", "validation_results", output2)
        
        # Test successful resolution
        s3_uri = registry.resolve_property_path("Steps.Training.ModelArtifacts.S3ModelArtifacts")
        assert s3_uri == "s3://bucket/model.tar.gz"
        
        s3_uri = registry.resolve_property_path("Steps.Validation.ProcessingOutputConfig.Outputs.validation")
        assert s3_uri == "s3://bucket/validation.json"
        
        # Test non-existent property path
        assert registry.resolve_property_path("Steps.NonExistent.Output") is None
    
    def test_create_registry_summary(self):
        """Test creating registry summary"""
        registry = S3OutputPathRegistry()
        registry.execution_metadata.pipeline_name = "test-pipeline"
        registry.execution_metadata.execution_id = "exec-123"
        
        output1 = S3OutputInfo(
            logical_name="model_artifacts",
            s3_uri="s3://bucket/model.tar.gz",
            property_path="Steps.Training.ModelArtifacts",
            data_type="ModelArtifacts",
            step_name="training-step",
            job_type="training"
        )
        
        output2 = S3OutputInfo(
            logical_name="validation_results",
            s3_uri="s3://bucket/validation.json",
            property_path="Steps.Validation.Output",
            data_type="S3Uri",
            step_name="validation-step",
            job_type="validation"
        )
        
        registry.register_step_output("training-step", "model_artifacts", output1)
        registry.register_step_output("validation-step", "validation_results", output2)
        
        summary = registry.create_registry_summary()
        
        assert summary["total_steps"] == 2
        assert summary["total_outputs"] == 2
        assert "ModelArtifacts" in summary["data_types"]
        assert "S3Uri" in summary["data_types"]
        assert "training" in summary["job_types"]
        assert "validation" in summary["job_types"]
        assert "execution_metadata" in summary
        assert "registry_created" in summary
    
    def test_export_to_dict(self):
        """Test exporting registry to dictionary"""
        registry = S3OutputPathRegistry()
        
        output_info = S3OutputInfo(
            logical_name="model_artifacts",
            s3_uri="s3://bucket/model.tar.gz",
            property_path="Steps.Training.ModelArtifacts",
            data_type="ModelArtifacts",
            step_name="training-step"
        )
        
        registry.register_step_output("training-step", "model_artifacts", output_info)
        
        data = registry.export_to_dict()
        
        assert "step_outputs" in data
        assert "execution_metadata" in data
        assert "training-step" in data["step_outputs"]
        assert "model_artifacts" in data["step_outputs"]["training-step"]
    
    def test_from_dict(self):
        """Test creating registry from dictionary"""
        data = {
            "step_outputs": {
                "training-step": {
                    "model_artifacts": {
                        "logical_name": "model_artifacts",
                        "s3_uri": "s3://bucket/model.tar.gz",
                        "property_path": "Steps.Training.ModelArtifacts",
                        "data_type": "ModelArtifacts",
                        "step_name": "training-step",
                        "job_type": None,
                        "timestamp": "2023-01-01T12:00:00",
                        "metadata": {}
                    }
                }
            },
            "execution_metadata": {
                "pipeline_name": "test-pipeline",
                "execution_id": "exec-123",
                "start_time": "2023-01-01T10:00:00",
                "end_time": None,
                "total_steps": 1,
                "completed_steps": 1
            }
        }
        
        registry = S3OutputPathRegistry.from_dict(data)
        
        assert "training-step" in registry.step_outputs
        assert "model_artifacts" in registry.step_outputs["training-step"]
        assert registry.execution_metadata.pipeline_name == "test-pipeline"
        assert registry.execution_metadata.execution_id == "exec-123"
    
    def test_merge_registry(self):
        """Test merging registries"""
        registry1 = S3OutputPathRegistry()
        registry2 = S3OutputPathRegistry()
        
        output1 = S3OutputInfo(
            logical_name="model_artifacts",
            s3_uri="s3://bucket/model.tar.gz",
            property_path="Steps.Training.ModelArtifacts",
            data_type="ModelArtifacts",
            step_name="training-step"
        )
        
        output2 = S3OutputInfo(
            logical_name="validation_results",
            s3_uri="s3://bucket/validation.json",
            property_path="Steps.Validation.Output",
            data_type="S3Uri",
            step_name="validation-step"
        )
        
        output3 = S3OutputInfo(
            logical_name="training_metrics",
            s3_uri="s3://bucket/metrics.json",
            property_path="Steps.Training.Metrics",
            data_type="S3Uri",
            step_name="training-step"
        )
        
        registry1.register_step_output("training-step", "model_artifacts", output1)
        registry2.register_step_output("validation-step", "validation_results", output2)
        registry2.register_step_output("training-step", "training_metrics", output3)
        
        # Set different execution metadata
        registry1.execution_metadata.total_steps = 2
        registry1.execution_metadata.completed_steps = 1
        registry2.execution_metadata.total_steps = 3
        registry2.execution_metadata.completed_steps = 2
        
        registry1.merge_registry(registry2)
        
        # Check merged outputs
        assert len(registry1.step_outputs) == 2
        assert "training-step" in registry1.step_outputs
        assert "validation-step" in registry1.step_outputs
        assert len(registry1.step_outputs["training-step"]) == 2
        assert "model_artifacts" in registry1.step_outputs["training-step"]
        assert "training_metrics" in registry1.step_outputs["training-step"]
        
        # Check merged execution metadata
        assert registry1.execution_metadata.total_steps == 3
        assert registry1.execution_metadata.completed_steps == 2
    
    def test_merge_registry_no_overwrite(self):
        """Test that merge doesn't overwrite existing outputs"""
        registry1 = S3OutputPathRegistry()
        registry2 = S3OutputPathRegistry()
        
        output1 = S3OutputInfo(
            logical_name="model_artifacts",
            s3_uri="s3://bucket/original_model.tar.gz",
            property_path="Steps.Training.ModelArtifacts",
            data_type="ModelArtifacts",
            step_name="training-step"
        )
        
        output2 = S3OutputInfo(
            logical_name="model_artifacts",
            s3_uri="s3://bucket/new_model.tar.gz",
            property_path="Steps.Training.ModelArtifacts",
            data_type="ModelArtifacts",
            step_name="training-step"
        )
        
        registry1.register_step_output("training-step", "model_artifacts", output1)
        registry2.register_step_output("training-step", "model_artifacts", output2)
        
        registry1.merge_registry(registry2)
        
        # Should keep original output, not overwrite
        merged_output = registry1.get_step_output_info("training-step", "model_artifacts")
        assert merged_output.s3_uri == "s3://bucket/original_model.tar.gz"
    
    def test_timestamp_explicit_setting(self):
        """Test that timestamps can be set explicitly"""
        fixed_time = datetime(2023, 1, 1, 12, 0, 0)
        
        output_info = S3OutputInfo(
            logical_name="test_output",
            s3_uri="s3://bucket/test",
            property_path="Steps.Test.Output",
            data_type="S3Uri",
            step_name="test-step",
            timestamp=fixed_time
        )
        
        assert output_info.timestamp == fixed_time
    
    def test_xgboost_training_to_package_dependency(self):
        """Test realistic XGBoost Training -> Package step dependency scenario"""
        registry = S3OutputPathRegistry()
        registry.execution_metadata.pipeline_name = "xgboost-training-pipeline"
        registry.execution_metadata.execution_id = "pipeline-2023-12-01-10-00-00-789"
        registry.execution_metadata.total_steps = 2
        
        # Register XGBoost Training step outputs
        xgboost_model_output = S3OutputInfo(
            logical_name="model_output",
            s3_uri="s3://sagemaker-us-west-2-123456789012/xgboost-training-2023-12-01-10-30-45-123/output/model.tar.gz",
            property_path="Steps.XGBoostTraining.ModelArtifacts.S3ModelArtifacts",
            data_type="ModelArtifacts",
            step_name="XGBoostTraining",
            job_type="training",
            metadata={
                "container_path": "/opt/ml/model",
                "artifacts": ["xgboost_model.bst", "risk_table_map.pkl", "impute_dict.pkl"]
            }
        )
        
        xgboost_eval_output = S3OutputInfo(
            logical_name="evaluation_output",
            s3_uri="s3://sagemaker-us-west-2-123456789012/xgboost-training-2023-12-01-10-30-45-123/output/data",
            property_path="Steps.XGBoostTraining.ProcessingOutputConfig.Outputs.evaluation_output.S3Output.S3Uri",
            data_type="S3Uri",
            step_name="XGBoostTraining",
            job_type="training",
            metadata={
                "container_path": "/opt/ml/output/data",
                "evaluation_files": ["val.tar.gz", "test.tar.gz"]
            }
        )
        
        registry.register_step_output("XGBoostTraining", "model_output", xgboost_model_output)
        registry.register_step_output("XGBoostTraining", "evaluation_output", xgboost_eval_output)
        
        # Register Package step output that depends on XGBoost Training
        package_output = S3OutputInfo(
            logical_name="packaged_model",
            s3_uri="s3://sagemaker-us-west-2-123456789012/package-step-2023-12-01-11-15-30-456/output/model.tar.gz",
            property_path="Steps.Package.ProcessingOutputConfig.Outputs.packaged_model.S3Output.S3Uri",
            data_type="S3Uri",
            step_name="Package",
            metadata={
                "container_path": "/opt/ml/processing/output",
                "depends_on": "XGBoostTraining",
                "input_model_path": "Steps.XGBoostTraining.ModelArtifacts.S3ModelArtifacts",
                "input_model_uri": "s3://sagemaker-us-west-2-123456789012/xgboost-training-2023-12-01-10-30-45-123/output/model.tar.gz"
            }
        )
        
        registry.register_step_output("Package", "packaged_model", package_output)
        
        # Verify the dependency relationship
        assert len(registry.step_outputs) == 2
        assert "XGBoostTraining" in registry.step_outputs
        assert "Package" in registry.step_outputs
        
        # Verify XGBoost Training outputs
        xgb_outputs = registry.get_all_step_outputs("XGBoostTraining")
        assert len(xgb_outputs) == 2
        assert "model_output" in xgb_outputs
        assert "evaluation_output" in xgb_outputs
        
        # Verify Package output references XGBoost Training
        package_info = registry.get_step_output_info("Package", "packaged_model")
        assert package_info.metadata["depends_on"] == "XGBoostTraining"
        assert package_info.metadata["input_model_path"] == "Steps.XGBoostTraining.ModelArtifacts.S3ModelArtifacts"
        
        # Test property path resolution for dependency
        resolved_model_uri = registry.resolve_property_path("Steps.XGBoostTraining.ModelArtifacts.S3ModelArtifacts")
        assert resolved_model_uri == "s3://sagemaker-us-west-2-123456789012/xgboost-training-2023-12-01-10-30-45-123/output/model.tar.gz"
        
        # Verify execution metadata
        assert registry.execution_metadata.completed_steps == 3  # 2 XGBoost outputs + 1 Package output
        assert registry.execution_metadata.is_complete()  # completed > total is considered complete
        
        # Test filtering by job type
        training_outputs = registry.get_outputs_by_job_type("training")
        assert len(training_outputs) == 2  # Both XGBoost outputs have job_type="training"
        
        # Test filtering by data type
        model_artifacts = registry.get_outputs_by_data_type("ModelArtifacts")
        assert len(model_artifacts) == 1
        assert model_artifacts[0].step_name == "XGBoostTraining"
        
        s3_uri_outputs = registry.get_outputs_by_data_type("S3Uri")
        assert len(s3_uri_outputs) == 2  # evaluation_output and packaged_model
        
        # Test registry summary
        summary = registry.create_registry_summary()
        assert summary["total_steps"] == 2
        assert summary["total_outputs"] == 3
        assert "ModelArtifacts" in summary["data_types"]
        assert "S3Uri" in summary["data_types"]
        assert "training" in summary["job_types"]
    
    def test_empty_registry_operations(self):
        """Test operations on empty registry"""
        registry = S3OutputPathRegistry()
        
        assert registry.list_all_steps() == []
        assert registry.get_outputs_by_data_type("S3Uri") == []
        assert registry.get_outputs_by_job_type("training") == []
        assert registry.resolve_property_path("Steps.Test.Output") is None
        
        summary = registry.create_registry_summary()
        assert summary["total_steps"] == 0
        assert summary["total_outputs"] == 0
        assert summary["data_types"] == []
        assert summary["job_types"] == []
