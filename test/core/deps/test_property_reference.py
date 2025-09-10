import pytest
from unittest.mock import Mock, MagicMock
import re

from cursus.core.deps import (
    PropertyReference,
    OutputSpec,
    DependencyType,
)

# SageMaker property path patterns for reference:
# - Training: properties.ModelArtifacts.S3ModelArtifacts
# - Processing: properties.ProcessingOutputConfig.Outputs['output_name'].S3Output.S3Uri
# - Transform: properties.TransformOutput.S3OutputPath
# - Evaluation: properties.DescribeEvaluationJobResponse.Statistics['metric_name']

class TestPropertyReference:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up common test fixtures."""
        # Create SageMaker output specs for testing
        
        # Processing output (most complex structure with dictionary access)
        self.processing_output_spec = OutputSpec(
            logical_name="processed_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
            data_type="S3Uri"
        )
        
        # Training output (simple nested properties)
        self.training_output_spec = OutputSpec(
            logical_name="model_artifacts",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts",
            data_type="S3Uri"
        )
        
        # Evaluation metrics output (dictionary access with metrics)
        self.metrics_output_spec = OutputSpec(
            logical_name="evaluation_metrics",
            output_type=DependencyType.CUSTOM_PROPERTY,
            property_path="properties.DescribeEvaluationJobResponse.Statistics['accuracy']",
            data_type="Float"
        )
        
        # Transform output with array indexing
        self.transform_output_spec = OutputSpec(
            logical_name="batch_predictions",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.TransformOutput.Outputs[0].S3OutputPath",
            data_type="S3Uri"
        )

    def test_init_and_validation(self):
        """Test initialization and validation."""
        # Test valid initialization
        prop_ref = PropertyReference(
            step_name="processing_step",
            output_spec=self.processing_output_spec
        )
        assert prop_ref.step_name == "processing_step"
        assert prop_ref.output_spec == self.processing_output_spec
        
        # Test with whitespace in step_name (should be stripped)
        prop_ref = PropertyReference(
            step_name="  processing_step  ",
            output_spec=self.processing_output_spec
        )
        assert prop_ref.step_name == "processing_step"
        
        # Test empty step_name (should raise ValueError)
        with pytest.raises(ValueError):
            PropertyReference(
                step_name="",
                output_spec=self.processing_output_spec
            )
            
        # Test whitespace-only step_name (should raise ValueError)
        with pytest.raises(ValueError):
            PropertyReference(
                step_name="   ",
                output_spec=self.processing_output_spec
            )
    
    def test_parse_property_path(self):
        """Test parsing of SageMaker property path formats."""
        prop_ref = PropertyReference(
            step_name="test_step",
            output_spec=self.processing_output_spec
        )
        
        # Test SageMaker training output path (simple nested structure)
        path_parts = prop_ref._parse_property_path("properties.ModelArtifacts.S3ModelArtifacts")
        assert len(path_parts) == 2
        assert path_parts[0] == "ModelArtifacts"
        assert path_parts[1] == "S3ModelArtifacts"
        
        # Test SageMaker processing output path (with dictionary access)
        path_parts = prop_ref._parse_property_path(
            "properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri"
        )
        assert len(path_parts) == 4
        assert path_parts[0] == "ProcessingOutputConfig"
        assert isinstance(path_parts[1], tuple)
        assert path_parts[1][0] == "Outputs"
        assert path_parts[1][1] == "processed_data"
        assert path_parts[2] == "S3Output"
        assert path_parts[3] == "S3Uri"
        
        # Test SageMaker batch transform output (with array indexing)
        path_parts = prop_ref._parse_property_path("properties.TransformOutput.Outputs[0].S3OutputPath")
        assert len(path_parts) == 3
        assert path_parts[0] == "TransformOutput"
        assert isinstance(path_parts[1], tuple)
        assert path_parts[1][0] == "Outputs"
        assert path_parts[1][1] == 0  # Should be converted to integer
        assert path_parts[2] == "S3OutputPath"
        
        # Test SageMaker evaluation metrics (with dictionary access)
        path_parts = prop_ref._parse_property_path("properties.DescribeEvaluationJobResponse.Statistics['accuracy']")
        assert len(path_parts) == 2
        assert path_parts[0] == "DescribeEvaluationJobResponse"
        assert isinstance(path_parts[1], tuple)
        assert path_parts[1][0] == "Statistics"
        assert path_parts[1][1] == "accuracy"
        
        # Test complex nested path with multiple dictionary and array accesses
        # Create path manually for testing since the parsing implementation varies
        # The original string is: "properties.Config.Outputs['data'].Sub[0].Value"
        path = "properties.Config.Outputs['data'].Sub[0].Value"
        parts = prop_ref._parse_property_path(path)
        
        # Validate expected structure without enforcing specific parsing implementation
        # The key point is that we can properly navigate this path structure
        
        # Check that we're extracting Config
        assert "Config" in parts or any(p[0] == "Config" if isinstance(p, tuple) else False for p in parts)
        
        # Check that we're extracting Outputs with key 'data'
        outputs_found = False
        for p in parts:
            if isinstance(p, tuple) and p[0] == "Outputs" and p[1] == "data":
                outputs_found = True
                break
        assert outputs_found, "Outputs['data'] not found in parsed parts"
        
        # Check that we're extracting Sub
        assert "Sub" in parts or any(p[0] == "Sub" if isinstance(p, tuple) else False for p in parts)
        
        # Check that we're extracting array index 0
        array_index_found = False
        for p in parts:
            if isinstance(p, tuple) and p[1] == 0:
                array_index_found = True
                break
        assert array_index_found, "Array index [0] not found in parsed parts"
        
        # Check that we're extracting Value
        assert "Value" in parts or any(p[0] == "Value" if isinstance(p, tuple) else False for p in parts)
    
    def test_to_sagemaker_property(self):
        """Test conversion to SageMaker property format."""
        # Test training output
        prop_ref = PropertyReference(
            step_name="training_step",
            output_spec=self.training_output_spec
        )
        sagemaker_prop = prop_ref.to_sagemaker_property()
        # The implementation removes the "properties." prefix
        assert sagemaker_prop == {"Get": "Steps.training_step.ModelArtifacts.S3ModelArtifacts"}
        
        # Test processing output
        prop_ref = PropertyReference(
            step_name="processing_step",
            output_spec=self.processing_output_spec
        )
        sagemaker_prop = prop_ref.to_sagemaker_property()
        assert sagemaker_prop == {"Get": "Steps.processing_step.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri"}
        
        # Test metrics output
        prop_ref = PropertyReference(
            step_name="eval_step",
            output_spec=self.metrics_output_spec
        )
        sagemaker_prop = prop_ref.to_sagemaker_property()
        assert sagemaker_prop == {"Get": "Steps.eval_step.DescribeEvaluationJobResponse.Statistics['accuracy']"}
        
        # Test transform output with array index
        prop_ref = PropertyReference(
            step_name="transform_step",
            output_spec=self.transform_output_spec
        )
        sagemaker_prop = prop_ref.to_sagemaker_property()
        assert sagemaker_prop == {"Get": "Steps.transform_step.TransformOutput.Outputs[0].S3OutputPath"}
    
    def test_get_property_value(self):
        """Test navigation through nested SageMaker property objects."""
        prop_ref = PropertyReference(
            step_name="test_step",
            output_spec=self.processing_output_spec
        )
        
        # Create a test object with nested structure
        test_obj = type('TestObject', (), {})()
        test_obj.ProcessingOutputConfig = type('ProcessingConfig', (), {})()
        test_obj.ProcessingOutputConfig.Outputs = {}
        test_obj.ProcessingOutputConfig.Outputs['processed_data'] = type('Output', (), {})()
        test_obj.ProcessingOutputConfig.Outputs['processed_data'].S3Output = type('S3Output', (), {})()
        test_obj.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri = "s3://test-bucket/output"
        
        # Test navigation with attribute and dictionary access
        path_parts = prop_ref._parse_property_path(prop_ref.output_spec.property_path)
        value = prop_ref._get_property_value(test_obj, path_parts)
        assert value == "s3://test-bucket/output"
        
        # Test with array index access
        test_obj_with_array = type('TestObject', (), {})()
        test_obj_with_array.Outputs = ["first", "second", "third"]
        
        path_parts = [("Outputs", 1)]  # Access second element
        value = prop_ref._get_property_value(test_obj_with_array, path_parts)
        assert value == "second"
        
        # Test error when attribute doesn't exist
        path_parts = ["NonExistentAttribute"]
        with pytest.raises(AttributeError):
            prop_ref._get_property_value(test_obj, path_parts)
        
        # Test error when dictionary key doesn't exist
        path_parts = [(None, "missing_key")]
        with pytest.raises(KeyError):
            prop_ref._get_property_value({}, path_parts)
        
        # Test error with invalid path part format
        invalid_path_part = object()  # Something that's neither a string nor a tuple
        path_parts = ["ProcessingOutputConfig", invalid_path_part]  # Use an attribute that exists
        with pytest.raises(ValueError):
            prop_ref._get_property_value(test_obj, path_parts)
    
    def test_to_runtime_property(self):
        """Test creation of SageMaker runtime properties."""
        # Test processing output reference
        processing_prop_ref = PropertyReference(
            step_name="processing_step",
            output_spec=self.processing_output_spec
        )
        
        # Create mock step instances with typical SageMaker property structures
        step_instances = {
            "processing_step": self._create_mock_processing_step(),
            "training_step": self._create_mock_training_step(),
            "eval_step": self._create_mock_eval_step(),
            "transform_step": self._create_mock_transform_step()
        }
        
        # Test processing output property reference
        processing_value = processing_prop_ref.to_runtime_property(step_instances)
        assert processing_value == "s3://test-bucket/processed-data"
        
        # Test training output property reference
        training_prop_ref = PropertyReference(
            step_name="training_step",
            output_spec=self.training_output_spec
        )
        training_value = training_prop_ref.to_runtime_property(step_instances)
        assert training_value == "s3://test-bucket/model"
        
        # Test eval metrics property reference
        metrics_prop_ref = PropertyReference(
            step_name="eval_step",
            output_spec=self.metrics_output_spec
        )
        metrics_value = metrics_prop_ref.to_runtime_property(step_instances)
        assert metrics_value == 0.92
        
        # Test transform output property reference
        transform_prop_ref = PropertyReference(
            step_name="transform_step",
            output_spec=self.transform_output_spec
        )
        transform_value = transform_prop_ref.to_runtime_property(step_instances)
        assert transform_value == "s3://test-bucket/predictions"
        
        # Test error when step doesn't exist
        with pytest.raises(ValueError):
            missing_prop_ref = PropertyReference(
                step_name="missing_step",
                output_spec=self.processing_output_spec
            )
            missing_prop_ref.to_runtime_property(step_instances)
    
    def test_string_representation(self):
        """Test string and repr methods."""
        prop_ref = PropertyReference(
            step_name="processing_step",
            output_spec=self.processing_output_spec
        )
        
        # Test __str__ method
        assert str(prop_ref) == "processing_step.processed_data"
        
        # Test __repr__ method
        assert repr(prop_ref) == "PropertyReference(step='processing_step', output='processed_data')"

    def _create_mock_processing_step(self):
        """Create a mock processing step instance with SageMaker property structure."""
        mock_step = Mock()
        
        # Create nested structure mimicking SageMaker processing step properties
        mock_step.properties = Mock()
        mock_step.properties.ProcessingOutputConfig = Mock()
        mock_step.properties.ProcessingOutputConfig.Outputs = {
            'processed_data': Mock()
        }
        mock_step.properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output = Mock()
        mock_step.properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri = "s3://test-bucket/processed-data"
        
        return mock_step
    
    def _create_mock_training_step(self):
        """Create a mock training step instance with SageMaker property structure."""
        mock_step = Mock()
        
        # Create nested structure mimicking SageMaker training step properties
        mock_step.properties = Mock()
        mock_step.properties.ModelArtifacts = Mock()
        mock_step.properties.ModelArtifacts.S3ModelArtifacts = "s3://test-bucket/model"
        
        return mock_step
    
    def _create_mock_eval_step(self):
        """Create a mock evaluation step instance with SageMaker property structure."""
        mock_step = Mock()
        
        # Create nested structure mimicking SageMaker evaluation step properties
        mock_step.properties = Mock()
        mock_step.properties.DescribeEvaluationJobResponse = Mock()
        mock_step.properties.DescribeEvaluationJobResponse.Statistics = {
            'accuracy': 0.92,
            'precision': 0.89,
            'recall': 0.85
        }
        
        return mock_step
    
    def _create_mock_transform_step(self):
        """Create a mock transform step instance with SageMaker property structure."""
        mock_step = Mock()
        
        # Create nested structure mimicking SageMaker transform step properties
        mock_step.properties = Mock()
        mock_step.properties.TransformOutput = Mock()
        mock_step.properties.TransformOutput.Outputs = [Mock()]
        mock_step.properties.TransformOutput.Outputs[0].S3OutputPath = "s3://test-bucket/predictions"
        
        return mock_step
