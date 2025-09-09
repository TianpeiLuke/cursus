"""
Unit tests for cursus.validation.alignment.loaders.specification_loader module.

Tests the SpecificationLoader class that handles loading and parsing of step
specification files from Python modules with robust sys.path management and
job type awareness.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import tempfile
import logging
from typing import Dict, List, Any, Optional

from cursus.validation.alignment.loaders.specification_loader import SpecificationLoader


class TestSpecificationLoader:
    """Test cases for SpecificationLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.specs_dir = Path(self.temp_dir) / "specs"
        self.specs_dir.mkdir(exist_ok=True)
        
        self.loader = SpecificationLoader(str(self.specs_dir))
        
        # Sample specification files
        self.sample_spec_files = [
            "model_training_spec.py",
            "model_training_validation_spec.py",
            "data_preprocessing_spec.py",
            "__init__.py"  # Should be ignored
        ]
        
        # Create sample spec files
        for spec_file in self.sample_spec_files:
            (self.specs_dir / spec_file).touch()
        
        # Sample StepSpecification object
        self.sample_spec_obj = Mock()
        self.sample_spec_obj.step_type = "ModelTraining"
        self.sample_spec_obj.node_type = Mock()
        self.sample_spec_obj.node_type.value = "ProcessingJob"
        
        # Mock dependencies
        mock_dep = Mock()
        mock_dep.logical_name = "training_data"
        mock_dep.dependency_type = Mock()
        mock_dep.dependency_type.value = "InputData"
        mock_dep.required = True
        mock_dep.compatible_sources = ["S3"]
        mock_dep.data_type = "tabular"
        mock_dep.description = "Training dataset"
        
        self.sample_spec_obj.dependencies = {"training_data": mock_dep}
        
        # Mock outputs
        mock_output = Mock()
        mock_output.logical_name = "model"
        mock_output.output_type = Mock()
        mock_output.output_type.value = "Model"
        mock_output.property_path = "/opt/ml/model"
        mock_output.data_type = "model"
        mock_output.description = "Trained model"
        
        self.sample_spec_obj.outputs = {"model": mock_output}
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test SpecificationLoader initialization."""
        loader = SpecificationLoader("/path/to/specs")
        
        assert loader.specs_dir == Path("/path/to/specs")
        assert loader.file_resolver is not None
    
    def test_find_specification_files_direct_matching(self):
        """Test finding specification files using direct matching."""
        spec_files = self.loader.find_specification_files("model_training")
        
        # Should find both the main spec and the validation variant
        expected_files = [
            self.specs_dir / "model_training_spec.py",
            self.specs_dir / "model_training_validation_spec.py"
        ]
        
        assert len(spec_files) == 2
        for expected_file in expected_files:
            assert expected_file in spec_files
    
    def test_find_specification_files_no_direct_match(self):
        """Test finding specification files when no direct match exists."""
        # Mock the file resolver fallback
        with patch.object(self.loader.file_resolver, 'find_spec_file') as mock_find:
            mock_find.return_value = str(self.specs_dir / "model_training_spec.py")
            
            spec_files = self.loader.find_specification_files("nonexistent_spec")
            
            assert len(spec_files) >= 1
            assert Path(mock_find.return_value) in spec_files
    
    def test_find_specification_files_no_match(self):
        """Test finding specification files when no match exists."""
        with patch.object(self.loader.file_resolver, 'find_spec_file') as mock_find:
            mock_find.return_value = None
            
            spec_files = self.loader.find_specification_files("nonexistent_spec")
            
            assert spec_files == []
    
    def test_extract_job_type_from_spec_file_default(self):
        """Test extracting job type from spec file with default pattern."""
        spec_file = Path("model_training_spec.py")
        job_type = self.loader.extract_job_type_from_spec_file(spec_file)
        
        assert job_type == "default"
    
    def test_extract_job_type_from_spec_file_with_job_type(self):
        """Test extracting job type from spec file with job type pattern."""
        spec_file = Path("model_training_validation_spec.py")
        job_type = self.loader.extract_job_type_from_spec_file(spec_file)
        
        assert job_type == "validation"
    
    def test_extract_job_type_from_spec_file_training(self):
        """Test extracting training job type from spec file."""
        spec_file = Path("data_preprocessing_training_spec.py")
        job_type = self.loader.extract_job_type_from_spec_file(spec_file)
        
        assert job_type == "training"
    
    def test_extract_job_type_from_spec_file_calibration(self):
        """Test extracting calibration job type from spec file."""
        spec_file = Path("model_evaluation_calibration_spec.py")
        job_type = self.loader.extract_job_type_from_spec_file(spec_file)
        
        assert job_type == "calibration"
    
    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_load_specification_from_python_success(self, mock_module_from_spec, mock_spec_from_file):
        """Test successful specification loading from Python file."""
        # Mock the module loading
        mock_spec = Mock()
        mock_loader = Mock()
        mock_spec.loader = mock_loader
        mock_spec_from_file.return_value = mock_spec
        
        mock_module = Mock()
        mock_module_from_spec.return_value = mock_module
        
        # Set up module with specification object
        mock_module.__dict__ = {'MODEL_TRAINING_SPEC': self.sample_spec_obj}
        
        with patch('builtins.dir', return_value=['MODEL_TRAINING_SPEC']):
            with patch.object(self.loader.file_resolver, 'find_spec_constant_name', return_value='MODEL_TRAINING_SPEC'):
                spec_path = Path("model_training_spec.py")
                result = self.loader.load_specification_from_python(spec_path, "model_training", "default")
                
                # Verify the result structure
                assert result['step_type'] == "ModelTraining"
                assert result['node_type'] == "ProcessingJob"
                assert 'dependencies' in result
                assert 'outputs' in result
    
    @patch('importlib.util.spec_from_file_location')
    def test_load_specification_from_python_no_spec(self, mock_spec_from_file):
        """Test specification loading when spec creation fails."""
        mock_spec_from_file.return_value = None
        
        spec_path = Path("test_spec.py")
        
        with pytest.raises(ValueError) as exc_info:
            self.loader.load_specification_from_python(spec_path, "test", "default")
        
        assert "Could not load specification module" in str(exc_info.value)
    
    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_load_specification_from_python_no_spec_object(self, mock_module_from_spec, mock_spec_from_file):
        """Test specification loading when no specification object is found."""
        # Mock the module loading
        mock_spec = Mock()
        mock_loader = Mock()
        mock_spec.loader = mock_loader
        mock_spec_from_file.return_value = mock_spec
        
        mock_module = Mock()
        mock_module_from_spec.return_value = mock_module
        
        # Set up module with no specification objects
        mock_module.__dict__ = {'other_attr': 'value'}
        
        with patch('builtins.dir', return_value=['other_attr']):
            with patch.object(self.loader.file_resolver, 'find_spec_constant_name', return_value=None):
                spec_path = Path("test_spec.py")
                
                with pytest.raises(ValueError) as exc_info:
                    self.loader.load_specification_from_python(spec_path, "test", "default")
                
                assert "No specification constant found" in str(exc_info.value)
    
    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_load_specification_from_python_sys_path_management(self, mock_module_from_spec, mock_spec_from_file):
        """Test that sys.path is properly managed during specification loading."""
        original_path = sys.path.copy()
        
        # Mock the module loading
        mock_spec = Mock()
        mock_loader = Mock()
        mock_spec.loader = mock_loader
        mock_spec_from_file.return_value = mock_spec
        
        mock_module = Mock()
        mock_module_from_spec.return_value = mock_module
        mock_module.__dict__ = {'TEST_SPEC': self.sample_spec_obj}
        
        with patch('builtins.dir', return_value=['TEST_SPEC']):
            with patch.object(self.loader.file_resolver, 'find_spec_constant_name', return_value='TEST_SPEC'):
                spec_path = Path("test_spec.py")
                self.loader.load_specification_from_python(spec_path, "test", "default")
                
                # Verify sys.path is restored
                assert sys.path == original_path
    
    def test_step_specification_to_dict_complete(self):
        """Test converting complete StepSpecification object to dictionary."""
        result = self.loader.step_specification_to_dict(self.sample_spec_obj)
        
        # Verify structure
        assert result['step_type'] == "ModelTraining"
        assert result['node_type'] == "ProcessingJob"
        
        # Verify dependencies conversion
        assert len(result['dependencies']) == 1
        dep = result['dependencies'][0]
        assert dep['logical_name'] == "training_data"
        assert dep['dependency_type'] == "InputData"
        assert dep['required'] is True
        assert dep['compatible_sources'] == ["S3"]
        assert dep['data_type'] == "tabular"
        assert dep['description'] == "Training dataset"
        
        # Verify outputs conversion
        assert len(result['outputs']) == 1
        output = result['outputs'][0]
        assert output['logical_name'] == "model"
        assert output['output_type'] == "Model"
        assert output['property_path'] == "/opt/ml/model"
        assert output['data_type'] == "model"
        assert output['description'] == "Trained model"
    
    def test_step_specification_to_dict_empty_collections(self):
        """Test converting StepSpecification with empty collections."""
        empty_spec = Mock()
        empty_spec.step_type = "EmptySpec"
        empty_spec.node_type = Mock()
        empty_spec.node_type.value = "ProcessingJob"
        empty_spec.dependencies = {}
        empty_spec.outputs = {}
        
        result = self.loader.step_specification_to_dict(empty_spec)
        
        assert result['step_type'] == "EmptySpec"
        assert result['dependencies'] == []
        assert result['outputs'] == []
    
    @patch('cursus.validation.alignment.loaders.specification_loader.DependencySpec')
    @patch('cursus.validation.alignment.loaders.specification_loader.OutputSpec')
    @patch('cursus.validation.alignment.loaders.specification_loader.StepSpecification')
    def test_dict_to_step_specification(self, mock_step_spec, mock_output_spec, mock_dep_spec):
        """Test converting specification dictionary back to StepSpecification object."""
        spec_dict = {
            'step_type': 'ModelTraining',
            'node_type': 'ProcessingJob',
            'dependencies': [{
                'logical_name': 'training_data',
                'dependency_type': 'InputData',
                'required': True,
                'compatible_sources': ['S3'],
                'data_type': 'tabular',
                'description': 'Training dataset'
            }],
            'outputs': [{
                'logical_name': 'model',
                'output_type': 'Model',
                'property_path': '/opt/ml/model',
                'data_type': 'model',
                'description': 'Trained model'
            }]
        }
        
        # Mock the spec objects
        mock_dep_instance = Mock()
        mock_dep_spec.return_value = mock_dep_instance
        
        mock_output_instance = Mock()
        mock_output_spec.return_value = mock_output_instance
        
        mock_step_instance = Mock()
        mock_step_spec.return_value = mock_step_instance
        
        result = self.loader.dict_to_step_specification(spec_dict)
        
        # Verify DependencySpec was created correctly
        mock_dep_spec.assert_called_once_with(
            logical_name='training_data',
            dependency_type='InputData',
            required=True,
            compatible_sources=['S3'],
            data_type='tabular',
            description='Training dataset',
            semantic_keywords=[]
        )
        
        # Verify OutputSpec was created correctly
        mock_output_spec.assert_called_once_with(
            logical_name='model',
            output_type='Model',
            property_path='/opt/ml/model',
            data_type='model',
            description='Trained model',
            aliases=[]
        )
        
        # Verify StepSpecification was created correctly
        mock_step_spec.assert_called_once()
        
        assert result == mock_step_instance
    
    def test_load_all_specifications_success(self):
        """Test loading all specifications from directory."""
        with patch.object(self.loader, 'load_specification_from_python') as mock_load:
            mock_load.return_value = {'step_type': 'TestSpec', 'node_type': 'ProcessingJob'}
            
            result = self.loader.load_all_specifications()
            
            # Should load specifications for each unique spec name
            assert 'model_training' in result
            assert 'data_preprocessing' in result
    
    def test_load_all_specifications_with_errors(self):
        """Test loading all specifications when some fail to load."""
        with patch.object(self.loader, 'load_specification_from_python') as mock_load:
            # First call succeeds, second fails
            mock_load.side_effect = [
                {'step_type': 'TestSpec', 'node_type': 'ProcessingJob'},
                Exception("Load failed")
            ]
            
            with patch('cursus.validation.alignment.loaders.specification_loader.logger') as mock_logger:
                result = self.loader.load_all_specifications()
                
                # Should still return the successful one
                assert len(result) >= 1
                # Should log the warning
                mock_logger.warning.assert_called()
    
    def test_load_all_specifications_empty_directory(self):
        """Test loading specifications from empty directory."""
        empty_dir = Path(self.temp_dir) / "empty_specs"
        empty_dir.mkdir()
        
        loader = SpecificationLoader(str(empty_dir))
        result = loader.load_all_specifications()
        
        assert result == {}
    
    def test_discover_specifications_success(self):
        """Test discovering specifications in directory."""
        specs = self.loader.discover_specifications()
        
        expected = ["data_preprocessing", "model_training"]
        assert sorted(specs) == expected
    
    def test_discover_specifications_ignores_init_files(self):
        """Test that __init__.py files are ignored during discovery."""
        specs = self.loader.discover_specifications()
        
        # Should not include __init__ in the results
        assert "__init__" not in specs
        assert "data_preprocessing" in specs
    
    def test_discover_specifications_empty_directory(self):
        """Test discovering specifications in empty directory."""
        empty_dir = Path(self.temp_dir) / "empty_specs"
        empty_dir.mkdir()
        
        loader = SpecificationLoader(str(empty_dir))
        specs = loader.discover_specifications()
        
        assert specs == []
    
    def test_find_specifications_by_contract_success(self):
        """Test finding specifications that reference a specific contract."""
        with patch.object(self.loader, 'load_specification_from_python') as mock_load:
            mock_load.return_value = {'step_type': 'ModelTraining'}
            
            with patch.object(self.loader, '_specification_references_contract', return_value=True):
                result = self.loader.find_specifications_by_contract("model_training")
                
                assert len(result) > 0
                # Should contain spec file paths as keys
                for spec_file in result.keys():
                    assert isinstance(spec_file, Path)
                    assert spec_file.suffix == '.py'
    
    def test_find_specifications_by_contract_no_match(self):
        """Test finding specifications when none reference the contract."""
        with patch.object(self.loader, 'load_specification_from_python') as mock_load:
            mock_load.return_value = {'step_type': 'DataPreprocessing'}
            
            with patch.object(self.loader, '_specification_references_contract', return_value=False):
                result = self.loader.find_specifications_by_contract("model_training")
                
                assert result == {}
    
    def test_find_specifications_by_contract_load_error(self):
        """Test finding specifications when some fail to load."""
        with patch.object(self.loader, 'load_specification_from_python') as mock_load:
            mock_load.side_effect = Exception("Load failed")
            
            result = self.loader.find_specifications_by_contract("model_training")
            
            # Should handle errors gracefully
            assert result == {}
    
    def test_specification_references_contract_direct_match(self):
        """Test specification contract reference with direct match."""
        spec_dict = {'step_type': 'model_training'}
        
        result = self.loader._specification_references_contract(spec_dict, "model_training")
        
        assert result is True
    
    def test_specification_references_contract_partial_match(self):
        """Test specification contract reference with partial match."""
        spec_dict = {'step_type': 'ModelTraining_XGBoost'}
        
        result = self.loader._specification_references_contract(spec_dict, "model_training")
        
        assert result is True
    
    def test_specification_references_contract_eval_evaluation_match(self):
        """Test specification contract reference with eval/evaluation substitution."""
        spec_dict = {'step_type': 'model_evaluation'}
        
        result = self.loader._specification_references_contract(spec_dict, "model_eval")
        
        assert result is True
    
    def test_specification_references_contract_no_match(self):
        """Test specification contract reference with no match."""
        spec_dict = {'step_type': 'data_preprocessing'}
        
        result = self.loader._specification_references_contract(spec_dict, "model_training")
        
        assert result is False
    
    def test_load_specification_with_spec_dict(self):
        """Test loading specification when spec_dict is already provided."""
        spec_info = {
            'spec_dict': {'step_type': 'TestSpec', 'node_type': 'ProcessingJob'}
        }
        
        result = self.loader.load_specification(Path("test_spec.py"), spec_info)
        
        assert result == spec_info['spec_dict']
    
    def test_load_specification_without_spec_dict(self):
        """Test loading specification when spec_dict is not provided."""
        spec_info = {
            'spec_name': 'test_spec',
            'job_type': 'training'
        }
        
        with patch.object(self.loader, 'load_specification_from_python') as mock_load:
            mock_load.return_value = {'step_type': 'TestSpec'}
            
            spec_file = Path("test_spec.py")
            result = self.loader.load_specification(spec_file, spec_info)
            
            mock_load.assert_called_once_with(spec_file, 'test_spec', 'training')
            assert result == {'step_type': 'TestSpec'}


class TestSpecificationLoaderIntegration:
    """Integration test cases for SpecificationLoader."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.specs_dir = Path(self.temp_dir) / "specs"
        self.specs_dir.mkdir(exist_ok=True)
        
        self.loader = SpecificationLoader(str(self.specs_dir))
    
    def teardown_method(self):
        """Clean up integration test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_specification_workflow(self):
        """Test complete specification loading workflow with real file operations."""
        # Create specification files
        spec_files = [
            "model_training_spec.py",
            "model_training_validation_spec.py",
            "data_preprocessing_spec.py"
        ]
        
        for spec_file in spec_files:
            (self.specs_dir / spec_file).touch()
        
        # Test discovery
        discovered_specs = self.loader.discover_specifications()
        expected_specs = ["data_preprocessing", "model_training"]
        
        assert sorted(discovered_specs) == expected_specs
        
        # Test file finding
        training_files = self.loader.find_specification_files("model_training")
        assert len(training_files) >= 1
    
    def test_job_type_extraction_patterns(self):
        """Test job type extraction with various file naming patterns."""
        test_cases = [
            ("model_training_spec.py", "default"),
            ("model_training_validation_spec.py", "validation"),
            ("data_preprocessing_training_spec.py", "training"),
            ("model_evaluation_testing_spec.py", "testing"),
            ("calibration_calibration_spec.py", "calibration")
        ]
        
        for filename, expected_job_type in test_cases:
            spec_file = Path(filename)
            job_type = self.loader.extract_job_type_from_spec_file(spec_file)
            assert job_type == expected_job_type, f"Failed for {filename}"
    
    def test_error_resilience_in_loading(self):
        """Test that loading continues even when some specifications fail."""
        # Create mix of valid and invalid spec files
        (self.specs_dir / "valid_spec.py").touch()
        (self.specs_dir / "another_valid_spec.py").touch()
        
        with patch.object(self.loader, 'load_specification_from_python') as mock_load:
            # Simulate some specs failing to load
            mock_load.side_effect = [
                {'step_type': 'ValidSpec'},
                Exception("Load failed"),
            ]
            
            result = self.loader.load_all_specifications()
            
            # Should still get the valid specification
            assert len(result) >= 1
            assert 'valid' in result or 'another_valid' in result


class TestSpecificationLoaderErrorScenarios:
    """Test cases for error scenarios and edge cases."""
    
    def setup_method(self):
        """Set up error scenario test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.specs_dir = Path(self.temp_dir) / "specs"
        self.specs_dir.mkdir(exist_ok=True)
        
        self.loader = SpecificationLoader(str(self.specs_dir))
    
    def teardown_method(self):
        """Clean up error scenario test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_step_specification_to_dict_with_string_enums(self):
        """Test converting StepSpecification when enums are already strings."""
        spec_obj = Mock()
        spec_obj.step_type = "ModelTraining"
        spec_obj.node_type = "ProcessingJob"  # String instead of enum
        spec_obj.dependencies = {}
        spec_obj.outputs = {}
        
        result = self.loader.step_specification_to_dict(spec_obj)
        
        assert result['node_type'] == "ProcessingJob"
    
    def test_dict_to_step_specification_with_missing_fields(self):
        """Test converting dict to StepSpecification with missing optional fields."""
        spec_dict = {
            'step_type': 'ModelTraining',
            'node_type': 'ProcessingJob',
            'dependencies': [{
                'logical_name': 'training_data',
                'dependency_type': 'InputData',
                'required': True,
                'data_type': 'tabular'
                # Missing optional fields
            }],
            'outputs': [{
                'logical_name': 'model',
                'output_type': 'Model',
                'property_path': '/opt/ml/model',
                'data_type': 'model'
                # Missing optional fields
            }]
        }
        
        with patch('cursus.validation.alignment.loaders.specification_loader.DependencySpec') as mock_dep_spec:
            with patch('cursus.validation.alignment.loaders.specification_loader.OutputSpec') as mock_output_spec:
                with patch('cursus.validation.alignment.loaders.specification_loader.StepSpecification') as mock_step_spec:
                    
                    self.loader.dict_to_step_specification(spec_dict)
                    
                    # Verify defaults are provided for missing fields
                    dep_call_args = mock_dep_spec.call_args[1]
                    assert dep_call_args['description'] == ''
                    assert dep_call_args['semantic_keywords'] == []
                    
                    output_call_args = mock_output_spec.call_args[1]
                    assert output_call_args['description'] == ''
                    assert output_call_args['aliases'] == []
    
    def test_specification_references_contract_edge_cases(self):
        """Test specification contract reference with edge cases."""
        # Test with empty step_type
        spec_dict = {'step_type': ''}
        result = self.loader._specification_references_contract(spec_dict, "model_training")
        assert result is False
        
        # Test with None step_type
        spec_dict = {'step_type': None}
        result = self.loader._specification_references_contract(spec_dict, "model_training")
        assert result is False
        
        # Test with missing step_type
        spec_dict = {}
        result = self.loader._specification_references_contract(spec_dict, "model_training")
        assert result is False


if __name__ == '__main__':
    pytest.main([__file__])
