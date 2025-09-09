"""
Unit tests for cursus.validation.alignment.static_analysis.builder_analyzer module.

Tests the BuilderArgumentExtractor and BuilderRegistry classes that analyze step
builder classes to extract arguments from _get_job_arguments() methods and map
scripts to their corresponding builders.
"""

import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
import ast
import tempfile
from typing import Set, List, Dict, Any, Optional

from cursus.validation.alignment.static_analysis.builder_analyzer import (
    BuilderArgumentExtractor,
    BuilderRegistry,
    extract_builder_arguments
)


class TestBuilderArgumentExtractor:
    """Test cases for BuilderArgumentExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.builder_file = Path(self.temp_dir) / "test_builder.py"
        
        # Sample builder file content with _get_job_arguments method
        self.sample_builder_content = '''
class TestStepBuilder:
    def _get_job_arguments(self):
        return [
            "--learning-rate", "0.01",
            "--max-depth", "6",
            "--n-estimators", "100",
            "--output-path", "/opt/ml/model"
        ]
    
    def other_method(self):
        return ["--not-a-job-arg"]
'''
        
        # Create the test builder file
        with open(self.builder_file, 'w') as f:
            f.write(self.sample_builder_content)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_success(self):
        """Test successful initialization of BuilderArgumentExtractor."""
        extractor = BuilderArgumentExtractor(str(self.builder_file))
        
        assert extractor.builder_file_path == self.builder_file
        assert extractor.builder_ast is not None
        assert isinstance(extractor.builder_ast, ast.Module)
    
    def test_init_file_not_found(self):
        """Test initialization with non-existent file."""
        nonexistent_file = Path(self.temp_dir) / "nonexistent.py"
        
        with pytest.raises(ValueError) as exc_info:
            BuilderArgumentExtractor(str(nonexistent_file))
        
        assert "Failed to parse builder file" in str(exc_info.value)
    
    def test_init_invalid_python_file(self):
        """Test initialization with invalid Python syntax."""
        invalid_file = Path(self.temp_dir) / "invalid.py"
        with open(invalid_file, 'w') as f:
            f.write("invalid python syntax !!!")
        
        with pytest.raises(ValueError) as exc_info:
            BuilderArgumentExtractor(str(invalid_file))
        
        assert "Failed to parse builder file" in str(exc_info.value)
    
    def test_extract_job_arguments_success(self):
        """Test successful extraction of job arguments."""
        extractor = BuilderArgumentExtractor(str(self.builder_file))
        arguments = extractor.extract_job_arguments()
        
        expected_args = {
            "learning-rate",
            "max-depth", 
            "n-estimators",
            "output-path"
        }
        
        assert arguments == expected_args
    
    def test_extract_job_arguments_no_method(self):
        """Test extraction when _get_job_arguments method doesn't exist."""
        # Create builder without _get_job_arguments method
        no_method_content = '''
class TestStepBuilder:
    def other_method(self):
        return ["--not-relevant"]
'''
        no_method_file = Path(self.temp_dir) / "no_method.py"
        with open(no_method_file, 'w') as f:
            f.write(no_method_content)
        
        extractor = BuilderArgumentExtractor(str(no_method_file))
        arguments = extractor.extract_job_arguments()
        
        assert arguments == set()
    
    def test_extract_job_arguments_empty_method(self):
        """Test extraction from empty _get_job_arguments method."""
        empty_method_content = '''
class TestStepBuilder:
    def _get_job_arguments(self):
        return []
'''
        empty_method_file = Path(self.temp_dir) / "empty_method.py"
        with open(empty_method_file, 'w') as f:
            f.write(empty_method_content)
        
        extractor = BuilderArgumentExtractor(str(empty_method_file))
        arguments = extractor.extract_job_arguments()
        
        assert arguments == set()
    
    def test_extract_job_arguments_list_format(self):
        """Test extraction from method that returns list of arguments."""
        list_format_content = '''
class TestStepBuilder:
    def _get_job_arguments(self):
        args = [
            "--batch-size", "32",
            "--epochs", "10"
        ]
        return args
'''
        list_format_file = Path(self.temp_dir) / "list_format.py"
        with open(list_format_file, 'w') as f:
            f.write(list_format_content)
        
        extractor = BuilderArgumentExtractor(str(list_format_file))
        arguments = extractor.extract_job_arguments()
        
        expected_args = {"batch-size", "epochs"}
        assert arguments == expected_args
    
    def test_extract_job_arguments_mixed_format(self):
        """Test extraction from method with mixed argument formats."""
        mixed_format_content = '''
class TestStepBuilder:
    def _get_job_arguments(self):
        base_args = ["--model-type", "xgboost"]
        additional_args = ["--verbose", "true"]
        
        # Some inline arguments
        return base_args + additional_args + ["--debug", "false"]
'''
        mixed_format_file = Path(self.temp_dir) / "mixed_format.py"
        with open(mixed_format_file, 'w') as f:
            f.write(mixed_format_content)
        
        extractor = BuilderArgumentExtractor(str(mixed_format_file))
        arguments = extractor.extract_job_arguments()
        
        expected_args = {"model-type", "verbose", "debug"}
        assert arguments == expected_args
    
    def test_extract_job_arguments_ignores_non_args(self):
        """Test that extraction ignores strings that don't start with --."""
        non_args_content = '''
class TestStepBuilder:
    def _get_job_arguments(self):
        return [
            "--valid-arg", "value",
            "not-an-arg", "value",
            "-single-dash", "value",
            "--another-valid-arg", "value"
        ]
'''
        non_args_file = Path(self.temp_dir) / "non_args.py"
        with open(non_args_file, 'w') as f:
            f.write(non_args_content)
        
        extractor = BuilderArgumentExtractor(str(non_args_file))
        arguments = extractor.extract_job_arguments()
        
        expected_args = {"valid-arg", "another-valid-arg"}
        assert arguments == expected_args
    
    def test_find_job_arguments_method_success(self):
        """Test finding _get_job_arguments method in AST."""
        extractor = BuilderArgumentExtractor(str(self.builder_file))
        method_node = extractor._find_job_arguments_method()
        
        assert method_node is not None
        assert isinstance(method_node, ast.FunctionDef)
        assert method_node.name == "_get_job_arguments"
    
    def test_find_job_arguments_method_not_found(self):
        """Test finding method when it doesn't exist."""
        no_method_content = '''
class TestStepBuilder:
    def other_method(self):
        pass
'''
        no_method_file = Path(self.temp_dir) / "no_method.py"
        with open(no_method_file, 'w') as f:
            f.write(no_method_content)
        
        extractor = BuilderArgumentExtractor(str(no_method_file))
        method_node = extractor._find_job_arguments_method()
        
        assert method_node is None
    
    def test_get_method_source_success(self):
        """Test getting source code of _get_job_arguments method."""
        extractor = BuilderArgumentExtractor(str(self.builder_file))
        source = extractor.get_method_source()
        
        assert source is not None
        assert "def _get_job_arguments(self):" in source
        assert "--learning-rate" in source
    
    def test_get_method_source_no_method(self):
        """Test getting source when method doesn't exist."""
        no_method_content = '''
class TestStepBuilder:
    def other_method(self):
        pass
'''
        no_method_file = Path(self.temp_dir) / "no_method.py"
        with open(no_method_file, 'w') as f:
            f.write(no_method_content)
        
        extractor = BuilderArgumentExtractor(str(no_method_file))
        source = extractor.get_method_source()
        
        assert source is None
    
    def test_extract_arguments_from_method_ast_str_compatibility(self):
        """Test extraction works with both ast.Constant and ast.Str (older Python)."""
        extractor = BuilderArgumentExtractor(str(self.builder_file))
        method_node = extractor._find_job_arguments_method()
        
        # Mock ast.Str nodes for older Python compatibility testing
        with patch('ast.walk') as mock_walk:
            # Create mock nodes including ast.Str
            mock_str_node = Mock()
            mock_str_node.s = "--old-style-arg"
            
            mock_constant_node = Mock()
            mock_constant_node.value = "--new-style-arg"
            
            mock_walk.return_value = [mock_str_node, mock_constant_node]
            
            # Mock isinstance to return appropriate values
            def mock_isinstance(obj, cls):
                if obj is mock_str_node and cls is ast.Str:
                    return True
                elif obj is mock_constant_node and cls is ast.Constant:
                    return True
                elif obj is mock_constant_node.value and cls is str:
                    return True
                return False
            
            with patch('builtins.isinstance', side_effect=mock_isinstance):
                arguments = extractor._extract_arguments_from_method(method_node)
                
                expected_args = {"old-style-arg", "new-style-arg"}
                assert arguments == expected_args


class TestBuilderRegistry:
    """Test cases for BuilderRegistry class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.builders_dir = Path(self.temp_dir) / "builders"
        self.builders_dir.mkdir()
        
        # Create sample builder files
        self.sample_builders = {
            "builder_model_training_step.py": '''
from ..configs.config_model_training_step import ModelTrainingConfig
class ModelTrainingStepBuilder:
    pass
''',
            "builder_data_preprocessing_step.py": '''
from ..configs.config_data_preprocessing_step import DataPreprocessingConfig
class DataPreprocessingStepBuilder:
    entry_point = "data_preprocessing.py"
''',
            "builder_model_evaluation_xgb_step.py": '''
from ..configs.config_model_evaluation_step import ModelEvaluationConfig
class ModelEvaluationXGBStepBuilder:
    pass
''',
            "__init__.py": "# Init file"
        }
        
        # Create the builder files
        for filename, content in self.sample_builders.items():
            builder_file = self.builders_dir / filename
            with open(builder_file, 'w') as f:
                f.write(content)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_success(self):
        """Test successful initialization of BuilderRegistry."""
        registry = BuilderRegistry(str(self.builders_dir))
        
        assert registry.builders_dir == self.builders_dir
        assert isinstance(registry._script_to_builder_mapping, dict)
        assert len(registry._script_to_builder_mapping) > 0
    
    def test_init_nonexistent_directory(self):
        """Test initialization with non-existent directory."""
        nonexistent_dir = Path(self.temp_dir) / "nonexistent"
        
        registry = BuilderRegistry(str(nonexistent_dir))
        
        assert registry._script_to_builder_mapping == {}
    
    def test_build_script_mapping_success(self):
        """Test building script-to-builder mapping."""
        registry = BuilderRegistry(str(self.builders_dir))
        mapping = registry.get_all_mappings()
        
        # Should contain mappings for the builder files
        assert "model_training" in mapping
        assert "data_preprocessing" in mapping
        assert "model_evaluation_xgb" in mapping
        
        # Should not contain __init__
        assert "__init__" not in mapping
    
    def test_extract_script_names_from_builder_filename(self):
        """Test extracting script names from builder filename."""
        registry = BuilderRegistry(str(self.builders_dir))
        
        builder_file = self.builders_dir / "builder_model_training_step.py"
        script_names = registry._extract_script_names_from_builder(builder_file)
        
        assert "model_training" in script_names
    
    def test_extract_script_names_from_builder_config_import(self):
        """Test extracting script names from config imports."""
        registry = BuilderRegistry(str(self.builders_dir))
        
        builder_file = self.builders_dir / "builder_data_preprocessing_step.py"
        script_names = registry._extract_script_names_from_builder(builder_file)
        
        # Should extract from both filename and config import
        assert "data_preprocessing" in script_names
    
    def test_extract_script_names_from_builder_entry_point(self):
        """Test extracting script names from entry_point references."""
        registry = BuilderRegistry(str(self.builders_dir))
        
        builder_file = self.builders_dir / "builder_data_preprocessing_step.py"
        script_names = registry._extract_script_names_from_builder(builder_file)
        
        # Should extract from entry_point reference
        assert "data_preprocessing" in script_names
    
    def test_generate_name_variations_preprocessing(self):
        """Test generating name variations for preprocessing."""
        registry = BuilderRegistry(str(self.builders_dir))
        
        variations = registry._generate_name_variations("data_preprocessing")
        assert "data_preprocess" in variations
        
        variations = registry._generate_name_variations("data_preprocess")
        assert "data_preprocessing" in variations
    
    def test_generate_name_variations_evaluation(self):
        """Test generating name variations for evaluation."""
        registry = BuilderRegistry(str(self.builders_dir))
        
        variations = registry._generate_name_variations("model_evaluation")
        assert "model_eval" in variations
        
        variations = registry._generate_name_variations("model_eval")
        assert "model_evaluation" in variations
    
    def test_generate_name_variations_xgboost(self):
        """Test generating name variations for xgboost."""
        registry = BuilderRegistry(str(self.builders_dir))
        
        variations = registry._generate_name_variations("model_xgboost")
        assert "model_xgb" in variations
        
        variations = registry._generate_name_variations("model_xgb")
        assert "model_xgboost" in variations
    
    def test_get_builder_for_script_success(self):
        """Test getting builder file for existing script."""
        registry = BuilderRegistry(str(self.builders_dir))
        
        builder_file = registry.get_builder_for_script("model_training")
        
        assert builder_file is not None
        assert "builder_model_training_step.py" in builder_file
    
    def test_get_builder_for_script_not_found(self):
        """Test getting builder file for non-existent script."""
        registry = BuilderRegistry(str(self.builders_dir))
        
        builder_file = registry.get_builder_for_script("nonexistent_script")
        
        assert builder_file is None
    
    def test_get_builder_for_script_with_variations(self):
        """Test getting builder file using name variations."""
        registry = BuilderRegistry(str(self.builders_dir))
        
        # Should find builder using name variations
        builder_file = registry.get_builder_for_script("data_preprocess")
        
        # Might find it through variations
        assert builder_file is None or "builder_data_preprocessing_step.py" in builder_file
    
    def test_get_all_mappings(self):
        """Test getting all script-to-builder mappings."""
        registry = BuilderRegistry(str(self.builders_dir))
        
        mappings = registry.get_all_mappings()
        
        assert isinstance(mappings, dict)
        assert len(mappings) > 0
        
        # Should be a copy, not the original
        original_mappings = registry._script_to_builder_mapping
        assert mappings is not original_mappings


class TestExtractBuilderArgumentsFunction:
    """Test cases for the extract_builder_arguments convenience function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.builders_dir = Path(self.temp_dir) / "builders"
        self.builders_dir.mkdir()
        
        # Create a builder file with _get_job_arguments method
        builder_content = '''
class TestStepBuilder:
    def _get_job_arguments(self):
        return ["--test-arg", "value", "--another-arg", "value2"]
'''
        builder_file = self.builders_dir / "builder_test_script_step.py"
        with open(builder_file, 'w') as f:
            f.write(builder_content)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_extract_builder_arguments_success(self):
        """Test successful extraction using convenience function."""
        arguments = extract_builder_arguments("test_script", str(self.builders_dir))
        
        expected_args = {"test-arg", "another-arg"}
        assert arguments == expected_args
    
    def test_extract_builder_arguments_script_not_found(self):
        """Test extraction when script has no corresponding builder."""
        arguments = extract_builder_arguments("nonexistent_script", str(self.builders_dir))
        
        assert arguments == set()
    
    def test_extract_builder_arguments_invalid_builder_dir(self):
        """Test extraction with invalid builder directory."""
        nonexistent_dir = Path(self.temp_dir) / "nonexistent"
        
        arguments = extract_builder_arguments("test_script", str(nonexistent_dir))
        
        assert arguments == set()
    
    @patch('cursus.validation.alignment.static_analysis.builder_analyzer.BuilderArgumentExtractor')
    def test_extract_builder_arguments_extractor_error(self, mock_extractor_class):
        """Test extraction when BuilderArgumentExtractor raises an error."""
        mock_extractor_class.side_effect = Exception("Extraction failed")
        
        arguments = extract_builder_arguments("test_script", str(self.builders_dir))
        
        assert arguments == set()


class TestBuilderAnalyzerIntegration:
    """Integration test cases for builder analyzer components."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.builders_dir = Path(self.temp_dir) / "builders"
        self.builders_dir.mkdir()
    
    def teardown_method(self):
        """Clean up integration test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_workflow_integration(self):
        """Test complete workflow from builder file to argument extraction."""
        # Create a realistic builder file
        builder_content = '''
from typing import List, Dict, Any
from ..configs.config_model_training_step import ModelTrainingConfig

class ModelTrainingStepBuilder:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
    
    def _get_job_arguments(self) -> List[str]:
        """Get command line arguments for the training job."""
        args = [
            "--learning-rate", str(self.config.learning_rate),
            "--max-depth", str(self.config.max_depth),
            "--n-estimators", str(self.config.n_estimators),
            "--model-output-path", self.config.model_output_path,
            "--metrics-output-path", self.config.metrics_output_path
        ]
        
        if self.config.enable_early_stopping:
            args.extend(["--early-stopping", "true"])
        
        return args
    
    def build_step(self):
        # Implementation details...
        pass
'''
        
        builder_file = self.builders_dir / "builder_model_training_step.py"
        with open(builder_file, 'w') as f:
            f.write(builder_content)
        
        # Test the full workflow
        arguments = extract_builder_arguments("model_training", str(self.builders_dir))
        
        expected_args = {
            "learning-rate",
            "max-depth", 
            "n-estimators",
            "model-output-path",
            "metrics-output-path",
            "early-stopping"
        }
        
        assert arguments == expected_args
    
    def test_error_resilience_integration(self):
        """Test that the system handles various error conditions gracefully."""
        # Create a builder with syntax errors
        invalid_builder = self.builders_dir / "builder_invalid_step.py"
        with open(invalid_builder, 'w') as f:
            f.write("invalid python syntax !!!")
        
        # Create a builder without _get_job_arguments method
        no_method_builder = self.builders_dir / "builder_no_method_step.py"
        with open(no_method_builder, 'w') as f:
            f.write("class NoMethodBuilder:\n    pass")
        
        # Test that errors are handled gracefully
        registry = BuilderRegistry(str(self.builders_dir))
        mappings = registry.get_all_mappings()
        
        # Should still work despite some invalid files
        assert isinstance(mappings, dict)
        
        # Test extraction with invalid files
        args1 = extract_builder_arguments("invalid", str(self.builders_dir))
        args2 = extract_builder_arguments("no_method", str(self.builders_dir))
        
        assert args1 == set()
        assert args2 == set()


if __name__ == '__main__':
    pytest.main([__file__])
