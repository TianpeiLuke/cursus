"""
Integration test for portable path resolution in step builders.

This test simulates the exact scenario from the MODS pipeline path resolution error analysis:
1. Create temp files and directories
2. Create config with source_dir pointing to temp file folder
3. Save config to JSON (similar to demo_config.ipynb)
4. Load config using load_config (which reconstructs portable paths)
5. Create processing step using loaded config (similar to demo_pipeline.ipynb)
6. Verify that SageMaker receives absolute paths instead of portable paths
"""

import pytest
import tempfile
import json
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging

# Import the classes we need to test
from cursus.steps.configs.config_tabular_preprocessing_step import TabularPreprocessingConfig
from cursus.steps.configs.config_xgboost_model_eval_step import XGBoostModelEvalConfig
from cursus.steps.configs.config_dummy_training_step import DummyTrainingConfig
from cursus.steps.configs.config_risk_table_mapping_step import RiskTableMappingConfig
from cursus.steps.configs.utils import merge_and_save_configs, load_configs
from cursus.core.base.config_base import BasePipelineConfig
from cursus.steps.configs.config_processing_step_base import ProcessingStepConfigBase

# Import step builders
from cursus.steps.builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
from cursus.steps.builders.builder_xgboost_model_eval_step import XGBoostModelEvalStepBuilder
from cursus.steps.builders.builder_dummy_training_step import DummyTrainingStepBuilder
from cursus.steps.builders.builder_risk_table_mapping_step import RiskTableMappingStepBuilder

# Import hyperparameters
from cursus.core.base.hyperparameters_base import ModelHyperparameters

logger = logging.getLogger(__name__)


class TestPortablePathResolutionIntegration:
    """Integration test for portable path resolution across the entire config->save->load->build cycle."""

    @pytest.fixture
    def temp_project_structure(self):
        """Create a temporary project structure similar to the real project."""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        # Create directory structure similar to real project
        dockers_dir = temp_path / "dockers" / "xgboost_atoz"
        scripts_dir = dockers_dir / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dummy script files
        (scripts_dir / "tabular_preprocessing.py").write_text("# Dummy preprocessing script")
        (scripts_dir / "xgboost_model_evaluation.py").write_text("# Dummy evaluation script")
        (scripts_dir / "dummy_training.py").write_text("# Dummy training script")
        (scripts_dir / "risk_table_mapping.py").write_text("# Dummy risk mapping script")
        
        # Create config directory
        config_dir = temp_path / "pipeline_config"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        yield {
            "temp_dir": temp_path,
            "dockers_dir": dockers_dir,
            "scripts_dir": scripts_dir,
            "config_dir": config_dir
        }
        
        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def base_hyperparameters(self):
        """Create base hyperparameters for testing."""
        return ModelHyperparameters(
            full_field_list=["field1", "field2", "label"],
            cat_field_list=["field1"],
            tab_field_list=["field2"],
            label_name="label",
            id_name="id",
            multiclass_categories=[0, 1]
        )

    @pytest.fixture
    def base_config(self, temp_project_structure):
        """Create base config pointing to temp directory."""
        return BasePipelineConfig(
            bucket="test-bucket",
            current_date="2025-09-22",
            region="NA",
            aws_region="us-east-1",
            author="test-author",
            role="arn:aws:iam::123456789012:role/test-role",
            service_name="TestService",
            pipeline_version="1.0.0",
            framework_version="1.7-1",
            py_version="py3",
            source_dir=str(temp_project_structure["dockers_dir"])
        )

    @pytest.fixture
    def processing_config(self, base_config, temp_project_structure):
        """Create processing config with processing_source_dir."""
        return ProcessingStepConfigBase.from_base_config(
            base_config,
            processing_source_dir=str(temp_project_structure["scripts_dir"]),
            processing_instance_type_large="ml.m5.12xlarge",
            processing_instance_type_small="ml.m5.4xlarge"
        )

    def test_tabular_preprocessing_portable_path_resolution(self, processing_config, temp_project_structure):
        """Test portable path resolution for TabularPreprocessingStepBuilder."""
        
        # Step 1: Create config with absolute paths (simulating demo_config.ipynb)
        config = TabularPreprocessingConfig.from_base_config(
            processing_config,
            job_type="training",
            label_name="label",
            processing_entry_point="tabular_preprocessing.py"
        )
        
        # Verify initial state - should have absolute paths
        assert config.processing_source_dir == str(temp_project_structure["scripts_dir"])
        assert config.get_script_path().endswith("tabular_preprocessing.py")
        
        # Step 2: Save config to JSON (simulating config serialization)
        config_file = temp_project_structure["config_dir"] / "test_config.json"
        config_list = [config]
        merged_config = merge_and_save_configs(config_list, str(config_file))
        
        # Verify config was saved
        assert config_file.exists()
        
        # Step 3: Load config from JSON (simulating load_config which creates portable paths)
        loaded_configs = load_configs(str(config_file))
        # load_configs returns a dictionary, get the first config
        loaded_config = list(loaded_configs.values())[0]
        
        # Verify loaded config has portable paths
        assert loaded_config.portable_processing_source_dir is not None
        assert loaded_config.portable_processing_source_dir.startswith("dockers/xgboost_atoz/scripts")
        
        # Step 4: Create step builder with loaded config (simulating demo_pipeline.ipynb)
        with patch('sagemaker.sklearn.SKLearnProcessor') as mock_processor_class:
            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_processor.run.return_value = Mock()
            
            # Mock the step creation
            with patch('sagemaker.workflow.steps.ProcessingStep') as mock_step_class:
                mock_step = Mock()
                mock_step_class.return_value = mock_step
                
                builder = TabularPreprocessingStepBuilder(
                    config=loaded_config,
                    role="test-role"
                )
                
                # Step 5: Create the step - this should resolve portable paths to absolute paths
                step = builder.create_step(
                    inputs={},
                    outputs={}
                )
                
                # Step 6: Verify that processor.run was called with absolute path, not portable path
                mock_processor.run.assert_called_once()
                call_args = mock_processor.run.call_args
                
                # Extract the source_dir argument
                source_dir_arg = call_args.kwargs.get('source_dir')
                
                # Verify it's an absolute path, not a portable path
                assert source_dir_arg is not None
                assert Path(source_dir_arg).is_absolute()
                assert not source_dir_arg.startswith("dockers/")  # Should not be portable
                assert source_dir_arg.endswith("scripts")  # Should point to scripts directory

    def test_xgboost_model_eval_portable_path_resolution(self, processing_config, temp_project_structure, base_hyperparameters):
        """Test portable path resolution for XGBoostModelEvalStepBuilder."""
        
        # Step 1: Create config with absolute paths
        config = XGBoostModelEvalConfig(
            **processing_config.model_dump(),
            processing_entry_point="xgboost_model_evaluation.py",
            job_type="calibration",
            hyperparameters=base_hyperparameters,
            xgboost_framework_version="1.7-1"
        )
        
        # Step 2: Save and load config to create portable paths
        config_file = temp_project_structure["config_dir"] / "test_xgb_eval_config.json"
        merged_config = merge_and_save_configs([config], str(config_file))
        loaded_configs = load_configs(str(config_file))
        loaded_config = list(loaded_configs.values())[0]
        
        # Verify portable path was created
        assert loaded_config.portable_processing_source_dir is not None
        
        # Step 3: Create step builder and verify path resolution
        with patch('sagemaker.xgboost.XGBoostProcessor') as mock_processor_class:
            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_processor.run.return_value = Mock()
            
            with patch('sagemaker.workflow.steps.ProcessingStep') as mock_step_class:
                mock_step = Mock()
                mock_step_class.return_value = mock_step
                
                builder = XGBoostModelEvalStepBuilder(
                    config=loaded_config,
                    role="test-role"
                )
                
                step = builder.create_step(
                    inputs={"input_path": "s3://test-bucket/input"},
                    outputs={}
                )
                
                # Verify absolute path was used
                call_args = mock_processor.run.call_args
                source_dir_arg = call_args.kwargs.get('source_dir')
                
                assert source_dir_arg is not None
                assert Path(source_dir_arg).is_absolute()
                assert not source_dir_arg.startswith("dockers/")

    def test_dummy_training_portable_effective_source_dir_resolution(self, processing_config, temp_project_structure):
        """Test portable_effective_source_dir resolution for DummyTrainingStepBuilder."""
        
        # Step 1: Create config
        config = DummyTrainingConfig.from_base_config(
            processing_config,
            processing_entry_point="dummy_training.py"
        )
        
        # Step 2: Save and load to create portable paths
        config_file = temp_project_structure["config_dir"] / "test_dummy_config.json"
        merged_config = merge_and_save_configs([config], str(config_file))
        loaded_configs = load_configs(str(config_file))
        loaded_config = list(loaded_configs.values())[0]
        
        # Verify portable_effective_source_dir is available
        assert loaded_config.portable_effective_source_dir is not None
        
        # Step 3: Test the new resolution method
        resolved_path = loaded_config.get_resolved_effective_source_dir()
        assert resolved_path is not None
        assert Path(resolved_path).is_absolute()
        
        # Step 4: Create step builder and verify it uses resolved path
        with patch('sagemaker.sklearn.SKLearnProcessor') as mock_processor_class:
            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_processor.run.return_value = Mock()
            
            with patch('sagemaker.workflow.steps.ProcessingStep') as mock_step_class:
                mock_step = Mock()
                mock_step_class.return_value = mock_step
                
                builder = DummyTrainingStepBuilder(
                    config=loaded_config,
                    role="test-role"
                )
                
                step = builder.create_step(
                    inputs={},
                    outputs={}
                )
                
                # Verify absolute path was used
                call_args = mock_processor.run.call_args
                source_dir_arg = call_args.kwargs.get('source_dir')
                
                assert source_dir_arg is not None
                assert Path(source_dir_arg).is_absolute()
                assert not source_dir_arg.startswith("dockers/")

    def test_risk_table_mapping_portable_effective_source_dir_resolution(self, processing_config, temp_project_structure):
        """Test portable_effective_source_dir resolution for RiskTableMappingStepBuilder."""
        
        # Step 1: Create config
        config = RiskTableMappingConfig.from_base_config(
            processing_config,
            processing_entry_point="risk_table_mapping.py",
            job_type="training",
            label_name="label",
            cat_field_list=["field1"]
        )
        
        # Step 2: Save and load to create portable paths
        config_file = temp_project_structure["config_dir"] / "test_risk_config.json"
        merged_config = merge_and_save_configs([config], str(config_file))
        loaded_configs = load_configs(str(config_file))
        loaded_config = list(loaded_configs.values())[0]
        
        # Step 3: Test the new resolution method
        resolved_path = loaded_config.get_resolved_effective_source_dir()
        assert resolved_path is not None
        assert Path(resolved_path).is_absolute()
        
        # Step 4: Create step builder and verify it uses resolved path
        with patch('sagemaker.sklearn.SKLearnProcessor') as mock_processor_class:
            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_processor.run.return_value = Mock()
            
            with patch('sagemaker.workflow.steps.ProcessingStep') as mock_step_class:
                mock_step = Mock()
                mock_step_class.return_value = mock_step
                
                builder = RiskTableMappingStepBuilder(
                    config=loaded_config,
                    role="test-role"
                )
                
                step = builder.create_step(
                    inputs={"data_input": "s3://test-bucket/data"},
                    outputs={}
                )
                
                # Verify absolute path was used
                call_args = mock_processor.run.call_args
                source_dir_arg = call_args.kwargs.get('source_dir')
                
                assert source_dir_arg is not None
                assert Path(source_dir_arg).is_absolute()
                assert not source_dir_arg.startswith("dockers/")

    def test_mods_pipeline_error_scenario_simulation(self, temp_project_structure):
        """
        Simulate the exact scenario from mods_pipeline_path_resolution_error_analysis.
        
        This test reproduces the error scenario where:
        1. Config is created with absolute paths
        2. Config is saved to JSON with portable paths
        3. Config is loaded from JSON (portable paths reconstructed)
        4. Step builder tries to use portable path directly -> SageMaker validation error
        5. With our fix, step builder resolves portable path to absolute path -> Success
        """
        
        # Step 1: Create the exact scenario from the error analysis
        base_config = BasePipelineConfig(
            bucket="test-bucket",
            current_date="2025-09-22",
            region="NA",
            aws_region="us-east-1",
            author="test-author",
            role="arn:aws:iam::123456789012:role/test-role",
            service_name="AtoZ",
            pipeline_version="1.0.0",
            framework_version="1.7-1",
            py_version="py3",
            source_dir=str(temp_project_structure["dockers_dir"])
        )
        
        processing_config = ProcessingStepConfigBase.from_base_config(
            base_config,
            processing_source_dir=str(temp_project_structure["scripts_dir"])
        )
        
        # Create the problematic config (TabularPreprocessingConfig)
        config = TabularPreprocessingConfig.from_base_config(
            processing_config,
            job_type="training",
            label_name="is_abuse",
            processing_entry_point="tabular_preprocessing.py"
        )
        
        # Step 2: Save config to JSON (this creates portable paths)
        config_file = temp_project_structure["config_dir"] / "mods_pipeline_config.json"
        merged_config = merge_and_save_configs([config], str(config_file))
        
        # Step 3: Load config from JSON (simulating MODS pipeline loading)
        loaded_configs = load_configs(str(config_file))
        loaded_config = list(loaded_configs.values())[0]
        
        # Step 4: Verify the problematic portable path exists
        portable_script_path = loaded_config.get_portable_script_path()
        assert portable_script_path is not None
        assert portable_script_path.startswith("dockers/xgboost_atoz/scripts/")
        assert portable_script_path.endswith("tabular_preprocessing.py")
        
        # This is the exact path that was causing the SageMaker validation error:
        # "mods_pipeline_adapter/dockers/xgboost_atoz/scripts/tabular_preprocessing.py is not a valid file"
        expected_problematic_path = "dockers/xgboost_atoz/scripts/tabular_preprocessing.py"
        assert portable_script_path == expected_problematic_path
        
        # Step 5: Create step builder and verify our fix works
        with patch('sagemaker.sklearn.SKLearnProcessor') as mock_processor_class:
            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_processor.run.return_value = Mock()
            
            with patch('sagemaker.workflow.steps.ProcessingStep') as mock_step_class:
                mock_step = Mock()
                mock_step_class.return_value = mock_step
                
                builder = TabularPreprocessingStepBuilder(
                    config=loaded_config,
                    role="test-role"
                )
                
                # This should NOT raise a SageMaker validation error anymore
                step = builder.create_step(
                    inputs={},
                    outputs={}
                )
                
                # Step 6: Verify that the step builder resolved the portable path to absolute path
                call_args = mock_processor.run.call_args
                code_arg = call_args.kwargs.get('code')
                source_dir_arg = call_args.kwargs.get('source_dir')
                
                # Verify code is just the filename (correct)
                assert code_arg == "tabular_preprocessing.py"
                
                # Verify source_dir is absolute path (our fix)
                assert source_dir_arg is not None
                assert Path(source_dir_arg).is_absolute()
                assert source_dir_arg.endswith("scripts")
                
                # Most importantly: verify it's NOT the portable path that caused the error
                assert not source_dir_arg.startswith("dockers/")
                assert source_dir_arg != expected_problematic_path
                
                # Verify the resolved path actually exists (would pass SageMaker validation)
                resolved_script_path = Path(source_dir_arg) / code_arg
                assert resolved_script_path.exists()

    def test_all_portable_path_types_resolution(self, temp_project_structure):
        """Test that all types of portable paths are properly resolved."""
        
        # Create configs that use different portable path types
        base_config = BasePipelineConfig(
            bucket="test-bucket",
            current_date="2025-09-22",
            region="NA",
            aws_region="us-east-1",
            author="test-author",
            role="arn:aws:iam::123456789012:role/test-role",
            service_name="TestService",
            pipeline_version="1.0.0",
            framework_version="1.7-1",
            py_version="py3",
            source_dir=str(temp_project_structure["dockers_dir"])
        )
        
        processing_config = ProcessingStepConfigBase.from_base_config(
            base_config,
            processing_source_dir=str(temp_project_structure["scripts_dir"])
        )
        
        # Test different config types
        configs_to_test = [
            # Processing steps using get_portable_script_path
            TabularPreprocessingConfig.from_base_config(
                processing_config,
                job_type="training",
                label_name="label",
                processing_entry_point="tabular_preprocessing.py"
            ),
            # Processing steps using portable_effective_source_dir
            DummyTrainingConfig.from_base_config(
                processing_config,
                processing_entry_point="dummy_training.py"
            ),
            # Processing steps using portable_processing_source_dir
            XGBoostModelEvalConfig(
                **processing_config.model_dump(),
                processing_entry_point="xgboost_model_evaluation.py",
                job_type="calibration",
                hyperparameters=ModelHyperparameters(
                    full_field_list=["field1", "field2"],
                    cat_field_list=["field1"],
                    tab_field_list=["field2"],
                    label_name="label",
                    id_name="id",
                    multiclass_categories=[0, 1]
                ),
                xgboost_framework_version="1.7-1"
            )
        ]
        
        for i, config in enumerate(configs_to_test):
            # Save and load each config
            config_file = temp_project_structure["config_dir"] / f"test_config_{i}.json"
            merged_config = merge_and_save_configs([config], str(config_file))
            loaded_configs = load_configs(str(config_file))
            loaded_config = list(loaded_configs.values())[0]
            
            # Verify portable paths were created
            if hasattr(loaded_config, 'get_portable_script_path'):
                portable_script = loaded_config.get_portable_script_path()
                if portable_script:
                    assert portable_script.startswith("dockers/")
                    # Verify resolution works
                    resolved_script = loaded_config.get_resolved_path(portable_script)
                    assert Path(resolved_script).is_absolute()
            
            if hasattr(loaded_config, 'portable_source_dir'):
                portable_source = loaded_config.portable_source_dir
                if portable_source:
                    assert portable_source.startswith("dockers/")
                    # Verify resolution works
                    resolved_source = loaded_config.get_resolved_path(portable_source)
                    assert Path(resolved_source).is_absolute()
            
            if hasattr(loaded_config, 'portable_processing_source_dir'):
                portable_processing = loaded_config.portable_processing_source_dir
                if portable_processing:
                    assert portable_processing.startswith("dockers/")
                    # Verify resolution works
                    resolved_processing = loaded_config.get_resolved_path(portable_processing)
                    assert Path(resolved_processing).is_absolute()
            
            if hasattr(loaded_config, 'portable_effective_source_dir'):
                portable_effective = loaded_config.portable_effective_source_dir
                if portable_effective:
                    assert portable_effective.startswith("dockers/")
                    # Verify new resolution method works
                    resolved_effective = loaded_config.get_resolved_effective_source_dir()
                    assert Path(resolved_effective).is_absolute()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
