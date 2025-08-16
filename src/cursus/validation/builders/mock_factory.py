"""
Consolidated Mock factory system for creating step type-specific mock objects.
This version combines the best features from both the original and enhanced versions.
"""

from typing import Dict, Any, Optional, Type, List
from types import SimpleNamespace
from unittest.mock import MagicMock
from pathlib import Path
import os
from ...core.base.builder_base import StepBuilderBase


class StepTypeMockFactory:
    """Consolidated factory for creating step type-specific mock objects with enhanced validation."""
    
    def __init__(self, step_info: Dict[str, Any], test_mode: bool = True):
        """
        Initialize factory with step information.
        
        Args:
            step_info: Step information from StepInfoDetector
            test_mode: Enable test mode for relaxed validation and better error handling
        """
        self.step_info = step_info
        self.sagemaker_step_type = step_info.get("sagemaker_step_type")
        self.framework = step_info.get("framework")
        self.test_pattern = step_info.get("test_pattern")
        self.test_mode = test_mode
        
        # Ensure test script directory exists
        self._ensure_test_script_directory()
    
    def _ensure_test_script_directory(self) -> str:
        """Ensure test script directory exists and create necessary script files."""
        test_script_dir = '/tmp/mock_scripts'
        Path(test_script_dir).mkdir(parents=True, exist_ok=True)
        
        # Create common script files that builders expect
        script_files = [
            'tabular_preprocess.py',
            'currency_conversion.py', 
            'risk_table_mapping.py',
            'model_calibration.py',
            'dummy_training.py',
            'model_evaluation_xgb.py',
            'payload.py',
            'package.py',
            'train_xgb.py',
            'train_pytorch.py',
            'inference.py',
            'process.py'
        ]
        
        for script_file in script_files:
            script_path = Path(test_script_dir) / script_file
            if not script_path.exists():
                script_path.write_text(f'# Mock script for testing: {script_file}\nprint("Mock script execution")\n')
        
        return test_script_dir
    
    def create_mock_config(self) -> Any:
        """Create appropriate mock config for the step type with enhanced validation."""
        builder_name = self.step_info.get("builder_class_name", "")
        
        # Try to create proper config instance for type-strict builders
        proper_config = self._try_create_proper_config_instance(builder_name)
        if proper_config:
            return proper_config
        
        # Fall back to enhanced SimpleNamespace for flexible builders
        mock_config = self._create_base_config()
        
        # Add step type-specific configuration
        if self.sagemaker_step_type == "Processing":
            self._add_processing_config(mock_config)
        elif self.sagemaker_step_type == "Training":
            self._add_training_config(mock_config)
        elif self.sagemaker_step_type == "Transform":
            self._add_transform_config(mock_config)
        elif self.sagemaker_step_type == "CreateModel":
            self._add_createmodel_config(mock_config)
        else:
            self._add_generic_config(mock_config)
        
        # Add framework-specific configuration
        self._add_framework_config(mock_config)
        
        return mock_config
    
    def _try_create_proper_config_instance(self, builder_name: str) -> Optional[Any]:
        """Try to create proper config instance with enhanced error handling."""
        try:
            # First create base config with all required fields
            base_config = self._create_base_pipeline_config()
            if not base_config:
                if self.test_mode:
                    print(f"INFO: Could not create base config for {builder_name}")
                return None
            
            # Then create specific config using from_base_config with enhanced parameters
            config_instance = None
            if "Payload" in builder_name:
                config_instance = self._create_payload_config_from_base(base_config)
            elif "Package" in builder_name:
                config_instance = self._create_package_config_from_base(base_config)
            elif "TabularPreprocessing" in builder_name:
                config_instance = self._create_tabular_preprocessing_config_from_base(base_config)
            elif "CurrencyConversion" in builder_name:
                config_instance = self._create_currency_conversion_config_from_base(base_config)
            elif "RiskTableMapping" in builder_name:
                config_instance = self._create_risk_table_mapping_config_from_base(base_config)
            elif "ModelCalibration" in builder_name:
                config_instance = self._create_model_calibration_config_from_base(base_config)
            elif "DummyTraining" in builder_name:
                config_instance = self._create_dummy_training_config_from_base(base_config)
            elif "XGBoostModelEval" in builder_name:
                config_instance = self._create_xgboost_model_eval_config_from_base(base_config)
            elif "XGBoostTraining" in builder_name:
                config_instance = self._create_xgboost_training_config_from_base(base_config)
            elif "PyTorchTraining" in builder_name:
                config_instance = self._create_pytorch_training_config_from_base(base_config)
            elif "XGBoostModel" in builder_name:
                config_instance = self._create_xgboost_model_config_from_base(base_config)
            elif "PyTorchModel" in builder_name:
                config_instance = self._create_pytorch_model_config_from_base(base_config)
            elif "BatchTransform" in builder_name:
                config_instance = self._create_batch_transform_config_from_base(base_config)
            elif "Registration" in builder_name:
                config_instance = self._create_registration_config_from_base(base_config)
            elif "CradleDataLoading" in builder_name:
                config_instance = self._create_cradle_data_loading_config_from_base(base_config)
            
            if config_instance:
                if self.test_mode:
                    print(f"INFO: Successfully created {type(config_instance).__name__} for {builder_name}")
                return config_instance
            else:
                if self.test_mode:
                    print(f"INFO: No specific config creator found for {builder_name}")
                return None
                
        except Exception as e:
            if self.test_mode:
                print(f"INFO: Could not create proper config for {builder_name}, using fallback: {e}")
            else:
                print(f"Failed to create proper config for {builder_name}: {e}")
            return None
        
        return None
    
    def _create_base_pipeline_config(self) -> Optional[Any]:
        """Create enhanced base pipeline config with proper test setup."""
        try:
            from ...core.base.config_base import BasePipelineConfig
            
            # Use the test script directory we created
            test_script_dir = '/tmp/mock_scripts'
            
            return BasePipelineConfig(
                author='test-author',
                bucket='test-bucket',
                role='arn:aws:iam::123456789012:role/test-role',
                region='NA',  # Use valid region code that passes validation
                service_name='test-service',
                pipeline_version='1.0.0',
                model_class='xgboost',
                current_date='2025-08-15',
                framework_version='1.7-1',
                py_version='py3',
                source_dir=test_script_dir
            )
        except Exception as e:
            if self.test_mode:
                print(f"INFO: Failed to create BasePipelineConfig, using fallback: {e}")
            else:
                print(f"Failed to create BasePipelineConfig: {e}")
            return None
    
    def _create_payload_config_from_base(self, base_config: Any) -> Any:
        """Create proper PayloadConfig instance using from_base_config."""
        try:
            from ...steps.configs.config_payload_step import PayloadConfig
            return PayloadConfig.from_base_config(
                base_config,
                # Payload-specific fields
                model_owner='test-team',
                model_domain='test-domain',
                model_objective='test-objective',
                source_model_inference_output_variable_list={'prediction': 'NUMERIC'},
                source_model_inference_input_variable_list={'feature1': 'NUMERIC', 'feature2': 'TEXT'},
                expected_tps=100,
                max_latency_in_millisecond=1000,
                framework='xgboost',
                processing_entry_point='payload.py',
                source_model_inference_content_types=['text/csv'],
                source_model_inference_response_types=['application/json'],
                max_acceptable_error_rate=0.2,
                default_numeric_value=0.0,
                default_text_value='DEFAULT_TEXT',
                processing_instance_count=1,
                processing_volume_size=30,
                processing_instance_type_large='ml.m5.xlarge',
                processing_instance_type_small='ml.m5.large',
                use_large_processing_instance=False,
                processing_framework_version='1.2-1'
            )
        except Exception as e:
            if self.test_mode:
                print(f"INFO: Failed to create PayloadConfig from base: {e}")
            else:
                print(f"Failed to create PayloadConfig from base: {e}")
            return None
    
    def _create_package_config_from_base(self, base_config: Any) -> Any:
        """Create proper PackageConfig instance using from_base_config."""
        try:
            from ...steps.configs.config_package_step import PackageConfig
            return PackageConfig.from_base_config(
                base_config,
                processing_entry_point='package.py',
                processing_source_dir=None,  # Don't set source dir to avoid validation issues
                processing_instance_count=1,
                processing_volume_size=30,
                processing_instance_type_large='ml.m5.xlarge',
                processing_instance_type_small='ml.m5.large',
                use_large_processing_instance=False,
                processing_framework_version='1.2-1'
            )
        except Exception as e:
            if self.test_mode:
                print(f"INFO: Failed to create PackageConfig from base: {e}")
            else:
                print(f"Failed to create PackageConfig from base: {e}")
            return None
    
    def _create_tabular_preprocessing_config_from_base(self, base_config: Any) -> Any:
        """Create proper TabularPreprocessingConfig instance using from_base_config."""
        try:
            from ...steps.configs.config_tabular_preprocessing_step import TabularPreprocessingConfig
            return TabularPreprocessingConfig.from_base_config(
                base_config,
                job_type='training',
                label_name='target',
                train_ratio=0.7,
                test_val_ratio=0.5,
                categorical_columns=['category_1', 'category_2'],
                numerical_columns=['numeric_1', 'numeric_2'],
                text_columns=['text_1', 'text_2'],
                date_columns=['date_1'],
                processing_entry_point='tabular_preprocess.py',
                processing_instance_count=1,
                processing_volume_size=30,
                processing_instance_type_large='ml.m5.xlarge',
                processing_instance_type_small='ml.m5.large',
                use_large_processing_instance=False,
                processing_framework_version='1.2-1'
            )
        except Exception as e:
            if self.test_mode:
                print(f"INFO: Failed to create TabularPreprocessingConfig from base: {e}")
            else:
                print(f"Failed to create TabularPreprocessingConfig from base: {e}")
            return None
    
    def _create_currency_conversion_config_from_base(self, base_config: Any) -> Any:
        """Create proper CurrencyConversionConfig instance using from_base_config."""
        try:
            from ...steps.configs.config_currency_conversion_step import CurrencyConversionConfig
            return CurrencyConversionConfig.from_base_config(
                base_config,
                job_type='training',
                label_field='target',
                marketplace_id_col='marketplace_id',
                currency_col='currency',
                default_currency='USD',
                mode='per_split',
                currency_conversion_var_list=['price', 'cost'],
                currency_conversion_dict={'USD': 1.0, 'EUR': 1.1, 'GBP': 1.3},  # Include USD with rate 1.0
                marketplace_info={'1': {'currency': 'USD'}, '2': {'currency': 'EUR'}},  # Proper dict format
                train_ratio=0.7,
                test_val_ratio=0.5,
                enable_currency_conversion=True,
                skip_invalid_currencies=False,
                processing_entry_point='currency_conversion.py',
                processing_instance_count=1,
                processing_volume_size=30,
                processing_instance_type_large='ml.m5.xlarge',
                processing_instance_type_small='ml.m5.large',
                use_large_processing_instance=False,
                processing_framework_version='1.2-1'
            )
        except Exception as e:
            if self.test_mode:
                print(f"INFO: Failed to create CurrencyConversionConfig from base: {e}")
            else:
                print(f"Failed to create CurrencyConversionConfig from base: {e}")
            return None
    
    def _create_risk_table_mapping_config_from_base(self, base_config: Any) -> Any:
        """Create proper RiskTableMappingConfig instance using from_base_config."""
        try:
            from ...steps.configs.config_risk_table_mapping_step import RiskTableMappingConfig
            return RiskTableMappingConfig.from_base_config(
                base_config,
                job_type='training',
                label_name='target',
                categorical_columns=['risk_category', 'market_segment'],
                processing_entry_point='risk_table_mapping.py',
                processing_instance_count=1,
                processing_volume_size=30,
                processing_instance_type_large='ml.m5.xlarge',
                processing_instance_type_small='ml.m5.large',
                use_large_processing_instance=False,
                processing_framework_version='1.2-1'
            )
        except Exception as e:
            if self.test_mode:
                print(f"INFO: Failed to create RiskTableMappingConfig from base: {e}")
            else:
                print(f"Failed to create RiskTableMappingConfig from base: {e}")
            return None
    
    def _create_model_calibration_config_from_base(self, base_config: Any) -> Any:
        """Create proper ModelCalibrationConfig instance using from_base_config."""
        try:
            from ...steps.configs.config_model_calibration_step import ModelCalibrationConfig
            return ModelCalibrationConfig.from_base_config(
                base_config,
                job_type='training',
                label_field='target',  # Note: ModelCalibration uses label_field, not label_name
                id_field='id',
                calibration_method='isotonic',
                processing_entry_point='model_calibration.py',
                processing_source_dir='/tmp/mock_scripts',  # Use the test script directory
                processing_instance_count=1,
                processing_volume_size=30,
                processing_instance_type_large='ml.m5.xlarge',
                processing_instance_type_small='ml.m5.large',
                use_large_processing_instance=False,
                processing_framework_version='1.2-1',
                # Add required fields for ModelCalibration
                score_field='score',
                is_binary=True,
                monotonic_constraint=True,
                gam_splines=10,
                error_threshold=0.1,
                num_classes=2,
                score_field_prefix='score_',
                multiclass_categories=['class_0', 'class_1']
            )
        except Exception as e:
            if self.test_mode:
                print(f"INFO: Failed to create ModelCalibrationConfig from base: {e}")
            else:
                print(f"Failed to create ModelCalibrationConfig from base: {e}")
            return None
    
    def _create_dummy_training_config_from_base(self, base_config: Any) -> Any:
        """Create proper DummyTrainingConfig instance using from_base_config."""
        try:
            from ...steps.configs.config_dummy_training_step import DummyTrainingConfig
            from ...core.base.hyperparameters_base import ModelHyperparameters
            from pathlib import Path
            
            # Create a temporary model file for testing with proper .tar.gz extension
            temp_model_dir = Path('/tmp/mock_models')
            temp_model_dir.mkdir(parents=True, exist_ok=True)
            temp_model_path = temp_model_dir / 'test_model.tar.gz'
            
            # Create empty tar.gz file for validation
            if not temp_model_path.exists():
                temp_model_path.write_bytes(b'mock tar.gz content')
            
            # Create proper ModelHyperparameters instance with required fields
            try:
                # Try to create a proper hyperparameters instance
                mock_hp = ModelHyperparameters(
                    full_field_list=['id', 'feature1', 'feature2', 'target'],
                    cat_field_list=['feature1'],
                    tab_field_list=['feature2'],
                    id_name='id',
                    label_name='target',
                    multiclass_categories=['class_0', 'class_1']
                )
            except Exception as hp_error:
                if self.test_mode:
                    print(f"INFO: Failed to create proper ModelHyperparameters, using default: {hp_error}")
                # Fall back to default constructor
                mock_hp = ModelHyperparameters()
            
            return DummyTrainingConfig.from_base_config(
                base_config,
                pretrained_model_path=str(temp_model_path),  # Use correct field name and ensure string
                hyperparameters=mock_hp,
                hyperparameters_s3_uri='s3://test-bucket/config/hyperparameters.json',
                processing_entry_point='dummy_training.py',
                processing_instance_count=1,
                processing_volume_size=30,
                processing_instance_type_large='ml.m5.xlarge',
                processing_instance_type_small='ml.m5.large',
                use_large_processing_instance=False,
                processing_framework_version='1.2-1'
            )
        except Exception as e:
            if self.test_mode:
                print(f"INFO: Failed to create DummyTrainingConfig from base: {e}")
                # Print more detailed error information for debugging
                import traceback
                print(f"DEBUG: Full traceback: {traceback.format_exc()}")
            else:
                print(f"Failed to create DummyTrainingConfig from base: {e}")
            return None
    
    def _create_xgboost_model_eval_config_from_base(self, base_config: Any) -> Any:
        """Create proper XGBoostModelEvalConfig instance using from_base_config."""
        try:
            from ...steps.configs.config_xgboost_model_eval_step import XGBoostModelEvalConfig
            
            # Create enhanced hyperparameters for model evaluation
            mock_hp = self._create_enhanced_xgboost_model_eval_hyperparameters()
            
            return XGBoostModelEvalConfig.from_base_config(
                base_config,
                job_type='training',
                processing_entry_point='model_evaluation_xgb.py',  # Use correct script name that exists
                processing_source_dir='/tmp/mock_scripts',  # Use test script directory
                xgboost_framework_version='1.7-1',
                hyperparameters=mock_hp,
                processing_instance_count=1,
                processing_volume_size=30,
                processing_instance_type_large='ml.m5.xlarge',
                processing_instance_type_small='ml.m5.large',
                use_large_processing_instance=False
            )
        except Exception as e:
            if self.test_mode:
                print(f"INFO: Failed to create XGBoostModelEvalConfig from base: {e}")
            else:
                print(f"Failed to create XGBoostModelEvalConfig from base: {e}")
            return None
    
    def _create_xgboost_training_config_from_base(self, base_config: Any) -> Any:
        """Create proper XGBoostTrainingConfig instance using from_base_config."""
        try:
            from ...steps.configs.config_xgboost_training_step import XGBoostTrainingConfig
            
            # Create enhanced hyperparameters that satisfy all validation rules
            mock_hp = self._create_enhanced_xgboost_hyperparameters()
            
            return XGBoostTrainingConfig.from_base_config(
                base_config,
                hyperparameters=mock_hp,
                hyperparameters_s3_uri='s3://test-bucket/config/hyperparameters.json',
                training_instance_type='ml.m5.xlarge',
                training_instance_count=1,
                training_volume_size=30,
                training_entry_point='train_xgb.py',
                framework_version='1.7-1',
                py_version='py3'
            )
        except Exception as e:
            if self.test_mode:
                print(f"INFO: Failed to create XGBoostTrainingConfig from base: {e}")
            else:
                print(f"Failed to create XGBoostTrainingConfig from base: {e}")
            return None
    
    def _create_pytorch_training_config_from_base(self, base_config: Any) -> Any:
        """Create proper PyTorchTrainingConfig instance using from_base_config."""
        try:
            from ...steps.configs.config_pytorch_training_step import PyTorchTrainingConfig
            
            # Create enhanced hyperparameters that satisfy all validation rules
            mock_hp = self._create_enhanced_pytorch_hyperparameters()
            
            return PyTorchTrainingConfig.from_base_config(
                base_config,
                hyperparameters=mock_hp,
                training_instance_type='ml.g5.12xlarge',  # Use valid instance type
                training_instance_count=1,
                training_volume_size=30,
                training_entry_point='train_pytorch.py',
                framework_version='1.12.0',
                py_version='py38'  # Use correct py version
            )
        except Exception as e:
            if self.test_mode:
                print(f"INFO: Failed to create PyTorchTrainingConfig from base: {e}")
            else:
                print(f"Failed to create PyTorchTrainingConfig from base: {e}")
            return None
    
    def _create_enhanced_xgboost_hyperparameters(self) -> Any:
        """Create enhanced XGBoost hyperparameters that satisfy all validation rules."""
        try:
            from ...steps.hyperparams.hyperparameters_xgboost import XGBoostModelHyperparameters
            
            # Create comprehensive field lists that satisfy validation
            full_field_list = ['id', 'feature1', 'feature2', 'feature3', 'feature4', 'target']
            cat_field_list = ['feature1', 'feature2']  # Subset of full_field_list
            tab_field_list = ['feature3', 'feature4']  # Subset of full_field_list
            
            return XGBoostModelHyperparameters(
                # Required fields from base ModelHyperparameters
                full_field_list=full_field_list,
                cat_field_list=cat_field_list,
                tab_field_list=tab_field_list,
                id_name='id',
                label_name='target',
                multiclass_categories=['class_0', 'class_1'],
                # Required XGBoost-specific fields
                num_round=100,
                max_depth=6,
                # Optional fields with defaults
                eta=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=1.0,
                gamma=0.0,
                # Additional fields that might be required
                objective='binary:logistic',
                eval_metric='auc',
                scale_pos_weight=1.0,
                alpha=0.0,
                lambda_reg=1.0,
                tree_method='auto',
                grow_policy='depthwise',
                max_leaves=0,
                max_bin=256,
                predictor='auto'
            )
        except Exception as e:
            if self.test_mode:
                print(f"INFO: Failed to create XGBoostModelHyperparameters, using fallback: {e}")
            else:
                print(f"Failed to create XGBoostModelHyperparameters: {e}")
            # Fallback to SimpleNamespace with all required attributes
            mock_hp = SimpleNamespace()
            mock_hp.full_field_list = ['id', 'feature1', 'feature2', 'feature3', 'feature4', 'target']
            mock_hp.cat_field_list = ['feature1', 'feature2']
            mock_hp.tab_field_list = ['feature3', 'feature4']
            mock_hp.id_name = 'id'
            mock_hp.label_name = 'target'
            mock_hp.multiclass_categories = ['class_0', 'class_1']
            mock_hp.num_round = 100
            mock_hp.max_depth = 6
            mock_hp.eta = 0.1
            mock_hp.model_dump = lambda: {
                'num_round': 100, 'max_depth': 6, 'eta': 0.1,
                'subsample': 0.8, 'colsample_bytree': 0.8
            }
            return mock_hp
    
    def _create_enhanced_pytorch_hyperparameters(self) -> Any:
        """Create enhanced PyTorch hyperparameters that satisfy all validation rules."""
        try:
            from ...steps.hyperparams.hyperparameters_bsm import BSMModelHyperparameters
            
            # Create comprehensive field lists that satisfy validation (include id_name)
            full_field_list = ['id', 'feature1', 'feature2', 'text_field', 'target']
            cat_field_list = ['feature1']  # Subset of full_field_list
            tab_field_list = ['feature2']  # Subset of full_field_list
            
            return BSMModelHyperparameters(
                # Required fields from base ModelHyperparameters
                full_field_list=full_field_list,
                cat_field_list=cat_field_list,
                tab_field_list=tab_field_list,
                id_name='id',
                label_name='target',
                multiclass_categories=['class_0', 'class_1'],
                # Required BSM-specific fields
                tokenizer='bert-base-uncased',
                text_name='text_field',
                # Optional fields with comprehensive defaults (removed extra_forbidden fields)
                lr_decay=0.05,
                adam_epsilon=1e-08,
                momentum=0.9,
                run_scheduler=True,
                val_check_interval=0.25,
                warmup_steps=300,
                weight_decay=0.0,
                gradient_clip_val=1.0,
                fp16=False,
                early_stop_metric='val_loss',
                early_stop_patience=3,
                load_ckpt=False,
                text_field_overwrite=False,
                chunk_trancate=True,
                max_total_chunks=3,
                max_sen_len=512,
                fixed_tokenizer_length=True,
                text_input_ids_key='input_ids',
                text_attention_mask_key='attention_mask',
                num_channels=[100, 100],
                num_layers=2,
                dropout_keep=0.1,
                kernel_size=[3, 5, 7],
                is_embeddings_trainable=True,
                pretrained_embedding=True,
                reinit_layers=2,
                reinit_pooler=True,
                hidden_common_dim=100,
                batch_size=32
                # Removed learning_rate and epochs as they cause extra_forbidden validation errors
            )
        except Exception as e:
            if self.test_mode:
                print(f"INFO: Failed to create BSMModelHyperparameters, using fallback: {e}")
            else:
                print(f"Failed to create BSMModelHyperparameters: {e}")
            # Fallback to SimpleNamespace with all required attributes
            mock_hp = SimpleNamespace()
            mock_hp.full_field_list = ['feature1', 'feature2', 'text_field', 'target']
            mock_hp.cat_field_list = ['feature1']
            mock_hp.tab_field_list = ['feature2']
            mock_hp.id_name = 'id'
            mock_hp.label_name = 'target'
            mock_hp.text_name = 'text_field'
            mock_hp.tokenizer = 'bert-base-uncased'
            mock_hp.model_dump = lambda: {
                'learning_rate': 0.001, 'batch_size': 32, 'epochs': 10
            }
            return mock_hp
    
    def _create_enhanced_xgboost_model_eval_hyperparameters(self) -> Any:
        """Create enhanced XGBoost model evaluation hyperparameters."""
        try:
            from ...steps.hyperparams.hyperparameters_xgboost import XGBoostModelHyperparameters
            
            # Create minimal but valid hyperparameters for model evaluation
            full_field_list = ['id', 'feature1', 'feature2', 'target']
            cat_field_list = ['feature1']
            tab_field_list = ['feature2']
            
            return XGBoostModelHyperparameters(
                full_field_list=full_field_list,
                cat_field_list=cat_field_list,
                tab_field_list=tab_field_list,
                id_name='id',
                label_name='target',
                multiclass_categories=['class_0', 'class_1'],
                num_round=100,
                max_depth=6,
                eta=0.1
            )
        except Exception as e:
            if self.test_mode:
                print(f"INFO: Failed to create XGBoost model eval hyperparameters, using fallback: {e}")
            else:
                print(f"Failed to create XGBoost model eval hyperparameters: {e}")
            # Fallback to SimpleNamespace
            mock_hp = SimpleNamespace()
            mock_hp.id_name = 'id'
            mock_hp.label_name = 'target'
            mock_hp.full_field_list = ['id', 'feature1', 'feature2', 'target']
            mock_hp.cat_field_list = ['feature1']
            mock_hp.tab_field_list = ['feature2']
            mock_hp.model_dump = lambda: {'id_name': 'id', 'label_name': 'target'}
            return mock_hp
    
    def _create_xgboost_model_config_from_base(self, base_config: Any) -> Any:
        """Create proper XGBoostModelStepConfig instance using from_base_config."""
        try:
            from ...steps.configs.config_xgboost_model_step import XGBoostModelStepConfig
            return XGBoostModelStepConfig.from_base_config(
                base_config,
                instance_type='ml.m5.large',
                entry_point='inference.py',
                framework_version='1.5-1',
                py_version='py3',
                accelerator_type=None,
                model_name='test-xgboost-model',
                tags=None,
                initial_instance_count=1,
                container_startup_health_check_timeout=300,
                container_memory_limit=6144,
                data_download_timeout=900,
                inference_memory_limit=6144,
                max_concurrent_invocations=10,
                max_payload_size=6
            )
        except Exception as e:
            if self.test_mode:
                print(f"INFO: Failed to create XGBoostModelStepConfig from base: {e}")
            else:
                print(f"Failed to create XGBoostModelStepConfig from base: {e}")
            return None
    
    def _create_pytorch_model_config_from_base(self, base_config: Any) -> Any:
        """Create proper PyTorchModelStepConfig instance using from_base_config."""
        try:
            from ...steps.configs.config_pytorch_model_step import PyTorchModelStepConfig
            return PyTorchModelStepConfig.from_base_config(
                base_config,
                instance_type='ml.m5.large',
                entry_point='inference.py',
                framework_version='1.12.0',
                py_version='py38',
                accelerator_type=None,
                model_name='test-pytorch-model',
                tags=None,
                initial_instance_count=1,
                container_startup_health_check_timeout=300,
                container_memory_limit=6144,
                data_download_timeout=900,
                inference_memory_limit=6144,
                max_concurrent_invocations=10,
                max_payload_size=6
            )
        except Exception as e:
            if self.test_mode:
                print(f"INFO: Failed to create PyTorchModelStepConfig from base: {e}")
            else:
                print(f"Failed to create PyTorchModelStepConfig from base: {e}")
            return None
    
    def _create_batch_transform_config_from_base(self, base_config: Any) -> Any:
        """Create proper BatchTransformConfig instance using from_base_config."""
        try:
            from ...steps.configs.config_batch_transform_step import BatchTransformStepConfig
            return BatchTransformStepConfig.from_base_config(
                base_config,
                job_type='training',  # Required field
                transform_instance_type='ml.m5.large',
                transform_instance_count=1,
                content_type='text/csv',
                accept='text/csv',
                split_type='Line',
                assemble_with='Line',
                input_filter='$[1:]',
                output_filter='$[-1]',
                join_source='Input'
            )
        except Exception as e:
            if self.test_mode:
                print(f"INFO: Failed to create BatchTransformStepConfig from base: {e}")
            else:
                print(f"Failed to create BatchTransformStepConfig from base: {e}")
            return None
    
    def _create_registration_config_from_base(self, base_config: Any) -> Any:
        """Create proper RegistrationConfig instance using from_base_config."""
        try:
            from ...steps.configs.config_registration_step import RegistrationConfig
            return RegistrationConfig.from_base_config(
                base_config,
                model_owner='test-team',
                model_domain='test-domain',
                model_objective='test-objective'
            )
        except Exception as e:
            if self.test_mode:
                print(f"INFO: Failed to create RegistrationConfig from base: {e}")
            else:
                print(f"Failed to create RegistrationConfig from base: {e}")
            return None
    
    def _create_cradle_data_loading_config_from_base(self, base_config: Any) -> Any:
        """Create proper CradleDataLoadConfig instance using cradle_config_factory."""
        try:
            from ...core.config_fields.cradle_config_factory import create_cradle_data_load_config
            return create_cradle_data_load_config(
                base_config=base_config,
                job_type='training',
                mds_field_list=['objectId', 'transactionDate', 'feature1', 'feature2'],
                start_date='2025-01-01T00:00:00',
                end_date='2025-01-31T23:59:59',
                tag_edx_provider='test-provider',
                tag_edx_subject='test-subject',
                tag_edx_dataset='test-dataset',
                etl_job_id='test-etl-job',
                edx_manifest_comment='test-comment',
                service_name='test-service',
                cradle_account='Buyer-Abuse-RnD-Dev',
                org_id=0,
                cluster_type='STANDARD',
                output_format='PARQUET',
                output_save_mode='ERRORIFEXISTS',
                split_job=False,
                days_per_split=7,
                merge_sql=None,
                s3_input_override=None,
                transform_sql=None,
                tag_schema=None,
                use_dedup_sql=None,
                mds_join_key='objectId',
                edx_join_key='order_id',
                join_type='JOIN'
            )
        except Exception as e:
            if self.test_mode:
                print(f"INFO: Failed to create CradleDataLoadConfig using factory: {e}")
            else:
                print(f"Failed to create CradleDataLoadConfig using factory: {e}")
            return None
    
    def create_step_type_mocks(self) -> Dict[str, Any]:
        """Create step type-specific mock objects."""
        mocks = {}
        
        if self.sagemaker_step_type == "Processing":
            mocks.update(self._create_processing_mocks())
        elif self.sagemaker_step_type == "Training":
            mocks.update(self._create_training_mocks())
        elif self.sagemaker_step_type == "Transform":
            mocks.update(self._create_transform_mocks())
        elif self.sagemaker_step_type == "CreateModel":
            mocks.update(self._create_createmodel_mocks())
        
        return mocks
    
    def get_expected_dependencies(self) -> List[str]:
        """Get expected dependencies based on step type and pattern."""
        if self.sagemaker_step_type == "Processing":
            return self._get_processing_dependencies()
        elif self.sagemaker_step_type == "Training":
            return self._get_training_dependencies()
        elif self.sagemaker_step_type == "Transform":
            return self._get_transform_dependencies()
        elif self.sagemaker_step_type == "CreateModel":
            return self._get_createmodel_dependencies()
        else:
            return ["input"]
    
    def _create_base_config(self) -> SimpleNamespace:
        """Create enhanced base configuration with better validation support."""
        mock_config = SimpleNamespace()
        mock_config.region = 'NA'  # Use valid region code
        mock_config.pipeline_name = 'test-pipeline'
        mock_config.pipeline_s3_loc = 's3://bucket/prefix'
        
        # Add common methods
        mock_config.get_image_uri = lambda: 'mock-image-uri'
        mock_config.get_script_path = lambda: 'mock_script.py'
        mock_config.get_script_contract = lambda: None
        
        return mock_config
    
    def _add_processing_config(self, mock_config: SimpleNamespace) -> None:
        """Add Processing step-specific configuration with enhanced validation."""
        mock_config.processing_instance_type = 'ml.m5.large'
        mock_config.processing_instance_type_large = 'ml.m5.xlarge'
        mock_config.processing_instance_type_small = 'ml.m5.large'
        mock_config.processing_instance_count = 1
        mock_config.processing_volume_size = 30
        mock_config.processing_entry_point = 'process.py'
        mock_config.source_dir = '/tmp/mock_scripts'  # Use the test script directory
        mock_config.use_large_processing_instance = False
        
        # Add processing-specific attributes based on builder type
        builder_name = self.step_info.get("builder_class_name", "")
        if "TabularPreprocessing" in builder_name:
            mock_config.job_type = 'training'
            mock_config.label_name = 'target'
            mock_config.train_ratio = 0.7
            mock_config.test_val_ratio = 0.5
            mock_config.categorical_columns = ['category_1', 'category_2']
            mock_config.numerical_columns = ['numeric_1', 'numeric_2']
            mock_config.text_columns = ['text_1', 'text_2']
            mock_config.date_columns = ['date_1']
            mock_config.processing_entry_point = 'tabular_preprocess.py'
            mock_config.processing_framework_version = '1.2-1'
        elif "RiskTableMapping" in builder_name:
            mock_config.job_type = 'training'
            mock_config.label_name = 'target'
            mock_config.categorical_columns = ['risk_category', 'market_segment']
            mock_config.processing_entry_point = 'risk_table_mapping.py'
            mock_config.processing_framework_version = '1.2-1'
            mock_config.risk_mapping_config = {'high': 3, 'medium': 2, 'low': 1}
        elif "CurrencyConversion" in builder_name:
            mock_config.job_type = 'training'
            mock_config.label_field = 'target'
            mock_config.marketplace_id_col = 'marketplace_id'
            mock_config.currency_col = 'currency'
            mock_config.default_currency = 'USD'
            mock_config.mode = 'per_split'
            mock_config.currency_conversion_var_list = ['price', 'cost']
            mock_config.currency_conversion_dict = {'USD': 1.0, 'EUR': 1.1, 'GBP': 1.3}
            mock_config.marketplace_info = {'1': {'currency': 'USD'}, '2': {'currency': 'EUR'}}
            mock_config.train_ratio = 0.7
            mock_config.test_val_ratio = 0.5
            mock_config.enable_currency_conversion = True
            mock_config.skip_invalid_currencies = False
            mock_config.processing_entry_point = 'currency_conversion.py'
            mock_config.processing_framework_version = '1.2-1'
        elif "DummyTraining" in builder_name:
            mock_config.job_type = 'training'
            mock_config.model_type = 'xgboost'
            mock_config.pretrained_model_s3_uri = 's3://bucket/pretrained/model.tar.gz'
            mock_config.processing_entry_point = 'dummy_training.py'
            mock_config.processing_framework_version = '1.2-1'
        elif "ModelEval" in builder_name or "XGBoostModelEval" in builder_name:
            mock_config.job_type = 'training'
            mock_config.processing_entry_point = 'model_evaluation_xgb.py'
            mock_config.processing_source_dir = '/tmp/mock_scripts'
            mock_config.xgboost_framework_version = '1.7-1'
            # Add enhanced hyperparameters mock
            mock_hp = self._create_enhanced_xgboost_model_eval_hyperparameters()
            mock_config.hyperparameters = mock_hp
        elif "ModelCalibration" in builder_name:
            mock_config.job_type = 'training'
            mock_config.label_field = 'target'  # Note: ModelCalibration uses label_field
            mock_config.id_field = 'id'
            mock_config.calibration_method = 'isotonic'
            mock_config.processing_entry_point = 'model_calibration.py'
            mock_config.processing_framework_version = '1.2-1'
        elif "Payload" in builder_name:
            mock_config.model_owner = 'test-team'
            mock_config.model_domain = 'test-domain'
            mock_config.model_objective = 'test-objective'
            mock_config.source_model_inference_output_variable_list = {'prediction': 'NUMERIC'}
            mock_config.source_model_inference_input_variable_list = {'feature1': 'NUMERIC', 'feature2': 'TEXT'}
            mock_config.expected_tps = 100
            mock_config.max_latency_in_millisecond = 1000
            mock_config.framework = 'xgboost'
            mock_config.processing_entry_point = 'payload.py'
            mock_config.source_model_inference_content_types = ['text/csv']
            mock_config.source_model_inference_response_types = ['application/json']
            mock_config.max_acceptable_error_rate = 0.2
            mock_config.default_numeric_value = 0.0
            mock_config.default_text_value = 'DEFAULT_TEXT'
            mock_config.bucket = 'test-bucket'
        elif "Package" in builder_name:
            mock_config.processing_entry_point = 'package.py'
            mock_config.processing_source_dir = '/tmp/mock_scripts'
        
        # Add processor-specific attributes based on framework
        if self.framework == "sklearn":
            mock_config.framework_version = '1.2-1'
            mock_config.py_version = 'py3'
            mock_config.processing_framework_version = '1.2-1'
        elif self.framework == "xgboost":
            mock_config.framework_version = '1.7-1'
            mock_config.py_version = 'py3'
            mock_config.processing_framework_version = '1.7-1'
    
    def _add_training_config(self, mock_config: SimpleNamespace) -> None:
        """Add Training step-specific configuration with enhanced validation."""
        mock_config.training_instance_type = 'ml.m5.xlarge'
        mock_config.training_instance_count = 1
        mock_config.training_volume_size = 30
        mock_config.training_entry_point = 'train.py'
        mock_config.source_dir = '/tmp/mock_scripts'  # Use the test script directory
        
        # Add enhanced hyperparameters based on framework
        builder_name = self.step_info.get("builder_class_name", "")
        if "XGBoostTraining" in builder_name:
            mock_hp = self._create_enhanced_xgboost_hyperparameters()
        elif "PyTorchTraining" in builder_name:
            mock_hp = self._create_enhanced_pytorch_hyperparameters()
        else:
            # Generic training hyperparameters
            mock_hp = SimpleNamespace()
            mock_hp.model_dump = lambda: {
                'learning_rate': 0.1, 
                'max_depth': 6,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
            mock_hp.is_binary = True
            mock_hp.num_classes = 2
            mock_hp.input_tab_dim = 10
            mock_hp.objective = 'binary:logistic'
            mock_hp.eval_metric = 'auc'
        
        mock_config.hyperparameters = mock_hp
        
        # Add S3 paths for hyperparameters
        mock_config.hyperparameters_s3_uri = 's3://test-bucket/config/hyperparameters.json'
        mock_config.bucket = 'test-bucket'
        mock_config.current_date = '2025-08-15'
        
        # Add framework-specific training config
        if self.framework == "xgboost":
            mock_config.framework_version = '1.7-1'
            mock_config.py_version = 'py3'
            mock_config.training_entry_point = 'train_xgb.py'
        elif self.framework == "pytorch":
            mock_config.framework_version = '1.12.0'
            mock_config.py_version = 'py38'
            mock_config.training_entry_point = 'train_pytorch.py'
        elif self.framework == "tensorflow":
            mock_config.framework_version = '2.11.0'
            mock_config.py_version = 'py39'
            mock_config.training_entry_point = 'train_tf.py'
    
    def _add_transform_config(self, mock_config: SimpleNamespace) -> None:
        """Add Transform step-specific configuration."""
        mock_config.transform_instance_type = 'ml.m5.large'
        mock_config.transform_instance_count = 1
        mock_config.transform_max_concurrent_transforms = 1
        mock_config.transform_max_payload = 6
        mock_config.transform_accept = 'text/csv'
        mock_config.transform_content_type = 'text/csv'
        mock_config.transform_strategy = 'MultiRecord'
        mock_config.transform_assemble_with = 'Line'
    
    def _add_createmodel_config(self, mock_config: SimpleNamespace) -> None:
        """Add CreateModel step-specific configuration."""
        mock_config.model_name = 'test-model'
        mock_config.primary_container_image = 'mock-image-uri'
        mock_config.model_data_url = 's3://bucket/model.tar.gz'
        mock_config.execution_role_arn = 'arn:aws:iam::123456789012:role/MockRole'
    
    def _add_generic_config(self, mock_config: SimpleNamespace) -> None:
        """Add generic configuration for unknown step types."""
        mock_config.instance_type = 'ml.m5.large'
        mock_config.instance_count = 1
        mock_config.volume_size = 30
        mock_config.entry_point = 'generic_script.py'
        mock_config.source_dir = '/tmp/mock_scripts'
    
    def _add_framework_config(self, mock_config: SimpleNamespace) -> None:
        """Add framework-specific configuration."""
        if self.framework == "xgboost":
            if not hasattr(mock_config, 'framework_version'):
                mock_config.framework_version = '1.7-1'
            if not hasattr(mock_config, 'py_version'):
                mock_config.py_version = 'py3'
        elif self.framework == "pytorch":
            if not hasattr(mock_config, 'framework_version'):
                mock_config.framework_version = '1.12.0'
            if not hasattr(mock_config, 'py_version'):
                mock_config.py_version = 'py38'
        elif self.framework == "tensorflow":
            if not hasattr(mock_config, 'framework_version'):
                mock_config.framework_version = '2.11.0'
            if not hasattr(mock_config, 'py_version'):
                mock_config.py_version = 'py39'
    
    def _create_processing_mocks(self) -> Dict[str, Any]:
        """Create Processing step-specific mocks."""
        mocks = {}
        
        # Mock ProcessingInput
        mock_processing_input = MagicMock()
        mock_processing_input.source = 's3://bucket/input'
        mock_processing_input.destination = '/opt/ml/processing/input'
        mocks['processing_input'] = mock_processing_input
        
        # Mock ProcessingOutput
        mock_processing_output = MagicMock()
        mock_processing_output.source = '/opt/ml/processing/output'
        mock_processing_output.destination = 's3://bucket/output'
        mocks['processing_output'] = mock_processing_output
        
        # Mock Processor based on framework
        if self.framework == "sklearn":
            from sagemaker.sklearn.processing import SKLearnProcessor
            mocks['processor_class'] = SKLearnProcessor
        elif self.framework == "xgboost":
            from sagemaker.xgboost.processing import XGBoostProcessor
            mocks['processor_class'] = XGBoostProcessor
        else:
            from sagemaker.processing import ScriptProcessor
            mocks['processor_class'] = ScriptProcessor
        
        return mocks
    
    def _create_training_mocks(self) -> Dict[str, Any]:
        """Create Training step-specific mocks."""
        mocks = {}
        
        # Mock TrainingInput
        mock_training_input = MagicMock()
        mock_training_input.config = {
            'DataSource': {
                'S3DataSource': {
                    'S3Uri': 's3://bucket/training-data',
                    'S3DataType': 'S3Prefix'
                }
            }
        }
        mocks['training_input'] = mock_training_input
        
        # Mock Estimator based on framework
        if self.framework == "xgboost":
            from sagemaker.xgboost.estimator import XGBoost
            mocks['estimator_class'] = XGBoost
        elif self.framework == "pytorch":
            from sagemaker.pytorch.estimator import PyTorch
            mocks['estimator_class'] = PyTorch
        elif self.framework == "tensorflow":
            from sagemaker.tensorflow.estimator import TensorFlow
            mocks['estimator_class'] = TensorFlow
        else:
            from sagemaker.estimator import Estimator
            mocks['estimator_class'] = Estimator
        
        return mocks
    
    def _create_transform_mocks(self) -> Dict[str, Any]:
        """Create Transform step-specific mocks."""
        mocks = {}
        
        # Mock TransformInput
        mock_transform_input = MagicMock()
        mock_transform_input.data = 's3://bucket/transform-input'
        mock_transform_input.content_type = 'text/csv'
        mocks['transform_input'] = mock_transform_input
        
        # Mock Transformer
        mock_transformer = MagicMock()
        mock_transformer.model_name = 'test-model'
        mocks['transformer'] = mock_transformer
        
        return mocks
    
    def _create_createmodel_mocks(self) -> Dict[str, Any]:
        """Create CreateModel step-specific mocks that don't interfere with SageMaker validation."""
        mocks = {}
        
        # Mock Model with proper string attributes to avoid MagicMock issues
        mock_model = MagicMock()
        mock_model.name = 'test-model'
        mock_model.image_uri = 'mock-image-uri'
        mock_model.model_data = 's3://bucket/model.tar.gz'
        
        # Ensure model.create() returns proper step arguments without conflicts
        # This prevents the "step_args and model are mutually exclusive" error
        mock_model.create.return_value = {
            'ModelName': 'test-model',
            'PrimaryContainer': {
                'Image': 'mock-image-uri',
                'ModelDataUrl': 's3://bucket/model.tar.gz',
                'Environment': {}
            },
            'ExecutionRoleArn': 'arn:aws:iam::123456789012:role/MockRole'
        }
        
        mocks['model'] = mock_model
        
        return mocks
    
    def _get_processing_dependencies(self) -> List[str]:
        """Get expected dependencies for Processing steps."""
        builder_name = self.step_info.get("builder_class_name", "")
        
        if "TabularPreprocessing" in builder_name:
            return ["DATA"]
        elif "RiskTableMapping" in builder_name:
            return ["risk_tables"]  # RiskTableMapping needs risk_tables dependency
        elif "XGBoostModelEval" in builder_name:
            return ["model_input"]  # XGBoostModelEval only needs model_input
        elif "ModelEval" in builder_name:
            return ["model_input", "eval_data_input"]
        else:
            return ["input_data"]
    
    def _get_training_dependencies(self) -> List[str]:
        """Get expected dependencies for Training steps."""
        return ["input_path"]
    
    def _get_transform_dependencies(self) -> List[str]:
        """Get expected dependencies for Transform steps."""
        return ["model_input", "transform_input"]
    
    def _get_createmodel_dependencies(self) -> List[str]:
        """Get expected dependencies for CreateModel steps."""
        return ["model_data"]
